import hashlib
import json
import pickle
from copy import deepcopy

import dashboard.launch as launch
import pytest
import torch
from dashboard.config import load_config, validate_startup
from dashboard.launch import (
    _prepare_parent_resume_link,
    _runtime_status_from_line,
    _training_arg,
    build_parser,
)
from dashboard.provenance import WorkingTreeState

from ascento_mjlab.control_contract import current_action_contract
from ascento_mjlab.plant_contract import current_plant_contract
from ascento_mjlab.task_contract import current_task_contract_for_task


def test_launcher_uses_same_default_artifact_root_as_dashboard(monkeypatch, tmp_path):
    monkeypatch.setenv("ASCENTO_ARTIFACT_ROOT", str(tmp_path))
    config = load_config()
    args = build_parser().parse_args([])

    assert args.artifact_root == config.artifact_root == tmp_path.resolve()


def _dirty_state() -> WorkingTreeState:
    return WorkingTreeState("abc123", "main", ("tracked.py",), ("new.py",))


def test_launcher_rejects_dirty_tree_without_override(monkeypatch, tmp_path):
    monkeypatch.setattr(launch, "working_tree_state", lambda _: _dirty_state())

    with pytest.raises(ValueError, match="allow-dirty-provenance"):
        launch.validate_source_provenance(tmp_path, allow_dirty_provenance=False)


def test_launcher_writes_bundle_only_when_explicitly_allowed(monkeypatch, tmp_path):
    monkeypatch.setattr(launch, "working_tree_state", lambda _: _dirty_state())
    monkeypatch.setattr(launch, "write_dirty_source_bundle", lambda *_: {"recorded": True})

    provenance = launch.validate_source_provenance(tmp_path, allow_dirty_provenance=True)

    assert provenance["mode"] == "dirty_bundle"


def test_launcher_argument_metadata_parser_supports_both_cli_forms():
    args = [
        "--seed=7",
        "--device",
        "cuda:0",
        "--env.sim.mujoco.timestep",
        "0.002",
        "--agent.seed",
        "11",
    ]

    assert _training_arg(args, "--seed", "--agent.seed") == "11"
    assert _training_arg(args, "--device") == "cuda:0"
    assert _training_arg(args, "--env.sim.mujoco.timestep") == "0.002"


def test_launcher_prepares_parent_checkpoint_link_before_resume(monkeypatch, tmp_path):
    checkpoint = tmp_path / "parent" / "ascento_balance" / "source" / "model_7999.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    run_dir = tmp_path / "child"
    run_dir.mkdir()
    monkeypatch.setattr(launch, "_validate_parent_checkpoint", lambda *_args: None)

    _prepare_parent_resume_link(
        run_dir,
        parent_checkpoint=str(checkpoint),
        training_args=["--agent.resume", "True", "--agent.load-run", "_resume_parent"],
        stage="balance",
        task="Ascento-Balance-Flat",
    )

    link = run_dir / "ascento_balance" / "_resume_parent"
    assert link.is_symlink()
    assert link.resolve() == checkpoint.parent.resolve()


def test_launcher_uses_the_task_rl_experiment_name_for_locomotion_resume(monkeypatch, tmp_path):
    checkpoint = (
        tmp_path / "parent" / "ascento_locomotion_flat" / "source" / "model_0.pt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    run_dir = tmp_path / "child"
    run_dir.mkdir()
    monkeypatch.setattr(launch, "_validate_parent_checkpoint", lambda *_args: None)

    _prepare_parent_resume_link(
        run_dir,
        parent_checkpoint=str(checkpoint),
        training_args=["--agent.resume", "True", "--agent.load-run", "_resume_parent"],
        stage="locomotion",
        task="Ascento-Locomotion-Flat",
    )

    link = run_dir / "ascento_locomotion_flat" / "_resume_parent"
    assert link.is_symlink()
    assert link.resolve() == checkpoint.parent.resolve()


def test_launcher_rejects_ambiguous_managed_resume(tmp_path):
    checkpoint = tmp_path / "ascento_balance" / "source" / "model_7999.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="load-run _resume_parent"):
        _prepare_parent_resume_link(
            tmp_path / "child",
            parent_checkpoint=str(checkpoint),
            training_args=["--agent.resume", "True"],
            stage="balance",
            task="Ascento-Balance-Flat",
        )


def test_launcher_rejects_parent_checkpoint_without_task_contract(tmp_path):
    checkpoint = tmp_path / "parent" / "ascento_balance" / "source" / "model_7999.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(
        {
            "infos": {
                "plant_contract": current_plant_contract(),
                "action_contract": current_action_contract(),
            }
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="task topology contract"):
        launch._validate_parent_checkpoint(checkpoint, "Ascento-Balance-Flat")


def test_launcher_accepts_parent_checkpoint_with_current_contracts(tmp_path):
    checkpoint = tmp_path / "parent" / "ascento_balance" / "source" / "model_7999.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(
        {
            "infos": {
                "plant_contract": current_plant_contract(),
                "action_contract": current_action_contract(),
                "task_contract": current_task_contract_for_task("Ascento-Balance-Flat"),
            }
        },
        checkpoint,
    )

    launch._validate_parent_checkpoint(checkpoint, "Ascento-Balance-Flat")


@pytest.mark.parametrize(
    "load_error",
    [EOFError("truncated checkpoint"), pickle.UnpicklingError("invalid checkpoint")],
)
def test_launcher_rejects_parent_checkpoint_deserialization_errors(
    tmp_path, monkeypatch, load_error
):
    checkpoint = tmp_path / "parent.pt"
    checkpoint.write_bytes(b"invalid")

    def fail_load(*_args, **_kwargs):
        raise load_error

    monkeypatch.setattr(torch, "load", fail_load)

    with pytest.raises(ValueError, match="cannot read parent checkpoint contracts"):
        launch._validate_parent_checkpoint(checkpoint, "Ascento-Balance-Flat")


@pytest.mark.parametrize(
    "load_error",
    [EOFError("truncated checkpoint"), pickle.UnpicklingError("invalid checkpoint")],
)
def test_finalization_records_checkpoint_deserialization_errors(
    tmp_path, monkeypatch, load_error
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "model_10.pt").write_bytes(b"invalid")
    manifest_path = run_dir / "experiment_manifest.json"
    manifest_path.write_text(json.dumps({"task": "synthetic"}), encoding="utf-8")

    def fail_load(*_args, **_kwargs):
        raise load_error

    monkeypatch.setattr(torch, "load", fail_load)

    launch._finalize_experiment_manifest(manifest_path, run_dir)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert str(load_error) in manifest["checkpoint"]["contract_read_error"]


def test_final_manifest_uses_actual_checkpoint_contracts_and_preserves_launch_contracts(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    current = current_task_contract_for_task("Ascento-Balance-Flat")
    checkpoint_task = deepcopy(current)
    checkpoint_task["topology"]["task_id"] = None
    checkpoint_task["topology_sha256"] = hashlib.sha256(
        json.dumps(checkpoint_task["topology"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    checkpoint = run_dir / "model_10.pt"
    torch.save(
        {
            "infos": {
                "plant_contract": current_plant_contract(),
                "action_contract": current_action_contract(),
                "task_contract": checkpoint_task,
            }
        },
        checkpoint,
    )
    manifest_path = run_dir / "experiment_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "plant_contract": current_plant_contract(),
                "action_contract": current_action_contract(),
                "task_contract": current,
            }
        ),
        encoding="utf-8",
    )

    launch._finalize_experiment_manifest(manifest_path, run_dir)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["launch_contracts"]["task_contract"] == current
    assert manifest["task_contract"] == checkpoint_task
    assert manifest["checkpoint"]["task_contract"] == checkpoint_task


def test_compose_does_not_override_image_build_provenance_at_runtime():
    compose = (launch.REPO_ROOT / "docker" / "compose.yaml").read_text(encoding="utf-8")

    assert "\n      ASCENTO_REPOSITORY_COMMIT:" not in compose
    assert "\n      ASCENTO_REPOSITORY_BRANCH:" not in compose


def test_launcher_accepts_dashboard_horizon_after_training_separator():
    args = build_parser().parse_args(
        [
            "--artifact-root",
            "/tmp/runs",
            "--name",
            "dashboard-run",
            "--preinitialized",
            "--",
            "--env.episode-length-s",
            "20.0",
            "--env.scene.num-envs",
            "512",
        ]
    )

    assert args.training_args == [
        "--",
        "--env.episode-length-s",
        "20.0",
        "--env.scene.num-envs",
        "512",
    ]


def test_launcher_extracts_runtime_device_seed_and_world_size():
    assert _runtime_status_from_line("[INFO] Training with: device=cuda:0, seed=42, rank=0") == {
        "device": "cuda:0",
        "seed": 42,
        "rank": 0,
    }
    assert _runtime_status_from_line("[INFO] Launching training with 2 GPUs") == {
        "gpu_world_size": 2
    }
    assert _runtime_status_from_line(
        "HORIZON_CURRICULUM horizon_s=60.0 stage=2 qualified_windows=0 timeout_fraction=0.9219"
    ) == {
        "episode_horizon_s": 60.0,
        "horizon_stage": 2,
        "horizon_qualified_windows": 0,
        "horizon_timeout_fraction": 0.9219,
    }
    assert _runtime_status_from_line(
        "HORIZON_CURRICULUM horizon_s=300.0 stage=4 qualified_windows=0 "
        "failed_windows=0 stage_windows=4 top_horizon_windows=3 transition=protected "
        "timeout_fraction=0.4219 candidate_checkpoint=model_best_long_horizon.pt"
    ) == {
        "episode_horizon_s": 300.0,
        "horizon_stage": 4,
        "horizon_qualified_windows": 0,
        "horizon_failed_windows": 0,
        "horizon_stage_windows": 4,
        "horizon_top_windows": 3,
        "horizon_transition": "protected",
        "horizon_timeout_fraction": 0.4219,
        "long_horizon_candidate_checkpoint": "model_best_long_horizon.pt",
    }


def test_launcher_uses_injected_repository_version_without_git(monkeypatch):
    monkeypatch.setattr(launch, "_git_value", lambda *args: None)
    monkeypatch.setenv("ASCENTO_REPOSITORY_COMMIT", "container-commit")
    monkeypatch.setenv("ASCENTO_REPOSITORY_BRANCH", "main")

    metadata = launch.git_metadata()
    assert metadata["commit"] == "container-commit"
    assert metadata["branch"] == "main"


def test_clean_container_provenance_is_marked_as_an_image_build(monkeypatch, tmp_path):
    monkeypatch.setattr(launch, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("ASCENTO_REPOSITORY_COMMIT", "container-commit")
    monkeypatch.setenv("ASCENTO_REPOSITORY_BRANCH", "main")
    monkeypatch.setenv("ASCENTO_REPOSITORY_DIRTY", "0")

    provenance = launch.validate_source_provenance(
        tmp_path, allow_dirty_provenance=False
    )

    assert provenance == {
        "mode": "clean_image_build",
        "commit": "container-commit",
        "branch": "main",
    }


def test_compose_bakes_clean_build_status_without_runtime_override():
    compose = (launch.REPO_ROOT / "docker" / "compose.yaml").read_text(encoding="utf-8")

    assert "REPOSITORY_DIRTY: ${ASCENTO_REPOSITORY_DIRTY:-unknown}" in compose
    assert "\n      ASCENTO_REPOSITORY_DIRTY:" not in compose


def test_startup_validation_reports_bad_artifact_root(tmp_path):
    bad_root = tmp_path / "artifact-file"
    bad_root.write_text("not a directory", encoding="utf-8")
    monkeypatch_config = load_config()
    config = type(monkeypatch_config)(
        repo_root=monkeypatch_config.repo_root,
        artifact_root=bad_root,
        frontend_dist=monkeypatch_config.frontend_dist,
        stale_after_seconds=monkeypatch_config.stale_after_seconds,
    )

    with pytest.raises(RuntimeError, match="artifact root is not a directory"):
        validate_startup(config)


def test_read_only_monitor_validation_does_not_create_missing_artifact_root(tmp_path):
    missing_root = tmp_path / "logs" / "rsl_rl"
    base = load_config()
    config = type(base)(
        repo_root=base.repo_root,
        artifact_root=missing_root,
        frontend_dist=base.frontend_dist,
        stale_after_seconds=base.stale_after_seconds,
    )

    warnings = validate_startup(config, create_artifact_root=False)

    assert missing_root.exists() is False
    assert any("does not exist yet" in warning for warning in warnings)
