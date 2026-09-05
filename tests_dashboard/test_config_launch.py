import dashboard.launch as launch
import pytest
from dashboard.config import load_config, validate_startup
from dashboard.launch import _runtime_status_from_line, _training_arg, build_parser


def test_launcher_uses_same_default_artifact_root_as_dashboard(monkeypatch, tmp_path):
    monkeypatch.setenv("ASCENTO_ARTIFACT_ROOT", str(tmp_path))
    config = load_config()
    args = build_parser().parse_args([])

    assert args.artifact_root == config.artifact_root == tmp_path.resolve()


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


def test_launcher_uses_injected_repository_version_without_git(monkeypatch):
    monkeypatch.setattr(launch, "_git_value", lambda *args: None)
    monkeypatch.setenv("ASCENTO_REPOSITORY_COMMIT", "container-commit")
    monkeypatch.setenv("ASCENTO_REPOSITORY_BRANCH", "main")

    metadata = launch.git_metadata()
    assert metadata["commit"] == "container-commit"
    assert metadata["branch"] == "main"


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
