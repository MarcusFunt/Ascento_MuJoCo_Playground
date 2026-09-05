import json
from pathlib import Path

import pytest
from dashboard.run_service import RunService


def _run(root: Path, name: str, *, state: str = "finished", reward: float | None = None) -> Path:
    run = root / name
    run.mkdir(parents=True)
    status = {
        "state": state,
        "task": "Ascento-Balance-Flat",
        "stage": "balance",
        "git_commit": "abc123",
    }
    if state in {"starting", "running"}:
        status.update({"launcher_pid": 12345, "process_group": 12345})
    (run / "run_status.json").write_text(json.dumps(status), encoding="utf-8")
    if reward is not None:
        (run / "telemetry.jsonl").write_text(
            json.dumps(
                {
                    "completed_steps": 4,
                    "total_steps": 10,
                    "wall_time": 1,
                    "metrics": {"Train/mean_reward": reward},
                }
            )
            + "\n",
            encoding="utf-8",
        )
    return run


def service_compare_ids(service: RunService, root: Path):
    from dashboard.health import discover_dashboard_runs

    return [ref.id for ref in discover_dashboard_runs(root)]


def test_metadata_can_annotate_existing_runs(tmp_path):
    run = _run(tmp_path, "legacy")
    service = RunService(tmp_path)
    run_id = service_compare_ids(service, tmp_path)[0]

    updated = service.update_metadata(
        run_id,
        {
            "display_name": "Recovery baseline after PR83",
            "notes": "Known-good comparison run.",
            "tags": ["recovery", "baseline", "recovery"],
            "purpose": "baseline",
        },
    )

    metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["display_name"] == "Recovery baseline after PR83"
    assert metadata["tags"] == ["recovery", "baseline"]
    assert updated["display_name"] == "Recovery baseline after PR83"
    assert updated["notes"] == "Known-good comparison run."


def test_nested_training_artifacts_use_launcher_metadata(tmp_path):
    launcher = tmp_path / "managed"
    launcher.mkdir()
    (launcher / "run_status.json").write_text(
        json.dumps({"state": "finished", "task": "Ascento-Recovery-Flat", "stage": "recovery"}),
        encoding="utf-8",
    )
    (launcher / "run_metadata.json").write_text(
        json.dumps({"display_name": "Managed recovery", "tags": ["recovery"]}),
        encoding="utf-8",
    )
    nested = launcher / "ascento_recovery" / "2026-09-05_12-00-00"
    nested.mkdir(parents=True)
    (nested / "telemetry.jsonl").write_text(
        json.dumps({"completed_steps": 1, "total_steps": 10, "wall_time": 1, "metrics": {}}) + "\n",
        encoding="utf-8",
    )

    service = RunService(tmp_path)
    run_id = service_compare_ids(service, tmp_path)[0]
    detail = service.detail(run_id)

    assert detail["display_name"] == "Managed recovery"
    assert detail["tags"] == ["recovery"]


def test_lineage_rejects_self_parent_and_accepts_other_run(tmp_path):
    _run(tmp_path, "parent")
    _run(tmp_path, "child")
    service = RunService(tmp_path)
    ids = {
        service.detail(run_id)["name"]: run_id for run_id in service_compare_ids(service, tmp_path)
    }

    with pytest.raises(ValueError, match="own parent"):
        service.update_metadata(ids["child"], {"parent_run_id": ids["child"]})

    detail = service.update_metadata(
        ids["child"],
        {"parent_run_id": ids["parent"], "parent_checkpoint": "model_7500.pt"},
    )
    assert detail["lineage"] == {
        "parent_run_id": ids["parent"],
        "parent_checkpoint": "model_7500.pt",
    }


def test_create_starts_detached_launcher_with_metadata_arguments(monkeypatch, tmp_path):
    captured = {}

    class FakeProcess:
        pid = 4321

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr("dashboard.run_service.subprocess.Popen", fake_popen)
    service = RunService(tmp_path)

    created = service.create(
        {
            "display_name": "Velocity validation",
            "task": "Ascento-Velocity-Flat",
            "purpose": "validation",
            "tags": ["velocity", "gate"],
            "notes": "Verify the current gate.",
            "training_args": ["--agent.max-iterations", "5000"],
        }
    )

    command = captured["command"]
    assert created["display_name"] == "Velocity validation"
    assert created["launcher_pid"] == 4321
    assert "dashboard.launch" in command
    assert "--display-name" in command
    assert "Velocity validation" in command
    assert command[-2:] == ["--agent.max-iterations", "5000"]
    assert captured["kwargs"]["start_new_session"] is True


def test_stop_marks_stopping_before_signalling(monkeypatch, tmp_path):
    run = _run(tmp_path, "active", state="running")
    service = RunService(tmp_path)
    run_id = service_compare_ids(service, tmp_path)[0]
    calls = []

    monkeypatch.setattr(
        "dashboard.run_service.os.killpg", lambda pgid, sig: calls.append((pgid, sig))
    )
    monkeypatch.setattr("dashboard.health.os.kill", lambda pid, sig: None)

    result = service.stop(run_id, reason="plateau")
    stored = json.loads((run / "run_status.json").read_text(encoding="utf-8"))

    assert calls and calls[0][0] == 12345
    assert stored["state"] == "stopping"
    assert stored["stop_reason"] == "plateau"
    assert result["state"] == "stopping"


def test_compare_uses_normalized_metrics_and_baseline_deltas(tmp_path):
    _run(tmp_path, "first", reward=2.0)
    _run(tmp_path, "second", reward=3.5)
    service = RunService(tmp_path)
    ids = service_compare_ids(service, tmp_path)
    by_name = {service.detail(run_id)["name"]: run_id for run_id in ids}

    result = service.compare([by_name["first"], by_name["second"]])
    rows = {row["display_name"]: row for row in result["runs"]}

    assert result["baseline_id"] == by_name["first"]
    assert rows["first"]["latest_metrics"]["reward"] == 2.0
    assert rows["second"]["delta_from_baseline"]["reward"] == pytest.approx(1.5)
