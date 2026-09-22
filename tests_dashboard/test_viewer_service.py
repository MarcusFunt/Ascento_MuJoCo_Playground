import json
from pathlib import Path
import pytest

import dashboard.viewer_service as viewer_service_module
from dashboard.health import discover_dashboard_runs
from dashboard.run_service import RunService
from dashboard.viewer_service import ViewerBusyError, ViewerService


def _run(root: Path) -> tuple[RunService, str, Path]:
    run = root / "run"
    run.mkdir(parents=True)
    (run / "run_status.json").write_text(
        json.dumps(
            {
                "state": "running",
                "task": "Ascento-Balance-Flat",
                "stage": "balance",
            }
        ),
        encoding="utf-8",
    )
    (run / "telemetry.jsonl").write_text(
        json.dumps(
            {
                "completed_steps": 140,
                "total_steps": 200,
                "wall_time": 1,
                "metrics": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint = run / "model_100.pt"
    checkpoint.write_bytes(b"checkpoint")
    service = RunService(root)
    run_id = discover_dashboard_runs(root)[0].id
    return service, run_id, run


class FakeProcess:
    def __init__(self, command):
        self.command = command
        self.pid = 4321
        self.exit_code = None
        self.terminated = False

    def poll(self):
        return self.exit_code

    def terminate(self):
        self.terminated = True


def test_viewer_service_launches_isolated_worker_and_rejects_duplicate(
    monkeypatch,
    tmp_path,
):
    run_service, run_id, _ = _run(tmp_path / "artifacts")
    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess(command)

    monkeypatch.setattr(viewer_service_module.subprocess, "Popen", fake_popen)
    service = ViewerService(
        run_service,
        logs_root=tmp_path / "viewer-logs",
        stable_age_seconds=0,
    )
    monkeypatch.setattr(service, "_port_open", lambda: False)

    started = service.start(run_id=run_id, checkpoint="latest", follow=True)

    assert started["state"] == "starting"
    assert started["checkpoint"] == "model_100.pt"
    assert "--follow" in captured["command"]
    assert "ascento_mjlab.viewer.worker" in captured["command"]
    assert captured["kwargs"]["start_new_session"] is True

    with pytest.raises(ViewerBusyError, match="already active"):
        service.start(run_id=run_id)


def test_runtime_status_updates_loaded_checkpoint_and_lag(monkeypatch, tmp_path):
    run_service, run_id, run_dir = _run(tmp_path / "artifacts")
    (run_dir / "model_130.pt").write_bytes(b"new-checkpoint")

    monkeypatch.setattr(
        viewer_service_module.subprocess,
        "Popen",
        lambda command, **kwargs: FakeProcess(command),
    )
    service = ViewerService(
        run_service,
        logs_root=tmp_path / "viewer-logs",
        stable_age_seconds=0,
    )
    monkeypatch.setattr(service, "_port_open", lambda: True)
    monkeypatch.setattr(
        run_service,
        "progress",
        lambda _run_id: {"telemetry": {"iteration": 140}},
    )

    started = service.start(run_id=run_id, checkpoint="model_100.pt")
    status_path = tmp_path / "viewer-logs" / f"{started['id']}.json"
    status_path.write_text(
        json.dumps(
            {
                "checkpoint": "model_130.pt",
                "checkpoint_iteration": 130,
                "loaded_at": 123.0,
            }
        ),
        encoding="utf-8",
    )

    status = service.get(started["id"])

    assert status["state"] == "running"
    assert status["checkpoint"] == "model_130.pt"
    assert status["checkpoint_iteration"] == 130
    assert status["training_iteration"] == 140
    assert status["lag_iterations"] == 10


def test_stop_signals_only_viewer_process_group(monkeypatch, tmp_path):
    run_service, run_id, _ = _run(tmp_path / "artifacts")
    monkeypatch.setattr(
        viewer_service_module.subprocess,
        "Popen",
        lambda command, **kwargs: FakeProcess(command),
    )
    signalled = []
    monkeypatch.setattr(
        viewer_service_module.os,
        "killpg",
        lambda pgid, sig: signalled.append((pgid, sig)),
    )

    service = ViewerService(
        run_service,
        logs_root=tmp_path / "viewer-logs",
        stable_age_seconds=0,
    )
    monkeypatch.setattr(service, "_port_open", lambda: True)
    started = service.start(run_id=run_id)

    stopped = service.stop(started["id"])

    assert stopped["state"] == "stopping"
    assert signalled and signalled[0][0] == 4321
