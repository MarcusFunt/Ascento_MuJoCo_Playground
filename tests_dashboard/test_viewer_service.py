import json
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace

import dashboard.viewer_service as viewer_service_module
import pytest
from dashboard.health import discover_dashboard_runs
from dashboard.run_service import RunService
from dashboard.viewer_service import ViewerBusyError, ViewerNotFoundError, ViewerService

from ascento_mjlab.viewer.ipc import IntrospectionIPC
from ascento_mjlab.viewer.replay import PolicyReplayRecorder


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


def _patch_popen(monkeypatch, popen) -> None:
    monkeypatch.setattr(
        viewer_service_module,
        "subprocess",
        SimpleNamespace(
            Popen=popen,
            DEVNULL=subprocess.DEVNULL,
            STDOUT=subprocess.STDOUT,
        ),
    )


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

    def kill(self):
        self.exit_code = -9


def test_viewer_service_launches_isolated_worker_and_rejects_duplicate(
    monkeypatch,
    tmp_path,
):
    run_service, run_id, run_dir = _run(tmp_path / "artifacts")
    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess(command)

    _patch_popen(monkeypatch, fake_popen)
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
    assert "--jacobian-hz" in captured["command"]
    assert "--introspection-dir" in captured["command"]
    assert "--viewer-id" in captured["command"]
    assert "--run-id" in captured["command"]
    assert "--capture-dir" in captured["command"]
    assert "ascento_mjlab.viewer.worker" in captured["command"]
    assert captured["kwargs"]["start_new_session"] is True

    with pytest.raises(ViewerBusyError, match="already active"):
        service.start(run_id=run_id)


def test_viewer_introspection_reads_only_its_managed_runtime_directory(monkeypatch, tmp_path):
    run_service, run_id, _ = _run(tmp_path / "artifacts")
    _patch_popen(monkeypatch, lambda command, **kwargs: FakeProcess(command))
    service = ViewerService(run_service, logs_root=tmp_path / "viewer-logs", stable_age_seconds=0)
    monkeypatch.setattr(service, "_port_open", lambda: False)
    started = service.start(run_id=run_id)

    schema_path = tmp_path / "viewer-logs" / started["id"] / "introspection" / "schema.json"
    latest_path = schema_path.with_name("latest.json")
    schema_path.write_text('{"schema_version":1,"actor":{"input_dim":41}}', encoding="utf-8")
    latest_path.write_text('{"sequence_id":7,"critic_value":0.5}', encoding="utf-8")

    assert service.introspection_schema(started["id"]) == {
        "schema_version": 1,
        "actor": {"input_dim": 41},
    }
    assert service.introspection_latest(started["id"]) == {
        "sequence_id": 7,
        "critic_value": 0.5,
    }
    with pytest.raises(ViewerNotFoundError):
        service.introspection_latest("not-the-managed-viewer")


def test_viewer_capture_artifacts_are_scoped_and_manual_requests_are_queued(
    monkeypatch,
    tmp_path,
):
    run_service, run_id, run_dir = _run(tmp_path / "artifacts")
    _patch_popen(monkeypatch, lambda command, **kwargs: FakeProcess(command))
    service = ViewerService(run_service, logs_root=tmp_path / "viewer-logs", stable_age_seconds=0)
    monkeypatch.setattr(service, "_port_open", lambda: False)
    started = service.start(run_id=run_id)
    viewer_id = started["id"]
    capture_dir = run_dir / "viewer_diagnostics" / viewer_id / "events"
    recorder = PolicyReplayRecorder(capture_dir, viewer_id=viewer_id, run_id=run_id)
    recorder.persist(
        event_type="manual",
        frames=[SimpleNamespace(to_dict=lambda: {"sequence_id": 7})],
        trigger_sequence=7,
        trigger_reason="operator request",
        checkpoint="model_100.pt",
        checkpoint_iteration=100,
        schema={"schema_version": 1},
    )

    listed = service.introspection_captures(viewer_id)
    event_id = listed["captures"][0]["event_id"]
    capture = service.introspection_capture(viewer_id, event_id)
    queued = service.request_introspection_capture(viewer_id)

    assert listed["captures"][0]["event_type"] == "manual"
    assert capture["frames"] == [{"sequence_id": 7}]
    assert queued["state"] == "queued"
    assert list((tmp_path / "viewer-logs" / viewer_id / "introspection").glob("capture-request-*.json"))


def test_introspection_explanation_is_validated_and_spooled_for_viewer(monkeypatch, tmp_path):
    run_service, run_id, _ = _run(tmp_path / "artifacts")
    _patch_popen(monkeypatch, lambda command, **kwargs: FakeProcess(command))
    service = ViewerService(run_service, logs_root=tmp_path / "viewer-logs", stable_age_seconds=0)
    monkeypatch.setattr(service, "_port_open", lambda: False)
    started = service.start(run_id=run_id)
    viewer_id = started["id"]
    introspection_dir = tmp_path / "viewer-logs" / viewer_id / "introspection"
    (introspection_dir / "schema.json").write_text(
        json.dumps({"checkpoint": "model_100.pt", "actor": {"input_dim": 2}}),
        encoding="utf-8",
    )
    (introspection_dir / "latest.json").write_text(
        json.dumps({"actor_output": [0.1, -0.2]}), encoding="utf-8"
    )

    queued = service.request_introspection_explanation(
        viewer_id,
        {
            "checkpoint": "model_100.pt",
            "input_raw": [0.2, -0.4],
            "action_index": 1,
            "action_name": "right_knee",
            "baseline_kind": "normalizer_mean",
            "n_steps": 32,
            "live_sequence": 9,
        },
    )
    request = IntrospectionIPC(introspection_dir).consume_explanation_request()

    assert queued["state"] == "queued"
    assert queued["explanation_id"] == request["explanation_id"]
    assert request["input_raw"] == [0.2, -0.4]
    assert service.introspection_explanation(viewer_id, queued["explanation_id"])["status"] == "pending"


def test_runtime_status_updates_loaded_checkpoint_and_lag(monkeypatch, tmp_path):
    run_service, run_id, run_dir = _run(tmp_path / "artifacts")
    (run_dir / "model_130.pt").write_bytes(b"new-checkpoint")

    _patch_popen(
        monkeypatch,
        lambda command, **kwargs: FakeProcess(command),
    )
    service = ViewerService(
        run_service,
        logs_root=tmp_path / "viewer-logs",
        stable_age_seconds=0,
    )
    port_ready = {"value": False}
    monkeypatch.setattr(service, "_port_open", lambda: port_ready["value"])
    monkeypatch.setattr(
        run_service,
        "progress",
        lambda _run_id: {"telemetry": {"iteration": 140}},
    )

    started = service.start(run_id=run_id, checkpoint="model_100.pt")
    port_ready["value"] = True
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


def test_start_rejects_an_already_occupied_viewer_port(monkeypatch, tmp_path):
    run_service, run_id, _ = _run(tmp_path / "artifacts")
    service = ViewerService(
        run_service,
        logs_root=tmp_path / "viewer-logs",
        stable_age_seconds=0,
    )
    monkeypatch.setattr(service, "_port_open", lambda: True)

    with pytest.raises(ViewerBusyError, match="port 8081 is already in use"):
        service.start(run_id=run_id)


def test_architecture_reports_the_checkpoint_actor_and_critic_shapes(tmp_path):
    import torch

    run_service, run_id, run_dir = _run(tmp_path / "artifacts")
    torch.save(
        {
            "actor_state_dict": {
                "mlp.0.weight": torch.ones(256, 41),
                "mlp.0.bias": torch.full((256,), 0.25),
                "mlp.2.weight": torch.zeros(256, 256),
                "mlp.2.bias": torch.zeros(256),
                "mlp.4.weight": torch.zeros(256, 256),
                "mlp.4.bias": torch.zeros(256),
                "mlp.6.weight": torch.zeros(6, 256),
                "mlp.6.bias": torch.zeros(6),
                "distribution.std_param": torch.zeros(6),
            },
            "critic_state_dict": {
                "mlp.0.weight": torch.zeros(256, 50),
                "mlp.0.bias": torch.zeros(256),
                "mlp.2.weight": torch.zeros(256, 256),
                "mlp.2.bias": torch.zeros(256),
                "mlp.4.weight": torch.zeros(256, 256),
                "mlp.4.bias": torch.zeros(256),
                "mlp.6.weight": torch.zeros(1, 256),
                "mlp.6.bias": torch.zeros(1),
            },
            "iter": 100,
        },
        run_dir / "model_100.pt",
    )
    params = run_dir / "params"
    params.mkdir()
    (params / "agent.yaml").write_text(
        "actor:\n  activation: elu\ncritic:\n  activation: elu\n",
        encoding="utf-8",
    )
    service = ViewerService(run_service, logs_root=tmp_path / "viewer-logs", stable_age_seconds=0)

    architecture = service.architecture(run_id)

    assert architecture["available"] is True
    assert architecture["checkpoint"] == "model_100.pt"
    assert architecture["iteration"] == 100
    assert architecture["actor"]["layers"] == [41, 256, 256, 256, 6]
    assert architecture["actor"]["activation"] == "ELU"
    assert architecture["actor"]["distribution"] == "Gaussian"
    assert architecture["actor"]["parameter_count"] > 140_000
    assert len(architecture["actor"]["linear_layers"]) == 4
    assert architecture["actor"]["linear_layers"][0]["input_size"] == 41
    assert architecture["actor"]["linear_layers"][0]["output_size"] == 256
    assert architecture["actor"]["linear_layers"][0]["weight_rms"] == pytest.approx(1.0)
    assert architecture["actor"]["linear_layers"][0]["bias_rms"] == pytest.approx(0.25)
    assert architecture["critic"]["layers"] == [50, 256, 256, 256, 1]
    assert architecture["critic"]["activation"] == "ELU"
    assert len(architecture["critic"]["linear_layers"]) == 4


def test_stop_signals_only_viewer_process_group(monkeypatch, tmp_path):
    run_service, run_id, _ = _run(tmp_path / "artifacts")
    _patch_popen(
        monkeypatch,
        lambda command, **kwargs: FakeProcess(command),
    )
    signalled = []
    monkeypatch.setattr(
        viewer_service_module.os,
        "killpg",
        lambda pgid, sig: signalled.append((pgid, sig)),
        raising=False,
    )

    service = ViewerService(
        run_service,
        logs_root=tmp_path / "viewer-logs",
        stable_age_seconds=0,
    )
    port_ready = {"value": False}
    monkeypatch.setattr(service, "_port_open", lambda: port_ready["value"])
    started = service.start(run_id=run_id)
    port_ready["value"] = True

    stopped = service.stop(started["id"])

    assert stopped["state"] == "stopping"
    assert signalled and signalled[0] == (4321, signal.SIGINT)


def test_stop_escalates_from_interrupt_to_term_and_kill(monkeypatch, tmp_path):
    run_service, run_id, _ = _run(tmp_path / "artifacts")
    _patch_popen(monkeypatch, lambda command, **kwargs: FakeProcess(command))
    signals = []
    monkeypatch.setattr(
        viewer_service_module.os,
        "killpg",
        lambda pgid, sig: signals.append((pgid, sig)),
        raising=False,
    )

    class ImmediateThread:
        def __init__(self, *, target, args, **kwargs):
            del kwargs
            self.target = target
            self.args = args

        def start(self):
            self.target(*self.args)

    monkeypatch.setattr(viewer_service_module.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(viewer_service_module.time, "sleep", lambda _seconds: None)

    service = ViewerService(
        run_service,
        logs_root=tmp_path / "viewer-logs",
        stable_age_seconds=0,
        stop_grace_seconds=0,
        stop_term_seconds=0,
    )
    monkeypatch.setattr(service, "_port_open", lambda: False)
    started = service.start(run_id=run_id)
    service.stop(started["id"])

    assert signals == [
        (4321, signal.SIGINT),
        (4321, signal.SIGTERM),
        (4321, viewer_service_module._FORCE_KILL_SIGNAL),
    ]
