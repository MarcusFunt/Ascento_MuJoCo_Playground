import importlib
import subprocess
from types import SimpleNamespace

from dashboard.provenance import WorkingTreeState
from fastapi.testclient import TestClient


def _load_app(monkeypatch, artifact_root):
    monkeypatch.setenv("ASCENTO_ARTIFACT_ROOT", str(artifact_root))
    import dashboard.app as dashboard_app

    return importlib.reload(dashboard_app)


def test_run_management_routes_are_registered(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    methods = {
        (route.path, method)
        for route in module.app.routes
        for method in getattr(route, "methods", set())
    }

    assert ("/api/runs", "POST") in methods
    assert ("/api/runs/{run_id}", "PATCH") in methods
    assert ("/api/runs/{run_id}/progress", "GET") in methods
    assert ("/api/runs/{run_id}/stop", "POST") in methods
    assert ("/api/runs/compare", "GET") in methods
    assert ("/api/runs/{run_id}/checkpoint-compatibility", "GET") in methods


def test_checkpoint_compatibility_uses_only_stable_checkpoints_and_reports_contract(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "model_100.pt").write_bytes(b"stable checkpoint")
    monkeypatch.setattr(module.RUN_SERVICE, "resolve", lambda _run_id: SimpleNamespace(path=run_dir))
    monkeypatch.setattr(
        module.VIEWER_SERVICE,
        "checkpoints",
        lambda _run_id: {
            "checkpoints": [
                {"relative_path": "model_100.pt", "iteration": 100, "stable": True},
                {"relative_path": "model_200.pt", "iteration": 200, "stable": False},
            ]
        },
    )
    monkeypatch.setattr(module, "validate_checkpoint_for_task", lambda _path, _task: None, raising=False)
    client = TestClient(module.app)

    compatible = client.get(
        "/api/runs/run-1/checkpoint-compatibility?checkpoint=model_100.pt&task=Ascento-Balance-Flat"
    )
    unstable = client.get(
        "/api/runs/run-1/checkpoint-compatibility?checkpoint=model_200.pt&task=Ascento-Balance-Flat"
    )

    assert compatible.status_code == 200
    assert compatible.json()["status"] == "COMPATIBLE"
    assert compatible.json()["compatible"] is True
    assert compatible.json()["checkpoint_sha256"]
    assert unstable.status_code == 404


def test_create_request_preserves_lineage_and_training_args(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    captured = {}
    approved_checkpoint = tmp_path / "run" / "model_7500.pt"
    monkeypatch.setattr(module, "_require_control_session", lambda _request: None)
    monkeypatch.setattr(module, "discover_experiments", lambda *_args: {"programs": [{"id": "locomotion-pilot"}]})
    monkeypatch.setattr(
        module,
        "_resolve_stable_checkpoint",
        lambda *_args: (approved_checkpoint, {"relative_path": "model_7500.pt"}, approved_checkpoint.parent),
    )
    monkeypatch.setattr(module, "validate_checkpoint_for_task", lambda *_args: None)

    def fake_create(payload):
        captured.update(payload)
        return {"id": "abc", "state": "starting"}

    monkeypatch.setattr(module.RUN_SERVICE, "create", fake_create)
    request = module.RunCreateRequest(
        display_name="Recovery validation",
        task="Ascento-Recovery-Flat",
        tags=["recovery", "validation"],
        experiment_id="locomotion-pilot",
        parent_run_id="parent123",
        parent_checkpoint="model_7500.pt",
        training_args=["--agent.max-iterations", "12000"],
    )

    result = module.create_run(request, http_request=None)

    assert result["id"] == "abc"
    assert captured["display_name"] == "Recovery validation"
    assert captured["parent_run_id"] == "parent123"
    assert captured["parent_checkpoint"] == str(approved_checkpoint)
    assert captured["experiment_id"] == "locomotion-pilot"
    assert captured["training_args"][-1] == "12000"


def test_create_request_preserves_explicit_dirty_provenance_override(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    captured = {}
    monkeypatch.setattr(module, "_require_control_session", lambda _request: None)
    monkeypatch.setattr(module.RUN_SERVICE, "create", lambda payload: captured.update(payload) or payload)

    module.create_run(module.RunCreateRequest(display_name="override", allow_dirty_provenance=True), http_request=None)

    assert captured["allow_dirty_provenance"] is True


def test_created_run_is_immediately_discoverable(monkeypatch, tmp_path):
    """The create response must not point at a run the detail route cannot read."""
    module = _load_app(monkeypatch, tmp_path)
    monkeypatch.setattr(module, "_require_control_session", lambda _request: None)

    class FakeProcess:
        pid = 4321

    monkeypatch.setattr(
        "dashboard.run_service.subprocess",
        SimpleNamespace(Popen=lambda *args, **kwargs: FakeProcess(), DEVNULL=subprocess.DEVNULL),
    )
    monkeypatch.setattr(
        "dashboard.run_service.working_tree_state",
        lambda _: WorkingTreeState("abc123", "main", (), ()),
    )
    monkeypatch.setattr(
        "dashboard.run_service.runtime_identity",
        lambda _, requested_device: {"device": requested_device, "runtime_kind": "test"},
    )
    created = module.create_run(
        module.RunCreateRequest(
            display_name="adaptive-balance-horizons",
            task="Ascento-Balance-Flat",
            episode_horizon_s=20,
            training_args=["--device", "cpu"],
        ),
        http_request=None,
    )

    detail = module.run_status(created["id"])

    assert detail["id"] == created["id"]
    assert detail["state"] == "starting"
    assert detail["status"]["launcher_pid"] == 4321


def test_control_session_replaces_the_static_control_header(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    secret = "test-control-token-that-is-at-least-32-chars"
    monkeypatch.setenv("ASCENTO_CONTROL_TOKEN", secret)
    monkeypatch.setattr(module.RUN_SERVICE, "create", lambda payload: {"id": "authorized-run", **payload})
    monkeypatch.setattr(module.DATABASE, "record_event", lambda *_args, **_kwargs: None)
    client = TestClient(module.app)
    headers = {"Origin": "http://testserver"}

    rejected = client.post(
        "/api/runs",
        headers={**headers, "X-Ascento-Control": "1"},
        json={"display_name": "blocked"},
    )
    wrong_token = client.post("/api/control/session", headers=headers, json={"token": "wrong"})
    opened = client.post("/api/control/session", headers=headers, json={"token": secret})
    created = client.post("/api/runs", headers=headers, json={"display_name": "allowed"})
    cross_origin = client.post(
        "/api/runs", headers={"Origin": "https://attacker.invalid"}, json={"display_name": "blocked"}
    )

    assert rejected.status_code == 403
    assert "control token" in rejected.json()["detail"]
    assert wrong_token.status_code == 403
    assert opened.status_code == 200
    assert "httponly" in opened.headers["set-cookie"].lower()
    assert "samesite=strict" in opened.headers["set-cookie"].lower()
    assert created.status_code == 202
    assert cross_origin.status_code == 403
