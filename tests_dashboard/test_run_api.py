import importlib
import subprocess
from types import SimpleNamespace


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
    assert ("/api/runs/{run_id}/stop", "POST") in methods
    assert ("/api/runs/compare", "GET") in methods


def test_create_request_preserves_lineage_and_training_args(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    captured = {}

    def fake_create(payload):
        captured.update(payload)
        return {"id": "abc", "state": "starting"}

    monkeypatch.setattr(module.RUN_SERVICE, "create", fake_create)
    request = module.RunCreateRequest(
        display_name="Recovery validation",
        task="Ascento-Recovery-Flat",
        tags=["recovery", "validation"],
        parent_run_id="parent123",
        parent_checkpoint="model_7500.pt",
        training_args=["--agent.max-iterations", "12000"],
    )

    result = module.create_run(request)

    assert result["id"] == "abc"
    assert captured["display_name"] == "Recovery validation"
    assert captured["parent_run_id"] == "parent123"
    assert captured["parent_checkpoint"] == "model_7500.pt"
    assert captured["training_args"][-1] == "12000"


def test_created_run_is_immediately_discoverable(monkeypatch, tmp_path):
    """The create response must not point at a run the detail route cannot read."""
    module = _load_app(monkeypatch, tmp_path)

    class FakeProcess:
        pid = 4321

    monkeypatch.setattr(
        "dashboard.run_service.subprocess",
        SimpleNamespace(Popen=lambda *args, **kwargs: FakeProcess(), DEVNULL=subprocess.DEVNULL),
    )
    created = module.create_run(
        module.RunCreateRequest(
            display_name="adaptive-balance-horizons",
            task="Ascento-Balance-Flat",
            episode_horizon_s=20,
        )
    )

    detail = module.run_status(created["id"])

    assert detail["id"] == created["id"]
    assert detail["state"] == "starting"
    assert detail["status"]["launcher_pid"] == 4321
