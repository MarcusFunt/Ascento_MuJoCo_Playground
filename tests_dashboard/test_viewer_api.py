import importlib

import pytest
from fastapi import HTTPException
from starlette.requests import Request


def _load_app(monkeypatch, artifact_root):
    monkeypatch.setenv("ASCENTO_ARTIFACT_ROOT", str(artifact_root))
    import dashboard.app as dashboard_app

    return importlib.reload(dashboard_app)


def _request(*, control: bool) -> Request:
    headers = [(b"x-ascento-control", b"1")] if control else []
    return Request({"type": "http", "headers": headers})


def test_viewer_routes_are_registered(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    methods = {
        (route.path, method)
        for route in module.app.routes
        for method in getattr(route, "methods", set())
    }

    assert ("/api/runs/{run_id}/checkpoints", "GET") in methods
    assert ("/api/viewers", "GET") in methods
    assert ("/api/viewers", "POST") in methods
    assert ("/api/viewers/{viewer_id}", "GET") in methods
    assert ("/api/viewers/{viewer_id}", "DELETE") in methods
    assert ("/api/viewers/{viewer_id}/logs", "GET") in methods


def test_viewer_start_requires_control_header(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)

    with pytest.raises(HTTPException) as exc:
        module.start_viewer(
            module.ViewerCreateRequest(run_id="run"),
            _request(control=False),
        )

    assert exc.value.status_code == 403


def test_viewer_start_preserves_checkpoint_follow_and_device(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    captured = {}
    monkeypatch.setattr(
        module.VIEWER_SERVICE,
        "start",
        lambda **kwargs: captured.update(kwargs) or {"id": "viewer", "state": "starting"},
    )

    result = module.start_viewer(
        module.ViewerCreateRequest(
            run_id="run",
            checkpoint="nested/model_500.pt",
            follow=True,
            device="cuda:0",
        ),
        _request(control=True),
    )

    assert result["id"] == "viewer"
    assert captured == {
        "run_id": "run",
        "checkpoint": "nested/model_500.pt",
        "follow": True,
        "device": "cuda:0",
    }
