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
    assert ("/api/viewers/{viewer_id}/introspection/schema", "GET") in methods
    assert ("/api/viewers/{viewer_id}/introspection/latest", "GET") in methods
    assert ("/api/viewers/{viewer_id}/introspection/captures", "GET") in methods
    assert ("/api/viewers/{viewer_id}/introspection/captures", "POST") in methods
    assert ("/api/viewers/{viewer_id}/introspection/captures/{event_id}", "GET") in methods
    assert ("/api/viewers/{viewer_id}/introspection/explanations", "POST") in methods
    assert ("/api/viewers/{viewer_id}/introspection/explanations/{explanation_id}", "GET") in methods
    assert ("/api/viewers/{viewer_id}/captures", "GET") in methods
    assert ("/api/viewers/{viewer_id}/captures", "POST") in methods
    assert ("/api/viewers/{viewer_id}/captures/{event_id}", "GET") in methods
    assert ("/api/viewers/{viewer_id}/explanations", "POST") in methods
    assert ("/api/viewers/{viewer_id}/explanations/{explanation_id}", "GET") in methods
    assert ("/api/viewers/{viewer_id}/introspection/stream", "GET") in methods


def test_viewer_start_requires_control_header(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)

    with pytest.raises(HTTPException) as exc:
        module.start_viewer(
            module.ViewerCreateRequest(run_id="run"),
            _request(control=False),
        )

    assert exc.value.status_code == 403


def test_manual_capture_route_requires_control_header(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)

    with pytest.raises(HTTPException) as exc:
        module.request_viewer_introspection_capture("viewer", _request(control=False))

    assert exc.value.status_code == 403


def test_policy_explanation_route_requires_control_header(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    payload = module.ViewerExplanationRequest(
        checkpoint="model_100.pt",
        input_raw=[0.0],
        action_index=0,
        action_name="hip",
        baseline_kind="normalizer_mean",
    )

    with pytest.raises(HTTPException) as exc:
        module.request_viewer_introspection_explanation("viewer", payload, _request(control=False))

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
        "jacobian_hz": 2.0,
    }
