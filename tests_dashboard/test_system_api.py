import importlib

import pytest
from fastapi import HTTPException

from dashboard.supervisor_client import SupervisorRejected, SupervisorUnavailable


def _load_app(monkeypatch, artifact_root):
    monkeypatch.setenv("ASCENTO_ARTIFACT_ROOT", str(artifact_root))
    import dashboard.app as dashboard_app

    return importlib.reload(dashboard_app)


def test_system_status_surfaces_supervisor_repository_and_tailnet(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)

    class FakeSupervisor:
        def status(self, *, refresh=False):
            assert refresh is True
            return {
                "repository": {"branch": "main", "behind_by": 2, "update_available": True},
                "can_update": True,
                "update_blockers": [],
                "active_runs": [],
                "update": {"status": "idle"},
                "tailscale": {
                    "enabled": True,
                    "connected": True,
                    "dns_name": "ascento-dashboard.example.ts.net",
                },
            }

        def update(self):
            return {"status": "running"}

    module.SUPERVISOR = FakeSupervisor()
    result = module.system_status(refresh=True)

    assert result["connected"] is True
    assert result["repository"]["behind_by"] == 2
    assert result["tailscale"]["connected"] is True
    assert result["can_update"] is True
    assert module.system_update()["status"] == "running"


def test_system_status_degrades_cleanly_when_supervisor_is_missing(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)

    class MissingSupervisor:
        def status(self, *, refresh=False):
            raise SupervisorUnavailable("socket missing")

    module.SUPERVISOR = MissingSupervisor()
    result = module.system_status()

    assert result["connected"] is False
    assert result["can_update"] is False
    assert result["update"]["status"] == "unavailable"
    assert "socket missing" in result["error"]


def test_system_update_maps_supervisor_rejection_to_conflict(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)

    class BlockedSupervisor:
        def update(self):
            raise SupervisorRejected("one or more training runs are active")

    module.SUPERVISOR = BlockedSupervisor()
    with pytest.raises(HTTPException) as error:
        module.system_update()

    assert error.value.status_code == 409
    assert "training runs" in error.value.detail


def test_system_update_maps_missing_supervisor_to_service_unavailable(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)

    class MissingSupervisor:
        def update(self):
            raise SupervisorUnavailable("not installed")

    module.SUPERVISOR = MissingSupervisor()
    with pytest.raises(HTTPException) as error:
        module.system_update()

    assert error.value.status_code == 503
