import asyncio
import importlib
import warnings
from types import SimpleNamespace

import dashboard.app as dashboard_app
import pytest


def _enter_lifespan(app):
    async def enter():
        async with app.router.lifespan_context(app):
            pass

    asyncio.run(enter())


def test_dashboard_lifespan_initializes_and_cleans_up_services_once(monkeypatch):
    module = importlib.reload(dashboard_app)
    events = []
    fallback_warning = "Dashboard database unavailable; using filesystem fallback: connection refused"
    warnings_list = []
    database = SimpleNamespace(
        enabled=True,
        error="connection refused",
        initialize=lambda: events.append("initialize"),
        dispose=lambda: events.append("dispose"),
    )

    monkeypatch.setattr(module, "DATABASE", database)
    monkeypatch.setattr(module.VIEWER_SERVICE, "stop_all", lambda: events.append("stop"))
    monkeypatch.setattr(module, "STARTUP_WARNINGS", warnings_list)

    _enter_lifespan(module.app)
    _enter_lifespan(module.app)

    assert events == ["initialize", "stop", "dispose"] * 2
    assert warnings_list == [fallback_warning]


def test_dashboard_lifespan_cleans_up_if_database_initialization_raises(monkeypatch):
    module = importlib.reload(dashboard_app)
    events = []

    def fail_initialize():
        events.append("initialize")
        raise RuntimeError("database initialization failed")

    database = SimpleNamespace(
        initialize=fail_initialize,
        dispose=lambda: events.append("dispose"),
    )
    monkeypatch.setattr(module, "DATABASE", database)
    monkeypatch.setattr(module.VIEWER_SERVICE, "stop_all", lambda: events.append("stop"))

    with pytest.raises(RuntimeError, match="database initialization failed"):
        _enter_lifespan(module.app)

    assert events == ["initialize", "stop", "dispose"]


def test_dashboard_app_does_not_register_deprecated_event_hooks():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        module = importlib.reload(dashboard_app)

    assert not [warning for warning in caught if "on_event is deprecated" in str(warning.message)]
    assert module.app.router.on_startup == []
    assert module.app.router.on_shutdown == []
