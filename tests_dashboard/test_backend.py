import importlib


def test_dashboard_backend_imports_and_registers_health_route(monkeypatch, tmp_path):
  monkeypatch.setenv("ASCENTO_ARTIFACT_ROOT", str(tmp_path))
  module = importlib.import_module("dashboard.app")

  assert module.app.title == "Ascento Training Monitor"
  assert any(route.path == "/api/health" for route in module.app.routes)
  assert module.health() == {
    "ok": True,
    "artifact_root": str(tmp_path.resolve()),
  }
