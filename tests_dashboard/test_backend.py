import importlib
import json


def _load_app(monkeypatch, artifact_root):
    monkeypatch.setenv("ASCENTO_ARTIFACT_ROOT", str(artifact_root))
    import dashboard.app as dashboard_app

    return importlib.reload(dashboard_app)


def test_dashboard_backend_registers_health_and_config_routes(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)

    assert module.app.title == "Ascento Training Monitor"
    paths = {route.path for route in module.app.routes}
    assert "/api/health" in paths
    assert "/api/config" in paths
    assert "/api/runs/{run_id}/summary.json" in paths

    health = module.health()
    assert health["ok"] is True
    assert health["artifact_root"] == str(tmp_path.resolve())
    assert health["config"]["artifact_root"] == str(tmp_path.resolve())
    assert module.configuration()["stale_after_seconds"] > 0


def test_dashboard_starts_before_read_only_artifact_root_exists(monkeypatch, tmp_path):
    artifact_root = tmp_path / "logs" / "rsl_rl"
    assert artifact_root.exists() is False

    module = _load_app(monkeypatch, artifact_root)
    health = module.health()

    assert artifact_root.exists() is False
    assert health["ok"] is True
    assert health["run_count"] == 0
    assert health["problems"] == []
    assert any("does not exist yet" in warning for warning in health["warnings"])
    assert module.runs() == {"runs": []}


def test_run_summary_download_is_json_safe(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_status.json").write_text(
        json.dumps(
            {
                "state": "finished",
                "task": "Ascento-Balance-Flat",
                "stage": "balance",
                "exit_code": 0,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "telemetry.jsonl").write_text(
        '{"completed_steps": 4, "total_steps": 10, "wall_time": 1, '
        '"metrics": {"Train/mean_reward": NaN}}\n',
        encoding="utf-8",
    )

    run_id = module.runs()["runs"][0]["id"]
    response = module.run_summary(run_id)
    payload = json.loads(response.body)

    assert response.headers["content-disposition"] == 'attachment; filename="run-summary.json"'
    assert payload["run_info"]["task"] == "Ascento-Balance-Flat"
    assert payload["training_health"]["non_finite_updates"] == 1
    assert payload["telemetry"]["metrics"]["Train/mean_reward"] is None
