import importlib
import json
import threading
import time
from types import SimpleNamespace

from dashboard.supervisor_client import SupervisorUnavailable
from fastapi.testclient import TestClient


def _load_app(monkeypatch, artifact_root):
    monkeypatch.setenv("ASCENTO_ARTIFACT_ROOT", str(artifact_root))
    import dashboard.app as dashboard_app

    return importlib.reload(dashboard_app)


def test_dashboard_backend_registers_health_and_config_routes(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)

    assert module.app.title == "Ascento Control"
    paths = {route.path for route in module.app.routes}
    assert "/api/health" in paths
    assert "/api/health/live" in paths
    assert "/api/health/ready" in paths
    assert "/api/system/components" in paths
    assert "/api/activity" in paths
    assert "/api/assessments" in paths
    assert "/api/evaluation-suites" in paths
    assert "/api/evaluations" in paths
    assert "/api/experiments" in paths
    assert "/api/runs/{run_id}/checkpoint-evidence" in paths
    assert "/api/runtime/preflight" in paths
    assert "/api/runtime/identity" in paths
    assert "/api/config" in paths
    assert "/api/runs/{run_id}/summary.json" in paths
    assert "/api/overview" in paths
    assert "/api/runs/index" in paths
    assert "/api/tasks" in paths
    assert "/api/runs/{run_id}/curriculum" in paths

    health = module.health()
    assert health["ready"] is True
    assert health["status"] in {"healthy", "degraded"}
    assert health["artifact_root"] == str(tmp_path.resolve())
    assert health["config"]["artifact_root"] == str(tmp_path.resolve())
    assert module.configuration()["stale_after_seconds"] > 0


def test_dashboard_starts_before_read_only_artifact_root_exists(monkeypatch, tmp_path):
    artifact_root = tmp_path / "logs" / "rsl_rl"
    assert artifact_root.exists() is False

    module = _load_app(monkeypatch, artifact_root)
    health = module.health()

    assert artifact_root.exists() is False
    assert health["ready"] is True
    assert health["run_count"] == 0
    assert health["problems"] == []
    assert any("does not exist yet" in warning for warning in health["warnings"])
    assert module.runs() == {"runs": []}


def test_health_does_not_trigger_an_expensive_run_summary_scan(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    monkeypatch.setattr(
        module,
        "_annotated_summaries",
        lambda: (_ for _ in ()).throw(AssertionError("health should not scan runs")),
    )

    health = module.health()

    assert health["ready"] is True
    assert health["run_count"] is None


def test_health_separates_liveness_readiness_and_degraded_dependencies(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    monkeypatch.setattr(
        module.DATABASE,
        "status",
        lambda: {
            "enabled": True,
            "available": False,
            "backend": "postgresql",
            "error": "connection refused",
            "source_conflicts": None,
            "last_successful_sync_at": None,
        },
    )

    class MissingSupervisor:
        def status(self, *, refresh=False):
            raise SupervisorUnavailable("socket missing")

    module.SUPERVISOR = MissingSupervisor()
    client = TestClient(module.app)

    live = client.get("/api/health/live")
    health = client.get("/api/health")
    ready = client.get("/api/health/ready")

    assert live.status_code == 200
    assert live.json()["live"] is True
    assert health.status_code == 200
    assert health.json()["status"] == "degraded"
    assert health.json()["ok"] is False
    assert health.json()["ready"] is True
    assert health.json()["components"]["database"]["status"] == "degraded"
    assert health.json()["components"]["supervisor"]["status"] == "unavailable"
    assert ready.status_code == 200


def test_readiness_fails_when_required_artifacts_are_unavailable(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    monkeypatch.setattr(module, "_artifact_health", lambda: ["artifact root is not readable"])

    response = TestClient(module.app).get("/api/health/ready")

    assert response.status_code == 503
    assert response.json()["ready"] is False
    assert response.json()["status"] == "unavailable"


def test_missing_supervisor_does_not_claim_idle_or_tailscale_disconnected(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)

    class MissingSupervisor:
        def status(self, *, refresh=False):
            raise SupervisorUnavailable("socket missing")

    module.SUPERVISOR = MissingSupervisor()
    response = TestClient(module.app).get("/api/system")

    assert response.status_code == 200
    payload = response.json()
    assert payload["connected"] is False
    assert payload["active_runs"] is None
    assert payload["tailscale"] == {
        "status": "unknown",
        "enabled": None,
        "connected": None,
        "error": "cannot verify without the host supervisor",
    }
    assert "host supervisor is unavailable" in payload["update_blockers"]


def test_system_components_reports_optional_and_unavailable_services(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    monkeypatch.setattr(module.DATABASE, "status", lambda: {"enabled": False, "available": False})

    class MissingSupervisor:
        def status(self, *, refresh=False):
            raise SupervisorUnavailable("socket missing")

    module.SUPERVISOR = MissingSupervisor()
    payload = TestClient(module.app).get("/api/system/components").json()

    assert payload["components"]["api"]["status"] == "healthy"
    assert payload["components"]["database"]["status"] == "optional"
    assert payload["components"]["supervisor"]["status"] == "unavailable"
    assert payload["components"]["tailscale"]["status"] == "unknown"


def test_runtime_preflight_is_read_only_and_uses_managed_runtime_policy(monkeypatch, tmp_path):
    artifact_root = tmp_path / "artifacts"
    module = _load_app(monkeypatch, artifact_root)
    observed = {}

    def runtime_identity(root, *, requested_device):
        observed["root"] = root
        observed["device"] = requested_device
        return {
            "runtime_kind": "canonical-wsl-checkout",
            "source_checkout_root": "/root/Ascento_MuJoCo_Playground",
            "execution_root": "/root/Ascento_MuJoCo_Playground",
            "source_commit": "abc123",
            "source_branch": "main",
            "source_dirty": False,
            "compute_backend": "cu128",
            "device": "cuda:0",
        }

    monkeypatch.setattr(module, "runtime_identity", runtime_identity)
    response = TestClient(module.app).get(
        "/api/runtime/preflight?task=Ascento-Locomotion-Flat&device=cuda:0"
    )

    assert response.status_code == 200
    assert response.json()["allowed"] is True
    assert response.json()["runtime"]["device"] == "cuda:0"
    assert observed == {"root": module.CONFIG.repo_root, "device": "cuda:0"}
    assert not artifact_root.exists()


def test_runtime_preflight_explains_policy_blockers(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    monkeypatch.setattr(
        module,
        "runtime_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            module.RuntimePolicyError("canonical checkout is dirty")
        ),
    )

    response = TestClient(module.app).get("/api/runtime/preflight?task=Ascento-Balance-Flat")

    assert response.status_code == 200
    assert response.json()["allowed"] is False
    assert response.json()["blockers"] == ["canonical checkout is dirty"]


def test_runtime_identity_keeps_checkout_image_and_run_revisions_separate(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    monkeypatch.setattr(
        module,
        "runtime_revision_report",
        lambda: {
            "checkout": {"commit": "source-commit"},
            "packaged_image": {"commit": "image-commit"},
            "reported_api": {"commit": "api-commit"},
            "comparisons": {"checkout_matches_image": False},
        },
    )
    monkeypatch.setattr(
        module,
        "_indexed_summaries",
        lambda: [
            {
                "id": "run-1",
                "name": "latest run",
                "state": "finished",
                "repository_version": {"run_commit": "run-commit", "status": "outdated"},
            }
        ],
    )

    response = TestClient(module.app).get("/api/runtime/identity")

    assert response.status_code == 200
    payload = response.json()
    assert payload["checkout"]["commit"] == "source-commit"
    assert payload["packaged_image"]["commit"] == "image-commit"
    assert payload["reported_api"]["commit"] == "api-commit"
    assert payload["latest_indexed_run"]["commit"] == "run-commit"
    assert payload["comparisons"]["checkout_matches_image"] is False


def test_activity_reports_unknown_host_processes_when_supervisor_is_missing(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)

    class MissingSupervisor:
        def status(self, *, refresh=False):
            raise SupervisorUnavailable("socket missing")

    module.SUPERVISOR = MissingSupervisor()
    monkeypatch.setattr(module, "_annotated_summaries", lambda: [])
    monkeypatch.setattr(module.VIEWER_SERVICE, "list", lambda: {"viewers": []})
    monkeypatch.setattr(module, "gpu_snapshot", lambda: {"available": False, "gpus": []})

    response = TestClient(module.app).get("/api/activity")

    assert response.status_code == 200
    payload = response.json()
    assert payload["trainer"]["status"] == "unknown"
    assert payload["trainer"]["verified"] is False
    assert payload["trainer"]["active_runs"] is None
    assert payload["evaluator"]["status"] == "unknown"
    assert payload["viewer"]["status"] == "idle"
    assert payload["viewer"]["verified"] is True
    assert payload["render"]["status"] == "unknown"


def test_index_cache_uses_database_snapshot_before_artifact_refresh(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    cached_rows = [{"id": "db-run", "state": "running"}]
    monkeypatch.setattr(module.DATABASE, "list_runs", lambda: cached_rows)
    monkeypatch.setattr(
        module,
        "_build_indexed_summaries",
        lambda: (_ for _ in ()).throw(AssertionError("cold request should use database cache")),
    )

    assert module._indexed_summaries() == cached_rows
    assert module._INDEX_CACHE_SOURCE == "database"
    assert module._INDEX_REFRESHING is True


def test_index_snapshot_serves_stale_data_while_refresh_runs(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    old_rows = [{"id": "old"}]
    refreshed = threading.Event()
    module._INDEX_CACHE = (time.monotonic() - module._INDEX_CACHE_TTL_S - 1, old_rows)

    def rebuild():
        refreshed.set()
        return [{"id": "new"}]

    monkeypatch.setattr(module, "_build_indexed_summaries", rebuild)

    assert module._indexed_summaries() == old_rows
    assert refreshed.wait(timeout=2)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        with module._INDEX_CACHE_LOCK:
            if module._INDEX_CACHE and module._INDEX_CACHE[1] == [{"id": "new"}]:
                break
        time.sleep(0.01)
    assert module._INDEX_CACHE[1] == [{"id": "new"}]


def test_activity_snapshot_is_reused_until_its_cache_expires(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    calls = []
    def build_snapshot():
        calls.append(True)
        return {"checked_at": 123.0, "trainer": {"status": "idle"}}
    monkeypatch.setattr(module, "_build_activity_snapshot", build_snapshot)
    module._ACTIVITY_CACHE = None
    first = module.activity_snapshot()
    second = module.activity_snapshot()
    assert first == second
    assert len(calls) == 1


def test_assessments_api_is_read_only_and_returns_deterministic_findings(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    monkeypatch.setattr(module, "_indexed_summaries", lambda: [])
    monkeypatch.setattr(module, "_system_components", lambda **_kwargs: {})
    monkeypatch.setattr(module, "activity_snapshot", lambda: {})
    monkeypatch.setattr(module, "runtime_revision_report", lambda: {})
    monkeypatch.setattr(module.DATABASE, "status", lambda: {"enabled": False, "available": False})
    monkeypatch.setattr(module, "discover_evaluations", lambda *_args, **_kwargs: [])

    response = TestClient(module.app).get("/api/assessments")

    assert response.status_code == 200
    assert response.json()["read_only"] is True
    assert response.json()["assessments"] == []


def test_assessment_response_reuses_activity_and_evaluation_snapshots(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    counts = {"activity": 0, "evaluations": 0}
    monkeypatch.setattr(module, "_indexed_summaries", lambda: [])
    monkeypatch.setattr(module, "_system_components", lambda **_kwargs: {})
    monkeypatch.setattr(module, "activity_snapshot", lambda: counts.__setitem__("activity", counts["activity"] + 1) or {})
    monkeypatch.setattr(module, "runtime_revision_report", lambda: {})
    monkeypatch.setattr(module.DATABASE, "status", lambda: {"enabled": False, "available": False})
    cached_rows = [{"id": "cached-evaluation"}]
    module._EVALUATION_CACHE = (time.monotonic(), cached_rows)
    def evaluations():
        counts["evaluations"] += 1
        return list(cached_rows)
    monkeypatch.setattr(module, "_evaluation_summaries", evaluations)
    client = TestClient(module.app)
    first = client.get("/api/assessments")
    second = client.get("/api/assessments")
    assert first.status_code == second.status_code == 200
    assert counts == {"activity": 1, "evaluations": 1}


def test_cold_evaluation_snapshot_refreshes_outside_the_request(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    started = threading.Event()
    release = threading.Event()
    rows = [{"id": "background-evaluation"}]

    def blocked_discovery(*_args, **_kwargs):
        started.set()
        assert release.wait(timeout=3)
        return rows

    monkeypatch.setattr(module, "discover_evaluations", blocked_discovery)
    response = TestClient(module.app).get("/api/evaluations")

    assert response.status_code == 200
    assert response.json()["total"] == 0
    assert started.wait(timeout=1)
    release.set()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if module._evaluation_summaries() == rows:
            break
        time.sleep(0.01)
    assert module._evaluation_summaries() == rows


def test_evaluation_registry_routes_are_read_only_and_preserve_incomplete_status(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    evaluation_root = tmp_path / "evaluations"
    partial = evaluation_root / "partial"
    partial.mkdir(parents=True)
    (partial / "suite.json").write_text(
        '{"suite_id":"dev_v1","task":"Ascento-Balance-Flat","gates":[{"hard":true}]}',
        encoding="utf-8",
    )
    suite_root = tmp_path / "suites"
    suite_root.mkdir()
    (suite_root / "dev_v1.toml").write_text(
        'suite_id = "dev_v1"\ntask = "Ascento-Balance-Flat"\n', encoding="utf-8"
    )
    monkeypatch.setattr(module, "EVALUATION_ROOT", evaluation_root)
    monkeypatch.setattr(module, "EVALUATION_SUITE_ROOT", suite_root)
    module._EVALUATION_CACHE = (
        time.monotonic(),
        module.discover_evaluations(evaluation_root, limit=5000),
    )
    client = TestClient(module.app)

    suite_response = client.get("/api/evaluation-suites")
    list_response = client.get("/api/evaluations?status=INCOMPLETE")
    detail_response = client.get("/api/evaluations/partial")
    gates_response = client.get("/api/evaluations/partial/gates")

    assert suite_response.json()["evaluation_launch_available"] is False
    assert suite_response.json()["suites"][0]["suite_id"] == "dev_v1"
    assert list_response.json()["total"] == 1
    assert list_response.json()["evaluations"][0]["status"] == "INCOMPLETE"
    assert detail_response.json()["evaluation"]["status"] == "INCOMPLETE"
    assert gates_response.json()["status"] == "INCOMPLETE"


def test_evaluation_registry_paginates_and_supports_etag_revalidation(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    evaluation_root = tmp_path / "evaluations"
    partial = evaluation_root / "partial"
    partial.mkdir(parents=True)
    (partial / "suite.json").write_text(
        '{"suite_id":"dev_v1","task":"Ascento-Balance-Flat","gates":[{"hard":true}]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "EVALUATION_ROOT", evaluation_root)
    module._EVALUATION_CACHE = (
        time.monotonic(),
        module.discover_evaluations(evaluation_root, limit=5000),
    )
    client = TestClient(module.app)

    first_page = client.get("/api/evaluations?limit=1&offset=0")
    cached_page = client.get(
        "/api/evaluations?limit=1&offset=0",
        headers={"If-None-Match": first_page.headers["etag"]},
    )
    second_page = client.get("/api/evaluations?limit=1&offset=1")

    assert first_page.status_code == 200
    assert first_page.json()["total"] == 1
    assert len(first_page.json()["evaluations"]) == 1
    assert cached_page.status_code == 304
    assert second_page.status_code == 200
    assert second_page.json()["total"] == 1
    assert second_page.json()["evaluations"] == []
    assert second_page.headers["etag"] != first_page.headers["etag"]


def test_checkpoint_evidence_api_hashes_the_selected_stable_file(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "model_200.pt").write_bytes(b"policy")
    monkeypatch.setattr(module.RUN_SERVICE, "resolve", lambda _run_id: SimpleNamespace(path=run_dir))
    monkeypatch.setattr(
        module.VIEWER_SERVICE,
        "checkpoints",
        lambda _run_id: {"checkpoints": [{"relative_path": "model_200.pt", "iteration": 200, "stable": True}]},
    )

    response = TestClient(module.app).get("/api/runs/run-1/checkpoint-evidence")

    assert response.status_code == 200
    assert response.json()["relative_path"] == "model_200.pt"
    assert response.json()["selection_status"] == "not_recorded"


def test_experiment_registry_uses_explicit_run_metadata_only(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    experiment_root = tmp_path / "experiments"
    experiment_root.mkdir()
    (experiment_root / "program.json").write_text(
        json.dumps({"plan_id": "program-1", "status": "declared"}), encoding="utf-8"
    )
    monkeypatch.setattr(module, "EXPERIMENT_ROOT", experiment_root)
    monkeypatch.setattr(
        module,
        "_annotated_summaries",
        lambda: [
            {"id": "linked", "metadata": {"experiment_id": "program-1"}},
            {"id": "unlinked", "metadata": {"display_name": "program-1-like-name"}},
        ],
    )

    response = TestClient(module.app).get("/api/experiments")

    assert response.status_code == 200
    program = response.json()["programs"][0]
    assert [item["run"]["id"] for item in program["linked_runs"]] == ["linked"]
    assert [item["id"] for item in program["unlinked_runs"]] == ["unlinked"]


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


def test_sampled_telemetry_reports_canonical_coverage(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    monkeypatch.setattr(module, "_run", lambda _run_id: SimpleNamespace(path=tmp_path))
    monkeypatch.setattr(
        module,
        "load_training_records",
        lambda *_args, **_kwargs: [
            {"completed_steps": 1, "metrics": {"Train/mean_reward": 1.0}},
            {"completed_steps": 2, "metrics": {"Train/mean_reward": 2.0}},
            {"completed_steps": 3, "metrics": {"Train/mean_reward": 3.0}},
        ],
    )

    payload = module.telemetry("run", max_points=2)

    assert len(payload["records"]) == 2
    assert payload["coverage"]["reward"] == {"present": 2, "missing": 0}
    assert payload["coverage"]["ppo_loss"] == {"present": 0, "missing": 2}
