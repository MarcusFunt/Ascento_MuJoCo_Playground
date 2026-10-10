import json

from dashboard.experiments import discover_experiments


def test_experiment_adapter_keeps_declared_state_and_explicit_links_separate(tmp_path):
    root = tmp_path / "experiments"
    root.mkdir()
    (root / "program.json").write_text(
        json.dumps({
            "schema_version": 1,
            "plan_id": "AG-PLAN-1",
            "status": "in_progress",
            "baseline": {"checkpoint": "model_000.pt"},
            "exp_retention_01": {
                "status": "complete_failed_retention",
                "conclusion": "retention failed",
                "run_attempts": [{"run_id": "r1", "outcome": "failed"}],
            },
        }),
        encoding="utf-8",
    )
    runs = [
        {"id": "r1", "name": "linked", "metadata": {"experiment_id": "AG-PLAN-1", "tags": []}},
        {"id": "r2", "name": "tagged", "metadata": {"tags": ["AG-PLAN-1"]}},
        {"id": "r3", "name": "similar name", "metadata": {"tags": [], "display_name": "AG-PLAN-1 continuation"}},
    ]

    result = discover_experiments(root, runs)

    program = result["programs"][0]
    assert program["status"] == "in_progress"
    assert program["arms"][0]["declared_status"] == "complete_failed_retention"
    assert program["arms"][0]["observed_outcome"] == "retention failed"
    assert [item["link_source"] for item in program["linked_runs"]] == ["experiment_id", "explicit_tag"]
    assert [run["id"] for run in program["unlinked_runs"]] == ["r3"]


def test_experiment_index_does_not_embed_run_telemetry(tmp_path):
    root = tmp_path / "experiments"
    root.mkdir()
    (root / "program.json").write_text(
        json.dumps({"schema_version": 1, "plan_id": "AG-PLAN-1", "status": "in_progress"}),
        encoding="utf-8",
    )
    huge_payload = "x" * 200_000
    runs = [
        {"id": "linked", "name": "linked", "state": "finished",
         "metadata": {"experiment_id": "AG-PLAN-1"}, "telemetry": huge_payload,
         "training_health": huge_payload, "unused": huge_payload},
        {"id": "unlinked", "name": "unlinked", "state": "failed",
         "metadata": {"tags": []}, "telemetry": huge_payload},
    ]
    result = discover_experiments(root, runs)
    assert result["programs"][0]["linked_runs"][0]["run"]["id"] == "linked"
    assert result["programs"][0]["unlinked_runs"][0]["id"] == "unlinked"
    assert len(json.dumps(result)) < 5000
    assert "telemetry" not in json.dumps(result)
