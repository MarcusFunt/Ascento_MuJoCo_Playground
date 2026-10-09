from dashboard.assessments import build_assessments


def test_assessments_prioritize_integrity_and_link_exact_evidence():
    assessments = build_assessments(
        checked_at=1000,
        components={"supervisor": {"status": "unavailable"}},
        database={"enabled": True, "available": False},
        runtime={"comparisons": {"checkout_matches_image": False}},
        activity={"trainer": {"status": "active", "verified": True}},
        run={"id": "run-a", "state": "running", "stale": True, "invalid_updates": 2},
        checkpoint={
            "run_id": "run-a",
            "relative_path": "model_12.pt",
            "sha256": "abc",
            "visual_review_status": "not_recorded",
            "evaluations": [],
        },
        evaluations=[
            {
                "evaluation_id": "eval/invalid",
                "status": "INVALID",
                "hard_gates_failed": 1,
            }
        ],
    )

    by_rule = {item["rule_id"]: item for item in assessments}
    assert by_rule["system.database_degraded"]["severity"] == "warning"
    assert by_rule["system.supervisor_missing"]["severity"] == "warning"
    assert by_rule["runtime.revision_mismatch"]["severity"] == "warning"
    assert by_rule["training.telemetry_stale"]["severity"] == "warning"
    assert by_rule["training.nonfinite"]["severity"] == "critical"
    assert by_rule["checkpoint.unevaluated"]["evidence_refs"][0]["href"] == "/runs/run-a"
    assert by_rule["checkpoint.newest_unreviewed"]["subject_id"] == "run-a:model_12.pt"
    assert by_rule["evaluation.invalid"]["evidence_refs"][0]["href"] == "/evaluations/eval%2Finvalid"


def test_assessments_do_not_infer_idle_or_stale_from_unverified_process_state():
    assessments = build_assessments(
        checked_at=1000,
        components={"supervisor": {"status": "unavailable"}},
        database={"enabled": False, "available": False},
        runtime={},
        activity={"trainer": {"status": "unknown", "verified": False}},
        run={"id": "run-a", "state": "running", "stale": True},
        checkpoint=None,
        evaluations=[],
    )

    rules = {item["rule_id"] for item in assessments}
    assert "training.telemetry_stale" not in rules
    assert "training.nonfinite" not in rules
    assert "system.supervisor_missing" in rules


def test_assessment_uses_hash_linked_quantitative_evidence_only():
    assessments = build_assessments(
        checked_at=1000,
        components={},
        database={"enabled": False},
        runtime={},
        activity={},
        run={"id": "run-a"},
        checkpoint={
            "run_id": "run-a",
            "relative_path": "model.pt",
            "sha256": "abc",
            "visual_review_status": "reviewed",
            "evaluations": [
                {"evaluation_id": "eval/diag", "status": "DIAGNOSTIC_ONLY", "evidence_class": "diagnostic_only"}
            ],
        },
        evaluations=[
            {"evaluation_id": "eval/fail", "status": "FAIL", "hard_gates_failed": 1},
            {"evaluation_id": "eval/partial", "status": "INCOMPLETE", "hard_gates_failed": 0},
        ],
    )

    rules = {item["rule_id"] for item in assessments}
    assert "checkpoint.unevaluated" in rules
    assert "checkpoint.newest_unreviewed" not in rules
    assert "evaluation.hard_gate_failed" in rules
    assert "evaluation.invalid" not in rules
