import json
import sqlite3

import pytest
from dashboard.evaluations import (
    compare_evaluations,
    discover_evaluations,
    evaluation_detail,
    evaluation_scenarios,
)


def _write_evaluation(root, evaluation_id, *, gate_status="PASS", gates=None, consistency=True):
    directory = root / evaluation_id
    directory.mkdir(parents=True)
    suite_gates = gates if gates is not None else [
        {"gate_id": "arrivals", "family": "waypoint", "metric": "arrival", "statistic": "mean", "op": ">=", "threshold": 0.9, "hard": True}
    ]
    suite = {
        "schema_version": 1,
        "suite_id": "test_dev_v1",
        "task": "Ascento-Locomotion-Flat",
        "root_seed": 42,
        "policy_mode": "deterministic",
        "families": [{"family_id": "waypoint", "kind": "fixed", "count": 2, "horizon_s": 1}],
        "gates": suite_gates,
        "required_capabilities": [],
    }
    (directory / "suite.json").write_text(json.dumps(suite), encoding="utf-8")
    (directory / "resolved_scenarios.jsonl").write_text('{"scenario_id":"s1"}\n', encoding="utf-8")
    (directory / "manifest.json").write_text(
        json.dumps({
            "suite_id": suite["suite_id"],
            "task": suite["task"],
            "suite_sha256": "a" * 64,
            "resolved_scenarios_sha256": "b" * 64,
            "scenario_count": 2,
            "checkpoint": "runs/model_100.pt",
            "checkpoint_sha256": "c" * 64,
            "repository_commit": "deadbeef",
            "finished_at_utc": "2026-10-09T12:00:00Z",
            "task_contract": {"actor_observations": ["orientation", "velocity"]},
            "checkpoint_task_compatibility_detail": {"compatible": True, "status": "current"},
        }),
        encoding="utf-8",
    )
    (directory / "summary.json").write_text(json.dumps({"families": {}}), encoding="utf-8")
    (directory / "gate.json").write_text(
        json.dumps({
            "status": gate_status,
            "gates": [{"gate_id": "arrivals", "hard": True, "passed": gate_status == "PASS", "observed": 1.0, "op": ">=", "threshold": 0.9}] if suite_gates else [],
        }),
        encoding="utf-8",
    )
    (directory / "consistency.json").write_text(json.dumps({"passed": consistency, "checks": []}), encoding="utf-8")
    connection = sqlite3.connect(directory / "results.sqlite")
    connection.executescript(
        """
        CREATE TABLE scenario (scenario_id TEXT, family TEXT, task TEXT, horizon_steps INTEGER, spec_json TEXT);
        CREATE TABLE episode (scenario_id TEXT, success INTEGER, termination_reason TEXT, episode_steps INTEGER);
        CREATE TABLE episode_metric (scenario_id TEXT, name TEXT, value REAL);
        CREATE TABLE event (scenario_id TEXT, name TEXT, value REAL);
        """
    )
    connection.executemany("INSERT INTO scenario VALUES (?, ?, ?, ?, ?)", [
        ("s1", "waypoint", suite["task"], 50, "{}"),
        ("s2", "waypoint", suite["task"], 50, "{}"),
    ])
    connection.executemany("INSERT INTO episode VALUES (?, ?, ?, ?)", [
        ("s1", 1, "arrived", 10),
        ("s2", 0, "timeout", 50),
    ])
    connection.executemany("INSERT INTO episode_metric VALUES (?, ?, ?)", [
        ("s1", "target_error_m", 0.01),
        ("s2", "target_error_m", 0.12),
    ])
    connection.commit()
    connection.close()
    return directory


def test_partial_evaluation_is_incomplete_not_a_policy_failure(tmp_path):
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "suite.json").write_text(
        '{"suite_id":"dev","gates":[{"hard":true}]}', encoding="utf-8"
    )

    result = discover_evaluations(tmp_path)

    assert len(result) == 1
    assert result[0]["status"] == "INCOMPLETE"
    assert result[0]["evidence_class"] == "quantitative"


def test_consistency_failure_overrides_a_passing_gate(tmp_path):
    _write_evaluation(tmp_path, "invalid", consistency=False)

    result = discover_evaluations(tmp_path)

    assert result[0]["status"] == "INVALID"
    assert result[0]["validity"] == "invalid"


def test_pass_with_missing_hard_gate_observation_is_incomplete(tmp_path):
    directory = _write_evaluation(tmp_path, "missing-observation")
    gate_path = directory / "gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["gates"][0]["observed"] = None
    gate["gates"][0]["passed"] = False
    gate_path.write_text(json.dumps(gate), encoding="utf-8")

    result = discover_evaluations(tmp_path)

    assert result[0]["status"] == "INCOMPLETE"
    assert result[0]["hard_gates_unavailable"] == 1
    assert "no finite observation" in result[0]["integrity_error"]


def test_gate_verdict_that_disagrees_with_threshold_is_invalid(tmp_path):
    directory = _write_evaluation(tmp_path, "false-pass")
    gate_path = directory / "gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["gates"][0]["observed"] = 0.1
    gate_path.write_text(json.dumps(gate), encoding="utf-8")

    result = discover_evaluations(tmp_path)

    assert result[0]["status"] == "INVALID"
    assert "disagrees with the frozen suite" in result[0]["integrity_error"]


def test_gate_free_diagnostic_never_receives_a_synthetic_pass(tmp_path):
    _write_evaluation(tmp_path, "morphology", gates=[])

    result = discover_evaluations(tmp_path)

    assert result[0]["status"] == "DIAGNOSTIC_ONLY"
    assert result[0]["evidence_class"] == "diagnostic_only"
    assert result[0]["hard_gates_passed"] == 0


def test_evaluation_details_and_scenarios_are_read_only_and_bounded(tmp_path):
    _write_evaluation(tmp_path, "complete")

    detail = evaluation_detail(tmp_path, "complete")
    worst = evaluation_scenarios(tmp_path, "complete", metric="target_error_m", limit=1, offset=0)

    assert detail["evaluation"]["evaluation_id"] == "complete"
    assert detail["gate"]["status"] == "PASS"
    assert worst["total"] == 2
    assert worst["scenarios"][0]["scenario_id"] == "s2"
    assert worst["scenarios"][0]["metric_value"] == 0.12


def test_evaluation_detail_rejects_traversal_outside_root(tmp_path):
    outside = tmp_path.parent / "outside"
    outside.mkdir()

    try:
        evaluation_detail(tmp_path, "../outside")
    except ValueError as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("evaluation path traversal should be rejected")


def test_paired_comparison_requires_matching_task_and_scenario_contracts(tmp_path, monkeypatch):
    import ascento_mjlab.evaluation.compare as comparison

    _write_evaluation(tmp_path, "baseline")
    candidate = _write_evaluation(tmp_path, "candidate")
    monkeypatch.setattr(comparison, "compare", lambda *_args: {"paired_scenarios": 2, "metrics": {}})

    payload = compare_evaluations(tmp_path, "baseline", "candidate")

    assert payload["baseline"]["evaluation_id"] == "baseline"
    assert payload["candidate"]["evaluation_id"] == "candidate"
    assert payload["comparison"]["paired_scenarios"] == 2
    assert payload["read_only"] is True
    assert payload["promotion_eligible"] is False

    manifest_path = candidate / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["task_contract"] = {"actor_observations": ["different"]}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="different observation or task contracts"):
        compare_evaluations(tmp_path, "baseline", "candidate")


def test_paired_comparison_rejects_incomplete_or_incompatible_artifacts(tmp_path):
    _write_evaluation(tmp_path, "baseline")
    _write_evaluation(tmp_path, "candidate", gate_status="INCOMPLETE")

    with pytest.raises(ValueError, match="complete, valid, quantitative"):
        compare_evaluations(tmp_path, "baseline", "candidate")
