import hashlib
import json

import pytest

from ascento_mjlab.control_contract import current_action_contract
from ascento_mjlab.evaluation.compare import ensure_compatible_plants, quality_baseline_verdict
from ascento_mjlab.plant_contract import current_plant_contract


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _quality_artifact(
    directory,
    *,
    status,
    suite_id="quality_v1",
    suite_sha256=None,
    resolved_scenarios_sha256=None,
    gates=None,
):
    directory.mkdir()
    (directory / "gate.json").write_text(
        json.dumps({"status": status, "gates": gates or []}), encoding="utf-8"
    )
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "suite_id": suite_id,
                "suite_sha256": suite_sha256 or _digest("suite-a"),
                "resolved_scenarios_sha256": resolved_scenarios_sha256
                or _digest("scenarios-a"),
            }
        ),
        encoding="utf-8",
    )


def _manifest(directory, contract):
    directory.mkdir()
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "plant_contract": contract,
                "checkpoint_plant_contract": contract,
                "action_contract": current_action_contract() if contract else None,
                "checkpoint_action_contract": current_action_contract() if contract else None,
            }
        ),
        encoding="utf-8",
    )


def test_comparison_rejects_legacy_or_mixed_plant_provenance(tmp_path):
    legacy = tmp_path / "legacy"
    current = tmp_path / "current"
    _manifest(legacy, None)
    _manifest(current, current_plant_contract())

    with pytest.raises(ValueError, match="legacy"):
        ensure_compatible_plants(legacy, current)


def test_comparison_accepts_exact_matching_plant_contracts(tmp_path):
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    contract = current_plant_contract()
    _manifest(base, contract)
    _manifest(candidate, contract)

    ensure_compatible_plants(base, candidate)


def test_quality_baseline_verdict_rejects_a_candidate_that_fails_a_hard_gate(tmp_path):
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    _quality_artifact(base, status="PASS")
    _quality_artifact(
        candidate,
        status="FAIL",
        gates=[{"gate_id": "heading", "hard": True, "passed": False}],
    )

    assert quality_baseline_verdict(base, candidate) == {
        "baseline_status": "PASS",
        "candidate_status": "FAIL",
        "verdict": "WORSE",
        "reason": "candidate failed hard quality gates",
        "failed_hard_gates": ["heading"],
    }


def test_quality_baseline_verdict_requires_matching_known_suites(tmp_path):
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    _quality_artifact(base, status="PASS")
    _quality_artifact(candidate, status="PASS", suite_id="other_quality_v1")

    assert quality_baseline_verdict(base, candidate) is None


def test_quality_baseline_verdict_requires_both_suite_ids(tmp_path):
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    _quality_artifact(base, status="PASS")
    _quality_artifact(candidate, status="PASS")
    (candidate / "manifest.json").write_text("{}", encoding="utf-8")

    assert quality_baseline_verdict(base, candidate) is None


@pytest.mark.parametrize(
    ("baseline_status", "candidate_status"),
    [
        ("PASS", "INVALID"),
        ("PASS", "INCOMPLETE"),
        ("INVALID", "PASS"),
        ("INCOMPLETE", "PASS"),
    ],
)
def test_quality_baseline_verdict_ignores_invalid_or_incomplete_artifacts(
    tmp_path, baseline_status, candidate_status
):
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    _quality_artifact(base, status=baseline_status)
    _quality_artifact(candidate, status=candidate_status)

    assert quality_baseline_verdict(base, candidate) is None


@pytest.mark.parametrize(
    ("field", "base_value", "candidate_value"),
    [
        ("suite_sha256", _digest("suite-a"), _digest("suite-b")),
        (
            "resolved_scenarios_sha256",
            _digest("scenarios-a"),
            _digest("scenarios-b"),
        ),
    ],
)
def test_quality_baseline_verdict_requires_matching_evaluation_content(
    tmp_path, field, base_value, candidate_value
):
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    _quality_artifact(base, status="PASS", **{field: base_value})
    _quality_artifact(candidate, status="FAIL", **{field: candidate_value})

    assert quality_baseline_verdict(base, candidate) is None


@pytest.mark.parametrize("missing_field", ["suite_sha256", "resolved_scenarios_sha256"])
def test_quality_baseline_verdict_requires_evaluation_content_hashes(
    tmp_path, missing_field
):
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    _quality_artifact(base, status="PASS")
    _quality_artifact(candidate, status="FAIL")
    manifest_path = candidate / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop(missing_field)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert quality_baseline_verdict(base, candidate) is None


@pytest.mark.parametrize("invalid_digest", ["unknown", "not-a-sha256"])
def test_quality_baseline_verdict_rejects_non_hash_content_ids(tmp_path, invalid_digest):
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    _quality_artifact(base, status="PASS")
    _quality_artifact(candidate, status="FAIL")
    manifest_path = candidate / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["resolved_scenarios_sha256"] = invalid_digest
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert quality_baseline_verdict(base, candidate) is None
