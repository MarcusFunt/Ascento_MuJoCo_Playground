import hashlib
import json

import pytest
from dashboard.checkpoint_registry import checkpoint_evidence


def test_checkpoint_evidence_links_by_content_hash_without_claiming_selection(tmp_path):
    run_dir = tmp_path / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "model_100.pt"
    checkpoint.write_bytes(b"stable policy bytes")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"selected_checkpoint": None}), encoding="utf-8"
    )
    evaluation_dir = tmp_path / "evaluations" / "eval-1"
    evaluation_dir.mkdir(parents=True)
    (evaluation_dir / "manifest.json").write_text(
        json.dumps({
            "suite_id": "dev_v1",
            "task": "Ascento-Balance-Flat",
            "checkpoint_sha256": digest,
            "finished_at_utc": "2026-10-09T12:00:00Z",
        }),
        encoding="utf-8",
    )
    (evaluation_dir / "suite.json").write_text(
        json.dumps({
            "suite_id": "dev_v1",
            "task": "Ascento-Balance-Flat",
            "gates": [{"id": "upright", "hard": True, "op": ">=", "threshold": 1.0}],
        }),
        encoding="utf-8",
    )
    (evaluation_dir / "gate.json").write_text(
        json.dumps({
            "status": "PASS",
            "gates": [{
                "gate_id": "upright",
                "hard": True,
                "op": ">=",
                "threshold": 1.0,
                "observed": 1.0,
                "passed": True,
            }],
        }),
        encoding="utf-8",
    )
    (evaluation_dir / "consistency.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
    for filename in ("resolved_scenarios.jsonl", "summary.json", "results.sqlite"):
        (evaluation_dir / filename).write_text("{}", encoding="utf-8")

    result = checkpoint_evidence(
        run_id="run-1",
        run_dir=run_dir,
        stable_checkpoints=[{"relative_path": "model_100.pt", "iteration": 100, "stable": True}],
        evaluation_root=tmp_path / "evaluations",
    )

    assert result["sha256"] == digest
    assert result["size_bytes"] == len(b"stable policy bytes")
    assert result["selection_status"] == "not_recorded"
    assert result["visual_review_status"] == "not_recorded"
    assert result["evaluations"][0]["status"] == "PASS"


def test_checkpoint_evidence_rejects_paths_outside_the_run(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "secret.pt"
    outside.write_bytes(b"not a checkpoint")

    with pytest.raises(ValueError, match="escapes"):
        checkpoint_evidence(
            run_id="run",
            run_dir=run_dir,
            stable_checkpoints=[{"relative_path": "../secret.pt", "stable": True}],
            evaluation_root=tmp_path / "evaluations",
        )
