"""Small deterministic, read-only dashboard assessments."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

_ACTIVE = {"starting", "running", "stopping"}


def _count(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0


def _assessment(
    *,
    rule_id: str,
    severity: str,
    headline: str,
    explanation: str,
    subject_type: str,
    subject_id: str | None,
    recommended_action: str,
    href: str,
    evidence_label: str,
    checked_at: float,
) -> dict[str, Any]:
    return {
        "id": f"{rule_id}:{subject_id or 'dashboard'}",
        "rule_id": rule_id,
        "severity": severity,
        "headline": headline,
        "explanation": explanation,
        "subject_type": subject_type,
        "subject_id": subject_id,
        "recommended_action": recommended_action,
        "evidence_refs": [{"label": evidence_label, "href": href}],
        "observed_at": checked_at,
        "freshness_seconds": 0,
        "source": "deterministic_dashboard_rule",
    }


def build_assessments(
    *,
    checked_at: float,
    components: dict[str, Any],
    database: dict[str, Any],
    runtime: dict[str, Any],
    activity: dict[str, Any],
    run: dict[str, Any] | None,
    checkpoint: dict[str, Any] | None,
    evaluations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build concise evidence-linked warnings without mutating project state."""
    items: list[dict[str, Any]] = []

    def add(**kwargs: Any) -> None:
        items.append(_assessment(checked_at=checked_at, **kwargs))

    if database.get("enabled") and not database.get("available"):
        add(
            rule_id="system.database_degraded",
            severity="warning",
            headline="Dashboard database is degraded",
            explanation=str(database.get("error") or "The filesystem index is serving as fallback."),
            subject_type="system",
            subject_id="database",
            recommended_action="Open System and inspect database connectivity and the last successful sync.",
            href="/system",
            evidence_label="Database component status",
        )

    supervisor = components.get("supervisor") if isinstance(components, dict) else None
    if isinstance(supervisor, dict) and supervisor.get("status") in {"unavailable", "degraded"}:
        add(
            rule_id="system.supervisor_missing",
            severity="warning",
            headline="Host activity cannot be verified",
            explanation=str(supervisor.get("message") or "The host supervisor is unavailable."),
            subject_type="system",
            subject_id="supervisor",
            recommended_action="Restore the configured host supervisor before relying on process activity state.",
            href="/system",
            evidence_label="Supervisor component status",
        )

    comparisons = runtime.get("comparisons") if isinstance(runtime, dict) else None
    mismatches = [
        name
        for name in ("checkout_matches_image", "image_matches_manifest")
        if isinstance(comparisons, dict) and comparisons.get(name) is False
    ]
    run_repo = (run or {}).get("repository_version") if isinstance(run, dict) else None
    run_outdated = isinstance(run_repo, dict) and run_repo.get("is_outdated") is True
    if mismatches or run_outdated:
        details = ", ".join(mismatches) if mismatches else "latest indexed run is outdated"
        add(
            rule_id="runtime.revision_mismatch",
            severity="warning",
            headline="Runtime revisions do not match",
            explanation=f"Revision provenance reports: {details}.",
            subject_type="runtime",
            subject_id="dashboard",
            recommended_action="Inspect the checkout, packaged image and run revisions before starting work.",
            href="/system",
            evidence_label="Runtime identity report",
        )

    trainer = activity.get("trainer") if isinstance(activity, dict) else None
    trainer_verified_active = (
        isinstance(trainer, dict)
        and trainer.get("verified") is True
        and trainer.get("status") == "active"
    )
    if trainer_verified_active and isinstance(run, dict) and run.get("state") in _ACTIVE and run.get("stale") is True:
        run_id = str(run.get("id") or "unknown")
        add(
            rule_id="training.telemetry_stale",
            severity="warning",
            headline="Training telemetry is stale",
            explanation="The supervisor verifies a running process, but the latest indexed run telemetry is marked stale.",
            subject_type="run",
            subject_id=run_id,
            recommended_action="Inspect the run log and artifact heartbeat before interpreting performance trends.",
            href=f"/runs/{quote(run_id, safe='')}",
            evidence_label="Verified trainer state and run freshness",
        )

    if isinstance(run, dict):
        training_health = run.get("training_health") if isinstance(run.get("training_health"), dict) else {}
        telemetry = run.get("telemetry") if isinstance(run.get("telemetry"), dict) else {}
        canonical = telemetry.get("canonical_metrics") if isinstance(telemetry.get("canonical_metrics"), dict) else {}
        invalid_updates = max(
            _count(run.get("invalid_updates")),
            _count(training_health.get("invalid_updates")),
            _count(training_health.get("non_finite_updates")),
            _count(canonical.get("invalid_update")),
        )
        if invalid_updates:
            run_id = str(run.get("id") or "unknown")
            add(
                rule_id="training.nonfinite",
                severity="critical",
                headline="Training reports invalid numerical updates",
                explanation=f"The run reports {invalid_updates} invalid or non-finite update(s).",
                subject_type="run",
                subject_id=run_id,
                recommended_action="Inspect the numerical-health telemetry and run log before using new checkpoints.",
                href=f"/runs/{quote(run_id, safe='')}",
                evidence_label="Run numerical-health telemetry",
            )

    if isinstance(checkpoint, dict):
        run_id = str(checkpoint.get("run_id") or (run or {}).get("id") or "unknown")
        checkpoint_id = f"{run_id}:{checkpoint.get('relative_path') or 'unknown'}"
        evidence_rows = checkpoint.get("evaluations")
        evidence_rows = evidence_rows if isinstance(evidence_rows, list) else []
        usable_evaluation = next(
            (
                row
                for row in evidence_rows
                if isinstance(row, dict)
                and row.get("status") in {"PASS", "FAIL"}
                and row.get("evidence_class") == "quantitative"
                and not any(
                    marker in str(row.get("suite_id") or "").lower()
                    for marker in ("heldout", "held_out", "promotion")
                )
            ),
            None,
        )
        if usable_evaluation is None:
            add(
                rule_id="checkpoint.unevaluated",
                severity="warning",
                headline="Newest stable checkpoint has no development evaluation",
                explanation="No valid quantitative, non-promotion evaluation is linked by this checkpoint's SHA-256.",
                subject_type="checkpoint",
                subject_id=checkpoint_id,
                recommended_action="Review the checkpoint, then use an appropriate development suite if evidence is needed.",
                href=f"/runs/{quote(run_id, safe='')}",
                evidence_label="Checkpoint SHA-256 and hash-linked evaluations",
            )
        if checkpoint.get("visual_review_status") != "reviewed":
            add(
                rule_id="checkpoint.newest_unreviewed",
                severity="info",
                headline="Newest stable checkpoint has no recorded visual review",
                explanation="No visual-review record is linked to this stable checkpoint.",
                subject_type="checkpoint",
                subject_id=checkpoint_id,
                recommended_action="Open this checkpoint in the visualizer and record a review if behavior inspection is needed.",
                href=f"/runs/{quote(run_id, safe='')}",
                evidence_label="Stable checkpoint identity",
            )

    evaluation_rows = [row for row in evaluations if isinstance(row, dict)]
    invalid = next((row for row in evaluation_rows if row.get("status") == "INVALID"), None)
    if invalid:
        evaluation_id = str(invalid.get("evaluation_id") or "unknown")
        add(
            rule_id="evaluation.invalid",
            severity="critical",
            headline="Evaluation evidence is invalid",
            explanation=str(invalid.get("integrity_error") or "An evaluator consistency check failed."),
            subject_type="evaluation",
            subject_id=evaluation_id,
            recommended_action="Investigate the consistency artifact before making any quality claim.",
            href=f"/evaluations/{quote(evaluation_id, safe='')}",
            evidence_label="Invalid evaluation artifact",
        )
    failed = next(
        (
            row
            for row in evaluation_rows
            if row.get("status") == "FAIL" and _count(row.get("hard_gates_failed")) > 0
        ),
        None,
    )
    if failed:
        evaluation_id = str(failed.get("evaluation_id") or "unknown")
        add(
            rule_id="evaluation.hard_gate_failed",
            severity="critical",
            headline="A saved evaluation failed a hard gate",
            explanation=f"{_count(failed.get('hard_gates_failed'))} observed hard gate(s) failed.",
            subject_type="evaluation",
            subject_id=evaluation_id,
            recommended_action="Inspect the failed gates and worst scenarios before selecting this checkpoint.",
            href=f"/evaluations/{quote(evaluation_id, safe='')}",
            evidence_label="Valid hard-gated evaluation",
        )

    severity_rank = {"critical": 0, "warning": 1, "info": 2}
    return sorted(items, key=lambda item: (severity_rank.get(item["severity"], 3), item["rule_id"]))
