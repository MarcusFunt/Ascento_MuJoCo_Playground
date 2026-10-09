"""Translate task/runtime state into UI-ready training curricula."""

from __future__ import annotations

import math
from typing import Any

HORIZON_SCHEDULE_S = (20.0, 60.0, 120.0, 300.0)
HORIZON_SUCCESS_WINDOWS = 6
HORIZON_FAILURE_WINDOWS = 4
HORIZON_SUCCESS_THRESHOLD = 0.90
HORIZON_FAILURE_THRESHOLD = 0.50

HORIZON_TASKS = {
    "Ascento-Balance-Flat",
    "Ascento-Balance-Quiet-Flat",
    "Ascento-Velocity-Flat",
    "Ascento-Balance-Recovery-Flat",
}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _integer(value: Any) -> int | None:
    value = _number(value)
    return int(value) if value is not None else None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


_GENERALIST_COHORTS = (
    ("precision_anchor", "Precision anchor"),
    ("recovery_retarget_anchor", "Recovery and retarget anchor"),
    ("generalist_navigation", "Generalist navigation"),
)
_GENERALIST_BANDS = ("short", "medium", "long")


def _generalist_curriculum(telemetry: dict[str, Any], task: str) -> dict[str, Any]:
    """Normalize task metrics while preserving their sample grain and source."""
    raw_metrics = telemetry.get("metrics")
    if not isinstance(raw_metrics, dict):
        raw_metrics = telemetry.get("canonical_metrics")
    raw_metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
    metrics: dict[str, Any] = {}
    for name, value in raw_metrics.items():
        # TensorBoard readers may preserve the ``Episode/`` namespace.
        metrics[str(name).rsplit("/", 1)[-1]] = value

    def metric(name: str) -> float | None:
        return _number(metrics.get(name))

    def count(name: str) -> int | None:
        value = metric(name)
        return max(0, int(value)) if value is not None else None

    stage_value = count("generalist_curriculum_goal_mix_stage")
    stage = min(2, stage_value) if stage_value is not None else None
    fractions = {
        "short": metric("generalist_curriculum_short_goal_fraction"),
        "medium": metric("generalist_curriculum_medium_goal_fraction"),
        "long": metric("generalist_curriculum_long_goal_fraction"),
    }
    gates = [
        {
            "id": "gate_recovery",
            "label": "Recovery on gate-like episodes",
            "estimate": metric("generalist_curriculum_stage_gate_recovery_lcb"),
            "threshold": 0.85,
            "successes": count("generalist_curriculum_window_gate_recovery_successes"),
            "samples": count("generalist_curriculum_window_gate_episodes"),
            "minimum_samples": 64,
            "grain": "gate-like episodes",
        },
        {
            "id": "short_arrival",
            "label": "Short target arrival",
            "estimate": metric("generalist_curriculum_stage_short_attempt_arrival_lcb"),
            "threshold": 0.50,
            "successes": count("generalist_curriculum_window_short_arrivals"),
            "samples": count("generalist_curriculum_window_short_attempts"),
            "minimum_samples": 64,
            "grain": "target attempts",
        },
        {
            "id": "long_arrival",
            "label": "Long target arrival",
            "estimate": metric("generalist_curriculum_stage_long_attempt_arrival_lcb"),
            "threshold": 0.50,
            "successes": count("generalist_curriculum_window_long_arrivals"),
            "samples": count("generalist_curriculum_window_long_attempts"),
            "minimum_samples": 64,
            "grain": "target attempts",
        },
    ]
    for gate in gates:
        samples = gate["samples"]
        gate["coverage"] = (
            min(1.0, samples / gate["minimum_samples"]) if samples is not None else None
        )
        gate["state"] = (
            "waiting"
            if gate["estimate"] is None or samples is None or samples < gate["minimum_samples"]
            else "pass"
            if gate["estimate"] >= gate["threshold"]
            else "below_threshold"
        )

    cohorts = []
    for cohort_id, label in _GENERALIST_COHORTS:
        prefix = f"generalist_cohort_{cohort_id}_"
        episode_count = count(prefix + "episode_count")
        arrivals = count(prefix + "arrival_count")
        cohorts.append(
            {
                "id": cohort_id,
                "label": label,
                "episode_count": episode_count,
                "arrival_count": arrivals,
                "arrival_rate": arrivals / episode_count
                if arrivals is not None and episode_count
                else None,
                "recovery_count": count(prefix + "recovery_count"),
                "settled_stop_count": count(prefix + "settled_stop_count"),
                "fall_count": count(prefix + "fall_count"),
                "timeout_count": count(prefix + "timeout_count"),
                "grain": "episodes",
            }
        )

    attempt_bands = []
    for band in _GENERALIST_BANDS:
        prefix = f"generalist_attempt_generalist_navigation_{band}_"
        attempts = count(prefix + "target_attempt_count")
        arrivals = count(prefix + "arrival_completed_count")
        settled = count(prefix + "settled_stop_completed_count")
        attempt_bands.append(
            {
                "id": band,
                "label": f"{band.title()} targets",
                "attempts": attempts,
                "arrivals": arrivals,
                "arrival_rate": arrivals / attempts if arrivals is not None and attempts else None,
                "settled_stops": settled,
                "settled_stop_rate": settled / attempts if settled is not None and attempts else None,
                "heading_valid_at_settle": count(prefix + "heading_valid_at_settle_count"),
                "fall_interruptions": count(prefix + "interrupted_by_fall_count"),
                "timeout_interruptions": count(prefix + "interrupted_by_timeout_count"),
                "mean_final_target_error_m": metric(prefix + "final_target_error_mean_m"),
                "p95_final_target_error_m": metric(prefix + "final_target_error_p95_m"),
                "mean_time_to_arrival_s": metric(prefix + "time_to_arrival_mean_s"),
                "p95_time_to_arrival_s": metric(prefix + "time_to_arrival_p95_s"),
                "mean_time_to_settle_s": metric(prefix + "time_to_settle_mean_s"),
                "p95_time_to_settle_s": metric(prefix + "time_to_settle_p95_s"),
                "mean_heading_error_at_settle_rad": metric(prefix + "heading_error_at_settle_mean_rad"),
                "grain": "target attempts",
            }
        )

    return {
        "kind": "generalist",
        "label": "Generalist navigation curriculum",
        "task": task,
        "stage": stage,
        "stage_count": 3,
        "progress": metric("generalist_curriculum_progress"),
        "target_distance_m": {
            "minimum": metric("generalist_curriculum_scheduled_target_min_distance_m"),
            "maximum": metric("generalist_curriculum_scheduled_target_max_distance_m"),
        },
        "goal_fractions": fractions,
        "gate_like_fraction": metric("generalist_curriculum_scheduled_gate_like_fraction"),
        "demotion_streak": count("generalist_curriculum_demotion_streak"),
        "gates": gates,
        "cohorts": cohorts,
        "attempt_bands": attempt_bands,
        "telemetry": {
            "iteration": _integer(telemetry.get("iteration")),
            "wall_time": _number(telemetry.get("wall_time")),
            "source": "latest TensorBoard episode metrics",
            "window_label": "latest emitted curriculum window; counters retain their task-defined grain",
        },
        "has_metrics": bool(metrics),
        "note": "Episode outcomes and target-attempt outcomes use separate denominators. Missing values mean not measured.",
    }


def _recovery_difficulty(
    iteration: int | None,
    rollout_steps: int | None,
    exact_control_steps: int | None = None,
) -> dict[str, Any]:
    ramp_steps = 120_000
    control_steps = (
        max(0, exact_control_steps)
        if exact_control_steps is not None
        else max(0, (iteration or 0) * (rollout_steps or 24))
    )
    progress = _clamp(control_steps / ramp_steps)
    hard_fraction = 0.10 + 0.20 * progress
    pitch_max = 0.10 + 0.05 * progress
    linear_x_max = 0.10 + 0.15 * progress
    angular_max = 0.20 + 0.30 * progress
    return {
        "kind": "recovery_difficulty",
        "label": "Recovery reset difficulty",
        "progress": progress,
        "control_steps": control_steps,
        "ramp_control_steps": ramp_steps,
        "hard_fraction": hard_fraction,
        "hard_fraction_start": 0.10,
        "hard_fraction_end": 0.30,
        "pitch_max_rad": pitch_max,
        "linear_x_max_m_s": linear_x_max,
        "angular_max_rad_s": angular_max,
    }


def curriculum_for_run(detail: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a compact, task-aware curriculum description for the dashboard."""
    if not detail:
        return None
    run_info = detail.get("run_info") if isinstance(detail.get("run_info"), dict) else {}
    status = detail.get("status") if isinstance(detail.get("status"), dict) else {}
    telemetry = detail.get("telemetry") if isinstance(detail.get("telemetry"), dict) else {}
    task = run_info.get("task") or status.get("task")
    iteration = _integer(telemetry.get("iteration"))

    if task in HORIZON_TASKS:
        current_horizon = _number(
            run_info.get("episode_horizon_s")
            if run_info.get("episode_horizon_s") is not None
            else run_info.get("requested_episode_horizon_s")
        )
        stage = _integer(run_info.get("horizon_stage"))
        if stage is None and current_horizon is not None:
            stage = next(
                (
                    index + 1
                    for index, horizon in enumerate(HORIZON_SCHEDULE_S)
                    if current_horizon <= horizon
                ),
                len(HORIZON_SCHEDULE_S),
            )
        stage = max(1, min(stage or 1, len(HORIZON_SCHEDULE_S)))
        stages = []
        for index, horizon in enumerate(HORIZON_SCHEDULE_S, start=1):
            if index < stage:
                state = "complete"
            elif index == stage:
                state = "current"
            else:
                state = "upcoming"
            stages.append(
                {"index": index, "label": f"{int(horizon)} s", "value": horizon, "state": state}
            )

        qualified = _integer(run_info.get("horizon_qualified_windows")) or 0
        failed = _integer(run_info.get("horizon_failed_windows")) or 0
        result: dict[str, Any] = {
            "kind": "horizon",
            "label": "Episode horizon curriculum",
            "task": task,
            "stage": stage,
            "stage_count": len(HORIZON_SCHEDULE_S),
            "current_horizon_s": current_horizon or HORIZON_SCHEDULE_S[stage - 1],
            "stages": stages,
            "promotion": {
                "qualified_windows": qualified,
                "required_windows": HORIZON_SUCCESS_WINDOWS,
                "timeout_fraction": _number(run_info.get("horizon_timeout_fraction")),
                "timeout_threshold": HORIZON_SUCCESS_THRESHOLD,
                "quality_fraction": _number(run_info.get("horizon_stationary_quality_fraction")),
                "quality_threshold": HORIZON_SUCCESS_THRESHOLD,
                "quality_failure_counts": run_info.get("horizon_quality_failure_counts"),
            },
            "demotion": {
                "failed_windows": failed,
                "required_windows": HORIZON_FAILURE_WINDOWS,
                "severe_timeout_threshold": HORIZON_FAILURE_THRESHOLD,
            },
            "stage_windows": _integer(run_info.get("horizon_stage_windows")),
            "top_horizon_windows": _integer(run_info.get("horizon_top_windows")),
            "transition": run_info.get("horizon_transition") or "held",
            "protected": stage == len(HORIZON_SCHEDULE_S),
            "candidate_checkpoint": run_info.get("long_horizon_candidate_checkpoint"),
        }
        if task == "Ascento-Balance-Recovery-Flat":
            result["secondary"] = _recovery_difficulty(
                iteration,
                _integer(run_info.get("rollout_steps_per_env")),
                _integer(run_info.get("horizon_control_steps")),
            )
        return result

    if task == "Ascento-Locomotion-Flat":
        return {
            "kind": "sequence",
            "label": "Locomotion training sequence",
            "task": task,
            "note": "Each vectorized environment moves through this sequence independently.",
            "stages": [
                {"label": "Settle", "detail": "Hold stable for 0.75 s"},
                {"label": "Push", "detail": "0.05–0.15 m/s impulse; 75% fore/aft"},
                {"label": "Recover", "detail": "Regain supported balance"},
                {"label": "Target", "detail": "Move 5–20 cm toward the world target"},
                {"label": "Stop", "detail": "Settle at the target"},
            ],
        }

    if task == "Ascento-Generalist-Locomotion-Flat":
        return _generalist_curriculum(telemetry, str(task))

    return {
        "kind": "static",
        "label": "Training program",
        "task": task,
        "stage": detail.get("stage"),
        "note": "This task does not currently emit an adaptive curriculum state.",
    }
