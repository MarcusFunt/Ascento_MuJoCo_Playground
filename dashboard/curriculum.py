"""Translate task/runtime state into UI-ready training curricula."""

from __future__ import annotations

from typing import Any

HORIZON_SCHEDULE_S = (20.0, 60.0, 120.0, 300.0)
HORIZON_SUCCESS_WINDOWS = 6
HORIZON_FAILURE_WINDOWS = 4
HORIZON_SUCCESS_THRESHOLD = 0.90
HORIZON_FAILURE_THRESHOLD = 0.50

HORIZON_TASKS = {
    "Ascento-Balance-Flat",
    "Ascento-Velocity-Flat",
    "Ascento-Balance-Recovery-Flat",
}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _integer(value: Any) -> int | None:
    value = _number(value)
    return int(value) if value is not None else None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


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

    return {
        "kind": "static",
        "label": "Training program",
        "task": task,
        "stage": detail.get("stage"),
        "note": "This task does not currently emit an adaptive curriculum state.",
    }
