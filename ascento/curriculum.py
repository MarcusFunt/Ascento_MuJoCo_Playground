"""Stage definitions and physical-metric acceptance checks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class StageSpec:
    name: str
    env: str
    max_vx: float
    max_yaw_rate: float
    height_range: float
    disturbance_tilt: float
    reset_angular_velocity: float = 0.0
    reset_linear_velocity: float = 0.0
    reset_leg_variation: float = 0.0
    action_scale: float = 0.35
    initial_noise_std: float = 0.10
    entropy_cost: float = 0.0
    jump_probability: float = 0.0
    max_jump_height: float = 0.0
    max_jump_distance: float = 0.0
    # A rollout contains many correlated transitions; several PPO epochs are
    # needed before the direct-torque action head receives a measurable update.
    learning_rate: float = 3e-4
    updates_per_batch: int = 4
    reset_wheel_velocity: float = 2.0


STAGES = (
    # Start where the passive equilibrium is reachable.  PPO first learns to
    # preserve that equilibrium before it is asked to reject real disturbances.
    # A standing rollout starts at the passive zero-torque equilibrium.  Keep
    # the wheel speed at zero here, then widen it in the disturbed stage.
    StageSpec("balance", "balance", 0.0, 0.0, 0.0, 0.02, 0.05, 0.03, 0.01, 0.35, 0.12, 0.0, reset_wheel_velocity=0.0),
    StageSpec("disturbed_balance", "balance", 0.0, 0.0, 0.0, 0.01, 0.03, 0.03, 0.01, 0.30, 0.08, 1e-4, reset_wheel_velocity=0.25),
    StageSpec("flat_commands", "balance", 0.4, 0.5, 0.08, 0.08, 0.25, 0.18, 0.05, 0.55, 0.12, 2e-4),
    StageSpec("recovery", "recovery", 0.5, 0.5, 0.08, 0.55, 3.0, 1.0, 0.30, 0.75, 0.18, 3e-4),
    StageSpec("jump_flat", "jump", 0.1, 0.0, 0.04, 0.30, 1.0, 0.4, 0.15, 0.75, 0.15, 2e-4, 0.55, 0.12, 0.0),
    StageSpec("high_landing", "jump", 0.2, 0.2, 0.08, 0.35, 1.5, 0.5, 0.20, 0.80, 0.18, 3e-4, 0.65, 0.22, 0.10),
    StageSpec("clearance", "jump", 0.4, 0.3, 0.08, 0.40, 1.8, 0.6, 0.25, 0.85, 0.20, 3e-4, 0.70, 0.28, 0.20),
    StageSpec("moving_jump", "jump", 1.0, 0.5, 0.10, 0.45, 2.0, 0.8, 0.28, 0.90, 0.22, 4e-4, 0.70, 0.30, 0.60),
    StageSpec("unified_fine_tune", "jump", 1.0, 0.6, 0.12, 0.55, 2.5, 1.0, 0.30, 1.0, 0.25, 5e-4, 0.60, 0.30, 0.75),
)


def stage_by_name(name: str) -> StageSpec:
    for stage in STAGES:
        if stage.name == name:
            return stage
    raise ValueError(f"unknown stage {name!r}; choose from {[s.name for s in STAGES]}")


def accepts_stage(stage: str, metrics: Mapping[str, float]) -> bool:
    """Uses physical metrics, never a lone total-reward threshold."""
    if stage in ("balance", "disturbed_balance", "flat_commands"):
        return (metrics.get("survival_rate", 0.0) >= 0.95
                and metrics.get("rms_tilt", float("inf")) <= 0.20
                and metrics.get("height_error", float("inf")) <= 0.08
                and metrics.get("action_jitter", float("inf")) <= 0.20)
    if stage == "recovery":
        return (metrics.get("recovery_success_rate", 0.0) >= 0.80
                and metrics.get("median_recovery_time", float("inf")) <= 3.0
                and metrics.get("failure_rate", 1.0) <= 0.15)
    return (metrics.get("takeoff_rate", 0.0) >= 0.70
            and metrics.get("wheel_clearance", 0.0) >= 0.02
            and metrics.get("landing_speed", float("inf")) <= 2.5
            and metrics.get("recovery_success_rate", 0.0) >= 0.65)
