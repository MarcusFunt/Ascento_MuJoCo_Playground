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
    jump_probability: float = 0.0
    max_jump_height: float = 0.0
    max_jump_distance: float = 0.0


STAGES = (
    StageSpec("balance", "balance", 0.0, 0.0, 0.0, 0.08),
    StageSpec("flat_commands", "balance", 0.4, 0.5, 0.08, 0.12),
    StageSpec("recovery", "recovery", 0.5, 0.5, 0.08, 0.55),
    StageSpec("jump_flat", "jump", 0.1, 0.0, 0.04, 0.30, 0.55, 0.12, 0.0),
    StageSpec("high_landing", "jump", 0.2, 0.2, 0.08, 0.35, 0.65, 0.22, 0.10),
    StageSpec("clearance", "jump", 0.4, 0.3, 0.08, 0.40, 0.70, 0.28, 0.20),
    StageSpec("moving_jump", "jump", 1.0, 0.5, 0.10, 0.45, 0.70, 0.30, 0.60),
    StageSpec("unified_fine_tune", "jump", 1.0, 0.6, 0.12, 0.55, 0.60, 0.30, 0.75),
)


def stage_by_name(name: str) -> StageSpec:
    for stage in STAGES:
        if stage.name == name:
            return stage
    raise ValueError(f"unknown stage {name!r}; choose from {[s.name for s in STAGES]}")


def accepts_stage(stage: str, metrics: Mapping[str, float]) -> bool:
    """Uses physical metrics, never a lone total-reward threshold."""
    if stage in ("balance", "flat_commands"):
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
