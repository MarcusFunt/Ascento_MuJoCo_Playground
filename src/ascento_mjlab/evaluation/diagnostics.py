"""Named, evaluator-authoritative physical predicates for diagnosing settling."""

from __future__ import annotations

import torch


def stop_conditions(
    *,
    target_error: torch.Tensor,
    heading_error: torch.Tensor,
    planar_speed: torch.Tensor,
    tilt: torch.Tensor,
    angular_xy: torch.Tensor,
    yaw_rate: torch.Tensor,
    both_supported: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return the *actual* waypoint evaluator thresholds as independent booleans."""
    return {
        "distance": target_error <= 0.10,
        "heading": heading_error <= 0.20,
        "speed": planar_speed <= 0.08,
        "tilt": tilt <= 0.08,
        "angular_xy": angular_xy <= 0.25,
        "yaw_rate": yaw_rate <= 0.25,
        "both_wheels": both_supported,
    }


def recovery_conditions(
    *,
    planar_speed: torch.Tensor,
    height: torch.Tensor,
    heading_error: torch.Tensor,
    tilt: torch.Tensor,
    angular_xy: torch.Tensor,
    yaw_rate: torch.Tensor,
    both_supported: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return the distinct evaluator post-disturbance recovery envelope."""
    return {
        "tilt": tilt <= 0.08,
        "speed": planar_speed <= 0.10,
        "height": torch.abs(height - 0.75) <= 0.05,
        "angular_xy": angular_xy <= 0.25,
        "yaw_rate": yaw_rate <= 0.25,
        "heading": heading_error <= 0.15,
        "both_wheels": both_supported,
    }


def all_conditions(conditions: dict[str, torch.Tensor]) -> torch.Tensor:
    """Combine named flags without silently relaxing any threshold."""
    if not conditions:
        raise ValueError("At least one condition is required")
    iterator = iter(conditions.values())
    result = next(iterator)
    for flag in iterator:
        result = result & flag
    return result
