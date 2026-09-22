"""Small, testable helpers for the live browser-viewer diagnostic HUD."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class BalanceConfidence:
    """State-derived fall margin used only for visualization/debugging."""

    confidence: float
    predicted_tilt_rad: float
    tilt_margin: float
    height_margin: float
    support_factor: float


def compute_balance_confidence(
    *,
    tilt_rad: float,
    tilt_rate_rad_s: float,
    height_m: float,
    left_contact: bool,
    right_contact: bool,
    fall_tilt_rad: float,
    min_height_m: float,
    nominal_height_m: float,
    lookahead_s: float = 0.25,
) -> BalanceConfidence:
    """Estimate remaining balance margin from state and the real fall boundary.

    This is deliberately a diagnostic heuristic, not a probability emitted by
    the policy. Only tilt that is currently increasing is projected forward, so
    an aggressive recovery motion is not penalized merely for having a large
    angular rate.
    """
    values = (
        tilt_rad,
        tilt_rate_rad_s,
        height_m,
        fall_tilt_rad,
        min_height_m,
        nominal_height_m,
        lookahead_s,
    )
    if not all(math.isfinite(value) for value in values):
        return BalanceConfidence(0.0, math.inf, 0.0, 0.0, 0.0)
    if fall_tilt_rad <= 0.0:
        raise ValueError("fall_tilt_rad must be positive")
    if nominal_height_m <= min_height_m:
        raise ValueError("nominal_height_m must be greater than min_height_m")
    if lookahead_s < 0.0:
        raise ValueError("lookahead_s must be non-negative")

    predicted_tilt = max(0.0, tilt_rad + max(0.0, tilt_rate_rad_s) * lookahead_s)
    tilt_margin = _clamp01((fall_tilt_rad - predicted_tilt) / fall_tilt_rad)
    height_margin = _clamp01(
        (height_m - min_height_m) / (nominal_height_m - min_height_m)
    )

    if left_contact and right_contact:
        support_factor = 1.0
    elif left_contact or right_contact:
        support_factor = 0.65
    else:
        support_factor = 0.25

    confidence = _clamp01(min(tilt_margin, height_margin) * support_factor)
    return BalanceConfidence(
        confidence=confidence,
        predicted_tilt_rad=predicted_tilt,
        tilt_margin=tilt_margin,
        height_margin=height_margin,
        support_factor=support_factor,
    )


def smooth_value(previous: float | None, current: float, smoothing: float) -> float:
    """Apply a simple GUI-tunable exponential smoother."""
    if not 0.0 <= smoothing <= 0.99:
        raise ValueError("smoothing must lie in [0, 0.99]")
    if previous is None or not math.isfinite(previous):
        return current
    return smoothing * previous + (1.0 - smoothing) * current


def sparkline(
    values: Sequence[float],
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> str:
    """Render a dependency-free unicode sparkline for compact live history."""
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return "—"

    low = min(finite) if minimum is None else float(minimum)
    high = max(finite) if maximum is None else float(maximum)
    if not math.isfinite(low) or not math.isfinite(high):
        return "—"
    if high <= low:
        return "▄" * len(finite)

    blocks = "▁▂▃▄▅▆▇█"
    scale = (len(blocks) - 1) / (high - low)
    result = []
    for value in finite:
        index = round((value - low) * scale)
        index = max(0, min(len(blocks) - 1, index))
        result.append(blocks[index])
    return "".join(result)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
