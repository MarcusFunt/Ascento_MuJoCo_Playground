"""Shared geometry helpers that do not require task registration."""

from __future__ import annotations

import torch


def projected_gravity_tilt(projected_gravity_b: torch.Tensor) -> torch.Tensor:
    """Return body tilt in radians: zero upright and pi inverted."""
    if projected_gravity_b.ndim != 2 or projected_gravity_b.shape[1] != 3:
        raise ValueError("projected_gravity_b must have shape [num_envs, 3]")
    return torch.atan2(
        torch.linalg.vector_norm(projected_gravity_b[:, :2], dim=1),
        -projected_gravity_b[:, 2],
    )
