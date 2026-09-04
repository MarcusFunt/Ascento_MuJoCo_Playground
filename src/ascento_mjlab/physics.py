"""Canonical simulation-only physics/timing contract for Ascento tasks.

Keep project-wide values that must agree across the plant, action mapping,
sensors, capture metadata, and tests here. Runtime code should still prefer the
active environment/model timestep when a task intentionally overrides it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicsProfile:
  name: str = "animation_high_authority"
  sim_dt_s: float = 0.002
  decimation: int = 5
  peak_effort_nm: float = 40.0
  default_root_height_m: float = 0.75

  @property
  def control_dt_s(self) -> float:
    return self.sim_dt_s * self.decimation


PHYSICS_PROFILE = PhysicsProfile()

__all__ = ["PHYSICS_PROFILE", "PhysicsProfile"]
