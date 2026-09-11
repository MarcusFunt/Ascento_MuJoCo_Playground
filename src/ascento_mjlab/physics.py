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
    contract_version: str = "v1"
    sim_dt_s: float = 0.002
    decimation: int = 5
    # Simulation-only authority; deliberately exceeds the nominal hardware rating.
    peak_effort_nm: float = 65.0
    default_root_height_m: float = 0.75

    @property
    def control_dt_s(self) -> float:
        return self.sim_dt_s * self.decimation

    @property
    def contract_id(self) -> str:
        """Stable identity for artifacts produced against this plant contract."""
        effort = f"{self.peak_effort_nm:g}".replace("-", "neg").replace(".", "p")
        return f"ascento-{self.name}-{effort}nm-{self.contract_version}"


PHYSICS_PROFILE = PhysicsProfile()
REWARD_SCHEMA_VERSION = "v3"

__all__ = ["PHYSICS_PROFILE", "PhysicsProfile", "REWARD_SCHEMA_VERSION"]
