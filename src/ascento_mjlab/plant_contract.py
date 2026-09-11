"""Immutable identity for the compiled Ascento simulation plant.

The task profile, robot MJCF, and evaluation evidence must describe one plant.
This module keeps that identity independent from mjlab so dashboard provenance
can be written before a trainer starts.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from .physics import PHYSICS_PROFILE

PLANT_CONTRACT_SCHEMA_VERSION = 1
ROBOT_MJCF = Path(__file__).parent / "assets" / "ascento_guard2" / "robot.xml"


def robot_mjcf_sha256(path: Path = ROBOT_MJCF) -> str:
    """Return the content identity of the robot model used by this checkout."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_plant_contract() -> dict[str, Any]:
    """Describe the current simulation-only plant without mutable runtime state."""
    return {
        "schema_version": PLANT_CONTRACT_SCHEMA_VERSION,
        "id": PHYSICS_PROFILE.contract_id,
        "profile": PHYSICS_PROFILE.name,
        "profile_version": PHYSICS_PROFILE.contract_version,
        "peak_effort_nm": PHYSICS_PROFILE.peak_effort_nm,
        "robot_mjcf_sha256": robot_mjcf_sha256(),
    }


def plant_contracts_compatible(
    left: Mapping[str, Any] | None, right: Mapping[str, Any] | None
) -> bool:
    """Require exact plant identity before comparing quantitative evidence."""
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return False
    fields = ("schema_version", "id", "peak_effort_nm", "robot_mjcf_sha256")
    return all(left.get(field) == right.get(field) for field in fields)


def require_current_plant_contract(contract: Mapping[str, Any] | None) -> None:
    """Reject a checkpoint compiled for another simulation plant."""
    if not isinstance(contract, Mapping):
        raise ValueError(
            "checkpoint lacks the plant contract and cannot be rolled out against the current "
            "simulation plant; retrain the policy"
        )
    if not plant_contracts_compatible(contract, current_plant_contract()):
        raise ValueError(
            "checkpoint plant contract is incompatible with the current simulation plant; retrain "
            "the policy"
        )


__all__ = [
    "PLANT_CONTRACT_SCHEMA_VERSION",
    "ROBOT_MJCF",
    "current_plant_contract",
    "plant_contracts_compatible",
    "require_current_plant_contract",
    "robot_mjcf_sha256",
]
