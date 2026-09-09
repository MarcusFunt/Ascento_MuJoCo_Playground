"""RSL-RL runner that writes the immutable plant contract into every checkpoint."""

from __future__ import annotations

from typing import Any

from mjlab.rl import MjlabOnPolicyRunner

from .plant_contract import current_plant_contract


class AscentoProvenanceRunner(MjlabOnPolicyRunner):
    """Persist the compiled simulation authority with all Ascento checkpoints."""

    def save(self, path: str, infos: dict[str, Any] | None = None) -> None:
        provenance = {**(infos or {}), "plant_contract": current_plant_contract()}
        super().save(path, infos=provenance)
