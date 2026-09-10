"""RSL-RL runner that writes the immutable plant contract into every checkpoint."""

from __future__ import annotations

from typing import Any

from mjlab.rl import MjlabOnPolicyRunner

from .control_contract import current_action_contract, require_current_action_contract
from .plant_contract import current_plant_contract


class AscentoProvenanceRunner(MjlabOnPolicyRunner):
    """Persist the compiled simulation authority with all Ascento checkpoints."""

    def save(self, path: str, infos: dict[str, Any] | None = None) -> None:
        provenance = {
            **(infos or {}),
            "plant_contract": current_plant_contract(),
            "action_contract": current_action_contract(),
        }
        super().save(path, infos=provenance)

    def load(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Refuse direct-effort checkpoints for resume, evaluation, or capture."""
        infos = super().load(*args, **kwargs)
        require_current_action_contract(infos.get("action_contract") if isinstance(infos, dict) else None)
        return infos
