"""RSL-RL runner that writes the immutable plant contract into every checkpoint."""

from __future__ import annotations

from typing import Any

from mjlab.rl import MjlabOnPolicyRunner

from .checkpoint_contract import require_current_checkpoint_contracts
from .control_contract import current_action_contract
from .plant_contract import current_plant_contract
from .task_contract import current_task_contract


class AscentoProvenanceRunner(MjlabOnPolicyRunner):
    """Persist the compiled simulation authority with all Ascento checkpoints."""

    def save(self, path: str, infos: dict[str, Any] | None = None) -> None:
        provenance = {
            **(infos or {}),
            "plant_contract": current_plant_contract(),
            "action_contract": current_action_contract(),
            "task_contract": current_task_contract(self.env.unwrapped.cfg),
        }
        super().save(path, infos=provenance)

    def load(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Refuse checkpoints whose plant, action, or task ABI no longer matches."""
        infos = super().load(*args, **kwargs)
        require_current_checkpoint_contracts(infos, self.env.unwrapped.cfg)
        return infos
