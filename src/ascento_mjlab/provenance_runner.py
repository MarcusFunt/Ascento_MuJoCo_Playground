"""RSL-RL runner that writes the immutable plant contract into every checkpoint."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from mjlab.rl import MjlabOnPolicyRunner

from .checkpoint_contract import require_current_checkpoint_contracts
from .control_contract import current_action_contract, require_current_action_contract
from .plant_contract import current_plant_contract, require_current_plant_contract
from .task_contract import (
    canonical_task_cfg_for_runtime_cfg,
    classify_actor_transfer_compatibility,
    current_task_contract,
)


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
        # Viewer/evaluation environments may be compiled from ``play=True``
        # configs that add recorder or presentation terms.  Those operational
        # terms are not part of the training ABI, so validate against the
        # registered canonical task topology instead of the mutated runtime
        # config.  Training/resume configs already carry the task id and use
        # the same canonical registration here.
        runtime_cfg = self.env.unwrapped.cfg
        contract_cfg = canonical_task_cfg_for_runtime_cfg(runtime_cfg)
        require_current_checkpoint_contracts(infos, contract_cfg)
        return infos

    def initialize_from_compatible_actor(self, source_checkpoint: str | Path) -> dict[str, Any]:
        """Initialize only actor weights/normalization for a new signed task.

        This does not weaken normal checkpoint loading.  Plant and structured
        action contracts must still match, then a separate actor-ABI comparison
        proves that the actor observation topology, action topology, control
        timestep, and reward-dt convention are identical.  Critic, optimizer,
        iteration, and environment state remain the fresh target-task values.
        """
        source = Path(source_checkpoint).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"source checkpoint does not exist: {source}")
        payload = torch.load(source, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("source checkpoint is not a supported RSL-RL checkpoint")
        infos = payload.get("infos")
        if not isinstance(infos, dict):
            raise ValueError("source checkpoint lacks provenance metadata")
        require_current_plant_contract(infos.get("plant_contract"))
        require_current_action_contract(infos.get("action_contract"))
        compatibility = classify_actor_transfer_compatibility(
            infos.get("task_contract"), current_task_contract(self.env.unwrapped.cfg)
        )
        if not compatibility.compatible:
            raise ValueError(f"actor transfer rejected: {compatibility.reason}")
        if "actor_state_dict" not in payload:
            raise ValueError("source checkpoint lacks actor weights")

        self.alg.load(
            payload,
            load_cfg={
                "actor": True,
                "critic": False,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
            },
            strict=True,
        )
        self.current_learning_iteration = 0
        self.env.unwrapped.common_step_counter = 0
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        return {
            "kind": "compatible_actor_transfer",
            "source_checkpoint": str(source),
            "source_checkpoint_sha256": digest,
            "source_task_contract": infos["task_contract"],
            "target_task_contract": current_task_contract(self.env.unwrapped.cfg),
            "actor_compatibility": compatibility.to_dict(),
            "copied": ["actor_weights", "actor_observation_normalization"],
            "fresh": ["critic", "optimizer", "iteration", "environment_state"],
        }
