"""RSL-RL runner that writes the immutable plant contract into every checkpoint."""

from __future__ import annotations

import hashlib
import os
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

_ENVIRONMENT_PROGRESS_SCHEMA_VERSION = 1


class AscentoProvenanceRunner(MjlabOnPolicyRunner):
    """Persist the compiled simulation authority with all Ascento checkpoints."""

    def _num_steps_per_env(self) -> int:
        """Return rollout length across supported RSL-RL runner versions."""
        value = getattr(self, "num_steps_per_env", None)
        if value is None:
            value = self.cfg.get("num_steps_per_env", 0)
        return int(value)

    def _environment_progress(self) -> dict[str, int]:
        """Return training progress needed to resume environment-side curricula exactly."""
        return {
            "schema_version": _ENVIRONMENT_PROGRESS_SCHEMA_VERSION,
            "common_step_counter": int(self.env.unwrapped.common_step_counter),
            "learning_iteration": int(self.current_learning_iteration),
            "num_steps_per_env": self._num_steps_per_env(),
        }

    def save(self, path: str, infos: dict[str, Any] | None = None) -> None:
        provenance = {
            **(infos or {}),
            "plant_contract": current_plant_contract(),
            "action_contract": current_action_contract(),
            "task_contract": current_task_contract(self.env.unwrapped.cfg),
            "environment_progress": self._environment_progress(),
        }
        target = Path(path).expanduser().resolve()
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        upload_model = bool(self.cfg.get("upload_model", False))
        try:
            self.cfg["upload_model"] = False
            super().save(str(temporary), infos=provenance)
            os.replace(temporary, target)
        finally:
            self.cfg["upload_model"] = upload_model
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        if upload_model:
            self.logger.save_model(str(target), self.current_learning_iteration)

    def _restore_environment_progress(
        self,
        infos: dict[str, Any],
        *,
        load_cfg: dict[str, Any] | None,
    ) -> None:
        """Restore the global control-step counter for true training resumes.

        RSL-RL restores current_learning_iteration but mjlab constructs a fresh
        environment whose common_step_counter starts at zero. Environment-side
        curricula and interval events key off that counter, so resetting it silently
        restarts their schedules after a process restart.

        New checkpoints carry the exact counter. Legacy checkpoints reconstruct it
        from the restored iteration and rollout length, which is exact for the normal
        fixed-length on-policy training loop.
        """
        if load_cfg is not None and not bool(load_cfg.get("iteration", False)):
            # Playback/evaluation commonly request actor weights only. Do not make
            # those fresh environments inherit training-time event progress.
            return

        progress = infos.get("environment_progress")
        restored_steps: int | None = None
        if isinstance(progress, dict):
            raw_steps = progress.get("common_step_counter")
            if isinstance(raw_steps, int) and not isinstance(raw_steps, bool) and raw_steps >= 0:
                restored_steps = raw_steps

        if restored_steps is None:
            iteration = max(0, int(self.current_learning_iteration))
            steps_per_iteration = max(0, self._num_steps_per_env())
            restored_steps = iteration * steps_per_iteration

        self.env.unwrapped.common_step_counter = restored_steps

    def load(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Refuse incompatible checkpoints and restore resumable environment progress."""
        infos = super().load(*args, **kwargs)
        # Viewer/evaluation environments may be compiled from play=True
        # configs that add recorder or presentation terms. Those operational
        # terms are not part of the training ABI, so validate against the
        # registered canonical task topology instead of the mutated runtime
        # config. Training/resume configs already carry the task id and use
        # the same canonical registration here.
        runtime_cfg = self.env.unwrapped.cfg
        contract_cfg = canonical_task_cfg_for_runtime_cfg(runtime_cfg)
        require_current_checkpoint_contracts(infos, contract_cfg)
        load_cfg = kwargs.get("load_cfg")
        if load_cfg is not None and not isinstance(load_cfg, dict):
            raise TypeError("load_cfg must be a mapping when provided")
        self._restore_environment_progress(infos, load_cfg=load_cfg)
        return infos

    def initialize_from_compatible_actor(self, source_checkpoint: str | Path) -> dict[str, Any]:
        """Initialize only actor weights/normalization for a new signed task.

        This does not weaken normal checkpoint loading. Plant and structured
        action contracts must still match, then a separate actor-ABI comparison
        proves that the actor observation topology, action topology, control
        timestep, and reward-dt convention are identical. Critic, optimizer,
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
