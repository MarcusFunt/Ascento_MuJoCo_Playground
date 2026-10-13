"""Version-specific optional actor-only handoff for the widened V5 task."""

from __future__ import annotations

import os
from typing import Any

from .provenance_runner import AscentoProvenanceRunner

_INITIAL_ACTOR_ENV = "ASCENTO_V5_INITIAL_ACTOR_CHECKPOINT"


class AscentoGoalSettleV5Runner(AscentoProvenanceRunner):
    """Copy compatible actor/normalizer before PPO starts; never copy optimizer.

    This is an opt-in, versioned training initialization, not a task resume.
    Log/checkpoint provenance retains the source checkpoint SHA-256 and the
    distinction between copied actor weights and freshly initialized critic.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._v5_initial_lineage: dict[str, Any] | None = None
        source = os.environ.get(_INITIAL_ACTOR_ENV, "").strip()
        if not source:
            return
        cfg = self.env.unwrapped.cfg
        if cfg.task_id != "Ascento-Generalist-Locomotion-Goal-Settle-V5-Flat":
            raise ValueError(f"{_INITIAL_ACTOR_ENV} is only supported for V5")
        if bool(self.cfg.get("resume", False)):
            raise ValueError(f"{_INITIAL_ACTOR_ENV} cannot be combined with PPO resume")
        lineage = self.initialize_from_compatible_actor(source)
        if lineage.get("fresh") != ["critic", "optimizer", "iteration", "environment_state"]:
            raise ValueError(
                "actor handoff must reset critic, optimizer, iteration and environment"
            )
        self._v5_initial_lineage = lineage
        print(
            "[V5 actor-only handoff] copied actor+normalizer; "
            f"fresh critic/optimizer/iteration; source sha256={lineage['source_checkpoint_sha256']}",
            flush=True,
        )

    def save(self, path: str, infos: dict[str, Any] | None = None) -> None:
        lineage = getattr(self, "_v5_initial_lineage", None)
        if lineage is not None:
            infos = {**(infos or {}), "lineage": lineage}
        super().save(path, infos=infos)
