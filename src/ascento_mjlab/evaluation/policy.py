"""Policy abstraction for deterministic and future recurrent/multimodal models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class PolicyMetadata:
    kind: str
    deterministic: bool
    recurrent: bool
    checkpoint: str


class PolicyAdapter:
    """Small interface between evaluation and a concrete policy implementation."""

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        del env_ids

    def act(self, observations: Any) -> torch.Tensor:
        raise NotImplementedError

    def metadata(self) -> PolicyMetadata:
        raise NotImplementedError


class RslRlPolicyAdapter(PolicyAdapter):
    """RSL-RL actor adapter with explicit deterministic inference semantics."""

    def __init__(self, runner: Any, checkpoint: str | Path, *, deterministic: bool = True):
        self.runner = runner
        self.actor = runner.alg.get_policy()
        self.runner.alg.eval_mode()
        self.checkpoint = str(checkpoint)
        self.deterministic = deterministic
        self.recurrent = bool(getattr(self.actor, "is_recurrent", False))

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        reset = getattr(self.actor, "reset", None)
        if not callable(reset):
            return
        if env_ids is None:
            try:
                reset()
            except TypeError:
                return
            return

        # RSL-RL recurrent policies reset from a boolean done mask. Feed only the
        # selected env IDs so a completed vector slot cannot contaminate another.
        num_envs = int(getattr(self.runner.env, "num_envs", 0))
        if num_envs <= 0:
            return
        done = torch.zeros(num_envs, dtype=torch.bool, device=self.runner.device)
        done[env_ids] = True
        try:
            reset(done)
        except TypeError:
            reset(dones=done)

    def act(self, observations: Any) -> torch.Tensor:
        # MLPModel and recurrent RSL-RL models accept stochastic_output. Making the
        # mode explicit prevents evaluation from silently sampling the Gaussian.
        try:
            return self.actor(
                observations,
                stochastic_output=not self.deterministic,
            )
        except TypeError as exc:
            if self.deterministic:
                raise RuntimeError(
                    "Policy does not expose an explicit deterministic inference path; "
                    "refusing to evaluate ambiguously."
                ) from exc
            return self.actor(observations)

    def metadata(self) -> PolicyMetadata:
        return PolicyMetadata(
            kind=type(self.actor).__name__,
            deterministic=self.deterministic,
            recurrent=self.recurrent,
            checkpoint=self.checkpoint,
        )
