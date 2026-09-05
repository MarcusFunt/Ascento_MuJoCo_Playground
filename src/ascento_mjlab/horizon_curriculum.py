"""Adaptive episode-horizon curriculum for long-horizon locomotion tasks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper


HORIZON_SCHEDULE_S = (20.0, 60.0, 120.0, 300.0)
"""Successive episode horizons used by balance and velocity training."""


class HorizonCurriculumRunner(MjlabOnPolicyRunner):
    """Promote a run after sustained timeout-based episode success.

    Each completion window contains a fixed number of finished vector slots.
    The horizon advances only after three consecutive windows have at least 90%
    timeouts, keeping frequent randomized resets while the policy is unstable.
    """

    env: RslRlVecEnvWrapper
    completion_window_episodes = 512
    required_success_windows = 3
    timeout_success_threshold = 0.90

    def __init__(
        self,
        env: RslRlVecEnvWrapper,
        train_cfg: dict[str, Any],
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(env, train_cfg, log_dir, device)
        self._completed_in_window = 0
        self._timeouts_in_window = 0
        self._successful_windows = 0
        self._schedule_index = self._schedule_index_for(env.unwrapped.max_episode_length_s)
        self._base_step: Callable[[torch.Tensor], tuple[Any, torch.Tensor, torch.Tensor, dict]] = env.step
        env.step = self._step  # type: ignore[method-assign]
        self._emit_status(timeout_fraction=None)

    @staticmethod
    def _schedule_index_for(horizon_s: float) -> int:
        """Start at the first configured horizon not below the requested value."""
        for index, candidate in enumerate(HORIZON_SCHEDULE_S):
            if horizon_s <= candidate:
                return index
        return len(HORIZON_SCHEDULE_S) - 1

    def _step(self, actions: torch.Tensor) -> tuple[Any, torch.Tensor, torch.Tensor, dict]:
        observations, rewards, dones, extras = self._base_step(actions)
        timeouts = self.env.unwrapped.reset_time_outs
        completed = int(dones.sum().item())
        if completed:
            self._completed_in_window += completed
            self._timeouts_in_window += int((timeouts & dones.bool()).sum().item())
            if self._completed_in_window >= self.completion_window_episodes:
                self._evaluate_completion_window()
        return observations, rewards, dones, extras

    def _evaluate_completion_window(self) -> None:
        timeout_fraction = self._timeouts_in_window / self._completed_in_window
        if timeout_fraction >= self.timeout_success_threshold:
            self._successful_windows += 1
        else:
            self._successful_windows = 0

        if (
            self._successful_windows >= self.required_success_windows
            and self._schedule_index < len(HORIZON_SCHEDULE_S) - 1
        ):
            self._schedule_index += 1
            self.env.unwrapped.cfg.episode_length_s = HORIZON_SCHEDULE_S[self._schedule_index]
            # RSL-RL consults this wrapper field for initial random episode
            # lengths; the environment itself derives timeout length dynamically
            # from cfg.episode_length_s on every step.
            self.env.max_episode_length = self.env.unwrapped.max_episode_length
            self._successful_windows = 0

        self._emit_status(timeout_fraction=timeout_fraction)
        self._completed_in_window = 0
        self._timeouts_in_window = 0

    def _emit_status(self, *, timeout_fraction: float | None) -> None:
        parts = [
            f"horizon_s={self.env.unwrapped.max_episode_length_s:.1f}",
            f"stage={self._schedule_index + 1}",
            f"qualified_windows={self._successful_windows}",
        ]
        if timeout_fraction is not None:
            parts.append(f"timeout_fraction={timeout_fraction:.4f}")
        print("HORIZON_CURRICULUM " + " ".join(parts), flush=True)
