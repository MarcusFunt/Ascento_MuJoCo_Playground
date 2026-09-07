"""Adaptive episode-horizon curriculum for long-horizon locomotion tasks."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

HORIZON_SCHEDULE_S = (20.0, 60.0, 120.0, 300.0)
"""Successive episode horizons used by balance and velocity training."""


class HorizonCurriculumRunner(MjlabOnPolicyRunner):
    """Adapt episode horizons while preserving the final long-horizon phase.

    The horizon advances after three consecutive 512-episode windows with at
    least 90% timeouts. Shorter practice stages can demote only after a grace
    period and sustained severe regression. Once the policy reaches 300 seconds,
    it remains there: stochastic PPO rollouts are useful learning data, but they
    are not authoritative grounds to throw away long-horizon practice. Stable
    300-second windows retain a candidate checkpoint; deterministic gate
    evaluation selects the final model.
    """

    env: RslRlVecEnvWrapper
    completion_window_episodes = 512
    required_success_windows = 3
    timeout_success_threshold = 0.90
    required_failure_windows = 4
    timeout_failure_threshold = 0.50
    minimum_stage_dwell_windows = 3
    required_top_horizon_candidate_windows = 3
    top_horizon_candidate_name = "model_best_long_horizon.pt"

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
        self._failing_windows = 0
        self._schedule_index = self._schedule_index_for(env.unwrapped.max_episode_length_s)
        self._stage_windows = 0
        self._top_horizon_success_windows = 0
        self._best_top_horizon_timeout_fraction = -1.0
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

    @property
    def _at_top_horizon(self) -> bool:
        return self._schedule_index == len(HORIZON_SCHEDULE_S) - 1

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
        transition = "held"
        was_top_horizon = self._at_top_horizon
        self._stage_windows += 1
        if timeout_fraction >= self.timeout_success_threshold:
            self._successful_windows += 1
            self._failing_windows = 0
        else:
            self._successful_windows = 0
            if timeout_fraction < self.timeout_failure_threshold:
                self._failing_windows += 1
            else:
                self._failing_windows = 0

        if was_top_horizon:
            if timeout_fraction >= self.timeout_success_threshold:
                self._top_horizon_success_windows += 1
            else:
                self._top_horizon_success_windows = 0

        if (
            self._successful_windows >= self.required_success_windows
            and not self._at_top_horizon
        ):
            self._schedule_index += 1
            self._apply_horizon()
            self._successful_windows = 0
            self._failing_windows = 0
            self._stage_windows = 0
            self._top_horizon_success_windows = 0
            transition = "promoted"
        elif (
            self._failing_windows >= self.required_failure_windows
            and self._schedule_index > 0
            and not was_top_horizon
            and self._stage_windows >= self.minimum_stage_dwell_windows
        ):
            self._schedule_index -= 1
            self._apply_horizon()
            self._successful_windows = 0
            self._failing_windows = 0
            self._stage_windows = 0
            self._top_horizon_success_windows = 0
            transition = "demoted"

        if was_top_horizon and self._failing_windows >= self.required_failure_windows:
            # Do not oscillate away from the hardest stage based on stochastic
            # training trajectories. Keep learning at 300 seconds and make the
            # regression visible for deterministic evaluation to adjudicate.
            self._failing_windows = 0
            transition = "protected"

        candidate_saved = False
        if (
            was_top_horizon
            and self._top_horizon_success_windows >= self.required_top_horizon_candidate_windows
            and timeout_fraction > self._best_top_horizon_timeout_fraction
        ):
            candidate_saved = self._save_top_horizon_candidate(timeout_fraction)
            if candidate_saved:
                self._best_top_horizon_timeout_fraction = timeout_fraction

        self._emit_status(
            timeout_fraction=timeout_fraction,
            transition=transition,
            candidate_saved=candidate_saved,
        )
        self._completed_in_window = 0
        self._timeouts_in_window = 0

    def _apply_horizon(self) -> None:
        self.env.unwrapped.cfg.episode_length_s = HORIZON_SCHEDULE_S[self._schedule_index]
        # RSL-RL consults this wrapper field for initial random episode lengths;
        # the environment derives its timeout dynamically from cfg above.
        self.env.max_episode_length = self.env.unwrapped.max_episode_length

    def _save_top_horizon_candidate(self, timeout_fraction: float) -> bool:
        """Persist the best training-derived long-horizon candidate atomically."""
        log_dir = getattr(self.logger, "log_dir", None)
        if not log_dir:
            return False
        output_dir = Path(log_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = output_dir / self.top_horizon_candidate_name
        temporary_checkpoint = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
        details = {
            "kind": "training_long_horizon_candidate",
            "horizon_s": HORIZON_SCHEDULE_S[-1],
            "timeout_fraction": timeout_fraction,
            "stable_windows": self._top_horizon_success_windows,
            "learning_iteration": self.current_learning_iteration,
            "selection": "requires deterministic balance_gate_v2 evaluation",
        }
        self.save(str(temporary_checkpoint), infos={"long_horizon_candidate": details})
        os.replace(temporary_checkpoint, checkpoint)

        metadata = output_dir / "long_horizon_candidate.json"
        temporary_metadata = metadata.with_suffix(metadata.suffix + ".tmp")
        temporary_metadata.write_text(
            json.dumps(details, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary_metadata, metadata)
        return True

    def _emit_status(
        self,
        *,
        timeout_fraction: float | None,
        transition: str = "held",
        candidate_saved: bool = False,
    ) -> None:
        parts = [
            f"horizon_s={self.env.unwrapped.max_episode_length_s:.1f}",
            f"stage={self._schedule_index + 1}",
            f"qualified_windows={self._successful_windows}",
            f"failed_windows={self._failing_windows}",
            f"stage_windows={self._stage_windows}",
            f"top_horizon_windows={self._top_horizon_success_windows}",
            f"transition={transition}",
        ]
        if timeout_fraction is not None:
            parts.append(f"timeout_fraction={timeout_fraction:.4f}")
        if candidate_saved:
            parts.append(f"candidate_checkpoint={self.top_horizon_candidate_name}")
        print("HORIZON_CURRICULUM " + " ".join(parts), flush=True)
