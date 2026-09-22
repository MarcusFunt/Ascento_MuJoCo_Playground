"""Adaptive episode-horizon curriculum for long-horizon locomotion tasks."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from mjlab.rl import RslRlVecEnvWrapper

from .geometry import projected_gravity_tilt
from .mdp.events import world_target_yaw, wrapped_angle_difference, yaw_from_quaternion_wxyz
from .provenance_runner import AscentoProvenanceRunner

HORIZON_SCHEDULE_S = (20.0, 60.0, 120.0, 300.0)
"""Successive episode horizons used by balance and velocity training."""


class HorizonCurriculumRunner(AscentoProvenanceRunner):
    """Adapt episode horizons while preserving the final long-horizon phase.

    The horizon advances after six consecutive 512-episode windows with at
    least 90% timeouts. Shorter practice stages can demote only after a grace
    period and sustained severe regression. Once the policy reaches 300 seconds,
    it remains there: stochastic PPO rollouts are useful learning data, but they
    are not authoritative grounds to throw away long-horizon practice. Stable
    300-second windows retain the best candidate checkpoint immediately;
    deterministic gate evaluation selects the final model.
    """

    env: RslRlVecEnvWrapper
    completion_window_episodes = 512
    # Prevent the observed post-promotion collapse: a short-lived timeout
    # streak is not enough evidence that the policy is ready for a longer
    # horizon. Promotion requires six consecutive qualifying windows.
    required_success_windows = 6
    timeout_success_threshold = 0.90
    quality_success_threshold = 0.90
    required_failure_windows = 4
    timeout_failure_threshold = 0.50
    minimum_stage_dwell_windows = 3
    # A policy can regress after its first successful 300-second window.  Keep
    # that first validated training candidate rather than requiring a streak
    # that may never complete after the regression has begun.
    required_top_horizon_candidate_windows = 1
    top_horizon_candidate_name = "model_best_long_horizon.pt"
    top_horizon_learning_rate = 1.0e-5
    selection_suite = "balance_gate_v5"
    stationary_min_samples = 50
    stationary_quality_thresholds = {
        "tilt_rms": 0.04,
        "planar_speed_rms": 0.03,
        "heading_error_rms": 0.03,
        "effort_rms": 3.2,
        "action_rate_rms": 0.01,
        "action_second_difference_rms": 0.01,
    }

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
        self._pending_completion_outcomes: list[bool] = []
        self._pending_quality_outcomes: list[bool] = []
        self._quality_passes_in_window = 0
        self._last_quality_fraction = 0.0
        self._stationary_quality_sums = torch.zeros(
            (env.num_envs, len(self.stationary_quality_thresholds)), device=env.device
        )
        self._stationary_quality_samples = torch.zeros(env.num_envs, device=env.device)
        self._base_step: Callable[[torch.Tensor], tuple[Any, torch.Tensor, torch.Tensor, dict]] = env.step
        env.step = self._step  # type: ignore[method-assign]
        if self._at_top_horizon:
            self._stabilize_top_horizon_optimizer()
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
        self._accumulate_stationary_quality(actions)
        observations, rewards, dones, extras = self._base_step(actions)
        timeouts = self.env.unwrapped.reset_time_outs
        done_mask = dones.bool()
        if bool(done_mask.any()):
            # Preserve environment-index order.  Reconstructing a vectorized
            # batch from aggregate counts can assign outcomes to the wrong
            # completion window at a boundary.
            done_ids = torch.nonzero(done_mask, as_tuple=False).flatten()
            if not hasattr(self, "_pending_quality_outcomes"):
                self._pending_quality_outcomes = []
            self._pending_quality_outcomes.extend(self._finish_quality_episodes(done_ids))
            self._record_completion_outcomes(timeouts[done_mask])
        return observations, rewards, dones, extras

    def _accumulate_stationary_quality(self, actions: torch.Tensor) -> None:
        """Accumulate evaluator-aligned quality metrics only while settled.

        A disturbance response is legitimate movement, so it must not block a
        horizon promotion merely by increasing action acceleration or speed.
        The same supported/near-equilibrium envelope as ``settled_balance``
        isolates the stationary portion where buzzing is harmful.
        """
        if not hasattr(self, "_stationary_quality_sums"):
            # Lightweight unit tests construct a runner without a simulator.
            return
        base_env = self.env.unwrapped
        robot = base_env.scene["robot"]
        tilt = projected_gravity_tilt(robot.data.projected_gravity_b)
        planar_speed = torch.linalg.vector_norm(robot.data.root_link_lin_vel_b[:, :2], dim=1)
        angular_speed = torch.linalg.vector_norm(robot.data.root_link_ang_vel_b, dim=1)
        heading_error = wrapped_angle_difference(
            world_target_yaw(base_env), yaw_from_quaternion_wxyz(robot.data.root_link_quat_w)
        )
        height_error = (robot.data.root_link_pos_w[:, 2] - 0.75).abs()
        left = base_env.scene["left_wheel_contact"].data.found.flatten(start_dim=1).any(dim=1)
        right = base_env.scene["right_wheel_contact"].data.found.flatten(start_dim=1).any(dim=1)
        stationary = (
            (tilt <= 0.08)
            & (planar_speed <= 0.10)
            & (angular_speed <= 0.25)
            & (heading_error.abs() <= 0.35)
            & (height_error <= 0.05)
            & left
            & right
        )
        if not bool(stationary.any()):
            return
        previous = base_env.action_manager.action
        previous_previous = base_env.action_manager.prev_action
        action_rate = torch.sqrt(torch.mean(torch.square(actions - previous), dim=1))
        action_second_difference = torch.sqrt(
            torch.mean(torch.square(actions - 2.0 * previous + previous_previous), dim=1)
        )
        effort = torch.sqrt(torch.mean(torch.square(robot.data.actuator_force), dim=1))
        values = torch.stack(
            (
                tilt,
                planar_speed,
                heading_error.abs(),
                effort,
                action_rate,
                action_second_difference,
            ),
            dim=1,
        )
        mask = stationary.unsqueeze(1)
        self._stationary_quality_sums += torch.where(mask, torch.square(values), 0.0)
        self._stationary_quality_samples += stationary.float()

    def _finish_quality_episodes(self, done_ids: torch.Tensor) -> list[bool]:
        """Convert accumulated per-environment stationarity into pass/fail outcomes."""
        if not hasattr(self, "_stationary_quality_sums"):
            return [True] * int(done_ids.numel())
        samples = self._stationary_quality_samples[done_ids]
        rms = torch.sqrt(
            self._stationary_quality_sums[done_ids] / samples.clamp_min(1.0).unsqueeze(1)
        )
        thresholds = torch.tensor(
            tuple(self.stationary_quality_thresholds.values()), device=rms.device, dtype=rms.dtype
        )
        passed = (samples >= self.stationary_min_samples) & torch.all(rms <= thresholds, dim=1)
        self._stationary_quality_sums[done_ids] = 0.0
        self._stationary_quality_samples[done_ids] = 0.0
        return [bool(value) for value in passed.cpu().tolist()]

    def _record_completion_outcomes(self, timeouts: torch.Tensor) -> None:
        """Queue ordered timeout outcomes and evaluate only complete windows."""
        if timeouts.ndim != 1:
            raise ValueError("completion outcomes must be a one-dimensional tensor")
        quality_queue = getattr(self, "_pending_quality_outcomes", None)
        if quality_queue is None:
            quality_queue = []
            self._pending_quality_outcomes = quality_queue
        timeout_values = [bool(value) for value in timeouts.cpu().tolist()]
        self._pending_completion_outcomes.extend(timeout_values)
        if len(quality_queue) < len(timeout_values):
            quality_queue.extend([True] * (len(timeout_values) - len(quality_queue)))
        while len(self._pending_completion_outcomes) >= self.completion_window_episodes:
            window = self._pending_completion_outcomes[: self.completion_window_episodes]
            del self._pending_completion_outcomes[: self.completion_window_episodes]
            quality_window = quality_queue[: self.completion_window_episodes]
            del quality_queue[: self.completion_window_episodes]
            self._completed_in_window = self.completion_window_episodes
            self._timeouts_in_window = sum(window)
            self._quality_passes_in_window = sum(quality_window)
            self._evaluate_completion_window()

    def _evaluate_completion_window(self) -> None:
        timeout_fraction = self._timeouts_in_window / self._completed_in_window
        quality_fraction = getattr(self, "_quality_passes_in_window", self._completed_in_window) / self._completed_in_window
        self._last_quality_fraction = quality_fraction
        quality_qualified = quality_fraction >= self.quality_success_threshold
        window_qualified = timeout_fraction >= self.timeout_success_threshold and quality_qualified
        transition = "held"
        was_top_horizon = self._at_top_horizon
        self._stage_windows += 1
        if window_qualified:
            self._successful_windows += 1
            self._failing_windows = 0
        else:
            self._successful_windows = 0
            if timeout_fraction < self.timeout_failure_threshold:
                self._failing_windows += 1
            else:
                self._failing_windows = 0

        if was_top_horizon:
            if window_qualified:
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
            and quality_qualified
        ):
            candidate_saved = self._save_top_horizon_candidate(timeout_fraction, quality_fraction)
            if candidate_saved:
                self._best_top_horizon_timeout_fraction = timeout_fraction

        self._emit_status(
            timeout_fraction=timeout_fraction,
            quality_fraction=quality_fraction,
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
        if self._at_top_horizon:
            self._stabilize_top_horizon_optimizer()

    def _stabilize_top_horizon_optimizer(self) -> None:
        """Freeze adaptive PPO rate changes for 300-second fine-tuning.

        The short-horizon stages may use RSL-RL's adaptive KL schedule to
        learn quickly.  At the final horizon, an adaptive increase after a
        low-KL minibatch can overwrite an already stable controller.  Lock the
        optimizer to the minimum supported rate when that stage begins.
        """
        algorithm = self.alg
        algorithm.schedule = "fixed"
        algorithm.learning_rate = min(float(algorithm.learning_rate), self.top_horizon_learning_rate)
        for param_group in algorithm.optimizer.param_groups:
            param_group["lr"] = algorithm.learning_rate

    def _save_top_horizon_candidate(self, timeout_fraction: float, quality_fraction: float) -> bool:
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
            "stationary_quality_pass_fraction": quality_fraction,
            "stable_windows": self._top_horizon_success_windows,
            "learning_iteration": self.current_learning_iteration,
            "selection": f"requires deterministic {self.selection_suite} evaluation",
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
        quality_fraction: float | None = None,
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
            f"control_steps={int(getattr(self.env.unwrapped, 'common_step_counter', 0))}",
            f"transition={transition}",
        ]
        if timeout_fraction is not None:
            parts.append(f"timeout_fraction={timeout_fraction:.4f}")
        if quality_fraction is not None:
            parts.append(f"quality_fraction={quality_fraction:.4f}")
        if candidate_saved:
            parts.append(f"candidate_checkpoint={self.top_horizon_candidate_name}")
        print("HORIZON_CURRICULUM " + " ".join(parts), flush=True)


class VelocityHorizonCurriculumRunner(HorizonCurriculumRunner):
    """Advance the velocity horizon only when commanded motion is tracked.

    Balance quality is defined around a fixed world heading and a 0.75 m root
    height.  Those are intentionally absent from ``Ascento-Velocity-Flat``:
    yaw rate and height are policy commands.  This runner therefore measures
    the same body-frame twist and root-height errors used by the velocity gate
    instead of inheriting balance's stationary-quality predicate.
    """

    tracking_min_samples = 50
    selection_suite = "velocity_gate_v1"
    tracking_quality_thresholds = {
        "velocity_tracking_rmse": 0.25,
        "height_tracking_rmse": 0.08,
    }

    def __init__(
        self,
        env: RslRlVecEnvWrapper,
        train_cfg: dict[str, Any],
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(env, train_cfg, log_dir, device)
        self._tracking_quality_sums = torch.zeros(
            (env.num_envs, len(self.tracking_quality_thresholds)), device=env.device
        )
        self._tracking_quality_samples = torch.zeros(env.num_envs, device=env.device)

    def _accumulate_stationary_quality(self, actions: torch.Tensor) -> None:
        """Accumulate command-tracking mean-square errors for every rollout step."""
        del actions
        if not hasattr(self, "_tracking_quality_sums"):
            # Lightweight unit tests may construct a runner without its normal
            # initializer.
            return
        base_env = self.env.unwrapped
        robot = base_env.scene["robot"]
        twist = base_env.command_manager.get_command("twist")
        height = base_env.command_manager.get_command("height")
        assert twist is not None and twist.shape[1] == 3
        assert height is not None and height.shape[1] == 1
        actual_twist = torch.stack(
            (
                robot.data.root_link_lin_vel_b[:, 0],
                robot.data.root_link_lin_vel_b[:, 1],
                robot.data.root_link_ang_vel_b[:, 2],
            ),
            dim=1,
        )
        velocity_error_sq = torch.mean(torch.square(actual_twist - twist), dim=1)
        height_error_sq = torch.square(robot.data.root_link_pos_w[:, 2] - height[:, 0])
        self._tracking_quality_sums += torch.stack((velocity_error_sq, height_error_sq), dim=1)
        self._tracking_quality_samples += 1.0

    def _finish_quality_episodes(self, done_ids: torch.Tensor) -> list[bool]:
        """Require adequate twist and height tracking before horizon promotion."""
        if not hasattr(self, "_tracking_quality_sums"):
            return [True] * int(done_ids.numel())
        samples = self._tracking_quality_samples[done_ids]
        rms = torch.sqrt(
            self._tracking_quality_sums[done_ids] / samples.clamp_min(1.0).unsqueeze(1)
        )
        thresholds = torch.tensor(
            tuple(self.tracking_quality_thresholds.values()), device=rms.device, dtype=rms.dtype
        )
        passed = (samples >= self.tracking_min_samples) & torch.all(rms <= thresholds, dim=1)
        self._tracking_quality_sums[done_ids] = 0.0
        self._tracking_quality_samples[done_ids] = 0.0
        return [bool(value) for value in passed.cpu().tolist()]
