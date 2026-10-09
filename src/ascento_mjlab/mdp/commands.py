"""Ascento velocity, height, and one-shot motion commands."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg


@dataclass(kw_only=True)
class AscentoHeightCommandCfg(CommandTermCfg):
    """Uniform scalar body-height command."""

    entity_name: str
    height_range: tuple[float, float] = (0.70, 0.80)

    def __post_init__(self) -> None:
        if self.height_range[0] > self.height_range[1]:
            raise ValueError("height_range must be ordered low <= high")

    def build(self, env) -> AscentoHeightCommand:
        return AscentoHeightCommand(self, env)


class AscentoHeightCommand(CommandTerm):
    """One-channel target body height in metres."""

    def __init__(self, cfg: AscentoHeightCommandCfg, env) -> None:
        super().__init__(cfg, env)
        self._command = torch.zeros((self.num_envs, 1), device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self) -> None:
        return

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        low, high = self.cfg.height_range
        samples = torch.rand(len(env_ids), device=self.device)
        self._command[env_ids, 0] = samples.mul(high - low).add(low)

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        del env_ids


@dataclass(kw_only=True)
class AscentoTargetSpeedCommandCfg(CommandTermCfg):
    """Bounded travel-speed command for one speed-conditioned locomotion policy."""

    entity_name: str
    max_speed_mps: float = 0.5
    max_speed_slew_rate_mps_per_s: float = 0.5
    standing_probability: float = 0.15

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_speed_mps) or self.max_speed_mps <= 0.0:
            raise ValueError("max_speed_mps must be finite and positive")
        if (
            not math.isfinite(self.max_speed_slew_rate_mps_per_s)
            or self.max_speed_slew_rate_mps_per_s <= 0.0
        ):
            raise ValueError("max_speed_slew_rate_mps_per_s must be finite and positive")
        if not 0.0 <= self.standing_probability <= 1.0:
            raise ValueError("standing_probability must be in [0, 1]")
        low, high = self.resampling_time_range
        if not math.isfinite(low) or not math.isfinite(high) or low <= 0.0 or high < low:
            raise ValueError("resampling_time_range must be finite, positive, and ordered")

    def build(self, env) -> AscentoTargetSpeedCommand:
        return AscentoTargetSpeedCommand(self, env)


class AscentoTargetSpeedCommand(CommandTerm):
    """One-channel nonnegative desired speed in m/s, resampled during training.

    Viser callbacks only enqueue user input. The queued value is copied to the
    device tensor from _update_command at the environment step boundary.
    """

    def __init__(self, cfg: AscentoTargetSpeedCommandCfg, env) -> None:
        super().__init__(cfg, env)
        self._command = torch.zeros((self.num_envs, 1), device=self.device)
        self._target_speed = torch.zeros_like(self._command)
        self._manual_override = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._pending_lock = threading.Lock()
        self._pending_speed: dict[int, float] = {}
        self._speed_display = None
        self._speed_display_env = 0
        self._last_display_update = 0.0

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _validate_speed(self, speed_mps: float) -> float:
        value = float(speed_mps)
        if not math.isfinite(value) or not 0.0 <= value <= self.cfg.max_speed_mps:
            raise ValueError(f"speed must be finite and within [0, {self.cfg.max_speed_mps}] m/s")
        return value

    def queue_speed_mps(self, speed_mps: float, *, env_id: int = 0) -> None:
        """Queue a Viser slider value for application on the next control step."""
        if not 0 <= int(env_id) < self.num_envs:
            raise IndexError("environment index is outside the speed command batch")
        value = self._validate_speed(speed_mps)
        with self._pending_lock:
            self._pending_speed[int(env_id)] = value

    def set_manual_speed_mps(self, speed_mps: float, *, env_id: int) -> None:
        """Set an evaluator-selected command on the current simulation thread."""
        if not 0 <= int(env_id) < self.num_envs:
            raise IndexError("environment index is outside the speed command batch")
        self._target_speed[int(env_id), 0] = self._validate_speed(speed_mps)
        self._manual_override[int(env_id)] = True

    def _update_metrics(self) -> None:
        if self._speed_display is None or time.monotonic() - self._last_display_update < 0.2:
            return
        env_id = self._speed_display_env
        robot = self._env.scene[self.cfg.entity_name]
        requested = float(self._target_speed[env_id, 0].item())
        applied = float(self._command[env_id, 0].item())
        achieved = float(
            torch.linalg.vector_norm(robot.data.root_link_lin_vel_w[env_id, :2]).item()
        )
        self._speed_display.content = (
            f"<div style='padding:0.25em'>Requested: <b>{requested:.2f} m/s</b>"
            f"<br>Applied command: <b>{applied:.2f} m/s</b>"
            f"<br>Achieved planar speed: <b>{achieved:.2f} m/s</b></div>"
        )
        self._last_display_update = time.monotonic()

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        eligible = env_ids[~self._manual_override[env_ids]]
        if len(eligible) == 0:
            return
        speeds = torch.rand(len(eligible), device=self.device) * self.cfg.max_speed_mps
        standing = torch.rand(len(eligible), device=self.device) < self.cfg.standing_probability
        self._target_speed[eligible, 0] = torch.where(standing, torch.zeros_like(speeds), speeds)

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        del env_ids
        with self._pending_lock:
            pending = self._pending_speed
            self._pending_speed = {}
        for env_id, speed_mps in pending.items():
            self.set_manual_speed_mps(speed_mps, env_id=env_id)

        max_step = self.cfg.max_speed_slew_rate_mps_per_s * float(self._env.step_dt)
        delta = self._target_speed - self._command
        self._command.add_(torch.clamp(delta, min=-max_step, max=max_step))

    def create_gui(self, name, server, get_env_idx, on_change=None, request_action=None) -> None:
        del name, on_change, request_action
        env_id = int(get_env_idx())
        self._speed_display_env = env_id
        with server.gui.add_folder("Runtime speed"):
            self._speed_display = server.gui.add_html("<div>Waiting for speed telemetry…</div>")
            slider = server.gui.add_slider(
                "Desired travel speed (m/s)",
                min=0.0,
                max=float(self.cfg.max_speed_mps),
                step=max(0.001, float(self.cfg.max_speed_mps) / 100.0),
                initial_value=float(self._command[env_id, 0].item()),
                hint="One policy supports the full trained range; zero requests a controlled stop.",
            )

            @slider.on_update
            def _(_) -> None:
                self.queue_speed_mps(float(slider.value), env_id=int(get_env_idx()))


@dataclass(kw_only=True)
class AscentoMotionCommandCfg(CommandTermCfg):
    """Combined velocity/height/jump target with a one-step jump pulse."""

    entity_name: str
    jump_probability: float = 0.5
    vx_range: tuple[float, float] = (0.0, 0.0)
    yaw_range: tuple[float, float] = (0.0, 0.0)
    height_range: tuple[float, float] = (0.70, 0.80)
    jump_height_range: tuple[float, float] = (0.15, 0.25)
    jump_distance_range: tuple[float, float] = (0.0, 0.30)

    def __post_init__(self) -> None:
        if not 0.0 <= self.jump_probability <= 1.0:
            raise ValueError("jump_probability must be in [0, 1]")

    def build(self, env) -> AscentoMotionCommand:
        return AscentoMotionCommand(self, env)


class AscentoMotionCommand(CommandTerm):
    """Six channels: vx, yaw, height, one-shot request, target height, distance.

    ``jump_generation`` is a monotonically increasing per-environment counter.
    It lets downstream jump semantics observe every request exactly once even if
    the visible request pulse has already been cleared by command-manager timing.
    """

    def __init__(self, cfg: AscentoMotionCommandCfg, env) -> None:
        super().__init__(cfg, env)
        self._command = torch.zeros((self.num_envs, 6), device=self.device)
        self._jump_pulse = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._pulse_is_new = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._jump_generation = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    @property
    def jump_generation(self) -> torch.Tensor:
        return self._jump_generation

    def _update_metrics(self) -> None:
        return

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        count = len(env_ids)
        samples = torch.rand((count, 6), device=self.device)
        self._command[env_ids, 0] = (
            samples[:, 0].mul(self.cfg.vx_range[1] - self.cfg.vx_range[0]).add(self.cfg.vx_range[0])
        )
        self._command[env_ids, 1] = (
            samples[:, 1]
            .mul(self.cfg.yaw_range[1] - self.cfg.yaw_range[0])
            .add(self.cfg.yaw_range[0])
        )
        self._command[env_ids, 2] = (
            samples[:, 2]
            .mul(self.cfg.height_range[1] - self.cfg.height_range[0])
            .add(self.cfg.height_range[0])
        )
        self._command[env_ids, 4] = (
            samples[:, 3]
            .mul(self.cfg.jump_height_range[1] - self.cfg.jump_height_range[0])
            .add(self.cfg.jump_height_range[0])
        )
        self._command[env_ids, 5] = (
            samples[:, 4]
            .mul(self.cfg.jump_distance_range[1] - self.cfg.jump_distance_range[0])
            .add(self.cfg.jump_distance_range[0])
        )
        self._jump_pulse[env_ids] = samples[:, 5] < self.cfg.jump_probability
        self._command[env_ids, 3] = self._jump_pulse[env_ids].float()
        self._pulse_is_new[env_ids] = True
        pulse_ids = env_ids[self._jump_pulse[env_ids]]
        self._jump_generation[pulse_ids] += 1

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None:
            consumed = ~self._pulse_is_new
            self._command[consumed, 3] = 0.0
            self._pulse_is_new[:] = False
        else:
            self._command[env_ids, 3] = self._jump_pulse[env_ids].float()
            self._pulse_is_new[env_ids] = False


__all__ = [
    "AscentoHeightCommand",
    "AscentoHeightCommandCfg",
    "AscentoMotionCommand",
    "AscentoMotionCommandCfg",
    "AscentoTargetSpeedCommand",
    "AscentoTargetSpeedCommandCfg",
    "UniformVelocityCommandCfg",
]
