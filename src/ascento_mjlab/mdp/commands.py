"""Ascento velocity and one-shot motion commands."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg


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
  """Six channels: vx, yaw, height, one-shot request, target height, distance."""

  def __init__(self, cfg: AscentoMotionCommandCfg, env) -> None:
    super().__init__(cfg, env)
    self._command = torch.zeros((self.num_envs, 6), device=self.device)
    self._jump_pulse = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self._pulse_is_new = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def _update_metrics(self) -> None:
    return

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    count = len(env_ids)
    samples = torch.rand((count, 6), device=self.device)
    self._command[env_ids, 0] = samples[:, 0].mul(self.cfg.vx_range[1] - self.cfg.vx_range[0]).add(self.cfg.vx_range[0])
    self._command[env_ids, 1] = samples[:, 1].mul(self.cfg.yaw_range[1] - self.cfg.yaw_range[0]).add(self.cfg.yaw_range[0])
    self._command[env_ids, 2] = samples[:, 2].mul(self.cfg.height_range[1] - self.cfg.height_range[0]).add(self.cfg.height_range[0])
    self._command[env_ids, 4] = samples[:, 3].mul(self.cfg.jump_height_range[1] - self.cfg.jump_height_range[0]).add(self.cfg.jump_height_range[0])
    self._command[env_ids, 5] = samples[:, 4].mul(self.cfg.jump_distance_range[1] - self.cfg.jump_distance_range[0]).add(self.cfg.jump_distance_range[0])
    self._jump_pulse[env_ids] = samples[:, 5] < self.cfg.jump_probability
    self._command[env_ids, 3] = self._jump_pulse[env_ids].float()
    self._pulse_is_new[env_ids] = True

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    if env_ids is None:
      # Keep a pulse sampled during this compute visible for the next policy
      # observation, then clear it on the following step.
      consumed = ~self._pulse_is_new
      self._command[consumed, 3] = 0.0
      self._pulse_is_new[:] = False
    else:
      # Reset keeps the freshly sampled pulse visible to the first policy action.
      self._command[env_ids, 3] = self._jump_pulse[env_ids].float()
      self._pulse_is_new[env_ids] = False


__all__ = ["AscentoMotionCommandCfg", "UniformVelocityCommandCfg"]
