"""Physically meaningful balance and motion-quality reward terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def alive(env: ManagerBasedRlEnv) -> torch.Tensor:
  return (~env.termination_manager.terminated).float()


def upright(env: ManagerBasedRlEnv, std: float = 0.35, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  tilt_sq = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return torch.exp(-tilt_sq / (std * std))


def height_tracking(
  env: ManagerBasedRlEnv,
  target: float = 0.75,
  std: float = 0.08,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  error = asset.data.root_link_pos_w[:, 2] - target
  return torch.exp(-torch.square(error) / (std * std))


def angular_rate_penalty(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def effort_penalty(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.mean(torch.square(asset.data.actuator_force[:, asset_cfg.actuator_ids]), dim=1)


def action_rate_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.mean(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def track_velocity(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  std: float = 0.5,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  error = torch.sum(torch.square(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]), dim=1)
  error += torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
  return torch.exp(-error / (std * std))


def jump_takeoff(env: ManagerBasedRlEnv) -> torch.Tensor:
  from .jump import takeoff_bonus

  return takeoff_bonus(env)


def jump_landing(env: ManagerBasedRlEnv) -> torch.Tensor:
  from .jump import landing_bonus

  return landing_bonus(env)


def airborne_height_progress(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  target = asset.data.root_link_pos_w[:, 2] - 0.75
  target_height = command[:, 4].clamp(min=0.05)
  progress = torch.clamp(target / target_height, min=0.0, max=1.0)
  return progress * env.ascento_jump_state["airborne"].float()
