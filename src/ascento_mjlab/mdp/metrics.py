"""Motion-quality metrics for selection and diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def tilt_radians(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    gravity_xy = torch.linalg.vector_norm(asset.data.projected_gravity_b[:, :2], dim=1)
    return torch.atan2(gravity_xy, -asset.data.projected_gravity_b[:, 2].clamp(max=-1.0e-6))


def commanded_effort(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Mean absolute effort requested by the action layer, in Nm."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.mean(torch.abs(asset.data.joint_effort_target[:, asset_cfg.actuator_ids]), dim=1)


def actuator_output_effort(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Mean absolute actuator-space output after the motor model, in Nm."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.mean(torch.abs(asset.data.actuator_force[:, asset_cfg.actuator_ids]), dim=1)


def joint_applied_effort(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Mean absolute generalized actuator force applied at the robot joints."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.mean(torch.abs(asset.data.qfrc_actuator[:, asset_cfg.joint_ids]), dim=1)


def applied_effort(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Backward-compatible alias for :func:`joint_applied_effort`."""
    return joint_applied_effort(env, asset_cfg)


def root_speed(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.linalg.vector_norm(asset.data.root_link_lin_vel_w, dim=1)
