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


def applied_effort(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.mean(torch.abs(asset.data.actuator_force[:, asset_cfg.actuator_ids]), dim=1)


def root_speed(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.linalg.vector_norm(asset.data.root_link_lin_vel_w, dim=1)
