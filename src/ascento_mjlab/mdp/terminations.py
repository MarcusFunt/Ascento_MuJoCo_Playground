"""Failure and numerical-health termination terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def fallen(
    env: ManagerBasedRlEnv,
    min_height: float = 0.35,
    max_gravity_z: float = -0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    # MuJoCo gravity expressed in an upright body frame points down, so its
    # normalized Z component is approximately -1 while a 90-degree fall tends
    # toward 0.  Falling therefore means gravity_z rises above the threshold.
    return (asset.data.root_link_pos_w[:, 2] < min_height) | (
        asset.data.projected_gravity_b[:, 2] > max_gravity_z
    )


def excessive_velocity(
    env: ManagerBasedRlEnv,
    max_linear: float = 12.0,
    max_angular: float = 25.0,
    max_joint: float = 40.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return (
        (torch.linalg.vector_norm(asset.data.root_link_lin_vel_w, dim=1) > max_linear)
        | (torch.linalg.vector_norm(asset.data.root_link_ang_vel_w, dim=1) > max_angular)
        | (torch.amax(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1) > max_joint)
    )


def nonfinite(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    values = (
        asset.data.joint_pos,
        asset.data.joint_vel,
        asset.data.root_link_pos_w,
        asset.data.root_link_quat_w,
    )
    return torch.stack(
        [~torch.isfinite(value).all(dim=tuple(range(1, value.ndim))) for value in values], dim=0
    ).any(dim=0)
