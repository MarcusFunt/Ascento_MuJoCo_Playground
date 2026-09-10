"""Motion-quality metrics for selection and diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def controller_requested_effort(asset: Entity) -> torch.Tensor:
    """Return pre-motor controller requests in canonical robot joint order."""
    actuators = asset.actuators
    if len(actuators) != 2:
        raise RuntimeError("Ascento structured control requires leg and wheel actuator groups")
    leg = actuators[0].controller_requested_effort
    wheel = actuators[1].controller_requested_effort
    return torch.stack(
        (leg[:, 0], leg[:, 1], wheel[:, 0], leg[:, 2], leg[:, 3], wheel[:, 1]), dim=1
    )


def tilt_radians(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    gravity_xy = torch.linalg.vector_norm(asset.data.projected_gravity_b[:, :2], dim=1)
    return torch.atan2(gravity_xy, -asset.data.projected_gravity_b[:, 2].clamp(max=-1.0e-6))


def commanded_effort(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Mean absolute torque requested by the structured controllers, in Nm."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.mean(
        torch.abs(controller_requested_effort(asset)[:, asset_cfg.actuator_ids]), dim=1
    )


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
