"""Observation terms owned by the Ascento task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from .events import (
    world_target_xy,
    world_target_yaw,
    wrapped_angle_difference,
    yaw_from_quaternion_wxyz,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def base_height(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2:3]


def actuator_effort(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.actuator_force[:, asset_cfg.actuator_ids]


def world_target_error_body(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Express the world-frame target displacement in the robot's yaw frame.

    The target remains an absolute, per-environment world coordinate. Rotating
    only this observation prevents a yaw-randomized policy from relearning the
    same return-to-target behavior for every world orientation.
    """
    asset: Entity = env.scene[asset_cfg.name]
    error_w = world_target_xy(env) - asset.data.root_link_pos_w[:, :2]
    quat_wxyz = asset.data.root_link_quat_w
    w, x, y, z = quat_wxyz.unbind(dim=1)
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y.square() + z.square())
    return torch.stack(
        [
            cos_yaw * error_w[:, 0] + sin_yaw * error_w[:, 1],
            -sin_yaw * error_w[:, 0] + cos_yaw * error_w[:, 1],
        ],
        dim=1,
    )


def world_target_heading_error(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Return signed target-minus-current yaw error in the principal branch.

    The target is the robot's supported reset yaw, not a fixed global heading.
    This makes the observation invariant to the reset yaw distribution while
    giving the policy a direct signal that distinguishes yaw drift from a
    stationary world-position target.
    """
    asset: Entity = env.scene[asset_cfg.name]
    current_yaw = yaw_from_quaternion_wxyz(asset.data.root_link_quat_w)
    return wrapped_angle_difference(world_target_yaw(env), current_yaw).unsqueeze(1)


def wheel_contacts(
    env: ManagerBasedRlEnv, left_sensor_name: str, right_sensor_name: str
) -> torch.Tensor:
    values = []
    for name in (left_sensor_name, right_sensor_name):
        sensor = env.scene[name]
        assert isinstance(sensor, ContactSensor)
        assert sensor.data.found is not None
        values.append(
            (sensor.data.found > 0).float().flatten(start_dim=1).amax(dim=1, keepdim=True)
        )
    return torch.cat(values, dim=1)


def wheel_contact_forces(
    env: ManagerBasedRlEnv, left_sensor_name: str, right_sensor_name: str
) -> torch.Tensor:
    values = []
    for name in (left_sensor_name, right_sensor_name):
        sensor = env.scene[name]
        assert isinstance(sensor, ContactSensor)
        assert sensor.data.force is not None
        values.append(
            torch.linalg.vector_norm(sensor.data.force, dim=-1)
            .flatten(start_dim=1)
            .amax(dim=1, keepdim=True)
        )
    return torch.cat(values, dim=1)


def motion_command(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    assert command is not None
    return command


def jump_state(env: ManagerBasedRlEnv) -> torch.Tensor:
    from .jump import phase_features

    return phase_features(env)
