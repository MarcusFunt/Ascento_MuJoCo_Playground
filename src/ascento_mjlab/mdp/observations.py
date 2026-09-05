"""Observation terms owned by the Ascento task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def base_height(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2:3]


def actuator_effort(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.actuator_force[:, asset_cfg.actuator_ids]


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
