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


def upright(
    env: ManagerBasedRlEnv,
    std: float = 0.35,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
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


def commanded_height_tracking(
    env: ManagerBasedRlEnv,
    command_name: str = "height",
    std: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Track a scalar body-height command without a competing fixed-height term."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None and command.shape[1] == 1
    error = asset.data.root_link_pos_w[:, 2] - command[:, 0]
    return torch.exp(-torch.square(error) / (std * std))


def angular_rate_penalty(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def planar_speed_penalty(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Penalize horizontal drift while still allowing wheel motion for recovery."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_link_lin_vel_b[:, :2]), dim=1)


def effort_penalty(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.mean(torch.square(asset.data.actuator_force[:, asset_cfg.actuator_ids]), dim=1)


def action_rate_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    return torch.mean(
        torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
    )


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


def _phase_weight(env: ManagerBasedRlEnv, values: tuple[float, ...]) -> torch.Tensor:
    from .jump import PHASE_RECOVERY

    if len(values) != PHASE_RECOVERY + 1:
        raise ValueError("phase-weight table must contain idle through recovery")
    phase = env.ascento_jump_state["phase"].clamp(min=0, max=PHASE_RECOVERY)
    table = torch.tensor(values, dtype=torch.float32, device=env.device)
    return table[phase]


def jump_nominal_height_tracking(
    env: ManagerBasedRlEnv,
    target: float = 0.75,
    std: float = 0.08,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Use nominal-height pressure only where it does not fight the jump motion."""
    weight = _phase_weight(env, (1.0, 0.05, 0.0, 0.0, 0.20, 0.70))
    return height_tracking(env, target=target, std=std, asset_cfg=asset_cfg) * weight


def jump_angular_rate_penalty(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    weight = _phase_weight(env, (1.0, 0.8, 0.5, 0.7, 1.0, 1.0))
    return angular_rate_penalty(env, asset_cfg=asset_cfg) * weight


def jump_planar_speed_penalty(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    weight = _phase_weight(env, (1.0, 0.5, 0.2, 0.0, 0.3, 1.0))
    return planar_speed_penalty(env, asset_cfg=asset_cfg) * weight


def jump_crouch(
    env: ManagerBasedRlEnv,
    target_height: float = 0.64,
    std: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward an actual lower body pose, not unsigned joint displacement."""
    from .jump import PHASE_CROUCH

    asset: Entity = env.scene[asset_cfg.name]
    error = asset.data.root_link_pos_w[:, 2] - target_height
    score = torch.exp(-torch.square(error) / (std * std))
    return score * (env.ascento_jump_state["phase"] == PHASE_CROUCH).float()


def jump_thrust(
    env: ManagerBasedRlEnv,
    target_vz: float = 1.2,
    std: float = 0.7,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    from .jump import PHASE_THRUST

    asset: Entity = env.scene[asset_cfg.name]
    error = asset.data.root_link_lin_vel_w[:, 2] - target_vz
    score = torch.exp(-torch.square(error) / (std * std))
    return score * (env.ascento_jump_state["phase"] == PHASE_THRUST).float()


def jump_takeoff(env: ManagerBasedRlEnv) -> torch.Tensor:
    from .jump import takeoff_bonus

    return takeoff_bonus(env)


def jump_landing(env: ManagerBasedRlEnv) -> torch.Tensor:
    from .jump import landing_bonus

    return landing_bonus(env)


def jump_distance_tracking(env: ManagerBasedRlEnv, std: float = 0.08) -> torch.Tensor:
    """Score heading-relative landing displacement against the requested distance."""
    state = env.ascento_jump_state
    score = torch.exp(-torch.square(state["landing_distance_error"]) / (std * std))
    return score * state["landing"]


def jump_landing_softness(env: ManagerBasedRlEnv, std: float = 1.0) -> torch.Tensor:
    """Reward low pre-contact vertical speed without using post-contact deceleration."""
    state = env.ascento_jump_state
    score = torch.exp(-torch.square(state["landing_preimpact_vz"]) / (std * std))
    return score * state["landing"]


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
