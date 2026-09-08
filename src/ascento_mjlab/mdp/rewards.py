"""Physically meaningful balance and motion-quality reward terms."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ascento_mjlab.geometry import projected_gravity_tilt

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
    if std <= 0.0:
        raise ValueError("std must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    tilt = projected_gravity_tilt(asset.data.projected_gravity_b)
    return torch.exp(-torch.square(tilt) / (std * std))


def height_tracking(
    env: ManagerBasedRlEnv,
    target: float = 0.75,
    std: float = 0.08,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    if std <= 0.0:
        raise ValueError("std must be positive")
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
    if std <= 0.0:
        raise ValueError("std must be positive")
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


def position_hold(
    env: ManagerBasedRlEnv,
    std: float = 0.50,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward remaining near the actual supported reset position in balance."""
    if not hasattr(env, "ascento_balance_state"):
        raise RuntimeError("balance origin must be initialized before position_hold")
    if std <= 0.0:
        raise ValueError("std must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    displacement = asset.data.root_link_pos_w[:, :2] - env.ascento_balance_state["origin_xy"]
    return torch.exp(-torch.sum(torch.square(displacement), dim=1) / (std * std))


def leg_pose_symmetry_penalty(
    env: ManagerBasedRlEnv,
    beta: float = 0.15,
    joint_pairs: tuple[tuple[str, str], ...] = (
        ("left_hip", "right_hip"),
        ("left_knee", "right_knee"),
    ),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Softly penalize named mirrored leg-joint mismatch without coupling actions."""
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    joint_names = getattr(asset, "joint_names", None)
    if joint_names is None:
        raise RuntimeError("leg symmetry requires robot joint names")
    try:
        indices = [(joint_names.index(left), joint_names.index(right)) for left, right in joint_pairs]
    except ValueError as error:
        raise RuntimeError("leg symmetry joint pair is absent from the robot") from error
    deltas = torch.stack(
        [asset.data.joint_pos[:, left] - asset.data.joint_pos[:, right] for left, right in indices],
        dim=1,
    ).abs()
    huber = torch.where(
        deltas <= beta,
        0.5 * torch.square(deltas) / beta,
        deltas - 0.5 * beta,
    )
    return torch.mean(huber, dim=1)


def settled_balance(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward the same low-motion, supported state required by the balance gate."""
    asset: Entity = env.scene[asset_cfg.name]
    tilt_sq = torch.square(projected_gravity_tilt(asset.data.projected_gravity_b))
    planar_speed_sq = torch.sum(torch.square(asset.data.root_link_lin_vel_b[:, :2]), dim=1)
    angular_speed_sq = torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)
    height_error_sq = torch.square(asset.data.root_link_pos_w[:, 2] - 0.75)
    left = env.scene["left_wheel_contact"].data.found
    right = env.scene["right_wheel_contact"].data.found
    assert left is not None and right is not None
    supported = left.flatten(start_dim=1).any(dim=1) & right.flatten(start_dim=1).any(dim=1)
    score = torch.exp(-tilt_sq / 0.08**2)
    score *= torch.exp(-planar_speed_sq / 0.10**2)
    score *= torch.exp(-angular_speed_sq / 0.25**2)
    score *= torch.exp(-height_error_sq / 0.05**2)
    return score * supported.float()


def effort_penalty(
    env: ManagerBasedRlEnv,
    peak_effort_nm: float = 65.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    if peak_effort_nm <= 0.0:
        raise ValueError("peak_effort_nm must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    utilisation = asset.data.actuator_force[:, asset_cfg.actuator_ids] / peak_effort_nm
    return torch.mean(torch.square(utilisation), dim=1)


def effort_target_barrier(
    env: ManagerBasedRlEnv,
    peak_effort_nm: float = 65.0,
    soft_limit_fraction: float = 0.75,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize commanded effort entering the actuator saturation region.

    ``effort_penalty`` regularizes *applied* torque, which can remain moderate
    while the policy repeatedly requests clipped commands.  This smooth hinge
    exposes that hidden saturation to PPO without creating a discontinuity:
    requests below ``soft_limit_fraction`` are free and a request at the hard
    limit contributes one unit.
    """
    if peak_effort_nm <= 0.0:
        raise ValueError("peak_effort_nm must be positive")
    if not 0.0 < soft_limit_fraction < 1.0:
        raise ValueError("soft_limit_fraction must lie strictly between 0 and 1")
    asset: Entity = env.scene[asset_cfg.name]
    request = asset.data.joint_effort_target[:, asset_cfg.actuator_ids]
    utilisation = torch.abs(request) / peak_effort_nm
    excess = torch.relu(utilisation - soft_limit_fraction)
    normalized = excess / (1.0 - soft_limit_fraction)
    return torch.mean(torch.square(normalized), dim=1)


def action_rate_penalty(env: ManagerBasedRlEnv, reference_dt: float = 0.01) -> torch.Tensor:
    if reference_dt <= 0.0 or env.step_dt <= 0.0:
        raise ValueError("time steps must be positive")
    delta = env.action_manager.action - env.action_manager.prev_action
    scaled_delta = delta * (reference_dt / float(env.step_dt))
    return torch.mean(torch.square(scaled_delta), dim=1)


def track_velocity(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    if std <= 0.0:
        raise ValueError("std must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    error = torch.sum(torch.square(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]), dim=1)
    error += torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
    return torch.exp(-error / (std * std))


def track_linear_velocity_xy(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Track body-frame XY velocity; ``std`` is in m/s."""
    if std <= 0.0:
        raise ValueError("std must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    error_sq = torch.sum(torch.square(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-error_sq / (std * std))


def track_yaw_rate(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Track body yaw rate; ``std`` is in rad/s."""
    if std <= 0.0:
        raise ValueError("std must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    error = command[:, 2] - asset.data.root_link_ang_vel_b[:, 2]
    return torch.exp(-torch.square(error) / (std * std))


def _phase_weight(env: ManagerBasedRlEnv, values: tuple[float, ...]) -> torch.Tensor:
    from .jump import PHASE_RECOVERY

    if len(values) != PHASE_RECOVERY + 1:
        raise ValueError("phase-weight table must contain idle through recovery")
    phase = env.ascento_jump_state["phase"].clamp(min=0, max=PHASE_RECOVERY)
    cache = getattr(env, "_ascento_phase_weight_tables", None)
    if cache is None:
        cache = {}
        env._ascento_phase_weight_tables = cache
    table = cache.get(values)
    if table is None or table.device != env.device:
        table = torch.tensor(values, dtype=torch.float32, device=env.device)
        cache[values] = table
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


def jump_commanded_height_tracking(
    env: ManagerBasedRlEnv,
    command_name: str = "motion",
    std: float = 0.08,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Track the motion command's nominal height outside active jump motion."""
    if std <= 0.0:
        raise ValueError("std must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None and command.shape[1] >= 3
    error = asset.data.root_link_pos_w[:, 2] - command[:, 2]
    score = torch.exp(-torch.square(error) / (std * std))
    return score * _phase_weight(env, (1.0, 0.05, 0.0, 0.0, 0.20, 0.70))


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


def lateral_speed_penalty(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Penalize lateral slip without opposing a commanded forward velocity."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_lin_vel_b[:, 1])


def track_motion_forward_velocity(
    env: ManagerBasedRlEnv,
    command_name: str = "motion",
    std: float = 0.35,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    if std <= 0.0:
        raise ValueError("std must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None and command.shape[1] >= 1
    error = command[:, 0] - asset.data.root_link_lin_vel_b[:, 0]
    return torch.exp(-torch.square(error) / (std * std))


def track_motion_yaw_rate(
    env: ManagerBasedRlEnv,
    command_name: str = "motion",
    std: float = 0.40,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    if std <= 0.0:
        raise ValueError("std must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None and command.shape[1] >= 2
    error = command[:, 1] - asset.data.root_link_ang_vel_b[:, 2]
    return torch.exp(-torch.square(error) / (std * std))


def jump_crouch(
    env: ManagerBasedRlEnv,
    command_name: str = "motion",
    crouch_depth: float = 0.11,
    std: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward an actual lower body pose, not unsigned joint displacement."""
    from .jump import PHASE_CROUCH

    asset: Entity = env.scene[asset_cfg.name]
    if std <= 0.0 or crouch_depth <= 0.0:
        raise ValueError("std and crouch_depth must be positive")
    command = env.command_manager.get_command(command_name)
    assert command is not None and command.shape[1] >= 3
    target_height = command[:, 2] - crouch_depth
    error = asset.data.root_link_pos_w[:, 2] - target_height
    score = torch.exp(-torch.square(error) / (std * std))
    return score * (env.ascento_jump_state["phase"] == PHASE_CROUCH).float()


def jump_thrust(
    env: ManagerBasedRlEnv,
    command_name: str = "motion",
    velocity_scale: float = 1.0,
    std: float = 0.60,
    gravity: float = 9.81,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Track take-off speed implied by the commanded jump height."""
    from .jump import PHASE_THRUST

    if std <= 0.0 or gravity <= 0.0 or velocity_scale <= 0.0:
        raise ValueError("std, gravity and velocity_scale must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None and command.shape[1] >= 5
    target_vz = velocity_scale * torch.sqrt(2.0 * gravity * command[:, 4].clamp(min=0.01))
    error = asset.data.root_link_lin_vel_w[:, 2] - target_vz
    score = torch.exp(-torch.square(error) / (std * std))
    return score * (env.ascento_jump_state["phase"] == PHASE_THRUST).float()


def _event_impulse(env: ManagerBasedRlEnv, score: torch.Tensor) -> torch.Tensor:
    """Convert a one-step score to an impulse under dt-scaled aggregation."""
    dt = float(env.step_dt)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"env.step_dt must be positive and finite, got {dt}")
    return score / dt


def jump_takeoff(env: ManagerBasedRlEnv) -> torch.Tensor:
    from .jump import takeoff_bonus

    return _event_impulse(env, takeoff_bonus(env))


def jump_landing(env: ManagerBasedRlEnv) -> torch.Tensor:
    from .jump import landing_bonus

    return _event_impulse(env, landing_bonus(env))


def jump_recovered_landing(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Reward a landing only after the post-impact recovery condition is held."""
    return _event_impulse(env, env.ascento_jump_state["recovered_landing"])


def jump_post_landing_stability(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Dense proximity/dwell shaping while completing post-landing recovery.

    The strict ``recovered_landing`` event remains the only binary success
    signal.  This reward only supplies per-step credit during the recovery
    phase, increasing as the continuous stable hold progresses.
    """
    from .jump import PHASE_RECOVERY, RECOVERY_HOLD_S
    from .recovery import recovery_proximity

    state = env.ascento_jump_state
    in_recovery = (state["phase"] == PHASE_RECOVERY).float()
    proximity = recovery_proximity(env, asset_cfg=asset_cfg)
    hold_progress = (state["recovery_stable_time"] / RECOVERY_HOLD_S).clamp(0.0, 1.0)
    return in_recovery * proximity * (0.25 + 0.75 * hold_progress)


def jump_distance_tracking(env: ManagerBasedRlEnv, std: float = 0.08) -> torch.Tensor:
    """Score heading-relative landing displacement against the requested distance."""
    if std <= 0.0:
        raise ValueError("std must be positive")
    state = env.ascento_jump_state
    score = torch.exp(-torch.square(state["landing_distance_error"]) / (std * std))
    return _event_impulse(env, score * state["landing"])


def jump_landing_softness(env: ManagerBasedRlEnv, std: float = 1.0) -> torch.Tensor:
    """Reward low pre-contact vertical speed without using post-contact deceleration."""
    if std <= 0.0:
        raise ValueError("std must be positive")
    state = env.ascento_jump_state
    score = torch.exp(-torch.square(state["landing_preimpact_vz"]) / (std * std))
    return _event_impulse(env, score * state["landing"])


def airborne_height_progress(
    env: ManagerBasedRlEnv,
    command_name: str = "motion",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    state = env.ascento_jump_state
    height_gain = asset.data.root_link_pos_w[:, 2] - state["takeoff_height"]
    target_height = command[:, 4].clamp(min=0.05)
    progress = torch.clamp(height_gain / target_height, min=0.0, max=1.0)
    return progress * state["airborne"].float()
