"""Reset and perturbation presets for Ascento tasks."""

from __future__ import annotations

import math

import torch
from mjlab.envs.mdp.events import reset_root_state_uniform
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

DEFAULT_WHEEL_RADIUS_M = 0.25
DEFAULT_WHEEL_HALF_WIDTH_M = 0.0025


def linear_curriculum_value(
    control_step: int,
    ramp_control_steps: int,
    start: float,
    end: float,
) -> float:
    # Clamp step progress so resumed or extended runs stay at the final stage.
    if ramp_control_steps <= 0:
        raise ValueError("ramp_control_steps must be positive")
    if not math.isfinite(float(start)) or not math.isfinite(float(end)):
        raise ValueError("curriculum values must be finite")
    progress = min(1.0, max(0.0, float(control_step) / float(ramp_control_steps)))
    return float(start) + (float(end) - float(start)) * progress


GENERALIST_SHORT_TARGET_MAX_DISTANCE_M = 0.50
GENERALIST_MEDIUM_TARGET_MAX_DISTANCE_M = 1.50
GENERALIST_TARGET_ARRIVAL_DISTANCE_M = 0.035


def generalist_curriculum_values(
    control_step: int,
    *,
    ramp_control_steps: int = 24_000,
    start_gate_like_fraction: float = 0.10,
    final_gate_like_fraction: float = 0.25,
) -> dict[str, float]:
    """Resolve the fixed stratified-distance envelope and gate-like schedule."""
    if ramp_control_steps <= 0:
        raise ValueError("ramp_control_steps must be positive")
    if not 0.0 <= start_gate_like_fraction <= final_gate_like_fraction <= 1.0:
        raise ValueError("gate-like curriculum fractions must satisfy 0 <= start <= final <= 1")
    progress = min(1.0, max(0.0, float(control_step) / float(ramp_control_steps)))
    return {
        "progress": progress,
        "target_min_distance_m": 0.15,
        "target_max_distance_m": 3.0,
        "gate_like_fraction": linear_curriculum_value(
            control_step,
            ramp_control_steps,
            start_gate_like_fraction,
            final_gate_like_fraction,
        ),
    }


def generalist_goal_mix_fractions(
    stage: int,
    *,
    medium_goal_fraction: float = 0.25,
    initial_long_goal_fraction: float = 0.10,
    max_long_goal_fraction: float = 0.40,
) -> dict[str, float]:
    """Return short/medium/long shares while preserving a short-goal floor."""
    long_goal_fractions = (
        initial_long_goal_fraction,
        (initial_long_goal_fraction + max_long_goal_fraction) / 2.0,
        max_long_goal_fraction,
    )
    if stage < 0 or stage >= len(long_goal_fractions):
        raise ValueError("generalist goal-mix stage must be 0, 1, or 2")
    long_fraction = long_goal_fractions[stage]
    short_fraction = 1.0 - medium_goal_fraction - long_fraction
    if not 0.0 < medium_goal_fraction < 1.0 or short_fraction <= 0.0:
        raise ValueError("goal-mix fractions must leave positive short, medium, and long shares")
    return {
        "short": short_fraction,
        "medium": medium_goal_fraction,
        "long": long_fraction,
    }


def sample_stratified_goal_distances(
    num_samples: int,
    *,
    device: torch.device | str,
    long_goal_fraction: float,
    medium_goal_fraction: float = 0.25,
    short_min_distance_m: float = 0.15,
    short_max_distance_m: float = 0.35,
    medium_min_distance_m: float = 0.50,
    medium_max_distance_m: float = 1.50,
    long_min_distance_m: float = 2.0,
    long_max_distance_m: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw a categorical distance mix and return distances plus 0/1/2 band ids."""
    if not 0.0 < medium_goal_fraction < 1.0:
        raise ValueError("medium_goal_fraction must be between zero and one")
    if not 0.0 <= long_goal_fraction < 1.0 - medium_goal_fraction:
        raise ValueError("long_goal_fraction must leave positive short and medium shares")
    if not 0.0 < short_min_distance_m <= short_max_distance_m:
        raise ValueError("short target distance requires 0 < min <= max")
    if not 0.0 < medium_min_distance_m <= medium_max_distance_m:
        raise ValueError("medium target distance requires 0 < min <= max")
    if not 0.0 < long_min_distance_m <= long_max_distance_m:
        raise ValueError("long target distance requires 0 < min <= max")
    short_goal_fraction = 1.0 - medium_goal_fraction - long_goal_fraction
    unit = torch.rand(num_samples, device=device)
    bands = torch.where(
        unit < short_goal_fraction,
        torch.zeros(num_samples, dtype=torch.long, device=device),
        torch.where(
            unit < short_goal_fraction + medium_goal_fraction,
            torch.ones(num_samples, dtype=torch.long, device=device),
            torch.full((num_samples,), 2, dtype=torch.long, device=device),
        ),
    )
    minima = torch.tensor(
        [short_min_distance_m, medium_min_distance_m, long_min_distance_m],
        dtype=torch.float32,
        device=device,
    )
    maxima = torch.tensor(
        [short_max_distance_m, medium_max_distance_m, long_max_distance_m],
        dtype=torch.float32,
        device=device,
    )
    distances = minima[bands] + torch.rand(num_samples, device=device) * (
        maxima[bands] - minima[bands]
    )
    return distances, bands


def wilson_lower_bound(successes: int, trials: int, *, z: float = 1.96) -> float:
    if trials <= 0:
        return 0.0
    rate = successes / trials
    z_sq = z * z
    denominator = 1.0 + z_sq / trials
    center = rate + z_sq / (2.0 * trials)
    margin = z * math.sqrt(rate * (1.0 - rate) / trials + z_sq / (4.0 * trials * trials))
    return (center - margin) / denominator


def advance_generalist_goal_mix_stage(
    stage: int,
    *,
    gate_recovery_successes: int,
    gate_episodes: int,
    short_arrival_successes: int,
    short_episodes: int,
    long_arrival_successes: int,
    long_episodes: int,
    minimum_episodes: int = 64,
    gate_recovery_lcb_threshold: float = 0.85,
    target_arrival_lcb_threshold: float = 0.50,
) -> int:
    """Advance the long-goal share only after Wilson-bound quality evidence."""
    if stage >= 2:
        return stage
    if minimum_episodes <= 0:
        raise ValueError("minimum_episodes must be positive")
    if gate_episodes < minimum_episodes or short_episodes < minimum_episodes:
        return stage
    if wilson_lower_bound(gate_recovery_successes, gate_episodes) < gate_recovery_lcb_threshold:
        return stage
    if wilson_lower_bound(short_arrival_successes, short_episodes) < target_arrival_lcb_threshold:
        return stage
    if stage == 1:
        if long_episodes < minimum_episodes:
            return stage
        if wilson_lower_bound(long_arrival_successes, long_episodes) < target_arrival_lcb_threshold:
            return stage
    return stage + 1


def generalist_goal_mix_state(
    env,
    *,
    initial_long_goal_fraction: float = 0.10,
    medium_goal_fraction: float = 0.25,
    max_long_goal_fraction: float = 0.40,
) -> dict[str, int | float]:
    """Shared run-local curriculum state used by reset sampling and the scheduler."""
    state = getattr(env, "ascento_generalist_goal_mix", None)
    if state is None:
        if not 0.0 <= initial_long_goal_fraction < max_long_goal_fraction < 1.0:
            raise ValueError("long-goal shares must increase within [0, 1)")
        if not 0.0 < medium_goal_fraction < 1.0 - max_long_goal_fraction:
            raise ValueError("goal-mix fractions must leave a positive short-goal share")
        state = {
            "stage": 0,
            "long_goal_fraction": float(initial_long_goal_fraction),
            "medium_goal_fraction": float(medium_goal_fraction),
            "initial_long_goal_fraction": float(initial_long_goal_fraction),
            "max_long_goal_fraction": float(max_long_goal_fraction),
            "gate_episodes": 0,
            "gate_recovery_successes": 0,
            "short_episodes": 0,
            "short_arrival_successes": 0,
            "long_episodes": 0,
            "long_arrival_successes": 0,
        }
        env.ascento_generalist_goal_mix = state
    return state


def generalist_episode_metrics_state(env) -> dict[str, torch.Tensor]:
    """Return device-resident episode outcome state used by generalist metrics."""
    state = getattr(env, "ascento_generalist_episode_metrics", None)
    if state is None:
        state = {
            "initial_target_distance_m": torch.zeros(
                env.num_envs, dtype=torch.float32, device=env.device
            ),
            "gate_like": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "arrived": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "recovery_completed": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "heading_error_at_arrival_rad": torch.zeros(
                env.num_envs, dtype=torch.float32, device=env.device
            ),
            "sampled_target_band_counts": torch.zeros(
                (env.num_envs, 3), dtype=torch.float32, device=env.device
            ),
        }
        env.ascento_generalist_episode_metrics = state
    return state


def reset_generalist_episode_metrics(
    env,
    env_ids: torch.Tensor | slice | None,
    *,
    initial_target_distance_m: torch.Tensor | None = None,
) -> None:
    """Reset slice outcomes for selected episodes and optionally record their sampled distance."""
    state = generalist_episode_metrics_state(env)
    ids = _resolved_env_ids(env, env_ids)
    state["initial_target_distance_m"][ids] = 0.0
    state["gate_like"][ids] = False
    state["arrived"][ids] = False
    state["recovery_completed"][ids] = False
    state["heading_error_at_arrival_rad"][ids] = 0.0
    state["sampled_target_band_counts"][ids] = 0.0
    if initial_target_distance_m is not None:
        state["initial_target_distance_m"][ids] = initial_target_distance_m.to(
            dtype=torch.float32, device=env.device
        ).reshape(-1)


def _scheduled_target_distance_range(
    env,
    *,
    final_minimum_m: float,
    final_maximum_m: float,
    start_minimum_m: float | None,
    start_maximum_m: float | None,
    ramp_control_steps: int | None,
) -> tuple[float, float]:
    if start_minimum_m is None and start_maximum_m is None:
        return final_minimum_m, final_maximum_m
    if start_minimum_m is None or start_maximum_m is None:
        raise ValueError("both curriculum target distance endpoints must be provided")
    if ramp_control_steps is None or ramp_control_steps <= 0:
        raise ValueError("curriculum_ramp_control_steps must be positive")
    if not 0.0 < start_minimum_m <= start_maximum_m:
        raise ValueError("curriculum start target distance requires 0 < min <= max")
    if not 0.0 < final_minimum_m <= final_maximum_m:
        raise ValueError("curriculum final target distance requires 0 < min <= max")
    if start_minimum_m > final_minimum_m or start_maximum_m > final_maximum_m:
        raise ValueError("target-distance curriculum must expand from near to far")
    control_step = int(getattr(env, "common_step_counter", 0))
    return (
        linear_curriculum_value(control_step, ramp_control_steps, start_minimum_m, final_minimum_m),
        linear_curriculum_value(control_step, ramp_control_steps, start_maximum_m, final_maximum_m),
    )


def _resolved_env_ids(env, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
    all_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    if env_ids is None:
        return all_ids
    if isinstance(env_ids, slice):
        return all_ids[env_ids]
    return env_ids.reshape(-1).to(dtype=torch.long, device=env.device)


def yaw_from_quaternion_wxyz(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    """Return the world-frame yaw for WXYZ quaternions.

    Balance targets intentionally retain the yaw that was sampled at reset, so
    an arbitrary world heading remains valid without making rotation free.
    """
    w, x, y, z = quaternion_wxyz.unbind(dim=-1)
    return torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y.square() + z.square()),
    )


def wrapped_angle_difference(target: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    """Return ``target - current`` wrapped to the closed yaw principal branch."""
    difference = target - current
    return torch.atan2(torch.sin(difference), torch.cos(difference))


def _world_target_state(env, *, asset_name: str = "robot") -> dict[str, torch.Tensor]:
    """Return initialized position-and-heading target state for every world."""
    asset = env.scene[asset_name]
    state = getattr(env, "ascento_world_target_state", None)
    if state is None:
        state = {}
        env.ascento_world_target_state = state
    if "target_xy" not in state:
        state["target_xy"] = asset.data.root_link_pos_w[:, :2].clone()
    if "target_yaw" not in state:
        state["target_yaw"] = yaw_from_quaternion_wxyz(asset.data.root_link_quat_w).clone()
    return state


def flat_ground_wheel_bottom_heights(
    env,
    *,
    asset_name: str = "robot",
    wheel_radius_m: float = DEFAULT_WHEEL_RADIUS_M,
    wheel_half_width_m: float = DEFAULT_WHEEL_HALF_WIDTH_M,
    env_ids: torch.Tensor | slice | None = None,
) -> torch.Tensor:
    """Return true outer-tire bottom heights above the flat support plane.

    The wheel collision geom is a cylinder whose axis is the wheel body's local Y
    axis.  A tilted cylinder does not extend a full radius in world Z, so simply
    subtracting ``wheel_radius_m`` from the wheel-centre height overestimates its
    downward extent.  The support extent of a cylinder with axis ``a`` is
    ``h*|a_z| + r*sqrt(1-a_z^2)``.
    """
    ids = _resolved_env_ids(env, env_ids)
    asset = env.scene[asset_name]
    try:
        left_id = asset.body_names.index("left_wheel")
        right_id = asset.body_names.index("right_wheel")
    except ValueError as exc:
        raise RuntimeError("Expected left_wheel and right_wheel bodies") from exc

    wheel_ids = torch.tensor(
        [left_id, right_id], dtype=torch.long, device=asset.data.body_link_pos_w.device
    )
    positions = asset.data.body_link_pos_w.index_select(0, ids).index_select(1, wheel_ids)
    quaternions = asset.data.body_link_quat_w.index_select(0, ids).index_select(1, wheel_ids)

    local_axis = torch.zeros_like(positions)
    local_axis[..., 1] = 1.0
    axis_w = quat_apply(quaternions, local_axis)
    axis_z = axis_w[..., 2].abs().clamp(max=1.0)
    radial_z = torch.sqrt(torch.clamp(1.0 - axis_z.square(), min=0.0))
    vertical_extent = float(wheel_half_width_m) * axis_z + float(wheel_radius_m) * radial_z
    return positions[..., 2] - vertical_extent


def reset_root_state_supported(
    env,
    env_ids: torch.Tensor | slice | None,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
    wheel_radius_m: float = DEFAULT_WHEEL_RADIUS_M,
    wheel_half_width_m: float = DEFAULT_WHEEL_HALF_WIDTH_M,
    support_clearance_m: float = 0.0,
) -> None:
    """Sample a root reset, then vertically align its lowest wheel with flat ground.

    The generic mjlab root reset perturbs orientation while leaving the nominal root
    height unchanged. For a wheel-legged robot that can introduce accidental wheel
    penetration or unsupported hovering before the first physics step. This helper
    preserves the sampled pose/velocity but translates the root in Z so the true
    lowest point of the outer wheel cylinder starts on the flat support plane.
    """
    reset_root_state_uniform(
        env,
        env_ids,
        pose_range=pose_range,
        velocity_range=velocity_range,
        asset_cfg=asset_cfg,
    )
    ids = _resolved_env_ids(env, env_ids)
    if ids.numel() == 0:
        return

    env.sim.forward()
    env.sim.sense()
    asset = env.scene[asset_cfg.name]
    bottoms = flat_ground_wheel_bottom_heights(
        env,
        asset_name=asset_cfg.name,
        wheel_radius_m=wheel_radius_m,
        wheel_half_width_m=wheel_half_width_m,
        env_ids=ids,
    )
    correction = float(support_clearance_m) - bottoms.amin(dim=1)
    pose = asset.data.root_link_pose_w.index_select(0, ids).clone()
    pose[:, 2] += correction
    asset.write_root_link_pose_to_sim(pose, env_ids=ids)
    env.sim.forward()
    env.sim.sense()


def mixed_balance_recovery_reset(
    env,
    env_ids: torch.Tensor | slice | None,
    *,
    asset_cfg: SceneEntityCfg,
    normal_pose_range: dict[str, tuple[float, float]],
    normal_velocity_range: dict[str, tuple[float, float]],
    hard_pose_range_start: dict[str, tuple[float, float]],
    hard_pose_range_end: dict[str, tuple[float, float]],
    hard_velocity_range_start: dict[str, tuple[float, float]],
    hard_velocity_range_end: dict[str, tuple[float, float]],
    hard_fraction_start: float = 0.10,
    hard_fraction_end: float = 0.30,
    ramp_control_steps: int = 120_000,
) -> None:
    """Mix ordinary balance resets with a progressive hard-recovery subset."""
    if not (0.0 <= hard_fraction_start <= 1.0 and 0.0 <= hard_fraction_end <= 1.0):
        raise ValueError("hard reset fractions must lie in [0, 1]")
    if ramp_control_steps <= 0:
        raise ValueError("ramp_control_steps must be positive")
    ids = _resolved_env_ids(env, env_ids)
    if ids.numel() == 0:
        return
    progress = min(
        1.0, max(0.0, float(getattr(env, "common_step_counter", 0)) / ramp_control_steps)
    )
    hard_fraction = hard_fraction_start + (hard_fraction_end - hard_fraction_start) * progress
    hard_mask = torch.rand(ids.numel(), device=env.device) < hard_fraction
    normal_ids = ids[~hard_mask]
    hard_ids = ids[hard_mask]
    if normal_ids.numel():
        reset_root_state_supported(
            env,
            normal_ids,
            pose_range=normal_pose_range,
            velocity_range=normal_velocity_range,
            asset_cfg=asset_cfg,
        )
    if hard_ids.numel():

        def lerp_ranges(start, end):
            keys = set(start) | set(end)
            resolved = {}
            for key in keys:
                a0, a1 = start.get(key, end[key])
                b0, b1 = end.get(key, start[key])
                resolved[key] = (a0 + (b0 - a0) * progress, a1 + (b1 - a1) * progress)
            return resolved

        reset_root_state_supported(
            env,
            hard_ids,
            pose_range=lerp_ranges(hard_pose_range_start, hard_pose_range_end),
            velocity_range=lerp_ranges(hard_velocity_range_start, hard_velocity_range_end),
            asset_cfg=asset_cfg,
        )


def reset_to_default_supported(
    env,
    env_ids: torch.Tensor | slice | None = None,
    *,
    asset_name: str = "robot",
    wheel_radius_m: float = DEFAULT_WHEEL_RADIUS_M,
    wheel_half_width_m: float = DEFAULT_WHEEL_HALF_WIDTH_M,
) -> None:
    """Reset selected worlds to the nominal, support-aligned robot state.

    This is deliberately deterministic: it bypasses randomized reset events,
    restores the configured root/joint defaults, and then aligns the actual
    wheel geometry to the plane.  Plant/controller probes use it to separate
    control behavior from reset-distribution behavior.
    """
    ids = _resolved_env_ids(env, env_ids)
    if ids.numel() == 0:
        return
    asset = env.scene[asset_name]
    default_root = asset.data.default_root_state.index_select(0, ids).clone()
    default_root[:, :3] += env.scene.env_origins.index_select(0, ids)
    asset.write_root_link_pose_to_sim(default_root[:, :7], env_ids=ids)
    asset.write_root_link_velocity_to_sim(default_root[:, 7:13], env_ids=ids)
    if asset.is_articulated:
        asset.write_joint_state_to_sim(
            asset.data.default_joint_pos.index_select(0, ids).clone(),
            asset.data.default_joint_vel.index_select(0, ids).clone(),
            env_ids=ids,
        )
    env.sim.forward()
    env.sim.sense()

    bottoms = flat_ground_wheel_bottom_heights(
        env,
        asset_name=asset_name,
        wheel_radius_m=wheel_radius_m,
        wheel_half_width_m=wheel_half_width_m,
        env_ids=ids,
    )
    pose = asset.data.root_link_pose_w.index_select(0, ids).clone()
    pose[:, 2] -= bottoms.amin(dim=1)
    asset.write_root_link_pose_to_sim(pose, env_ids=ids)
    env.sim.forward()
    env.sim.sense()
    initialize_world_target(env, ids, asset_name=asset_name)


def initialize_world_target(
    env,
    env_ids: torch.Tensor | slice | None = None,
    *,
    asset_name: str = "robot",
) -> None:
    """Set each reset slot's world-frame position and heading targets.

    The balance task randomizes its initial XY position slightly.  Tracking the
    actual post-reset position, rather than an environment-grid origin, keeps the
    target free of reset-dependent bias. The matching reset yaw is retained as
    a heading target: a balance policy must not spin while holding position,
    yet each yaw-randomized reset remains equally valid. Navigation can later
    replace both targets with the next directional world-frame gate without
    changing observations or rewards.
    """
    ids = _resolved_env_ids(env, env_ids)
    if ids.numel() == 0:
        return
    asset = env.scene[asset_name]
    state = _world_target_state(env, asset_name=asset_name)
    state["target_xy"][ids] = asset.data.root_link_pos_w[ids, :2]
    state["target_yaw"][ids] = yaw_from_quaternion_wxyz(asset.data.root_link_quat_w[ids])


def initialize_random_world_target(
    env,
    env_ids: torch.Tensor | slice | None = None,
    *,
    asset_name: str = "robot",
    min_distance_m: float = 0.15,
    max_distance_m: float = 0.35,
    arena_half_extent_m: float = 0.65,
    curriculum_start_min_target_distance_m: float | None = None,
    curriculum_start_max_target_distance_m: float | None = None,
    curriculum_ramp_control_steps: int | None = None,
    record_training_metrics: bool = False,
    stratified_goal_mix: bool = False,
    short_min_distance_m: float = 0.15,
    short_max_distance_m: float = 0.35,
    medium_min_distance_m: float = 0.50,
    medium_max_distance_m: float = 1.50,
    long_min_distance_m: float = 2.0,
    long_max_distance_m: float = 3.0,
    initial_long_goal_fraction: float = 0.10,
    medium_goal_fraction: float = 0.25,
    max_long_goal_fraction: float = 0.40,
) -> None:
    """Assign each selected environment an immediate bounded random XY target.

    Locomotion training needs task-relevant reward from the first rollout step.
    Targets remain close enough for the existing proximity reward to stay
    informative, while the arena bound prevents repeated target sampling from
    turning into an unbounded random walk across neighbouring cloned worlds.
    """
    ids = _resolved_env_ids(env, env_ids)
    if ids.numel() == 0:
        return
    asset = env.scene[asset_name]
    if stratified_goal_mix:
        mix = generalist_goal_mix_state(
            env,
            initial_long_goal_fraction=initial_long_goal_fraction,
            medium_goal_fraction=medium_goal_fraction,
            max_long_goal_fraction=max_long_goal_fraction,
        )
        if record_training_metrics:
            reset_generalist_episode_metrics(env, ids)
        _set_stratified_random_world_targets(
            env,
            asset,
            ids,
            arena_half_extent_m=arena_half_extent_m,
            long_goal_fraction=float(mix["long_goal_fraction"]),
            medium_goal_fraction=medium_goal_fraction,
            short_min_distance_m=short_min_distance_m,
            short_max_distance_m=short_max_distance_m,
            medium_min_distance_m=medium_min_distance_m,
            medium_max_distance_m=medium_max_distance_m,
            long_min_distance_m=long_min_distance_m,
            long_max_distance_m=long_max_distance_m,
            record_training_metrics=record_training_metrics,
        )
    else:
        min_distance_m, max_distance_m = _scheduled_target_distance_range(
            env,
            final_minimum_m=min_distance_m,
            final_maximum_m=max_distance_m,
            start_minimum_m=curriculum_start_min_target_distance_m,
            start_maximum_m=curriculum_start_max_target_distance_m,
            ramp_control_steps=curriculum_ramp_control_steps,
        )
        _set_bounded_random_world_targets(
            env,
            asset,
            ids,
            min_distance_m=min_distance_m,
            max_distance_m=max_distance_m,
            arena_half_extent_m=arena_half_extent_m,
        )
    if record_training_metrics:
        state = generalist_episode_metrics_state(env)
        state["initial_target_distance_m"][ids] = torch.linalg.vector_norm(
            world_target_xy(env)[ids] - asset.data.root_link_pos_w[ids, :2], dim=1
        )


def world_target_xy(env) -> torch.Tensor:
    """Return the current per-environment world-frame XY target.

    Observation-manager construction queries terms before reset events run. In
    that phase, seed the target from the current robot pose so construction and
    an initial observation are finite; the reset event replaces it with the
    exact supported pose before the first rollout step.
    """
    return _world_target_state(env)["target_xy"]


def world_target_yaw(env) -> torch.Tensor:
    """Return the reset-relative world-frame heading target for each environment."""
    return _world_target_state(env)["target_yaw"]


class OneShotPlanarVelocityPush:
    """Apply one gate-shaped planar velocity kick per episode.

    The built-in interval push samples all six root-velocity components every
    few seconds. That makes a 300-second training episode a qualitatively
    different task from the balance gate, which applies one cardinal planar
    disturbance. This class uses EventManager's interval scheduling for the
    first trigger, then ignores subsequent triggers until reset.
    """

    def __init__(self, cfg, env) -> None:
        del cfg
        self._pushed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            self._pushed.fill_(False)
        else:
            self._pushed[env_ids] = False

    def __call__(
        self,
        env,
        env_ids: torch.Tensor | None,
        *,
        min_delta_v: float = 0.15,
        max_delta_v: float = 0.45,
        asset_cfg: SceneEntityCfg,
    ) -> None:
        if not (0.0 < min_delta_v <= max_delta_v):
            raise ValueError("planar push requires 0 < min_delta_v <= max_delta_v")
        ids = _resolved_env_ids(env, env_ids)
        ids = ids[~self._pushed[ids]]
        if ids.numel() == 0:
            return

        asset = env.scene[asset_cfg.name]
        velocity = asset.data.root_link_vel_w[ids].clone()
        magnitudes = torch.empty(ids.numel(), device=env.device).uniform_(min_delta_v, max_delta_v)
        directions = torch.randint(0, 4, (ids.numel(),), device=env.device)
        delta = torch.zeros((ids.numel(), 3), dtype=velocity.dtype, device=env.device)
        axes = directions.remainder(2)
        signs = torch.where(directions < 2, 1.0, -1.0).to(dtype=velocity.dtype)
        delta[torch.arange(ids.numel(), device=env.device), axes] = magnitudes * signs
        velocity[:, :3] += delta
        asset.write_root_link_velocity_to_sim(velocity, env_ids=ids)
        self._pushed[ids] = True


class RepeatedRandomWorldTargetSequence:
    """Mix sustained waypoint travel with gate-like retarget-and-stop episodes.

    A target is replaced only after the robot is both close to it and settled
    for a short dwell. This keeps the locomotion objective dense without
    rewarding fly-through behavior, and provides many movement attempts per
    long episode instead of a single target step. A configurable subset of
    episodes instead receives one mild planar push, one short forward target,
    and then holds that target for the remainder of the episode.
    """

    def __init__(self, cfg, env) -> None:
        params = getattr(cfg, "params", {}) if cfg is not None else {}
        self._env = env
        self._stratified_goal_mix = bool(params.get("stratified_goal_mix", False))
        self._minimum_episodes_per_stage = int(params.get("minimum_episodes_per_stage", 64))
        self._goal_mix_state = generalist_goal_mix_state(
            env,
            initial_long_goal_fraction=float(params.get("initial_long_goal_fraction", 0.10)),
            medium_goal_fraction=float(params.get("medium_goal_fraction", 0.25)),
            max_long_goal_fraction=float(params.get("max_long_goal_fraction", 0.40)),
        )
        self._settled_at_target_s = torch.zeros(
            env.num_envs, dtype=torch.float32, device=env.device
        )
        self._episode_elapsed_s = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self._episode_initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._gate_like = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._gate_pushed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._gate_retargeted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._gate_recovery_stable_time_s = torch.zeros(
            env.num_envs, dtype=torch.float32, device=env.device
        )
        self._push = OneShotPlanarVelocityPush(None, env)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            self._settled_at_target_s.zero_()
            self._episode_elapsed_s.zero_()
            self._episode_initialized.zero_()
            self._gate_like.zero_()
            self._gate_pushed.zero_()
            self._gate_retargeted.zero_()
            self._gate_recovery_stable_time_s.zero_()
            self._push.reset()
            return
        ids = _resolved_env_ids_placeholder(
            env_ids, self._settled_at_target_s.device, self._settled_at_target_s.numel()
        )
        self._settled_at_target_s[ids] = 0.0
        self._episode_elapsed_s[ids] = 0.0
        self._episode_initialized[ids] = False
        self._gate_like[ids] = False
        self._gate_pushed[ids] = False
        self._gate_retargeted[ids] = False
        self._gate_recovery_stable_time_s[ids] = 0.0
        self._push.reset(ids)

    def __call__(
        self,
        env,
        env_ids: torch.Tensor | slice | None,
        *,
        target_reached_distance_m: float = 0.04,
        target_hold_s: float = 0.35,
        min_target_distance_m: float = 0.15,
        max_target_distance_m: float = 0.35,
        curriculum_start_min_target_distance_m: float | None = None,
        curriculum_start_max_target_distance_m: float | None = None,
        curriculum_start_gate_like_fraction: float | None = None,
        curriculum_ramp_control_steps: int | None = None,
        arena_half_extent_m: float = 0.65,
        gate_like_fraction: float = 0.25,
        track_training_metrics: bool = False,
        stratified_goal_mix: bool = False,
        short_min_distance_m: float = 0.15,
        short_max_distance_m: float = 0.35,
        medium_min_distance_m: float = 0.50,
        medium_max_distance_m: float = 1.50,
        long_min_distance_m: float = 2.0,
        long_max_distance_m: float = 3.0,
        medium_goal_fraction: float = 0.25,
        initial_long_goal_fraction: float = 0.10,
        max_long_goal_fraction: float = 0.40,
        minimum_episodes_per_stage: int = 64,
        gate_recovery_lcb_threshold: float = 0.85,
        target_arrival_lcb_threshold: float = 0.50,
        gate_push_time_s: float = 4.0,
        gate_min_delta_v: float = 0.05,
        gate_max_delta_v: float = 0.15,
        gate_retarget_time_s: float = 9.0,
        gate_min_target_distance_m: float = 0.10,
        gate_max_target_distance_m: float = 0.20,
        asset_cfg: SceneEntityCfg,
    ) -> None:
        if target_reached_distance_m <= 0.0:
            raise ValueError("target_reached_distance_m must be positive")
        if target_hold_s <= 0.0:
            raise ValueError("target_hold_s must be positive")
        if not 0.0 <= gate_like_fraction <= 1.0:
            raise ValueError("gate_like_fraction must be in [0, 1]")
        if stratified_goal_mix:
            mix = generalist_goal_mix_state(
                env,
                initial_long_goal_fraction=initial_long_goal_fraction,
                medium_goal_fraction=medium_goal_fraction,
                max_long_goal_fraction=max_long_goal_fraction,
            )
            long_goal_fraction = float(mix["long_goal_fraction"])
        else:
            min_target_distance_m, max_target_distance_m = _scheduled_target_distance_range(
                env,
                final_minimum_m=min_target_distance_m,
                final_maximum_m=max_target_distance_m,
                start_minimum_m=curriculum_start_min_target_distance_m,
                start_maximum_m=curriculum_start_max_target_distance_m,
                ramp_control_steps=curriculum_ramp_control_steps,
            )
        current_gate_like_fraction = gate_like_fraction
        if curriculum_start_gate_like_fraction is not None:
            if not 0.0 <= curriculum_start_gate_like_fraction <= gate_like_fraction:
                raise ValueError(
                    "curriculum start gate-like fraction must be between zero and final fraction"
                )
            if curriculum_ramp_control_steps is None or curriculum_ramp_control_steps <= 0:
                raise ValueError("curriculum_ramp_control_steps must be positive")
            current_gate_like_fraction = linear_curriculum_value(
                int(getattr(env, "common_step_counter", 0)),
                curriculum_ramp_control_steps,
                curriculum_start_gate_like_fraction,
                gate_like_fraction,
            )
        if gate_push_time_s < 0.0 or gate_retarget_time_s < gate_push_time_s:
            raise ValueError("gate retarget time must be no earlier than the push time")
        if not 0.0 < gate_min_delta_v <= gate_max_delta_v:
            raise ValueError("gate push requires 0 < min_delta_v <= max_delta_v")
        if not 0.0 < gate_min_target_distance_m <= gate_max_target_distance_m:
            raise ValueError("gate target distance requires 0 < min <= max")

        ids = _resolved_env_ids(env, env_ids)
        if ids.numel() == 0:
            return
        asset = env.scene[asset_cfg.name]
        new_ids = ids[~self._episode_initialized[ids]]
        if new_ids.numel() > 0:
            self._gate_like[new_ids] = (
                torch.rand(new_ids.numel(), device=env.device) < current_gate_like_fraction
            )
            self._episode_initialized[new_ids] = True
            if track_training_metrics:
                generalist_episode_metrics_state(env)["gate_like"][new_ids] = self._gate_like[
                    new_ids
                ]
        self._episode_elapsed_s[ids] += float(env.step_dt)

        gate_ids = ids[self._gate_like[ids]]
        push_ids = gate_ids[
            (self._episode_elapsed_s[gate_ids] >= gate_push_time_s) & ~self._gate_pushed[gate_ids]
        ]
        if push_ids.numel() > 0:
            self._push(
                env,
                push_ids,
                min_delta_v=gate_min_delta_v,
                max_delta_v=gate_max_delta_v,
                asset_cfg=asset_cfg,
            )
            self._gate_pushed[push_ids] = True

        retarget_ids = gate_ids[
            (self._episode_elapsed_s[gate_ids] >= gate_retarget_time_s)
            & ~self._gate_retargeted[gate_ids]
        ]
        if retarget_ids.numel() > 0:
            distances = torch.empty(retarget_ids.numel(), device=env.device).uniform_(
                gate_min_target_distance_m, gate_max_target_distance_m
            )
            forward_b = torch.zeros((retarget_ids.numel(), 3), device=env.device)
            forward_b[:, 0] = 1.0
            forward_w = quat_apply(asset.data.root_link_quat_w[retarget_ids], forward_b)
            forward_xy = forward_w[:, :2]
            forward_xy /= torch.linalg.vector_norm(forward_xy, dim=1, keepdim=True).clamp_min(
                1.0e-6
            )
            state = _world_target_state(env, asset_name=asset_cfg.name)
            state["target_xy"][retarget_ids] = (
                asset.data.root_link_pos_w[retarget_ids, :2] + distances.unsqueeze(1) * forward_xy
            )
            state["target_yaw"][retarget_ids] = yaw_from_quaternion_wxyz(
                asset.data.root_link_quat_w[retarget_ids]
            )
            self._gate_retargeted[retarget_ids] = True
            if track_training_metrics:
                state = generalist_episode_metrics_state(env)
                state["arrived"][retarget_ids] = False
                state["recovery_completed"][retarget_ids] = False
                state["heading_error_at_arrival_rad"][retarget_ids] = 0.0
                state["sampled_target_band_counts"][retarget_ids, 0] += 1.0
                self._gate_recovery_stable_time_s[retarget_ids] = 0.0

        if track_training_metrics:
            state = generalist_episode_metrics_state(env)
            recovering_ids = gate_ids[
                self._gate_pushed[gate_ids] & ~self._gate_retargeted[gate_ids]
            ]
            if recovering_ids.numel() > 0:
                settled = generalist_gate_recovery_stable(env, asset, recovering_ids)
                self._gate_recovery_stable_time_s[recovering_ids] = torch.where(
                    settled,
                    self._gate_recovery_stable_time_s[recovering_ids] + float(env.step_dt),
                    torch.zeros_like(self._gate_recovery_stable_time_s[recovering_ids]),
                )
                recovered = recovering_ids[
                    self._gate_recovery_stable_time_s[recovering_ids] >= 0.50
                ]
                state["recovery_completed"][recovered] = True

            target_distance = torch.linalg.vector_norm(
                asset.data.root_link_pos_w[ids, :2] - world_target_xy(env)[ids], dim=1
            )
            newly_arrived = (target_distance <= GENERALIST_TARGET_ARRIVAL_DISTANCE_M) & ~state[
                "arrived"
            ][ids]
            arrived_ids = ids[newly_arrived]
            if arrived_ids.numel() > 0:
                current_yaw = yaw_from_quaternion_wxyz(asset.data.root_link_quat_w[arrived_ids])
                heading_error = wrapped_angle_difference(
                    world_target_yaw(env)[arrived_ids], current_yaw
                ).abs()
                state["heading_error_at_arrival_rad"][arrived_ids] = heading_error
            state["arrived"][ids] |= target_distance <= GENERALIST_TARGET_ARRIVAL_DISTANCE_M

        if self._stratified_goal_mix and stratified_goal_mix:
            self._record_goal_mix_outcomes(
                env,
                minimum_episodes=minimum_episodes_per_stage,
                gate_recovery_lcb_threshold=gate_recovery_lcb_threshold,
                target_arrival_lcb_threshold=target_arrival_lcb_threshold,
            )

        regular_ids = ids[~self._gate_like[ids]]
        if regular_ids.numel() == 0:
            return
        distance = torch.linalg.vector_norm(
            asset.data.root_link_pos_w[regular_ids, :2] - world_target_xy(env)[regular_ids], dim=1
        )
        settled = _is_settled_for_locomotion(env, asset, regular_ids)
        at_target = (distance <= target_reached_distance_m) & settled
        self._settled_at_target_s[regular_ids] = torch.where(
            at_target,
            self._settled_at_target_s[regular_ids] + float(env.step_dt),
            torch.zeros_like(self._settled_at_target_s[regular_ids]),
        )
        ready = regular_ids[self._settled_at_target_s[regular_ids] >= target_hold_s]
        if ready.numel() == 0:
            return

        if stratified_goal_mix:
            _set_stratified_random_world_targets(
                env,
                asset,
                ready,
                arena_half_extent_m=arena_half_extent_m,
                long_goal_fraction=long_goal_fraction,
                medium_goal_fraction=medium_goal_fraction,
                short_min_distance_m=short_min_distance_m,
                short_max_distance_m=short_max_distance_m,
                medium_min_distance_m=medium_min_distance_m,
                medium_max_distance_m=medium_max_distance_m,
                long_min_distance_m=long_min_distance_m,
                long_max_distance_m=long_max_distance_m,
                record_training_metrics=track_training_metrics,
            )
        else:
            _set_bounded_random_world_targets(
                env,
                asset,
                ready,
                min_distance_m=min_target_distance_m,
                max_distance_m=max_target_distance_m,
                arena_half_extent_m=arena_half_extent_m,
            )
        self._settled_at_target_s[ready] = 0.0

    def _record_goal_mix_outcomes(
        self,
        env,
        *,
        minimum_episodes: int,
        gate_recovery_lcb_threshold: float,
        target_arrival_lcb_threshold: float,
    ) -> None:
        done_ids = env.termination_manager.dones.nonzero(as_tuple=False).flatten()
        if done_ids.numel() == 0:
            return
        metrics_state = generalist_episode_metrics_state(env)
        gate = self._gate_like[done_ids]
        regular = ~gate
        distance = metrics_state["initial_target_distance_m"][done_ids]
        arrived = metrics_state["arrived"][done_ids]
        recovered = metrics_state["recovery_completed"][done_ids]

        gate_count = int(gate.sum().item())
        gate_recovery_count = int((gate & recovered).sum().item())
        short = regular & (distance <= GENERALIST_SHORT_TARGET_MAX_DISTANCE_M)
        long = regular & (distance > GENERALIST_MEDIUM_TARGET_MAX_DISTANCE_M)
        self._goal_mix_state["gate_episodes"] = (
            int(self._goal_mix_state["gate_episodes"]) + gate_count
        )
        self._goal_mix_state["gate_recovery_successes"] = (
            int(self._goal_mix_state["gate_recovery_successes"]) + gate_recovery_count
        )
        self._goal_mix_state["short_episodes"] = int(self._goal_mix_state["short_episodes"]) + int(
            short.sum().item()
        )
        self._goal_mix_state["short_arrival_successes"] = int(
            self._goal_mix_state["short_arrival_successes"]
        ) + int((short & arrived).sum().item())
        self._goal_mix_state["long_episodes"] = int(self._goal_mix_state["long_episodes"]) + int(
            long.sum().item()
        )
        self._goal_mix_state["long_arrival_successes"] = int(
            self._goal_mix_state["long_arrival_successes"]
        ) + int((long & arrived).sum().item())
        next_stage = advance_generalist_goal_mix_stage(
            int(self._goal_mix_state["stage"]),
            gate_recovery_successes=int(self._goal_mix_state["gate_recovery_successes"]),
            gate_episodes=int(self._goal_mix_state["gate_episodes"]),
            short_arrival_successes=int(self._goal_mix_state["short_arrival_successes"]),
            short_episodes=int(self._goal_mix_state["short_episodes"]),
            long_arrival_successes=int(self._goal_mix_state["long_arrival_successes"]),
            long_episodes=int(self._goal_mix_state["long_episodes"]),
            minimum_episodes=minimum_episodes,
            gate_recovery_lcb_threshold=gate_recovery_lcb_threshold,
            target_arrival_lcb_threshold=target_arrival_lcb_threshold,
        )
        if next_stage != int(self._goal_mix_state["stage"]):
            self._goal_mix_state["stage"] = next_stage
            fractions = generalist_goal_mix_fractions(
                next_stage,
                medium_goal_fraction=float(self._goal_mix_state["medium_goal_fraction"]),
                initial_long_goal_fraction=float(
                    self._goal_mix_state["initial_long_goal_fraction"]
                ),
                max_long_goal_fraction=float(self._goal_mix_state["max_long_goal_fraction"]),
            )
            self._goal_mix_state["long_goal_fraction"] = fractions["long"]


class SettleTriggeredLocomotionSequence:
    """One settled-state push followed by one nearby world-frame target step.

    The event is evaluated at a fixed short interval, but neither transition is
    clock-triggered.  Each environment must hold the same low-motion/support
    envelope used by the balance objective before it receives its mild push,
    and must settle again before its 5--20 cm target changes.  This produces a
    clean first locomotion curriculum: settle -> push -> recover -> go-to-pose
    -> stop.
    """

    _PHASE_WAITING_FOR_INITIAL_SETTLE = 0
    _PHASE_WAITING_FOR_RECOVERY = 1
    _PHASE_WAITING_FOR_TARGET_STOP = 2
    _PHASE_COMPLETE = 3

    def __init__(self, cfg, env) -> None:
        del cfg
        self._phase = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
        self._settled_time_s = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            self._phase.zero_()
            self._settled_time_s.zero_()
            return
        ids = _resolved_env_ids_placeholder(env_ids, self._phase.device, self._phase.numel())
        self._phase[ids] = self._PHASE_WAITING_FOR_INITIAL_SETTLE
        self._settled_time_s[ids] = 0.0

    def __call__(
        self,
        env,
        env_ids: torch.Tensor | slice | None,
        *,
        settle_hold_s: float = 0.75,
        min_delta_v: float = 0.05,
        max_delta_v: float = 0.15,
        fore_aft_probability: float = 0.75,
        min_target_distance_m: float = 0.05,
        max_target_distance_m: float = 0.20,
        target_reached_distance_m: float = 0.035,
        asset_cfg: SceneEntityCfg,
    ) -> None:
        if settle_hold_s <= 0.0:
            raise ValueError("settle_hold_s must be positive")
        if not 0.0 < min_delta_v <= max_delta_v:
            raise ValueError("locomotion push requires 0 < min_delta_v <= max_delta_v")
        if not 0.0 < fore_aft_probability <= 1.0:
            raise ValueError("fore_aft_probability must be in (0, 1]")
        if not 0.0 < min_target_distance_m <= max_target_distance_m:
            raise ValueError("target distance requires 0 < min <= max")
        if target_reached_distance_m <= 0.0:
            raise ValueError("target_reached_distance_m must be positive")

        ids = _resolved_env_ids(env, env_ids)
        if ids.numel() == 0:
            return
        asset = env.scene[asset_cfg.name]
        settled = _is_settled_for_locomotion(env, asset, ids)
        self._settled_time_s[ids] = torch.where(
            settled,
            self._settled_time_s[ids] + float(env.step_dt),
            torch.zeros_like(self._settled_time_s[ids]),
        )
        eligible = ids[self._settled_time_s[ids] >= settle_hold_s]
        if eligible.numel() == 0:
            return
        # Snapshot phases so a transition cannot cascade (settle -> push ->
        # target) within one event tick.
        eligible_phase = self._phase[eligible].clone()

        initial = eligible[eligible_phase == self._PHASE_WAITING_FOR_INITIAL_SETTLE]
        if initial.numel() > 0:
            _apply_cardinal_planar_push(
                env,
                asset,
                initial,
                min_delta_v=min_delta_v,
                max_delta_v=max_delta_v,
                fore_aft_probability=fore_aft_probability,
            )
            self._phase[initial] = self._PHASE_WAITING_FOR_RECOVERY
            self._settled_time_s[initial] = 0.0

        recovered = eligible[eligible_phase == self._PHASE_WAITING_FOR_RECOVERY]
        if recovered.numel() > 0:
            _set_nearby_world_targets(
                env,
                asset,
                recovered,
                min_distance_m=min_target_distance_m,
                max_distance_m=max_target_distance_m,
            )
            self._phase[recovered] = self._PHASE_WAITING_FOR_TARGET_STOP
            self._settled_time_s[recovered] = 0.0

        stopping = eligible[eligible_phase == self._PHASE_WAITING_FOR_TARGET_STOP]
        if stopping.numel() > 0:
            distance = torch.linalg.vector_norm(
                asset.data.root_link_pos_w[stopping, :2] - world_target_xy(env)[stopping], dim=1
            )
            complete = stopping[distance <= target_reached_distance_m]
            self._phase[complete] = self._PHASE_COMPLETE


def _resolved_env_ids_placeholder(
    env_ids: torch.Tensor | slice, device: torch.device, num_envs: int
) -> torch.Tensor:
    """Resolve ids for an EventTerm.reset callback, which lacks ``env``."""
    ids = torch.arange(num_envs, dtype=torch.long, device=device)
    if isinstance(env_ids, slice):
        return ids[env_ids]
    return env_ids.reshape(-1).to(dtype=torch.long, device=device)


def _is_settled_for_locomotion(env, asset, ids: torch.Tensor) -> torch.Tensor:
    """Boolean low-motion/support envelope matching ``settled_balance``."""
    tilt = torch.acos((-asset.data.projected_gravity_b[ids, 2]).clamp(-1.0, 1.0))
    planar_speed = torch.linalg.vector_norm(asset.data.root_link_lin_vel_b[ids, :2], dim=1)
    angular_speed = torch.linalg.vector_norm(asset.data.root_link_ang_vel_b[ids], dim=1)
    heading_error = wrapped_angle_difference(
        world_target_yaw(env)[ids], yaw_from_quaternion_wxyz(asset.data.root_link_quat_w[ids])
    ).abs()
    height_error = (asset.data.root_link_pos_w[ids, 2] - 0.75).abs()
    left = env.scene["left_wheel_contact"].data.found[ids].flatten(start_dim=1).any(dim=1)
    right = env.scene["right_wheel_contact"].data.found[ids].flatten(start_dim=1).any(dim=1)
    return (
        (tilt <= 0.08)
        & (planar_speed <= 0.10)
        & (angular_speed <= 0.25)
        & (heading_error <= 0.35)
        & (height_error <= 0.05)
        & left
        & right
    )


def generalist_gate_recovery_stable(env, asset, ids: torch.Tensor) -> torch.Tensor:
    """Match the sequence evaluator's 0.5 s post-disturbance recovery envelope."""
    tilt = torch.acos((-asset.data.projected_gravity_b[ids, 2]).clamp(-1.0, 1.0))
    planar_speed = torch.linalg.vector_norm(asset.data.root_link_lin_vel_b[ids, :2], dim=1)
    angular_xy = torch.linalg.vector_norm(asset.data.root_link_ang_vel_b[ids, :2], dim=1)
    yaw_rate = asset.data.root_link_ang_vel_b[ids, 2].abs()
    heading_error = wrapped_angle_difference(
        world_target_yaw(env)[ids], yaw_from_quaternion_wxyz(asset.data.root_link_quat_w[ids])
    ).abs()
    height_error = (asset.data.root_link_pos_w[ids, 2] - 0.75).abs()
    left = env.scene["left_wheel_contact"].data.found[ids].flatten(start_dim=1).any(dim=1)
    right = env.scene["right_wheel_contact"].data.found[ids].flatten(start_dim=1).any(dim=1)
    return (
        (tilt <= 0.08)
        & (planar_speed <= 0.10)
        & (height_error <= 0.05)
        & (angular_xy <= 0.25)
        & (yaw_rate <= 0.25)
        & (heading_error <= 0.15)
        & left
        & right
    )


def _apply_cardinal_planar_push(
    env,
    asset,
    ids: torch.Tensor,
    *,
    min_delta_v: float,
    max_delta_v: float,
    fore_aft_probability: float,
) -> None:
    velocity = asset.data.root_link_vel_w[ids].clone()
    magnitudes = torch.empty(ids.numel(), device=env.device).uniform_(min_delta_v, max_delta_v)
    delta = torch.zeros((ids.numel(), 3), dtype=velocity.dtype, device=env.device)
    # Use the longitudinal axis most often during the first curriculum phase,
    # while retaining some lateral recovery examples.
    axes = (torch.rand(ids.numel(), device=env.device) >= fore_aft_probability).long()
    signs = torch.where(torch.rand(ids.numel(), device=env.device) < 0.5, 1.0, -1.0).to(
        dtype=velocity.dtype
    )
    body_direction = torch.zeros_like(delta)
    body_direction[torch.arange(ids.numel(), device=env.device), axes] = signs
    world_direction = quat_apply(asset.data.root_link_quat_w[ids], body_direction)
    planar_direction = world_direction[:, :2]
    planar_norm = torch.linalg.vector_norm(planar_direction, dim=1, keepdim=True).clamp_min(
        torch.finfo(velocity.dtype).eps
    )
    delta[:, :2] = planar_direction / planar_norm * magnitudes.unsqueeze(1)
    velocity[:, :3] += delta
    asset.write_root_link_velocity_to_sim(velocity, env_ids=ids)


def _set_bounded_random_world_targets(
    env,
    asset,
    ids: torch.Tensor,
    *,
    min_distance_m: float,
    max_distance_m: float,
    arena_half_extent_m: float,
    target_distances_m: torch.Tensor | None = None,
) -> None:
    """Sample reachable local targets while keeping each clone in its own arena."""
    if not 0.0 < min_distance_m <= max_distance_m:
        raise ValueError("target distance requires 0 < min <= max")
    if arena_half_extent_m <= 0.0:
        raise ValueError("arena_half_extent_m must be positive")
    if target_distances_m is not None:
        target_distances_m = target_distances_m.to(device=env.device, dtype=torch.float32).reshape(
            -1
        )
        if target_distances_m.numel() != ids.numel() or torch.any(target_distances_m <= 0.0):
            raise ValueError(
                "target_distances_m must provide one positive distance per environment"
            )

    current = asset.data.root_link_pos_w[ids, :2]
    origins = env.scene.env_origins[ids, :2]
    targets = current.clone()
    pending = torch.arange(ids.numel(), dtype=torch.long, device=env.device)

    # Local polar sampling produces a directionally diverse waypoint set.
    # Rejection against the per-clone arena prevents a long chain of targets
    # from drifting into neighbouring environments.
    for _ in range(8):
        if pending.numel() == 0:
            break
        angles = torch.empty(pending.numel(), device=env.device).uniform_(-torch.pi, torch.pi)
        distances = (
            torch.empty(pending.numel(), device=env.device).uniform_(min_distance_m, max_distance_m)
            if target_distances_m is None
            else target_distances_m[pending]
        )
        offsets = distances.unsqueeze(1) * torch.stack(
            (torch.cos(angles), torch.sin(angles)), dim=1
        )
        candidates = current[pending] + offsets
        local = candidates - origins[pending]
        valid = torch.all(local.abs() <= arena_half_extent_m, dim=1)
        if bool(valid.any()):
            targets[pending[valid]] = candidates[valid]
        pending = pending[~valid]

    if pending.numel() > 0:
        # Rare boundary fallback: move back toward the clone origin. This keeps
        # the target useful even if overshoot left the robot near/outside the
        # nominal arena.
        to_origin = origins[pending] - current[pending]
        norm = torch.linalg.vector_norm(to_origin, dim=1, keepdim=True)
        default_direction = torch.zeros_like(to_origin)
        default_direction[:, 0] = 1.0
        direction = torch.where(
            norm > 1.0e-6,
            to_origin / norm.clamp_min(1.0e-6),
            default_direction,
        )
        fallback_distance = (
            0.5 * (min_distance_m + max_distance_m)
            if target_distances_m is None
            else target_distances_m[pending].unsqueeze(1)
        )
        targets[pending] = current[pending] + direction * fallback_distance

    state = _world_target_state(env)
    state["target_xy"][ids] = targets
    displacement = targets - current
    # Locomotion waypoints own their arrival heading: face along the segment
    # toward the target rather than preserving the yaw at target assignment.
    # Keeping this bearing fixed for the segment avoids a 180-degree heading
    # discontinuity if the robot slightly overshoots the waypoint.
    state["target_yaw"][ids] = torch.atan2(displacement[:, 1], displacement[:, 0])


def _set_stratified_random_world_targets(
    env,
    asset,
    ids: torch.Tensor,
    *,
    arena_half_extent_m: float,
    long_goal_fraction: float,
    medium_goal_fraction: float,
    short_min_distance_m: float,
    short_max_distance_m: float,
    medium_min_distance_m: float,
    medium_max_distance_m: float,
    long_min_distance_m: float,
    long_max_distance_m: float,
    record_training_metrics: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    distances, bands = sample_stratified_goal_distances(
        ids.numel(),
        device=env.device,
        long_goal_fraction=long_goal_fraction,
        medium_goal_fraction=medium_goal_fraction,
        short_min_distance_m=short_min_distance_m,
        short_max_distance_m=short_max_distance_m,
        medium_min_distance_m=medium_min_distance_m,
        medium_max_distance_m=medium_max_distance_m,
        long_min_distance_m=long_min_distance_m,
        long_max_distance_m=long_max_distance_m,
    )
    _set_bounded_random_world_targets(
        env,
        asset,
        ids,
        min_distance_m=short_min_distance_m,
        max_distance_m=long_max_distance_m,
        arena_half_extent_m=arena_half_extent_m,
        target_distances_m=distances,
    )
    if record_training_metrics:
        state = generalist_episode_metrics_state(env)
        for band in range(3):
            state["sampled_target_band_counts"][ids, band] += (bands == band).float()
    return distances, bands


def _set_nearby_world_targets(
    env, asset, ids: torch.Tensor, *, min_distance_m: float, max_distance_m: float
) -> None:
    state = _world_target_state(env)
    angles = torch.empty(ids.numel(), device=env.device).uniform_(-torch.pi, torch.pi)
    distances = torch.empty(ids.numel(), device=env.device).uniform_(min_distance_m, max_distance_m)
    offset = distances.unsqueeze(1) * torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    state["target_xy"][ids] = asset.data.root_link_pos_w[ids, :2] + offset
    state["target_yaw"][ids] = yaw_from_quaternion_wxyz(asset.data.root_link_quat_w[ids])


__all__ = [
    "DEFAULT_WHEEL_HALF_WIDTH_M",
    "DEFAULT_WHEEL_RADIUS_M",
    "flat_ground_wheel_bottom_heights",
    "generalist_curriculum_values",
    "generalist_goal_mix_fractions",
    "sample_stratified_goal_distances",
    "advance_generalist_goal_mix_stage",
    "generalist_goal_mix_state",
    "generalist_episode_metrics_state",
    "reset_generalist_episode_metrics",
    "generalist_gate_recovery_stable",
    "initialize_random_world_target",
    "initialize_world_target",
    "mixed_balance_recovery_reset",
    "OneShotPlanarVelocityPush",
    "RepeatedRandomWorldTargetSequence",
    "SettleTriggeredLocomotionSequence",
    "reset_to_default_supported",
    "reset_root_state_supported",
    "reset_root_state_uniform",
    "wrapped_angle_difference",
    "world_target_xy",
    "world_target_yaw",
    "yaw_from_quaternion_wxyz",
]
