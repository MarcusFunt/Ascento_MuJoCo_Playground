"""Reset and perturbation presets for Ascento tasks."""

from __future__ import annotations

import torch
from mjlab.envs.mdp.events import reset_root_state_uniform
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

DEFAULT_WHEEL_RADIUS_M = 0.25
DEFAULT_WHEEL_HALF_WIDTH_M = 0.0025


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
    _set_bounded_random_world_targets(
        env,
        asset,
        ids,
        min_distance_m=min_distance_m,
        max_distance_m=max_distance_m,
        arena_half_extent_m=arena_half_extent_m,
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
        del cfg
        self._settled_at_target_s = torch.zeros(
            env.num_envs, dtype=torch.float32, device=env.device
        )
        self._episode_elapsed_s = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self._episode_initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._gate_like = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._gate_pushed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._gate_retargeted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._push = OneShotPlanarVelocityPush(None, env)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            self._settled_at_target_s.zero_()
            self._episode_elapsed_s.zero_()
            self._episode_initialized.zero_()
            self._gate_like.zero_()
            self._gate_pushed.zero_()
            self._gate_retargeted.zero_()
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
        arena_half_extent_m: float = 0.65,
        gate_like_fraction: float = 0.25,
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
                torch.rand(new_ids.numel(), device=env.device) < gate_like_fraction
            )
            self._episode_initialized[new_ids] = True
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

        _set_bounded_random_world_targets(
            env,
            asset,
            ready,
            min_distance_m=min_target_distance_m,
            max_distance_m=max_target_distance_m,
            arena_half_extent_m=arena_half_extent_m,
        )
        self._settled_at_target_s[ready] = 0.0


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
) -> None:
    """Sample reachable local targets while keeping each clone in its own arena."""
    if not 0.0 < min_distance_m <= max_distance_m:
        raise ValueError("target distance requires 0 < min <= max")
    if arena_half_extent_m <= 0.0:
        raise ValueError("arena_half_extent_m must be positive")

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
        distances = torch.empty(pending.numel(), device=env.device).uniform_(
            min_distance_m, max_distance_m
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
        distance = 0.5 * (min_distance_m + max_distance_m)
        targets[pending] = current[pending] + direction * distance

    state = _world_target_state(env)
    state["target_xy"][ids] = targets
    displacement = targets - current
    # Locomotion waypoints own their arrival heading: face along the segment
    # toward the target rather than preserving the yaw at target assignment.
    # Keeping this bearing fixed for the segment avoids a 180-degree heading
    # discontinuity if the robot slightly overshoots the waypoint.
    state["target_yaw"][ids] = torch.atan2(displacement[:, 1], displacement[:, 0])


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
