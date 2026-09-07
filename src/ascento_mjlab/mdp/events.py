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


def initialize_balance_origin(
    env,
    env_ids: torch.Tensor | slice | None = None,
    *,
    asset_name: str = "robot",
) -> None:
    """Remember the supported root XY pose used as this episode's balance target.

    The balance task randomizes its initial XY position slightly.  Tracking the
    actual post-reset position, rather than the environment-grid origin, keeps a
    position-hold reward free of a reset-dependent bias.  The state is stored on
    the environment because rewards run after the reset event has completed.
    """
    ids = _resolved_env_ids(env, env_ids)
    if ids.numel() == 0:
        return
    if not hasattr(env, "ascento_balance_state"):
        env.ascento_balance_state = {
            "origin_xy": torch.zeros((env.num_envs, 2), dtype=torch.float32, device=env.device)
        }
    origin_xy = env.ascento_balance_state["origin_xy"]
    asset = env.scene[asset_name]
    origin_xy[ids] = asset.data.root_link_pos_w[ids, :2]


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


__all__ = [
    "DEFAULT_WHEEL_HALF_WIDTH_M",
    "DEFAULT_WHEEL_RADIUS_M",
    "flat_ground_wheel_bottom_heights",
    "initialize_balance_origin",
    "OneShotPlanarVelocityPush",
    "reset_root_state_supported",
    "reset_root_state_uniform",
]
