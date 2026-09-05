"""Reset and perturbation presets for Ascento tasks."""

from __future__ import annotations

import torch
from mjlab.envs.mdp.events import reset_root_state_uniform
from mjlab.managers.scene_entity_config import SceneEntityCfg


DEFAULT_WHEEL_RADIUS_M = 0.25


def _resolved_env_ids(env, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
  if env_ids is None or isinstance(env_ids, slice):
    return torch.arange(env.num_envs, dtype=torch.long, device=env.device)[env_ids]
  return env_ids


def flat_ground_wheel_bottom_heights(
  env,
  *,
  asset_name: str = "robot",
  wheel_radius_m: float = DEFAULT_WHEEL_RADIUS_M,
  env_ids: torch.Tensor | slice | None = None,
) -> torch.Tensor:
  """Return [left, right] wheel-bottom heights above the flat support plane."""
  ids = _resolved_env_ids(env, env_ids)
  asset = env.scene[asset_name]
  left_ids, _ = asset.find_bodies(r"^left_wheel$")
  right_ids, _ = asset.find_bodies(r"^right_wheel$")
  if len(left_ids) != 1 or len(right_ids) != 1:
    raise RuntimeError("Expected exactly one left_wheel and one right_wheel body")
  wheel_centres = asset.data.body_link_pos_w[ids][:, [left_ids[0], right_ids[0]], 2]
  return wheel_centres - float(wheel_radius_m)


def reset_root_state_supported(
  env,
  env_ids: torch.Tensor | slice | None,
  pose_range: dict[str, tuple[float, float]],
  velocity_range: dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg,
  wheel_radius_m: float = DEFAULT_WHEEL_RADIUS_M,
  support_clearance_m: float = 0.0,
) -> None:
  """Sample a root reset, then vertically align its lowest wheel with flat ground.

  The generic mjlab root reset perturbs orientation while leaving the nominal root
  height unchanged.  For a wheel-legged robot that can introduce accidental wheel
  penetration or unsupported hovering before the first physics step.  This helper
  preserves the sampled pose/velocity but translates the root in Z so the lowest
  wheel starts exactly on the flat support plane.
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
    env_ids=ids,
  )
  correction = float(support_clearance_m) - bottoms.amin(dim=1)
  pose = asset.data.root_link_pose_w[ids].clone()
  pose[:, 2] += correction
  asset.write_root_link_pose_to_sim(pose, env_ids=ids)
  env.sim.forward()
  env.sim.sense()


__all__ = [
  "DEFAULT_WHEEL_RADIUS_M",
  "flat_ground_wheel_bottom_heights",
  "reset_root_state_supported",
  "reset_root_state_uniform",
]
