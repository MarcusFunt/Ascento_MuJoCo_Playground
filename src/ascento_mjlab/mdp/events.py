"""Reset and perturbation presets for Ascento tasks."""

from __future__ import annotations

import torch
from mjlab.envs.mdp.events import reset_root_state_uniform
from mjlab.managers.scene_entity_config import SceneEntityCfg


DEFAULT_WHEEL_RADIUS_M = 0.25


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
  env_ids: torch.Tensor | slice | None = None,
) -> torch.Tensor:
  """Return [left, right] wheel-bottom heights above the flat support plane."""
  ids = _resolved_env_ids(env, env_ids)
  asset = env.scene[asset_name]
  try:
    left_id = asset.body_names.index("left_wheel")
    right_id = asset.body_names.index("right_wheel")
  except ValueError as exc:
    raise RuntimeError("Expected left_wheel and right_wheel bodies") from exc

  positions = asset.data.body_link_pos_w.index_select(0, ids)
  wheel_ids = torch.tensor([left_id, right_id], dtype=torch.long, device=positions.device)
  wheel_positions = positions.index_select(1, wheel_ids)
  return wheel_positions[..., 2] - float(wheel_radius_m)


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
  height unchanged. For a wheel-legged robot that can introduce accidental wheel
  penetration or unsupported hovering before the first physics step. This helper
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
  pose = asset.data.root_link_pose_w.index_select(0, ids).clone()
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
