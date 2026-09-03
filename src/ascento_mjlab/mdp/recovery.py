"""Recovery envelope definitions."""

from dataclasses import dataclass

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg


@dataclass(frozen=True)
class RecoveryEnvelope:
  min_height: float = 0.45
  max_tilt_radians: float = 0.55
  max_linear_speed: float = 3.0
  max_angular_speed: float = 5.0
  stable_duration_s: float = 0.25


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def recovery_progress(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Progress toward an upright, supported, low-velocity recovery envelope."""
  asset: Entity = env.scene[asset_cfg.name]
  upright = torch.exp(-torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1) / 0.35**2)
  height = torch.exp(-torch.square(asset.data.root_link_pos_w[:, 2] - 0.75) / 0.10**2)
  speed = torch.linalg.vector_norm(asset.data.root_link_lin_vel_b, dim=1)
  return upright * height * torch.exp(-torch.square(speed) / 2.0**2)
