"""Recovery shaping and executable recovery-success semantics."""

from __future__ import annotations

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


def _sensor_contact(env, name: str) -> torch.Tensor:
  found = env.scene[name].data.found
  assert found is not None
  return found.flatten(start_dim=1).any(dim=1)


def recovery_condition(
  env,
  envelope: RecoveryEnvelope = RecoveryEnvelope(),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Return whether each environment is presently inside the recovery envelope."""
  asset: Entity = env.scene[asset_cfg.name]
  gravity_xy = torch.linalg.vector_norm(asset.data.projected_gravity_b[:, :2], dim=1)
  tilt = torch.atan2(
    gravity_xy,
    -asset.data.projected_gravity_b[:, 2].clamp(max=-1.0e-6),
  )
  linear_speed = torch.linalg.vector_norm(asset.data.root_link_lin_vel_b, dim=1)
  angular_speed = torch.linalg.vector_norm(asset.data.root_link_ang_vel_b, dim=1)
  supported = _sensor_contact(env, "left_wheel_contact") & _sensor_contact(
    env, "right_wheel_contact"
  )
  return (
    (asset.data.root_link_pos_w[:, 2] >= envelope.min_height)
    & (tilt <= envelope.max_tilt_radians)
    & (linear_speed <= envelope.max_linear_speed)
    & (angular_speed <= envelope.max_angular_speed)
    & supported
  )


class RecoverySuccess:
  """Stateful metric requiring the envelope continuously for ``stable_duration_s``."""

  def __init__(self, cfg, env) -> None:
    self.envelope = cfg.params.get("envelope", RecoveryEnvelope())
    self._stable_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self._required_steps = max(1, round(self.envelope.stable_duration_s / env.step_dt))

  def __call__(self, env, envelope: RecoveryEnvelope | None = None) -> torch.Tensor:
    active_envelope = self.envelope if envelope is None else envelope
    stable = recovery_condition(env, active_envelope)
    self._stable_steps = torch.where(
      stable, self._stable_steps + 1, torch.zeros_like(self._stable_steps)
    )
    return (self._stable_steps >= self._required_steps).float()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self._stable_steps[ids] = 0


def recovery_progress(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Smooth shaping toward an upright, low-velocity recovery state."""
  asset: Entity = env.scene[asset_cfg.name]
  upright = torch.exp(
    -torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1) / 0.35**2
  )
  height = torch.exp(-torch.square(asset.data.root_link_pos_w[:, 2] - 0.75) / 0.10**2)
  speed = torch.linalg.vector_norm(asset.data.root_link_lin_vel_b, dim=1)
  return upright * height * torch.exp(-torch.square(speed) / 2.0**2)
