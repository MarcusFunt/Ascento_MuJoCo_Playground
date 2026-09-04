"""mjlab-dependent implementation of the Ascento actuator model.

Kept separate from :mod:`ascento_mjlab.actuator` so importing the public
actuator specification does not import mjlab while its entry-point task loader
is still starting.  This avoids the actuator-first task-registration cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import torch
from mjlab.actuator import Actuator, ActuatorCfg, ActuatorCmd
from mjlab.utils.spec import create_motor_actuator

if TYPE_CHECKING:
  from mjlab.entity import Entity


def torque_speed_limit(
  peak_torque: float,
  no_load_speed: float,
  controller_speed_limit: float,
  velocity: torch.Tensor,
  requested: torch.Tensor,
) -> torch.Tensor:
  """Apply the transient torque-speed envelope and motoring speed guard."""
  speed = torch.abs(velocity)
  envelope = torch.clamp(1.0 - speed / no_load_speed, min=0.0)
  output = torch.clamp(requested, -peak_torque * envelope, peak_torque * envelope)
  over_speed = speed > controller_speed_limit
  motoring = torch.sign(output) * velocity > 0.0
  return torch.where(over_speed & motoring, torch.zeros_like(output), output)


@dataclass(kw_only=True)
class AscentoTorqueActuatorCfg(ActuatorCfg):
  """Direct-effort actuator with motion-relevant transient limits only."""

  peak_torque: float
  no_load_speed: float
  controller_speed_limit: float
  response_time: float

  def __post_init__(self) -> None:
    super().__post_init__()
    if self.peak_torque <= 0.0:
      raise ValueError("peak_torque must be positive")
    if self.no_load_speed <= 0.0:
      raise ValueError("no_load_speed must be positive")
    if self.controller_speed_limit <= 0.0:
      raise ValueError("controller_speed_limit must be positive")
    if self.response_time < 0.0:
      raise ValueError("response_time must be non-negative")

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> AscentoTorqueActuator:
    return AscentoTorqueActuator(self, entity, target_ids, target_names)


class AscentoTorqueActuator(Actuator[AscentoTorqueActuatorCfg]):
  """Stateful torque response with a one-sided high-speed motor guard."""

  def __init__(
    self,
    cfg: AscentoTorqueActuatorCfg,
    entity: Entity,
    target_ids: list[int],
    target_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, target_ids, target_names)
    self._filtered: torch.Tensor | None = None
    self._physics_dt: float | None = None

  def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
    for target_name in target_names:
      self._mjs_actuators.append(
        create_motor_actuator(
          spec,
          target_name,
          effort_limit=self.cfg.peak_torque,
          transmission_type=self.cfg.transmission_type,
        )
      )

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model,
    data,
    device: str,
  ) -> None:
    self._physics_dt = float(mj_model.opt.timestep)
    super().initialize(mj_model, model, data, device)
    self._filtered = torch.zeros(
      (data.nworld, len(self.target_ids)), dtype=torch.float32, device=device
    )

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    assert self._filtered is not None
    requested = torch.clamp(cmd.effort_target, -self.cfg.peak_torque, self.cfg.peak_torque)
    if self._physics_dt is None:
      raise RuntimeError("Actuator must be initialized before compute()")
    dt = self._physics_dt
    alpha = 1.0 if self.cfg.response_time == 0.0 else 1.0 - torch.exp(
      torch.tensor(-dt / self.cfg.response_time, device=requested.device)
    )
    self._filtered.add_(alpha * (requested - self._filtered))
    return torque_speed_limit(
      self.cfg.peak_torque,
      self.cfg.no_load_speed,
      self.cfg.controller_speed_limit,
      cmd.vel,
      self._filtered,
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    assert self._filtered is not None
    self._filtered[env_ids if env_ids is not None else slice(None)] = 0.0
