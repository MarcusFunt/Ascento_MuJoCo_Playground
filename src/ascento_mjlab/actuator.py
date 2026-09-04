"""Public Ascento actuator specification with lazy mjlab implementation loading.

This module intentionally stays free of mjlab imports at import time. mjlab
auto-discovers task entry points while it imports, and importing mjlab from this
module before the actuator constants existed could re-enter Ascento task
registration through ``robot_cfg`` and leave the registry empty.

The actual actuator classes are loaded lazily from ``actuator_impl`` when a
consumer asks for them. Existing imports from ``ascento_mjlab.actuator`` remain
compatible while actuator-first imports are now safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .physics import PHYSICS_PROFILE


@dataclass(frozen=True)
class ActuatorDocumentation:
  peak_torque_nm: float
  continuous_torque_nm_documentation: float
  no_load_speed_rad_s: float
  controller_speed_limit_rad_s: float
  response_time_s: float


LEG_ACTUATOR = ActuatorDocumentation(
  PHYSICS_PROFILE.peak_effort_nm, 15.0, 12.0, 4.0, 0.004
)
WHEEL_ACTUATOR = ActuatorDocumentation(
  PHYSICS_PROFILE.peak_effort_nm, 5.0, 20.0, 10.0, 0.003
)

_LAZY_EXPORTS = {
  "AscentoTorqueActuator",
  "AscentoTorqueActuatorCfg",
  "torque_speed_limit",
}


def __getattr__(name: str) -> Any:
  if name not in _LAZY_EXPORTS:
    raise AttributeError(name)
  from . import actuator_impl

  value = getattr(actuator_impl, name)
  globals()[name] = value
  return value


__all__ = [
  "ActuatorDocumentation",
  "LEG_ACTUATOR",
  "WHEEL_ACTUATOR",
  "AscentoTorqueActuator",
  "AscentoTorqueActuatorCfg",
  "torque_speed_limit",
]
