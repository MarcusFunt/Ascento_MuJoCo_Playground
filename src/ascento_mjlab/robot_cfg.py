"""Ascento Guard-2 robot entity configuration.

The MJCF is deliberately limited to the robot.  The floor, lighting, cameras,
and environment spacing belong to the mjlab scene configuration.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from .actuator import (
  LEG_ACTUATOR,
  WHEEL_ACTUATOR,
  AscentoTorqueActuatorCfg,
)

ROBOT_XML = Path(__file__).parent / "assets" / "ascento_guard2" / "robot.xml"
JOINT_NAMES = (
  "left_hip",
  "left_knee",
  "left_wheel_joint",
  "right_hip",
  "right_knee",
  "right_wheel_joint",
)
LEG_JOINT_NAMES = ("left_hip", "left_knee", "right_hip", "right_knee")
WHEEL_JOINT_NAMES = ("left_wheel_joint", "right_wheel_joint")

# The source model is a two-link serial leg with 0.25 m wheels.  -pi is the
# straight-down nominal pose and 0.75 m leaves the wheels supported without
# penetrating the plane.
DEFAULT_POSE = {
  "left_hip": -3.141592653589793,
  "left_knee": -3.141592653589793,
  "left_wheel_joint": 0.0,
  "right_hip": -3.141592653589793,
  "right_knee": -3.141592653589793,
  "right_wheel_joint": 0.0,
}
DEFAULT_ROOT_HEIGHT = 0.75

def get_spec() -> mujoco.MjSpec:
  """Load a fresh robot spec for each entity variant."""
  return mujoco.MjSpec.from_file(str(ROBOT_XML))


ARTICULATION_CFG = EntityArticulationInfoCfg(
  actuators=(
    AscentoTorqueActuatorCfg(
      target_names_expr=LEG_JOINT_NAMES,
      peak_torque=LEG_ACTUATOR.peak_torque_nm,
      no_load_speed=LEG_ACTUATOR.no_load_speed_rad_s,
      controller_speed_limit=LEG_ACTUATOR.controller_speed_limit_rad_s,
      response_time=LEG_ACTUATOR.response_time_s,
    ),
    AscentoTorqueActuatorCfg(
      target_names_expr=WHEEL_JOINT_NAMES,
      peak_torque=WHEEL_ACTUATOR.peak_torque_nm,
      no_load_speed=WHEEL_ACTUATOR.no_load_speed_rad_s,
      controller_speed_limit=WHEEL_ACTUATOR.controller_speed_limit_rad_s,
      response_time=WHEEL_ACTUATOR.response_time_s,
    ),
  ),
)

DEFAULT_ASCENTO_CFG = EntityCfg(
  spec_fn=get_spec,
  init_state=EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, DEFAULT_ROOT_HEIGHT),
    joint_pos=DEFAULT_POSE,
    joint_vel={".*": 0.0},
  ),
  articulation=ARTICULATION_CFG,
)

VIEWER_CONFIG = ViewerConfig(
  origin_type=ViewerConfig.OriginType.ASSET_BODY,
  entity_name="robot",
  body_name="base",
  distance=2.5,
  elevation=-15.0,
  azimuth=90.0,
)

SIM_CFG = SimulationCfg(
  mujoco=MujocoCfg(
    timestep=0.002,
    iterations=10,
    ls_iterations=20,
  ),
  nconmax=256,
  njmax=512,
)
