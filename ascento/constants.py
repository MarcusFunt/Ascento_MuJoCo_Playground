"""Central Guard 2.0 simulation constants; no deployment transport model."""
from dataclasses import dataclass
import numpy as np

JOINT_NAMES = (
    "left_hip", "left_knee", "left_wheel_joint",
    "right_hip", "right_knee", "right_wheel_joint",
)
LEG_INDEX = np.array((0, 1, 3, 4), dtype=np.int32)
WHEEL_INDEX = np.array((2, 5), dtype=np.int32)
N_ACTUATORS = 6
N_SUBSTEPS = 5
SIM_DT = 0.002
CTRL_DT = SIM_DT * N_SUBSTEPS
STANCE = np.array((-3.14, -3.14, 0.0, -3.14, -3.14, 0.0), dtype=np.float32)
LEG_Q_MIN = -1.5 * np.pi
LEG_Q_MAX = -0.5 * np.pi
LEG_LIMIT_MARGIN = np.deg2rad(5.0)
TIRE_FRICTION = (0.80, 0.020, 0.002)
TIRE_CONDIM = 6
BASE_HEIGHT = 0.75
FALL_HEIGHT = 0.10
UPRIGHT_GRAVITY_Z_MIN = 0.25

@dataclass(frozen=True)
class ActuatorSpec:
    peak_torque_nm: float
    continuous_torque_nm: float
    no_load_speed_rad_s: float
    controller_speed_limit_rad_s: float
    torque_time_constant_s: float

LEG_ACTUATOR = ActuatorSpec(40.0, 15.0, 12.0, 4.0, 0.004)
WHEEL_ACTUATOR = ActuatorSpec(8.0, 5.0, 20.0, 10.0, 0.003)
ACTUATOR_SPECS = (LEG_ACTUATOR, LEG_ACTUATOR, WHEEL_ACTUATOR,
                  LEG_ACTUATOR, LEG_ACTUATOR, WHEEL_ACTUATOR)
PEAK_TORQUE = np.array([s.peak_torque_nm for s in ACTUATOR_SPECS], dtype=np.float32)
