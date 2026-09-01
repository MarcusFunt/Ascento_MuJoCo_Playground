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
# A free-base height below this value cannot represent a wheel-supported,
# recoverable posture for this model.  The former 10 cm threshold let a robot
# lie on its side while its root remained technically "alive".
FALL_HEIGHT = 0.35
# ``gravity`` is world gravity expressed in the base frame: -1 is upright and
# 0 is a 90 degree fall.  End an episode before the robot can collect tracking
# rewards while resting sideways on its wheels.
FALL_GRAVITY_Z_MAX = -0.50
UPRIGHT_GRAVITY_Z_MIN = 0.25
WHEEL_RADIUS = 0.25

# Safety limits are intentionally above normal operation.  They terminate
# numerically explosive trajectories instead of feeding unbounded velocities
# into PPO, while clipped actor features keep ordinary recovery transients
# well-scaled.
MAX_BASE_LINEAR_VELOCITY = 12.0
MAX_BASE_ANGULAR_VELOCITY = 25.0
MAX_JOINT_VELOCITY = 40.0

# The actor always receives these six command channels, including during early
# balance training where they are all zero.  Keeping this schema stable allows
# staged checkpoint transfer without changing the network's first layer.
COMMAND_SIZE = 6
COMMAND_VX = 0
COMMAND_YAW_RATE = 1
COMMAND_HEIGHT = 2
COMMAND_JUMP_TRIGGER = 3
COMMAND_JUMP_HEIGHT = 4
COMMAND_JUMP_DISTANCE = 5

# Six logical jump phases.  They classify state and gate rewards only; no phase
# ever produces motor commands.
PHASE_IDLE = 0
PHASE_CROUCH = 1
PHASE_THRUST = 2
PHASE_FLIGHT = 3
PHASE_LANDING = 4
PHASE_RECOVERY = 5
NUM_JUMP_PHASES = 6

# 3 gravity + 3 local linear velocity + 3 local angular velocity + height +
# 4 leg positions + 6 joint velocities + 2 contacts + 2 contact forces +
# 6 applied torques + 6 previous actions + 6 commands + 6 phase one-hot + time.
OBS_SIZE = 49

@dataclass(frozen=True)
class ActuatorSpec:
    peak_torque_nm: float
    continuous_torque_nm: float
    no_load_speed_rad_s: float
    controller_speed_limit_rad_s: float
    torque_time_constant_s: float

LEG_ACTUATOR = ActuatorSpec(40.0, 15.0, 12.0, 4.0, 0.004)
# The simulation intentionally gives every learned direct-effort channel the
# same requested peak authority.  Continuous wheel rating/speed remain wheel
# specific; only the transient peak is raised for this simulation task.
WHEEL_ACTUATOR = ActuatorSpec(40.0, 5.0, 20.0, 10.0, 0.003)
ACTUATOR_SPECS = (LEG_ACTUATOR, LEG_ACTUATOR, WHEEL_ACTUATOR,
                  LEG_ACTUATOR, LEG_ACTUATOR, WHEEL_ACTUATOR)
PEAK_TORQUE = np.array([s.peak_torque_nm for s in ACTUATOR_SPECS], dtype=np.float32)
