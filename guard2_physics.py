"""Guard 2.0-oriented low-level physics helpers.

The rigid-body masses/inertias come from the supplied Ascento description and are
left unchanged.  Guard 2.0 does NOT use a passive leg spring, so no joint spring
or elastic energy-storage element is added here.

Public Guard 2.0 actuator curves are not available.  Values marked ESTIMATE are
therefore deliberately centralized below so they can be replaced from measured
hardware logs without touching the environment/controller code.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

# Actuator order used throughout the project.
JOINT_NAMES = (
    "left_hip", "left_knee", "left_wheel_joint",
    "right_hip", "right_knee", "right_wheel_joint",
)
LEG_INDEX = np.array([0, 1, 3, 4], dtype=np.int32)
WHEEL_INDEX = np.array([2, 5], dtype=np.int32)

# Finite operational joint envelope around the straight-down configuration q=-pi.
# ±pi/2 gives ~0.25...0.50 m vertical serial-leg reach for the supplied 0.25 m
# thigh + 0.25 m shank geometry.  This also includes the source robot's commonly
# used wrapped joint configurations while preventing unlimited revolutions during RL.
LEG_Q_MIN = -1.5 * math.pi
LEG_Q_MAX = -0.5 * math.pi
LEG_LIMIT_MARGIN = math.radians(5.0)

# Contact parameters.  MuJoCo geom friction = sliding, torsional, rolling.
TIRE_FRICTION = (0.80, 0.020, 0.002)
TIRE_CONDIM = 6
# Soft but stiff contact for a pneumatic/rubber tyre rather than a rigid cylinder.
TIRE_SOLREF = (0.006, 1.0)                  # time constant [s], damping ratio
TIRE_SOLIMP = (0.90, 0.97, 0.001, 0.5, 2.0)

@dataclass(frozen=True)
class ActuatorSpec:
    peak_torque_nm: float
    continuous_torque_nm: float
    no_load_speed_rad_s: float
    controller_speed_limit_rad_s: float
    torque_time_constant_s: float

# Legs: supplied URDF already specifies 40 Nm effort and 4 rad/s velocity.
# The 12 rad/s electromechanical no-load point is used only for torque-speed
# derating; controller speed is still limited to 4 rad/s.
LEG_ACTUATOR = ActuatorSpec(
    peak_torque_nm=40.0,
    continuous_torque_nm=15.0,
    no_load_speed_rad_s=12.0,
    controller_speed_limit_rad_s=4.0,
    torque_time_constant_s=0.004,   # ESTIMATE: finite torque-control bandwidth
)

# Wheels: do NOT reuse the leg's 40 Nm limit.  Exact Guard 2.0 wheel torque is
# not publicly documented.  This conservative current-platform estimate is kept
# explicit so it can be calibrated from acceleration/current logs.
WHEEL_ACTUATOR = ActuatorSpec(
    peak_torque_nm=8.0,             # ESTIMATE
    continuous_torque_nm=5.0,       # ESTIMATE
    no_load_speed_rad_s=20.0,       # source URDF joint velocity envelope
    controller_speed_limit_rad_s=10.0, # source ROS velocity interface limit
    torque_time_constant_s=0.003,   # ESTIMATE
)

ACTUATOR_SPECS = (LEG_ACTUATOR, LEG_ACTUATOR, WHEEL_ACTUATOR,
                  LEG_ACTUATOR, LEG_ACTUATOR, WHEEL_ACTUATOR)

class Guard2ActuatorModel:
    """Torque bandwidth + motor torque/speed + speed/position guards.

    Input/output units are joint torque [Nm]. There is deliberately NO command
    transport delay: this policy is simulation-only and is not intended for
    deployment through a real robot communication/control stack.

    The model intentionally has no passive spring state. Guard 2.0 has no
    physical leg spring. Mechanical realism retained here comes from finite
    actuator torque response, torque-speed limits, hard saturation, joint travel
    limits, and compliant tyre contact.
    """
    def __init__(self, dt: float, specs=ACTUATOR_SPECS):
        self.dt = float(dt)
        self.specs = tuple(specs)
        self.filtered = np.zeros(6, dtype=np.float64)

    def reset(self):
        self.filtered[:] = 0.0

    @staticmethod
    def _torque_speed_limit(spec: ActuatorSpec, omega: float) -> float:
        # First-order BLDC torque-speed envelope: stall/peak torque at zero speed,
        # linearly falling to zero at the no-load speed.
        return spec.peak_torque_nm * max(0.0, 1.0 - abs(omega) / spec.no_load_speed_rad_s)

    def step(self, requested_torque, joint_q, joint_qd):
        req = np.asarray(requested_torque, dtype=np.float64).reshape(6)
        q = np.asarray(joint_q, dtype=np.float64).reshape(6)
        qd = np.asarray(joint_qd, dtype=np.float64).reshape(6)
        out = np.zeros(6, dtype=np.float64)

        for i, spec in enumerate(self.specs):
            # No artificial command/transport delay. The requested simulation
            # action reaches the actuator model in the same physics step.
            commanded = float(req[i])

            # Hard peak saturation first.
            commanded = float(np.clip(commanded, -spec.peak_torque_nm, spec.peak_torque_nm))

            # First-order torque-loop response (exact discretization). This is
            # retained as electromechanical actuator dynamics, not communication latency.
            if spec.torque_time_constant_s > 0.0:
                alpha = 1.0 - math.exp(-self.dt / spec.torque_time_constant_s)
            else:
                alpha = 1.0
            self.filtered[i] += alpha * (commanded - self.filtered[i])

            # Torque-speed envelope.
            speed_tau = self._torque_speed_limit(spec, qd[i])
            tau = float(np.clip(self.filtered[i], -speed_tau, speed_tau))

            # Controller velocity protection: braking remains available, but torque
            # that would accelerate farther past the speed limit is removed.
            if abs(qd[i]) >= spec.controller_speed_limit_rad_s and tau * qd[i] > 0.0:
                tau = 0.0

            # Mechanical leg travel guard.  Use a 5 deg soft zone before the hard
            # MuJoCo/URDF stop so policies do not hammer the constraint continuously.
            if i in LEG_INDEX:
                if q[i] <= LEG_Q_MIN + LEG_LIMIT_MARGIN and tau < 0.0:
                    tau = 0.0
                elif q[i] >= LEG_Q_MAX - LEG_LIMIT_MARGIN and tau > 0.0:
                    tau = 0.0

            out[i] = tau
        return out


def apply_torque(model, data, actuator_model: Guard2ActuatorModel, requested_torque):
    """Apply one physics-step worth of Guard 2.0 actuator dynamics."""
    q = data.qpos[7:13]
    qd = data.qvel[6:12]
    data.ctrl[:] = actuator_model.step(requested_torque, q, qd)
    return data.ctrl.copy()
