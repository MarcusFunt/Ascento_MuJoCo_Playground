"""JAX-traceable direct-torque actuator dynamics for Guard 2.0 MJX."""
import jax.numpy as jp
from mujoco import mjx
from .constants import (
    ACTUATOR_SPECS, LEG_INDEX, LEG_LIMIT_MARGIN, LEG_Q_MAX, LEG_Q_MIN,
    PEAK_TORQUE,
)
PEAK = jp.asarray(PEAK_TORQUE)
NO_LOAD = jp.asarray([s.no_load_speed_rad_s for s in ACTUATOR_SPECS])
SPEED_LIMIT = jp.asarray([s.controller_speed_limit_rad_s for s in ACTUATOR_SPECS])
TIME_CONSTANT = jp.asarray([s.torque_time_constant_s for s in ACTUATOR_SPECS])
ALPHA = jp.asarray(1.0 - jp.exp(-0.002 / TIME_CONSTANT))
LEG_MASK = jp.asarray([True, True, False, True, True, False])

def torque_speed_envelope(torque, velocity):
    """Derate motoring torque while preserving opposing braking authority."""
    motoring = torque * velocity >= 0.0
    motoring_limit = PEAK * jp.clip(1.0 - jp.abs(velocity) / NO_LOAD, 0.0, 1.0)
    return jp.where(motoring, motoring_limit, PEAK)

def _project_leg_limits(data, mjx_model):
    """Project leg state onto the MJCF range after a constraint overshoot.

    MuJoCo range constraints are compliant, so a fast RL-driven state can
    penetrate a limit by a small amount. Keep the exposed MJX state inside the
    declared range and remove only outward stop velocity, then refresh all
    derived kinematics/contact fields.
    """
    q = data.qpos[7:13]
    qd = data.qvel[6:12]
    clipped_q = jp.clip(q, LEG_Q_MIN, LEG_Q_MAX)
    q = jp.where(LEG_MASK, clipped_q, q)
    outward = LEG_MASK & (
        ((q <= LEG_Q_MIN) & (qd < 0.0))
        | ((q >= LEG_Q_MAX) & (qd > 0.0))
    )
    qd = jp.where(outward, 0.0, qd)
    next_data = data.replace(
        qpos=data.qpos.at[7:13].set(q),
        qvel=data.qvel.at[6:12].set(qd),
    )
    return mjx.forward(mjx_model, next_data)


def substep(data, torque_state, action, mjx_model):
    """Apply one 2 ms plant step; no command/transport delay is modeled."""
    requested = jp.clip(action, -1.0, 1.0) * PEAK
    requested = jp.clip(requested, -PEAK, PEAK)
    filtered = torque_state + ALPHA * (requested - torque_state)
    limits = torque_speed_envelope(filtered, data.qvel[6:12])
    torque = jp.clip(filtered, -limits, limits)
    qd = data.qvel[6:12]
    overspeed = (jp.abs(qd) >= SPEED_LIMIT) & (torque * qd > 0.0)
    torque = jp.where(overspeed, 0.0, torque)
    q = data.qpos[7:13]
    lower_stop = LEG_MASK & (q <= LEG_Q_MIN + LEG_LIMIT_MARGIN) & (torque < 0.0)
    upper_stop = LEG_MASK & (q >= LEG_Q_MAX - LEG_LIMIT_MARGIN) & (torque > 0.0)
    torque = jp.where(lower_stop | upper_stop, 0.0, torque)
    next_data = mjx.step(mjx_model, data.replace(ctrl=torque))
    next_data = _project_leg_limits(next_data, mjx_model)
    return next_data, filtered, torque

def rollout_substeps(data, torque_state, action, mjx_model, n_substeps):
    def one(carry, _):
        data, torque_state, torque = substep(carry[0], carry[1], action, mjx_model)
        return (data, torque_state), torque
    (data, torque_state), torques = __import__("jax").lax.scan(
        one, (data, torque_state), None, length=n_substeps
    )
    return data, torque_state, torques[-1]
