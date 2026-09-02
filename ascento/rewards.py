"""Separate reward terms used by the direct-torque Ascento tasks."""
from __future__ import annotations

import jax.numpy as jp

from .constants import (
    BASE_HEIGHT,
    COMMAND_HEIGHT,
    COMMAND_VX,
    COMMAND_YAW_RATE,
    LEG_Q_MAX,
    LEG_Q_MIN,
    PHASE_CROUCH,
    PHASE_FLIGHT,
    PHASE_LANDING,
    PHASE_RECOVERY,
    PHASE_THRUST,
    STANCE,
    LEG_INDEX,
)


def base_terms(data, action, info, gravity, local_linear_velocity, local_angular_velocity,
               nonwheel_collision):
    """Returns transparent, logged reward terms; positive values are better."""
    command = info["command"]
    upright_error = jp.sum(jp.square(gravity - jp.asarray([0.0, 0.0, -1.0])))
    upright = jp.exp(-4.0 * upright_error)
    vx_tracking = jp.exp(-jp.square(local_linear_velocity[0] - command[COMMAND_VX]) / 0.25)
    yaw_tracking = jp.exp(-jp.square(local_angular_velocity[2] - command[COMMAND_YAW_RATE]) / 0.25)
    height = jp.exp(-25.0 * jp.square(data.qpos[2] - command[COMMAND_HEIGHT]))
    # Tracking a zero command must not pay a fallen robot for merely lying
    # still.  The gate also gives the policy a sharp, unambiguous recovery
    # signal before the orientation termination boundary is reached.
    upright_gate = gravity[2] < -0.80
    recovered_gate = upright_gate & (jp.abs(data.qpos[2] - command[COMMAND_HEIGHT]) < 0.10)
    leg_q = data.qpos[7:13][jp.asarray(LEG_INDEX)]
    leg_stance = jp.asarray(STANCE)[jp.asarray(LEG_INDEX)]
    posture = jp.exp(-0.25 * jp.sum(jp.square(leg_q - leg_stance))) * upright_gate
    stable = jp.exp(-0.10 * jp.sum(jp.square(data.qvel[:6]))) * recovered_gate
    action_rate = -0.015 * jp.sum(jp.square(action - info["last_action"]))
    action_smooth = -0.007 * jp.sum(
        jp.square(action - 2.0 * info["last_action"] + info["last_last_action"])
    )
    # The rate terms prevent jitter; this term keeps a quiet equilibrium from
    # being rewarded for applying a constant but unnecessary torque.
    action_magnitude = -0.002 * jp.mean(jp.square(action))
    lateral_drift = -0.05 * jp.square(local_linear_velocity[1])
    vertical_velocity = -0.05 * jp.square(local_linear_velocity[2])
    angular_velocity = -0.03 * jp.sum(jp.square(local_angular_velocity[:2]))
    joint_q = data.qpos[7:13][jp.asarray([0, 1, 3, 4])]
    margin = jp.minimum(joint_q - LEG_Q_MIN, LEG_Q_MAX - joint_q)
    joint_limit = -0.05 * jp.sum(jp.square(jp.clip(0.10 - margin, 0.0)))
    torque_saturation = -0.003 * jp.mean(jp.square(info["last_torque"] / 40.0))
    collision = jp.where(nonwheel_collision, -5.0, 0.0)
    return {
        "upright": 2.0 * upright,
        "vx_tracking": 0.70 * vx_tracking * upright_gate,
        "yaw_tracking": 0.35 * yaw_tracking * upright_gate,
        "height": 0.60 * height * upright_gate,
        "posture": 0.30 * posture,
        "stable": 0.20 * stable,
        "action_rate": action_rate,
        "action_smooth": action_smooth,
        "action_magnitude": action_magnitude,
        "lateral_drift": lateral_drift,
        "vertical_velocity": vertical_velocity,
        "angular_velocity": angular_velocity,
        "joint_limit": joint_limit,
        "torque_saturation": torque_saturation,
        "collision": collision,
    }


def jump_terms(data, info, gravity, wheel_contact):
    """Phase-gated jump terms; the state machine never commands the motors."""
    phase = info["jump_phase"]
    is_crouch = phase == PHASE_CROUCH
    is_thrust = phase == PHASE_THRUST
    is_flight = phase == PHASE_FLIGHT
    is_landing = phase == PHASE_LANDING
    is_recovery = phase == PHASE_RECOVERY
    leg_displacement = jp.mean(jp.abs(data.qpos[7:13][jp.asarray([0, 1, 3, 4])] - STANCE[[0, 1, 3, 4]]))
    crouch = jp.where(is_crouch, jp.exp(-8.0 * jp.square(leg_displacement - 0.45)), 0.0)
    thrust = jp.where(is_thrust, jp.clip(data.qvel[2], 0.0, 3.0), 0.0)
    achieved_height = jp.maximum(0.0, info["jump_apex_height"] - info["jump_takeoff_height"])
    height = jp.where(
        is_flight | is_landing | is_recovery,
        jp.exp(-20.0 * jp.square(achieved_height - info["command"][4])),
        0.0,
    )
    clearance = jp.where(is_flight, jp.clip(info["jump_wheel_apex_height"] - 0.25, 0.0, 0.50), 0.0)
    landing = jp.where(info["jump_landing_event"] > 0, -0.25 * jp.abs(info["jump_landing_vz"]), 0.0)
    landing += jp.where(is_landing, 0.20 * (gravity[2] < -0.75) * jp.all(wheel_contact), 0.0)
    recovery = jp.where(info["jump_success_event"] > 0, 3.0, 0.0)
    failure = jp.where(info["jump_failure_event"] > 0, -3.0, 0.0)
    return {
        "jump_crouch": crouch,
        "jump_thrust": 0.5 * thrust,
        "jump_height": 1.5 * height,
        "jump_clearance": clearance,
        "jump_landing": landing,
        "jump_recovery": recovery,
        "jump_failure": failure,
    }
