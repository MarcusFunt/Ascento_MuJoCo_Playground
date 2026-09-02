"""Pure JAX bookkeeping for the six-phase direct-torque jump task."""
from __future__ import annotations

import jax.numpy as jp

from .constants import (
    COMMAND_JUMP_TRIGGER,
    COMMAND_JUMP_DISTANCE,
    NUM_JUMP_PHASES,
    PHASE_CROUCH,
    PHASE_FLIGHT,
    PHASE_IDLE,
    PHASE_LANDING,
    PHASE_RECOVERY,
    PHASE_THRUST,
)


def initial_jump_info():
    """Fixed-shape state used by every environment, including balance."""
    zero = jp.asarray(0.0, jp.float32)
    return {
        "jump_phase": jp.asarray(PHASE_IDLE, jp.int32),
        "phase_steps": jp.asarray(0, jp.int32),
        "no_contact_steps": jp.asarray(0, jp.int32),
        "landing_contact_steps": jp.asarray(0, jp.int32),
        "recovery_stable_steps": jp.asarray(0, jp.int32),
        "jump_start_height": zero,
        "jump_start_x": zero,
        "jump_target_x": zero,
        "jump_takeoff_height": zero,
        "jump_takeoff_x": zero,
        "jump_takeoff_vx": zero,
        "jump_takeoff_vz": zero,
        "jump_apex_height": zero,
        "jump_wheel_apex_height": zero,
        "jump_landing_vx": zero,
        "jump_landing_vz": zero,
        "jump_landing_x": zero,
        "jump_air_time": zero,
        "jump_takeoff_event": zero,
        "jump_landing_event": zero,
        "jump_success_event": zero,
        "jump_failure_event": zero,
    }


def update_jump_info(info: dict, data, wheel_contact, gravity, wheel_height):
    """Advances phase/event state using contacts, not a reference controller."""
    phase = info["jump_phase"]
    steps = info["phase_steps"] + 1
    all_contact = jp.all(wheel_contact)
    any_contact = jp.any(wheel_contact)
    # Flight requires both wheels to be clear.  A one-wheel lift must not
    # count as airborne, while landing still requires both wheels down.
    no_contact_steps = jp.where(any_contact, 0, info["no_contact_steps"] + 1)
    landing_steps = jp.where(all_contact, info["landing_contact_steps"] + 1, 0)
    requested = info["command"][COMMAND_JUMP_TRIGGER] > 0.5
    upright = gravity[2] < -0.80
    stable = upright & all_contact & (data.qpos[2] > 0.55) & (jp.linalg.norm(data.qvel[:6]) < 1.5)
    recovery_steps = jp.where(stable, info["recovery_stable_steps"] + 1, 0)

    to_crouch = (phase == PHASE_IDLE) & requested
    to_thrust = (phase == PHASE_CROUCH) & (steps >= 15)
    to_flight = (phase == PHASE_THRUST) & (no_contact_steps >= 2)
    thrust_failed = (phase == PHASE_THRUST) & (steps >= 75) & ~to_flight
    to_landing = (phase == PHASE_FLIGHT) & (landing_steps >= 2) & (data.qvel[2] < 0.0)
    to_recovery = ((phase == PHASE_LANDING) & (steps >= 10)) | thrust_failed
    recovered = (phase == PHASE_RECOVERY) & (recovery_steps >= 25)
    recovery_failed = (phase == PHASE_RECOVERY) & (steps >= 300)

    next_phase = phase
    next_phase = jp.where(to_crouch, PHASE_CROUCH, next_phase)
    next_phase = jp.where(to_thrust, PHASE_THRUST, next_phase)
    next_phase = jp.where(to_flight, PHASE_FLIGHT, next_phase)
    next_phase = jp.where(to_landing, PHASE_LANDING, next_phase)
    next_phase = jp.where(to_recovery, PHASE_RECOVERY, next_phase)
    next_phase = jp.where(recovered | recovery_failed, PHASE_IDLE, next_phase)
    phase_steps = jp.where(next_phase != phase, 0, steps)

    in_jump = phase != PHASE_IDLE
    apex_height = jp.where(in_jump, jp.maximum(info["jump_apex_height"], data.qpos[2]), info["jump_apex_height"])
    wheel_apex = jp.where(in_jump, jp.maximum(info["jump_wheel_apex_height"], jp.max(wheel_height)), info["jump_wheel_apex_height"])
    start_height = jp.where(to_crouch, data.qpos[2], info["jump_start_height"])
    start_x = jp.where(to_crouch, data.qpos[0], info["jump_start_x"])
    target_x = jp.where(to_crouch, data.qpos[0] + info["command"][COMMAND_JUMP_DISTANCE], info["jump_target_x"])
    takeoff_height = jp.where(to_flight, data.qpos[2], info["jump_takeoff_height"])
    takeoff_x = jp.where(to_flight, data.qpos[0], info["jump_takeoff_x"])
    takeoff_vx = jp.where(to_flight, data.qvel[0], info["jump_takeoff_vx"])
    takeoff_vz = jp.where(to_flight, data.qvel[2], info["jump_takeoff_vz"])
    landing_x = jp.where(to_landing, data.qpos[0], info["jump_landing_x"])
    landing_vx = jp.where(to_landing, data.qvel[0], info["jump_landing_vx"])
    landing_vz = jp.where(to_landing, data.qvel[2], info["jump_landing_vz"])
    air_time = info["jump_air_time"] + jp.where(phase == PHASE_FLIGHT, 0.01, 0.0)

    return dict(
        info,
        jump_phase=next_phase,
        phase_steps=phase_steps,
        no_contact_steps=no_contact_steps,
        landing_contact_steps=landing_steps,
        recovery_stable_steps=recovery_steps,
        jump_start_height=start_height,
        jump_start_x=start_x,
        jump_target_x=target_x,
        jump_takeoff_height=takeoff_height,
        jump_takeoff_x=takeoff_x,
        jump_takeoff_vx=takeoff_vx,
        jump_takeoff_vz=takeoff_vz,
        jump_apex_height=apex_height,
        jump_wheel_apex_height=wheel_apex,
        jump_landing_x=landing_x,
        jump_air_time=air_time,
        jump_landing_vx=landing_vx,
        jump_landing_vz=landing_vz,
        jump_takeoff_event=to_flight.astype(jp.float32),
        jump_landing_event=to_landing.astype(jp.float32),
        jump_success_event=recovered.astype(jp.float32),
        jump_failure_event=(thrust_failed | recovery_failed).astype(jp.float32),
    )


def phase_one_hot(phase):
    return jp.eye(NUM_JUMP_PHASES, dtype=jp.float32)[phase]
