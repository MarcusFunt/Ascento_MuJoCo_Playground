from types import SimpleNamespace

import jax.numpy as jp

from ascento import jump
from ascento.constants import PHASE_CROUCH, PHASE_FLIGHT, PHASE_IDLE, PHASE_LANDING, PHASE_RECOVERY, PHASE_THRUST


def _data(z=0.75, vz=0.0):
    return SimpleNamespace(qpos=jp.asarray([0.0, 0.0, z]), qvel=jp.asarray([0.0, 0.0, vz]))


def test_synthetic_jump_transitions_and_success_event():
    info = dict(jump.initial_jump_info(), command=jp.asarray([0.0, 0.0, 0.75, 1.0, 0.1, 0.0]))
    gravity = jp.asarray([0.0, 0.0, -1.0])
    contact = jp.asarray([True, True])
    info = jump.update_jump_info(info, _data(), contact, gravity, jp.asarray([0.25, 0.25]))
    assert int(info["jump_phase"]) == PHASE_CROUCH
    for _ in range(16):
        info = jump.update_jump_info(info, _data(), contact, gravity, jp.asarray([0.25, 0.25]))
    assert int(info["jump_phase"]) == PHASE_THRUST
    for _ in range(2):
        info = jump.update_jump_info(info, _data(0.82, 1.0), jp.asarray([False, False]), gravity, jp.asarray([0.32, 0.32]))
    assert int(info["jump_phase"]) == PHASE_FLIGHT
    for _ in range(2):
        info = jump.update_jump_info(info, _data(0.80, -1.0), contact, gravity, jp.asarray([0.25, 0.25]))
    assert int(info["jump_phase"]) == PHASE_LANDING
    for _ in range(11):
        info = jump.update_jump_info(info, _data(), contact, gravity, jp.asarray([0.25, 0.25]))
    assert int(info["jump_phase"]) == PHASE_RECOVERY
    for _ in range(26):
        info = jump.update_jump_info(info, _data(), contact, gravity, jp.asarray([0.25, 0.25]))
        if float(info["jump_success_event"]) == 1.0:
            break
    assert int(info["jump_phase"]) == PHASE_IDLE
    assert float(info["jump_success_event"]) == 1.0
