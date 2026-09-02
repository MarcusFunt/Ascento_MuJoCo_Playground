from types import SimpleNamespace

import jax.numpy as jp
import numpy as np

from ascento import jump
from ascento.constants import PHASE_FLIGHT, PHASE_THRUST, STANCE
from ascento.observations import bounded_kinematics
from ascento.rewards import base_terms


def test_angular_velocity_observation_preserves_body_local_free_joint_velocity():
    # A 90-degree yaw makes an accidental world-to-body transform visible.
    body_to_world = jp.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    data = SimpleNamespace(qvel=jp.asarray([0.0, 0.0, 0.0, 1.0, -2.0, 3.0] + [0.0] * 6))

    _linear, angular, _joint = bounded_kinematics(data, body_to_world)

    np.testing.assert_allclose(angular, jp.asarray([0.1, -0.2, 0.3]))


def test_posture_reward_is_invariant_to_wheel_angle():
    data = SimpleNamespace(
        qpos=jp.asarray([0.0, 0.0, 0.75, 1.0, 0.0, 0.0, 0.0] + STANCE.tolist()),
        qvel=jp.zeros(12),
    )
    info = {
        "command": jp.asarray([0.0, 0.0, 0.75, 0.0, 0.0, 0.0]),
        "last_action": jp.zeros(6),
        "last_last_action": jp.zeros(6),
        "last_torque": jp.zeros(6),
    }
    kwargs = dict(
        action=jp.zeros(6),
        info=info,
        gravity=jp.asarray([0.0, 0.0, -1.0]),
        local_linear_velocity=jp.zeros(3),
        local_angular_velocity=jp.zeros(3),
        nonwheel_collision=jp.asarray(False),
    )
    nominal = base_terms(data, **kwargs)
    spun = data.qpos.at[9].set(123.0).at[12].set(-87.0)
    rotated = base_terms(data.__class__(qpos=spun, qvel=data.qvel), **kwargs)

    assert float(nominal["posture"]) == float(rotated["posture"])


def test_one_wheel_lift_does_not_trigger_flight():
    info = dict(jump.initial_jump_info(), command=jp.asarray([0.0, 0.0, 0.75, 1.0, 0.1, 0.0]))
    gravity = jp.asarray([0.0, 0.0, -1.0])
    data = SimpleNamespace(qpos=jp.asarray([0.0, 0.0, 0.82]), qvel=jp.asarray([0.0, 0.0, 1.0]))

    info = jump.update_jump_info(info, data, jp.asarray([True, True]), gravity, jp.asarray([0.25, 0.25]))
    for _ in range(16):
        info = jump.update_jump_info(info, data, jp.asarray([True, True]), gravity, jp.asarray([0.25, 0.25]))
    assert int(info["jump_phase"]) == PHASE_THRUST

    for _ in range(3):
        info = jump.update_jump_info(info, data, jp.asarray([True, False]), gravity, jp.asarray([0.32, 0.25]))
    assert int(info["jump_phase"]) == PHASE_THRUST
    assert float(info["jump_takeoff_event"]) == 0.0


def test_two_wheel_no_contact_triggers_flight_after_hysteresis():
    info = dict(jump.initial_jump_info(), command=jp.asarray([0.0, 0.0, 0.75, 1.0, 0.1, 0.0]))
    gravity = jp.asarray([0.0, 0.0, -1.0])
    contact = jp.asarray([True, True])
    data = SimpleNamespace(qpos=jp.asarray([0.0, 0.0, 0.82]), qvel=jp.asarray([0.0, 0.0, 1.0]))

    info = jump.update_jump_info(info, data, contact, gravity, jp.asarray([0.25, 0.25]))
    for _ in range(16):
        info = jump.update_jump_info(info, data, contact, gravity, jp.asarray([0.25, 0.25]))
    assert int(info["jump_phase"]) == PHASE_THRUST

    for _ in range(2):
        info = jump.update_jump_info(info, data, jp.asarray([False, False]), gravity, jp.asarray([0.32, 0.32]))
    assert int(info["jump_phase"]) == PHASE_FLIGHT
