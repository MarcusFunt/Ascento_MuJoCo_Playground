from math import cos, sin
from types import SimpleNamespace

import torch

from ascento_mjlab.mdp.recovery import RecoveryEnvelope, RecoverySuccess, recovery_condition


def _recovery_env(*, step_dt: float = 0.01):
    left = torch.ones((1, 1), dtype=torch.bool)
    right = torch.ones((1, 1), dtype=torch.bool)
    robot_data = SimpleNamespace(
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
        root_link_pos_w=torch.tensor([[0.0, 0.0, 0.75]]),
        root_link_lin_vel_b=torch.zeros((1, 3)),
        root_link_ang_vel_b=torch.zeros((1, 3)),
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=step_dt,
        scene={
            "robot": SimpleNamespace(data=robot_data),
            "left_wheel_contact": SimpleNamespace(data=SimpleNamespace(found=left)),
            "right_wheel_contact": SimpleNamespace(data=SimpleNamespace(found=right)),
        },
    )
    return env, robot_data, left, right


def _metric(env, envelope: RecoveryEnvelope | None = None):
    envelope = RecoveryEnvelope() if envelope is None else envelope
    cfg = SimpleNamespace(params={"envelope": envelope})
    return RecoverySuccess(cfg, env)


def test_recovery_condition_rejects_each_failure_dimension_independently():
    envelope = RecoveryEnvelope()

    env, robot, left, right = _recovery_env()
    assert recovery_condition(env, envelope).item()

    right.zero_()
    assert not recovery_condition(env, envelope).item()
    right.fill_(True)

    robot.root_link_ang_vel_b[0, 0] = envelope.max_angular_speed + 0.1
    assert not recovery_condition(env, envelope).item()
    robot.root_link_ang_vel_b.zero_()

    robot.root_link_lin_vel_b[0, 0] = envelope.max_linear_speed + 0.1
    assert not recovery_condition(env, envelope).item()
    robot.root_link_lin_vel_b.zero_()

    angle = envelope.max_tilt_radians + 0.05
    robot.projected_gravity_b[0] = torch.tensor([sin(angle), 0.0, -cos(angle)])
    assert not recovery_condition(env, envelope).item()
    robot.projected_gravity_b[0] = torch.tensor([0.0, 0.0, -1.0])

    robot.root_link_pos_w[0, 2] = envelope.min_height - 0.01
    assert not recovery_condition(env, envelope).item()


def test_recovery_success_requires_the_full_continuous_duration_after_interruption():
    envelope = RecoveryEnvelope(stable_duration_s=0.25)
    env, robot, left, right = _recovery_env(step_dt=0.001)
    metric = _metric(env, envelope)

    for _ in range(249):
        assert metric(env).item() == 0.0

    right.zero_()
    assert metric(env).item() == 0.0
    right.fill_(True)

    for _ in range(249):
        assert metric(env).item() == 0.0
    assert metric(env).item() == 1.0


def test_recovery_success_uses_an_overridden_envelope_duration():
    env, robot, left, right = _recovery_env(step_dt=0.01)
    metric = _metric(env, RecoveryEnvelope(stable_duration_s=0.25))
    short = RecoveryEnvelope(stable_duration_s=0.03)

    assert metric(env, short).item() == 0.0
    assert metric(env, short).item() == 0.0
    assert metric(env, short).item() == 1.0
