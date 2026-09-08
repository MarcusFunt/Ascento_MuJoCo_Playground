from types import SimpleNamespace

import pytest
import torch

from ascento_mjlab.mdp.jump import PHASE_CROUCH, PHASE_THRUST
from ascento_mjlab.mdp.recovery import recovery_progress
from ascento_mjlab.mdp.rewards import (
    action_rate_penalty,
    effort_penalty,
    effort_target_barrier,
    jump_commanded_height_tracking,
    jump_crouch,
    jump_thrust,
    lateral_speed_penalty,
    track_linear_velocity_xy,
    track_motion_forward_velocity,
    track_motion_yaw_rate,
    track_yaw_rate,
)


def _env(*, step_dt: float = 0.01):
    motion_command = torch.tensor([[0.50, 0.25, 0.80, 1.0, 0.20, 0.10]])
    twist_command = torch.tensor([[0.50, 0.0, 0.50]])
    robot = SimpleNamespace(
        joint_names=("left_hip", "left_knee", "left_wheel", "right_hip", "right_knee", "right_wheel"),
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.0, 0.0, 0.80]]),
            root_link_lin_vel_b=torch.tensor([[0.50, 0.30, 0.0]]),
            root_link_lin_vel_w=torch.tensor([[0.0, 0.0, (2.0 * 9.81 * 0.20) ** 0.5]]),
            root_link_ang_vel_b=torch.tensor([[0.0, 0.0, 0.25]]),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
            actuator_force=torch.full((1, 6), 40.0),
            joint_effort_target=torch.full((1, 6), 40.0),
            joint_pos=torch.zeros((1, 6)),
        ),
    )
    contact = SimpleNamespace(data=SimpleNamespace(found=torch.ones((1, 1), dtype=torch.bool)))
    return SimpleNamespace(
        device="cpu",
        step_dt=step_dt,
        scene={"robot": robot, "left_wheel_contact": contact, "right_wheel_contact": contact},
        command_manager=SimpleNamespace(
            get_command=lambda name: {"motion": motion_command, "twist": twist_command}.get(name)
        ),
        action_manager=SimpleNamespace(action=torch.tensor([[0.20]]), prev_action=torch.zeros((1, 1))),
        ascento_jump_state={
            "phase": torch.tensor([PHASE_CROUCH]),
            "recovered_landing": torch.zeros(1),
        },
    )


def test_velocity_tracking_terms_are_independently_tunable():
    env = _env()
    asset_cfg = SimpleNamespace(name="robot")
    env.scene["robot"].data.root_link_lin_vel_b[0, 0] = 0.0
    env.scene["robot"].data.root_link_lin_vel_b[0, 1] = 0.0
    env.scene["robot"].data.root_link_ang_vel_b[0, 2] = 0.0

    assert track_linear_velocity_xy(env, "twist", std=0.5, asset_cfg=asset_cfg).item() == pytest.approx(
        torch.exp(torch.tensor(-1.0)).item()
    )
    assert track_yaw_rate(env, "twist", std=0.25, asset_cfg=asset_cfg).item() == pytest.approx(
        torch.exp(torch.tensor(-4.0)).item()
    )


def test_jump_objectives_consume_the_motion_command_without_forward_conflict():
    env = _env()
    asset_cfg = SimpleNamespace(name="robot")

    assert jump_commanded_height_tracking(env, asset_cfg=asset_cfg).item() == pytest.approx(0.05)
    assert lateral_speed_penalty(env, asset_cfg=asset_cfg).item() == pytest.approx(0.09)
    assert track_motion_forward_velocity(env, asset_cfg=asset_cfg).item() == pytest.approx(1.0)
    assert track_motion_yaw_rate(env, asset_cfg=asset_cfg).item() == pytest.approx(1.0)
    assert jump_crouch(env, asset_cfg=asset_cfg).item() == pytest.approx(
        torch.exp(torch.tensor(-(0.11 / 0.05) ** 2)).item()
    )

    env.ascento_jump_state["phase"][0] = PHASE_THRUST
    assert jump_thrust(env, asset_cfg=asset_cfg).item() == pytest.approx(1.0)


def test_regularizers_preserve_100_hz_scale_and_are_frequency_aware():
    env = _env(step_dt=0.01)
    asset_cfg = SimpleNamespace(name="robot", actuator_ids=[0, 1, 2, 3, 4, 5])

    assert effort_penalty(env, asset_cfg=asset_cfg, peak_effort_nm=40.0).item() == pytest.approx(1.0)
    assert action_rate_penalty(env).item() == pytest.approx(0.04)

    slower = _env(step_dt=0.02)
    slower.action_manager.action[0, 0] = 0.40
    assert action_rate_penalty(slower).item() == pytest.approx(0.04)


def test_effort_target_barrier_is_zero_below_soft_limit_and_one_at_limit():
    env = _env()
    asset_cfg = SimpleNamespace(name="robot", actuator_ids=[0, 1, 2, 3, 4, 5])
    env.scene["robot"].data.joint_effort_target[:] = 30.0
    assert effort_target_barrier(env, asset_cfg=asset_cfg, peak_effort_nm=40.0).item() == pytest.approx(0.0)
    env.scene["robot"].data.joint_effort_target[:] = 40.0
    assert effort_target_barrier(env, asset_cfg=asset_cfg, peak_effort_nm=40.0).item() == pytest.approx(1.0)


def test_recovery_progress_requires_support_and_accounts_for_angular_speed():
    env = _env()
    asset_cfg = SimpleNamespace(name="robot")
    supported = recovery_progress(env, asset_cfg=asset_cfg)
    env.scene["right_wheel_contact"].data.found.zero_()
    unsupported = recovery_progress(env, asset_cfg=asset_cfg)

    assert supported.item() > 0.0
    assert unsupported.item() == pytest.approx(0.0)
