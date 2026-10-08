from types import SimpleNamespace

import pytest
import torch

from ascento_mjlab.mdp.jump import PHASE_CROUCH, PHASE_FLIGHT, PHASE_IDLE, PHASE_THRUST
from ascento_mjlab.mdp.recovery import recovery_progress
from ascento_mjlab.mdp.rewards import (
    GeneralistTargetArrivalSettledStopBonus,
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
    world_target_heading,
    world_target_progress_velocity,
    world_target_proximity,
    world_target_speed_penalty,
)


def _env(*, step_dt: float = 0.01):
    motion_command = torch.tensor([[0.50, 0.25, 0.80, 1.0, 0.20, 0.10]])
    twist_command = torch.tensor([[0.50, 0.0, 0.50]])
    robot = SimpleNamespace(
        actuators=(
            SimpleNamespace(controller_requested_effort=torch.full((1, 4), 40.0)),
            SimpleNamespace(controller_requested_effort=torch.full((1, 2), 40.0)),
        ),
        joint_names=(
            "left_hip",
            "left_knee",
            "left_wheel",
            "right_hip",
            "right_knee",
            "right_wheel",
        ),
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.0, 0.0, 0.80]]),
            root_link_lin_vel_b=torch.tensor([[0.50, 0.30, 0.0]]),
            root_link_lin_vel_w=torch.tensor([[0.0, 0.0, (2.0 * 9.81 * 0.20) ** 0.5]]),
            root_link_ang_vel_b=torch.tensor([[0.0, 0.0, 0.25]]),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
            actuator_force=torch.full((1, 6), 40.0),
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
        action_manager=SimpleNamespace(
            action=torch.full((1, 6), 0.20), prev_action=torch.zeros((1, 6))
        ),
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

    assert track_linear_velocity_xy(
        env, "twist", std=0.5, asset_cfg=asset_cfg
    ).item() == pytest.approx(torch.exp(torch.tensor(-1.0)).item())
    assert track_yaw_rate(env, "twist", std=0.25, asset_cfg=asset_cfg).item() == pytest.approx(
        torch.exp(torch.tensor(-4.0)).item()
    )


def test_jump_objectives_consume_the_motion_command_without_forward_conflict():
    env = _env()
    asset_cfg = SimpleNamespace(name="robot")

    assert jump_commanded_height_tracking(env, asset_cfg=asset_cfg).item() == pytest.approx(0.05)
    assert lateral_speed_penalty(env, asset_cfg=asset_cfg).item() == pytest.approx(0.09)
    assert track_motion_forward_velocity(env, asset_cfg=asset_cfg).item() == pytest.approx(0.0)
    assert track_motion_yaw_rate(env, asset_cfg=asset_cfg).item() == pytest.approx(1.0)
    assert jump_crouch(env, asset_cfg=asset_cfg).item() == pytest.approx(
        torch.exp(torch.tensor(-((0.11 / 0.05) ** 2))).item()
    )

    env.ascento_jump_state["phase"][0] = PHASE_THRUST
    assert jump_thrust(env, asset_cfg=asset_cfg).item() == pytest.approx(1.0)

    env.ascento_jump_state["phase"][0] = PHASE_IDLE
    assert track_motion_forward_velocity(env, asset_cfg=asset_cfg).item() == pytest.approx(1.0)
    env.ascento_jump_state["phase"][0] = PHASE_FLIGHT
    assert track_motion_forward_velocity(env, asset_cfg=asset_cfg).item() == pytest.approx(0.0)


def test_regularizers_preserve_100_hz_scale_and_are_frequency_aware():
    env = _env(step_dt=0.01)
    asset_cfg = SimpleNamespace(name="robot", actuator_ids=[0, 1, 2, 3, 4, 5])

    assert effort_penalty(env, asset_cfg=asset_cfg, peak_effort_nm=40.0).item() == pytest.approx(
        1.0
    )
    assert action_rate_penalty(env).item() == pytest.approx(0.04)

    slower = _env(step_dt=0.02)
    slower.action_manager.action[:] = 0.40
    assert action_rate_penalty(slower).item() == pytest.approx(0.04)


def test_effort_target_barrier_is_zero_below_soft_limit_and_one_at_limit():
    env = _env()
    asset_cfg = SimpleNamespace(name="robot", actuator_ids=[0, 1, 2, 3, 4, 5])
    env.scene["robot"].actuators[0].controller_requested_effort[:] = 30.0
    env.scene["robot"].actuators[1].controller_requested_effort[:] = 30.0
    assert effort_target_barrier(
        env, asset_cfg=asset_cfg, peak_effort_nm=40.0
    ).item() == pytest.approx(0.0)
    env.scene["robot"].actuators[0].controller_requested_effort[:] = 40.0
    env.scene["robot"].actuators[1].controller_requested_effort[:] = 40.0
    assert effort_target_barrier(
        env, asset_cfg=asset_cfg, peak_effort_nm=40.0
    ).item() == pytest.approx(1.0)


def test_recovery_progress_requires_support_and_accounts_for_angular_speed():
    env = _env()
    asset_cfg = SimpleNamespace(name="robot")
    supported = recovery_progress(env, asset_cfg=asset_cfg)
    env.scene["right_wheel_contact"].data.found.zero_()
    unsupported = recovery_progress(env, asset_cfg=asset_cfg)

    assert supported.item() > 0.0
    assert unsupported.item() == pytest.approx(0.0)


def test_world_target_rewards_distinguish_arrival_from_near_target_departure():
    env = _env()
    robot = env.scene["robot"]
    robot.data.root_link_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    robot.data.root_link_pos_w[0, :2] = torch.tensor([0.0, 0.0])
    env.ascento_world_target_state = {
        "target_xy": torch.tensor([[0.03, 0.0]]),
        "target_yaw": torch.tensor([0.0]),
    }
    robot.data.root_link_lin_vel_w.zero_()
    robot.data.root_link_lin_vel_b.zero_()

    arrived_progress = world_target_progress_velocity(env).item()
    arrived_proximity = world_target_proximity(env).item()
    arrived_speed_cost = world_target_speed_penalty(env).item()

    robot.data.root_link_lin_vel_w[0, 0] = -0.25
    robot.data.root_link_lin_vel_b[0, 0] = -0.25
    away_progress = world_target_progress_velocity(env).item()
    away_proximity = world_target_proximity(env).item()
    away_speed_cost = world_target_speed_penalty(env).item()

    assert arrived_progress == pytest.approx(0.0)
    assert arrived_proximity > 0.99
    assert arrived_speed_cost == pytest.approx(0.0)
    assert away_progress < 0.0
    assert away_proximity == pytest.approx(arrived_proximity)
    assert away_speed_cost > 0.0

    env.ascento_world_target_state["target_yaw"][0] = 0.5
    assert world_target_heading(env).item() < 1.0


def test_generalist_arrival_settle_bonus_is_once_per_attempt_and_requires_dwell():
    env = _env()
    env.num_envs = 1
    robot = env.scene["robot"]
    robot.data.root_link_pos_w[:] = torch.tensor([[0.0, 0.0, 0.75]])
    robot.data.root_link_lin_vel_w.zero_()
    robot.data.root_link_lin_vel_b.zero_()
    robot.data.root_link_ang_vel_b.zero_()
    robot.data.root_link_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    robot.data.projected_gravity_b[:] = torch.tensor([[0.0, 0.0, -1.0]])
    env.ascento_world_target_state = {
        "target_xy": torch.tensor([[0.03, 0.0]]),
        "target_yaw": torch.tensor([0.0]),
    }
    env.ascento_generalist_episode_metrics = {"attempt_id": torch.tensor([1])}
    term = GeneralistTargetArrivalSettledStopBonus(None, env)

    returns = [term(env, hold_s=0.35, target_reached_distance_m=0.035).item() for _ in range(40)]
    assert sum(returns) == pytest.approx(1.0 / env.step_dt)
    assert returns.count(1.0 / env.step_dt) == 1

    env.ascento_generalist_episode_metrics["attempt_id"][0] = 2
    second_attempt = [
        term(env, hold_s=0.35, target_reached_distance_m=0.035).item() for _ in range(40)
    ]
    assert sum(second_attempt) == pytest.approx(1.0 / env.step_dt)

    env.ascento_generalist_episode_metrics["attempt_id"][0] = 3
    env.ascento_world_target_state["target_xy"][0, 0] = 0.15
    overshoot = [term(env, hold_s=0.35, target_reached_distance_m=0.035).item() for _ in range(40)]
    assert sum(overshoot) == pytest.approx(0.0)


def test_generalist_arrival_settle_bonus_changes_reward_contract():
    from ascento_mjlab.physics import REWARD_SCHEMA_VERSION
    from ascento_mjlab.tasks.generalist_locomotion.env_cfg import (
        ascento_generalist_locomotion_env_cfg,
    )

    cfg = ascento_generalist_locomotion_env_cfg(play=False, num_envs=2)
    term = cfg.rewards["target_arrival_settled_stop"]
    assert term.weight == pytest.approx(5.0)
    assert term.params["hold_s"] == pytest.approx(0.35)
    assert REWARD_SCHEMA_VERSION == "v5"
