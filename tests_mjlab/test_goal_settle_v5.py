"""V5 expanded training tilt acceptance regression tests."""

from math import cos, radians, sin
from types import SimpleNamespace

import pytest
import torch
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import ascento_mjlab.tasks  # noqa: F401
from ascento_mjlab.mdp import events, rewards
from ascento_mjlab.task_contract import current_task_contract
from ascento_mjlab.tasks.generalist_locomotion.env_cfg import (
    ascento_generalist_goal_settle_v4_env_cfg,
    ascento_generalist_goal_settle_v5_env_cfg,
)

V5 = "Ascento-Generalist-Locomotion-Goal-Settle-V5-Flat"


def _mock_env(tilt_deg=7.0):
    a = radians(tilt_deg)
    data = SimpleNamespace(
        root_link_pos_w=torch.tensor([[0.0, 0.0, 0.75]]),
        root_link_lin_vel_b=torch.zeros((1, 3)),
        root_link_ang_vel_b=torch.zeros((1, 3)),
        root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        projected_gravity_b=torch.tensor([[sin(a), 0.0, -cos(a)]]),
    )
    robot = SimpleNamespace(data=data)
    left = SimpleNamespace(data=SimpleNamespace(found=torch.ones((1, 1), dtype=torch.bool)))
    right = SimpleNamespace(data=SimpleNamespace(found=torch.ones((1, 1), dtype=torch.bool)))
    return SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.01,
        scene={"robot": robot, "left_wheel_contact": left, "right_wheel_contact": right},
        ascento_world_target_state={
            "target_xy": torch.tensor([[0.025, 0.0]]),
            "target_yaw": torch.tensor([0.0]),
        },
        ascento_generalist_episode_metrics={"attempt_id": torch.tensor([1])},
    )


def test_v5_contract_preserves_v4_except_envelope():
    v4 = ascento_generalist_goal_settle_v4_env_cfg(num_envs=3)
    v5 = ascento_generalist_goal_settle_v5_env_cfg(num_envs=3)
    assert v4.task_id != v5.task_id == V5
    assert v4.events["repeated_random_world_targets"].params.get("target_max_tilt_rad") is None
    assert v5.events["repeated_random_world_targets"].params["target_max_tilt_rad"] == 0.16
    assert v5.events["repeated_random_world_targets"].params["target_split_angular_limits"] is True
    assert "target_split_angular_limits" not in v4.events["repeated_random_world_targets"].params
    assert v5.rewards["target_arrival_settled_stop"].params["max_tilt_rad"] == 0.16
    assert v5.rewards["target_settled_quality"].params["max_tilt_rad"] == 0.24
    assert v5.rewards["target_settled_quality"].params["max_heading_error_rad"] == 0.35
    assert v5.rewards["target_arrival_settled_stop"].params["hold_s"] == 0.5
    assert v5.events["repeated_random_world_targets"].params["target_hold_s"] == 0.5
    assert v5.rewards["target_arrival_settled_stop"].weight == 30
    assert v5.rewards["target_settled_quality"].weight == 6
    assert v5.rewards["pending_goal_time_cost"].weight == -1
    assert v5.rewards["failure_termination_impulse"].weight == -30
    assert set(v4.observations["actor"].terms) == set(v5.observations["actor"].terms)
    assert set(v4.actions) == set(v5.actions)
    assert current_task_contract(v5) != current_task_contract(v4)
    assert load_env_cfg(V5, play=True).task_id == V5
    assert load_rl_cfg(V5).experiment_name == "ascento_generalist_goal_settle_v5_flat"


def test_v5_training_event_accepts_7_degrees_but_rejects_12():
    env = _mock_env(7)
    ids = torch.tensor([0])
    robot = env.scene["robot"]
    assert not events._is_settled_for_locomotion(env, robot, ids).item()
    assert events._is_settled_for_locomotion(
        env, robot, ids, max_heading_error_rad=0.2, max_tilt_rad=0.16
    ).item()
    a = radians(12)
    robot.data.projected_gravity_b[:] = torch.tensor([[sin(a), 0.0, -cos(a)]])
    assert not events._is_settled_for_locomotion(
        env, robot, ids, max_heading_error_rad=0.2, max_tilt_rad=0.16
    ).item()


def test_v5_event_and_reward_agree_on_low_angular_motion():
    env = _mock_env(7)
    ids = torch.tensor([0])
    env.scene["robot"].data.root_link_ang_vel_b[:] = torch.tensor([[0.20, 0.0, 0.20]])
    assert not events._is_settled_for_locomotion(
        env, env.scene["robot"], ids, max_tilt_rad=0.16
    ).item()
    assert events._is_settled_for_locomotion(
        env, env.scene["robot"], ids, max_tilt_rad=0.16, split_angular_limits=True
    ).item()
    env.scene["robot"].data.root_link_ang_vel_b[:] = torch.tensor([[0.26, 0.0, 0.0]])
    assert not events._is_settled_for_locomotion(
        env, env.scene["robot"], ids, max_tilt_rad=0.16, split_angular_limits=True
    ).item()


def test_v5_bonus_and_event_match_on_tilt_and_dwell():
    env = _mock_env(7)
    term = rewards.GeneralistTargetArrivalSettledStopBonus(None, env)
    params = (
        ascento_generalist_goal_settle_v5_env_cfg(num_envs=3)
        .rewards["target_arrival_settled_stop"]
        .params
    )
    received = [term(env, **params).item() for _ in range(60)]
    assert sum(received) * env.step_dt == pytest.approx(1.0)
    assert all(term(env, **params).item() == 0.0 for _ in range(80))

    env = _mock_env(12)
    term = rewards.GeneralistTargetArrivalSettledStopBonus(None, env)
    assert all(term(env, **params).item() == 0.0 for _ in range(80))
    env = _mock_env(7)
    env.scene["right_wheel_contact"].data.found[:] = False
    term = rewards.GeneralistTargetArrivalSettledStopBonus(None, env)
    assert all(term(env, **params).item() == 0.0 for _ in range(80))


def test_v5_wider_quality_signal_is_nonzero_and_still_finite():
    env = _mock_env(7)
    cfg = ascento_generalist_goal_settle_v5_env_cfg(num_envs=3)
    term = rewards.GeneralistTargetSettledQuality(None, env)
    params = cfg.rewards["target_settled_quality"].params
    credit = [term(env, **params).item() * env.step_dt for _ in range(150)]
    assert 0 < sum(credit) <= 1
    assert term(env, **params).item() == 0
    env.scene["right_wheel_contact"].data.found[:] = False
    term.reset()
    assert term(env, **params).item() == 0


def test_v5_real_mjlab_step_reward_parity():
    from mjlab.envs import ManagerBasedRlEnv

    env = ManagerBasedRlEnv(ascento_generalist_goal_settle_v5_env_cfg(num_envs=3), device="cpu")
    try:
        obs, _ = env.reset()
        assert obs["actor"].shape == (3, 42)
        for _ in range(5):
            _, reward, _, _, _ = env.step(torch.zeros((3, 6)))
            assert torch.isfinite(reward).all()
            assert torch.allclose(
                reward,
                env.reward_manager._step_reward.sum(dim=1) * env.step_dt,
                atol=1e-7,
                rtol=1e-5,
            )
    finally:
        env.close()
