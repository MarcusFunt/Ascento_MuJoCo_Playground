"""Regression tests for completion-first, non-farmable generalist rewards."""

from types import SimpleNamespace

import pytest
import torch
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import ascento_mjlab.tasks  # noqa: F401
from ascento_mjlab.mdp import rewards
from ascento_mjlab.task_contract import current_task_contract
from ascento_mjlab.tasks.generalist_locomotion.env_cfg import (
    ascento_generalist_goal_settle_v2_env_cfg,
    ascento_generalist_goal_settle_v3_env_cfg,
)

V2 = "Ascento-Generalist-Locomotion-Goal-Settle-V2-Flat"
V3 = "Ascento-Generalist-Locomotion-Goal-Settle-V3-Flat"


def sample_env(*, target_distance: float = 0.01, velocity: float = 0.0):
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.0, 0.0, 0.75]]),
            root_link_lin_vel_w=torch.tensor([[velocity, 0.0, 0.0]]),
            root_link_lin_vel_b=torch.tensor([[velocity, 0.0, 0.0]]),
            root_link_ang_vel_b=torch.zeros((1, 3)),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
        )
    )
    contact = SimpleNamespace(data=SimpleNamespace(found=torch.ones((1, 1), dtype=torch.bool)))
    return SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.01,
        scene={"robot": robot, "left_wheel_contact": contact, "right_wheel_contact": contact},
        ascento_world_target_state={
            "target_xy": torch.tensor([[target_distance, 0.0]]),
            "target_yaw": torch.tensor([0.0]),
        },
        ascento_generalist_episode_metrics={"attempt_id": torch.tensor([1])},
    )


def test_v3_registration_and_isolated_reward_topology():
    old = ascento_generalist_goal_settle_v2_env_cfg(num_envs=2)
    new = ascento_generalist_goal_settle_v3_env_cfg(num_envs=2)
    assert old.task_id == V2
    assert new.task_id == V3
    assert "pending_goal_time_cost" not in old.rewards
    assert "target_settled_quality" not in old.rewards
    assert new.rewards["pending_goal_time_cost"].func is rewards.GeneralistPendingGoalTimeCost
    assert new.rewards["target_settled_quality"].func is rewards.GeneralistTargetSettledQuality
    assert new.rewards["target_arrival_settled_stop"].weight == 30.0
    assert old.rewards["target_arrival_settled_stop"].weight == 8.0
    assert old.rewards["alive"].weight == 1.0
    assert new.rewards["alive"].weight == 0.15
    assert new.rewards["world_target_progress"].weight == 14.0
    assert old.rewards["world_target_progress"].weight == 10.0
    assert new.events["repeated_random_world_targets"].params == (
        old.events["repeated_random_world_targets"].params
    )
    assert set(new.observations["actor"].terms) == set(old.observations["actor"].terms)
    assert set(new.actions) == set(old.actions)
    assert current_task_contract(old) != current_task_contract(new)
    assert load_env_cfg(V3, play=True).task_id == V3
    assert load_rl_cfg(V3).experiment_name == "ascento_generalist_goal_settle_v3_flat"


def test_unfinished_stationary_goal_cannot_earn_positive_perpetual_reward():
    cfg = ascento_generalist_goal_settle_v3_env_cfg(num_envs=2)
    passive = (
        "alive",
        "upright",
        "height",
        "world_target_heading",
        "track_world_target_yaw_rate",
        "settled_balance",
    )
    maximum_passive_rate = sum(cfg.rewards[name].weight for name in passive)
    pending = cfg.rewards["pending_goal_time_cost"]
    assert pending.weight == -1.0
    assert maximum_passive_rate < pending.params["base_rate"]

    env = sample_env()
    cost = rewards.GeneralistPendingGoalTimeCost(None, env)
    rates = [cost(env, **pending.params).item() for _ in range(1200)]
    assert rates[0] > maximum_passive_rate
    assert rates[-1] == pytest.approx(3.0)
    # Even maximally favorable passive reward cannot pay for 12s of idling.
    assert (sum(maximum_passive_rate - rate for rate in rates) * env.step_dt) < 0


def test_pending_clock_resets_on_a_new_attempt_even_at_identical_coordinates():
    env = sample_env()
    cost = rewards.GeneralistPendingGoalTimeCost(None, env)
    first = cost(env, base_rate=1.7, ramp_rate=1.3, ramp_seconds=8).item()
    for _ in range(250):
        last = cost(env, base_rate=1.7, ramp_rate=1.3, ramp_seconds=8).item()
    assert last > first + 0.3
    env.ascento_generalist_episode_metrics["attempt_id"][:] = 2
    restarted = cost(env, base_rate=1.7, ramp_rate=1.3, ramp_seconds=8).item()
    assert restarted == pytest.approx(first)
    env.ascento_world_target_state["target_xy"][:] = torch.tensor([[0.22, 0.0]])
    assert cost(env, base_rate=1.7, ramp_rate=1.3, ramp_seconds=8).item() == pytest.approx(first)
    for _ in range(10):
        cost(env)
    cost.reset()
    assert cost(env, base_rate=1.7, ramp_rate=1.3, ramp_seconds=8).item() == pytest.approx(first)


def test_pending_clock_ignores_inactive_recovery_hold():
    env = sample_env()
    env.ascento_generalist_episode_metrics["attempt_id"][:] = -1
    cost = rewards.GeneralistPendingGoalTimeCost(None, env)
    assert all(cost(env).item() == 0.0 for _ in range(20))
    env.ascento_generalist_episode_metrics["attempt_id"][:] = 1
    assert cost(env).item() == pytest.approx(1.7 + 1.3 * 0.01 / 8)
    env.ascento_generalist_episode_metrics["attempt_id"][:] = -1
    assert cost(env).item() == 0.0


@pytest.mark.parametrize(
    "base,ramp,seconds", [(-1, 1, 8), (1, -1, 8), (1, 1, 0), (1, 1, float("nan"))]
)
def test_pending_cost_rejects_invalid_config(base, ramp, seconds):
    env = sample_env()
    cost = rewards.GeneralistPendingGoalTimeCost(None, env)
    with pytest.raises(ValueError):
        cost(env, base_rate=base, ramp_rate=ramp, ramp_seconds=seconds)


def test_target_quality_is_capped_and_requires_dual_wheel_support():
    env = sample_env(target_distance=0.01)
    quality = rewards.GeneralistTargetSettledQuality(None, env)
    right = env.scene["right_wheel_contact"].data.found
    right[:] = False
    assert sum(quality(env).item() * env.step_dt for _ in range(80)) == 0.0
    right[:] = True
    total = sum(quality(env).item() * env.step_dt for _ in range(1500))
    assert 0 < total <= 1.0001
    assert quality(env).item() == 0.0


def test_oscillating_inside_circle_does_not_receive_completion_reward():
    env = sample_env(target_distance=0.01)
    bonus = rewards.GeneralistTargetArrivalSettledStopBonus(None, env)
    for step in range(500):
        # Never hold two-wheel support continuously for the required 0.5s.
        env.scene["right_wheel_contact"].data.found[:] = step % 20 < 10
        assert (
            bonus(
                env, hold_s=0.50, target_reached_distance_m=0.10, max_heading_error_rad=0.20
            ).item()
            == 0.0
        )
    env.scene["right_wheel_contact"].data.found[:] = True
    paid = [
        bonus(env, hold_s=0.50, target_reached_distance_m=0.10, max_heading_error_rad=0.20).item()
        for _ in range(60)
    ]
    assert sum(paid) * env.step_dt == pytest.approx(1.0)
    assert all(
        bonus(env, hold_s=0.50, target_reached_distance_m=0.10, max_heading_error_rad=0.20).item()
        == 0.0
        for _ in range(100)
    )


def test_failure_impulse_deters_early_reset_without_punishing_timeout():
    env = sample_env()
    env.termination_manager = SimpleNamespace(
        terminated=torch.tensor([False]),
        time_outs=torch.tensor([False]),
    )
    assert rewards.generalist_failure_termination_impulse(env).item() == 0.0
    env.termination_manager.time_outs[:] = True
    assert rewards.generalist_failure_termination_impulse(env).item() == 0.0
    env.termination_manager.terminated[:] = True
    assert rewards.generalist_failure_termination_impulse(
        env
    ).item() * env.step_dt == pytest.approx(1.0)
    cfg = ascento_generalist_goal_settle_v3_env_cfg(num_envs=2)
    assert cfg.rewards["failure_termination_impulse"].weight == -200.0


def test_progress_remains_signed_and_zero_when_parked():
    env = sample_env(target_distance=0.15, velocity=0.1)
    forward = rewards.goal_settle_progress_velocity(env).item()
    assert forward > 0
    env.scene["robot"].data.root_link_lin_vel_w[:, 0] = -0.1
    assert rewards.goal_settle_progress_velocity(env).item() == pytest.approx(-forward)
    env.scene["robot"].data.root_link_lin_vel_w.zero_()
    assert rewards.goal_settle_progress_velocity(env).item() == 0


def test_real_mjlab_v3_reward_manager_recomposes_step_rewards():
    from mjlab.envs import ManagerBasedRlEnv

    cfg = ascento_generalist_goal_settle_v3_env_cfg(num_envs=3)
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        obs, _ = env.reset()
        assert obs["actor"].shape == (3, 42)
        for _ in range(5):
            _, reward, terminated, truncated, _ = env.step(torch.zeros((3, 6)))
            weighted_rates = env.reward_manager._step_reward
            assert torch.isfinite(weighted_rates).all()
            assert torch.isfinite(reward).all()
            assert torch.allclose(
                reward, weighted_rates.sum(dim=1) * env.step_dt, atol=1e-7, rtol=1e-5
            )
            assert terminated.shape == truncated.shape == (3,)
    finally:
        env.close()
