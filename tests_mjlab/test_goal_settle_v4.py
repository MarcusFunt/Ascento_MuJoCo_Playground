"""Versioned goal-settle-v4 reward contract and short-run failure safeguards."""

import pytest
import torch
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import ascento_mjlab.tasks  # noqa: F401
from ascento_mjlab.task_contract import current_task_contract
from ascento_mjlab.tasks.generalist_locomotion.env_cfg import (
    ascento_generalist_goal_settle_v2_env_cfg,
    ascento_generalist_goal_settle_v3_env_cfg,
    ascento_generalist_goal_settle_v4_env_cfg,
)

V4 = "Ascento-Generalist-Locomotion-Goal-Settle-V4-Flat"


def test_v4_is_isolated_and_preserves_actor_and_goal_completion_abi():
    v2 = ascento_generalist_goal_settle_v2_env_cfg(num_envs=3)
    v3 = ascento_generalist_goal_settle_v3_env_cfg(num_envs=3)
    v4 = ascento_generalist_goal_settle_v4_env_cfg(num_envs=3)
    assert v4.task_id == V4
    assert v2.rewards["target_arrival_settled_stop"].weight == 8.0
    assert v3.rewards["failure_termination_impulse"].weight == -200.0
    assert v4.rewards["failure_termination_impulse"].weight == -30.0
    assert "failure_termination_impulse" not in v2.rewards
    assert set(v3.rewards) == set(v4.rewards)
    for key in v3.rewards:
        if key == "failure_termination_impulse":
            continue
        old, new = v3.rewards[key], v4.rewards[key]
        assert old.func is new.func
        assert old.weight == new.weight
        assert old.params == new.params
    previous_event = v3.events["repeated_random_world_targets"].params
    next_event = v4.events["repeated_random_world_targets"].params
    assert next_event["navigation_stage_independent_of_recovery"] is True
    assert "navigation_stage_independent_of_recovery" not in previous_event
    assert {
        key: value
        for key, value in next_event.items()
        if key != "navigation_stage_independent_of_recovery"
    } == previous_event
    assert set(v2.observations["actor"].terms) == set(v4.observations["actor"].terms)
    assert set(v2.actions) == set(v4.actions)
    assert current_task_contract(v2) != current_task_contract(v4)
    assert current_task_contract(v3) != current_task_contract(v4)
    assert load_env_cfg(V4, play=True).task_id == V4
    assert load_rl_cfg(V4).experiment_name == "ascento_generalist_goal_settle_v4_flat"


def test_unfinished_goal_still_has_negative_passive_rate():
    cfg = ascento_generalist_goal_settle_v4_env_cfg(num_envs=3)
    passive_names = (
        "alive",
        "upright",
        "height",
        "world_target_heading",
        "track_world_target_yaw_rate",
        "settled_balance",
    )
    passive_max = sum(cfg.rewards[name].weight for name in passive_names)
    params = cfg.rewards["pending_goal_time_cost"].params
    assert passive_max == pytest.approx(1.40)
    assert params["base_rate"] == pytest.approx(1.70)
    assert params["base_rate"] + params["ramp_rate"] == pytest.approx(3.0)
    assert cfg.rewards["target_settled_quality"].weight == pytest.approx(6.0)
    assert cfg.rewards["target_arrival_settled_stop"].weight == pytest.approx(30.0)


def test_terminal_penalty_exceeds_maximum_discounted_pending_cost():
    # This avoids the deliberately extreme -200 critic target in v3, while
    # preserving a locally negative incentive for early termination. It is
    # a heuristic bound, not proof against every deliberate-reset strategy.
    cfg = ascento_generalist_goal_settle_v4_env_cfg(num_envs=3)
    gamma = load_rl_cfg(V4).algorithm.gamma
    pending = cfg.rewards["pending_goal_time_cost"].params
    max_discounted_pending = (pending["base_rate"] + pending["ramp_rate"]) * 0.01 / (1.0 - gamma)
    assert gamma == pytest.approx(0.998)
    assert max_discounted_pending == pytest.approx(15.0)
    assert -cfg.rewards["failure_termination_impulse"].weight > max_discounted_pending


def test_real_mjlab_v4_reward_manager_recomposes_step_rewards():
    from mjlab.envs import ManagerBasedRlEnv

    cfg = ascento_generalist_goal_settle_v4_env_cfg(num_envs=3)
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        obs, _ = env.reset()
        assert obs["actor"].shape == (3, 42)
        for _ in range(5):
            _, reward, terminated, truncated, _ = env.step(torch.zeros((3, 6)))
            weighted = env.reward_manager._step_reward
            assert torch.isfinite(weighted).all()
            assert torch.isfinite(reward).all()
            assert torch.allclose(reward, weighted.sum(dim=1) * env.step_dt, atol=1e-7, rtol=1e-5)
            assert terminated.shape == truncated.shape == (3,)
    finally:
        env.close()


def test_v4_navigation_promotion_does_not_depend_on_unlearned_push_recovery():
    from ascento_mjlab.mdp.events import update_generalist_goal_mix_stage

    inputs = dict(
        stage=0,
        gate_recovery_successes=1,
        gate_episodes=100,
        short_arrival_successes=88,
        short_episodes=100,
        long_arrival_successes=0,
        long_episodes=0,
        minimum_attempts=64,
    )
    assert update_generalist_goal_mix_stage(**inputs)[0] == 0
    assert (
        update_generalist_goal_mix_stage(
            **inputs,
            promotion_gate_lcb_threshold=0.0,
            demotion_gate_lcb_threshold=0.0,
        )[0]
        == 1
    )
    inputs["short_arrival_successes"] = 10
    assert (
        update_generalist_goal_mix_stage(
            **inputs,
            promotion_gate_lcb_threshold=0.0,
            demotion_gate_lcb_threshold=0.0,
        )[0]
        == 0
    )
    inputs.update(stage=1, short_arrival_successes=90, long_arrival_successes=90, long_episodes=100)
    assert (
        update_generalist_goal_mix_stage(
            **inputs,
            promotion_gate_lcb_threshold=0.0,
            demotion_gate_lcb_threshold=0.0,
        )[0]
        == 2
    )
