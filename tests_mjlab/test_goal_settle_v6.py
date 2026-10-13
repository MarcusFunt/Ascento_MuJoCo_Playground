"""Temporary V6 completion-envelope ablation; strict evaluation stays unchanged."""

from math import cos, radians, sin
from types import SimpleNamespace

import pytest
import torch
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import ascento_mjlab.tasks  # noqa: F401
from ascento_mjlab.mdp import rewards
from ascento_mjlab.task_contract import current_task_contract
from ascento_mjlab.tasks.generalist_locomotion.env_cfg import (
    ascento_generalist_goal_settle_v5_env_cfg,
    ascento_generalist_goal_settle_v6_env_cfg,
)

V5 = "Ascento-Generalist-Locomotion-Goal-Settle-V5-Flat"
V6 = "Ascento-Generalist-Locomotion-Goal-Settle-V6-Flat"


def mock_env(*, distance=0.11, tilt_deg=10, yaw=0):
    a = radians(tilt_deg)
    data = SimpleNamespace(
        root_link_pos_w=torch.tensor([[0.0, 0.0, 0.75]]),
        root_link_lin_vel_b=torch.zeros((1, 3)),
        root_link_ang_vel_b=torch.zeros((1, 3)),
        root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        projected_gravity_b=torch.tensor([[sin(a), 0.0, -cos(a)]]),
    )

    def contact():
        return SimpleNamespace(data=SimpleNamespace(found=torch.ones((1, 1), dtype=torch.bool)))

    return SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.01,
        scene={
            "robot": SimpleNamespace(data=data),
            "left_wheel_contact": contact(),
            "right_wheel_contact": contact(),
        },
        ascento_world_target_state={
            "target_xy": torch.tensor([[distance, 0.0]]),
            "target_yaw": torch.tensor([float(yaw)]),
        },
        ascento_generalist_episode_metrics={"attempt_id": torch.tensor([1])},
    )


def test_v6_isolation_and_exact_event_bonus_match():
    v5 = ascento_generalist_goal_settle_v5_env_cfg(num_envs=3)
    v6 = ascento_generalist_goal_settle_v6_env_cfg(num_envs=3)
    v6_event = v6.events["repeated_random_world_targets"].params
    v6_bonus = v6.rewards["target_arrival_settled_stop"]
    v5_event = v5.events["repeated_random_world_targets"].params
    assert v6.task_id == V6 and v5.task_id == V5
    assert current_task_contract(v6) != current_task_contract(v5)
    assert v5_event["target_reached_distance_m"] == 0.10
    assert v5_event["target_hold_s"] == 0.50
    assert v5_event["target_max_tilt_rad"] == 0.16
    for a, b in [
        ("target_reached_distance_m", 0.12),
        ("target_heading_tolerance_rad", 0.28),
        ("target_hold_s", 0.25),
        ("target_max_tilt_rad", 0.20),
    ]:
        assert v6_event[a] == pytest.approx(b)
    assert v6_bonus.params["target_reached_distance_m"] == v6_event["target_reached_distance_m"]
    assert v6_bonus.params["max_heading_error_rad"] == v6_event["target_heading_tolerance_rad"]
    assert v6_bonus.params["hold_s"] == v6_event["target_hold_s"]
    assert v6_bonus.params["max_tilt_rad"] == v6_event["target_max_tilt_rad"]
    assert v6_bonus.weight == v5.rewards["target_arrival_settled_stop"].weight == 30.0
    assert v6.rewards["target_settled_quality"].weight == 6.0
    assert v6.rewards["failure_termination_impulse"].weight == -30.0
    assert load_rl_cfg(V6).experiment_name == "ascento_generalist_goal_settle_v6_flat"
    assert set(v5.observations["actor"].terms) == set(v6.observations["actor"].terms)
    assert set(v5.actions) == set(v6.actions)
    # Play/evaluation is NOT loosened and still has no training-specific bonus.
    play_v5 = load_env_cfg(V5, play=True)
    play_v6 = load_env_cfg(V6, play=True)
    assert set(play_v5.events) == set(play_v6.events)
    assert set(play_v5.rewards) == set(play_v6.rewards)
    assert play_v5.rewards["settled_balance"].params == play_v6.rewards["settled_balance"].params


def test_v6_credits_a_near_goal_10deg_pose_once_not_forever():
    env = mock_env()
    v5 = ascento_generalist_goal_settle_v5_env_cfg(num_envs=3)
    v6 = ascento_generalist_goal_settle_v6_env_cfg(num_envs=3)
    t5 = rewards.GeneralistTargetArrivalSettledStopBonus(None, env)
    t6 = rewards.GeneralistTargetArrivalSettledStopBonus(None, env)
    assert all(
        t5(env, **v5.rewards["target_arrival_settled_stop"].params).item() == 0 for _ in range(55)
    )
    out = [t6(env, **v6.rewards["target_arrival_settled_stop"].params).item() for _ in range(50)]
    assert sum(out) * env.step_dt == pytest.approx(1.0)
    assert out.index(max(out)) == 24
    assert all(
        t6(env, **v6.rewards["target_arrival_settled_stop"].params).item() == 0 for _ in range(200)
    )


@pytest.mark.parametrize("change", ["distance", "heading", "support", "speed", "tilt"])
def test_v6_still_does_not_reward_unfinished_or_unsafe_goals(change):
    env = mock_env()
    data = env.scene["robot"].data
    if change == "distance":
        env.ascento_world_target_state["target_xy"][:] = torch.tensor([[0.13, 0.0]])
    if change == "heading":
        env.ascento_world_target_state["target_yaw"][:] = 0.60
    if change == "support":
        env.scene["right_wheel_contact"].data.found[:] = False
    if change == "speed":
        data.root_link_lin_vel_b[:] = torch.tensor([[0.12, 0.0, 0.0]])
    if change == "tilt":
        a = radians(13)
        data.projected_gravity_b[:] = torch.tensor([[sin(a), 0.0, -cos(a)]])
    params = (
        ascento_generalist_goal_settle_v6_env_cfg(num_envs=3)
        .rewards["target_arrival_settled_stop"]
        .params
    )
    term = rewards.GeneralistTargetArrivalSettledStopBonus(None, env)
    assert all(term(env, **params).item() == 0 for _ in range(55))


def test_v6_dwell_resets_after_one_bad_frame():
    env = mock_env(distance=0.05, tilt_deg=4)
    params = (
        ascento_generalist_goal_settle_v6_env_cfg(num_envs=3)
        .rewards["target_arrival_settled_stop"]
        .params
    )
    term = rewards.GeneralistTargetArrivalSettledStopBonus(None, env)
    for _ in range(20):
        assert term(env, **params).item() == 0
    env.scene["right_wheel_contact"].data.found[:] = False
    assert term(env, **params).item() == 0
    env.scene["right_wheel_contact"].data.found[:] = True
    for _ in range(24):
        assert term(env, **params).item() == 0
    assert term(env, **params).item() == pytest.approx(100.0)


def test_v6_real_cpu_reward_parity():
    from mjlab.envs import ManagerBasedRlEnv

    env = ManagerBasedRlEnv(ascento_generalist_goal_settle_v6_env_cfg(num_envs=3), device="cpu")
    try:
        for _ in range(3):
            _, reward, _, _, _ = env.step(torch.zeros((3, 6)))
            assert torch.isfinite(reward).all()
            assert torch.allclose(
                reward, env.reward_manager._step_reward.sum(dim=1) * env.step_dt, atol=1e-6
            )
    finally:
        env.close()
