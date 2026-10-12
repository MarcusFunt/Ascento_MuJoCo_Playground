"""Regression contract for the opt-in near-goal navigation experiment."""

from types import SimpleNamespace

import pytest
import torch
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import ascento_mjlab.tasks  # noqa: F401
from ascento_mjlab.mdp import rewards
from ascento_mjlab.task_contract import current_task_contract
from ascento_mjlab.tasks.generalist_locomotion.env_cfg import (
    ascento_generalist_goal_settle_v2_env_cfg,
    ascento_generalist_locomotion_env_cfg,
)

V1 = "Ascento-Generalist-Locomotion-Flat"
V2 = "Ascento-Generalist-Locomotion-Goal-Settle-V2-Flat"


def sample_env(*, distance=0.115, velocity=0.05, yaw=0.0):
    q = torch.tensor([[__import__("math").cos(yaw / 2), 0.0, 0.0, __import__("math").sin(yaw / 2)]])
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.0, 0.0, 0.75]]),
            root_link_lin_vel_w=torch.tensor([[velocity, 0.0, 0.0]]),
            root_link_lin_vel_b=torch.tensor([[velocity, 0.0, 0.0]]),
            root_link_ang_vel_b=torch.zeros((1, 3)),
            root_link_quat_w=q,
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
            "target_xy": torch.tensor([[distance, 0.0]]),
            "target_yaw": torch.tensor([0.0]),
        },
        ascento_generalist_episode_metrics={"attempt_id": torch.tensor([1])},
    )


def test_new_task_is_versioned_without_modifying_old_reward_or_goal_thresholds():
    old = ascento_generalist_locomotion_env_cfg(num_envs=2)
    new = ascento_generalist_goal_settle_v2_env_cfg(num_envs=2)
    assert old.task_id == V1 and new.task_id == V2
    assert old.rewards["world_target_progress"].func is rewards.world_target_progress_velocity
    assert new.rewards["world_target_progress"].func is rewards.goal_settle_progress_velocity
    assert old.events["repeated_random_world_targets"].params["target_reached_distance_m"] == pytest.approx(0.035)
    assert new.events["repeated_random_world_targets"].params["target_reached_distance_m"] == pytest.approx(0.10)
    assert new.events["repeated_random_world_targets"].params["target_hold_s"] == pytest.approx(0.50)
    assert new.events["repeated_random_world_targets"].params["target_heading_tolerance_rad"] == pytest.approx(0.20)
    assert new.rewards["target_arrival_settled_stop"].params["target_reached_distance_m"] == pytest.approx(0.10)
    assert new.rewards["target_arrival_settled_stop"].params["hold_s"] == pytest.approx(0.50)
    assert set(old.observations["actor"].terms) == set(new.observations["actor"].terms)
    assert set(old.actions) == set(new.actions)
    assert current_task_contract(old) != current_task_contract(new)
    assert load_env_cfg(V2, play=True).task_id == V2
    assert load_rl_cfg(V2).experiment_name == "ascento_generalist_goal_settle_v2_flat"


def test_progress_reward_remains_signed_and_still_matters_at_115mm():
    env = sample_env()
    new = rewards.goal_settle_progress_velocity(env).item()
    old = rewards.world_target_progress_velocity(env, speed_scale=0.30, stop_distance_m=0.35).item()
    assert new > old * 2
    env.scene["robot"].data.root_link_lin_vel_w[0, 0] = -0.05
    env.scene["robot"].data.root_link_lin_vel_b[0, 0] = -0.05
    assert rewards.goal_settle_progress_velocity(env).item() == pytest.approx(-new)
    env.scene["robot"].data.root_link_lin_vel_w.zero_()
    assert rewards.goal_settle_progress_velocity(env).item() == 0.0


def test_braking_penalty_is_delayed_until_inside_goal_envelope():
    env = sample_env()
    new = rewards.goal_settle_speed_penalty(env).item()
    old = rewards.world_target_speed_penalty(env).item()
    assert 0.0 < new < old
    env.ascento_world_target_state["target_xy"][:] = torch.tensor([[0.03, 0.0]])
    assert rewards.goal_settle_speed_penalty(env).item() > new
    env.ascento_world_target_state["target_xy"][:] = torch.tensor([[0.25, 0.0]])
    assert rewards.goal_settle_speed_penalty(env).item() == 0.0


def test_distant_heading_and_settling_are_not_as_rewarding_as_near_goal():
    far = sample_env(distance=0.5, velocity=0.0)
    near = sample_env(distance=0.02, velocity=0.0)
    h_far = rewards.goal_settle_final_heading(far).item()
    h_near = rewards.goal_settle_final_heading(near).item()
    assert h_near > h_far * 4
    b_far = rewards.goal_settle_settled_balance(far).item()
    b_near = rewards.goal_settle_settled_balance(near).item()
    assert b_near > b_far * 5


def test_completion_bonus_pays_once_only_after_dwell_with_correct_yaw():
    env = sample_env(distance=0.08, velocity=0.0)
    bonus = rewards.GeneralistTargetArrivalSettledStopBonus(None, env)
    values = [bonus(env, hold_s=0.50, target_reached_distance_m=0.10,
                    max_heading_error_rad=0.20).item() for _ in range(60)]
    assert sum(values) * env.step_dt == pytest.approx(1.0)
    env.ascento_generalist_episode_metrics["attempt_id"][:] = 2
    env.scene["robot"].data.root_link_quat_w[:] = torch.tensor([[0.98, 0., 0., 0.199]])
    assert all(bonus(env, hold_s=0.50, target_reached_distance_m=0.10,
                     max_heading_error_rad=0.20).item() == 0 for _ in range(55))


def test_reward_calculation_is_finite_at_goal_and_outside_envelope():
    for distance in (0.0, 0.001, 0.08, 0.115, 0.3, 3.0):
        env = sample_env(distance=distance)
        for f in (rewards.goal_settle_progress_velocity, rewards.goal_settle_speed_penalty,
                  rewards.goal_settle_final_heading, rewards.goal_settle_settled_balance):
            assert torch.isfinite(f(env)).all()



def test_episode_reward_ranks_settled_completion_above_parking_outside():
    """Scripted state traces test objective arithmetic, not physics feasibility."""
    def trace(*, finish: bool, wrong_heading: bool = False) -> float:
        env = sample_env(distance=0.115, velocity=0.0)
        term = rewards.GeneralistTargetArrivalSettledStopBonus(None, env)
        reward = 0.0
        dt = env.step_dt
        for step in range(1200):  # 12-second identical horizon
            if finish and step < 120:
                # Approach the remaining 7 cm at about 0.058 m/s and settle.
                # Target coordinate stays fixed and we move the robot toward it.
                env.scene["robot"].data.root_link_pos_w[0, 0] = 0.070 * (step / 120.0)
                speed = 0.070 / 1.20
            elif finish:
                env.scene["robot"].data.root_link_pos_w[0, 0] = 0.070
                speed = 0.0
            else:
                speed = 0.0
            env.scene["robot"].data.root_link_lin_vel_w[0, 0] = speed
            env.scene["robot"].data.root_link_lin_vel_b[0, 0] = speed
            if wrong_heading:
                # Approximately 0.3 rad, outside the 0.2 rad completion tolerance.
                env.scene["robot"].data.root_link_quat_w[:] = torch.tensor(
                    [[0.9888, 0.0, 0.0, 0.1494]]
                )
            r = (
                10.0 * rewards.goal_settle_progress_velocity(env)
                - 2.0 * rewards.goal_settle_speed_penalty(env)
                + 1.0 * rewards.goal_settle_final_heading(env)
                + 1.0 * rewards.goal_settle_settled_balance(env)
            )
            r += 8.0 * term(
                env,
                hold_s=0.50,
                target_reached_distance_m=0.10,
                max_heading_error_rad=0.20,
            )
            reward += r.item() * dt
        return reward

    parked = trace(finish=False)
    completed = trace(finish=True)
    wrong_heading = trace(finish=True, wrong_heading=True)
    assert completed > parked + 5.0
    assert completed > wrong_heading + 5.0



def test_real_mjlab_rollout_reward_terms_recompose_and_remain_finite():
    """Exercise the actual reward manager and event loop, not just mock tensors."""
    from mjlab.envs import ManagerBasedRlEnv

    cfg = ascento_generalist_goal_settle_v2_env_cfg(num_envs=3)
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        obs, _ = env.reset()
        assert obs["actor"].shape == (3, 42)
        for _ in range(8):
            _, reward, terminated, truncated, _ = env.step(torch.zeros((3, 6)))
            weighted_rates = env.reward_manager._step_reward
            assert weighted_rates.shape == (3, len(env.reward_manager.active_terms))
            assert torch.isfinite(weighted_rates).all()
            assert torch.isfinite(reward).all()
            assert torch.allclose(
                reward, weighted_rates.sum(dim=1) * env.step_dt, atol=1e-7, rtol=1e-5
            )
            assert terminated.shape == truncated.shape == (3,)
    finally:
        env.close()
