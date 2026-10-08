from __future__ import annotations

import pytest
import torch
from scripts.reward_freedom_probe import build_report
from scripts.reward_probe import _make_probe_env

import ascento_mjlab.tasks  # noqa: F401 - complete task registration before MDP imports
from ascento_mjlab.mdp.rewards import (
    height_tracking,
    leg_pose_hold_penalty,
    leg_pose_symmetry_penalty,
    settled_balance,
)
from ascento_mjlab.tasks.generalist_locomotion.env_cfg import (
    ascento_generalist_locomotion_env_cfg,
)


def test_wp07b_probe_quantifies_every_required_pose_and_reward_component():
    report = build_report()
    cases = {case["case_id"]: case for case in report["cases"]}

    assert {
        "nominal_flat_settle",
        "symmetric_height_change_with_good_balance",
        "asymmetric_split_wheel_height_pose_with_good_balance",
        "mild_slope_posture_adaptation",
        "scripted_low_obstacle_leg_excursion",
        "flat_mode_unnecessary_pseudo_step",
    } <= cases.keys()
    for case in cases.values():
        assert {
            "leg_pose_symmetry",
            "leg_pose_hold",
            "height",
            "settled_balance",
        } <= case["weighted_return_rate_by_term_per_s"].keys()
        assert "settled_balance_fixed_height_component_delta_per_s" in case


def test_wp07b_probe_identifies_posture_and_height_conflicts_without_weakening_flat_mode():
    report = build_report()
    cases = {case["case_id"]: case for case in report["cases"]}
    nominal = cases["nominal_flat_settle"]
    asymmetric = cases["asymmetric_split_wheel_height_pose_with_good_balance"]
    raised = cases["symmetric_height_change_with_good_balance"]
    flat_step = cases["flat_mode_unnecessary_pseudo_step"]
    obstacle = cases["scripted_low_obstacle_leg_excursion"]

    assert nominal["weighted_return_rate_by_term_per_s"]["leg_pose_symmetry"] == pytest.approx(0)
    assert nominal["weighted_return_rate_by_term_per_s"]["leg_pose_hold"] == pytest.approx(0)
    assert asymmetric["weighted_return_rate_by_term_per_s"]["leg_pose_symmetry"] < -1.0
    assert asymmetric["weighted_return_rate_by_term_per_s"]["leg_pose_hold"] < -0.1
    assert asymmetric["obstacle_mode"] == 1
    assert asymmetric["candidate_mode_conditioned_weighted_return_rate_per_s"][
        "leg_pose_symmetry"
    ] == pytest.approx(0)
    assert asymmetric["candidate_mode_conditioned_weighted_return_rate_per_s"][
        "leg_pose_hold"
    ] == pytest.approx(0)
    assert raised["fixed_height_reward_deficit_per_s"] < -0.5
    assert raised["settled_balance_fixed_height_component_delta_per_s"] < -0.9
    assert flat_step["obstacle_mode"] == 0
    assert flat_step["candidate_mode_conditioned_return_rate_per_s"] == pytest.approx(
        flat_step["current_return_rate_per_s"]
    )
    assert obstacle["obstacle_mode"] == 1
    assert obstacle["candidate_mode_conditioned_weighted_return_rate_per_s"][
        "leg_pose_symmetry"
    ] == pytest.approx(0)
    assert obstacle["candidate_mode_conditioned_weighted_return_rate_per_s"][
        "leg_pose_hold"
    ] == pytest.approx(0)


def test_generalist_reward_freedom_changes_only_the_obstacle_mode_profile():
    cfg = ascento_generalist_locomotion_env_cfg(play=True, num_envs=1)
    pose = [-2.4, -2.5, 0.0, -3.7, -4.0, 0.0]

    env = _make_probe_env()
    env.scene["robot"].data.root_link_pos_w[0, 2] = 0.85
    env.scene["robot"].data.joint_pos = torch.tensor([pose], dtype=torch.float32)
    env.ascento_obstacle_mode = torch.zeros((1, 1))
    assert float(
        cfg.rewards["leg_pose_symmetry"].func(env, **cfg.rewards["leg_pose_symmetry"].params)[0]
    ) == pytest.approx(float(leg_pose_symmetry_penalty(env)[0]))
    assert float(
        cfg.rewards["leg_pose_hold"].func(env, **cfg.rewards["leg_pose_hold"].params)[0]
    ) == pytest.approx(float(leg_pose_hold_penalty(env)[0]))
    assert float(
        cfg.rewards["height"].func(env, **cfg.rewards["height"].params)[0]
    ) == pytest.approx(float(height_tracking(env)[0]))
    assert float(
        cfg.rewards["settled_balance"].func(env, **cfg.rewards["settled_balance"].params)[0]
    ) == pytest.approx(float(settled_balance(env)[0]))

    env.ascento_obstacle_mode = torch.ones((1, 1))
    assert (
        float(
            cfg.rewards["leg_pose_symmetry"].func(env, **cfg.rewards["leg_pose_symmetry"].params)[0]
        )
        == 0.0
    )
    assert (
        float(cfg.rewards["leg_pose_hold"].func(env, **cfg.rewards["leg_pose_hold"].params)[0])
        == 0.0
    )
    assert float(
        cfg.rewards["height"].func(env, **cfg.rewards["height"].params)[0]
    ) == pytest.approx(1.0)
    assert float(
        cfg.rewards["settled_balance"].func(env, **cfg.rewards["settled_balance"].params)[0]
    ) == pytest.approx(1.0)
