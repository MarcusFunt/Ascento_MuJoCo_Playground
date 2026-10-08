#!/usr/bin/env python3
"""Quantify posture and fixed-height reward terms on scripted WP-07B probes.

The states are arithmetic probes only. They do not assert dynamic feasibility,
physical safety, or policy performance.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from scripts.reward_probe import (
    DEFAULT_JOINT_POSITION,
    _make_probe_env,
    _projected_gravity_b,
    _quaternion_wxyz,
)

import ascento_mjlab.tasks  # noqa: F401 - finish task registration before MDP imports
from ascento_mjlab.tasks.generalist_locomotion.env_cfg import (
    ascento_generalist_locomotion_env_cfg,
)

_HEIGHT_RANGE_TOLERANCE_M = 0.10
_AUDITED_TERMS = ("leg_pose_symmetry", "leg_pose_hold", "height", "settled_balance")
_PROBES = (
    {
        "case_id": "nominal_flat_settle",
        "obstacle_mode": 0,
        "height_m": 0.75,
        "roll_rad": 0.0,
        "pitch_rad": 0.0,
        "joint_positions_rad": DEFAULT_JOINT_POSITION[0].tolist(),
    },
    {
        "case_id": "symmetric_height_change_with_good_balance",
        "obstacle_mode": 0,
        "height_m": 0.85,
        "roll_rad": 0.0,
        "pitch_rad": 0.0,
        "joint_positions_rad": DEFAULT_JOINT_POSITION[0].tolist(),
    },
    {
        "case_id": "asymmetric_split_wheel_height_pose_with_good_balance",
        "obstacle_mode": 1,
        "height_m": 0.80,
        "roll_rad": 0.0,
        "pitch_rad": 0.0,
        "joint_positions_rad": [-2.4, -2.5, 0.0, -3.7, -4.0, 0.0],
    },
    {
        "case_id": "mild_slope_posture_adaptation",
        "obstacle_mode": 0,
        "height_m": 0.80,
        "roll_rad": 0.0,
        "pitch_rad": math.radians(5.0),
        "joint_positions_rad": [-2.7, -2.8, 0.0, -2.7, -2.8, 0.0],
    },
    {
        "case_id": "scripted_low_obstacle_leg_excursion",
        "obstacle_mode": 1,
        "height_m": 0.83,
        "roll_rad": 0.0,
        "pitch_rad": 0.0,
        "joint_positions_rad": [-1.9, -2.1, 0.0, -math.pi, -math.pi, 0.0],
    },
    {
        "case_id": "flat_mode_unnecessary_pseudo_step",
        "obstacle_mode": 0,
        "height_m": 0.75,
        "roll_rad": 0.0,
        "pitch_rad": 0.0,
        "joint_positions_rad": [-2.4, -2.4, 0.0, -3.8, -3.8, 0.0],
    },
)


def _pose_env(pose: dict[str, Any], *, mode: int = 0):
    env = _make_probe_env()
    env.ascento_obstacle_mode = torch.full((1, 1), float(mode))
    data = env.scene["robot"].data
    data.root_link_pos_w[0, 2] = float(pose["height_m"])
    data.root_link_quat_w = _quaternion_wxyz(float(pose["roll_rad"]), float(pose["pitch_rad"]), 0.0)
    data.projected_gravity_b = _projected_gravity_b(
        float(pose["roll_rad"]), float(pose["pitch_rad"])
    )
    data.joint_pos = torch.tensor([pose["joint_positions_rad"]], dtype=torch.float32)
    return env


def _weighted_terms(cfg, env) -> dict[str, float]:
    result = {}
    for name in _AUDITED_TERMS:
        term_cfg = cfg.rewards[name]
        value = term_cfg.func(env, **term_cfg.params)
        result[name] = float(term_cfg.weight) * float(value.reshape(-1)[0])
    return result


def _probe(cfg, pose: dict[str, Any]) -> dict[str, Any]:
    baseline_env = _pose_env(pose, mode=0)
    current = _weighted_terms(cfg, baseline_env)
    candidate_env = _pose_env(pose, mode=int(pose["obstacle_mode"]))
    candidate = _weighted_terms(cfg, candidate_env)
    height_error = abs(float(pose["height_m"]) - 0.75)
    height_score = current["height"] / float(cfg.rewards["height"].weight)
    settled_height_factor = math.exp(-(height_error**2) / 0.05**2)
    settled_without_height = current["settled_balance"] / settled_height_factor
    weighted_settled_weight = float(cfg.rewards["settled_balance"].weight)
    current_total = sum(current.values())
    candidate_total = sum(candidate.values())
    return {
        **pose,
        "height_error_m": height_error,
        "weighted_return_rate_by_term_per_s": current,
        "current_return_rate_per_s": current_total,
        "fixed_height_reward_deficit_per_s": float(cfg.rewards["height"].weight)
        * (height_score - 1.0),
        "settled_balance_fixed_height_component_delta_per_s": weighted_settled_weight
        * settled_without_height
        * (settled_height_factor - 1.0),
        "candidate_mode_conditioned_weighted_return_rate_per_s": candidate,
        "candidate_mode_conditioned_return_rate_per_s": candidate_total,
    }


def build_report() -> dict[str, Any]:
    """Return deterministic per-term reward geometry for the required poses."""
    cfg = ascento_generalist_locomotion_env_cfg(play=True, num_envs=1)
    cases = [_probe(cfg, pose) for pose in _PROBES]
    case_by_id = {case["case_id"]: case for case in cases}
    nominal = case_by_id["nominal_flat_settle"]["current_return_rate_per_s"]
    pseudo_step = case_by_id["flat_mode_unnecessary_pseudo_step"]
    obstacle = case_by_id["scripted_low_obstacle_leg_excursion"]
    return {
        "schema_version": 1,
        "work_package": "WP-07B",
        "probe_type": "scripted_static_reward_arithmetic_not_physics_or_policy_rollouts",
        "task": "Ascento-Generalist-Locomotion-Flat",
        "reward_schema": "v4",
        "audited_terms": {
            "leg_pose_symmetry": "leg_pose_symmetry_penalty",
            "leg_pose_hold": "leg_pose_hold_penalty",
            "height": "fixed_height_tracking",
            "settled_balance_fixed_height_component": "fixed_0.75m_factor_inside_settled_balance",
        },
        "candidate_mode_conditioned_profile": {
            "mode_zero": "leave all existing reward weights and terms unchanged",
            "mode_one_leg_pose_symmetry_weight": 0.0,
            "mode_one_leg_pose_hold_weight": 0.0,
            "mode_one_height_objective": f"no fixed-height penalty within a +/-{_HEIGHT_RANGE_TOLERANCE_M:.2f}m band; penalize only range violations",
            "mode_one_settled_balance_height_objective": f"remove the fixed-height factor within the same +/-{_HEIGHT_RANGE_TOLERANCE_M:.2f}m band; keep tilt, speed, heading, and support factors",
            "other_rewards": "unchanged",
        },
        "flat_mode_unnecessary_pseudo_step_penalty_rate_per_s": pseudo_step[
            "candidate_mode_conditioned_return_rate_per_s"
        ]
        - nominal,
        "scripted_obstacle_mode_posture_prior_removed": all(
            obstacle["candidate_mode_conditioned_weighted_return_rate_per_s"][name] == 0.0
            for name in ("leg_pose_symmetry", "leg_pose_hold")
        ),
        "cases": cases,
        "interpretation_limit": (
            "The mode-conditioned profile is implemented, but this is only static reward arithmetic. "
            "These probes do not establish safe or dynamically feasible postures. A separate mode-1 "
            "terrain ablation is still required before promotion."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write JSON to this path instead of stdout")
    args = parser.parse_args()
    payload = json.dumps(build_report(), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
