"""Shared-policy locomotion with a staged command and recovery curriculum."""

from __future__ import annotations

import math
import os

from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from ascento_mjlab import mdp as ascento_mdp
from ascento_mjlab.tasks.locomotion.env_cfg import ascento_locomotion_env_cfg

CURRICULUM_RAMP_CONTROL_STEPS = 24_000


def ascento_generalist_locomotion_env_cfg(play: bool = False, num_envs: int = 512):
    """Mix precision, medium-range, long-range, and gate-like training episodes."""
    cfg = ascento_locomotion_env_cfg(play=play, num_envs=num_envs)
    gate_like_fraction = float(os.environ.get("ASCENTO_GENERALIST_GATE_LIKE_FRACTION", "0.25"))
    if not 0.10 <= gate_like_fraction <= 1.0:
        raise ValueError("ASCENTO_GENERALIST_GATE_LIKE_FRACTION must be in [0.10, 1.0]")
    reference_bc_weight = float(os.environ.get("ASCENTO_GENERALIST_REFERENCE_BC_WEIGHT", "0.0"))
    if not math.isfinite(reference_bc_weight) or reference_bc_weight < 0.0:
        raise ValueError("ASCENTO_GENERALIST_REFERENCE_BC_WEIGHT must be finite and nonnegative")
    reference_checkpoint = os.environ.get("ASCENTO_GENERALIST_REFERENCE_CHECKPOINT", "").strip()
    if not play and reference_bc_weight > 0.0:
        if not reference_checkpoint:
            raise ValueError(
                "ASCENTO_GENERALIST_REFERENCE_CHECKPOINT is required when the reference BC weight is positive"
            )
        cfg.rewards["reference_actor_action_mse"] = RewardTermCfg(
            func=ascento_mdp.rewards.reference_actor_action_mse,
            weight=-reference_bc_weight,
            params={"checkpoint_path": reference_checkpoint, "action_clip": 1.0},
        )
    goal_band_params = {
        "stratified_goal_mix": not play,
        "short_min_distance_m": 0.15,
        "short_max_distance_m": 0.35,
        "medium_min_distance_m": 0.50,
        "medium_max_distance_m": 1.50,
        "long_min_distance_m": 2.0,
        "long_max_distance_m": 3.0,
        "initial_long_goal_fraction": 0.10,
        "medium_goal_fraction": 0.25,
        "max_long_goal_fraction": 0.40,
    }
    cfg.events["initialize_world_target"].params.update(
        {**goal_band_params, "record_training_metrics": not play}
    )
    if not play:
        cfg.events["repeated_random_world_targets"].params.update(
            {
                **goal_band_params,
                "minimum_episodes_per_stage": 64,
                "gate_recovery_lcb_threshold": 0.85,
                "target_arrival_lcb_threshold": 0.50,
                "curriculum_start_gate_like_fraction": 0.10,
                "curriculum_ramp_control_steps": CURRICULUM_RAMP_CONTROL_STEPS,
                "gate_like_fraction": gate_like_fraction,
                "track_training_metrics": True,
            }
        )
    metric_params = {
        "curriculum_ramp_control_steps": CURRICULUM_RAMP_CONTROL_STEPS,
        "curriculum_start_gate_like_fraction": 0.10,
        "curriculum_final_gate_like_fraction": gate_like_fraction,
    }
    for metric_name in (
        "control_step",
        "progress",
        "scheduled_target_min_distance_m",
        "scheduled_target_max_distance_m",
        "scheduled_gate_like_fraction",
        "gate_like_episode_count",
        "initial_target_distance_m",
        "sampled_short_goal_targets",
        "sampled_medium_goal_targets",
        "sampled_long_goal_targets",
        "goal_mix_stage",
        "short_goal_fraction",
        "medium_goal_fraction",
        "long_goal_fraction",
        "gate_recovery_lcb",
        "short_arrival_lcb",
        "long_arrival_lcb",
        "gate_episode_samples",
        "short_episode_samples",
        "long_episode_samples",
    ):
        cfg.metrics[f"generalist_curriculum_{metric_name}"] = MetricsTermCfg(
            func=ascento_mdp.metrics.generalist_training_metric,
            params={"metric_name": metric_name, **metric_params},
            reduce="last",
        )
    for slice_name in ("gate_like", "short", "medium", "long"):
        for metric_name in (
            "episode_count",
            "arrival_count",
            "recovery_count",
            "heading_error_sum_rad",
            "heading_error_samples",
            "fall_count",
            "timeout_count",
        ):
            key = f"generalist_{slice_name}_{metric_name}"
            cfg.metrics[key] = MetricsTermCfg(
                func=ascento_mdp.metrics.generalist_training_metric,
                params={
                    "metric_name": metric_name,
                    "slice_name": slice_name,
                    **metric_params,
                },
                reduce="last",
            )
    cfg.task_id = "Ascento-Generalist-Locomotion-Flat"
    return cfg
