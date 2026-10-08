"""Shared-policy locomotion with a staged command and recovery curriculum."""

from __future__ import annotations

import math
import os

from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from ascento_mjlab import mdp as ascento_mdp
from ascento_mjlab.tasks.locomotion.env_cfg import ascento_locomotion_env_cfg

CURRICULUM_RAMP_CONTROL_STEPS = 24_000
OBSTACLE_HEIGHT_TOLERANCE_M = 0.10
UNSETTLED_REGULARIZATION_FLOOR = 0.20


def ascento_generalist_locomotion_env_cfg(play: bool = False, num_envs: int = 512):
    """Mix precision, medium-range, long-range, and gate-like training episodes."""
    cfg = ascento_locomotion_env_cfg(play=play, num_envs=num_envs)
    for term_name, function in (
        ("leg_pose_symmetry", ascento_mdp.rewards.mode_conditioned_leg_pose_symmetry_penalty),
        ("leg_pose_hold", ascento_mdp.rewards.mode_conditioned_leg_pose_hold_penalty),
    ):
        term = cfg.rewards[term_name]
        cfg.rewards[term_name] = RewardTermCfg(
            func=function,
            weight=term.weight,
            params=dict(term.params),
        )
    height_term = cfg.rewards["height"]
    cfg.rewards["height"] = RewardTermCfg(
        func=ascento_mdp.rewards.mode_conditioned_height_tracking,
        weight=height_term.weight,
        params={
            **dict(height_term.params),
            "obstacle_height_tolerance_m": OBSTACLE_HEIGHT_TOLERANCE_M,
        },
    )
    settled_term = cfg.rewards["settled_balance"]
    cfg.rewards["settled_balance"] = RewardTermCfg(
        func=settled_term.func,
        weight=settled_term.weight,
        params={
            **dict(settled_term.params),
            "obstacle_height_tolerance_m": OBSTACLE_HEIGHT_TOLERANCE_M,
        },
    )
    # The settled score is intentionally strict: a single large motion, heading,
    # height, or support error can drive it close to zero. Keep the high-frequency
    # action and rocking costs partly active outside that envelope so PPO cannot
    # make those costs vanish by leaving the settled regime.
    for term_name in ("settled_action_second_difference", "settled_body_rocking"):
        term = cfg.rewards[term_name]
        cfg.rewards[term_name] = RewardTermCfg(
            func=term.func,
            weight=term.weight,
            params={
                **dict(term.params),
                "unsettled_penalty_floor": UNSETTLED_REGULARIZATION_FLOOR,
            },
        )
    actor_group = cfg.observations["actor"]
    critic_group = cfg.observations["critic"]
    actor_term_names = tuple(actor_group.terms)
    mode_term = ObservationTermCfg(func=ascento_mdp.observations.obstacle_mode)
    actor_group.terms = {**actor_group.terms, "obstacle_mode": mode_term}
    critic_terms = critic_group.terms
    critic_extras = {
        name: term for name, term in critic_terms.items() if name not in actor_term_names
    }
    critic_group.terms = {
        **{name: critic_terms[name] for name in actor_term_names},
        "obstacle_mode": mode_term,
        **critic_extras,
    }
    precision_anchor_fraction = float(
        os.environ.get("ASCENTO_GENERALIST_PRECISION_ANCHOR_FRACTION", "0.20")
    )
    recovery_retarget_anchor_fraction = float(
        os.environ.get(
            "ASCENTO_GENERALIST_RECOVERY_RETARGET_ANCHOR_FRACTION",
            os.environ.get("ASCENTO_GENERALIST_GATE_LIKE_FRACTION", "0.20"),
        )
    )
    navigation_fraction = 1.0 - precision_anchor_fraction - recovery_retarget_anchor_fraction
    if (
        not math.isfinite(precision_anchor_fraction)
        or not math.isfinite(recovery_retarget_anchor_fraction)
        or precision_anchor_fraction <= 0.0
        or recovery_retarget_anchor_fraction <= 0.0
        or navigation_fraction <= 0.0
    ):
        raise ValueError(
            "generalist precision and recovery anchor fractions must be positive and sum to less than one"
        )
    # Fixed cohorts make the recovery share a constant allocation, not a curriculum schedule.
    gate_like_fraction = recovery_retarget_anchor_fraction
    gate_like_start_fraction = recovery_retarget_anchor_fraction
    legacy_gate_start = os.environ.get("ASCENTO_GENERALIST_GATE_LIKE_START_FRACTION")
    if legacy_gate_start is not None and not math.isclose(
        float(legacy_gate_start), recovery_retarget_anchor_fraction
    ):
        raise ValueError(
            "ASCENTO_GENERALIST_GATE_LIKE_START_FRACTION must equal the fixed recovery anchor fraction"
        )
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
    target_achievement_weight = float(
        os.environ.get("ASCENTO_GENERALIST_TARGET_ACHIEVEMENT_WEIGHT", "5.0")
    )
    if not math.isfinite(target_achievement_weight) or target_achievement_weight < 0.0:
        raise ValueError(
            "ASCENTO_GENERALIST_TARGET_ACHIEVEMENT_WEIGHT must be finite and nonnegative"
        )
    if not play and target_achievement_weight > 0.0:
        cfg.rewards["target_arrival_settled_stop"] = RewardTermCfg(
            func=ascento_mdp.rewards.GeneralistTargetArrivalSettledStopBonus,
            weight=target_achievement_weight,
            params={"hold_s": 0.35, "target_reached_distance_m": 0.035},
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
    fixed_cohort_params = {
        "precision_anchor_fraction": precision_anchor_fraction,
        "recovery_retarget_anchor_fraction": recovery_retarget_anchor_fraction,
    }
    cfg.events["initialize_world_target"].params.update(
        {
            **goal_band_params,
            **({} if play else fixed_cohort_params),
            "record_training_metrics": not play,
        }
    )
    if not play:
        cfg.events["repeated_random_world_targets"].params.update(
            {
                **goal_band_params,
                **fixed_cohort_params,
                "minimum_attempts_per_window": 64,
                "gate_recovery_lcb_threshold": 0.85,
                "target_arrival_lcb_threshold": 0.50,
                "curriculum_start_gate_like_fraction": gate_like_start_fraction,
                "curriculum_ramp_control_steps": CURRICULUM_RAMP_CONTROL_STEPS,
                "gate_like_fraction": gate_like_fraction,
                "track_training_metrics": True,
            }
        )
    metric_params = {
        "curriculum_ramp_control_steps": CURRICULUM_RAMP_CONTROL_STEPS,
        "curriculum_start_gate_like_fraction": gate_like_start_fraction,
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
        "short_attempt_arrival_lcb",
        "long_attempt_arrival_lcb",
        "stage_gate_recovery_lcb",
        "stage_short_attempt_arrival_lcb",
        "stage_long_attempt_arrival_lcb",
        "short_attempts",
        "short_attempt_arrivals",
        "long_attempts",
        "long_attempt_arrivals",
        "window_gate_episodes",
        "window_gate_recovery_successes",
        "window_short_attempts",
        "window_short_arrivals",
        "window_long_attempts",
        "window_long_arrivals",
        "demotion_streak",
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
            "push_received_count",
            "retarget_count",
            "target_arrival_count",
            "second_target_arrival_count",
            "settled_stop_count",
            "post_retarget_settled_stop_count",
            "heading_error_sum_rad",
            "heading_error_samples",
            "post_retarget_heading_error_sum_rad",
            "post_retarget_heading_error_samples",
            "post_arrival_heading_error_sum_rad",
            "post_arrival_heading_error_samples",
            "post_retarget_settled_heading_error_sum_rad",
            "post_retarget_settled_heading_error_samples",
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
    cohort_episode_metrics = (
        "episode_count",
        "arrival_count",
        "recovery_count",
        "push_received_count",
        "retarget_count",
        "target_arrival_count",
        "second_target_arrival_count",
        "settled_stop_count",
        "post_retarget_settled_stop_count",
        "heading_error_sum_rad",
        "heading_error_samples",
        "post_retarget_heading_error_sum_rad",
        "post_retarget_heading_error_samples",
        "post_arrival_heading_error_sum_rad",
        "post_arrival_heading_error_samples",
        "post_retarget_settled_heading_error_sum_rad",
        "post_retarget_settled_heading_error_samples",
        "fall_count",
        "timeout_count",
    )
    for cohort_name in (
        "precision_anchor",
        "recovery_retarget_anchor",
        "generalist_navigation",
    ):
        slice_name = f"cohort_{cohort_name}"
        for metric_name in cohort_episode_metrics:
            cfg.metrics[f"generalist_{slice_name}_{metric_name}"] = MetricsTermCfg(
                func=ascento_mdp.metrics.generalist_training_metric,
                params={
                    "metric_name": metric_name,
                    "slice_name": slice_name,
                    **metric_params,
                },
                reduce="last",
            )
    cfg.metrics["generalist_episode_cohort_id"] = MetricsTermCfg(
        func=ascento_mdp.metrics.generalist_training_metric,
        params={"metric_name": "cohort_id", **metric_params},
        reduce="last",
    )
    for cohort_name in (
        "precision_anchor",
        "recovery_retarget_anchor",
        "generalist_navigation",
        "regular",
        "gate",
    ):
        for band_name in ("short", "medium", "long"):
            attempt_slice = f"attempt_{cohort_name}_{band_name}"
            for metric_name in (
                "target_attempt_count",
                "arrival_completed_count",
                "settled_stop_completed_count",
                "heading_valid_at_settle_count",
                "interrupted_by_fall_count",
                "interrupted_by_timeout_count",
                "interrupted_by_reset_count",
                "target_distance_sum_m",
                "time_to_arrival_sum_s",
                "time_to_arrival_samples",
                "time_to_arrival_mean_s",
                "time_to_arrival_p95_s",
                "time_to_settle_sum_s",
                "time_to_settle_samples",
                "time_to_settle_mean_s",
                "time_to_settle_p95_s",
                "heading_error_at_settle_sum_rad",
                "heading_error_at_settle_samples",
                "heading_error_at_settle_mean_rad",
                "final_target_error_sum_m",
                "final_target_error_samples",
                "final_target_error_mean_m",
                "final_target_error_p95_m",
                "root_speed_at_settle_sum_mps",
                "root_speed_at_settle_mean_mps",
                "angular_speed_at_settle_sum_radps",
                "angular_speed_at_settle_mean_radps",
            ):
                key = f"generalist_attempt_{cohort_name}_{band_name}_{metric_name}"
                cfg.metrics[key] = MetricsTermCfg(
                    func=ascento_mdp.metrics.generalist_training_metric,
                    params={
                        "metric_name": metric_name,
                        "slice_name": attempt_slice,
                        **metric_params,
                    },
                    reduce="last",
                )
    cfg.task_id = "Ascento-Generalist-Locomotion-Flat"
    return cfg
