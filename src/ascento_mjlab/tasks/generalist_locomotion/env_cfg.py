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
            params={
                "hold_s": 0.35,
                "target_reached_distance_m": (
                    ascento_mdp.events.GENERALIST_TARGET_ARRIVAL_DISTANCE_M
                ),
                "max_heading_error_rad": (
                    ascento_mdp.events.GENERALIST_SETTLED_HEADING_VALID_THRESHOLD_RAD
                ),
            },
        )
    progress_reward_mode = (
        os.environ.get("ASCENTO_GENERALIST_PROGRESS_REWARD_MODE", "velocity").strip().lower()
    )
    if progress_reward_mode not in {"velocity", "potential"}:
        raise ValueError(
            "ASCENTO_GENERALIST_PROGRESS_REWARD_MODE must be 'velocity' or 'potential'"
        )
    if not play and progress_reward_mode == "potential":
        from ascento_mjlab.tasks.generalist_locomotion.rl_cfg import (
            AscentoGeneralistLocomotionRlCfg,
        )

        potential_scale = float(os.environ.get("ASCENTO_GENERALIST_POTENTIAL_SCALE", "20.0"))
        if not math.isfinite(potential_scale) or potential_scale <= 0.0:
            raise ValueError("ASCENTO_GENERALIST_POTENTIAL_SCALE must be finite and positive")
        velocity_term = cfg.rewards["world_target_progress"]
        cfg.rewards["world_target_progress"] = RewardTermCfg(
            func=ascento_mdp.rewards.PotentialBasedTargetDistanceReward,
            weight=1.0,
            params={
                "potential_scale": potential_scale,
                "discount_factor": float(AscentoGeneralistLocomotionRlCfg.algorithm.gamma),
                "asset_cfg": velocity_term.params["asset_cfg"],
            },
        )
    target_quality_weight = float(os.environ.get("ASCENTO_GENERALIST_TARGET_QUALITY_WEIGHT", "0.0"))
    if not math.isfinite(target_quality_weight) or target_quality_weight < 0.0:
        raise ValueError("ASCENTO_GENERALIST_TARGET_QUALITY_WEIGHT must be finite and nonnegative")
    if not play and target_quality_weight > 0.0:
        cfg.rewards["target_settled_quality"] = RewardTermCfg(
            func=ascento_mdp.rewards.GeneralistTargetSettledQuality,
            weight=target_quality_weight,
            params={
                "hold_s": 0.35,
                "target_height_m": 0.75,
                "height_tolerance_m": 0.05,
                "max_target_error_m": 0.10,
                "max_heading_error_rad": 0.20,
                "max_planar_speed_mps": 0.08,
                "max_tilt_rad": 0.08,
                "max_angular_xy_radps": 0.25,
                "max_yaw_rate_radps": 0.25,
                "asset_cfg": cfg.rewards["world_target_progress"].params["asset_cfg"],
            },
        )
    quarter_turn_heading_fraction = float(
        os.environ.get("ASCENTO_GENERALIST_QUARTER_TURN_HEADING_FRACTION", "0.20")
    )
    if (
        not math.isfinite(quarter_turn_heading_fraction)
        or not 0.0 <= quarter_turn_heading_fraction <= 1.0
    ):
        raise ValueError("generalist quarter-turn heading fraction must be finite and in [0, 1]")
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
        "quarter_turn_heading_fraction": (quarter_turn_heading_fraction if not play else None),
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
        fixed_goal_mix_stage_value = os.environ.get("ASCENTO_GENERALIST_FIXED_GOAL_MIX_STAGE")
        fixed_goal_mix_stage = None
        if fixed_goal_mix_stage_value is not None:
            try:
                fixed_goal_mix_stage = int(fixed_goal_mix_stage_value)
            except ValueError as exc:
                raise ValueError(
                    "ASCENTO_GENERALIST_FIXED_GOAL_MIX_STAGE must be 0, 1, or 2"
                ) from exc
            if fixed_goal_mix_stage not in (0, 1, 2):
                raise ValueError("ASCENTO_GENERALIST_FIXED_GOAL_MIX_STAGE must be 0, 1, or 2")
        cfg.events["repeated_random_world_targets"].params.update(
            {
                "target_reached_distance_m": (
                    ascento_mdp.events.GENERALIST_TARGET_ARRIVAL_DISTANCE_M
                ),
                "target_heading_tolerance_rad": (
                    ascento_mdp.events.GENERALIST_SETTLED_HEADING_VALID_THRESHOLD_RAD
                ),
                **goal_band_params,
                **fixed_cohort_params,
                **(
                    {}
                    if fixed_goal_mix_stage is None
                    else {"fixed_goal_mix_stage": fixed_goal_mix_stage}
                ),
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


def ascento_generalist_goal_settle_v2_env_cfg(play: bool = False, num_envs: int = 512):
    """Experimental approach-to-settle reward/arrival topology; never mutates v1.

    Keep the 42-D actor, 6-action plant and fixed 20/20/60 cohorts unchanged.
    This task has a distinct ID and must start via actor-only transfer.
    """
    cfg = ascento_generalist_locomotion_env_cfg(play=play, num_envs=num_envs)
    if not play:
        progress = cfg.rewards["world_target_progress"]
        cfg.rewards["world_target_progress"] = RewardTermCfg(
            func=ascento_mdp.rewards.goal_settle_progress_velocity,
            weight=10.0,
            params={
                "speed_scale": 0.30,
                "stop_distance_m": 0.35,
                "minimum_near_gain": 0.80,
                "asset_cfg": progress.params["asset_cfg"],
            },
        )
        speed = cfg.rewards["world_target_speed_penalty"]
        cfg.rewards["world_target_speed_penalty"] = RewardTermCfg(
            func=ascento_mdp.rewards.goal_settle_speed_penalty,
            weight=-2.0,
            params={
                "inner_m": 0.065,
                "outer_m": 0.14,
                "speed_scale": 0.30,
                "asset_cfg": speed.params["asset_cfg"],
            },
        )
        heading = cfg.rewards["world_target_heading"]
        cfg.rewards["world_target_heading"] = RewardTermCfg(
            func=ascento_mdp.rewards.goal_settle_final_heading,
            weight=heading.weight,
            params={
                "inner_m": 0.15,
                "outer_m": 0.35,
                "far_gain": 0.20,
                **dict(heading.params),
            },
        )
        settled = cfg.rewards["settled_balance"]
        cfg.rewards["settled_balance"] = RewardTermCfg(
            func=ascento_mdp.rewards.goal_settle_settled_balance,
            weight=settled.weight,
            params={
                "near_goal_m": 0.10,
                "outer_m": 0.17,
                "far_gain": 0.15,
                **dict(settled.params),
            },
        )
        yaw = cfg.rewards["track_world_target_yaw_rate"]
        cfg.rewards["track_world_target_yaw_rate"] = RewardTermCfg(
            func=yaw.func, weight=0.30, params=dict(yaw.params)
        )
        # Reward and retarget event agree on the same position, heading and
        # dwell thresholds. The official evaluation limits remain unchanged.
        cfg.events["repeated_random_world_targets"].params.update(
            {
                "target_reached_distance_m": 0.10,
                "target_heading_tolerance_rad": 0.20,
                "target_hold_s": 0.50,
            }
        )
        cfg.rewards["target_arrival_settled_stop"] = RewardTermCfg(
            func=ascento_mdp.rewards.GeneralistTargetArrivalSettledStopBonus,
            weight=8.0,
            params={
                "hold_s": 0.50,
                "target_reached_distance_m": 0.10,
                "max_heading_error_rad": 0.20,
            },
        )
    cfg.task_id = "Ascento-Generalist-Locomotion-Goal-Settle-V2-Flat"
    return cfg


def ascento_generalist_goal_settle_v3_env_cfg(play: bool = False, num_envs: int = 512):
    """Opt-in completion-first navigation: do not pay indefinitely for balancing.

    v1/v2 are preserved for checkpoint reproducibility. The 42D observation,
    6-action controller, obstacle modes, goal ranges, and completion gates stay
    the same as v2. Completion, not circle occupancy, resets the attempt clock.
    """
    cfg = ascento_generalist_goal_settle_v2_env_cfg(play=play, num_envs=num_envs)
    if not play:
        # The maximum passive rates below total 1.40/s, while an unfinished
        # target costs at least 1.70/s. This prevents perpetual static
        # balancing from being a positive-return strategy. Falling remains
        # discouraged through termination and the existing motion penalties.
        passive_weights = {
            "alive": 0.15,
            "upright": 0.45,
            "height": 0.20,
            "world_target_heading": 0.25,
            "track_world_target_yaw_rate": 0.10,
            "settled_balance": 0.25,
        }
        for name, weight in passive_weights.items():
            term = cfg.rewards[name]
            cfg.rewards[name] = RewardTermCfg(
                func=term.func, weight=weight, params=dict(term.params)
            )

        # Positive progress still requires actually closing distance; reverse
        # motion is penalized. The approach incentive remains active near goal.
        progress = cfg.rewards["world_target_progress"]
        cfg.rewards["world_target_progress"] = RewardTermCfg(
            func=progress.func, weight=14.0, params=dict(progress.params)
        )
        speed = cfg.rewards["world_target_speed_penalty"]
        cfg.rewards["world_target_speed_penalty"] = RewardTermCfg(
            func=speed.func, weight=-3.0, params=dict(speed.params)
        )

        # Bounded quality credit provides a smooth physical settling signal.
        # It can pay at most 6 units per attempt, never per second forever.
        cfg.rewards["target_settled_quality"] = RewardTermCfg(
            func=ascento_mdp.rewards.GeneralistTargetSettledQuality,
            weight=6.0,
            params={
                "hold_s": 0.50,
                "max_target_error_m": 0.10,
                "max_heading_error_rad": 0.20,
                "max_planar_speed_mps": 0.08,
            },
        )
        completion = cfg.rewards["target_arrival_settled_stop"]
        cfg.rewards["target_arrival_settled_stop"] = RewardTermCfg(
            func=completion.func,
            weight=30.0,
            params=dict(completion.params),
        )
        cfg.rewards["pending_goal_time_cost"] = RewardTermCfg(
            func=ascento_mdp.rewards.GeneralistPendingGoalTimeCost,
            weight=-1.0,
            params={
                "base_rate": 1.70,
                "ramp_rate": 1.30,
                "ramp_seconds": 8.0,
            },
        )
        # Without this, the negative pending-time return could make falling
        # early preferable to finishing a difficult target. Timeout is not
        # a failure; only true termination receives this one-time impulse.
        cfg.rewards["failure_termination_impulse"] = RewardTermCfg(
            func=ascento_mdp.rewards.generalist_failure_termination_impulse,
            weight=-200.0,
        )
    cfg.task_id = "Ascento-Generalist-Locomotion-Goal-Settle-V3-Flat"
    return cfg


def ascento_generalist_goal_settle_v4_env_cfg(play: bool = False, num_envs: int = 512):
    """Completion-first v4 with critic-safe failure shaping.

    Preserve v2 and v3 reward contracts and their checkpoint provenance.
    Keep v3's per-target pending cost, signed progress, capped settling quality,
    and one-shot 0.5 s completion. Avoid its -200 failure impulse, which
    destabilized the value function in matched 50-update training.
    """
    cfg = ascento_generalist_goal_settle_v3_env_cfg(play=play, num_envs=num_envs)
    if not play:
        # The old gate required >=85% push recovery before *navigation*
        # targets could grow, even when short-waypoint success exceeded 95%.
        # Keep the recovery cohort and independent telemetry, but let the
        # goal-distance curriculum follow navigation competence.
        cfg.events["repeated_random_world_targets"].params[
            "navigation_stage_independent_of_recovery"
        ] = True
        failure = cfg.rewards["failure_termination_impulse"]
        cfg.rewards["failure_termination_impulse"] = RewardTermCfg(
            func=failure.func,
            weight=-30.0,
            params=dict(failure.params),
        )
    cfg.task_id = "Ascento-Generalist-Locomotion-Goal-Settle-V4-Flat"
    return cfg


def ascento_generalist_goal_settle_v5_env_cfg(play: bool = False, num_envs: int = 512):
    """V5: widen training tilt acceptance, preserving strict evaluation.

    In V4, a robot often parked within 4 cm of its target at about 7 degrees
    of tilt. Its 4.6-degree training completion threshold never triggered.
    V5 uses a 9.2-degree event/bonus threshold and a broader differentiable
    13.8-degree quality envelope while preserving all other completion gates.
    """
    cfg = ascento_generalist_goal_settle_v4_env_cfg(play=play, num_envs=num_envs)
    if not play:
        completion_tilt_rad = 0.16
        shaping_tilt_rad = 0.24

        # Real curriculum retarget depends on this event, not reward alone.
        cfg.events["repeated_random_world_targets"].params["target_max_tilt_rad"] = (
            completion_tilt_rad
        )
        # Match the existing completion bonus XY/yaw angular gates.
        cfg.events["repeated_random_world_targets"].params["target_split_angular_limits"] = True

        bonus = cfg.rewards["target_arrival_settled_stop"]
        bonus_params = dict(bonus.params)
        bonus_params["max_tilt_rad"] = completion_tilt_rad
        cfg.rewards["target_arrival_settled_stop"] = RewardTermCfg(
            func=bonus.func, weight=bonus.weight, params=bonus_params
        )

        quality = cfg.rewards["target_settled_quality"]
        quality_params = dict(quality.params)
        quality_params.update(
            {
                "max_tilt_rad": shaping_tilt_rad,
                "max_heading_error_rad": 0.35,
                "max_planar_speed_mps": 0.12,
                "max_angular_xy_radps": 0.40,
                "max_yaw_rate_radps": 0.35,
            }
        )
        cfg.rewards["target_settled_quality"] = RewardTermCfg(
            func=quality.func, weight=quality.weight, params=quality_params
        )
    cfg.task_id = "Ascento-Generalist-Locomotion-Goal-Settle-V5-Flat"
    return cfg
