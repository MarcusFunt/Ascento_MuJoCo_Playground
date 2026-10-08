"""Motion-quality metrics for selection and diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from . import events as ascento_events
from .events import world_target_yaw, wrapped_angle_difference, yaw_from_quaternion_wxyz

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def controller_requested_effort(asset: Entity) -> torch.Tensor:
    """Return pre-motor controller requests in canonical robot joint order."""
    actuators = asset.actuators
    if len(actuators) != 2:
        raise RuntimeError("Ascento structured control requires leg and wheel actuator groups")
    leg = actuators[0].controller_requested_effort
    wheel = actuators[1].controller_requested_effort
    return torch.stack(
        (leg[:, 0], leg[:, 1], wheel[:, 0], leg[:, 2], leg[:, 3], wheel[:, 1]), dim=1
    )


def tilt_radians(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    gravity_xy = torch.linalg.vector_norm(asset.data.projected_gravity_b[:, :2], dim=1)
    return torch.atan2(gravity_xy, -asset.data.projected_gravity_b[:, 2].clamp(max=-1.0e-6))


def commanded_effort(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Mean absolute torque requested by the structured controllers, in Nm."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.mean(
        torch.abs(controller_requested_effort(asset)[:, asset_cfg.actuator_ids]), dim=1
    )


def actuator_output_effort(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Mean absolute actuator-space output after the motor model, in Nm."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.mean(torch.abs(asset.data.actuator_force[:, asset_cfg.actuator_ids]), dim=1)


def joint_applied_effort(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Mean absolute generalized actuator force applied at the robot joints."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.mean(torch.abs(asset.data.qfrc_actuator[:, asset_cfg.joint_ids]), dim=1)


def applied_effort(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Backward-compatible alias for :func:`joint_applied_effort`."""
    return joint_applied_effort(env, asset_cfg)


def root_speed(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.linalg.vector_norm(asset.data.root_link_lin_vel_w, dim=1)


def world_target_heading_error_radians(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Absolute reset-relative world-heading error in radians."""
    asset: Entity = env.scene[asset_cfg.name]
    current_yaw = yaw_from_quaternion_wxyz(asset.data.root_link_quat_w)
    return torch.abs(wrapped_angle_difference(world_target_yaw(env), current_yaw))


def generalist_training_metric(
    env: ManagerBasedRlEnv,
    metric_name: str,
    slice_name: str = "all",
    *,
    curriculum_ramp_control_steps: int = 24_000,
    curriculum_start_gate_like_fraction: float = 0.10,
    curriculum_final_gate_like_fraction: float = 0.25,
) -> torch.Tensor:
    """Expose curriculum progress, episode reliability, and target-attempt outcomes."""
    state = ascento_events.generalist_episode_metrics_state(env)
    distances = state["initial_target_distance_m"]
    if metric_name == "cohort_id":
        return state["cohort_id"].float()
    fixed_cohorts = getattr(env, "ascento_generalist_cohort_ids", None)
    cohort_indices = {
        "precision_anchor": ascento_events.GENERALIST_COHORT_PRECISION_ANCHOR,
        "recovery_retarget_anchor": ascento_events.GENERALIST_COHORT_RECOVERY_RETARGET_ANCHOR,
        "generalist_navigation": ascento_events.GENERALIST_COHORT_GENERALIST_NAVIGATION,
        "regular": ascento_events.GENERALIST_COHORT_GENERALIST_NAVIGATION,
        "gate": ascento_events.GENERALIST_COHORT_RECOVERY_RETARGET_ANCHOR,
    }
    attempt_cohort_band: tuple[int, int] | None = None
    if slice_name == "all":
        selected = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    elif slice_name == "gate_like":
        selected = (
            state["cohort_id"] == ascento_events.GENERALIST_COHORT_RECOVERY_RETARGET_ANCHOR
            if fixed_cohorts is not None
            else state["gate_like"]
        )
    elif slice_name.startswith("cohort_"):
        cohort_name = slice_name.removeprefix("cohort_")
        if cohort_name not in cohort_indices:
            raise ValueError(f"unknown generalist cohort slice: {slice_name}")
        selected = state["cohort_id"] == cohort_indices[cohort_name]
    elif slice_name.startswith("attempt_"):
        parts = slice_name.split("_")
        if len(parts) < 3:
            raise ValueError(f"unknown generalist attempt slice: {slice_name}")
        band_indices = {"short": 0, "medium": 1, "long": 2}
        band_name = parts[-1]
        cohort_name = "_".join(parts[1:-1])
        if band_name not in band_indices or cohort_name not in cohort_indices:
            raise ValueError(f"unknown generalist attempt slice: {slice_name}")
        attempt_cohort_band = (
            cohort_indices[cohort_name],
            band_indices[band_name],
        )
        selected = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    elif slice_name == "short":
        selected = (
            state["cohort_id"] == ascento_events.GENERALIST_COHORT_GENERALIST_NAVIGATION
            if fixed_cohorts is not None
            else ~state["gate_like"]
        ) & (distances <= ascento_events.GENERALIST_SHORT_TARGET_MAX_DISTANCE_M)
    elif slice_name == "medium":
        selected = (
            state["cohort_id"] == ascento_events.GENERALIST_COHORT_GENERALIST_NAVIGATION
            if fixed_cohorts is not None
            else ~state["gate_like"]
        ) & (
            (distances >= ascento_events.GENERALIST_MEDIUM_TARGET_MIN_DISTANCE_M)
            & (distances <= ascento_events.GENERALIST_MEDIUM_TARGET_MAX_DISTANCE_M)
        )
    elif slice_name == "long":
        selected = (
            state["cohort_id"] == ascento_events.GENERALIST_COHORT_GENERALIST_NAVIGATION
            if fixed_cohorts is not None
            else ~state["gate_like"]
        ) & (distances >= ascento_events.GENERALIST_LONG_TARGET_MIN_DISTANCE_M)
    else:
        raise ValueError(f"unknown generalist training slice: {slice_name}")

    selected_float = selected.to(dtype=torch.float32)
    arrived = state["arrived"].to(dtype=torch.float32)
    if attempt_cohort_band is not None:
        cohort, band = attempt_cohort_band
        attempt_metric_fields = {
            "target_attempt_count": "target_attempt_count_by_cohort_band",
            "arrival_completed_count": "arrival_completed_count_by_cohort_band",
            "settled_stop_completed_count": "settled_stop_completed_count_by_cohort_band",
            "heading_valid_at_settle_count": "heading_valid_at_settle_count_by_cohort_band",
            "interrupted_by_fall_count": "interrupted_by_fall_count_by_cohort_band",
            "interrupted_by_timeout_count": "interrupted_by_timeout_count_by_cohort_band",
            "interrupted_by_reset_count": "interrupted_by_reset_count_by_cohort_band",
            "target_distance_sum_m": "target_distance_sum_m_by_cohort_band",
            "time_to_arrival_sum_s": "time_to_arrival_sum_s_by_cohort_band",
            "time_to_arrival_samples": "time_to_arrival_samples_by_cohort_band",
            "time_to_settle_sum_s": "time_to_settle_sum_s_by_cohort_band",
            "time_to_settle_samples": "time_to_settle_samples_by_cohort_band",
            "heading_error_at_settle_sum_rad": "heading_error_at_settle_sum_rad_by_cohort_band",
            "heading_error_at_settle_samples": "heading_error_at_settle_samples_by_cohort_band",
            "final_target_error_sum_m": "target_error_at_settle_sum_m_by_cohort_band",
            "final_target_error_samples": "target_error_at_settle_samples_by_cohort_band",
            "root_speed_at_settle_sum_mps": "root_speed_at_settle_sum_mps_by_cohort_band",
            "angular_speed_at_settle_sum_radps": "angular_speed_at_settle_sum_radps_by_cohort_band",
        }
        if metric_name in attempt_metric_fields:
            return state[attempt_metric_fields[metric_name]][:, cohort, band]

        def average(sum_name: str, sample_name: str) -> torch.Tensor:
            sums = state[sum_name][:, cohort, band]
            samples = state[sample_name][:, cohort, band]
            return sums / samples.clamp_min(1.0)

        if metric_name == "time_to_arrival_mean_s":
            return average(
                "time_to_arrival_sum_s_by_cohort_band",
                "time_to_arrival_samples_by_cohort_band",
            )
        if metric_name == "time_to_settle_mean_s":
            return average(
                "time_to_settle_sum_s_by_cohort_band",
                "time_to_settle_samples_by_cohort_band",
            )
        if metric_name == "heading_error_at_settle_mean_rad":
            return average(
                "heading_error_at_settle_sum_rad_by_cohort_band",
                "heading_error_at_settle_samples_by_cohort_band",
            )
        if metric_name == "final_target_error_mean_m":
            return average(
                "target_error_at_settle_sum_m_by_cohort_band",
                "target_error_at_settle_samples_by_cohort_band",
            )
        if metric_name == "root_speed_at_settle_mean_mps":
            return average(
                "root_speed_at_settle_sum_mps_by_cohort_band",
                "settled_stop_completed_count_by_cohort_band",
            )
        if metric_name == "angular_speed_at_settle_mean_radps":
            return average(
                "angular_speed_at_settle_sum_radps_by_cohort_band",
                "settled_stop_completed_count_by_cohort_band",
            )
        history_fields = {
            "time_to_arrival_p95_s": "time_to_arrival_s",
            "time_to_settle_p95_s": "time_to_settle_s",
            "final_target_error_p95_m": "target_error_at_settle_m",
        }
        if metric_name in history_fields:
            history = state["target_attempt_history"]
            values = history[history_fields[metric_name]]
            valid = (
                (history["cohort_id"] == cohort)
                & (history["target_band"] == band)
                & (values >= 0.0)
            )
            counts = valid.sum(dim=1)
            ordered = (
                torch.where(valid, values, torch.full_like(values, float("inf"))).sort(dim=1).values
            )
            rank = torch.ceil(counts.float() * 0.95).to(torch.long).sub(1).clamp_min(0)
            p95 = ordered.gather(1, rank.unsqueeze(1)).squeeze(1)
            return torch.where(counts > 0, p95, torch.zeros_like(p95))
        raise ValueError(f"unknown generalist target-attempt metric: {metric_name}")
    if metric_name == "episode_count":
        return selected_float
    if metric_name == "arrival_count":
        return selected_float * arrived
    if metric_name == "recovery_count":
        return selected_float * state["recovery_completed"].float()
    if metric_name == "push_received_count":
        return selected_float * state["push_received"].float()
    if metric_name == "retarget_count":
        return selected_float * state["retargeted"].float()
    if metric_name == "second_target_arrival_count":
        return selected_float * state["second_target_arrived"].float()
    if metric_name == "target_arrival_count":
        return selected_float * state["target_arrival_count"]
    if metric_name == "settled_stop_count":
        return selected_float * state["settled_stop_count"]
    if metric_name == "post_retarget_settled_stop_count":
        return selected_float * state["post_retarget_settled_stop"].float()
    if metric_name == "heading_error_sum_rad":
        return selected_float * state["arrival_heading_error_sum_rad"]
    if metric_name == "heading_error_samples":
        return selected_float * state["arrival_heading_error_samples"]
    if metric_name == "post_retarget_heading_error_sum_rad":
        return selected_float * state["post_retarget_heading_error_sum_rad"]
    if metric_name == "post_retarget_heading_error_samples":
        return selected_float * state["post_retarget_heading_error_samples"]
    if metric_name == "post_arrival_heading_error_sum_rad":
        return selected_float * state["settled_heading_error_sum_rad"]
    if metric_name == "post_arrival_heading_error_samples":
        return selected_float * state["settled_heading_error_samples"]
    if metric_name == "post_retarget_settled_heading_error_sum_rad":
        return selected_float * state["post_retarget_settled_heading_error_sum_rad"]
    if metric_name == "post_retarget_settled_heading_error_samples":
        return selected_float * state["post_retarget_settled_heading_error_samples"]
    if metric_name == "initial_target_distance_sum_m":
        return selected_float * distances
    if metric_name == "initial_target_distance_samples":
        return selected_float
    if metric_name == "fall_count":
        return selected_float * env.termination_manager.terminated.float()
    if metric_name == "timeout_count":
        return selected_float * env.termination_manager.time_outs.float()
    band_indices = {
        "sampled_short_goal_targets": 0,
        "sampled_medium_goal_targets": 1,
        "sampled_long_goal_targets": 2,
    }
    if metric_name in band_indices:
        index = band_indices[metric_name]
        return state["sampled_target_band_counts"][:, index]

    values = ascento_events.generalist_curriculum_values(
        int(getattr(env, "common_step_counter", 0)),
        ramp_control_steps=curriculum_ramp_control_steps,
        start_gate_like_fraction=curriculum_start_gate_like_fraction,
        final_gate_like_fraction=curriculum_final_gate_like_fraction,
    )
    if metric_name == "control_step":
        return torch.full_like(distances, float(getattr(env, "common_step_counter", 0)))
    if metric_name == "progress":
        return torch.full_like(distances, values["progress"])
    if metric_name == "scheduled_target_min_distance_m":
        return torch.full_like(distances, values["target_min_distance_m"])
    if metric_name == "scheduled_target_max_distance_m":
        return torch.full_like(distances, values["target_max_distance_m"])
    if metric_name == "scheduled_gate_like_fraction":
        return torch.full_like(distances, values["gate_like_fraction"])
    if metric_name == "gate_like_episode_count":
        return state["gate_like"].float()
    if metric_name == "initial_target_distance_m":
        return distances

    mix = ascento_events.generalist_goal_mix_state(env)
    counters = (
        "gate_recovery_successes",
        "gate_episodes",
        "short_arrival_successes",
        "short_episodes",
        "long_arrival_successes",
        "long_episodes",
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
    )
    summary = {key: int(mix[key]) for key in counters}
    summary["goal_mix_stage"] = int(mix["stage"])
    summary["short_goal_fraction"] = (
        1.0 - float(mix["medium_goal_fraction"]) - float(mix["long_goal_fraction"])
    )
    summary["medium_goal_fraction"] = float(mix["medium_goal_fraction"])
    summary["long_goal_fraction"] = float(mix["long_goal_fraction"])
    summary["gate_recovery_lcb"] = ascento_events.wilson_lower_bound(
        summary["gate_recovery_successes"], summary["gate_episodes"]
    )
    summary["short_arrival_lcb"] = ascento_events.wilson_lower_bound(
        summary["short_arrival_successes"], summary["short_episodes"]
    )
    summary["long_arrival_lcb"] = ascento_events.wilson_lower_bound(
        summary["long_arrival_successes"], summary["long_episodes"]
    )
    summary["short_attempt_arrival_lcb"] = ascento_events.wilson_lower_bound(
        summary["short_attempt_arrivals"], summary["short_attempts"]
    )
    summary["long_attempt_arrival_lcb"] = ascento_events.wilson_lower_bound(
        summary["long_attempt_arrivals"], summary["long_attempts"]
    )
    summary["stage_gate_recovery_lcb"] = ascento_events.wilson_lower_bound(
        summary["window_gate_recovery_successes"], summary["window_gate_episodes"]
    )
    summary["stage_short_attempt_arrival_lcb"] = ascento_events.wilson_lower_bound(
        summary["window_short_arrivals"], summary["window_short_attempts"]
    )
    summary["stage_long_attempt_arrival_lcb"] = ascento_events.wilson_lower_bound(
        summary["window_long_arrivals"], summary["window_long_attempts"]
    )
    summary["gate_episode_samples"] = summary["gate_episodes"]
    summary["short_episode_samples"] = summary["short_episodes"]
    summary["long_episode_samples"] = summary["long_episodes"]
    if metric_name in summary:
        return torch.full_like(distances, float(summary[metric_name]))
    raise ValueError(f"unknown generalist training metric: {metric_name}")


def yaw_rate_radians_per_s(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Absolute body-frame yaw rate in rad/s."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.abs(asset.data.root_link_ang_vel_b[:, 2])
