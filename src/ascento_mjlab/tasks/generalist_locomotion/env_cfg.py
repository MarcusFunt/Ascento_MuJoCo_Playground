"""Shared-policy locomotion with a staged command and recovery curriculum."""

from __future__ import annotations

from ascento_mjlab.tasks.locomotion.env_cfg import ascento_locomotion_env_cfg

CURRICULUM_RAMP_CONTROL_STEPS = 24_000


def ascento_generalist_locomotion_env_cfg(play: bool = False, num_envs: int = 512):
    """Progress from nearby go-to-pose tasks to sustained travel and recovery."""
    cfg = ascento_locomotion_env_cfg(play=play, num_envs=num_envs)
    curriculum = {
        "curriculum_start_min_target_distance_m": 0.15,
        "curriculum_start_max_target_distance_m": 0.35,
        "curriculum_ramp_control_steps": CURRICULUM_RAMP_CONTROL_STEPS,
    }
    cfg.events["initialize_world_target"].params.update(curriculum)
    if not play:
        cfg.events["repeated_random_world_targets"].params.update(
            {
                **curriculum,
                "curriculum_start_gate_like_fraction": 0.10,
                "gate_like_fraction": 0.25,
            }
        )
    cfg.task_id = "Ascento-Generalist-Locomotion-Flat"
    return cfg
