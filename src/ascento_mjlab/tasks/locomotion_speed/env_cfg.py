"""Runtime-selectable speed-conditioned world-target locomotion task."""

from __future__ import annotations

import math
import os

from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from ascento_mjlab import mdp as ascento_mdp
from ascento_mjlab.tasks.locomotion.env_cfg import ascento_locomotion_env_cfg


def ascento_locomotion_speed_env_cfg(play: bool = False, num_envs: int = 512):
    """Build a distinct speed-conditioned task; the frozen locomotion task is unchanged."""
    try:
        max_speed_mps = float(os.environ.get("ASCENTO_LOCOMOTION_MAX_SPEED_COMMAND_MPS", "0.5"))
    except ValueError as error:
        raise ValueError("ASCENTO_LOCOMOTION_MAX_SPEED_COMMAND_MPS must be numeric") from error
    if not math.isfinite(max_speed_mps) or max_speed_mps <= 0.0:
        raise ValueError("maximum speed command must be finite and positive")

    cfg = ascento_locomotion_env_cfg(play=play, num_envs=num_envs)
    cfg.commands = {
        "speed": ascento_mdp.commands.AscentoTargetSpeedCommandCfg(
            entity_name="robot",
            max_speed_mps=max_speed_mps,
            max_speed_slew_rate_mps_per_s=0.5,
            standing_probability=0.15,
            resampling_time_range=(3.0, 6.0),
            debug_vis=False,
        )
    }

    command_observation = ObservationTermCfg(
        func=ascento_mdp.observations.speed_command_fraction,
        params={"command_name": "speed", "max_speed_mps": max_speed_mps},
    )
    for group in ("actor", "critic"):
        cfg.observations[group].terms["speed_command"] = command_observation

    # Track requested speed only while a world target is active. The inherited
    # bounded progress reward still teaches heading toward that target.
    cfg.rewards["commanded_travel_speed"] = RewardTermCfg(
        func=ascento_mdp.rewards.track_world_target_speed,
        weight=3.0,
        params={
            "command_name": "speed",
            "std": max(0.10, min(0.25, max_speed_mps * 0.25)),
            "stop_distance_m": 0.35,
        },
    )
    cfg.task_id = "Ascento-Locomotion-Speed-Flat"
    return cfg


def configure_speed_command_cap(cfg, max_speed_mps: float | None = None):
    """Apply a managed run's cap to an already-registered config copy.

    Task registries cache configs when the task package is imported. Evaluation
    discovers the managed cap later, so changing the environment alone cannot
    rebuild that cached object. This updates every cap-dependent term on the
    per-use config returned by `load_env_cfg`.
    """
    if getattr(cfg, "task_id", None) != "Ascento-Locomotion-Speed-Flat":
        return cfg

    if max_speed_mps is None:
        raw_cap = os.environ.get("ASCENTO_LOCOMOTION_MAX_SPEED_COMMAND_MPS")
        if raw_cap is None:
            return cfg
    else:
        raw_cap = max_speed_mps

    try:
        cap = float(raw_cap)
    except (TypeError, ValueError) as error:
        raise ValueError("maximum speed command must be numeric") from error
    if not math.isfinite(cap) or cap <= 0.0:
        raise ValueError("maximum speed command must be finite and positive")

    cfg.commands["speed"].max_speed_mps = cap
    for group_name in ("actor", "critic"):
        cfg.observations[group_name].terms["speed_command"].params["max_speed_mps"] = cap
    cfg.rewards["commanded_travel_speed"].params["std"] = max(
        0.10, min(0.25, cap * 0.25)
    )
    return cfg
