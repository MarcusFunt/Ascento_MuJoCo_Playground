"""World-target locomotion built directly on the balance observation contract."""

from __future__ import annotations

import os

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from ascento_mjlab import mdp as ascento_mdp
from ascento_mjlab.tasks.balance.env_cfg import ROBOT_CFG
from ascento_mjlab.tasks.balance_quiet.env_cfg import ascento_balance_quiet_env_cfg


def ascento_locomotion_env_cfg(play: bool = False, num_envs: int = 512):
    """Train sustained locomotion through repeated long-range random world targets.

    This intentionally preserves the 41-dimensional balance actor observation
    topology, including world-frame target XY and heading error. For locomotion,
    each waypoint's heading is the fixed bearing from the robot to that waypoint
    at assignment time, so heading shaping turns with the travel objective
    instead of preserving the pre-trip yaw. It does not introduce velocity or
    height commands, so compatible balance actor weights can be transferred
    without a policy input/output adapter.

    During training, each reset starts with an immediate 2--3 m target. After
    the robot arrives and settles briefly, another bounded target is sampled.
    Training episodes default to 60 seconds so one episode contains many
    independent go-to-pose attempts. In play mode the target starts at the
    reset pose and remains under evaluator or viewer control.
    """
    cfg = ascento_balance_quiet_env_cfg(play=play, num_envs=num_envs)
    cfg.scene.env_spacing = 10.0
    cfg.events.pop("balance_push", None)
    cfg.events["initialize_world_target"] = EventTermCfg(
        func=ascento_mdp.events.initialize_random_world_target,
        mode="reset",
        params={
            "asset_name": "robot",
            "min_distance_m": 2.0,
            "max_distance_m": 3.0,
            "arena_half_extent_m": 4.0,
        },
    )
    if not play:
        cfg.events["repeated_random_world_targets"] = EventTermCfg(
            func=ascento_mdp.events.RepeatedRandomWorldTargetSequence,
            mode="interval",
            interval_range_s=(0.01, 0.01),
            params={
                "target_reached_distance_m": 0.04,
                "target_hold_s": 0.35,
                "min_target_distance_m": 2.0,
                "max_target_distance_m": 3.0,
                "arena_half_extent_m": 4.0,
                "gate_like_fraction": 0.25,
                "gate_push_time_s": 4.0,
                "gate_min_delta_v": 0.05,
                "gate_max_delta_v": 0.15,
                "gate_retarget_time_s": 9.0,
                "gate_min_target_distance_m": 0.10,
                "gate_max_target_distance_m": 0.20,
                "asset_cfg": ROBOT_CFG,
            },
        )
    cfg.rewards["world_target_progress"] = RewardTermCfg(
        func=ascento_mdp.rewards.world_target_progress_velocity,
        weight=8.0,
        params={"speed_scale": 0.30, "stop_distance_m": 0.35, "asset_cfg": ROBOT_CFG},
    )
    cfg.rewards["world_target_speed_penalty"] = RewardTermCfg(
        func=ascento_mdp.rewards.world_target_speed_penalty,
        weight=-2.0,
        params={"std": 0.35, "speed_scale": 0.30, "asset_cfg": ROBOT_CFG},
    )
    if not play:
        episode_length_s = float(os.environ.get("ASCENTO_LOCOMOTION_EPISODE_LENGTH_S", "60.0"))
        if episode_length_s <= 0.0:
            raise ValueError("ASCENTO_LOCOMOTION_EPISODE_LENGTH_S must be positive")
        cfg.episode_length_s = episode_length_s
    cfg.task_id = "Ascento-Locomotion-Flat"
    return cfg


def ascento_locomotion_gate_hold_env_cfg(play: bool = False, num_envs: int = 512):
    """Use a stationary reset target for the existing gate-like episode subset.

    This opt-in task differs from ``Ascento-Locomotion-Flat`` only in the
    initial target of its gate-like episodes. The 25% mixture, 4 s push,
    9 s retarget, regular 2--3 m target sequence, rewards, observations,
    and PPO configuration remain identical.
    """
    cfg = ascento_locomotion_env_cfg(play=play, num_envs=num_envs)
    if not play:
        target_params = dict(cfg.events["initialize_world_target"].params)
        sequence_params = cfg.events["repeated_random_world_targets"].params
        target_params["gate_like_fraction"] = sequence_params["gate_like_fraction"]
        cfg.events["initialize_world_target"] = EventTermCfg(
            func=ascento_mdp.events.initialize_gate_hold_locomotion_target,
            mode="reset",
            params=target_params,
        )
        sequence_params["gate_like_initial_hold"] = True
    cfg.task_id = "Ascento-Locomotion-Gate-Hold-Flat"
    return cfg


def ascento_locomotion_gate_hold_heading_mix_env_cfg(
    *,
    quarter_turn_heading_fraction: float,
    task_id: str,
    play: bool = False,
    num_envs: int = 512,
):
    """Gate-hold locomotion with a controlled mix of independent goal headings.

    Regular targets retain their sampled 2--3 m positions and travel directions.
    For the configured fraction, only the requested final heading is shifted by
    a signed quarter turn. The original gate-hold task and reward structure stay
    untouched.
    """
    if not 0.0 <= quarter_turn_heading_fraction <= 1.0:
        raise ValueError("quarter_turn_heading_fraction must be in [0, 1]")
    cfg = ascento_locomotion_env_cfg(play=play, num_envs=num_envs)
    if not play:
        target_params = dict(cfg.events["initialize_world_target"].params)
        sequence_params = cfg.events["repeated_random_world_targets"].params
        target_params["gate_like_fraction"] = sequence_params["gate_like_fraction"]
        target_params["quarter_turn_heading_fraction"] = quarter_turn_heading_fraction
        cfg.events["initialize_world_target"] = EventTermCfg(
            func=ascento_mdp.events.initialize_gate_hold_locomotion_target,
            mode="reset",
            params=target_params,
        )
        sequence_params["gate_like_initial_hold"] = True
        sequence_params["quarter_turn_heading_fraction"] = quarter_turn_heading_fraction
    cfg.task_id = task_id
    return cfg
