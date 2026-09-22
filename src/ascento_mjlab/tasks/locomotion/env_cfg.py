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
    topology, including world-frame target XY and heading error.  It does not
    introduce velocity or height commands, so compatible balance actor weights
    can be transferred without a policy input/output adapter.

    Each reset starts with an immediate 2--3 m target. After the robot
    arrives and settles briefly, another bounded target is sampled. Training
    episodes default to 60 seconds so one episode contains many independent
    go-to-pose attempts instead of a single short movement sequence.
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
            "asset_cfg": ROBOT_CFG,
        },
    )
    cfg.rewards["world_target_progress"] = RewardTermCfg(
        func=ascento_mdp.rewards.world_target_progress_velocity,
        weight=8.0,
        params={"speed_scale": 0.30, "asset_cfg": ROBOT_CFG},
    )
    if not play:
        episode_length_s = float(os.environ.get("ASCENTO_LOCOMOTION_EPISODE_LENGTH_S", "60.0"))
        if episode_length_s <= 0.0:
            raise ValueError("ASCENTO_LOCOMOTION_EPISODE_LENGTH_S must be positive")
        cfg.episode_length_s = episode_length_s
    cfg.task_id = "Ascento-Locomotion-Flat"
    return cfg
