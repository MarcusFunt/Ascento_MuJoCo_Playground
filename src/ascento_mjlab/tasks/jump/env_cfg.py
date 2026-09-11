"""Flat-ground jump specialist configuration.

Terrain remains intentionally disabled until flat-ground takeoff, flight,
landing, distance tracking, and recovery are all quantitatively sound.
"""

import os

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from ascento_mjlab import mdp as ascento_mdp
from ascento_mjlab.tasks.balance.env_cfg import ROBOT_CFG, ascento_balance_env_cfg


def ascento_jump_env_cfg(play: bool = False, num_envs: int = 512):
    cfg = ascento_balance_env_cfg(play=play, num_envs=num_envs)
    cfg.rewards.pop("world_target_proximity", None)
    cfg.rewards.pop("settled_balance", None)
    cfg.events.pop("balance_push", None)
    cfg.events.pop("initialize_world_target", None)
    cfg.observations["actor"].terms.pop("world_target_error", None)
    cfg.observations["critic"].terms.pop("world_target_error", None)
    cfg.commands = {
        "motion": ascento_mdp.commands.AscentoMotionCommandCfg(
            entity_name="robot",
            resampling_time_range=(4.0, 6.0),
            jump_probability=0.5,
            vx_range=(0.0, 0.25),
            yaw_range=(-0.25, 0.25),
            height_range=(0.72, 0.80),
            jump_height_range=(0.15, 0.25),
            jump_distance_range=(0.0, 0.30),
            debug_vis=False,
        )
    }
    cfg.observations["actor"].terms["motion_command"] = ObservationTermCfg(
        func=ascento_mdp.observations.motion_command,
        params={"command_name": "motion"},
    )
    cfg.observations["actor"].terms["jump_state"] = ObservationTermCfg(
        func=ascento_mdp.observations.jump_state,
    )
    cfg.observations["critic"].terms["motion_command"] = cfg.observations["actor"].terms[
        "motion_command"
    ]
    cfg.observations["critic"].terms["jump_state"] = cfg.observations["actor"].terms["jump_state"]
    cfg.events["initialize_jump_state"] = EventTermCfg(
        func=ascento_mdp.jump.initialize_jump_state,
        mode="reset",
    )

    # Synchronization must be the first reward term. Phase-dependent inherited
    # rewards otherwise see the previous contact sample/phase on a transition.
    base_rewards = dict(cfg.rewards)
    base_rewards["height"] = RewardTermCfg(
        func=ascento_mdp.rewards.jump_commanded_height_tracking,
        weight=1.0,
        params={"command_name": "motion", "std": 0.08, "asset_cfg": ROBOT_CFG},
    )
    base_rewards["angular_rate"] = RewardTermCfg(
        func=ascento_mdp.rewards.jump_angular_rate_penalty,
        weight=-0.04,
        params={"asset_cfg": ROBOT_CFG},
    )
    base_rewards.pop("planar_speed", None)
    base_rewards["lateral_speed"] = RewardTermCfg(
        func=ascento_mdp.rewards.lateral_speed_penalty,
        weight=-0.2,
        params={"asset_cfg": ROBOT_CFG},
    )
    cfg.rewards = {
        "jump_state_sync": RewardTermCfg(
            func=ascento_mdp.jump.sync_jump_state_reward,
            weight=1.0,
        ),
        **base_rewards,
        "crouch": RewardTermCfg(
            func=ascento_mdp.rewards.jump_crouch,
            weight=0.8,
            params={"asset_cfg": ROBOT_CFG},
        ),
        "thrust": RewardTermCfg(
            func=ascento_mdp.rewards.jump_thrust,
            weight=1.2,
            params={"command_name": "motion", "asset_cfg": ROBOT_CFG},
        ),
        "track_forward_velocity": RewardTermCfg(
            func=ascento_mdp.rewards.track_motion_forward_velocity,
            weight=1.0,
            params={"command_name": "motion", "std": 0.35, "asset_cfg": ROBOT_CFG},
        ),
        "track_yaw_rate": RewardTermCfg(
            func=ascento_mdp.rewards.track_motion_yaw_rate,
            weight=0.5,
            params={"command_name": "motion", "std": 0.40, "asset_cfg": ROBOT_CFG},
        ),
        "takeoff": RewardTermCfg(func=ascento_mdp.rewards.jump_takeoff, weight=2.0),
        "landing": RewardTermCfg(func=ascento_mdp.rewards.jump_landing, weight=2.0),
        "recovered_landing": RewardTermCfg(
            func=ascento_mdp.rewards.jump_recovered_landing,
            weight=1.0,
        ),
        **(
            {
                "post_landing_stability": RewardTermCfg(
                    func=ascento_mdp.rewards.jump_post_landing_stability,
                    weight=1.5,
                )
            }
            if os.environ.get("ASCENTO_DISABLE_DENSE_SHAPING", "").strip() != "1"
            else {}
        ),
        "landing_softness": RewardTermCfg(
            func=ascento_mdp.rewards.jump_landing_softness,
            weight=1.0,
        ),
        "distance_tracking": RewardTermCfg(
            func=ascento_mdp.rewards.jump_distance_tracking,
            weight=3.0,
        ),
        "height_progress": RewardTermCfg(
            func=ascento_mdp.rewards.airborne_height_progress,
            weight=0.5,
            params={"command_name": "motion"},
        ),
    }
    if os.environ.get("ASCENTO_DISABLE_DENSE_SHAPING", "").strip() == "1":
        cfg.rewards.pop("post_landing_stability", None)
    cfg.episode_length_s = 8.0 if not play else 10000.0
    return cfg
