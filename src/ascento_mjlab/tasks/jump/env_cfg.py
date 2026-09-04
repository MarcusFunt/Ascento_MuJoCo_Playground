"""Flat-ground jump specialist configuration placeholder.

Terrain is intentionally disabled here. This task is not expanded until
takeoff, flight, landing, and post-landing recovery are sound on a plane.
"""

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from ascento_mjlab import mdp as ascento_mdp
from ascento_mjlab.tasks.balance.env_cfg import ascento_balance_env_cfg


def ascento_jump_env_cfg(play: bool = False, num_envs: int = 512):
  cfg = ascento_balance_env_cfg(play=play, num_envs=num_envs)
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
  cfg.observations["critic"].terms["jump_state"] = cfg.observations["actor"].terms[
    "jump_state"
  ]
  cfg.events["initialize_jump_state"] = EventTermCfg(
    func=ascento_mdp.jump.initialize_jump_state,
    mode="reset",
  )

  # Reward computation happens before mjlab's step events. Synchronize the
  # contact-derived jump state here so takeoff/landing rewards correspond to
  # the same transition that produced the returned contact observation.
  cfg.rewards["jump_state_sync"] = RewardTermCfg(
    func=ascento_mdp.jump.sync_jump_state_reward,
    weight=1.0,
  )
  cfg.rewards["takeoff"] = RewardTermCfg(func=ascento_mdp.rewards.jump_takeoff, weight=2.0)
  cfg.rewards["landing"] = RewardTermCfg(func=ascento_mdp.rewards.jump_landing, weight=3.0)
  cfg.rewards["height_progress"] = RewardTermCfg(
    func=ascento_mdp.rewards.airborne_height_progress,
    weight=0.5,
    params={"command_name": "motion"},
  )
  cfg.episode_length_s = 8.0 if not play else 10000.0
  return cfg
