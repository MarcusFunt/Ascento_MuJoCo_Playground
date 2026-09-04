"""Flat-ground velocity/yaw/height task built from the balance plant."""

from __future__ import annotations

from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from ascento_mjlab import mdp as ascento_mdp
from ascento_mjlab.tasks.balance.env_cfg import ascento_balance_env_cfg


def ascento_velocity_env_cfg(play: bool = False, num_envs: int = 512):
  cfg = ascento_balance_env_cfg(play=play, num_envs=num_envs)
  cfg.commands = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 6.0),
      rel_standing_envs=0.2,
      rel_heading_envs=0.0,
      heading_command=False,
      debug_vis=not play,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.5, 0.5), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.6, 0.6)
      ),
    ),
    "height": ascento_mdp.commands.AscentoHeightCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 6.0),
      height_range=(0.68, 0.82),
      debug_vis=False,
    ),
  }
  command_obs = ascento_mdp.observations.motion_command
  twist_term = ObservationTermCfg(func=command_obs, params={"command_name": "twist"})
  height_term = ObservationTermCfg(func=command_obs, params={"command_name": "height"})
  cfg.observations["actor"].terms["twist_command"] = twist_term
  cfg.observations["actor"].terms["height_command"] = height_term
  cfg.observations["critic"].terms["twist_command"] = twist_term
  cfg.observations["critic"].terms["height_command"] = height_term

  # The balance objective rewards a fixed 0.75 m height and penalizes planar
  # speed. Both conflict directly with this stage, where translation and body
  # height are commanded. Keep the stabilizing terms but remove those two
  # contradictory objectives before adding command tracking.
  cfg.rewards.pop("height", None)
  cfg.rewards.pop("planar_speed", None)
  cfg.rewards["track_velocity"] = RewardTermCfg(
    func=ascento_mdp.rewards.track_velocity,
    weight=2.0,
    params={"command_name": "twist", "std": 0.5},
  )
  cfg.rewards["track_height"] = RewardTermCfg(
    func=ascento_mdp.rewards.commanded_height_tracking,
    weight=1.0,
    params={"command_name": "height", "std": 0.05},
  )
  return cfg
