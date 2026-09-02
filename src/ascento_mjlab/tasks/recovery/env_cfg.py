"""Recovery specialist configuration."""

from mjlab.managers.reward_manager import RewardTermCfg

from ascento_mjlab import mdp as ascento_mdp
from ascento_mjlab.tasks.balance.env_cfg import ascento_balance_env_cfg


def ascento_recovery_env_cfg(play: bool = False, num_envs: int = 512):
  cfg = ascento_balance_env_cfg(play=play, num_envs=num_envs)
  cfg.events["reset_supported_pose"].params["pose_range"].update(
    {"roll": (-0.35, 0.35), "pitch": (-0.45, 0.45)}
  )
  cfg.events["reset_supported_pose"].params["velocity_range"].update(
    {"x": (-0.8, 0.8), "y": (-0.4, 0.4), "roll": (-1.5, 1.5), "pitch": (-2.0, 2.0), "yaw": (-0.5, 0.5)}
  )
  cfg.rewards["recovery_progress"] = RewardTermCfg(
    func=ascento_mdp.recovery.recovery_progress,
    weight=2.0,
  )
  cfg.episode_length_s = 5.0 if not play else 10000.0
  return cfg
