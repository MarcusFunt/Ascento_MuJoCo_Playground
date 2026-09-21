"""ABI-versioned balance task with equilibrium-only anti-rocking rewards."""

from __future__ import annotations

from mjlab.managers.reward_manager import RewardTermCfg

from ascento_mjlab import mdp as ascento_mdp
from ascento_mjlab.tasks.balance.env_cfg import ROBOT_CFG, ascento_balance_env_cfg


def ascento_balance_quiet_env_cfg(play: bool = False, num_envs: int = 512):
    """Keep the frozen balance ABI intact while adding calibrated quietness costs.

    Calibration against the frozen ``model_79999`` reference and the
    pathological ``model_best_long_horizon`` checkpoint showed roughly three
    orders of magnitude separation in stationary action second difference.
    These terms are therefore versioned as a new task instead of silently
    changing the old checkpoint's reward contract.
    """
    cfg = ascento_balance_env_cfg(play=play, num_envs=num_envs)
    cfg.rewards["settled_action_second_difference"] = RewardTermCfg(
        func=ascento_mdp.rewards.settled_action_second_difference_penalty,
        weight=-0.50,
        params={"asset_cfg": ROBOT_CFG},
    )
    cfg.rewards["settled_body_rocking"] = RewardTermCfg(
        func=ascento_mdp.rewards.settled_body_rocking_penalty,
        weight=-0.10,
        params={"asset_cfg": ROBOT_CFG},
    )
    cfg.task_id = "Ascento-Balance-Quiet-Flat"
    return cfg
