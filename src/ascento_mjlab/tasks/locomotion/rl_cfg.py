"""PPO settings for the fixed-interface flat locomotion curriculum."""

from copy import deepcopy

from ascento_mjlab.tasks.balance_quiet.rl_cfg import AscentoBalanceQuietRlCfg

AscentoLocomotionRlCfg = deepcopy(AscentoBalanceQuietRlCfg)
AscentoLocomotionRlCfg.experiment_name = "ascento_locomotion_flat"
