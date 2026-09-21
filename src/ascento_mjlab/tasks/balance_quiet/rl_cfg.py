"""PPO configuration for the quiet-balance continuation task."""

from copy import deepcopy

from ascento_mjlab.tasks.balance.rl_cfg import AscentoBalanceRlCfg

AscentoBalanceQuietRlCfg = deepcopy(AscentoBalanceRlCfg)
AscentoBalanceQuietRlCfg.experiment_name = "ascento_balance_quiet"
