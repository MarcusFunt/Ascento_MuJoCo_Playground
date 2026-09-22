"""PPO settings for the balance recovery curriculum."""

from copy import deepcopy

from ascento_mjlab.tasks.balance_quiet.rl_cfg import AscentoBalanceQuietRlCfg

AscentoBalanceRecoveryRlCfg = deepcopy(AscentoBalanceQuietRlCfg)
AscentoBalanceRecoveryRlCfg.experiment_name = "ascento_balance_recovery"
