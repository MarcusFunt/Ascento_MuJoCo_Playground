"""RSL-RL recovery settings derived from the balance specialist."""

from copy import deepcopy

from ascento_mjlab.tasks.balance.rl_cfg import AscentoBalanceRlCfg

AscentoRecoveryRlCfg = deepcopy(AscentoBalanceRlCfg)
AscentoRecoveryRlCfg.experiment_name = "ascento_recovery"
