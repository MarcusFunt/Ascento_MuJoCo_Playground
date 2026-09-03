"""RSL-RL flat-ground jump settings derived from the balance specialist."""

from copy import deepcopy

from ascento_mjlab.tasks.balance.rl_cfg import AscentoBalanceRlCfg

AscentoJumpRlCfg = deepcopy(AscentoBalanceRlCfg)
AscentoJumpRlCfg.experiment_name = "ascento_jump"
