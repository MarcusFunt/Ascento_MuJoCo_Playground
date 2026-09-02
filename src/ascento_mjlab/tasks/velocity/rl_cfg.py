"""RSL-RL settings for the flat-ground velocity specialist."""

from copy import deepcopy

from ascento_mjlab.tasks.balance.rl_cfg import AscentoBalanceRlCfg

AscentoVelocityRlCfg = deepcopy(AscentoBalanceRlCfg)
AscentoVelocityRlCfg.experiment_name = "ascento_velocity"
