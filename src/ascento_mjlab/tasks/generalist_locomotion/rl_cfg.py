"""PPO settings for the Roadrunner-inspired generalist locomotion curriculum."""

from copy import deepcopy

from ascento_mjlab.tasks.locomotion.rl_cfg import AscentoLocomotionRlCfg

AscentoGeneralistLocomotionRlCfg = deepcopy(AscentoLocomotionRlCfg)
AscentoGeneralistLocomotionRlCfg.experiment_name = "ascento_generalist_locomotion_flat"
