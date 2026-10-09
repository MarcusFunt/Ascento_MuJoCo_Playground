"""PPO settings for runtime-selectable speed-conditioned locomotion."""

from copy import deepcopy

from ascento_mjlab.tasks.locomotion.rl_cfg import AscentoLocomotionRlCfg

AscentoLocomotionSpeedRlCfg = deepcopy(AscentoLocomotionRlCfg)
AscentoLocomotionSpeedRlCfg.experiment_name = "ascento_locomotion_speed_flat"
