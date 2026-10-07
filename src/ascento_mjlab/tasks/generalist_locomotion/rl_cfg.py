"""PPO settings and controlled drift ablations for generalist locomotion."""

from __future__ import annotations

import os
from copy import deepcopy

from ascento_mjlab.tasks.locomotion.rl_cfg import AscentoLocomotionRlCfg


def configure_generalist_optimizer_profile(cfg, profile: str):
    """Apply one reproducible optimizer profile to a copied PPO config."""
    if profile == "default":
        cfg.algorithm.learning_rate = 1.0e-4
        cfg.algorithm.desired_kl = 0.01
    elif profile == "conservative":
        # RSL-RL shares one learning rate across the actor and critic. A lower
        # common rate is the closest supported actor drift control.
        cfg.algorithm.learning_rate = 3.0e-5
        cfg.algorithm.desired_kl = 0.003
    else:
        raise ValueError(
            f"unknown generalist optimizer profile {profile!r}; "
            "expected 'default' or 'conservative'"
        )
    return cfg


AscentoGeneralistLocomotionRlCfg = deepcopy(AscentoLocomotionRlCfg)
AscentoGeneralistLocomotionRlCfg.experiment_name = "ascento_generalist_locomotion_flat"
AscentoGeneralistLocomotionRlCfg = configure_generalist_optimizer_profile(
    AscentoGeneralistLocomotionRlCfg,
    os.environ.get("ASCENTO_GENERALIST_OPTIMIZER_PROFILE", "default").strip().lower(),
)
