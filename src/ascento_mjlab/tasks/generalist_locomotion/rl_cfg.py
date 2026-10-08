"""PPO settings and controlled drift ablations for generalist locomotion."""

from __future__ import annotations

import os
from copy import deepcopy

from rsl_rl.models.mlp_model import MLPModel
from tensordict import TensorDict

from ascento_mjlab.semantic_normalization import semantic_model_class_name
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


class FrozenActorMLPModel(MLPModel):
    """Keep the transferred actor's empirical observation statistics fixed."""

    def update_normalization(self, obs: TensorDict) -> None:
        """Preserve loaded actor statistics while PPO adapts the policy weights."""
        del obs


def configure_generalist_normalizer_profile(cfg, profile: str):
    """Choose adaptive, frozen, or semantic observation normalization."""
    if profile == "adaptive":
        cfg.actor.class_name = "MLPModel"
        cfg.critic.class_name = "MLPModel"
        return cfg
    if profile == "frozen_transfer":
        if not cfg.actor.obs_normalization:
            raise ValueError("frozen-transfer normalizer profile requires actor normalization")
        cfg.actor.class_name = f"{__name__}:FrozenActorMLPModel"
        cfg.critic.class_name = "MLPModel"
        return cfg
    if profile == "semantic_command":
        class_name = semantic_model_class_name()
        cfg.actor.class_name = class_name
        cfg.critic.class_name = class_name
        return cfg
    raise ValueError(
        f"unknown generalist normalizer profile {profile!r}; "
        "expected 'adaptive', 'frozen_transfer', or 'semantic_command'"
    )


AscentoGeneralistLocomotionRlCfg = deepcopy(AscentoLocomotionRlCfg)
AscentoGeneralistLocomotionRlCfg.experiment_name = "ascento_generalist_locomotion_flat"
AscentoGeneralistLocomotionRlCfg = configure_generalist_optimizer_profile(
    AscentoGeneralistLocomotionRlCfg,
    os.environ.get("ASCENTO_GENERALIST_OPTIMIZER_PROFILE", "default").strip().lower(),
)
AscentoGeneralistLocomotionRlCfg = configure_generalist_normalizer_profile(
    AscentoGeneralistLocomotionRlCfg,
    os.environ.get("ASCENTO_GENERALIST_NORMALIZER_PROFILE", "adaptive").strip().lower(),
)
