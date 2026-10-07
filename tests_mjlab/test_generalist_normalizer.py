from copy import deepcopy
from importlib import import_module

import pytest
import torch

from ascento_mjlab.normalizer_diagnostics import (
    compare_target_error_normalizers,
    observation_term_slice,
    target_error_normalizer_summary,
)


def test_observation_term_slice_uses_configured_term_order_and_dimensions():
    names = ["gravity", "actions", "world_target_error", "world_target_heading_error"]
    dimensions = [(3,), (6,), (2,), (1,)]

    assert observation_term_slice(names, dimensions, "world_target_error") == (9, 11)


def test_normalizer_summary_reports_target_error_mean_and_scale_drift():
    reference = {
        "actor_state_dict": {
            "obs_normalizer._mean": torch.tensor([1.0, 2.0, 3.0, 4.0]),
            "obs_normalizer._std": torch.tensor([2.0, 3.0, 4.0, 5.0]),
        }
    }
    candidate = {
        "actor_state_dict": {
            "obs_normalizer._mean": torch.tensor([1.0, 2.5, 3.5, 4.0]),
            "obs_normalizer._std": torch.tensor([2.0, 4.0, 2.0, 5.0]),
        }
    }

    baseline = target_error_normalizer_summary(reference, (1, 3))
    comparison = compare_target_error_normalizers(reference, candidate, (1, 3))

    assert baseline["mean"] == pytest.approx([2.0, 3.0])
    assert baseline["scale"] == pytest.approx([3.0, 4.0])
    assert comparison["mean_delta"] == pytest.approx([0.5, 0.5])
    assert comparison["scale_ratio"] == pytest.approx([4.0 / 3.0, 0.5])


def test_generalist_optimizer_profiles_change_learning_rate_and_kl_together():
    from mjlab.tasks.registry import load_rl_cfg

    import ascento_mjlab.tasks  # noqa: F401
    from ascento_mjlab.tasks.generalist_locomotion.rl_cfg import (
        configure_generalist_optimizer_profile,
    )

    default = configure_generalist_optimizer_profile(
        deepcopy(load_rl_cfg("Ascento-Generalist-Locomotion-Flat")), "default"
    )
    conservative = configure_generalist_optimizer_profile(
        deepcopy(load_rl_cfg("Ascento-Generalist-Locomotion-Flat")), "conservative"
    )

    assert default.algorithm.learning_rate == pytest.approx(1.0e-4)
    assert default.algorithm.desired_kl == pytest.approx(0.01)
    assert conservative.algorithm.learning_rate == pytest.approx(3.0e-5)
    assert conservative.algorithm.desired_kl == pytest.approx(0.003)
    with pytest.raises(ValueError, match="optimizer profile"):
        configure_generalist_optimizer_profile(conservative, "unrecognized")


def test_frozen_transfer_profile_freezes_only_the_actor_model():
    from mjlab.tasks.registry import load_rl_cfg
    from rsl_rl.utils import resolve_callable

    import ascento_mjlab.tasks  # noqa: F401

    rl_cfg = import_module("ascento_mjlab.tasks.generalist_locomotion.rl_cfg")
    configure = getattr(rl_cfg, "configure_generalist_normalizer_profile", None)
    assert callable(configure), "generalist normalizer profile configuration is missing"

    cfg = deepcopy(load_rl_cfg("Ascento-Generalist-Locomotion-Flat"))
    critic_model = cfg.critic.class_name
    frozen = configure(cfg, "frozen_transfer")

    assert frozen.actor.obs_normalization is True
    assert frozen.actor.class_name.endswith(":FrozenActorMLPModel")
    assert resolve_callable(frozen.actor.class_name) is rl_cfg.FrozenActorMLPModel
    assert frozen.critic.class_name == critic_model
    assert frozen.critic.obs_normalization is True
    adaptive = configure(frozen, "adaptive")
    assert adaptive.actor.class_name == "MLPModel"


def test_frozen_actor_model_preserves_loaded_normalizer_statistics():
    from tensordict import TensorDict

    rl_cfg = import_module("ascento_mjlab.tasks.generalist_locomotion.rl_cfg")
    model_cls = getattr(rl_cfg, "FrozenActorMLPModel", None)
    assert model_cls is not None, "frozen actor model is missing"

    observations = TensorDict(
        {
            "actor": torch.full((8, 2), 20.0),
            "critic": torch.full((8, 2), -20.0),
        },
        batch_size=[8],
    )
    model = model_cls(
        observations,
        obs_groups={"actor": ["actor"], "critic": ["critic"]},
        obs_set="actor",
        output_dim=2,
        hidden_dims=(4,),
        obs_normalization=True,
    )
    model.train()
    with torch.no_grad():
        model.obs_normalizer._mean.copy_(torch.tensor([[1.0, -2.0]]))
        model.obs_normalizer._std.copy_(torch.tensor([[3.0, 4.0]]))
        model.obs_normalizer.count.fill_(128)

    before = {key: value.clone() for key, value in model.obs_normalizer.state_dict().items()}
    model.update_normalization(observations)

    for key, expected in before.items():
        torch.testing.assert_close(model.obs_normalizer.state_dict()[key], expected)
