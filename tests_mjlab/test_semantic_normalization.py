from __future__ import annotations

import pytest
import torch
from rsl_rl.models.mlp_model import MLPModel
from tensordict import TensorDict

from ascento_mjlab.semantic_normalization import (
    SemanticNormalizationMLP,
    configure_rl_cfg_for_normalizer_contract,
    migrate_checkpoint_payload,
    migrate_semantic_state_dict,
    require_model_normalizer_contract,
    semantic_normalizer_contract,
)


def _model(model_cls, *, obs_set: str, obs_dim: int):
    obs = TensorDict({obs_set: torch.zeros((2, obs_dim))}, batch_size=[2])
    return model_cls(
        obs=obs,
        obs_groups={obs_set: [obs_set]},
        obs_set=obs_set,
        output_dim=6 if obs_set == "actor" else 1,
        hidden_dims=(16, 12),
        activation="elu",
        obs_normalization=True,
        distribution_cfg=(
            {"class_name": "GaussianDistribution", "init_std": 0.7, "std_type": "scalar"}
            if obs_set == "actor"
            else None
        ),
    )


@pytest.mark.parametrize(("obs_set", "obs_dim"), [("actor", 41), ("critic", 50)])
def test_affine_migration_preserves_pinned_rsl_model_outputs(obs_set: str, obs_dim: int):
    torch.manual_seed(41)
    source = _model(MLPModel, obs_set=obs_set, obs_dim=obs_dim)
    with torch.no_grad():
        source.obs_normalizer._mean.copy_(torch.linspace(-0.4, 0.5, obs_dim).reshape(1, -1))
        source.obs_normalizer._std.copy_(torch.linspace(0.08, 1.9, obs_dim).reshape(1, -1))
        source.obs_normalizer._var.copy_(source.obs_normalizer._std.square())
    migrated = _model(SemanticNormalizationMLP, obs_set=obs_set, obs_dim=obs_dim)
    migrated.load_state_dict(migrate_semantic_state_dict(source.state_dict()), strict=True)

    raw = torch.randn((257, obs_dim)) * 1.7
    source_output = source(TensorDict({obs_set: raw}, batch_size=[257]))
    migrated_output = migrated(TensorDict({obs_set: raw}, batch_size=[257]))

    torch.testing.assert_close(migrated_output, source_output, rtol=1e-6, atol=2e-6)


def test_semantic_normalizer_updates_only_running_proprioceptive_channels():
    model = _model(SemanticNormalizationMLP, obs_set="actor", obs_dim=41)
    before_mean = model.obs_normalizer._mean.clone()
    before_std = model.obs_normalizer._std.clone()
    observations = TensorDict({"actor": torch.randn((64, 41))}, batch_size=[64])

    model.update_normalization(observations)

    assert not torch.equal(model.obs_normalizer._mean[:, :22], before_mean[:, :22])
    assert torch.equal(model.obs_normalizer._mean[:, 22:], before_mean[:, 22:])
    assert not torch.equal(model.obs_normalizer._std[:, :22], before_std[:, :22])
    assert torch.equal(model.obs_normalizer._std[:, 22:], before_std[:, 22:])


def test_semantic_contract_fixes_command_units_and_critic_privileged_channels():
    contract = semantic_normalizer_contract()

    assert contract["schema"] == "ascento_semantic_normalization/v1"
    assert contract["epsilon"] == pytest.approx(0.01)
    assert contract["fixed_scaled"]["world_target_error_xy_m"]["scale"] == [1.0, 1.0]
    assert contract["fixed_scaled"]["world_target_heading_error_rad"]["scale"] == [
        3.141592653589793
    ]
    assert contract["critic_extra_running_channels"] == [
        "root_pos_xyz",
        "root_lin_vel_w_xyz",
        "root_ang_vel_w_xyz",
    ]


def test_checkpoint_migration_records_contract_and_leaves_optimizer_fresh():
    actor = _model(MLPModel, obs_set="actor", obs_dim=41)
    critic = _model(MLPModel, obs_set="critic", obs_dim=50)
    payload = {
        "iter": 0,
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "infos": {"task_contract": {"topology_sha256": "frozen"}},
    }

    migrated = migrate_checkpoint_payload(payload)

    assert migrated["infos"]["normalizer_contract"] == semantic_normalizer_contract()
    assert migrated["infos"]["task_contract"] == payload["infos"]["task_contract"]
    assert migrated["optimizer_state_dict"] == payload["optimizer_state_dict"]
    assert not torch.equal(
        migrated["actor_state_dict"]["mlp.0.weight"], payload["actor_state_dict"]["mlp.0.weight"]
    )
    assert not torch.equal(
        migrated["critic_state_dict"]["mlp.0.weight"], payload["critic_state_dict"]["mlp.0.weight"]
    )


def test_checkpoint_normalizer_contract_selects_matching_model_and_rejects_missing_semantic_metadata():
    cfg = type(
        "AgentCfg",
        (),
        {
            "actor": type("ModelCfg", (), {"class_name": "MLPModel"})(),
            "critic": type("ModelCfg", (), {"class_name": "MLPModel"})(),
        },
    )()
    configure_rl_cfg_for_normalizer_contract(cfg, semantic_normalizer_contract())
    assert cfg.actor.class_name.endswith(":SemanticNormalizationMLP")
    assert cfg.critic.class_name == cfg.actor.class_name

    semantic_model = _model(SemanticNormalizationMLP, obs_set="actor", obs_dim=41)
    with pytest.raises(ValueError, match="lacks the semantic normalizer contract"):
        require_model_normalizer_contract({}, semantic_model)
