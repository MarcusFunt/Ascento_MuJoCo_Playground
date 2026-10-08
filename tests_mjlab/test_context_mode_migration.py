from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

import ascento_mjlab.semantic_normalization as normalization
import ascento_mjlab.tasks  # noqa: F401 - complete mjlab task registration before MDP imports
from ascento_mjlab.mdp.observations import obstacle_mode
from ascento_mjlab.task_contract import (
    classify_task_contract_compatibility,
    current_task_contract,
)
from ascento_mjlab.tasks.generalist_locomotion.env_cfg import (
    ascento_generalist_locomotion_env_cfg,
)


def _task_contract(*, with_mode: bool) -> dict:
    actor_terms = {"state": {}}
    if with_mode:
        actor_terms["obstacle_mode"] = {}
    critic_terms = {**actor_terms, "privileged": {}}
    return {
        "schema_version": 1,
        "id": "ascento_task_topology_v1",
        "topology_sha256": "context-mode-test",
        "topology": {
            "task_id": "Ascento-Generalist-Locomotion-Flat",
            "observations": {
                "actor": {"terms": actor_terms},
                "critic": {"terms": critic_terms},
            },
        },
    }


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


def test_obstacle_mode_is_an_explicit_fixed_float_command(monkeypatch):
    env = SimpleNamespace(num_envs=3, device="cpu")
    monkeypatch.delenv("ASCENTO_GENERALIST_OBSTACLE_MODE", raising=False)
    assert torch.equal(obstacle_mode(env), torch.zeros((3, 1)))

    monkeypatch.setenv("ASCENTO_GENERALIST_OBSTACLE_MODE", "1")
    assert torch.equal(obstacle_mode(env), torch.ones((3, 1)))

    monkeypatch.setenv("ASCENTO_GENERALIST_OBSTACLE_MODE", "0.5")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        obstacle_mode(env)


def test_context_normalizer_v2_keeps_mode_identity_and_critic_extras_running():
    contract = normalization.context_mode_normalizer_contract()
    actor = normalization.SemanticEmpiricalNormalization(42)
    critic = normalization.SemanticEmpiricalNormalization(51)

    assert contract["schema"] == "ascento_semantic_normalization/v2"
    assert contract["actor_observation_dim"] == 42
    assert contract["critic_observation_dim"] == 51
    assert contract["identity_or_fixed"]["obstacle_mode"]["index"] == 41
    assert not actor.running_mask[0, 41]
    assert not critic.running_mask[0, 41]
    assert torch.all(critic.running_mask[0, 42:])
    assert actor.fixed_scale[0, 41] == 1
    assert critic.fixed_scale[0, 41] == 1


@pytest.mark.parametrize(("obs_set", "old_dim", "new_dim"), [("actor", 41, 42), ("critic", 50, 51)])
def test_context_mode_checkpoint_migration_preserves_mode_zero_model_outputs(
    obs_set: str, old_dim: int, new_dim: int
):
    torch.manual_seed(1729)
    source = _model(normalization.SemanticNormalizationMLP, obs_set=obs_set, obs_dim=old_dim)
    if obs_set == "critic":
        with torch.no_grad():
            source.obs_normalizer._mean[:, 41:] = torch.linspace(-0.3, 0.4, old_dim - 41)
            source.obs_normalizer._std[:, 41:] = torch.linspace(0.5, 1.5, old_dim - 41)
            source.obs_normalizer._var[:, 41:] = source.obs_normalizer._std[:, 41:].square()
    target_contract = _task_contract(with_mode=True)
    payload = {
        "iter": 0,
        f"{obs_set}_state_dict": source.state_dict(),
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "infos": {
            "normalizer_contract": normalization.semantic_normalizer_contract(),
            "task_contract": _task_contract(with_mode=False),
        },
    }
    # Include both models because the migration contract covers actor and critic together.
    other_set = "critic" if obs_set == "actor" else "actor"
    other_dim = 50 if other_set == "critic" else 41
    other = _model(normalization.SemanticNormalizationMLP, obs_set=other_set, obs_dim=other_dim)
    payload[f"{other_set}_state_dict"] = other.state_dict()

    migrated = normalization.migrate_context_mode_checkpoint_payload(
        payload, target_task_contract=target_contract
    )
    target = _model(normalization.SemanticNormalizationMLP, obs_set=obs_set, obs_dim=new_dim)
    target.load_state_dict(migrated[f"{obs_set}_state_dict"], strict=True)

    raw_old = torch.randn((257, old_dim)) * 1.7
    raw_new = (
        torch.cat((raw_old, torch.zeros((257, 1))), dim=1)
        if obs_set == "actor"
        else torch.cat((raw_old[:, :41], torch.zeros((257, 1)), raw_old[:, 41:]), dim=1)
    )
    source_output = source(TensorDict({obs_set: raw_old}, batch_size=[257]))
    migrated_output = target(TensorDict({obs_set: raw_new}, batch_size=[257]))

    torch.testing.assert_close(migrated_output, source_output, rtol=1e-6, atol=1e-6)
    assert (
        migrated["infos"]["normalizer_contract"] == normalization.context_mode_normalizer_contract()
    )
    assert migrated["infos"]["task_contract"] == target_contract
    assert migrated["optimizer_state_dict"] == payload["optimizer_state_dict"]


def test_context_mode_migration_rejects_nonzero_iteration_and_wrong_contract():
    payload = {
        "iter": 1,
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "infos": {"normalizer_contract": normalization.semantic_normalizer_contract()},
    }
    with pytest.raises(ValueError, match="iteration-zero"):
        normalization.migrate_context_mode_checkpoint_payload(
            payload, target_task_contract=_task_contract(with_mode=True)
        )

    payload["iter"] = 0
    payload["infos"]["normalizer_contract"] = normalization.context_mode_normalizer_contract()
    with pytest.raises(ValueError, match="semantic-normalization v1"):
        normalization.migrate_context_mode_checkpoint_payload(
            payload, target_task_contract=_task_contract(with_mode=True)
        )


def test_context_normalizer_contract_rejects_41d_and_42d_mixing():
    legacy = _model(normalization.SemanticNormalizationMLP, obs_set="actor", obs_dim=41)
    contextual = _model(normalization.SemanticNormalizationMLP, obs_set="actor", obs_dim=42)

    assert (
        normalization.model_normalizer_contract(legacy)["schema"]
        == "ascento_semantic_normalization/v1"
    )
    assert (
        normalization.model_normalizer_contract(contextual)
        == normalization.context_mode_normalizer_contract()
    )
    with pytest.raises(ValueError, match="does not match"):
        normalization.require_model_normalizer_contract(
            {"normalizer_contract": normalization.semantic_normalizer_contract()}, contextual
        )


def test_generalist_task_contract_rejects_the_old_41d_observation_topology():
    current_cfg = ascento_generalist_locomotion_env_cfg()
    actor_terms = current_cfg.observations["actor"].terms
    critic_terms = current_cfg.observations["critic"].terms
    assert tuple(actor_terms)[-1] == "obstacle_mode"
    assert tuple(critic_terms)[len(actor_terms) - 1] == "obstacle_mode"

    del actor_terms["obstacle_mode"]
    del critic_terms["obstacle_mode"]
    old_contract = current_task_contract(current_cfg)
    # Rebuild the active contract from the registered 42D configuration.
    active_contract = current_task_contract(ascento_generalist_locomotion_env_cfg())

    assert not classify_task_contract_compatibility(old_contract, active_contract).is_compatible
