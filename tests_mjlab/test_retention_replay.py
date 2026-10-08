from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from rsl_rl.models.mlp_model import MLPModel
from tensordict import TensorDict
from torch import nn

from ascento_mjlab.evaluation.schema import CommandPoint, DisturbanceSpec, ScenarioSpec
from ascento_mjlab.ppo import apply_actor_reference_mse
from ascento_mjlab.retention_replay import (
    AnchorReplayDataset,
    AnchorReplayRecorder,
    load_anchor_replay,
    save_anchor_replay,
    validate_anchor_replay_contract,
)


def _scenario() -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id="suite/sequence/000001",
        family="sequence",
        task="Ascento-Generalist-Locomotion-Flat",
        horizon_steps=10,
        reset={},
        disturbances=(
            DisturbanceSpec(
                start_step=2,
                duration_steps=1,
                direction="+x",
                equivalent_delta_v=0.1,
            ),
        ),
        commands=(CommandPoint(step=4, name="world_target_offset", values=(0.15, 0.0, 0.0)),),
    )


def test_anchor_replay_recorder_keeps_scenario_attempt_and_cohort_identity() -> None:
    recorder = AnchorReplayRecorder(stride_steps=2)
    scenario = _scenario()

    for step in range(5):
        recorder.record(
            observations=torch.tensor([[float(step), 1.0]]),
            teacher_actions=torch.tensor([[0.2, -0.1]]),
            scenarios=[scenario],
            step=step,
            active=torch.tensor([True]),
        )

    arrays = recorder.arrays()
    assert arrays["steps"].tolist() == [0, 2, 4]
    assert arrays["attempt_ids"].tolist() == [0, 0, 1]
    assert arrays["cohort_ids"].tolist() == [0, 1, 1]
    assert arrays["scenario_ids"].tolist() == ["suite/sequence/000001"] * 3
    assert arrays["observations"].shape == (3, 2)
    assert arrays["teacher_actions"].shape == (3, 2)


def test_anchor_replay_recorder_omits_inactive_environments() -> None:
    recorder = AnchorReplayRecorder(stride_steps=1)
    scenarios = [_scenario(), _scenario()]
    recorder.record(
        observations=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        teacher_actions=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        scenarios=scenarios,
        step=0,
        active=torch.tensor([True, False]),
    )

    arrays = recorder.arrays()
    assert arrays["observations"].tolist() == [[1.0, 2.0]]
    assert torch.allclose(torch.from_numpy(arrays["teacher_actions"]), torch.tensor([[0.1, 0.2]]))
    assert len(arrays["scenario_ids"]) == 1


def test_anchor_replay_round_trip_is_versioned_and_hash_checked(tmp_path) -> None:
    recorder = AnchorReplayRecorder(stride_steps=1)
    recorder.record(
        observations=torch.tensor([[1.0, 2.0]]),
        teacher_actions=torch.tensor([[0.1, -0.2]]),
        scenarios=[_scenario()],
        step=0,
        active=torch.tensor([True]),
    )
    recorder.record(
        observations=torch.tensor([[3.0, 4.0]]),
        teacher_actions=torch.tensor([[0.3, -0.4]]),
        scenarios=[_scenario()],
        step=2,
        active=torch.tensor([True]),
    )
    metadata = {
        "source_checkpoint_sha256": "a" * 64,
        "source_actor_normalizer_sha256": "b" * 64,
        "source_task_contract_sha256": "c" * 64,
        "source_action_contract_sha256": "d" * 64,
        "source_plant_contract_sha256": "e" * 64,
        "actor_observation_contract_sha256": "f" * 64,
        "suite_id": "roadrunner_generalist_sequence_gate_v1",
        "suite_definition_sha256": "1" * 64,
        "resolved_scenario_sha256": "2" * 64,
    }
    path = tmp_path / "anchor_replay.npz"
    save_anchor_replay(path, recorder.arrays(), metadata)

    loaded = load_anchor_replay(path, expected_observation_dim=2, expected_action_dim=2)
    assert loaded.metadata["schema_version"] == 1
    assert loaded.metadata["dataset_sha256"]
    assert loaded.observations.shape == (2, 2)
    assert torch.equal(loaded.teacher_actions[0], torch.tensor([0.1, -0.2]))
    assert loaded.cohort_ids.tolist() == [0, 1]

    manifest_path = path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["suite_id"] = "tampered"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata hash"):
        load_anchor_replay(path, expected_observation_dim=2, expected_action_dim=2)

    save_anchor_replay(path, recorder.arrays(), metadata)

    arrays = dict(recorder.arrays())
    arrays["teacher_actions"][0] = np.asarray([0.9, 0.8], dtype=np.float32)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    with pytest.raises(ValueError, match="dataset hash"):
        load_anchor_replay(path, expected_observation_dim=2, expected_action_dim=2)


def test_anchor_replay_sampling_balances_precision_and_recovery() -> None:
    dataset = AnchorReplayDataset(
        observations=torch.arange(16, dtype=torch.float32).reshape(4, 4),
        teacher_actions=torch.zeros(4, 2),
        cohort_ids=torch.tensor([0, 0, 1, 1]),
        scenario_ids=("a", "b", "c", "d"),
        attempt_ids=torch.zeros(4, dtype=torch.long),
        steps=torch.arange(4),
        metadata={},
    )

    observations, actions = dataset.sample(
        20,
        device="cpu",
        generator=torch.Generator().manual_seed(3),
    )

    assert observations.shape == (20, 4)
    assert actions.shape == (20, 2)
    assert torch.isin(observations[:, 0], torch.tensor([0.0, 4.0])).sum().item() == 10
    assert torch.isin(observations[:, 0], torch.tensor([8.0, 12.0])).sum().item() == 10


def test_anchor_replay_contract_rejects_actor_or_plant_abi_drift() -> None:
    metadata = {
        "actor_observation_contract_sha256": "actor",
        "source_action_contract_sha256": "action",
        "source_plant_contract_sha256": "plant",
        "observation_dim": 41,
        "action_dim": 6,
    }
    validate_anchor_replay_contract(
        metadata,
        actor_observation_contract_sha256="actor",
        action_contract_sha256="action",
        plant_contract_sha256="plant",
        observation_dim=41,
        action_dim=6,
    )
    with pytest.raises(ValueError, match="actor_observation_contract_sha256"):
        validate_anchor_replay_contract(
            metadata,
            actor_observation_contract_sha256="other",
            action_contract_sha256="action",
            plant_contract_sha256="plant",
            observation_dim=41,
            action_dim=6,
        )


def test_actor_reference_mse_zero_coefficient_is_exact_noop() -> None:
    torch.manual_seed(7)
    actor = nn.Linear(3, 2)
    critic = nn.Linear(3, 1)
    optimizer = torch.optim.Adam((*actor.parameters(), *critic.parameters()), lr=0.01)
    inputs = torch.randn(5, 3)
    teacher = torch.randn(5, 2)
    actor_before = {name: value.detach().clone() for name, value in actor.state_dict().items()}
    critic_before = {name: value.detach().clone() for name, value in critic.state_dict().items()}
    optimizer_before = json.dumps(optimizer.state_dict(), sort_keys=True, default=str)

    metrics = apply_actor_reference_mse(
        actor,
        optimizer,
        inputs,
        teacher,
        coefficient=0.0,
    )

    assert metrics == {
        "reference_aux_loss": 0.0,
        "reference_coefficient": 0.0,
        "reference_sample_count": 0.0,
        "anchor_student_teacher_action_rms": 0.0,
        "reference_actor_gradient_norm": 0.0,
        "reference_policy_std_delta_rms": 0.0,
    }
    assert all(torch.equal(value, actor_before[name]) for name, value in actor.state_dict().items())
    assert all(
        torch.equal(value, critic_before[name]) for name, value in critic.state_dict().items()
    )
    assert json.dumps(optimizer.state_dict(), sort_keys=True, default=str) == optimizer_before


def test_actor_reference_mse_updates_actor_without_touching_critic() -> None:
    torch.manual_seed(11)
    actor = nn.Linear(3, 2)
    critic = nn.Linear(3, 1)
    optimizer = torch.optim.Adam((*actor.parameters(), *critic.parameters()), lr=0.01)
    inputs = torch.randn(8, 3)
    teacher = torch.ones(8, 2)
    actor_before = {name: value.detach().clone() for name, value in actor.state_dict().items()}
    critic_before = {name: value.detach().clone() for name, value in critic.state_dict().items()}

    metrics = apply_actor_reference_mse(
        actor,
        optimizer,
        inputs,
        teacher,
        coefficient=0.25,
    )

    assert metrics["reference_aux_loss"] > 0.0
    assert metrics["reference_coefficient"] == 0.25
    assert metrics["reference_sample_count"] == 8.0
    assert metrics["anchor_student_teacher_action_rms"] >= 0.0
    assert metrics["reference_actor_gradient_norm"] > 0.0
    assert any(
        not torch.equal(value, actor_before[name]) for name, value in actor.state_dict().items()
    )
    assert all(
        torch.equal(value, critic_before[name]) for name, value in critic.state_dict().items()
    )
    assert all(parameter not in optimizer.state for parameter in critic.parameters())


def test_actor_reference_mse_accepts_the_rsl_rl_actor_tensor_dict_abi() -> None:
    actor_observations = torch.zeros(1, 3)
    actor = MLPModel(
        obs=TensorDict({"actor": actor_observations}, batch_size=[1]),
        obs_groups={"policy": ["actor"]},
        obs_set="policy",
        output_dim=2,
        hidden_dims=(8,),
        obs_normalization=False,
        distribution_cfg=None,
    )
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.01)
    observations = torch.randn(6, 3)
    teacher_actions = torch.ones(6, 2)

    metrics = apply_actor_reference_mse(
        actor,
        optimizer,
        observations,
        teacher_actions,
        coefficient=0.1,
    )

    assert metrics["reference_sample_count"] == 6.0
    assert metrics["reference_policy_std_delta_rms"] == 0.0
