from types import SimpleNamespace

import pytest
import torch

from ascento_mjlab.mdp import rewards


def _teacher_checkpoint(path, *, bias=1.25):
    state = {
        "obs_normalizer._mean": torch.zeros((1, 41)),
        "obs_normalizer._std": torch.ones((1, 41)),
    }
    for layer, shape in (
        (0, (256, 41)),
        (2, (256, 256)),
        (4, (256, 256)),
        (6, (6, 256)),
    ):
        state[f"mlp.{layer}.weight"] = torch.zeros(shape)
        state[f"mlp.{layer}.bias"] = torch.zeros(shape[0])
    state["mlp.6.bias"][0] = bias
    torch.save({"actor_state_dict": state}, path)


def test_reference_actor_action_mse_matches_clipped_frozen_actor(tmp_path):
    checkpoint = tmp_path / "teacher.pt"
    _teacher_checkpoint(checkpoint)
    env = SimpleNamespace(
        device="cpu",
        obs_buf={"actor": torch.zeros((2, 41))},
        observation_manager=SimpleNamespace(compute_group=lambda group: torch.full((2, 41), 5.0)),
        action_manager=SimpleNamespace(
            action=torch.tensor([[1.0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        ),
    )

    penalty = rewards.reference_actor_action_mse(
        env, checkpoint_path=str(checkpoint), action_clip=1.0
    )

    assert penalty.tolist() == pytest.approx([0.0, 1.0 / 6.0])


def test_generalist_reference_regularizer_is_opt_in_and_part_of_task_contract(monkeypatch):
    from ascento_mjlab.task_contract import current_task_contract
    from ascento_mjlab.tasks.generalist_locomotion.env_cfg import (
        ascento_generalist_locomotion_env_cfg,
    )

    monkeypatch.delenv("ASCENTO_GENERALIST_REFERENCE_BC_WEIGHT", raising=False)
    monkeypatch.delenv("ASCENTO_GENERALIST_REFERENCE_CHECKPOINT", raising=False)
    default_cfg = ascento_generalist_locomotion_env_cfg()
    default_contract = current_task_contract(default_cfg)
    assert "reference_actor_action_mse" not in default_cfg.rewards

    monkeypatch.setenv("ASCENTO_GENERALIST_REFERENCE_BC_WEIGHT", "2.0")
    monkeypatch.setenv("ASCENTO_GENERALIST_REFERENCE_CHECKPOINT", "/tmp/teacher.pt")
    cfg = ascento_generalist_locomotion_env_cfg()
    contract = current_task_contract(cfg)

    term = cfg.rewards["reference_actor_action_mse"]
    assert term.weight == pytest.approx(-2.0)
    assert term.params["checkpoint_path"] == "/tmp/teacher.pt"
    assert term.params["action_clip"] == pytest.approx(1.0)
    assert contract["topology"]["rewards"] != default_contract["topology"]["rewards"]


def test_generalist_reference_regularizer_requires_checkpoint(monkeypatch):
    from ascento_mjlab.tasks.generalist_locomotion.env_cfg import (
        ascento_generalist_locomotion_env_cfg,
    )

    monkeypatch.setenv("ASCENTO_GENERALIST_REFERENCE_BC_WEIGHT", "2.0")
    monkeypatch.delenv("ASCENTO_GENERALIST_REFERENCE_CHECKPOINT", raising=False)

    with pytest.raises(ValueError, match="ASCENTO_GENERALIST_REFERENCE_CHECKPOINT"):
        ascento_generalist_locomotion_env_cfg()
