from types import SimpleNamespace

import pytest

from ascento_mjlab.checkpoint_contract import require_current_checkpoint_contracts
from ascento_mjlab.control_contract import current_action_contract
from ascento_mjlab.plant_contract import current_plant_contract
from ascento_mjlab.task_contract import (
    TASK_CONTRACT_ID,
    current_task_contract,
    current_task_contract_for_task,
    require_current_task_contract,
    task_contracts_compatible,
)


def test_registered_task_contract_is_stable_and_names_the_task():
    first = current_task_contract_for_task("Ascento-Balance-Flat")
    second = current_task_contract_for_task("Ascento-Balance-Flat")

    assert first["id"] == TASK_CONTRACT_ID
    assert first["topology"]["task_id"] == "Ascento-Balance-Flat"
    assert first["topology_sha256"] == second["topology_sha256"]
    assert task_contracts_compatible(first, second)


def test_task_contract_rejects_changed_reward_or_observation_topology():
    cfg = SimpleNamespace(
        task_id="synthetic",
        decimation=5,
        scale_rewards_by_dt=True,
        observations={"actor": {"gravity": "projected_gravity"}},
        actions={"targets": "structured_targets"},
        commands={},
        rewards={"upright": {"weight": 2.0}},
        terminations={},
        events={},
        metrics={},
    )
    contract = current_task_contract(cfg)
    cfg.rewards["upright"]["weight"] = 1.0

    with pytest.raises(ValueError, match="task topology"):
        require_current_task_contract(contract, cfg)


def test_checkpoint_validation_reports_missing_task_contract_before_rollout():
    cfg = SimpleNamespace(
        task_id="synthetic",
        decimation=5,
        scale_rewards_by_dt=True,
        observations={},
        actions={},
        commands={},
        rewards={},
        terminations={},
        events={},
        metrics={},
    )
    infos = {
        "plant_contract": current_plant_contract(),
        "action_contract": current_action_contract(),
    }

    with pytest.raises(ValueError, match="task topology contract"):
        require_current_checkpoint_contracts(infos, cfg)
