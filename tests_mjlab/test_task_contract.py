import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from ascento_mjlab.checkpoint_contract import require_current_checkpoint_contracts
from ascento_mjlab.control_contract import current_action_contract
from ascento_mjlab.plant_contract import current_plant_contract
from ascento_mjlab.task_contract import (
    TASK_CONTRACT_ID,
    TaskContractStatus,
    canonical_task_cfg_for_runtime_cfg,
    classify_actor_transfer_compatibility,
    classify_task_contract_compatibility,
    current_task_contract,
    current_task_contract_for_task,
    require_current_task_contract,
    task_contracts_compatible,
)


def _legacy_null_task_id(contract):
    legacy = deepcopy(contract)
    legacy["topology"]["task_id"] = None
    legacy["topology_sha256"] = hashlib.sha256(
        json.dumps(legacy["topology"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return legacy


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


def test_v1_null_task_id_migration_requires_every_other_topology_field_to_match():
    current = current_task_contract_for_task("Ascento-Balance-Flat")
    legacy = _legacy_null_task_id(current)

    compatibility = classify_task_contract_compatibility(legacy, current)

    assert compatibility.status == TaskContractStatus.MIGRATED_V1_NULL_TASK_ID
    assert compatibility.is_compatible is True
    assert task_contracts_compatible(legacy, current)

    changed = deepcopy(legacy)
    changed["topology"]["rewards"]["upright"]["weight"] = 123.0
    changed["topology_sha256"] = hashlib.sha256(
        json.dumps(changed["topology"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    assert classify_task_contract_compatibility(changed, current).status == TaskContractStatus.INCOMPATIBLE
    assert not task_contracts_compatible(changed, current)


def test_task_contract_classifier_marks_missing_contract_legacy():
    current = current_task_contract_for_task("Ascento-Balance-Flat")

    compatibility = classify_task_contract_compatibility(None, current)

    assert compatibility.status == TaskContractStatus.LEGACY
    assert compatibility.is_compatible is False


def test_actor_transfer_allows_new_reward_and_event_contracts_but_not_actor_abi_changes():
    source = _legacy_null_task_id(current_task_contract_for_task("Ascento-Balance-Flat"))
    target = current_task_contract_for_task("Ascento-Locomotion-Flat")

    compatibility = classify_actor_transfer_compatibility(source, target)

    assert compatibility.compatible is True
    assert "exactly match" in compatibility.reason

    incompatible = deepcopy(target)
    incompatible["topology"]["observations"]["actor"]["terms"]["world_target_error"]["clip"] = (
        -2.0,
        2.0,
    )
    compatibility = classify_actor_transfer_compatibility(source, incompatible)

    assert compatibility.compatible is False
    assert "actor observations" in compatibility.reason


def test_runtime_contract_recovery_requires_one_exact_registered_topology():
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg("Ascento-Balance-Quiet-Flat", play=False)
    delattr(cfg, "task_id")

    recovered = canonical_task_cfg_for_runtime_cfg(cfg)

    assert recovered.task_id == "Ascento-Balance-Quiet-Flat"


def test_locomotion_uses_the_balance_actor_abi_and_a_fore_aft_biased_sequence():
    from mjlab.tasks.registry import load_env_cfg

    balance = current_task_contract_for_task("Ascento-Balance-Quiet-Flat")
    locomotion = current_task_contract_for_task("Ascento-Locomotion-Flat")
    assert classify_actor_transfer_compatibility(balance, locomotion).compatible is True

    cfg = load_env_cfg("Ascento-Locomotion-Flat", play=False)
    sequence = cfg.events["settle_triggered_sequence"]
    assert sequence.interval_range_s == (0.01, 0.01)
    assert sequence.params["settle_hold_s"] == 0.75
    assert sequence.params["min_delta_v"] == 0.05
    assert sequence.params["max_delta_v"] == 0.15
    assert sequence.params["fore_aft_probability"] == 0.75
    assert sequence.params["min_target_distance_m"] == 0.05
    assert sequence.params["max_target_distance_m"] == 0.20
    assert "settle_triggered_sequence" not in load_env_cfg(
        "Ascento-Locomotion-Flat", play=True
    ).events


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
