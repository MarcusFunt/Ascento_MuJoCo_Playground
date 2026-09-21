import torch
from mjlab.tasks.registry import load_env_cfg

from ascento_mjlab.control_contract import current_action_contract
from ascento_mjlab.plant_contract import (
    PLANT_CONTRACT_SCHEMA_VERSION,
    ROBOT_MJCF,
    current_plant_contract,
    plant_contracts_compatible,
    robot_mjcf_sha256,
)
from ascento_mjlab.provenance_runner import AscentoProvenanceRunner
from ascento_mjlab.task_contract import current_task_contract, task_contracts_compatible


def test_current_plant_contract_hashes_the_real_robot_asset():
    contract = current_plant_contract()

    assert contract["schema_version"] == PLANT_CONTRACT_SCHEMA_VERSION
    assert contract["peak_effort_nm"] == 65.0
    assert contract["robot_mjcf_sha256"] == robot_mjcf_sha256(ROBOT_MJCF)
    assert plant_contracts_compatible(contract, dict(contract))


def test_checkpoint_embeds_the_plant_contract_and_supports_safe_actor_transfer(
    tmp_path, monkeypatch
):
    runner = object.__new__(AscentoProvenanceRunner)
    import ascento_mjlab.tasks  # noqa: F401

    cfg = load_env_cfg("Ascento-Balance-Flat", play=False)
    runner.env = type(
        "Env",
        (),
        {"unwrapped": type("Base", (), {"common_step_counter": 7, "cfg": cfg})()},
    )()
    runner.current_learning_iteration = 3
    runner.cfg = {"upload_model": False}
    runner.alg = type("Algorithm", (), {"save": lambda self: {"actor_state_dict": {}}})()
    runner.logger = type("Logger", (), {})()
    path = tmp_path / "model.pt"

    runner.save(str(path), infos={"purpose": "test"})

    payload = torch.load(path, weights_only=False)
    assert plant_contracts_compatible(payload["infos"]["plant_contract"], current_plant_contract())
    assert payload["infos"]["action_contract"] == current_action_contract()
    assert task_contracts_compatible(payload["infos"]["task_contract"], current_task_contract(cfg))

    transfer_runner = object.__new__(AscentoProvenanceRunner)
    transfer_runner.env = runner.env
    transfer_runner.current_learning_iteration = 12
    transfer_runner.env.unwrapped.common_step_counter = 91
    loaded = {}
    transfer_runner.alg = type(
        "Algorithm",
        (),
        {"load": lambda self, state, load_cfg, strict: loaded.update(state)},
    )()
    original_load = torch.load

    def safe_load(*args, **kwargs):
        assert kwargs.get("weights_only") is True
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", safe_load)
    lineage = transfer_runner.initialize_from_compatible_actor(path)

    assert lineage["kind"] == "compatible_actor_transfer"
    assert transfer_runner.current_learning_iteration == 0
    assert transfer_runner.env.unwrapped.common_step_counter == 0
    assert loaded["actor_state_dict"] == {}
