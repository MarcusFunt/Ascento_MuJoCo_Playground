import torch
from mjlab.rl import MjlabOnPolicyRunner
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


def _checkpoint_infos(cfg):
    return {
        "plant_contract": current_plant_contract(),
        "action_contract": current_action_contract(),
        "task_contract": current_task_contract(cfg),
    }


def _resume_runner(cfg):
    runner = object.__new__(AscentoProvenanceRunner)
    runner.env = type(
        "Env",
        (),
        {"unwrapped": type("Base", (), {"common_step_counter": 0, "cfg": cfg})()},
    )()
    runner.current_learning_iteration = 0
    runner.num_steps_per_env = 24
    return runner


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
    runner.num_steps_per_env = 24
    runner.cfg = {"upload_model": False}
    runner.alg = type("Algorithm", (), {"save": lambda self: {"actor_state_dict": {}}})()
    runner.logger = type("Logger", (), {})()
    path = tmp_path / "model.pt"

    runner.save(str(path), infos={"purpose": "test"})

    payload = torch.load(path, weights_only=False)
    assert plant_contracts_compatible(payload["infos"]["plant_contract"], current_plant_contract())
    assert payload["infos"]["action_contract"] == current_action_contract()
    assert task_contracts_compatible(payload["infos"]["task_contract"], current_task_contract(cfg))
    assert payload["infos"]["environment_progress"] == {
        "schema_version": 1,
        "common_step_counter": 7,
        "learning_iteration": 3,
        "num_steps_per_env": 24,
    }

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


def test_training_resume_restores_exact_environment_step_counter(monkeypatch):
    import ascento_mjlab.tasks  # noqa: F401

    cfg = load_env_cfg("Ascento-Balance-Recovery-Flat", play=False)
    runner = _resume_runner(cfg)
    infos = {
        **_checkpoint_infos(cfg),
        "environment_progress": {
            "schema_version": 1,
            "common_step_counter": 118_321,
            "learning_iteration": 4_930,
            "num_steps_per_env": 24,
        },
    }

    def fake_load(self, *args, **kwargs):
        self.current_learning_iteration = 4_930
        return infos

    monkeypatch.setattr(MjlabOnPolicyRunner, "load", fake_load)

    runner.load("unused.pt")

    assert runner.current_learning_iteration == 4_930
    assert runner.env.unwrapped.common_step_counter == 118_321


def test_training_resume_reconstructs_environment_progress_for_legacy_checkpoint(monkeypatch):
    import ascento_mjlab.tasks  # noqa: F401

    cfg = load_env_cfg("Ascento-Balance-Recovery-Flat", play=False)
    runner = _resume_runner(cfg)
    infos = _checkpoint_infos(cfg)

    def fake_load(self, *args, **kwargs):
        self.current_learning_iteration = 5_000
        return infos

    monkeypatch.setattr(MjlabOnPolicyRunner, "load", fake_load)

    runner.load("legacy.pt")

    assert runner.current_learning_iteration == 5_000
    assert runner.env.unwrapped.common_step_counter == 120_000


def test_actor_only_load_does_not_restore_training_environment_progress(monkeypatch):
    import ascento_mjlab.tasks  # noqa: F401

    cfg = load_env_cfg("Ascento-Balance-Recovery-Flat", play=True)
    runner = _resume_runner(cfg)
    runner.env.unwrapped.common_step_counter = 17
    infos = {
        **_checkpoint_infos(cfg),
        "environment_progress": {
            "schema_version": 1,
            "common_step_counter": 120_000,
            "learning_iteration": 5_000,
            "num_steps_per_env": 24,
        },
    }

    def fake_load(self, *args, **kwargs):
        return infos

    monkeypatch.setattr(MjlabOnPolicyRunner, "load", fake_load)

    runner.load("play.pt", load_cfg={"actor": True})

    assert runner.env.unwrapped.common_step_counter == 17
