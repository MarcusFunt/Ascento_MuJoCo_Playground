from types import SimpleNamespace

import torch

from ascento_mjlab.evaluation.runner import _apply_commands, task_capabilities
from ascento_mjlab.evaluation.schema import CommandPoint, ScenarioSpec


def test_speed_fraction_evaluation_command_scales_to_checkpoint_range():
    calls = []
    term = SimpleNamespace(
        cfg=SimpleNamespace(max_speed_mps=0.8),
        command=torch.zeros((1, 1)),
        set_manual_speed_mps=lambda value, *, env_id: calls.append((value, env_id)),
    )
    lookups = []

    def get_term(name):
        lookups.append(name)
        if name != "speed":
            raise KeyError(name)
        return term

    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=get_term),
        num_envs=1,
        device=torch.device("cpu"),
    )
    scenario = ScenarioSpec(
        scenario_id="speed/0",
        family="speed",
        task="Ascento-Locomotion-Speed-Flat",
        horizon_steps=10,
        reset={},
        commands=(CommandPoint(step=2, name="speed_fraction", values=(0.5,)),),
    )

    _apply_commands(env, [scenario], 2)

    assert calls == [(0.4, 0)]
    assert lookups == ["speed"]


def test_speed_conditioned_locomotion_advertises_gate_capabilities():
    capabilities = task_capabilities("Ascento-Locomotion-Speed-Flat")

    assert {"locomotion", "command:speed", "speed_tracking"} <= capabilities


def test_speed_gate_scenario_is_versioned_and_covers_runtime_levels():
    from ascento_mjlab.evaluation.cli import resolve_suite_path
    from ascento_mjlab.evaluation.runner import task_step_dt
    from ascento_mjlab.evaluation.scenarios import materialize_suite
    from ascento_mjlab.evaluation.schema import load_suite

    suite = load_suite(resolve_suite_path("locomotion_speed_command_gate_v1"))
    scenarios = materialize_suite(suite, task_step_dt(suite.task))

    assert suite.task == "Ascento-Locomotion-Speed-Flat"
    assert [gate.metric for gate in suite.gates] == [
        "success",
        "speed_tracking_rmse_fraction",
        "zero_command_speed_rms_mps",
    ]
    assert [
        point.values[0] for point in scenarios[0].commands if point.name == "speed_fraction"
    ] == [
        0.25,
        0.70,
        0.0,
    ]


def test_evaluation_uses_and_restores_managed_speed_cap(monkeypatch, tmp_path):
    import json

    from ascento_mjlab.evaluation import cli

    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "models"
    checkpoint_dir.mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(json.dumps({"max_speed_mps": 0.8}), encoding="utf-8")
    checkpoint = checkpoint_dir / "model_5000.pt"
    checkpoint.write_bytes(b"checkpoint")
    observed = {}
    monkeypatch.setenv("ASCENTO_LOCOMOTION_MAX_SPEED_COMMAND_MPS", "9")

    def fake_evaluate(**kwargs):
        observed["cap"] = cli.os.environ["ASCENTO_LOCOMOTION_MAX_SPEED_COMMAND_MPS"]
        return "ok", tmp_path

    monkeypatch.setattr(cli, "_evaluate", fake_evaluate)
    result = cli.evaluate(
        checkpoint=checkpoint,
        suite_path=cli.resolve_suite_path("locomotion_speed_command_gate_v1"),
        output_base=tmp_path,
        batch_size=1,
        device="cpu",
    )

    assert observed["cap"] == "0.8"
    assert cli.os.environ["ASCENTO_LOCOMOTION_MAX_SPEED_COMMAND_MPS"] == "9"
    assert result == ("ok", tmp_path)


def test_cached_registry_config_uses_managed_speed_cap_for_contract(monkeypatch, tmp_path):
    from mjlab.tasks.registry import load_env_cfg

    import ascento_mjlab.tasks  # noqa: F401
    from ascento_mjlab.evaluation import cli
    from ascento_mjlab.task_contract import current_task_contract, current_task_contract_for_task
    from ascento_mjlab.tasks.locomotion_speed.env_cfg import configure_speed_command_cap

    task = "Ascento-Locomotion-Speed-Flat"
    registered_cfg = load_env_cfg(task, play=False)
    registered_cap = registered_cfg.commands["speed"].max_speed_mps
    managed_cap = 0.8 if registered_cap != 0.8 else 0.9
    monkeypatch.setenv("ASCENTO_LOCOMOTION_MAX_SPEED_COMMAND_MPS", str(managed_cap))

    cfg = configure_speed_command_cap(load_env_cfg(task, play=False))

    assert cfg.commands["speed"].max_speed_mps == managed_cap
    assert cfg.observations["actor"].terms["speed_command"].params["max_speed_mps"] == managed_cap
    assert cfg.observations["critic"].terms["speed_command"].params["max_speed_mps"] == managed_cap
    assert cfg.rewards["commanded_travel_speed"].params["std"] == max(
        0.10, min(0.25, managed_cap * 0.25)
    )
    assert current_task_contract(cfg) == current_task_contract_for_task(task)

    from ascento_mjlab.control_contract import current_action_contract
    from ascento_mjlab.plant_contract import current_plant_contract

    checkpoint = tmp_path / "model_5000.pt"
    torch.save(
        {
            "infos": {
                "plant_contract": current_plant_contract(),
                "action_contract": current_action_contract(),
                "task_contract": current_task_contract(cfg),
            }
        },
        checkpoint,
    )
    preflight = cli._checkpoint_contract_preflight(checkpoint, task)

    assert preflight["status"] == "passed"
