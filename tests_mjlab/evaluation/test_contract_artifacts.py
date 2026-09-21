from types import SimpleNamespace

import torch

from ascento_mjlab.evaluation import cli as evaluation_cli
from ascento_mjlab.evaluation.schema import EvaluationStatus, ScenarioSpec, SuiteSpec
from ascento_mjlab.task_contract import current_task_contract


def test_rejected_checkpoint_writes_complete_invalid_evaluation_artifacts(monkeypatch, tmp_path):
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
    suite = SuiteSpec(
        schema_version=1,
        suite_id="synthetic_gate",
        task="Synthetic-Task",
        root_seed=1,
        policy_mode="deterministic",
        families=(),
        gates=(),
    )
    checkpoint = tmp_path / "model_1.pt"
    torch.save({"infos": {}}, checkpoint)
    scenario = ScenarioSpec(
        scenario_id="scenario",
        family="nominal",
        task=suite.task,
        horizon_steps=10,
        reset={},
    )

    monkeypatch.setattr(evaluation_cli, "load_suite", lambda _path: suite)
    monkeypatch.setattr(evaluation_cli, "task_capabilities", lambda _task: set())
    monkeypatch.setattr(evaluation_cli, "task_step_dt", lambda _task: 0.01)
    monkeypatch.setattr(evaluation_cli, "materialize_suite", lambda *_args: [scenario])
    monkeypatch.setattr(evaluation_cli, "load_env_cfg", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(evaluation_cli, "current_task_contract_for_task", lambda _task: current_task_contract(cfg))
    monkeypatch.setattr(
        evaluation_cli,
        "_manifest",
        lambda **_kwargs: {"task": suite.task, "checkpoint": str(checkpoint)},
    )
    monkeypatch.setattr(
        evaluation_cli,
        "run_scenarios",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("rollout must not start")),
    )
    monkeypatch.setattr(
        evaluation_cli,
        "render_html",
        lambda path, **_kwargs: path.write_text("<html>invalid</html>", encoding="utf-8"),
    )

    status, output = evaluation_cli.evaluate(
        checkpoint=checkpoint,
        suite_path=tmp_path / "synthetic.toml",
        output_base=tmp_path / "evaluations",
        batch_size=1,
        device="cpu",
    )

    assert status == EvaluationStatus.INVALID
    assert (output / "manifest.json").is_file()
    assert (output / "gate.json").is_file()
    assert (output / "consistency.json").is_file()
    assert (output / "report.html").is_file()
