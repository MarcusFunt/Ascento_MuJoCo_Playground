from dataclasses import dataclass

from ascento_mjlab.evaluation import runner
from ascento_mjlab.evaluation.schema import EpisodeResult, ScenarioSpec


@dataclass
class _FakeRuntime:
    capacity: int
    closed: bool = False

    def close(self) -> None:
        self.closed = True


def _scenario(scenario_id: str, family: str) -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id=scenario_id,
        family=family,
        task="Ascento-Balance-Flat",
        horizon_steps=10,
        reset={},
    )


def test_run_scenarios_reuses_one_fixed_capacity_runtime(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    created: list[_FakeRuntime] = []
    batches: list[list[str]] = []

    def create_runtime(*args, **kwargs):
        runtime = _FakeRuntime(capacity=kwargs["capacity"])
        created.append(runtime)
        return runtime

    def run_batch(*args, **kwargs):
        scenario_batch = args[2]
        batches.append([scenario.scenario_id for scenario in scenario_batch])
        return (
            [
                EpisodeResult(
                    scenario_id=scenario.scenario_id,
                    family=scenario.family,
                    success=True,
                    termination_reason="horizon",
                    episode_steps=scenario.horizon_steps,
                    metrics={},
                )
                for scenario in scenario_batch
            ],
            {
                "step_dt": 0.01,
                "checkpoint_plant_contract": {"schema_id": "plant"},
                "checkpoint_action_contract": {"schema_id": "structured_targets_v1"},
            },
        )

    monkeypatch.setattr(runner, "_create_runtime", create_runtime)
    monkeypatch.setattr(runner, "_run_batch", run_batch)
    scenarios = [
        _scenario("a-0", "a"),
        _scenario("a-1", "a"),
        _scenario("b-0", "b"),
        _scenario("b-1", "b"),
        _scenario("b-2", "b"),
    ]

    results, metadata = runner.run_scenarios(
        "Ascento-Balance-Flat", checkpoint, scenarios, batch_size=3, device="cpu"
    )

    assert len(created) == 1
    assert created[0].closed
    assert batches == [["a-0", "a-1", "a-1"], ["b-0", "b-1", "b-2"]]
    assert [result.scenario_id for result in results] == ["a-0", "a-1", "b-0", "b-1", "b-2"]
    assert metadata["batches"] == 2
