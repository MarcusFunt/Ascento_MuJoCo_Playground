from ascento_mjlab.evaluation.scenarios import materialize_suite, scenario_seed
from ascento_mjlab.evaluation.schema import FamilySpec, SuiteSpec


def _suite():
    return SuiteSpec(
        schema_version=1,
        suite_id="test_suite",
        task="Ascento-Balance-Flat",
        root_seed=123,
        policy_mode="deterministic",
        families=(
            FamilySpec(
                family_id="nominal",
                kind="uniform_reset",
                count=8,
                horizon_s=2.0,
                config={"reset": {"roll": [-0.1, 0.1], "pitch": [-0.2, 0.2]}},
            ),
        ),
        gates=(),
    )


def test_scenario_materialization_is_deterministic():
    suite = _suite()
    assert materialize_suite(suite, 0.01) == materialize_suite(suite, 0.01)


def test_scenario_seed_is_index_stable():
    suite = _suite()
    seeds = [scenario_seed(suite, "nominal", index) for index in range(8)]
    assert len(set(seeds)) == len(seeds)
    assert seeds[3] == scenario_seed(suite, "nominal", 3)


def test_horizon_is_resolved_to_integer_control_steps():
    scenarios = materialize_suite(_suite(), 0.01)
    assert {scenario.horizon_steps for scenario in scenarios} == {200}
