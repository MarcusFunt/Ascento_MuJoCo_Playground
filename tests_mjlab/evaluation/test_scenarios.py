from pathlib import Path

from ascento_mjlab.evaluation.scenarios import materialize_suite, scenario_seed
from ascento_mjlab.evaluation.schema import FamilySpec, SuiteSpec, load_suite


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


def test_balance_gate_v2_preserves_v1_and_adds_symmetry_gates():
    suite = load_suite(Path("benchmarks/suites/balance_gate_v2.toml"))

    assert suite.suite_id == "balance_gate_v2"
    assert {gate.gate_id for gate in suite.gates} >= {
        "nominal_p95_net_displacement",
        "disturbance_p95_recovery_time",
        "nominal_p95_leg_hip_mismatch_rms",
        "nominal_p95_leg_knee_mismatch_rms",
    }


def test_balance_gate_v3_uses_authoritative_joint_applied_effort_metrics():
    suite = load_suite(Path("benchmarks/suites/balance_gate_v3.toml"))

    assert suite.suite_id == "balance_gate_v3"
    metrics = {gate.metric for gate in suite.gates}
    assert "joint_applied_effort_rms" in metrics
    assert "joint_applied_saturation_fraction" in metrics


def test_balance_gate_v4_rejects_long_horizon_world_target_drift():
    suite = load_suite(Path("benchmarks/suites/balance_gate_v4.toml"))

    assert suite.suite_id == "balance_gate_v4"
    assert {
        (gate.gate_id, gate.family, gate.metric, gate.statistic, gate.threshold)
        for gate in suite.gates
    } >= {
        ("long_endurance_p95_max_target_error", "long_endurance", "max_target_error", "p95", 0.75)
    }


def test_specialist_suites_cover_binary_hold_and_shaping_metrics():
    recovery = load_suite(Path("benchmarks/suites/recovery_gate_v1.toml"))
    jump = load_suite(Path("benchmarks/suites/jump_gate_v1.toml"))

    recovery_metrics = {gate.metric for gate in recovery.gates}
    jump_metrics = {gate.metric for gate in jump.gates}
    assert {"recovery_success", "recovery_from_start_s", "max_recovery_hold_s"} <= recovery_metrics
    assert {
        "jump_takeoff",
        "jump_landing",
        "jump_recovered_landing",
        "post_landing_hold_s",
    } <= jump_metrics
    assert "shaping_reward_abs_mean" in recovery_metrics
    assert "shaping_reward_abs_mean" in jump_metrics
