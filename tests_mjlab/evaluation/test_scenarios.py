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


def test_balance_gate_v5_rejects_heading_drift_and_persistent_spin():
    suite = load_suite(Path("benchmarks/suites/balance_gate_v5.toml"))

    assert suite.suite_id == "balance_gate_v5"
    assert {
        (gate.gate_id, gate.family, gate.metric, gate.statistic, gate.threshold)
        for gate in suite.gates
    } >= {
        (
            "long_endurance_p95_max_heading_error",
            "long_endurance",
            "max_heading_error",
            "p95",
            0.35,
        ),
        ("nominal_p95_heading_error_rms", "nominal", "heading_error_rms", "p95", 0.15),
        ("nominal_p95_yaw_rate_rms", "nominal", "yaw_rate_rms", "p95", 0.15),
    }


def test_balance_dev_v2_includes_fast_heading_screen():
    suite = load_suite(Path("benchmarks/suites/balance_dev_v2.toml"))

    assert suite.suite_id == "balance_dev_v2"
    assert {gate.metric for gate in suite.gates} >= {"heading_error_rms", "yaw_rate_rms"}


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


def test_guard_morphology_baseline_suite_covers_precision_recovery_medium_and_long():
    suite = load_suite(Path("benchmarks/suites/guard_generalist_morphology_baseline_v1.toml"))
    scenarios = materialize_suite(suite, step_dt=0.02)

    assert suite.task == "Ascento-Generalist-Locomotion-Flat"
    assert not suite.gates
    assert len(scenarios) == 256
    families = {scenario.family for scenario in scenarios}
    assert families == {
        "flat_precision",
        "push_recovery_retarget",
        "medium_target",
        "long_target",
    }
    by_family = {
        name: [scenario for scenario in scenarios if scenario.family == name] for name in families
    }
    assert {len(items) for items in by_family.values()} == {64}
    command_offsets = {name: items[0].commands[0].values[0] for name, items in by_family.items()}
    assert command_offsets == {
        "flat_precision": 0.15,
        "push_recovery_retarget": 0.15,
        "medium_target": 0.75,
        "long_target": 2.5,
    }
    assert by_family["push_recovery_retarget"][0].disturbances[0].equivalent_delta_v == 0.05


def test_guard_morphology_replay_smoke_suite_is_small_and_diagnostic():
    suite = load_suite(
        Path("benchmarks/suites/guard_generalist_morphology_replay_smoke_v1.toml")
    )
    scenarios = materialize_suite(suite, step_dt=0.02)

    assert suite.task == "Ascento-Generalist-Locomotion-Flat"
    assert suite.policy_mode == "deterministic"
    assert not suite.gates
    assert len(scenarios) == 4
    assert {scenario.family for scenario in scenarios} == {
        "flat_precision",
        "push_recovery_retarget",
        "medium_target",
        "long_target",
    }
