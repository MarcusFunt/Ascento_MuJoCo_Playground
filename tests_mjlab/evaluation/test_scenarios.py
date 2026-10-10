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



def test_waypoint_gate_suite_has_versioned_pose_routes_and_hard_stop_gates():
    from ascento_mjlab.evaluation.runner import task_capabilities

    suite = load_suite(Path("benchmarks/suites/locomotion_waypoint_gate_v2.toml"))
    scenarios = materialize_suite(suite, step_dt=0.02)

    assert suite.task == "Ascento-Locomotion-Flat"
    assert len(scenarios) == 240
    assert {"command:world_target_pose", "waypoint_dwell"} <= set(suite.required_capabilities)
    assert {"command:world_target_pose", "waypoint_dwell"} <= task_capabilities(suite.task)
    assert {"waypoint_sequence_complete", "waypoint_stop_window_action_second_difference_rms"} <= {
        gate.metric for gate in suite.gates
    }
    route = next(s for s in scenarios if s.family == "route_push")
    assert [point.name for point in route.commands] == ["world_target_pose"] * 3


def test_evaluator_world_target_pose_applies_origin_and_marks_commanded():
    from types import SimpleNamespace

    import pytest
    import torch

    from ascento_mjlab.evaluation.runner import _apply_commands
    from ascento_mjlab.evaluation.schema import CommandPoint, ScenarioSpec

    origin = torch.tensor([[10.0, -2.0, 0.0]])
    class Scene(dict):
        pass

    scene = Scene(
        robot=SimpleNamespace(
            data=SimpleNamespace(
                root_link_pos_w=torch.tensor([[10.0, -2.0, 0.75]]),
                root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            )
        )
    )
    scene.env_origins = origin
    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        scene=scene,
        ascento_world_target_state={
            "target_xy": torch.zeros((1, 2)),
            "target_yaw": torch.zeros(1),
        },
    )
    scenario = ScenarioSpec(
        scenario_id="pose/0",
        family="pose",
        task="Ascento-Locomotion-Flat",
        horizon_steps=100,
        reset={},
        commands=(CommandPoint(step=4, name="world_target_pose", values=(1.0, 0.5, 1.2)),),
    )

    commanded = _apply_commands(env, [scenario], 4)

    assert commanded.tolist() == [True]
    assert env.ascento_world_target_state["target_xy"].tolist() == [[11.0, -1.5]]
    assert env.ascento_world_target_state["target_yaw"].item() == pytest.approx(1.2)


def test_evaluator_rejects_two_targets_for_one_environment_on_one_step():
    from types import SimpleNamespace

    import torch

    from ascento_mjlab.evaluation.runner import _apply_commands
    from ascento_mjlab.evaluation.schema import CommandPoint, ScenarioSpec

    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        scene=SimpleNamespace(env_origins=torch.zeros((1, 3))),
        ascento_world_target_state={
            "target_xy": torch.zeros((1, 2)),
            "target_yaw": torch.zeros(1),
        },
    )
    scenario = ScenarioSpec(
        scenario_id="ambiguous/0",
        family="pose",
        task="Ascento-Locomotion-Flat",
        horizon_steps=10,
        reset={},
        commands=(
            CommandPoint(step=0, name="world_target_offset", values=(1.0, 0.0, 0.0)),
            CommandPoint(step=0, name="world_target_pose", values=(1.0, 0.0, 0.0)),
        ),
    )

    import pytest

    with pytest.raises(RuntimeError, match="two world targets"):
        _apply_commands(env, [scenario], 0)
def test_generalist_driver_suite_covers_long_goals_turns_routes_recovery_and_control_gates():
    from math import hypot
    from pathlib import Path

    import pytest

    from ascento_mjlab.evaluation.scenarios import materialize_suite
    from ascento_mjlab.evaluation.schema import load_suite

    suite = load_suite(Path("benchmarks/suites/roadrunner_generalist_driver_gate_v1.toml"))
    scenarios = materialize_suite(suite, step_dt=0.01)

    assert suite.task == "Ascento-Generalist-Locomotion-Flat"
    assert len(scenarios) == 256
    assert {
        "locomotion",
        "exact_reset",
        "force_disturbance",
        "deterministic_policy",
        "command:world_target_pose",
        "waypoint_dwell",
    } <= set(suite.required_capabilities)

    by_family = {
        family: [scenario for scenario in scenarios if scenario.family == family]
        for family in ("forward_2m", "turn_3m", "route_turns", "route_recovery")
    }
    assert {family: len(items) for family, items in by_family.items()} == {
        "forward_2m": 64,
        "turn_3m": 64,
        "route_turns": 64,
        "route_recovery": 64,
    }

    first_commands = {family: items[0].commands for family, items in by_family.items()}
    assert first_commands["forward_2m"][0].step == 0
    assert first_commands["forward_2m"][0].values == pytest.approx((2.0, 0.0, 0.0))
    assert first_commands["turn_3m"][0].step == 0
    assert first_commands["turn_3m"][0].values == pytest.approx((3.0, 0.0, 1.5707963267948966))
    for family in ("route_turns", "route_recovery"):
        commands = first_commands[family]
        assert [point.step for point in commands] == [0, 1700, 3400]
        assert [point.values[2] for point in commands] == pytest.approx(
            (0.0, 1.5707963267948966, 3.141592653589793)
        )
        coordinates = [(point.values[0], point.values[1]) for point in commands]
        leg_lengths = [
            hypot(right[0] - left[0], right[1] - left[1])
            for left, right in zip(coordinates, coordinates[1:], strict=False)
        ]
        assert leg_lengths == pytest.approx([2.0, 2.0])

    recovery = by_family["route_recovery"][0]
    assert len(recovery.disturbances) == 1
    assert recovery.disturbances[0].start_step == 3000
    assert recovery.disturbances[0].equivalent_delta_v in {0.05, 0.10, 0.15}

    gates = {(gate.family, gate.metric): gate for gate in suite.gates}
    for family in by_family:
        assert gates[(family, "success")].threshold == 0.90
        assert gates[(family, "waypoint_sequence_complete")].threshold == 0.80
        assert gates[(family, "waypoint_final_error")].threshold == 0.05
        assert gates[(family, "waypoint_stop_window_error_rms")].threshold == 0.05
        assert gates[(family, "action_clip_fraction")].threshold == 0.05
        assert gates[(family, "physical_saturation_fraction")].threshold == 0.01
    assert gates[("route_recovery", "recovered")].threshold == 0.85
