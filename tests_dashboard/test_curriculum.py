import math

from dashboard.curriculum import curriculum_for_run


def test_horizon_curriculum_exposes_promotion_and_failure_gates():
    detail = {
        "run_info": {
            "task": "Ascento-Balance-Flat",
            "episode_horizon_s": 60.0,
            "horizon_stage": 2,
            "horizon_qualified_windows": 4,
            "horizon_failed_windows": 0,
            "horizon_timeout_fraction": 0.941,
            "horizon_stationary_quality_fraction": 0.927,
            "horizon_transition": "held",
        },
        "telemetry": {"iteration": 4833},
    }

    curriculum = curriculum_for_run(detail)

    assert curriculum["kind"] == "horizon"
    assert curriculum["stage"] == 2
    assert [stage["state"] for stage in curriculum["stages"]] == [
        "complete",
        "current",
        "upcoming",
        "upcoming",
    ]
    assert curriculum["promotion"]["qualified_windows"] == 4
    assert curriculum["promotion"]["required_windows"] == 6
    assert curriculum["promotion"]["timeout_threshold"] == 0.90
    assert curriculum["promotion"]["quality_threshold"] == 0.90
    assert curriculum["demotion"]["required_windows"] == 4
    assert curriculum["protected"] is False


def test_top_horizon_is_explicitly_protected():
    curriculum = curriculum_for_run(
        {
            "run_info": {
                "task": "Ascento-Velocity-Flat",
                "episode_horizon_s": 300.0,
                "horizon_stage": 4,
                "horizon_transition": "protected",
            }
        }
    )

    assert curriculum["stage"] == 4
    assert curriculum["protected"] is True
    assert curriculum["transition"] == "protected"


def test_balance_recovery_includes_reset_difficulty_ramp():
    curriculum = curriculum_for_run(
        {
            "run_info": {
                "task": "Ascento-Balance-Recovery-Flat",
                "episode_horizon_s": 20.0,
                "horizon_stage": 1,
                "rollout_steps_per_env": 24,
                "horizon_control_steps": 30000,
            },
            "telemetry": {"iteration": 2500},
        }
    )

    secondary = curriculum["secondary"]
    assert secondary["kind"] == "recovery_difficulty"
    assert secondary["control_steps"] == 30_000
    assert secondary["progress"] == 0.25
    assert math.isclose(secondary["hard_fraction"], 0.15)
    assert math.isclose(secondary["pitch_max_rad"], 0.1125)


def test_locomotion_curriculum_describes_training_sequence():
    curriculum = curriculum_for_run({"run_info": {"task": "Ascento-Locomotion-Flat"}})

    assert curriculum["kind"] == "sequence"
    assert [stage["label"] for stage in curriculum["stages"]] == [
        "Settle",
        "Push",
        "Recover",
        "Target",
        "Stop",
    ]
