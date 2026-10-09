import math

from dashboard.curriculum import curriculum_for_run


def test_horizon_curriculum_exposes_promotion_and_failure_gates():
    detail = {
        "run_info": {
            "task": "Ascento-Balance-Quiet-Flat",
            "episode_horizon_s": 60.0,
            "horizon_stage": 2,
            "horizon_qualified_windows": 4,
            "horizon_failed_windows": 0,
            "horizon_timeout_fraction": 0.941,
            "horizon_stationary_quality_fraction": 0.927,
            "horizon_quality_failure_counts": {
                "action_rate_rms": 81,
                "action_second_difference_rms": 230,
                "insufficient_stationary_samples": 12,
            },
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
    assert curriculum["promotion"]["quality_failure_counts"] == {
        "action_rate_rms": 81,
        "action_second_difference_rms": 230,
        "insufficient_stationary_samples": 12,
    }
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


def test_generalist_curriculum_uses_episode_and_attempt_denominators_separately():
    curriculum = curriculum_for_run(
        {
            "run_info": {"task": "Ascento-Generalist-Locomotion-Flat"},
            "telemetry": {
                "iteration": 1200,
                "wall_time": 1234.5,
                "metrics": {
                    "Episode/generalist_curriculum_goal_mix_stage": 1,
                    "Episode/generalist_curriculum_progress": 0.5,
                    "Episode/generalist_curriculum_long_goal_fraction": 0.25,
                    "Episode/generalist_curriculum_stage_gate_recovery_lcb": 0.86,
                    "Episode/generalist_curriculum_window_gate_recovery_successes": 59,
                    "Episode/generalist_curriculum_window_gate_episodes": 64,
                    "Episode/generalist_curriculum_stage_short_attempt_arrival_lcb": 0.52,
                    "Episode/generalist_curriculum_window_short_arrivals": 40,
                    "Episode/generalist_curriculum_window_short_attempts": 80,
                    "Episode/generalist_curriculum_stage_long_attempt_arrival_lcb": 0.2,
                    "Episode/generalist_curriculum_window_long_arrivals": 8,
                    "Episode/generalist_curriculum_window_long_attempts": 64,
                    "Episode/generalist_cohort_generalist_navigation_episode_count": 100,
                    "Episode/generalist_cohort_generalist_navigation_arrival_count": 40,
                    "Episode/generalist_cohort_generalist_navigation_fall_count": 5,
                    "Episode/generalist_attempt_generalist_navigation_short_target_attempt_count": 50,
                    "Episode/generalist_attempt_generalist_navigation_short_arrival_completed_count": 20,
                    "Episode/generalist_attempt_generalist_navigation_short_settled_stop_completed_count": 12,
                    "Episode/generalist_attempt_generalist_navigation_short_final_target_error_p95_m": 0.07,
                },
            },
        }
    )

    assert curriculum["kind"] == "generalist"
    assert curriculum["stage"] == 1
    assert curriculum["goal_fractions"]["long"] == 0.25
    assert curriculum["gates"][0]["samples"] == 64
    assert curriculum["gates"][0]["successes"] == 59
    assert curriculum["gates"][0]["state"] == "pass"
    navigation = next(row for row in curriculum["cohorts"] if row["id"] == "generalist_navigation")
    assert navigation["episode_count"] == 100
    assert navigation["arrival_rate"] == 0.4
    short = curriculum["attempt_bands"][0]
    assert short["attempts"] == 50
    assert short["arrivals"] == 20
    assert short["arrival_rate"] == 0.4
    assert short["settled_stop_rate"] == 0.24
    assert short["p95_final_target_error_m"] == 0.07


def test_generalist_missing_telemetry_is_waiting_not_zero():
    curriculum = curriculum_for_run(
        {"run_info": {"task": "Ascento-Generalist-Locomotion-Flat"}}
    )

    assert curriculum["kind"] == "generalist"
    assert curriculum["stage"] is None
    assert curriculum["has_metrics"] is False
    assert all(gate["state"] == "waiting" for gate in curriculum["gates"])
    assert curriculum["cohorts"][0]["episode_count"] is None
