from pathlib import Path

from ascento_mjlab.evaluation.gates import evaluate_gates
from ascento_mjlab.evaluation.schema import EvaluationStatus, load_suite


def _summary(**metrics: float):
    return {
        "nominal": {
            name: ({"success_rate": value} if name == "success" else {"p95": value, "p05": value})
            for name, value in metrics.items()
        }
    }


def test_quality_regression_separates_79999_from_long_horizon_buzzing():
    """Lock thresholds to the frozen 256-scenario calibration evidence.

    The values are the measured p95s from ``balance_quality_measure_v1``:
    model_79999 is the quiet behavioral reference and model_best_long_horizon
    is the known A-B-A-B / 50-Hz negative fixture.
    """
    suite = load_suite(Path("benchmarks/suites/balance_quality_regression_v1.toml"))
    assert suite.suite_id == "balance_quality_regression_v1"

    reference = _summary(
        stationary_tilt_rms=0.0153,
        stationary_planar_speed_rms=0.0054,
        stationary_heading_error_rms=0.0062,
        stationary_effort_rms=2.27,
        stationary_action_rate_rms=0.00036,
        stationary_action_second_difference_rms=0.00020,
        stationary_high_frequency_action_power_ratio=0.0357,
        stationary_nyquist_action_power_ratio=0.0000051,
        stationary_body_rocking_rms=0.0076,
        stationary_roll_pitch_reversal_rate_hz=0.0,
        settling_time_s=1.12,
        post_settle_angular_velocity_rms=0.0076,
        stationary_both_supported_fraction=1.0,
        stationary_net_displacement=0.0242,
    )
    pathological = _summary(
        stationary_tilt_rms=0.0066,
        stationary_planar_speed_rms=0.0126,
        stationary_heading_error_rms=0.0161,
        stationary_effort_rms=3.53,
        stationary_action_rate_rms=0.0675,
        stationary_action_second_difference_rms=0.1351,
        stationary_high_frequency_action_power_ratio=0.5153,
        stationary_nyquist_action_power_ratio=0.1122,
        stationary_body_rocking_rms=0.0069,
        stationary_roll_pitch_reversal_rate_hz=5.79,
        settling_time_s=1.08,
        post_settle_angular_velocity_rms=0.0069,
        stationary_both_supported_fraction=1.0,
        stationary_net_displacement=0.0568,
    )

    reference_status, reference_gates = evaluate_gates(suite.gates, reference)
    pathology_status, pathology_gates = evaluate_gates(suite.gates, pathological)

    assert reference_status == EvaluationStatus.PASS
    assert all(gate.passed for gate in reference_gates)
    assert pathology_status == EvaluationStatus.FAIL
    failed = {gate.gate_id for gate in pathology_gates if not gate.passed}
    assert {
        "nominal_stationary_action_second_difference",
        "nominal_stationary_high_frequency_action_power",
        "nominal_stationary_nyquist_action_power",
        "nominal_stationary_roll_pitch_reversals",
    } <= failed
