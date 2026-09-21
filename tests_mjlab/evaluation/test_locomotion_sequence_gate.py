from pathlib import Path

from ascento_mjlab.evaluation.report import summarize_results
from ascento_mjlab.evaluation.schema import EpisodeResult, load_suite


def test_target_arrived_is_aggregated_as_a_binary_gate_metric():
    summary = summarize_results(
        [
            EpisodeResult("sequence/0", "sequence", True, "horizon", 100, {"target_arrived": 1.0}),
            EpisodeResult("sequence/1", "sequence", True, "horizon", 100, {"target_arrived": 0.0}),
        ]
    )

    assert summary["sequence"]["target_arrived"]["success_rate"] == 0.5


def test_locomotion_sequence_gate_declares_target_arrival_and_quiet_stop_guards():
    suite = load_suite(Path("benchmarks/suites/locomotion_sequence_gate_v1.toml"))

    assert suite.task == "Ascento-Locomotion-Flat"
    gates = {gate.gate_id for gate in suite.gates}
    assert {
        "sequence_survival_lcb",
        "sequence_recovery_lcb",
        "sequence_target_arrival_lcb",
        "sequence_final_target_error",
        "sequence_post_target_speed",
        "sequence_post_target_heading",
        "sequence_stationary_action_second_difference",
    } <= gates
    thresholds = {gate.gate_id: gate.threshold for gate in suite.gates}
    # Calibrated from the frozen 79,999 actor transfer.  The continuation at
    # 299 exceeds both values on the same scenarios.
    assert thresholds["sequence_stationary_action_second_difference"] == 0.18
    assert thresholds["sequence_stationary_body_rocking"] == 0.03
