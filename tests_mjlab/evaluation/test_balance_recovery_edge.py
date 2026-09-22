from pathlib import Path

from ascento_mjlab.evaluation.scenarios import materialize_suite
from ascento_mjlab.evaluation.schema import load_suite


def test_balance_recovery_edge_suite_targets_measured_failure_region():
    suite = load_suite(Path("benchmarks/suites/balance_recovery_edge_v1.toml"))
    assert suite.task == "Ascento-Balance-Recovery-Flat"
    assert sum(family.count for family in suite.families) == 256
    scenarios = materialize_suite(suite, 0.01)
    assert all(0.08 <= item.reset["pitch"] <= 0.15 for item in scenarios)
    assert max(abs(item.reset["wx"]) for item in scenarios) > 0.45
    assert max(abs(item.reset["wy"]) for item in scenarios) > 0.45
    gates = {gate.gate_id: gate.threshold for gate in suite.gates}
    assert gates["edge_survival"] == 0.98
    assert gates["edge_p95_max_tilt"] == 0.25
