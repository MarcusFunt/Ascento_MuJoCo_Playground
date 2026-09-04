from ascento_mjlab.evaluation.gates import evaluate_gates
from ascento_mjlab.evaluation.schema import EvaluationStatus, GateSpec


def test_gate_pass_and_fail():
  gates = (
    GateSpec("reliable", "nominal", "success", "wilson_lower", ">=", 0.99),
    GateSpec("tilt", "nominal", "max_tilt", "p95", "<=", 0.15),
  )
  summary = {
    "nominal": {
      "success": {"wilson_lower": 0.997},
      "max_tilt": {"p95": 0.10},
    }
  }
  status, results = evaluate_gates(gates, summary)
  assert status == EvaluationStatus.PASS
  assert all(result.passed for result in results)

  summary["nominal"]["max_tilt"]["p95"] = 0.20
  status, _ = evaluate_gates(gates, summary)
  assert status == EvaluationStatus.FAIL


def test_missing_required_metric_is_incomplete():
  gates = (GateSpec("missing", "jump", "landing_speed", "p95", "<=", 1.0),)
  status, results = evaluate_gates(gates, {"jump": {}})
  assert status == EvaluationStatus.INCOMPLETE
  assert results[0].observed is None
