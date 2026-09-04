"""Gate evaluation over per-family statistical summaries."""

from __future__ import annotations

import operator
from dataclasses import asdict, dataclass

from .schema import EvaluationStatus, GateSpec


OPS = {
  ">=": operator.ge,
  "<=": operator.le,
  ">": operator.gt,
  "<": operator.lt,
  "==": operator.eq,
}


@dataclass(frozen=True)
class GateResult:
  gate_id: str
  family: str
  metric: str
  statistic: str
  op: str
  threshold: float
  observed: float | None
  passed: bool
  hard: bool
  reason: str = ""

  def to_dict(self) -> dict:
    return asdict(self)


def evaluate_gates(
  gates: tuple[GateSpec, ...],
  family_summary: dict[str, dict[str, dict[str, float]]],
) -> tuple[EvaluationStatus, list[GateResult]]:
  results: list[GateResult] = []
  incomplete = False
  hard_failed = False
  for gate in gates:
    metric_summary = family_summary.get(gate.family, {}).get(gate.metric)
    if metric_summary is None or gate.statistic not in metric_summary:
      incomplete = True
      results.append(
        GateResult(
          gate_id=gate.gate_id,
          family=gate.family,
          metric=gate.metric,
          statistic=gate.statistic,
          op=gate.op,
          threshold=gate.threshold,
          observed=None,
          passed=False,
          hard=gate.hard,
          reason="required metric/statistic unavailable",
        )
      )
      continue
    observed = float(metric_summary[gate.statistic])
    comparator = OPS.get(gate.op)
    if comparator is None:
      raise ValueError(f"Unsupported gate operator: {gate.op}")
    passed = bool(comparator(observed, gate.threshold))
    if gate.hard and not passed:
      hard_failed = True
    results.append(
      GateResult(
        gate_id=gate.gate_id,
        family=gate.family,
        metric=gate.metric,
        statistic=gate.statistic,
        op=gate.op,
        threshold=gate.threshold,
        observed=observed,
        passed=passed,
        hard=gate.hard,
      )
    )
  if incomplete:
    return EvaluationStatus.INCOMPLETE, results
  if hard_failed:
    return EvaluationStatus.FAIL, results
  return EvaluationStatus.PASS, results
