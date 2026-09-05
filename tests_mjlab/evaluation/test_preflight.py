from dataclasses import replace

import pytest

from ascento_mjlab.evaluation.preflight import (
  _assert_repeatable,
  _assert_results_well_formed,
  _mixed_horizon_subset,
)
from ascento_mjlab.evaluation.schema import EpisodeResult, ScenarioSpec


def _result(**overrides):
  result = EpisodeResult(
    scenario_id="scenario-0",
    family="family",
    success=True,
    termination_reason="horizon",
    episode_steps=20,
    metrics={"tilt_rms": 0.1, "recovery_success": 0.0, "recovery_from_start_s": float("nan")},
  )
  for key, value in overrides.items():
    setattr(result, key, value)
  return result


def test_well_formed_allows_only_inapplicable_recovery_nan():
  _assert_results_well_formed([_result()])

  bad = _result()
  bad.metrics["tilt_rms"] = float("nan")
  with pytest.raises(RuntimeError, match="non-finite metric tilt_rms"):
    _assert_results_well_formed([bad])


def test_repeatability_compares_discrete_and_numeric_results():
  first = _result()
  second = _result()
  _assert_repeatable([first], [second])

  changed = _result()
  changed.metrics["tilt_rms"] += 1.0e-3
  with pytest.raises(RuntimeError, match="Deterministic repeat changed"):
    _assert_repeatable([first], [changed])


def test_mixed_horizon_subset_contains_distinct_lengths():
  base = ScenarioSpec(
    scenario_id="s",
    family="f",
    task="Ascento-Balance-Flat",
    horizon_steps=100,
    reset={},
  )
  scenarios = [replace(base, scenario_id=f"s-{index}") for index in range(4)]
  mixed = _mixed_horizon_subset(scenarios)

  assert [scenario.horizon_steps for scenario in mixed] == [20, 40, 70, 100]
  assert len({scenario.scenario_id for scenario in mixed}) == 4
