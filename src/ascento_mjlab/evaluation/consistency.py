"""Cross-checks that make telemetry contradictions invalidate an evaluation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .schema import EpisodeResult, ScenarioSpec


@dataclass(frozen=True)
class ConsistencyCheck:
  check_id: str
  passed: bool
  detail: str

  def to_dict(self) -> dict:
    return asdict(self)


def check_episode(
  result: EpisodeResult,
  scenario: ScenarioSpec,
  *,
  step_dt: float,
  physical_effort_limit: float = 40.0,
  atol: float = 1.0e-5,
) -> list[ConsistencyCheck]:
  metrics = result.metrics
  checks: list[ConsistencyCheck] = []

  expected_time = result.episode_steps * step_dt
  measured_time = metrics.get("episode_time_s", expected_time)
  checks.append(
    ConsistencyCheck(
      "episode_time_matches_steps",
      math.isclose(measured_time, expected_time, rel_tol=0.0, abs_tol=max(atol, step_dt * 0.51)),
      f"measured={measured_time:.9g}, expected={expected_time:.9g}",
    )
  )
  checks.append(
    ConsistencyCheck(
      "episode_does_not_exceed_horizon",
      result.episode_steps <= scenario.horizon_steps,
      f"steps={result.episode_steps}, horizon={scenario.horizon_steps}",
    )
  )

  path_length = metrics.get("path_length")
  displacement = metrics.get("net_displacement")
  if path_length is not None and displacement is not None:
    checks.append(
      ConsistencyCheck(
        "displacement_le_path_length",
        displacement <= path_length + 1.0e-4,
        f"displacement={displacement:.9g}, path_length={path_length:.9g}",
      )
    )

  mean_abs = metrics.get("effort_mean_abs")
  rms = metrics.get("effort_rms")
  peak = metrics.get("effort_max_abs")
  if mean_abs is not None and rms is not None:
    checks.append(
      ConsistencyCheck(
        "effort_mean_abs_le_rms",
        mean_abs <= rms + 1.0e-5,
        f"mean_abs={mean_abs:.9g}, rms={rms:.9g}",
      )
    )
  if rms is not None and peak is not None:
    checks.append(
      ConsistencyCheck(
        "effort_rms_le_peak",
        rms <= peak + 1.0e-5,
        f"rms={rms:.9g}, peak={peak:.9g}",
      )
    )

  request_peak = metrics.get("physical_request_max_abs")
  if request_peak is not None:
    checks.append(
      ConsistencyCheck(
        "physical_request_within_limit",
        request_peak <= physical_effort_limit + 1.0e-4,
        f"request_peak={request_peak:.9g}, limit={physical_effort_limit:.9g}",
      )
    )

  for key in (
    "both_supported_fraction",
    "airborne_fraction",
    "action_clip_fraction",
    "physical_saturation_fraction",
  ):
    value = metrics.get(key)
    if value is not None:
      checks.append(
        ConsistencyCheck(
          f"{key}_is_fraction",
          -atol <= value <= 1.0 + atol,
          f"{key}={value:.9g}",
        )
      )

  if result.success:
    checks.append(
      ConsistencyCheck(
        "success_reaches_horizon",
        result.episode_steps == scenario.horizon_steps,
        f"success steps={result.episode_steps}, horizon={scenario.horizon_steps}",
      )
    )
  return checks


def check_collection(
  results: list[EpisodeResult],
  scenarios: list[ScenarioSpec],
  *,
  step_dt: float,
) -> tuple[bool, list[ConsistencyCheck]]:
  checks: list[ConsistencyCheck] = []
  result_ids = [result.scenario_id for result in results]
  scenario_ids = [scenario.scenario_id for scenario in scenarios]
  checks.append(
    ConsistencyCheck(
      "scenario_count_matches_results",
      len(results) == len(scenarios),
      f"results={len(results)}, scenarios={len(scenarios)}",
    )
  )
  checks.append(
    ConsistencyCheck(
      "scenario_ids_unique",
      len(set(scenario_ids)) == len(scenario_ids),
      f"unique={len(set(scenario_ids))}, total={len(scenario_ids)}",
    )
  )
  checks.append(
    ConsistencyCheck(
      "result_ids_match_scenarios",
      set(result_ids) == set(scenario_ids),
      f"missing={len(set(scenario_ids)-set(result_ids))}, extra={len(set(result_ids)-set(scenario_ids))}",
    )
  )

  scenario_by_id = {scenario.scenario_id: scenario for scenario in scenarios}
  for result in results:
    scenario = scenario_by_id.get(result.scenario_id)
    if scenario is None:
      continue
    checks.extend(check_episode(result, scenario, step_dt=step_dt))
  return all(check.passed for check in checks), checks
