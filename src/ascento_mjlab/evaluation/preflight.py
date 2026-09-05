"""Long-run evaluator preflight and determinism checks."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import torch

from .runner import run_scenarios, task_capabilities, task_step_dt
from .scenarios import materialize_suite
from .schema import EpisodeResult, ScenarioSpec, load_suite

DEFAULT_SUITES = {
    "balance": "balance_gate_v1.toml",
    "velocity": "velocity_gate_v1.toml",
    "recovery": "recovery_gate_v1.toml",
}

_ALLOWED_NAN = {
    "recovery_time_s": "recovered",
    "recovery_from_start_s": "recovery_success",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _suite_path(name: str) -> Path:
    return _repo_root() / "benchmarks" / "suites" / DEFAULT_SUITES[name]


def _assert_results_well_formed(results: list[EpisodeResult]) -> None:
    if not results:
        raise RuntimeError("Evaluator returned no episodes")
    for result in results:
        if result.termination_reason in {"", "unknown", "done"}:
            raise RuntimeError(
                f"Scenario {result.scenario_id} has ambiguous termination reason "
                f"{result.termination_reason!r}"
            )
        if result.episode_steps <= 0:
            raise RuntimeError(f"Scenario {result.scenario_id} has no valid steps")
        for name, value in result.metrics.items():
            if math.isfinite(value):
                continue
            gate_metric = _ALLOWED_NAN.get(name)
            if gate_metric is not None and not bool(result.metrics.get(gate_metric, 0.0)):
                continue
            raise RuntimeError(
                f"Scenario {result.scenario_id} produced non-finite metric {name}={value}"
            )


def _assert_repeatable(
    first: list[EpisodeResult], second: list[EpisodeResult], *, atol: float = 1.0e-6
) -> None:
    if len(first) != len(second):
        raise RuntimeError("Deterministic repeat changed episode count")
    for left, right in zip(first, second, strict=True):
        if (
            left.scenario_id != right.scenario_id
            or left.family != right.family
            or left.success != right.success
            or left.termination_reason != right.termination_reason
            or left.episode_steps != right.episode_steps
        ):
            raise RuntimeError(
                f"Deterministic repeat changed discrete result for {left.scenario_id}"
            )
        if set(left.metrics) != set(right.metrics):
            raise RuntimeError(f"Deterministic repeat changed metric schema for {left.scenario_id}")
        for name in left.metrics:
            a, b = left.metrics[name], right.metrics[name]
            if math.isnan(a) and math.isnan(b):
                continue
            if not math.isclose(a, b, rel_tol=0.0, abs_tol=atol):
                raise RuntimeError(
                    f"Deterministic repeat changed {left.scenario_id}.{name}: {a} vs {b}"
                )


def _mixed_horizon_subset(scenarios: list[ScenarioSpec]) -> list[ScenarioSpec]:
    if not scenarios:
        return []
    source = scenarios[: min(4, len(scenarios))]
    longest = max(s.horizon_steps for s in source)
    fractions = (0.20, 0.40, 0.70, 1.0)
    return [
        replace(
            scenario,
            scenario_id=f"{scenario.scenario_id}__mixed_{index}",
            horizon_steps=max(5, round(longest * fractions[index])),
        )
        for index, scenario in enumerate(source)
    ]


def _assert_mixed_horizon(
    task: str,
    checkpoint: Path,
    scenarios: list[ScenarioSpec],
    *,
    device: str,
) -> dict[str, int]:
    mixed = _mixed_horizon_subset(scenarios)
    results, _ = run_scenarios(
        task,
        checkpoint,
        mixed,
        batch_size=max(1, len(mixed)),
        device=device,
        deterministic=True,
    )
    _assert_results_well_formed(results)
    requested = {scenario.scenario_id: scenario.horizon_steps for scenario in mixed}
    for result in results:
        horizon = requested[result.scenario_id]
        if result.episode_steps > horizon:
            raise RuntimeError(
                f"Mixed-horizon scenario {result.scenario_id} ran past {horizon} steps"
            )
        if result.success and result.episode_steps != horizon:
            raise RuntimeError(
                f"Successful mixed-horizon scenario {result.scenario_id} stopped early"
            )
    return requested


def preflight_suite(
    suite_name: str,
    checkpoint: Path,
    *,
    device: str,
    max_scenarios: int,
    batch_size: int,
) -> dict:
    suite = load_suite(_suite_path(suite_name))
    missing = sorted(set(suite.required_capabilities) - task_capabilities(suite.task))
    if missing:
        raise RuntimeError(f"{suite_name} suite missing capabilities: {', '.join(missing)}")
    scenarios = materialize_suite(suite, task_step_dt(suite.task))
    scenarios = scenarios[:max_scenarios]
    if not scenarios:
        raise RuntimeError(f"{suite_name} suite materialized no scenarios")

    first, first_runtime = run_scenarios(
        suite.task,
        checkpoint,
        scenarios,
        batch_size=batch_size,
        device=device,
        deterministic=True,
    )
    _assert_results_well_formed(first)
    second, second_runtime = run_scenarios(
        suite.task,
        checkpoint,
        scenarios,
        batch_size=batch_size,
        device=device,
        deterministic=True,
    )
    _assert_results_well_formed(second)
    _assert_repeatable(first, second)
    mixed = _assert_mixed_horizon(suite.task, checkpoint, scenarios, device=device)
    return {
        "suite": suite.suite_id,
        "task": suite.task,
        "checkpoint": str(checkpoint),
        "scenario_count": len(scenarios),
        "repeatable": True,
        "mixed_horizon": mixed,
        "first_runtime": first_runtime,
        "second_runtime": second_runtime,
        "termination_reasons": sorted({result.termination_reason for result in first}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--balance-checkpoint", type=Path, required=True)
    parser.add_argument("--velocity-checkpoint", type=Path, required=True)
    parser.add_argument("--recovery-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-scenarios", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("preflight/evaluator.json"))
    args = parser.parse_args()
    if args.max_scenarios < 1 or args.batch_size < 1:
        parser.error("--max-scenarios and --batch-size must be positive")

    checkpoints = {
        "balance": args.balance_checkpoint,
        "velocity": args.velocity_checkpoint,
        "recovery": args.recovery_checkpoint,
    }
    for name, checkpoint in checkpoints.items():
        if not checkpoint.is_file():
            parser.error(f"{name} checkpoint does not exist: {checkpoint}")

    device = (
        "cuda:0"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    payload = {
        name: preflight_suite(
            name,
            checkpoint,
            device=device,
            max_scenarios=args.max_scenarios,
            batch_size=args.batch_size,
        )
        for name, checkpoint in checkpoints.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Preflight evidence: {args.output}")


if __name__ == "__main__":
    main()
