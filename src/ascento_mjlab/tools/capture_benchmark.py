"""Capture exact seeded benchmark scenarios into NPZ and optional Rerun RRD.

Example:
  python -m ascento_mjlab.tools.capture_benchmark \\
    --suite goal_settle_v5_comprehensive_v1 --checkpoint /path/model_16300.pt \\
    --family precision_short --index 0 --index 1 --rrd

Captures use the evaluation runner, not the unrelated stochastic play sequence.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ascento_mjlab.evaluation.cli import resolve_suite_path
from ascento_mjlab.evaluation.replay_capture import BenchmarkReplayRecorder
from ascento_mjlab.evaluation.runner import run_scenarios, task_step_dt
from ascento_mjlab.evaluation.scenarios import materialize_suite
from ascento_mjlab.evaluation.schema import ScenarioSpec, load_suite


def select_scenarios(
    scenarios: list[ScenarioSpec],
    *,
    scenario_ids: list[str] | None = None,
    families: list[str] | None = None,
    indices: list[int] | None = None,
) -> list[ScenarioSpec]:
    """Never silently fall back to an unrelated suite sample."""
    scenario_ids = scenario_ids or []
    families = families or []
    indices = [0] if indices is None else indices
    if not scenario_ids and not families:
        raise ValueError("Specify --scenario-id and/or --family")
    by_id = {s.scenario_id: s for s in scenarios}
    unknown = sorted(set(scenario_ids) - by_id.keys())
    if unknown:
        raise ValueError(f"Unknown scenario IDs: {unknown}")
    known_families = {s.family for s in scenarios}
    unknown_families = sorted(set(families) - known_families)
    if unknown_families:
        raise ValueError(f"Unknown scenario families: {unknown_families}")
    if any(index < 0 for index in indices):
        raise ValueError("Scenario indices must be nonnegative")
    selected_ids = set(scenario_ids)
    for family in families:
        for index in indices:
            matches = [
                s.scenario_id
                for s in scenarios
                if s.family == family and s.scenario_id.endswith(f"/{index:06d}")
            ]
            if not matches:
                raise ValueError(f"No such scenario: {family}/{index:06d}")
            selected_ids.update(matches)
    return [s for s in scenarios if s.scenario_id in selected_ids]


def capture_benchmark(
    *,
    suite_path: Path,
    checkpoint: Path,
    output_dir: Path,
    scenario_ids: list[str] | None = None,
    families: list[str] | None = None,
    indices: list[int] | None = None,
    device: str = "cuda:0",
    batch_size: int = 1,
    export_rrd: bool = False,
    obstacle_mode: int = 0,
) -> dict:
    if obstacle_mode not in (0, 1):
        raise ValueError("obstacle_mode must be 0 or 1")
    # Mirror evaluation.cli.evaluate: checkpoint topology is conditioned on
    # this command mode, even when the suite itself uses no obstacle case.
    os.environ["ASCENTO_GENERALIST_OBSTACLE_MODE"] = str(obstacle_mode)
    suite_path = Path(suite_path).resolve()
    checkpoint = Path(checkpoint).resolve()
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    suite = load_suite(suite_path)
    all_scenarios = materialize_suite(suite, task_step_dt(suite.task))
    selected = select_scenarios(
        all_scenarios,
        scenario_ids=scenario_ids,
        families=families,
        indices=indices,
    )
    recorder = BenchmarkReplayRecorder(
        {s.scenario_id for s in selected},
        output_dir,
        checkpoint=checkpoint,
        suite_path=suite_path,
        suite_sha256=suite.sha256(),
    )
    results, runtime = run_scenarios(
        suite.task,
        checkpoint,
        selected,
        batch_size=batch_size,
        device=device,
        deterministic=suite.policy_mode == "deterministic",
        replay_recorder=recorder,
    )
    npz_paths = recorder.save()
    rrd_paths: list[str] = []
    if export_rrd:
        from ascento_mjlab.tools.export_rerun import export_rerun

        for path in npz_paths:
            target = path.with_suffix(".rrd")
            export_rerun(path, target)
            rrd_paths.append(str(target))
    outcome = {
        "suite_id": suite.suite_id,
        "suite_sha256": suite.sha256(),
        "checkpoint": str(checkpoint),
        "runtime": {
            "checkpoint_sha256": runtime.get("checkpoint_sha256"),
            "step_dt": runtime.get("step_dt"),
            "batches": runtime.get("batches"),
            "checkpoint_task_topology_sha256": (
                (runtime.get("checkpoint_task_contract") or {}).get("topology_sha256")
            ),
            "checkpoint_plant_contract_id": (
                (runtime.get("checkpoint_plant_contract") or {}).get("id")
            ),
        },
        "scenarios": [s.scenario_id for s in selected],
        "npz": [str(path) for path in npz_paths],
        "rrd": rrd_paths,
        "results": [
            {
                "scenario_id": r.scenario_id,
                "success": r.success,
                "termination_reason": r.termination_reason,
                "episode_steps": r.episode_steps,
                "target_arrived": r.metrics.get("target_arrived"),
                "waypoint_final_stopped": r.metrics.get("waypoint_final_stopped"),
                "waypoint_max_consecutive_dwell_s": r.metrics.get(
                    "waypoint_max_consecutive_dwell_s"
                ),
                "stop_condition_pass_fraction": {
                    key.removeprefix("waypoint_stop_").removesuffix("_pass_fraction"): value
                    for key, value in r.metrics.items()
                    if key.startswith("waypoint_stop_") and key.endswith("_pass_fraction")
                },
            }
            for r in results
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark_capture_manifest.json").write_text(
        json.dumps(outcome, indent=2, sort_keys=True) + "\n"
    )
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, help="Benchmark suite ID or TOML path")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scenario-id", action="append", default=[])
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--index", type=int, action="append")
    parser.add_argument("--output-dir", type=Path, default=Path("captures/benchmark-rerun"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--rrd", action="store_true", help="Export Rerun recordings")
    parser.add_argument("--obstacle-mode", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    try:
        outcome = capture_benchmark(
            suite_path=resolve_suite_path(args.suite),
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            scenario_ids=args.scenario_id,
            families=args.family,
            indices=args.index,
            device=args.device,
            batch_size=args.batch_size,
            export_rrd=args.rrd,
            obstacle_mode=args.obstacle_mode,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(outcome, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
