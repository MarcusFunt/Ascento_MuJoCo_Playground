"""Collect immutable teacher labels from a passing frozen development suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from ascento_mjlab.evaluation.cli import _checkpoint_contract_preflight
from ascento_mjlab.evaluation.consistency import check_collection
from ascento_mjlab.evaluation.gates import evaluate_gates
from ascento_mjlab.evaluation.report import summarize_results
from ascento_mjlab.evaluation.runner import run_scenarios, task_capabilities, task_step_dt
from ascento_mjlab.evaluation.scenarios import materialize_suite
from ascento_mjlab.evaluation.schema import EvaluationStatus, load_suite, scenarios_sha256
from ascento_mjlab.retention_replay import (
    AnchorReplayRecorder,
    actor_normalizer_sha256,
    canonical_sha256,
    save_anchor_replay,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(
    *,
    checkpoint: Path,
    suite_path: Path,
    output: Path,
    batch_size: int,
    device: str,
    stride_steps: int,
) -> dict[str, Any]:
    suite = load_suite(suite_path)
    if suite.suite_id != "roadrunner_generalist_sequence_gate_v1":
        raise ValueError("anchor replay collection requires the frozen development suite")
    if suite.policy_mode != "deterministic":
        raise ValueError("anchor replay teacher labels require deterministic policy mode")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    preflight = _checkpoint_contract_preflight(checkpoint, suite.task)
    if preflight["status"] != "passed":
        raise ValueError(f"source checkpoint contract preflight failed: {preflight['reason']}")

    step_dt = task_step_dt(suite.task)
    scenarios = materialize_suite(suite, step_dt)
    missing = sorted(set(suite.required_capabilities) - task_capabilities(suite.task))
    if missing:
        raise ValueError(f"development suite requires unavailable capabilities: {missing}")
    recorder = AnchorReplayRecorder(stride_steps=stride_steps)
    results, runtime = run_scenarios(
        suite.task,
        checkpoint,
        scenarios,
        batch_size=batch_size,
        device=device,
        deterministic=True,
        replay_recorder=recorder,
    )
    summary = summarize_results(results, bootstrap_seed=suite.root_seed)
    gate_status, gates = evaluate_gates(suite.gates, summary)
    consistent, consistency_checks = check_collection(results, scenarios, step_dt=step_dt)
    if gate_status != EvaluationStatus.PASS or not consistent:
        failed_gates = [gate.to_dict() for gate in gates if gate.hard and not gate.passed]
        failed_checks = [check.to_dict() for check in consistency_checks if not check.passed]
        raise RuntimeError(
            "source checkpoint did not pass the frozen development evaluation; "
            f"gate_status={gate_status.value}, failed_gates={failed_gates}, "
            f"failed_consistency_checks={failed_checks}"
        )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    infos = payload.get("infos")
    actor_state = payload.get("actor_state_dict")
    if not isinstance(infos, dict) or not isinstance(actor_state, dict):
        raise ValueError("source checkpoint lacks signed actor and task provenance")
    task_contract = infos.get("task_contract")
    if not isinstance(task_contract, dict):
        raise ValueError("source checkpoint lacks a task contract")
    actor_observation_contract = (
        task_contract.get("topology", {}).get("observations", {}).get("actor")
    )
    if actor_observation_contract is None:
        raise ValueError("source checkpoint lacks the actor-observation contract")

    metadata = {
        "source_checkpoint_sha256": _sha256(checkpoint),
        "source_actor_normalizer_sha256": actor_normalizer_sha256(actor_state),
        "source_task_contract_sha256": canonical_sha256(task_contract),
        "source_action_contract_sha256": canonical_sha256(infos["action_contract"]),
        "source_plant_contract_sha256": canonical_sha256(infos["plant_contract"]),
        "actor_observation_contract_sha256": canonical_sha256(actor_observation_contract),
        "suite_id": suite.suite_id,
        "suite_definition_sha256": _sha256(suite_path),
        "resolved_scenario_sha256": scenarios_sha256(scenarios),
        "evaluation_status": gate_status.value,
        "evaluation_consistency_passed": consistent,
        "evaluation_checkpoint_sha256": runtime["checkpoint_sha256"],
        "evaluation_target_arrivals": sum(
            bool(item.metrics.get("target_arrived", False)) for item in results
        ),
        "evaluation_scenario_count": len(results),
        "collection_stride_steps": stride_steps,
        "teacher_action_semantics": "deterministic_gaussian_mean_raw_action",
        "cohort_semantics": {
            "0": "precision_anchor_before_first_disturbance",
            "1": "recovery_retarget_anchor_from_first_disturbance_onward",
        },
    }
    manifest = save_anchor_replay(output, recorder.arrays(), metadata)
    return {
        "dataset": str(output.resolve()),
        "manifest": str(output.with_suffix(".json").resolve()),
        "evaluation_status": gate_status.value,
        "evaluation_scenario_count": len(results),
        "dataset_sample_count": manifest["sample_count"],
        "samples_by_cohort": manifest["samples_by_cohort"],
        "dataset_sha256": manifest["dataset_sha256"],
        "resolved_scenario_sha256": manifest["resolved_scenario_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stride-steps", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError("refusing to overwrite an existing immutable replay corpus")
    result = collect(
        checkpoint=args.checkpoint,
        suite_path=args.suite,
        output=args.output,
        batch_size=args.batch_size,
        device=args.device,
        stride_steps=args.stride_steps,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
