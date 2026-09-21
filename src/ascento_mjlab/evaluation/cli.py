"""Command-line entry point for versioned quantitative evaluation."""

from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
from mjlab.tasks.registry import load_env_cfg

from ascento_mjlab.checkpoint_contract import require_current_checkpoint_contracts
from ascento_mjlab.control_contract import action_contracts_compatible, current_action_contract
from ascento_mjlab.plant_contract import current_plant_contract, plant_contracts_compatible
from ascento_mjlab.task_contract import (
    classify_task_contract_compatibility,
    current_task_contract_for_task,
)

from .consistency import check_collection
from .gates import evaluate_gates
from .report import render_html, select_worst_scenarios, summarize_results
from .runner import (
    checkpoint_sha256,
    physics_timestep,
    run_scenarios,
    task_capabilities,
    task_step_dt,
)
from .scenarios import materialize_suite
from .schema import EvaluationStatus, load_suite, scenarios_sha256
from .store import (
    write_json,
    write_resolved_scenarios,
    write_results_database,
    write_suite_snapshot,
)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [here, *here.parents]
    for parent in candidates:
        if (parent / "pyproject.toml").is_file() and (parent / ".git").exists():
            return parent
    # Docker images intentionally omit .git but retain the project root.
    for parent in candidates:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def resolve_suite_path(value: str, repo_root: Path | None = None) -> Path:
    direct = Path(value)
    if direct.is_file():
        return direct
    root = repo_root or _repo_root()
    candidates = [
        root / "benchmarks" / "suites" / value,
        root / "benchmarks" / "suites" / f"{value}.toml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Evaluation suite {value!r} not found; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def _git(args: list[str], cwd: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip()


def _package_versions() -> dict[str, str]:
    names = ["mjlab", "mujoco", "mujoco-warp", "warp-lang", "torch", "rsl-rl"]
    output = {}
    for name in names:
        try:
            output[name] = version(name)
        except PackageNotFoundError:
            output[name] = "unknown"
    return output


def _device(value: str) -> str:
    if value != "auto":
        return value
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _make_output_dir(base: Path, suite_id: str, checkpoint: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = base / f"{stamp}_{suite_id}_{checkpoint.stem}"
    suffix = 1
    candidate = path
    while candidate.exists():
        candidate = Path(f"{path}_{suffix}")
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _manifest(
    *,
    suite,
    suite_path: Path,
    scenarios,
    checkpoint: Path,
    device: str,
    batch_size: int,
    step_dt: float,
    repo_root: Path,
) -> dict[str, Any]:
    dirty = _git(["status", "--porcelain"], repo_root)
    repository_commit = _git(["rev-parse", "HEAD"], repo_root)
    repository_branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    if repository_commit == "unknown":
        repository_commit = os.environ.get("ASCENTO_REPOSITORY_COMMIT", "unknown")
    if repository_branch == "unknown":
        repository_branch = os.environ.get("ASCENTO_REPOSITORY_BRANCH", "unknown")
    env_cfg = load_env_cfg(suite.task, play=False)
    timestep = physics_timestep(env_cfg)
    decimation = int(env_cfg.decimation)
    return {
        "evaluation_schema_version": suite.schema_version,
        "suite_id": suite.suite_id,
        "suite_sha256": suite.sha256(),
        "suite_path": str(suite_path),
        "resolved_scenarios_sha256": scenarios_sha256(scenarios),
        "scenario_count": len(scenarios),
        "task": suite.task,
        "policy_mode": suite.policy_mode,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "repository_commit": repository_commit,
        "repository_branch": repository_branch,
        "repository_dirty": None if dirty == "unknown" else bool(dirty),
        "device": device,
        "gpu": torch.cuda.get_device_name(0)
        if device.startswith("cuda") and torch.cuda.is_available()
        else None,
        "batch_size": batch_size,
        "step_dt": step_dt,
        "physics_timestep": timestep,
        "decimation": decimation,
        "plant_contract": current_plant_contract(),
        "action_contract": current_action_contract(),
        "task_contract": current_task_contract_for_task(suite.task),
        "packages": _package_versions(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
    }


def _checkpoint_contract_preflight(checkpoint: Path, task: str) -> dict[str, Any]:
    """Read and validate checkpoint contracts before allocating a simulator world."""
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except (EOFError, OSError, pickle.UnpicklingError, RuntimeError, ValueError) as error:
        return {
            "status": "invalid",
            "reason": f"cannot read checkpoint contracts: {error}",
            "task_contract": None,
        }
    infos = payload.get("infos") if isinstance(payload, dict) else None
    checkpoint_task_contract = infos.get("task_contract") if isinstance(infos, dict) else None
    try:
        import ascento_mjlab.tasks  # noqa: F401

        cfg = load_env_cfg(task, play=False)
        compatibility = classify_task_contract_compatibility(
            checkpoint_task_contract, current_task_contract_for_task(task)
        )
        require_current_checkpoint_contracts(infos, cfg)
    except (OSError, RuntimeError, ValueError) as error:
        return {
            "status": "invalid",
            "reason": str(error),
            "task_contract": compatibility.to_dict() if "compatibility" in locals() else None,
        }
    return {
        "status": "passed",
        "reason": "checkpoint contracts are compatible with the requested task",
        "task_contract": compatibility.to_dict(),
    }


def _write_invalid_preflight_artifacts(
    output_dir: Path, manifest: dict[str, Any], preflight: dict[str, Any]
) -> None:
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["preflight"] = preflight
    task_contract = preflight.get("task_contract")
    if isinstance(task_contract, dict):
        manifest["checkpoint_task_compatibility"] = task_contract.get("status")
        manifest["checkpoint_task_compatibility_detail"] = task_contract
    gate_payload = {
        "status": EvaluationStatus.INVALID.value,
        "reason": preflight["reason"],
        "gates": [],
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "summary.json", {})
    write_json(
        output_dir / "consistency.json",
        {
            "passed": False,
            "checks": [
                {
                    "id": "checkpoint_contract_preflight",
                    "passed": False,
                    "reason": preflight["reason"],
                }
            ],
        },
    )
    write_json(output_dir / "gate.json", gate_payload)
    write_json(output_dir / "failures.json", {})
    render_html(
        output_dir / "report.html",
        manifest=manifest,
        summary={},
        gate_payload=gate_payload,
        worst={},
    )


def evaluate(
    *,
    checkpoint: Path,
    suite_path: Path,
    output_base: Path,
    batch_size: int,
    device: str,
    render_clips: bool = False,
    clip_takes: int = 3,
    clip_steps: int = 600,
) -> tuple[EvaluationStatus, Path]:
    suite = load_suite(suite_path)
    capabilities = task_capabilities(suite.task)
    missing = sorted(set(suite.required_capabilities) - capabilities)
    step_dt = task_step_dt(suite.task)
    scenarios = materialize_suite(suite, step_dt)
    output_dir = _make_output_dir(output_base, suite.suite_id, checkpoint)
    repo_root = _repo_root()
    manifest = _manifest(
        suite=suite,
        suite_path=suite_path,
        scenarios=scenarios,
        checkpoint=checkpoint,
        device=device,
        batch_size=batch_size,
        step_dt=step_dt,
        repo_root=repo_root,
    )
    write_suite_snapshot(output_dir / "suite.json", suite)
    write_resolved_scenarios(output_dir / "resolved_scenarios.jsonl", scenarios)

    if missing:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["missing_capabilities"] = missing
        gate_payload = {
            "status": EvaluationStatus.INCOMPLETE.value,
            "reason": f"missing capabilities: {', '.join(missing)}",
            "gates": [],
        }
        write_json(output_dir / "manifest.json", manifest)
        write_json(output_dir / "summary.json", {})
        write_json(output_dir / "gate.json", gate_payload)
        render_html(
            output_dir / "report.html",
            manifest=manifest,
            summary={},
            gate_payload=gate_payload,
            worst={},
        )
        return EvaluationStatus.INCOMPLETE, output_dir

    preflight = _checkpoint_contract_preflight(checkpoint, suite.task)
    if preflight["status"] != "passed":
        _write_invalid_preflight_artifacts(output_dir, manifest, preflight)
        return EvaluationStatus.INVALID, output_dir

    deterministic = suite.policy_mode == "deterministic"
    results, runtime = run_scenarios(
        suite.task,
        checkpoint,
        scenarios,
        batch_size=batch_size,
        device=device,
        deterministic=deterministic,
    )
    family_summary = summarize_results(results, bootstrap_seed=suite.root_seed)
    gate_status, gate_results = evaluate_gates(suite.gates, family_summary)
    consistent, checks = check_collection(results, scenarios, step_dt=step_dt)
    status = gate_status if consistent else EvaluationStatus.INVALID

    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["runtime"] = runtime
    checkpoint_contract = runtime.get("checkpoint_plant_contract")
    checkpoint_action_contract = runtime.get("checkpoint_action_contract")
    checkpoint_task_contract = runtime.get("checkpoint_task_contract")
    manifest["checkpoint_plant_contract"] = checkpoint_contract
    manifest["checkpoint_action_contract"] = checkpoint_action_contract
    manifest["checkpoint_task_contract"] = checkpoint_task_contract
    manifest["preflight"] = preflight
    manifest["checkpoint_plant_compatibility"] = (
        "current"
        if plant_contracts_compatible(checkpoint_contract, current_plant_contract())
        else "legacy"
        if checkpoint_contract is None
        else "incompatible"
    )
    manifest["checkpoint_action_compatibility"] = (
        "current"
        if action_contracts_compatible(checkpoint_action_contract, current_action_contract())
        else "legacy"
        if checkpoint_action_contract is None
        else "incompatible"
    )
    task_compatibility = classify_task_contract_compatibility(
        checkpoint_task_contract, current_task_contract_for_task(suite.task)
    )
    manifest["checkpoint_task_compatibility"] = task_compatibility.status.value
    manifest["checkpoint_task_compatibility_detail"] = task_compatibility.to_dict()
    manifest["capabilities"] = sorted(capabilities)
    write_json(output_dir / "manifest.json", manifest)
    write_results_database(output_dir / "results.sqlite", scenarios, results)
    write_json(output_dir / "summary.json", family_summary)
    write_json(
        output_dir / "consistency.json",
        {
            "passed": consistent,
            "checks": [check.to_dict() for check in checks],
        },
    )
    gate_payload = {
        "status": status.value,
        "gates": [result.to_dict() for result in gate_results],
    }
    write_json(output_dir / "gate.json", gate_payload)
    worst = select_worst_scenarios(results)
    write_json(output_dir / "failures.json", worst)
    render_html(
        output_dir / "report.html",
        manifest=manifest,
        summary=family_summary,
        gate_payload=gate_payload,
        worst=worst,
    )
    if render_clips:
        from ascento_mjlab.tools.capture_motion import capture

        clips_dir = output_dir / "clips"
        clip_payload: dict[str, object] = {
            "requested": True,
            "task": suite.task,
            "checkpoint": str(checkpoint),
            "takes": clip_takes,
            "steps": clip_steps,
            "device": device,
        }
        try:
            clip_payload.update(
                {
                    "status": "ok",
                    "captures": capture(
                        task=suite.task,
                        checkpoint=checkpoint,
                        takes=clip_takes,
                        steps=clip_steps,
                        output_dir=clips_dir,
                        video_dir=clips_dir / "videos",
                        device=device,
                    ),
                }
            )
        except (OSError, RuntimeError, ValueError) as error:
            clip_payload.update({"status": "error", "error": str(error)})
        write_json(output_dir / "clips_manifest.json", clip_payload)
    return status, output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite", required=True, help="Suite ID or TOML path")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", type=Path, default=Path("evaluations"))
    parser.add_argument(
        "--render-clips", action="store_true", help="render representative policy videos"
    )
    parser.add_argument("--clip-takes", type=int, default=3)
    parser.add_argument("--clip-steps", type=int, default=600)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.clip_takes < 1 or args.clip_steps < 1:
        parser.error("--clip-takes and --clip-steps must be positive")
    suite_path = resolve_suite_path(args.suite)
    status, output_dir = evaluate(
        checkpoint=args.checkpoint,
        suite_path=suite_path,
        output_base=args.output_root,
        batch_size=args.batch_size,
        device=_device(args.device),
        render_clips=args.render_clips,
        clip_takes=args.clip_takes,
        clip_steps=args.clip_steps,
    )
    print(f"\nEvaluation: {status.value}")
    print(f"Artifacts: {output_dir}")
    print(f"Report: {output_dir / 'report.html'}")
    raise SystemExit(0 if status == EvaluationStatus.PASS else 2)


if __name__ == "__main__":
    main()
