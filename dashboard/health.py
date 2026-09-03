"""Training health, process/GPU status, and reproducibility metadata for the dashboard."""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dashboard import monitor

ALIASES: dict[str, tuple[str, ...]] = {
    "reward": ("Train/mean_reward", "train/mean_reward", "eval/episode_reward", "Episode/reward", "episode_reward"),
    "episode_length": ("Train/mean_episode_length", "train/mean_episode_length", "eval/avg_episode_length", "Episode/length", "episode_length"),
    "ppo_loss": ("Loss/surrogate", "loss/surrogate", "training/policy_loss", "training/total_loss", "ppo_loss"),
    "value_loss": ("Loss/value", "Loss/value_function", "loss/value_function", "training/v_loss", "value_loss"),
    "entropy": ("Loss/entropy", "loss/entropy", "Policy/entropy", "training/entropy", "training/entropy_loss", "entropy"),
    "kl": ("Loss/kl", "loss/kl", "Policy/kl", "training/kl", "training/approx_kl", "approx_kl", "kl"),
    "clip_fraction": ("Loss/clip_fraction", "Policy/clip_fraction", "training/clip_fraction", "clip_fraction"),
    "learning_rate": ("Loss/learning_rate", "training/learning_rate", "learning_rate"),
    "invalid_update": ("training/invalid_update", "Train/invalid_update", "invalid_update"),
    "throughput": ("Perf/total_fps", "perf/total_fps", "training/fps", "steps_per_second"),
}
CHECKPOINT_RE = re.compile(r"(?:model|checkpoint)[_-]?(\d+)", re.IGNORECASE)
TRAINING_RUNTIME_RE = re.compile(
    r"Training with:\s*device=([^,\s]+),\s*seed=([^,\s]+),\s*rank=(\d+)"
)
GPU_WORLD_RE = re.compile(r"Launching training with\s+(\d+)\s+GPUs?", re.IGNORECASE)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _timestamp(value: Any) -> float | None:
    if _finite(value):
        return float(value)
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _pid_namespace() -> str | None:
    try:
        return os.readlink("/proc/self/ns/pid")
    except OSError:
        return None


def canonical_metrics(metrics: dict[str, Any]) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {}
    for canonical, aliases in ALIASES.items():
        result[canonical] = None
        for alias in aliases:
            value = metrics.get(alias)
            if isinstance(value, bool):
                result[canonical] = int(value)
                break
            if _finite(value):
                result[canonical] = value
                break
    return result


def _training_limit(run_dir: Path) -> int | None:
    path = run_dir / "params" / "agent.yaml"
    try:
        contents = path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"^\s*max_iterations:\s*(\d+)\s*$", contents, re.MULTILINE)
    return int(match.group(1)) if match else None


def _yaml_path_scalar(path: Path, keys: tuple[str, ...]) -> str | None:
    """Read one scalar from a simple nested YAML mapping without a YAML dependency."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None

    stack: list[tuple[int, str]] = []
    item_re = re.compile(r"^(\s*)([^:#][^:]*):(?:\s*(.*?))?\s*$")
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("-"):
            continue
        match = item_re.match(line)
        if not match:
            continue
        indent = len(match.group(1))
        key = match.group(2).strip()
        value = (match.group(3) or "").strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        stack.append((indent, key))
        if tuple(part for _, part in stack) == keys and value:
            return value.split("#", 1)[0].strip().strip("\"'")
    return None


def _training_shape(run_dir: Path) -> tuple[int | None, int | None]:
    agent = run_dir / "params" / "agent.yaml"
    env = run_dir / "params" / "env.yaml"
    steps = _yaml_path_scalar(agent, ("num_steps_per_env",)) if agent.is_file() else None
    envs = _yaml_path_scalar(env, ("scene", "num_envs")) if env.is_file() else None
    try:
        steps_value = int(steps) if steps is not None else None
    except ValueError:
        steps_value = None
    try:
        envs_value = int(envs) if envs is not None else None
    except ValueError:
        envs_value = None
    return steps_value, envs_value


def decorate_records(records: list[dict[str, Any]], run_dir: Path) -> list[dict[str, Any]]:
    """Return JSON-safe telemetry with explicit iteration and environment-step units."""
    decorated: list[dict[str, Any]] = []
    total = _training_limit(run_dir)
    steps_per_env, num_envs = _training_shape(run_dir)
    for source in records:
        record = dict(source)
        raw_metrics = source.get("metrics") if isinstance(source.get("metrics"), dict) else {}
        safe_metrics: dict[str, Any] = {}
        non_finite: list[str] = []
        for key, value in raw_metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(float(value)):
                safe_metrics[str(key)] = None
                non_finite.append(str(key))
            else:
                safe_metrics[str(key)] = value
        canonical = canonical_metrics(safe_metrics)
        if canonical["invalid_update"] is None and non_finite:
            canonical["invalid_update"] = 1
        record["metrics"] = safe_metrics
        record["canonical_metrics"] = canonical
        record["non_finite_metrics"] = non_finite
        record["has_non_finite"] = bool(non_finite)

        iteration = record.get(
            "iteration", record.get("completed_iterations", record.get("completed_steps"))
        )
        if _finite(iteration):
            iteration = int(iteration)
            record["iteration"] = iteration
            record["completed_iterations"] = iteration
        record.pop("completed_steps", None)

        configured_total = record.get("total_iterations", record.get("total_steps", total))
        if _finite(configured_total):
            configured_total = int(configured_total)
            record["total_iterations"] = configured_total
            if _finite(iteration):
                record["percent_complete"] = min(
                    100.0, 100.0 * float(iteration) / max(configured_total, 1)
                )
        record.pop("total_steps", None)

        if _finite(iteration) and steps_per_env and num_envs:
            record["environment_steps"] = int(iteration) * steps_per_env * num_envs
            if _finite(configured_total):
                record["total_environment_steps"] = configured_total * steps_per_env * num_envs

        throughput = canonical.get("throughput")
        if _finite(throughput) and float(throughput) > 0:
            record["environment_steps_per_second"] = float(throughput)
        record.pop("steps_per_second", None)
        decorated.append(record)

    if not decorated:
        return decorated
    first_wall = decorated[0].get("wall_time")
    for index, record in enumerate(decorated):
        wall = record.get("wall_time")
        if _finite(first_wall) and _finite(wall):
            record["elapsed_seconds"] = max(0.0, float(wall) - float(first_wall))
        iteration = record.get("iteration")
        total_iterations = record.get("total_iterations")
        if not (_finite(iteration) and _finite(total_iterations)):
            continue
        if float(iteration) >= float(total_iterations):
            record["eta_seconds"] = 0.0
            continue
        start = decorated[max(0, index - 20)]
        start_iteration = start.get("iteration")
        start_wall = start.get("wall_time")
        if not (_finite(start_iteration) and _finite(start_wall) and _finite(wall)):
            continue
        delta_i = float(iteration) - float(start_iteration)
        delta_t = float(wall) - float(start_wall)
        if delta_i > 0 and delta_t > 0:
            rate = delta_i / delta_t
            record["iterations_per_second"] = rate
            record["eta_seconds"] = max(
                0.0, (float(total_iterations) - float(iteration)) / rate
            )
    return decorated


def load_dashboard_records(run_dir: Path, limit: int | None = 2000) -> list[dict[str, Any]]:
    # Load enough recent history to compute a stable rolling iteration rate.
    raw_limit = None if limit is None else max(limit, 32)
    records = decorate_records(monitor.load_training_records(run_dir, limit=raw_limit), run_dir)
    return records[-limit:] if limit is not None else records


def run_status_path(run_dir: Path, root: Path) -> Path:
    for directory in (run_dir, *run_dir.parents):
        if not _inside(directory, root):
            break
        candidate = directory / "run_status.json"
        if candidate.is_file():
            return candidate
    return run_dir / "run_status.json"


def _has_training_signal(path: Path) -> bool:
    if (path / "telemetry.jsonl").is_file() or (path / "training_manifest.json").is_file():
        return True
    if (path / "params" / "agent.yaml").is_file():
        return True
    if any(path.glob("events.out.tfevents.*")) or any(path.glob("model_*.pt")):
        return True
    return False


def discover_dashboard_runs(root: Path) -> list[monitor.RunRef]:
    refs = monitor.discover_runs(root)
    paths = {ref.path.resolve() for ref in refs}
    filtered: list[monitor.RunRef] = []
    for ref in refs:
        path = ref.path.resolve()
        launcher_only = (path / "run_status.json").is_file() and not _has_training_signal(path)
        has_nested_signal = any(
            other != path and _inside(other, path) and _has_training_signal(other)
            for other in paths
        )
        if launcher_only and has_nested_signal:
            continue
        filtered.append(ref)
    return filtered


def resolve_dashboard_run(root: Path, run_id: str) -> monitor.RunRef:
    for ref in discover_dashboard_runs(root):
        if ref.id == run_id:
            return ref
    raise KeyError(run_id)


def process_status(pid: Any, source_pid_namespace: str | None = None) -> dict[str, Any]:
    if not isinstance(pid, int) or pid <= 0:
        return {"pid": None, "alive": None, "same_namespace": None}

    current_namespace = _pid_namespace()
    if (
        source_pid_namespace is not None
        and current_namespace is not None
        and source_pid_namespace != current_namespace
    ):
        return {
            "pid": pid,
            "alive": None,
            "same_namespace": False,
            "reason": "different_pid_namespace",
        }

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        alive = False
    except PermissionError:
        alive = True
    except OSError:
        alive = None
    else:
        alive = True
    return {
        "pid": pid,
        "alive": alive,
        "same_namespace": True if source_pid_namespace and current_namespace else None,
    }


def gpu_snapshot(pid: int | None = None) -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "gpus": [], "process_gpu_memory_mb": None}
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": False, "gpus": [], "process_gpu_memory_mb": None, "error": str(error)}
    gpus: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 7:
            continue
        try:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "uuid": parts[2],
                    "utilization_percent": float(parts[3]),
                    "memory_used_mb": float(parts[4]),
                    "memory_total_mb": float(parts[5]),
                    "temperature_c": float(parts[6]),
                }
            )
        except ValueError:
            continue
    process_memory: float | None = None
    if isinstance(pid, int) and pid > 0:
        try:
            result = subprocess.run(
                [
                    executable,
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            for line in result.stdout.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pid:
                    try:
                        process_memory = (process_memory or 0.0) + float(parts[1])
                    except ValueError:
                        pass
        except (OSError, subprocess.SubprocessError):
            pass
    return {"available": True, "gpus": gpus, "process_gpu_memory_mb": process_memory}


def _yaml_scalar(path: Path, key: str) -> str | None:
    return _yaml_path_scalar(path, (key,))


def _sim_timestep(run_dir: Path) -> float | None:
    for name in ("env.yaml", "env_cfg.yaml", "environment.yaml"):
        path = run_dir / "params" / name
        if not path.is_file():
            continue
        for keys in (
            ("sim", "mujoco", "timestep"),
            ("sim", "dt"),
            ("simulation", "dt"),
        ):
            value = _yaml_path_scalar(path, keys)
            if value is not None:
                try:
                    return float(value)
                except ValueError:
                    pass
    return None


def _runtime_metadata(run_dir: Path, root: Path) -> dict[str, Any]:
    log_path = monitor.training_log_path(run_dir, root)
    metadata: dict[str, Any] = {}
    for line in monitor.tail_lines(log_path, 4000):
        runtime = TRAINING_RUNTIME_RE.search(line)
        if runtime:
            metadata["device"] = runtime.group(1)
            try:
                metadata["seed"] = int(runtime.group(2))
            except ValueError:
                metadata["seed"] = runtime.group(2)
            metadata["rank"] = int(runtime.group(3))
        world = GPU_WORLD_RE.search(line)
        if world:
            metadata["gpu_world_size"] = int(world.group(1))
    return metadata


def _latest_checkpoint(run_dir: Path) -> str | None:
    candidates = list(run_dir.glob("model_*.pt"))
    checkpoint_dir = run_dir / "checkpoint"
    if checkpoint_dir.is_dir():
        candidates.extend(checkpoint_dir.rglob("*.pt"))
    if not candidates:
        return None

    def key(path: Path) -> tuple[int, float]:
        match = CHECKPOINT_RE.search(path.stem)
        try:
            modified = path.stat().st_mtime
        except OSError:
            modified = 0.0
        return (int(match.group(1)) if match else -1, modified)

    return max(candidates, key=key).relative_to(run_dir).as_posix()


def _configuration_files(run_dir: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".yaml", ".yml", ".json", ".toml"}:
            continue
        relative = path.relative_to(run_dir)
        is_params = bool(relative.parts) and relative.parts[0] == "params"
        looks_config = any(token in path.stem.lower() for token in ("config", "cfg"))
        if not is_params and not looks_config:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        files.append({"path": relative.as_posix(), "size_bytes": size})
        if len(files) >= 64:
            break
    return files


def _first(sources: tuple[dict[str, Any], ...], *keys: str) -> Any:
    for source in sources:
        for key in keys:
            value = source.get(key)
            if value is not None:
                return value
    return None


def build_run_info(run_dir: Path, root: Path, stage: str) -> dict[str, Any]:
    status = _json(run_status_path(run_dir, root)) or {}
    manifest = _json(run_dir / "training_manifest.json") or {}
    runtime = _runtime_metadata(run_dir, root)
    git_data = _first((status, manifest), "git")
    git_data = git_data if isinstance(git_data, dict) else {}
    agent = run_dir / "params" / "agent.yaml"
    env = run_dir / "params" / "env.yaml"

    seed = _first((status, manifest, runtime), "seed")
    if seed is None and agent.is_file():
        raw = _yaml_scalar(agent, "seed")
        if raw is not None:
            try:
                seed = int(raw)
            except ValueError:
                seed = raw

    device = _first((status, manifest, runtime), "device")
    if device is None:
        for path in (agent, env):
            raw = _yaml_scalar(path, "device") if path.is_file() else None
            if raw:
                device = raw
                break

    sim_timestep = _first((status, manifest), "simulation_timestep", "sim_timestep", "sim_dt")
    if sim_timestep is None:
        sim_timestep = _sim_timestep(run_dir)

    return {
        "git_commit": _first((status, manifest, git_data), "git_commit", "commit", "sha"),
        "git_branch": _first((status, manifest, git_data), "git_branch", "branch"),
        "task": _first((status, manifest), "task"),
        "stage": stage,
        "seed": seed,
        "command": _first((status, manifest), "command", "command_line"),
        "simulation_timestep": sim_timestep,
        "device": device,
        "gpu_world_size": _first((status, manifest, runtime), "gpu_world_size"),
        "checkpoint_path": _first((status, manifest), "checkpoint_path", "model_path")
        or _latest_checkpoint(run_dir),
        "model_path": _first((status, manifest), "model_path"),
        "configuration_files": _configuration_files(run_dir),
        "started_at": _first((status, manifest), "started_at", "start_time"),
        "finished_at": _first((status, manifest), "finished_at", "end_time"),
        "exit_code": _first((status, manifest), "exit_code"),
    }


def training_health(records: list[dict[str, Any]]) -> dict[str, Any]:
    invalid_updates = 0
    non_finite_updates = 0
    names: set[str] = set()
    for record in records:
        if record.get("has_non_finite"):
            non_finite_updates += 1
            names.update(record.get("non_finite_metrics") or [])
        invalid = (record.get("canonical_metrics") or {}).get("invalid_update")
        if _finite(invalid) and float(invalid) > 0:
            invalid_updates += int(max(1, round(float(invalid))))
    latest = records[-1] if records else {}
    return {
        "invalid_updates": invalid_updates,
        "non_finite_updates": non_finite_updates,
        "non_finite_metrics": sorted(names),
        "latest": latest.get("canonical_metrics") or {},
    }


def summarize_dashboard_run(
    run_dir: Path,
    root: Path,
    *,
    stale_after_seconds: float = 90.0,
    detailed: bool = False,
    now: float | None = None,
) -> dict[str, Any]:
    now = time.time() if now is None else now
    base = monitor.summarize_run(run_dir, root, now=now)
    records = load_dashboard_records(run_dir, limit=500 if detailed else 1)
    latest = records[-1] if records else None
    status_file = run_status_path(run_dir, root)
    status = _json(status_file) or base.get("status") or {}
    modified_at = float(base.get("modified_at") or 0.0)
    wall = latest.get("wall_time") if latest else None
    freshness = (
        max(0.0, now - float(wall))
        if _finite(wall)
        else max(0.0, now - modified_at)
    )
    state = status.get("state") or base.get("state") or "unknown"
    if latest is not None:
        started = _timestamp(status.get("started_at"))
        event_time = _timestamp(latest.get("wall_time"))
        if started is not None and event_time is not None:
            latest["elapsed_seconds"] = max(0.0, event_time - started)

    process = process_status(status.get("pid"), status.get("pid_namespace"))
    stale = state in {"starting", "running"} and freshness > stale_after_seconds
    if state == "running" and process["alive"] is False:
        stale = True

    base.update(
        {
            "state": state,
            "status": status,
            "telemetry": latest,
            "stale": stale,
            "stale_after_seconds": stale_after_seconds,
            "freshness_seconds": freshness,
            "process": process,
            "status_source": status_file.relative_to(root).as_posix()
            if status_file.is_file() and _inside(status_file, root)
            else None,
        }
    )
    if detailed:
        base["training_health"] = training_health(records)
        base["run_info"] = build_run_info(run_dir, root, str(base.get("stage") or "unknown"))
        process_pid = status.get("pid") if process.get("same_namespace") is not False else None
        base["system"] = gpu_snapshot(process_pid)
    return base


def list_dashboard_summaries(
    root: Path, *, stale_after_seconds: float = 90.0
) -> list[dict[str, Any]]:
    return [
        summarize_dashboard_run(ref.path, root, stale_after_seconds=stale_after_seconds)
        for ref in discover_dashboard_runs(root)
    ]
