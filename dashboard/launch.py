"""Launch an mjlab/RSL-RL run with durable dashboard metadata and console logging."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from uuid import uuid4

from dashboard.config import REPO_ROOT, load_config

TRAINING_RUNTIME_RE = re.compile(
    r"Training with:\s*device=([^,\s]+),\s*seed=([^,\s]+),\s*rank=(\d+)"
)
GPU_WORLD_RE = re.compile(r"Launching training with\s+(\d+)\s+GPUs?", re.IGNORECASE)
HORIZON_CURRICULUM_RE = re.compile(
    r"HORIZON_CURRICULUM\s+horizon_s=(?P<horizon>\S+)\s+stage=(?P<stage>\d+)"
    r"\s+qualified_windows=(?P<qualified>\d+)"
    r"(?:\s+failed_windows=(?P<failed>\d+))?"
    r"(?:\s+stage_windows=(?P<stage_windows>\d+))?"
    r"(?:\s+top_horizon_windows=(?P<top_windows>\d+))?"
    r"(?:\s+transition=(?P<transition>\S+))?"
    r"(?:\s+timeout_fraction=(?P<timeout>\S+))?"
    r"(?:\s+candidate_checkpoint=(?P<candidate>\S+))?"
)


def write_status(path: Path, **values: Any) -> None:
    current: dict[str, Any] = {}
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = {}
    current.update(values)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(current, indent=2), encoding="utf-8")
    tmp.replace(path)


def write_metadata(path: Path, **values: Any) -> None:
    current: dict[str, Any] = {}
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = {}
    current.update(values)
    current["updated_at"] = datetime.now(timezone.utc).isoformat()
    current.setdefault("created_at", current["updated_at"])
    current.setdefault("schema_version", 1)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def _repository_env(name: str) -> str | None:
    value = os.environ.get(name)
    if not value or value.strip().lower() in {"unknown", "none", "null"}:
        return None
    return value.strip()


def _pid_namespace() -> str | None:
    try:
        return os.readlink("/proc/self/ns/pid")
    except OSError:
        return None


def git_metadata() -> dict[str, Any]:
    # A source checkout has .git and should report its exact live state. The
    # maintained Docker image intentionally excludes .git, so it receives the
    # immutable build commit/branch through environment variables instead.
    commit = _git_value("rev-parse", "HEAD") or _repository_env("ASCENTO_REPOSITORY_COMMIT")
    branch = _git_value("branch", "--show-current") or _repository_env("ASCENTO_REPOSITORY_BRANCH")
    dirty = None
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
        dirty = bool(result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return {"commit": commit, "branch": branch, "dirty": dirty}


def _training_arg(training_args: list[str], *names: str) -> str | None:
    """Return the last value supplied for one of the CLI option names."""
    value: str | None = None
    for index, token in enumerate(training_args):
        for name in names:
            if token == name and index + 1 < len(training_args):
                value = training_args[index + 1]
            elif token.startswith(name + "="):
                value = token.split("=", 1)[1]
    return value


def _number(value: str | None) -> int | float | str | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _runtime_status_from_line(line: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    runtime = TRAINING_RUNTIME_RE.search(line)
    if runtime:
        values["device"] = runtime.group(1)
        values["seed"] = _number(runtime.group(2))
        values["rank"] = int(runtime.group(3))
    world = GPU_WORLD_RE.search(line)
    if world:
        values["gpu_world_size"] = int(world.group(1))
    horizon = HORIZON_CURRICULUM_RE.search(line)
    if horizon:
        values["episode_horizon_s"] = float(horizon.group("horizon"))
        values["horizon_stage"] = int(horizon.group("stage"))
        values["horizon_qualified_windows"] = int(horizon.group("qualified"))
        if horizon.group("failed") is not None:
            values["horizon_failed_windows"] = int(horizon.group("failed"))
        if horizon.group("stage_windows") is not None:
            values["horizon_stage_windows"] = int(horizon.group("stage_windows"))
        if horizon.group("top_windows") is not None:
            values["horizon_top_windows"] = int(horizon.group("top_windows"))
        if horizon.group("transition") is not None:
            values["horizon_transition"] = horizon.group("transition")
        if horizon.group("timeout") is not None:
            values["horizon_timeout_fraction"] = float(horizon.group("timeout"))
        if horizon.group("candidate") is not None:
            values["long_horizon_candidate_checkpoint"] = horizon.group("candidate")
    return values


def _latest_checkpoint(run_dir: Path) -> str | None:
    candidates = list(run_dir.rglob("model_*.pt"))
    if not candidates:
        return None

    def key(path: Path) -> tuple[int, float]:
        number = -1
        stem = path.stem
        suffix = stem.rsplit("_", 1)[-1]
        if suffix.isdigit():
            number = int(suffix)
        try:
            modified = path.stat().st_mtime
        except OSError:
            modified = 0.0
        return number, modified

    latest = max(candidates, key=key)
    return latest.relative_to(run_dir).as_posix()


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    names = ("mjlab", "mujoco", "mujoco-warp", "warp-lang", "torch", "rsl-rl")
    values: dict[str, str] = {}
    for name in names:
        try:
            values[name] = version(name)
        except PackageNotFoundError:
            values[name] = "unknown"
    return values


def _experiment_reward_terms(task: str) -> dict[str, dict[str, Any]]:
    try:
        from mjlab.tasks.registry import load_env_cfg

        cfg = load_env_cfg(task, play=False)
        return {
            name: {
                "weight": float(term.weight),
                "params": {
                    key: value
                    for key, value in (term.params or {}).items()
                    if isinstance(value, (str, int, float, bool, type(None)))
                },
            }
            for name, term in cfg.rewards.items()
        }
    except Exception as error:  # pragma: no cover - defensive metadata path
        return {"__error__": {"weight": 0.0, "error": str(error)}}


def _training_arg_value(training_args: list[str], *names: str) -> str | None:
    return _training_arg(training_args, *names)


def _write_experiment_manifest(
    path: Path,
    *,
    task: str,
    training_args: list[str],
    git: dict[str, Any],
    run_dir: Path,
    sim_timestep: int | float | str | None,
    device: str | None,
) -> None:
    env_count = _training_arg_value(training_args, "--env.scene.num-envs", "--num-envs")
    manifest = {
        "schema_version": 1,
        "task": task,
        "task_config_id": task,
        "reward_terms": _experiment_reward_terms(task),
        "dense_shaping_enabled": os.environ.get("ASCENTO_DISABLE_DENSE_SHAPING", "") != "1",
        "experiment_overrides": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith("ASCENTO_")
            and key
            not in {"ASCENTO_ARTIFACT_ROOT", "ASCENTO_REPOSITORY_COMMIT", "ASCENTO_REPOSITORY_BRANCH"}
        },
        "seed": _number(_training_arg_value(training_args, "--seed", "--agent.seed")),
        "environment_count": _number(env_count),
        "simulation_timestep": sim_timestep,
        "device": device,
        "packages": _package_versions(),
        "git": git,
        "checkpoint": {"path": None, "sha256": None},
        "evaluation": {"suite": None, "result": None},
        "run_directory": str(run_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_metadata(path, **manifest)


def _finalize_experiment_manifest(path: Path, run_dir: Path) -> None:
    if not path.is_file():
        return
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    checkpoint = _latest_checkpoint(run_dir)
    if checkpoint:
        checkpoint_path = run_dir / checkpoint
        manifest["checkpoint"] = {
            "path": checkpoint,
            "sha256": _file_sha256(checkpoint_path),
        }
    write_metadata(path, **manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run mjlab training while capturing console output for the dashboard."
    )
    default_artifact_root = load_config().artifact_root
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=default_artifact_root,
        help=f"dashboard/training artifact root (default: {default_artifact_root})",
    )
    parser.add_argument("--name", help="run directory name; defaults to timestamp_stage")
    parser.add_argument("--task", default="Ascento-Balance-Flat")
    parser.add_argument(
        "--run-id", help="stable dashboard run id; generated automatically when omitted"
    )
    parser.add_argument(
        "--preinitialized",
        action="store_true",
        help="reuse the empty run directory initialized synchronously by the dashboard API",
    )
    parser.add_argument("--display-name", help="human-readable run name shown by the dashboard")
    parser.add_argument("--notes", default="", help="human notes stored with the run")
    parser.add_argument("--tag", action="append", default=[], help="repeatable run tag")
    parser.add_argument("--purpose", default="", help="run purpose, e.g. baseline or validation")
    parser.add_argument("--parent-run-id", help="dashboard id of the parent run")
    parser.add_argument("--parent-checkpoint", help="checkpoint inherited from the parent run")
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    return parser


def _interrupt_on_sigterm(_signum: int, _frame: Any) -> None:
    """Route an external supervisor stop through the existing SIGINT cleanup."""
    raise KeyboardInterrupt


def _graceful_stop(process: subprocess.Popen[str]) -> int:
    """Give the trainer its normal interrupt cleanup before escalating."""
    try:
        process.send_signal(signal.SIGINT)
    except ProcessLookupError:
        pass
    try:
        return process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            return process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            return process.wait()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    training_args = list(args.training_args)
    if training_args and training_args[0] == "--":
        training_args = training_args[1:]
    if "--output" in training_args:
        parser.error("do not pass --output; dashboard.launch assigns an isolated run directory")

    stage = args.task.removeprefix("Ascento-").removesuffix("-Flat").lower()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.name or f"{stamp}_{stage}"
    run_dir = (args.artifact_root.expanduser().resolve() / run_name).resolve()

    try:
        expected_files = {"run_metadata.json", "run_status.json"}
        existing_files = {path.name for path in run_dir.iterdir()} if run_dir.exists() else set()
        if args.preinitialized and existing_files != expected_files:
            parser.error(f"preinitialized run directory is incomplete: {run_dir}")
        if not args.preinitialized and existing_files:
            parser.error(f"run directory already exists and is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        parser.error(f"cannot create run directory {run_dir}: {error}")

    status_path = run_dir / "run_status.json"
    metadata_path = run_dir / "run_metadata.json"
    log_path = run_dir / "training.log"
    command = [
        sys.executable,
        "-u",
        "-m",
        "mjlab.scripts.train",
        args.task,
        *training_args,
        "--log-root",
        str(run_dir),
    ]

    seed = _number(_training_arg(training_args, "--seed", "--agent.seed"))
    device = _training_arg(
        training_args,
        "--device",
        "--env.device",
        "--agent.device",
    )
    sim_timestep = _number(
        _training_arg(
            training_args,
            "--env.sim.mujoco.timestep",
            "--env.sim.dt",
            "--sim.dt",
            "--simulation.dt",
        )
    )
    episode_horizon_s = _number(_training_arg(training_args, "--env.episode-length-s"))
    started = datetime.now(timezone.utc).isoformat()
    git = git_metadata()
    display_name = (args.display_name or run_name).strip()
    stable_run_id = (args.run_id or uuid4().hex[:12]).strip()
    tags = list(dict.fromkeys(tag.strip() for tag in args.tag if tag.strip()))

    write_metadata(
        metadata_path,
        run_id=stable_run_id,
        display_name=display_name,
        notes=args.notes.strip(),
        tags=tags,
        purpose=args.purpose.strip(),
        parent_run_id=args.parent_run_id,
        parent_checkpoint=args.parent_checkpoint,
    )

    # The API starts launchers in a new session so their process group can be
    # signalled without touching uvicorn. Direct CLI launches are still safe:
    # they simply omit process_group and the launcher forwards SIGINT itself.
    process_group = os.getpgrp() if os.getpid() == os.getpgrp() else None
    write_status(
        status_path,
        schema_version=4,
        state="starting",
        run_id=stable_run_id,
        task=args.task,
        stage=stage,
        display_name=display_name,
        seed=seed,
        device=device,
        simulation_timestep=sim_timestep,
        requested_episode_horizon_s=episode_horizon_s,
        started_at=started,
        command=command,
        command_line=" ".join(command),
        python_executable=sys.executable,
        artifact_root=str(args.artifact_root.expanduser().resolve()),
        run_directory=str(run_dir),
        git=git,
        git_commit=git.get("commit"),
        git_branch=git.get("branch"),
        launcher_pid=os.getpid(),
        process_group=process_group,
        pid_namespace=_pid_namespace(),
        exit_code=None,
        stop_requested_at=None,
        stop_reason=None,
    )
    experiment_manifest_path = run_dir / "experiment_manifest.json"
    _write_experiment_manifest(
        experiment_manifest_path,
        task=args.task,
        training_args=training_args,
        git=git,
        run_dir=run_dir,
        sim_timestep=sim_timestep,
        device=device,
    )

    exit_code = 127
    stop_requested = False
    process: subprocess.Popen[str] | None = None
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _interrupt_on_sigterm)
    try:
        try:
            with log_path.open("a", encoding="utf-8", buffering=1) as log:
                log.write("DASHBOARD_LAUNCH " + " ".join(command) + "\n")
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=REPO_ROOT,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                except OSError as error:
                    message = f"Dashboard launcher could not start training: {error}"
                    log.write(message + "\n")
                    print(message, file=sys.stderr)
                    write_status(
                        status_path,
                        state="error",
                        launch_error=str(error),
                        exit_code=127,
                        finished_at=datetime.now(timezone.utc).isoformat(),
                    )
                    return 127

                write_status(status_path, state="running", pid=process.pid)
                try:
                    assert process.stdout is not None
                    for line in process.stdout:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                        log.write(line)
                        runtime_values = _runtime_status_from_line(line)
                        if runtime_values:
                            write_status(status_path, **runtime_values)
                    exit_code = process.wait()
                except KeyboardInterrupt:
                    stop_requested = True
                    exit_code = _graceful_stop(process)
        except KeyboardInterrupt:
            stop_requested = True
            exit_code = _graceful_stop(process) if process is not None else 130
    except OSError as error:
        message = f"Dashboard launcher cannot write {log_path}: {error}"
        print(message, file=sys.stderr)
        write_status(
            status_path,
            state="error",
            launch_error=str(error),
            exit_code=127,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        return 127
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)

    _finalize_experiment_manifest(experiment_manifest_path, run_dir)
    finished = datetime.now(timezone.utc).isoformat()
    state = "stopped" if stop_requested else ("finished" if exit_code == 0 else "error")
    write_status(
        status_path,
        state=state,
        exit_code=exit_code,
        finished_at=finished,
        checkpoint_path=_latest_checkpoint(run_dir),
        stop_reason="user_requested" if stop_requested else None,
    )
    return 130 if stop_requested else exit_code


if __name__ == "__main__":
    raise SystemExit(main())
