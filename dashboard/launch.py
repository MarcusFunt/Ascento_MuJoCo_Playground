"""Launch an mjlab/RSL-RL run with durable dashboard metadata and console logging."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from dashboard.config import REPO_ROOT, load_config

TRAINING_RUNTIME_RE = re.compile(
    r"Training with:\s*device=([^,\s]+),\s*seed=([^,\s]+),\s*rank=(\d+)"
)
GPU_WORLD_RE = re.compile(r"Launching training with\s+(\d+)\s+GPUs?", re.IGNORECASE)
HORIZON_CURRICULUM_RE = re.compile(
    r"HORIZON_CURRICULUM\s+horizon_s=(\S+)\s+stage=(\d+)\s+qualified_windows=(\d+)"
    r"(?:\s+timeout_fraction=(\S+))?"
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
        values["episode_horizon_s"] = float(horizon.group(1))
        values["horizon_stage"] = int(horizon.group(2))
        values["horizon_qualified_windows"] = int(horizon.group(3))
        if horizon.group(4) is not None:
            values["horizon_timeout_fraction"] = float(horizon.group(4))
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

    exit_code = 127
    stop_requested = False
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
                # SIGINT is the graceful request for both native and API-managed
                # launches. Give the trainer time to run its interrupt cleanup
                # before escalating to terminate/kill.
                try:
                    process.send_signal(signal.SIGINT)
                except ProcessLookupError:
                    pass
                try:
                    exit_code = process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        exit_code = process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        exit_code = process.wait()
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
