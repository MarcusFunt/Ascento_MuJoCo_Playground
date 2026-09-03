"""Launch an mjlab/RSL-RL run with durable dashboard metadata and console logging."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dashboard.config import REPO_ROOT, load_config

TRAINING_RUNTIME_RE = re.compile(
    r"Training with:\s*device=([^,\s]+),\s*seed=([^,\s]+),\s*rank=(\d+)"
)
GPU_WORLD_RE = re.compile(r"Launching training with\s+(\d+)\s+GPUs?", re.IGNORECASE)


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
        if run_dir.exists() and any(run_dir.iterdir()):
            parser.error(f"run directory already exists and is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        parser.error(f"cannot create run directory {run_dir}: {error}")

    status_path = run_dir / "run_status.json"
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
    started = datetime.now(timezone.utc).isoformat()
    git = git_metadata()

    write_status(
        status_path,
        schema_version=3,
        state="starting",
        task=args.task,
        stage=stage,
        seed=seed,
        device=device,
        simulation_timestep=sim_timestep,
        started_at=started,
        command=command,
        command_line=" ".join(command),
        python_executable=sys.executable,
        artifact_root=str(args.artifact_root.expanduser().resolve()),
        run_directory=str(run_dir),
        git=git,
        git_commit=git.get("commit"),
        git_branch=git.get("branch"),
        pid_namespace=_pid_namespace(),
        exit_code=None,
    )

    exit_code = 127
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
                process.terminate()
                exit_code = process.wait()
                write_status(
                    status_path,
                    state="cancelled",
                    exit_code=exit_code,
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    checkpoint_path=_latest_checkpoint(run_dir),
                )
                raise
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
    state = "finished" if exit_code == 0 else "error"
    write_status(
        status_path,
        state=state,
        exit_code=exit_code,
        finished_at=finished,
        checkpoint_path=_latest_checkpoint(run_dir),
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
