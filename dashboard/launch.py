"""Launch an mjlab/RSL-RL run with durable console logging."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def write_status(path: Path, **values) -> None:
    current = {}
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = {}
    current.update(values)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(current, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run mjlab training while capturing console output for the dashboard."
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("logs/rsl_rl"))
    parser.add_argument("--name", help="run directory name; defaults to timestamp_stage")
    parser.add_argument("--task", default="Ascento-Balance-Flat")
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    training_args = list(args.training_args)
    if training_args and training_args[0] == "--":
        training_args = training_args[1:]
    if "--output" in training_args:
        parser.error("do not pass --output; dashboard.launch assigns an isolated run directory")

    stage = args.task.removeprefix("Ascento-").removesuffix("-Flat").lower()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.name or f"{stamp}_{stage}"
    run_dir = (args.artifact_root / run_name).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        parser.error(f"run directory already exists and is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    status_path = run_dir / "run_status.json"
    log_path = run_dir / "training.log"
    command = [
        sys.executable,
        "-u",
        "-m", "mjlab.scripts.train",
        args.task,
        *training_args,
        "--log-root", str(run_dir),
    ]
    started = datetime.now(timezone.utc).isoformat()
    write_status(
        status_path,
        state="starting",
        stage=stage,
        started_at=started,
        command=command,
        exit_code=None,
    )

    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write("DASHBOARD_LAUNCH " + " ".join(command) + "\n")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        write_status(status_path, state="running", pid=process.pid)
        try:
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
            exit_code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            exit_code = process.wait()
            write_status(status_path, state="cancelled", exit_code=exit_code)
            raise

    finished = datetime.now(timezone.utc).isoformat()
    state = "finished" if exit_code == 0 else "error"
    write_status(status_path, state=state, exit_code=exit_code, finished_at=finished)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
