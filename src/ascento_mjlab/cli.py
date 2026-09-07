"""Unified Ascento operations CLI.

The CLI intentionally delegates run lifecycle work to the dashboard's
filesystem-backed RunService, so command-line, web, and MCP operations share
the same metadata, process-group handling, and provenance.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def _artifact_root(value: str | None) -> Path:
    return Path(value or os.environ.get("ASCENTO_ARTIFACT_ROOT", "logs/rsl_rl")).expanduser().resolve()


def _service(root: Path):
    from dashboard.run_service import RunService

    return RunService(root)


def _print(value: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(value, indent=2, sort_keys=True, default=str))
    elif isinstance(value, list):
        for item in value:
            print(json.dumps(item, sort_keys=True, default=str))
    else:
        print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _start(args: argparse.Namespace) -> int:
    root = _artifact_root(args.artifact_root)
    training_args = list(args.training_args or [])
    if args.envs is not None:
        training_args += ["--env.scene.num-envs", str(args.envs)]
    if args.iterations is not None:
        training_args += ["--agent.max-iterations", str(args.iterations)]
    if args.seed is not None:
        training_args += ["--agent.seed", str(args.seed)]
    request = {
        "display_name": args.display_name or args.name or args.task,
        "task": args.task,
        "purpose": args.purpose,
        "tags": args.tag,
        "notes": args.notes,
        "episode_horizon_s": args.episode_horizon_s,
        "training_args": training_args,
    }
    created = _service(root).create(request)
    _print(created, args.json)
    if args.foreground:
        while True:
            progress = _service(root).progress(created["id"])
            _print(progress, args.json)
            if progress.get("state") not in {"starting", "running", "stopping"}:
                return 0 if progress.get("status", {}).get("exit_code", 1) == 0 else 1
            time.sleep(args.interval)
    return 0


def _list(args: argparse.Namespace) -> int:
    from dashboard.health import list_dashboard_summaries

    runs = list_dashboard_summaries(_artifact_root(args.artifact_root))
    if args.active:
        runs = [run for run in runs if run.get("state") in {"starting", "running", "stopping"}]
    _print(runs, args.json)
    return 0


def _status(args: argparse.Namespace, progress: bool = False) -> int:
    value = (_service(_artifact_root(args.artifact_root)).progress(args.run_id)
             if progress else _service(_artifact_root(args.artifact_root)).detail(args.run_id))
    _print(value, args.json)
    return 0


def _stop(args: argparse.Namespace) -> int:
    value = _service(_artifact_root(args.artifact_root)).stop(args.run_id, reason=args.reason)
    _print(value, args.json)
    return 0


def _monitor(args: argparse.Namespace) -> int:
    service = _service(_artifact_root(args.artifact_root))
    previous: tuple[Any, ...] | None = None
    while True:
        value = service.progress(args.run_id)
        telemetry = value.get("telemetry") or {}
        canonical = telemetry.get("canonical_metrics") or {}
        signature = (
            value.get("state"),
            telemetry.get("iteration"),
            telemetry.get("percent_complete"),
            canonical.get("reward"),
            canonical.get("invalid_update"),
            value.get("stale"),
        )
        if signature != previous or args.once:
            _print(value if args.json else {
                "id": value.get("id"), "state": value.get("state"),
                "iteration": telemetry.get("iteration"),
                "percent_complete": telemetry.get("percent_complete"),
                "reward": canonical.get("reward"),
                "throughput": canonical.get("throughput"),
                "stale": value.get("stale"),
                "invalid_updates": (value.get("training_health") or {}).get("invalid_updates"),
            }, args.json)
            previous = signature
        if args.once or value.get("state") not in {"starting", "running", "stopping"}:
            return 0
        time.sleep(args.interval)


def _maintain(args: argparse.Namespace) -> int:
    command = ["bash", str(_repo_root() / "scripts" / "maintain.sh"), *args.maintain_args]
    return subprocess.run(command, cwd=_repo_root(), check=False).returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ascento", description="Ascento training operations CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="start and manage training runs")
    run_sub = run.add_subparsers(dest="run_command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--artifact-root")
    common.add_argument("--json", action="store_true")
    start = run_sub.add_parser("start", parents=[common], help="start a managed run")
    start.add_argument("--task", default="Ascento-Balance-Flat")
    start.add_argument("--name")
    start.add_argument("--display-name")
    start.add_argument("--purpose", default="exploratory")
    start.add_argument("--tag", action="append", default=[])
    start.add_argument("--notes", default="")
    start.add_argument("--envs", type=int)
    start.add_argument("--iterations", type=int)
    start.add_argument("--seed", type=int)
    start.add_argument("--episode-horizon-s", type=float)
    start.add_argument("--foreground", action="store_true")
    start.add_argument("--interval", type=float, default=15.0)
    start.add_argument("training_args", nargs=argparse.REMAINDER)
    start.set_defaults(handler=_start)
    listed = run_sub.add_parser("list", parents=[common], help="list runs")
    listed.add_argument("--active", action="store_true")
    listed.set_defaults(handler=_list)
    for name, handler, help_text in (("status", _status, "show detailed run status"), ("progress", lambda a: _status(a, True), "show cheap live progress")):
        command = run_sub.add_parser(name, parents=[common], help=help_text)
        command.add_argument("run_id")
        command.set_defaults(handler=handler)
    monitor = run_sub.add_parser("monitor", parents=[common], help="print only changed progress snapshots")
    monitor.add_argument("run_id")
    monitor.add_argument("--interval", type=float, default=30.0)
    monitor.add_argument("--once", action="store_true")
    monitor.set_defaults(handler=_monitor)
    stop = run_sub.add_parser("stop", parents=[common], help="gracefully stop an active run")
    stop.add_argument("run_id")
    stop.add_argument("--reason", default="user_requested")
    stop.set_defaults(handler=_stop)

    maintain = sub.add_parser("maintain", help="run scripts/maintain.sh")
    maintain.add_argument("maintain_args", nargs=argparse.REMAINDER)
    maintain.set_defaults(handler=_maintain)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "training_args", None) and args.training_args[0] == "--":
        args.training_args = args.training_args[1:]
    try:
        raise SystemExit(args.handler(args))
    except (KeyError, ValueError, OSError) as error:
        print(f"ascento: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
