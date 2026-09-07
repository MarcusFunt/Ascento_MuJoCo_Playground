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
import urllib.request
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


def _logs(args: argparse.Namespace) -> int:
    from dashboard.monitor import tail_lines, training_log_path

    service = _service(_artifact_root(args.artifact_root))
    ref = service.resolve(args.run_id)
    _print({"run_id": args.run_id, "lines": tail_lines(training_log_path(ref.path), args.tail)}, args.json)
    return 0


def _telemetry(args: argparse.Namespace) -> int:
    from dashboard.health import load_dashboard_records

    service = _service(_artifact_root(args.artifact_root))
    ref = service.resolve(args.run_id)
    _print({"run_id": args.run_id, "records": load_dashboard_records(ref.path, limit=args.limit)}, args.json)
    return 0


def _compare(args: argparse.Namespace) -> int:
    _print(_service(_artifact_root(args.artifact_root)).compare(args.run_ids), args.json)
    return 0


def _dashboard_status(args: argparse.Namespace) -> int:
    url = f"http://{args.host}:{args.port}/api/health"
    try:
        with urllib.request.urlopen(url, timeout=args.timeout) as response:
            payload = json.load(response)
    except (OSError, ValueError) as error:
        _print({"ok": False, "url": url, "error": str(error)}, args.json)
        return 1
    _print(payload, args.json)
    return 0


def _dashboard_start(args: argparse.Namespace) -> int:
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "dashboard.app:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.reload:
        command.append("--reload")
    if args.detach:
        process = subprocess.Popen(
            command,
            cwd=_repo_root(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        _print({"state": "starting", "pid": process.pid, "host": args.host, "port": args.port}, args.json)
        return 0
    return subprocess.run(command, cwd=_repo_root(), check=False).returncode


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
    logs = run_sub.add_parser("logs", parents=[common], help="show a bounded training-log tail")
    logs.add_argument("run_id")
    logs.add_argument("--tail", type=int, default=200)
    logs.set_defaults(handler=_logs)
    telemetry = run_sub.add_parser("telemetry", parents=[common], help="show bounded normalized telemetry")
    telemetry.add_argument("run_id")
    telemetry.add_argument("--limit", type=int, default=50)
    telemetry.set_defaults(handler=_telemetry)
    compare = run_sub.add_parser("compare", parents=[common], help="compare normalized latest metrics")
    compare.add_argument("run_ids", nargs="+", metavar="RUN_ID")
    compare.set_defaults(handler=_compare)

    maintain = sub.add_parser("maintain", help="run scripts/maintain.sh")
    maintain.add_argument("maintain_args", nargs=argparse.REMAINDER)
    maintain.set_defaults(handler=_maintain)

    dashboard = sub.add_parser("dashboard", help="start or inspect the dashboard")
    dashboard_sub = dashboard.add_subparsers(dest="dashboard_command", required=True)
    dashboard_common = argparse.ArgumentParser(add_help=False)
    dashboard_common.add_argument("--host", default="127.0.0.1")
    dashboard_common.add_argument("--port", type=int, default=8000)
    dashboard_common.add_argument("--json", action="store_true")
    dashboard_status = dashboard_sub.add_parser("status", parents=[dashboard_common])
    dashboard_status.add_argument("--timeout", type=float, default=5.0)
    dashboard_status.set_defaults(handler=_dashboard_status)
    dashboard_start = dashboard_sub.add_parser("start", parents=[dashboard_common])
    dashboard_start.add_argument("--foreground", dest="detach", action="store_false")
    dashboard_start.set_defaults(detach=True)
    dashboard_start.add_argument("--reload", action="store_true")
    dashboard_start.set_defaults(handler=_dashboard_start)
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
