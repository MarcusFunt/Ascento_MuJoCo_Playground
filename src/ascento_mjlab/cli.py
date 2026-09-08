"""Unified Ascento operations CLI.

The CLI intentionally delegates run lifecycle work to the dashboard's
filesystem-backed RunService, so command-line, web, and MCP operations share
the same metadata, process-group handling, and provenance.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from ascento_mjlab.operations import (
    archive_evaluation,
    artifact_root,
    default_capture_dir,
    evaluation_details,
    evaluation_root,
    ensure_checkout_import_path,
    list_evaluation_suites,
    list_evaluations,
    repo_root,
    resolve_checkpoint,
    resolve_evaluation_dir,
)


# The dashboard is a checkout-local companion package.  Console entry points
# start with ``.venv/bin`` on sys.path, so establish the repository path before
# any lazy dashboard import below.
ensure_checkout_import_path()


def _repo_root() -> Path:
    return repo_root()


def _artifact_root(value: str | None) -> Path:
    return artifact_root(value)


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
        "parent_run_id": args.parent_run_id,
        "parent_checkpoint": args.parent_checkpoint,
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


def _annotate(args: argparse.Namespace) -> int:
    changes: dict[str, Any] = {}
    for field in ("display_name", "notes", "purpose", "parent_run_id", "parent_checkpoint"):
        value = getattr(args, field)
        if value is not None:
            changes[field] = value
    if args.tag is not None:
        changes["tags"] = args.tag
    if not changes:
        raise ValueError("supply at least one metadata field to update")
    value = _service(_artifact_root(args.artifact_root)).update_metadata(args.run_id, changes)
    _print(value, args.json)
    return 0


def _checkpoint_from_args(args: argparse.Namespace) -> Path:
    return resolve_checkpoint(
        checkpoint=args.checkpoint,
        run_id=args.run_id,
        artifacts=getattr(args, "artifact_root", None),
    )


def _evaluate(args: argparse.Namespace) -> int:
    from ascento_mjlab.evaluation.cli import _device, evaluate, resolve_suite_path

    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if args.clip_takes < 1 or args.clip_steps < 1:
        raise ValueError("clip_takes and clip_steps must be positive")
    checkpoint = _checkpoint_from_args(args)
    output = evaluation_root(args.output_root)
    status, directory = evaluate(
        checkpoint=checkpoint,
        suite_path=resolve_suite_path(args.suite),
        output_base=output,
        batch_size=args.batch_size,
        device=_device(args.device),
        render_clips=args.render_clips,
        clip_takes=args.clip_takes,
        clip_steps=args.clip_steps,
    )
    payload = evaluation_details(directory, output)
    payload["status"] = status.value
    _print(payload, args.json)
    return 0 if status.value == "PASS" else 2


def _screen_checkpoints(args: argparse.Namespace) -> int:
    from ascento_mjlab.evaluation.checkpoints import _expand, _score
    from ascento_mjlab.evaluation.cli import _device, evaluate, resolve_suite_path

    if args.batch_size < 1 or args.top < 1:
        raise ValueError("batch_size and top must be positive")
    checkpoints = _expand(args.checkpoints)
    if not checkpoints:
        raise ValueError("no checkpoints matched")
    output = evaluation_root(args.output_root)
    suite_path = resolve_suite_path(args.suite)
    rows = []
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            continue
        status, directory = evaluate(
            checkpoint=checkpoint.resolve(),
            suite_path=suite_path,
            output_base=output,
            batch_size=args.batch_size,
            device=_device(args.device),
        )
        row = evaluation_details(directory, output)
        row["status"] = status.value
        row["score"] = _score(directory)
        rows.append(row)
    rows.sort(key=lambda row: row["score"], reverse=True)
    _print({"suite": args.suite, "ranked": rows[: args.top]}, args.json)
    return 0


def _compare_evaluations(args: argparse.Namespace) -> int:
    from ascento_mjlab.evaluation.compare import compare

    output = evaluation_root(args.output_root)
    baseline = resolve_evaluation_dir(args.baseline, output)
    candidate = resolve_evaluation_dir(args.candidate, output)
    payload = compare(baseline, candidate)
    if args.output is not None:
        path = Path(args.output).expanduser()
        if not path.is_absolute():
            path = output / path
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        payload["output"] = str(path)
    _print(payload, args.json)
    return 0


def _preflight(args: argparse.Namespace) -> int:
    from ascento_mjlab.evaluation.cli import _device
    from ascento_mjlab.evaluation.preflight import preflight_suite

    if args.max_scenarios < 1 or args.batch_size < 1:
        raise ValueError("max_scenarios and batch_size must be positive")
    checkpoints = {
        "balance": args.balance_checkpoint,
        "velocity": args.velocity_checkpoint,
        "recovery": args.recovery_checkpoint,
    }
    for name, checkpoint in checkpoints.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{name} checkpoint does not exist: {checkpoint}")
    device = _device(args.device)
    payload = {
        name: preflight_suite(
            name,
            checkpoint.resolve(),
            device=device,
            max_scenarios=args.max_scenarios,
            batch_size=args.batch_size,
        )
        for name, checkpoint in checkpoints.items()
    }
    if args.output is not None:
        path = Path(args.output).expanduser()
        if not path.is_absolute():
            path = _repo_root() / path
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        payload["output"] = str(path)
    _print(payload, args.json)
    return 0


def _list_evaluations(args: argparse.Namespace) -> int:
    _print(list_evaluations(args.output_root, limit=args.limit), args.json)
    return 0


def _evaluation_details(args: argparse.Namespace) -> int:
    _print(evaluation_details(args.evaluation, args.output_root), args.json)
    return 0


def _archive_evaluation(args: argparse.Namespace) -> int:
    archive = archive_evaluation(args.evaluation, root=args.output_root, output=args.output)
    _print({"archive": str(archive)}, args.json)
    return 0


def _list_suites(args: argparse.Namespace) -> int:
    _print(list_evaluation_suites(), args.json)
    return 0


def _capture(args: argparse.Namespace) -> int:
    from ascento_mjlab.evaluation.cli import _device
    from ascento_mjlab.tools.capture_motion import capture

    checkpoint = _checkpoint_from_args(args)
    task = args.task
    if task is None and args.run_id is not None:
        task = _service(_artifact_root(args.artifact_root)).detail(args.run_id).get("task")
    task = str(task or "Ascento-Balance-Flat")
    output = default_capture_dir(checkpoint) if args.output_dir is None else Path(args.output_dir).expanduser()
    if not output.is_absolute():
        output = _repo_root() / output
    video = None
    if args.video_dir is not None:
        video = Path(args.video_dir).expanduser()
        if not video.is_absolute():
            video = _repo_root() / video
    payload = {
        "task": task,
        "checkpoint": str(checkpoint),
        "output_dir": str(output),
        "captures": capture(
            task=task,
            checkpoint=checkpoint,
            takes=args.takes,
            steps=args.steps,
            output_dir=output,
            video_dir=video,
            device=_device(args.device),
        ),
    }
    _print(payload, args.json)
    return 0


def _delegate_tool(args: argparse.Namespace) -> int:
    """Keep specialist, interactive tools reachable from the unified CLI."""
    tool_args = list(args.tool_args)
    if tool_args[:1] == ["--"]:
        tool_args = tool_args[1:]
    command = [sys.executable, "-m", args.module, *tool_args]
    return subprocess.run(command, cwd=_repo_root(), check=False).returncode


def _serve_mcp(_args: argparse.Namespace) -> int:
    from ascento_mjlab.mcp_server import main as serve

    serve()
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
    start.add_argument("--parent-run-id")
    start.add_argument("--parent-checkpoint")
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
    annotate = run_sub.add_parser("annotate", parents=[common], help="update run lineage and metadata")
    annotate.add_argument("run_id")
    annotate.add_argument("--display-name")
    annotate.add_argument("--notes")
    annotate.add_argument("--purpose")
    annotate.add_argument("--tag", action="append", default=None)
    annotate.add_argument("--parent-run-id")
    annotate.add_argument("--parent-checkpoint")
    annotate.set_defaults(handler=_annotate)

    evaluate = sub.add_parser("evaluate", help="run, inspect, compare, and archive immutable evaluations")
    evaluate_sub = evaluate.add_subparsers(dest="evaluate_command", required=True)
    evaluate_common = argparse.ArgumentParser(add_help=False)
    evaluate_common.add_argument("--artifact-root")
    evaluate_common.add_argument("--output-root", default="evaluations")
    evaluate_common.add_argument("--json", action="store_true")
    evaluation = evaluate_sub.add_parser(
        "run", parents=[evaluate_common], help="evaluate a checkpoint or managed run"
    )
    evaluation_source = evaluation.add_mutually_exclusive_group(required=True)
    evaluation_source.add_argument("--checkpoint", type=Path)
    evaluation_source.add_argument("--run-id")
    evaluation.add_argument("--suite", required=True, help="suite ID or TOML path")
    evaluation.add_argument("--batch-size", type=int, default=512)
    evaluation.add_argument("--device", default="auto")
    evaluation.add_argument("--render-clips", action="store_true")
    evaluation.add_argument("--clip-takes", type=int, default=3)
    evaluation.add_argument("--clip-steps", type=int, default=600)
    evaluation.set_defaults(handler=_evaluate)
    screen = evaluate_sub.add_parser(
        "screen", parents=[evaluate_common], help="rank checkpoints against one immutable suite"
    )
    screen.add_argument("checkpoints", nargs="+", help="checkpoint paths or glob patterns")
    screen.add_argument("--suite", required=True, help="suite ID or TOML path")
    screen.add_argument("--batch-size", type=int, default=256)
    screen.add_argument("--device", default="auto")
    screen.add_argument("--top", type=int, default=3)
    screen.set_defaults(handler=_screen_checkpoints)
    evaluation_compare = evaluate_sub.add_parser(
        "compare", parents=[evaluate_common], help="paired comparison of two completed evaluations"
    )
    evaluation_compare.add_argument("baseline")
    evaluation_compare.add_argument("candidate")
    evaluation_compare.add_argument("--output")
    evaluation_compare.set_defaults(handler=_compare_evaluations)
    evaluation_list = evaluate_sub.add_parser(
        "list", parents=[evaluate_common], help="list saved evaluation reports"
    )
    evaluation_list.add_argument("--limit", type=int, default=50)
    evaluation_list.set_defaults(handler=_list_evaluations)
    evaluation_report = evaluate_sub.add_parser(
        "report", parents=[evaluate_common], help="show gates, consistency, failures, and clips"
    )
    evaluation_report.add_argument("evaluation")
    evaluation_report.set_defaults(handler=_evaluation_details)
    evaluation_archive = evaluate_sub.add_parser(
        "archive", parents=[evaluate_common], help="ZIP one complete evaluation artifact"
    )
    evaluation_archive.add_argument("evaluation")
    evaluation_archive.add_argument("--output")
    evaluation_archive.set_defaults(handler=_archive_evaluation)
    suites = evaluate_sub.add_parser("suites", parents=[evaluate_common], help="list immutable suites")
    suites.set_defaults(handler=_list_suites)
    preflight = evaluate_sub.add_parser(
        "preflight", parents=[evaluate_common], help="run deterministic evaluator preflight"
    )
    preflight.add_argument("--balance-checkpoint", type=Path, required=True)
    preflight.add_argument("--velocity-checkpoint", type=Path, required=True)
    preflight.add_argument("--recovery-checkpoint", type=Path, required=True)
    preflight.add_argument("--device", default="auto")
    preflight.add_argument("--max-scenarios", type=int, default=64)
    preflight.add_argument("--batch-size", type=int, default=64)
    preflight.add_argument("--output", default="preflight/evaluator.json")
    preflight.set_defaults(handler=_preflight)

    capture = sub.add_parser("capture", help="capture policy state channels and optional MP4 clips")
    capture.add_argument("--artifact-root")
    capture.add_argument("--json", action="store_true")
    capture_source = capture.add_mutually_exclusive_group(required=True)
    capture_source.add_argument("--checkpoint", type=Path)
    capture_source.add_argument("--run-id")
    capture.add_argument("--task", default=None)
    capture.add_argument("--takes", type=int, default=3)
    capture.add_argument("--steps", type=int, default=1000)
    capture.add_argument("--output-dir", default=None)
    capture.add_argument("--video-dir", default=None)
    capture.add_argument("--device", default="auto")
    capture.set_defaults(handler=_capture)

    tools = sub.add_parser("tools", help="reach specialist motion, reward, and replay tools")
    tools_sub = tools.add_subparsers(dest="tool_command", required=True)
    for name, module, help_text in (
        ("clip-motion", "ascento_mjlab.tools.clip_motion", "trim or resample a capture"),
        ("rank-motion", "ascento_mjlab.tools.motion_quality", "rank captures for visual review"),
        ("reward-probe", "ascento_mjlab.tools.reward_probe", "inspect reward terms"),
        ("replay-evaluation", "ascento_mjlab.evaluation.replay", "open an exact evaluator replay"),
    ):
        tool = tools_sub.add_parser(name, help=help_text)
        tool.add_argument("tool_args", nargs=argparse.REMAINDER)
        tool.set_defaults(handler=_delegate_tool, module=module)

    mcp = sub.add_parser("mcp", help="serve the project MCP integration over stdio")
    mcp_sub = mcp.add_subparsers(dest="mcp_command", required=True)
    mcp_serve = mcp_sub.add_parser("serve", help="run the stdio MCP server")
    mcp_serve.set_defaults(handler=_serve_mcp)

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
