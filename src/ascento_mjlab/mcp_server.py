"""MCP tools for Ascento run management and low-noise monitoring."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ascento_mjlab.operations import (
    archive_evaluation,
    default_capture_dir,
    evaluation_details,
    evaluation_root,
    ensure_checkout_import_path,
    list_evaluations,
    repo_root,
    resolve_checkpoint,
    resolve_evaluation_dir,
)
from ascento_mjlab.operations import (
    list_evaluation_suites as discover_evaluation_suites,
)


# See ``ensure_checkout_import_path``: MCP is also installed as a console
# entry point, so it must not rely on the caller having cd'ed to the checkout.
ensure_checkout_import_path()

from dashboard.health import list_dashboard_summaries, load_dashboard_records  # noqa: E402
from dashboard.monitor import tail_lines, training_log_path  # noqa: E402
from dashboard.run_service import RunService  # noqa: E402

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as error:  # pragma: no cover - exercised when extra is absent
    FastMCP = None  # type: ignore[assignment]
    _MCP_IMPORT_ERROR = error


def _root() -> Path:
    return Path(os.environ.get("ASCENTO_ARTIFACT_ROOT", "logs/rsl_rl")).expanduser().resolve()


def _service() -> RunService:
    return RunService(_root())


def _evaluation_root(value: str | None = None) -> Path:
    return evaluation_root(value)


def _checkpoint(checkpoint: str | None, run_id: str | None) -> Path:
    return resolve_checkpoint(checkpoint=checkpoint, run_id=run_id, artifacts=_root())


def _evaluate(
    *,
    checkpoint: str | None,
    run_id: str | None,
    suite: str,
    batch_size: int,
    device: str,
    output_root: str | None,
    render_clips: bool,
    clip_takes: int,
    clip_steps: int,
) -> dict[str, Any]:
    from ascento_mjlab.evaluation.cli import _device, evaluate, resolve_suite_path

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if clip_takes < 1 or clip_steps < 1:
        raise ValueError("clip_takes and clip_steps must be positive")
    model = _checkpoint(checkpoint, run_id)
    root = _evaluation_root(output_root)
    status, directory = evaluate(
        checkpoint=model,
        suite_path=resolve_suite_path(suite),
        output_base=root,
        batch_size=batch_size,
        device=_device(device),
        render_clips=render_clips,
        clip_takes=clip_takes,
        clip_steps=clip_steps,
    )
    payload = evaluation_details(directory, root)
    payload["status"] = status.value
    return payload


def _capture(
    *,
    checkpoint: str | None,
    run_id: str | None,
    task: str | None,
    takes: int,
    steps: int,
    output_dir: str | None,
    video_dir: str | None,
    device: str,
) -> dict[str, Any]:
    from ascento_mjlab.evaluation.cli import _device
    from ascento_mjlab.tools.capture_motion import capture

    model = _checkpoint(checkpoint, run_id)
    if task is None and run_id is not None:
        task = _service().detail(run_id).get("task")
    project_root = repo_root().resolve()
    resolved_output = default_capture_dir(model) if output_dir is None else Path(output_dir).expanduser()
    if not resolved_output.is_absolute():
        resolved_output = project_root / resolved_output
    resolved_output = resolved_output.resolve()
    resolved_video = None if video_dir is None else Path(video_dir).expanduser()
    if resolved_video is not None and not resolved_video.is_absolute():
        resolved_video = project_root / resolved_video
    if resolved_video is not None:
        resolved_video = resolved_video.resolve()
    for path in (resolved_output, resolved_video):
        if path is None:
            continue
        try:
            path.relative_to(project_root)
        except ValueError as error:
            raise ValueError(f"capture paths must be below {project_root}") from error
    return {
        "task": task or "Ascento-Balance-Flat",
        "checkpoint": str(model),
        "output_dir": str(resolved_output),
        "captures": capture(
            task=str(task or "Ascento-Balance-Flat"),
            checkpoint=model,
            takes=takes,
            steps=steps,
            output_dir=resolved_output,
            video_dir=resolved_video,
            device=_device(device),
        ),
    }


def _preflight(
    *,
    balance_checkpoint: str,
    velocity_checkpoint: str,
    recovery_checkpoint: str,
    device: str,
    max_scenarios: int,
    batch_size: int,
) -> dict[str, Any]:
    from ascento_mjlab.evaluation.cli import _device
    from ascento_mjlab.evaluation.preflight import preflight_suite

    if max_scenarios < 1 or batch_size < 1:
        raise ValueError("max_scenarios and batch_size must be positive")
    checkpoints = {
        "balance": _checkpoint(balance_checkpoint, None),
        "velocity": _checkpoint(velocity_checkpoint, None),
        "recovery": _checkpoint(recovery_checkpoint, None),
    }
    resolved_device = _device(device)
    return {
        name: preflight_suite(
            name,
            checkpoint,
            device=resolved_device,
            max_scenarios=max_scenarios,
            batch_size=batch_size,
        )
        for name, checkpoint in checkpoints.items()
    }


if FastMCP is not None:
    mcp = FastMCP("ascento")

    @mcp.tool()
    def list_runs(active_only: bool = False) -> list[dict[str, Any]]:
        """List Ascento runs with state, task, progress, and health fields."""
        runs = list_dashboard_summaries(_root())
        if active_only:
            runs = [run for run in runs if run.get("state") in {"starting", "running", "stopping"}]
        return runs

    @mcp.tool()
    def get_run_progress(run_id: str) -> dict[str, Any]:
        """Read a cheap latest progress snapshot for one run."""
        return _service().progress(run_id)

    @mcp.tool()
    def get_run_details(run_id: str) -> dict[str, Any]:
        """Read detailed telemetry, health, provenance, and artifact metadata."""
        return _service().detail(run_id)

    @mcp.tool()
    def get_run_logs(run_id: str, tail: int = 200) -> dict[str, Any]:
        """Read a bounded tail of a run log without loading the full file."""
        service = _service()
        ref = service.resolve(run_id)
        return {"run_id": run_id, "lines": tail_lines(training_log_path(ref.path), max(1, min(tail, 5000)))}

    @mcp.tool()
    def get_run_telemetry(run_id: str, limit: int = 50) -> dict[str, Any]:
        """Read bounded normalized telemetry history for a run."""
        service = _service()
        ref = service.resolve(run_id)
        return {"run_id": run_id, "records": load_dashboard_records(ref.path, limit=max(1, min(limit, 2000)))}

    @mcp.tool()
    def get_dashboard_health() -> dict[str, Any]:
        """Return dashboard health and repository-version status."""
        from dashboard.app import health

        return health()

    @mcp.tool()
    def start_run(
        task: str = "Ascento-Balance-Flat",
        display_name: str = "Ascento run",
        purpose: str = "exploratory",
        tags: list[str] | None = None,
        notes: str = "",
        parent_run_id: str | None = None,
        parent_checkpoint: str | None = None,
        envs: int | None = None,
        iterations: int | None = None,
        seed: int | None = None,
        episode_horizon_s: float | None = None,
        training_args: list[str] | None = None,
    ) -> dict[str, Any]:
        """Start a managed training run and return its stable run ID."""
        args = list(training_args or [])
        if envs is not None:
            args += ["--env.scene.num-envs", str(envs)]
        if iterations is not None:
            args += ["--agent.max-iterations", str(iterations)]
        if seed is not None:
            args += ["--agent.seed", str(seed)]
        return _service().create({
            "display_name": display_name,
            "task": task,
            "purpose": purpose,
            "tags": tags or [],
            "notes": notes,
            "parent_run_id": parent_run_id,
            "parent_checkpoint": parent_checkpoint,
            "episode_horizon_s": episode_horizon_s,
            "training_args": args,
        })

    @mcp.tool()
    def stop_run(run_id: str, reason: str = "user_requested") -> dict[str, Any]:
        """Gracefully stop an active run."""
        return _service().stop(run_id, reason=reason)

    @mcp.tool()
    def update_run_metadata(
        run_id: str,
        display_name: str | None = None,
        notes: str | None = None,
        purpose: str | None = None,
        tags: list[str] | None = None,
        parent_run_id: str | None = None,
        parent_checkpoint: str | None = None,
    ) -> dict[str, Any]:
        """Update a run's display metadata, tags, and lineage without moving its artifacts."""
        changes: dict[str, Any] = {}
        for name, value in (
            ("display_name", display_name),
            ("notes", notes),
            ("purpose", purpose),
            ("parent_run_id", parent_run_id),
            ("parent_checkpoint", parent_checkpoint),
        ):
            if value is not None:
                changes[name] = value
        if tags is not None:
            changes["tags"] = tags
        if not changes:
            raise ValueError("supply at least one metadata field to update")
        return _service().update_metadata(run_id, changes)

    @mcp.tool()
    def compare_runs(run_ids: list[str]) -> dict[str, Any]:
        """Compare latest normalized telemetry for 2-8 runs."""
        return _service().compare(run_ids)

    @mcp.tool()
    def get_run_checkpoint(run_id: str) -> dict[str, str]:
        """Resolve the latest usable checkpoint for a managed run."""
        return {"run_id": run_id, "checkpoint": str(_checkpoint(None, run_id))}

    @mcp.tool()
    def list_evaluation_suites() -> list[dict[str, Any]]:
        """List immutable quantitative suites and their task/gate coverage."""
        return discover_evaluation_suites()

    @mcp.tool()
    def list_evaluation_reports(
        output_root: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List saved evaluator reports, gate outcomes, and clip availability."""
        return list_evaluations(_evaluation_root(output_root), limit=max(1, min(limit, 500)))

    @mcp.tool()
    def get_evaluation_report(
        evaluation: str,
        output_root: str | None = None,
    ) -> dict[str, Any]:
        """Read one evaluation's gates, consistency checks, failure modes, and clips manifest."""
        return evaluation_details(evaluation, _evaluation_root(output_root))

    @mcp.tool()
    def evaluate_policy(
        suite: str,
        checkpoint: str | None = None,
        run_id: str | None = None,
        batch_size: int = 512,
        device: str = "auto",
        output_root: str | None = None,
        render_clips: bool = False,
        clip_takes: int = 3,
        clip_steps: int = 600,
    ) -> dict[str, Any]:
        """Run a complete deterministic evaluation and write its HTML/JSON/SQLite report.

        This operation is intentionally synchronous: use it when the result,
        rather than a background process, is needed before choosing a policy.
        Supply exactly one of ``checkpoint`` or ``run_id``.
        """
        return _evaluate(
            checkpoint=checkpoint,
            run_id=run_id,
            suite=suite,
            batch_size=batch_size,
            device=device,
            output_root=output_root,
            render_clips=render_clips,
            clip_takes=clip_takes,
            clip_steps=clip_steps,
        )

    @mcp.tool()
    def compare_evaluation_reports(
        baseline: str,
        candidate: str,
        output_root: str | None = None,
    ) -> dict[str, Any]:
        """Compute paired metric deltas for two saved evaluations of the same scenarios."""
        from ascento_mjlab.evaluation.compare import compare

        root = _evaluation_root(output_root)
        return compare(
            resolve_evaluation_dir(baseline, root),
            resolve_evaluation_dir(candidate, root),
        )

    @mcp.tool()
    def archive_evaluation_report(
        evaluation: str,
        output_root: str | None = None,
        output: str | None = None,
    ) -> dict[str, str]:
        """Create a ZIP archive containing the complete evaluation evidence and rendered clips."""
        return {
            "archive": str(
                archive_evaluation(evaluation, root=_evaluation_root(output_root), output=output)
            )
        }

    @mcp.tool()
    def capture_policy(
        checkpoint: str | None = None,
        run_id: str | None = None,
        task: str | None = None,
        takes: int = 3,
        steps: int = 1000,
        output_dir: str | None = None,
        video_dir: str | None = None,
        device: str = "auto",
    ) -> dict[str, Any]:
        """Capture policy state channels and optional MP4 clips from a checkpoint or managed run."""
        return _capture(
            checkpoint=checkpoint,
            run_id=run_id,
            task=task,
            takes=takes,
            steps=steps,
            output_dir=output_dir,
            video_dir=video_dir,
            device=device,
        )

    @mcp.tool()
    def run_evaluator_preflight(
        balance_checkpoint: str,
        velocity_checkpoint: str,
        recovery_checkpoint: str,
        device: str = "auto",
        max_scenarios: int = 64,
        batch_size: int = 64,
    ) -> dict[str, Any]:
        """Run deterministic, mixed-horizon evaluator preflight for the three core tasks."""
        return _preflight(
            balance_checkpoint=balance_checkpoint,
            velocity_checkpoint=velocity_checkpoint,
            recovery_checkpoint=recovery_checkpoint,
            device=device,
            max_scenarios=max_scenarios,
            batch_size=batch_size,
        )


def main() -> None:
    if FastMCP is None:
        raise SystemExit(
            "MCP support is not installed; run `uv sync --extra mcp` first: "
            f"{_MCP_IMPORT_ERROR}"
        )
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
