"""MCP tools for Ascento run management and low-noise monitoring."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dashboard.health import list_dashboard_summaries
from dashboard.run_service import RunService

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as error:  # pragma: no cover - exercised when extra is absent
    FastMCP = None  # type: ignore[assignment]
    _MCP_IMPORT_ERROR = error


def _root() -> Path:
    return Path(os.environ.get("ASCENTO_ARTIFACT_ROOT", "logs/rsl_rl")).expanduser()


def _service() -> RunService:
    return RunService(_root())


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
    def start_run(
        task: str = "Ascento-Balance-Flat",
        display_name: str = "Ascento run",
        purpose: str = "exploratory",
        tags: list[str] | None = None,
        envs: int | None = None,
        iterations: int | None = None,
        seed: int | None = None,
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
            "training_args": args,
        })

    @mcp.tool()
    def stop_run(run_id: str, reason: str = "user_requested") -> dict[str, Any]:
        """Gracefully stop an active run."""
        return _service().stop(run_id, reason=reason)

    @mcp.tool()
    def compare_runs(run_ids: list[str]) -> dict[str, Any]:
        """Compare latest normalized telemetry for 2-8 runs."""
        return _service().compare(run_ids)


def main() -> None:
    if FastMCP is None:
        raise SystemExit(
            "MCP support is not installed; run `uv sync --extra mcp` first: "
            f"{_MCP_IMPORT_ERROR}"
        )
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
