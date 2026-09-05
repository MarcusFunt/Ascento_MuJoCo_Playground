"""FastAPI service for remotely monitoring and managing Ascento PPO training."""
from __future__ import annotations

import asyncio
import json
import os
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from dashboard.config import load_config, validate_startup
from dashboard.health import (
    discover_dashboard_runs,
    list_dashboard_summaries,
    load_dashboard_records,
)
from dashboard.monitor import tail_lines, training_log_path
from dashboard.run_service import RunService
from dashboard.supervisor_client import (
    SupervisorClient,
    SupervisorRejected,
    SupervisorUnavailable,
)
from dashboard.versioning import current_repository_version

CONFIG = load_config()
STARTUP_WARNINGS = validate_startup(CONFIG, create_artifact_root=False)
ARTIFACT_ROOT = CONFIG.artifact_root
FRONTEND_DIST = CONFIG.frontend_dist
RUN_SERVICE = RunService(ARTIFACT_ROOT, stale_after_seconds=CONFIG.stale_after_seconds)
SUPERVISOR = SupervisorClient()

app = FastAPI(title="Ascento Training Monitor", version="2.1")


class RunCreateRequest(BaseModel):
    display_name: str
    task: str = "Ascento-Balance-Flat"
    notes: str = ""
    tags: list[str] = Field(default_factory=list)
    purpose: str = ""
    parent_run_id: str | None = None
    parent_checkpoint: str | None = None
    training_args: list[str] = Field(default_factory=list)


class RunUpdateRequest(BaseModel):
    display_name: str | None = None
    notes: str | None = None
    tags: list[str] | None = None
    purpose: str | None = None
    parent_run_id: str | None = None
    parent_checkpoint: str | None = None


class RunStopRequest(BaseModel):
    reason: str = "user_requested"


def _run(run_id: str):
    try:
        return RUN_SERVICE.resolve(run_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="training run not found") from error
    except OSError as error:
        raise HTTPException(
            status_code=500,
            detail=f"failed to scan artifact root {ARTIFACT_ROOT}: {error}",
        ) from error


def _artifact_health() -> list[str]:
    problems: list[str] = []
    if ARTIFACT_ROOT.exists() and not ARTIFACT_ROOT.is_dir():
        problems.append(f"artifact root is not a directory: {ARTIFACT_ROOT}")
    elif ARTIFACT_ROOT.exists():
        if not os.access(ARTIFACT_ROOT, os.R_OK):
            problems.append(f"artifact root is not readable: {ARTIFACT_ROOT}")
        if not os.access(ARTIFACT_ROOT, os.X_OK):
            problems.append(f"artifact root is not searchable: {ARTIFACT_ROOT}")
    return problems


def _annotated_summaries() -> list[dict]:
    summaries = list_dashboard_summaries(
        ARTIFACT_ROOT,
        stale_after_seconds=CONFIG.stale_after_seconds,
    )
    refs = {ref.id: ref for ref in discover_dashboard_runs(ARTIFACT_ROOT)}
    for summary in summaries:
        ref = refs.get(summary.get("id"))
        if ref is not None:
            RUN_SERVICE.annotate(summary, ref.path)
    return summaries


@app.get("/api/health")
def health():
    problems = _artifact_health()
    try:
        run_count = len(_annotated_summaries())
    except OSError as error:
        problems.append(f"artifact scan failed: {error}")
        run_count = None
    return {
        "ok": not problems,
        "status": "ok" if not problems else "error",
        "checked_at": time.time(),
        "artifact_root": str(ARTIFACT_ROOT),
        "run_count": run_count,
        "problems": problems,
        "warnings": STARTUP_WARNINGS,
        "config": CONFIG.public_dict(),
        "repository_version": current_repository_version(),
    }


@app.get("/api/config")
def configuration():
    return {
        **CONFIG.public_dict(),
        "startup_warnings": STARTUP_WARNINGS,
        "repository_version": current_repository_version(),
    }


@app.get("/api/system")
def system_status(refresh: bool = False):
    """Return host repository/update/Tailnet state through the restricted supervisor."""
    try:
        return {"connected": True, **SUPERVISOR.status(refresh=refresh)}
    except SupervisorUnavailable as error:
        return {
            "connected": False,
            "error": str(error),
            "repository": None,
            "active_runs": [],
            "update": {"status": "unavailable"},
            "tailscale": {"enabled": False, "connected": False},
            "can_update": False,
            "update_blockers": ["host supervisor is unavailable"],
        }


@app.post("/api/system/update", status_code=202)
def system_update(request: Request):
    """Ask the host supervisor to update to the newest origin/main."""
    # Requiring a non-simple custom header prevents cross-site forms and simple
    # browser requests from invoking the privileged host boundary. Tailnet
    # grants/ACLs remain the authentication boundary for reaching this service.
    if request.headers.get("x-ascento-control") != "1":
        raise HTTPException(status_code=403, detail="missing dashboard control header")
    try:
        return SUPERVISOR.update()
    except SupervisorUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except SupervisorRejected as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/api/runs")
def runs():
    try:
        summaries = _annotated_summaries()
    except OSError as error:
        raise HTTPException(
            status_code=500,
            detail=f"failed to scan artifact root {ARTIFACT_ROOT}: {error}",
        ) from error
    return {"runs": summaries}


@app.post("/api/runs", status_code=202)
def create_run(request: RunCreateRequest):
    try:
        return RUN_SERVICE.create(request.model_dump())
    except KeyError as error:
        raise HTTPException(status_code=400, detail=f"parent run not found: {error.args[0]}") from error
    except (ValueError, PermissionError, OSError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/runs/compare")
def compare_runs(run_ids: str):
    ids = [value.strip() for value in run_ids.split(",") if value.strip()]
    try:
        return RUN_SERVICE.compare(ids)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=f"training run not found: {error.args[0]}") from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/runs/{run_id}")
def run_status(run_id: str):
    try:
        return RUN_SERVICE.detail(run_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="training run not found") from error


@app.patch("/api/runs/{run_id}")
def update_run(run_id: str, request: RunUpdateRequest):
    try:
        return RUN_SERVICE.update_metadata(run_id, request.model_dump(exclude_unset=True))
    except KeyError as error:
        raise HTTPException(status_code=404, detail="training run or parent run not found") from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/runs/{run_id}/stop", status_code=202)
def stop_run(run_id: str, request: RunStopRequest):
    try:
        return RUN_SERVICE.stop(run_id, reason=request.reason.strip() or "user_requested")
    except KeyError as error:
        raise HTTPException(status_code=404, detail="training run not found") from error
    except ProcessLookupError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/api/runs/{run_id}/summary.json")
def run_summary(run_id: str):
    try:
        summary = RUN_SERVICE.detail(run_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="training run not found") from error
    return JSONResponse(
        content=summary,
        headers={"Content-Disposition": 'attachment; filename="run-summary.json"'},
    )


@app.get("/api/runs/{run_id}/telemetry")
def telemetry(run_id: str, limit: int = 2000):
    ref = _run(run_id)
    limit = max(1, min(limit, 20_000))
    return {"records": load_dashboard_records(ref.path, limit=limit)}


@app.get("/api/runs/{run_id}/logs")
def logs(run_id: str, tail: int = 500):
    ref = _run(run_id)
    tail = max(1, min(tail, 5000))
    log_path = training_log_path(ref.path, ARTIFACT_ROOT)
    return {"lines": [line.rstrip("\n") for line in tail_lines(log_path, tail)]}


@app.get("/api/runs/{run_id}/logs/stream")
async def stream_logs(run_id: str, request: Request):
    ref = _run(run_id)
    log_path = training_log_path(ref.path, ARTIFACT_ROOT)

    async def event_stream():
        try:
            position = log_path.stat().st_size
        except OSError:
            position = 0
        inode = None
        while True:
            if await request.is_disconnected():
                return
            try:
                stat = log_path.stat()
                current_inode = getattr(stat, "st_ino", None)
                if inode is not None and current_inode != inode:
                    position = 0
                inode = current_inode
                if stat.st_size < position:
                    position = 0
                with log_path.open("r", encoding="utf-8", errors="replace") as handle:
                    handle.seek(position)
                    chunk = handle.read()
                    position = handle.tell()
                for line in chunk.splitlines():
                    yield f"data: {json.dumps({'line': line})}\n\n"
            except FileNotFoundError:
                pass
            except OSError as error:
                yield (
                    "event: monitor_error\n"
                    f"data: {json.dumps({'message': str(error)})}\n\n"
                )
            yield ": keepalive\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
def index():
    index_path = FRONTEND_DIST / "index.html"
    if index_path.is_file():
        return FileResponse(index_path)
    return JSONResponse(
        {
            "message": "Dashboard frontend is not built yet.",
            "build": "cd dashboard/frontend && npm install && npm run build",
            "api": "/api/runs",
            "health": "/api/health",
            "config": "/api/config",
            "system": "/api/system",
        }
    )


if (FRONTEND_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")
