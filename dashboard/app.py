"""FastAPI service for remotely monitoring Ascento PPO training."""
from __future__ import annotations

import asyncio
import json
import os
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from dashboard.config import load_config, validate_startup
from dashboard.health import (
    discover_dashboard_runs,
    list_dashboard_summaries,
    load_dashboard_records,
    resolve_dashboard_run,
    summarize_dashboard_run,
)
from dashboard.monitor import tail_lines, training_log_path
from dashboard.versioning import annotate_run_summary, current_repository_version

CONFIG = load_config()
STARTUP_WARNINGS = validate_startup(CONFIG)
ARTIFACT_ROOT = CONFIG.artifact_root
FRONTEND_DIST = CONFIG.frontend_dist

app = FastAPI(title="Ascento Training Monitor", version="1.3")


def _run(run_id: str):
    try:
        return resolve_dashboard_run(ARTIFACT_ROOT, run_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="training run not found") from error
    except OSError as error:
        raise HTTPException(
            status_code=500,
            detail=f"failed to scan artifact root {ARTIFACT_ROOT}: {error}",
        ) from error


def _artifact_health() -> list[str]:
    problems: list[str] = []
    if not ARTIFACT_ROOT.exists():
        problems.append(f"artifact root does not exist: {ARTIFACT_ROOT}")
    elif not ARTIFACT_ROOT.is_dir():
        problems.append(f"artifact root is not a directory: {ARTIFACT_ROOT}")
    else:
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
            annotate_run_summary(summary, ref.path, ARTIFACT_ROOT)
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


@app.get("/api/runs/{run_id}")
def run_status(run_id: str):
    ref = _run(run_id)
    summary = summarize_dashboard_run(
        ref.path,
        ARTIFACT_ROOT,
        stale_after_seconds=CONFIG.stale_after_seconds,
        detailed=True,
    )
    return annotate_run_summary(summary, ref.path, ARTIFACT_ROOT)


@app.get("/api/runs/{run_id}/summary.json")
def run_summary(run_id: str):
    ref = _run(run_id)
    summary = summarize_dashboard_run(
        ref.path,
        ARTIFACT_ROOT,
        stale_after_seconds=CONFIG.stale_after_seconds,
        detailed=True,
    )
    annotate_run_summary(summary, ref.path, ARTIFACT_ROOT)
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
        }
    )


if (FRONTEND_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")
