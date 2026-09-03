"""FastAPI service for remotely monitoring Ascento PPO training."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from dashboard.monitor import (
    list_run_summaries,
    load_training_records,
    resolve_run,
    summarize_run,
    tail_lines,
    training_log_path,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path(
    os.environ.get("ASCENTO_ARTIFACT_ROOT", ROOT / "training" / "artifacts")
).expanduser().resolve()
FRONTEND_DIST = Path(
    os.environ.get("ASCENTO_DASHBOARD_DIST", Path(__file__).parent / "frontend" / "dist")
).expanduser().resolve()

app = FastAPI(title="Ascento Training Monitor", version="1.0")


def _run(run_id: str):
    try:
        return resolve_run(ARTIFACT_ROOT, run_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="training run not found") from error


@app.get("/api/health")
def health():
    return {"ok": True, "artifact_root": str(ARTIFACT_ROOT)}


@app.get("/api/runs")
def runs():
    return {"runs": list_run_summaries(ARTIFACT_ROOT)}


@app.get("/api/runs/{run_id}")
def run_status(run_id: str):
    ref = _run(run_id)
    return summarize_run(ref.path, ARTIFACT_ROOT)


@app.get("/api/runs/{run_id}/telemetry")
def telemetry(run_id: str, limit: int = 2000):
    ref = _run(run_id)
    limit = max(1, min(limit, 20_000))
    return {"records": load_training_records(ref.path, limit=limit)}


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
                yield f"event: monitor_error\ndata: {json.dumps({'message': str(error)})}\n\n"
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
        }
    )


if (FRONTEND_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")
