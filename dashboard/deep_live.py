"""Read-only PPO stream discovery across explicitly configured worktrees."""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse

from dashboard.live_training import find_live_frame_file, read_live_frame
from dashboard.monitor import RunRef, discover_runs

_ALIAS = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_RUNID = re.compile(r"^[a-f0-9]{12}$")


def configured_roots(primary: Path, raw: str | None = None) -> dict[str, Path]:
    """Operator-defined 'alias=/read-only/artifact-root' sources separated by semicolons."""
    roots = {"main": primary.resolve()}
    for entry in filter(
        None,
        (
            s.strip()
            for s in (os.getenv("ASCENTO_DEEP_LIVE_ROOTS", "") if raw is None else raw).split(";")
        ),
    ):
        alias, sep, directory = entry.partition("=")
        if (
            not sep
            or not _ALIAS.fullmatch(alias)
            or alias in roots
            or not Path(directory).is_absolute()
        ):
            raise ValueError("Invalid Deep Analysis live source root configuration")
        roots[alias] = Path(directory).resolve()
    return roots


def _sources(alias: str, root: Path) -> list[tuple[str, RunRef, Path]]:
    if not root.is_dir():
        return []
    discovered = []
    seen = set()
    for ref in discover_runs(root):
        f = find_live_frame_file(ref.path)
        if f is None or not f.resolve().is_relative_to(root) or f.resolve() in seen:
            continue
        seen.add(f.resolve())
        discovered.append((alias + "~" + ref.id, ref, f))
    return discovered


def frame_for_run(run: Path, *, now: float | None = None) -> dict[str, Any]:
    data = read_live_frame(run, now=now)
    if data.get("available") and data.get("stale"):
        return {
            **data,
            "available": False,
            "state": "stale",
            "reason": "PPO telemetry has stopped updating",
        }
    if not data.get("available"):
        return {**data, "state": "unavailable"}
    return {**data, "state": "streaming"}


def discover_live_sources(
    primary: Path, raw: str | None = None, *, now: float | None = None
) -> dict[str, list[dict]]:
    live = []
    recent = []
    for alias, root in configured_roots(primary, raw).items():
        for identity, ref, _ in _sources(alias, root):
            frame = frame_for_run(ref.path, now=now)
            item = {
                "id": identity,
                "root": alias,
                "label": alias.upper() + " / " + ref.relative_path,
                "lineage": ref.relative_path,
                "state": frame["state"],
                "available": frame["available"],
                "reason": frame.get("reason"),
                "sampled_at": frame.get("sampled_at"),
                "age_seconds": frame.get("age_seconds"),
                "iteration": frame.get("global_step"),
                "sequence": frame.get("sequence"),
            }
            (live if frame["available"] else recent).append(item)
    live.sort(key=lambda x: x.get("sampled_at") or 0, reverse=True)
    recent.sort(key=lambda x: x.get("sampled_at") or 0, reverse=True)
    # Old experiments should not overwhelm the source selector. Show the two
    # most recent stale monitors per worktree, not a long history of dead runs.
    concise = []
    per_root: dict[str, int] = {}
    for item in recent:
        alias = str(item["root"])
        if per_root.get(alias, 0) >= 2:
            continue
        concise.append(item)
        per_root[alias] = per_root.get(alias, 0) + 1
        if len(concise) == 8:
            break
    return {"live_runs": live, "recent_live_sources": concise}


def resolve_live_run(source_id: str, primary: Path, raw: str | None = None) -> Path:
    if len(source_id) > 64 or "~" not in source_id:
        raise HTTPException(404, "Live source not found")
    alias, runid = source_id.split("~", 1)
    if not _ALIAS.fullmatch(alias) or not _RUNID.fullmatch(runid):
        raise HTTPException(404, "Live source not found")
    root = configured_roots(primary, raw).get(alias)
    if root is None or not root.is_dir():
        raise HTTPException(404, "Live source not found")
    for identity, ref, _ in _sources(alias, root):
        if identity == source_id:
            return ref.path
    raise HTTPException(404, "Live source not found")


def current_frame(source_id: str, primary: Path, raw: str | None = None) -> dict:
    return frame_for_run(resolve_live_run(source_id, primary, raw))


def _sse(event: str, payload: dict) -> str:
    return (
        "event: "
        + event
        + "\ndata: "
        + json.dumps(payload, separators=(",", ":"), allow_nan=False)
        + "\n\n"
    )


async def live_events(run: Path, request: Request, *, poll_s: float = 0.75):
    """Emit new samples or clear status transitions; never replay a stale frame."""
    last_key = None
    last_state = None
    while not await request.is_disconnected():
        frame = frame_for_run(run)
        state = frame["state"]
        if frame["available"]:
            key = (frame.get("sampled_at"), frame.get("sequence"), frame.get("global_step"))
            if key != last_key:
                last_key, last_state = key, "streaming"
                yield _sse("frame", frame)
            else:
                yield ": keepalive\n\n"
        elif state != last_state:
            last_state, last_key = state, None
            yield _sse(
                "status",
                {
                    "available": False,
                    "state": state,
                    "reason": frame.get("reason"),
                    "age_seconds": frame.get("age_seconds"),
                },
            )
        else:
            yield ": keepalive\n\n"
        await asyncio.sleep(poll_s)


def streaming_response(
    source_id: str, primary: Path, request: Request, raw: str | None = None
) -> StreamingResponse:
    return StreamingResponse(
        live_events(resolve_live_run(source_id, primary, raw), request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
