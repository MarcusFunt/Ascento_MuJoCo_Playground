"""Phase 2: multi-root live PPO discovery, freshness and SSE changes."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest
from dashboard import deep_live
from dashboard.monitor import invalidate_discovery_cache
from fastapi import HTTPException


def _sample(run: Path, *, sampled_at: float, sequence: int = 3, step: int = 200):
    root = run / "task" / "stamp"
    root.mkdir(parents=True, exist_ok=True)
    (root / "training.log").write_text("Learning iteration 5/999\n")
    (root / "live_monitor.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": "actual_ppo_rollout",
                "sampled_at": sampled_at,
                "sequence": sequence,
                "global_step": step,
                "environments": [{"environment_id": 0}],
            }
        )
    )
    return root


def test_worktree_sources_are_unique_and_only_fresh_is_live(tmp_path):
    now = time.time()
    primary = tmp_path / "main"
    v6 = tmp_path / "v6"
    _sample(primary / "v5_run", sampled_at=now - 500)
    _sample(v6 / "overnight_v6", sampled_at=now)
    roots = "v6=" + str(v6)
    result = deep_live.discover_live_sources(primary, roots, now=now)
    assert len(result["live_runs"]) == 1
    assert result["live_runs"][0]["root"] == "v6"
    assert result["live_runs"][0]["state"] == "streaming"
    assert result["live_runs"][0]["id"].startswith("v6~")
    assert any(
        entry["root"] == "main" and entry["state"] == "stale"
        for entry in result["recent_live_sources"]
    )
    source_id = result["live_runs"][0]["id"]
    assert deep_live.resolve_live_run(source_id, primary, roots).is_relative_to(v6)
    actual = deep_live.current_frame(source_id, primary, roots)
    assert actual["available"] is True and actual["source"] == "actual_ppo_rollout"
    assert len({x["id"] for x in result["live_runs"] + result["recent_live_sources"]}) == 2
    stale = deep_live.discover_live_sources(primary, roots, now=now + 30)
    assert stale["live_runs"] == []
    assert any(x["root"] == "v6" for x in stale["recent_live_sources"])
    invalidate_discovery_cache(primary)
    invalidate_discovery_cache(v6)


@pytest.mark.parametrize(
    "source_id",
    [
        "../../root",
        "v6~../../outside",
        "v6~12345678901x",
        "bad/name~0123456789ab",
        "v6~nope",
        "v6~0123456789ab/../../x",
        "bad~0123456789ab",
    ],
)
def test_rejects_unconfigured_sources(tmp_path, source_id):
    with pytest.raises(HTTPException) as err:
        deep_live.resolve_live_run(source_id, tmp_path, "v6=" + str(tmp_path / "v6"))
    assert err.value.status_code == 404


@pytest.mark.parametrize(
    "config", ["bad", "main=/tmp", "v6=relative", "V6=/tmp", "v6=/tmp;v6=/else"]
)
def test_bad_root_configuration_rejected(tmp_path, config):
    with pytest.raises(ValueError):
        deep_live.configured_roots(tmp_path, config)


def test_missing_and_stale_never_called_live(tmp_path):
    assert deep_live.frame_for_run(tmp_path)["state"] == "unavailable"
    run = _sample(tmp_path / "run", sampled_at=time.time() - 20)
    result = deep_live.frame_for_run(run)
    assert not result["available"]
    assert result["state"] == "stale"
    assert result["age_seconds"] >= 10


def test_stream_emits_stale_transition_then_resume(tmp_path, monkeypatch):
    states = iter(
        [
            {
                "available": True,
                "state": "streaming",
                "source": "actual_ppo_rollout",
                "sampled_at": 100.0,
                "sequence": 1,
                "global_step": 10,
            },
            {"available": False, "state": "stale", "reason": "stopped"},
            {"available": False, "state": "stale", "reason": "stopped"},
            {
                "available": True,
                "state": "streaming",
                "source": "actual_ppo_rollout",
                "sampled_at": 130.0,
                "sequence": 1,
                "global_step": 2,
            },
        ]
    )
    monkeypatch.setattr(deep_live, "frame_for_run", lambda *_args: next(states))

    class Request:
        async def is_disconnected(self):
            return False

    async def sample():
        stream = deep_live.live_events(tmp_path, Request(), poll_s=0)
        data = [await anext(stream) for _ in range(4)]
        await stream.aclose()
        return data

    messages = asyncio.run(sample())
    assert messages[0].startswith("event: frame")
    assert messages[1].startswith("event: status")
    assert messages[2].startswith(": keepalive")
    assert messages[3].startswith("event: frame")
    assert json.loads(messages[3].split("data: ", 1)[1])["global_step"] == 2
