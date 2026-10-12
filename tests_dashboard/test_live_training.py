"""Live training API contract, SSE framing, and stale-data tests."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from dashboard import app as dashboard_app
from dashboard.live_training import read_live_frame


def _create_frame(run_path):
    directory = run_path / "ascento_task" / "2026-10-12_13-00-00"
    directory.mkdir(parents=True)
    (directory / "live_monitor.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": "actual_ppo_rollout",
                "sampled_at": 100.0,
                "sequence": 42,
                "global_step": 105,
                "environments": [{"environment_id": 0}],
            }
        )
    )


def test_get_live_training_without_capture(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard_app, "_run", lambda _id: SimpleNamespace(path=tmp_path))
    response = dashboard_app.live_training_monitor("example")
    assert response == {"available": False, "reason": "No sampled PPO telemetry from this run"}


def test_live_training_endpoint_and_sse(tmp_path, monkeypatch):
    _create_frame(tmp_path)
    monkeypatch.setattr(dashboard_app, "_run", lambda _id: SimpleNamespace(path=tmp_path))
    response = dashboard_app.live_training_monitor("example")
    assert response["available"] and response["sequence"] == 42

    class Request:
        async def is_disconnected(self):
            return False

    async def first_event():
        stream = await dashboard_app.live_training_monitor_stream("example", Request())
        assert stream.media_type == "text/event-stream"
        iterator = stream.body_iterator
        payload = await anext(iterator)
        await iterator.aclose()
        return payload

    message = asyncio.run(first_event())
    assert message.startswith("id: 42\nevent: frame\ndata: ")
    assert message.endswith("\n\n")
    assert json.loads(message.split("data: ", 1)[1])["sequence"] == 42


def test_live_snapshot_rejects_oversized_data(tmp_path):
    path = tmp_path / "live_monitor.json"
    path.write_text(" " * 66000)
    assert read_live_frame(tmp_path)["available"] is False
