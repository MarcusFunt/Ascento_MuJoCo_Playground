"""Fresh read-only live summary tests, including missing/stale metadata."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone

import pytest
from dashboard import live_overview


def _source() -> dict:
    return {
        "id": "v7~abc123def456",
        "label": "V7 / overnight_v7/task/2026-10-15_03-51-57",
        "root": "v7",
        "state": "streaming",
        "sampled_at": time.time(),
    }


def test_empty_source_has_no_gpu_query(monkeypatch, tmp_path):
    monkeypatch.setattr(live_overview.deep_live, "discover_live_sources", lambda _: {"live_runs": []})
    monkeypatch.setattr(live_overview, "_gpu", lambda: pytest.fail("Do not poll GPU without a live run"))
    assert live_overview.summary(tmp_path) == {"runs": [], "gpu": None}


def test_live_v7_reads_fresh_status_reward_trend_and_curriculum(monkeypatch, tmp_path):
    root = tmp_path / "metadata"
    root.mkdir()
    now = datetime.now(timezone.utc)
    status = {
        "updated_at": now.isoformat(),
        "last_iteration": 14800,
        "last_mean_reward": -25.5,
        "training_cutoff_local": (now + timedelta(hours=17)).isoformat(),
        "deadline_local": (now + timedelta(hours=19)).isoformat(),
    }
    (root / "status.json").write_text(json.dumps(status))
    (root / "trainer.log").write_text(
        "Learning iteration 14799/50000\n"
        "Mean reward: -28.5\n"
        "Episode_Metrics/generalist_curriculum_goal_mix_stage: 1.0000\n"
        "Learning iteration 14800/50000\n"
        "Mean reward: -25.5\n"
        "Episode_Metrics/generalist_curriculum_goal_mix_stage: 2.0000\n"
    )
    monkeypatch.setenv("ASCENTO_DEEP_LIVE_METADATA_ROOTS", "v7=" + str(root))
    monkeypatch.setattr(live_overview.deep_live, "discover_live_sources", lambda _: {"live_runs": [_source()]})
    monkeypatch.setattr(live_overview, "_gpu", lambda: {"utilization_percent": 35.0})
    result = live_overview.summary(tmp_path)
    row = result["runs"][0]
    assert row["id"] == _source()["id"]
    assert row["iteration"] == 14800
    assert row["mean_reward"] == -25.5
    assert row["reward_trend"] == [
        {"iteration": 14799, "reward": -28.5}, {"iteration": 14800, "reward": -25.5}
    ]
    assert row["curriculum_stage"] == 2
    assert 16 * 3600 < row["training_remaining_seconds"] <= 17 * 3600
    assert result["gpu"]["utilization_percent"] == 35.0


def test_stale_and_unmounted_metrics_never_presented_as_fresh(monkeypatch, tmp_path):
    root = tmp_path / "metadata"
    root.mkdir()
    (root / "status.json").write_text(json.dumps({
        "updated_at": (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(),
        "last_iteration": 10000, "last_mean_reward": 5000.0,
    }))
    log = root / "trainer.log"
    log.write_text("Learning iteration 10/50000\nMean reward: 999\n")
    os.utime(log, (time.time() - 600, time.time() - 600))
    monkeypatch.setenv("ASCENTO_DEEP_LIVE_METADATA_ROOTS", "v7=" + str(root))
    monkeypatch.setattr(live_overview.deep_live, "discover_live_sources", lambda _: {"live_runs": [_source()]})
    monkeypatch.setattr(live_overview, "_gpu", lambda: None)
    row = live_overview.summary(tmp_path)["runs"][0]
    assert row["iteration"] is None and row["mean_reward"] is None
    assert row["reward_trend"] == [] and row["curriculum_stage"] is None


@pytest.mark.parametrize("raw", ["badalias", "v7=relative", "v7=/tmp;v7=/root", "!bad=/tmp"])
def test_invalid_metadata_roots_rejected(raw):
    with pytest.raises(ValueError):
        live_overview.metadata_roots(raw)
