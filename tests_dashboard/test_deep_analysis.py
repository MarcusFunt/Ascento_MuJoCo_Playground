"""Deep Analysis is read-only and never confuses similarly numbered checkpoints."""

from __future__ import annotations

import json

import numpy as np
import pytest
from dashboard import deep_analysis
from fastapi import HTTPException


@pytest.fixture
def fixture(tmp_path):
    root = tmp_path / "logs" / "rsl_rl"
    captures = tmp_path / "captures"
    a = root / "run-A" / "task" / "2026-10-14"
    b = root / "run-B" / "task" / "2026-10-14"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    captures.mkdir()
    checkpoint_a = a / "model_16800.pt"
    checkpoint_b = b / "model_16800.pt"
    checkpoint_a.write_bytes(b"A-policy")
    checkpoint_b.write_bytes(b"B-policy")
    npz = captures / "recorded.npz"
    n = 1200
    np.savez_compressed(
        npz,
        meta_checkpoint=np.asarray(str(checkpoint_a)),
        meta_checkpoint_sha256=np.asarray("sha-a"),
        meta_scenario_id=np.asarray("goal_settle_v5/precision_short/0"),
        meta_task=np.asarray("Ascento-Generalist-Flat"),
        time=np.arange(n, dtype=np.float64) * 0.01,
        root_pos=np.zeros((n, 3), dtype=np.float32),
        target_xy=np.ones((n, 2), dtype=np.float32) * 0.025,
        tilt_rad=np.ones(n, dtype=np.float32) * 0.12,
        target_error_m=np.linspace(1, 0.026, n).astype(np.float32),
        planar_speed_mps=np.zeros(n, dtype=np.float32),
        contacts=np.ones((n, 2), dtype=np.float32),
        evaluator_stop_tilt=np.zeros(n, dtype=np.float32),
        evaluator_stop_dwell_s=np.zeros(n, dtype=np.float32),
        reward_total=np.ones(n, dtype=np.float32),
        reward_terms_rate=np.ones((n, 1), dtype=np.float32),
        reward_dt=np.ones(n, dtype=np.float32) * 0.01,
        meta_reward_term_names=np.asarray(json.dumps(["alive"])),
    )
    return root, captures, checkpoint_a, checkpoint_b


def test_catalog_does_not_mix_same_iteration_between_runs(fixture):
    root, captures, a, b = fixture
    deep_analysis._CATALOG_CACHE = None
    c = deep_analysis.catalog(
        root,
        captures,
        [
            {
                "id": "live-A",
                "state": "running",
                "display_name": "Running in-memory A",
                "iteration": 17700,
            },
            {"id": "stale", "state": "running", "display_name": "Old", "stale": True},
        ],
    )
    assert len(c["checkpoints"]) == 2
    key_a = a.relative_to(root).as_posix()
    key_b = b.relative_to(root).as_posix()
    assert c["checkpoints"][0]["iteration"] == c["checkpoints"][1]["iteration"] == 16800
    assert len(c["recordings"][key_a]) == 1
    assert key_b not in c["recordings"]
    assert c["live_runs"][0]["id"] == "live-A"
    assert len(c["live_runs"]) == 1


def test_capture_is_decimated_and_provenance_verified(fixture):
    root, captures, a, b = fixture
    key_a = a.relative_to(root).as_posix()
    key_b = b.relative_to(root).as_posix()
    result = deep_analysis.episode(captures, root, key_a, "recorded.npz")
    assert result["frames"] == 1200
    assert len(result["time"]) <= 601
    assert result["stats"]["tilt_rad"]["mean"] == pytest.approx(0.12, abs=1e-6)
    assert result["stats"]["target_error_m_final_2s"]["mean"] < 0.2
    assert result["has_evaluator_signals"]
    assert "00 alive" in result["reward_terms"]
    with pytest.raises(HTTPException) as exc:
        deep_analysis.episode(captures, root, key_b, "recorded.npz")
    assert exc.value.status_code == 409


def test_capture_traversal_and_symlink_rejected(fixture, tmp_path):
    root, captures, a, b = fixture
    key = a.relative_to(root).as_posix()
    with pytest.raises(HTTPException) as exc:
        deep_analysis.episode(captures, root, key, "../outside.npz")
    assert exc.value.status_code == 404
    other = tmp_path / "outside.npz"
    other.write_bytes(b"secret")
    (captures / "evil.npz").symlink_to(other)
    with pytest.raises(HTTPException) as exc:
        deep_analysis.episode(captures, root, key, "evil.npz")
    assert exc.value.status_code == 404


def test_missing_and_unprovenanced_data_never_borrowed(fixture):
    root, captures, a, b = fixture
    deep_analysis._CATALOG_CACHE = None
    (captures / "stray.npz").write_bytes(b"invalid")
    catalog = deep_analysis.catalog(root, captures)
    assert len(catalog["recordings"]) == 1
    with pytest.raises(HTTPException):
        deep_analysis.episode(captures, root, "nonexistent/model_16800.pt", "recorded.npz")
