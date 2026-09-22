import os
import time
from pathlib import Path

import pytest

from ascento_mjlab.viewer.checkpoints import discover_checkpoints, resolve_run_checkpoint


def _checkpoint(root: Path, name: str, *, age: float = 10.0, size: int = 16) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    stamp = time.time() - age
    os.utime(path, (stamp, stamp))
    return path


def test_checkpoint_discovery_orders_by_iteration_and_ignores_unstable_files(tmp_path):
    _checkpoint(tmp_path, "nested/model_200.pt", age=20)
    _checkpoint(tmp_path, "model_100.pt", age=20)
    _checkpoint(tmp_path, "model_300.pt", age=0.1)
    _checkpoint(tmp_path, "other.pt", age=20)

    items = discover_checkpoints(tmp_path, stable_age_seconds=2.0)

    assert [item.relative_path for item in items] == [
        "model_100.pt",
        "nested/model_200.pt",
    ]
    assert [item.iteration for item in items] == [100, 200]


def test_latest_and_explicit_checkpoint_resolution_are_run_scoped(tmp_path):
    _checkpoint(tmp_path, "model_100.pt")
    _checkpoint(tmp_path, "nested/model_200.pt")

    latest = resolve_run_checkpoint(tmp_path, "latest", stable_age_seconds=0)
    explicit = resolve_run_checkpoint(
        tmp_path,
        "model_100.pt",
        stable_age_seconds=0,
    )

    assert latest.relative_path == "nested/model_200.pt"
    assert explicit.iteration == 100


@pytest.mark.parametrize("selection", ["../escape.pt", "/tmp/escape.pt"])
def test_checkpoint_resolution_rejects_path_escape(tmp_path, selection):
    _checkpoint(tmp_path, "model_100.pt")

    with pytest.raises(ValueError, match="run-relative"):
        resolve_run_checkpoint(tmp_path, selection, stable_age_seconds=0)
