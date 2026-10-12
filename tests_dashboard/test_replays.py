"""Replay API tests: safe same-origin RRD serving and controlled NPZ export."""

from __future__ import annotations

from pathlib import Path

import pytest
from dashboard import replays
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    root = tmp_path / "captures"
    root.mkdir()
    monkeypatch.setattr(replays, "CAPTURE_ROOT", root)
    app = FastAPI()
    app.include_router(replays.router)
    return TestClient(app), root


def test_replay_listing_and_http_range(client):
    browser, root = client
    episode = root / "episode"
    episode.mkdir()
    (episode / "take_000.npz").write_bytes(b"numpy")
    (episode / "take_000.rrd").write_bytes(b"rerun-data")
    response = browser.get("/api/replays")
    assert response.status_code == 200
    item = response.json()["replays"][0]
    assert item["name"] == "episode/take_000"
    assert item["source"] == "episode/take_000.npz"
    assert item["rrd_url"] == "/api/replays/files/episode/take_000.rrd"
    stream = browser.get(item["rrd_url"], headers={"Range": "bytes=0-4"})
    assert stream.status_code == 206
    assert stream.content == b"rerun"
    assert stream.headers["content-range"] == "bytes 0-4/10"


def test_missing_root_is_empty(client):
    browser, root = client
    root.rmdir()
    assert browser.get("/api/replays").json() == {"replays": []}


def test_traversal_and_symlinks_never_escape_root(client, tmp_path):
    browser, root = client
    forbidden = tmp_path / "secret.rrd"
    forbidden.write_bytes(b"secret")
    (root / "outside.rrd").symlink_to(forbidden)
    assert browser.get("/api/replays/files/outside.rrd").status_code == 404
    with pytest.raises(HTTPException) as exc:
        replays._safe_file("../secret.rrd", ".rrd")
    assert exc.value.status_code == 404


def test_npz_export_requires_explicit_request_and_returns_viewable_url(client, monkeypatch):
    browser, root = client
    path = root / "sample.npz"
    path.write_bytes(b"test-data")
    called = []

    def fake_export(source: Path, target: Path):
        called.append(source)
        target.write_bytes(b"rrd-data")

    monkeypatch.setattr("ascento_mjlab.tools.export_rerun.export_rerun", fake_export)
    assert browser.post("/api/replays/export/sample.npz").status_code == 403
    response = browser.post("/api/replays/export/sample.npz", headers={"X-Ascento-Control": "1"})
    assert response.status_code == 200, response.text
    assert response.json()["rrd_url"] == "/api/replays/files/sample.rrd"
    assert browser.get(response.json()["rrd_url"]).content == b"rrd-data"
    assert called == [path]
    browser.post("/api/replays/export/sample.npz", headers={"X-Ascento-Control": "1"})
    assert len(called) == 1


def test_replay_export_rejects_arbitrary_paths(client):
    browser, root = client
    (root / "other.txt").write_text("nope")
    assert (
        browser.post(
            "/api/replays/export/other.txt", headers={"X-Ascento-Control": "1"}
        ).status_code
        == 404
    )
