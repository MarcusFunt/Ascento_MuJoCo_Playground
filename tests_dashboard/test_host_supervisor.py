import json

from scripts.host_supervisor import HostSupervisor


def test_repository_status_reports_remote_delta_and_incoming_commits(monkeypatch, tmp_path):
    supervisor = HostSupervisor(tmp_path)

    def fake_git(*args, **_kwargs):
        if args[:4] == ("fetch", "--quiet", "--prune", "origin"):
            return ""
        if args == ("rev-parse", "HEAD"):
            return "local123"
        if args == ("branch", "--show-current"):
            return "main"
        if args == ("status", "--porcelain", "--untracked-files=no"):
            return ""
        if args == ("rev-parse", "origin/main"):
            return "remote456"
        if args == ("rev-list", "--left-right", "--count", "HEAD...origin/main"):
            return "0\t2"
        if args[:3] == ("log", "--max-count=8", "--format=%H%x09%s"):
            return "remote456\tFix recovery gate\nremote455\tImprove dashboard"
        raise AssertionError(args)

    monkeypatch.setattr(supervisor, "_git", fake_git)
    monkeypatch.setattr(supervisor, "active_runs", lambda: [])
    monkeypatch.setattr(supervisor, "update_state", lambda: {"status": "idle"})
    monkeypatch.setattr(
        supervisor,
        "_tailscale_status",
        lambda: {"enabled": True, "connected": True, "dns_name": "ascento.example.ts.net"},
    )

    status = supervisor.repository_status(refresh=True)

    assert status["repository"]["local_commit"] == "local123"
    assert status["repository"]["remote_commit"] == "remote456"
    assert status["repository"]["behind_by"] == 2
    assert status["repository"]["ahead_by"] == 0
    assert status["repository"]["update_available"] is True
    assert status["repository"]["incoming_commits"][0]["subject"] == "Fix recovery gate"
    assert status["can_update"] is True
    assert status["update_blockers"] == []


def test_repository_update_is_blocked_by_active_training(monkeypatch, tmp_path):
    supervisor = HostSupervisor(tmp_path)

    def fake_git(*args, **_kwargs):
        values = {
            ("rev-parse", "HEAD"): "local123",
            ("branch", "--show-current"): "main",
            ("status", "--porcelain", "--untracked-files=no"): "",
            ("rev-parse", "origin/main"): "remote456",
            ("rev-list", "--left-right", "--count", "HEAD...origin/main"): "0 1",
        }
        if args[:4] == ("fetch", "--quiet", "--prune", "origin"):
            return ""
        if args[:3] == ("log", "--max-count=8", "--format=%H%x09%s"):
            return "remote456\tUpdate"
        return values[args]

    monkeypatch.setattr(supervisor, "_git", fake_git)
    monkeypatch.setattr(
        supervisor,
        "active_runs",
        lambda: [{"name": "Recovery long run", "state": "running", "path": "run-a"}],
    )
    monkeypatch.setattr(supervisor, "update_state", lambda: {"status": "idle"})
    monkeypatch.setattr(
        supervisor, "_tailscale_status", lambda: {"enabled": False, "connected": False}
    )

    status = supervisor.repository_status(refresh=True)

    assert status["repository"]["update_available"] is True
    assert status["can_update"] is False
    assert "one or more training runs are active" in status["update_blockers"]


def test_active_run_scan_reads_launcher_statuses(tmp_path):
    root = tmp_path / "logs" / "rsl_rl"
    active = root / "run-a"
    finished = root / "run-b"
    active.mkdir(parents=True)
    finished.mkdir(parents=True)
    (active / "run_status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
    (active / "run_metadata.json").write_text(
        json.dumps({"display_name": "Recovery validation"}), encoding="utf-8"
    )
    (finished / "run_status.json").write_text(json.dumps({"state": "finished"}), encoding="utf-8")

    runs = HostSupervisor(tmp_path).active_runs()

    assert runs == [{"name": "Recovery validation", "state": "running", "path": "run-a"}]


def test_dispatch_exposes_only_status_and_update(monkeypatch, tmp_path):
    supervisor = HostSupervisor(tmp_path)
    monkeypatch.setattr(supervisor, "repository_status", lambda **_kwargs: {"can_update": False})

    assert supervisor.dispatch({"op": "status"}) == {
        "ok": True,
        "result": {"can_update": False},
    }
    rejected = supervisor.dispatch({"op": "shell", "command": "rm -rf /"})
    assert rejected["ok"] is False
    assert rejected["code"] == "unsupported"
