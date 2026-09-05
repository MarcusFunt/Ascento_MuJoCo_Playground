#!/usr/bin/env python3
"""Minimal host-side control boundary for repository status and maintenance updates.

The dashboard container talks to this process over a Unix-domain socket.  The
protocol deliberately exposes only a small fixed command set; it is not a shell
proxy and never accepts arbitrary commands, paths, branches, or Docker actions.
"""
from __future__ import annotations

import argparse
import json
import os
import socketserver
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ACTIVE_RUN_STATES = {"starting", "running", "stopping"}
STATUS_CACHE_SECONDS = 30.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


class HostSupervisor:
    """Fixed-purpose host controller used by the dashboard system API."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root.expanduser().resolve()
        self.maintenance_root = self.repo_root / ".maintenance"
        self.update_state_path = self.maintenance_root / "update-state.json"
        self.update_log_path = self.maintenance_root / "update.log"
        self.tailscale_marker = self.maintenance_root / "tailscale-enabled"
        self._update_lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._status_cache: tuple[float, dict[str, Any]] | None = None

    def _run(
        self,
        command: list[str],
        *,
        timeout: float = 10.0,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=self.repo_root,
            check=check,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def _git(self, *args: str, timeout: float = 10.0, check: bool = True) -> str:
        result = self._run(["git", *args], timeout=timeout, check=check)
        return result.stdout.strip()

    def active_runs(self) -> list[dict[str, str]]:
        root = self.repo_root / "logs" / "rsl_rl"
        if not root.is_dir():
            return []
        active: list[dict[str, str]] = []
        for path in root.rglob("run_status.json"):
            status = _load_json(path)
            state = str(status.get("state") or "")
            if state not in ACTIVE_RUN_STATES:
                continue
            run_dir = path.parent
            metadata = _load_json(run_dir / "run_metadata.json")
            active.append(
                {
                    "name": str(metadata.get("display_name") or run_dir.name),
                    "state": state,
                    "path": run_dir.relative_to(root).as_posix(),
                }
            )
        return active

    def update_state(self) -> dict[str, Any]:
        state = _load_json(self.update_state_path)
        if not state:
            return {"status": "idle"}
        if state.get("status") == "running" and isinstance(state.get("pid"), int):
            try:
                os.kill(int(state["pid"]), 0)
            except ProcessLookupError:
                # A supervisor crash can lose the waiter that records the final
                # exit code. Do not claim success when the outcome is unknown.
                state = {
                    **state,
                    "status": "unknown",
                    "finished_at": _now(),
                    "message": "update process exited while supervisor was not tracking it",
                }
                _write_json(self.update_state_path, state)
            except (PermissionError, OSError):
                pass
        return state

    def _tailscale_status(self) -> dict[str, Any]:
        enabled = self.tailscale_marker.is_file()
        result: dict[str, Any] = {
            "enabled": enabled,
            "connected": False,
            "dns_name": None,
            "ips": [],
            "url": None,
        }
        if not enabled:
            return result
        try:
            container_id = self._run(
                [
                    "docker",
                    "ps",
                    "--filter",
                    "label=com.docker.compose.service=tailscale-dashboard",
                    "--format",
                    "{{.ID}}",
                ],
                timeout=4.0,
            ).stdout.splitlines()[0].strip()
            raw = self._run(
                ["docker", "exec", container_id, "tailscale", "status", "--json"],
                timeout=5.0,
            ).stdout
            status = json.loads(raw)
            self_status = status.get("Self") if isinstance(status.get("Self"), dict) else {}
            dns_name = str(self_status.get("DNSName") or "").rstrip(".") or None
            ips = self_status.get("TailscaleIPs") if isinstance(self_status.get("TailscaleIPs"), list) else []
            connected = str(status.get("BackendState") or "").lower() == "running" or bool(ips)
            port = _env_file(self.maintenance_root / "compose.env").get(
                "ASCENTO_DASHBOARD_PORT", "8000"
            )
            result.update(
                {
                    "connected": connected,
                    "dns_name": dns_name,
                    "ips": [str(value) for value in ips],
                    "url": f"http://{dns_name}:{port}" if dns_name else None,
                }
            )
        except (IndexError, json.JSONDecodeError, OSError, subprocess.SubprocessError) as error:
            result["error"] = str(error)
        return result

    def repository_status(self, *, refresh: bool = False) -> dict[str, Any]:
        with self._status_lock:
            if not refresh and self._status_cache is not None:
                cached_at, cached = self._status_cache
                if time.monotonic() - cached_at < STATUS_CACHE_SECONDS:
                    return cached

            remote_error: str | None = None
            try:
                # Fetching the remote ref lets us calculate ahead/behind and list
                # incoming commits without modifying the checked-out files.
                self._git("fetch", "--quiet", "--prune", "origin", "main", timeout=30.0)
            except (OSError, subprocess.SubprocessError) as error:
                remote_error = str(error)

            try:
                local_commit = self._git("rev-parse", "HEAD")
                branch = self._git("branch", "--show-current") or None
                dirty = bool(self._git("status", "--porcelain", "--untracked-files=no"))
            except (OSError, subprocess.SubprocessError) as error:
                return {
                    "ok": False,
                    "error": f"cannot inspect local repository: {error}",
                    "update": self.update_state(),
                    "tailscale": self._tailscale_status(),
                }

            remote_commit: str | None = None
            ahead = behind = None
            incoming: list[dict[str, str]] = []
            try:
                remote_commit = self._git("rev-parse", "origin/main")
                counts = self._git("rev-list", "--left-right", "--count", "HEAD...origin/main")
                left, right = counts.split()
                ahead, behind = int(left), int(right)
                log_output = self._git(
                    "log",
                    "--max-count=8",
                    "--format=%H%x09%s",
                    "HEAD..origin/main",
                )
                for line in log_output.splitlines():
                    if "\t" in line:
                        commit, subject = line.split("\t", 1)
                        incoming.append({"commit": commit, "subject": subject})
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                remote_error = remote_error or str(error)

            active = self.active_runs()
            update = self.update_state()
            update_running = update.get("status") == "running"
            blockers: list[str] = []
            if branch != "main":
                blockers.append("checkout is not on main")
            if dirty:
                blockers.append("tracked files have local modifications")
            if isinstance(ahead, int) and ahead > 0:
                blockers.append("checkout has local-only commits")
            if active:
                blockers.append("one or more training runs are active")
            if update_running:
                blockers.append("an update is already running")
            if remote_error:
                blockers.append("remote repository status is unavailable")

            update_available = bool(isinstance(behind, int) and behind > 0)
            result = {
                "ok": True,
                "checked_at": _now(),
                "repository": {
                    "branch": branch,
                    "local_commit": local_commit,
                    "remote_branch": "main",
                    "remote_commit": remote_commit,
                    "dirty": dirty,
                    "ahead_by": ahead,
                    "behind_by": behind,
                    "update_available": update_available,
                    "incoming_commits": incoming,
                    "remote_error": remote_error,
                },
                "active_runs": active,
                "update": update,
                "can_update": update_available and not blockers,
                "update_blockers": blockers,
                "tailscale": self._tailscale_status(),
            }
            self._status_cache = (time.monotonic(), result)
            return result

    def _compute_backend(self) -> str:
        state = _load_json(self.maintenance_root / "repository-version.json")
        compute = str(state.get("compute") or "auto")
        return compute if compute in {"auto", "cpu", "cu128"} else "auto"

    def start_update(self) -> dict[str, Any]:
        with self._update_lock:
            status = self.repository_status(refresh=True)
            if not status.get("can_update"):
                blockers = status.get("update_blockers") or ["update is not currently allowed"]
                raise RuntimeError("; ".join(str(value) for value in blockers))

            repository = status["repository"]
            command = [
                "bash",
                str(self.repo_root / "scripts" / "maintain.sh"),
                "--install-dir",
                str(self.repo_root),
                "--branch",
                "main",
                "--compute",
                self._compute_backend(),
            ]
            self.maintenance_root.mkdir(parents=True, exist_ok=True)
            log = self.update_log_path.open("a", encoding="utf-8", buffering=1)
            log.write(f"\n[{_now()}] dashboard-requested update\n")
            process = subprocess.Popen(
                command,
                cwd=self.repo_root,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            state = {
                "status": "running",
                "pid": process.pid,
                "started_at": _now(),
                "from_commit": repository.get("local_commit"),
                "target_commit": repository.get("remote_commit"),
                "log_path": str(self.update_log_path),
            }
            _write_json(self.update_state_path, state)
            self._status_cache = None

            def wait_for_update() -> None:
                return_code = process.wait()
                finished = {
                    **state,
                    "status": "success" if return_code == 0 else "error",
                    "return_code": return_code,
                    "finished_at": _now(),
                }
                _write_json(self.update_state_path, finished)
                try:
                    log.write(f"[{_now()}] update exited with code {return_code}\n")
                finally:
                    log.close()
                with self._status_lock:
                    self._status_cache = None

            threading.Thread(target=wait_for_update, daemon=True, name="ascento-update-waiter").start()
            return state

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        operation = request.get("op")
        if operation == "status":
            return {"ok": True, "result": self.repository_status(refresh=bool(request.get("refresh")))}
        if operation == "update":
            try:
                return {"ok": True, "result": self.start_update()}
            except RuntimeError as error:
                return {"ok": False, "error": str(error), "code": "update_blocked"}
        return {"ok": False, "error": "unsupported supervisor operation", "code": "unsupported"}


class _ThreadingUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        raw = self.rfile.readline(65_537)
        if not raw or len(raw) > 65_536:
            return
        try:
            request = json.loads(raw.decode("utf-8"))
            if not isinstance(request, dict):
                raise ValueError("request must be an object")
            response = self.server.supervisor.dispatch(request)  # type: ignore[attr-defined]
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            response = {"ok": False, "error": str(error), "code": "bad_request"}
        self.wfile.write((json.dumps(response) + "\n").encode("utf-8"))


def serve(repo_root: Path, socket_path: Path) -> None:
    supervisor = HostSupervisor(repo_root)
    socket_path = socket_path.expanduser().resolve()
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        socket_path.unlink()
    except FileNotFoundError:
        pass
    server = _ThreadingUnixServer(str(socket_path), _Handler)
    server.supervisor = supervisor  # type: ignore[attr-defined]
    os.chmod(socket_path, 0o660)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Ascento host maintenance supervisor")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    args = parser.parse_args()
    serve(args.repo, args.socket)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
