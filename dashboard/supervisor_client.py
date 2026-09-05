"""Restricted Unix-socket client for the host maintenance supervisor."""
from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any

DEFAULT_SOCKET = Path("/run/ascento-supervisor/supervisor.sock")


class SupervisorUnavailable(RuntimeError):
    pass


class SupervisorRejected(RuntimeError):
    pass


class SupervisorClient:
    def __init__(self, socket_path: Path | None = None, *, timeout: float = 3.0):
        configured = os.environ.get("ASCENTO_SUPERVISOR_SOCKET")
        self.socket_path = Path(configured) if configured else (socket_path or DEFAULT_SOCKET)
        self.timeout = timeout

    def _request(self, operation: str, **values: Any) -> dict[str, Any]:
        payload = json.dumps({"op": operation, **values}, separators=(",", ":")) + "\n"
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(self.timeout)
                connection.connect(str(self.socket_path))
                connection.sendall(payload.encode("utf-8"))
                chunks = bytearray()
                while len(chunks) <= 1_048_576:
                    chunk = connection.recv(65_536)
                    if not chunk:
                        break
                    chunks.extend(chunk)
                    if b"\n" in chunk:
                        break
        except (FileNotFoundError, ConnectionRefusedError, socket.timeout, OSError) as error:
            raise SupervisorUnavailable(
                f"host supervisor is unavailable at {self.socket_path}: {error}"
            ) from error

        try:
            response = json.loads(bytes(chunks).split(b"\n", 1)[0].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SupervisorUnavailable("host supervisor returned an invalid response") from error
        if not isinstance(response, dict):
            raise SupervisorUnavailable("host supervisor returned an invalid response")
        if not response.get("ok"):
            raise SupervisorRejected(str(response.get("error") or "supervisor rejected request"))
        result = response.get("result")
        return result if isinstance(result, dict) else {}

    def status(self, *, refresh: bool = False) -> dict[str, Any]:
        return self._request("status", refresh=refresh)

    def update(self) -> dict[str, Any]:
        return self._request("update")
