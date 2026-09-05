#!/usr/bin/env python3
"""Validate that a Tailscale status snapshot represents a usable logged-in node."""

from __future__ import annotations

import json
import sys
from typing import Any


def is_running(status: dict[str, Any]) -> bool:
    """Return True only after containerboot has reached an authenticated Running state."""
    if status.get("BackendState") != "Running":
        return False
    self_node = status.get("Self")
    if not isinstance(self_node, dict):
        return False
    tailscale_ips = self_node.get("TailscaleIPs")
    return isinstance(tailscale_ips, list) and bool(tailscale_ips)


def main() -> int:
    try:
        status = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError):
        return 2
    if not isinstance(status, dict):
        return 2
    return 0 if is_running(status) else 1


if __name__ == "__main__":
    raise SystemExit(main())
