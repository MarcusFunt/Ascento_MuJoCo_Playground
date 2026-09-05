import json
import subprocess
import sys


SCRIPT = "scripts/tailscale_status.py"


def _run(payload):
    return subprocess.run(
        [sys.executable, SCRIPT],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    ).returncode


def test_tailscale_status_rejects_needs_login_even_with_valid_json():
    assert _run({"BackendState": "NeedsLogin", "Self": {"TailscaleIPs": []}}) == 1


def test_tailscale_status_rejects_starting_state():
    assert _run({"BackendState": "Starting", "Self": {"TailscaleIPs": ["100.64.0.1"]}}) == 1


def test_tailscale_status_rejects_running_without_assigned_tailnet_ip():
    assert _run({"BackendState": "Running", "Self": {"TailscaleIPs": []}}) == 1


def test_tailscale_status_accepts_authenticated_running_node():
    assert _run({"BackendState": "Running", "Self": {"TailscaleIPs": ["100.64.0.1"]}}) == 0


def test_tailscale_status_rejects_invalid_json():
    result = subprocess.run(
        [sys.executable, SCRIPT],
        input="not-json",
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
