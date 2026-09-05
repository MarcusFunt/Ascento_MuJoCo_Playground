from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_supervisor_uses_a_short_systemd_runtime_socket_path():
    installer = (REPO_ROOT / "scripts" / "install_supervisor.sh").read_text(encoding="utf-8")
    compose = (REPO_ROOT / "docker" / "compose.yaml").read_text(encoding="utf-8")

    assert 'SOCKET_DIR="/run/ascento-supervisor"' in installer
    assert "RuntimeDirectory=ascento-supervisor" in installer
    assert "RuntimeDirectoryMode=0770" in installer
    assert "/run/ascento-supervisor:/run/ascento-supervisor" in compose
    assert len("/run/ascento-supervisor/supervisor.sock".encode()) < 108
