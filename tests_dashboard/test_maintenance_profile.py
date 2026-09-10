from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_wsl_cuda_maintenance_profile_uses_a_native_install_path():
    profile = (REPO_ROOT / "config" / "maintenance.wsl-cu128.env").read_text(encoding="utf-8")

    assert 'export ASCENTO_INSTALL_DIR="/root/Ascento_MuJoCo_Playground"' in profile
    assert 'export ASCENTO_COMPUTE="cu128"' in profile
