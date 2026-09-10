from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_native_wsl_mcp_registration_targets_the_native_checkout_and_artifacts():
    script = (REPO_ROOT / "scripts" / "register_codex_mcp.ps1").read_text(encoding="utf-8")

    assert '$nativeRoot = "/root/Ascento_MuJoCo_Playground"' in script
    assert '$artifactRoot = "$nativeRoot/logs/rsl_rl"' in script
    assert '"ASCENTO_ARTIFACT_ROOT=$artifactRoot"' in script
    assert "Convert-ToWslPath" not in script
