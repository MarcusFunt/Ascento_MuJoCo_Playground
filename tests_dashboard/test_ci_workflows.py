from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def workflow_text(name: str) -> str:
    return (REPOSITORY_ROOT / ".github/workflows" / name).read_text(encoding="utf-8")


def test_ci_uses_python_312():
    assert workflow_text("dashboard-ci.yml").count("python-version: '3.12'") == 3


def test_gpu_smoke_is_manual_and_labeled():
    text = workflow_text("gpu-smoke.yml")

    assert "workflow_dispatch:" in text
    assert "[self-hosted, linux, gpu]" in text


def test_maintainer_and_ci_provide_verified_clean_image_provenance():
    workflow = workflow_text("dashboard-ci.yml")
    maintainer = (REPOSITORY_ROOT / "scripts" / "maintain.sh").read_text(encoding="utf-8")

    assert "ASCENTO_REPOSITORY_DIRTY=0" in workflow
    assert "status --porcelain --untracked-files=all" in maintainer
    assert "Refusing to bake repository provenance from a dirty or untracked source tree." in maintainer
