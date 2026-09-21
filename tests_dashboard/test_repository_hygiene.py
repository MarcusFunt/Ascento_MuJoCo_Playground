import subprocess
from pathlib import Path
from shutil import copy2

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _is_ignored(path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", path],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    return result.returncode == 0


def test_generated_artifact_directories_are_ignored():
    """Generated artifacts cannot re-enter reviewable source history."""
    assert _is_ignored("reports")
    assert _is_ignored("transfers")
    assert _is_ignored(".worktrees")
    assert _is_ignored(".codex/recovery")


def test_recovered_locomotion_contract_is_present_for_review():
    """The locomotion task and its acceptance gate are source, not local artifacts."""
    assert (REPOSITORY_ROOT / "benchmarks/suites/locomotion_sequence_gate_v1.toml").is_file()
    assert (REPOSITORY_ROOT / "src/ascento_mjlab/tasks/locomotion/env_cfg.py").is_file()


def test_hook_installer_uses_the_common_git_directory_for_a_worktree(tmp_path):
    """Installing from a worktree writes to Git's shared hooks directory."""
    primary = tmp_path / "primary"
    worktree = tmp_path / "worktree"
    subprocess.run(["git", "init", str(primary)], check=True)
    subprocess.run(["git", "-C", str(primary), "config", "user.name", "Test User"], check=True)
    subprocess.run(["git", "-C", str(primary), "config", "user.email", "test@example.com"], check=True)
    (primary / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(primary), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(primary), "commit", "-m", "fixture"], check=True)
    subprocess.run(["git", "-C", str(primary), "worktree", "add", "-b", "fixture", str(worktree)], check=True)

    hook_source = worktree / "scripts/git-hooks/pre-commit"
    hook_source.parent.mkdir(parents=True)
    copy2(REPOSITORY_ROOT / "scripts/git-hooks/pre-commit", hook_source)
    installer = worktree / "scripts/install_git_hooks.sh"
    copy2(REPOSITORY_ROOT / "scripts/install_git_hooks.sh", installer)

    installation = subprocess.run(
        ["bash", str(installer)],
        cwd=worktree,
        check=True,
        capture_output=True,
        text=True,
    )
    hook_path = subprocess.run(
        ["git", "rev-parse", "--git-path", "hooks/pre-commit"],
        cwd=worktree,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert Path(hook_path).read_text(encoding="utf-8") == hook_source.read_text(encoding="utf-8")
    assert installation.stdout.endswith("\n")
    assert not installation.stdout.endswith("\\n")
