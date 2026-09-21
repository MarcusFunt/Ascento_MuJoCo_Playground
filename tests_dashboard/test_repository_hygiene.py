import subprocess
from pathlib import Path


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
