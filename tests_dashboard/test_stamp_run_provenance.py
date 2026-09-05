import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "stamp_run_provenance.py"


def test_stamp_run_provenance_runs_directly_without_pythonpath(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(runs),
            "--commit",
            "test-commit",
            "--branch",
            "main",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "Stamped repository provenance for 0 legacy run(s)." in result.stdout
