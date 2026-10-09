"""Dependency warning regressions that run imports in isolated Python processes."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _assert_import_is_warning_free(statement: str, category: str | None = None) -> None:
    warning_args = ["-W", f"error::{category}"] if category else ["-W", "error"]
    result = subprocess.run(
        [sys.executable, *warning_args, "-c", statement],
        cwd=ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_mjlab_import_does_not_emit_warp_deprecations():
    _assert_import_is_warning_free("import mjlab", "DeprecationWarning")


def test_mjlab_math_import_does_not_emit_torch_jit_deprecations():
    _assert_import_is_warning_free("import mjlab.utils.lab_api.math")


def test_mjlab_overlay_preserves_installed_assets_and_task_registration():
    statement = (
        "from importlib.metadata import distribution, entry_points; "
        "from pathlib import Path; "
        "import mjlab; "
        "root = Path(distribution('mjlab').locate_file('mjlab')).resolve(); "
        "assert mjlab.MJLAB_SRC_PATH == root; "
        "assert (root / 'asset_zoo/robots/unitree_go1/xmls/go1.xml').is_file(); "
        "assert any(ep.name == 'ascento_mjlab' and ep.load() for ep in "
        "entry_points().select(group='mjlab.tasks'))"
    )
    _assert_import_is_warning_free(statement)
