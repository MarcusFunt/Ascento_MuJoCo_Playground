import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
  "source",
  [
    """
import ascento_mjlab.tasks
from mjlab.tasks.registry import list_tasks
assert 'Ascento-Jump-Flat' in list_tasks()
""",
    """
from mjlab.tasks.registry import list_tasks
import ascento_mjlab.tasks
assert 'Ascento-Jump-Flat' in list_tasks()
""",
    """
from ascento_mjlab.tasks.jump.env_cfg import ascento_jump_env_cfg
from mjlab.tasks.registry import list_tasks
assert 'Ascento-Jump-Flat' in list_tasks()
assert ascento_jump_env_cfg(num_envs=1).scene.num_envs == 1
""",
    """
import ascento_mjlab.actuator
from mjlab.tasks.registry import load_env_cfg, list_tasks
assert 'Ascento-Balance-Flat' in list_tasks()
assert load_env_cfg('Ascento-Balance-Flat').scene.num_envs == 512
""",
  ],
)
def test_task_registration_is_independent_of_import_order(source):
  env = os.environ.copy()
  src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
  env["PYTHONPATH"] = os.pathsep.join(
    part for part in (src_dir, env.get("PYTHONPATH", "")) if part
  )
  result = subprocess.run(
    [sys.executable, "-c", source],
    cwd=os.path.dirname(os.path.dirname(__file__)),
    env=env,
    capture_output=True,
    text=True,
    check=False,
  )

  assert result.returncode == 0, result.stderr
