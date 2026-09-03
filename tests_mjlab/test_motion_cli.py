import json
import os
import subprocess
import sys

import numpy as np

from ascento_mjlab.tools.clip_motion import save_capture


def _capture() -> dict[str, np.ndarray]:
  time = np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float64)
  return {
    "time": time,
    "root_pos": np.column_stack((time, np.zeros((len(time), 2)))),
    "root_quat": np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (len(time), 1)),
    "joint_pos": np.zeros((len(time), 6), dtype=np.float32),
    "action": np.zeros((len(time), 6), dtype=np.float32),
    "effort": np.zeros((len(time), 6), dtype=np.float32),
    "contacts": np.asarray([[1, 1], [1, 1], [0, 0], [0, 0]], dtype=np.float32),
    "jump_state": np.zeros((len(time), 4), dtype=np.float32),
    "meta_task": np.asarray("cli-fixture"),
  }


def _run_module(*args: str) -> subprocess.CompletedProcess[str]:
  env = os.environ.copy()
  env["MUJOCO_GL"] = "disable"
  return subprocess.run(
    [sys.executable, "-m", *args],
    env=env,
    capture_output=True,
    text=True,
    check=False,
  )


def test_capture_cli_imports_on_cpu_safe_help_path():
  result = _run_module("ascento_mjlab.tools.capture_motion", "--help")
  assert result.returncode == 0, result.stderr
  assert "--device" in result.stdout


def test_clip_and_rank_clis_process_representative_npz(tmp_path):
  source = tmp_path / "capture.npz"
  clipped = tmp_path / "capture_clip.npz"
  report = tmp_path / "ranking.json"
  save_capture(source, _capture())

  clip_result = _run_module(
    "ascento_mjlab.tools.clip_motion",
    str(source),
    "--output",
    str(clipped),
    "--fps",
    "20",
  )
  assert clip_result.returncode == 0, clip_result.stderr
  assert clipped.is_file()

  rank_result = _run_module(
    "ascento_mjlab.tools.motion_quality",
    str(clipped),
    "--output",
    str(report),
  )
  assert rank_result.returncode == 0, rank_result.stderr
  payload = json.loads(report.read_text(encoding="utf-8"))
  assert len(payload) == 1
  assert payload[0]["path"] == str(clipped)
  assert payload[0]["finite"] is True
