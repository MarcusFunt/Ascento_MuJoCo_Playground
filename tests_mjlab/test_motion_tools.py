import json

import numpy as np

from ascento_mjlab.tools.clip_motion import detect_events, process_capture, save_capture
from ascento_mjlab.tools.motion_quality import compute_motion_quality, rank_capture_files


def _capture() -> dict[str, np.ndarray]:
  time = np.arange(0.0, 1.0, 0.1)
  contacts = np.ones((len(time), 2), dtype=np.float32)
  contacts[3:7] = 0.0
  return {
    "time": time,
    "root_pos": np.column_stack((time, np.zeros((len(time), 2)))),
    "root_quat": np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (len(time), 1)),
    "joint_pos": np.zeros((len(time), 6)),
    "action": np.zeros((len(time), 6)),
    "effort": np.zeros((len(time), 6)),
    "contacts": contacts,
    "meta_task": np.asarray("test"),
  }


def test_detect_events_uses_two_wheel_contact_transitions():
  events = detect_events(_capture())
  assert [(event["name"], event["frame"]) for event in events] == [
    ("start", 0),
    ("end", 9),
    ("takeoff", 3),
    ("landing", 7),
  ]


def test_process_capture_trims_resamples_and_separates_motion():
  result = process_capture(_capture(), event="takeoff", pre_roll=0.1, post_roll=0.3, fps=20.0)
  assert len(result["time"]) == 9
  assert result["time"][0] == 0.0
  assert result["root_motion"].shape == result["root_pos"].shape
  assert result["joint_pos_local"].shape == result["joint_pos"].shape
  markers = json.loads(str(result["clip_events_json"].item()))
  assert any(marker["name"] == "takeoff" for marker in markers)


def test_motion_quality_reports_smoothness_separately_from_success():
  report = compute_motion_quality(_capture())
  assert report["finite"] is True
  assert report["duration_s"] == 0.9
  assert report["contact_toggle_count"] == 2
  assert 0.0 < report["quality_score"] <= 1.0


def test_rank_capture_files_returns_best_first(tmp_path):
  first = tmp_path / "first.npz"
  second = tmp_path / "second.npz"
  save_capture(first, _capture())
  noisy = _capture()
  noisy["root_pos"][:, 0] = np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
  save_capture(second, noisy)
  reports = rank_capture_files([second, first])
  assert {report["path"] for report in reports} == {str(first), str(second)}
  assert reports[0]["quality_score"] >= reports[1]["quality_score"]
