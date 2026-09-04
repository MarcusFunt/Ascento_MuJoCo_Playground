import numpy as np

from ascento_mjlab.evaluation.statistics import iqm, summarize_binary, wilson_interval


def test_wilson_all_successes_is_below_one():
  low, high = wilson_interval(1024, 1024)
  assert 0.995 < low < 1.0
  assert high == 1.0


def test_iqm_rejects_extreme_quartiles():
  values = np.array([-100.0, 1.0, 2.0, 3.0, 4.0, 100.0])
  assert 1.0 < iqm(values) < 5.0


def test_binary_summary():
  summary = summarize_binary([True, True, False, True])
  assert summary["successes"] == 3.0
  assert summary["success_rate"] == 0.75
  assert summary["wilson_lower"] < summary["success_rate"] < summary["wilson_upper"]
