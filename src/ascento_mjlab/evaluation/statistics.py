"""Statistical summaries for benchmark results."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
  if total <= 0:
    return float("nan"), float("nan")
  p = successes / total
  z2 = z * z
  denom = 1.0 + z2 / total
  center = (p + z2 / (2.0 * total)) / denom
  half = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total) / denom
  return max(0.0, center - half), min(1.0, center + half)


def iqm(values: np.ndarray) -> float:
  values = np.asarray(values, dtype=float)
  if values.size == 0:
    return float("nan")
  values = np.sort(values)
  lo = int(math.floor(0.25 * values.size))
  hi = int(math.ceil(0.75 * values.size))
  middle = values[lo:hi]
  return float(np.mean(middle if middle.size else values))


def bootstrap_ci(
  values: np.ndarray,
  statistic: Callable[[np.ndarray], float] = iqm,
  *,
  seed: int = 0,
  samples: int = 1000,
  alpha: float = 0.05,
) -> tuple[float, float]:
  values = np.asarray(values, dtype=float)
  values = values[np.isfinite(values)]
  if values.size == 0:
    return float("nan"), float("nan")
  if values.size == 1:
    value = float(values[0])
    return value, value
  rng = np.random.default_rng(seed)
  estimates = np.empty(samples, dtype=float)
  for idx in range(samples):
    sample = rng.choice(values, size=values.size, replace=True)
    estimates[idx] = statistic(sample)
  return (
    float(np.quantile(estimates, alpha / 2.0)),
    float(np.quantile(estimates, 1.0 - alpha / 2.0)),
  )


def summarize_numeric(values: list[float], *, seed: int = 0) -> dict[str, float]:
  array = np.asarray(values, dtype=float)
  array = array[np.isfinite(array)]
  if array.size == 0:
    return {
      "count": 0.0,
      "mean": float("nan"),
      "median": float("nan"),
      "iqm": float("nan"),
      "p05": float("nan"),
      "p50": float("nan"),
      "p95": float("nan"),
      "min": float("nan"),
      "max": float("nan"),
      "iqm_ci95_low": float("nan"),
      "iqm_ci95_high": float("nan"),
      "sum": 0.0,
    }
  ci_low, ci_high = bootstrap_ci(array, seed=seed)
  return {
    "count": float(array.size),
    "mean": float(np.mean(array)),
    "median": float(np.median(array)),
    "iqm": iqm(array),
    "p05": float(np.quantile(array, 0.05)),
    "p50": float(np.quantile(array, 0.50)),
    "p95": float(np.quantile(array, 0.95)),
    "min": float(np.min(array)),
    "max": float(np.max(array)),
    "iqm_ci95_low": ci_low,
    "iqm_ci95_high": ci_high,
    "sum": float(np.sum(array)),
  }


def summarize_binary(values: list[bool]) -> dict[str, float]:
  total = len(values)
  successes = sum(bool(value) for value in values)
  low, high = wilson_interval(successes, total)
  return {
    "count": float(total),
    "successes": float(successes),
    "success_rate": successes / total if total else float("nan"),
    "wilson_lower": low,
    "wilson_upper": high,
  }
