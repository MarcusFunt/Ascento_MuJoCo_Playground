"""Compute motion-quality metrics and rank captured animation candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .clip_motion import load_capture


def _derivative(values: np.ndarray, time: np.ndarray) -> np.ndarray:
    return np.gradient(values, time, axis=0, edge_order=1)


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def compute_motion_quality(capture: dict[str, np.ndarray]) -> dict[str, Any]:
    """Return quality metrics without conflating them with task success."""
    time = np.asarray(capture["time"], dtype=np.float64)
    if len(time) < 2 or np.any(np.diff(time) <= 0.0):
        raise ValueError("capture time must be strictly increasing")
    report: dict[str, Any] = {
        "finite": all(
            np.isfinite(value).all() for value in capture.values() if value.dtype.kind in "fiu"
        ),
        "duration_s": float(time[-1] - time[0]),
        "frame_count": len(time),
    }
    if "root_pos" in capture:
        velocity = _derivative(capture["root_pos"].astype(np.float64), time)
        acceleration = _derivative(velocity, time)
        jerk = _derivative(acceleration, time)
        report.update(
            root_speed_rms_mps=_rms(np.linalg.norm(velocity, axis=1)),
            root_acceleration_rms_mps2=_rms(np.linalg.norm(acceleration, axis=1)),
            root_jerk_rms_mps3=_rms(np.linalg.norm(jerk, axis=1)),
        )
    for source, prefix in (("action", "action"), ("effort", "torque")):
        if source not in capture:
            continue
        values = capture[source].astype(np.float64)
        first = _derivative(values, time)
        second = _derivative(first, time)
        third = _derivative(second, time)
        report[f"{prefix}_rms"] = _rms(values)
        report[f"{prefix}_peak"] = float(np.max(np.abs(values)))
        report[f"{prefix}_jerk_rms"] = _rms(third)
    if "joint_pos" in capture:
        joint_acceleration = _derivative(
            _derivative(capture["joint_pos"].astype(np.float64), time), time
        )
        report["joint_acceleration_rms_rad_s2"] = _rms(joint_acceleration)
    if "contacts" in capture:
        contacts = capture["contacts"] > 0.5
        report["contact_toggle_count"] = int(
            np.count_nonzero(np.any(contacts[1:] != contacts[:-1], axis=1))
        )
        report["contact_toggle_rate_hz"] = float(
            report["contact_toggle_count"] / report["duration_s"]
        )
    report["quality_score"] = _quality_score(report)
    return report


def _quality_score(report: dict[str, Any]) -> float:
    """Higher is smoother; this score is for ranking, not task acceptance."""
    penalty = 0.0
    penalty += report.get("root_jerk_rms_mps3", 0.0) / 10.0
    penalty += report.get("torque_jerk_rms", 0.0) / 1000.0
    penalty += report.get("action_jerk_rms", 0.0) / 10.0
    penalty += report.get("contact_toggle_rate_hz", 0.0) / 20.0
    return 0.0 if not report["finite"] else float(1.0 / (1.0 + penalty))


def rank_capture_files(paths: Iterable[Path]) -> list[dict[str, Any]]:
    reports = []
    for path in sorted(paths):
        report = compute_motion_quality(load_capture(path))
        reports.append({"path": str(path), **report})
    return sorted(reports, key=lambda item: (-item["quality_score"], item["path"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Capture files or directories")
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    paths = []
    for item in args.inputs:
        paths.extend(sorted(item.glob("*.npz")) if item.is_dir() else [item])
    if not paths or any(not path.is_file() for path in paths):
        parser.error("inputs must contain existing NPZ captures")
    reports = rank_capture_files(paths)
    if args.top is not None:
        if args.top < 1:
            parser.error("--top must be positive")
        reports = reports[: args.top]
    payload = json.dumps(reports, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
