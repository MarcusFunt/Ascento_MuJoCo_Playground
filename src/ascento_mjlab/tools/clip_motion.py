"""Trim and resample RecorderManager motion captures for animation use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_capture(path: Path) -> dict[str, np.ndarray]:
    """Load a capture without enabling pickle for metadata or object arrays."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _frame_count(capture: dict[str, np.ndarray]) -> int:
    if "time" not in capture or capture["time"].ndim != 1:
        raise ValueError("capture must contain a one-dimensional time array")
    count = len(capture["time"])
    if count < 2:
        raise ValueError("capture must contain at least two frames")
    return count


def _event_indices(capture: dict[str, np.ndarray]) -> dict[str, list[int]]:
    """Return all detected event frame indices from jump state or contact fallback."""
    count = _frame_count(capture)
    events: dict[str, list[int]] = {"start": [0], "end": [count - 1]}
    jump_state = capture.get("jump_state")
    if jump_state is not None and jump_state.ndim == 2 and jump_state.shape[1] >= 3:
        takeoff = np.flatnonzero(jump_state[:, 1] > 0.5).tolist()
        landing = np.flatnonzero(jump_state[:, 2] > 0.5).tolist()
        if takeoff:
            events["takeoff"] = takeoff
        if landing:
            events["landing"] = landing
    contacts = capture.get("contacts")
    if contacts is not None and contacts.ndim == 2 and contacts.shape[1] >= 2:
        supported = np.all(contacts[:, :2] > 0.5, axis=1)
        takeoff = (np.flatnonzero(supported[:-1] & ~supported[1:]) + 1).tolist()
        landing = (np.flatnonzero(~supported[:-1] & supported[1:]) + 1).tolist()
        # Jump-state event pulses are authoritative when present. Contacts fill only
        # missing event types so an empty jump_state channel cannot suppress fallback.
        if takeoff and not events.get("takeoff"):
            events["takeoff"] = takeoff
        if landing and not events.get("landing"):
            events["landing"] = landing
    return {name: indices for name, indices in events.items() if indices}


def detect_events(capture: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """Return named event markers with frame and source-time fields."""
    times = np.asarray(capture["time"], dtype=np.float64)
    return [
        {"name": name, "frame": frame, "source_time": float(times[frame])}
        for name, indices in _event_indices(capture).items()
        for frame in indices
    ]


def _metadata(capture: dict[str, np.ndarray]) -> dict[str, str]:
    metadata = {}
    for key, value in capture.items():
        if key.startswith("meta_") and value.ndim == 0:
            metadata[key[5:]] = str(value.item())
    return metadata


def _slice_capture(capture: dict[str, np.ndarray], start: int, end: int) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    count = end - start + 1
    for key, value in capture.items():
        if key.startswith("meta_") or value.ndim == 0 or len(value) != _frame_count(capture):
            result[key] = value.copy()
        elif len(value) == count and start == 0 and end == len(value) - 1:
            result[key] = value.copy()
        else:
            result[key] = value[start : end + 1].copy()
    result["source_time"] = capture["time"][start : end + 1].copy()
    result["time"] = result["source_time"] - result["source_time"][0]
    return result


def _resample_capture(capture: dict[str, np.ndarray], fps: float) -> dict[str, np.ndarray]:
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    source_time = np.asarray(capture["time"], dtype=np.float64)
    if np.any(np.diff(source_time) <= 0.0):
        raise ValueError("capture time must be strictly increasing")
    target_time = np.arange(0.0, source_time[-1] + 1.0e-9, 1.0 / fps)
    if target_time[-1] < source_time[-1] - 1.0e-9:
        target_time = np.append(target_time, source_time[-1])
    result: dict[str, np.ndarray] = {"time": target_time}
    discrete = {"contacts", "jump_state"}
    for key, value in capture.items():
        if key in {"time", "source_time"} or key.startswith("meta_") or value.ndim == 0:
            if key not in {"time", "source_time"}:
                result[key] = value.copy()
            continue
        if len(value) != len(source_time):
            result[key] = value.copy()
            continue
        if key in discrete:
            # Zero-order hold: keep the most recent source sample until its successor's
            # timestamp. side="right" preserves transitions at the original sample time.
            indices = np.searchsorted(source_time, target_time, side="right") - 1
            indices = indices.clip(min=0, max=len(source_time) - 1)
            result[key] = value[indices]
            continue
        flat = value.reshape(len(value), -1).astype(np.float64)
        interpolated = np.column_stack(
            [
                np.interp(target_time, source_time, flat[:, column])
                for column in range(flat.shape[1])
            ]
        ).reshape((len(target_time),) + value.shape[1:])
        if key == "root_quat":
            norms = np.linalg.norm(interpolated, axis=1, keepdims=True).clip(min=1.0e-12)
            interpolated /= norms
        result[key] = interpolated.astype(value.dtype, copy=False)
    return result


def process_capture(
    capture: dict[str, np.ndarray],
    *,
    fps: float | None = None,
    event: str = "all",
    pre_roll: float = 0.5,
    post_roll: float = 0.5,
) -> dict[str, np.ndarray]:
    """Trim around the first named event and optionally resample the result."""
    if pre_roll < 0.0 or post_roll < 0.0:
        raise ValueError("pre_roll and post_roll must be non-negative")
    events = detect_events(capture)
    if event in {"all", "start", "end"}:
        start, end = 0, _frame_count(capture) - 1
    else:
        matches = [marker for marker in events if marker["name"] == event]
        if not matches:
            raise ValueError(f"event {event!r} was not found in the capture")
        source_time = np.asarray(capture["time"], dtype=np.float64)
        center = matches[0]["frame"]
        # Account for decimal timestamp representation at exact frame boundaries.
        epsilon = 1.0e-9
        start = int(
            np.searchsorted(source_time, source_time[center] - pre_roll - epsilon, side="left")
        )
        end = int(
            np.searchsorted(source_time, source_time[center] + post_roll + epsilon, side="right")
            - 1
        )
        end = min(end, len(source_time) - 1)
    result = _slice_capture(capture, start, end)
    if fps is not None:
        result = _resample_capture(result, fps)
    if "root_pos" in result:
        result["root_motion"] = result["root_pos"] - result["root_pos"][0]
    if "joint_pos" in result:
        result["joint_pos_local"] = result["joint_pos"] - result["joint_pos"][0]
    selected_start = float(capture["time"][start])
    selected_end = float(capture["time"][end])
    markers = [
        {**marker, "clip_time": marker["source_time"] - selected_start}
        for marker in events
        if selected_start <= marker["source_time"] <= selected_end
    ]
    result["clip_events_json"] = np.asarray(json.dumps(markers, sort_keys=True))
    result["clip_metadata_json"] = np.asarray(
        json.dumps(
            {
                **_metadata(capture),
                "source_frame_count": _frame_count(capture),
                "source_duration_s": float(capture["time"][-1] - capture["time"][0]),
                "clip_duration_s": float(result["time"][-1]),
                "clip_fps": (
                    float(1.0 / np.median(np.diff(result["time"])))
                    if len(result["time"]) > 1
                    else None
                ),
                "trim_event": event,
            },
            sort_keys=True,
        )
    )
    return result


def save_capture(path: Path, capture: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **capture)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=None, help="Optional animation output FPS")
    parser.add_argument("--event", default="all", choices=("all", "takeoff", "landing"))
    parser.add_argument("--pre-roll", type=float, default=0.5)
    parser.add_argument("--post-roll", type=float, default=0.5)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"capture does not exist: {args.input}")
    output = args.output or args.input.with_name(f"{args.input.stem}_clip.npz")
    save_capture(
        output,
        process_capture(
            load_capture(args.input),
            fps=args.fps,
            event=args.event,
            pre_roll=args.pre_roll,
            post_roll=args.post_roll,
        ),
    )
    print(output)


if __name__ == "__main__":
    main()
