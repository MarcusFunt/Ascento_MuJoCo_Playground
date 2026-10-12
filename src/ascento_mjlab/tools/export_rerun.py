"""Convert Ascento motion NPZ captures into spatial Rerun recordings."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

import mjlab  # noqa: F401 - initialize task plugin registry before robot_cfg import
from ascento_mjlab.robot_cfg import JOINT_NAMES, ROBOT_XML


def _meta(data: Any, key: str, default: str = "") -> str:
    name = f"meta_{key}"
    return str(data[name].item()) if name in data else default


def validate_capture(data: Any, model: mujoco.MjModel) -> dict[str, Any]:
    """Validate geometry mapping, time and reward accounting before rendering."""
    required = ("time", "root_pos", "root_quat", "joint_pos")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Missing capture channels: {missing}")
    frames = {name: np.asarray(data[name], dtype=np.float64) for name in required}
    time, root, quat, joints = (frames[key] for key in required)
    n = len(time)
    if n == 0 or time.shape != (n,) or root.shape != (n, 3) or quat.shape != (n, 4):
        raise ValueError("Invalid frame count, timestamp, root position or quaternion shape")
    if joints.ndim != 2 or len(joints) != n:
        raise ValueError("joint_pos must have shape (frame_count, joint_count)")
    if any(not np.isfinite(a).all() for a in frames.values()) or np.any(np.diff(time) <= 0):
        raise ValueError("Nonfinite capture data or non-monotonic timestamps")
    if np.any(np.abs(np.linalg.norm(quat, axis=1) - 1.0) > 0.01):
        raise ValueError("Root quaternions are not normalized")

    names = json.loads(_meta(data, "joint_names", json.dumps(JOINT_NAMES)))
    if (
        not isinstance(names, list)
        or len(names) != joints.shape[1]
        or len(set(names)) != len(names)
    ):
        raise ValueError("Joint names do not match joint positions")
    addrs = []
    for name in names:
        if not isinstance(name, str):
            raise ValueError("Invalid joint name")
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"Unknown or non-hinge joint {name!r}")
        addrs.append(int(model.jnt_qposadr[joint_id]))
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
        raise ValueError("Expected exactly one floating root joint")
    frames["joint_addrs"] = addrs
    frames["root_addr"] = int(model.jnt_qposadr[int(free[0])])
    frames["count"] = n

    for channel, width in (("target_xy", 2), ("root_lin_vel_w", 3)):
        if channel in data:
            values = np.asarray(data[channel], dtype=np.float64)
            if values.shape != (n, width) or not np.isfinite(values).all():
                raise ValueError(f"Invalid {channel} channel")
            frames[channel] = values
    if "speed_command" in data:
        values = np.asarray(data["speed_command"], dtype=np.float64)
        if values.shape not in ((n,), (n, 1)) or not np.isfinite(values).all():
            raise ValueError("Invalid speed_command channel")
        frames["speed_command"] = values.reshape(n)

    if "reward_total" in data or "reward_terms_rate" in data:
        if not all(key in data for key in ("reward_total", "reward_terms_rate", "reward_dt")):
            raise ValueError("Incomplete reward recording")
        rate = np.asarray(data["reward_terms_rate"], dtype=np.float64)
        total = np.asarray(data["reward_total"], dtype=np.float64)
        dt = np.asarray(data["reward_dt"], dtype=np.float64)
        labels = json.loads(_meta(data, "reward_term_names", "[]"))
        if (
            rate.ndim != 2
            or rate.shape[0] != n
            or total.shape != (n,)
            or dt.shape != (n,)
            or not isinstance(labels, list)
            or len(labels) != rate.shape[1]
            or not all(isinstance(x, str) for x in labels)
        ):
            raise ValueError("Malformed reward labels or channels")
        if (
            not np.isfinite(rate).all()
            or not np.isfinite(total).all()
            or not np.isfinite(dt).all()
            or np.any(dt <= 0)
        ):
            raise ValueError("Nonfinite reward values or invalid dt")
        if not np.allclose(rate.sum(axis=1) * dt, total, atol=1e-5, rtol=1e-4):
            raise ValueError("Reward terms do not reproduce the actual training step reward")
        frames.update(reward_total=total, reward_terms_rate=rate, reward_dt=dt)
        frames["reward_term_names"] = labels
    return frames


def export_rerun(
    capture: Path,
    output: Path,
    *,
    model_path: Path = ROBOT_XML,
    stride: int = 1,
    collision: bool = False,
) -> dict[str, Any]:
    """Export a saved capture offline; never run the PPO environment."""
    if stride < 1:
        raise ValueError("Stride must be a positive integer")
    capture = Path(capture).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    if output.suffix != ".rrd":
        raise ValueError("Output must end in .rrd")
    if not capture.is_file() or not model_path.is_file():
        raise FileNotFoundError("Capture or MJCF model path does not exist")
    try:
        import rerun as rr
        import rerun.blueprint as rrb
        from rerun_loader_mjcf import MJCFLogger
    except ImportError as exc:
        raise RuntimeError("Install the optional dependencies with: uv sync --extra rerun") from exc
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    with np.load(capture, allow_pickle=False) as data:
        captured_sha = _meta(data, "robot_xml_sha256")
        if captured_sha and captured_sha != model_sha:
            raise ValueError("Replay MJCF differs from the captured model checksum")
        ep = validate_capture(data, model)
        output.parent.mkdir(parents=True, exist_ok=True)
        rr.init("ascento-episode-replay")
        rr.save(
            str(output),
            default_blueprint=rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Spatial3DView(origin="/robot", name="Ascento 3D"),
                    rrb.Vertical(
                        rrb.Spatial2DView(origin="/spatial", name="World XY"),
                        rrb.TimeSeriesView(origin="/reward", name="Reward components"),
                    ),
                ),
                collapse_panels=True,
            ),
        )
        logger = MJCFLogger(model, entity_path_prefix="robot", log_collision=collision)
        rr.set_time("sim_time", duration=float(ep["time"][0]))
        logger.log_model()
        path_xy = ep["root_pos"][:, :2]
        rr.log("spatial/trajectory", rr.LineStrips2D([path_xy]), static=True)
        body_data = mujoco.MjData(model)
        labels = ep.get("reward_term_names", [])
        for index in range(0, ep["count"], stride):
            rr.set_time("sim_time", duration=float(ep["time"][index]))
            mujoco.mj_resetData(model, body_data)
            addr = ep["root_addr"]
            body_data.qpos[addr : addr + 3] = ep["root_pos"][index]
            body_data.qpos[addr + 3 : addr + 7] = ep["root_quat"][index]
            for j, joint_addr in enumerate(ep["joint_addrs"]):
                body_data.qpos[joint_addr] = ep["joint_pos"][index, j]
            body_data.time = float(ep["time"][index])
            mujoco.mj_forward(model, body_data)
            logger.log_data(body_data)
            xy = path_xy[index]
            rr.log("spatial/robot", rr.Points2D([xy]))
            if "target_xy" in ep:
                target = ep["target_xy"][index]
                rr.log("spatial/target", rr.Points2D([target]))
                if "speed_command" in ep:
                    delta = target - xy
                    norm = np.linalg.norm(delta)
                    vector = (
                        delta / norm * ep["speed_command"][index] if norm > 1e-9 else np.zeros(2)
                    )
                    rr.log(
                        "spatial/requested_velocity", rr.Arrows2D(origins=[xy], vectors=[vector])
                    )
            if "root_lin_vel_w" in ep:
                rr.log(
                    "spatial/actual_velocity",
                    rr.Arrows2D(origins=[xy], vectors=[ep["root_lin_vel_w"][index, :2]]),
                )
            if "reward_total" in ep:
                rr.log("reward/total_step", rr.Scalars(float(ep["reward_total"][index])))
                for j, label in enumerate(labels):
                    safe = re.sub(r"[^A-Za-z0-9_-]", "_", label)
                    rr.log(
                        f"reward/terms/{j:03d}_{safe}",
                        rr.Scalars(
                            float(ep["reward_terms_rate"][index, j] * ep["reward_dt"][index])
                        ),
                    )
        rr.disconnect()
    return {
        "capture": str(capture),
        "output": str(output),
        "frames": ep["count"],
        "frames_logged": len(range(0, ep["count"], stride)),
        "reward_terms": len(labels),
        "target": "target_xy" in ep,
        "model_sha256": model_sha,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="Existing NPZ motion take")
    parser.add_argument("--output", type=Path, help="Output RRD file (default: beside input)")
    parser.add_argument("--model", type=Path, default=ROBOT_XML, help="MuJoCo robot XML")
    parser.add_argument("--stride", type=int, default=1, help="Save every Nth step")
    parser.add_argument("--collision", action="store_true", help="Include collision geometry")
    args = parser.parse_args()
    try:
        result = export_rerun(
            args.capture,
            args.output or args.capture.with_suffix(".rrd"),
            model_path=args.model,
            stride=args.stride,
            collision=args.collision,
        )
    except (OSError, ValueError, RuntimeError, mujoco.FatalError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
