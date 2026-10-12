"""Low-overhead sampled telemetry from actual PPO environments (never a viewer)."""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

import torch

LOG = logging.getLogger(__name__)
SCHEMA_VERSION = 1
MAX_SELECTED_ENVS = 4


def selected_environment_ids(raw: str, num_envs: int) -> tuple[int, ...]:
    """Validate a small, stable cohort; never copy the full vectorized simulation."""
    try:
        ids = tuple(int(part.strip()) for part in raw.split(","))
    except ValueError as exc:
        raise ValueError("ASCENTO_LIVE_MONITOR_ENVS must be comma-separated indices") from exc
    if (
        not ids
        or len(ids) > MAX_SELECTED_ENVS
        or len(set(ids)) != len(ids)
        or any(index < 0 or index >= num_envs for index in ids)
    ):
        raise ValueError("Live monitor needs 1-4 distinct, valid environment IDs")
    return ids


class LatestFrameWriter:
    """Bounded, replaceable latest-frame mailbox; no I/O on the PPO step thread."""

    def __init__(self, output: Path, *, capacity: int = 4) -> None:
        self.output = Path(output)
        self.pending: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=capacity)
        self.dropped = 0
        self._stopped = threading.Event()
        self._worker = threading.Thread(
            target=self._run, name="ascento-live-telemetry", daemon=True
        )
        self._worker.start()

    def offer(self, frame: dict[str, Any]) -> None:
        if self._stopped.is_set():
            return
        try:
            self.pending.put_nowait(frame)
        except queue.Full:
            try:
                self.pending.get_nowait()
            except queue.Empty:
                pass
            self.dropped += 1
            try:
                self.pending.put_nowait(frame)
            except queue.Full:
                self.dropped += 1

    def _run(self) -> None:
        try:
            self.output.parent.mkdir(parents=True, exist_ok=True)
            while True:
                frame = self.pending.get()
                if frame is None:
                    return
                frame["dropped_display_frames"] = self.dropped
                tmp = self.output.with_name(f".{self.output.name}.{os.getpid()}.tmp")
                tmp.write_text(
                    json.dumps(frame, allow_nan=False, separators=(",", ":")), encoding="utf-8"
                )
                os.replace(tmp, self.output)
        except (OSError, TypeError, ValueError):
            # Display telemetry must never affect the trainer.
            LOG.warning("Live telemetry writer disabled after filesystem error", exc_info=True)
            self._stopped.set()

    def close(self) -> None:
        if self._stopped.is_set():
            return
        # Make sure the last offered frame can be flushed.
        try:
            self.pending.put_nowait(None)
        except queue.Full:
            try:
                self.pending.get_nowait()
            except queue.Empty:
                pass
            try:
                self.pending.put_nowait(None)
            except queue.Full:
                pass
        self._worker.join(timeout=1.0)
        self._stopped.set()


class PpoLiveMonitor:
    """Sample a few environment rows after an actual RSL-RL rollout step."""

    def __init__(
        self,
        env: Any,
        output: Path,
        *,
        environment_ids: tuple[int, ...] = (0,),
        interval_s: float = 0.5,
        clock=time.monotonic,
        writer: LatestFrameWriter | None = None,
    ) -> None:
        if not 0.2 <= interval_s <= 10:
            raise ValueError("Live sampling interval must be 0.2-10 seconds")
        source = env.unwrapped
        if any(i < 0 or i >= source.num_envs for i in environment_ids):
            raise ValueError("Live environment index is outside the vectorized environment")
        self.env = env
        self.source = source
        self.indices = environment_ids
        self.interval_s = interval_s
        self.clock = clock
        self.next_sample = clock()  # first post-step snapshot is available immediately
        self.writer = writer if writer is not None else LatestFrameWriter(output)
        self.sequence = 0
        self._original_step = env.step
        self._had_instance_step = "step" in vars(env)
        self._disabled = False
        env.step = self._step

    def _step(self, actions: torch.Tensor):
        result = self._original_step(actions)
        now = self.clock()
        if not self._disabled and now >= self.next_sample:
            self.next_sample = now + self.interval_s
            try:
                self._capture(result, now)
            except Exception:
                # Monitoring is strictly best-effort. A malformed new task's
                # telemetry must never abort its policy rollout.
                self._disabled = True
                LOG.warning("Live training monitor disabled (training unaffected)", exc_info=True)
        return result

    def _capture(self, result: Any, now: float) -> None:
        start = now
        source = self.source
        robot = source.scene["robot"]
        dev = robot.data.root_link_pos_w.device
        row_ids = torch.tensor(self.indices, device=dev, dtype=torch.long)

        def selected(values: torch.Tensor) -> torch.Tensor:
            return values.index_select(0, row_ids).reshape(len(self.indices), -1).float()

        def sensor_flags(name: str) -> torch.Tensor:
            try:
                found = source.scene[name].data.found
                if found is not None:
                    return selected(found).amax(dim=1, keepdim=True)
            except (KeyError, AttributeError):
                pass
            return torch.full((len(self.indices), 1), -1, device=dev)

        target_state = getattr(source, "ascento_world_target_state", None)
        target = target_state.get("target_xy") if isinstance(target_state, dict) else None
        command = None
        try:
            command = source.command_manager.get_command("speed")
        except (AttributeError, KeyError, ValueError):
            pass

        reward_manager = source.reward_manager
        reward_terms = getattr(reward_manager, "_step_reward", None)
        labels = tuple(str(label) for label in reward_manager.active_terms)
        if not isinstance(reward_terms, torch.Tensor) or reward_terms.ndim != 2:
            reward_terms = None
            labels = ()

        channels = [
            ("root_pos", selected(robot.data.root_link_pos_w)),
            ("root_quat", selected(robot.data.root_link_quat_w)),
            ("velocity_world", selected(robot.data.root_link_lin_vel_w)),
            ("projected_gravity", selected(robot.data.projected_gravity_b)),
            ("joint_pos", selected(robot.data.joint_pos)),
            ("episode_step", selected(source.episode_length_buf)),
            ("done", selected(result[2])),
            ("step_reward", selected(result[1])),
            (
                "target",
                selected(target)
                if isinstance(target, torch.Tensor)
                else torch.full((len(self.indices), 2), float("nan"), device=dev),
            ),
            (
                "speed_command",
                selected(command)
                if isinstance(command, torch.Tensor)
                else torch.full((len(self.indices), 1), float("nan"), device=dev),
            ),
            ("left_contact", sensor_flags("left_wheel_contact")),
            ("right_contact", sensor_flags("right_wheel_contact")),
        ]
        if reward_terms is not None:
            channels.append(("reward_terms_rate", selected(reward_terms)))
        widths = [values.shape[1] for _, values in channels]
        # One bounded GPU->CPU synchronization for all selected rows/channels.
        combined = torch.cat([values for _, values in channels], dim=1).detach().cpu().tolist()
        self.sequence += 1
        elapsed = self.clock() - start
        items = []
        for env_id, values in zip(self.indices, combined, strict=True):
            offsets = {}
            at = 0
            for (name, _), width in zip(channels, widths, strict=True):
                offsets[name] = values[at : at + width]
                at += width
            pos, velocity = offsets["root_pos"], offsets["velocity_world"]
            target_xy = offsets["target"]
            has_target = all(math.isfinite(value) for value in target_xy)
            distance = math.dist(pos[:2], target_xy) if has_target else None
            delta = [target_xy[i] - pos[i] for i in range(2)] if has_target else [0.0, 0.0]
            closing = (
                sum(velocity[i] * delta[i] for i in range(2)) / max(distance, 1e-6)
                if distance is not None and distance > 1e-6
                else None
            )
            gravity_z = offsets["projected_gravity"][2]
            tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, -gravity_z))))
            step_reward = offsets["step_reward"][0]
            reward_rates = offsets.get("reward_terms_rate", [])
            # The reward manager already applied term weights; do not call its functions.
            item = {
                "environment_id": env_id,
                "episode_step": int(offsets["episode_step"][0]),
                "reset": bool(offsets["done"][0] > 0.5),
                "position_xyz": pos,
                "quaternion_wxyz": offsets["root_quat"],
                "velocity_xy": velocity[:2],
                "speed_mps": math.hypot(*velocity[:2]),
                "target_xy": target_xy if has_target else None,
                "target_distance_m": distance,
                "closing_speed_mps": closing,
                "speed_command_mps": offsets["speed_command"][0]
                if math.isfinite(offsets["speed_command"][0])
                else None,
                "tilt_deg": tilt_deg,
                "root_height_m": pos[2],
                "joint_positions_rad": offsets["joint_pos"],
                "wheel_contacts": {
                    "left": None
                    if offsets["left_contact"][0] < 0
                    else bool(offsets["left_contact"][0]),
                    "right": None
                    if offsets["right_contact"][0] < 0
                    else bool(offsets["right_contact"][0]),
                },
                "step_reward": step_reward,
                "reward_terms_rate": dict(zip(labels, reward_rates, strict=True)),
            }
            items.append(item)
        frame = {
            "schema_version": SCHEMA_VERSION,
            "source": "actual_ppo_rollout",
            "sampled_at": time.time(),
            "sequence": self.sequence,
            "global_step": int(source.common_step_counter),
            "simulation_time_s": float(source.common_step_counter * source.step_dt),
            "step_dt_s": float(source.step_dt),
            "sample_wall_interval_s": self.interval_s,
            "capture_overhead_ms": round(elapsed * 1000, 3),
            "environments": items,
        }
        # Drop the diagnostic frame rather than emit JSON NaN/Inf or affect PPO.
        if not all(
            math.isfinite(float(value))
            for item in items
            for value in [
                item["speed_mps"],
                item["tilt_deg"],
                item["root_height_m"],
                item["step_reward"],
            ]
        ):
            return
        self.writer.offer(frame)

    def close(self) -> None:
        if self.env.__dict__.get("step") == self._step:
            if self._had_instance_step:
                self.env.step = self._original_step
            else:
                del self.env.step
        self.writer.close()


def install_training_monitor(runner: Any, log_dir: str | None) -> PpoLiveMonitor | None:
    """Only training runs with persistent logs should install sampling."""
    if not log_dir or os.getenv("ASCENTO_LIVE_MONITOR", "1").lower() in {"0", "false", "off"}:
        return None
    selected = selected_environment_ids(
        os.getenv("ASCENTO_LIVE_MONITOR_ENVS", "0"), runner.env.unwrapped.num_envs
    )
    interval = float(os.getenv("ASCENTO_LIVE_MONITOR_INTERVAL_S", "0.5"))
    return PpoLiveMonitor(
        runner.env,
        Path(log_dir) / "live_monitor.json",
        environment_ids=selected,
        interval_s=interval,
    )
