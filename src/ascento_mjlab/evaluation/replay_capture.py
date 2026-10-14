"""Bounded, opt-in, exact-suite rollout captures for Rerun.

This recorder is called inside the evaluator after applying scenario-specific
resets, commands and disturbance forces. It never samples training environments.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ascento_mjlab.robot_cfg import JOINT_NAMES, ROBOT_XML

from .schema import ScenarioSpec


def _sample(tensor: torch.Tensor, index: int) -> torch.Tensor:
    # Keep only selected-scenario frames on GPU. Thousands of per-channel
    # device-to-host synchronizations otherwise stall MuJoCo Warp playback.
    return tensor[index].detach().clone()


class BenchmarkReplayRecorder:
    """Record only explicitly chosen scenarios, with opt-in GPU readbacks."""

    def __init__(
        self,
        scenario_ids: set[str],
        output_dir: Path,
        *,
        checkpoint: Path,
        suite_path: Path,
        suite_sha256: str,
    ) -> None:
        if not scenario_ids:
            raise ValueError("Choose at least one scenario to record")
        self.selected = set(scenario_ids)
        self.output_dir = Path(output_dir)
        self.checkpoint = Path(checkpoint).resolve()
        self.suite_path = Path(suite_path).resolve()
        self.suite_sha256 = suite_sha256
        self.frames: dict[str, list[dict[str, torch.Tensor]]] = {}
        self.scenarios: dict[str, ScenarioSpec] = {}
        self.step_dt: float | None = None
        self.reward_names: tuple[str, ...] | None = None

    def record_step(
        self,
        *,
        scenarios: list[ScenarioSpec],
        step: int,
        active: torch.Tensor,
        robot: Any,
        raw_action: torch.Tensor,
        step_rewards: torch.Tensor,
        reward_terms: torch.Tensor,
        reward_term_names: list[str],
        step_dt: float,
        target_xy: torch.Tensor,
        target_yaw: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
        target_error: torch.Tensor,
        heading_error: torch.Tensor,
        planar_speed: torch.Tensor,
        tilt: torch.Tensor,
        angular_xy: torch.Tensor,
        yaw_rate: torch.Tensor,
        waypoint_active: torch.Tensor,
        waypoint_dwell_count: torch.Tensor,
        stop_flags: dict[str, torch.Tensor],
        stop_now: torch.Tensor,
        recovery_flags: dict[str, torch.Tensor],
        recovery_stable_count: torch.Tensor,
        recovered: torch.Tensor,
        after_disturbance: torch.Tensor,
    ) -> None:
        if self.step_dt is not None and self.step_dt != step_dt:
            raise RuntimeError("Capture step_dt changed between batches")
        if self.reward_names is not None and self.reward_names != tuple(reward_term_names):
            raise RuntimeError("Capture reward schema changed between batches")
        self.step_dt = step_dt
        self.reward_names = tuple(reward_term_names)
        for i, scenario in enumerate(scenarios):
            if scenario.scenario_id not in self.selected or not bool(active[i].item()):
                continue
            key = scenario.scenario_id
            self.scenarios[key] = scenario
            frame: dict[str, torch.Tensor] = {
                "time": torch.tensor(
                    (step + 1) * step_dt, dtype=torch.float64, device=active.device
                ),
                "root_pos": _sample(robot.data.root_link_pos_w, i),
                "root_quat": _sample(robot.data.root_link_quat_w, i),
                "root_lin_vel_w": _sample(robot.data.root_link_lin_vel_w, i),
                "root_ang_vel_b": _sample(robot.data.root_link_ang_vel_b, i),
                "joint_pos": _sample(robot.data.joint_pos, i),
                "joint_vel": _sample(robot.data.joint_vel, i),
                "effort": _sample(robot.data.actuator_force, i),
                "action": _sample(raw_action, i),
                "target_xy": _sample(target_xy, i),
                "target_yaw": _sample(target_yaw, i),
                "contacts": torch.stack((left[i], right[i])).float(),
                "reward_total": _sample(step_rewards, i),
                "reward_terms_rate": _sample(reward_terms, i),
                "reward_dt": torch.tensor(step_dt, dtype=torch.float32, device=active.device),
                "target_error_m": _sample(target_error, i),
                "heading_error_rad": _sample(heading_error, i),
                "planar_speed_mps": _sample(planar_speed, i),
                "tilt_rad": _sample(tilt, i),
                "angular_xy_radps": _sample(angular_xy, i),
                "yaw_rate_radps": _sample(yaw_rate, i),
                "evaluator_stop_active": _sample(waypoint_active.float(), i),
                "evaluator_stop_all_pass": _sample(stop_now.float(), i),
                "evaluator_stop_dwell_s": _sample(waypoint_dwell_count.float() * step_dt, i),
                "evaluator_recovery_after_disturbance": _sample(after_disturbance.float(), i),
                "evaluator_recovery_stable_dwell_s": _sample(
                    recovery_stable_count.float() * step_dt, i
                ),
                "evaluator_recovered": _sample(recovered.float(), i),
            }
            for name, flag in stop_flags.items():
                frame[f"evaluator_stop_{name}"] = _sample(flag.float(), i)
            for name, flag in recovery_flags.items():
                frame[f"evaluator_recovery_{name}"] = _sample(flag.float(), i)
            self.frames.setdefault(key, []).append(frame)

    def save(self) -> list[Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        robot_sha = hashlib.sha256(ROBOT_XML.read_bytes()).hexdigest()
        checkpoint_sha = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        paths: list[Path] = []
        for scenario_id in sorted(self.selected):
            frames = self.frames.get(scenario_id)
            if not frames:
                raise RuntimeError(f"Selected scenario has no recorded frames: {scenario_id}")
            scenario = self.scenarios[scenario_id]
            arrays: dict[str, np.ndarray] = {
                key: torch.stack([frame[key] for frame in frames]).detach().to(device="cpu").numpy()
                for key in frames[0]
            }
            metadata = {
                "task": scenario.task,
                "scenario_id": scenario_id,
                "scenario": json.dumps(asdict(scenario), sort_keys=True),
                "suite": str(self.suite_path),
                "suite_sha256": self.suite_sha256,
                "checkpoint": str(self.checkpoint),
                "checkpoint_sha256": checkpoint_sha,
                "robot_xml_sha256": robot_sha,
                "joint_names": json.dumps(JOINT_NAMES),
                "reward_term_names": json.dumps(self.reward_names),
                "reward_term_units": "weighted_rate_before_dt",
                "capture_schema": "benchmark_evaluator_v1",
                "step_dt": str(self.step_dt),
                "physics_source": "exact_benchmark_evaluator",
            }
            arrays.update({f"meta_{key}": np.asarray(value) for key, value in metadata.items()})
            family, index = scenario.family, scenario_id.rsplit("/", 1)[-1]
            path = self.output_dir / f"{family}_{index}.npz"
            np.savez_compressed(path, **arrays)
            paths.append(path)
        return paths
