"""World-frame waypoint commands for the one-environment simulation viewer.

HTTP, Viser, and Python callers use one command format. Goals write the
existing target_xy and target_yaw tensors consumed by the policy observation.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

# MJLab discovers registered tasks during import; initialize it before Ascento MDP imports.
import mjlab  # noqa: F401
import torch

from ascento_mjlab.mdp.events import (
    world_target_xy,
    world_target_yaw,
    yaw_from_quaternion_wxyz,
)

WAYPOINT_OPERATIONS = frozenset({"set", "queue", "hold", "resume", "cancel", "speed"})


def normalize_waypoint_command(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the shared command object before it reaches the simulator."""
    if not isinstance(payload, dict):
        raise ValueError("waypoint command must be an object")
    operation = payload.get("operation")
    if operation not in WAYPOINT_OPERATIONS:
        raise ValueError("operation must be set, queue, hold, resume, cancel, or speed")
    result: dict[str, Any] = {"operation": operation}
    if operation == "speed":
        value = payload.get("speed_mps")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("speed_mps must be a finite number of metres per second")
        result["speed_mps"] = float(value)
        return result
    if operation in {"set", "queue"}:
        for name in ("x_m", "y_m"):
            value = payload.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number of metres")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number of metres")
            result[name] = value
        yaw = payload.get("yaw_rad")
        if yaw is not None:
            if isinstance(yaw, bool) or not isinstance(yaw, (int, float)):
                raise ValueError("yaw_rad must be a finite angle in radians")
            yaw = float(yaw)
            if not math.isfinite(yaw):
                raise ValueError("yaw_rad must be a finite angle in radians")
            yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        result["yaw_rad"] = yaw
    return result


@dataclass(frozen=True)
class Waypoint:
    id: str
    x_m: float
    y_m: float
    yaw_rad: float | None

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "x_m": self.x_m, "y_m": self.y_m, "yaw_rad": self.yaw_rad}


class WaypointController:
    """Apply world goals and advance a route after stable pose arrival."""

    def __init__(
        self,
        env: Any,
        *,
        arena_half_extent_m: float = 4.0,
        max_segment_m: float = 3.0,
        arrival_distance_m: float = 0.08,
        arrival_heading_rad: float = 0.20,
        arrival_speed_m_s: float = 0.08,
        arrival_yaw_rate_rad_s: float = 0.20,
        arrival_dwell_s: float = 0.50,
        max_queued: int = 32,
    ) -> None:
        if env.num_envs != 1:
            raise ValueError("waypoint viewer requires exactly one environment")
        self.env = env
        self.arena_half_extent_m = float(arena_half_extent_m)
        self.max_segment_m = float(max_segment_m)
        self.arrival_distance_m = float(arrival_distance_m)
        self.arrival_heading_rad = float(arrival_heading_rad)
        self.arrival_speed_m_s = float(arrival_speed_m_s)
        self.arrival_yaw_rate_rad_s = float(arrival_yaw_rate_rad_s)
        self.arrival_dwell_s = float(arrival_dwell_s)
        self.max_queued = int(max_queued)
        command_manager = getattr(env, "command_manager", None)
        get_term = getattr(command_manager, "get_term", None)
        try:
            self.speed_command = get_term("speed") if callable(get_term) else None
        except (KeyError, ValueError):
            self.speed_command = None
        self.active: Waypoint | None = None
        self.queued: deque[Waypoint] = deque()
        self.paused = False
        self.arrived = False
        self.dwell_s = 0.0
        self.completed = 0
        self.last_command: dict[str, Any] | None = None
        self._hold_at_current_pose()

    def _robot(self) -> dict[str, Any]:
        asset = self.env.scene["robot"]
        yaw = yaw_from_quaternion_wxyz(asset.data.root_link_quat_w[:1])
        velocity = asset.data.root_link_vel_w[0]
        gravity = asset.data.projected_gravity_b[0]
        gravity_xy = gravity[:2].norm()
        tilt = torch.atan2(gravity_xy, -gravity[2].clamp(max=-1.0e-6))
        left_support = self.env.scene["left_wheel_contact"].data.found
        right_support = self.env.scene["right_wheel_contact"].data.found
        if left_support is None or right_support is None:
            raise RuntimeError("waypoint control requires wheel contact sensors")
        return {
            "x_m": float(asset.data.root_link_pos_w[0, 0].item()),
            "y_m": float(asset.data.root_link_pos_w[0, 1].item()),
            "yaw_rad": float(yaw[0].item()),
            "speed_m_s": float(velocity[:2].norm().item()),
            "yaw_rate_rad_s": float(asset.data.root_link_ang_vel_b[0, 2].abs().item()),
            "tilt_rad": float(tilt.item()),
            "both_wheels_supported": bool(
                left_support[0].any().item() and right_support[0].any().item()
            ),
        }

    def _write_target(self, x_m: float, y_m: float, yaw_rad: float) -> None:
        target_xy = world_target_xy(self.env)
        target_xy[0, 0] = x_m
        target_xy[0, 1] = y_m
        world_target_yaw(self.env)[0] = yaw_rad

    def _hold_at_current_pose(self) -> None:
        robot = self._robot()
        self._write_target(robot["x_m"], robot["y_m"], robot["yaw_rad"])

    def _activate(self, waypoint: Waypoint) -> None:
        robot = self._robot()
        yaw = waypoint.yaw_rad
        if yaw is None:
            dx = waypoint.x_m - robot["x_m"]
            dy = waypoint.y_m - robot["y_m"]
            yaw = math.atan2(dy, dx) if math.hypot(dx, dy) > 1.0e-6 else robot["yaw_rad"]
        self.active = Waypoint(waypoint.id, waypoint.x_m, waypoint.y_m, yaw)
        self.paused = False
        self.arrived = False
        self.dwell_s = 0.0
        self._write_target(waypoint.x_m, waypoint.y_m, yaw)

    def _validate_destination(self, x_m: float, y_m: float, operation: str) -> None:
        origin = self.env.scene.env_origins[0, :2]
        origin_x, origin_y = float(origin[0].item()), float(origin[1].item())
        if max(abs(x_m - origin_x), abs(y_m - origin_y)) > self.arena_half_extent_m:
            raise ValueError(
                f"waypoint lies outside the {self.arena_half_extent_m:g} m simulation arena"
            )
        if operation == "queue" and self.queued:
            source_x, source_y = self.queued[-1].x_m, self.queued[-1].y_m
        elif operation == "queue" and self.active is not None and not self.arrived:
            source_x, source_y = self.active.x_m, self.active.y_m
        else:
            robot = self._robot()
            source_x, source_y = robot["x_m"], robot["y_m"]
        segment = math.hypot(x_m - source_x, y_m - source_y)
        if segment > self.max_segment_m + 1.0e-6:
            raise ValueError(
                f"waypoint segment is {segment:.2f} m; maximum is {self.max_segment_m:g} m"
            )

    def apply(
        self,
        payload: dict[str, Any],
        *,
        request_id: str | None = None,
        source: str = "code",
    ) -> dict[str, Any]:
        """Apply one command on the viewer simulation thread."""
        command = normalize_waypoint_command(payload)
        operation = command["operation"]
        if operation in {"set", "queue"}:
            x_m, y_m = command["x_m"], command["y_m"]
            self._validate_destination(x_m, y_m, operation)
            waypoint = Waypoint(uuid4().hex[:12], x_m, y_m, command["yaw_rad"])
            if operation == "set":
                self.queued.clear()
                self._activate(waypoint)
            elif self.active is None or (self.arrived and not self.paused):
                self._activate(waypoint)
            else:
                if len(self.queued) >= self.max_queued:
                    raise ValueError(f"route already has {self.max_queued} queued waypoints")
                self.queued.append(waypoint)
        elif operation == "speed":
            if self.speed_command is None:
                raise ValueError("speed command is unsupported for this viewer task")
            value = command["speed_mps"]
            cap = float(self.speed_command.cfg.max_speed_mps)
            if value < 0.0 or value > cap:
                raise ValueError(f"speed must be within [0, {cap:g}] m/s")
            self.speed_command.set_manual_speed_mps(value, env_id=0)
        elif operation == "hold":
            self.paused = True
            self.dwell_s = 0.0
            self._hold_at_current_pose()
        elif operation == "resume":
            if self.active is not None:
                self.paused = False
                self.arrived = False
                self.dwell_s = 0.0
                self._write_target(
                    self.active.x_m, self.active.y_m, float(self.active.yaw_rad)
                )
            elif self.queued:
                self._activate(self.queued.popleft())
        else:
            self.active = None
            self.queued.clear()
            self.paused = False
            self.arrived = False
            self.dwell_s = 0.0
            self._hold_at_current_pose()
        self.last_command = {
            "request_id": request_id,
            "source": source,
            "operation": operation,
            "state": "accepted",
            "at": time.time(),
        }
        return self.snapshot()

    def reject(
        self, payload: dict[str, Any], error: Exception, *, request_id: str | None, source: str
    ) -> None:
        self.last_command = {
            "request_id": request_id,
            "source": source,
            "operation": payload.get("operation"),
            "state": "rejected",
            "error": str(error),
            "at": time.time(),
        }

    def advance(self, *, reset: bool = False) -> None:
        """Update dwell and progress the queue after one simulation step."""
        if reset:
            self.active = None
            self.queued.clear()
            self.paused = False
            self.arrived = False
            self.dwell_s = 0.0
            self._hold_at_current_pose()
            return
        if self.active is None or self.paused or self.arrived:
            return
        robot = self._robot()
        distance = math.hypot(
            self.active.x_m - robot["x_m"], self.active.y_m - robot["y_m"]
        )
        heading_error = math.atan2(
            math.sin(float(self.active.yaw_rad) - robot["yaw_rad"]),
            math.cos(float(self.active.yaw_rad) - robot["yaw_rad"]),
        )
        settled = (
            distance <= self.arrival_distance_m
            and abs(heading_error) <= self.arrival_heading_rad
            and robot["speed_m_s"] <= self.arrival_speed_m_s
            and robot["yaw_rate_rad_s"] <= self.arrival_yaw_rate_rad_s
            and robot["tilt_rad"] <= 0.08
            and robot["both_wheels_supported"]
        )
        self.dwell_s = self.dwell_s + float(self.env.step_dt) if settled else 0.0
        if self.dwell_s >= self.arrival_dwell_s:
            self.completed += 1
            self.arrived = True
            if self.queued:
                self._activate(self.queued.popleft())

    def snapshot(self) -> dict[str, Any]:
        robot = self._robot()
        active = self.active.as_dict() if self.active is not None else None
        target_xy = world_target_xy(self.env)[0]
        target = {
            "x_m": float(target_xy[0].item()),
            "y_m": float(target_xy[1].item()),
            "yaw_rad": float(world_target_yaw(self.env)[0].item()),
        }
        distance = (
            math.hypot(active["x_m"] - robot["x_m"], active["y_m"] - robot["y_m"])
            if active is not None else 0.0
        )
        state = (
            "holding" if self.active is None else
            "paused" if self.paused else
            "arrived" if self.arrived else
            "driving"
        )
        snapshot = {
            "available": True,
            "frame": "sim_world",
            "state": state,
            "robot": robot,
            "target": target,
            "active": active,
            "queue": [item.as_dict() for item in self.queued],
            "distance_m": distance,
            "dwell_s": self.dwell_s,
            "completed": self.completed,
            "arrival": {
                "distance_m": self.arrival_distance_m,
                "heading_rad": self.arrival_heading_rad,
                "speed_m_s": self.arrival_speed_m_s,
                "yaw_rate_rad_s": self.arrival_yaw_rate_rad_s,
                "dwell_s": self.arrival_dwell_s,
            },
            "limits": {
                "arena_half_extent_m": self.arena_half_extent_m,
                "max_segment_m": self.max_segment_m,
                "max_queued": self.max_queued,
            },
            "last_command": self.last_command,
            "updated_at": time.time(),
        }
        if self.speed_command is not None:
            speed_state = self.speed_command.state_snapshot(env_id=0)
            speed_state["measured_mps"] = robot["speed_m_s"]
            snapshot["speed_command"] = speed_state
        origin = self.env.scene.env_origins[0, :2]
        snapshot["origin"] = {"x_m": float(origin[0].item()), "y_m": float(origin[1].item())}
        return snapshot
