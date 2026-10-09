import math
from types import SimpleNamespace

import pytest
import torch

from ascento_mjlab.viewer.waypoints import (
    WaypointController,
    normalize_waypoint_command,
)


class _Scene(dict):
    def __init__(self, *args, env_origins, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_origins = env_origins


def _env():
    position = torch.tensor([[10.0, -2.0, 0.75]])
    asset = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=position,
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            root_link_vel_w=torch.zeros((1, 6)),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
            root_link_ang_vel_b=torch.zeros((1, 3)),
        )
    )
    found = torch.ones((1, 1), dtype=torch.bool)
    scene = _Scene(
        {
            "robot": asset,
            "left_wheel_contact": SimpleNamespace(data=SimpleNamespace(found=found.clone())),
            "right_wheel_contact": SimpleNamespace(data=SimpleNamespace(found=found.clone())),
        },
        env_origins=torch.tensor([[10.0, -2.0, 0.0]]),
    )
    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        scene=scene,
        step_dt=0.1,
        ascento_world_target_state={
            "target_xy": position[:, :2].clone(),
            "target_yaw": torch.zeros(1),
        },
    )
    return env


def _speed_env(max_speed_mps=0.5):
    env = _env()

    class SpeedCommand:
        def __init__(self):
            self.cfg = SimpleNamespace(max_speed_mps=max_speed_mps, max_speed_slew_rate_mps_per_s=0.5)
            self._target_speed = torch.zeros((1, 1))
            self._command = torch.zeros((1, 1))
            self._manual_override = torch.zeros(1, dtype=torch.bool)

        @property
        def command(self):
            return self._command

        def set_manual_speed_mps(self, speed_mps, *, env_id):
            assert env_id == 0
            self._target_speed[0, 0] = speed_mps
            self._manual_override[0] = True

        def state_snapshot(self, *, env_id):
            return {
                "requested_mps": float(self._target_speed[env_id, 0]),
                "applied_mps": float(self._command[env_id, 0]),
                "max_speed_mps": self.cfg.max_speed_mps,
                "max_slew_rate_mps_per_s": self.cfg.max_speed_slew_rate_mps_per_s,
            }

    speed = SpeedCommand()
    env.command_manager = SimpleNamespace(get_term=lambda name: speed if name == "speed" else None)
    env.speed_command = speed
    return env


def test_waypoint_command_normalizes_heading_and_rejects_invalid_coordinates():
    command = normalize_waypoint_command(
        {"operation": "set", "x_m": 11, "y_m": -2, "yaw_rad": 3 * math.pi}
    )

    assert abs(command["yaw_rad"]) == pytest.approx(math.pi)
    with pytest.raises(ValueError, match="x_m"):
        normalize_waypoint_command({"operation": "set", "x_m": True, "y_m": 0})
    with pytest.raises(ValueError, match="finite"):
        normalize_waypoint_command({"operation": "queue", "x_m": float("nan"), "y_m": 0})
    with pytest.raises(ValueError, match="operation"):
        normalize_waypoint_command({"operation": "teleport"})


def test_speed_command_is_finite_and_range_checked_against_task_cap():
    assert normalize_waypoint_command({"operation": "speed", "speed_mps": 0.25}) == {
        "operation": "speed",
        "speed_mps": 0.25,
    }
    with pytest.raises(ValueError, match="speed_mps"):
        normalize_waypoint_command({"operation": "speed", "speed_mps": float("nan")})

    controller = WaypointController(_speed_env(max_speed_mps=0.4))
    with pytest.raises(ValueError, match=r"within \[0, 0.4\]"):
        controller.apply({"operation": "speed", "speed_mps": 0.5})


def test_speed_command_updates_requested_value_and_exposes_measured_state():
    env = _speed_env()
    env.scene["robot"].data.root_link_vel_w[0, :2] = torch.tensor([0.3, 0.4])
    controller = WaypointController(env)

    state = controller.apply({"operation": "speed", "speed_mps": 0.25})

    assert env.speed_command._target_speed[0, 0] == pytest.approx(0.25)
    assert state["speed_command"]["requested_mps"] == pytest.approx(0.25)
    assert state["speed_command"]["measured_mps"] == pytest.approx(0.5)


def test_waypoint_controller_enforces_arena_and_segment_limits():
    controller = WaypointController(_env(), arena_half_extent_m=4.0, max_segment_m=3.0)

    with pytest.raises(ValueError, match="outside"):
        controller.apply({"operation": "set", "x_m": 14.1, "y_m": -2.0})
    with pytest.raises(ValueError, match="maximum is 3"):
        controller.apply({"operation": "set", "x_m": 13.2, "y_m": -2.0})


def test_waypoint_queue_advances_only_after_stable_arrival_dwell():
    env = _env()
    controller = WaypointController(
        env,
        arrival_distance_m=0.08,
        arrival_dwell_s=0.3,
        max_queued=1,
    )
    controller.apply({"operation": "set", "x_m": 11.0, "y_m": -2.0})
    controller.apply({"operation": "queue", "x_m": 12.0, "y_m": -2.0})
    with pytest.raises(ValueError, match="already has 1"):
        controller.apply({"operation": "queue", "x_m": 12.5, "y_m": -2.0})

    env.scene["robot"].data.root_link_pos_w[0, :2] = torch.tensor([11.0, -2.0])
    controller.advance()
    controller.advance()
    assert controller.completed == 0
    controller.advance()

    assert controller.completed == 1
    assert controller.active is not None
    assert controller.active.x_m == pytest.approx(12.0)
    assert controller.snapshot()["state"] == "driving"

    controller.advance(reset=True)
    assert controller.active is None
    assert controller.snapshot()["state"] == "holding"
