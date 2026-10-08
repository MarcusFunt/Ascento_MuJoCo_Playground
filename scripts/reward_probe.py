"""Integrate generalist reward terms over deterministic scripted state probes.

These probes isolate reward semantics from policy performance and physics. They
are useful for detecting missing or conflicting incentives, but they are not
simulator rollouts and must never be used as policy-quality evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from ascento_mjlab.tasks.generalist_locomotion.env_cfg import (
    ascento_generalist_locomotion_env_cfg,
)

DT = 0.01
DEFAULT_JOINT_POSITION = torch.tensor(
    [[-math.pi, -math.pi, 0.0, -math.pi, -math.pi, 0.0]], dtype=torch.float32
)


def _quaternion_wxyz(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return torch.tensor(
        [
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ]
        ],
        dtype=torch.float32,
    )


def _projected_gravity_b(roll: float, pitch: float) -> torch.Tensor:
    quaternion = _quaternion_wxyz(roll, pitch, 0.0)[0]
    w, x, y, z = quaternion
    return torch.tensor(
        [
            [
                -2.0 * (x * z - w * y),
                -2.0 * (y * z + w * x),
                -(1.0 - 2.0 * (x.square() + y.square())),
            ]
        ],
        dtype=torch.float32,
    )


def _make_probe_env() -> SimpleNamespace:
    leg_controller = SimpleNamespace(controller_requested_effort=torch.zeros((1, 4)))
    wheel_controller = SimpleNamespace(controller_requested_effort=torch.zeros((1, 2)))
    data = SimpleNamespace(
        root_link_pos_w=torch.tensor([[0.0, 0.0, 0.75]]),
        root_link_lin_vel_w=torch.zeros((1, 3)),
        root_link_lin_vel_b=torch.zeros((1, 3)),
        root_link_ang_vel_b=torch.zeros((1, 3)),
        root_link_quat_w=_quaternion_wxyz(0.0, 0.0, 0.0),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
        joint_pos=DEFAULT_JOINT_POSITION.clone(),
        joint_vel=torch.zeros((1, 6)),
        default_joint_pos=DEFAULT_JOINT_POSITION.clone(),
        actuator_force=torch.zeros((1, 6)),
        qfrc_actuator=torch.zeros((1, 6)),
    )
    robot = SimpleNamespace(
        data=data,
        actuators=[leg_controller, wheel_controller],
        joint_names=(
            "left_hip",
            "left_knee",
            "left_wheel_joint",
            "right_hip",
            "right_knee",
            "right_wheel_joint",
        ),
    )
    contact = SimpleNamespace(data=SimpleNamespace(found=torch.ones((1, 1), dtype=torch.bool)))
    action = torch.zeros((1, 6))
    env = SimpleNamespace(
        device=torch.device("cpu"),
        num_envs=1,
        step_dt=DT,
        scene={
            "robot": robot,
            "left_wheel_contact": contact,
            "right_wheel_contact": contact,
        },
        action_manager=SimpleNamespace(
            action=action.clone(),
            prev_action=action.clone(),
            prev_prev_action=action.clone(),
        ),
        termination_manager=SimpleNamespace(terminated=torch.zeros((1,), dtype=torch.bool)),
        ascento_world_target_state={
            "target_xy": torch.zeros((1, 2)),
            "target_yaw": torch.zeros((1,)),
        },
    )
    return env


def _state_for(case_id: str, step: int, steps: int, target_distance_m: float) -> dict[str, float]:
    time_s = (step + 1) * DT
    travel_s = {
        "successful_15cm_arrival_and_settle": 1.0,
        "near_target_overshoot": 0.7,
        "arrive_but_wrong_heading": 1.0,
        "arrive_but_keep_rocking": 1.0,
        "medium_target_success": 3.0,
        "long_target_success": 10.0,
    }[case_id]

    if case_id == "near_target_overshoot":
        overshoot_distance = max(0.10, target_distance_m * 0.40)
        if time_s <= travel_s:
            x = target_distance_m * time_s / travel_s
            vx = target_distance_m / travel_s
        elif time_s <= travel_s + 0.30:
            overshoot_time = time_s - travel_s
            x = target_distance_m + overshoot_distance * overshoot_time / 0.30
            vx = overshoot_distance / 0.30
        else:
            x, vx = target_distance_m + overshoot_distance, 0.0
    elif time_s <= travel_s:
        x = target_distance_m * time_s / travel_s
        vx = target_distance_m / travel_s
    else:
        x, vx = target_distance_m, 0.0

    roll = pitch = roll_rate = yaw = 0.0
    if case_id == "arrive_but_wrong_heading":
        yaw = 0.50
    elif case_id == "arrive_but_keep_rocking" and time_s > travel_s:
        frequency_hz = 4.0
        roll = 0.035 * math.sin(2.0 * math.pi * frequency_hz * time_s)
        roll_rate = (
            0.035 * 2.0 * math.pi * frequency_hz * math.cos(2.0 * math.pi * frequency_hz * time_s)
        )

    wheel_action = max(-1.0, min(1.0, vx / 0.5))
    leg_action = 0.0
    if case_id == "arrive_but_keep_rocking" and time_s > travel_s:
        leg_action = 0.04 * math.sin(2.0 * math.pi * 4.0 * time_s)
    return {
        "time_s": time_s,
        "x": x,
        "vx": vx,
        "roll": roll,
        "pitch": pitch,
        "roll_rate": roll_rate,
        "yaw": yaw,
        "wheel_action": wheel_action,
        "leg_action": leg_action,
    }


def _apply_state(env: SimpleNamespace, state: dict[str, float]) -> None:
    robot = env.scene["robot"]
    data = robot.data
    data.root_link_pos_w[0] = torch.tensor([state["x"], 0.0, 0.75])
    data.root_link_lin_vel_w[0] = torch.tensor([state["vx"], 0.0, 0.0])
    yaw = state["yaw"]
    data.root_link_lin_vel_b[0] = torch.tensor(
        [math.cos(yaw) * state["vx"], -math.sin(yaw) * state["vx"], 0.0]
    )
    data.root_link_ang_vel_b[0] = torch.tensor([state["roll_rate"], 0.0, 0.0])
    data.root_link_quat_w = _quaternion_wxyz(state["roll"], state["pitch"], yaw)
    data.projected_gravity_b = _projected_gravity_b(state["roll"], state["pitch"])

    actions = torch.tensor(
        [
            [
                state["leg_action"],
                state["leg_action"],
                state["wheel_action"],
                state["leg_action"],
                state["leg_action"],
                state["wheel_action"],
            ]
        ],
        dtype=torch.float32,
    )
    env.action_manager.action = actions
    data.joint_pos = data.default_joint_pos + torch.tensor(
        [
            [
                state["leg_action"] * math.pi / 2.0,
                state["leg_action"] * math.pi / 2.0,
                0.0,
                state["leg_action"] * math.pi / 2.0,
                state["leg_action"] * math.pi / 2.0,
                0.0,
            ]
        ],
        dtype=torch.float32,
    )
    requested = torch.tensor(
        [
            [
                state["leg_action"] * 10.0,
                state["leg_action"] * 10.0,
                state["wheel_action"] * 30.0,
                state["leg_action"] * 10.0,
                state["leg_action"] * 10.0,
                state["wheel_action"] * 30.0,
            ]
        ],
        dtype=torch.float32,
    )
    robot.actuators[0].controller_requested_effort = requested[:, [0, 1, 3, 4]]
    robot.actuators[1].controller_requested_effort = requested[:, [2, 5]]
    data.actuator_force = requested * 0.75
    data.qfrc_actuator = requested * 0.75


def _integrate_case(case_id: str, distance_m: float, duration_s: float) -> dict[str, Any]:
    env = _make_probe_env()
    env.ascento_world_target_state["target_xy"][0] = torch.tensor([distance_m, 0.0])
    env.ascento_world_target_state["target_yaw"][0] = 0.0
    cfg = ascento_generalist_locomotion_env_cfg(play=True, num_envs=1)
    term_states: dict[str, Any] = {}
    integrated: dict[str, float] = {name: 0.0 for name in cfg.rewards}
    steps = round(duration_s / DT)

    for step in range(steps):
        state = _state_for(case_id, step, steps, distance_m)
        _apply_state(env, state)
        for name, term_cfg in cfg.rewards.items():
            func = term_states.get(name)
            if func is None:
                candidate = term_cfg.func
                func = candidate(term_cfg, env) if isinstance(candidate, type) else candidate
                term_states[name] = func
            value = func(env, **term_cfg.params)
            integrated[name] += float(term_cfg.weight) * float(value.reshape(-1)[0]) * DT
        env.action_manager.prev_prev_action = env.action_manager.prev_action.clone()
        env.action_manager.prev_action = env.action_manager.action.clone()

    total = sum(integrated.values())
    return {
        "case_id": case_id,
        "target_distance_m": distance_m,
        "duration_s": steps * DT,
        "steps": steps,
        "final_target_error_m": abs(distance_m - state["x"]),
        "final_heading_error_rad": abs(state["yaw"]),
        "final_roll_rate_rad_s": state["roll_rate"],
        "weighted_integrated_return_by_term": integrated,
        "weighted_return_rate_by_term_per_s": {
            name: value / (steps * DT) for name, value in integrated.items()
        },
        "weighted_integrated_return_total": total,
        "weighted_return_rate_total_per_s": total / (steps * DT),
    }


def build_report() -> dict[str, Any]:
    probes = (
        ("successful_15cm_arrival_and_settle", 0.15, 11.0),
        ("near_target_overshoot", 0.15, 11.0),
        ("arrive_but_wrong_heading", 0.15, 11.0),
        ("arrive_but_keep_rocking", 0.15, 11.0),
        ("medium_target_success", 0.75, 11.0),
        ("long_target_success", 2.50, 11.0),
    )
    return {
        "schema_version": 1,
        "probe_type": "scripted_state_trajectories_not_policy_rollouts",
        "timestep_s": DT,
        "task": "Ascento-Generalist-Locomotion-Flat",
        "reward_schema": "v3",
        "cases": [
            _integrate_case(case_id, distance_m, duration_s)
            for case_id, distance_m, duration_s in probes
        ],
        "interpretation_limit": (
            "This isolates the configured reward arithmetic over hand-authored states. "
            "It does not establish dynamic feasibility or learned-policy performance."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write JSON to this path instead of stdout")
    args = parser.parse_args()
    payload = json.dumps(build_report(), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
