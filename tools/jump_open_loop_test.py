"""Direct-torque jump feasibility sweep; it is a diagnostic, not a controller."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from guard2_physics import Guard2ActuatorModel



def parse_values(text: str):
    return tuple(float(value) for value in text.split(","))


def run_trial(model, crouch_torque, crouch_steps, thrust_torque, thrust_steps, wheel_torque):
    data = mujoco.MjData(model)
    data.qpos[:7] = [0.0, 0.0, 0.75, 1.0, 0.0, 0.0, 0.0]
    data.qpos[7:] = [-np.pi, -np.pi, 0.0, -np.pi, -np.pi, 0.0]
    mujoco.mj_forward(model, data)
    motors = Guard2ActuatorModel(model.opt.timestep)
    left_wheel = model.body("left_wheel").id
    right_wheel = model.body("right_wheel").id
    takeoff = False
    takeoff_z = data.qpos[2]
    apex_z = takeoff_z
    wheel_apex = max(data.xpos[left_wheel, 2], data.xpos[right_wheel, 2])
    landing_vz = None
    max_torque = 0.0
    sequence = ((-crouch_torque, crouch_steps), (thrust_torque, thrust_steps), (0.0, 400))
    for leg_torque, steps in sequence:
        for _ in range(steps):
            requested = np.array([leg_torque, leg_torque, wheel_torque, leg_torque, leg_torque, -wheel_torque])
            data.ctrl[:] = motors.step(requested, data.qpos[7:13], data.qvel[6:12])
            max_torque = max(max_torque, float(np.abs(data.ctrl).max()))
            mujoco.mj_step(model, data)
            wheel_z = np.array([data.xpos[left_wheel, 2], data.xpos[right_wheel, 2]])
            airborne = bool(np.all(wheel_z > 0.265))
            if airborne and not takeoff:
                takeoff, takeoff_z = True, float(data.qpos[2])
            if takeoff and not airborne and landing_vz is None:
                landing_vz = float(data.qvel[2])
            apex_z = max(apex_z, float(data.qpos[2]))
            wheel_apex = max(wheel_apex, float(wheel_z.max()))
    return {
        "crouch_torque": crouch_torque,
        "crouch_steps": crouch_steps,
        "thrust_torque": thrust_torque,
        "thrust_steps": thrust_steps,
        "wheel_torque": wheel_torque,
        "takeoff": takeoff,
        "takeoff_vz": float(data.qvel[2]) if takeoff else 0.0,
        "body_rise": apex_z - takeoff_z if takeoff else 0.0,
        "wheel_clearance": max(0.0, wheel_apex - 0.25),
        "landing_vz": landing_vz,
        "max_applied_torque": max_torque,
        "joint_limit_ok": bool(np.all(data.qpos[[7, 8, 10, 11]] >= -1.5 * np.pi - 1e-4) and np.all(data.qpos[[7, 8, 10, 11]] <= -0.5 * np.pi + 1e-4)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "tools" / "jump_open_loop_results")
    parser.add_argument("--crouch-torques", default="10,20")
    parser.add_argument("--crouch-steps", default="40,80")
    parser.add_argument("--thrust-torques", default="20,35")
    parser.add_argument("--thrust-steps", default="40,80")
    parser.add_argument("--wheel-torques", default="0,4")
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(ROOT / "model" / "ascento_guard2_mjx.xml"))
    results = [run_trial(model, *trial) for trial in itertools.product(
        parse_values(args.crouch_torques), (int(value) for value in parse_values(args.crouch_steps)),
        parse_values(args.thrust_torques), (int(value) for value in parse_values(args.thrust_steps)), parse_values(args.wheel_torques),
    )]
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"trials": len(results), "takeoffs": sum(row["takeoff"] for row in results)}, indent=2))


if __name__ == "__main__":
    main()
