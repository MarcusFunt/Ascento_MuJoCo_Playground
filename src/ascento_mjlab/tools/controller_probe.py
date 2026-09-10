"""Characterize the deterministic structured-target controller before training.

The probe uses the production Balance environment with its stochastic push event
disabled.  It is intentionally a plant/controller check, not a policy score:
it starts from the exact nominal supported pose and verifies neutral hold,
wheel-direction conventions, and indexed PI-state reset behavior.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import ascento_mjlab.tasks  # noqa: F401
from ascento_mjlab.mdp.events import reset_to_default_supported
from ascento_mjlab.mdp.metrics import tilt_radians
from ascento_mjlab.physics import PHYSICS_PROFILE

NEUTRAL_MAX_TILT_RAD = 0.05
NEUTRAL_MAX_PLANAR_SPEED_M_S = 0.05
NEUTRAL_MAX_DISPLACEMENT_M = 0.05
MIN_FORWARD_DISPLACEMENT_M = 0.01
MIN_TURN_ANGLE_RAD = 0.01
# With left-wheel target positive and right-wheel target negative, base yaw is
# negative: a clockwise turn when viewed from above.
OPPOSING_TARGET_YAW_SIGN = -1.0


def _body_forward_xy(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    """Return the base +X axis in world XY for one or more WXYZ quaternions."""
    w, x, y, z = quaternion_wxyz.unbind(dim=-1)
    forward = torch.stack(
        (1.0 - 2.0 * (y.square() + z.square()), 2.0 * (x * y + w * z)), dim=-1
    )
    return forward / torch.linalg.vector_norm(forward, dim=-1, keepdim=True).clamp(min=1.0e-8)


def _yaw(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quaternion_wxyz.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))


def _wrapped_difference(after: torch.Tensor, before: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(after - before), torch.cos(after - before))


def _step_count(duration_s: float, step_dt_s: float) -> int:
    if duration_s <= 0.0 or step_dt_s <= 0.0:
        raise ValueError("durations and step timestep must be positive")
    return max(1, int(round(duration_s / step_dt_s)))


def characterize(
    *,
    device: str = "cpu",
    duration_s: float = 20.0,
    direction_duration_s: float = 0.10,
    wheel_action: float = 0.05,
) -> dict[str, Any]:
    """Run and return the controller's deterministic physical characterization.

    Four worlds share one production environment: neutral hold, equal positive
    targets, opposing targets, and a PI partial-reset witness.  The target
    action stays well inside the normalized action bounds so direction checks
    measure sign conventions rather than clipping behavior.
    """
    if not 0.0 < wheel_action <= 1.0:
        raise ValueError("wheel_action must be in (0, 1]")
    if direction_duration_s > duration_s:
        raise ValueError("direction_duration_s cannot exceed duration_s")

    cfg = load_env_cfg("Ascento-Balance-Flat", play=True)
    cfg.scene.num_envs = 4
    cfg.auto_reset = False
    cfg.events.pop("balance_push", None)
    cfg.episode_length_s = duration_s + PHYSICS_PROFILE.control_dt_s
    env = ManagerBasedRlEnv(cfg, device=device, render_mode=None)
    try:
        env.reset()
        reset_to_default_supported(env)
        robot = env.scene["robot"]
        step_dt = float(env.step_dt)
        steps = _step_count(duration_s, step_dt)
        direction_steps = _step_count(direction_duration_s, step_dt)
        actions = torch.zeros((env.num_envs, 6), device=env.device)
        # Both positive policy targets must move forward in the body frame.
        actions[1, (2, 5)] = wheel_action
        # Left positive, right negative is the documented negative-yaw turn.
        actions[2, 2] = wheel_action
        actions[2, 5] = -wheel_action
        # This independent world provides a non-zero wheel PI state before an
        # indexed environment reset.
        actions[3, (2, 5)] = wheel_action

        initial_pose = robot.data.root_link_pose_w.clone()
        neutral_initial_xy = initial_pose[0, :2].clone()
        neutral_max_tilt = torch.zeros((), device=env.device)
        neutral_max_speed = torch.zeros((), device=env.device)
        terminated_any = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        forward_displacement = torch.zeros((), device=env.device)
        forward_velocity = torch.zeros((), device=env.device)
        turn_angle = torch.zeros((), device=env.device)
        turn_rate = torch.zeros((), device=env.device)
        direction_terminated = torch.zeros(2, dtype=torch.bool, device=env.device)
        integral_before_reset: torch.Tensor | None = None
        integral_after_reset: torch.Tensor | None = None
        partial_reset_ok = False

        for step in range(steps):
            _, _, terminated, truncated, _ = env.step(actions)
            terminated_any |= terminated | truncated
            neutral_max_tilt = torch.maximum(neutral_max_tilt, tilt_radians(env)[0])
            neutral_max_speed = torch.maximum(
                neutral_max_speed,
                torch.linalg.vector_norm(robot.data.root_link_lin_vel_b[0, :2]),
            )

            if step == direction_steps - 1:
                direction_terminated.copy_(terminated_any[1:3])
                forward_delta = robot.data.root_link_pos_w[1, :2] - initial_pose[1, :2]
                forward_displacement = torch.dot(
                    forward_delta, _body_forward_xy(initial_pose[1:2, 3:7])[0]
                )
                forward_velocity = torch.dot(
                    robot.data.root_link_lin_vel_w[1, :2],
                    _body_forward_xy(robot.data.root_link_quat_w[1:2])[0],
                )
                turn_angle = _wrapped_difference(
                    _yaw(robot.data.root_link_quat_w[2:3])[0], _yaw(initial_pose[2:3, 3:7])[0]
                )
                turn_rate = robot.data.root_link_ang_vel_b[2, 2].clone()
                # Direction is an impulse-style sign check.  Do not keep
                # driving the independent worlds while the neutral 20-second
                # equilibrium check continues.
                actions[1:] = 0.0

            if step == 9:
                wheel_actuator = robot.actuators[1]
                integral_before_reset = wheel_actuator._integral.detach().clone()
                reset_id = torch.tensor([3], dtype=torch.long, device=env.device)
                env.reset(env_ids=reset_id)
                integral_after_reset = wheel_actuator._integral.detach().clone()
                partial_reset_ok = bool(
                    torch.allclose(integral_after_reset[3], torch.zeros_like(integral_after_reset[3]))
                    # A reset executes a synchronize/forward pass, which can
                    # make a sub-step-scale controller update in live worlds.
                    # Verify they remain materially intact rather than demand
                    # bitwise identity across that simulator boundary.
                    and torch.allclose(
                        integral_after_reset[:3],
                        integral_before_reset[:3],
                        rtol=2.0e-2,
                        atol=2.0e-4,
                    )
                )

            done_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:
                # The probe records any early termination as a failed check,
                # but must reset that slot before mjlab permits the other
                # independent worlds to continue their characterization.
                env.reset(env_ids=done_ids)
                reset_to_default_supported(env, done_ids)

        neutral_displacement = torch.linalg.vector_norm(
            robot.data.root_link_pos_w[0, :2] - neutral_initial_xy
        )
        neutral_ok = bool(
            not terminated_any[0].item()
            and neutral_max_tilt <= NEUTRAL_MAX_TILT_RAD
            and neutral_max_speed <= NEUTRAL_MAX_PLANAR_SPEED_M_S
            and neutral_displacement <= NEUTRAL_MAX_DISPLACEMENT_M
        )
        forward_ok = bool(
            not direction_terminated[0].item()
            and (
                forward_displacement >= MIN_FORWARD_DISPLACEMENT_M
                or forward_velocity >= MIN_FORWARD_DISPLACEMENT_M
            )
        )
        turn_ok = bool(
            not direction_terminated[1].item()
            and (
                OPPOSING_TARGET_YAW_SIGN * turn_angle >= MIN_TURN_ANGLE_RAD
                or OPPOSING_TARGET_YAW_SIGN * turn_rate >= MIN_TURN_ANGLE_RAD
            )
        )
        controller_checks = {
            "equal_positive_moves_forward": forward_ok,
            "opposing_targets_turn_negative_yaw": turn_ok,
            "partial_wheel_pi_reset": partial_reset_ok,
        }
        return {
            "schema_version": 1,
            "task": "Ascento-Balance-Flat",
            "device": str(env.device),
            "duration_s": duration_s,
            "step_dt_s": step_dt,
            "steps": steps,
            "wheel_action": wheel_action,
            "neutral": {
                "terminated": bool(terminated_any[0].item()),
                "max_tilt_rad": float(neutral_max_tilt.item()),
                "max_planar_speed_m_s": float(neutral_max_speed.item()),
                "net_displacement_m": float(neutral_displacement.item()),
                "held": neutral_ok,
                "interpretation": (
                    "A false result means the neutral target is not open-loop self-balancing; "
                    "it does not invalidate an active balancing policy."
                ),
            },
            "equal_positive": {
                "terminated_during_direction_probe": bool(direction_terminated[0].item()),
                "body_forward_displacement_m": float(forward_displacement.item()),
                "body_forward_velocity_m_s": float(forward_velocity.item()),
            },
            "opposing_targets": {
                "terminated_during_direction_probe": bool(direction_terminated[1].item()),
                "yaw_change_rad": float(turn_angle.item()),
                "yaw_rate_rad_s": float(turn_rate.item()),
                "convention": "left positive, right negative => negative base yaw (clockwise from above)",
            },
            "partial_wheel_pi_reset": {
                "integral_nonzero_before_reset": bool(
                    integral_before_reset is not None
                    and torch.any(torch.abs(integral_before_reset[3]) > 1.0e-7).item()
                ),
                "integral_before_reset": (
                    integral_before_reset.detach().cpu().tolist()
                    if integral_before_reset is not None
                    else None
                ),
                "integral_after_reset": (
                    integral_after_reset.detach().cpu().tolist()
                    if integral_after_reset is not None
                    else None
                ),
                "passed": partial_reset_ok,
            },
            "controller_checks": controller_checks,
            "passed": all(controller_checks.values()),
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="mjlab device, such as cpu or cuda:0")
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--direction-duration-s", type=float, default=0.10)
    parser.add_argument("--wheel-action", type=float, default=0.05)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = characterize(
        device=args.device,
        duration_s=args.duration_s,
        direction_duration_s=args.direction_duration_s,
        wheel_action=args.wheel_action,
    )
    print(json.dumps(payload, indent=2, sort_keys=True) if args.json else json.dumps(payload))
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
