"""Headless vectorized quantitative evaluation runner."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

import ascento_mjlab.tasks  # noqa: F401
from .policy import RslRlPolicyAdapter
from .schema import EpisodeResult, ScenarioSpec


DEFAULT_PHYSICAL_EFFORT_LIMIT = 40.0


def task_capabilities(task: str) -> set[str]:
  capabilities = {
    "generic_body_metrics",
    "generic_control_metrics",
    "exact_reset",
    "force_disturbance",
    "deterministic_policy",
  }
  if "Balance" in task:
    capabilities.add("balance")
  if "Velocity" in task:
    capabilities.update(
      {
        "velocity",
        "command:twist",
        "command:height",
        "velocity_tracking",
        "height_tracking",
      }
    )
  if "Recovery" in task:
    capabilities.update({"recovery", "recovery_success"})
  if "Jump" in task:
    capabilities.update({"jump_events", "command:motion"})
  return capabilities


def checkpoint_sha256(path: str | Path) -> str:
  digest = hashlib.sha256()
  with Path(path).open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def task_step_dt(task: str) -> float:
  cfg = load_env_cfg(task, play=False)
  return float(cfg.sim.timestep) * int(cfg.decimation)


def _exact_reset(base_env: ManagerBasedRlEnv, scenarios: list[ScenarioSpec]) -> torch.Tensor:
  """Overwrite stochastic training resets with resolved scenario states."""
  robot = base_env.scene["robot"]
  device = base_env.device
  count = len(scenarios)
  env_ids = torch.arange(count, device=device, dtype=torch.long)

  default_root = robot.data.default_root_state[env_ids].clone()
  default_root[:, 0:3] += base_env.scene.env_origins[env_ids]

  offsets = torch.tensor(
    [
      [
        scenario.reset.get("x", 0.0),
        scenario.reset.get("y", 0.0),
        scenario.reset.get("z", 0.0),
      ]
      for scenario in scenarios
    ],
    dtype=torch.float32,
    device=device,
  )
  euler = torch.tensor(
    [
      [
        scenario.reset.get("roll", 0.0),
        scenario.reset.get("pitch", 0.0),
        scenario.reset.get("yaw", 0.0),
      ]
      for scenario in scenarios
    ],
    dtype=torch.float32,
    device=device,
  )
  velocity = torch.tensor(
    [
      [
        scenario.reset.get("vx", 0.0),
        scenario.reset.get("vy", 0.0),
        scenario.reset.get("vz", 0.0),
        scenario.reset.get("wx", 0.0),
        scenario.reset.get("wy", 0.0),
        scenario.reset.get("wz", 0.0),
      ]
      for scenario in scenarios
    ],
    dtype=torch.float32,
    device=device,
  )

  positions = default_root[:, 0:3] + offsets
  delta_quat = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
  orientation = quat_mul(default_root[:, 3:7], delta_quat)
  root_velocity = default_root[:, 7:13] + velocity

  robot.write_root_link_pose_to_sim(
    torch.cat([positions, orientation], dim=-1), env_ids=env_ids
  )
  robot.write_root_link_velocity_to_sim(root_velocity, env_ids=env_ids)
  if robot.is_articulated:
    robot.write_joint_state_to_sim(
      robot.data.default_joint_pos[env_ids].clone(),
      robot.data.default_joint_vel[env_ids].clone(),
      env_ids=env_ids,
    )

  base_env.sim.forward()
  base_env.sim.sense()
  return positions[:, :2].clone()


def _refresh_exact_observation_history(base_env: ManagerBasedRlEnv) -> None:
  """Seed stateful observation buffers from the exact scenario state only.

  ``env.reset()`` must run first to reset managers, but it also writes a
  stochastic training reset into history/delay buffers. After `_exact_reset`
  overwrites that state, clear those buffers and seed them with the resolved
  deterministic frame so history-dependent policies do not see a hidden random
  pre-scenario observation.
  """
  env_ids = torch.arange(base_env.num_envs, device=base_env.device, dtype=torch.long)
  base_env.observation_manager.reset(env_ids)
  base_env.obs_buf = base_env.observation_manager.compute(
    update_history=True, env_ids=env_ids
  )


def _reset_finished_slots(base_env: ManagerBasedRlEnv, policy, finish: torch.Tensor) -> None:
  """Clear mjlab manual-reset state for completed vector slots.

  mjlab 1.6 requires every done slot to be explicitly reset before the next
  vector ``step()`` when ``auto_reset=False``. Horizon-complete slots are reset
  too so all inactive slots have the same benign lifecycle.
  """
  if not bool(finish.any().item()):
    return
  env_ids = finish.nonzero(as_tuple=False).squeeze(-1)
  base_env.reset(env_ids=env_ids)
  policy.reset(env_ids)


def _command_values_for_step(
  scenarios: list[ScenarioSpec], step: int
) -> dict[str, list[tuple[int, tuple[float, ...]]]]:
  updates: dict[str, list[tuple[int, tuple[float, ...]]]] = {}
  for env_id, scenario in enumerate(scenarios):
    latest: dict[str, tuple[float, ...]] = {}
    for point in scenario.commands:
      if point.step > step:
        break
      latest[point.name] = point.values
    for name, values in latest.items():
      updates.setdefault(name, []).append((env_id, values))
  return updates


def _apply_commands(base_env: ManagerBasedRlEnv, scenarios: list[ScenarioSpec], step: int) -> None:
  for name, updates in _command_values_for_step(scenarios, step).items():
    try:
      term = base_env.command_manager.get_term(name)
    except (KeyError, AttributeError) as exc:
      raise RuntimeError(f"Scenario requires unavailable command term {name!r}") from exc
    command = term.command
    for env_id, values in updates:
      if len(values) != command.shape[1]:
        raise RuntimeError(
          f"Command {name!r} expects {command.shape[1]} values, got {len(values)}"
        )
      command[env_id] = torch.tensor(values, device=command.device, dtype=command.dtype)


def _command_target(
  scenarios: list[ScenarioSpec],
  step: int,
  *,
  name: str,
  dim: int,
  device: str,
) -> torch.Tensor:
  target = torch.zeros((len(scenarios), dim), dtype=torch.float32, device=device)
  updates = _command_values_for_step(scenarios, step).get(name, [])
  for env_id, values in updates:
    if len(values) != dim:
      raise RuntimeError(f"Command {name!r} expects {dim} values, got {len(values)}")
    target[env_id] = torch.tensor(values, dtype=torch.float32, device=device)
  return target


def _robot_total_mass(base_env: ManagerBasedRlEnv) -> float:
  robot = base_env.scene["robot"]
  body_ids = robot.data.indexing.body_ids.detach().cpu().numpy()
  return float(np.asarray(base_env.sim.mj_model.body_mass)[body_ids].sum())


def _disturbance_wrench(
  scenarios: list[ScenarioSpec],
  step: int,
  *,
  mass: float,
  step_dt: float,
  device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  count = len(scenarios)
  forces = torch.zeros((count, 1, 3), device=device)
  torques = torch.zeros((count, 1, 3), device=device)
  active = torch.zeros(count, dtype=torch.bool, device=device)
  direction_vectors = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
  }
  for env_id, scenario in enumerate(scenarios):
    for disturbance in scenario.disturbances:
      if not (
        disturbance.start_step
        <= step
        < disturbance.start_step + disturbance.duration_steps
      ):
        continue
      duration_s = disturbance.duration_steps * step_dt
      magnitude = (
        disturbance.force_n
        if disturbance.force_n is not None
        else mass * disturbance.equivalent_delta_v / max(duration_s, step_dt)
      )
      if disturbance.direction not in direction_vectors:
        raise ValueError(f"Unsupported disturbance direction: {disturbance.direction}")
      vector = direction_vectors[disturbance.direction]
      forces[env_id, 0] = torch.tensor(vector, device=device) * float(magnitude)
      torques[env_id, 0] = torch.tensor(disturbance.torque_nm, device=device)
      active[env_id] = True
  return forces, torques, active


def _termination_reasons(base_env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> dict[int, str]:
  manager = base_env.termination_manager
  reasons: dict[int, str] = {}
  for env_id in env_ids.detach().cpu().tolist():
    names = [
      name
      for name in manager.active_terms
      if bool(manager.get_term(name)[env_id].item())
    ]
    non_timeout = [
      name for name in names if not bool(manager.get_term_cfg(name).time_out)
    ]
    reasons[env_id] = non_timeout[0] if non_timeout else (names[0] if names else "done")
  return reasons


def _run_batch(
  task: str,
  checkpoint: Path,
  scenarios: list[ScenarioSpec],
  *,
  device: str,
  deterministic: bool,
  physical_effort_limit: float,
) -> tuple[list[EpisodeResult], dict[str, Any]]:
  if not scenarios:
    return [], {}
  cfg = load_env_cfg(task, play=False)
  cfg.scene.num_envs = len(scenarios)
  cfg.auto_reset = False
  cfg.seed = 0
  step_dt = float(cfg.sim.timestep) * int(cfg.decimation)
  max_horizon = max(scenario.horizon_steps for scenario in scenarios)
  cfg.episode_length_s = (max_horizon + 2) * step_dt

  base_env = ManagerBasedRlEnv(cfg, device=device, render_mode=None)
  agent_cfg = load_rl_cfg(task)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  policy = RslRlPolicyAdapter(runner, checkpoint, deterministic=deterministic)

  env.reset()
  initial_xy = _exact_reset(base_env, scenarios)
  _refresh_exact_observation_history(base_env)
  policy.reset()
  count = len(scenarios)
  dev = base_env.device
  is_velocity_task = "Velocity" in task
  is_recovery_task = "Recovery" in task
  active = torch.ones(count, dtype=torch.bool, device=dev)
  finished_success = torch.zeros(count, dtype=torch.bool, device=dev)
  episode_steps = torch.zeros(count, dtype=torch.long, device=dev)
  horizon = torch.tensor(
    [scenario.horizon_steps for scenario in scenarios],
    dtype=torch.long,
    device=dev,
  )
  termination_reason = [""] * count

  sum_tilt = torch.zeros(count, device=dev)
  sum_tilt_sq = torch.zeros(count, device=dev)
  max_tilt = torch.zeros(count, device=dev)
  sum_planar_speed_sq = torch.zeros(count, device=dev)
  max_planar_speed = torch.zeros(count, device=dev)
  sum_height = torch.zeros(count, device=dev)
  min_height = torch.full((count,), float("inf"), device=dev)
  sum_effort_abs = torch.zeros(count, device=dev)
  sum_effort_sq = torch.zeros(count, device=dev)
  max_effort_abs = torch.zeros(count, device=dev)
  sum_request_sq = torch.zeros(count, device=dev)
  max_request_abs = torch.zeros(count, device=dev)
  action_clip_count = torch.zeros(count, device=dev)
  action_count = torch.zeros(count, device=dev)
  saturation_count = torch.zeros(count, device=dev)
  request_count = torch.zeros(count, device=dev)
  support_count = torch.zeros(count, device=dev)
  airborne_count = torch.zeros(count, device=dev)
  path_length = torch.zeros(count, device=dev)
  sum_velocity_tracking_sq = torch.zeros(count, device=dev)
  sum_height_tracking_sq = torch.zeros(count, device=dev)
  recovery_stable_count = torch.zeros(count, dtype=torch.long, device=dev)
  recovery_success = torch.zeros(count, dtype=torch.bool, device=dev)
  recovery_from_start_s = torch.full((count,), float("nan"), device=dev)
  prev_xy = initial_xy.clone()
  final_xy_snapshot = initial_xy.clone()

  stable_count = torch.zeros(count, dtype=torch.long, device=dev)
  recovered = torch.zeros(count, dtype=torch.bool, device=dev)
  recovery_time = torch.full((count,), float("nan"), device=dev)
  stable_steps_required = max(1, round(0.50 / step_dt))
  disturbance_end = torch.tensor(
    [
      max(
        (d.start_step + d.duration_steps for d in scenario.disturbances),
        default=-1,
      )
      for scenario in scenarios
    ],
    dtype=torch.long,
    device=dev,
  )

  mass = _robot_total_mass(base_env)
  all_env_ids = torch.arange(count, device=dev, dtype=torch.long)
  last_wrench_active = torch.zeros(count, dtype=torch.bool, device=dev)

  try:
    for step in range(max_horizon):
      if not bool(active.any().item()):
        break

      _apply_commands(base_env, scenarios, step)
      twist_target = (
        _command_target(scenarios, step, name="twist", dim=3, device=dev)
        if is_velocity_task
        else None
      )
      height_target = (
        _command_target(scenarios, step, name="height", dim=1, device=dev)
        if is_velocity_task
        else None
      )
      forces, torques, wrench_active = _disturbance_wrench(
        scenarios, step, mass=mass, step_dt=step_dt, device=dev
      )
      if bool((wrench_active | last_wrench_active).any().item()):
        base_env.scene["robot"].write_external_wrench_to_sim(
          forces,
          torques,
          env_ids=all_env_ids,
          body_ids=[0],
        )
      last_wrench_active = wrench_active

      observations = env.get_observations()
      raw_action = policy.act(observations)
      if not torch.isfinite(raw_action).all():
        raise RuntimeError("Policy emitted NaN/Inf action during evaluation")

      clip_limit = float(agent_cfg.clip_actions)
      clipped_fraction = (torch.abs(raw_action) > clip_limit).float().mean(dim=-1)
      action_clip_count += clipped_fraction * active.float()
      action_count += active.float()

      action = torch.where(active[:, None], raw_action, torch.zeros_like(raw_action))
      _, _, dones, _ = env.step(action)

      robot = base_env.scene["robot"]
      gravity_xy = torch.linalg.vector_norm(robot.data.projected_gravity_b[:, :2], dim=1)
      tilt = torch.atan2(
        gravity_xy,
        -robot.data.projected_gravity_b[:, 2].clamp(max=-1.0e-6),
      )
      planar_speed = torch.linalg.vector_norm(robot.data.root_link_lin_vel_w[:, :2], dim=1)
      angular_xy = torch.linalg.vector_norm(robot.data.root_link_ang_vel_b[:, :2], dim=1)
      linear_speed = torch.linalg.vector_norm(robot.data.root_link_lin_vel_b, dim=1)
      angular_speed = torch.linalg.vector_norm(robot.data.root_link_ang_vel_b, dim=1)
      height = robot.data.root_link_pos_w[:, 2]
      effort = robot.data.actuator_force
      effort_abs = torch.mean(torch.abs(effort), dim=1)
      effort_sq = torch.mean(torch.square(effort), dim=1)
      effort_peak = torch.amax(torch.abs(effort), dim=1)
      request = robot.data.joint_effort_target
      request_sq = torch.mean(torch.square(request), dim=1)
      request_peak = torch.amax(torch.abs(request), dim=1)
      request_sat = torch.mean(
        (torch.abs(request) >= physical_effort_limit - 1.0e-4).float(), dim=1
      )
      xy = robot.data.root_link_pos_w[:, :2]
      segment = torch.linalg.vector_norm(xy - prev_xy, dim=1)
      prev_xy = xy.clone()

      left_data = base_env.scene["left_wheel_contact"].data.found
      right_data = base_env.scene["right_wheel_contact"].data.found
      assert left_data is not None and right_data is not None
      left = left_data.flatten(start_dim=1).any(dim=1)
      right = right_data.flatten(start_dim=1).any(dim=1)
      both_supported = left & right
      airborne = ~left & ~right

      weight = active.float()
      sum_tilt += tilt * weight
      sum_tilt_sq += tilt.square() * weight
      max_tilt = torch.maximum(max_tilt, torch.where(active, tilt, torch.zeros_like(tilt)))
      sum_planar_speed_sq += planar_speed.square() * weight
      max_planar_speed = torch.maximum(
        max_planar_speed, torch.where(active, planar_speed, torch.zeros_like(planar_speed))
      )
      sum_height += height * weight
      min_height = torch.minimum(min_height, torch.where(active, height, min_height))
      sum_effort_abs += effort_abs * weight
      sum_effort_sq += effort_sq * weight
      max_effort_abs = torch.maximum(
        max_effort_abs, torch.where(active, effort_peak, torch.zeros_like(effort_peak))
      )
      sum_request_sq += request_sq * weight
      max_request_abs = torch.maximum(
        max_request_abs, torch.where(active, request_peak, torch.zeros_like(request_peak))
      )
      saturation_count += request_sat * weight
      request_count += weight
      support_count += both_supported.float() * weight
      airborne_count += airborne.float() * weight
      path_length += segment * weight
      if twist_target is not None:
        actual_twist = torch.stack(
          [
            robot.data.root_link_lin_vel_b[:, 0],
            robot.data.root_link_lin_vel_b[:, 1],
            robot.data.root_link_ang_vel_b[:, 2],
          ],
          dim=1,
        )
        sum_velocity_tracking_sq += (
          torch.mean(torch.square(actual_twist - twist_target), dim=1) * weight
        )
      if height_target is not None:
        sum_height_tracking_sq += torch.square(height - height_target[:, 0]) * weight
      episode_steps += active.long()

      if is_recovery_task:
        from ascento_mjlab.mdp.recovery import RecoveryEnvelope, recovery_condition

        envelope = RecoveryEnvelope()
        recovery_stable = recovery_condition(base_env, envelope)
        recovery_stable_count = torch.where(
          active & recovery_stable,
          recovery_stable_count + 1,
          torch.where(active, torch.zeros_like(recovery_stable_count), recovery_stable_count),
        )
        required = max(1, round(envelope.stable_duration_s / step_dt))
        newly_recovered_from_start = (
          active & ~recovery_success & (recovery_stable_count >= required)
        )
        if bool(newly_recovered_from_start.any().item()):
          first_stable_step = step - required + 1
          recovery_from_start_s[newly_recovered_from_start] = first_stable_step * step_dt
          recovery_success |= newly_recovered_from_start

      after_disturbance = active & (disturbance_end >= 0) & (step >= disturbance_end)
      stable_now = (
        (tilt <= 0.08)
        & (planar_speed <= 0.10)
        & (torch.abs(height - 0.75) <= 0.05)
        & (angular_xy <= 0.25)
        & both_supported
      )
      stable_count = torch.where(
        after_disturbance & stable_now,
        stable_count + 1,
        torch.where(after_disturbance, torch.zeros_like(stable_count), stable_count),
      )
      newly_recovered = after_disturbance & ~recovered & (
        stable_count >= stable_steps_required
      )
      if bool(newly_recovered.any().item()):
        first_stable_step = step - stable_steps_required + 1
        recovery_time[newly_recovered] = (
          first_stable_step - disturbance_end[newly_recovered]
        ) * step_dt
        recovered |= newly_recovered

      reached_horizon = active & (episode_steps >= horizon)
      active_done = active & dones.bool()
      non_timeout_done = torch.zeros_like(active_done)
      for name in base_env.termination_manager.active_terms:
        if not base_env.termination_manager.get_term_cfg(name).time_out:
          non_timeout_done |= active & base_env.termination_manager.get_term(name)

      failed = active_done & non_timeout_done
      succeeded = reached_horizon & ~failed
      finish = failed | succeeded

      if bool(finish.any().item()):
        final_xy_snapshot[finish] = xy[finish]
        failed_ids = failed.nonzero(as_tuple=False).squeeze(-1)
        if len(failed_ids) > 0:
          for env_id, reason in _termination_reasons(base_env, failed_ids).items():
            termination_reason[env_id] = reason
        success_ids = succeeded.nonzero(as_tuple=False).squeeze(-1)
        for env_id in success_ids.detach().cpu().tolist():
          if not termination_reason[env_id]:
            timeout_names = [
              name
              for name in base_env.termination_manager.active_terms
              if bool(base_env.termination_manager.get_term_cfg(name).time_out)
              and bool(base_env.termination_manager.get_term(name)[env_id].item())
            ]
            termination_reason[env_id] = timeout_names[0] if timeout_names else "horizon"
        finished_success |= succeeded
        active &= ~finish
        _reset_finished_slots(base_env, policy, finish)

    if bool(active.any().item()):
      raise RuntimeError("Evaluation loop ended with unfinished scenarios")

    denom = episode_steps.clamp(min=1).float()
    net_displacement = torch.linalg.vector_norm(final_xy_snapshot - initial_xy, dim=1)

    arrays = {
      "success": finished_success,
      "episode_steps": episode_steps,
      "tilt_mean": sum_tilt / denom,
      "tilt_rms": torch.sqrt(sum_tilt_sq / denom),
      "max_tilt": max_tilt,
      "planar_speed_rms": torch.sqrt(sum_planar_speed_sq / denom),
      "max_planar_speed": max_planar_speed,
      "height_mean": sum_height / denom,
      "height_min": min_height,
      "effort_mean_abs": sum_effort_abs / denom,
      "effort_rms": torch.sqrt(sum_effort_sq / denom),
      "effort_max_abs": max_effort_abs,
      "physical_request_rms": torch.sqrt(sum_request_sq / denom),
      "physical_request_max_abs": max_request_abs,
      "action_clip_fraction": action_clip_count / action_count.clamp(min=1.0),
      "physical_saturation_fraction": saturation_count / request_count.clamp(min=1.0),
      "both_supported_fraction": support_count / denom,
      "airborne_fraction": airborne_count / denom,
      "path_length": path_length,
      "net_displacement": net_displacement,
      "recovered": recovered.float(),
      "recovery_time_s": recovery_time,
    }
    if is_velocity_task:
      arrays["velocity_tracking_rmse"] = torch.sqrt(sum_velocity_tracking_sq / denom)
      arrays["height_tracking_rmse"] = torch.sqrt(sum_height_tracking_sq / denom)
    if is_recovery_task:
      arrays["recovery_success"] = recovery_success.float()
      arrays["recovery_from_start_s"] = recovery_from_start_s
    cpu = {key: value.detach().cpu().numpy() for key, value in arrays.items()}

    results: list[EpisodeResult] = []
    for index, scenario in enumerate(scenarios):
      metrics = {
        key: float(values[index])
        for key, values in cpu.items()
        if key not in {"success", "episode_steps"}
      }
      metrics["episode_time_s"] = int(cpu["episode_steps"][index]) * step_dt
      metrics["nonfinite"] = float(termination_reason[index] == "nonfinite")
      if scenario.disturbances:
        disturbance = scenario.disturbances[0]
        duration_s = disturbance.duration_steps * step_dt
        force_n = (
          disturbance.force_n
          if disturbance.force_n is not None
          else mass * disturbance.equivalent_delta_v / max(duration_s, step_dt)
        )
        metrics["disturbance_force_n"] = float(force_n)
        metrics["disturbance_equivalent_delta_v"] = float(
          disturbance.equivalent_delta_v
        )
      results.append(
        EpisodeResult(
          scenario_id=scenario.scenario_id,
          family=scenario.family,
          success=bool(cpu["success"][index]),
          termination_reason=termination_reason[index] or "unknown",
          episode_steps=int(cpu["episode_steps"][index]),
          metrics=metrics,
        )
      )

    metadata = {
      "step_dt": step_dt,
      "num_envs": count,
      "robot_total_mass_kg": mass,
      "policy": asdict(policy.metadata()),
    }
    return results, metadata
  finally:
    env.close()


def run_scenarios(
  task: str,
  checkpoint: str | Path,
  scenarios: list[ScenarioSpec],
  *,
  batch_size: int = 512,
  device: str = "cuda:0",
  deterministic: bool = True,
  physical_effort_limit: float = DEFAULT_PHYSICAL_EFFORT_LIMIT,
) -> tuple[list[EpisodeResult], dict[str, Any]]:
  checkpoint = Path(checkpoint)
  if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
  if batch_size < 1:
    raise ValueError("batch_size must be positive")

  results: list[EpisodeResult] = []
  batch_metadata: list[dict[str, Any]] = []
  family_order = list(dict.fromkeys(scenario.family for scenario in scenarios))
  for family in family_order:
    family_scenarios = [scenario for scenario in scenarios if scenario.family == family]
    for start in range(0, len(family_scenarios), batch_size):
      batch = family_scenarios[start : start + batch_size]
      batch_results, metadata = _run_batch(
        task,
        checkpoint,
        batch,
        device=device,
        deterministic=deterministic,
        physical_effort_limit=physical_effort_limit,
      )
      results.extend(batch_results)
      batch_metadata.append({"family": family, **metadata})

  step_dts = {float(item["step_dt"]) for item in batch_metadata if item}
  if len(step_dts) > 1:
    raise RuntimeError(f"Inconsistent step_dt across batches: {sorted(step_dts)}")
  return results, {
    "checkpoint_sha256": checkpoint_sha256(checkpoint),
    "batches": len(batch_metadata),
    "step_dt": next(iter(step_dts)) if step_dts else task_step_dt(task),
    "batch_metadata": batch_metadata,
  }
