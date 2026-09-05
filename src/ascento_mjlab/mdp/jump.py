"""Flat-ground jump state semantics and one-attempt phase tracking.

All contact-derived events are synchronized during reward computation from the
same sensor sample that is returned in the next observation. A single state
owner also carries the active jump attempt, heading-relative distance target,
phase, landing impact sample, and simultaneous two-wheel clearance.
"""

from dataclasses import dataclass

import torch

WHEEL_RADIUS_M = 0.25

PHASE_IDLE = 0
PHASE_CROUCH = 1
PHASE_THRUST = 2
PHASE_FLIGHT = 3
PHASE_LANDING = 4
PHASE_RECOVERY = 5

CROUCH_DURATION_S = 0.20
THRUST_TIMEOUT_S = 1.00
LANDING_HOLD_S = 0.10
RECOVERY_HOLD_S = 0.50


def initialize_jump_state(env, env_ids: torch.Tensor | None = None) -> None:
  """Initialize the single jump-state owner used by rewards, observations, and metrics."""
  ids = slice(None) if env_ids is None else env_ids
  if not hasattr(env, "ascento_jump_state"):
    env.ascento_jump_state = {
      "supported": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
      "airborne": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
      "takeoff": torch.zeros(env.num_envs, device=env.device),
      "landing": torch.zeros(env.num_envs, device=env.device),
      "air_time": torch.zeros(env.num_envs, device=env.device),
      "phase": torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
      "phase_time": torch.zeros(env.num_envs, device=env.device),
      "attempt_active": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
      "has_taken_off": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
      "last_jump_generation": torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
      "target_distance": torch.zeros(env.num_envs, device=env.device),
      "takeoff_xy": torch.zeros((env.num_envs, 2), device=env.device),
      "takeoff_forward_xy": torch.zeros((env.num_envs, 2), device=env.device),
      "remaining_distance": torch.zeros(env.num_envs, device=env.device),
      "landing_distance_error": torch.ones(env.num_envs, device=env.device),
      "limiting_wheel_clearance": torch.zeros(env.num_envs, device=env.device),
      "takeoff_height": torch.zeros(env.num_envs, device=env.device),
      "last_airborne_vz": torch.zeros(env.num_envs, device=env.device),
      "landing_preimpact_vz": torch.zeros(env.num_envs, device=env.device),
    }
  state = env.ascento_jump_state
  state["supported"][ids] = True
  state["airborne"][ids] = False
  state["takeoff"][ids] = 0.0
  state["landing"][ids] = 0.0
  state["air_time"][ids] = 0.0
  state["phase"][ids] = PHASE_IDLE
  state["phase_time"][ids] = 0.0
  state["attempt_active"][ids] = False
  state["has_taken_off"][ids] = False
  state["last_jump_generation"][ids] = 0
  state["target_distance"][ids] = 0.0
  state["takeoff_xy"][ids] = 0.0
  state["takeoff_forward_xy"][ids] = 0.0
  state["remaining_distance"][ids] = 0.0
  state["landing_distance_error"][ids] = 1.0
  state["limiting_wheel_clearance"][ids] = 0.0
  state["takeoff_height"][ids] = 0.0
  state["last_airborne_vz"][ids] = 0.0
  state["landing_preimpact_vz"][ids] = 0.0


def _wheel_contacts(env) -> tuple[torch.Tensor, torch.Tensor]:
  left = env.scene["left_wheel_contact"].data.found
  right = env.scene["right_wheel_contact"].data.found
  assert left is not None and right is not None
  return (
    left.flatten(start_dim=1).any(dim=1),
    right.flatten(start_dim=1).any(dim=1),
  )


def _motion_term(env):
  try:
    return env.command_manager.get_term("motion")
  except (AttributeError, KeyError):
    return None


def _forward_xy_from_quat(quat_wxyz: torch.Tensor) -> torch.Tensor:
  w, x, y, z = quat_wxyz.unbind(dim=1)
  forward = torch.stack(
    [1.0 - 2.0 * (y.square() + z.square()), 2.0 * (x * y + w * z)], dim=1
  )
  return forward / torch.linalg.vector_norm(forward, dim=1, keepdim=True).clamp(min=1.0e-8)


def _wheel_bottom_heights(env) -> torch.Tensor:
  robot = env.scene["robot"]
  try:
    left_id = robot.body_names.index("left_wheel")
    right_id = robot.body_names.index("right_wheel")
  except ValueError as exc:
    raise RuntimeError("Jump metrics require left_wheel and right_wheel bodies") from exc
  centres = robot.data.body_link_pos_w[:, [left_id, right_id], 2]
  return centres - WHEEL_RADIUS_M


def update_jump_state(
  env, env_ids: torch.Tensor | None = None, dt: float | None = None
) -> None:
  """Synchronize jump events and attempt state from the current policy transition."""
  if not hasattr(env, "ascento_jump_state"):
    initialize_jump_state(env)
  dt = env.step_dt if dt is None else dt
  left, right = _wheel_contacts(env)
  supported = left & right
  any_contact = left | right
  airborne = ~any_contact
  state = env.ascento_jump_state

  ids = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
  if env_ids is not None:
    ids.zero_()
    ids[env_ids] = True

  robot = env.scene["robot"]
  root_pos = robot.data.root_link_pos_w
  root_vz = robot.data.root_link_lin_vel_w[:, 2]
  root_quat = robot.data.root_link_quat_w

  motion_term = _motion_term(env)
  command = motion_term.command if motion_term is not None else None
  if motion_term is not None and hasattr(motion_term, "jump_generation"):
    generation = motion_term.jump_generation
    generation_changed = ids & (generation != state["last_jump_generation"])
    state["last_jump_generation"][generation_changed] = generation[generation_changed]
    new_request = generation_changed & ~state["attempt_active"]
  else:
    request = (
      command[:, 3] > 0.5
      if command is not None and command.shape[1] >= 6
      else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    )
    new_request = ids & request & ~state["attempt_active"]

  if bool(new_request.any().item()):
    state["attempt_active"][new_request] = True
    state["has_taken_off"][new_request] = False
    state["phase"][new_request] = PHASE_CROUCH
    state["phase_time"][new_request] = 0.0
    state["air_time"][new_request] = 0.0
    state["limiting_wheel_clearance"][new_request] = 0.0
    state["landing_preimpact_vz"][new_request] = 0.0
    state["landing_distance_error"][new_request] = 1.0
    if command is not None:
      state["target_distance"][new_request] = command[new_request, 5]
      state["remaining_distance"][new_request] = command[new_request, 5]

  ticking = ids & state["attempt_active"] & ~new_request
  state["phase_time"][ticking] += dt

  crouch_to_thrust = (
    ids
    & state["attempt_active"]
    & (state["phase"] == PHASE_CROUCH)
    & (state["phase_time"] >= CROUCH_DURATION_S)
  )
  state["phase"][crouch_to_thrust] = PHASE_THRUST
  state["phase_time"][crouch_to_thrust] = 0.0

  previous_supported = state["supported"].clone()
  previous_airborne = state["airborne"].clone()
  takeoff = ids & state["attempt_active"] & previous_supported & airborne
  landing = ids & state["attempt_active"] & previous_airborne & any_contact

  state["takeoff"][ids] = 0.0
  state["landing"][ids] = 0.0
  state["takeoff"][takeoff] = 1.0
  state["landing"][landing] = 1.0

  state["air_time"][ids] = torch.where(
    airborne[ids], state["air_time"][ids] + dt, state["air_time"][ids]
  )
  state["air_time"][takeoff] = dt

  if bool(takeoff.any().item()):
    state["has_taken_off"][takeoff] = True
    state["phase"][takeoff] = PHASE_FLIGHT
    state["phase_time"][takeoff] = 0.0
    state["takeoff_height"][takeoff] = root_pos[takeoff, 2]
    state["takeoff_xy"][takeoff] = root_pos[takeoff, :2]
    state["takeoff_forward_xy"][takeoff] = _forward_xy_from_quat(root_quat[takeoff])

  has_reference = ids & state["attempt_active"] & state["has_taken_off"]
  if bool(has_reference.any().item()):
    displacement = root_pos[:, :2] - state["takeoff_xy"]
    forward_distance = torch.sum(displacement * state["takeoff_forward_xy"], dim=1)
    state["remaining_distance"][has_reference] = (
      state["target_distance"][has_reference] - forward_distance[has_reference]
    )

  active_airborne = ids & state["attempt_active"] & airborne
  if bool(active_airborne.any().item()):
    limiting_now = _wheel_bottom_heights(env).amin(dim=1)
    state["limiting_wheel_clearance"][active_airborne] = torch.maximum(
      state["limiting_wheel_clearance"][active_airborne], limiting_now[active_airborne]
    )
    state["last_airborne_vz"][active_airborne] = root_vz[active_airborne]

  if bool(landing.any().item()):
    state["landing_preimpact_vz"][landing] = state["last_airborne_vz"][landing]
    state["landing_distance_error"][landing] = -state["remaining_distance"][landing]
    state["phase"][landing] = PHASE_LANDING
    state["phase_time"][landing] = 0.0

  landing_to_recovery = (
    ids
    & state["attempt_active"]
    & (state["phase"] == PHASE_LANDING)
    & (state["phase_time"] >= LANDING_HOLD_S)
  )
  state["phase"][landing_to_recovery] = PHASE_RECOVERY
  state["phase_time"][landing_to_recovery] = 0.0

  recovery_to_idle = (
    ids
    & state["attempt_active"]
    & (state["phase"] == PHASE_RECOVERY)
    & (state["phase_time"] >= RECOVERY_HOLD_S)
  )
  failed_attempt = (
    ids
    & state["attempt_active"]
    & ~state["has_taken_off"]
    & (state["phase"] == PHASE_THRUST)
    & (state["phase_time"] >= THRUST_TIMEOUT_S)
  )
  finished = recovery_to_idle | failed_attempt
  state["attempt_active"][finished] = False
  state["phase"][finished] = PHASE_IDLE
  state["phase_time"][finished] = 0.0

  state["supported"][ids] = supported[ids]
  state["airborne"][ids] = airborne[ids]


def sync_jump_state_reward(env) -> torch.Tensor:
  """Update jump state before all jump-dependent rewards, contributing zero reward."""
  update_jump_state(env)
  return torch.zeros(env.num_envs, device=env.device)


def phase_features(env) -> torch.Tensor:
  """Return contact/event state plus phase and target-relative distance."""
  if not hasattr(env, "ascento_jump_state"):
    initialize_jump_state(env)
  state = env.ascento_jump_state
  return torch.stack(
    [
      state["airborne"].float(),
      state["takeoff"],
      state["landing"],
      state["air_time"].clamp(max=2.0),
      state["phase"].float() / float(PHASE_RECOVERY),
      state["remaining_distance"].clamp(min=-1.0, max=1.0),
    ],
    dim=1,
  )


def takeoff_bonus(env) -> torch.Tensor:
  return env.ascento_jump_state["takeoff"]


def landing_bonus(env) -> torch.Tensor:
  return env.ascento_jump_state["landing"]


@dataclass(frozen=True)
class JumpSemantics:
  """Definitions shared by jump rewards, evaluation, and capture metrics."""

  takeoff_requires_both_wheels_airborne: bool = True
  landing_is_first_subsequent_wheel_contact: bool = True
  landing_impact_uses_precontact_vertical_speed: bool = True
  jump_distance_is_takeoff_heading_relative: bool = True
  clearance_uses_simultaneous_limiting_wheel: bool = True
  terrain_enabled: bool = False
