"""Flat-ground jump state semantics.

Jump events are synchronized during reward computation from one contact sample
instead of through a later step event.  This keeps reward events and the next
observation on the same policy transition.
"""

from dataclasses import dataclass

import torch


def initialize_jump_state(env, env_ids: torch.Tensor | None = None) -> None:
  """Initialize the single jump-state owner used by rewards and metrics."""
  ids = slice(None) if env_ids is None else env_ids
  if not hasattr(env, "ascento_jump_state"):
    env.ascento_jump_state = {
      "supported": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
      "airborne": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
      "takeoff": torch.zeros(env.num_envs, device=env.device),
      "landing": torch.zeros(env.num_envs, device=env.device),
      "air_time": torch.zeros(env.num_envs, device=env.device),
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


def update_jump_state(
  env, env_ids: torch.Tensor | None = None, dt: float | None = None
) -> None:
  """Synchronize takeoff/landing state from the current wheel-contact sample.

  Takeoff requires both wheels to leave the ground. Landing is the first
  subsequent contact by either wheel.  ``landing_preimpact_vz`` is sampled from
  the final prior airborne policy sample, independently from any contact
  hysteresis or later two-wheel support.
  """
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

  previous_supported = state["supported"].clone()
  previous_airborne = state["airborne"].clone()
  takeoff = ids & previous_supported & airborne
  landing = ids & previous_airborne & any_contact

  state["takeoff"][ids] = 0.0
  state["landing"][ids] = 0.0
  state["takeoff"][takeoff] = 1.0
  state["landing"][landing] = 1.0

  state["air_time"][ids] = torch.where(
    airborne[ids], state["air_time"][ids] + dt, state["air_time"][ids]
  )
  state["air_time"][takeoff] = dt

  robot = env.scene["robot"]
  root_height = robot.data.root_link_pos_w[:, 2]
  root_vz = robot.data.root_link_lin_vel_w[:, 2]
  state["takeoff_height"][takeoff] = root_height[takeoff]
  state["landing_preimpact_vz"][landing] = state["last_airborne_vz"][landing]
  state["last_airborne_vz"][ids & airborne] = root_vz[ids & airborne]

  state["supported"][ids] = supported[ids]
  state["airborne"][ids] = airborne[ids]


def sync_jump_state_reward(env) -> torch.Tensor:
  """Update jump state before jump-dependent rewards, contributing zero reward."""
  update_jump_state(env)
  return torch.zeros(env.num_envs, device=env.device)


def phase_features(env) -> torch.Tensor:
  """Return [airborne, takeoff, landing, air_time] for observations."""
  if not hasattr(env, "ascento_jump_state"):
    initialize_jump_state(env)
  state = env.ascento_jump_state
  return torch.stack(
    [
      state["airborne"].float(),
      state["takeoff"],
      state["landing"],
      state["air_time"].clamp(max=2.0),
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
  terrain_enabled: bool = False
