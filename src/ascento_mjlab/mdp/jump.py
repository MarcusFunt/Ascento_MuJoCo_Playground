"""Flat-ground jump state semantics.

The initial port keeps jump state derived from contact and root motion.  A
persistent state owner will be added only if observations/rewards demonstrate
that derived signals are insufficient; no parallel FSM is introduced here.
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
    }
  state = env.ascento_jump_state
  state["supported"][ids] = True
  state["airborne"][ids] = False
  state["takeoff"][ids] = 0.0
  state["landing"][ids] = 0.0
  state["air_time"][ids] = 0.0
  state["takeoff_height"][ids] = 0.0


def update_jump_state(env, env_ids: torch.Tensor | None, dt: float) -> None:
  """Detect both-wheel takeoff and first subsequent contact."""
  del env_ids
  left = env.scene["left_wheel_contact"].data.found > 0
  right = env.scene["right_wheel_contact"].data.found > 0
  left = left.flatten(start_dim=1).any(dim=1)
  right = right.flatten(start_dim=1).any(dim=1)
  supported = left & right
  airborne = ~left & ~right
  state = env.ascento_jump_state
  state["takeoff"].zero_()
  state["landing"].zero_()
  takeoff = state["supported"] & airborne
  landing = state["airborne"] & supported
  state["takeoff"][takeoff] = 1.0
  state["landing"][landing] = 1.0
  state["air_time"] = torch.where(airborne, state["air_time"] + dt, state["air_time"])
  state["air_time"][takeoff] = dt
  root_height = env.scene["robot"].data.root_link_pos_w[:, 2]
  state["takeoff_height"][takeoff] = root_height[takeoff]
  state["supported"] = supported
  state["airborne"] = airborne


def phase_features(env) -> torch.Tensor:
  """Return [airborne, takeoff, landing, air_time] for observations."""
  if not hasattr(env, "ascento_jump_state"):
    initialize_jump_state(env)
  state = env.ascento_jump_state
  return torch.stack(
    [state["airborne"].float(), state["takeoff"], state["landing"], state["air_time"].clamp(max=2.0)], dim=1
  )


def takeoff_bonus(env) -> torch.Tensor:
  return env.ascento_jump_state["takeoff"]


def landing_bonus(env) -> torch.Tensor:
  return env.ascento_jump_state["landing"]


@dataclass(frozen=True)
class JumpSemantics:
  """Definitions shared by future jump rewards and capture metrics."""

  takeoff_requires_both_wheels_airborne: bool = True
  landing_is_first_subsequent_wheel_contact: bool = True
  terrain_enabled: bool = False
