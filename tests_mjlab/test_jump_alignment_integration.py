import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz

import ascento_mjlab.tasks  # noqa: F401


def _contacts(env):
  values = []
  for name in ("left_wheel_contact", "right_wheel_contact"):
    found = env.scene[name].data.found
    assert found is not None
    values.append(bool(found[0].flatten().any().item()))
  return tuple(values)


def _reward_terms(env):
  return {
    name: float(values[0])
    for name, values in env.reward_manager.get_active_iterable_terms(0)
  }


def _set_root_pose(env, *, z: float, roll: float = 0.0) -> None:
  robot = env.scene["robot"]
  pose = robot.data.root_link_pose_w[:1].clone()
  pose[0, 2] = z
  zero = torch.zeros(1, device=env.device)
  pose[0, 3:7] = quat_from_euler_xyz(
    torch.tensor([roll], device=env.device), zero, zero
  )[0]
  robot.write_root_link_pose_to_sim(pose, env_ids=torch.tensor([0], device=env.device))
  robot.write_root_link_velocity_to_sim(
    torch.zeros((1, 6), device=env.device), env_ids=torch.tensor([0], device=env.device)
  )
  env.sim.forward()
  env.sim.sense()


def _transition(env):
  obs, reward, terminated, truncated, _ = env.step(torch.zeros((1, 6), device=env.device))
  state = env.ascento_jump_state
  row = {
    "contacts": _contacts(env),
    "airborne": bool(state["airborne"][0].item()),
    "takeoff": float(state["takeoff"][0].item()),
    "landing": float(state["landing"][0].item()),
    "rewards": _reward_terms(env),
    "terminated": bool(terminated[0].item()),
    "truncated": bool(truncated[0].item()),
  }
  assert torch.isfinite(obs["actor"]).all()
  assert torch.isfinite(reward).all()
  return row


def test_real_jump_rollout_aligns_contacts_state_and_reward_pulses():
  cfg = load_env_cfg("Ascento-Jump-Flat", play=True)
  cfg.scene.num_envs = 1
  cfg.auto_reset = False
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    env.reset()
    assert _contacts(env) == (True, True)

    # Force exactly one new jump request without relying on the visible pulse
    # surviving command-manager update order.
    motion = env.command_manager.get_term("motion")
    motion._command[0] = torch.tensor([0.0, 0.0, 0.75, 1.0, 0.20, 0.20])
    motion._jump_pulse[0] = True
    motion._jump_generation[0] += 1

    # Lift the actual simulated robot clear of the plane, then take one real
    # ManagerBasedRlEnv policy transition.
    _set_root_pose(env, z=1.05, roll=0.0)
    takeoff_row = _transition(env)

    assert takeoff_row["contacts"] == (False, False)
    assert takeoff_row["airborne"] is True
    assert takeoff_row["takeoff"] == 1.0
    assert takeoff_row["landing"] == 0.0
    assert takeoff_row["rewards"]["takeoff"] > 0.0
    assert takeoff_row["rewards"]["landing"] == 0.0

    # Find a static tilted pose with exactly one wheel contacting the plane.
    # Sensor probing does not update the jump state; the subsequent env.step is
    # therefore the first policy transition after the airborne sample.
    candidate = None
    for z in torch.linspace(0.80, 0.50, 61).tolist():
      _set_root_pose(env, z=float(z), roll=0.30)
      if sum(_contacts(env)) == 1:
        candidate = float(z)
        break
    assert candidate is not None, "could not construct a one-wheel first-contact pose"

    _set_root_pose(env, z=candidate, roll=0.30)
    landing_row = _transition(env)

    assert sum(landing_row["contacts"]) == 1
    assert landing_row["airborne"] is False
    assert landing_row["takeoff"] == 0.0
    assert landing_row["landing"] == 1.0
    assert landing_row["rewards"]["landing"] > 0.0
  finally:
    env.close()
