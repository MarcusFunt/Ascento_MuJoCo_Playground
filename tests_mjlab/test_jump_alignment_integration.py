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
  env_id = torch.tensor([0], device=env.device)
  robot.write_root_link_pose_to_sim(pose, env_ids=env_id)
  robot.write_root_link_velocity_to_sim(
    torch.zeros((1, 6), device=env.device), env_ids=env_id
  )
  env.sim.forward()
  env.sim.sense()


def _find_static_pose_with_contact_count(env, *, roll: float, count: int) -> float:
  for z in torch.linspace(0.90, 0.45, 181).tolist():
    _set_root_pose(env, z=float(z), roll=roll)
    if sum(_contacts(env)) == count:
      return float(z)
  raise AssertionError(f"could not construct a static pose with {count} wheel contacts")


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
  # Eliminate stochastic command requests; this test injects one exact request.
  cfg.commands["motion"].jump_probability = 0.0
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    env.reset()

    # Establish a genuinely supported state through a real policy transition.
    # Geometric tangency alone need not set a contact sensor until physics runs.
    supported_z = _find_static_pose_with_contact_count(env, roll=0.0, count=2)
    _set_root_pose(env, z=supported_z, roll=0.0)
    supported_row = _transition(env)
    assert supported_row["contacts"] == (True, True)
    assert supported_row["airborne"] is False
    assert supported_row["takeoff"] == 0.0
    assert supported_row["landing"] == 0.0

    # Force exactly one new jump request without relying on the visible pulse
    # surviving command-manager update order.
    motion = env.command_manager.get_term("motion")
    motion._command[0] = torch.tensor([0.0, 0.0, 0.75, 1.0, 0.20, 0.20])
    motion._jump_pulse[0] = True
    motion._jump_generation[0] += 1

    # Lift the actual simulated robot clear of the plane, then take one real
    # ManagerBasedRlEnv policy transition. Contact observation, jump state and
    # reward pulse must describe this same transition.
    _set_root_pose(env, z=1.05, roll=0.0)
    takeoff_row = _transition(env)

    assert takeoff_row["contacts"] == (False, False)
    assert takeoff_row["airborne"] is True
    assert takeoff_row["takeoff"] == 1.0
    assert takeoff_row["landing"] == 0.0
    assert takeoff_row["rewards"]["takeoff"] > 0.0
    assert takeoff_row["rewards"]["landing"] == 0.0

    # Probe for a one-wheel first-contact pose without updating jump state;
    # only the subsequent env.step is allowed to create the landing event.
    one_wheel_z = _find_static_pose_with_contact_count(env, roll=0.30, count=1)
    _set_root_pose(env, z=one_wheel_z, roll=0.30)
    landing_row = _transition(env)

    assert sum(landing_row["contacts"]) == 1
    assert landing_row["airborne"] is False
    assert landing_row["takeoff"] == 0.0
    assert landing_row["landing"] == 1.0
    assert landing_row["rewards"]["landing"] > 0.0
  finally:
    env.close()
