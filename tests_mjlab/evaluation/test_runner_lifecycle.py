from types import SimpleNamespace

import torch

from ascento_mjlab.evaluation.runner import (
  _refresh_exact_observation_history,
  _reset_finished_slots,
)


class _ObservationManager:
  def __init__(self):
    self.reset_ids = None
    self.compute_ids = None

  def reset(self, env_ids):
    self.reset_ids = env_ids.clone()

  def compute(self, *, update_history, env_ids):
    assert update_history is True
    self.compute_ids = env_ids.clone()
    return {"actor": torch.arange(len(env_ids), dtype=torch.float32)[:, None]}


def test_exact_reset_reseeds_observation_history_from_resolved_state():
  manager = _ObservationManager()
  env = SimpleNamespace(
    num_envs=3,
    device="cpu",
    observation_manager=manager,
    obs_buf=None,
  )

  _refresh_exact_observation_history(env)

  expected = torch.tensor([0, 1, 2])
  assert torch.equal(manager.reset_ids, expected)
  assert torch.equal(manager.compute_ids, expected)
  assert torch.equal(env.obs_buf["actor"].flatten(), torch.tensor([0.0, 1.0, 2.0]))


def test_finished_vector_slots_are_reset_before_next_step():
  reset_calls = []
  policy_calls = []
  env = SimpleNamespace(reset=lambda *, env_ids: reset_calls.append(env_ids.clone()))
  policy = SimpleNamespace(reset=lambda env_ids: policy_calls.append(env_ids.clone()))
  finish = torch.tensor([False, True, False, True])

  _reset_finished_slots(env, policy, finish)

  expected = torch.tensor([1, 3])
  assert len(reset_calls) == len(policy_calls) == 1
  assert torch.equal(reset_calls[0], expected)
  assert torch.equal(policy_calls[0], expected)


def test_no_finished_slots_do_not_trigger_reset():
  env = SimpleNamespace(reset=lambda **kwargs: (_ for _ in ()).throw(AssertionError(kwargs)))
  policy = SimpleNamespace(reset=lambda *args: (_ for _ in ()).throw(AssertionError(args)))

  _reset_finished_slots(env, policy, torch.zeros(4, dtype=torch.bool))
