import torch

from ascento_mjlab.evaluation.runner import _disturbance_wrench
from ascento_mjlab.evaluation.schema import DisturbanceSpec, ScenarioSpec


def test_equivalent_delta_v_resolves_to_force_pulse():
  scenario = ScenarioSpec(
    scenario_id="s/f/0",
    family="f",
    task="Ascento-Balance-Flat",
    horizon_steps=100,
    reset={},
    disturbances=(
      DisturbanceSpec(
        start_step=10,
        duration_steps=10,
        direction="+x",
        equivalent_delta_v=0.5,
      ),
    ),
  )
  forces, torques, active = _disturbance_wrench(
    [scenario], 10, mass=20.0, step_dt=0.01, device="cpu"
  )
  # J=m*dv=10 N*s over 0.1 s => 100 N.
  assert torch.allclose(forces[0, 0], torch.tensor([100.0, 0.0, 0.0]))
  assert torch.count_nonzero(torques) == 0
  assert bool(active[0])

  forces, _, active = _disturbance_wrench(
    [scenario], 20, mass=20.0, step_dt=0.01, device="cpu"
  )
  assert torch.count_nonzero(forces) == 0
  assert not bool(active[0])
