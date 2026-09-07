from types import SimpleNamespace

import pytest
import torch

from ascento_mjlab.mdp.rewards import (
    airborne_height_progress,
    jump_distance_tracking,
    jump_landing,
    jump_landing_softness,
    jump_takeoff,
)


def _env(*, step_dt: float = 0.01):
    command = torch.tensor([[0.0, 0.0, 0.75, 1.0, 0.20, 0.0]])
    return SimpleNamespace(
        step_dt=step_dt,
        ascento_jump_state={
            "takeoff": torch.tensor([1.0]),
            "landing": torch.tensor([1.0]),
            "landing_distance_error": torch.tensor([0.0]),
            "landing_preimpact_vz": torch.tensor([0.0]),
            "takeoff_height": torch.tensor([0.80]),
            "airborne": torch.tensor([1.0]),
        },
        command_manager=SimpleNamespace(get_command=lambda _: command),
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(root_link_pos_w=torch.tensor([[0.0, 0.0, 0.85]]))
            )
        },
    )


def test_jump_events_are_impulses_under_dt_scaled_reward_aggregation():
    env = _env()

    assert jump_takeoff(env).item() * env.step_dt == pytest.approx(1.0)
    assert jump_landing(env).item() * env.step_dt == pytest.approx(1.0)
    assert jump_distance_tracking(env).item() * env.step_dt == pytest.approx(1.0)
    assert jump_landing_softness(env).item() * env.step_dt == pytest.approx(1.0)


@pytest.mark.parametrize("step_dt", [0.005, 0.01, 0.02])
def test_takeoff_impulse_is_independent_of_control_period(step_dt):
    env = _env(step_dt=step_dt)

    assert jump_takeoff(env).item() * step_dt == pytest.approx(1.0)


def test_airborne_height_progress_is_relative_to_actual_takeoff_height():
    env = _env()
    asset_cfg = SimpleNamespace(name="robot")

    assert airborne_height_progress(env, asset_cfg=asset_cfg).item() == pytest.approx(0.25)
