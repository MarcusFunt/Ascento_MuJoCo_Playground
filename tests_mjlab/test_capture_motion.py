from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ascento_mjlab.tools.capture_motion import (
    _configure_capture_cfg,
    _jump_state_array,
    _run_capture_steps,
)


def _jump_state(device: str) -> dict[str, torch.Tensor]:
    return {
        "airborne": torch.tensor([True], device=device),
        "takeoff": torch.tensor([False], device=device),
        "landing": torch.tensor([True], device=device),
        "air_time": torch.tensor([0.25], device=device),
    }


def test_jump_state_serialization_copies_tensors_to_cpu():
    result = _jump_state_array(_jump_state("cpu"))
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, np.asarray([1.0, 0.0, 1.0, 0.25], dtype=np.float32))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_jump_state_serialization_accepts_cuda_tensors():
    result = _jump_state_array(_jump_state("cuda:0"))
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, np.asarray([1.0, 0.0, 1.0, 0.25], dtype=np.float32))


def test_capture_configuration_disables_auto_reset():
    cfg = SimpleNamespace(
        scene=SimpleNamespace(num_envs=8),
        seed=None,
        auto_reset=True,
        recorders={},
    )
    configured = _configure_capture_cfg(cfg, take=3)
    assert configured is cfg
    assert configured.scene.num_envs == 1
    assert configured.seed == 3
    assert configured.auto_reset is False
    assert set(configured.recorders) == {"motion"}


def test_capture_rollout_stops_at_first_done_state():
    class FakeEnv:
        def __init__(self):
            self.steps = 0

        def reset(self):
            return torch.zeros((1, 1)), {}

        def step(self, action):
            del action
            self.steps += 1
            obs = torch.full((1, 1), float(self.steps))
            reward = torch.zeros(1)
            done = torch.tensor([self.steps >= 2], dtype=torch.long)
            return obs, reward, done, {}

    env = FakeEnv()
    captured_steps, ended_on_done = _run_capture_steps(
        env,
        lambda obs: torch.zeros((len(obs), 6)),
        steps=10,
    )
    assert captured_steps == 2
    assert ended_on_done is True
    assert env.steps == 2
