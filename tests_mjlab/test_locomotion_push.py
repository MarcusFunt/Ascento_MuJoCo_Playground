from types import SimpleNamespace

import torch

from ascento_mjlab.mdp.events import _apply_cardinal_planar_push


class _Asset:
    def __init__(self, yaw: float):
        self.data = SimpleNamespace(
            root_link_vel_w=torch.zeros((1, 3)),
            root_link_quat_w=torch.tensor(
                [[torch.cos(torch.tensor(yaw / 2)), 0.0, 0.0, torch.sin(torch.tensor(yaw / 2))]]
            ),
        )
        self.written_velocity = None

    def write_root_link_velocity_to_sim(self, velocity, *, env_ids):
        assert env_ids.tolist() == [0]
        self.written_velocity = velocity.clone()


def test_fore_aft_push_is_relative_to_robot_heading():
    asset = _Asset(yaw=torch.pi / 2)
    env = SimpleNamespace(device=torch.device("cpu"))

    _apply_cardinal_planar_push(
        env,
        asset,
        torch.tensor([0]),
        min_delta_v=0.1,
        max_delta_v=0.1,
        fore_aft_probability=1.0,
    )

    delta = asset.written_velocity[0]
    assert abs(float(delta[0])) < 1e-6
    assert abs(abs(float(delta[1])) - 0.1) < 1e-6
    assert float(delta[2]) == 0.0
