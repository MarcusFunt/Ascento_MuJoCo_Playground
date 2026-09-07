from types import SimpleNamespace

import pytest
import torch

from ascento_mjlab.mdp.terminations import nonfinite


def _env():
    data = SimpleNamespace(
        joint_pos=torch.zeros((1, 6)),
        joint_vel=torch.zeros((1, 6)),
        root_link_pos_w=torch.zeros((1, 3)),
        root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        root_link_lin_vel_w=torch.zeros((1, 3)),
        root_link_ang_vel_w=torch.zeros((1, 3)),
        root_link_lin_vel_b=torch.zeros((1, 3)),
        root_link_ang_vel_b=torch.zeros((1, 3)),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
        actuator_force=torch.zeros((1, 6)),
    )
    return SimpleNamespace(scene={"robot": SimpleNamespace(data=data)}), data


@pytest.mark.parametrize(
    "field",
    (
        "root_link_lin_vel_w",
        "root_link_ang_vel_w",
        "root_link_lin_vel_b",
        "root_link_ang_vel_b",
        "projected_gravity_b",
        "actuator_force",
    ),
)
def test_nonfinite_catches_critical_dynamic_state(field):
    env, data = _env()
    asset_cfg = SimpleNamespace(name="robot")

    assert not nonfinite(env, asset_cfg=asset_cfg).item()
    getattr(data, field)[0, 0] = torch.nan
    assert nonfinite(env, asset_cfg=asset_cfg).item()
