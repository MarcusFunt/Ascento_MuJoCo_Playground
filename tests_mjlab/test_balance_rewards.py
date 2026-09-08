from types import SimpleNamespace

import pytest
import torch

from ascento_mjlab.mdp.events import OneShotPlanarVelocityPush, initialize_balance_origin
from ascento_mjlab.mdp.rewards import (
    leg_pose_hold_penalty,
    leg_pose_symmetry_penalty,
    position_hold,
    settled_balance,
    upright,
)


def _env():
    robot = SimpleNamespace(
        joint_names=("left_hip", "left_knee", "left_wheel", "right_hip", "right_knee", "right_wheel"),
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.02, -0.01, 0.75]]),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
            root_link_lin_vel_b=torch.zeros((1, 3)),
            root_link_ang_vel_b=torch.zeros((1, 3)),
            joint_pos=torch.tensor([[-3.14159, -3.14159, 0.0, -3.14159, -3.14159, 0.0]]),
        )
    )
    contact = SimpleNamespace(data=SimpleNamespace(found=torch.ones((1, 1), dtype=torch.bool)))
    return SimpleNamespace(
        num_envs=1,
        device="cpu",
        scene={"robot": robot, "left_wheel_contact": contact, "right_wheel_contact": contact},
    )


def test_balance_origin_tracks_the_actual_supported_reset_position():
    env = _env()
    initialize_balance_origin(env)

    assert torch.allclose(env.ascento_balance_state["origin_xy"], torch.tensor([[0.02, -0.01]]))


def test_position_hold_is_reset_relative_and_decays_with_drift():
    env = _env()
    initialize_balance_origin(env)
    asset_cfg = SimpleNamespace(name="robot")

    assert position_hold(env, asset_cfg=asset_cfg).item() == pytest.approx(1.0)
    env.scene["robot"].data.root_link_pos_w[0, 0] += 0.5
    assert position_hold(env, asset_cfg=asset_cfg).item() == pytest.approx(torch.exp(torch.tensor(-1.0)).item())


def test_soft_leg_symmetry_penalty_distinguishes_a_persistent_knee_offset():
    env = _env()
    asset_cfg = SimpleNamespace(name="robot", joint_ids=[0, 1, 2, 3, 4, 5])

    symmetric = leg_pose_symmetry_penalty(env, asset_cfg=asset_cfg)
    env.scene["robot"].data.joint_pos[0, 4] -= 0.55
    asymmetric = leg_pose_symmetry_penalty(env, asset_cfg=asset_cfg)

    assert symmetric.item() == pytest.approx(0.0)
    assert asymmetric.item() > 0.20


def test_leg_pose_hold_penalty_rejects_synchronized_excursion():
    env = _env()
    asset_cfg = SimpleNamespace(name="robot")
    assert leg_pose_hold_penalty(env, asset_cfg=asset_cfg).item() == pytest.approx(0.0, abs=1.0e-8)
    env.scene["robot"].data.joint_pos[:, [0, 1, 3, 4]] += 0.35
    assert leg_pose_hold_penalty(env, asset_cfg=asset_cfg).item() == pytest.approx(1.0, rel=2.0e-5)


def test_leg_symmetry_resolves_joint_pairs_by_name_not_model_order():
    env = _env()
    asset_cfg = SimpleNamespace(name="robot")
    robot = env.scene["robot"]
    robot.joint_names = (
        "right_knee",
        "left_wheel",
        "right_hip",
        "left_hip",
        "right_wheel",
        "left_knee",
    )
    robot.data.joint_pos[0] = torch.tensor([0.4, 0.0, 0.2, 0.2, 0.0, 0.4])

    assert leg_pose_symmetry_penalty(env, asset_cfg=asset_cfg).item() == pytest.approx(0.0)
    robot.data.joint_pos[0, 0] += 0.55
    assert leg_pose_symmetry_penalty(env, asset_cfg=asset_cfg).item() > 0.20


def test_settled_balance_requires_both_wheels_and_gate_quality_motion():
    env = _env()
    asset_cfg = SimpleNamespace(name="robot")

    assert settled_balance(env, asset_cfg=asset_cfg).item() == pytest.approx(1.0)
    env.scene["right_wheel_contact"].data.found.zero_()
    assert settled_balance(env, asset_cfg=asset_cfg).item() == pytest.approx(0.0)


def test_signed_tilt_rewards_reject_an_inverted_robot():
    env = _env()
    asset_cfg = SimpleNamespace(name="robot")

    upright_score = upright(env, asset_cfg=asset_cfg)
    settled_score = settled_balance(env, asset_cfg=asset_cfg)
    env.scene["robot"].data.projected_gravity_b[0] = torch.tensor([0.0, 0.0, 1.0])

    assert upright_score.item() == pytest.approx(1.0)
    assert settled_score.item() == pytest.approx(1.0)
    assert upright(env, asset_cfg=asset_cfg).item() < 1.0e-12
    assert settled_balance(env, asset_cfg=asset_cfg).item() < 1.0e-12


def test_one_shot_planar_push_matches_the_gate_disturbance_shape():
    root_velocity = torch.zeros((4, 6))
    writes = []

    def write_root_link_velocity_to_sim(velocity, env_ids):
        root_velocity[env_ids] = velocity
        writes.append(env_ids.clone())

    env = SimpleNamespace(
        num_envs=4,
        device="cpu",
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(root_link_vel_w=root_velocity),
                write_root_link_velocity_to_sim=write_root_link_velocity_to_sim,
            )
        },
    )
    push = OneShotPlanarVelocityPush(SimpleNamespace(), env)
    asset_cfg = SimpleNamespace(name="robot")
    torch.manual_seed(7)

    push(env, torch.arange(4), min_delta_v=0.15, max_delta_v=0.45, asset_cfg=asset_cfg)

    planar = root_velocity[:, :2]
    assert torch.all((planar != 0).sum(dim=1) == 1)
    assert torch.all((planar.abs().sum(dim=1) >= 0.15) & (planar.abs().sum(dim=1) <= 0.45))
    assert torch.equal(root_velocity[:, 2:], torch.zeros((4, 4)))
    assert len(writes) == 1

    push(env, torch.arange(4), min_delta_v=0.15, max_delta_v=0.45, asset_cfg=asset_cfg)
    assert len(writes) == 1

    push.reset(torch.tensor([1, 3]))
    push(env, torch.arange(4), min_delta_v=0.15, max_delta_v=0.45, asset_cfg=asset_cfg)
    assert torch.equal(writes[-1], torch.tensor([1, 3]))
