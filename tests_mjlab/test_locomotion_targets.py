from types import SimpleNamespace

import torch

from ascento_mjlab.mdp.events import (
    RepeatedRandomWorldTargetSequence,
    initialize_random_world_target,
    world_target_xy,
)


class _Scene(dict):
    def __init__(self, *args, env_origins: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_origins = env_origins


class _Asset:
    def __init__(self, positions: torch.Tensor):
        count = positions.shape[0]
        self.data = SimpleNamespace(
            root_link_pos_w=positions.clone(),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(count, 1),
            root_link_lin_vel_b=torch.zeros((count, 3)),
            root_link_ang_vel_b=torch.zeros((count, 3)),
        )


def _env(count: int = 4):
    origins = torch.stack(
        (torch.arange(count, dtype=torch.float32) * 2.0, torch.zeros(count), torch.zeros(count)),
        dim=1,
    )
    asset = _Asset(origins + torch.tensor([0.0, 0.0, 0.75]))
    found = torch.ones((count, 1), dtype=torch.bool)
    scene = _Scene(
        {
            "robot": asset,
            "left_wheel_contact": SimpleNamespace(data=SimpleNamespace(found=found.clone())),
            "right_wheel_contact": SimpleNamespace(data=SimpleNamespace(found=found.clone())),
        },
        env_origins=origins,
    )
    return SimpleNamespace(
        num_envs=count,
        device=torch.device("cpu"),
        scene=scene,
        step_dt=0.1,
    )


def test_random_world_target_is_immediately_reachable_and_arena_bounded():
    torch.manual_seed(7)
    env = _env()

    initialize_random_world_target(
        env,
        min_distance_m=0.15,
        max_distance_m=0.35,
        arena_half_extent_m=0.65,
    )

    current = env.scene["robot"].data.root_link_pos_w[:, :2]
    targets = world_target_xy(env)
    distance = torch.linalg.vector_norm(targets - current, dim=1)
    local_target = targets - env.scene.env_origins[:, :2]

    assert torch.all(distance >= 0.15 - 1e-6)
    assert torch.all(distance <= 0.35 + 1e-6)
    assert torch.all(local_target.abs() <= 0.65 + 1e-6)


def test_repeated_target_sequence_requires_a_settled_hold_before_resampling():
    torch.manual_seed(11)
    env = _env(count=1)
    asset = env.scene["robot"]
    env.ascento_world_target_state = {
        "target_xy": asset.data.root_link_pos_w[:, :2].clone(),
        "target_yaw": torch.zeros(1),
    }
    sequence = RepeatedRandomWorldTargetSequence(None, env)
    cfg = SimpleNamespace(name="robot")

    for _ in range(3):
        sequence(
            env,
            None,
            target_reached_distance_m=0.04,
            target_hold_s=0.35,
            min_target_distance_m=0.15,
            max_target_distance_m=0.35,
            arena_half_extent_m=0.65,
            asset_cfg=cfg,
        )
        assert torch.allclose(world_target_xy(env), asset.data.root_link_pos_w[:, :2])

    sequence(
        env,
        None,
        target_reached_distance_m=0.04,
        target_hold_s=0.35,
        min_target_distance_m=0.15,
        max_target_distance_m=0.35,
        arena_half_extent_m=0.65,
        asset_cfg=cfg,
    )

    distance = torch.linalg.vector_norm(
        world_target_xy(env) - asset.data.root_link_pos_w[:, :2], dim=1
    )
    assert float(distance[0]) >= 0.15 - 1e-6
