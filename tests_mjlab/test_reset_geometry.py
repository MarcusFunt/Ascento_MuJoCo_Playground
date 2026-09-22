import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import load_env_cfg

import ascento_mjlab.tasks  # noqa: F401
from ascento_mjlab.mdp import events as ascento_events
from ascento_mjlab.mdp.events import flat_ground_wheel_bottom_heights


def test_recovery_resets_are_support_consistent_across_thousands_of_samples():
    cfg = load_env_cfg("Ascento-Recovery-Flat")
    cfg.scene.num_envs = 128
    cfg.scene.sensors = (
        *cfg.scene.sensors,
        ContactSensorCfg(
            name="nonwheel_ground_contact",
            primary=ContactMatch(
                mode="body",
                pattern=r".*",
                entity="robot",
                exclude=(r".*_wheel$",),
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "dist"),
            reduce="mindist",
        ),
    )
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        observed_lowest = []
        observed_nonwheel = []
        # 128 envs * 32 resets = 4096 independently sampled recovery states.
        for _ in range(32):
            env.reset()
            bottoms = flat_ground_wheel_bottom_heights(env)
            observed_lowest.append(bottoms.amin(dim=1))

            sensor = env.scene["nonwheel_ground_contact"].data
            assert sensor.found is not None
            observed_nonwheel.append(sensor.found.gt(0).any(dim=1))

        lowest = torch.cat(observed_lowest)
        nonwheel = torch.cat(observed_nonwheel)
        assert lowest.numel() == 4096

        # No accidental penetration or whole-robot hovering from a fixed root
        # height: the true lower outer-wheel surface is anchored to the plane.
        assert torch.max(torch.abs(lowest)).item() <= 2.0e-3

        # Recovery difficulty should come from the configured pose/velocity
        # disturbance, not the chassis/thigh/shank starting inside the floor.
        assert not torch.any(nonwheel).item()
    finally:
        env.close()


def test_all_flat_tasks_use_support_aware_root_resets():
    for task in (
        "Ascento-Balance-Flat",
        "Ascento-Velocity-Flat",
        "Ascento-Recovery-Flat",
        "Ascento-Jump-Flat",
    ):
        cfg = load_env_cfg(task)
        assert cfg.events["reset_supported_pose"].func.__name__ == "reset_root_state_supported"


def test_balance_recovery_reset_curriculum_ramps_hard_fraction_and_ranges(monkeypatch):
    class FakeEnv:
        num_envs = 4
        device = "cpu"
        common_step_counter = 0

    calls = []

    def fake_supported(env, env_ids, *, pose_range, velocity_range, asset_cfg):
        calls.append((env_ids.clone(), pose_range, velocity_range, asset_cfg))

    samples = torch.tensor([0.01, 0.15, 0.50, 0.90])
    monkeypatch.setattr(ascento_events, "reset_root_state_supported", fake_supported)
    monkeypatch.setattr(ascento_events.torch, "rand", lambda n, device=None: samples[:n].clone())

    kwargs = dict(
        asset_cfg=object(),
        normal_pose_range={"pitch": (-0.08, 0.08)},
        normal_velocity_range={"pitch": (-0.10, 0.10)},
        hard_pose_range_start={"pitch": (0.08, 0.10)},
        hard_pose_range_end={"pitch": (0.08, 0.15)},
        hard_velocity_range_start={"pitch": (-0.20, 0.20)},
        hard_velocity_range_end={"pitch": (-0.50, 0.50)},
        hard_fraction_start=0.10,
        hard_fraction_end=0.30,
        ramp_control_steps=100,
    )

    env = FakeEnv()
    ascento_events.mixed_balance_recovery_reset(env, None, **kwargs)
    assert len(calls) == 2
    normal, hard = calls
    assert normal[0].tolist() == [1, 2, 3]
    assert hard[0].tolist() == [0]
    assert hard[1]["pitch"] == (0.08, 0.10)
    assert hard[2]["pitch"] == (-0.20, 0.20)

    calls.clear()
    env.common_step_counter = 100
    ascento_events.mixed_balance_recovery_reset(env, None, **kwargs)
    normal, hard = calls
    assert normal[0].tolist() == [2, 3]
    assert hard[0].tolist() == [0, 1]
    assert hard[1]["pitch"] == (0.08, 0.15)
    assert hard[2]["pitch"] == (-0.50, 0.50)


def test_balance_recovery_task_uses_mixed_support_aware_reset():
    cfg = load_env_cfg("Ascento-Balance-Recovery-Flat")
    assert cfg.events["reset_supported_pose"].func.__name__ == "mixed_balance_recovery_reset"
