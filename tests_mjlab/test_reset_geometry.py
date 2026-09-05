import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import load_env_cfg

import ascento_mjlab.tasks  # noqa: F401
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
