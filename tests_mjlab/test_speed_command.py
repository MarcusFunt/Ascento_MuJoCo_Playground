import threading
from types import SimpleNamespace

import pytest
import torch

import ascento_mjlab.mdp.commands as commands


def test_speed_command_term_is_available():
    assert hasattr(commands, "AscentoTargetSpeedCommandCfg")
    assert hasattr(commands, "AscentoTargetSpeedCommand")


def _bare_speed_term():
    term = object.__new__(commands.AscentoTargetSpeedCommand)
    term.cfg = SimpleNamespace(max_speed_mps=1.0, max_speed_slew_rate_mps_per_s=0.5)
    term._env = SimpleNamespace(num_envs=1, device=torch.device("cpu"), step_dt=0.1)
    term._command = torch.zeros((1, 1))
    term._target_speed = torch.zeros((1, 1))
    term._manual_override = torch.zeros(1, dtype=torch.bool)
    term._pending_lock = threading.Lock()
    term._pending_speed = {}
    return term


def test_runtime_speed_change_is_queued_and_rate_limited_at_control_step():
    term = _bare_speed_term()

    term.queue_speed_mps(0.8)

    assert term.command[0, 0].item() == pytest.approx(0.0)
    term._update_command(None)

    assert term.command[0, 0].item() == pytest.approx(0.05)


def test_speed_tracking_reward_matches_requested_planar_speed():
    from ascento_mjlab.mdp.rewards import track_world_target_speed

    speeds = torch.tensor([[0.4, 0.0, 0.0], [0.2, 0.0, 0.0]])
    env = SimpleNamespace(
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    root_link_pos_w=torch.zeros((2, 3)),
                    root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2),
                    root_link_lin_vel_w=speeds,
                    projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]] * 2),
                )
            )
        },
        command_manager=SimpleNamespace(get_command=lambda _: torch.tensor([[0.4], [0.4]])),
        ascento_world_target_state={
            "target_xy": torch.tensor([[2.0, 0.0], [2.0, 0.0]]),
        },
    )

    reward = track_world_target_speed(env, std=0.1)

    assert reward[0].item() == pytest.approx(1.0)
    assert reward[1].item() == pytest.approx(torch.exp(torch.tensor(-4.0)).item())


def test_speed_telemetry_distinguishes_requested_applied_and_achieved_speed():
    term = _bare_speed_term()
    term.cfg.entity_name = "robot"
    term._target_speed[0, 0] = 0.8
    term._command[0, 0] = 0.05
    term._speed_display = SimpleNamespace(content="")
    term._speed_display_env = 0
    term._last_display_update = 0.0
    term._env.scene = {
        "robot": SimpleNamespace(
            data=SimpleNamespace(root_link_lin_vel_w=torch.tensor([[0.25, 0.0, 0.0]]))
        )
    }

    term._update_metrics()

    assert "Requested: <b>0.80 m/s</b>" in term._speed_display.content
    assert "Applied command: <b>0.05 m/s</b>" in term._speed_display.content
    assert "Achieved planar speed: <b>0.25 m/s</b>" in term._speed_display.content

def test_speed_locomotion_env_builds_and_applies_runtime_command():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    import ascento_mjlab.tasks  # noqa: F401

    cfg = load_env_cfg("Ascento-Locomotion-Speed-Flat", play=True)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        env.reset()
        term = env.command_manager.get_term("speed")
        term.set_manual_speed_mps(0.2, env_id=0)
        env.step(torch.zeros((1, env.action_manager.total_action_dim), device=env.device))

        assert term.command.shape == (1, 1)
        assert 0.0 < term.command[0, 0].item() <= 0.2
    finally:
        env.close()

@pytest.mark.parametrize("invalid_speed", [-0.01, 1.01, float("nan"), float("inf")])
def test_manual_speed_command_rejects_nonfinite_or_out_of_range_values(invalid_speed):
    term = _bare_speed_term()

    with pytest.raises(ValueError, match="speed must be finite"):
        term.set_manual_speed_mps(invalid_speed, env_id=0)

    assert term._target_speed[0, 0].item() == pytest.approx(0.0)
