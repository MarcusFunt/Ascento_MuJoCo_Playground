"""CPU-only tests for sampled real-training telemetry and read-only monitoring."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from dashboard.live_training import read_live_frame

from ascento_mjlab.live_training_monitor import (
    LatestFrameWriter,
    PpoLiveMonitor,
    selected_environment_ids,
)


class Frames:
    def __init__(self):
        self.frames: list[dict] = []
        self.closed = False

    def offer(self, frame: dict) -> None:
        self.frames.append(frame)

    def close(self):
        self.closed = True


def build_env():
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.0, 0.0, 0.75], [1.0, 0.0, 0.7]]),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
            root_link_lin_vel_w=torch.tensor([[0.3, 0.0, 0.0], [0.1, 0.0, 0.0]]),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
            joint_pos=torch.zeros(2, 6),
        )
    )
    source = SimpleNamespace(
        scene={
            "robot": robot,
            "left_wheel_contact": SimpleNamespace(
                data=SimpleNamespace(found=torch.tensor([[True], [False]]))
            ),
            "right_wheel_contact": SimpleNamespace(
                data=SimpleNamespace(found=torch.tensor([[True], [True]]))
            ),
        },
        num_envs=2,
        common_step_counter=0,
        step_dt=0.01,
        episode_length_buf=torch.tensor([1, 20]),
        ascento_world_target_state={"target_xy": torch.tensor([[2.0, 0.0], [2.0, 1.0]])},
        command_manager=SimpleNamespace(
            get_command=lambda name: torch.tensor([[0.5], [0.2]]) if name == "speed" else None
        ),
        reward_manager=SimpleNamespace(
            active_terms=["progress", "penalty"],
            _step_reward=torch.tensor([[2.0, -0.5], [1.0, -0.3]]),
        ),
    )

    def step(actions):
        source.common_step_counter += 1
        robot.data.root_link_pos_w[0, 0] += 0.1
        source.episode_length_buf += 1
        return ({}, torch.tensor([0.015, 0.007]), torch.tensor([0, 1]), {})

    return SimpleNamespace(unwrapped=source, step=step)


def test_sampling_is_wall_clock_limited_and_does_not_change_steps():
    env = build_env()
    original = env.step
    writer = Frames()
    now = [0.0]
    sampler = PpoLiveMonitor(
        env,
        Path("/tmp/unused.json"),
        environment_ids=(0, 1),
        interval_s=0.5,
        clock=lambda: now[0],
        writer=writer,
    )
    actions = torch.zeros(2, 6)
    first = env.step(actions)
    assert len(writer.frames) == 1
    assert first[1].tolist() == pytest.approx([0.015, 0.007])
    a, b = writer.frames[0]["environments"]
    assert writer.frames[0]["source"] == "actual_ppo_rollout"
    assert a["environment_id"] == 0 and b["environment_id"] == 1
    assert a["target_xy"] == pytest.approx([2, 0])
    assert a["speed_mps"] == pytest.approx(0.3)
    assert a["closing_speed_mps"] == pytest.approx(0.3)
    assert a["step_reward"] == pytest.approx(0.015)
    assert a["reward_terms_rate"] == {"progress": 2, "penalty": -0.5}
    assert b["reset"] is True
    now[0] = 0.1
    env.step(actions)
    assert len(writer.frames) == 1
    now[0] = 0.6
    env.step(actions)
    assert len(writer.frames) == 2
    sampler.close()
    assert writer.closed
    assert env.step == original


def test_sampler_failure_never_prevents_training_step():
    env = build_env()
    del env.unwrapped.scene["robot"]
    writer = Frames()
    sampler = PpoLiveMonitor(env, Path("/tmp/unused.json"), writer=writer)
    env.step(torch.zeros(2, 6))
    assert sampler._disabled
    assert not writer.frames
    env.step(torch.zeros(2, 6))
    sampler.close()


@pytest.mark.parametrize("value", ["", "0,0", "-1", "2", "0,1,0", "0,1,2,3,4", "alpha"])
def test_invalid_environment_selection(value):
    with pytest.raises(ValueError):
        selected_environment_ids(value, 2)


def test_writer_persists_atomically_and_close_flushes(tmp_path):
    path = tmp_path / "live_monitor.json"
    writer = LatestFrameWriter(path)
    writer.offer({"schema_version": 1, "sequence": 3, "sampled_at": time.time()})
    writer.close()
    assert json.loads(path.read_text())["sequence"] == 3


def test_nested_monitor_lookup_and_stale_state(tmp_path):
    path = tmp_path / "ascento_flat" / "2026-10-12_12-00-00"
    path.mkdir(parents=True)
    snapshot = {
        "schema_version": 1,
        "source": "actual_ppo_rollout",
        "sampled_at": 100.0,
        "sequence": 4,
        "environments": [],
    }
    (path / "live_monitor.json").write_text(json.dumps(snapshot))
    assert read_live_frame(tmp_path, now=102.0)["stale"] is False
    assert read_live_frame(tmp_path, now=115.0)["stale"] is True
    snapshot["source"] = "untrusted_data"
    (path / "live_monitor.json").write_text(json.dumps(snapshot))
    assert read_live_frame(tmp_path)["available"] is False
