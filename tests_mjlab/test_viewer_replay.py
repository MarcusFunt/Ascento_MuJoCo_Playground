from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from ascento_mjlab.viewer.replay import (
    EventTriggerDetector,
    PolicyReplayBuffer,
    PolicyReplayRecorder,
    TransitionSnapshot,
)
from ascento_mjlab.viewer.worker import ViewerTransitionObserver


@dataclass
class Frame:
    sequence_id: int
    transition: TransitionSnapshot

    def to_dict(self):
        return {"sequence_id": self.sequence_id, "transition": self.transition.to_dict()}


def make_frame(sequence: int, *, fallen: bool = False, margin: float = 0.8, tilt: float = 0.1):
    return Frame(
        sequence,
        TransitionSnapshot(
            episode_id=1,
            episode_step=sequence,
            sim_time_s=sequence * 0.01,
            reward_rate=1.0,
            step_reward=0.01,
            reward_terms={"survive": 1.0},
            tilt_rad=tilt,
            tilt_rate_rad_s=0.0,
            height_m=0.7,
            contacts={"left": True, "right": True},
            balance_margin=margin,
            fallen=fallen,
            reset=fallen,
        ),
    )


def test_replay_buffer_keeps_recent_window_and_persists_gzip_capture(tmp_path):
    buffer = PolicyReplayBuffer(step_dt=0.01, capture_seconds=0.05, margin_frames=0)
    for sequence in range(1, 9):
        buffer.append(make_frame(sequence))

    frames = buffer.snapshot()
    assert [frame.sequence_id for frame in frames] == [4, 5, 6, 7, 8]

    recorder = PolicyReplayRecorder(tmp_path, viewer_id="viewer", run_id="run")
    path = recorder.persist(
        event_type="manual",
        frames=frames,
        trigger_sequence=8,
        trigger_reason="manual capture",
        checkpoint="model_8.pt",
        checkpoint_iteration=8,
        schema={"version": 1},
    )
    assert path.name.startswith("event_") and path.suffix == ".gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["version"] == 1
    assert payload["event_type"] == "manual"
    assert payload["trigger_sequence"] == 8
    assert [frame["sequence_id"] for frame in payload["frames"]] == [4, 5, 6, 7, 8]
    assert recorder.list_summaries()[0]["frame_count"] == 5


def test_replay_recorder_prunes_only_oldest_viewer_capture_at_retention_limit(tmp_path):
    recorder = PolicyReplayRecorder(tmp_path, viewer_id="viewer", run_id="run", max_events=2)
    for sequence in range(1, 4):
        recorder.persist(
            event_type="manual",
            frames=[make_frame(sequence)],
            trigger_sequence=sequence,
            trigger_reason="manual",
            checkpoint=f"model_{sequence}.pt",
            checkpoint_iteration=sequence,
            schema={"version": 1},
        )

    assert len(list(tmp_path.glob("event_*.json.gz"))) == 2
    assert len(recorder.list_summaries()) == 2
    assert len(recorder.list_events()) == 2


def test_event_trigger_detector_marks_fall_and_requires_recovery_dwell():
    detector = EventTriggerDetector(recovery_dwell_s=0.5)

    assert detector.update(make_frame(1, fallen=True), fall_boundary_rad=1.0) == "fall"

    detector = EventTriggerDetector(recovery_dwell_s=0.5)
    assert detector.update(make_frame(2, margin=0.2), fall_boundary_rad=1.0) is None
    assert detector.update(make_frame(3, margin=0.8), fall_boundary_rad=1.0) is None
    assert detector.update(make_frame(52, margin=0.8), fall_boundary_rad=1.0) is None
    assert detector.update(make_frame(54, margin=0.8), fall_boundary_rad=1.0) == "recovery"


def test_transition_observer_pairs_reward_with_pre_reset_terminal_state():
    class TerminationManager:
        def __init__(self):
            self.fallen = torch.tensor([False])

        def compute(self):
            self.fallen.fill_(True)
            return torch.tensor([True])

        def get_term(self, name):
            assert name == "fallen"
            return self.fallen

        def get_term_cfg(self, name):
            assert name == "fallen"
            return SimpleNamespace(params={"min_height": 0.35, "max_gravity_z": -0.5})

    class Env:
        def __init__(self):
            self.step_dt = 0.01
            self.common_step_counter = 0
            self.episode_length_buf = torch.tensor([0])
            self.termination_manager = TerminationManager()
            self.reward_manager = SimpleNamespace(
                _step_reward=torch.tensor([[2.0, -0.5]]),
                active_terms=("survive", "tilt"),
                compute=lambda **_kwargs: torch.tensor([0.015]),
            )
            angle = math.radians(20)
            robot = SimpleNamespace(data=SimpleNamespace(
                projected_gravity_b=torch.tensor([[math.sin(angle), 0.0, -math.cos(angle)]]),
                root_link_pos_w=torch.tensor([[0.0, 0.0, 0.40]]),
                root_link_ang_vel_b=torch.zeros((1, 3)),
            ))
            self.scene = {
                "robot": robot,
                "left_wheel_contact": SimpleNamespace(data=SimpleNamespace(found=torch.tensor([[True]]))),
                "right_wheel_contact": SimpleNamespace(data=SimpleNamespace(found=torch.tensor([[False]]))),
            }

        def step(self, actions):
            self.episode_length_buf += 1
            self.common_step_counter += 1
            self.termination_manager.compute()
            self.reward_manager.compute(dt=self.step_dt)
            # The environment auto-resets before returning, as MJLab does.
            self.scene["robot"].data.projected_gravity_b.copy_(torch.tensor([[0.0, 0.0, -1.0]]))
            self.scene["robot"].data.root_link_pos_w[0, 2] = 0.8
            self.episode_length_buf.zero_()
            return (
                torch.tensor([[9.0]]),
                torch.tensor([0.015]),
                torch.tensor([True]),
                torch.tensor([False]),
                {},
            )

    env = Env()
    observer = ViewerTransitionObserver(env)
    env.step(torch.zeros((1, 1)))

    transition = observer.latest
    assert transition is not None
    assert transition.episode_step == 1
    assert transition.sim_time_s == 0.01
    assert transition.step_reward == pytest.approx(0.015)
    assert transition.reward_terms == {"survive": 2.0, "tilt": -0.5}
    assert transition.tilt_rad == pytest.approx(math.radians(20))
    assert transition.height_m == pytest.approx(0.40)
    assert transition.contacts == {"left": True, "right": False}
    assert transition.fallen is True and transition.reset is True
    observer.close()
    assert env.episode_length_buf.item() == 0
