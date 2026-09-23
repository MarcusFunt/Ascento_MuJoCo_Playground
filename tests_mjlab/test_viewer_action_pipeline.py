from __future__ import annotations

from types import SimpleNamespace

import torch

from ascento_mjlab.control_contract import LEG_POSITION_SCALE_RAD
from ascento_mjlab.robot_cfg import JOINT_NAMES
from ascento_mjlab.structured_action import StructuredTargetAction
from ascento_mjlab.viewer.introspection import (
    StructuredActionObserver,
    build_action_pipeline_snapshot,
)


def test_action_pipeline_reports_wrapper_action_term_and_joint_limit_clipping():
    action_term = StructuredTargetAction.__new__(StructuredTargetAction)
    action_term._raw_actions = torch.tensor([[1.0, -1.0, 0.5, 0.1, -1.0, 0.8]])
    action_term._leg_slots = torch.tensor([0, 1, 3, 4])
    action_term._wheel_slots = torch.tensor([2, 5])
    action_term._joint_ids = torch.arange(6)
    action_term._velocity_targets = torch.tensor([[2.0, -3.2]])
    action_term.cfg = SimpleNamespace(actuator_names=JOINT_NAMES)
    nominal = torch.tensor([[0.3, 0.2, 0.0, -0.2, 0.1, 0.0]])
    requested = nominal[0, action_term._leg_slots] + torch.tensor(
        [1.0, -1.0, 0.1, -1.0]
    ) * LEG_POSITION_SCALE_RAD
    action_term._position_targets = torch.tensor(
        [[0.4, requested[1], requested[2], requested[3]]]
    )
    limits = torch.tensor(
        [[
            [0.0, 0.4], [-1.0, 1.0], [-10.0, 10.0], [-1.0, 1.0],
            [-1.0, 1.0], [-10.0, 10.0],
        ]]
    )
    action_term._entity = SimpleNamespace(
        data=SimpleNamespace(default_joint_pos=nominal, joint_pos_limits=limits)
    )
    actor_output = torch.tensor([[1.5, -1.2, 0.5, 0.1, -2.0, 0.8]])

    snapshot = build_action_pipeline_snapshot(
        actor_output,
        action_term,
        wrapper_clip=1.0,
    )

    assert torch.allclose(
        snapshot.wrapper_clipped,
        torch.tensor([1.0, -1.0, 0.5, 0.1, -1.0, 0.8]),
    )
    assert torch.allclose(
        snapshot.processed_action,
        torch.tensor([1.0, -1.0, 0.5, 0.1, -1.0, 0.8]),
    )
    assert snapshot.wrapper_clipped_flags == (True, True, False, False, True, False)
    assert snapshot.action_term_clipped_flags == (False,) * 6
    assert abs(snapshot.targets[0].value - 0.4) < 1e-6
    assert snapshot.targets[0].joint_limit_clipped is True
    assert snapshot.targets[1].joint_limit_clipped is False
    assert snapshot.targets[0].unit == "rad"
    assert snapshot.targets[2].unit == "rad/s"


def test_action_pipeline_detects_action_term_clip_when_wrapper_does_not_clip():
    action_term = StructuredTargetAction.__new__(StructuredTargetAction)
    action_term._raw_actions = torch.tensor([[1.0, -1.0]])
    action_term._leg_slots = torch.tensor([0])
    action_term._wheel_slots = torch.tensor([1])
    action_term._joint_ids = torch.arange(2)
    action_term._position_targets = torch.tensor([[LEG_POSITION_SCALE_RAD]])
    action_term._velocity_targets = torch.zeros((1, 1))
    action_term.cfg = SimpleNamespace(actuator_names=("hip", "wheel"))
    action_term._entity = SimpleNamespace(
        data=SimpleNamespace(
            default_joint_pos=torch.zeros((1, 2)),
            joint_pos_limits=torch.tensor([[[-2.0, 2.0], [-10.0, 10.0]]]),
        )
    )

    snapshot = build_action_pipeline_snapshot(
        torch.tensor([[1.4, -1.3]]),
        action_term,
        wrapper_clip=None,
    )

    assert torch.allclose(snapshot.wrapper_clipped, torch.tensor([1.4, -1.3]))
    assert snapshot.action_term_clipped_flags == (True, True)
    assert snapshot.processed_action.tolist() == [1.0, -1.0]


def test_viewer_action_observer_preserves_actual_step_outputs_across_env_reset():
    action_term = StructuredTargetAction.__new__(StructuredTargetAction)
    action_term._raw_actions = torch.zeros((1, 2))
    action_term._leg_slots = torch.tensor([0])
    action_term._wheel_slots = torch.tensor([1])
    action_term._joint_ids = torch.arange(2)
    action_term._position_targets = torch.zeros((1, 1))
    action_term._velocity_targets = torch.zeros((1, 1))
    action_term.cfg = SimpleNamespace(actuator_names=("hip", "wheel"))
    action_term._entity = SimpleNamespace(
        data=SimpleNamespace(
            default_joint_pos=torch.zeros((1, 2)),
            joint_pos_limits=torch.tensor([[[-2.0, 2.0], [-10.0, 10.0]]]),
        )
    )

    def process_actions(actions):
        action_term._raw_actions.copy_(actions.clamp(-1.0, 1.0))
        action_term._position_targets.copy_(action_term._raw_actions[:, :1] * LEG_POSITION_SCALE_RAD)
        action_term._velocity_targets.copy_(action_term._raw_actions[:, 1:2] * 2.0)

    action_term.process_actions = process_actions
    observer = StructuredActionObserver(action_term)
    action_term.process_actions(torch.tensor([[0.5, -0.75]]))
    saved = observer.latest
    action_term._raw_actions.zero_()
    action_term._position_targets.zero_()
    action_term._velocity_targets.zero_()

    assert saved is not None
    assert torch.allclose(saved.processed_action, torch.tensor([0.5, -0.75]))
    assert torch.allclose(saved.position_targets, torch.tensor([0.5 * LEG_POSITION_SCALE_RAD]))
    assert torch.allclose(saved.velocity_targets, torch.tensor([-1.5]))
    observer.close()
    assert action_term.process_actions is process_actions
