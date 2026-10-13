"""Exact benchmark replay selection, evaluator flags and persisted trace tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from ascento_mjlab.evaluation.diagnostics import (
    all_conditions,
    recovery_conditions,
    stop_conditions,
)
from ascento_mjlab.evaluation.replay_capture import BenchmarkReplayRecorder
from ascento_mjlab.evaluation.schema import ScenarioSpec
from ascento_mjlab.robot_cfg import ROBOT_XML
from ascento_mjlab.tools.capture_benchmark import select_scenarios
from ascento_mjlab.tools.export_rerun import validate_capture


def test_stop_predicates_distinguish_contact_and_dwell_requirements():
    x = torch.tensor([0.0, 0.0])
    conditions = stop_conditions(
        target_error=x,
        heading_error=x,
        planar_speed=x,
        tilt=x,
        angular_xy=x,
        yaw_rate=x,
        both_supported=torch.tensor([True, False]),
    )
    assert list(conditions) == [
        "distance",
        "heading",
        "speed",
        "tilt",
        "angular_xy",
        "yaw_rate",
        "both_wheels",
    ]
    assert all_conditions(conditions).tolist() == [True, False]
    assert conditions["speed"].all()
    assert conditions["both_wheels"].tolist() == [True, False]


def test_recovery_predicates_have_different_speed_and_heading_envelope():
    x = torch.tensor([0.09])
    stop = stop_conditions(
        target_error=torch.zeros(1),
        heading_error=torch.zeros(1),
        planar_speed=x,
        tilt=torch.zeros(1),
        angular_xy=torch.zeros(1),
        yaw_rate=torch.zeros(1),
        both_supported=torch.ones(1, dtype=torch.bool),
    )
    recovery = recovery_conditions(
        planar_speed=x,
        height=torch.tensor([0.75]),
        heading_error=torch.zeros(1),
        tilt=torch.zeros(1),
        angular_xy=torch.zeros(1),
        yaw_rate=torch.zeros(1),
        both_supported=torch.ones(1, dtype=torch.bool),
    )
    assert not bool(all_conditions(stop)[0])
    assert bool(all_conditions(recovery)[0])


def test_scenario_selection_is_exact_and_rejects_unknown():
    cases = [
        ScenarioSpec(
            scenario_id=f"suite/precision/{i:06d}",
            family="precision",
            task="task",
            horizon_steps=100,
            reset={"yaw": i * 0.1},
        )
        for i in range(3)
    ]
    chosen = select_scenarios(cases, families=["precision"], indices=[1])
    assert [x.scenario_id for x in chosen] == ["suite/precision/000001"]
    assert chosen[0].reset["yaw"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="Unknown scenario IDs"):
        select_scenarios(cases, scenario_ids=["suite/precision/999999"])
    with pytest.raises(ValueError, match="No such scenario"):
        select_scenarios(cases, families=["precision"], indices=[7])
    with pytest.raises(ValueError, match="Specify"):
        select_scenarios(cases)


def test_exact_capture_persists_condition_time_series_and_provenance(tmp_path):
    case = ScenarioSpec(
        scenario_id="suite/precision/000000",
        family="precision",
        task="test",
        horizon_steps=2,
        reset={"yaw": 0.5},
    )
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"synthetic-checkpoint")
    suite = tmp_path / "suite.toml"
    suite.write_text("test")
    rec = BenchmarkReplayRecorder(
        {case.scenario_id},
        tmp_path / "captures",
        checkpoint=ckpt,
        suite_path=suite,
        suite_sha256="ab" * 32,
    )
    ones = torch.tensor([True])
    zeros = torch.zeros(1)
    root = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.0, 0.0, 0.75]]),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            root_link_lin_vel_w=torch.zeros(1, 3),
            root_link_ang_vel_b=torch.zeros(1, 3),
            joint_pos=torch.zeros(1, 6),
            joint_vel=torch.zeros(1, 6),
            actuator_force=torch.ones(1, 6),
        )
    )
    for step in range(2):
        rec.record_step(
            scenarios=[case],
            step=step,
            active=ones,
            robot=root,
            raw_action=torch.full((1, 6), float(step)),
            step_rewards=torch.tensor([0.015]),
            reward_terms=torch.tensor([[2.0, -0.5]]),
            reward_term_names=["progress", "penalty"],
            step_dt=0.01,
            target_xy=torch.tensor([[0.25, 0.0]]),
            target_yaw=zeros,
            left=ones,
            right=ones,
            target_error=torch.tensor([0.25]),
            heading_error=zeros,
            planar_speed=zeros,
            tilt=zeros,
            angular_xy=zeros,
            yaw_rate=zeros,
            waypoint_active=ones,
            waypoint_dwell_count=torch.tensor([step + 1]),
            stop_flags={"distance": torch.tensor([False]), "both_wheels": ones},
            stop_now=torch.tensor([False]),
            recovery_flags={"heading": ones},
            recovery_stable_count=torch.tensor([step + 1]),
            recovered=torch.tensor([False]),
            after_disturbance=ones,
        )
    [path] = rec.save()
    with np.load(path) as data:
        assert data["contacts"].shape == (2, 2)
        assert data["evaluator_stop_dwell_s"].tolist() == pytest.approx([0.01, 0.02])
        assert data["evaluator_stop_distance"].tolist() == [0.0, 0.0]
        assert data["evaluator_stop_both_wheels"].tolist() == [1.0, 1.0]
        assert json.loads(str(data["meta_scenario"].item()))["reset"]["yaw"] == 0.5
        model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
        assert validate_capture(data, model)["count"] == 2
    pytest.importorskip("rerun")
    pytest.importorskip("rerun_loader_mjcf")
    from ascento_mjlab.tools.export_rerun import export_rerun

    rrd = path.with_suffix(".rrd")
    result = export_rerun(path, rrd)
    assert result["contacts"] is True
    assert result["evaluator_diagnostics"] is True
    assert rrd.is_file() and rrd.stat().st_size > 256


def test_export_rejects_invalid_optional_contact_flags():
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    from tests_mjlab.test_rerun_export import _episode_data

    data = _episode_data()
    data["contacts"] = np.array([[0, 1], [1, 1], [0.5, 0]])
    with pytest.raises(ValueError, match="binary"):
        validate_capture(data, model)


def test_checkpoint_handoff_restores_saved_quarter_turn_fraction():
    """An earlier V5 checkpoint must remain loadable after a changed default."""
    from copy import deepcopy

    from mjlab.tasks.registry import load_env_cfg

    import ascento_mjlab.tasks  # noqa: F401
    from ascento_mjlab.checkpoint_contract import apply_checkpoint_curriculum_contract
    from ascento_mjlab.task_contract import current_task_contract

    task = "Ascento-Generalist-Locomotion-Goal-Settle-V5-Flat"
    cfg = load_env_cfg(task, play=False)
    snapshot = deepcopy(current_task_contract(cfg))
    saved_fraction = 0.05
    for event in ("initialize_world_target", "repeated_random_world_targets"):
        snapshot["topology"]["events"][event]["params"]["quarter_turn_heading_fraction"] = (
            saved_fraction
        )
    assert apply_checkpoint_curriculum_contract(cfg, {"task_contract": snapshot}, task)
    assert current_task_contract(cfg)["topology"] == snapshot["topology"]
    bad = deepcopy(snapshot)
    bad["topology"]["events"]["initialize_world_target"]["params"][
        "quarter_turn_heading_fraction"
    ] = 1.5
    assert not apply_checkpoint_curriculum_contract(
        load_env_cfg(task, play=False), {"task_contract": bad}, task
    )
