import math

import pytest
import torch

from ascento_mjlab.evaluation.runner import _morphology_metric_arrays


def test_morphology_telemetry_arrays_preserve_rates_contacts_and_inactive_rows():
    metrics = _morphology_metric_arrays(
        denom=torch.tensor([4.0, 0.0]),
        support_count=torch.tensor([3.0, 0.0]),
        left_only_contact_count=torch.tensor([1.0, 0.0]),
        right_only_contact_count=torch.tensor([0.0, 0.0]),
        max_single_wheel_support_s=torch.tensor([0.08, 0.0]),
        wheel_contact_transition_count=torch.tensor([2.0, 0.0]),
        sum_leg_pose_asymmetry_sq=torch.tensor([0.02, 0.0]),
        leg_pose_asymmetry_trace=torch.tensor([[0.1, 0.2, 0.3, float("nan")], [float("nan")] * 4]),
        sum_leg_target_offset_sq=torch.tensor([0.04, 0.0]),
        leg_target_offset_count=torch.tensor([4.0, 0.0]),
        sum_leg_target_rate_sq=torch.tensor([1.0, 0.0]),
        leg_target_rate_count=torch.tensor([2.0, 0.0]),
    )

    assert metrics["dual_wheel_contact_fraction"].tolist() == pytest.approx([0.75, 0.0])
    assert metrics["left_only_contact_fraction"].tolist() == pytest.approx([0.25, 0.0])
    assert metrics["right_only_contact_fraction"].tolist() == pytest.approx([0.0, 0.0])
    assert metrics["max_continuous_single_wheel_support_s"].tolist() == pytest.approx([0.08, 0.0])
    assert metrics["wheel_contact_transition_count"].tolist() == pytest.approx([2.0, 0.0])
    assert metrics["left_right_leg_pose_asymmetry_rms_rad"][0].item() == pytest.approx(
        math.sqrt(0.02 / 4.0)
    )
    assert metrics["left_right_leg_pose_asymmetry_p95_rad"][0].item() == pytest.approx(0.29)
    assert math.isnan(metrics["left_right_leg_pose_asymmetry_p95_rad"][1].item())
    assert metrics["leg_target_offset_rms_rad"].tolist() == pytest.approx([0.1, 0.0])
    assert metrics["leg_target_rate_rms_rad_s"].tolist() == pytest.approx([math.sqrt(0.5), 0.0])


def test_guard_morphology_replay_smoke_suite_is_small_and_diagnostic():
    from pathlib import Path

    from ascento_mjlab.evaluation.scenarios import materialize_suite
    from ascento_mjlab.evaluation.schema import load_suite

    suite = load_suite(
        Path("benchmarks/suites/guard_generalist_morphology_replay_smoke_v1.toml")
    )
    scenarios = materialize_suite(suite, step_dt=0.02)

    assert suite.task == "Ascento-Generalist-Locomotion-Flat"
    assert suite.policy_mode == "deterministic"
    assert not suite.gates
    assert len(scenarios) == 4
    assert {scenario.family for scenario in scenarios} == {
        "flat_precision",
        "push_recovery_retarget",
        "medium_target",
        "long_target",
    }
