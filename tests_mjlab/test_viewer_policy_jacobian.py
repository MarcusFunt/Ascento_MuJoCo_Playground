from __future__ import annotations

from dataclasses import replace

import torch

from ascento_mjlab.viewer.introspection import (
    JacobianCadence,
    PolicyIntrospector,
    compute_policy_jacobian,
)
from ascento_mjlab.viewer.introspection_contract import resolve_runtime_handles
from tests_mjlab.test_policy_introspection import _runner


def test_raw_observation_jacobian_matches_direct_autograd_and_has_action_by_input_shape():
    runner, actor, critic = _runner()
    runner.alg.critic = critic
    handles = resolve_runtime_handles(runner)
    raw = torch.tensor([2.0, 4.0, 0.0])

    jacobian = compute_policy_jacobian(handles, raw)
    expected = torch.autograd.functional.jacobian(
        lambda value: actor(
            {"actor": value.reshape(1, -1)},
            stochastic_output=False,
        )[0],
        raw.clone().requires_grad_(True),
    )

    assert jacobian.shape == (2, 3)
    assert torch.allclose(jacobian, expected, atol=1e-6, rtol=1e-5)


def test_jacobian_cadence_accepts_documented_rates_and_throttles():
    assert [JacobianCadence(rate).frequency_hz for rate in (0, 1, 2, 5, 10)] == [0, 1, 2, 5, 10]
    cadence = JacobianCadence(2)
    assert cadence.should_calculate(now=10.0)
    assert not cadence.should_calculate(now=10.2)
    assert cadence.should_calculate(now=10.5)
    disabled = JacobianCadence(0)
    assert not disabled.should_calculate(now=10.0)


def test_jacobian_cadence_rejects_unsupported_rate():
    try:
        JacobianCadence(3)
    except ValueError as error:
        assert "0, 1, 2, 5, 10" in str(error)
    else:
        raise AssertionError("unsupported Jacobian rate must be rejected")


def test_introspector_publishes_jacobian_with_sequence_and_age_metadata():
    runner, actor, critic = _runner()
    runner.alg.critic = critic
    handles = resolve_runtime_handles(runner)
    introspector = PolicyIntrospector(handles, jacobian_hz=2)
    observations = {
        "actor": torch.tensor([[2.0, 4.0, 0.0]]),
        "critic": torch.ones((1, 4)),
    }
    capture = introspector.capture_action(
        observations,
        lambda: actor(observations, stochastic_output=False),
        sequence_id=100,
    )

    with torch.no_grad():
        first = introspector.maybe_calculate_jacobian(capture.frame, now=10.0)
        skipped_frame = replace(capture.frame, sequence_id=101)
        skipped = introspector.maybe_calculate_jacobian(skipped_frame, now=10.2)
        recalculated_frame = replace(capture.frame, sequence_id=102)
        recalculated = introspector.maybe_calculate_jacobian(recalculated_frame, now=10.5)

    assert first.jacobian.shape == (2, 3)
    assert first.jacobian_calculated_at_sequence == 100
    assert first.jacobian_age_steps == 0
    assert skipped.jacobian_calculated_at_sequence == 100
    assert skipped.jacobian_age_steps == 1
    assert abs(skipped.jacobian_age_ms - 200.0) < 1e-6
    assert recalculated.jacobian_calculated_at_sequence == 102
    assert recalculated.jacobian_age_steps == 0
    introspector.close()
