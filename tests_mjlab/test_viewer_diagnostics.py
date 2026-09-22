import math

import pytest

from ascento_mjlab.viewer.diagnostics import (
    compute_balance_confidence,
    smooth_value,
    sparkline,
)


def test_balance_confidence_is_full_when_upright_supported_and_at_nominal_height():
    result = compute_balance_confidence(
        tilt_rad=0.0,
        tilt_rate_rad_s=0.0,
        height_m=0.75,
        left_contact=True,
        right_contact=True,
        fall_tilt_rad=math.radians(60.0),
        min_height_m=0.35,
        nominal_height_m=0.75,
    )

    assert result.confidence == pytest.approx(1.0)
    assert result.predicted_tilt_rad == pytest.approx(0.0)


def test_balance_confidence_tracks_real_tilt_margin():
    result = compute_balance_confidence(
        tilt_rad=math.radians(30.0),
        tilt_rate_rad_s=0.0,
        height_m=0.75,
        left_contact=True,
        right_contact=True,
        fall_tilt_rad=math.radians(60.0),
        min_height_m=0.35,
        nominal_height_m=0.75,
    )

    assert result.tilt_margin == pytest.approx(0.5)
    assert result.confidence == pytest.approx(0.5)


def test_worsening_tilt_is_projected_forward_but_recovery_motion_is_not():
    kwargs = {
        "tilt_rad": math.radians(30.0),
        "height_m": 0.75,
        "left_contact": True,
        "right_contact": True,
        "fall_tilt_rad": math.radians(60.0),
        "min_height_m": 0.35,
        "nominal_height_m": 0.75,
        "lookahead_s": 0.25,
    }

    worsening = compute_balance_confidence(
        tilt_rate_rad_s=math.radians(60.0),
        **kwargs,
    )
    recovering = compute_balance_confidence(
        tilt_rate_rad_s=math.radians(-60.0),
        **kwargs,
    )

    assert math.degrees(worsening.predicted_tilt_rad) == pytest.approx(45.0)
    assert worsening.confidence == pytest.approx(0.25)
    assert math.degrees(recovering.predicted_tilt_rad) == pytest.approx(30.0)
    assert recovering.confidence == pytest.approx(0.5)


def test_missing_wheel_support_reduces_confidence_without_claiming_certain_failure():
    both = compute_balance_confidence(
        tilt_rad=0.0,
        tilt_rate_rad_s=0.0,
        height_m=0.75,
        left_contact=True,
        right_contact=True,
        fall_tilt_rad=math.radians(60.0),
        min_height_m=0.35,
        nominal_height_m=0.75,
    )
    one = compute_balance_confidence(
        tilt_rad=0.0,
        tilt_rate_rad_s=0.0,
        height_m=0.75,
        left_contact=True,
        right_contact=False,
        fall_tilt_rad=math.radians(60.0),
        min_height_m=0.35,
        nominal_height_m=0.75,
    )
    none = compute_balance_confidence(
        tilt_rad=0.0,
        tilt_rate_rad_s=0.0,
        height_m=0.75,
        left_contact=False,
        right_contact=False,
        fall_tilt_rad=math.radians(60.0),
        min_height_m=0.35,
        nominal_height_m=0.75,
    )

    assert both.confidence == pytest.approx(1.0)
    assert one.confidence == pytest.approx(0.65)
    assert none.confidence == pytest.approx(0.25)


def test_height_margin_can_be_the_limiting_factor():
    result = compute_balance_confidence(
        tilt_rad=0.0,
        tilt_rate_rad_s=0.0,
        height_m=0.55,
        left_contact=True,
        right_contact=True,
        fall_tilt_rad=math.radians(60.0),
        min_height_m=0.35,
        nominal_height_m=0.75,
    )

    assert result.height_margin == pytest.approx(0.5)
    assert result.confidence == pytest.approx(0.5)


def test_nonfinite_state_returns_zero_confidence():
    result = compute_balance_confidence(
        tilt_rad=math.nan,
        tilt_rate_rad_s=0.0,
        height_m=0.75,
        left_contact=True,
        right_contact=True,
        fall_tilt_rad=math.radians(60.0),
        min_height_m=0.35,
        nominal_height_m=0.75,
    )

    assert result.confidence == 0.0
    assert result.tilt_margin == 0.0


def test_smoothing_and_sparkline_helpers_are_predictable():
    assert smooth_value(None, 0.8, 0.9) == pytest.approx(0.8)
    assert smooth_value(0.8, 0.2, 0.5) == pytest.approx(0.5)
    assert sparkline([0.0, 0.5, 1.0], minimum=0.0, maximum=1.0) == "▁▅█"
