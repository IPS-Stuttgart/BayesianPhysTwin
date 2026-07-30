from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.cloth_action_phase_diagnostic import (
    apply_action_phase_correction,
    apply_action_phase_translation_delta,
    fit_action_phase_profile,
    fit_action_phase_translation_delta,
    projected_residual_scale,
)


def test_projected_residual_scale_recovers_known_amplitude() -> None:
    physical = np.zeros((4, 3), dtype=np.float64)
    correction = np.asarray(
        [
            [0.1, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, -0.1],
            [0.1, 0.1, 0.0],
        ]
    )
    observed = physical - 0.75 * correction

    assert projected_residual_scale(correction, physical, observed) == pytest.approx(
        -0.75
    )


def test_action_phase_profile_interpolates_smooths_and_clips() -> None:
    result = fit_action_phase_profile(
        (
            np.asarray([0.0, 2.0, 0.0]),
            np.asarray([0.0, 4.0, 0.0]),
        ),
        target_length=5,
        smoothing_window=1,
        maximum_absolute_scale=1.5,
    )

    assert np.array_equal(
        result,
        np.asarray([0.0, 1.5, 1.5, 1.5, 0.0]),
    )
    assert not result.flags.writeable


def test_zero_correction_is_exact_fallback() -> None:
    physical = np.asarray(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ],
        dtype=np.float64,
    )
    candidate = apply_action_phase_correction(
        physical,
        np.zeros((2, 3), dtype=np.float64),
        np.asarray([-1.0, 1.5]),
    )

    assert np.array_equal(candidate, physical)


def test_action_phase_scale_applies_per_frame() -> None:
    physical = np.zeros((2, 2, 3), dtype=np.float64)
    correction = np.ones((2, 3), dtype=np.float64)

    candidate = apply_action_phase_correction(
        physical,
        correction,
        np.asarray([-0.5, 1.25]),
    )

    assert np.array_equal(candidate[0], -0.5 * correction)
    assert np.array_equal(candidate[1], 1.25 * correction)


def test_translation_delta_is_anchored_to_each_training_prefix() -> None:
    correction_a = np.full((4, 3), [0.01, -0.02, 0.0])
    correction_b = np.full((4, 3), [-0.03, 0.01, 0.02])
    dynamic_delta = np.asarray(
        [
            [0.0, 0.01, 0.0],
            [0.02, -0.01, 0.0],
            [-0.01, 0.0, 0.03],
        ]
    )
    residual_a = correction_a[None] + dynamic_delta[:, None]
    residual_b = correction_b[None] + dynamic_delta[:, None]

    result = fit_action_phase_translation_delta(
        (residual_a, residual_b),
        (correction_a, correction_b),
        target_length=3,
        smoothing_window=1,
    )

    assert np.allclose(result, dynamic_delta)


def test_translation_delta_respects_total_node_cap() -> None:
    physical = np.zeros((2, 2, 3), dtype=np.float64)
    correction = np.full((2, 3), [0.08, 0.0, 0.0])
    delta = np.full((2, 3), [0.08, 0.0, 0.0])

    candidate = apply_action_phase_translation_delta(
        physical,
        correction,
        delta,
        maximum_correction_m=0.10,
    )

    assert np.allclose(candidate[..., 0], 0.10)
    assert np.allclose(candidate[..., 1:], 0.0)
