from __future__ import annotations

import inspect

import numpy as np

from bayesian_phystwin.deform360_raw_camera_cycle_uncertainty import (
    build_raw_camera_cycle_uncertainty_case,
    inflate_covariance_from_cycle,
)


def test_cycle_error_inflates_only_jacobian_term() -> None:
    jacobian = np.eye(3) * 1.0e-6
    jackknife = np.eye(3) * 2.0e-6

    combined, diagnostic = inflate_covariance_from_cycle(
        jacobian,
        jackknife,
        1.0,
        np.array([2.0, 2.0, 2.0]),
        pixel_noise_floor_px=0.5,
    )

    assert diagnostic["effective_pixel_sigma"] > 1.0
    np.testing.assert_allclose(
        combined,
        jacobian * diagnostic["jacobian_covariance_scale"] + jackknife,
    )


def test_cycle_floor_never_shrinks_covariance() -> None:
    jacobian = np.eye(3) * 1.0e-6
    jackknife = np.eye(3) * 2.0e-6

    combined, diagnostic = inflate_covariance_from_cycle(
        jacobian,
        jackknife,
        1.0,
        np.array([0.1, 0.2]),
        pixel_noise_floor_px=0.5,
    )

    assert diagnostic["jacobian_covariance_scale"] == 1.0
    np.testing.assert_allclose(combined, jacobian + jackknife)


def test_cycle_builder_has_no_target_or_outcome_argument() -> None:
    parameters = inspect.signature(build_raw_camera_cycle_uncertainty_case).parameters

    assert "target" not in parameters
    assert "outcome" not in parameters
    assert set(parameters) == {
        "panel_case_dir",
        "processed_episode_dir",
        "measurement_dir",
        "uncertainty_dir",
        "output_dir",
        "runtime",
        "config",
    }
