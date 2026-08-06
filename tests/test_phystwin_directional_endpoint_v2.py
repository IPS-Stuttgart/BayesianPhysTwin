from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.phystwin_directional_endpoint import (
    robust_directional_endpoint,
)
from bayesian_phystwin.phystwin_directional_endpoint_v2 import (
    DirectionalEndpointConfigV2,
    DirectionalEndpointNumericalError,
    PHYSTWIN_DIRECTIONAL_ENDPOINT_VERSION,
    robust_directional_endpoint_v2,
)


def _tangent_projectors(point_count: int) -> np.ndarray:
    projector = np.diag([1.0, 1.0, 0.0])
    return np.repeat(projector[None], point_count, axis=0)


def _run_v1(
    source: np.ndarray,
    source_valid: np.ndarray,
    multiview: np.ndarray,
    multiview_valid: np.ndarray,
    priority: np.ndarray,
):
    return robust_directional_endpoint(
        source,
        source_valid,
        multiview,
        multiview_valid,
        _tangent_projectors(source.shape[1]),
        priority,
        end_frame=len(source),
        process_variance=0.0,
        observation_variance=1e-4,
        initial_variance=1e-3,
        inlier_prior=0.95,
        outlier_variance_multiplier=100.0,
    )


def _run_v2(
    source: np.ndarray,
    source_valid: np.ndarray,
    multiview: np.ndarray,
    multiview_valid: np.ndarray,
    priority: np.ndarray,
    *,
    projectors: np.ndarray | None = None,
    config: DirectionalEndpointConfigV2 | None = None,
    observation_variance: float = 1e-4,
    initial_variance: float = 1e-3,
):
    return robust_directional_endpoint_v2(
        source,
        source_valid,
        multiview,
        multiview_valid,
        (
            _tangent_projectors(source.shape[1])
            if projectors is None
            else projectors
        ),
        priority,
        end_frame=len(source),
        process_variance=0.0,
        observation_variance=observation_variance,
        initial_variance=initial_variance,
        inlier_prior=0.95,
        outlier_variance_multiplier=100.0,
        config=config,
    )


def test_v2_retains_legacy_mean_but_not_anti_conservative_isotropization() -> None:
    source = np.array([[[0.1, 0.0, 0.0]]])
    valid = np.ones((1, 1), dtype=bool)
    multiview = np.zeros_like(source)
    priority = np.array([False])

    legacy = _run_v1(
        source,
        valid,
        multiview,
        np.zeros_like(valid),
        priority,
    )
    prospective = _run_v2(
        source,
        valid,
        multiview,
        np.zeros_like(valid),
        priority,
    )

    np.testing.assert_allclose(prospective.mean, legacy.mean, atol=1e-14)
    np.testing.assert_allclose(
        prospective.final_inlier_probability,
        legacy.final_inlier_probability,
        atol=1e-14,
    )
    assert prospective.variance[0] > legacy.variance[0]
    assert prospective.variance[0] == pytest.approx(
        np.max(np.linalg.eigvalsh(prospective.covariance[0]))
    )
    assert legacy.variance[0] == pytest.approx(
        np.trace(prospective.covariance[0]) / 3.0
    )
    scalar_upper_bound = (
        prospective.variance[0] * np.eye(3) - prospective.covariance[0]
    )
    assert np.min(np.linalg.eigvalsh(scalar_upper_bound)) >= -1e-15


def test_v2_covariance_is_symmetric_positive_definite_after_mixed_updates() -> None:
    source = np.array(
        [
            [[0.0, 0.0, 0.02]],
            [[0.0, 0.0, 0.03]],
            [[0.0, 0.0, 0.04]],
        ]
    )
    multiview = np.array(
        [
            [[0.01, -0.02, 0.0]],
            [[0.02, -0.01, 0.0]],
            [[0.03, 0.00, 0.0]],
        ]
    )
    valid = np.ones((3, 1), dtype=bool)

    result = _run_v2(
        source,
        valid,
        multiview,
        valid,
        np.array([True]),
    )

    np.testing.assert_allclose(
        result.covariance,
        np.swapaxes(result.covariance, 1, 2),
        atol=0.0,
    )
    assert np.all(np.linalg.eigvalsh(result.covariance) > 0.0)
    np.testing.assert_array_equal(result.source_update_count, [3])
    np.testing.assert_array_equal(result.tangent_update_count, [3])
    assert result.maximum_innovation_condition_number[0] >= 1.0
    assert result.maximum_posterior_condition_number[0] >= 1.0


def test_v2_is_invariant_to_an_orthogonal_coordinate_change() -> None:
    source = np.array(
        [
            [[0.01, -0.02, 0.03]],
            [[0.02, -0.01, 0.04]],
        ]
    )
    multiview = np.array(
        [
            [[0.03, -0.01, 0.0]],
            [[0.04, 0.01, 0.0]],
        ]
    )
    valid = np.ones((2, 1), dtype=bool)
    priority = np.array([True])
    projectors = _tangent_projectors(1)
    axis = np.asarray([1.0, 2.0, 3.0])
    axis /= np.linalg.norm(axis)
    angle = 0.7
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rotation = (
        np.cos(angle) * np.eye(3)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )

    reference = _run_v2(
        source,
        valid,
        multiview,
        valid,
        priority,
        projectors=projectors,
    )
    rotated = _run_v2(
        source @ rotation.T,
        valid,
        multiview @ rotation.T,
        valid,
        priority,
        projectors=np.einsum(
            "ij,njk,lk->nil",
            rotation,
            projectors,
            rotation,
        ),
    )
    restored_mean = rotated.mean @ rotation
    restored_covariance = np.einsum(
        "ij,njk,kl->nil",
        rotation.T,
        rotated.covariance,
        rotation,
    )

    np.testing.assert_allclose(restored_mean, reference.mean, atol=2e-14)
    np.testing.assert_allclose(
        restored_covariance,
        reference.covariance,
        atol=2e-14,
    )
    np.testing.assert_allclose(rotated.variance, reference.variance, atol=2e-14)


def test_v2_rejects_a_nearly_idempotent_projector_outside_absolute_tolerance() -> None:
    source = np.zeros((1, 1, 3), dtype=np.float64)
    valid = np.ones((1, 1), dtype=bool)
    near_projector = np.diag([1.0 + 5e-6, 1.0, 0.0])[None]

    with pytest.raises(ValueError, match="idempotent"):
        _run_v2(
            source,
            valid,
            np.zeros_like(source),
            np.zeros_like(valid),
            np.array([True]),
            projectors=near_projector,
        )


def test_v2_fails_closed_when_the_posterior_exceeds_condition_limit() -> None:
    source = np.array([[[0.0, 0.0, 0.01]]])
    multiview = np.zeros_like(source)
    valid = np.ones((1, 1), dtype=bool)

    with pytest.raises(DirectionalEndpointNumericalError, match="SPD admission"):
        _run_v2(
            source,
            valid,
            multiview,
            np.zeros_like(valid),
            np.array([True]),
            config=DirectionalEndpointConfigV2(
                maximum_condition_number=1e8,
            ),
            observation_variance=1e-16,
            initial_variance=1.0,
        )


def test_v2_results_are_immutable_and_report_numerical_semantics() -> None:
    source = np.zeros((1, 1, 3))
    valid = np.ones((1, 1), dtype=bool)
    result = _run_v2(
        source,
        valid,
        np.zeros_like(source),
        np.zeros_like(valid),
        np.array([False]),
    )

    for array in (
        result.mean,
        result.covariance,
        result.variance,
        result.final_inlier_probability,
        result.update_count,
        result.source_update_count,
        result.tangent_update_count,
        result.maximum_innovation_condition_number,
        result.maximum_posterior_condition_number,
    ):
        assert not array.flags.writeable
    with pytest.raises(ValueError):
        result.mean[0, 0] = 1.0

    diagnostics = result.diagnostics()
    assert diagnostics["schema_version"] == PHYSTWIN_DIRECTIONAL_ENDPOINT_VERSION
    assert diagnostics["component_covariance_update"] == "joseph-form"
    assert diagnostics["mixture_covariance_update"] == "exact-moment-matching"
    assert diagnostics["full_source_covariance_retained"] is True
    assert diagnostics["trace_average_isotropization"] is False
    assert diagnostics["implicit_jitter"] is False
    assert diagnostics["eigenvalue_clipping"] is False
    assert diagnostics["pseudoinverse_fallback"] is False
