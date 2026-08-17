from __future__ import annotations

import numpy as np

from bayesian_phystwin.contracts.fixed_anchor import FixedBayesianAnchorConfigV1
from bayesian_phystwin.dynamic_endpoint_model_average import (
    DampedTrendEndpointComponentV2,
    DynamicEndpointComponentV2,
    DynamicEndpointModelAverageConfigV2,
    PersistenceEndpointComponentV2,
    infer_dynamic_endpoint_model_average,
    infer_full_covariance_dynamic_endpoint_model_average,
)


def _single(
    component: DynamicEndpointComponentV2,
) -> DynamicEndpointModelAverageConfigV2:
    return DynamicEndpointModelAverageConfigV2(components=(component,))


def _metric_covariance(
    frame_count: int,
    track_count: int,
    covariance: np.ndarray,
) -> np.ndarray:
    return np.broadcast_to(
        covariance,
        (frame_count, track_count, 3, 3),
    ).copy()


def test_persistence_preserves_anisotropy_and_cross_axis_covariance() -> None:
    residual = np.array([[[0.02, -0.01, 0.03]]], dtype=np.float64)
    valid = np.ones((1, 1), dtype=np.bool_)
    metric = np.array(
        [
            [
                [
                    [1.0e-5, 2.0e-6, -1.0e-6],
                    [2.0e-6, 4.0e-5, 3.0e-6],
                    [-1.0e-6, 3.0e-6, 9.0e-5],
                ]
            ]
        ],
        dtype=np.float64,
    )
    config = _single(
        PersistenceEndpointComponentV2(
            process_std_m=0.0,
            observation_std_m=0.001,
        )
    )

    posterior = infer_full_covariance_dynamic_endpoint_model_average(
        residual,
        valid,
        metric,
        end_frame=1,
        config=config,
    )

    np.testing.assert_array_equal(posterior.mean_m, residual[-1])
    covariance = posterior.covariance_m2[0]
    np.testing.assert_allclose(covariance[0, 1], metric[0, 0, 0, 1])
    np.testing.assert_allclose(covariance[0, 2], metric[0, 0, 0, 2])
    np.testing.assert_allclose(
        covariance[2, 2] - covariance[0, 0],
        metric[0, 0, 2, 2] - metric[0, 0, 0, 0],
    )
    assert not np.allclose(np.diag(covariance), covariance[0, 0])


def test_filter_is_rotation_equivariant() -> None:
    residual = np.array(
        [
            [[0.01, 0.02, -0.01], [-0.02, 0.00, 0.03]],
            [[0.02, 0.01, 0.00], [-0.01, 0.01, 0.02]],
            [[0.03, 0.00, 0.01], [0.00, 0.02, 0.01]],
        ],
        dtype=np.float64,
    )
    valid = np.ones((3, 2), dtype=np.bool_)
    base = np.array(
        [
            [4.0e-5, 1.0e-5, -0.5e-5],
            [1.0e-5, 8.0e-5, 0.4e-5],
            [-0.5e-5, 0.4e-5, 2.0e-5],
        ],
        dtype=np.float64,
    )
    metric = _metric_covariance(3, 2, base)
    angle = 0.63
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rotated_residual = residual @ rotation.T
    rotated_metric = np.einsum(
        "ab,tnbc,dc->tnad",
        rotation,
        metric,
        rotation,
    )

    original = infer_full_covariance_dynamic_endpoint_model_average(
        residual,
        valid,
        metric,
        end_frame=3,
    )
    rotated = infer_full_covariance_dynamic_endpoint_model_average(
        rotated_residual,
        valid,
        rotated_metric,
        end_frame=3,
    )

    np.testing.assert_allclose(
        rotated.mean_m,
        original.mean_m @ rotation.T,
        atol=2e-12,
        rtol=2e-12,
    )
    expected_covariance = np.einsum(
        "ab,nbc,dc->nad",
        rotation,
        original.covariance_m2,
        rotation,
    )
    np.testing.assert_allclose(
        rotated.covariance_m2,
        expected_covariance,
        atol=2e-12,
        rtol=2e-12,
    )
    np.testing.assert_allclose(
        rotated.component_weights,
        original.component_weights,
        atol=2e-12,
        rtol=2e-12,
    )


def test_isotropic_first_update_matches_v2_filter_statistics() -> None:
    residual = np.array(
        [[[0.01, 0.00, -0.01], [0.00, 0.02, 0.01]]],
        dtype=np.float64,
    )
    valid = np.ones((1, 2), dtype=np.bool_)
    metric_variance = np.array([[2.0e-5, 3.0e-5]], dtype=np.float64)
    metric_covariance = metric_variance[:, :, None, None] * np.eye(3)
    component = FixedBayesianAnchorConfigV1(
        process_std_m=0.001,
        observation_std_m=0.0025,
    )
    config = _single(component)

    scalar = infer_dynamic_endpoint_model_average(
        residual,
        valid,
        end_frame=1,
        config=config,
        observation_variance_m2=metric_variance,
    )
    full = infer_full_covariance_dynamic_endpoint_model_average(
        residual,
        valid,
        metric_covariance,
        end_frame=1,
        config=config,
    )

    np.testing.assert_allclose(full.mean_m, scalar.mean_m, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(
        full.final_nominal_probability,
        scalar.final_nominal_probability,
        atol=2e-14,
        rtol=2e-14,
    )
    np.testing.assert_allclose(
        full.component_log_evidence,
        scalar.component_log_evidence,
        atol=2e-14,
        rtol=2e-14,
    )
    np.testing.assert_allclose(full.component_weights, scalar.component_weights)
    np.testing.assert_array_equal(full.update_count, scalar.update_count)
    np.testing.assert_allclose(
        full.component_state_mean,
        scalar.component_state_mean,
        atol=2e-14,
        rtol=2e-14,
    )
    state_covariance = full.component_state_covariance_m2[0]
    averaged = np.empty_like(scalar.component_state_covariance[0])
    averaged[:, 0, 0] = np.trace(state_covariance[:, :3, :3], axis1=1, axis2=2) / 3
    averaged[:, 0, 1] = np.trace(state_covariance[:, :3, 3:], axis1=1, axis2=2) / 3
    averaged[:, 1, 0] = np.trace(state_covariance[:, 3:, :3], axis1=1, axis2=2) / 3
    averaged[:, 1, 1] = np.trace(state_covariance[:, 3:, 3:], axis1=1, axis2=2) / 3
    np.testing.assert_allclose(
        averaged,
        scalar.component_state_covariance[0],
        atol=2e-14,
        rtol=2e-14,
    )


def test_object_evidence_pooling_shares_component_weights() -> None:
    residual = np.zeros((4, 2, 3), dtype=np.float64)
    residual[:, 0, 0] = [0.0, 0.01, 0.02, 0.03]
    residual[:, 1, 1] = [0.02, -0.02, 0.02, -0.02]
    valid = np.ones((4, 2), dtype=np.bool_)
    covariance = _metric_covariance(4, 2, np.diag([1e-5, 2e-5, 3e-5]))
    config = DynamicEndpointModelAverageConfigV2(
        components=(
            PersistenceEndpointComponentV2(),
            DampedTrendEndpointComponentV2(observation_std_m=0.001),
        ),
        evidence_pooling="object",
    )

    posterior = infer_full_covariance_dynamic_endpoint_model_average(
        residual,
        valid,
        covariance,
        end_frame=4,
        config=config,
    )

    np.testing.assert_array_equal(
        posterior.component_weights[0],
        posterior.component_weights[1],
    )


def test_causal_cutoff_ignores_future_values_and_covariance() -> None:
    residual = np.array(
        [
            [[0.00, 0.00, 0.00]],
            [[0.01, 0.02, -0.01]],
            [[0.02, 0.04, -0.02]],
            [[9.00, 9.00, 9.00]],
        ],
        dtype=np.float64,
    )
    valid = np.ones((4, 1), dtype=np.bool_)
    covariance = _metric_covariance(4, 1, np.eye(3) * 1e-5)
    changed_residual = residual.copy()
    changed_residual[3] = -100.0
    changed_covariance = covariance.copy()
    changed_covariance[3] = np.eye(3) * 100.0

    first = infer_full_covariance_dynamic_endpoint_model_average(
        residual,
        valid,
        covariance,
        end_frame=3,
    )
    second = infer_full_covariance_dynamic_endpoint_model_average(
        changed_residual,
        valid,
        changed_covariance,
        end_frame=3,
    )

    np.testing.assert_array_equal(first.mean_m, second.mean_m)
    np.testing.assert_array_equal(first.covariance_m2, second.covariance_m2)
    np.testing.assert_array_equal(
        first.component_log_evidence,
        second.component_log_evidence,
    )


def test_no_observations_preserve_component_priors() -> None:
    config = DynamicEndpointModelAverageConfigV2(
        components=(
            PersistenceEndpointComponentV2(),
            FixedBayesianAnchorConfigV1(),
        ),
        component_prior_probability=(0.2, 0.8),
        evidence_pooling="object",
    )
    posterior = infer_full_covariance_dynamic_endpoint_model_average(
        np.zeros((3, 2, 3)),
        np.zeros((3, 2), dtype=np.bool_),
        _metric_covariance(3, 2, np.zeros((3, 3))),
        end_frame=3,
        config=config,
    )

    np.testing.assert_allclose(
        posterior.component_weights,
        [[0.2, 0.8], [0.2, 0.8]],
    )
    assert not np.any(posterior.updated_mask)
