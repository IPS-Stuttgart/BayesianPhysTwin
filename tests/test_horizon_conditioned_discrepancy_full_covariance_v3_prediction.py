from __future__ import annotations

import numpy as np
import pytest

import bayesian_phystwin._full_covariance_dynamic_endpoint_v3 as v3_module
from bayesian_phystwin.dynamic_endpoint_model_average import (
    DampedTrendEndpointComponentV2,
    DynamicEndpointComponentV2,
    DynamicEndpointModelAverageConfigV2,
    PersistenceEndpointComponentV2,
    infer_full_covariance_dynamic_endpoint_model_average,
    predict_full_covariance_dynamic_endpoint_model_average,
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


def test_prediction_retains_full_covariance_and_uses_logarithmic_power() -> None:
    residual = np.array(
        [
            [[0.00, 0.00, 0.00]],
            [[0.01, 0.02, -0.01]],
            [[0.02, 0.04, -0.02]],
        ],
        dtype=np.float64,
    )
    valid = np.ones((3, 1), dtype=np.bool_)
    metric = _metric_covariance(
        3,
        1,
        np.array(
            [
                [5e-5, 1e-5, 0.0],
                [1e-5, 8e-5, 2e-5],
                [0.0, 2e-5, 6e-5],
            ]
        ),
    )
    config = _single(
        DampedTrendEndpointComponentV2(
            velocity_retention=0.8,
            level_process_std_m=1e-5,
            velocity_process_std_m_per_step=1e-5,
        )
    )
    posterior = infer_full_covariance_dynamic_endpoint_model_average(
        residual,
        valid,
        metric,
        end_frame=3,
        config=config,
    )
    now = predict_full_covariance_dynamic_endpoint_model_average(
        posterior,
        horizon_steps=0,
    )
    future = predict_full_covariance_dynamic_endpoint_model_average(
        posterior,
        horizon_steps=1_000_000,
    )

    np.testing.assert_allclose(now.mean_m, posterior.mean_m)
    np.testing.assert_allclose(now.covariance_m2, posterior.covariance_m2)
    assert np.all(np.isfinite(future.mean_m))
    assert np.all(np.isfinite(future.covariance_m2))
    assert np.min(np.linalg.eigvalsh(future.covariance_m2)) >= -1e-12
    assert abs(now.covariance_m2[0, 0, 1]) > 0.0


def test_transition_power_matches_iterated_propagation() -> None:
    component = DampedTrendEndpointComponentV2(
        velocity_retention=0.87,
        level_process_std_m=0.001,
        velocity_process_std_m_per_step=0.0002,
    )
    transition, process, *_ = v3_module._expanded_component_matrices(component)
    for horizon in (0, 1, 2, 5, 17):
        powered_transition, powered_process = v3_module._transition_power(
            transition,
            process,
            horizon,
        )
        expected_transition = np.eye(6)
        expected_process = np.zeros((6, 6))
        for _ in range(horizon):
            expected_process = transition @ expected_process @ transition.T + process
            expected_transition = transition @ expected_transition
        np.testing.assert_allclose(powered_transition, expected_transition)
        np.testing.assert_allclose(powered_process, expected_process)


def test_inputs_are_not_mutated_and_outputs_are_read_only() -> None:
    residual = np.array([[[0.01, 0.02, 0.03]]], dtype=np.float64)
    valid = np.ones((1, 1), dtype=np.bool_)
    covariance = _metric_covariance(1, 1, np.eye(3) * 1e-5)
    residual_before = residual.copy()
    covariance_before = covariance.copy()

    posterior = infer_full_covariance_dynamic_endpoint_model_average(
        residual,
        valid,
        covariance,
        end_frame=1,
        config=_single(PersistenceEndpointComponentV2()),
    )
    prediction = predict_full_covariance_dynamic_endpoint_model_average(
        posterior,
        horizon_steps=2,
    )

    np.testing.assert_array_equal(residual, residual_before)
    np.testing.assert_array_equal(covariance, covariance_before)
    for value in (
        posterior.mean_m,
        posterior.covariance_m2,
        posterior.component_state_covariance_m2,
        prediction.mean_m,
        prediction.component_covariance_m2,
    ):
        assert not value.flags.writeable
    with pytest.raises(ValueError):
        posterior.mean_m[0, 0] = 0.0
