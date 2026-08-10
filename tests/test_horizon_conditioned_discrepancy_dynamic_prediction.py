from __future__ import annotations

import numpy as np
import pytest

import bayesian_phystwin._dynamic_endpoint_prediction as prediction_module
from bayesian_phystwin.dynamic_endpoint_model_average import (
    DampedTrendEndpointComponentV2,
    DynamicEndpointModelAverageConfigV2,
    DynamicEndpointPosteriorV2,
    DynamicEndpointPredictionV2,
    PersistenceEndpointComponentV2,
    infer_dynamic_endpoint_model_average,
    predict_dynamic_endpoint_model_average,
)


def _single(
    component: object,
    *,
    pooling: str = "per_track",
) -> DynamicEndpointModelAverageConfigV2:
    return DynamicEndpointModelAverageConfigV2(
        components=(component,),  # type: ignore[arg-type]
        evidence_pooling=pooling,  # type: ignore[arg-type]
    )


def _posterior_kwargs(
    *,
    config: DynamicEndpointModelAverageConfigV2 | None = None,
) -> dict[str, object]:
    settings = config or _single(PersistenceEndpointComponentV2())
    component_count = len(settings.components)
    return {
        "mean_m": np.zeros((1, 3)),
        "covariance_m2": np.eye(3)[None],
        "final_nominal_probability": np.array([0.5]),
        "update_count": np.array([1], dtype=np.int64),
        "component_weights": np.full((1, component_count), 1.0 / component_count),
        "component_log_evidence": np.zeros((1, component_count)),
        "component_state_mean": np.zeros((component_count, 1, 2, 3)),
        "component_state_covariance": np.repeat(
            np.eye(2)[None, None, :, :],
            component_count,
            axis=0,
        ),
        "config": settings,
        "end_frame": 1,
    }


def _prediction_kwargs() -> dict[str, object]:
    return {
        "mean_m": np.zeros((1, 3)),
        "covariance_m2": np.eye(3)[None],
        "component_weights": np.ones((1, 1)),
        "component_mean_m": np.zeros((1, 1, 3)),
        "component_variance_m2": np.ones((1, 1)),
        "component_velocity_mean_m_per_step": np.zeros((1, 1, 3)),
        "horizon_steps": 0,
    }


def test_persistence_sample_and_hold_mean_survives_forecast() -> None:
    component = PersistenceEndpointComponentV2(process_std_m=0.002)
    posterior = infer_dynamic_endpoint_model_average(
        np.array([[[0.01, -0.02, 0.03]]]),
        np.ones((1, 1), dtype=bool),
        end_frame=1,
        config=_single(component),
    )
    future = predict_dynamic_endpoint_model_average(posterior, horizon_steps=25)

    assert np.array_equal(future.mean_m, posterior.mean_m)
    assert np.trace(future.covariance_m2[0]) > np.trace(posterior.covariance_m2[0])


def test_damped_trend_learns_motion_and_forecasts_horizon_mean() -> None:
    residual = np.zeros((8, 1, 3))
    residual[:, 0, 0] = np.arange(8) * 0.01
    component = DampedTrendEndpointComponentV2(
        velocity_retention=0.9,
        level_process_std_m=0.0001,
        velocity_process_std_m_per_step=0.0001,
        observation_std_m=0.0005,
        initial_velocity_std_m_per_step=0.02,
    )
    posterior = infer_dynamic_endpoint_model_average(
        residual,
        np.ones((8, 1), dtype=bool),
        end_frame=8,
        config=_single(component),
    )
    one = predict_dynamic_endpoint_model_average(posterior, horizon_steps=1)
    ten = predict_dynamic_endpoint_model_average(posterior, horizon_steps=10)

    velocity_now = posterior.component_state_mean[0, 0, 1, 0]
    assert velocity_now > 0.0
    assert one.mean_m[0, 0] > posterior.mean_m[0, 0]
    assert ten.mean_m[0, 0] > one.mean_m[0, 0]
    assert ten.component_velocity_mean_m_per_step[0, 0, 0] < velocity_now
    assert np.isclose(
        ten.component_velocity_mean_m_per_step[0, 0, 0],
        component.velocity_retention**10 * velocity_now,
    )


def test_zero_retention_trend_uses_current_velocity_once() -> None:
    component = DampedTrendEndpointComponentV2(velocity_retention=0.0)
    config = _single(component)
    kwargs = _posterior_kwargs(config=config)
    kwargs["component_state_mean"] = np.array([[[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]])
    posterior = DynamicEndpointPosteriorV2(**kwargs)

    one = predict_dynamic_endpoint_model_average(posterior, horizon_steps=1)
    two = predict_dynamic_endpoint_model_average(posterior, horizon_steps=2)
    assert np.isclose(one.mean_m[0, 0], 3.0)
    assert np.isclose(two.mean_m[0, 0], 3.0)
    assert np.isclose(two.component_velocity_mean_m_per_step[0, 0, 0], 0.0)


def test_horizon_zero_reproduces_posterior_moments() -> None:
    residual = np.array([[[0.0, 0.0, 0.0]], [[0.01, 0.0, 0.0]]])
    posterior = infer_dynamic_endpoint_model_average(
        residual,
        np.ones((2, 1), dtype=bool),
        end_frame=2,
    )
    prediction = predict_dynamic_endpoint_model_average(posterior, horizon_steps=0)

    assert np.allclose(prediction.mean_m, posterior.mean_m)
    assert np.allclose(prediction.covariance_m2, posterior.covariance_m2)
    assert np.array_equal(prediction.component_weights, posterior.component_weights)


def test_transition_power_matches_repeated_propagation() -> None:
    transition = np.array([[1.0, 1.0], [0.0, 0.87]])
    process = np.diag([0.2, 0.05])
    for horizon in (0, 1, 2, 3, 9, 32):
        powered_transition, powered_process = prediction_module._transition_power(
            transition,
            process,
            horizon,
        )
        expected_transition = np.eye(2)
        expected_process = np.zeros((2, 2))
        for _ in range(horizon):
            expected_process = transition @ expected_process @ transition.T + process
            expected_transition = transition @ expected_transition
        assert np.allclose(powered_transition, expected_transition)
        assert np.allclose(powered_process, expected_process)


def test_large_horizon_is_finite_and_logarithmic_path() -> None:
    posterior = infer_dynamic_endpoint_model_average(
        np.zeros((1, 1, 3)),
        np.ones((1, 1), dtype=bool),
        end_frame=1,
        config=_single(
            DampedTrendEndpointComponentV2(
                velocity_retention=0.95,
                level_process_std_m=1e-6,
                velocity_process_std_m_per_step=1e-6,
            )
        ),
    )
    prediction = predict_dynamic_endpoint_model_average(
        posterior,
        horizon_steps=1_000_000,
    )
    assert np.all(np.isfinite(prediction.mean_m))
    assert np.all(np.isfinite(prediction.covariance_m2))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("horizon_steps", True, "nonnegative integer"),
        ("horizon_steps", -1, "nonnegative integer"),
        ("mean_m", np.zeros((0, 3)), "mean_m"),
        ("covariance_m2", np.zeros((2, 3, 3)), "shape"),
        ("covariance_m2", -np.eye(3)[None], "positive semidefinite"),
        ("component_weights", np.ones((2, 1)), "shape"),
        ("component_mean_m", np.zeros((1, 2, 3)), "mean/velocity"),
        ("component_velocity_mean_m_per_step", np.zeros((1, 2, 3)), "mean/velocity"),
        ("component_variance_m2", np.zeros((1, 2)), "variance"),
        ("component_variance_m2", np.array([[-1.0]]), "nonnegative"),
        ("component_mean_m", np.full((1, 1, 3), np.nan), "non-finite"),
        ("component_weights", np.array([[0.5]]), "row-normalized"),
    ],
)
def test_prediction_contract_rejects_invalid_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    kwargs = _prediction_kwargs()
    kwargs[field] = value
    with pytest.raises(ValueError, match=message):
        DynamicEndpointPredictionV2(**kwargs)  # type: ignore[arg-type]
