from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.dynamic_endpoint_model_average import (
    FULL_COVARIANCE_DYNAMIC_ENDPOINT_CONTRACT_VERSION,
    DynamicEndpointComponentV2,
    DynamicEndpointModelAverageConfigV2,
    FullCovarianceDynamicEndpointPosteriorV3,
    FullCovarianceDynamicEndpointPredictionV3,
    PersistenceEndpointComponentV2,
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


def test_contract_version_and_public_types_are_exposed() -> None:
    assert FULL_COVARIANCE_DYNAMIC_ENDPOINT_CONTRACT_VERSION == 3
    assert FullCovarianceDynamicEndpointPosteriorV3.__name__.endswith("V3")
    assert FullCovarianceDynamicEndpointPredictionV3.__name__.endswith("V3")


def _valid_posterior_fields() -> dict[str, object]:
    config = _single(PersistenceEndpointComponentV2())
    return {
        "mean_m": np.zeros((1, 3)),
        "covariance_m2": np.eye(3)[None],
        "final_nominal_probability": np.array([0.5]),
        "update_count": np.array([1], dtype=np.int64),
        "component_weights": np.ones((1, 1)),
        "component_log_evidence": np.zeros((1, 1)),
        "component_state_mean": np.zeros((1, 1, 2, 3)),
        "component_state_covariance_m2": np.eye(6)[None, None],
        "config": config,
        "end_frame": 1,
    }


def _valid_prediction_fields() -> dict[str, object]:
    return {
        "mean_m": np.zeros((1, 3)),
        "covariance_m2": np.eye(3)[None],
        "component_weights": np.ones((1, 1)),
        "component_mean_m": np.zeros((1, 1, 3)),
        "component_covariance_m2": np.eye(3)[None, None],
        "component_velocity_mean_m_per_step": np.zeros((1, 1, 3)),
        "horizon_steps": 0,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("config", object(), "config"),
        ("end_frame", True, "positive integer"),
        ("end_frame", 0, "positive integer"),
        ("mean_m", np.zeros(3), "mean_m"),
        ("covariance_m2", np.zeros((1, 2, 2)), "covariance_m2"),
        ("final_nominal_probability", np.zeros(2), "probability shape"),
        ("update_count", np.array([1.0]), "contain integers"),
        ("update_count", np.array([-1]), "nonnegative track vector"),
        ("component_weights", np.ones((1, 2)), "weight/evidence shape"),
        ("component_state_mean", np.zeros((1, 1, 3, 2)), "state_mean"),
        (
            "component_state_covariance_m2",
            np.zeros((1, 1, 5, 5)),
            "state_covariance",
        ),
        ("mean_m", np.full((1, 3), np.nan), "non-finite"),
        ("final_nominal_probability", np.array([1.1]), "lie in"),
        ("component_weights", np.array([[-0.1]]), "row-normalized"),
    ],
)
def test_posterior_contract_rejects_malformed_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    fields = _valid_posterior_fields()
    fields[field] = value
    with pytest.raises((TypeError, ValueError), match=message):
        FullCovarianceDynamicEndpointPosteriorV3(**fields)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mean_m", np.zeros(3), "mean_m"),
        ("covariance_m2", np.zeros((1, 2, 2)), "covariance_m2"),
        ("component_weights", np.ones(1), "component_weights"),
        ("component_mean_m", np.zeros((1, 2, 3)), "mean/velocity"),
        ("component_covariance_m2", np.zeros((1, 1, 2, 2)), "covariance"),
        ("component_mean_m", np.full((1, 1, 3), np.nan), "non-finite"),
        ("component_weights", np.array([[0.5]]), "row-normalized"),
    ],
)
def test_prediction_contract_rejects_malformed_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    fields = _valid_prediction_fields()
    fields[field] = value
    with pytest.raises(ValueError, match=message):
        FullCovarianceDynamicEndpointPredictionV3(**fields)  # type: ignore[arg-type]
