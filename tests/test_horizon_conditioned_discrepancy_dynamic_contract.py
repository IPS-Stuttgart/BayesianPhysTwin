from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import bayesian_phystwin._dynamic_endpoint_contract as contract_module
from bayesian_phystwin.dynamic_endpoint_model_average import (
    DEFAULT_DYNAMIC_ENDPOINT_MODEL_AVERAGE_CONFIG_V2,
    DampedTrendEndpointComponentV2,
    DynamicEndpointModelAverageConfigV2,
    DynamicEndpointPosteriorV2,
    PersistenceEndpointComponentV2,
    component_kind,
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


def test_default_family_and_component_labels_are_stable() -> None:
    config = DEFAULT_DYNAMIC_ENDPOINT_MODEL_AVERAGE_CONFIG_V2
    assert len(config.components) == 7
    assert config.component_kinds == (
        "persistence",
        "local-level",
        "local-level",
        "local-level",
        "damped-trend",
        "damped-trend",
        "damped-trend",
    )
    assert np.isclose(sum(config.component_prior_probability), 1.0)
    assert np.isclose(config.component_prior_probability[0], 1.0 / 3.0)
    assert np.allclose(config.component_prior_probability[1:], 1.0 / 9.0)
    assert component_kind(config.components[0]) == "persistence"
    assert component_kind(config.components[1]) == "local-level"
    assert component_kind(config.components[-1]) == "damped-trend"
    with pytest.raises(TypeError, match="unsupported"):
        component_kind(object())  # type: ignore[arg-type]


def test_component_validation_failures() -> None:
    invalid_persistence = [
        {"process_std_m": -1.0},
        {"observation_std_m": 0.0},
        {"initial_std_m": 0.0},
        {"inlier_prior": 0.0},
        {"inlier_prior": 1.0},
        {"outlier_variance_multiplier": 1.0},
        {"process_std_m": True},
    ]
    for values in invalid_persistence:
        with pytest.raises(ValueError):
            PersistenceEndpointComponentV2(**values)

    invalid_trend = [
        {"velocity_retention": -0.1},
        {"velocity_retention": 1.1},
        {"level_process_std_m": -1.0},
        {"velocity_process_std_m_per_step": -1.0},
        {"initial_level_std_m": 0.0},
        {"initial_velocity_std_m_per_step": 0.0},
        {"observation_std_m": np.inf},
    ]
    for values in invalid_trend:
        with pytest.raises(ValueError):
            DampedTrendEndpointComponentV2(**values)


def test_config_validation_failures() -> None:
    with pytest.raises(ValueError, match="at least one"):
        DynamicEndpointModelAverageConfigV2(components=())
    with pytest.raises(TypeError, match="unsupported"):
        DynamicEndpointModelAverageConfigV2(
            components=(object(),)  # type: ignore[arg-type]
        )
    component = PersistenceEndpointComponentV2()
    with pytest.raises(ValueError, match="unique"):
        DynamicEndpointModelAverageConfigV2(components=(component, component))
    with pytest.raises(ValueError, match="component count"):
        DynamicEndpointModelAverageConfigV2(
            components=(component,),
            component_prior_probability=(0.5, 0.5),
        )
    for prior in ((0.0,), (np.nan,)):
        with pytest.raises(ValueError, match="finite and positive"):
            DynamicEndpointModelAverageConfigV2(
                components=(component,),
                component_prior_probability=prior,
            )
    with pytest.raises(TypeError, match="balance_component_families"):
        DynamicEndpointModelAverageConfigV2(
            components=(component,),
            balance_component_families=1,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="evidence_pooling"):
        DynamicEndpointModelAverageConfigV2(
            components=(component,),
            evidence_pooling="track",  # type: ignore[arg-type]
        )


def test_component_uniform_prior_can_be_requested_explicitly() -> None:
    config = DynamicEndpointModelAverageConfigV2(
        components=(
            PersistenceEndpointComponentV2(),
            DampedTrendEndpointComponentV2(),
            DampedTrendEndpointComponentV2(velocity_retention=0.8),
        ),
        balance_component_families=False,
    )
    assert np.allclose(config.component_prior_probability, 1.0 / 3.0)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("config", object(), "config"),
        ("end_frame", 0, "positive integer"),
        ("end_frame", True, "positive integer"),
        ("mean_m", np.zeros((0, 3)), "mean_m"),
        ("covariance_m2", np.zeros((2, 3, 3)), "shape"),
        ("covariance_m2", np.full((1, 3, 3), np.nan), "finite"),
        (
            "covariance_m2",
            np.array([[[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]),
            "symmetric",
        ),
        ("covariance_m2", -np.eye(3)[None], "positive semidefinite"),
        ("final_nominal_probability", np.zeros(2), "probability shape"),
        ("final_nominal_probability", np.array([2.0]), r"\[0, 1\]"),
        ("update_count", np.array([1.0]), "integers"),
        ("update_count", np.array([-1]), "nonnegative"),
        ("component_weights", np.ones((1, 2)), "weight/evidence"),
        ("component_log_evidence", np.ones((1, 2)), "weight/evidence"),
        ("component_state_mean", np.zeros((1, 1, 1, 3)), "state_mean"),
        ("component_state_covariance", np.zeros((1, 1, 3, 3)), "state_covariance"),
        ("component_state_covariance", -np.eye(2)[None, None], "positive semidefinite"),
        ("component_log_evidence", np.array([[np.nan]]), "non-finite"),
        ("component_weights", np.array([[-1.0]]), "row-normalized"),
        ("component_weights", np.array([[0.5]]), "row-normalized"),
    ],
)
def test_posterior_contract_rejects_invalid_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    kwargs = _posterior_kwargs()
    kwargs[field] = value
    with pytest.raises((TypeError, ValueError), match=message):
        DynamicEndpointPosteriorV2(**kwargs)  # type: ignore[arg-type]


def test_config_replace_keeps_immutable_semantics() -> None:
    config = DynamicEndpointModelAverageConfigV2(
        components=(PersistenceEndpointComponentV2(),),
    )
    changed = replace(config, evidence_pooling="object")
    assert config.evidence_pooling == "per_track"
    assert changed.evidence_pooling == "object"


def test_covariance_helper_rejects_wrong_rank() -> None:
    with pytest.raises(ValueError, match="invalid covariance shape"):
        contract_module._validate_covariance(
            np.zeros(1),
            name="probe",
            final_shape=(2, 2),
        )
