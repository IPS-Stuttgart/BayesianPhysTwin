import numpy as np
import pytest

import bayesian_phystwin.endpoint_model_average as endpoint_module
from bayesian_phystwin.causal4d_belief_provider_v2 import (
    CAUSAL4D_BELIEF_PROVIDER_V2_API_VERSION,
    causal4d_belief_provider_v2_manifest,
    infer_model_averaged_bayesian_anchor_endpoint,
)
from bayesian_phystwin.contracts.fixed_anchor import FixedBayesianAnchorConfigV1
from bayesian_phystwin.endpoint_model_average import (
    ModelAveragedEndpointConfigV1,
    ModelAveragedEndpointPosteriorV1,
    ModelAveragedEndpointPredictionV1,
    TemperedModelAveragedEndpointConfigV2,
    infer_model_averaged_endpoint,
    infer_tempered_model_averaged_endpoint,
    predict_model_averaged_endpoint,
    predict_tempered_model_averaged_endpoint,
)
from bayesian_phystwin.phystwin_bayesian_anchor import robust_random_walk_endpoint


def _single_config(
    component: FixedBayesianAnchorConfigV1,
) -> ModelAveragedEndpointConfigV1:
    return ModelAveragedEndpointConfigV1(components=(component,))


def test_single_component_matches_fixed_endpoint_exactly() -> None:
    residual = np.array(
        [
            [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0]],
            [[0.012, 0.0, 0.0], [0.0, 0.008, 0.0]],
            [[0.011, 0.0, 0.0], [0.0, 0.009, 0.0]],
        ]
    )
    valid = np.array([[True, True], [True, False], [True, True]])
    component = FixedBayesianAnchorConfigV1(
        process_std_m=0.001,
        observation_std_m=0.0025,
    )
    posterior = infer_model_averaged_endpoint(
        residual,
        valid,
        end_frame=3,
        config=_single_config(component),
    )
    reference = robust_random_walk_endpoint(
        residual,
        valid,
        end_frame=3,
        process_variance=component.process_std_m**2,
        observation_variance=component.observation_std_m**2,
        initial_variance=component.initial_std_m**2,
        inlier_prior=component.inlier_prior,
        outlier_variance_multiplier=component.outlier_variance_multiplier,
    )

    assert np.array_equal(posterior.update_count, reference.update_count)
    assert np.allclose(posterior.mean_m, reference.mean)
    assert np.allclose(
        posterior.component_variance_m2[0],
        reference.variance,
    )
    assert np.allclose(
        posterior.final_nominal_probability,
        reference.final_inlier_probability,
    )
    assert np.allclose(posterior.component_weights, 1.0)
    expected = reference.variance[:, None, None] * np.eye(3)
    assert np.allclose(posterior.covariance_m2, expected)


def test_between_model_disagreement_increases_covariance() -> None:
    residual = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[0.01, 0.0, 0.0]],
            [[0.02, 0.0, 0.0]],
        ]
    )
    valid = np.ones((3, 1), dtype=bool)
    config = ModelAveragedEndpointConfigV1(
        components=(
            FixedBayesianAnchorConfigV1(
                process_std_m=0.0,
                observation_std_m=0.001,
            ),
            FixedBayesianAnchorConfigV1(
                process_std_m=0.01,
                observation_std_m=0.01,
            ),
        )
    )
    posterior = infer_model_averaged_endpoint(
        residual,
        valid,
        end_frame=3,
        config=config,
    )

    weighted_within_x = float(
        np.sum(posterior.component_weights[0] * posterior.component_variance_m2[:, 0])
    )
    assert posterior.covariance_m2[0, 0, 0] >= weighted_within_x
    assert posterior.covariance_m2[0, 0, 0] > posterior.covariance_m2[0, 1, 1]


def test_no_observations_preserve_model_prior_weights() -> None:
    config = ModelAveragedEndpointConfigV1(
        components=(
            FixedBayesianAnchorConfigV1(process_std_m=0.0),
            FixedBayesianAnchorConfigV1(process_std_m=0.001),
        ),
        component_prior_probability=(0.25, 0.75),
    )
    posterior = infer_model_averaged_endpoint(
        np.zeros((4, 2, 3)),
        np.zeros((4, 2), dtype=bool),
        end_frame=4,
        config=config,
    )

    assert np.allclose(posterior.component_weights, [[0.25, 0.75]] * 2)
    assert np.allclose(posterior.mean_m, 0.0)
    assert not np.any(posterior.updated_mask)


def test_horizon_prediction_increases_covariance_trace() -> None:
    config = ModelAveragedEndpointConfigV1(
        components=(
            FixedBayesianAnchorConfigV1(process_std_m=0.001),
            FixedBayesianAnchorConfigV1(process_std_m=0.005),
        )
    )
    posterior = infer_model_averaged_endpoint(
        np.zeros((2, 1, 3)),
        np.ones((2, 1), dtype=bool),
        end_frame=2,
        config=config,
    )
    now = predict_model_averaged_endpoint(posterior, horizon_steps=0)
    future = predict_model_averaged_endpoint(posterior, horizon_steps=20)

    assert np.allclose(now.mean_m, future.mean_m)
    assert np.trace(future.covariance_m2[0]) > np.trace(now.covariance_m2[0])


def test_result_arrays_are_read_only() -> None:
    posterior = infer_model_averaged_endpoint(
        np.zeros((1, 1, 3)),
        np.ones((1, 1), dtype=bool),
        end_frame=1,
        config=_single_config(FixedBayesianAnchorConfigV1()),
    )
    assert not posterior.mean_m.flags.writeable
    assert not posterior.covariance_m2.flags.writeable
    assert not posterior.component_weights.flags.writeable
    with pytest.raises(ValueError):
        posterior.mean_m[0, 0] = 1.0


def test_default_configuration_uses_historical_noise_grid() -> None:
    config = ModelAveragedEndpointConfigV1()
    assert len(config.components) == 15
    assert np.isclose(sum(config.component_prior_probability), 1.0)


def test_tempered_large_cap_matches_historical_mixture() -> None:
    residual = np.array(
        [
            [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]],
            [[0.004, 0.0, 0.0], [0.012, 0.0, 0.0]],
            [[0.009, 0.0, 0.0], [0.011, 0.0, 0.0]],
        ]
    )
    valid = np.ones((3, 2), dtype=bool)
    base = infer_model_averaged_endpoint(residual, valid, end_frame=3)
    tempered = infer_tempered_model_averaged_endpoint(
        residual,
        valid,
        end_frame=3,
        config=TemperedModelAveragedEndpointConfigV2(
            effective_evidence_count_cap=100.0
        ),
    )

    assert np.array_equal(tempered.evidence_power, np.ones(2))
    assert np.allclose(tempered.component_weights, base.component_weights)
    assert np.allclose(tempered.mean_m, base.mean_m)
    assert np.allclose(tempered.covariance_m2, base.covariance_m2)


def test_tempered_evidence_power_tracks_effective_observation_count() -> None:
    residual = np.zeros((5, 3, 3))
    valid = np.array(
        [
            [True, True, False],
            [True, True, False],
            [True, False, False],
            [True, False, False],
            [True, False, False],
        ]
    )
    prior = tuple(float(value) for value in np.arange(1, 16) / 120.0)
    base_config = ModelAveragedEndpointConfigV1(
        component_prior_probability=prior
    )
    posterior = infer_tempered_model_averaged_endpoint(
        residual,
        valid,
        end_frame=5,
        config=TemperedModelAveragedEndpointConfigV2(
            base_config=base_config,
            effective_evidence_count_cap=2.0,
        ),
    )

    assert np.allclose(posterior.evidence_power, [0.4, 1.0, 1.0])
    assert np.allclose(posterior.component_weights[2], prior)
    assert np.array_equal(posterior.update_count, [5, 2, 0])


def test_tempering_preserves_component_diversity_and_predictive_growth() -> None:
    residual = np.linspace(0.0, 0.03, 20)[:, None, None] * np.array(
        [[[1.0, 0.0, 0.0]]]
    )
    valid = np.ones((20, 1), dtype=bool)
    historical = infer_model_averaged_endpoint(residual, valid, end_frame=20)
    tempered = infer_tempered_model_averaged_endpoint(
        residual,
        valid,
        end_frame=20,
        config=TemperedModelAveragedEndpointConfigV2(
            effective_evidence_count_cap=2.0
        ),
    )
    historical_effective = 1.0 / np.sum(np.square(historical.component_weights[0]))
    tempered_effective = 1.0 / np.sum(np.square(tempered.component_weights[0]))

    assert tempered_effective > historical_effective
    now = predict_tempered_model_averaged_endpoint(tempered, horizon_steps=0)
    future = predict_tempered_model_averaged_endpoint(tempered, horizon_steps=20)
    assert np.trace(future.covariance_m2[0]) > np.trace(now.covariance_m2[0])
    assert not tempered.mean_m.flags.writeable
    assert not future.covariance_m2.flags.writeable


def test_tempered_configuration_and_prediction_fail_closed() -> None:
    with pytest.raises(TypeError, match="base_config"):
        TemperedModelAveragedEndpointConfigV2(base_config=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite and positive"):
        TemperedModelAveragedEndpointConfigV2(
            effective_evidence_count_cap=float("inf")
        )
    with pytest.raises(TypeError, match="TemperedModelAveragedEndpointPosteriorV2"):
        predict_tempered_model_averaged_endpoint(object(), horizon_steps=0)  # type: ignore[arg-type]
    posterior = infer_tempered_model_averaged_endpoint(
        np.zeros((1, 1, 3)),
        np.ones((1, 1), dtype=bool),
        end_frame=1,
    )
    with pytest.raises(ValueError, match="nonnegative integer"):
        predict_tempered_model_averaged_endpoint(posterior, horizon_steps=-1)


def test_invalid_configurations_fail() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ModelAveragedEndpointConfigV1(components=())
    with pytest.raises(ValueError, match="unique"):
        ModelAveragedEndpointConfigV1(components=(FixedBayesianAnchorConfigV1(),) * 2)
    with pytest.raises(ValueError, match="component count"):
        ModelAveragedEndpointConfigV1(
            components=(FixedBayesianAnchorConfigV1(),),
            component_prior_probability=(0.5, 0.5),
        )


def test_provider_v2_wraps_endpoint_and_declares_calibration_boundary() -> None:
    posterior = infer_model_averaged_bayesian_anchor_endpoint(
        np.zeros((2, 1, 3)),
        np.ones((2, 1), dtype=bool),
        end_frame=2,
        config=_single_config(FixedBayesianAnchorConfigV1()),
    )
    manifest = causal4d_belief_provider_v2_manifest(provider_revision="revision-test")

    assert posterior.update_count[0] == 2
    assert manifest["schema_version"] == CAUSAL4D_BELIEF_PROVIDER_V2_API_VERSION
    assert "horizon_dependent_predictive_covariance" in manifest["capabilities"]
    assert "calibration" in manifest["metadata"]["raw_covariance_claim"]


def test_invalid_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="shape"):
        infer_model_averaged_endpoint(
            np.zeros((2, 3)),
            np.zeros(2, dtype=bool),
            end_frame=1,
        )
    with pytest.raises(ValueError, match="integer"):
        infer_model_averaged_endpoint(
            np.zeros((2, 1, 3)),
            np.zeros((2, 1), dtype=bool),
            end_frame=1.5,
        )
    posterior = infer_model_averaged_endpoint(
        np.zeros((2, 1, 3)),
        np.zeros((2, 1), dtype=bool),
        end_frame=1,
        config=_single_config(FixedBayesianAnchorConfigV1()),
    )
    with pytest.raises(ValueError, match="nonnegative integer"):
        predict_model_averaged_endpoint(posterior, horizon_steps=-1)


def _posterior_contract_kwargs() -> dict[str, object]:
    config = _single_config(FixedBayesianAnchorConfigV1())
    return {
        "mean_m": np.zeros((1, 3)),
        "covariance_m2": np.eye(3)[None],
        "final_nominal_probability": np.array([0.5]),
        "update_count": np.array([1], dtype=np.int64),
        "component_weights": np.ones((1, 1)),
        "component_log_evidence": np.zeros((1, 1)),
        "component_mean_m": np.zeros((1, 1, 3)),
        "component_variance_m2": np.ones((1, 1)),
        "component_process_variance_m2": np.zeros(1),
        "config": config,
        "end_frame": 1,
    }


def _prediction_contract_kwargs() -> dict[str, object]:
    return {
        "mean_m": np.zeros((1, 3)),
        "covariance_m2": np.eye(3)[None],
        "component_weights": np.ones((1, 1)),
        "horizon_steps": 0,
    }


def test_config_rejects_wrong_component_type_and_invalid_priors() -> None:
    with pytest.raises(TypeError, match="FixedBayesianAnchorConfigV1"):
        ModelAveragedEndpointConfigV1(components=(object(),))  # type: ignore[arg-type]
    component = FixedBayesianAnchorConfigV1()
    with pytest.raises(ValueError, match="finite and positive"):
        ModelAveragedEndpointConfigV1(
            components=(component,),
            component_prior_probability=(np.nan,),
        )
    with pytest.raises(ValueError, match="finite and positive"):
        ModelAveragedEndpointConfigV1(
            components=(component,),
            component_prior_probability=(0.0,),
        )


@pytest.mark.parametrize(
    ("covariance", "message"),
    [
        (np.zeros((1, 2, 2)), "shape"),
        (np.full((1, 3, 3), np.nan), "finite"),
        (
            np.array([[[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]),
            "symmetric",
        ),
        (-np.eye(3)[None], "positive semidefinite"),
    ],
)
def test_prediction_rejects_invalid_covariance(
    covariance: np.ndarray,
    message: str,
) -> None:
    kwargs = _prediction_contract_kwargs()
    kwargs["covariance_m2"] = covariance
    with pytest.raises(ValueError, match=message):
        ModelAveragedEndpointPredictionV1(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("config", object(), "config"),
        ("end_frame", 1.5, "integer"),
        ("end_frame", 0, "positive"),
        ("mean_m", np.zeros((0, 3)), "mean_m"),
        ("covariance_m2", np.repeat(np.eye(3)[None], 2, axis=0), "track count"),
        ("final_nominal_probability", np.zeros(2), "probability shape"),
        ("update_count", np.array([1.0]), "integers"),
        ("update_count", np.array([-1]), "nonnegative"),
        ("component_weights", np.ones((1, 2)), "weight/evidence"),
        ("component_log_evidence", np.ones((1, 2)), "weight/evidence"),
        ("component_mean_m", np.zeros((1, 2, 3)), "component_mean"),
        ("component_variance_m2", np.zeros((1, 2)), "component_variance"),
        ("component_process_variance_m2", np.zeros(2), "process_variance"),
        ("component_log_evidence", np.array([[np.nan]]), "non-finite"),
        ("final_nominal_probability", np.array([2.0]), r"\[0, 1\]"),
        ("component_weights", np.array([[0.5]]), "row-normalized"),
        ("component_variance_m2", np.array([[-1.0]]), "nonnegative"),
        ("component_process_variance_m2", np.array([-1.0]), "nonnegative"),
    ],
)
def test_posterior_contract_rejects_invalid_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    kwargs = _posterior_contract_kwargs()
    kwargs[field] = value
    with pytest.raises((TypeError, ValueError), match=message):
        ModelAveragedEndpointPosteriorV1(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("horizon_steps", True, "nonnegative integer"),
        ("horizon_steps", 0.5, "nonnegative integer"),
        ("mean_m", np.zeros((0, 3)), "mean_m"),
        ("covariance_m2", np.repeat(np.eye(3)[None], 2, axis=0), "track count"),
        ("component_weights", np.ones((2, 1)), "shape"),
        ("mean_m", np.full((1, 3), np.nan), "non-finite"),
        ("component_weights", np.array([[np.nan]]), "non-finite"),
        ("component_weights", np.array([[-1.0]]), "row-normalized"),
        ("component_weights", np.array([[0.5]]), "row-normalized"),
    ],
)
def test_prediction_contract_rejects_invalid_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    kwargs = _prediction_contract_kwargs()
    kwargs[field] = value
    with pytest.raises(ValueError, match=message):
        ModelAveragedEndpointPredictionV1(**kwargs)  # type: ignore[arg-type]


def test_additional_input_and_type_contracts_fail_closed() -> None:
    with pytest.raises(ValueError, match="valid must match"):
        infer_model_averaged_endpoint(
            np.zeros((2, 1, 3)),
            np.zeros((2, 2), dtype=bool),
            end_frame=1,
        )
    residual = np.zeros((2, 1, 3))
    residual[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        infer_model_averaged_endpoint(
            residual,
            np.ones((2, 1), dtype=bool),
            end_frame=1,
        )
    with pytest.raises(ValueError, match="inside"):
        infer_model_averaged_endpoint(
            np.zeros((2, 1, 3)),
            np.ones((2, 1), dtype=bool),
            end_frame=0,
        )
    with pytest.raises(TypeError, match="config"):
        infer_model_averaged_endpoint(
            np.zeros((2, 1, 3)),
            np.ones((2, 1), dtype=bool),
            end_frame=1,
            config=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="posterior"):
        predict_model_averaged_endpoint(
            object(),  # type: ignore[arg-type]
            horizon_steps=0,
        )


def test_component_observation_mismatch_is_detected(monkeypatch) -> None:
    original = endpoint_module._filter_component
    calls = 0

    def mismatched(*args, **kwargs):
        nonlocal calls
        values = original(*args, **kwargs)
        calls += 1
        if calls == 2:
            count = values[3].copy()
            count[0] += 1
            return (*values[:3], count, values[4])
        return values

    monkeypatch.setattr(endpoint_module, "_filter_component", mismatched)
    config = ModelAveragedEndpointConfigV1(
        components=(
            FixedBayesianAnchorConfigV1(process_std_m=0.0),
            FixedBayesianAnchorConfigV1(process_std_m=0.001),
        )
    )
    with pytest.raises(AssertionError, match="different observations"):
        infer_model_averaged_endpoint(
            np.zeros((1, 1, 3)),
            np.ones((1, 1), dtype=bool),
            end_frame=1,
            config=config,
        )


def test_provider_default_and_invalid_config_paths() -> None:
    posterior = infer_model_averaged_bayesian_anchor_endpoint(
        np.zeros((1, 1, 3)),
        np.ones((1, 1), dtype=bool),
        end_frame=1,
    )
    assert posterior.update_count[0] == 1
    with pytest.raises(TypeError, match="config"):
        infer_model_averaged_bayesian_anchor_endpoint(
            np.zeros((1, 1, 3)),
            np.ones((1, 1), dtype=bool),
            end_frame=1,
            config=object(),  # type: ignore[arg-type]
        )
