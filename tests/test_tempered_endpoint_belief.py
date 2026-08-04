import math

import numpy as np
import pytest

from bayesian_phystwin.causal4d_belief_provider_v3 import (
    CAUSAL4D_BELIEF_PROVIDER_V3_API_VERSION,
    causal4d_belief_provider_v3_manifest,
    infer_tempered_bayesian_anchor_endpoint,
)
from bayesian_phystwin.contracts.fixed_anchor import FixedBayesianAnchorConfigV1
from bayesian_phystwin.endpoint_model_average import (
    ModelAveragedEndpointConfigV1,
    ModelAveragedEndpointPosteriorV1,
    infer_model_averaged_endpoint,
)
from bayesian_phystwin.tempered_endpoint_belief import (
    EndpointRegretGuardFeaturesV1,
    TemperedEndpointConfigV2,
    apply_endpoint_regret_guard,
    fit_endpoint_grouped_calibration,
    fit_endpoint_regret_guard,
    fit_source_component_prior,
    minimum_groups_for_finite_conformal_quantile,
    predict_tempered_endpoint,
    temper_model_averaged_endpoint,
)


def _base_config() -> ModelAveragedEndpointConfigV1:
    return ModelAveragedEndpointConfigV1(
        components=(
            FixedBayesianAnchorConfigV1(
                process_std_m=0.0,
                observation_std_m=0.001,
            ),
            FixedBayesianAnchorConfigV1(
                process_std_m=0.005,
                observation_std_m=0.005,
            ),
        )
    )


def _base_posterior(
    *,
    log_evidence: np.ndarray | None = None,
    update_count: np.ndarray | None = None,
) -> ModelAveragedEndpointPosteriorV1:
    config = _base_config()
    evidence = (
        np.array([[100.0, 0.0]])
        if log_evidence is None
        else np.asarray(log_evidence, dtype=float)
    )
    count = (
        np.array([100], dtype=np.int64)
        if update_count is None
        else np.asarray(update_count)
    )
    component_mean = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[0.01, 0.0, 0.0]],
        ]
    )
    component_variance = np.array([[1e-6], [25e-6]])
    weights = np.array([[1.0, 0.0]])
    return ModelAveragedEndpointPosteriorV1(
        mean_m=np.zeros((1, 3)),
        covariance_m2=np.eye(3)[None] * 1e-6,
        final_nominal_probability=np.array([0.9]),
        update_count=count,
        component_weights=weights,
        component_log_evidence=evidence,
        component_mean_m=component_mean,
        component_variance_m2=component_variance,
        component_process_variance_m2=np.array([0.0, 25e-6]),
        config=config,
        end_frame=100,
    )


def _feature(value: float, horizon: float) -> EndpointRegretGuardFeaturesV1:
    return EndpointRegretGuardFeaturesV1(
        validation_relative_improvement=value,
        mean_component_entropy_nats=0.8 + value,
        median_effective_component_count=2.0 + value,
        mean_predictive_std_m=0.005 + 0.001 * value,
        correction_rms_m=0.004 + 0.001 * value,
        correction_saturated_fraction=0.05,
        normalized_horizon=horizon,
    )


def test_neutral_tempering_reproduces_v1_moments() -> None:
    residual = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[0.01, 0.0, 0.0]],
            [[0.012, 0.0, 0.0]],
        ]
    )
    valid = np.ones((3, 1), dtype=bool)
    base = infer_model_averaged_endpoint(
        residual,
        valid,
        end_frame=3,
        config=_base_config(),
    )
    result = temper_model_averaged_endpoint(
        base,
        config=TemperedEndpointConfigV2(
            maximum_effective_observations=None,
        ),
    )

    assert np.allclose(result.component_weights, base.component_weights)
    assert np.allclose(result.mean_m, base.mean_m)
    assert np.allclose(result.covariance_m2, base.covariance_m2)
    assert np.array_equal(
        result.final_nominal_probability,
        base.final_nominal_probability,
    )


def test_effective_evidence_cap_prevents_complete_component_collapse() -> None:
    base = _base_posterior()
    raw = temper_model_averaged_endpoint(
        base,
        config=TemperedEndpointConfigV2(
            maximum_effective_observations=None,
        ),
    )
    capped = temper_model_averaged_endpoint(
        base,
        config=TemperedEndpointConfigV2(
            maximum_effective_observations=1.0,
        ),
    )

    assert capped.temperature_by_track[0] == 100.0
    assert capped.component_entropy_nats[0] > raw.component_entropy_nats[0]
    assert (
        capped.effective_component_count[0]
        > raw.effective_component_count[0]
    )
    assert capped.between_model_covariance_fraction[0] > 0.0


def test_covariance_inflation_is_source_explicit_and_horizon_aware() -> None:
    base = _base_posterior(log_evidence=np.zeros((1, 2)))
    posterior = temper_model_averaged_endpoint(
        base,
        config=TemperedEndpointConfigV2(
            maximum_effective_observations=None,
            covariance_scale=4.0,
            isotropic_floor_std_m=0.01,
        ),
    )
    now = predict_tempered_endpoint(posterior, horizon_steps=0)
    future = predict_tempered_endpoint(posterior, horizon_steps=20)

    assert np.trace(now.covariance_m2[0]) > np.trace(base.covariance_m2[0])
    assert np.trace(future.covariance_m2[0]) > np.trace(now.covariance_m2[0])
    assert now.config_id == posterior.config.config_id


def test_source_component_prior_is_group_balanced_and_immutable() -> None:
    scores = np.array(
        [
            [0.0, -4.0],
            [-4.0, 0.0],
            [0.0, -4.0],
        ]
    )
    prior = fit_source_component_prior(
        scores,
        ("group-a", "group-b", "group-c"),
        score_temperature=2.0,
        uniform_pseudocount=3.0,
    )
    prior_id = prior.prior_id
    scores[0, 0] = 100.0

    assert prior.probability[0] > prior.probability[1]
    assert np.isclose(np.sum(prior.probability), 1.0)
    assert prior.prior_id == prior_id
    assert not prior.probability.flags.writeable
    with pytest.raises(ValueError):
        prior.probability[0] = 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"evidence_temperature": 0.5}, "at least"),
        ({"maximum_effective_observations": 0.0}, "at least"),
        ({"covariance_scale": 0.5}, "at least"),
        ({"isotropic_floor_std_m": -1.0}, "at least"),
        ({"component_prior_probability": ()}, "must not be empty"),
        (
            {"component_prior_probability": (0.0, 1.0)},
            "finite and positive",
        ),
    ],
)
def test_tempered_config_rejects_invalid_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        TemperedEndpointConfigV2(**kwargs)  # type: ignore[arg-type]


def test_source_component_prior_rejects_group_and_shape_errors() -> None:
    with pytest.raises(ValueError, match="two unique"):
        fit_source_component_prior(
            np.zeros((1, 2)),
            ("one",),
        )
    with pytest.raises(ValueError, match="shape"):
        fit_source_component_prior(
            np.zeros((2, 2)),
            ("one", "two", "three"),
        )
    values = np.zeros((2, 2))
    values[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_source_component_prior(values, ("one", "two"))


def test_candidate_specific_guard_accepts_only_negative_source_regret() -> None:
    rows = tuple(
        _feature(value, horizon)
        for value, horizon in (
            (0.00, 0.0),
            (0.01, 1.0),
            (0.02, 0.0),
            (0.03, 1.0),
            (0.04, 0.0),
            (0.05, 1.0),
        )
    )
    groups = ("a", "a", "b", "b", "c", "c")
    fallback_loss = np.ones(len(rows))
    accepted_guard = fit_endpoint_regret_guard(
        rows,
        fallback_loss - 0.2,
        fallback_loss,
        groups,
        nominal_coverage=0.5,
    )
    accepted = apply_endpoint_regret_guard(
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        rows[0],
        accepted_guard,
    )

    assert accepted.candidate_accepted
    assert np.array_equal(accepted.selected_value, [3.0, 4.0])

    rejected_guard = fit_endpoint_regret_guard(
        rows,
        fallback_loss + 0.2,
        fallback_loss,
        groups,
        nominal_coverage=0.5,
    )
    rejected = apply_endpoint_regret_guard(
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        rows[0],
        rejected_guard,
    )

    assert not rejected.candidate_accepted
    assert np.array_equal(rejected.selected_value, [1.0, 2.0])
    assert "fallback" in rejected.reason


def test_guard_rejects_non_source_safe_inputs() -> None:
    row = _feature(0.0, 0.5)
    with pytest.raises(ValueError, match="match feature"):
        fit_endpoint_regret_guard(
            (row, row, row),
            np.ones(2),
            np.ones(3),
            ("a", "b", "c"),
        )
    with pytest.raises(ValueError, match="nonnegative"):
        fit_endpoint_regret_guard(
            (row, row, row),
            np.array([-1.0, 1.0, 1.0]),
            np.ones(3),
            ("a", "b", "c"),
        )


def test_grouped_calibration_fails_closed_with_too_few_groups() -> None:
    errors = tuple(np.array([2.0, 1.0]) for _ in range(3))
    predictions = tuple(np.ones(2) for _ in range(3))
    calibration = fit_endpoint_grouped_calibration(
        errors,
        predictions,
        ("a", "b", "c"),
        coverage=0.90,
    )

    assert not calibration.finite
    assert math.isinf(calibration.quantile)
    assert calibration.required_group_count_for_finite_quantile == 9
    assert np.all(np.isinf(calibration.upper_bound(np.array([1.0, 2.0]))))


def test_grouped_calibration_is_finite_with_nine_groups() -> None:
    errors = tuple(np.array([2.0]) for _ in range(9))
    predictions = tuple(np.array([1.0]) for _ in range(9))
    calibration = fit_endpoint_grouped_calibration(
        errors,
        predictions,
        tuple(f"group-{index}" for index in range(9)),
        coverage=0.90,
    )

    assert calibration.finite
    assert calibration.quantile == 2.0
    assert np.array_equal(
        calibration.upper_bound(np.array([3.0])),
        np.array([6.0]),
    )
    assert minimum_groups_for_finite_conformal_quantile(0.90) == 9


def test_provider_v3_exposes_tempering_and_calibration_boundaries() -> None:
    posterior = infer_tempered_bayesian_anchor_endpoint(
        np.zeros((2, 1, 3)),
        np.ones((2, 1), dtype=bool),
        end_frame=2,
        config=TemperedEndpointConfigV2(
            maximum_effective_observations=None,
        ),
    )
    manifest = causal4d_belief_provider_v3_manifest(
        provider_revision="revision-test"
    )

    assert posterior.update_count[0] == 2
    assert manifest["schema_version"] == CAUSAL4D_BELIEF_PROVIDER_V3_API_VERSION
    assert "effective_evidence_tempering" in manifest["capabilities"]
    assert "independent-group" in manifest["metadata"]["calibration_boundary"]


def test_tempered_outputs_and_features_are_immutable() -> None:
    posterior = temper_model_averaged_endpoint(_base_posterior())
    feature = _feature(0.0, 0.5)
    vector = feature.as_array()

    assert not posterior.mean_m.flags.writeable
    assert not posterior.component_weights.flags.writeable
    assert not vector.flags.writeable
    with pytest.raises(ValueError):
        posterior.mean_m[0, 0] = 1.0


def test_invalid_tempering_and_prediction_inputs_fail_closed() -> None:
    base = _base_posterior()
    with pytest.raises(ValueError, match="component count"):
        temper_model_averaged_endpoint(
            base,
            config=TemperedEndpointConfigV2(
                component_prior_probability=(1.0,),
            ),
        )
    with pytest.raises(TypeError, match="posterior"):
        temper_model_averaged_endpoint(object())  # type: ignore[arg-type]
    posterior = temper_model_averaged_endpoint(base)
    with pytest.raises(ValueError, match="nonnegative integer"):
        predict_tempered_endpoint(posterior, horizon_steps=-1)
    with pytest.raises(TypeError, match="posterior"):
        predict_tempered_endpoint(  # type: ignore[arg-type]
            object(),
            horizon_steps=1,
        )
