from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin._gauge_aware_contracts import (
    GaugeAwareBeliefResult,
    GaugeAwareObservationBatch,
    _block_diagonal,
    _regularized_precision,
)
from bayesian_phystwin._prior_aware_gauge_math import (
    _group_layout,
    _prior_covariances,
    _student_t_mixture_statistics,
    _whiten,
)
from bayesian_phystwin.observed_information_covariance import (
    ObservedInformationCovarianceResultV1,
    observed_information_covariance_from_prior_aware_result,
)
from bayesian_phystwin.prior_aware_gauge_belief import (
    PriorAwareGaugeConfigV1,
    update_prior_aware_gauge_belief,
)


def _empty(count: int) -> np.ndarray:
    return np.zeros((count, 3, 0), dtype=np.float64)


def _batch() -> GaugeAwareObservationBatch:
    count = 12
    state = np.zeros((count, 3, 1), dtype=np.float64)
    state[:, 0, 0] = 1.0
    innovation = np.zeros((count, 3), dtype=np.float64)
    innovation[:, 0] = 0.01
    innovation[-4:, 0] = 0.014
    groups = tuple(
        "source-a" if index < 8 else "source-b" for index in range(count)
    )
    return GaugeAwareObservationBatch(
        innovation_m=innovation,
        observation_covariance_m2=np.tile(
            np.eye(3, dtype=np.float64) * 1e-6,
            (count, 1, 1),
        ),
        state_jacobian=state,
        gauge_jacobian=state.copy(),
        shared_bias_jacobian=_empty(count),
        view_bias_jacobian=_empty(count),
        query_state_jacobian=state.copy(),
        gauge_prior_covariance=np.asarray([[1e-10]], dtype=np.float64),
        correlation_group_ids=groups,
        prior_reliability=np.ones(count, dtype=np.float64),
        prior_nominal_probability=np.asarray(
            [0.95] * 8 + [0.80] * 4,
            dtype=np.float64,
        ),
        composite_weight=np.ones(count, dtype=np.float64),
        physical_response_scale_m=0.05,
        state_prior_covariance_m2=np.asarray([[0.01]], dtype=np.float64),
        metadata={"protocol": "observed-information-test-v1"},
    )


def _config() -> PriorAwareGaugeConfigV1:
    return PriorAwareGaugeConfigV1(
        effective_samples_per_correlation_group=12,
        minimum_identifiable_fraction=0.01,
    )


def _analysis() -> tuple[
    GaugeAwareObservationBatch,
    GaugeAwareBeliefResult,
    ObservedInformationCovarianceResultV1,
]:
    batch = _batch()
    config = _config()
    result = update_prior_aware_gauge_belief(batch, config=config)
    assert result.inference_admissible
    analysis = observed_information_covariance_from_prior_aware_result(
        batch,
        result,
        config=config,
        metadata={"study": "focused-contract"},
    )
    return batch, result, analysis


def _finite_difference_hessian(
    batch: GaugeAwareObservationBatch,
    result: GaugeAwareBeliefResult,
    config: PriorAwareGaugeConfigV1,
) -> np.ndarray:
    nuisance = np.concatenate(
        (
            batch.gauge_jacobian,
            batch.shared_bias_jacobian,
            batch.view_bias_jacobian,
        ),
        axis=2,
    )
    target, (state_white, nuisance_white), _ = _whiten(
        batch.innovation_m,
        batch.observation_covariance_m2,
        (batch.state_jacobian, nuisance),
        name="finite-difference-observation",
    )
    state_mapping = np.asarray(result.identifiable_state_transform)
    design = np.concatenate(
        (
            np.einsum("mcs,sr->mcr", state_white, state_mapping),
            nuisance_white,
        ),
        axis=2,
    )
    (
        _,
        indices,
        _,
        prior_nominal,
        group_power,
    ) = _group_layout(
        batch.correlation_group_ids,
        batch.prior_reliability,
        np.asarray(batch.prior_nominal_probability),
        np.asarray(batch.composite_weight),
        config.effective_samples_per_correlation_group,
        composite_weight_mode=batch.composite_weight_mode,
    )
    state_prior, nuisance_prior, _ = _prior_covariances(batch, config)
    reduced_prior = _block_diagonal(
        [np.eye(state_mapping.shape[1]), nuisance_prior]
    )
    prior_precision = _regularized_precision(
        reduced_prior,
        "finite-difference reduced prior covariance",
        eigenvalue_floor=config.prior_eigenvalue_floor,
    )
    reduced_state = np.linalg.lstsq(
        state_mapping,
        result.state_coefficients,
        rcond=None,
    )[0]
    solution = np.concatenate((reduced_state, result.gauge_delta))

    def gradient(value: np.ndarray) -> np.ndarray:
        residual = target - np.einsum("mci,i->mc", design, value)
        output = prior_precision @ value
        for position, selected in enumerate(indices):
            active = selected[batch.prior_reliability[selected] > 0.0]
            squared = float(
                np.sum(
                    batch.prior_reliability[active]
                    * np.sum(np.square(residual[active]), axis=1)
                )
            )
            statistics = _student_t_mixture_statistics(
                squared,
                3 * len(active),
                float(prior_nominal[position]),
                config,
            )
            score = np.einsum(
                "m,mci,mc->i",
                batch.prior_reliability[active],
                design[active],
                residual[active],
            )
            output -= (
                group_power[position]
                * statistics.expected_precision
                * score
            )
        return output

    step = 1e-7
    identity = np.eye(len(solution), dtype=np.float64)
    return np.column_stack(
        tuple(
            (
                gradient(solution + step * direction)
                - gradient(solution - step * direction)
            )
            / (2.0 * step)
            for direction in identity
        )
    )


def test_observed_information_reconstructs_solver_and_is_immutable() -> None:
    _, result, analysis = _analysis()

    assert analysis.covariance_semantics.method == (
        "laplace_observed_information"
    )
    assert analysis.covariance_semantics.mixture_curvature_exact
    assert not analysis.covariance_semantics.calibrated
    assert analysis.metadata["input_lineage"] == result.input_lineage
    assert np.linalg.eigvalsh(analysis.observed_information).min() > 0.0
    np.testing.assert_allclose(
        analysis.observed_information @ analysis.reduced_covariance,
        np.eye(analysis.reduced_dimension),
        atol=1e-8,
        rtol=1e-8,
    )
    assert analysis.metadata["minimum_eigenvalue"] == pytest.approx(
        result.diagnostics[
            "exact_reduced_mixture_hessian_minimum_eigenvalue"
        ]
    )
    assert analysis.artifact_id == replace(analysis).artifact_id
    assert not analysis.full_covariance.flags.writeable
    with pytest.raises(ValueError):
        analysis.full_covariance[0, 0] = 0.0


def test_observed_information_matches_finite_difference_gradient() -> None:
    batch, result, analysis = _analysis()
    finite_difference = _finite_difference_hessian(
        batch,
        result,
        _config(),
    )

    np.testing.assert_allclose(
        analysis.observed_information,
        finite_difference,
        atol=0.25,
        rtol=5e-7,
    )


def test_observed_information_rejects_tampered_robust_weights() -> None:
    batch = _batch()
    config = _config()
    result = update_prior_aware_gauge_belief(batch, config=config)
    assert result.inference_admissible
    tampered = replace(
        result,
        robust_weights=np.asarray(result.robust_weights) * 0.5,
    )

    with pytest.raises(ValueError, match="robust weights do not match"):
        observed_information_covariance_from_prior_aware_result(
            batch,
            tampered,
            config=config,
        )


def test_observed_information_requires_the_exact_unfloored_objective() -> None:
    batch = _batch()
    result = update_prior_aware_gauge_belief(batch, config=_config())
    assert result.inference_admissible
    floored = PriorAwareGaugeConfigV1(
        effective_samples_per_correlation_group=12,
        minimum_identifiable_fraction=0.01,
        minimum_robust_precision=0.05,
    )

    with pytest.raises(ValueError, match="minimum_robust_precision=0"):
        observed_information_covariance_from_prior_aware_result(
            batch,
            result,
            config=floored,
        )


def test_result_contract_rejects_indefinite_observed_information() -> None:
    _, _, analysis = _analysis()
    indefinite = -np.eye(analysis.reduced_dimension, dtype=np.float64)

    with pytest.raises(ValueError, match="positive definite"):
        replace(
            analysis,
            observed_information=indefinite,
            artifact_id=None,
        )


def test_result_contract_rejects_artifact_id_tampering() -> None:
    _, _, analysis = _analysis()

    with pytest.raises(ValueError, match="artifact_id does not match"):
        replace(analysis, artifact_id="0" * 64)
