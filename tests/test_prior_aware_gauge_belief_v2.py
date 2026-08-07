from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import bayesian_phystwin.prior_aware_gauge_belief_v2 as strict_v2
from bayesian_phystwin._gauge_aware_contracts import (
    GaugeAwareBeliefResult,
    GaugeAwareObservationBatch,
)
from bayesian_phystwin._prior_aware_gauge_math import PriorAwareGaugeConfigV1
from bayesian_phystwin.prior_aware_gauge_belief_v2 import (
    PRIOR_AWARE_GAUGE_BELIEF_V2_IMPLEMENTATION,
    PriorAwareGaugeAdmissionConfigV2,
    PriorAwareGaugeBeliefResultV2,
    update_prior_aware_gauge_belief_v2,
    update_sparse_prior_aware_gauge_belief_v2,
)
from bayesian_phystwin.sparse_prior_aware_gauge_belief import SparseGaugeDesignV1


def _empty_design(count: int) -> np.ndarray:
    return np.zeros((count, 3, 0), dtype=np.float64)


def _batches() -> tuple[
    GaugeAwareObservationBatch,
    GaugeAwareObservationBatch,
    SparseGaugeDesignV1,
]:
    count = 12
    mode = np.linspace(-1.0, 1.0, count)
    innovation = np.zeros((count, 3), dtype=np.float64)
    innovation[:, 0] = 0.006 * mode
    state = np.zeros((count, 3, 1), dtype=np.float64)
    state[:, 0, 0] = mode
    covariance = np.repeat((np.eye(3) * 0.01)[None], count, axis=0)
    common = dict(
        innovation_m=innovation,
        observation_covariance_m2=covariance,
        state_jacobian=state,
        shared_bias_jacobian=_empty_design(count),
        view_bias_jacobian=_empty_design(count),
        query_state_jacobian=state.copy(),
        correlation_group_ids=("group-0",) * count,
        prior_reliability=np.ones(count),
        prior_nominal_probability=np.full(count, 0.99),
        composite_weight=np.ones(count),
        physical_response_scale_m=1.0,
        state_prior_covariance_m2=np.asarray([[0.04]]),
        metadata={"fixture": "strict-prior-aware-v2"},
    )
    local_gauge = np.zeros((count, 3, 1), dtype=np.float64)
    dense = GaugeAwareObservationBatch(
        **common,
        gauge_jacobian=local_gauge,
        gauge_prior_covariance=np.asarray([[0.09]]),
    )
    sparse_batch = GaugeAwareObservationBatch(
        **common,
        gauge_jacobian=_empty_design(count),
        gauge_prior_covariance=np.zeros((0, 0)),
    )
    sparse = SparseGaugeDesignV1(
        local_gauge_jacobian=local_gauge,
        gauge_indices=np.zeros(count, dtype=np.int64),
        gauge_prior_covariance=np.asarray([[0.09]]),
        gauge_ids=("window-0",),
    )
    return dense, sparse_batch, sparse


def _config(**changes: object) -> PriorAwareGaugeConfigV1:
    values: dict[str, object] = {
        "effective_samples_per_correlation_group": 12.0,
        "maximum_iterations": 100,
        "convergence_tolerance": 1.0e-12,
        "minimum_conditional_information_fraction": 0.0,
        "minimum_identifiable_fraction": 1.0e-8,
        "minimum_query_sensitivity_fraction": 0.0,
        "maximum_state_update_m": 1.0,
        "maximum_update_to_physical_response_ratio": 100.0,
    }
    values.update(changes)
    return replace(PriorAwareGaugeConfigV1(), **values)


def _diagnostics(
    *,
    converged: bool = True,
    objective: str = "exact-group-mixture-gradient",
    minimum_eigenvalue: float = 1.0,
    maximum_eigenvalue: float = 2.0,
) -> dict[str, object]:
    return {
        "robust_likelihood_objective": objective,
        "mixture_fixed_point_converged": converged,
        "mixture_solution_delta": 0.0,
        "mixture_stationarity_norm": 0.0,
        "exact_reduced_mixture_hessian_minimum_eigenvalue": minimum_eigenvalue,
        "exact_reduced_mixture_hessian_maximum_eigenvalue": maximum_eigenvalue,
        "exact_reduced_mixture_hessian_positive_definite": (minimum_eigenvalue > 0.0),
    }


def _synthetic_result(
    batch: GaugeAwareObservationBatch,
    *,
    gauge_count: int,
    diagnostics: dict[str, object],
    admissible: bool = True,
) -> GaugeAwareBeliefResult:
    state_count = batch.state_jacobian.shape[2]
    dimension = state_count + gauge_count
    return GaugeAwareBeliefResult(
        inference_admissible=admissible,
        reason=("inference-admissible" if admissible else "synthetic-rejection"),
        state_coefficients=np.full(state_count, 0.005 if admissible else 0.0),
        gauge_delta=np.zeros(gauge_count),
        shared_bias_coefficients=np.zeros(0),
        view_bias_coefficients=np.zeros(0),
        anchor_bias_coefficients=np.zeros(0),
        posterior_covariance=np.eye(dimension) * 0.01,
        identifiable_state_transform=(
            np.ones((state_count, 1)) if admissible else np.zeros((state_count, 0))
        ),
        identifiable_fractions=(np.ones(1) if admissible else np.zeros(0)),
        query_sensitivity_fractions=(np.ones(1) if admissible else np.zeros(0)),
        robust_weights=np.ones(len(batch.innovation_m)),
        anchor_robust_weights=np.zeros(0),
        diagnostics=diagnostics,
        input_lineage=batch.metadata,
    )


def test_dense_and_sparse_v2_admit_converged_positive_curvature() -> None:
    dense_batch, sparse_batch, sparse_design = _batches()
    config = _config()

    dense = update_prior_aware_gauge_belief_v2(dense_batch, config=config)
    sparse = update_sparse_prior_aware_gauge_belief_v2(
        sparse_batch,
        sparse_design,
        config=config,
    )

    assert isinstance(dense, PriorAwareGaugeBeliefResultV2)
    assert isinstance(sparse, PriorAwareGaugeBeliefResultV2)
    assert dense.inference_admissible
    assert sparse.inference_admissible
    assert dense.diagnostics["strict_admission_passed"] is True
    assert sparse.diagnostics["strict_admission_passed"] is True
    assert dense.diagnostics["implementation_id"] == (
        PRIOR_AWARE_GAUGE_BELIEF_V2_IMPLEMENTATION
    )
    np.testing.assert_allclose(sparse.state_coefficients, dense.state_coefficients)
    np.testing.assert_allclose(sparse.posterior_covariance, dense.posterior_covariance)


def test_v2_fails_closed_when_fixed_point_does_not_converge() -> None:
    dense_batch, sparse_batch, sparse_design = _batches()
    config = _config(maximum_iterations=1, convergence_tolerance=1.0e-15)

    dense = update_prior_aware_gauge_belief_v2(dense_batch, config=config)
    sparse = update_sparse_prior_aware_gauge_belief_v2(
        sparse_batch,
        sparse_design,
        config=config,
    )

    for result in (dense, sparse):
        assert not result.inference_admissible
        assert result.reason == "strict-v2-fixed-point-not-converged"
        assert result.diagnostics["underlying_inference_admissible"] is True
        np.testing.assert_array_equal(result.state_coefficients, 0.0)
        np.testing.assert_array_equal(result.robust_weights, 0.0)
    np.testing.assert_allclose(dense.posterior_covariance, np.diag([0.04, 0.09]))
    np.testing.assert_allclose(sparse.posterior_covariance, np.diag([0.04, 0.09]))


def test_v2_rejects_precision_floored_approximate_objective() -> None:
    dense_batch, _, _ = _batches()
    result = update_prior_aware_gauge_belief_v2(
        dense_batch,
        config=_config(minimum_robust_precision=0.1),
    )

    assert not result.inference_admissible
    assert result.reason == "strict-v2-non-exact-mixture-objective"
    np.testing.assert_array_equal(result.state_coefficients, 0.0)


def test_v2_rejects_nonpositive_exact_curvature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dense_batch, _, _ = _batches()
    underlying = _synthetic_result(
        dense_batch,
        gauge_count=1,
        diagnostics=_diagnostics(
            minimum_eigenvalue=-0.1,
            maximum_eigenvalue=2.0,
        ),
    )
    monkeypatch.setattr(
        strict_v2,
        "update_prior_aware_gauge_belief",
        lambda *_args, **_kwargs: underlying,
    )

    result = update_prior_aware_gauge_belief_v2(dense_batch, config=_config())

    assert not result.inference_admissible
    assert result.reason == "strict-v2-non-positive-exact-mixture-curvature"
    np.testing.assert_allclose(result.posterior_covariance, np.diag([0.04, 0.09]))


def test_v2_rejects_ill_conditioned_exact_curvature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dense_batch, _, _ = _batches()
    underlying = _synthetic_result(
        dense_batch,
        gauge_count=1,
        diagnostics=_diagnostics(
            minimum_eigenvalue=1.0e-8,
            maximum_eigenvalue=1.0,
        ),
    )
    monkeypatch.setattr(
        strict_v2,
        "update_prior_aware_gauge_belief",
        lambda *_args, **_kwargs: underlying,
    )

    result = update_prior_aware_gauge_belief_v2(
        dense_batch,
        config=_config(),
        admission_config=PriorAwareGaugeAdmissionConfigV2(
            maximum_exact_hessian_condition_number=1.0e6
        ),
    )

    assert not result.inference_admissible
    assert result.reason == "strict-v2-ill-conditioned-exact-mixture-curvature"
    assert result.diagnostics["strict_exact_hessian_condition_number"] == (
        pytest.approx(1.0e8)
    )


def test_v2_rejects_inconsistent_curvature_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dense_batch, _, _ = _batches()
    diagnostics = _diagnostics()
    diagnostics["exact_reduced_mixture_hessian_positive_definite"] = False
    underlying = _synthetic_result(
        dense_batch,
        gauge_count=1,
        diagnostics=diagnostics,
    )
    monkeypatch.setattr(
        strict_v2,
        "update_prior_aware_gauge_belief",
        lambda *_args, **_kwargs: underlying,
    )

    result = update_prior_aware_gauge_belief_v2(dense_batch, config=_config())

    assert not result.inference_admissible
    assert result.reason == "strict-v2-invalid-admission-diagnostics"


def test_v2_preserves_underlying_rejection_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dense_batch, _, _ = _batches()
    underlying = _synthetic_result(
        dense_batch,
        gauge_count=1,
        diagnostics={},
        admissible=False,
    )
    monkeypatch.setattr(
        strict_v2,
        "update_prior_aware_gauge_belief",
        lambda *_args, **_kwargs: underlying,
    )

    result = update_prior_aware_gauge_belief_v2(dense_batch, config=_config())

    assert not result.inference_admissible
    assert result.reason == "synthetic-rejection"
    assert result.diagnostics["strict_admission_reason"] == (
        "underlying-inference-rejected"
    )
