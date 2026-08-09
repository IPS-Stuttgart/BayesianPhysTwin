from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

import bayesian_phystwin.prior_aware_gauge_belief_v2 as strict_v2
import bayesian_phystwin.prospective_prob4d_update as prospective_update
import bayesian_phystwin.tree_sparse_structured_gauge_prob4d as structured_update
from bayesian_phystwin._gauge_aware_contracts import GaugeAwareObservationBatch
from bayesian_phystwin._prior_aware_gauge_math import PriorAwareGaugeConfigV1
from bayesian_phystwin.prior_aware_gauge_belief import (
    update_prior_aware_gauge_belief,
)
from bayesian_phystwin.prior_aware_gauge_belief_v2 import (
    PriorAwareGaugeAdmissionConfigV2,
    update_prior_aware_gauge_belief_v2,
    update_sparse_prior_aware_gauge_belief_structured_v2,
    update_sparse_prior_aware_gauge_belief_v2,
)
from bayesian_phystwin.sparse_prior_aware_gauge_belief import (
    TreeSparseGaugeDesignV1,
    update_sparse_prior_aware_gauge_belief,
    update_sparse_prior_aware_gauge_belief_structured,
)
from bayesian_phystwin.structured_gauge_aware_result import (
    PRECISION_BACKED_COVARIANCE_REPRESENTATION,
)


def _tree_fixture() -> tuple[GaugeAwareObservationBatch, TreeSparseGaugeDesignV1]:
    count = 12
    mode = np.linspace(-1.0, 1.0, count)
    innovation = np.zeros((count, 3), dtype=np.float64)
    innovation[:, 0] = 0.006 * mode
    state = np.zeros((count, 3, 1), dtype=np.float64)
    state[:, 0, 0] = mode
    local_gauge = np.zeros((count, 3, 1), dtype=np.float64)
    empty = np.zeros((count, 3, 0), dtype=np.float64)
    batch = GaugeAwareObservationBatch(
        innovation_m=innovation,
        observation_covariance_m2=np.repeat(
            (np.eye(3) * 0.01)[None],
            count,
            axis=0,
        ),
        state_jacobian=state,
        gauge_jacobian=empty,
        shared_bias_jacobian=empty,
        view_bias_jacobian=empty,
        query_state_jacobian=state.copy(),
        correlation_group_ids=("group-0",) * count,
        prior_reliability=np.ones(count),
        prior_nominal_probability=np.full(count, 0.99),
        composite_weight=np.ones(count),
        physical_response_scale_m=1.0,
        gauge_prior_covariance=np.zeros((0, 0)),
        state_prior_covariance_m2=np.asarray([[0.04]]),
        metadata={"fixture": "claim-bearing-strict-admission"},
    )
    tree = TreeSparseGaugeDesignV1(
        local_gauge_jacobian=local_gauge,
        gauge_indices=np.zeros(count, dtype=np.int64),
        parent_indices=np.asarray([-1], dtype=np.int64),
        transition_matrices=np.zeros((1, 1, 1), dtype=np.float64),
        innovation_scale_tril=np.asarray([[[0.3]]], dtype=np.float64),
        gauge_ids=("window-0",),
        prior_id="0" * 64,
    )
    return batch, tree


def _exhausted_config() -> PriorAwareGaugeConfigV1:
    return replace(
        PriorAwareGaugeConfigV1(),
        effective_samples_per_correlation_group=12.0,
        maximum_iterations=1,
        convergence_tolerance=1.0e-15,
        minimum_conditional_information_fraction=0.0,
        minimum_identifiable_fraction=1.0e-8,
        minimum_query_sensitivity_fraction=0.0,
        maximum_state_update_m=1.0,
        maximum_update_to_physical_response_ratio=100.0,
    )


def _converged_config() -> PriorAwareGaugeConfigV1:
    return replace(
        _exhausted_config(),
        maximum_iterations=100,
        convergence_tolerance=1.0e-12,
    )


def _certificate(result: Any) -> dict[str, object]:
    return dict(result.diagnostics["strict_admission_certificate"])


def test_all_strict_v2_paths_reject_exhausted_v1_fixed_point() -> None:
    batch, tree = _tree_fixture()
    config = _exhausted_config()

    # One iteration intentionally exercises exhaustion before the fixed point.
    historical_dense = update_prior_aware_gauge_belief(batch, config=config)
    historical_sparse = update_sparse_prior_aware_gauge_belief(
        batch,
        tree,
        config=config,
    )
    dense = update_prior_aware_gauge_belief_v2(batch, config=config)
    sparse = update_sparse_prior_aware_gauge_belief_v2(
        batch,
        tree,
        config=config,
    )
    structured = update_sparse_prior_aware_gauge_belief_structured_v2(
        batch,
        tree,
        config=config,
    )

    for historical in (historical_dense, historical_sparse):
        assert historical.inference_admissible
        assert historical.diagnostics["mixture_fixed_point_converged"] is False
    for result in (dense, sparse, structured):
        assert not result.inference_admissible
        assert result.reason == "strict-v2-fixed-point-not-converged"
        assert result.diagnostics["underlying_inference_admissible"] is True
        assert result.diagnostics["strict_admission_passed"] is False
        assert _certificate(result)["fixed_point_converged"] is False
        np.testing.assert_array_equal(result.state_coefficients, 0.0)
        np.testing.assert_array_equal(result.gauge_delta, 0.0)
    np.testing.assert_allclose(dense.posterior_covariance, np.asarray([[0.04]]))
    np.testing.assert_allclose(sparse.posterior_covariance, np.diag([0.04, 0.09]))
    assert structured.covariance_representation == (
        PRECISION_BACKED_COVARIANCE_REPRESENTATION
    )
    assert not structured.dense_covariance_materialized
    assert structured.diagnostics["result_dense_covariance_materialized"] is False


def test_claim_bearing_entry_points_bind_strict_v2_solvers() -> None:
    assert prospective_update.update_prior_aware_gauge_belief is (
        update_prior_aware_gauge_belief_v2
    )
    assert structured_update.update_sparse_prior_aware_gauge_belief_structured is (
        update_sparse_prior_aware_gauge_belief_structured_v2
    )


def test_tree_sparse_structured_v2_admits_converged_result() -> None:
    batch, tree = _tree_fixture()
    result = update_sparse_prior_aware_gauge_belief_structured_v2(
        batch,
        tree,
        config=_converged_config(),
    )
    certificate = _certificate(result)
    assert result.inference_admissible
    assert result.diagnostics["strict_admission_passed"] is True
    assert result.diagnostics["strict_admission_reason"] == "strict-admission-passed"
    assert certificate["passed"] is True
    assert certificate["diagnostics_valid"] is True
    assert certificate["positive_exact_mixture_curvature"] is True
    assert certificate["condition_number_within_limit"] is True
    assert certificate["mixture_solution_delta"] is not None
    assert certificate["mixture_stationarity_norm"] is not None


def test_tree_sparse_structured_v2_preserves_underlying_rejection() -> None:
    batch, tree = _tree_fixture()
    config = replace(_converged_config(), maximum_state_update_m=1.0e-12)
    result = update_sparse_prior_aware_gauge_belief_structured_v2(
        batch,
        tree,
        config=config,
    )
    assert not result.inference_admissible
    assert result.diagnostics["strict_admission_reason"] == (
        "underlying-inference-rejected"
    )
    assert result.diagnostics["underlying_inference_admissible"] is False
    assert _certificate(result)["underlying_inference_admissible"] is False


def test_dense_v2_reconstructs_prior_for_malformed_underlying_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, _ = _tree_fixture()
    historical = update_prior_aware_gauge_belief(
        batch,
        config=_converged_config(),
    )
    assert historical.inference_admissible
    assert np.count_nonzero(historical.state_coefficients)
    malformed = replace(
        historical,
        inference_admissible=False,
        reason="synthetic-underlying-rejection",
    )
    monkeypatch.setattr(
        strict_v2,
        "update_prior_aware_gauge_belief",
        lambda *_args, **_kwargs: malformed,
    )

    result = update_prior_aware_gauge_belief_v2(
        batch,
        config=_converged_config(),
    )

    assert not result.inference_admissible
    assert result.reason == "synthetic-underlying-rejection"
    np.testing.assert_array_equal(result.state_coefficients, 0.0)
    np.testing.assert_allclose(result.posterior_covariance, np.asarray([[0.04]]))
    assert result.diagnostics["strict_admission_reason"] == (
        "underlying-inference-rejected"
    )


def _valid_admission_diagnostics(result: Any) -> dict[str, object]:
    diagnostics = dict(result.diagnostics)
    diagnostics.update(
        {
            "robust_likelihood_objective": "exact-group-mixture-gradient",
            "mixture_fixed_point_converged": True,
            "mixture_solution_delta": 0.0,
            "mixture_stationarity_norm": 0.0,
            "exact_reduced_mixture_hessian_minimum_eigenvalue": 1.0,
            "exact_reduced_mixture_hessian_maximum_eigenvalue": 2.0,
            "exact_reduced_mixture_hessian_positive_definite": True,
        }
    )
    return diagnostics


@pytest.mark.parametrize(
    ("updates", "expected_reason"),
    [
        (
            {"robust_likelihood_objective": "quadratic-surrogate"},
            "strict-v2-non-exact-mixture-objective",
        ),
        (
            {"mixture_fixed_point_converged": False},
            "strict-v2-fixed-point-not-converged",
        ),
        (
            {"mixture_solution_delta": -1.0},
            "strict-v2-invalid-admission-diagnostics",
        ),
        (
            {"mixture_stationarity_norm": True},
            "strict-v2-invalid-admission-diagnostics",
        ),
        (
            {
                "exact_reduced_mixture_hessian_minimum_eigenvalue": 2.0,
                "exact_reduced_mixture_hessian_maximum_eigenvalue": 1.0,
            },
            "strict-v2-invalid-admission-diagnostics",
        ),
        (
            {
                "exact_reduced_mixture_hessian_minimum_eigenvalue": 0.0,
                "exact_reduced_mixture_hessian_maximum_eigenvalue": 1.0,
                "exact_reduced_mixture_hessian_positive_definite": False,
            },
            "strict-v2-non-positive-exact-mixture-curvature",
        ),
        (
            {
                "exact_reduced_mixture_hessian_minimum_eigenvalue": 1.0e-16,
                "exact_reduced_mixture_hessian_maximum_eigenvalue": 1.0,
            },
            "strict-v2-ill-conditioned-exact-mixture-curvature",
        ),
        (
            {
                "exact_reduced_mixture_hessian_minimum_eigenvalue": 1.0e-320,
                "exact_reduced_mixture_hessian_maximum_eigenvalue": 1.0e308,
            },
            "strict-v2-invalid-admission-diagnostics",
        ),
    ],
)
def test_admission_certificate_localizes_each_strict_failure(
    updates: dict[str, object],
    expected_reason: str,
) -> None:
    batch, tree = _tree_fixture()
    historical = update_sparse_prior_aware_gauge_belief(
        batch,
        tree,
        config=_converged_config(),
    )
    assert historical.inference_admissible
    diagnostics = _valid_admission_diagnostics(historical)
    diagnostics.update(updates)
    candidate = replace(historical, diagnostics=diagnostics)

    certificate = strict_v2._build_admission_certificate(  # noqa: SLF001
        candidate,
        PriorAwareGaugeAdmissionConfigV2(),
    )

    assert not certificate.passed
    assert certificate.reason == expected_reason


def test_admission_certificate_accepts_complete_stationary_diagnostics() -> None:
    batch, tree = _tree_fixture()
    historical = update_sparse_prior_aware_gauge_belief(
        batch,
        tree,
        config=_converged_config(),
    )
    candidate = replace(
        historical,
        diagnostics=_valid_admission_diagnostics(historical),
    )

    certificate = strict_v2._build_admission_certificate(  # noqa: SLF001
        candidate,
        PriorAwareGaugeAdmissionConfigV2(),
    )

    assert certificate.passed
    assert certificate.reason == "strict-admission-passed"
    assert certificate.exact_hessian_condition_number == 2.0


def test_admission_certificate_prioritizes_underlying_rejection() -> None:
    batch, tree = _tree_fixture()
    historical = update_sparse_prior_aware_gauge_belief(
        batch,
        tree,
        config=_converged_config(),
    )
    rejected = replace(
        historical,
        inference_admissible=False,
        reason="synthetic-rejection",
        diagnostics={},
    )

    certificate = strict_v2._build_admission_certificate(  # noqa: SLF001
        rejected,
        PriorAwareGaugeAdmissionConfigV2(),
    )

    assert not certificate.passed
    assert certificate.reason == "underlying-inference-rejected"
    assert certificate.underlying_inference_reason == "synthetic-rejection"


def test_v2_result_rejects_forged_certificate_and_nonzero_fallback() -> None:
    batch, tree = _tree_fixture()
    accepted = update_sparse_prior_aware_gauge_belief_v2(
        batch,
        tree,
        config=_converged_config(),
    )
    forged_diagnostics = dict(accepted.diagnostics)
    forged_certificate = _certificate(accepted)
    forged_certificate["fixed_point_converged"] = False
    forged_diagnostics["strict_admission_certificate"] = forged_certificate
    with pytest.raises(ValueError, match="certificate pass invariant"):
        replace(accepted, diagnostics=forged_diagnostics)

    rejected = update_sparse_prior_aware_gauge_belief_v2(
        batch,
        tree,
        config=_exhausted_config(),
    )
    with pytest.raises(ValueError, match="zero candidate coefficients"):
        replace(
            rejected,
            state_coefficients=np.ones_like(rejected.state_coefficients),
        )


def test_v2_result_rejects_malformed_certificate_metadata() -> None:
    batch, tree = _tree_fixture()
    accepted = update_sparse_prior_aware_gauge_belief_v2(
        batch,
        tree,
        config=_converged_config(),
    )

    missing = dict(accepted.diagnostics)
    missing["strict_admission_certificate"] = "not-a-certificate"
    with pytest.raises(ValueError, match="do not contain an admission certificate"):
        replace(accepted, diagnostics=missing)

    wrong_schema = dict(accepted.diagnostics)
    certificate = _certificate(accepted)
    certificate["schema_version"] = 99
    wrong_schema["strict_admission_certificate"] = certificate
    with pytest.raises(ValueError, match="unsupported schema"):
        replace(accepted, diagnostics=wrong_schema)

    wrong_reason = dict(accepted.diagnostics)
    certificate = _certificate(accepted)
    certificate["reason"] = "forged"
    wrong_reason["strict_admission_certificate"] = certificate
    with pytest.raises(ValueError, match="reason invariant"):
        replace(accepted, diagnostics=wrong_reason)


def test_dense_v2_rejects_invalid_argument_types() -> None:
    batch, _ = _tree_fixture()
    with pytest.raises(TypeError, match="batch must"):
        update_prior_aware_gauge_belief_v2(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="config must"):
        update_prior_aware_gauge_belief_v2(
            batch,
            config=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="admission_config must"):
        update_prior_aware_gauge_belief_v2(
            batch,
            admission_config=object(),  # type: ignore[arg-type]
        )


def test_sparse_v2_rejects_invalid_argument_types() -> None:
    batch, tree = _tree_fixture()
    with pytest.raises(TypeError, match="batch must"):
        update_sparse_prior_aware_gauge_belief_v2(  # type: ignore[arg-type]
            object(),
            tree,
        )
    with pytest.raises(TypeError, match="gauge must"):
        update_sparse_prior_aware_gauge_belief_v2(  # type: ignore[arg-type]
            batch,
            object(),
        )
    with pytest.raises(TypeError, match="config must"):
        update_sparse_prior_aware_gauge_belief_v2(
            batch,
            tree,
            config=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="admission_config must"):
        update_sparse_prior_aware_gauge_belief_v2(
            batch,
            tree,
            admission_config=object(),  # type: ignore[arg-type]
        )


def test_tree_sparse_structured_v2_rejects_invalid_argument_types() -> None:
    batch, tree = _tree_fixture()
    with pytest.raises(TypeError, match="batch must"):
        update_sparse_prior_aware_gauge_belief_structured_v2(  # type: ignore[arg-type]
            object(),
            tree,
        )
    with pytest.raises(TypeError, match="gauge must"):
        update_sparse_prior_aware_gauge_belief_structured_v2(  # type: ignore[arg-type]
            batch,
            object(),
        )
    with pytest.raises(TypeError, match="config must"):
        update_sparse_prior_aware_gauge_belief_structured_v2(
            batch,
            tree,
            config=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="admission_config must"):
        update_sparse_prior_aware_gauge_belief_structured_v2(
            batch,
            tree,
            admission_config=object(),  # type: ignore[arg-type]
        )


def test_dense_sparse_v2_rejects_structured_fallback_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, tree = _tree_fixture()
    config = _exhausted_config()
    historical = update_sparse_prior_aware_gauge_belief(batch, tree, config=config)
    structured = update_sparse_prior_aware_gauge_belief_structured(
        batch,
        tree,
        config=config,
    )
    assert historical.inference_admissible
    monkeypatch.setattr(
        strict_v2,
        "update_sparse_prior_aware_gauge_belief",
        lambda *_args, **_kwargs: historical,
    )
    monkeypatch.setattr(
        strict_v2,
        "_sparse_fallback_result",
        lambda *_args, **_kwargs: structured,
    )
    with pytest.raises(RuntimeError, match="returned a structured result"):
        update_sparse_prior_aware_gauge_belief_v2(batch, tree, config=config)
