from __future__ import annotations

from dataclasses import replace

import numpy as np

import bayesian_phystwin.prospective_prob4d_update as prospective_update
import bayesian_phystwin.tree_sparse_structured_gauge_prob4d as structured_update
from bayesian_phystwin._gauge_aware_contracts import GaugeAwareObservationBatch
from bayesian_phystwin._prior_aware_gauge_math import PriorAwareGaugeConfigV1
from bayesian_phystwin.prior_aware_gauge_belief_v2 import (
    update_prior_aware_gauge_belief_v2,
    update_sparse_prior_aware_gauge_belief_structured_v2,
    update_sparse_prior_aware_gauge_belief_v2,
)
from bayesian_phystwin.sparse_prior_aware_gauge_belief import (
    TreeSparseGaugeDesignV1,
    update_sparse_prior_aware_gauge_belief,
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


def test_tree_sparse_v2_rejects_exhausted_v1_fixed_point() -> None:
    batch, tree = _tree_fixture()
    config = _exhausted_config()

    historical = update_sparse_prior_aware_gauge_belief(
        batch,
        tree,
        config=config,
    )
    strict = update_sparse_prior_aware_gauge_belief_v2(
        batch,
        tree,
        config=config,
    )
    structured = update_sparse_prior_aware_gauge_belief_structured_v2(
        batch,
        tree,
        config=config,
    )

    assert historical.inference_admissible
    assert historical.diagnostics["mixture_fixed_point_converged"] is False
    for result in (strict, structured):
        assert not result.inference_admissible
        assert result.reason == "strict-v2-fixed-point-not-converged"
        assert result.diagnostics["underlying_inference_admissible"] is True
        assert result.diagnostics["strict_admission_passed"] is False
        np.testing.assert_array_equal(result.state_coefficients, 0.0)
        np.testing.assert_array_equal(result.gauge_delta, 0.0)
    np.testing.assert_allclose(strict.posterior_covariance, np.diag([0.04, 0.09]))
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
