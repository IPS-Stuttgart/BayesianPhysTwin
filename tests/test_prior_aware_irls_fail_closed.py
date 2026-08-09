from __future__ import annotations

from dataclasses import replace

import numpy as np

from bayesian_phystwin._gauge_aware_contracts import GaugeAwareObservationBatch
from bayesian_phystwin.prior_aware_gauge_belief import (
    PriorAwareGaugeConfigV1,
    update_prior_aware_gauge_belief,
)
from bayesian_phystwin.sparse_prior_aware_gauge_belief import (
    SparseGaugeDesignV1,
    TreeSparseGaugeDesignV1,
    update_sparse_prior_aware_gauge_belief,
)


def _fixture() -> tuple[
    GaugeAwareObservationBatch,
    GaugeAwareObservationBatch,
    SparseGaugeDesignV1,
    TreeSparseGaugeDesignV1,
]:
    count = 4
    state = np.zeros((count, 3, 1))
    state[:, 0, 0] = 1.0
    local_gauge = np.zeros((count, 3, 1))
    innovation = np.zeros((count, 3))
    innovation[:, 0] = np.asarray([0.08, 0.10, 0.12, 0.09])
    covariance = np.repeat((np.eye(3) * 1.0e-3)[None], count, axis=0)
    query = np.zeros((1, 3, 1))
    query[0, 0, 0] = 1.0
    gauge_prior = np.asarray([[0.04]])

    common = dict(
        innovation_m=innovation,
        observation_covariance_m2=covariance,
        state_jacobian=state,
        shared_bias_jacobian=np.zeros((count, 3, 0)),
        view_bias_jacobian=np.zeros((count, 3, 0)),
        query_state_jacobian=query,
        correlation_group_ids=("group-0", "group-1", "group-2", "group-3"),
        prior_reliability=np.ones(count),
        prior_nominal_probability=np.full(count, 0.8),
        composite_weight=np.ones(count),
        physical_response_scale_m=1.0,
        state_prior_covariance_m2=np.asarray([[0.04]]),
    )
    dense = GaugeAwareObservationBatch(
        **common,
        gauge_jacobian=local_gauge,
        gauge_prior_covariance=gauge_prior,
    )
    sparse_batch = GaugeAwareObservationBatch(
        **common,
        gauge_jacobian=np.zeros((count, 3, 0)),
        gauge_prior_covariance=np.zeros((0, 0)),
    )
    sparse = SparseGaugeDesignV1(
        local_gauge_jacobian=local_gauge,
        gauge_indices=np.zeros(count, dtype=np.int64),
        gauge_prior_covariance=gauge_prior,
        gauge_ids=("window-0",),
    )
    tree = TreeSparseGaugeDesignV1(
        local_gauge_jacobian=local_gauge,
        gauge_indices=np.zeros(count, dtype=np.int64),
        parent_indices=np.asarray([-1], dtype=np.int64),
        transition_matrices=np.zeros((1, 1, 1)),
        innovation_scale_tril=np.asarray([[[0.2]]]),
        gauge_ids=("window-0",),
        prior_id="0" * 64,
    )
    return dense, sparse_batch, sparse, tree


def test_prior_aware_solvers_fail_closed_when_mixture_iterations_are_exhausted() -> None:
    dense_batch, sparse_batch, sparse_design, tree_design = _fixture()
    config = replace(
        PriorAwareGaugeConfigV1(),
        maximum_iterations=1,
        convergence_tolerance=1.0e-15,
        minimum_conditional_information_fraction=0.0,
        minimum_identifiable_fraction=1.0e-8,
        minimum_query_sensitivity_fraction=0.0,
        maximum_state_update_m=1.0,
        maximum_update_to_physical_response_ratio=100.0,
    )

    dense = update_prior_aware_gauge_belief(dense_batch, config=config)
    sparse_results = [
        update_sparse_prior_aware_gauge_belief(
            sparse_batch,
            design,
            config=config,
        )
        for design in (sparse_design, tree_design)
    ]

    for result in (dense, *sparse_results):
        assert not result.inference_admissible
        assert result.reason == "mixture-fixed-point-not-converged"
        assert result.diagnostics["iterations"] == 1
        assert result.diagnostics["mixture_fixed_point_converged"] is False
        np.testing.assert_array_equal(result.state_coefficients, np.zeros(1))
        np.testing.assert_array_equal(result.gauge_delta, np.zeros(1))
        np.testing.assert_array_equal(result.robust_weights, np.zeros(4))
        assert len(result.robust_weights) == 4

    for result in sparse_results:
        np.testing.assert_allclose(result.posterior_covariance, dense.posterior_covariance)
