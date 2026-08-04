from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin._gauge_aware_contracts import (
    COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,
    GaugeAwareBeliefResult,
    GaugeAwareObservationBatch,
)
from bayesian_phystwin.prior_aware_gauge_belief import (
    PriorAwareGaugeConfigV1,
    update_prior_aware_gauge_belief,
)
from bayesian_phystwin.prior_aware_sparse_gauge_belief import (
    SparseGaugeAwareObservationBatch,
    update_prior_aware_sparse_gauge_belief,
)


def _random_batches() -> tuple[
    GaugeAwareObservationBatch,
    SparseGaugeAwareObservationBatch,
]:
    rng = np.random.default_rng(20260804)
    row_count = 18
    state_count = 4
    gauge_block_size = 3
    gauge_group_count = 3
    gauge_count = gauge_block_size * gauge_group_count
    shared_count = 2
    view_count = 2
    anchor_count = 4

    covariance_factor = rng.normal(size=(row_count, 3, 3)) * 8e-4
    covariance = (
        np.einsum("mij,mkj->mik", covariance_factor, covariance_factor)
        + np.eye(3)[None] * 4e-6
    )
    state = rng.normal(size=(row_count, 3, state_count)) * 0.2
    local_gauge = rng.normal(size=(row_count, 3, gauge_block_size)) * 0.15
    gauge_indices = np.arange(row_count, dtype=np.int64) % gauge_group_count
    dense_gauge = np.zeros((row_count, 3, gauge_count), dtype=np.float64)
    for gauge_index in range(gauge_group_count):
        selected = gauge_indices == gauge_index
        start = gauge_index * gauge_block_size
        dense_gauge[selected, :, start : start + gauge_block_size] = local_gauge[
            selected
        ]
    shared = rng.normal(size=(row_count, 3, shared_count)) * 0.1
    view = rng.normal(size=(row_count, 3, view_count)) * 0.1
    query = rng.normal(size=(5, 3, state_count)) * 0.2
    prior_factor = rng.normal(size=(gauge_count, gauge_count)) * 4e-4
    gauge_prior = prior_factor @ prior_factor.T + np.eye(gauge_count) * 5e-7
    innovation = rng.normal(size=(row_count, 3)) * 8e-4

    groups = tuple(f"group-{index // 3}" for index in range(row_count))
    group_nominal = np.asarray([0.97, 0.91, 0.85, 0.79, 0.73, 0.67])
    nominal = np.repeat(group_nominal, 3)
    group_composite = np.asarray([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    composite = np.repeat(group_composite, 3)
    reliability = rng.uniform(0.35, 0.95, size=row_count)

    anchor_covariance_factor = rng.normal(size=(anchor_count, 3, 3)) * 6e-4
    anchor_covariance = (
        np.einsum("aij,akj->aik", anchor_covariance_factor, anchor_covariance_factor)
        + np.eye(3)[None] * 3e-6
    )
    anchor_state = rng.normal(size=(anchor_count, 3, state_count)) * 0.2
    anchor_bias = rng.normal(size=(anchor_count, 3, 1)) * 0.1
    anchor_innovation = rng.normal(size=(anchor_count, 3)) * 5e-4
    anchor_groups = ("anchor-a", "anchor-a", "anchor-b", "anchor-b")
    anchor_nominal = np.asarray([0.9, 0.9, 0.8, 0.8])
    anchor_composite = np.asarray([1.0, 1.0, 0.7, 0.7])

    common = {
        "innovation_m": innovation,
        "observation_covariance_m2": covariance,
        "state_jacobian": state,
        "shared_bias_jacobian": shared,
        "view_bias_jacobian": view,
        "query_state_jacobian": query,
        "correlation_group_ids": groups,
        "prior_reliability": reliability,
        "prior_nominal_probability": nominal,
        "composite_weight": composite,
        "physical_response_scale_m": 0.05,
        "state_prior_covariance_m2": np.eye(state_count) * 3e-4,
        "anchor_innovation_m": anchor_innovation,
        "anchor_covariance_m2": anchor_covariance,
        "anchor_state_jacobian": anchor_state,
        "anchor_correlation_group_ids": anchor_groups,
        "anchor_prior_reliability": np.asarray([0.8, 0.7, 0.9, 0.6]),
        "anchor_prior_nominal_probability": anchor_nominal,
        "anchor_composite_weight": anchor_composite,
        "anchor_bias_jacobian": anchor_bias,
        "anchor_bias_prior_covariance": np.asarray([[2e-5]]),
        "composite_weight_mode": COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,
        "anchor_composite_weight_mode": COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,
        "metadata": {"fixture": "random-correlated"},
    }
    dense = GaugeAwareObservationBatch(
        **common,
        gauge_jacobian=dense_gauge,
        gauge_prior_covariance=gauge_prior,
    )
    base = GaugeAwareObservationBatch(
        **common,
        gauge_jacobian=np.zeros((row_count, 3, 0)),
        gauge_prior_covariance=np.zeros((0, 0)),
    )
    sparse = SparseGaugeAwareObservationBatch(
        base=base,
        local_gauge_jacobian=local_gauge,
        gauge_indices=gauge_indices,
        gauge_prior_covariance=gauge_prior,
        gauge_block_size=gauge_block_size,
    )
    return dense, sparse


def _assert_results_close(
    sparse: GaugeAwareBeliefResult,
    dense: GaugeAwareBeliefResult,
) -> None:
    assert sparse.inference_admissible is dense.inference_admissible
    assert sparse.reason == dense.reason
    for name in (
        "state_coefficients",
        "gauge_delta",
        "shared_bias_coefficients",
        "view_bias_coefficients",
        "anchor_bias_coefficients",
        "posterior_covariance",
        "identifiable_state_transform",
        "identifiable_fractions",
        "query_sensitivity_fractions",
        "robust_weights",
        "anchor_robust_weights",
    ):
        np.testing.assert_allclose(
            getattr(sparse, name),
            getattr(dense, name),
            rtol=2e-9,
            atol=2e-11,
        )


def test_native_sparse_solver_matches_dense_random_correlated_problem() -> None:
    dense_batch, sparse_batch = _random_batches()
    config = PriorAwareGaugeConfigV1(
        minimum_conditional_information_fraction=0.0,
        minimum_identifiable_fraction=0.01,
        minimum_query_sensitivity_fraction=0.0,
        maximum_state_update_m=0.5,
        maximum_update_to_physical_response_ratio=10.0,
    )

    dense = update_prior_aware_gauge_belief(dense_batch, config=config)
    sparse = update_prior_aware_sparse_gauge_belief(sparse_batch, config=config)

    _assert_results_close(sparse, dense)


def test_native_sparse_solver_matches_dense_trust_region_fallback() -> None:
    dense_batch, sparse_batch = _random_batches()
    dense_batch = replace(dense_batch, physical_response_scale_m=1e-9)
    sparse_batch = replace(
        sparse_batch,
        base=replace(sparse_batch.base, physical_response_scale_m=1e-9),
    )

    dense = update_prior_aware_gauge_belief(dense_batch)
    sparse = update_prior_aware_sparse_gauge_belief(sparse_batch)

    _assert_results_close(sparse, dense)
    assert sparse.inference_admissible is False
    assert sparse.reason == "implausible-state-update"


def test_sparse_batch_rejects_noninteger_or_unknown_gauge_indices() -> None:
    _, batch = _random_batches()
    with pytest.raises(ValueError, match="integer vector"):
        replace(batch, gauge_indices=batch.gauge_indices.astype(np.float64))
    changed = np.asarray(batch.gauge_indices).copy()
    changed[0] = batch.gauge_group_count
    with pytest.raises(ValueError, match="unknown local block"):
        replace(batch, gauge_indices=changed)


def test_sparse_batch_rejects_incomplete_or_nonsymmetric_joint_prior() -> None:
    _, batch = _random_batches()
    with pytest.raises(ValueError, match="complete local blocks"):
        replace(
            batch,
            gauge_prior_covariance=np.eye(batch.gauge_parameter_count + 1),
        )
    changed = np.asarray(batch.gauge_prior_covariance).copy()
    changed[0, 1] += 1e-4
    with pytest.raises(ValueError, match="symmetric"):
        replace(batch, gauge_prior_covariance=changed)
