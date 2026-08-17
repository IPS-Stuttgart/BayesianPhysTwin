from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin._gauge_aware_contracts import (
    GaugeAwareBeliefResult,
    GaugeAwareObservationBatch,
)
from bayesian_phystwin.prior_aware_gauge_belief import (
    PriorAwareGaugeConfigV1,
    update_prior_aware_gauge_belief,
)
from bayesian_phystwin.sparse_prior_aware_gauge_belief import (
    SparseGaugeDesignV1,
    update_sparse_prior_aware_gauge_belief,
)


def _dense_gauge_design(
    local: np.ndarray,
    indices: np.ndarray,
    gauge_count: int,
) -> np.ndarray:
    block_size = local.shape[2]
    result = np.zeros((len(local), 3, gauge_count * block_size))
    for gauge_index in range(gauge_count):
        selected = indices == gauge_index
        start = gauge_index * block_size
        result[selected, :, start : start + block_size] = local[selected]
    return result


def _fixture(
    *,
    include_anchor: bool = True,
    include_bias: bool = True,
) -> tuple[GaugeAwareObservationBatch, GaugeAwareObservationBatch, SparseGaugeDesignV1]:
    rng = np.random.default_rng(20260804)
    count = 12
    state_count = 3
    gauge_count = 3
    block_size = 2
    gauge_indices = np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    local_gauge = rng.normal(scale=0.25, size=(count, 3, block_size))
    state = rng.normal(scale=0.15, size=(count, 3, state_count))
    state[:, 0, 0] += np.linspace(0.8, 1.4, count)
    state[:, 1, 1] += np.linspace(1.3, 0.7, count)
    state[:, 2, 2] += np.linspace(0.5, 1.7, count)
    shared_count = 1 if include_bias else 0
    view_count = 2 if include_bias else 0
    shared = rng.normal(scale=0.12, size=(count, 3, shared_count))
    view = rng.normal(scale=0.10, size=(count, 3, view_count))

    gauge_matrix = rng.normal(size=(gauge_count * block_size,) * 2)
    gauge_prior = gauge_matrix @ gauge_matrix.T * 2.0e-5
    gauge_prior += np.eye(gauge_count * block_size) * 5.0e-5
    state_prior = np.diag([5.0e-4, 7.0e-4, 9.0e-4])
    covariance = np.empty((count, 3, 3))
    for index in range(count):
        factor = rng.normal(size=(3, 3))
        covariance[index] = factor @ factor.T * 2.0e-5 + np.eye(3) * 8.0e-5

    state_truth = np.asarray([0.004, -0.003, 0.002])
    gauge_truth = rng.normal(scale=0.0015, size=(gauge_count, block_size))
    shared_truth = rng.normal(scale=0.001, size=shared_count)
    view_truth = rng.normal(scale=0.001, size=view_count)
    innovation = np.einsum("mcs,s->mc", state, state_truth)
    innovation += np.einsum(
        "mcg,mg->mc",
        local_gauge,
        gauge_truth[gauge_indices],
    )
    if shared_count:
        innovation += np.einsum("mcb,b->mc", shared, shared_truth)
    if view_count:
        innovation += np.einsum("mcv,v->mc", view, view_truth)
    innovation += rng.normal(scale=3.0e-4, size=(count, 3))

    groups = tuple(f"group-{index // 2}" for index in range(count))
    group_nominal = np.asarray([0.97, 0.90, 0.82, 0.94, 0.86, 0.91])
    group_composite = np.asarray([0.8, 0.6, 0.9, 0.7, 0.5, 1.0])
    nominal = np.repeat(group_nominal, 2)
    composite = np.repeat(group_composite, 2)
    reliability = np.linspace(0.55, 1.0, count)
    query = np.zeros((3, 3, state_count))
    query[0, 0, 0] = 1.0
    query[1, 1, 1] = 1.0
    query[2, 2, 2] = 1.0

    if include_anchor:
        anchor_count = 4
        anchor_state = rng.normal(scale=0.2, size=(anchor_count, 3, state_count))
        anchor_state[:, 0, 0] += 1.1
        anchor_state[:, 1, 1] += 0.8
        anchor_state[:, 2, 2] += 1.4
        anchor_bias = rng.normal(scale=0.08, size=(anchor_count, 3, 1))
        anchor_bias_truth = np.asarray([0.0007])
        anchor_innovation = np.einsum("acs,s->ac", anchor_state, state_truth)
        anchor_innovation += np.einsum("acb,b->ac", anchor_bias, anchor_bias_truth)
        anchor_innovation += rng.normal(scale=2.0e-4, size=(anchor_count, 3))
        anchor_covariance = np.repeat((np.eye(3) * 6.0e-5)[None], anchor_count, axis=0)
        anchor_groups = ("anchor-0", "anchor-0", "anchor-1", "anchor-1")
        anchor_reliability = np.asarray([1.0, 0.9, 0.8, 0.75])
        anchor_nominal = np.asarray([0.96, 0.96, 0.88, 0.88])
        anchor_composite = np.asarray([0.7, 0.7, 0.9, 0.9])
        anchor_bias_prior = np.asarray([[2.5e-5]])
    else:
        anchor_innovation = None
        anchor_covariance = None
        anchor_state = None
        anchor_groups = None
        anchor_reliability = None
        anchor_nominal = None
        anchor_composite = None
        anchor_bias = None
        anchor_bias_prior = None

    common = dict(
        innovation_m=innovation,
        observation_covariance_m2=covariance,
        state_jacobian=state,
        shared_bias_jacobian=shared,
        view_bias_jacobian=view,
        query_state_jacobian=query,
        correlation_group_ids=groups,
        prior_reliability=reliability,
        prior_nominal_probability=nominal,
        composite_weight=composite,
        physical_response_scale_m=0.1,
        state_prior_covariance_m2=state_prior,
        anchor_innovation_m=anchor_innovation,
        anchor_covariance_m2=anchor_covariance,
        anchor_state_jacobian=anchor_state,
        anchor_correlation_group_ids=anchor_groups,
        anchor_prior_reliability=anchor_reliability,
        anchor_prior_nominal_probability=anchor_nominal,
        anchor_composite_weight=anchor_composite,
        anchor_bias_jacobian=anchor_bias,
        anchor_bias_prior_covariance=anchor_bias_prior,
        metadata={"fixture": "native-sparse-parity"},
    )
    dense = GaugeAwareObservationBatch(
        **common,
        gauge_jacobian=_dense_gauge_design(
            local_gauge,
            gauge_indices,
            gauge_count,
        ),
        gauge_prior_covariance=gauge_prior,
    )
    sparse_batch = GaugeAwareObservationBatch(
        **common,
        gauge_jacobian=np.zeros((count, 3, 0)),
        gauge_prior_covariance=np.zeros((0, 0)),
    )
    sparse = SparseGaugeDesignV1(
        local_gauge_jacobian=local_gauge,
        gauge_indices=gauge_indices,
        gauge_prior_covariance=gauge_prior,
        gauge_ids=("window-0", "window-1", "window-2"),
    )
    return dense, sparse_batch, sparse


def _config(**changes: object) -> PriorAwareGaugeConfigV1:
    values: dict[str, object] = {
        "minimum_conditional_information_fraction": 0.0,
        "minimum_identifiable_fraction": 1.0e-8,
        "minimum_query_sensitivity_fraction": 0.0,
        "maximum_state_update_m": 1.0,
        "maximum_update_to_physical_response_ratio": 100.0,
    }
    values.update(changes)
    return replace(PriorAwareGaugeConfigV1(), **values)


def _assert_result_parity(
    dense: GaugeAwareBeliefResult,
    sparse: GaugeAwareBeliefResult,
) -> None:
    assert dense.inference_admissible == sparse.inference_admissible
    assert dense.reason == sparse.reason
    for name in (
        "state_coefficients",
        "gauge_delta",
        "shared_bias_coefficients",
        "view_bias_coefficients",
        "anchor_bias_coefficients",
        "posterior_covariance",
        "identifiable_fractions",
        "query_sensitivity_fractions",
        "robust_weights",
        "anchor_robust_weights",
    ):
        np.testing.assert_allclose(
            getattr(sparse, name),
            getattr(dense, name),
            atol=2.0e-9,
            rtol=2.0e-8,
        )
    dense_transform = dense.identifiable_state_transform
    sparse_transform = sparse.identifiable_state_transform
    np.testing.assert_allclose(
        sparse_transform @ sparse_transform.T,
        dense_transform @ dense_transform.T,
        atol=2.0e-9,
        rtol=2.0e-8,
    )


def test_native_sparse_solver_matches_dense_reference_with_anchors() -> None:
    dense_batch, sparse_batch, sparse_design = _fixture()
    config = _config()

    dense = update_prior_aware_gauge_belief(dense_batch, config=config)
    sparse = update_sparse_prior_aware_gauge_belief(
        sparse_batch,
        sparse_design,
        config=config,
    )

    assert dense.inference_admissible
    _assert_result_parity(dense, sparse)
    assert sparse.diagnostics["native_sparse_gauge_design_materialized"] is False
    assert sparse.diagnostics["dense_gauge_design_avoided_bytes"] == (
        sparse_design.equivalent_dense_design_bytes
    )


def test_native_sparse_solver_matches_dense_reference_without_optional_nuisance() -> (
    None
):
    dense_batch, sparse_batch, sparse_design = _fixture(
        include_anchor=False,
        include_bias=False,
    )
    config = _config(minimum_robust_precision=0.15)

    dense = update_prior_aware_gauge_belief(dense_batch, config=config)
    sparse = update_sparse_prior_aware_gauge_belief(
        sparse_batch,
        sparse_design,
        config=config,
    )

    assert dense.inference_admissible
    _assert_result_parity(dense, sparse)


def _permute_observation_rows(
    batch: GaugeAwareObservationBatch,
    order: np.ndarray,
) -> GaugeAwareObservationBatch:
    return GaugeAwareObservationBatch(
        innovation_m=batch.innovation_m[order],
        observation_covariance_m2=batch.observation_covariance_m2[order],
        state_jacobian=batch.state_jacobian[order],
        gauge_jacobian=batch.gauge_jacobian[order],
        shared_bias_jacobian=batch.shared_bias_jacobian[order],
        view_bias_jacobian=batch.view_bias_jacobian[order],
        query_state_jacobian=batch.query_state_jacobian,
        gauge_prior_covariance=batch.gauge_prior_covariance,
        correlation_group_ids=tuple(
            batch.correlation_group_ids[index] for index in order
        ),
        prior_reliability=batch.prior_reliability[order],
        prior_nominal_probability=batch.prior_nominal_probability[order],
        composite_weight=batch.composite_weight[order],
        physical_response_scale_m=batch.physical_response_scale_m,
        state_prior_covariance_m2=batch.state_prior_covariance_m2,
        anchor_innovation_m=batch.anchor_innovation_m,
        anchor_covariance_m2=batch.anchor_covariance_m2,
        anchor_state_jacobian=batch.anchor_state_jacobian,
        anchor_correlation_group_ids=batch.anchor_correlation_group_ids,
        anchor_prior_reliability=batch.anchor_prior_reliability,
        anchor_prior_nominal_probability=batch.anchor_prior_nominal_probability,
        anchor_composite_weight=batch.anchor_composite_weight,
        anchor_bias_jacobian=batch.anchor_bias_jacobian,
        anchor_bias_prior_covariance=batch.anchor_bias_prior_covariance,
        metadata=batch.metadata,
        composite_weight_mode=batch.composite_weight_mode,
        anchor_composite_weight_mode=batch.anchor_composite_weight_mode,
    )


def test_native_sparse_solver_is_observation_row_permutation_invariant() -> None:
    _dense, batch, design = _fixture()
    config = _config()
    reference = update_sparse_prior_aware_gauge_belief(batch, design, config=config)
    order = np.asarray([7, 2, 11, 0, 9, 5, 1, 8, 3, 10, 4, 6])
    permuted_batch = _permute_observation_rows(batch, order)
    permuted_design = SparseGaugeDesignV1(
        local_gauge_jacobian=design.local_gauge_jacobian[order],
        gauge_indices=design.gauge_indices[order],
        gauge_prior_covariance=design.gauge_prior_covariance,
        gauge_ids=design.gauge_ids,
    )

    permuted = update_sparse_prior_aware_gauge_belief(
        permuted_batch,
        permuted_design,
        config=config,
    )

    for name in (
        "state_coefficients",
        "gauge_delta",
        "shared_bias_coefficients",
        "view_bias_coefficients",
        "anchor_bias_coefficients",
        "posterior_covariance",
    ):
        np.testing.assert_allclose(
            getattr(permuted, name),
            getattr(reference, name),
            atol=2.0e-9,
            rtol=2.0e-8,
        )
    inverse = np.argsort(order)
    np.testing.assert_allclose(
        permuted.robust_weights[inverse],
        reference.robust_weights,
        atol=2.0e-9,
        rtol=2.0e-8,
    )


def test_native_sparse_solver_is_gauge_block_permutation_invariant() -> None:
    _dense, batch, design = _fixture()
    config = _config()
    reference = update_sparse_prior_aware_gauge_belief(batch, design, config=config)
    permutation = np.asarray([2, 0, 1])
    inverse = np.argsort(permutation)
    block_size = design.block_size
    coordinate_order = np.concatenate(
        [np.arange(old * block_size, (old + 1) * block_size) for old in permutation]
    )
    permuted_design = SparseGaugeDesignV1(
        local_gauge_jacobian=design.local_gauge_jacobian,
        gauge_indices=inverse[design.gauge_indices],
        gauge_prior_covariance=design.gauge_prior_covariance[
            np.ix_(coordinate_order, coordinate_order)
        ],
        gauge_ids=tuple(design.gauge_ids[index] for index in permutation),
    )

    permuted = update_sparse_prior_aware_gauge_belief(
        batch,
        permuted_design,
        config=config,
    )

    np.testing.assert_allclose(
        permuted.state_coefficients,
        reference.state_coefficients,
        atol=2.0e-9,
        rtol=2.0e-8,
    )
    permuted_gauge = permuted.gauge_delta.reshape(design.gauge_count, block_size)
    np.testing.assert_allclose(
        permuted_gauge[inverse].reshape(-1),
        reference.gauge_delta,
        atol=2.0e-9,
        rtol=2.0e-8,
    )
    state_count = len(reference.state_coefficients)
    gauge_parameter_count = design.gauge_parameter_count
    trailing = len(reference.posterior_covariance) - state_count - gauge_parameter_count
    covariance_order = list(range(state_count))
    for old_index in range(design.gauge_count):
        new_index = int(inverse[old_index])
        covariance_order.extend(
            range(
                state_count + new_index * block_size,
                state_count + (new_index + 1) * block_size,
            )
        )
    covariance_order.extend(
        range(
            state_count + gauge_parameter_count,
            state_count + gauge_parameter_count + trailing,
        )
    )
    np.testing.assert_allclose(
        permuted.posterior_covariance[np.ix_(covariance_order, covariance_order)],
        reference.posterior_covariance,
        atol=2.0e-9,
        rtol=2.0e-8,
    )


def test_native_sparse_solver_matches_dense_no_identifiable_fallback() -> None:
    dense_batch, sparse_batch, design = _fixture()
    zeros = np.zeros_like(dense_batch.state_jacobian)
    zero_anchor = (
        None
        if dense_batch.anchor_state_jacobian is None
        else np.zeros_like(dense_batch.anchor_state_jacobian)
    )
    dense_batch = replace(
        dense_batch,
        state_jacobian=zeros,
        anchor_state_jacobian=zero_anchor,
    )
    sparse_batch = replace(
        sparse_batch,
        state_jacobian=zeros,
        anchor_state_jacobian=zero_anchor,
    )

    dense = update_prior_aware_gauge_belief(dense_batch, config=_config())
    sparse = update_sparse_prior_aware_gauge_belief(
        sparse_batch,
        design,
        config=_config(),
    )

    assert dense.reason == sparse.reason == "no-identifiable-query-state"
    _assert_result_parity(dense, sparse)


def test_native_sparse_solver_matches_dense_implausible_update_fallback() -> None:
    dense_batch, sparse_batch, design = _fixture()
    config = _config(
        maximum_state_update_m=1.0e-10,
        maximum_update_to_physical_response_ratio=1.0e-8,
    )

    dense = update_prior_aware_gauge_belief(dense_batch, config=config)
    sparse = update_sparse_prior_aware_gauge_belief(
        sparse_batch,
        design,
        config=config,
    )

    assert dense.reason == sparse.reason == "implausible-state-update"
    _assert_result_parity(dense, sparse)


def test_native_sparse_solver_fails_closed_on_ownership_and_types() -> None:
    dense_batch, sparse_batch, design = _fixture()
    with pytest.raises(TypeError, match="GaugeAwareObservationBatch"):
        update_sparse_prior_aware_gauge_belief(
            object(),
            design,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="SparseGaugeDesignV1"):
        update_sparse_prior_aware_gauge_belief(
            sparse_batch,
            object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="leave gauge ownership"):
        update_sparse_prior_aware_gauge_belief(dense_batch, design)
    short_design = SparseGaugeDesignV1(
        local_gauge_jacobian=design.local_gauge_jacobian[:-1],
        gauge_indices=design.gauge_indices[:-1],
        gauge_prior_covariance=design.gauge_prior_covariance,
        gauge_ids=design.gauge_ids,
    )
    with pytest.raises(ValueError, match="row count"):
        update_sparse_prior_aware_gauge_belief(sparse_batch, short_design)


@pytest.mark.parametrize(
    ("changes", "error", "match"),
    (
        (
            {"local_gauge_jacobian": np.zeros((3, 2, 7))},
            ValueError,
            "shape",
        ),
        (
            {"gauge_indices": np.asarray([0.0] * 12)},
            ValueError,
            "integer vector",
        ),
        (
            {"gauge_indices": np.asarray([3] * 12)},
            ValueError,
            "unknown gauge",
        ),
        (
            {"gauge_prior_covariance": np.eye(5)},
            ValueError,
            "changed shape",
        ),
        (
            {"gauge_ids": ("window-0", "window-0", "window-2")},
            ValueError,
            "unique",
        ),
        (
            {"gauge_ids": ("window-0", 1, "window-2")},
            ValueError,
            "tuple of strings",
        ),
    ),
)
def test_sparse_gauge_design_rejects_invalid_contracts(
    changes: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    _dense, _batch, design = _fixture()
    values = {
        "local_gauge_jacobian": design.local_gauge_jacobian,
        "gauge_indices": design.gauge_indices,
        "gauge_prior_covariance": design.gauge_prior_covariance,
        "gauge_ids": design.gauge_ids,
        **changes,
    }
    with pytest.raises(error, match=match):
        SparseGaugeDesignV1(**values)  # type: ignore[arg-type]


def test_sparse_gauge_design_accounts_for_large_dense_allocation_without_it() -> None:
    observation_count = 30_000
    gauge_count = 64
    block_size = 7
    design = SparseGaugeDesignV1(
        local_gauge_jacobian=np.zeros((observation_count, 3, block_size)),
        gauge_indices=np.arange(observation_count) % gauge_count,
        gauge_prior_covariance=np.eye(gauge_count * block_size),
        gauge_ids=tuple(f"window-{index}" for index in range(gauge_count)),
    )

    assert design.equivalent_dense_design_bytes > 256 * 1024 * 1024
    assert design.local_gauge_jacobian.nbytes < 8 * 1024 * 1024
    assert not design.local_gauge_jacobian.flags.writeable
    assert not design.gauge_indices.flags.writeable
