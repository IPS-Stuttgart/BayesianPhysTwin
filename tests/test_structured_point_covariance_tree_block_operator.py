from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from bayesian_phystwin.tree_block_gaussian import (
    TreeBlockFactorizationV1,
    TreeBlockNormalSystemV1,
)
from bayesian_phystwin.tree_block_posterior_operator import (
    TREE_BLOCK_POSTERIOR_OPERATOR_BOUNDARY,
    TREE_BLOCK_POSTERIOR_OPERATOR_SCHEMA,
    TREE_BLOCK_POSTERIOR_OPERATOR_VERSION,
    TreeBlockPosteriorOperatorV1,
    apply_tree_block_posterior_covariance,
    tree_block_cross_covariance,
    tree_block_linear_query_covariance,
    tree_block_selected_marginal_covariance,
)
from bayesian_phystwin.tree_block_sparse_gauge_belief import (
    TreeBlockPosteriorCovarianceV1,
)


def _factorization(
    *,
    node_count: int = 3,
    block_size: int = 2,
    global_size: int = 4,
) -> TreeBlockFactorizationV1:
    parents = np.asarray(
        [-1] + [0 if index < 3 else index - 1 for index in range(1, node_count)],
        dtype=np.int64,
    )
    node_precision = np.repeat(
        (np.eye(block_size, dtype=np.float64) * 6.0)[None],
        node_count,
        axis=0,
    )
    parent_coupling = np.zeros_like(node_precision)
    for index in range(1, node_count):
        parent_coupling[index] = np.eye(block_size) * (0.05 + 0.01 * index)
        if block_size >= 2:
            parent_coupling[index, 0, 1] = 0.012
            parent_coupling[index, 1, 0] = -0.008
    global_coupling = np.zeros(
        (node_count, block_size, global_size),
        dtype=np.float64,
    )
    for index in range(node_count):
        for row in range(block_size):
            for column in range(global_size):
                global_coupling[index, row, column] = (
                    0.006 * (index + 1) * (row + 1) / (column + 1)
                )
    system = TreeBlockNormalSystemV1(
        parent_indices=parents,
        node_precision=node_precision,
        parent_coupling=parent_coupling,
        global_coupling=global_coupling,
        global_precision=np.eye(global_size, dtype=np.float64) * 8.0,
        node_right=np.zeros((node_count, block_size), dtype=np.float64),
        global_right=np.zeros(global_size, dtype=np.float64),
    )
    return system.eliminate_nodes(
        maximum_condition_number=1.0e12
    ).factor_global(maximum_condition_number=1.0e12)


def _covariance(
    *,
    node_count: int = 3,
    block_size: int = 2,
    bias_count: int = 2,
) -> TreeBlockPosteriorCovarianceV1:
    state_prior = np.asarray(
        [
            [4.0e-4, 1.0e-4, 0.0],
            [1.0e-4, 9.0e-4, 1.0e-4],
            [0.0, 1.0e-4, 1.6e-3],
        ],
        dtype=np.float64,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(state_prior)
    retained = 2
    mapping = eigenvectors[:, -retained:] * np.sqrt(eigenvalues[-retained:])
    return TreeBlockPosteriorCovarianceV1(
        state_prior_covariance=state_prior,
        state_mapping=mapping,
        factorization=_factorization(
            node_count=node_count,
            block_size=block_size,
            global_size=retained + bias_count,
        ),
        bias_count=bias_count,
    )


def _assert_irreversibly_readonly(value: np.ndarray) -> None:
    assert not value.flags.writeable
    with pytest.raises(ValueError):
        value.setflags(write=True)


def test_operator_applies_vector_and_matrix_with_dense_parity() -> None:
    covariance = _covariance()
    dense = covariance.materialize()
    operator = TreeBlockPosteriorOperatorV1(covariance)
    rng = np.random.default_rng(14)
    vector = rng.normal(size=covariance.dimension)
    matrix = rng.normal(size=(covariance.dimension, 5))

    vector_result = operator.apply(vector)
    matrix_result = operator.apply(matrix)
    np.testing.assert_allclose(vector_result, dense @ vector, rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(matrix_result, dense @ matrix, rtol=2e-11, atol=2e-12)
    assert vector_result.shape == vector.shape
    assert matrix_result.shape == matrix.shape
    _assert_irreversibly_readonly(vector_result)
    _assert_irreversibly_readonly(matrix_result)

    np.testing.assert_allclose(
        apply_tree_block_posterior_covariance(covariance, matrix),
        matrix_result,
        rtol=0.0,
        atol=0.0,
    )
    assert operator.schema == TREE_BLOCK_POSTERIOR_OPERATOR_SCHEMA
    assert operator.schema_version == TREE_BLOCK_POSTERIOR_OPERATOR_VERSION
    assert operator.dimension == covariance.dimension
    assert operator.state_count == covariance.state_count
    assert operator.gauge_parameter_count == covariance.gauge_parameter_count
    assert operator.bias_count == covariance.bias_count
    assert operator.dense_covariance_avoided_bytes == (
        covariance.estimated_dense_covariance_bytes
    )
    assert "deployment safety" in TREE_BLOCK_POSTERIOR_OPERATOR_BOUNDARY


def test_linear_cross_and_selected_covariances_match_dense_reference() -> None:
    covariance = _covariance()
    dense = covariance.materialize()
    operator = TreeBlockPosteriorOperatorV1(covariance)
    rng = np.random.default_rng(19)
    query = rng.normal(size=(4, covariance.dimension))
    left = rng.normal(size=(3, covariance.dimension))
    right = rng.normal(size=(5, covariance.dimension))
    indices = np.asarray([0, 4, covariance.dimension - 1, 2], dtype=np.int64)

    expected_query = query @ dense @ query.T
    expected_cross = left @ dense @ right.T
    expected_marginal = dense[np.ix_(indices, indices)]
    query_result = operator.linear_covariance(query)
    cross_result = operator.cross_covariance(left, right)
    marginal_result = operator.marginal_covariance(indices)

    np.testing.assert_allclose(query_result, expected_query, rtol=3e-11, atol=3e-12)
    np.testing.assert_allclose(cross_result, expected_cross, rtol=3e-11, atol=3e-12)
    np.testing.assert_allclose(
        marginal_result,
        expected_marginal,
        rtol=3e-11,
        atol=3e-12,
    )
    np.testing.assert_allclose(query_result, query_result.T, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        marginal_result,
        marginal_result.T,
        atol=0.0,
        rtol=0.0,
    )
    _assert_irreversibly_readonly(query_result)
    _assert_irreversibly_readonly(cross_result)
    _assert_irreversibly_readonly(marginal_result)

    np.testing.assert_allclose(
        tree_block_linear_query_covariance(covariance, query),
        expected_query,
        rtol=3e-11,
        atol=3e-12,
    )
    np.testing.assert_allclose(
        tree_block_cross_covariance(covariance, left, right),
        expected_cross,
        rtol=3e-11,
        atol=3e-12,
    )
    np.testing.assert_allclose(
        tree_block_selected_marginal_covariance(covariance, indices),
        expected_marginal,
        rtol=3e-11,
        atol=3e-12,
    )


def test_operator_preserves_covariance_identity_and_public_order() -> None:
    covariance = _covariance()
    descriptor_before = dict(covariance.descriptor())
    dense = covariance.materialize()
    operator = TreeBlockPosteriorOperatorV1(covariance)

    for index in range(covariance.dimension):
        basis = np.zeros(covariance.dimension, dtype=np.float64)
        basis[index] = 1.0
        np.testing.assert_allclose(
            operator.apply(basis),
            dense[:, index],
            rtol=2e-11,
            atol=2e-12,
        )
    assert dict(covariance.descriptor()) == descriptor_before


def test_operator_handles_no_bias_variables() -> None:
    covariance = _covariance(bias_count=0)
    dense = covariance.materialize()
    rng = np.random.default_rng(22)
    right = rng.normal(size=(covariance.dimension, 2))
    operator = TreeBlockPosteriorOperatorV1(covariance)

    np.testing.assert_allclose(
        operator.apply(right),
        dense @ right,
        rtol=2e-11,
        atol=2e-12,
    )


def test_large_tree_queries_do_not_call_dense_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node_count = 1_024
    block_size = 2
    state_prior = np.diag(np.asarray([2.5e-4, 4.0e-4]))
    covariance = TreeBlockPosteriorCovarianceV1(
        state_prior_covariance=state_prior,
        state_mapping=np.diag(np.sqrt(np.diag(state_prior))),
        factorization=_factorization(
            node_count=node_count,
            block_size=block_size,
            global_size=2,
        ),
        bias_count=0,
    )
    operator = TreeBlockPosteriorOperatorV1(covariance)

    def forbidden(*args: object, **kwargs: object) -> np.ndarray:
        del args, kwargs
        raise AssertionError("dense covariance materialization was attempted")

    monkeypatch.setattr(TreeBlockFactorizationV1, "materialize_covariance", forbidden)
    rng = np.random.default_rng(31)
    right = rng.normal(size=(covariance.dimension, 3))
    query = rng.normal(size=(3, covariance.dimension))
    selected = np.asarray([0, 3, covariance.dimension - 1], dtype=np.int64)

    applied = operator.apply(right)
    query_covariance = operator.linear_covariance(query)
    marginal = operator.marginal_covariance(selected)
    assert applied.shape == right.shape
    assert query_covariance.shape == (3, 3)
    assert marginal.shape == (3, 3)
    assert covariance.estimated_dense_covariance_bytes > 100 * covariance.stored_nbytes


@pytest.mark.parametrize(
    ("right", "message"),
    [
        (np.asarray([True]), "must be numeric"),
        (np.asarray(["1.0"]), "must be numeric"),
        (np.asarray([np.nan]), "must be finite"),
        (np.zeros((1, 1, 1)), "right must have shape"),
    ],
)
def test_apply_rejects_invalid_numeric_inputs(right: object, message: str) -> None:
    operator = TreeBlockPosteriorOperatorV1(_covariance())
    with pytest.raises(ValueError, match=message):
        operator.apply(right)


def test_apply_rejects_wrong_dimension_and_empty_matrix() -> None:
    covariance = _covariance()
    operator = TreeBlockPosteriorOperatorV1(covariance)
    with pytest.raises(ValueError, match="right must have shape"):
        operator.apply(np.zeros(covariance.dimension - 1))
    with pytest.raises(ValueError, match="at least one column"):
        operator.apply(np.zeros((covariance.dimension, 0)))


@pytest.mark.parametrize(
    ("method", "value", "message"),
    [
        ("linear", np.asarray([True]), "query must be numeric"),
        ("linear", np.asarray([np.nan]), "query must be finite"),
        ("linear", np.zeros(3), "query must have two dimensions"),
        ("linear", np.zeros((1, 3)), "query must have shape"),
        ("linear", np.zeros((0, 1)), "query must have shape"),
        ("cross-left", np.zeros(3), "left_query must have two dimensions"),
        ("cross-right", np.zeros(3), "right_query must have two dimensions"),
    ],
)
def test_query_methods_reject_invalid_inputs(
    method: str,
    value: object,
    message: str,
) -> None:
    covariance = _covariance()
    operator = TreeBlockPosteriorOperatorV1(covariance)
    valid = np.zeros((1, covariance.dimension), dtype=np.float64)
    with pytest.raises(ValueError, match=message):
        if method == "linear":
            operator.linear_covariance(value)
        elif method == "cross-left":
            operator.cross_covariance(value, valid)
        else:
            operator.cross_covariance(valid, value)


def test_query_methods_reject_empty_rows_after_dimension_validation() -> None:
    covariance = _covariance()
    operator = TreeBlockPosteriorOperatorV1(covariance)
    empty = np.zeros((0, covariance.dimension), dtype=np.float64)
    with pytest.raises(ValueError, match="at least one row"):
        operator.linear_covariance(empty)
    with pytest.raises(ValueError, match="at least one row"):
        operator.cross_covariance(empty, np.zeros((1, covariance.dimension)))


@pytest.mark.parametrize(
    ("indices", "message"),
    [
        (np.asarray([0.0]), "integer vector"),
        (np.asarray([True]), "integer vector"),
        (np.zeros((1, 1), dtype=np.int64), "integer vector"),
        (np.zeros(0, dtype=np.int64), "must not be empty"),
        (np.asarray([-1], dtype=np.int64), "out-of-range"),
        (np.asarray([0, 0], dtype=np.int64), "must not contain duplicates"),
    ],
)
def test_selected_marginal_rejects_invalid_indices(
    indices: object,
    message: str,
) -> None:
    operator = TreeBlockPosteriorOperatorV1(_covariance())
    with pytest.raises(ValueError, match=message):
        operator.marginal_covariance(indices)


def test_selected_marginal_rejects_upper_out_of_range() -> None:
    covariance = _covariance()
    with pytest.raises(ValueError, match="out-of-range"):
        TreeBlockPosteriorOperatorV1(covariance).marginal_covariance(
            np.asarray([covariance.dimension], dtype=np.int64)
        )


def test_operator_rejects_non_covariance_and_forged_factor() -> None:
    with pytest.raises(TypeError, match="TreeBlockPosteriorCovarianceV1"):
        TreeBlockPosteriorOperatorV1(object())  # type: ignore[arg-type]

    covariance = _covariance()
    forged_factor = replace(
        covariance.factorization,
        global_condition_number=(
            covariance.factorization.global_condition_number * 2.0
        ),
    )
    forged_covariance = replace(covariance, factorization=forged_factor)
    with pytest.raises(ValueError, match="global condition number does not match"):
        TreeBlockPosteriorOperatorV1(forged_covariance)


def test_wrapper_argument_type_is_validated() -> None:
    with pytest.raises(TypeError, match="TreeBlockPosteriorCovarianceV1"):
        apply_tree_block_posterior_covariance(
            object(),  # type: ignore[arg-type]
            np.zeros(1),
        )
