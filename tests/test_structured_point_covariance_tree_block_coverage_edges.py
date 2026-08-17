from __future__ import annotations

import numpy as np
import pytest

import bayesian_phystwin.tree_block_gaussian as gaussian
from bayesian_phystwin.tree_block_gaussian import TreeBlockNormalSystemV1


def _one_node_system(*, global_precision: np.ndarray) -> TreeBlockNormalSystemV1:
    global_size = len(global_precision)
    return TreeBlockNormalSystemV1(
        parent_indices=np.asarray([-1], dtype=np.int64),
        node_precision=np.asarray([[[1.0]]], dtype=np.float64),
        parent_coupling=np.zeros((1, 1, 1), dtype=np.float64),
        global_coupling=np.zeros((1, 1, global_size), dtype=np.float64),
        global_precision=global_precision,
        node_right=np.zeros((1, 1), dtype=np.float64),
        global_right=np.zeros(global_size, dtype=np.float64),
    )


def test_private_symmetric_matrix_positive_definite_validation() -> None:
    result = gaussian._symmetric_matrix(  # noqa: SLF001
        np.eye(2, dtype=np.float64),
        name="matrix",
        positive_definite=True,
    )
    np.testing.assert_array_equal(result, np.eye(2))

    with pytest.raises(ValueError, match="matrix must be positive definite"):
        gaussian._symmetric_matrix(  # noqa: SLF001
            np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64),
            name="matrix",
            positive_definite=True,
        )


def test_tree_block_elimination_rejects_ill_conditioned_node() -> None:
    system = TreeBlockNormalSystemV1(
        parent_indices=np.asarray([-1], dtype=np.int64),
        node_precision=np.asarray(
            [[[1.0, 0.0], [0.0, 1.0e-16]]],
            dtype=np.float64,
        ),
        parent_coupling=np.zeros((1, 2, 2), dtype=np.float64),
        global_coupling=np.zeros((1, 2, 1), dtype=np.float64),
        global_precision=np.eye(1, dtype=np.float64),
        node_right=np.zeros((1, 2), dtype=np.float64),
        global_right=np.zeros(1, dtype=np.float64),
    )

    with pytest.raises(
        np.linalg.LinAlgError,
        match="tree block 0 is ill-conditioned",
    ):
        system.eliminate_nodes(maximum_condition_number=1.0e12)


def test_tree_block_factorization_rejects_ill_conditioned_global_schur() -> None:
    system = _one_node_system(
        global_precision=np.asarray(
            [[1.0, 0.0], [0.0, 1.0e-16]],
            dtype=np.float64,
        )
    )
    elimination = system.eliminate_nodes(maximum_condition_number=1.0e12)

    with pytest.raises(
        np.linalg.LinAlgError,
        match="global Schur complement is ill-conditioned",
    ):
        elimination.factor_global(maximum_condition_number=1.0e12)


def test_dense_covariance_budget_edges() -> None:
    factorization = (
        _one_node_system(global_precision=np.eye(1, dtype=np.float64))
        .eliminate_nodes(maximum_condition_number=1.0e12)
        .factor_global(maximum_condition_number=1.0e12)
    )

    with pytest.raises(MemoryError, match="dense covariance requires"):
        factorization.materialize_covariance(maximum_bytes=0)
    covariance = factorization.materialize_covariance(maximum_bytes=None)
    np.testing.assert_array_equal(covariance, np.eye(2))
