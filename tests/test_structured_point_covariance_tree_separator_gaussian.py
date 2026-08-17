from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.tree_separator_gaussian import (
    TREE_SEPARATOR_GAUSSIAN_IMPLEMENTATION,
    TreeSeparatorGaussianError,
    TreeSeparatorGaussianSystemV1,
    solve_tree_separator_gaussian,
)


def _system(
    *,
    node_count: int = 7,
    block_size: int = 2,
    separator_size: int = 3,
    forest: bool = False,
    seed: int = 7,
) -> TreeSeparatorGaussianSystemV1:
    rng = np.random.default_rng(seed)
    parents = np.full(node_count, -1, dtype=np.int64)
    for index in range(1, node_count):
        parents[index] = int(rng.integers(0, index))
    if forest and node_count >= 4:
        parents[3] = -1
        for index in range(4, node_count):
            if parents[index] < 3:
                parents[index] = 3

    node_precision = np.empty(
        (node_count, block_size, block_size),
        dtype=np.float64,
    )
    parent_cross = np.zeros_like(node_precision)
    separator_cross = rng.normal(
        scale=0.15,
        size=(node_count, block_size, separator_size),
    ).astype(np.float64)
    for index in range(node_count):
        factor = rng.normal(size=(block_size, block_size))
        node_precision[index] = factor.T @ factor + 3.0 * np.eye(block_size)
        if parents[index] >= 0:
            parent_cross[index] = rng.normal(
                scale=0.15,
                size=(block_size, block_size),
            )
    separator_precision = np.eye(separator_size, dtype=np.float64) * 3.0
    node_information = rng.normal(size=(node_count, block_size)).astype(np.float64)
    separator_information = rng.normal(size=separator_size).astype(np.float64)

    provisional = TreeSeparatorGaussianSystemV1(
        parent_indices=parents,
        node_precision=node_precision,
        parent_cross_precision=parent_cross,
        separator_cross_precision=separator_cross,
        separator_precision=separator_precision,
        node_information=node_information,
        separator_information=separator_information,
    )
    dense, _ = provisional.to_dense(maximum_bytes=10_000_000)
    minimum_eigenvalue = float(np.linalg.eigvalsh(dense)[0])
    if minimum_eigenvalue <= 1.0:
        shift = 1.0 - minimum_eigenvalue
        node_precision = node_precision + shift * np.eye(block_size)
        separator_precision = separator_precision + shift * np.eye(
            separator_size, dtype=np.float64
        )
    return TreeSeparatorGaussianSystemV1(
        parent_indices=parents,
        node_precision=node_precision,
        parent_cross_precision=parent_cross,
        separator_cross_precision=separator_cross,
        separator_precision=separator_precision,
        node_information=node_information,
        separator_information=separator_information,
    )


def _assert_dense_parity(
    system: TreeSeparatorGaussianSystemV1,
) -> None:
    precision, information = system.to_dense(
        maximum_bytes=10_000_000,
    )
    expected_mean = np.linalg.solve(precision, information)
    expected_covariance = np.linalg.inv(precision)
    sign, expected_log_determinant = np.linalg.slogdet(precision)
    assert sign == 1.0

    result = solve_tree_separator_gaussian(system)
    node_dimension = system.node_count * system.block_size
    np.testing.assert_allclose(
        result.node_mean,
        expected_mean[:node_dimension].reshape(
            system.node_count,
            system.block_size,
        ),
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        result.separator_mean,
        expected_mean[node_dimension:],
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    for index in range(system.node_count):
        start = index * system.block_size
        stop = start + system.block_size
        np.testing.assert_allclose(
            result.node_covariance[index],
            expected_covariance[start:stop, start:stop],
            rtol=1.0e-11,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            result.node_separator_cross_covariance[index],
            expected_covariance[start:stop, node_dimension:],
            rtol=1.0e-11,
            atol=1.0e-12,
        )
    np.testing.assert_allclose(
        result.separator_covariance,
        expected_covariance[node_dimension:, node_dimension:],
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    assert result.log_determinant_precision == pytest.approx(
        expected_log_determinant,
        rel=1.0e-12,
        abs=1.0e-12,
    )


def test_solver_matches_dense_for_branched_tree_with_separator() -> None:
    system = _system()
    _assert_dense_parity(system)
    result = solve_tree_separator_gaussian(system)

    assert result.implementation == TREE_SEPARATOR_GAUSSIAN_IMPLEMENTATION
    assert result.node_count == system.node_count
    assert result.block_size == system.block_size
    assert result.separator_size == system.separator_size
    assert result.stored_nbytes > 0
    assert not result.node_mean.flags.writeable
    assert not result.node_covariance.flags.writeable
    assert not result.separator_covariance.flags.writeable

    mean, covariance, separator_cross = result.node_marginal(2)
    np.testing.assert_array_equal(mean, result.node_mean[2])
    np.testing.assert_array_equal(covariance, result.node_covariance[2])
    np.testing.assert_array_equal(
        separator_cross,
        result.node_separator_cross_covariance[2],
    )
    mean[0] += 1.0
    assert mean[0] != result.node_mean[2, 0]


def test_solver_matches_dense_for_forest_without_separator() -> None:
    system = _system(
        separator_size=0,
        forest=True,
    )
    _assert_dense_parity(system)
    result = solve_tree_separator_gaussian(system)

    assert result.separator_mean.shape == (0,)
    assert result.separator_covariance.shape == (0, 0)
    assert result.node_separator_cross_covariance.shape == (
        system.node_count,
        system.block_size,
        0,
    )


def test_sparse_storage_and_dense_budget_scale_differently() -> None:
    node_count = 1_024
    block_size = 2
    separator_size = 3
    parents = np.arange(-1, node_count - 1, dtype=np.int64)
    node_precision = np.repeat(
        np.eye(block_size, dtype=np.float64)[None],
        node_count,
        axis=0,
    )
    parent_cross = np.zeros_like(node_precision)
    separator_cross = np.zeros(
        (node_count, block_size, separator_size),
        dtype=np.float64,
    )
    system = TreeSeparatorGaussianSystemV1(
        parent_indices=parents,
        node_precision=node_precision,
        parent_cross_precision=parent_cross,
        separator_cross_precision=separator_cross,
        separator_precision=np.eye(
            separator_size,
            dtype=np.float64,
        ),
        node_information=np.zeros(
            (node_count, block_size),
            dtype=np.float64,
        ),
        separator_information=np.zeros(
            separator_size,
            dtype=np.float64,
        ),
    )

    assert system.dimension == node_count * block_size + separator_size
    assert system.estimated_dense_precision_bytes > 100 * system.stored_nbytes
    with pytest.raises(MemoryError, match="dense precision requires"):
        system.to_dense(maximum_bytes=system.stored_nbytes)
    with pytest.raises(TypeError, match="must be an integer"):
        system.to_dense(maximum_bytes=True)
    with pytest.raises(TypeError, match="must be an integer"):
        system.to_dense(maximum_bytes=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be non-negative"):
        system.to_dense(maximum_bytes=-1)


@pytest.mark.parametrize(
    ("field_name", "replacement", "message"),
    [
        (
            "parent_indices",
            np.asarray([[-1]], dtype=np.int64),
            "rank 1",
        ),
        (
            "parent_indices",
            np.asarray([-1.0], dtype=np.float64),
            "dtype int64",
        ),
        (
            "node_precision",
            np.ones((1, 1, 1), dtype=np.float32),
            "dtype float64",
        ),
        (
            "node_precision",
            np.asarray([[[np.nan]]], dtype=np.float64),
            "finite values",
        ),
        (
            "node_precision",
            np.asarray([[[1.0, 1.0], [0.0, 1.0]]]),
            "must be symmetric",
        ),
        (
            "node_precision",
            np.zeros((0, 1, 1), dtype=np.float64),
            "node count is inconsistent",
        ),
        (
            "node_precision",
            np.zeros((1, 0, 0), dtype=np.float64),
            "nonempty and square",
        ),
        (
            "parent_cross_precision",
            np.zeros((1, 2, 1), dtype=np.float64),
            "incompatible shape",
        ),
        (
            "separator_cross_precision",
            np.zeros((1, 2, 2), dtype=np.float64),
            "incompatible shape",
        ),
        (
            "separator_precision",
            np.zeros((1, 2), dtype=np.float64),
            "must be square",
        ),
        (
            "node_information",
            np.zeros((1, 1), dtype=np.float64),
            "incompatible shape",
        ),
        (
            "separator_information",
            np.zeros(2, dtype=np.float64),
            "incompatible shape",
        ),
    ],
)
def test_system_rejects_invalid_arrays(
    field_name: str,
    replacement: np.ndarray,
    message: str,
) -> None:
    system = _system(node_count=1, block_size=2, separator_size=1)
    values = {
        "parent_indices": system.parent_indices,
        "node_precision": system.node_precision,
        "parent_cross_precision": system.parent_cross_precision,
        "separator_cross_precision": system.separator_cross_precision,
        "separator_precision": system.separator_precision,
        "node_information": system.node_information,
        "separator_information": system.separator_information,
    }
    values[field_name] = replacement
    with pytest.raises(ValueError, match=message):
        TreeSeparatorGaussianSystemV1(**values)


def test_system_rejects_empty_noncanonical_and_root_cross_trees() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        TreeSeparatorGaussianSystemV1(
            parent_indices=np.zeros(0, dtype=np.int64),
            node_precision=np.zeros((0, 1, 1), dtype=np.float64),
            parent_cross_precision=np.zeros((0, 1, 1), dtype=np.float64),
            separator_cross_precision=np.zeros((0, 1, 0), dtype=np.float64),
            separator_precision=np.zeros((0, 0), dtype=np.float64),
            node_information=np.zeros((0, 1), dtype=np.float64),
            separator_information=np.zeros(0, dtype=np.float64),
        )

    system = _system(node_count=2, separator_size=0)
    with pytest.raises(ValueError, match="parents before children"):
        replace(
            system,
            parent_indices=np.asarray([-1, 1], dtype=np.int64),
        )
    root_cross = np.array(system.parent_cross_precision, copy=True)
    root_cross[0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="root parent-cross"):
        replace(system, parent_cross_precision=root_cross)


def test_asymmetric_separator_and_indefinite_systems_fail_closed() -> None:
    system = _system(node_count=1, separator_size=2)
    asymmetric = np.asarray(
        [[2.0, 1.0], [0.0, 2.0]],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="separator_precision must be symmetric"):
        replace(system, separator_precision=asymmetric)

    no_separator = _system(node_count=1, separator_size=0)
    with pytest.raises(
        TreeSeparatorGaussianError,
        match="eliminated node precision 0 is not positive definite",
    ):
        solve_tree_separator_gaussian(
            replace(
                no_separator,
                node_precision=np.asarray([[[-1.0, 0.0], [0.0, 1.0]]]),
            )
        )

    separator_system = _system(node_count=1, block_size=1, separator_size=1)
    with pytest.raises(
        TreeSeparatorGaussianError,
        match="separator Schur complement is not positive definite",
    ):
        solve_tree_separator_gaussian(
            replace(
                separator_system,
                separator_cross_precision=np.zeros(
                    (1, 1, 1),
                    dtype=np.float64,
                ),
                separator_precision=np.asarray([[-1.0]], dtype=np.float64),
            )
        )


def test_public_type_and_index_guards() -> None:
    with pytest.raises(TypeError, match="system must"):
        solve_tree_separator_gaussian(object())  # type: ignore[arg-type]

    result = solve_tree_separator_gaussian(_system(node_count=2))
    with pytest.raises(TypeError, match="node_index must be an integer"):
        result.node_marginal(True)
    with pytest.raises(TypeError, match="node_index must be an integer"):
        result.node_marginal(1.5)  # type: ignore[arg-type]
    with pytest.raises(IndexError, match="out of range"):
        result.node_marginal(-1)
    with pytest.raises(IndexError, match="out of range"):
        result.node_marginal(result.node_count)
