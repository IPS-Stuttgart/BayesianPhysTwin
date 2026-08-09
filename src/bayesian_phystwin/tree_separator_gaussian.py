"""Exact block-tree Gaussian elimination with a small dense separator.

The solver stores one fixed-size block per tree node and a small global separator.
It never constructs the complete tree covariance or precision matrix during
``solve_tree_separator_gaussian``. A budgeted dense conversion is retained only
for compatibility tests and small diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Final

import numpy as np

TREE_SEPARATOR_GAUSSIAN_IMPLEMENTATION: Final = "block-tree-schur-message-passing-v1"


class TreeSeparatorGaussianError(ValueError):
    """Raised when a block-tree Gaussian system cannot be factored safely."""


def _readonly_float64(
    value: object,
    *,
    name: str,
    ndim: int,
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float64:
        raise ValueError(f"{name} must have dtype float64")
    if array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    result = np.array(array, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def _readonly_int64(
    value: object,
    *,
    name: str,
    ndim: int,
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.int64:
        raise ValueError(f"{name} must have dtype int64")
    if array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    result = np.array(array, dtype=np.int64, order="C", copy=True)
    result.setflags(write=False)
    return result


def _symmetric(value: np.ndarray, *, name: str) -> None:
    if not np.allclose(
        value,
        np.swapaxes(value, -1, -2),
        rtol=1.0e-12,
        atol=1.0e-14,
    ):
        raise ValueError(f"{name} must be symmetric")


def _solve_cholesky(cholesky: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    intermediate = np.linalg.solve(cholesky, rhs)
    return np.linalg.solve(cholesky.T, intermediate)


def _cholesky(value: np.ndarray, *, name: str) -> np.ndarray:
    symmetric = 0.5 * (value + value.T)
    try:
        return np.linalg.cholesky(symmetric)
    except np.linalg.LinAlgError as error:
        raise TreeSeparatorGaussianError(f"{name} is not positive definite") from error


@dataclass(frozen=True, slots=True)
class TreeSeparatorGaussianSystemV1:
    """Sparse precision and information vector for a canonical block forest.

    ``parent_cross_precision[i]`` is the precision block in row ``i`` and the
    column of its parent. Roots use parent index ``-1`` and must have a zero
    parent-cross block. Parents must precede children, which makes reverse index
    order a valid leaf-to-root elimination schedule.
    """

    parent_indices: np.ndarray
    node_precision: np.ndarray
    parent_cross_precision: np.ndarray
    separator_cross_precision: np.ndarray
    separator_precision: np.ndarray
    node_information: np.ndarray
    separator_information: np.ndarray

    def __post_init__(self) -> None:
        parents = _readonly_int64(
            self.parent_indices,
            name="parent_indices",
            ndim=1,
        )
        node_precision = _readonly_float64(
            self.node_precision,
            name="node_precision",
            ndim=3,
        )
        parent_cross = _readonly_float64(
            self.parent_cross_precision,
            name="parent_cross_precision",
            ndim=3,
        )
        separator_cross = _readonly_float64(
            self.separator_cross_precision,
            name="separator_cross_precision",
            ndim=3,
        )
        separator_precision = _readonly_float64(
            self.separator_precision,
            name="separator_precision",
            ndim=2,
        )
        node_information = _readonly_float64(
            self.node_information,
            name="node_information",
            ndim=2,
        )
        separator_information = _readonly_float64(
            self.separator_information,
            name="separator_information",
            ndim=1,
        )

        node_count = len(parents)
        if node_count == 0:
            raise ValueError("parent_indices must not be empty")
        if node_precision.shape[0] != node_count:
            raise ValueError("node_precision node count is inconsistent")
        block_size = node_precision.shape[1]
        if block_size == 0 or node_precision.shape[2] != block_size:
            raise ValueError("node_precision blocks must be nonempty and square")
        if parent_cross.shape != (node_count, block_size, block_size):
            raise ValueError("parent_cross_precision has an incompatible shape")
        if node_information.shape != (node_count, block_size):
            raise ValueError("node_information has an incompatible shape")
        if separator_precision.shape[0] != separator_precision.shape[1]:
            raise ValueError("separator_precision must be square")
        separator_size = separator_precision.shape[0]
        if separator_cross.shape != (
            node_count,
            block_size,
            separator_size,
        ):
            raise ValueError("separator_cross_precision has an incompatible shape")
        if separator_information.shape != (separator_size,):
            raise ValueError("separator_information has an incompatible shape")

        _symmetric(node_precision, name="node_precision")
        _symmetric(separator_precision, name="separator_precision")
        for index, parent in enumerate(parents):
            if parent < -1 or parent >= index:
                raise ValueError(
                    "parent_indices must use -1 roots and parents before children"
                )
            if parent == -1 and np.any(parent_cross[index] != 0.0):
                raise ValueError("root parent-cross precision blocks must be zero")

        object.__setattr__(self, "parent_indices", parents)
        object.__setattr__(self, "node_precision", node_precision)
        object.__setattr__(self, "parent_cross_precision", parent_cross)
        object.__setattr__(
            self,
            "separator_cross_precision",
            separator_cross,
        )
        object.__setattr__(self, "separator_precision", separator_precision)
        object.__setattr__(self, "node_information", node_information)
        object.__setattr__(
            self,
            "separator_information",
            separator_information,
        )

    @property
    def node_count(self) -> int:
        return int(self.parent_indices.shape[0])

    @property
    def block_size(self) -> int:
        return int(self.node_precision.shape[1])

    @property
    def separator_size(self) -> int:
        return int(self.separator_precision.shape[0])

    @property
    def dimension(self) -> int:
        return self.node_count * self.block_size + self.separator_size

    @property
    def stored_nbytes(self) -> int:
        arrays = (
            self.parent_indices,
            self.node_precision,
            self.parent_cross_precision,
            self.separator_cross_precision,
            self.separator_precision,
            self.node_information,
            self.separator_information,
        )
        return int(sum(array.nbytes for array in arrays))

    @property
    def estimated_dense_precision_bytes(self) -> int:
        return self.dimension * self.dimension * np.dtype(np.float64).itemsize

    def to_dense(
        self,
        *,
        maximum_bytes: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Materialize the full precision only under an explicit byte budget."""

        if isinstance(maximum_bytes, bool) or not isinstance(maximum_bytes, Integral):
            raise TypeError("maximum_bytes must be an integer")
        if maximum_bytes < 0:
            raise ValueError("maximum_bytes must be non-negative")
        required = self.estimated_dense_precision_bytes
        if required > int(maximum_bytes):
            raise MemoryError(
                f"dense precision requires {required} bytes, budget is "
                f"{int(maximum_bytes)}"
            )
        precision = np.zeros(
            (self.dimension, self.dimension),
            dtype=np.float64,
        )
        information = np.zeros(self.dimension, dtype=np.float64)
        block_size = self.block_size
        separator_start = self.node_count * block_size
        for index, parent in enumerate(self.parent_indices):
            node_slice = slice(index * block_size, (index + 1) * block_size)
            precision[node_slice, node_slice] = self.node_precision[index]
            information[node_slice] = self.node_information[index]
            if parent >= 0:
                parent_slice = slice(
                    int(parent) * block_size,
                    (int(parent) + 1) * block_size,
                )
                cross = self.parent_cross_precision[index]
                precision[node_slice, parent_slice] = cross
                precision[parent_slice, node_slice] = cross.T
            precision[node_slice, separator_start:] = self.separator_cross_precision[
                index
            ]
            precision[separator_start:, node_slice] = self.separator_cross_precision[
                index
            ].T
        precision[separator_start:, separator_start:] = self.separator_precision
        information[separator_start:] = self.separator_information
        return precision, information


@dataclass(frozen=True, slots=True)
class TreeSeparatorGaussianSolutionV1:
    """Means and marginal covariances from exact block-tree elimination."""

    node_mean: np.ndarray
    separator_mean: np.ndarray
    node_covariance: np.ndarray
    node_separator_cross_covariance: np.ndarray
    separator_covariance: np.ndarray
    log_determinant_precision: float
    implementation: str = TREE_SEPARATOR_GAUSSIAN_IMPLEMENTATION

    @property
    def node_count(self) -> int:
        return int(self.node_mean.shape[0])

    @property
    def block_size(self) -> int:
        return int(self.node_mean.shape[1])

    @property
    def separator_size(self) -> int:
        return int(self.separator_mean.shape[0])

    @property
    def stored_nbytes(self) -> int:
        arrays = (
            self.node_mean,
            self.separator_mean,
            self.node_covariance,
            self.node_separator_cross_covariance,
            self.separator_covariance,
        )
        return int(sum(array.nbytes for array in arrays))

    def node_marginal(
        self,
        node_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(node_index, bool) or not isinstance(node_index, Integral):
            raise TypeError("node_index must be an integer")
        index = int(node_index)
        if index < 0 or index >= self.node_count:
            raise IndexError("node_index is out of range")
        return (
            self.node_mean[index].copy(),
            self.node_covariance[index].copy(),
            self.node_separator_cross_covariance[index].copy(),
        )


def solve_tree_separator_gaussian(
    system: TreeSeparatorGaussianSystemV1,
) -> TreeSeparatorGaussianSolutionV1:
    """Solve one SPD block-tree precision system without dense tree materialization."""

    if not isinstance(system, TreeSeparatorGaussianSystemV1):
        raise TypeError("system must be a TreeSeparatorGaussianSystemV1")

    node_count = system.node_count
    block_size = system.block_size
    separator_size = system.separator_size
    diagonal = np.array(system.node_precision, copy=True)
    separator_cross = np.array(
        system.separator_cross_precision,
        copy=True,
    )
    node_information = np.array(system.node_information, copy=True)
    separator_precision = np.array(system.separator_precision, copy=True)
    separator_information = np.array(
        system.separator_information,
        copy=True,
    )

    conditional_covariance = np.empty_like(diagonal)
    parent_coefficient = np.zeros_like(system.parent_cross_precision)
    separator_coefficient = np.empty_like(separator_cross)
    conditional_mean = np.empty_like(node_information)
    log_determinant = 0.0

    identity = np.eye(block_size, dtype=np.float64)
    for index in range(node_count - 1, -1, -1):
        cholesky = _cholesky(
            diagonal[index],
            name=f"eliminated node precision {index}",
        )
        conditional_covariance[index] = _solve_cholesky(cholesky, identity)
        conditional_mean[index] = _solve_cholesky(
            cholesky,
            node_information[index],
        )
        separator_coefficient[index] = -_solve_cholesky(
            cholesky,
            separator_cross[index],
        )
        parent = int(system.parent_indices[index])
        if parent >= 0:
            edge = system.parent_cross_precision[index]
            parent_coefficient[index] = -_solve_cholesky(
                cholesky,
                edge,
            )
            diagonal[parent] += edge.T @ parent_coefficient[index]
            separator_cross[parent] += edge.T @ separator_coefficient[index]
            node_information[parent] -= edge.T @ conditional_mean[index]
        separator_precision += separator_cross[index].T @ separator_coefficient[index]
        separator_information -= separator_cross[index].T @ conditional_mean[index]
        log_determinant += 2.0 * float(np.sum(np.log(np.diag(cholesky))))

    if separator_size:
        separator_cholesky = _cholesky(
            separator_precision,
            name="separator Schur complement",
        )
        separator_mean = _solve_cholesky(
            separator_cholesky,
            separator_information,
        )
        separator_covariance = _solve_cholesky(
            separator_cholesky,
            np.eye(separator_size, dtype=np.float64),
        )
        log_determinant += 2.0 * float(np.sum(np.log(np.diag(separator_cholesky))))
    else:
        separator_mean = np.zeros(0, dtype=np.float64)
        separator_covariance = np.zeros((0, 0), dtype=np.float64)

    node_mean = np.empty((node_count, block_size), dtype=np.float64)
    node_covariance = np.empty_like(diagonal)
    node_separator_cross_covariance = np.empty(
        (node_count, block_size, separator_size),
        dtype=np.float64,
    )

    for index, parent_value in enumerate(system.parent_indices):
        parent = int(parent_value)
        mean = conditional_mean[index]
        covariance = conditional_covariance[index]
        separator_mapping = separator_coefficient[index]
        if parent >= 0:
            parent_mapping = parent_coefficient[index]
            mean = mean + parent_mapping @ node_mean[parent]
            covariance = (
                covariance + parent_mapping @ node_covariance[parent] @ parent_mapping.T
            )
            if separator_size:
                mean = mean + separator_mapping @ separator_mean
                parent_cross = node_separator_cross_covariance[parent]
                covariance = (
                    covariance
                    + separator_mapping @ separator_covariance @ separator_mapping.T
                    + parent_mapping @ parent_cross @ separator_mapping.T
                    + separator_mapping @ parent_cross.T @ parent_mapping.T
                )
                cross = (
                    parent_mapping @ parent_cross
                    + separator_mapping @ separator_covariance
                )
            else:
                cross = np.zeros((block_size, 0), dtype=np.float64)
        elif separator_size:
            mean = mean + separator_mapping @ separator_mean
            covariance = (
                covariance
                + separator_mapping @ separator_covariance @ separator_mapping.T
            )
            cross = separator_mapping @ separator_covariance
        else:
            cross = np.zeros((block_size, 0), dtype=np.float64)
        node_mean[index] = mean
        node_covariance[index] = 0.5 * (covariance + covariance.T)
        node_separator_cross_covariance[index] = cross

    arrays = (
        node_mean,
        separator_mean,
        node_covariance,
        node_separator_cross_covariance,
        separator_covariance,
    )
    for array in arrays:
        array.setflags(write=False)
    return TreeSeparatorGaussianSolutionV1(
        node_mean=node_mean,
        separator_mean=separator_mean,
        node_covariance=node_covariance,
        node_separator_cross_covariance=(node_separator_cross_covariance),
        separator_covariance=separator_covariance,
        log_determinant_precision=log_determinant,
    )


__all__ = [
    "TREE_SEPARATOR_GAUSSIAN_IMPLEMENTATION",
    "TreeSeparatorGaussianError",
    "TreeSeparatorGaussianSolutionV1",
    "TreeSeparatorGaussianSystemV1",
    "solve_tree_separator_gaussian",
]
