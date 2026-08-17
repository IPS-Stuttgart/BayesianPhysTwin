"""Exact leaf-to-root block elimination for Gaussian tree systems."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._canonical_contracts import frozen_finite_json_mapping
from ._gauge_aware_contracts import _finite_array, _readonly, _require

TREE_BLOCK_GAUSSIAN_SCHEMA = "bayesian_phystwin.tree_block_gaussian"
TREE_BLOCK_GAUSSIAN_VERSION = 1


def _integer_vector(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    _require(
        raw.ndim == 1
        and np.issubdtype(raw.dtype, np.integer)
        and raw.dtype.kind != "b",
        f"{name} must be an integer vector",
    )
    return _readonly(raw, dtype=np.int64)


def _symmetric_matrix(
    value: object,
    *,
    name: str,
    positive_definite: bool,
) -> np.ndarray:
    matrix = _finite_array(value, name, 2)
    _require(matrix.shape[0] == matrix.shape[1], f"{name} must be square")
    _require(
        np.allclose(matrix, matrix.T, atol=1e-10, rtol=1e-10),
        f"{name} must be symmetric",
    )
    if positive_definite and len(matrix):
        try:
            np.linalg.cholesky(0.5 * (matrix + matrix.T))
        except np.linalg.LinAlgError as error:
            raise ValueError(f"{name} must be positive definite") from error
    return _readonly(matrix)


def _cholesky_solve(factor: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.linalg.solve(factor.T, np.linalg.solve(factor, right))


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _rhs_matrix(
    value: object,
    *,
    name: str,
    leading_shape: tuple[int, ...],
) -> tuple[np.ndarray, bool]:
    array = np.asarray(value, dtype=np.float64)
    _require(np.all(np.isfinite(array)), f"{name} must be finite")
    if array.shape == leading_shape:
        return array[..., None], True
    _require(
        array.ndim == len(leading_shape) + 1
        and array.shape[: len(leading_shape)] == leading_shape,
        f"{name} has changed shape",
    )
    return array, False


def _validate_condition_limit(value: object) -> float:
    raw = np.asarray(value)
    _require(
        raw.ndim == 0 and raw.dtype.kind not in {"b", "O", "U", "S"},
        "maximum_condition_number must be a numeric scalar",
    )
    limit = float(raw.item())
    _require(
        np.isfinite(limit) and limit > 0.0,
        "maximum_condition_number must be positive",
    )
    return limit


@dataclass(frozen=True, slots=True)
class TreeBlockNormalSystemV1:
    """One dense global block coupled to block-local variables on a tree."""

    parent_indices: np.ndarray
    node_precision: np.ndarray
    parent_coupling: np.ndarray
    global_coupling: np.ndarray
    global_precision: np.ndarray
    node_right: np.ndarray
    global_right: np.ndarray

    def __post_init__(self) -> None:
        parents = _integer_vector(self.parent_indices, name="parent_indices")
        node_precision = _finite_array(
            self.node_precision,
            "node_precision",
            3,
        )
        parent_coupling = _finite_array(
            self.parent_coupling,
            "parent_coupling",
            3,
        )
        global_coupling = _finite_array(
            self.global_coupling,
            "global_coupling",
            3,
        )
        global_precision = _symmetric_matrix(
            self.global_precision,
            name="global_precision",
            positive_definite=False,
        )
        node_right = _finite_array(self.node_right, "node_right", 2)
        global_right = _finite_array(self.global_right, "global_right", 1)

        _require(
            node_precision.shape[1] == node_precision.shape[2]
            and node_precision.shape[1] >= 1,
            "node_precision must have shape (K, B, B) with B >= 1",
        )
        node_count = len(node_precision)
        block_size = node_precision.shape[1]
        global_size = len(global_precision)
        _require(node_count >= 1, "the tree must contain at least one node")
        _require(
            parents.shape == (node_count,),
            "parent_indices must identify every node",
        )
        _require(
            parent_coupling.shape == node_precision.shape,
            "parent_coupling shape changed",
        )
        _require(
            global_coupling.shape == (node_count, block_size, global_size),
            "global_coupling shape changed",
        )
        _require(
            node_right.shape == (node_count, block_size),
            "node_right shape changed",
        )
        _require(
            global_right.shape == (global_size,),
            "global_right shape changed",
        )
        _require(parents[0] == -1, "the first node must be the tree root")
        for index in range(1, node_count):
            _require(
                0 <= int(parents[index]) < index,
                "each non-root parent must precede its child",
            )
        _require(
            np.allclose(parent_coupling[0], 0.0, atol=1e-14, rtol=0.0),
            "root parent coupling must be zero",
        )
        for index in range(node_count):
            _require(
                np.allclose(
                    node_precision[index],
                    node_precision[index].T,
                    atol=1e-10,
                    rtol=1e-10,
                ),
                f"node precision {index} must be symmetric",
            )

        object.__setattr__(self, "parent_indices", parents)
        object.__setattr__(self, "node_precision", _readonly(node_precision))
        object.__setattr__(
            self,
            "parent_coupling",
            _readonly(parent_coupling),
        )
        object.__setattr__(
            self,
            "global_coupling",
            _readonly(global_coupling),
        )
        object.__setattr__(self, "global_precision", global_precision)
        object.__setattr__(self, "node_right", _readonly(node_right))
        object.__setattr__(self, "global_right", _readonly(global_right))

    @property
    def node_count(self) -> int:
        return len(self.node_precision)

    @property
    def block_size(self) -> int:
        return self.node_precision.shape[1]

    @property
    def global_size(self) -> int:
        return len(self.global_precision)

    @property
    def dimension(self) -> int:
        return self.global_size + self.node_count * self.block_size

    @property
    def stored_nbytes(self) -> int:
        return int(
            self.parent_indices.nbytes
            + self.node_precision.nbytes
            + self.parent_coupling.nbytes
            + self.global_coupling.nbytes
            + self.global_precision.nbytes
            + self.node_right.nbytes
            + self.global_right.nbytes
        )

    @property
    def estimated_dense_precision_bytes(self) -> int:
        return self.dimension**2 * np.dtype(np.float64).itemsize

    def eliminate_nodes(
        self,
        *,
        maximum_condition_number: float,
    ) -> TreeBlockEliminationV1:
        """Eliminate every tree node without assembling a dense tree matrix."""

        limit = _validate_condition_limit(maximum_condition_number)
        diagonal = np.array(self.node_precision, copy=True)
        global_coupling = np.array(self.global_coupling, copy=True)
        global_schur = np.array(self.global_precision, copy=True)
        factors = np.empty_like(diagonal)
        condition_numbers: np.ndarray = np.empty(self.node_count, dtype=np.float64)

        for index in range(self.node_count - 1, -1, -1):
            diagonal[index] = 0.5 * (diagonal[index] + diagonal[index].T)
            condition_number = float(np.linalg.cond(diagonal[index]))
            condition_numbers[index] = condition_number
            if not np.isfinite(condition_number) or condition_number > limit:
                raise np.linalg.LinAlgError(
                    f"tree block {index} is ill-conditioned ({condition_number})"
                )
            factor = np.linalg.cholesky(diagonal[index])
            factors[index] = factor
            inverse_global = _cholesky_solve(
                factor,
                global_coupling[index],
            )
            global_schur -= global_coupling[index].T @ inverse_global
            parent = int(self.parent_indices[index])
            if parent < 0:
                continue
            coupling = self.parent_coupling[index]
            inverse_parent = _cholesky_solve(factor, coupling)
            diagonal[parent] -= coupling.T @ inverse_parent
            global_coupling[parent] -= coupling.T @ inverse_global

        return TreeBlockEliminationV1(
            parent_indices=self.parent_indices,
            node_cholesky=factors,
            parent_coupling=self.parent_coupling,
            global_coupling=global_coupling,
            global_schur=0.5 * (global_schur + global_schur.T),
            node_condition_numbers=condition_numbers,
        )

    def residual(
        self,
        global_solution: np.ndarray,
        node_solution: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the normal-equation residual in structured form."""

        global_vector = np.asarray(global_solution, dtype=np.float64)
        nodes = np.asarray(node_solution, dtype=np.float64)
        _require(
            global_vector.shape == (self.global_size,),
            "global solution shape changed",
        )
        _require(
            nodes.shape == (self.node_count, self.block_size),
            "node solution shape changed",
        )
        global_residual = self.global_precision @ global_vector - self.global_right
        node_residual = (
            np.einsum(
                "kij,kj->ki",
                self.node_precision,
                nodes,
                optimize=True,
            )
            - self.node_right
        )
        global_residual += np.einsum(
            "kid,ki->d",
            self.global_coupling,
            nodes,
            optimize=True,
        )
        node_residual += np.einsum(
            "kid,d->ki",
            self.global_coupling,
            global_vector,
            optimize=True,
        )
        for index in range(1, self.node_count):
            parent = int(self.parent_indices[index])
            coupling = self.parent_coupling[index]
            node_residual[index] += coupling @ nodes[parent]
            node_residual[parent] += coupling.T @ nodes[index]
        return global_residual, node_residual


@dataclass(frozen=True, slots=True)
class TreeBlockEliminationV1:
    """Node factors and the dense global Schur complement."""

    parent_indices: np.ndarray
    node_cholesky: np.ndarray
    parent_coupling: np.ndarray
    global_coupling: np.ndarray
    global_schur: np.ndarray
    node_condition_numbers: np.ndarray

    def __post_init__(self) -> None:
        parents = _integer_vector(self.parent_indices, name="parent_indices")
        factors = _finite_array(self.node_cholesky, "node_cholesky", 3)
        parent_coupling = _finite_array(
            self.parent_coupling,
            "parent_coupling",
            3,
        )
        global_coupling = _finite_array(
            self.global_coupling,
            "global_coupling",
            3,
        )
        global_schur = _symmetric_matrix(
            self.global_schur,
            name="global_schur",
            positive_definite=False,
        )
        condition_numbers = _finite_array(
            self.node_condition_numbers,
            "node_condition_numbers",
            1,
        )
        node_count, block_size, second = factors.shape
        _require(block_size == second, "node Cholesky blocks must be square")
        _require(
            parents.shape == (node_count,),
            "parent_indices shape changed",
        )
        _require(
            parent_coupling.shape == factors.shape,
            "parent_coupling shape changed",
        )
        _require(
            global_coupling.shape == (node_count, block_size, len(global_schur)),
            "global_coupling shape changed",
        )
        _require(
            condition_numbers.shape == (node_count,),
            "node condition shape changed",
        )
        object.__setattr__(self, "parent_indices", parents)
        object.__setattr__(self, "node_cholesky", _readonly(factors))
        object.__setattr__(
            self,
            "parent_coupling",
            _readonly(parent_coupling),
        )
        object.__setattr__(
            self,
            "global_coupling",
            _readonly(global_coupling),
        )
        object.__setattr__(self, "global_schur", global_schur)
        object.__setattr__(
            self,
            "node_condition_numbers",
            _readonly(condition_numbers),
        )

    @property
    def node_count(self) -> int:
        return len(self.node_cholesky)

    @property
    def block_size(self) -> int:
        return self.node_cholesky.shape[1]

    @property
    def global_size(self) -> int:
        return len(self.global_schur)

    @property
    def maximum_node_condition_number(self) -> float:
        return float(np.max(self.node_condition_numbers, initial=0.0))

    def factor_global(
        self,
        *,
        maximum_condition_number: float,
    ) -> TreeBlockFactorizationV1:
        """Factor the small global Schur complement."""

        limit = _validate_condition_limit(maximum_condition_number)
        condition_number = float(np.linalg.cond(self.global_schur))
        if not np.isfinite(condition_number) or condition_number > limit:
            raise np.linalg.LinAlgError(
                f"global Schur complement is ill-conditioned ({condition_number})"
            )
        factor = np.linalg.cholesky(self.global_schur)
        return TreeBlockFactorizationV1(
            parent_indices=self.parent_indices,
            node_cholesky=self.node_cholesky,
            parent_coupling=self.parent_coupling,
            global_coupling=self.global_coupling,
            global_cholesky=factor,
            node_condition_numbers=self.node_condition_numbers,
            global_condition_number=condition_number,
        )


@dataclass(frozen=True, slots=True)
class TreeBlockFactorizationV1:
    """Reusable Cholesky factors for solves and covariance queries."""

    parent_indices: np.ndarray
    node_cholesky: np.ndarray
    parent_coupling: np.ndarray
    global_coupling: np.ndarray
    global_cholesky: np.ndarray
    node_condition_numbers: np.ndarray
    global_condition_number: float

    def __post_init__(self) -> None:
        parents = _integer_vector(self.parent_indices, name="parent_indices")
        factors = _finite_array(self.node_cholesky, "node_cholesky", 3)
        parent_coupling = _finite_array(
            self.parent_coupling,
            "parent_coupling",
            3,
        )
        global_coupling = _finite_array(
            self.global_coupling,
            "global_coupling",
            3,
        )
        global_factor = _finite_array(
            self.global_cholesky,
            "global_cholesky",
            2,
        )
        condition_numbers = _finite_array(
            self.node_condition_numbers,
            "node_condition_numbers",
            1,
        )
        node_count, block_size, second = factors.shape
        _require(block_size == second, "node Cholesky blocks must be square")
        _require(
            parents.shape == (node_count,),
            "parent_indices shape changed",
        )
        _require(
            parent_coupling.shape == factors.shape,
            "parent_coupling shape changed",
        )
        global_size = len(global_factor)
        _require(
            global_factor.shape == (global_size, global_size),
            "global_cholesky must be square",
        )
        _require(
            global_coupling.shape == (node_count, block_size, global_size),
            "global_coupling shape changed",
        )
        _require(
            condition_numbers.shape == (node_count,),
            "node condition shape changed",
        )
        _require(
            np.isfinite(self.global_condition_number)
            and self.global_condition_number > 0.0,
            "global_condition_number must be positive",
        )
        object.__setattr__(self, "parent_indices", parents)
        object.__setattr__(self, "node_cholesky", _readonly(factors))
        object.__setattr__(
            self,
            "parent_coupling",
            _readonly(parent_coupling),
        )
        object.__setattr__(
            self,
            "global_coupling",
            _readonly(global_coupling),
        )
        object.__setattr__(self, "global_cholesky", _readonly(global_factor))
        object.__setattr__(
            self,
            "node_condition_numbers",
            _readonly(condition_numbers),
        )
        object.__setattr__(
            self,
            "global_condition_number",
            float(self.global_condition_number),
        )

    @property
    def node_count(self) -> int:
        return len(self.node_cholesky)

    @property
    def block_size(self) -> int:
        return self.node_cholesky.shape[1]

    @property
    def global_size(self) -> int:
        return len(self.global_cholesky)

    @property
    def dimension(self) -> int:
        return self.global_size + self.node_count * self.block_size

    @property
    def maximum_node_condition_number(self) -> float:
        return float(np.max(self.node_condition_numbers, initial=0.0))

    @property
    def stored_nbytes(self) -> int:
        return int(
            self.parent_indices.nbytes
            + self.node_cholesky.nbytes
            + self.parent_coupling.nbytes
            + self.global_coupling.nbytes
            + self.global_cholesky.nbytes
            + self.node_condition_numbers.nbytes
        )

    @property
    def estimated_dense_precision_bytes(self) -> int:
        return self.dimension**2 * np.dtype(np.float64).itemsize

    def solve(
        self,
        global_right: object,
        node_right: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve one or several right-hand sides by elimination/backsolve."""

        global_matrix, global_vector = _rhs_matrix(
            global_right,
            name="global_right",
            leading_shape=(self.global_size,),
        )
        node_matrix, node_vector = _rhs_matrix(
            node_right,
            name="node_right",
            leading_shape=(self.node_count, self.block_size),
        )
        _require(
            global_matrix.shape[-1] == node_matrix.shape[-1],
            "global and node right-hand sides have different column counts",
        )
        updated_nodes = np.array(node_matrix, copy=True)
        updated_global = np.array(global_matrix, copy=True)
        for index in range(self.node_count - 1, -1, -1):
            solved = _cholesky_solve(
                self.node_cholesky[index],
                updated_nodes[index],
            )
            updated_global -= self.global_coupling[index].T @ solved
            parent = int(self.parent_indices[index])
            if parent >= 0:
                updated_nodes[parent] -= self.parent_coupling[index].T @ solved
        global_solution = _cholesky_solve(
            self.global_cholesky,
            updated_global,
        )
        node_solution = np.empty_like(updated_nodes)
        for index in range(self.node_count):
            right = updated_nodes[index] - self.global_coupling[index] @ global_solution
            parent = int(self.parent_indices[index])
            if parent >= 0:
                right -= self.parent_coupling[index] @ node_solution[parent]
            node_solution[index] = _cholesky_solve(
                self.node_cholesky[index],
                right,
            )
        if global_vector and node_vector:
            return global_solution[:, 0], node_solution[:, :, 0]
        return global_solution, node_solution

    def global_marginal_covariance(self) -> np.ndarray:
        """Return only the dense marginal covariance of global variables."""

        identity = np.eye(self.global_size, dtype=np.float64)
        covariance = _cholesky_solve(self.global_cholesky, identity)
        return _readonly(0.5 * (covariance + covariance.T))

    def materialize_covariance(
        self,
        *,
        maximum_bytes: int | None = None,
    ) -> np.ndarray:
        """Materialize the complete covariance only after an explicit budget."""

        required_bytes = self.dimension**2 * np.dtype(np.float64).itemsize
        if maximum_bytes is not None:
            _require(
                type(maximum_bytes) is int and maximum_bytes >= 0,
                "maximum_bytes must be a nonnegative integer or None",
            )
            if required_bytes > maximum_bytes:
                raise MemoryError(
                    f"dense covariance requires {required_bytes} bytes, "
                    f"exceeding the {maximum_bytes}-byte limit"
                )
        identity = np.eye(self.dimension, dtype=np.float64)
        global_right = identity[: self.global_size]
        node_right = identity[self.global_size :].reshape(
            self.node_count,
            self.block_size,
            self.dimension,
        )
        global_solution, node_solution = self.solve(
            global_right,
            node_right,
        )
        covariance = np.concatenate(
            (
                global_solution,
                node_solution.reshape(-1, self.dimension),
            ),
            axis=0,
        )
        return _readonly(0.5 * (covariance + covariance.T))

    def descriptor(self) -> Mapping[str, Any]:
        """Return a content-addressable description of every numeric factor."""

        return frozen_finite_json_mapping(
            {
                "schema": TREE_BLOCK_GAUSSIAN_SCHEMA,
                "schema_version": TREE_BLOCK_GAUSSIAN_VERSION,
                "node_count": self.node_count,
                "block_size": self.block_size,
                "global_size": self.global_size,
                "maximum_node_condition_number": (self.maximum_node_condition_number),
                "global_condition_number": self.global_condition_number,
                "stored_nbytes": self.stored_nbytes,
                "estimated_dense_precision_bytes": (
                    self.estimated_dense_precision_bytes
                ),
                "parent_indices_sha256": _array_sha256(self.parent_indices),
                "node_cholesky_sha256": _array_sha256(self.node_cholesky),
                "parent_coupling_sha256": _array_sha256(self.parent_coupling),
                "global_coupling_sha256": _array_sha256(self.global_coupling),
                "global_cholesky_sha256": _array_sha256(self.global_cholesky),
            },
            name="tree-block factorization descriptor",
        )


__all__ = [
    "TREE_BLOCK_GAUSSIAN_SCHEMA",
    "TREE_BLOCK_GAUSSIAN_VERSION",
    "TreeBlockEliminationV1",
    "TreeBlockFactorizationV1",
    "TreeBlockNormalSystemV1",
]
