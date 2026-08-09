"""Native block-tree information factorization without dense graph assembly.

The factor stores one symmetric diagonal block per tree node and one child-to-
parent off-diagonal block per non-root edge.  Reverse topological block
elimination yields an exact Cholesky-backed solver whose storage is linear in
node count for a fixed block size.  Dense materialization is available only as
an explicit, byte-budgeted compatibility operation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

TREE_BLOCK_INFORMATION_SCHEMA = "bayesian_phystwin.tree_block_information"
TREE_BLOCK_INFORMATION_VERSION = 1


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _readonly(value: object, *, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True, order="C")
    array.setflags(write=False)
    return array


def _symmetric(value: np.ndarray) -> np.ndarray:
    return 0.5 * (value + np.swapaxes(value, -1, -2))


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _cholesky_solve(factor: np.ndarray, right: np.ndarray) -> np.ndarray:
    intermediate = np.linalg.solve(factor, right)
    return np.linalg.solve(factor.T, intermediate)


def _validate_maximum_bytes(maximum_bytes: int | None, required_bytes: int) -> None:
    if maximum_bytes is None:
        return
    if type(maximum_bytes) is not int or maximum_bytes < 0:
        raise ValueError("maximum_bytes must be a nonnegative integer or None")
    if required_bytes > maximum_bytes:
        raise MemoryError(
            "dense tree information materialization requires "
            f"{required_bytes} bytes, exceeding the {maximum_bytes}-byte limit"
        )


@dataclass(frozen=True, slots=True)
class TreeBlockInformationFactorV1:
    """Exact SPD information factor for a topologically ordered block tree.

    ``child_parent_blocks[i]`` is the information block in row ``i`` and the
    column of ``parent_indices[i]``.  The root must be node zero and use parent
    index ``-1`` with a zero child-parent block.  Parent indices must precede
    children, so reverse index order is a valid leaf-elimination order even for
    non-chain trees.
    """

    parent_indices: np.ndarray
    diagonal_blocks: np.ndarray
    child_parent_blocks: np.ndarray
    _eliminated_cholesky: np.ndarray = field(init=False, repr=False, compare=False)
    _result_id: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        raw_parents = np.asarray(self.parent_indices)
        _require(
            raw_parents.ndim == 1
            and np.issubdtype(raw_parents.dtype, np.integer)
            and raw_parents.dtype.kind != "b",
            "parent_indices must be an integer vector",
        )
        parents = np.asarray(raw_parents, dtype=np.int64)
        _require(len(parents) >= 1, "the block tree must contain at least one node")
        _require(parents[0] == -1, "the first block must be the tree root")
        for index in range(1, len(parents)):
            _require(
                0 <= int(parents[index]) < index,
                "each non-root parent must precede its child",
            )

        diagonal = np.asarray(self.diagonal_blocks, dtype=np.float64)
        edges = np.asarray(self.child_parent_blocks, dtype=np.float64)
        _require(
            diagonal.ndim == 3
            and diagonal.shape[0] == len(parents)
            and diagonal.shape[1] == diagonal.shape[2]
            and diagonal.shape[1] >= 1,
            "diagonal_blocks must have shape (N, D, D) with D >= 1",
        )
        _require(
            edges.shape == diagonal.shape,
            "child_parent_blocks must match diagonal_blocks shape",
        )
        _require(
            np.all(np.isfinite(diagonal)) and np.all(np.isfinite(edges)),
            "tree information blocks must be finite",
        )
        _require(
            np.allclose(
                diagonal,
                np.swapaxes(diagonal, 1, 2),
                atol=1e-10,
                rtol=1e-10,
            ),
            "diagonal_blocks must be symmetric",
        )
        _require(
            np.count_nonzero(edges[0]) == 0,
            "the root child-parent block must be zero",
        )

        eliminated = np.array(diagonal, dtype=np.float64, copy=True, order="C")
        factors = np.empty_like(eliminated)
        try:
            for index in range(len(parents) - 1, 0, -1):
                eliminated[index] = _symmetric(eliminated[index])
                factor = np.linalg.cholesky(eliminated[index])
                factors[index] = factor
                edge_solve = _cholesky_solve(factor, edges[index])
                parent = int(parents[index])
                eliminated[parent] -= edges[index].T @ edge_solve
            eliminated[0] = _symmetric(eliminated[0])
            factors[0] = np.linalg.cholesky(eliminated[0])
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "tree information matrix must be positive definite"
            ) from error

        parents = _readonly(parents, dtype=np.dtype(np.int64))
        diagonal = _readonly(diagonal, dtype=np.dtype(np.float64))
        edges = _readonly(edges, dtype=np.dtype(np.float64))
        factors = _readonly(factors, dtype=np.dtype(np.float64))
        object.__setattr__(self, "parent_indices", parents)
        object.__setattr__(self, "diagonal_blocks", diagonal)
        object.__setattr__(self, "child_parent_blocks", edges)
        object.__setattr__(self, "_eliminated_cholesky", factors)
        object.__setattr__(self, "_result_id", self._compute_result_id())

    @classmethod
    def from_transition_innovation(
        cls,
        *,
        parent_indices: np.ndarray,
        transition_matrices: np.ndarray,
        innovation_scale_tril: np.ndarray,
        local_information_blocks: np.ndarray | None = None,
    ) -> TreeBlockInformationFactorV1:
        """Build exact tree information from transition/innovation factors.

        The model is ``g_i - T_i g_parent ~ N(0, L_i L_i.T)`` for each
        non-root node and ``g_0 ~ N(0, L_0 L_0.T)`` for the root.  Optional
        local information is added to each node diagonal before factorization.
        """

        parents = np.asarray(parent_indices)
        transitions = np.asarray(transition_matrices, dtype=np.float64)
        scales = np.asarray(innovation_scale_tril, dtype=np.float64)
        _require(
            parents.ndim == 1
            and np.issubdtype(parents.dtype, np.integer)
            and parents.dtype.kind != "b",
            "parent_indices must be an integer vector",
        )
        count = len(parents)
        _require(
            transitions.ndim == 3
            and transitions.shape[0] == count
            and transitions.shape[1] == transitions.shape[2]
            and transitions.shape[1] >= 1,
            "transition_matrices must have shape (N, D, D) with D >= 1",
        )
        _require(
            scales.shape == transitions.shape,
            "innovation_scale_tril must match transition_matrices shape",
        )
        _require(
            np.all(np.isfinite(transitions)) and np.all(np.isfinite(scales)),
            "transition and innovation factors must be finite",
        )
        _require(
            np.allclose(scales, np.tril(scales), atol=1e-14, rtol=0.0),
            "innovation_scale_tril must be lower triangular",
        )
        _require(
            np.all(np.diagonal(scales, axis1=1, axis2=2) > 0.0),
            "innovation_scale_tril must have positive diagonal",
        )
        block_size = transitions.shape[1]
        if local_information_blocks is None:
            diagonal = np.zeros((count, block_size, block_size), dtype=np.float64)
        else:
            diagonal = np.asarray(local_information_blocks, dtype=np.float64)
            _require(
                diagonal.shape == transitions.shape,
                "local_information_blocks must match transition_matrices shape",
            )
            _require(
                np.all(np.isfinite(diagonal)),
                "local_information_blocks must be finite",
            )
            _require(
                np.allclose(
                    diagonal,
                    np.swapaxes(diagonal, 1, 2),
                    atol=1e-10,
                    rtol=1e-10,
                ),
                "local_information_blocks must be symmetric",
            )
            diagonal = np.array(diagonal, dtype=np.float64, copy=True, order="C")
        edges = np.zeros_like(diagonal)
        identity = np.eye(block_size, dtype=np.float64)
        normalized_parents = np.asarray(parents, dtype=np.int64)
        _require(count >= 1, "the block tree must contain at least one node")
        _require(normalized_parents[0] == -1, "the first block must be the tree root")
        for index in range(count):
            if index:
                _require(
                    0 <= int(normalized_parents[index]) < index,
                    "each non-root parent must precede its child",
                )
            inverse_scale = np.linalg.solve(scales[index], identity)
            innovation_precision = inverse_scale.T @ inverse_scale
            diagonal[index] += innovation_precision
            parent = int(normalized_parents[index])
            if parent < 0:
                continue
            transition = transitions[index]
            edges[index] = -innovation_precision @ transition
            diagonal[parent] += transition.T @ innovation_precision @ transition
        return cls(
            parent_indices=normalized_parents,
            diagonal_blocks=_symmetric(diagonal),
            child_parent_blocks=edges,
        )

    @property
    def node_count(self) -> int:
        return len(self.parent_indices)

    @property
    def block_size(self) -> int:
        return self.diagonal_blocks.shape[1]

    @property
    def dimension(self) -> int:
        return self.node_count * self.block_size

    @property
    def estimated_dense_bytes(self) -> int:
        return self.dimension**2 * np.dtype(np.float64).itemsize

    @property
    def stored_nbytes(self) -> int:
        return int(
            self.parent_indices.nbytes
            + self.diagonal_blocks.nbytes
            + self.child_parent_blocks.nbytes
            + self._eliminated_cholesky.nbytes
        )

    @property
    def dense_information_materialized(self) -> bool:
        return False

    @property
    def log_determinant(self) -> float:
        diagonal = np.diagonal(self._eliminated_cholesky, axis1=1, axis2=2)
        return float(2.0 * np.sum(np.log(diagonal)))

    @property
    def maximum_eliminated_block_condition(self) -> float:
        conditions = [
            float(np.linalg.cond(factor) ** 2)
            for factor in self._eliminated_cholesky
        ]
        return max(conditions, default=1.0)

    @property
    def result_id(self) -> str:
        return self._result_id

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": TREE_BLOCK_INFORMATION_SCHEMA,
            "schema_version": TREE_BLOCK_INFORMATION_VERSION,
            "node_count": self.node_count,
            "block_size": self.block_size,
            "dimension": self.dimension,
            "estimated_dense_bytes": self.estimated_dense_bytes,
            "stored_nbytes": self.stored_nbytes,
            "parent_indices_sha256": _array_sha256(self.parent_indices),
            "diagonal_blocks_sha256": _array_sha256(self.diagonal_blocks),
            "child_parent_blocks_sha256": _array_sha256(
                self.child_parent_blocks
            ),
        }

    def _compute_result_id(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.descriptor(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    def _normalize_right(
        self,
        right: object,
    ) -> tuple[np.ndarray, str, tuple[int, ...]]:
        array = np.asarray(right, dtype=np.float64)
        _require(np.all(np.isfinite(array)), "right-hand side must be finite")
        original_shape = array.shape
        if array.ndim == 1:
            _require(
                array.shape == (self.dimension,),
                "flat right-hand side dimension changed",
            )
            blocked = array.reshape(self.node_count, self.block_size, 1)
            return blocked, "flat-vector", original_shape
        if array.ndim == 2:
            if array.shape == (self.node_count, self.block_size):
                return array[:, :, None], "blocked-vector", original_shape
            _require(
                array.shape[0] == self.dimension,
                "flat multi-right-hand side dimension changed",
            )
            blocked = array.reshape(
                self.node_count,
                self.block_size,
                array.shape[1],
            )
            return blocked, "flat-matrix", original_shape
        _require(
            array.ndim == 3
            and array.shape[:2] == (self.node_count, self.block_size),
            "blocked right-hand side must have shape (N, D, K)",
        )
        return array, "blocked-matrix", original_shape

    def solve(self, right: object) -> np.ndarray:
        """Apply the inverse information matrix to one or more right-hand sides."""

        blocked, representation, original_shape = self._normalize_right(right)
        work = np.array(blocked, dtype=np.float64, copy=True, order="C")
        for index in range(self.node_count - 1, 0, -1):
            solved = _cholesky_solve(self._eliminated_cholesky[index], work[index])
            parent = int(self.parent_indices[index])
            work[parent] -= self.child_parent_blocks[index].T @ solved
        solution = np.empty_like(work)
        solution[0] = _cholesky_solve(self._eliminated_cholesky[0], work[0])
        for index in range(1, self.node_count):
            parent = int(self.parent_indices[index])
            local_right = (
                work[index]
                - self.child_parent_blocks[index] @ solution[parent]
            )
            solution[index] = _cholesky_solve(
                self._eliminated_cholesky[index],
                local_right,
            )
        if representation == "flat-vector":
            return solution.reshape(original_shape)
        if representation == "blocked-vector":
            return solution[:, :, 0]
        if representation == "flat-matrix":
            return solution.reshape(original_shape)
        return solution

    def marginal_covariance(self, block_indices: object) -> np.ndarray:
        """Return the exact covariance among selected tree blocks."""

        raw = np.asarray(block_indices)
        _require(
            raw.ndim == 1
            and np.issubdtype(raw.dtype, np.integer)
            and raw.dtype.kind != "b",
            "block_indices must be an integer vector",
        )
        indices = np.asarray(raw, dtype=np.int64)
        _require(len(indices) >= 1, "block_indices must be nonempty")
        _require(
            len(np.unique(indices)) == len(indices),
            "block_indices must be unique",
        )
        _require(
            np.all((indices >= 0) & (indices < self.node_count)),
            "block_indices reference an unknown node",
        )
        column_count = len(indices) * self.block_size
        right = np.zeros(
            (self.node_count, self.block_size, column_count),
            dtype=np.float64,
        )
        identity = np.eye(self.block_size, dtype=np.float64)
        for position, index in enumerate(indices):
            selected = slice(
                position * self.block_size,
                (position + 1) * self.block_size,
            )
            right[int(index), :, selected] = identity
        solved = self.solve(right)
        rows = np.concatenate([solved[int(index)] for index in indices], axis=0)
        return _readonly(_symmetric(rows), dtype=np.dtype(np.float64))

    def multiply(self, right: object) -> np.ndarray:
        """Apply the tree information matrix without materializing it."""

        blocked, representation, original_shape = self._normalize_right(right)
        result = np.einsum(
            "nij,njk->nik",
            self.diagonal_blocks,
            blocked,
            optimize=True,
        )
        for index in range(1, self.node_count):
            parent = int(self.parent_indices[index])
            result[index] += self.child_parent_blocks[index] @ blocked[parent]
            result[parent] += self.child_parent_blocks[index].T @ blocked[index]
        if representation == "flat-vector":
            return result.reshape(original_shape)
        if representation == "blocked-vector":
            return result[:, :, 0]
        if representation == "flat-matrix":
            return result.reshape(original_shape)
        return result

    def quadratic_form(self, vector: object) -> float:
        """Evaluate ``v.T H v`` without constructing the dense information matrix."""

        value = np.asarray(vector, dtype=np.float64)
        _require(
            value.shape in {
                (self.dimension,),
                (self.node_count, self.block_size),
            },
            "vector dimension changed",
        )
        _require(np.all(np.isfinite(value)), "vector must be finite")
        blocked = value.reshape(self.node_count, self.block_size)
        result = 0.0
        for index in range(self.node_count):
            result += float(
                blocked[index]
                @ self.diagonal_blocks[index]
                @ blocked[index]
            )
            parent = int(self.parent_indices[index])
            if parent >= 0:
                result += 2.0 * float(
                    blocked[index]
                    @ self.child_parent_blocks[index]
                    @ blocked[parent]
                )
        return result

    def materialize(self, *, maximum_bytes: int | None = None) -> np.ndarray:
        """Explicitly construct the complete dense information matrix."""

        _validate_maximum_bytes(maximum_bytes, self.estimated_dense_bytes)
        result = np.zeros((self.dimension, self.dimension), dtype=np.float64)
        block_size = self.block_size
        for index in range(self.node_count):
            child = slice(index * block_size, (index + 1) * block_size)
            result[child, child] = self.diagonal_blocks[index]
            parent_index = int(self.parent_indices[index])
            if parent_index < 0:
                continue
            parent = slice(
                parent_index * block_size,
                (parent_index + 1) * block_size,
            )
            result[child, parent] = self.child_parent_blocks[index]
            result[parent, child] = self.child_parent_blocks[index].T
        return _readonly(_symmetric(result), dtype=np.dtype(np.float64))


__all__ = [
    "TREE_BLOCK_INFORMATION_SCHEMA",
    "TREE_BLOCK_INFORMATION_VERSION",
    "TreeBlockInformationFactorV1",
]
