"""Structured accepted-posterior covariance for tree-sparse gauge inference.

The representation stores a small Schur-complement covariance for retained
physical-state coordinates and global nuisance variables, an exact block-tree
information factor for local gauge variables, and the linear-size solve
``G^-1 B`` coupling the tree to the small core.  It supports covariance-vector,
query, and selected-marginal operations without constructing the complete dense
posterior.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .tree_block_information import TreeBlockInformationFactorV1

TREE_SCHUR_COVARIANCE_SCHEMA = "bayesian_phystwin.tree_schur_covariance"
TREE_SCHUR_COVARIANCE_VERSION = 1
TREE_SCHUR_COVARIANCE_REPRESENTATION = (
    "state-prior-plus-tree-schur-posterior-v1"
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _readonly(value: object, *, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True, order="C")
    array.setflags(write=False)
    return array


def _symmetric(value: np.ndarray) -> np.ndarray:
    return 0.5 * (value + value.T)


def _symmetric_matrix(
    value: object,
    *,
    name: str,
    positive_semidefinite: bool,
) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    _require(matrix.ndim == 2, f"{name} must have two dimensions")
    _require(matrix.shape[0] == matrix.shape[1], f"{name} must be square")
    _require(np.all(np.isfinite(matrix)), f"{name} must be finite")
    _require(
        np.allclose(matrix, matrix.T, atol=1e-10, rtol=1e-10),
        f"{name} must be symmetric",
    )
    matrix = _symmetric(matrix)
    if positive_semidefinite and len(matrix):
        minimum = float(np.min(np.linalg.eigvalsh(matrix)))
        _require(minimum >= -1e-9, f"{name} must be positive semidefinite")
    return _readonly(matrix, dtype=np.dtype(np.float64))


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _validate_maximum_bytes(maximum_bytes: int | None, required_bytes: int) -> None:
    if maximum_bytes is None:
        return
    if type(maximum_bytes) is not int or maximum_bytes < 0:
        raise ValueError("maximum_bytes must be a nonnegative integer or None")
    if required_bytes > maximum_bytes:
        raise MemoryError(
            "dense posterior covariance materialization requires "
            f"{required_bytes} bytes, exceeding the {maximum_bytes}-byte limit"
        )


@dataclass(frozen=True, slots=True)
class TreeSchurCovarianceV1:
    """Linear-size covariance operator for an accepted tree-sparse posterior.

    Coefficients are ordered as full physical state, all tree gauge blocks, then
    global nuisance variables.  The small Schur core is ordered as retained
    physical-state coordinates followed by those global nuisance variables.
    ``tree_core_solve`` stores ``G^-1 B`` in blocked ``(N, D, C)`` form, where
    ``G`` is the tree information matrix and ``B`` couples gauges to the core.
    """

    state_prior_covariance: np.ndarray
    state_mapping: np.ndarray
    core_covariance: np.ndarray
    tree_factor: TreeBlockInformationFactorV1
    tree_core_solve: np.ndarray
    _result_id: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.tree_factor, TreeBlockInformationFactorV1):
            raise TypeError("tree_factor must be a TreeBlockInformationFactorV1")
        state_prior = _symmetric_matrix(
            self.state_prior_covariance,
            name="state_prior_covariance",
            positive_semidefinite=True,
        )
        mapping = np.asarray(self.state_mapping, dtype=np.float64)
        _require(
            mapping.ndim == 2 and mapping.shape[0] == len(state_prior),
            "state_mapping must have one row per physical-state coefficient",
        )
        _require(np.all(np.isfinite(mapping)), "state_mapping must be finite")
        mapping = _readonly(mapping, dtype=np.dtype(np.float64))
        core = _symmetric_matrix(
            self.core_covariance,
            name="core_covariance",
            positive_semidefinite=True,
        )
        retained = mapping.shape[1]
        _require(
            len(core) >= retained,
            "core_covariance is smaller than the retained state basis",
        )
        tree_core = np.asarray(self.tree_core_solve, dtype=np.float64)
        _require(
            tree_core.shape
            == (
                self.tree_factor.node_count,
                self.tree_factor.block_size,
                len(core),
            ),
            "tree_core_solve must have shape (N, D, C)",
        )
        _require(np.all(np.isfinite(tree_core)), "tree_core_solve must be finite")
        tree_core = _readonly(tree_core, dtype=np.dtype(np.float64))

        residual_prior = state_prior - mapping @ mapping.T
        residual_minimum = float(
            np.min(np.linalg.eigvalsh(_symmetric(residual_prior)), initial=0.0)
        )
        _require(
            residual_minimum >= -1e-8,
            "state_mapping exceeds the supplied physical-state prior",
        )

        object.__setattr__(self, "state_prior_covariance", state_prior)
        object.__setattr__(self, "state_mapping", mapping)
        object.__setattr__(self, "core_covariance", core)
        object.__setattr__(self, "tree_core_solve", tree_core)
        object.__setattr__(self, "_result_id", self._compute_result_id())

    @property
    def representation(self) -> str:
        return TREE_SCHUR_COVARIANCE_REPRESENTATION

    @property
    def state_count(self) -> int:
        return len(self.state_prior_covariance)

    @property
    def retained_state_count(self) -> int:
        return self.state_mapping.shape[1]

    @property
    def gauge_parameter_count(self) -> int:
        return self.tree_factor.dimension

    @property
    def global_nuisance_count(self) -> int:
        return len(self.core_covariance) - self.retained_state_count

    @property
    def dimension(self) -> int:
        return (
            self.state_count
            + self.gauge_parameter_count
            + self.global_nuisance_count
        )

    @property
    def estimated_dense_bytes(self) -> int:
        return self.dimension**2 * np.dtype(np.float64).itemsize

    @property
    def stored_nbytes(self) -> int:
        return int(
            self.state_prior_covariance.nbytes
            + self.state_mapping.nbytes
            + self.core_covariance.nbytes
            + self.tree_factor.stored_nbytes
            + self.tree_core_solve.nbytes
        )

    @property
    def dense_materialized(self) -> bool:
        return False

    @property
    def result_id(self) -> str:
        return self._result_id

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": TREE_SCHUR_COVARIANCE_SCHEMA,
            "schema_version": TREE_SCHUR_COVARIANCE_VERSION,
            "representation": self.representation,
            "dimension": self.dimension,
            "state_count": self.state_count,
            "retained_state_count": self.retained_state_count,
            "gauge_parameter_count": self.gauge_parameter_count,
            "global_nuisance_count": self.global_nuisance_count,
            "estimated_dense_bytes": self.estimated_dense_bytes,
            "stored_nbytes": self.stored_nbytes,
            "state_prior_covariance_sha256": _array_sha256(
                self.state_prior_covariance
            ),
            "state_mapping_sha256": _array_sha256(self.state_mapping),
            "core_covariance_sha256": _array_sha256(self.core_covariance),
            "tree_factor_id": self.tree_factor.result_id,
            "tree_core_solve_sha256": _array_sha256(self.tree_core_solve),
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
    ) -> tuple[np.ndarray, bool]:
        array = np.asarray(right, dtype=np.float64)
        _require(
            np.all(np.isfinite(array)),
            "covariance right-hand side must be finite",
        )
        if array.ndim == 1:
            _require(
                array.shape == (self.dimension,),
                "covariance right-hand side dimension changed",
            )
            return array[:, None], True
        _require(
            array.ndim == 2 and array.shape[0] == self.dimension,
            "covariance right-hand side must have shape (D,) or (D, K)",
        )
        return array, False

    def apply(self, right: object) -> np.ndarray:
        """Apply the complete posterior covariance without materializing it."""

        array, squeeze = self._normalize_right(right)
        state_stop = self.state_count
        gauge_stop = state_stop + self.gauge_parameter_count
        state_right = array[:state_stop]
        gauge_right = array[state_stop:gauge_stop]
        global_right = array[gauge_stop:]

        retained_right = self.state_mapping.T @ state_right
        core_right = np.concatenate((retained_right, global_right), axis=0)
        tree_core_flat = self.tree_core_solve.reshape(
            self.gauge_parameter_count,
            len(self.core_covariance),
        )
        adjusted_core_right = core_right - tree_core_flat.T @ gauge_right
        core_result = self.core_covariance @ adjusted_core_right

        tree_result = self.tree_factor.solve(gauge_right)
        tree_result -= tree_core_flat @ core_result

        retained_result = core_result[: self.retained_state_count]
        global_result = core_result[self.retained_state_count :]
        state_result = self.state_prior_covariance @ state_right
        state_result -= self.state_mapping @ retained_right
        state_result += self.state_mapping @ retained_result
        result = np.concatenate((state_result, tree_result, global_result), axis=0)
        return result[:, 0] if squeeze else result

    def query_covariance(self, query: object) -> np.ndarray:
        """Return ``Q C Q.T`` for one or more registered linear queries."""

        matrix = np.asarray(query, dtype=np.float64)
        _require(
            matrix.ndim == 2 and matrix.shape[1] == self.dimension,
            "query must have shape (Q, D)",
        )
        _require(np.all(np.isfinite(matrix)), "query must be finite")
        result = matrix @ self.apply(matrix.T)
        return _readonly(_symmetric(result), dtype=np.dtype(np.float64))

    def marginal_covariance(self, coefficient_indices: object) -> np.ndarray:
        """Return an exact selected coefficient marginal."""

        raw = np.asarray(coefficient_indices)
        _require(
            raw.ndim == 1
            and np.issubdtype(raw.dtype, np.integer)
            and raw.dtype.kind != "b",
            "coefficient_indices must be an integer vector",
        )
        indices = np.asarray(raw, dtype=np.int64)
        _require(len(indices) >= 1, "coefficient_indices must be nonempty")
        _require(
            len(np.unique(indices)) == len(indices),
            "coefficient_indices must be unique",
        )
        _require(
            np.all((indices >= 0) & (indices < self.dimension)),
            "coefficient_indices reference an unknown coefficient",
        )
        right = np.zeros((self.dimension, len(indices)), dtype=np.float64)
        right[indices, np.arange(len(indices))] = 1.0
        solved = self.apply(right)
        return _readonly(
            _symmetric(solved[indices]),
            dtype=np.dtype(np.float64),
        )

    def materialize(self, *, maximum_bytes: int | None = None) -> np.ndarray:
        """Explicitly construct the complete dense posterior covariance."""

        _validate_maximum_bytes(maximum_bytes, self.estimated_dense_bytes)
        result = self.apply(np.eye(self.dimension, dtype=np.float64))
        return _readonly(_symmetric(result), dtype=np.dtype(np.float64))

    def __array__(self, dtype: Any | None = None) -> np.ndarray:
        materialized = self.materialize()
        return materialized if dtype is None else np.asarray(materialized, dtype=dtype)


__all__ = [
    "TREE_SCHUR_COVARIANCE_REPRESENTATION",
    "TREE_SCHUR_COVARIANCE_SCHEMA",
    "TREE_SCHUR_COVARIANCE_VERSION",
    "TreeSchurCovarianceV1",
]
