"""Exact covariance queries for factorized tree-block posteriors.

The historical dense covariance order is

``state, gauge nodes, shared/view/anchor biases``.

``TreeBlockPosteriorCovarianceV1`` instead stores a prior-preserving physical
state expansion and a factorization whose internal order is

``retained state, biases, gauge nodes``.

This module applies the public posterior covariance, evaluates arbitrary linear
query covariance, and extracts selected marginals without constructing the
complete joint covariance.  It is additive and does not change the version-1
factor or result identities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from ._gauge_aware_contracts import _readonly, _require
from .tree_block_claim_contract import validate_tree_block_covariance
from .tree_block_sparse_gauge_belief import TreeBlockPosteriorCovarianceV1

TREE_BLOCK_POSTERIOR_OPERATOR_SCHEMA: Final = (
    "bayesian_phystwin.tree_block_posterior_operator"
)
TREE_BLOCK_POSTERIOR_OPERATOR_VERSION: Final = 1
TREE_BLOCK_POSTERIOR_OPERATOR_BOUNDARY: Final = (
    "Exact numerical covariance access only. It does not establish observation "
    "competence, calibrated uncertainty, physical-query benefit, intervention "
    "benefit, deployment safety, or state of the art."
)


def _numeric_array(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    _require(
        raw.dtype.kind in {"i", "u", "f"},
        f"{name} must be numeric",
    )
    result = np.asarray(raw, dtype=np.float64)
    _require(np.all(np.isfinite(result)), f"{name} must be finite")
    return result


def _right_hand_side(
    value: object,
    *,
    dimension: int,
) -> tuple[np.ndarray, bool]:
    right = _numeric_array(value, name="right")
    if right.shape == (dimension,):
        return right[:, None], True
    _require(
        right.ndim == 2 and right.shape[0] == dimension,
        f"right must have shape ({dimension},) or ({dimension}, K)",
    )
    _require(right.shape[1] >= 1, "right must contain at least one column")
    return right, False


def _query_matrix(
    value: object,
    *,
    name: str,
    dimension: int,
) -> np.ndarray:
    query = _numeric_array(value, name=name)
    _require(query.ndim == 2, f"{name} must have two dimensions")
    _require(
        query.shape[1] == dimension,
        f"{name} must have shape (Q, {dimension})",
    )
    _require(query.shape[0] >= 1, f"{name} must contain at least one row")
    return query


def _selected_indices(value: object, *, dimension: int) -> np.ndarray:
    raw = np.asarray(value)
    _require(
        raw.ndim == 1
        and np.issubdtype(raw.dtype, np.integer)
        and raw.dtype.kind != "b",
        "indices must be an integer vector",
    )
    indices = np.asarray(raw, dtype=np.int64)
    _require(len(indices) >= 1, "indices must not be empty")
    _require(
        np.all((indices >= 0) & (indices < dimension)),
        "indices contain an out-of-range coefficient",
    )
    _require(
        len(np.unique(indices)) == len(indices),
        "indices must not contain duplicates",
    )
    return indices


@dataclass(frozen=True, slots=True)
class TreeBlockPosteriorOperatorV1:
    """Reusable exact covariance operator for one validated posterior."""

    covariance: TreeBlockPosteriorCovarianceV1

    def __post_init__(self) -> None:
        validate_tree_block_covariance(self.covariance)

    @property
    def schema(self) -> str:
        return TREE_BLOCK_POSTERIOR_OPERATOR_SCHEMA

    @property
    def schema_version(self) -> int:
        return TREE_BLOCK_POSTERIOR_OPERATOR_VERSION

    @property
    def dimension(self) -> int:
        return self.covariance.dimension

    @property
    def state_count(self) -> int:
        return self.covariance.state_count

    @property
    def gauge_parameter_count(self) -> int:
        return self.covariance.gauge_parameter_count

    @property
    def bias_count(self) -> int:
        return self.covariance.bias_count

    @property
    def dense_covariance_avoided_bytes(self) -> int:
        return self.covariance.estimated_dense_covariance_bytes

    def apply(self, right: object) -> np.ndarray:
        """Apply the complete public covariance to one or more vectors.

        The returned coefficient order is the historical public order:
        physical state, flattened gauge nodes, then all global bias variables.
        """

        matrix, was_vector = _right_hand_side(right, dimension=self.dimension)
        state_stop = self.state_count
        gauge_stop = state_stop + self.gauge_parameter_count
        state_right = matrix[:state_stop]
        gauge_right = matrix[state_stop:gauge_stop]
        bias_right = matrix[gauge_stop:]

        mapping = self.covariance.state_mapping
        retained = self.covariance.retained_state_count
        retained_right = mapping.T @ state_right
        global_right = np.concatenate((retained_right, bias_right), axis=0)
        node_right = gauge_right.reshape(
            self.covariance.factorization.node_count,
            self.covariance.factorization.block_size,
            matrix.shape[1],
        )
        global_solution, node_solution = self.covariance.factorization.solve(
            global_right,
            node_right,
        )

        state_solution = self.covariance.state_prior_covariance @ state_right
        state_solution += mapping @ (global_solution[:retained] - retained_right)
        result = np.concatenate(
            (
                state_solution,
                node_solution.reshape(self.gauge_parameter_count, matrix.shape[1]),
                global_solution[retained:],
            ),
            axis=0,
        )
        return _readonly(result[:, 0] if was_vector else result)

    def linear_covariance(self, query: object) -> np.ndarray:
        """Return ``Q P Q.T`` for a public-order linear query ``Q``."""

        query_matrix = _query_matrix(
            query,
            name="query",
            dimension=self.dimension,
        )
        covariance_times_query = self.apply(query_matrix.T)
        result = query_matrix @ covariance_times_query
        return _readonly(0.5 * (result + result.T))

    def cross_covariance(
        self,
        left_query: object,
        right_query: object,
    ) -> np.ndarray:
        """Return ``L P R.T`` without materializing ``P``."""

        left = _query_matrix(
            left_query,
            name="left_query",
            dimension=self.dimension,
        )
        right = _query_matrix(
            right_query,
            name="right_query",
            dimension=self.dimension,
        )
        return _readonly(left @ self.apply(right.T))

    def marginal_covariance(self, indices: object) -> np.ndarray:
        """Return one unique selected coefficient marginal in caller order."""

        selected = _selected_indices(indices, dimension=self.dimension)
        basis: np.ndarray = np.zeros((self.dimension, len(selected)), dtype=np.float64)
        basis[selected, np.arange(len(selected))] = 1.0
        columns = self.apply(basis)
        result = columns[selected]
        return _readonly(0.5 * (result + result.T))


def apply_tree_block_posterior_covariance(
    covariance: TreeBlockPosteriorCovarianceV1,
    right: object,
) -> np.ndarray:
    """Apply a validated factorized posterior covariance."""

    return TreeBlockPosteriorOperatorV1(covariance).apply(right)


def tree_block_linear_query_covariance(
    covariance: TreeBlockPosteriorCovarianceV1,
    query: object,
) -> np.ndarray:
    """Evaluate an arbitrary linear query covariance from the factors."""

    return TreeBlockPosteriorOperatorV1(covariance).linear_covariance(query)


def tree_block_cross_covariance(
    covariance: TreeBlockPosteriorCovarianceV1,
    left_query: object,
    right_query: object,
) -> np.ndarray:
    """Evaluate a cross covariance between two public-order queries."""

    return TreeBlockPosteriorOperatorV1(covariance).cross_covariance(
        left_query,
        right_query,
    )


def tree_block_selected_marginal_covariance(
    covariance: TreeBlockPosteriorCovarianceV1,
    indices: object,
) -> np.ndarray:
    """Extract selected coefficient covariance without complete materialization."""

    return TreeBlockPosteriorOperatorV1(covariance).marginal_covariance(indices)


__all__ = [
    "TREE_BLOCK_POSTERIOR_OPERATOR_BOUNDARY",
    "TREE_BLOCK_POSTERIOR_OPERATOR_SCHEMA",
    "TREE_BLOCK_POSTERIOR_OPERATOR_VERSION",
    "TreeBlockPosteriorOperatorV1",
    "apply_tree_block_posterior_covariance",
    "tree_block_cross_covariance",
    "tree_block_linear_query_covariance",
    "tree_block_selected_marginal_covariance",
]
