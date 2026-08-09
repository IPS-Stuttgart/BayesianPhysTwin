"""Fail-closed validation for claim-bearing tree-block posterior factors.

The numerical tree solver stores Cholesky factors and compact parameter-layout
metadata rather than a dense covariance.  Dataclass construction verifies array
shapes, but a manually assembled object could otherwise pair inconsistent
factors, condition diagnostics, or coefficient partitions.  Claim-bearing
consumers call this module before admitting such a result.
"""

from __future__ import annotations

from numbers import Real
from typing import Final

import numpy as np

from .tree_block_gaussian import TreeBlockFactorizationV1
from .tree_block_sparse_gauge_belief import (
    TreeBlockGaugeAwareBeliefResultV1,
    TreeBlockPosteriorCovarianceV1,
)

TREE_BLOCK_CLAIM_CONTRACT_SCHEMA: Final = (
    "bayesian_phystwin.tree_block_claim_contract"
)
TREE_BLOCK_CLAIM_CONTRACT_VERSION: Final = 1
TREE_BLOCK_CLAIM_CONTRACT_BOUNDARY: Final = (
    "Structural and numerical-integrity validation only. Passing this contract "
    "does not establish observation competence, uncertainty calibration, "
    "physical-query benefit, intervention benefit, deployment safety, or state "
    "of the art."
)
_STRICT_IMPLEMENTATION_ID: Final = "tree-block-group-mixture-strict-admission-v2"


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _finite_array(value: object, *, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    _require(array.ndim == ndim, f"{name} must have {ndim} dimensions")
    _require(
        array.dtype.kind not in {"b", "O", "U", "S"},
        f"{name} must be numeric",
    )
    result = np.asarray(array, dtype=np.float64)
    _require(np.all(np.isfinite(result)), f"{name} must be finite")
    return result


def _validate_tree_topology(
    parents: np.ndarray,
    parent_coupling: np.ndarray,
) -> None:
    node_count = len(parents)
    _require(node_count >= 1, "the factorization must contain at least one node")
    _require(parents.shape == (node_count,), "parent_indices shape changed")
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


def _validate_cholesky_blocks(factors: np.ndarray) -> None:
    _require(
        factors.shape[1] == factors.shape[2],
        "node Cholesky blocks must be square",
    )
    _require(
        np.allclose(factors, np.tril(factors), atol=1e-14, rtol=0.0),
        "node Cholesky factors must be lower triangular",
    )
    _require(
        np.all(np.diagonal(factors, axis1=1, axis2=2) > 0.0),
        "node Cholesky factors must have positive diagonal",
    )


def _validate_cholesky_matrix(factor: np.ndarray) -> None:
    _require(
        factor.shape[0] == factor.shape[1],
        "global Cholesky factor must be square",
    )
    _require(
        np.allclose(factor, np.tril(factor), atol=1e-14, rtol=0.0),
        "global Cholesky factor must be lower triangular",
    )
    _require(
        np.all(np.diag(factor) > 0.0),
        "global Cholesky factor must have positive diagonal",
    )


def _condition_number(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    _require(
        np.isfinite(result) and result >= 1.0,
        f"{name} must be finite and at least one",
    )
    return result


def validate_tree_block_factorization(
    factorization: TreeBlockFactorizationV1,
) -> TreeBlockFactorizationV1:
    """Validate that stored factors and their diagnostics are self-consistent."""

    if not isinstance(factorization, TreeBlockFactorizationV1):
        raise TypeError("factorization must be a TreeBlockFactorizationV1")
    parents_raw = np.asarray(factorization.parent_indices)
    _require(
        parents_raw.ndim == 1
        and np.issubdtype(parents_raw.dtype, np.integer)
        and parents_raw.dtype.kind != "b",
        "parent_indices must be an integer vector",
    )
    parents = np.asarray(parents_raw, dtype=np.int64)
    node_factors = _finite_array(
        factorization.node_cholesky,
        name="node_cholesky",
        ndim=3,
    )
    node_count, block_size, second = node_factors.shape
    _require(block_size == second, "node Cholesky blocks must be square")
    parent_coupling = _finite_array(
        factorization.parent_coupling,
        name="parent_coupling",
        ndim=3,
    )
    global_coupling = _finite_array(
        factorization.global_coupling,
        name="global_coupling",
        ndim=3,
    )
    global_factor = _finite_array(
        factorization.global_cholesky,
        name="global_cholesky",
        ndim=2,
    )
    _require(
        parent_coupling.shape == (node_count, block_size, block_size),
        "parent_coupling shape changed",
    )
    global_size = len(global_factor)
    _require(
        global_coupling.shape == (node_count, block_size, global_size),
        "global_coupling shape changed",
    )
    _validate_tree_topology(parents, parent_coupling)
    _validate_cholesky_blocks(node_factors)
    _validate_cholesky_matrix(global_factor)

    reported = _finite_array(
        factorization.node_condition_numbers,
        name="node_condition_numbers",
        ndim=1,
    )
    _require(reported.shape == (node_count,), "node condition shape changed")
    _require(
        np.all(reported >= 1.0),
        "node condition numbers must be at least one",
    )
    actual = np.asarray(
        [np.linalg.cond(factor @ factor.T) for factor in node_factors],
        dtype=np.float64,
    )
    _require(
        np.allclose(reported, actual, atol=1e-10, rtol=1e-8),
        "node condition numbers do not match the Cholesky factors",
    )
    global_condition = _condition_number(
        factorization.global_condition_number,
        name="global_condition_number",
    )
    actual_global = float(np.linalg.cond(global_factor @ global_factor.T))
    _require(
        np.isclose(global_condition, actual_global, atol=1e-10, rtol=1e-8),
        "global condition number does not match the Cholesky factor",
    )
    return factorization


def validate_tree_block_covariance(
    covariance: TreeBlockPosteriorCovarianceV1,
) -> TreeBlockPosteriorCovarianceV1:
    """Validate the state expansion and its compact tree factorization."""

    if not isinstance(covariance, TreeBlockPosteriorCovarianceV1):
        raise TypeError("covariance must be a TreeBlockPosteriorCovarianceV1")
    state_prior = _finite_array(
        covariance.state_prior_covariance,
        name="state_prior_covariance",
        ndim=2,
    )
    _require(
        state_prior.shape[0] == state_prior.shape[1],
        "state_prior_covariance must be square",
    )
    _require(
        np.allclose(state_prior, state_prior.T, atol=1e-10, rtol=1e-10),
        "state_prior_covariance must be symmetric",
    )
    if len(state_prior):
        _require(
            np.min(np.linalg.eigvalsh(state_prior)) >= -1e-9,
            "state_prior_covariance must be positive semidefinite",
        )
    mapping = _finite_array(covariance.state_mapping, name="state_mapping", ndim=2)
    _require(
        mapping.shape[0] == len(state_prior),
        "state_mapping has changed state dimension",
    )
    if type(covariance.bias_count) is not int or covariance.bias_count < 0:
        raise ValueError("bias_count must be a nonnegative integer")
    factorization = validate_tree_block_factorization(covariance.factorization)
    _require(
        factorization.global_size == mapping.shape[1] + covariance.bias_count,
        "factorization global dimension differs from state/bias layout",
    )
    return covariance


def validate_tree_block_result(
    result: TreeBlockGaugeAwareBeliefResultV1,
    *,
    require_strict_admission: bool = False,
) -> TreeBlockGaugeAwareBeliefResultV1:
    """Validate coefficient layout, factor integrity, and strict-admission binding."""

    if not isinstance(result, TreeBlockGaugeAwareBeliefResultV1):
        raise TypeError("result must be a TreeBlockGaugeAwareBeliefResultV1")
    covariance = validate_tree_block_covariance(result.covariance)
    _require(
        covariance.state_count == len(result.state_coefficients),
        "covariance state dimension differs from state coefficients",
    )
    _require(
        covariance.gauge_parameter_count == len(result.gauge_delta),
        "covariance gauge dimension differs from gauge coefficients",
    )
    bias_count = sum(
        len(value)
        for value in (
            result.shared_bias_coefficients,
            result.view_bias_coefficients,
            result.anchor_bias_coefficients,
        )
    )
    _require(
        covariance.bias_count == bias_count,
        "covariance bias dimension differs from bias coefficients",
    )
    if result.inference_admissible:
        _require(
            covariance.retained_state_count
            == result.identifiable_state_transform.shape[1],
            "retained-state covariance differs from identifiability",
        )
    if type(require_strict_admission) is not bool:
        raise TypeError("require_strict_admission must be a bool")
    if require_strict_admission:
        diagnostics = result.diagnostics
        _require(
            diagnostics.get("implementation_id") == _STRICT_IMPLEMENTATION_ID,
            "result is not bound to strict tree-block admission",
        )
        _require(
            diagnostics.get("strict_admission_version") == 2,
            "result lacks strict tree-block admission version 2",
        )
        _require(
            diagnostics.get("strict_admission_passed")
            is result.inference_admissible,
            "strict admission status differs from result admissibility",
        )
    return result


__all__ = [
    "TREE_BLOCK_CLAIM_CONTRACT_BOUNDARY",
    "TREE_BLOCK_CLAIM_CONTRACT_SCHEMA",
    "TREE_BLOCK_CLAIM_CONTRACT_VERSION",
    "validate_tree_block_covariance",
    "validate_tree_block_factorization",
    "validate_tree_block_result",
]
