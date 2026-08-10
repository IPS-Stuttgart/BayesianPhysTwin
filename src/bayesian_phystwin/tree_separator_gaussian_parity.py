"""Shadow parity checks for the production tree-block Gaussian solver.

The accepted robust path continues to use ``TreeBlockNormalSystemV1`` and its
reusable factorization.  This module converts the same normal system to the
independent separator implementation and compares exact means, selected
covariances, log determinants, and structured residuals without materializing
the complete precision or covariance.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Final

import numpy as np

from ._canonical_contracts import frozen_finite_json_mapping, plain_json
from .tree_block_gaussian import (
    TREE_BLOCK_GAUSSIAN_SCHEMA,
    TREE_BLOCK_GAUSSIAN_VERSION,
    TreeBlockFactorizationV1,
    TreeBlockNormalSystemV1,
)
from .tree_separator_gaussian import (
    TREE_SEPARATOR_GAUSSIAN_IMPLEMENTATION,
    TreeSeparatorGaussianSystemV1,
    solve_tree_separator_gaussian,
)

TREE_SEPARATOR_GAUSSIAN_PARITY_SCHEMA: Final = (
    "bayesian_phystwin.tree_separator_gaussian_parity"
)
TREE_SEPARATOR_GAUSSIAN_PARITY_VERSION: Final = 1
TREE_SEPARATOR_GAUSSIAN_PARITY_IMPLEMENTATION: Final = (
    "production-tree-block-shadow-parity-v1"
)
TREE_SEPARATOR_GAUSSIAN_PARITY_BOUNDARY: Final = (
    "Exact numerical parity evidence only. It does not establish provider "
    "competence, calibrated uncertainty, physical-query benefit, intervention "
    "benefit, deployment safety, or state of the art."
)

_METRIC_NAMES = frozenset(
    {
        "mean_maximum_absolute_error",
        "mean_maximum_scaled_error",
        "separator_covariance_maximum_absolute_error",
        "separator_covariance_maximum_scaled_error",
        "node_covariance_maximum_absolute_error",
        "node_covariance_maximum_scaled_error",
        "node_separator_cross_maximum_absolute_error",
        "node_separator_cross_maximum_scaled_error",
        "log_determinant_absolute_error",
        "log_determinant_scaled_error",
        "structured_residual_norm",
        "structured_residual_scaled_error",
    }
)

_PASS_METRIC_NAMES = frozenset(
    {
        "mean_maximum_scaled_error",
        "separator_covariance_maximum_scaled_error",
        "node_covariance_maximum_scaled_error",
        "node_separator_cross_maximum_scaled_error",
        "log_determinant_scaled_error",
        "structured_residual_scaled_error",
    }
)


class TreeSeparatorGaussianParityError(ValueError):
    """Raised when the independent solver differs from the production path."""


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _condition_limit(value: object) -> float:
    result = _finite_nonnegative(value, name="maximum_condition_number")
    if result <= 0.0:
        raise ValueError("maximum_condition_number must be positive")
    return result


def _canonical_id(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            plain_json(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def tree_block_normal_system_id(system: TreeBlockNormalSystemV1) -> str:
    """Return an exact identity for one production normal system."""

    if not isinstance(system, TreeBlockNormalSystemV1):
        raise TypeError("system must be a TreeBlockNormalSystemV1")
    return _canonical_id(
        {
            "schema": "bayesian_phystwin.tree_block_normal_system_identity",
            "schema_version": 1,
            "production_solver_schema": TREE_BLOCK_GAUSSIAN_SCHEMA,
            "production_solver_version": TREE_BLOCK_GAUSSIAN_VERSION,
            "node_count": system.node_count,
            "block_size": system.block_size,
            "separator_size": system.global_size,
            "parent_indices_sha256": _array_sha256(system.parent_indices),
            "node_precision_sha256": _array_sha256(system.node_precision),
            "parent_coupling_sha256": _array_sha256(system.parent_coupling),
            "separator_coupling_sha256": _array_sha256(system.global_coupling),
            "separator_precision_sha256": _array_sha256(system.global_precision),
            "node_information_sha256": _array_sha256(system.node_right),
            "separator_information_sha256": _array_sha256(system.global_right),
        }
    )


def _maximum_absolute_error(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("parity arrays have different shapes")
    return 0.0 if not left.size else float(np.max(np.abs(left - right)))


def _maximum_scaled_error(
    left: np.ndarray,
    right: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> float:
    if left.shape != right.shape:
        raise ValueError("parity arrays have different shapes")
    if not left.size:
        return 0.0
    scale = atol + rtol * np.abs(right)
    return float(
        np.max(np.abs(left - right) / np.maximum(scale, np.finfo(np.float64).tiny))
    )


def _default_node_indices(node_count: int) -> tuple[int, ...]:
    if node_count <= 8:
        return tuple(range(node_count))
    return tuple(
        sorted({int(round(value)) for value in np.linspace(0, node_count - 1, num=8)})
    )


def _node_indices(
    values: Sequence[int] | None,
    *,
    node_count: int,
) -> tuple[int, ...]:
    if values is None:
        return _default_node_indices(node_count)
    if isinstance(values, (str, bytes)):
        raise TypeError("node_indices must be a sequence of integers")
    result: list[int] = []
    for position, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"node_indices[{position}] must be an integer")
        index = int(value)
        if index < 0 or index >= node_count:
            raise IndexError(f"node_indices[{position}] is out of range")
        result.append(index)
    if not result:
        raise ValueError("node_indices must not be empty")
    if len(result) != len(set(result)):
        raise ValueError("node_indices must not contain duplicates")
    return tuple(result)


def tree_block_normal_system_to_tree_separator(
    system: TreeBlockNormalSystemV1,
) -> TreeSeparatorGaussianSystemV1:
    """Convert one production normal system without changing its mathematics."""

    if not isinstance(system, TreeBlockNormalSystemV1):
        raise TypeError("system must be a TreeBlockNormalSystemV1")
    node_precision = 0.5 * (
        system.node_precision + np.swapaxes(system.node_precision, 1, 2)
    )
    separator_precision = 0.5 * (system.global_precision + system.global_precision.T)
    return TreeSeparatorGaussianSystemV1(
        parent_indices=np.asarray(system.parent_indices, dtype=np.int64),
        node_precision=np.asarray(node_precision, dtype=np.float64),
        parent_cross_precision=np.asarray(
            system.parent_coupling,
            dtype=np.float64,
        ),
        separator_cross_precision=np.asarray(
            system.global_coupling,
            dtype=np.float64,
        ),
        separator_precision=np.asarray(separator_precision, dtype=np.float64),
        node_information=np.asarray(system.node_right, dtype=np.float64),
        separator_information=np.asarray(system.global_right, dtype=np.float64),
    )


def _production_factorization(
    system: TreeBlockNormalSystemV1,
    *,
    maximum_condition_number: float,
) -> TreeBlockFactorizationV1:
    elimination = system.eliminate_nodes(
        maximum_condition_number=maximum_condition_number
    )
    if elimination.global_size:
        return elimination.factor_global(
            maximum_condition_number=maximum_condition_number
        )
    return TreeBlockFactorizationV1(
        parent_indices=elimination.parent_indices,
        node_cholesky=elimination.node_cholesky,
        parent_coupling=elimination.parent_coupling,
        global_coupling=elimination.global_coupling,
        global_cholesky=np.empty((0, 0), dtype=np.float64),
        node_condition_numbers=elimination.node_condition_numbers,
        global_condition_number=1.0,
    )


def _factorization_log_determinant(
    factorization: TreeBlockFactorizationV1,
) -> float:
    node_diagonal = np.diagonal(
        factorization.node_cholesky,
        axis1=1,
        axis2=2,
    )
    result = 2.0 * float(np.sum(np.log(node_diagonal)))
    if factorization.global_size:
        result += 2.0 * float(np.sum(np.log(np.diag(factorization.global_cholesky))))
    return result


def _reference_node_covariance(
    factorization: TreeBlockFactorizationV1,
    node_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    block_size = factorization.block_size
    global_right = np.zeros(
        (factorization.global_size, block_size),
        dtype=np.float64,
    )
    node_right = np.zeros(
        (factorization.node_count, block_size, block_size),
        dtype=np.float64,
    )
    node_right[node_index] = np.eye(block_size, dtype=np.float64)
    global_solution, node_solution = factorization.solve(
        global_right,
        node_right,
    )
    covariance = node_solution[node_index]
    return 0.5 * (covariance + covariance.T), global_solution.T


@dataclass(frozen=True, slots=True)
class TreeSeparatorGaussianParityV1:
    """Content-addressed parity diagnostics for one production normal system."""

    normal_system_id: str
    node_count: int
    block_size: int
    separator_size: int
    selected_node_indices: tuple[int, ...]
    maximum_condition_number: float
    relative_tolerance: float
    absolute_tolerance: float
    metrics: Mapping[str, float]
    maximum_node_condition_number: float
    separator_condition_number: float
    dense_precision_avoided_bytes: int
    passed: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "normal_system_id",
            _require_sha256(self.normal_system_id, name="normal_system_id"),
        )
        for name in ("node_count", "block_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.separator_size) is not int or self.separator_size < 0:
            raise ValueError("separator_size must be a non-negative integer")
        object.__setattr__(
            self,
            "selected_node_indices",
            _node_indices(
                self.selected_node_indices,
                node_count=self.node_count,
            ),
        )
        condition_limit = _condition_limit(self.maximum_condition_number)
        relative = _finite_nonnegative(
            self.relative_tolerance,
            name="relative_tolerance",
        )
        absolute = _finite_nonnegative(
            self.absolute_tolerance,
            name="absolute_tolerance",
        )
        object.__setattr__(self, "maximum_condition_number", condition_limit)
        object.__setattr__(self, "relative_tolerance", relative)
        object.__setattr__(self, "absolute_tolerance", absolute)
        if not isinstance(self.metrics, Mapping):
            raise TypeError("metrics must be a mapping")
        if set(self.metrics) != _METRIC_NAMES:
            raise ValueError("parity metric fields changed")
        metrics = {
            name: _finite_nonnegative(value, name=name)
            for name, value in self.metrics.items()
        }
        object.__setattr__(
            self,
            "metrics",
            frozen_finite_json_mapping(metrics, name="parity metrics"),
        )
        maximum_node_condition = _finite_nonnegative(
            self.maximum_node_condition_number,
            name="maximum_node_condition_number",
        )
        separator_condition = _finite_nonnegative(
            self.separator_condition_number,
            name="separator_condition_number",
        )
        object.__setattr__(
            self,
            "maximum_node_condition_number",
            maximum_node_condition,
        )
        object.__setattr__(
            self,
            "separator_condition_number",
            separator_condition,
        )
        if type(self.dense_precision_avoided_bytes) is not int:
            raise TypeError("dense_precision_avoided_bytes must be an integer")
        if self.dense_precision_avoided_bytes < 0:
            raise ValueError("dense_precision_avoided_bytes must be non-negative")
        if type(self.passed) is not bool:
            raise TypeError("passed must be a bool")
        expected_passed = all(metrics[name] <= 1.0 for name in _PASS_METRIC_NAMES)
        if self.passed is not expected_passed:
            raise ValueError("passed disagrees with scaled parity metrics")

    @property
    def parity_id(self) -> str:
        return _canonical_id(self.descriptor())

    def descriptor(self) -> Mapping[str, object]:
        return frozen_finite_json_mapping(
            {
                "schema": TREE_SEPARATOR_GAUSSIAN_PARITY_SCHEMA,
                "schema_version": TREE_SEPARATOR_GAUSSIAN_PARITY_VERSION,
                "implementation": TREE_SEPARATOR_GAUSSIAN_PARITY_IMPLEMENTATION,
                "production_solver_schema": TREE_BLOCK_GAUSSIAN_SCHEMA,
                "production_solver_version": TREE_BLOCK_GAUSSIAN_VERSION,
                "normal_system_id": self.normal_system_id,
                "independent_solver_implementation": (
                    TREE_SEPARATOR_GAUSSIAN_IMPLEMENTATION
                ),
                "node_count": self.node_count,
                "block_size": self.block_size,
                "separator_size": self.separator_size,
                "selected_node_indices": list(self.selected_node_indices),
                "maximum_condition_number": self.maximum_condition_number,
                "relative_tolerance": self.relative_tolerance,
                "absolute_tolerance": self.absolute_tolerance,
                "metrics": dict(self.metrics),
                "maximum_node_condition_number": (self.maximum_node_condition_number),
                "separator_condition_number": self.separator_condition_number,
                "dense_precision_avoided_bytes": (self.dense_precision_avoided_bytes),
                "passed": self.passed,
                "claim_boundary": TREE_SEPARATOR_GAUSSIAN_PARITY_BOUNDARY,
            },
            name="tree-separator Gaussian parity descriptor",
        )

    def to_dict(self) -> dict[str, object]:
        return {**dict(self.descriptor()), "parity_id": self.parity_id}


def evaluate_tree_separator_gaussian_parity(
    system: TreeBlockNormalSystemV1,
    *,
    maximum_condition_number: float,
    node_indices: Sequence[int] | None = None,
    relative_tolerance: float = 3.0e-11,
    absolute_tolerance: float = 3.0e-12,
) -> TreeSeparatorGaussianParityV1:
    """Compare the independent solver to the admitted production factorization."""

    if not isinstance(system, TreeBlockNormalSystemV1):
        raise TypeError("system must be a TreeBlockNormalSystemV1")
    condition_limit = _condition_limit(maximum_condition_number)
    rtol = _finite_nonnegative(relative_tolerance, name="relative_tolerance")
    atol = _finite_nonnegative(absolute_tolerance, name="absolute_tolerance")
    selected = _node_indices(node_indices, node_count=system.node_count)

    factorization = _production_factorization(
        system,
        maximum_condition_number=condition_limit,
    )
    reference_global, reference_nodes = factorization.solve(
        system.global_right,
        system.node_right,
    )
    independent = solve_tree_separator_gaussian(
        tree_block_normal_system_to_tree_separator(system)
    )

    candidate_mean = np.concatenate(
        (independent.separator_mean, independent.node_mean.reshape(-1))
    )
    reference_mean = np.concatenate((reference_global, reference_nodes.reshape(-1)))
    reference_separator_covariance = factorization.global_marginal_covariance()
    node_covariance_absolute = 0.0
    node_covariance_scaled = 0.0
    node_cross_absolute = 0.0
    node_cross_scaled = 0.0
    for node_index in selected:
        reference_covariance, reference_cross = _reference_node_covariance(
            factorization,
            node_index,
        )
        candidate_covariance = independent.node_covariance[node_index]
        candidate_cross = independent.node_separator_cross_covariance[node_index]
        node_covariance_absolute = max(
            node_covariance_absolute,
            _maximum_absolute_error(candidate_covariance, reference_covariance),
        )
        node_covariance_scaled = max(
            node_covariance_scaled,
            _maximum_scaled_error(
                candidate_covariance,
                reference_covariance,
                rtol=rtol,
                atol=atol,
            ),
        )
        node_cross_absolute = max(
            node_cross_absolute,
            _maximum_absolute_error(candidate_cross, reference_cross),
        )
        node_cross_scaled = max(
            node_cross_scaled,
            _maximum_scaled_error(
                candidate_cross,
                reference_cross,
                rtol=rtol,
                atol=atol,
            ),
        )

    reference_log_determinant = _factorization_log_determinant(factorization)
    log_absolute = abs(
        independent.log_determinant_precision - reference_log_determinant
    )
    log_scaled = log_absolute / max(
        atol + rtol * abs(reference_log_determinant),
        np.finfo(np.float64).tiny,
    )
    global_residual, node_residual = system.residual(
        independent.separator_mean,
        independent.node_mean,
    )
    residual_norm = float(
        np.linalg.norm(np.concatenate((global_residual, node_residual.reshape(-1))))
    )
    right_norm = float(
        np.linalg.norm(
            np.concatenate((system.global_right, system.node_right.reshape(-1)))
        )
    )
    residual_scaled = residual_norm / max(
        atol + rtol * (1.0 + right_norm),
        np.finfo(np.float64).tiny,
    )

    metrics = {
        "mean_maximum_absolute_error": _maximum_absolute_error(
            candidate_mean,
            reference_mean,
        ),
        "mean_maximum_scaled_error": _maximum_scaled_error(
            candidate_mean,
            reference_mean,
            rtol=rtol,
            atol=atol,
        ),
        "separator_covariance_maximum_absolute_error": (
            _maximum_absolute_error(
                independent.separator_covariance,
                reference_separator_covariance,
            )
        ),
        "separator_covariance_maximum_scaled_error": _maximum_scaled_error(
            independent.separator_covariance,
            reference_separator_covariance,
            rtol=rtol,
            atol=atol,
        ),
        "node_covariance_maximum_absolute_error": node_covariance_absolute,
        "node_covariance_maximum_scaled_error": node_covariance_scaled,
        "node_separator_cross_maximum_absolute_error": node_cross_absolute,
        "node_separator_cross_maximum_scaled_error": node_cross_scaled,
        "log_determinant_absolute_error": log_absolute,
        "log_determinant_scaled_error": log_scaled,
        "structured_residual_norm": residual_norm,
        "structured_residual_scaled_error": residual_scaled,
    }
    passed = all(metrics[name] <= 1.0 for name in _PASS_METRIC_NAMES)
    return TreeSeparatorGaussianParityV1(
        normal_system_id=tree_block_normal_system_id(system),
        node_count=system.node_count,
        block_size=system.block_size,
        separator_size=system.global_size,
        selected_node_indices=selected,
        maximum_condition_number=condition_limit,
        relative_tolerance=rtol,
        absolute_tolerance=atol,
        metrics=metrics,
        maximum_node_condition_number=(factorization.maximum_node_condition_number),
        separator_condition_number=factorization.global_condition_number,
        dense_precision_avoided_bytes=system.estimated_dense_precision_bytes,
        passed=passed,
    )


def require_tree_separator_gaussian_parity(
    system: TreeBlockNormalSystemV1,
    *,
    maximum_condition_number: float,
    node_indices: Sequence[int] | None = None,
    relative_tolerance: float = 3.0e-11,
    absolute_tolerance: float = 3.0e-12,
) -> TreeSeparatorGaussianParityV1:
    """Return the parity report or fail closed on numerical disagreement."""

    report = evaluate_tree_separator_gaussian_parity(
        system,
        maximum_condition_number=maximum_condition_number,
        node_indices=node_indices,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    if not report.passed:
        raise TreeSeparatorGaussianParityError(
            "tree-separator Gaussian shadow parity failed: "
            f"parity_id={report.parity_id}, metrics={dict(report.metrics)}"
        )
    return report


__all__ = [
    "TREE_SEPARATOR_GAUSSIAN_PARITY_BOUNDARY",
    "TREE_SEPARATOR_GAUSSIAN_PARITY_IMPLEMENTATION",
    "TREE_SEPARATOR_GAUSSIAN_PARITY_SCHEMA",
    "TREE_SEPARATOR_GAUSSIAN_PARITY_VERSION",
    "TreeSeparatorGaussianParityError",
    "TreeSeparatorGaussianParityV1",
    "evaluate_tree_separator_gaussian_parity",
    "require_tree_separator_gaussian_parity",
    "tree_block_normal_system_id",
    "tree_block_normal_system_to_tree_separator",
]
