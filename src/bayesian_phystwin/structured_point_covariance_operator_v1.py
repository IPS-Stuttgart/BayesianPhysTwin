"""Matrix-free linear algebra for structured metric point covariance.

The underlying :class:`StructuredPointCovarianceV1` stores a block-diagonal
conditional covariance plus labeled low-rank shared roots. This module exposes
exact matrix actions, quadratic forms, positive-definite solves, log
determinants, and zero-mean sampling without constructing the complete
``(3N, 3N)`` matrix.

Solves and log determinants use the Woodbury and matrix-determinant identities.
They fail closed unless every local ``3 x 3`` block is strictly positive
definite. Matrix actions and sampling remain available for positive
semidefinite local blocks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_integer,
    immutable_array,
    literal_lower_hex,
    plain_json,
)
from ._portable_contracts import content_id
from .structured_point_covariance import (
    ProjectedStructuredPointCovarianceV1,
    StructuredPointCovarianceV1,
)

STRUCTURED_POINT_COVARIANCE_OPERATOR_SCHEMA: Final = (
    "bayesian_phystwin.structured_point_covariance_operator"
)
STRUCTURED_POINT_COVARIANCE_OPERATOR_VERSION: Final = 1
STRUCTURED_POINT_COVARIANCE_OPERATOR_SEMANTICS: Final = (
    "exact-block-local-plus-low-rank-matrix-actions-woodbury-v1"
)
STRUCTURED_POINT_COVARIANCE_OPERATOR_CLAIM_BOUNDARY: Final = (
    "Exact numerical operations over an already supplied covariance only. "
    "This operator does not calibrate uncertainty, establish physical-state "
    "identifiability, authorize an update, or establish downstream benefit."
)


def _real_float64_array(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    array = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _immutable_float64(value: object) -> np.ndarray:
    return immutable_array(value, dtype=np.float64)


def _finite_result(value: np.ndarray, *, name: str) -> np.ndarray:
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} overflowed finite float64 representation")
    return value


def _rhs_matrix(
    value: object,
    *,
    dimension: int,
    name: str,
) -> tuple[np.ndarray, bool]:
    rhs = _real_float64_array(value, name=name)
    vector = rhs.ndim == 1
    if vector:
        rhs = rhs[:, None]
    elif rhs.ndim != 2:
        raise ValueError(f"{name} must have shape (dimension,) or (dimension, batch)")
    if rhs.shape[0] != dimension:
        raise ValueError(f"{name} first dimension must equal {dimension}")
    if rhs.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one right-hand side")
    return rhs, vector


def _restore_rhs_shape(value: np.ndarray, *, vector: bool) -> np.ndarray:
    restored = value[:, 0] if vector else value
    return _immutable_float64(restored)


def _shared_factor_matrix(covariance: StructuredPointCovarianceV1) -> np.ndarray:
    if not covariance.shared_factors_m:
        return np.zeros((covariance.state_dimension, 0), dtype=np.float64)
    factors = tuple(
        factor.reshape((covariance.state_dimension, factor.shape[2]))
        for factor in covariance.shared_factors_m.values()
    )
    return np.ascontiguousarray(np.concatenate(factors, axis=1), dtype=np.float64)


def _local_cholesky_factors(
    covariance: StructuredPointCovarianceV1,
) -> np.ndarray:
    factors = np.empty_like(covariance.local_covariance_m2)
    for index, block in enumerate(covariance.local_covariance_m2):
        try:
            factors[index] = np.linalg.cholesky(block)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "solve and logdet require strictly positive-definite local blocks"
            ) from error
    return factors


def _solve_cholesky(factor: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    intermediate = np.linalg.solve(factor, rhs)
    return np.linalg.solve(factor.T, intermediate)


def _solve_local_blocks(
    factors: np.ndarray,
    rhs: np.ndarray,
    *,
    point_count: int,
) -> np.ndarray:
    batch = rhs.shape[1]
    blocks = rhs.reshape((point_count, 3, batch))
    result = np.empty_like(blocks)
    for index, factor in enumerate(factors):
        result[index] = _solve_cholesky(factor, blocks[index])
    return result.reshape((3 * point_count, batch))


@dataclass(frozen=True, slots=True)
class StructuredPointCovarianceOperatorV1:
    """Content-addressed matrix-free view of one structured covariance."""

    covariance: StructuredPointCovarianceV1
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.covariance, StructuredPointCovarianceV1):
            raise TypeError("covariance must be a StructuredPointCovarianceV1")
        if self.covariance.artifact_id is None:
            raise ValueError("covariance must have a content identity")
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="structured covariance operator metadata",
        )
        object.__setattr__(self, "metadata", metadata)
        expected_id = content_id(self.descriptor())
        supplied_id = self.artifact_id
        if supplied_id is not None:
            supplied_id = literal_lower_hex(
                supplied_id,
                name="artifact_id",
                lengths={64},
            )
            if supplied_id != expected_id:
                raise ValueError(
                    "structured covariance operator artifact_id does not match content"
                )
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def point_count(self) -> int:
        return self.covariance.point_count

    @property
    def state_dimension(self) -> int:
        return self.covariance.state_dimension

    @property
    def shared_rank(self) -> int:
        return self.covariance.shared_rank

    @property
    def shared_component_names(self) -> tuple[str, ...]:
        return self.covariance.shared_component_names

    @property
    def supports_woodbury_solve(self) -> bool:
        """Return whether the local base admits exact Woodbury solves."""

        try:
            _local_cholesky_factors(self.covariance)
        except ValueError:
            return False
        return True

    def descriptor(self) -> dict[str, object]:
        covariance_id = self.covariance.artifact_id
        if covariance_id is None:
            raise ValueError("covariance must have a content identity")
        return {
            "schema": STRUCTURED_POINT_COVARIANCE_OPERATOR_SCHEMA,
            "schema_version": STRUCTURED_POINT_COVARIANCE_OPERATOR_VERSION,
            "semantics": STRUCTURED_POINT_COVARIANCE_OPERATOR_SEMANTICS,
            "covariance_artifact_id": covariance_id,
            "state_dimension": self.state_dimension,
            "metadata": plain_json(self.metadata),
            "claim_boundary": (STRUCTURED_POINT_COVARIANCE_OPERATOR_CLAIM_BOUNDARY),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}

    def summary(self) -> dict[str, object]:
        return {
            "schema": STRUCTURED_POINT_COVARIANCE_OPERATOR_SCHEMA,
            "schema_version": STRUCTURED_POINT_COVARIANCE_OPERATOR_VERSION,
            "artifact_id": self.artifact_id,
            "covariance_artifact_id": self.covariance.artifact_id,
            "point_count": self.point_count,
            "state_dimension": self.state_dimension,
            "shared_rank": self.shared_rank,
            "shared_component_names": list(self.shared_component_names),
            "supports_woodbury_solve": self.supports_woodbury_solve,
            "claim_boundary": (STRUCTURED_POINT_COVARIANCE_OPERATOR_CLAIM_BOUNDARY),
        }

    def matmul(self, rhs: object) -> np.ndarray:
        """Return ``Sigma @ rhs`` for one vector or a batch of columns."""

        matrix, vector = _rhs_matrix(
            rhs,
            dimension=self.state_dimension,
            name="rhs",
        )
        batch = matrix.shape[1]
        blocks = matrix.reshape((self.point_count, 3, batch))
        with np.errstate(over="ignore", invalid="ignore"):
            local = np.einsum(
                "nij,njk->nik",
                self.covariance.local_covariance_m2,
                blocks,
                optimize=True,
            ).reshape((self.state_dimension, batch))
        result = np.array(local, copy=True)
        with np.errstate(over="ignore", invalid="ignore"):
            for factor in self.covariance.shared_factors_m.values():
                flat = factor.reshape((self.state_dimension, factor.shape[2]))
                result += flat @ (flat.T @ matrix)
        _finite_result(result, name="structured covariance matrix action")
        return _restore_rhs_shape(result, vector=vector)

    def component_matmul(self, component: str, rhs: object) -> np.ndarray:
        """Apply only the local or one named shared covariance component."""

        if type(component) is not str or not component:
            raise ValueError("component must be nonempty literal text")
        matrix, vector = _rhs_matrix(
            rhs,
            dimension=self.state_dimension,
            name="rhs",
        )
        batch = matrix.shape[1]
        if component == "local":
            blocks = matrix.reshape((self.point_count, 3, batch))
            with np.errstate(over="ignore", invalid="ignore"):
                result = np.einsum(
                    "nij,njk->nik",
                    self.covariance.local_covariance_m2,
                    blocks,
                    optimize=True,
                ).reshape((self.state_dimension, batch))
        else:
            try:
                factor = self.covariance.shared_factors_m[component]
            except KeyError as error:
                raise ValueError(
                    "component must be 'local' or a retained shared component"
                ) from error
            flat = factor.reshape((self.state_dimension, factor.shape[2]))
            with np.errstate(over="ignore", invalid="ignore"):
                result = flat @ (flat.T @ matrix)
        _finite_result(
            result,
            name=f"structured covariance {component} action",
        )
        return _restore_rhs_shape(result, vector=vector)

    def quadratic_form(self, vector: object) -> float:
        """Return the exact scalar ``vector.T @ Sigma @ vector``."""

        value, was_vector = _rhs_matrix(
            vector,
            dimension=self.state_dimension,
            name="vector",
        )
        if not was_vector:
            raise ValueError("quadratic_form requires one vector")
        applied = self.matmul(value[:, 0])
        result = float(value[:, 0] @ applied)
        if not np.isfinite(result):
            raise ValueError("structured covariance quadratic form is not finite")
        return result

    def solve(self, rhs: object) -> np.ndarray:
        """Solve ``Sigma @ x = rhs`` with block Cholesky and Woodbury."""

        matrix, vector = _rhs_matrix(
            rhs,
            dimension=self.state_dimension,
            name="rhs",
        )
        local_factors = _local_cholesky_factors(self.covariance)
        local_solution = _solve_local_blocks(
            local_factors,
            matrix,
            point_count=self.point_count,
        )
        shared = _shared_factor_matrix(self.covariance)
        if shared.shape[1] == 0:
            _finite_result(
                local_solution,
                name="structured covariance solve",
            )
            return _restore_rhs_shape(local_solution, vector=vector)

        local_shared = _solve_local_blocks(
            local_factors,
            shared,
            point_count=self.point_count,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            core = np.eye(shared.shape[1], dtype=np.float64) + shared.T @ local_shared
        core = 0.5 * (core + core.T)
        _finite_result(core, name="Woodbury core")
        try:
            core_factor = np.linalg.cholesky(core)
        except np.linalg.LinAlgError as error:
            raise ValueError("Woodbury core must be positive definite") from error
        with np.errstate(over="ignore", invalid="ignore"):
            core_rhs = shared.T @ local_solution
            correction_weight = _solve_cholesky(core_factor, core_rhs)
            result = local_solution - local_shared @ correction_weight
        _finite_result(result, name="structured covariance solve")
        return _restore_rhs_shape(result, vector=vector)

    def logdet(self) -> float:
        """Return the exact log determinant for a positive-definite local base."""

        local_factors = _local_cholesky_factors(self.covariance)
        local_logdet = 0.0
        for factor in local_factors:
            local_logdet += 2.0 * float(np.sum(np.log(np.diag(factor))))
        shared = _shared_factor_matrix(self.covariance)
        if shared.shape[1] == 0:
            if not np.isfinite(local_logdet):
                raise ValueError("structured covariance log determinant is not finite")
            return local_logdet

        local_shared = _solve_local_blocks(
            local_factors,
            shared,
            point_count=self.point_count,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            core = np.eye(shared.shape[1], dtype=np.float64) + shared.T @ local_shared
        core = 0.5 * (core + core.T)
        _finite_result(core, name="Woodbury determinant core")
        try:
            core_factor = np.linalg.cholesky(core)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "Woodbury determinant core must be positive definite"
            ) from error
        result = local_logdet + 2.0 * float(np.sum(np.log(np.diag(core_factor))))
        if not np.isfinite(result):
            raise ValueError("structured covariance log determinant is not finite")
        return result

    def sample(
        self,
        generator: np.random.Generator,
        sample_count: int,
    ) -> np.ndarray:
        """Draw zero-mean samples while preserving every shared component."""

        if not isinstance(generator, np.random.Generator):
            raise TypeError("generator must be a numpy.random.Generator")
        count = genuine_integer(
            sample_count,
            name="sample_count",
            minimum=1,
        )
        samples = np.zeros(
            (self.state_dimension, count),
            dtype=np.float64,
        )
        for index, block in enumerate(self.covariance.local_covariance_m2):
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(block)
            except np.linalg.LinAlgError as error:
                raise ValueError("local covariance square root failed") from error
            scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
            tolerance = 1e-12 + 1e-10 * scale
            if float(np.min(eigenvalues)) < -tolerance:
                raise ValueError("local covariance must be positive semidefinite")
            root = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))[None, :]
            noise = generator.standard_normal((3, count))
            samples[3 * index : 3 * index + 3] = root @ noise
        for factor in self.covariance.shared_factors_m.values():
            flat = factor.reshape((self.state_dimension, factor.shape[2]))
            noise = generator.standard_normal((factor.shape[2], count))
            samples += flat @ noise
        _finite_result(samples, name="structured covariance samples")
        return _immutable_float64(samples.T)

    def project_query_covariance(
        self,
        query_jacobian: np.ndarray,
    ) -> ProjectedStructuredPointCovarianceV1:
        """Delegate exact component-preserving query projection."""

        return self.covariance.project_query_covariance(query_jacobian)


__all__ = [
    "STRUCTURED_POINT_COVARIANCE_OPERATOR_CLAIM_BOUNDARY",
    "STRUCTURED_POINT_COVARIANCE_OPERATOR_SCHEMA",
    "STRUCTURED_POINT_COVARIANCE_OPERATOR_SEMANTICS",
    "STRUCTURED_POINT_COVARIANCE_OPERATOR_VERSION",
    "StructuredPointCovarianceOperatorV1",
]
