"""Contracts and numerical primitives for query-covariance cross-fitting."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._canonical_contracts import (
    canonical_string_tuple,
    genuine_integer,
    literal_lower_hex,
)

QUERY_COVARIANCE_CROSSFIT_SCHEMA = "bayesian_phystwin.query_covariance_crossfit"
QUERY_COVARIANCE_CROSSFIT_VERSION = 1
QUERY_COVARIANCE_CROSSFIT_SCORE = "group_balanced_gaussian_nll"
QUERY_COVARIANCE_CROSSFIT_CLAIM_BOUNDARY = (
    "Source-development covariance selection only. It preserves point means, "
    "does not use target outcomes, does not establish finite-sample coverage, "
    "and requires a separate frozen calibration and confirmation cohort."
)


def _finite_real(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real number")
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real number")
    result = float(raw.item())
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    if minimum is not None:
        invalid = result <= minimum if strict_minimum else result < minimum
        if invalid:
            relation = "greater than" if strict_minimum else "at least"
            raise ValueError(f"{name} must be {relation} {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _sha256(value: object, *, name: str) -> str:
    return literal_lower_hex(value, name=name, lengths={64})


def _immutable_array(value: object, *, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True, order="C")
    if array.dtype.hasobject:
        raise TypeError("query covariance arrays must not contain Python objects")
    payload = bytes(array.tobytes(order="C"))
    result = np.ndarray(
        shape=array.shape,
        dtype=array.dtype,
        buffer=payload,
        order="C",
    )
    if result.flags.writeable:
        raise AssertionError("bytes-backed query covariance array became writeable")
    return result


def _canonical_json_bytes(values: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(values),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _content_id(values: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(values)).hexdigest()


def _numeric_array(value: object, *, name: str) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain real numeric values") from error
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    result = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _matrix_tolerance(value: np.ndarray) -> float:
    scale = max(1.0, float(np.max(np.abs(value), initial=0.0)))
    return 64.0 * np.finfo(np.float64).eps * value.shape[-1] * scale


def _admit_psd(value: object, *, name: str) -> np.ndarray:
    matrix = _numeric_array(value, name=name)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a nonempty square matrix")
    tolerance = _matrix_tolerance(matrix)
    if float(np.max(np.abs(matrix - matrix.T), initial=0.0)) > tolerance:
        raise ValueError(f"{name} must be symmetric")
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if float(np.min(eigenvalues, initial=0.0)) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    clipped = np.maximum(eigenvalues, 0.0)
    result = (eigenvectors * clipped[None, :]) @ eigenvectors.T
    return 0.5 * (result + result.T)


def _cholesky(value: np.ndarray, *, name: str) -> np.ndarray:
    try:
        factor = np.linalg.cholesky(value)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error
    if not np.all(np.isfinite(factor)):
        raise ValueError(f"{name} produced a non-finite Cholesky factor")
    return factor


def _query_group_arrays(
    residual: object,
    covariance: object,
    *,
    name: str,
    dimension: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    residual_array = _numeric_array(residual, name=f"{name} residual")
    covariance_array = _numeric_array(covariance, name=f"{name} covariance")
    if residual_array.ndim == 1:
        residual_array = residual_array[None, :]
    if covariance_array.ndim == 2:
        covariance_array = covariance_array[None, :, :]
    if residual_array.ndim != 2 or residual_array.shape[0] == 0:
        raise ValueError(f"{name} residual must have shape (m>=1, d)")
    if residual_array.shape[1] == 0:
        raise ValueError(f"{name} query dimension must be positive")
    if dimension is not None and residual_array.shape[1] != dimension:
        raise ValueError(f"{name} query dimension changed")
    expected = (
        residual_array.shape[0],
        residual_array.shape[1],
        residual_array.shape[1],
    )
    if covariance_array.shape != expected:
        raise ValueError(f"{name} covariance must have shape {expected}")
    admitted = np.stack(
        [
            _admit_psd(matrix, name=f"{name} covariance {index}")
            for index, matrix in enumerate(covariance_array)
        ]
    )
    return residual_array, admitted


def _canonical_groups(
    group_ids: Sequence[str],
    residual_groups: Sequence[object],
    covariance_groups: Sequence[object],
) -> tuple[tuple[str, ...], tuple[np.ndarray, ...], tuple[np.ndarray, ...], int]:
    ids = canonical_string_tuple(
        group_ids,
        name="development_group_ids",
        allow_empty=False,
    )
    if len(ids) < 2:
        raise ValueError("at least two independent development groups are required")
    if len(set(ids)) != len(ids):
        raise ValueError("development_group_ids must be unique")
    if len(residual_groups) != len(ids):
        raise ValueError("residual_groups must contain one entry per group")
    if len(covariance_groups) != len(ids):
        raise ValueError("covariance_groups must contain one entry per group")

    dimension: int | None = None
    rows: list[tuple[str, np.ndarray, np.ndarray]] = []
    for group_id, residual, covariance in zip(
        ids,
        residual_groups,
        covariance_groups,
        strict=True,
    ):
        residual_array, covariance_array = _query_group_arrays(
            residual,
            covariance,
            name=f"development group {group_id}",
            dimension=dimension,
        )
        if dimension is None:
            dimension = residual_array.shape[1]
        rows.append((group_id, residual_array, covariance_array))
    rows.sort(key=lambda row: row[0])
    if dimension is None:  # pragma: no cover - guarded by nonempty groups.
        raise AssertionError("query dimension was not initialized")
    return (
        tuple(row[0] for row in rows),
        tuple(row[1] for row in rows),
        tuple(row[2] for row in rows),
        dimension,
    )


@dataclass(frozen=True, slots=True)
class StructuredQueryCovarianceCandidateV1:
    """Low-dimensional PSD covariance-transform hyperparameters."""

    covariance_scale: float = 1.0
    diagonal_shrinkage: float = 0.0
    isotropic_variance: float = 0.0
    low_rank_rank: int = 0
    low_rank_fraction: float = 0.0

    def __post_init__(self) -> None:
        scale = _finite_real(
            self.covariance_scale,
            name="covariance_scale",
            minimum=0.0,
            strict_minimum=True,
        )
        shrinkage = _finite_real(
            self.diagonal_shrinkage,
            name="diagonal_shrinkage",
            minimum=0.0,
            maximum=1.0,
        )
        nugget = _finite_real(
            self.isotropic_variance,
            name="isotropic_variance",
            minimum=0.0,
        )
        rank = genuine_integer(
            self.low_rank_rank,
            name="low_rank_rank",
            minimum=0,
        )
        fraction = _finite_real(
            self.low_rank_fraction,
            name="low_rank_fraction",
            minimum=0.0,
            maximum=1.0,
        )
        if rank == 0 and fraction != 0.0:
            raise ValueError(
                "low_rank_fraction must be zero when low_rank_rank is zero"
            )
        if rank > 0 and fraction == 0.0:
            raise ValueError(
                "low_rank_fraction must be positive when low_rank_rank is positive"
            )
        object.__setattr__(self, "covariance_scale", scale)
        object.__setattr__(self, "diagonal_shrinkage", shrinkage)
        object.__setattr__(self, "isotropic_variance", nugget)
        object.__setattr__(self, "low_rank_rank", rank)
        object.__setattr__(self, "low_rank_fraction", fraction)

    def descriptor(self) -> dict[str, Any]:
        return {
            "covariance_scale": self.covariance_scale,
            "diagonal_shrinkage": self.diagonal_shrinkage,
            "isotropic_variance": self.isotropic_variance,
            "low_rank_rank": self.low_rank_rank,
            "low_rank_fraction": self.low_rank_fraction,
        }

    @property
    def candidate_id(self) -> str:
        return _content_id(self.descriptor())


@dataclass(frozen=True, slots=True)
class StructuredQueryCovarianceTransformV1:
    """A source-fitted PSD transform that leaves all point means unchanged."""

    candidate: StructuredQueryCovarianceCandidateV1
    dimension: int
    low_rank_covariance: np.ndarray
    source_group_ids: tuple[str, ...]
    source_evidence_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, StructuredQueryCovarianceCandidateV1):
            raise TypeError("candidate must be StructuredQueryCovarianceCandidateV1")
        dimension = genuine_integer(self.dimension, name="dimension", minimum=1)
        if self.candidate.low_rank_rank > dimension:
            raise ValueError("low_rank_rank cannot exceed the query dimension")
        covariance = _admit_psd(
            self.low_rank_covariance,
            name="low_rank_covariance",
        )
        if covariance.shape != (dimension, dimension):
            raise ValueError("low_rank_covariance shape must match dimension")
        tolerance = _matrix_tolerance(covariance)
        numerical_rank = int(
            np.sum(np.linalg.eigvalsh(covariance) > tolerance)
        )
        if numerical_rank > self.candidate.low_rank_rank:
            raise ValueError("low_rank_covariance exceeds the candidate rank")
        if self.candidate.low_rank_rank == 0 and np.any(covariance != 0.0):
            raise ValueError("rank-zero candidates require zero low_rank_covariance")
        group_ids = canonical_string_tuple(
            self.source_group_ids,
            name="source_group_ids",
            allow_empty=False,
        )
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("source_group_ids must be unique")
        canonical_groups = tuple(sorted(group_ids))
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(
            self,
            "low_rank_covariance",
            _immutable_array(covariance, dtype=np.dtype(np.float64)),
        )
        object.__setattr__(self, "source_group_ids", canonical_groups)
        object.__setattr__(
            self,
            "source_evidence_id",
            _sha256(self.source_evidence_id, name="source_evidence_id"),
        )

    @property
    def numerical_low_rank(self) -> int:
        tolerance = _matrix_tolerance(self.low_rank_covariance)
        return int(np.sum(np.linalg.eigvalsh(self.low_rank_covariance) > tolerance))

    def descriptor(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.descriptor(),
            "candidate_id": self.candidate.candidate_id,
            "dimension": self.dimension,
            "low_rank_covariance": self.low_rank_covariance.tolist(),
            "numerical_low_rank": self.numerical_low_rank,
            "source_group_ids": list(self.source_group_ids),
            "source_evidence_id": self.source_evidence_id,
        }

    @property
    def transform_id(self) -> str:
        return _content_id(self.descriptor())


def _base_covariance(
    covariance: np.ndarray,
    candidate: StructuredQueryCovarianceCandidateV1,
) -> np.ndarray:
    diagonal = np.diag(np.diag(covariance))
    result = candidate.covariance_scale * (
        (1.0 - candidate.diagonal_shrinkage) * covariance
        + candidate.diagonal_shrinkage * diagonal
    )
    result = result + candidate.isotropic_variance * np.eye(
        covariance.shape[0],
        dtype=np.float64,
    )
    return _admit_psd(result, name="structured base covariance")


def _fit_transform(
    group_ids: tuple[str, ...],
    residual_groups: tuple[np.ndarray, ...],
    covariance_groups: tuple[np.ndarray, ...],
    candidate: StructuredQueryCovarianceCandidateV1,
    *,
    source_evidence_id: str,
) -> StructuredQueryCovarianceTransformV1:
    dimension = residual_groups[0].shape[1]
    if candidate.low_rank_rank > dimension:
        raise ValueError("low_rank_rank cannot exceed the query dimension")
    low_rank = np.zeros((dimension, dimension), dtype=np.float64)
    if candidate.low_rank_rank > 0:
        group_excess: list[np.ndarray] = []
        for residual, covariance in zip(
            residual_groups,
            covariance_groups,
            strict=True,
        ):
            residual_second_moment = np.einsum(
                "ni,nj->ij",
                residual,
                residual,
            ) / len(residual)
            average_base = np.mean(
                np.stack(
                    [_base_covariance(matrix, candidate) for matrix in covariance]
                ),
                axis=0,
            )
            group_excess.append(residual_second_moment - average_base)
        excess = np.mean(np.stack(group_excess), axis=0)
        excess = 0.5 * (excess + excess.T)
        eigenvalues, eigenvectors = np.linalg.eigh(excess)
        order = np.argsort(eigenvalues)[::-1]
        tolerance = _matrix_tolerance(excess)
        retained = [
            int(index)
            for index in order
            if eigenvalues[index] > tolerance
        ][: candidate.low_rank_rank]
        if retained:
            basis = eigenvectors[:, retained]
            variance = candidate.low_rank_fraction * eigenvalues[retained]
            low_rank = (basis * variance[None, :]) @ basis.T
            low_rank = _admit_psd(low_rank, name="fitted low-rank covariance")
    return StructuredQueryCovarianceTransformV1(
        candidate=candidate,
        dimension=dimension,
        low_rank_covariance=low_rank,
        source_group_ids=group_ids,
        source_evidence_id=source_evidence_id,
    )


def apply_structured_query_covariance(
    covariance: object,
    transform: StructuredQueryCovarianceTransformV1,
) -> np.ndarray:
    """Apply a frozen PSD transform without changing any point prediction."""

    if not isinstance(transform, StructuredQueryCovarianceTransformV1):
        raise TypeError("transform must be StructuredQueryCovarianceTransformV1")
    raw = _numeric_array(covariance, name="covariance")
    if raw.ndim < 2 or raw.shape[-2:] != (transform.dimension, transform.dimension):
        raise ValueError("covariance shape does not match the transform dimension")
    flattened = raw.reshape((-1, transform.dimension, transform.dimension))
    result = np.empty_like(flattened)
    for index, matrix in enumerate(flattened):
        admitted = _admit_psd(matrix, name=f"covariance {index}")
        transformed = _base_covariance(admitted, transform.candidate)
        transformed = transformed + transform.low_rank_covariance
        result[index] = _admit_psd(
            transformed,
            name=f"transformed covariance {index}",
        )
    return _immutable_array(
        result.reshape(raw.shape),
        dtype=np.dtype(np.float64),
    )


__all__ = [
    "QUERY_COVARIANCE_CROSSFIT_CLAIM_BOUNDARY",
    "QUERY_COVARIANCE_CROSSFIT_SCHEMA",
    "QUERY_COVARIANCE_CROSSFIT_SCORE",
    "QUERY_COVARIANCE_CROSSFIT_VERSION",
    "StructuredQueryCovarianceCandidateV1",
    "StructuredQueryCovarianceTransformV1",
    "apply_structured_query_covariance",
]
