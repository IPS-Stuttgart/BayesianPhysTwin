"""Shared validation for posterior covariance portfolios."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    immutable_array,
    literal_lower_hex,
    plain_json,
)
from ._portable_contracts import content_id
from .posterior_covariance_semantics import (
    POSTERIOR_COVARIANCE_METHODS,
    PosteriorCovarianceMethod,
    PosteriorCovarianceSemanticsV1,
)

POSTERIOR_COVARIANCE_SOURCE_SCHEMA = "bayesian_phystwin.posterior_covariance_source"
POSTERIOR_COVARIANCE_SOURCE_VERSION = 1
POSTERIOR_QUERY_COVARIANCE_PORTFOLIO_SCHEMA = (
    "bayesian_phystwin.posterior_query_covariance_portfolio"
)
POSTERIOR_QUERY_COVARIANCE_PORTFOLIO_VERSION = 1

ACCEPTED_METHODS: tuple[PosteriorCovarianceMethod, ...] = (
    "irls_working",
    "laplace_observed_information",
    "group_sandwich",
)
METHOD_ORDER = {
    method: index for index, method in enumerate(POSTERIOR_COVARIANCE_METHODS)
}

FloatArray: TypeAlias = NDArray[np.float64]


def sha256_id(value: object, *, name: str) -> str:
    return literal_lower_hex(value, name=name, lengths={64})


def canonical_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def covariance_method(
    value: object,
    *,
    name: str,
) -> PosteriorCovarianceMethod:
    if type(value) is not str or value not in POSTERIOR_COVARIANCE_METHODS:
        allowed = list(POSTERIOR_COVARIANCE_METHODS)
        raise ValueError(f"{name} must be one of {allowed}")
    return cast(PosteriorCovarianceMethod, value)


def validated_covariance(value: object, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    matrix = np.asarray(raw, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if not len(matrix) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be nonempty and finite")
    if not np.allclose(matrix, matrix.T, atol=1e-11, rtol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    scale = max(1.0, float(np.linalg.norm(symmetric, ord=2)))
    tolerance = 16.0 * max(1, len(symmetric)) * np.finfo(np.float64).eps * scale
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    return immutable_array(symmetric, dtype=np.dtype("<f8"))


def validated_query_matrix(value: object) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError("query_matrix must contain real numeric values")
    matrix = np.asarray(raw, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError("query_matrix must be a nonempty matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("query_matrix must be finite")
    return immutable_array(matrix, dtype=np.dtype("<f8"))


def array_record(value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(value, dtype=np.dtype("<f8"))
    return {
        "dtype": "<f8",
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def query_matrix_id(query: FloatArray) -> str:
    return content_id(
        {
            "schema": "bayesian_phystwin.posterior_covariance_query_matrix",
            "schema_version": 1,
            "query_matrix": array_record(query),
        }
    )


def portfolio_metadata(
    value: Mapping[str, Any] | None,
    *,
    name: str,
    required: Mapping[str, object] | None = None,
) -> Mapping[str, Any]:
    details = {} if value is None else dict(value)
    for key, expected in (required or {}).items():
        actual = details.get(key)
        if actual is not None and actual != expected:
            raise ValueError(f"metadata contradicts {key}")
        details[key] = expected
    return frozen_finite_json_mapping(details, name=name)


def projected_semantics(
    source: PosteriorCovarianceSemanticsV1,
    *,
    dimension: int,
    source_id: str,
    query_id: str,
) -> PosteriorCovarianceSemanticsV1:
    details = dict(plain_json(source.metadata))
    required = {
        "source_covariance_semantics_id": source.artifact_id,
        "source_covariance_source_id": source_id,
        "query_matrix_sha256": query_id,
    }
    for key, expected in required.items():
        actual = details.get(key)
        if actual is not None and actual != expected:
            raise ValueError(f"source covariance metadata contradicts {key}")
        details[key] = expected
    return PosteriorCovarianceSemanticsV1(
        method=source.method,
        dimension=dimension,
        likelihood_power_semantics=source.likelihood_power_semantics,
        prior_included=source.prior_included,
        generalized_bayes=source.generalized_bayes,
        mixture_curvature_exact=source.mixture_curvature_exact,
        group_score_correction=source.group_score_correction,
        calibrated=False,
        metadata=details,
    )


__all__ = [
    "ACCEPTED_METHODS",
    "METHOD_ORDER",
    "POSTERIOR_COVARIANCE_SOURCE_SCHEMA",
    "POSTERIOR_COVARIANCE_SOURCE_VERSION",
    "POSTERIOR_QUERY_COVARIANCE_PORTFOLIO_SCHEMA",
    "POSTERIOR_QUERY_COVARIANCE_PORTFOLIO_VERSION",
    "FloatArray",
    "array_record",
    "canonical_string",
    "covariance_method",
    "portfolio_metadata",
    "projected_semantics",
    "query_matrix_id",
    "sha256_id",
    "validated_covariance",
    "validated_query_matrix",
]
