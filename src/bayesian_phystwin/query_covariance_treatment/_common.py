"""Shared validation and metadata keys for query covariance treatment."""

from __future__ import annotations

import hashlib
import math
from typing import Final

import numpy as np

from .._portable_contracts import sha256_digest

EVALUATION_BASELINE_BELIEF_ID_METADATA_KEY: Final = "covariance_only_baseline_belief_id"
EVALUATION_CANDIDATE_BELIEF_ID_METADATA_KEY: Final = (
    "covariance_only_candidate_belief_id"
)
EVALUATION_COMMON_DOMAIN_ID_METADATA_KEY: Final = "covariance_only_common_domain_id"
EVALUATION_HYBRID_RECORD_ID_METADATA_KEY: Final = "covariance_only_hybrid_record_id"
EVALUATION_CANDIDATE_COVARIANCE_ID_METADATA_KEY: Final = (
    "candidate_covariance_artifact_id"
)
EVALUATION_CALIBRATION_APPLICATION_ID_METADATA_KEY: Final = (
    "domain_covariance_calibration_application_id"
)
EVALUATION_HARM_RISK_CERTIFICATE_ID_METADATA_KEY: Final = (
    "guard_harm_risk_artifact_certificate_id"
)
EVALUATION_QUERY_RELEVANCE_CERTIFICATE_ID_METADATA_KEY: Final = (
    "query_covariance_relevance_certificate_id"
)
EVALUATION_QUERY_ID_METADATA_KEY: Final = "registered_query_id"
EVALUATION_CALIBRATION_PARTITION_ID_METADATA_KEY: Final = "calibration_partition_id"
EVALUATION_STATISTICAL_UNIT_METADATA_KEY: Final = "statistical_unit"
EVALUATION_MEAN_IDENTITY_VERIFIED_METADATA_KEY: Final = "exact_mean_identity_verified"
EVALUATION_SIMULTANEOUS_COVERAGE_METADATA_KEY: Final = "simultaneous_query_coverage"
EVALUATION_MEAN_FULL_WIDTH_RATIO_METADATA_KEY: Final = "mean_full_width_ratio"

COVARIANCE_MODES: Final = frozenset({"marginal", "explicit-joint"})


def canonical_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    if any(character in value for character in "\x00\r\n"):
        raise ValueError(f"{name} must be a single canonical line")
    return value


def finite_float(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite number")
    result = float(raw.item())
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def real_array(value: object, *, name: str, ndim: int) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain real numeric values") from error
    if raw.dtype.kind not in "iuf" or raw.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}-dimensional real numeric array")
    result = np.array(raw, dtype=np.float64, copy=True, order="C")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    return result


def symmetric_psd(value: object, *, name: str) -> np.ndarray:
    matrix = real_array(value, name=name, ndim=2)
    if matrix.shape[0] < 1 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if not np.allclose(matrix, matrix.T, atol=1e-10, rtol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    with np.errstate(over="ignore", invalid="ignore"):
        symmetric = 0.5 * matrix + 0.5 * matrix.T
    if not np.all(np.isfinite(symmetric)):
        raise ValueError(f"{name} symmetrization overflowed finite float64")
    normalization = max(float(np.max(np.abs(symmetric))), 1.0)
    try:
        eigenvalues = np.linalg.eigvalsh(symmetric / normalization)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} eigenvalues could not be evaluated") from error
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError(f"{name} eigenvalues must be finite")
    tolerance = 1e-12 + 1e-10 * max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    return symmetric


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype=np.dtype("<f8"))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0<f8\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def required_artifact_id(value: object, *, name: str) -> str:
    return sha256_digest(value, name=name)
