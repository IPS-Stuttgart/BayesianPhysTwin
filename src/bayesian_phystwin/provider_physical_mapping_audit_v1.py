"""Fail-closed provider-to-physical-query mapping diagnostics.

The audit verifies only that one frozen provider artifact can be mapped into one
frozen physical query under an explicit frame, unit, time, support, and optional
covariance policy. It does not establish provider competence, covariance
calibration, estimator benefit, target authorization, or deployment safety.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Final, cast

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    immutable_array,
    plain_json,
)
from ._portable_contracts import content_id, sha256_digest

PROVIDER_PHYSICAL_MAPPING_AUDIT_SCHEMA: Final = (
    "bayesian_phystwin.provider_physical_mapping_audit"
)
PROVIDER_PHYSICAL_MAPPING_AUDIT_VERSION: Final = 1
PROVIDER_PHYSICAL_MAPPING_AUDIT_INFORMATION_BOUNDARY: Final = (
    "Target-blind geometric and contract-support evidence only. An admissible "
    "mapping does not establish provider competence, association quality, "
    "covariance calibration, physical-estimation benefit, Causal4D benefit, "
    "target access, deployment safety, or state of the art."
)

_ADMISSIBLE_REASON: Final = "provider-physical-mapping-admissible"
_REASON_ORDER: Final[tuple[str, ...]] = (
    "invalid-provider-to-physical-transform",
    "nonfinite-effective-physical-query-bounds",
    "nonfinite-declared-valid-provider-points",
    "nonfinite-transformed-provider-points",
    "required-provider-timestamps-missing",
    "nonfinite-declared-valid-provider-timestamps",
    "required-provider-covariance-missing",
    "invalid-declared-valid-provider-covariance",
    "insufficient-provider-valid-support",
    "insufficient-physical-query-overlap",
)
_TECHNICAL_REASONS: Final[frozenset[str]] = frozenset(_REASON_ORDER[:8])
_PROVIDER_SUPPORT_REASON: Final = "insufficient-provider-valid-support"
_QUERY_SUPPORT_REASON: Final = "insufficient-physical-query-overlap"
_MINIMUM_SAFE_UNIT_SCALE_M: Final = math.sqrt(float(np.finfo(float).tiny))
_MAXIMUM_SAFE_UNIT_SCALE_M: Final = math.sqrt(float(np.finfo(float).max))


def _canonical_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def _finite_real(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _optional_finite_real(
    value: object | None,
    *,
    name: str,
    minimum: float | None = None,
) -> float | None:
    if value is None:
        return None
    return _finite_real(value, name=name, minimum=minimum)


def _float_array(values: object, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    array = immutable_array(values, dtype=np.dtype("<f8"))
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return array


def _point_array(values: object, *, name: str) -> np.ndarray:
    array = immutable_array(values, dtype=np.dtype("<f8"))
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] < 1:
        raise ValueError(f"{name} must have shape (N, 3) with N >= 1")
    return array


def _strict_boolean_vector(values: object, *, name: str, length: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype != np.dtype(np.bool_):
        raise ValueError(f"{name} must have boolean dtype")
    array = immutable_array(raw, dtype=np.dtype(np.bool_))
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},)")
    return array


def _array_descriptor(array: np.ndarray) -> dict[str, object]:
    canonical = np.ascontiguousarray(array)
    return {
        "shape": list(canonical.shape),
        "dtype": canonical.dtype.str,
        "sha256": hashlib.sha256(canonical.tobytes(order="C")).hexdigest(),
    }


def _optional_bbox(
    points: np.ndarray,
    selected: np.ndarray,
) -> dict[str, object] | None:
    if not np.any(selected):
        return None
    retained = points[selected]
    return {
        "lower": [float(value) for value in np.min(retained, axis=0)],
        "upper": [float(value) for value in np.max(retained, axis=0)],
    }


def _safe_fraction(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


@dataclass(frozen=True, slots=True)
class ProviderPhysicalMappingPolicyV1:
    """Frozen thresholds for one provider-to-query mapping decision."""

    minimum_valid_point_count: int = 1
    minimum_valid_fraction: float = 0.0
    minimum_mapped_point_count: int = 1
    minimum_mapped_fraction: float = 0.5
    boundary_tolerance_m: float = 0.0
    require_timestamps: bool = False
    require_covariance: bool = False
    maximum_rotation_orthogonality_error: float = 1e-6
    maximum_rotation_determinant_error: float = 1e-6
    maximum_homogeneous_row_error: float = 1e-12
    maximum_covariance_symmetry_error_m2: float = 1e-9
    minimum_covariance_eigenvalue_m2: float = -1e-12
    maximum_covariance_condition_number: float | None = None

    def __post_init__(self) -> None:
        for name in ("minimum_valid_point_count", "minimum_mapped_point_count"):
            object.__setattr__(
                self,
                name,
                genuine_integer(getattr(self, name), name=name, minimum=1),
            )
        for name in ("minimum_valid_fraction", "minimum_mapped_fraction"):
            object.__setattr__(
                self,
                name,
                _finite_real(
                    getattr(self, name),
                    name=name,
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
        for name in (
            "boundary_tolerance_m",
            "maximum_rotation_orthogonality_error",
            "maximum_rotation_determinant_error",
            "maximum_homogeneous_row_error",
            "maximum_covariance_symmetry_error_m2",
        ):
            object.__setattr__(
                self,
                name,
                _finite_real(getattr(self, name), name=name, minimum=0.0),
            )
        object.__setattr__(
            self,
            "minimum_covariance_eigenvalue_m2",
            _finite_real(
                self.minimum_covariance_eigenvalue_m2,
                name="minimum_covariance_eigenvalue_m2",
            ),
        )
        object.__setattr__(
            self,
            "maximum_covariance_condition_number",
            _optional_finite_real(
                self.maximum_covariance_condition_number,
                name="maximum_covariance_condition_number",
                minimum=1.0,
            ),
        )
        for name in ("require_timestamps", "require_covariance"):
            object.__setattr__(
                self,
                name,
                genuine_boolean(getattr(self, name), name=name),
            )

    @property
    def policy_id(self) -> str:
        return content_id(self.descriptor())

    def descriptor(self) -> dict[str, object]:
        return {
            "minimum_valid_point_count": self.minimum_valid_point_count,
            "minimum_valid_fraction": self.minimum_valid_fraction,
            "minimum_mapped_point_count": self.minimum_mapped_point_count,
            "minimum_mapped_fraction": self.minimum_mapped_fraction,
            "boundary_tolerance_m": self.boundary_tolerance_m,
            "require_timestamps": self.require_timestamps,
            "require_covariance": self.require_covariance,
            "maximum_rotation_orthogonality_error": (
                self.maximum_rotation_orthogonality_error
            ),
            "maximum_rotation_determinant_error": (
                self.maximum_rotation_determinant_error
            ),
            "maximum_homogeneous_row_error": self.maximum_homogeneous_row_error,
            "maximum_covariance_symmetry_error_m2": (
                self.maximum_covariance_symmetry_error_m2
            ),
            "minimum_covariance_eigenvalue_m2": (self.minimum_covariance_eigenvalue_m2),
            "maximum_covariance_condition_number": (
                self.maximum_covariance_condition_number
            ),
        }


@dataclass(frozen=True, slots=True)
class ProviderPhysicalMappingCaseV1:
    """Immutable provider geometry and one frozen physical-query support region."""

    case_id: str
    provider_artifact_id: str
    physical_query_id: str
    mapping_protocol_id: str
    provider_frame: str
    physical_frame: str
    points_native: np.ndarray
    valid_mask: np.ndarray
    provider_unit_scale_m: float
    provider_to_physical: np.ndarray
    query_bounds_m: np.ndarray
    timestamps_s: np.ndarray | None = None
    query_time_window_s: np.ndarray | None = None
    covariances_native: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "case_id",
            _canonical_text(self.case_id, name="case_id"),
        )
        for name in (
            "provider_artifact_id",
            "physical_query_id",
            "mapping_protocol_id",
        ):
            object.__setattr__(
                self,
                name,
                sha256_digest(getattr(self, name), name=name),
            )
        for name in ("provider_frame", "physical_frame"):
            object.__setattr__(
                self,
                name,
                _canonical_text(getattr(self, name), name=name),
            )

        points = _point_array(self.points_native, name="points_native")
        object.__setattr__(self, "points_native", points)
        object.__setattr__(
            self,
            "valid_mask",
            _strict_boolean_vector(
                self.valid_mask,
                name="valid_mask",
                length=points.shape[0],
            ),
        )
        object.__setattr__(
            self,
            "provider_unit_scale_m",
            _finite_real(
                self.provider_unit_scale_m,
                name="provider_unit_scale_m",
                minimum=_MINIMUM_SAFE_UNIT_SCALE_M,
                maximum=_MAXIMUM_SAFE_UNIT_SCALE_M,
            ),
        )

        transform = _float_array(
            self.provider_to_physical,
            name="provider_to_physical",
            shape=(4, 4),
        )
        if not np.all(np.isfinite(transform)):
            raise ValueError("provider_to_physical must contain finite values")
        object.__setattr__(self, "provider_to_physical", transform)

        bounds = _float_array(
            self.query_bounds_m,
            name="query_bounds_m",
            shape=(2, 3),
        )
        if not np.all(np.isfinite(bounds)):
            raise ValueError("query_bounds_m must contain finite values")
        if np.any(bounds[1] < bounds[0]):
            raise ValueError(
                "query_bounds_m upper bounds must not be below lower bounds"
            )
        object.__setattr__(self, "query_bounds_m", bounds)

        timestamps = self.timestamps_s
        time_window = self.query_time_window_s
        if (timestamps is None) != (time_window is None):
            raise ValueError(
                "timestamps_s and query_time_window_s must be supplied together"
            )
        if timestamps is not None:
            timestamp_array = immutable_array(timestamps, dtype=np.dtype("<f8"))
            if timestamp_array.shape != (points.shape[0],):
                raise ValueError(f"timestamps_s must have shape ({points.shape[0]},)")
            window_array = _float_array(
                cast(np.ndarray, time_window),
                name="query_time_window_s",
                shape=(2,),
            )
            if not np.all(np.isfinite(window_array)):
                raise ValueError("query_time_window_s must contain finite values")
            if window_array[1] < window_array[0]:
                raise ValueError("query_time_window_s end must not be before its start")
            object.__setattr__(self, "timestamps_s", timestamp_array)
            object.__setattr__(self, "query_time_window_s", window_array)

        covariance = self.covariances_native
        if covariance is not None:
            covariance_array = immutable_array(covariance, dtype=np.dtype("<f8"))
            expected_shape = (points.shape[0], 3, 3)
            if covariance_array.shape != expected_shape:
                raise ValueError(f"covariances_native must have shape {expected_shape}")
            object.__setattr__(self, "covariances_native", covariance_array)

        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="provider physical mapping case metadata",
            ),
        )
        expected = content_id(self.descriptor())
        if self.artifact_id is not None:
            supplied = sha256_digest(self.artifact_id, name="artifact_id")
            if supplied != expected:
                raise ValueError("artifact_id does not match mapping case contents")
        object.__setattr__(self, "artifact_id", expected)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": "bayesian_phystwin.provider_physical_mapping_case",
            "schema_version": 1,
            "case_id": self.case_id,
            "provider_artifact_id": self.provider_artifact_id,
            "physical_query_id": self.physical_query_id,
            "mapping_protocol_id": self.mapping_protocol_id,
            "provider_frame": self.provider_frame,
            "physical_frame": self.physical_frame,
            "points_native": _array_descriptor(self.points_native),
            "valid_mask": _array_descriptor(self.valid_mask),
            "provider_unit_scale_m": self.provider_unit_scale_m,
            "provider_to_physical": _array_descriptor(self.provider_to_physical),
            "query_bounds_m": _array_descriptor(self.query_bounds_m),
            "timestamps_s": (
                None
                if self.timestamps_s is None
                else _array_descriptor(self.timestamps_s)
            ),
            "query_time_window_s": (
                None
                if self.query_time_window_s is None
                else _array_descriptor(self.query_time_window_s)
            ),
            "covariances_native": (
                None
                if self.covariances_native is None
                else _array_descriptor(self.covariances_native)
            ),
            "metadata": plain_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ProviderPhysicalMappingAuditV1:
    """Content-addressed mapping decision and diagnostic accounting."""

    case_id: str
    case_artifact_id: str
    provider_artifact_id: str
    physical_query_id: str
    mapping_protocol_id: str
    provider_frame: str
    physical_frame: str
    policy_id: str
    mapping_admissible: bool
    technical_valid: bool
    provider_support_complete: bool
    query_support_sufficient: bool
    result_reason: str
    rejection_reasons: Sequence[str]
    diagnostics: Mapping[str, Any]
    audit_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "case_id",
            _canonical_text(self.case_id, name="case_id"),
        )
        for name in (
            "case_artifact_id",
            "provider_artifact_id",
            "physical_query_id",
            "mapping_protocol_id",
            "policy_id",
        ):
            object.__setattr__(
                self,
                name,
                sha256_digest(getattr(self, name), name=name),
            )
        for name in ("provider_frame", "physical_frame"):
            object.__setattr__(
                self,
                name,
                _canonical_text(getattr(self, name), name=name),
            )
        for name in (
            "mapping_admissible",
            "technical_valid",
            "provider_support_complete",
            "query_support_sufficient",
        ):
            object.__setattr__(
                self,
                name,
                genuine_boolean(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "result_reason",
            _canonical_text(self.result_reason, name="result_reason"),
        )
        if isinstance(self.rejection_reasons, (str, bytes)):
            raise ValueError("rejection_reasons must be a sequence")
        reasons = tuple(
            _canonical_text(value, name=f"rejection_reasons[{index}]")
            for index, value in enumerate(self.rejection_reasons)
        )
        if len(reasons) != len(set(reasons)):
            raise ValueError("rejection_reasons must not contain duplicates")
        if any(reason not in _REASON_ORDER for reason in reasons):
            raise ValueError("rejection_reasons contain an unsupported reason")
        ordered = tuple(reason for reason in _REASON_ORDER if reason in reasons)
        if reasons != ordered:
            raise ValueError("rejection_reasons must use canonical precedence order")
        expected_technical_valid = not any(
            reason in _TECHNICAL_REASONS for reason in reasons
        )
        expected_provider_support = _PROVIDER_SUPPORT_REASON not in reasons
        expected_query_support = _QUERY_SUPPORT_REASON not in reasons
        if self.technical_valid != expected_technical_valid:
            raise ValueError("technical_valid does not match rejection reasons")
        if self.provider_support_complete != expected_provider_support:
            raise ValueError(
                "provider_support_complete does not match rejection reasons"
            )
        if self.query_support_sufficient != expected_query_support:
            raise ValueError(
                "query_support_sufficient does not match rejection reasons"
            )
        expected_reason = _ADMISSIBLE_REASON if not reasons else reasons[0]
        if self.result_reason != expected_reason:
            raise ValueError("result_reason does not match rejection reasons")
        if self.mapping_admissible != (not reasons):
            raise ValueError("mapping_admissible does not match rejection reasons")
        object.__setattr__(self, "rejection_reasons", reasons)
        object.__setattr__(
            self,
            "diagnostics",
            frozen_finite_json_mapping(
                self.diagnostics,
                name="provider physical mapping diagnostics",
            ),
        )
        expected = content_id(self.descriptor())
        if self.audit_id is not None:
            supplied = sha256_digest(self.audit_id, name="audit_id")
            if supplied != expected:
                raise ValueError("audit_id does not match mapping audit contents")
        object.__setattr__(self, "audit_id", expected)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": PROVIDER_PHYSICAL_MAPPING_AUDIT_SCHEMA,
            "schema_version": PROVIDER_PHYSICAL_MAPPING_AUDIT_VERSION,
            "case_id": self.case_id,
            "case_artifact_id": self.case_artifact_id,
            "provider_artifact_id": self.provider_artifact_id,
            "physical_query_id": self.physical_query_id,
            "mapping_protocol_id": self.mapping_protocol_id,
            "provider_frame": self.provider_frame,
            "physical_frame": self.physical_frame,
            "policy_id": self.policy_id,
            "mapping_admissible": self.mapping_admissible,
            "technical_valid": self.technical_valid,
            "provider_support_complete": self.provider_support_complete,
            "query_support_sufficient": self.query_support_sufficient,
            "result_reason": self.result_reason,
            "rejection_reasons": list(self.rejection_reasons),
            "diagnostics": plain_json(self.diagnostics),
            "information_boundary": (
                PROVIDER_PHYSICAL_MAPPING_AUDIT_INFORMATION_BOUNDARY
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.descriptor(), "audit_id": self.audit_id}

    def provider_failure_signal_patch(self) -> dict[str, bool]:
        """Return only existing source-diagnostic signals owned by this audit."""

        return {
            "technical_valid": self.technical_valid,
            "provider_support_complete": (
                self.provider_support_complete and self.query_support_sufficient
            ),
        }


def _transform_diagnostics(
    transform: np.ndarray,
    policy: ProviderPhysicalMappingPolicyV1,
) -> tuple[dict[str, object], bool]:
    rotation = transform[:3, :3]
    with np.errstate(over="ignore", invalid="ignore"):
        determinant = float(np.linalg.det(rotation))
        orthogonality_error = float(
            np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro")
        )
        determinant_error = abs(determinant - 1.0)
        homogeneous_row_error = float(
            np.max(np.abs(transform[3] - np.array([0.0, 0.0, 0.0, 1.0], dtype=float)))
        )
    finite = all(
        math.isfinite(value)
        for value in (
            determinant,
            orthogonality_error,
            determinant_error,
            homogeneous_row_error,
        )
    )
    passed = (
        finite
        and orthogonality_error <= policy.maximum_rotation_orthogonality_error
        and determinant_error <= policy.maximum_rotation_determinant_error
        and homogeneous_row_error <= policy.maximum_homogeneous_row_error
    )
    return (
        {
            "rotation_determinant": _finite_or_none(determinant),
            "rotation_orthogonality_error": _finite_or_none(orthogonality_error),
            "rotation_determinant_error": _finite_or_none(determinant_error),
            "homogeneous_row_error": _finite_or_none(homogeneous_row_error),
            "finite": finite,
            "passed": passed,
        },
        passed,
    )


def _covariance_diagnostics(
    case: ProviderPhysicalMappingCaseV1,
    policy: ProviderPhysicalMappingPolicyV1,
) -> tuple[dict[str, object], bool]:
    covariance = case.covariances_native
    declared_valid_count = int(np.sum(case.valid_mask))
    scale2 = case.provider_unit_scale_m**2
    if covariance is None:
        return (
            {
                "available": False,
                "required": policy.require_covariance,
                "declared_valid_count": declared_valid_count,
                "provider_unit_scale_squared_m2": scale2,
            },
            not policy.require_covariance,
        )

    rotation = case.provider_to_physical[:3, :3]
    valid_indices = np.flatnonzero(case.valid_mask)
    counts = {
        "nonfinite_native_count": 0,
        "nonfinite_transformed_count": 0,
        "eigendecomposition_failure_count": 0,
        "symmetry_failure_count": 0,
        "eigenvalue_failure_count": 0,
        "condition_failure_count": 0,
    }
    invalid_indices: set[int] = set()
    min_eigenvalues: list[float] = []
    max_eigenvalues: list[float] = []
    finite_conditions: list[float] = []
    symmetry_errors: list[float] = []

    for raw_index in valid_indices:
        index = int(raw_index)
        native = covariance[index]
        if not np.all(np.isfinite(native)):
            counts["nonfinite_native_count"] += 1
            invalid_indices.add(index)
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            physical = scale2 * (rotation @ native @ rotation.T)
        if not np.all(np.isfinite(physical)):
            counts["nonfinite_transformed_count"] += 1
            invalid_indices.add(index)
            continue

        with np.errstate(over="ignore", invalid="ignore"):
            symmetry_error = float(np.linalg.norm(physical - physical.T, ord="fro"))
        if not math.isfinite(symmetry_error):
            counts["nonfinite_transformed_count"] += 1
            invalid_indices.add(index)
            continue
        symmetry_errors.append(symmetry_error)
        if symmetry_error > policy.maximum_covariance_symmetry_error_m2:
            counts["symmetry_failure_count"] += 1
            invalid_indices.add(index)

        with np.errstate(over="ignore", invalid="ignore"):
            symmetric = 0.5 * (physical + physical.T)
        try:
            eigenvalues = np.linalg.eigvalsh(symmetric)
        except np.linalg.LinAlgError:
            counts["eigendecomposition_failure_count"] += 1
            invalid_indices.add(index)
            continue
        if not np.all(np.isfinite(eigenvalues)):
            counts["eigendecomposition_failure_count"] += 1
            invalid_indices.add(index)
            continue
        minimum = float(eigenvalues[0])
        maximum = float(eigenvalues[-1])
        min_eigenvalues.append(minimum)
        max_eigenvalues.append(maximum)
        if minimum < policy.minimum_covariance_eigenvalue_m2:
            counts["eigenvalue_failure_count"] += 1
            invalid_indices.add(index)

        maximum_condition = policy.maximum_covariance_condition_number
        if maximum_condition is not None:
            if minimum <= 0.0:
                counts["condition_failure_count"] += 1
                invalid_indices.add(index)
            else:
                condition = maximum / minimum
                if not math.isfinite(condition) or condition > maximum_condition:
                    counts["condition_failure_count"] += 1
                    invalid_indices.add(index)
                if math.isfinite(condition):
                    finite_conditions.append(float(condition))

    diagnostics: dict[str, object] = {
        "available": True,
        "required": policy.require_covariance,
        "declared_valid_count": len(valid_indices),
        "provider_unit_scale_squared_m2": scale2,
        **counts,
        "invalid_declared_valid_count": len(invalid_indices),
        "minimum_eigenvalue_m2": (
            None if not min_eigenvalues else min(min_eigenvalues)
        ),
        "maximum_eigenvalue_m2": (
            None if not max_eigenvalues else max(max_eigenvalues)
        ),
        "maximum_symmetry_error_m2": (
            None if not symmetry_errors else max(symmetry_errors)
        ),
        "maximum_finite_condition_number": (
            None if not finite_conditions else max(finite_conditions)
        ),
        "passed": not invalid_indices,
    }
    return diagnostics, not invalid_indices


def audit_provider_physical_mapping(
    case: ProviderPhysicalMappingCaseV1,
    policy: ProviderPhysicalMappingPolicyV1,
) -> ProviderPhysicalMappingAuditV1:
    """Audit one frozen provider mapping without authorizing inference or targets."""

    if not isinstance(case, ProviderPhysicalMappingCaseV1):
        raise TypeError("case must be ProviderPhysicalMappingCaseV1")
    if not isinstance(policy, ProviderPhysicalMappingPolicyV1):
        raise TypeError("policy must be ProviderPhysicalMappingPolicyV1")

    transform_diagnostics, transform_valid = _transform_diagnostics(
        case.provider_to_physical,
        policy,
    )
    rotation = case.provider_to_physical[:3, :3]
    translation = case.provider_to_physical[:3, 3]
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = case.points_native * case.provider_unit_scale_m
        physical = scaled @ rotation.T + translation

    declared_valid = case.valid_mask
    finite_points = np.all(np.isfinite(case.points_native), axis=1)
    finite_physical = np.all(np.isfinite(physical), axis=1)
    finite_declared_valid = declared_valid & finite_points
    finite_transformed_declared_valid = finite_declared_valid & finite_physical
    nonfinite_declared_valid_count = int(np.sum(declared_valid & ~finite_points))
    nonfinite_transformed_declared_valid_count = int(
        np.sum(finite_declared_valid & ~finite_physical)
    )
    point_count = case.points_native.shape[0]
    declared_valid_count = int(np.sum(declared_valid))
    declared_valid_fraction = _safe_fraction(declared_valid_count, point_count)
    provider_support_complete = (
        declared_valid_count >= policy.minimum_valid_point_count
        and declared_valid_fraction >= policy.minimum_valid_fraction
    )

    query_candidate = finite_transformed_declared_valid.copy()
    timestamp_technical_valid = True
    if case.timestamps_s is None:
        timestamp_technical_valid = not policy.require_timestamps
        time_diagnostics: dict[str, object] = {
            "available": False,
            "required": policy.require_timestamps,
            "declared_valid_count": declared_valid_count,
        }
    else:
        timestamps = case.timestamps_s
        window = cast(np.ndarray, case.query_time_window_s)
        finite_timestamps = np.isfinite(timestamps)
        nonfinite_timestamp_count = int(np.sum(declared_valid & ~finite_timestamps))
        timestamp_technical_valid = nonfinite_timestamp_count == 0
        within_window = (
            finite_timestamps & (timestamps >= window[0]) & (timestamps <= window[1])
        )
        query_candidate &= within_window
        finite_valid_timestamps = declared_valid & finite_timestamps
        finite_timestamp_values = timestamps[finite_valid_timestamps]
        within_count = int(np.sum(declared_valid & within_window))
        time_diagnostics = {
            "available": True,
            "required": policy.require_timestamps,
            "query_window_s": [float(window[0]), float(window[1])],
            "provider_timestamp_range_s": (
                None
                if finite_timestamp_values.size == 0
                else [
                    float(np.min(finite_timestamp_values)),
                    float(np.max(finite_timestamp_values)),
                ]
            ),
            "declared_valid_count": declared_valid_count,
            "finite_declared_valid_timestamp_count": int(
                np.sum(finite_valid_timestamps)
            ),
            "nonfinite_declared_valid_timestamp_count": (nonfinite_timestamp_count),
            "within_window_declared_valid_count": within_count,
            "within_window_fraction_of_declared_valid": _safe_fraction(
                within_count,
                declared_valid_count,
            ),
        }

    with np.errstate(over="ignore", invalid="ignore"):
        lower = case.query_bounds_m[0] - policy.boundary_tolerance_m
        upper = case.query_bounds_m[1] + policy.boundary_tolerance_m
    effective_query_bounds_finite = bool(
        np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
    )
    if effective_query_bounds_finite:
        spatially_inside = np.all((physical >= lower) & (physical <= upper), axis=1)
    else:
        spatially_inside = np.zeros_like(declared_valid)
    mapped = query_candidate & spatially_inside
    mapped_count = int(np.sum(mapped))
    mapped_fraction = _safe_fraction(mapped_count, declared_valid_count)
    query_support_sufficient = (
        effective_query_bounds_finite
        and mapped_count >= policy.minimum_mapped_point_count
        and mapped_fraction >= policy.minimum_mapped_fraction
    )

    covariance_diagnostics, covariance_valid = _covariance_diagnostics(case, policy)
    point_technical_valid = nonfinite_declared_valid_count == 0
    transformed_point_technical_valid = nonfinite_transformed_declared_valid_count == 0
    technical_valid = (
        transform_valid
        and effective_query_bounds_finite
        and point_technical_valid
        and transformed_point_technical_valid
        and timestamp_technical_valid
        and covariance_valid
    )

    reasons: list[str] = []
    if not transform_valid:
        reasons.append("invalid-provider-to-physical-transform")
    if not effective_query_bounds_finite:
        reasons.append("nonfinite-effective-physical-query-bounds")
    if not point_technical_valid:
        reasons.append("nonfinite-declared-valid-provider-points")
    if not transformed_point_technical_valid:
        reasons.append("nonfinite-transformed-provider-points")
    if policy.require_timestamps and case.timestamps_s is None:
        reasons.append("required-provider-timestamps-missing")
    elif case.timestamps_s is not None and not timestamp_technical_valid:
        reasons.append("nonfinite-declared-valid-provider-timestamps")
    if policy.require_covariance and case.covariances_native is None:
        reasons.append("required-provider-covariance-missing")
    elif case.covariances_native is not None and not covariance_valid:
        reasons.append("invalid-declared-valid-provider-covariance")
    if not provider_support_complete:
        reasons.append("insufficient-provider-valid-support")
    if not query_support_sufficient:
        reasons.append("insufficient-physical-query-overlap")
    canonical_reasons = tuple(reason for reason in _REASON_ORDER if reason in reasons)

    diagnostics: dict[str, object] = {
        "point_accounting": {
            "point_count": point_count,
            "declared_valid_count": declared_valid_count,
            "declared_valid_fraction": declared_valid_fraction,
            "finite_declared_valid_count": int(np.sum(finite_declared_valid)),
            "nonfinite_declared_valid_count": nonfinite_declared_valid_count,
            "finite_transformed_declared_valid_count": int(
                np.sum(finite_transformed_declared_valid)
            ),
            "nonfinite_transformed_declared_valid_count": (
                nonfinite_transformed_declared_valid_count
            ),
            "query_candidate_count": int(np.sum(query_candidate)),
            "mapped_point_count": mapped_count,
            "mapped_fraction_of_declared_valid": mapped_fraction,
            "outside_spatial_bounds_query_candidate_count": int(
                np.sum(query_candidate & ~spatially_inside)
            ),
        },
        "provider_unit_scale_m": case.provider_unit_scale_m,
        "provider_bbox_native": _optional_bbox(
            case.points_native,
            finite_declared_valid,
        ),
        "physical_bbox_m": _optional_bbox(
            physical,
            finite_transformed_declared_valid,
        ),
        "mapped_bbox_m": _optional_bbox(physical, mapped),
        "query_bounds_m": {
            "lower": [float(value) for value in case.query_bounds_m[0]],
            "upper": [float(value) for value in case.query_bounds_m[1]],
            "boundary_tolerance_m": policy.boundary_tolerance_m,
            "effective_lower": [_finite_or_none(float(value)) for value in lower],
            "effective_upper": [_finite_or_none(float(value)) for value in upper],
            "effective_finite": effective_query_bounds_finite,
        },
        "transform": transform_diagnostics,
        "time": time_diagnostics,
        "covariance": covariance_diagnostics,
    }
    mapping_admissible = (
        technical_valid and provider_support_complete and query_support_sufficient
    )
    return ProviderPhysicalMappingAuditV1(
        case_id=case.case_id,
        case_artifact_id=cast(str, case.artifact_id),
        provider_artifact_id=case.provider_artifact_id,
        physical_query_id=case.physical_query_id,
        mapping_protocol_id=case.mapping_protocol_id,
        provider_frame=case.provider_frame,
        physical_frame=case.physical_frame,
        policy_id=policy.policy_id,
        mapping_admissible=mapping_admissible,
        technical_valid=technical_valid,
        provider_support_complete=provider_support_complete,
        query_support_sufficient=query_support_sufficient,
        result_reason=(
            _ADMISSIBLE_REASON if not canonical_reasons else canonical_reasons[0]
        ),
        rejection_reasons=canonical_reasons,
        diagnostics=diagnostics,
    )


__all__ = [
    "PROVIDER_PHYSICAL_MAPPING_AUDIT_INFORMATION_BOUNDARY",
    "PROVIDER_PHYSICAL_MAPPING_AUDIT_SCHEMA",
    "PROVIDER_PHYSICAL_MAPPING_AUDIT_VERSION",
    "ProviderPhysicalMappingAuditV1",
    "ProviderPhysicalMappingCaseV1",
    "ProviderPhysicalMappingPolicyV1",
    "audit_provider_physical_mapping",
]
