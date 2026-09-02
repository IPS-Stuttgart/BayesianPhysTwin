"""Quantitative cause-query estimation and diagnostic intervention design.

This module complements the binary interventional cause-identifiability
certificate.  Given an already registered finite cause family and a whitened
stacked residual, it estimates only the cause-query component that is supported
after every competing cause and declared nuisance direction is projected out.

The exact-model linear result is constructive.  For cause ``c``, let

    A_c = Q_c.T @ S_c,
    y_c = Q_c.T @ r,

where columns of ``Q_c`` span the orthogonal complement of the declared nuisance
and all competing cause signatures.  For the registered cause query ``B_c``,

    M_c = B_c @ pinv(A_c)

is the minimum-covariance linear unbiased estimator of every identifiable query
component under unit-covariance whitened noise.  Unresolved query components are
reported explicitly rather than silently set to zero.

The intervention planner enumerates a bounded finite roster and finds the
minimum-cost subset that identifies every required cause query.  When the budget
is insufficient it returns the best partial portfolio and records that the full
attribution target remains unresolved.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral, Real
from statistics import NormalDist
from types import MappingProxyType
from typing import Any, Final, Protocol, cast

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

INTERVENTIONAL_CAUSE_ESTIMATION_SCHEMA: Final = (
    "bayesian_phystwin.interventional_cause_estimation"
)
INTERVENTIONAL_CAUSE_ESTIMATION_VERSION: Final = 1
INTERVENTIONAL_CAUSE_ESTIMATION_SEMANTICS: Final = (
    "blue-cause-query-estimation-and-budgeted-intervention-design-v1"
)
INTERVENTIONAL_CAUSE_ESTIMATION_CLAIM_BOUNDARY: Final = (
    "A result is conditional on the exact registered finite cause family, "
    "intervention-response signatures, declared nuisance design, whitening, "
    "query maps, and linearized noise model.  Identifiability and a narrow "
    "confidence interval do not prove that the cause family is complete, that "
    "the supplied signatures are physically correct, or that the selected label "
    "is the unique data-generating mechanism.  The result does not establish "
    "global nonlinear identification, unseen-object transfer, online control, "
    "deployment safety, or state of the art."
)


class _InterventionBlockLike(Protocol):
    intervention_id: str
    whitened_response_signature: np.ndarray


class _CauseSignatureLike(Protocol):
    cause_id: str
    cause_query_map: np.ndarray
    intervention_blocks: Sequence[_InterventionBlockLike]
    stacked_response_signature: np.ndarray


class _CauseCertificateLike(Protocol):
    artifact_id: str
    cause_signatures: Sequence[_CauseSignatureLike]
    intervention_ids: Sequence[str]
    joint_whitened_nuisance_design: np.ndarray
    relative_rank_tolerance: float
    absolute_rank_tolerance: float
    identifiability_tolerance: float


class CauseQueryEstimateStatus(str, Enum):
    """Status of one quantitative cause-query estimate."""

    IDENTIFIABLE = "identifiable"
    PARTIALLY_IDENTIFIABLE = "partially_identifiable"
    CONFOUNDED = "confounded"
    TRIVIAL_QUERY = "trivial_query"


class InterventionPlanStatus(str, Enum):
    """Outcome of finite diagnostic-intervention planning."""

    FULL_IDENTIFICATION = "full_identification"
    BUDGET_LIMITED_PARTIAL = "budget_limited_partial_identification"


def _literal_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty literal string")
    return value


def _digest(value: object, *, name: str) -> str:
    result = _literal_string(value, name=name)
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{name} must be a 64-character lowercase hexadecimal digest")
    return result


def _finite_real(value: object, *, name: str, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        qualifier = "nonnegative " if nonnegative else ""
        raise ValueError(f"{name} must be a finite {qualifier}real number")
    result = float(value)
    if not np.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "nonnegative " if nonnegative else ""
        raise ValueError(f"{name} must be a finite {qualifier}real number")
    return result


def _probability(value: object, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie strictly between zero and one")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _matrix(value: object, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    matrix = np.ascontiguousarray(raw, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    return matrix


def _vector(value: object, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    vector = np.ascontiguousarray(raw, dtype=np.float64)
    if vector.ndim != 1 or not len(vector):
        raise ValueError(f"{name} must be a nonempty vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    return vector


def _immutable(value: np.ndarray) -> np.ndarray:
    canonical = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(canonical.tobytes(order="C"), dtype=np.float64).reshape(
        canonical.shape
    )


def _frozen_json(value: object, *, name: str) -> object:
    if value is None or type(value) in {str, bool, int}:
        return value
    if isinstance(value, Real):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError(f"{name} must contain only finite JSON values")
        return result
    if isinstance(value, Mapping):
        output: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"{name} keys must be strings")
            output[key] = _frozen_json(item, name=name)
        return MappingProxyType(output)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_frozen_json(item, name=name) for item in value)
    raise ValueError(f"{name} must contain only finite JSON values")


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


def _array_record(value: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def _content_id(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rank_tolerance(
    singular_values: np.ndarray,
    *,
    relative: float,
    absolute: float,
) -> float:
    scale = float(singular_values[0]) if len(singular_values) else 0.0
    return max(absolute, relative * scale)


def _orthogonal_complement(
    design: FloatArray,
    *,
    relative: float,
    absolute: float,
) -> FloatArray:
    """Return an orthonormal basis for the complement of ``col(design)``."""

    row_count = design.shape[0]
    if design.shape[1] == 0:
        return np.eye(row_count, dtype=np.float64)
    left, singular_values, _ = np.linalg.svd(design, full_matrices=True)
    tolerance = _rank_tolerance(
        singular_values,
        relative=relative,
        absolute=absolute,
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    return np.ascontiguousarray(left[:, rank:], dtype=np.float64)


def _pseudoinverse(
    design: FloatArray,
    *,
    relative: float,
    absolute: float,
) -> tuple[FloatArray, int, np.ndarray]:
    left, singular_values, right_transpose = np.linalg.svd(
        design,
        full_matrices=False,
    )
    tolerance = _rank_tolerance(
        singular_values,
        relative=relative,
        absolute=absolute,
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    if rank:
        inverse = 1.0 / singular_values[:rank]
        pseudoinverse = (right_transpose[:rank].T * inverse[None, :]) @ left[:, :rank].T
    else:
        pseudoinverse = np.zeros(
            (design.shape[1], design.shape[0]),
            dtype=np.float64,
        )
    return pseudoinverse, rank, singular_values


def _certificate_parts(
    certificate: _CauseCertificateLike,
) -> tuple[tuple[str, ...], tuple[_CauseSignatureLike, ...], tuple[int, ...]]:
    artifact_id = _digest(certificate.artifact_id, name="certificate artifact_id")
    del artifact_id
    intervention_ids = tuple(certificate.intervention_ids)
    if not intervention_ids or any(
        type(value) is not str or not value for value in intervention_ids
    ):
        raise ValueError("certificate intervention IDs must be nonempty strings")
    if len(intervention_ids) != len(set(intervention_ids)):
        raise ValueError("certificate intervention IDs must be unique")
    causes = tuple(certificate.cause_signatures)
    if len(causes) < 2:
        raise ValueError("certificate must contain at least two causes")
    cause_ids = tuple(cause.cause_id for cause in causes)
    if cause_ids != tuple(sorted(cause_ids)) or len(cause_ids) != len(set(cause_ids)):
        raise ValueError("certificate causes must be uniquely sorted by cause_id")
    dimensions = tuple(
        int(block.whitened_response_signature.shape[0])
        for block in causes[0].intervention_blocks
    )
    if len(dimensions) != len(intervention_ids) or any(
        value <= 0 for value in dimensions
    ):
        raise ValueError("certificate intervention dimensions are invalid")
    for cause in causes:
        if (
            tuple(block.intervention_id for block in cause.intervention_blocks)
            != intervention_ids
        ):
            raise ValueError(
                "all causes must share the certificate intervention roster"
            )
        if (
            tuple(
                int(block.whitened_response_signature.shape[0])
                for block in cause.intervention_blocks
            )
            != dimensions
        ):
            raise ValueError("all causes must share intervention row dimensions")
    nuisance = _matrix(
        certificate.joint_whitened_nuisance_design,
        name="joint_whitened_nuisance_design",
    )
    if nuisance.shape[0] != sum(dimensions):
        raise ValueError("certificate nuisance row count does not match interventions")
    return intervention_ids, causes, dimensions


def _selected_rows(
    all_ids: tuple[str, ...],
    dimensions: tuple[int, ...],
    selected_ids: Sequence[str] | None,
) -> tuple[tuple[str, ...], np.ndarray]:
    selected = all_ids if selected_ids is None else tuple(selected_ids)
    if not selected:
        raise ValueError("at least one intervention must be selected")
    if len(selected) != len(set(selected)):
        raise ValueError("selected intervention IDs must be unique")
    unknown = set(selected) - set(all_ids)
    if unknown:
        raise ValueError(f"unknown intervention IDs: {sorted(unknown)}")
    canonical = tuple(value for value in all_ids if value in set(selected))
    if selected != canonical:
        raise ValueError("selected intervention IDs must follow certificate order")
    offsets = np.cumsum((0, *dimensions))
    rows = np.concatenate(
        [
            np.arange(offsets[index], offsets[index + 1], dtype=np.int64)
            for index, value in enumerate(all_ids)
            if value in set(selected)
        ]
    )
    return canonical, rows


@dataclass(frozen=True, slots=True)
class _CauseGeometry:
    cause_id: str
    status: CauseQueryEstimateStatus
    selected_intervention_ids: tuple[str, ...]
    query_map: FloatArray
    identified_query_map: FloatArray
    unresolved_query_map: FloatArray
    factor_operator_reduced: FloatArray
    factor_operator_stacked: FloatArray
    complement_basis: FloatArray
    residualized_rank: int
    query_rank: int
    identifiable_energy_fraction: float
    normalized_unresolved_query: float
    noise_amplification: float


def _cause_geometry(
    certificate: _CauseCertificateLike,
    cause: _CauseSignatureLike,
    causes: tuple[_CauseSignatureLike, ...],
    rows: np.ndarray,
    selected_ids: tuple[str, ...],
) -> _CauseGeometry:
    relative = _finite_real(
        certificate.relative_rank_tolerance,
        name="relative_rank_tolerance",
        nonnegative=True,
    )
    absolute = _finite_real(
        certificate.absolute_rank_tolerance,
        name="absolute_rank_tolerance",
        nonnegative=True,
    )
    identifiability = _finite_real(
        certificate.identifiability_tolerance,
        name="identifiability_tolerance",
        nonnegative=True,
    )
    if relative == absolute == 0.0 or identifiability == 0.0:
        raise ValueError("certificate numerical tolerances are invalid")

    own = _matrix(
        cause.stacked_response_signature,
        name=f"{cause.cause_id} stacked_response_signature",
    )[rows]
    query = _matrix(cause.cause_query_map, name=f"{cause.cause_id} query map")
    if query.shape[1] != own.shape[1] or query.shape[0] == 0:
        raise ValueError(f"{cause.cause_id} query map has incompatible dimensions")

    nuisance_parts = [
        _matrix(
            certificate.joint_whitened_nuisance_design,
            name="joint_whitened_nuisance_design",
        )[rows]
    ]
    nuisance_parts.extend(
        _matrix(
            other.stacked_response_signature,
            name=f"{other.cause_id} stacked_response_signature",
        )[rows]
        for other in causes
        if other.cause_id != cause.cause_id
    )
    competing = np.hstack(nuisance_parts)
    complement = _orthogonal_complement(
        competing,
        relative=relative,
        absolute=absolute,
    )
    residualized = complement.T @ own
    pseudoinverse, residualized_rank, _ = _pseudoinverse(
        residualized,
        relative=relative,
        absolute=absolute,
    )
    factor_reduced = query @ pseudoinverse
    identified = factor_reduced @ residualized
    unresolved = query - identified
    factor_stacked = factor_reduced @ complement.T

    query_singular_values = np.linalg.svd(query, compute_uv=False)
    query_tolerance = _rank_tolerance(
        query_singular_values,
        relative=relative,
        absolute=absolute,
    )
    query_rank = int(np.count_nonzero(query_singular_values > query_tolerance))
    query_norm = float(np.linalg.norm(query, ord="fro"))
    unresolved_norm = float(np.linalg.norm(unresolved, ord="fro"))
    if query_norm == 0.0:
        status = CauseQueryEstimateStatus.TRIVIAL_QUERY
        energy_fraction = 1.0
        normalized_unresolved = 0.0
    else:
        normalized_unresolved = unresolved_norm / query_norm
        tolerance = absolute + identifiability * query_norm
        if unresolved_norm <= tolerance:
            status = CauseQueryEstimateStatus.IDENTIFIABLE
        else:
            energy_fraction_raw = 1.0 - (unresolved_norm**2 / query_norm**2)
            if energy_fraction_raw > identifiability:
                status = CauseQueryEstimateStatus.PARTIALLY_IDENTIFIABLE
            else:
                status = CauseQueryEstimateStatus.CONFOUNDED
        energy_fraction = float(
            np.clip(1.0 - (unresolved_norm**2 / query_norm**2), 0.0, 1.0)
        )
    noise_amplification = (
        float(np.linalg.norm(factor_reduced, ord=2)) if factor_reduced.size else 0.0
    )
    return _CauseGeometry(
        cause_id=cause.cause_id,
        status=status,
        selected_intervention_ids=selected_ids,
        query_map=_immutable(query),
        identified_query_map=_immutable(identified),
        unresolved_query_map=_immutable(unresolved),
        factor_operator_reduced=_immutable(factor_reduced),
        factor_operator_stacked=_immutable(factor_stacked),
        complement_basis=_immutable(complement),
        residualized_rank=residualized_rank,
        query_rank=query_rank,
        identifiable_energy_fraction=energy_fraction,
        normalized_unresolved_query=normalized_unresolved,
        noise_amplification=noise_amplification,
    )


@dataclass(frozen=True, slots=True)
class CauseQueryEstimateV1:
    """One identified cause-query component and its exact-model uncertainty."""

    cause_id: str
    status: CauseQueryEstimateStatus
    selected_intervention_ids: tuple[str, ...]
    estimate: np.ndarray
    covariance: np.ndarray
    marginal_lower: np.ndarray
    marginal_upper: np.ndarray
    identified_query_map: np.ndarray
    unresolved_query_map: np.ndarray
    factor_operator: np.ndarray
    residualized_cause_rank: int
    query_rank: int
    identifiable_energy_fraction: float
    normalized_unresolved_query: float
    noise_variance: float
    confidence_level: float
    noise_amplification: float
    deterministic_error_radius: float | None
    full_query_interval_valid: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        cause_id = _literal_string(self.cause_id, name="cause_id")
        if not isinstance(self.status, CauseQueryEstimateStatus):
            raise TypeError("status must be CauseQueryEstimateStatus")
        interventions = tuple(self.selected_intervention_ids)
        if not interventions or len(interventions) != len(set(interventions)):
            raise ValueError("selected intervention IDs must be nonempty and unique")
        estimate = _vector(self.estimate, name="estimate")
        covariance = _matrix(self.covariance, name="covariance")
        lower = _vector(self.marginal_lower, name="marginal_lower")
        upper = _vector(self.marginal_upper, name="marginal_upper")
        identified = _matrix(self.identified_query_map, name="identified_query_map")
        unresolved = _matrix(self.unresolved_query_map, name="unresolved_query_map")
        factor = _matrix(self.factor_operator, name="factor_operator")
        query_dimension = len(estimate)
        if (
            covariance.shape != (query_dimension, query_dimension)
            or lower.shape != estimate.shape
            or upper.shape != estimate.shape
            or identified.shape != unresolved.shape
            or identified.shape[0] != query_dimension
            or factor.shape[0] != query_dimension
            or np.any(lower > upper)
        ):
            raise ValueError("cause-query estimate array dimensions are inconsistent")
        if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=1e-12):
            raise ValueError("covariance must be symmetric")
        if float(np.min(np.linalg.eigvalsh(covariance))) < -1e-10:
            raise ValueError("covariance must be positive semidefinite")
        for name in ("residualized_cause_rank", "query_rank"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be a nonnegative integer")
            if int(value) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
            object.__setattr__(self, name, int(value))
        energy = _finite_real(
            self.identifiable_energy_fraction,
            name="identifiable_energy_fraction",
            nonnegative=True,
        )
        if energy > 1.0:
            raise ValueError("identifiable_energy_fraction must not exceed one")
        normalized = _finite_real(
            self.normalized_unresolved_query,
            name="normalized_unresolved_query",
            nonnegative=True,
        )
        noise_variance = _finite_real(
            self.noise_variance,
            name="noise_variance",
            nonnegative=True,
        )
        confidence = _probability(self.confidence_level, name="confidence_level")
        amplification = _finite_real(
            self.noise_amplification,
            name="noise_amplification",
            nonnegative=True,
        )
        radius = self.deterministic_error_radius
        if radius is not None:
            radius = _finite_real(
                radius,
                name="deterministic_error_radius",
                nonnegative=True,
            )
        if type(self.full_query_interval_valid) is not bool:
            raise ValueError("full_query_interval_valid must be a literal boolean")
        metadata = _frozen_json(self.metadata, name="estimate metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("estimate metadata must be a mapping")
        object.__setattr__(self, "cause_id", cause_id)
        object.__setattr__(self, "selected_intervention_ids", interventions)
        for name, value in (
            ("estimate", estimate),
            ("covariance", covariance),
            ("marginal_lower", lower),
            ("marginal_upper", upper),
            ("identified_query_map", identified),
            ("unresolved_query_map", unresolved),
            ("factor_operator", factor),
        ):
            object.__setattr__(self, name, _immutable(value))
        object.__setattr__(self, "identifiable_energy_fraction", energy)
        object.__setattr__(self, "normalized_unresolved_query", normalized)
        object.__setattr__(self, "noise_variance", noise_variance)
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(self, "noise_amplification", amplification)
        object.__setattr__(self, "deterministic_error_radius", radius)
        object.__setattr__(self, "metadata", metadata)
        expected = _content_id(self.descriptor())
        if self.artifact_id is not None:
            supplied = _digest(self.artifact_id, name="artifact_id")
            if supplied != expected:
                raise ValueError("estimate artifact_id does not match content")
        object.__setattr__(self, "artifact_id", expected)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": INTERVENTIONAL_CAUSE_ESTIMATION_SCHEMA,
            "schema_version": INTERVENTIONAL_CAUSE_ESTIMATION_VERSION,
            "artifact_kind": "CauseQueryEstimateV1",
            "semantics": INTERVENTIONAL_CAUSE_ESTIMATION_SEMANTICS,
            "cause_id": self.cause_id,
            "status": self.status.value,
            "selected_intervention_ids": list(self.selected_intervention_ids),
            "estimate": _array_record(self.estimate),
            "covariance": _array_record(self.covariance),
            "marginal_lower": _array_record(self.marginal_lower),
            "marginal_upper": _array_record(self.marginal_upper),
            "identified_query_map": _array_record(self.identified_query_map),
            "unresolved_query_map": _array_record(self.unresolved_query_map),
            "factor_operator": _array_record(self.factor_operator),
            "residualized_cause_rank": self.residualized_cause_rank,
            "query_rank": self.query_rank,
            "identifiable_energy_fraction": self.identifiable_energy_fraction,
            "normalized_unresolved_query": self.normalized_unresolved_query,
            "noise_variance": self.noise_variance,
            "confidence_level": self.confidence_level,
            "noise_amplification": self.noise_amplification,
            "deterministic_error_radius": self.deterministic_error_radius,
            "full_query_interval_valid": self.full_query_interval_valid,
            "metadata": _plain_json(self.metadata),
            "claim_boundary": INTERVENTIONAL_CAUSE_ESTIMATION_CLAIM_BOUNDARY,
        }

    def to_record(self) -> dict[str, object]:
        record = self.descriptor()
        record.update(
            {
                "estimate_values": self.estimate.tolist(),
                "covariance_values": self.covariance.tolist(),
                "marginal_lower_values": self.marginal_lower.tolist(),
                "marginal_upper_values": self.marginal_upper.tolist(),
                "artifact_id": self.artifact_id,
            }
        )
        return record


@dataclass(frozen=True, slots=True)
class InterventionalCauseEstimateBundleV1:
    """Quantitative estimates for every registered cause query."""

    certificate_id: str
    stacked_residual_id: str
    selected_intervention_ids: tuple[str, ...]
    cause_estimates: tuple[CauseQueryEstimateV1, ...]
    noise_variance: float
    confidence_level: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        certificate_id = _digest(self.certificate_id, name="certificate_id")
        residual_id = _digest(self.stacked_residual_id, name="stacked_residual_id")
        interventions = tuple(self.selected_intervention_ids)
        estimates = tuple(self.cause_estimates)
        if not estimates or any(
            not isinstance(item, CauseQueryEstimateV1) for item in estimates
        ):
            raise TypeError("cause_estimates must contain CauseQueryEstimateV1 values")
        cause_ids = tuple(item.cause_id for item in estimates)
        if cause_ids != tuple(sorted(cause_ids)) or len(cause_ids) != len(
            set(cause_ids)
        ):
            raise ValueError("cause estimates must be uniquely sorted by cause_id")
        if any(item.selected_intervention_ids != interventions for item in estimates):
            raise ValueError(
                "all cause estimates must share the bundle intervention set"
            )
        noise_variance = _finite_real(
            self.noise_variance,
            name="noise_variance",
            nonnegative=True,
        )
        confidence = _probability(self.confidence_level, name="confidence_level")
        metadata = _frozen_json(self.metadata, name="bundle metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("bundle metadata must be a mapping")
        object.__setattr__(self, "certificate_id", certificate_id)
        object.__setattr__(self, "stacked_residual_id", residual_id)
        object.__setattr__(self, "selected_intervention_ids", interventions)
        object.__setattr__(self, "cause_estimates", estimates)
        object.__setattr__(self, "noise_variance", noise_variance)
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(self, "metadata", metadata)
        expected = _content_id(self.descriptor())
        if self.artifact_id is not None:
            supplied = _digest(self.artifact_id, name="artifact_id")
            if supplied != expected:
                raise ValueError("bundle artifact_id does not match content")
        object.__setattr__(self, "artifact_id", expected)

    def by_cause(self, cause_id: str) -> CauseQueryEstimateV1:
        for item in self.cause_estimates:
            if item.cause_id == cause_id:
                return item
        raise KeyError(cause_id)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": INTERVENTIONAL_CAUSE_ESTIMATION_SCHEMA,
            "schema_version": INTERVENTIONAL_CAUSE_ESTIMATION_VERSION,
            "artifact_kind": "InterventionalCauseEstimateBundleV1",
            "semantics": INTERVENTIONAL_CAUSE_ESTIMATION_SEMANTICS,
            "certificate_id": self.certificate_id,
            "stacked_residual_id": self.stacked_residual_id,
            "selected_intervention_ids": list(self.selected_intervention_ids),
            "cause_estimate_ids": [item.artifact_id for item in self.cause_estimates],
            "noise_variance": self.noise_variance,
            "confidence_level": self.confidence_level,
            "metadata": _plain_json(self.metadata),
            "claim_boundary": INTERVENTIONAL_CAUSE_ESTIMATION_CLAIM_BOUNDARY,
        }

    def to_record(self) -> dict[str, object]:
        return {
            **self.descriptor(),
            "cause_estimates": [item.to_record() for item in self.cause_estimates],
            "artifact_id": self.artifact_id,
        }


def estimate_cause_queries(
    certificate: _CauseCertificateLike,
    stacked_whitened_residual: object,
    *,
    intervention_ids: Sequence[str] | None = None,
    residual_noise_variance: float = 1.0,
    residual_noise_radius: float | None = None,
    confidence_level: float = 0.95,
    metadata: Mapping[str, Any] | None = None,
) -> InterventionalCauseEstimateBundleV1:
    """Estimate every supported cause-query component.

    The residual is always supplied on the certificate's complete stacked row
    roster.  ``intervention_ids`` selects the rows used by the estimator without
    changing that residual identity.  Intervals cover the identified component
    under the exact whitened Gaussian model.  They cover the complete registered
    query only when ``full_query_interval_valid`` is true.
    """

    all_ids, causes, dimensions = _certificate_parts(certificate)
    selected_ids, rows = _selected_rows(all_ids, dimensions, intervention_ids)
    residual = _vector(stacked_whitened_residual, name="stacked_whitened_residual")
    if len(residual) != sum(dimensions):
        raise ValueError("stacked residual length does not match certificate rows")
    noise_variance = _finite_real(
        residual_noise_variance,
        name="residual_noise_variance",
        nonnegative=True,
    )
    if noise_variance <= 0.0:
        raise ValueError("residual_noise_variance must be positive")
    confidence = _probability(confidence_level, name="confidence_level")
    radius = None
    if residual_noise_radius is not None:
        radius = _finite_real(
            residual_noise_radius,
            name="residual_noise_radius",
            nonnegative=True,
        )
    residual_id = hashlib.sha256(residual.tobytes(order="C")).hexdigest()
    selected_residual = residual[rows]
    z_value = NormalDist().inv_cdf(0.5 + 0.5 * confidence)
    estimates: list[CauseQueryEstimateV1] = []
    for cause in causes:
        geometry = _cause_geometry(
            certificate,
            cause,
            causes,
            rows,
            selected_ids,
        )
        reduced_residual = geometry.complement_basis.T @ selected_residual
        estimate = geometry.factor_operator_reduced @ reduced_residual
        covariance = noise_variance * (
            geometry.factor_operator_reduced @ geometry.factor_operator_reduced.T
        )
        standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        lower = estimate - z_value * standard_error
        upper = estimate + z_value * standard_error
        error_radius = None if radius is None else geometry.noise_amplification * radius
        estimates.append(
            CauseQueryEstimateV1(
                cause_id=cause.cause_id,
                status=geometry.status,
                selected_intervention_ids=selected_ids,
                estimate=estimate,
                covariance=covariance,
                marginal_lower=lower,
                marginal_upper=upper,
                identified_query_map=geometry.identified_query_map,
                unresolved_query_map=geometry.unresolved_query_map,
                factor_operator=geometry.factor_operator_stacked,
                residualized_cause_rank=geometry.residualized_rank,
                query_rank=geometry.query_rank,
                identifiable_energy_fraction=(geometry.identifiable_energy_fraction),
                normalized_unresolved_query=(geometry.normalized_unresolved_query),
                noise_variance=noise_variance,
                confidence_level=confidence,
                noise_amplification=geometry.noise_amplification,
                deterministic_error_radius=error_radius,
                full_query_interval_valid=(
                    geometry.status
                    in {
                        CauseQueryEstimateStatus.IDENTIFIABLE,
                        CauseQueryEstimateStatus.TRIVIAL_QUERY,
                    }
                ),
                metadata={
                    "certificate_id": str(certificate.artifact_id),
                    "estimator": "minimum-covariance-linear-unbiased-v1",
                },
            )
        )
    return InterventionalCauseEstimateBundleV1(
        certificate_id=str(certificate.artifact_id),
        stacked_residual_id=residual_id,
        selected_intervention_ids=selected_ids,
        cause_estimates=tuple(estimates),
        noise_variance=noise_variance,
        confidence_level=confidence,
        metadata={} if metadata is None else metadata,
    )


@dataclass(frozen=True, slots=True)
class InterventionSubsetScoreV1:
    """One feasible diagnostic intervention portfolio."""

    intervention_ids: tuple[str, ...]
    total_cost: float
    fully_identified_cause_count: int
    partially_identified_cause_count: int
    required_cause_count: int
    minimum_identifiable_energy_fraction: float
    mean_identifiable_energy_fraction: float
    worst_query_variance: float | None
    worst_noise_amplification: float | None
    all_required_causes_identified: bool

    def to_record(self) -> dict[str, object]:
        return {
            "intervention_ids": list(self.intervention_ids),
            "total_cost": self.total_cost,
            "fully_identified_cause_count": self.fully_identified_cause_count,
            "partially_identified_cause_count": self.partially_identified_cause_count,
            "required_cause_count": self.required_cause_count,
            "minimum_identifiable_energy_fraction": (
                self.minimum_identifiable_energy_fraction
            ),
            "mean_identifiable_energy_fraction": (
                self.mean_identifiable_energy_fraction
            ),
            "worst_query_variance": self.worst_query_variance,
            "worst_noise_amplification": self.worst_noise_amplification,
            "all_required_causes_identified": self.all_required_causes_identified,
        }


@dataclass(frozen=True, slots=True)
class InterventionDesignResultV1:
    """Minimum-cost full attribution plan or best budget-limited partial plan."""

    certificate_id: str
    status: InterventionPlanStatus
    required_cause_ids: tuple[str, ...]
    required_intervention_ids: tuple[str, ...]
    selected_intervention_ids: tuple[str, ...]
    selected_total_cost: float
    selected_score: InterventionSubsetScoreV1
    feasible_subset_count: int
    fully_identifying_subset_count: int
    pareto_frontier: tuple[InterventionSubsetScoreV1, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        certificate_id = _digest(self.certificate_id, name="certificate_id")
        if not isinstance(self.status, InterventionPlanStatus):
            raise TypeError("status must be InterventionPlanStatus")
        for name in (
            "required_cause_ids",
            "required_intervention_ids",
            "selected_intervention_ids",
        ):
            values = tuple(getattr(self, name))
            if len(values) != len(set(values)) or any(
                type(value) is not str or not value for value in values
            ):
                raise ValueError(f"{name} must contain unique nonempty strings")
            object.__setattr__(self, name, values)
        cost = _finite_real(
            self.selected_total_cost,
            name="selected_total_cost",
            nonnegative=True,
        )
        feasible = _positive_integer(
            self.feasible_subset_count, name="feasible_subset_count"
        )
        identifying = int(self.fully_identifying_subset_count)
        if identifying < 0 or identifying > feasible:
            raise ValueError("fully_identifying_subset_count is invalid")
        frontier = tuple(self.pareto_frontier)
        if not frontier or any(
            not isinstance(item, InterventionSubsetScoreV1) for item in frontier
        ):
            raise TypeError("pareto_frontier must contain subset scores")
        metadata = _frozen_json(self.metadata, name="plan metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("plan metadata must be a mapping")
        object.__setattr__(self, "certificate_id", certificate_id)
        object.__setattr__(self, "selected_total_cost", cost)
        object.__setattr__(self, "feasible_subset_count", feasible)
        object.__setattr__(self, "fully_identifying_subset_count", identifying)
        object.__setattr__(self, "pareto_frontier", frontier)
        object.__setattr__(self, "metadata", metadata)
        expected = _content_id(self.descriptor())
        if self.artifact_id is not None:
            supplied = _digest(self.artifact_id, name="artifact_id")
            if supplied != expected:
                raise ValueError("plan artifact_id does not match content")
        object.__setattr__(self, "artifact_id", expected)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": INTERVENTIONAL_CAUSE_ESTIMATION_SCHEMA,
            "schema_version": INTERVENTIONAL_CAUSE_ESTIMATION_VERSION,
            "artifact_kind": "InterventionDesignResultV1",
            "semantics": INTERVENTIONAL_CAUSE_ESTIMATION_SEMANTICS,
            "certificate_id": self.certificate_id,
            "status": self.status.value,
            "required_cause_ids": list(self.required_cause_ids),
            "required_intervention_ids": list(self.required_intervention_ids),
            "selected_intervention_ids": list(self.selected_intervention_ids),
            "selected_total_cost": self.selected_total_cost,
            "selected_score": self.selected_score.to_record(),
            "feasible_subset_count": self.feasible_subset_count,
            "fully_identifying_subset_count": self.fully_identifying_subset_count,
            "pareto_frontier": [item.to_record() for item in self.pareto_frontier],
            "metadata": _plain_json(self.metadata),
            "claim_boundary": INTERVENTIONAL_CAUSE_ESTIMATION_CLAIM_BOUNDARY,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}


def _score_subset(
    certificate: _CauseCertificateLike,
    causes: tuple[_CauseSignatureLike, ...],
    all_ids: tuple[str, ...],
    dimensions: tuple[int, ...],
    selected_ids: tuple[str, ...],
    costs: Mapping[str, float],
    required_causes: tuple[str, ...],
) -> InterventionSubsetScoreV1:
    _, rows = _selected_rows(all_ids, dimensions, selected_ids)
    geometries = {
        cause.cause_id: _cause_geometry(
            certificate,
            cause,
            causes,
            rows,
            selected_ids,
        )
        for cause in causes
        if cause.cause_id in set(required_causes)
    }
    full = sum(
        geometry.status is CauseQueryEstimateStatus.IDENTIFIABLE
        for geometry in geometries.values()
    )
    partial = sum(
        geometry.status
        in {
            CauseQueryEstimateStatus.IDENTIFIABLE,
            CauseQueryEstimateStatus.PARTIALLY_IDENTIFIABLE,
        }
        for geometry in geometries.values()
    )
    energies = [
        geometry.identifiable_energy_fraction for geometry in geometries.values()
    ]
    all_identified = full == len(required_causes)
    worst_variance: float | None = None
    worst_amplification: float | None = None
    if all_identified:
        covariance_eigenvalues: list[float] = []
        amplifications: list[float] = []
        for geometry in geometries.values():
            covariance = (
                geometry.factor_operator_reduced @ geometry.factor_operator_reduced.T
            )
            covariance_eigenvalues.append(float(np.max(np.linalg.eigvalsh(covariance))))
            amplifications.append(geometry.noise_amplification)
        worst_variance = max(covariance_eigenvalues, default=0.0)
        worst_amplification = max(amplifications, default=0.0)
    return InterventionSubsetScoreV1(
        intervention_ids=selected_ids,
        total_cost=float(sum(costs[value] for value in selected_ids)),
        fully_identified_cause_count=full,
        partially_identified_cause_count=partial,
        required_cause_count=len(required_causes),
        minimum_identifiable_energy_fraction=min(energies, default=0.0),
        mean_identifiable_energy_fraction=float(np.mean(energies)) if energies else 0.0,
        worst_query_variance=worst_variance,
        worst_noise_amplification=worst_amplification,
        all_required_causes_identified=all_identified,
    )


def _pareto_frontier(
    scores: Sequence[InterventionSubsetScoreV1],
) -> tuple[InterventionSubsetScoreV1, ...]:
    frontier: list[InterventionSubsetScoreV1] = []
    for candidate in scores:
        dominated = False
        for other in scores:
            if other is candidate:
                continue
            not_worse = (
                other.total_cost <= candidate.total_cost
                and other.fully_identified_cause_count
                >= candidate.fully_identified_cause_count
                and other.mean_identifiable_energy_fraction
                >= candidate.mean_identifiable_energy_fraction
            )
            strictly_better = (
                other.total_cost < candidate.total_cost
                or other.fully_identified_cause_count
                > candidate.fully_identified_cause_count
                or other.mean_identifiable_energy_fraction
                > candidate.mean_identifiable_energy_fraction
            )
            if not_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return tuple(
        sorted(
            frontier,
            key=lambda item: (
                item.total_cost,
                -item.fully_identified_cause_count,
                -item.mean_identifiable_energy_fraction,
                item.intervention_ids,
            ),
        )
    )


def plan_diagnostic_interventions(
    certificate: _CauseCertificateLike,
    intervention_costs: Mapping[str, object],
    *,
    required_intervention_ids: Sequence[str] = (),
    required_cause_ids: Sequence[str] | None = None,
    maximum_total_cost: float | None = None,
    maximum_interventions: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> InterventionDesignResultV1:
    """Select a finite diagnostic intervention portfolio.

    Full-identification portfolios are ranked by total cost and then worst-case
    exact-model query variance.  When no full portfolio fits the budget, the
    planner maximizes fully identified causes, then partial cause coverage, then
    identifiable query energy, and finally minimizes cost.
    """

    all_ids, causes, dimensions = _certificate_parts(certificate)
    if set(intervention_costs) != set(all_ids):
        raise ValueError("intervention costs must cover exactly the certificate roster")
    costs = {
        intervention_id: _finite_real(
            intervention_costs[intervention_id],
            name=f"cost[{intervention_id}]",
            nonnegative=True,
        )
        for intervention_id in all_ids
    }
    required_interventions = tuple(required_intervention_ids)
    if len(required_interventions) != len(set(required_interventions)):
        raise ValueError("required intervention IDs must be unique")
    if set(required_interventions) - set(all_ids):
        raise ValueError("required intervention is absent from certificate")
    required_interventions = tuple(
        value for value in all_ids if value in set(required_interventions)
    )
    all_cause_ids = tuple(cause.cause_id for cause in causes)
    required_causes = (
        all_cause_ids if required_cause_ids is None else tuple(required_cause_ids)
    )
    if not required_causes or len(required_causes) != len(set(required_causes)):
        raise ValueError("required cause IDs must be nonempty and unique")
    if set(required_causes) - set(all_cause_ids):
        raise ValueError("required cause is absent from certificate")
    required_causes = tuple(
        value for value in all_cause_ids if value in set(required_causes)
    )
    cost_limit = None
    if maximum_total_cost is not None:
        cost_limit = _finite_real(
            maximum_total_cost,
            name="maximum_total_cost",
            nonnegative=True,
        )
    action_limit = len(all_ids)
    if maximum_interventions is not None:
        action_limit = _positive_integer(
            maximum_interventions,
            name="maximum_interventions",
        )
        if action_limit > len(all_ids):
            raise ValueError("maximum_interventions exceeds the registered roster")
    if len(all_ids) > 15:
        raise ValueError("exact intervention planning is limited to 15 actions")

    scores: list[InterventionSubsetScoreV1] = []
    for count in range(1, action_limit + 1):
        for indices in itertools.combinations(range(len(all_ids)), count):
            subset = tuple(all_ids[index] for index in indices)
            if not set(required_interventions).issubset(subset):
                continue
            total_cost = sum(costs[value] for value in subset)
            if cost_limit is not None and total_cost > cost_limit + 1e-12:
                continue
            scores.append(
                _score_subset(
                    certificate,
                    causes,
                    all_ids,
                    dimensions,
                    subset,
                    costs,
                    required_causes,
                )
            )
    if not scores:
        raise ValueError("no intervention subset satisfies the registered constraints")
    fully_identifying = [item for item in scores if item.all_required_causes_identified]
    if fully_identifying:
        selected = min(
            fully_identifying,
            key=lambda item: (
                item.total_cost,
                cast(float, item.worst_query_variance),
                cast(float, item.worst_noise_amplification),
                len(item.intervention_ids),
                item.intervention_ids,
            ),
        )
        status = InterventionPlanStatus.FULL_IDENTIFICATION
    else:
        selected = min(
            scores,
            key=lambda item: (
                -item.fully_identified_cause_count,
                -item.partially_identified_cause_count,
                -item.minimum_identifiable_energy_fraction,
                -item.mean_identifiable_energy_fraction,
                item.total_cost,
                len(item.intervention_ids),
                item.intervention_ids,
            ),
        )
        status = InterventionPlanStatus.BUDGET_LIMITED_PARTIAL
    return InterventionDesignResultV1(
        certificate_id=str(certificate.artifact_id),
        status=status,
        required_cause_ids=required_causes,
        required_intervention_ids=required_interventions,
        selected_intervention_ids=selected.intervention_ids,
        selected_total_cost=selected.total_cost,
        selected_score=selected,
        feasible_subset_count=len(scores),
        fully_identifying_subset_count=len(fully_identifying),
        pareto_frontier=_pareto_frontier(scores),
        metadata={} if metadata is None else metadata,
    )
