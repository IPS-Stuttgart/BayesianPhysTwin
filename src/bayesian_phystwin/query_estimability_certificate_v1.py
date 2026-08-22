"""Stability of an identifiable query under whitened observation noise.

The certificate consumes one exact
:class:`~bayesian_phystwin.query_identifiability_certificate_v2.QueryIdentifiabilityCertificateV2`.
For the source factor operator ``M`` and nuisance projector ``P_N``, the effective
map from unit-covariance whitened observation noise to the registered query is
``G = M (I - P_N)``. A supplied positive-definite query scale ``S_q`` converts
that map to dimensionless coordinates. The maximum singular value of
``chol(S_q)^{-1} G`` is the worst-direction normalized noise gain.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import Any, Final, cast

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    immutable_array,
    literal_lower_hex,
    plain_json,
)
from ._portable_contracts import content_id
from .query_identifiability_certificate_v2 import (
    QUERY_IDENTIFIABILITY_CERTIFICATE_SCHEMA,
    QUERY_IDENTIFIABILITY_CERTIFICATE_VERSION,
    QueryIdentifiabilityCertificateV2,
    QueryIdentifiabilityStatus,
)

QUERY_ESTIMABILITY_CERTIFICATE_SCHEMA: Final = (
    "bayesian_phystwin.query_estimability_certificate"
)
QUERY_ESTIMABILITY_CERTIFICATE_VERSION: Final = 1
QUERY_ESTIMABILITY_CERTIFICATE_SEMANTICS: Final = (
    "query-estimability-by-query-scaled-whitened-noise-gain-v1"
)
QUERY_ESTIMABILITY_CERTIFICATE_CLAIM_BOUNDARY: Final = (
    "Local linear query stability under unit-covariance noise in the exact "
    "whitened observation coordinates, nuisance projector, factor operator, "
    "positive-definite query scale, and source-frozen gain limit only. The "
    "certificate does not establish model correctness, global nonlinear "
    "identifiability, provider competence, uncertainty calibration, unseen-object "
    "transfer, deployment safety, or Causal4D benefit."
)


class QueryEstimabilityStatus(str, Enum):
    """Decision after identifiability and normalized noise-gain checks."""

    STABLY_ESTIMABLE = "stably_estimable"
    IDENTIFIABLE_BUT_UNSTABLE = "identifiable_but_unstable"
    NONIDENTIFIABLE = "nonidentifiable"
    TRIVIAL_QUERY = "trivial_query"


def _real_float64_matrix(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    matrix = np.ascontiguousarray(raw, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    return matrix


def _immutable_float64(value: object) -> np.ndarray:
    return cast(np.ndarray, immutable_array(value, dtype=np.float64))


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative real number")
    return result


def _array_record(value: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


@dataclass(frozen=True, slots=True)
class QueryEstimabilityCertificateV1:
    """Content-addressed local query noise-amplification certificate.

    ``query_scale`` is a positive-definite covariance-like scale in the exact
    registered query coordinates. ``noise_gain_limit`` is dimensionless and must
    be frozen before target outcomes are opened.
    """

    identifiability_certificate: QueryIdentifiabilityCertificateV2 = field(repr=False)
    query_scale_id: str
    query_scale: np.ndarray = field(repr=False)
    noise_gain_limit: float
    relative_scale_tolerance: float = 1e-10
    absolute_scale_tolerance: float = 1e-12
    stability_tolerance: float = 1e-12
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    effective_factor_operator: np.ndarray = field(init=False, repr=False)
    query_noise_covariance: np.ndarray = field(init=False, repr=False)
    query_scale_cholesky: np.ndarray = field(init=False, repr=False)
    normalized_factor_operator: np.ndarray = field(init=False, repr=False)
    normalized_query_noise_covariance: np.ndarray = field(init=False, repr=False)
    query_scale_eigenvalues: np.ndarray = field(init=False, repr=False)
    normalized_noise_singular_values: np.ndarray = field(init=False, repr=False)
    query_scale_rank_tolerance: float = field(init=False)
    query_scale_condition_number: float = field(init=False)
    nuisance_leakage_frobenius: float = field(init=False)
    maximum_normalized_noise_gain: float = field(init=False)
    rms_normalized_noise_gain: float = field(init=False)
    noise_gain_margin: float = field(init=False)
    admission_bound: float = field(init=False)
    status: QueryEstimabilityStatus = field(init=False)

    def __post_init__(self) -> None:
        source = self.identifiability_certificate
        if not isinstance(source, QueryIdentifiabilityCertificateV2):
            raise ValueError(
                "identifiability_certificate must be a "
                "QueryIdentifiabilityCertificateV2"
            )
        query_scale_id = cast(
            str,
            literal_lower_hex(
                self.query_scale_id,
                name="query_scale_id",
                lengths={64},
            ),
        )
        scale = _real_float64_matrix(self.query_scale, name="query_scale")
        expected_shape = (source.query_dimension, source.query_dimension)
        if scale.shape != expected_shape:
            raise ValueError(
                "query_scale must be square with one row and column per query "
                "coordinate"
            )
        if not np.array_equal(scale, scale.T):
            raise ValueError("query_scale must be exactly symmetric")

        relative = _finite_nonnegative(
            self.relative_scale_tolerance,
            name="relative_scale_tolerance",
        )
        absolute = _finite_nonnegative(
            self.absolute_scale_tolerance,
            name="absolute_scale_tolerance",
        )
        stability = _finite_nonnegative(
            self.stability_tolerance,
            name="stability_tolerance",
        )
        gain_limit = _finite_nonnegative(
            self.noise_gain_limit,
            name="noise_gain_limit",
        )
        if relative == absolute == 0.0:
            raise ValueError("at least one query-scale tolerance must be positive")

        scale_eigenvalues = np.linalg.eigvalsh(scale)
        largest_scale_eigenvalue = float(scale_eigenvalues[-1])
        scale_tolerance = max(absolute, relative * largest_scale_eigenvalue)
        smallest_scale_eigenvalue = float(scale_eigenvalues[0])
        if smallest_scale_eigenvalue <= scale_tolerance:
            raise ValueError(
                "query_scale must be positive definite above the frozen "
                "query-scale tolerance"
            )
        scale_condition_number = largest_scale_eigenvalue / smallest_scale_eigenvalue
        scale_cholesky = np.linalg.cholesky(scale)

        factor = source.factor_operator
        nuisance_leakage = factor @ source.nuisance_projector
        effective_factor = factor - nuisance_leakage
        query_noise_covariance = effective_factor @ effective_factor.T
        query_noise_covariance = 0.5 * (
            query_noise_covariance + query_noise_covariance.T
        )
        normalized_factor = np.linalg.solve(scale_cholesky, effective_factor)
        normalized_covariance = normalized_factor @ normalized_factor.T
        normalized_covariance = 0.5 * (normalized_covariance + normalized_covariance.T)
        normalized_singular_values = np.linalg.svd(
            normalized_factor,
            compute_uv=False,
        )

        maximum_gain = float(normalized_singular_values[0])
        rms_gain = float(
            np.linalg.norm(normalized_factor, ord="fro")
            / np.sqrt(float(source.query_dimension))
        )
        leakage_frobenius = float(np.linalg.norm(nuisance_leakage, ord="fro"))
        admission_bound = gain_limit + stability
        gain_margin = gain_limit - maximum_gain

        if source.status is QueryIdentifiabilityStatus.NONIDENTIFIABLE:
            status = QueryEstimabilityStatus.NONIDENTIFIABLE
        elif source.status is QueryIdentifiabilityStatus.TRIVIAL_QUERY:
            status = QueryEstimabilityStatus.TRIVIAL_QUERY
        elif maximum_gain <= admission_bound:
            status = QueryEstimabilityStatus.STABLY_ESTIMABLE
        else:
            status = QueryEstimabilityStatus.IDENTIFIABLE_BUT_UNSTABLE

        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="query estimability certificate metadata",
        )
        for name, value in (
            ("query_scale", scale),
            ("effective_factor_operator", effective_factor),
            ("query_noise_covariance", query_noise_covariance),
            ("query_scale_cholesky", scale_cholesky),
            ("normalized_factor_operator", normalized_factor),
            ("normalized_query_noise_covariance", normalized_covariance),
            ("query_scale_eigenvalues", scale_eigenvalues),
            ("normalized_noise_singular_values", normalized_singular_values),
        ):
            object.__setattr__(self, name, _immutable_float64(value))
        for name, value in (
            ("noise_gain_limit", gain_limit),
            ("relative_scale_tolerance", relative),
            ("absolute_scale_tolerance", absolute),
            ("stability_tolerance", stability),
            ("query_scale_rank_tolerance", scale_tolerance),
            ("query_scale_condition_number", scale_condition_number),
            ("nuisance_leakage_frobenius", leakage_frobenius),
            ("maximum_normalized_noise_gain", maximum_gain),
            ("rms_normalized_noise_gain", rms_gain),
            ("noise_gain_margin", gain_margin),
            ("admission_bound", admission_bound),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "query_scale_id", query_scale_id)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "status", status)

        expected_id = cast(str, content_id(self.descriptor()))
        supplied_id = self.artifact_id
        if supplied_id is not None:
            supplied_id = cast(
                str,
                literal_lower_hex(
                    supplied_id,
                    name="artifact_id",
                    lengths={64},
                ),
            )
            if supplied_id != expected_id:
                raise ValueError(
                    "query estimability certificate artifact_id does not match content"
                )
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def identifiable(self) -> bool:
        return self.identifiability_certificate.identifiable

    @property
    def stably_estimable(self) -> bool:
        return self.status is QueryEstimabilityStatus.STABLY_ESTIMABLE

    @property
    def passes_stability_gate(self) -> bool:
        return self.stably_estimable

    @property
    def observation_dimension(self) -> int:
        return int(self.effective_factor_operator.shape[1])

    @property
    def query_dimension(self) -> int:
        return int(self.effective_factor_operator.shape[0])

    def arrays(self) -> Mapping[str, np.ndarray]:
        """Return immutable input and derived arrays bound by the certificate."""

        return {
            "query_scale": self.query_scale,
            "effective_factor_operator": self.effective_factor_operator,
            "query_noise_covariance": self.query_noise_covariance,
            "query_scale_cholesky": self.query_scale_cholesky,
            "normalized_factor_operator": self.normalized_factor_operator,
            "normalized_query_noise_covariance": (
                self.normalized_query_noise_covariance
            ),
            "query_scale_eigenvalues": self.query_scale_eigenvalues,
            "normalized_noise_singular_values": (self.normalized_noise_singular_values),
        }

    def descriptor(self) -> dict[str, object]:
        source = self.identifiability_certificate
        return {
            "schema": QUERY_ESTIMABILITY_CERTIFICATE_SCHEMA,
            "schema_version": QUERY_ESTIMABILITY_CERTIFICATE_VERSION,
            "semantics": QUERY_ESTIMABILITY_CERTIFICATE_SEMANTICS,
            "identifiability_certificate_id": source.artifact_id,
            "identifiability_schema": QUERY_IDENTIFIABILITY_CERTIFICATE_SCHEMA,
            "identifiability_schema_version": (
                QUERY_IDENTIFIABILITY_CERTIFICATE_VERSION
            ),
            "identifiability_status": source.status.value,
            "physical_response_id": source.physical_response_id,
            "observation_mapping_id": source.observation_mapping_id,
            "nuisance_design_id": source.nuisance_design_id,
            "query_id": source.query_id,
            "query_scale_id": self.query_scale_id,
            "query_scale": _array_record(self.query_scale),
            "effective_factor_operator": _array_record(self.effective_factor_operator),
            "query_noise_covariance": _array_record(self.query_noise_covariance),
            "query_scale_cholesky": _array_record(self.query_scale_cholesky),
            "normalized_factor_operator": _array_record(
                self.normalized_factor_operator
            ),
            "normalized_query_noise_covariance": _array_record(
                self.normalized_query_noise_covariance
            ),
            "query_scale_eigenvalues": _array_record(self.query_scale_eigenvalues),
            "normalized_noise_singular_values": _array_record(
                self.normalized_noise_singular_values
            ),
            "noise_gain_limit": self.noise_gain_limit,
            "relative_scale_tolerance": self.relative_scale_tolerance,
            "absolute_scale_tolerance": self.absolute_scale_tolerance,
            "stability_tolerance": self.stability_tolerance,
            "query_scale_rank_tolerance": self.query_scale_rank_tolerance,
            "query_scale_condition_number": self.query_scale_condition_number,
            "nuisance_leakage_frobenius": self.nuisance_leakage_frobenius,
            "maximum_normalized_noise_gain": (self.maximum_normalized_noise_gain),
            "rms_normalized_noise_gain": self.rms_normalized_noise_gain,
            "noise_gain_margin": self.noise_gain_margin,
            "admission_bound": self.admission_bound,
            "status": self.status.value,
            "metadata": plain_json(self.metadata),
            "claim_boundary": QUERY_ESTIMABILITY_CERTIFICATE_CLAIM_BOUNDARY,
        }

    def summary(self) -> dict[str, object]:
        return {
            "schema": QUERY_ESTIMABILITY_CERTIFICATE_SCHEMA,
            "schema_version": QUERY_ESTIMABILITY_CERTIFICATE_VERSION,
            "artifact_id": self.artifact_id,
            "identifiability_certificate_id": (
                self.identifiability_certificate.artifact_id
            ),
            "status": self.status.value,
            "identifiable": self.identifiable,
            "stably_estimable": self.stably_estimable,
            "passes_stability_gate": self.passes_stability_gate,
            "observation_dimension": self.observation_dimension,
            "query_dimension": self.query_dimension,
            "noise_gain_limit": self.noise_gain_limit,
            "maximum_normalized_noise_gain": (self.maximum_normalized_noise_gain),
            "rms_normalized_noise_gain": self.rms_normalized_noise_gain,
            "noise_gain_margin": self.noise_gain_margin,
            "query_scale_condition_number": self.query_scale_condition_number,
            "nuisance_leakage_frobenius": self.nuisance_leakage_frobenius,
            "claim_boundary": QUERY_ESTIMABILITY_CERTIFICATE_CLAIM_BOUNDARY,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}


__all__ = [
    "QUERY_ESTIMABILITY_CERTIFICATE_CLAIM_BOUNDARY",
    "QUERY_ESTIMABILITY_CERTIFICATE_SCHEMA",
    "QUERY_ESTIMABILITY_CERTIFICATE_SEMANTICS",
    "QUERY_ESTIMABILITY_CERTIFICATE_VERSION",
    "QueryEstimabilityCertificateV1",
    "QueryEstimabilityStatus",
]
