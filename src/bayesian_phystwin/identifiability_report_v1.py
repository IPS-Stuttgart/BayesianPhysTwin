"""Content-addressed reporting for physical-state identifiability.

The report records which physically reachable response modes remain distinguishable
from a declared observation-bias subspace. It does not reinterpret predictive
readout discrepancy as latent physical state and does not itself authorize a
Bayesian update.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
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
from .bias_aware_belief import IdentifiableStateBasis, PhysicalResponseBasis

IDENTIFIABILITY_REPORT_SCHEMA: Final = "bayesian_phystwin.identifiability_report"
IDENTIFIABILITY_REPORT_VERSION: Final = 1
IDENTIFIABILITY_REPORT_SEMANTICS: Final = (
    "reachable-state-modes-distinguishable-from-declared-observation-bias-v1"
)
IDENTIFIABILITY_REPORT_CLAIM_BOUNDARY: Final = (
    "Subspace distinguishability under the supplied physical response, query, "
    "observation mapping, and bias design only. The report does not establish a "
    "unique physical cause, provider competence, calibration, safe deployment, "
    "or Causal4D benefit."
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


def _finite_scalar(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_exclusive: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    if minimum is not None:
        invalid = result <= minimum if minimum_exclusive else result < minimum
        if invalid:
            relation = ">" if minimum_exclusive else ">="
            raise ValueError(f"{name} must be {relation} {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def _array_record(value: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


@dataclass(frozen=True, slots=True)
class IdentifiabilityReportV1:
    """Immutable audit of one registered physical-query identifiability boundary."""

    physical_response_id: str
    observation_mapping_id: str
    bias_design_id: str
    query_id: str
    physical_singular_values_m: np.ndarray
    coefficient_transform: np.ndarray
    identifiable_fractions: np.ndarray
    supported_point_count: int
    maximum_response_m: float
    explained_energy_fraction: float
    minimum_identifiable_fraction_required: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "physical_response_id",
            "observation_mapping_id",
            "bias_design_id",
            "query_id",
        ):
            object.__setattr__(
                self,
                name,
                literal_lower_hex(getattr(self, name), name=name, lengths={64}),
            )

        singular_values = _real_float64_array(
            self.physical_singular_values_m,
            name="physical_singular_values_m",
        )
        if singular_values.ndim != 1 or len(singular_values) == 0:
            raise ValueError("physical_singular_values_m must be a nonempty vector")
        if np.any(singular_values <= 0.0):
            raise ValueError("physical_singular_values_m must be positive")
        if np.any(singular_values[1:] > singular_values[:-1]):
            raise ValueError("physical_singular_values_m must be nonincreasing")

        transform = _real_float64_array(
            self.coefficient_transform,
            name="coefficient_transform",
        )
        if transform.ndim != 2 or transform.shape[0] != len(singular_values):
            raise ValueError(
                "coefficient_transform must have one row per physical response mode"
            )
        if transform.shape[1] == 0 or transform.shape[1] > transform.shape[0]:
            raise ValueError(
                "coefficient_transform must retain between one and all physical modes"
            )

        fractions = _real_float64_array(
            self.identifiable_fractions,
            name="identifiable_fractions",
        )
        if fractions.shape != (transform.shape[1],):
            raise ValueError(
                "identifiable_fractions must have one entry per retained mode"
            )
        if np.any((fractions <= 0.0) | (fractions > 1.0 + 1e-12)):
            raise ValueError("identifiable_fractions must lie in (0, 1]")

        minimum_required = _finite_scalar(
            self.minimum_identifiable_fraction_required,
            name="minimum_identifiable_fraction_required",
            minimum=0.0,
            maximum=1.0,
            minimum_exclusive=True,
        )
        if np.any(fractions < minimum_required - 1e-12):
            raise ValueError("identifiable_fractions contradict the registered minimum")
        supported_point_count = genuine_integer(
            self.supported_point_count,
            name="supported_point_count",
            minimum=1,
        )
        maximum_response = _finite_scalar(
            self.maximum_response_m,
            name="maximum_response_m",
            minimum=0.0,
            minimum_exclusive=True,
        )
        explained_energy = _finite_scalar(
            self.explained_energy_fraction,
            name="explained_energy_fraction",
            minimum=0.0,
            maximum=1.0,
            minimum_exclusive=True,
        )
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="identifiability report metadata",
        )

        object.__setattr__(
            self,
            "physical_singular_values_m",
            _immutable_float64(singular_values),
        )
        object.__setattr__(
            self,
            "coefficient_transform",
            _immutable_float64(transform),
        )
        object.__setattr__(
            self,
            "identifiable_fractions",
            _immutable_float64(np.minimum(fractions, 1.0)),
        )
        object.__setattr__(self, "supported_point_count", supported_point_count)
        object.__setattr__(self, "maximum_response_m", maximum_response)
        object.__setattr__(
            self,
            "explained_energy_fraction",
            explained_energy,
        )
        object.__setattr__(
            self,
            "minimum_identifiable_fraction_required",
            minimum_required,
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
                    "identifiability report artifact_id does not match content"
                )
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def physical_mode_count(self) -> int:
        return int(len(self.physical_singular_values_m))

    @property
    def retained_mode_count(self) -> int:
        return int(self.coefficient_transform.shape[1])

    @property
    def discarded_mode_count(self) -> int:
        return self.physical_mode_count - self.retained_mode_count

    @property
    def retained_mode_fraction(self) -> float:
        return self.retained_mode_count / self.physical_mode_count

    @property
    def discarded_mode_fraction(self) -> float:
        return self.discarded_mode_count / self.physical_mode_count

    @property
    def state_bias_overlap_fractions(self) -> np.ndarray:
        overlap = np.sqrt(np.maximum(0.0, 1.0 - np.square(self.identifiable_fractions)))
        return _immutable_float64(overlap)

    @property
    def minimum_identifiable_fraction_observed(self) -> float:
        return float(np.min(self.identifiable_fractions))

    @property
    def mean_identifiable_fraction(self) -> float:
        return float(np.mean(self.identifiable_fractions))

    @property
    def maximum_state_bias_overlap(self) -> float:
        return float(np.max(self.state_bias_overlap_fractions))

    def arrays(self) -> Mapping[str, np.ndarray]:
        """Return the immutable numerical arrays bound by this report."""

        return {
            "physical_singular_values_m": self.physical_singular_values_m,
            "coefficient_transform": self.coefficient_transform,
            "identifiable_fractions": self.identifiable_fractions,
        }

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": IDENTIFIABILITY_REPORT_SCHEMA,
            "schema_version": IDENTIFIABILITY_REPORT_VERSION,
            "semantics": IDENTIFIABILITY_REPORT_SEMANTICS,
            "physical_response_id": self.physical_response_id,
            "observation_mapping_id": self.observation_mapping_id,
            "bias_design_id": self.bias_design_id,
            "query_id": self.query_id,
            "physical_singular_values_m": _array_record(
                self.physical_singular_values_m
            ),
            "coefficient_transform": _array_record(self.coefficient_transform),
            "identifiable_fractions": _array_record(self.identifiable_fractions),
            "supported_point_count": self.supported_point_count,
            "maximum_response_m": self.maximum_response_m,
            "explained_energy_fraction": self.explained_energy_fraction,
            "minimum_identifiable_fraction_required": (
                self.minimum_identifiable_fraction_required
            ),
            "metadata": plain_json(self.metadata),
            "claim_boundary": IDENTIFIABILITY_REPORT_CLAIM_BOUNDARY,
        }

    def summary(self) -> dict[str, object]:
        return {
            "schema": IDENTIFIABILITY_REPORT_SCHEMA,
            "schema_version": IDENTIFIABILITY_REPORT_VERSION,
            "artifact_id": self.artifact_id,
            "physical_mode_count": self.physical_mode_count,
            "retained_mode_count": self.retained_mode_count,
            "discarded_mode_count": self.discarded_mode_count,
            "retained_mode_fraction": self.retained_mode_fraction,
            "discarded_mode_fraction": self.discarded_mode_fraction,
            "minimum_identifiable_fraction_required": (
                self.minimum_identifiable_fraction_required
            ),
            "minimum_identifiable_fraction_observed": (
                self.minimum_identifiable_fraction_observed
            ),
            "mean_identifiable_fraction": self.mean_identifiable_fraction,
            "maximum_state_bias_overlap": self.maximum_state_bias_overlap,
            "supported_point_count": self.supported_point_count,
            "maximum_response_m": self.maximum_response_m,
            "explained_energy_fraction": self.explained_energy_fraction,
            "claim_boundary": IDENTIFIABILITY_REPORT_CLAIM_BOUNDARY,
        }

    def to_record(self) -> dict[str, object]:
        return {
            **self.descriptor(),
            "artifact_id": self.artifact_id,
            "derived": {
                "physical_mode_count": self.physical_mode_count,
                "retained_mode_count": self.retained_mode_count,
                "discarded_mode_count": self.discarded_mode_count,
                "retained_mode_fraction": self.retained_mode_fraction,
                "discarded_mode_fraction": self.discarded_mode_fraction,
                "state_bias_overlap_fractions": (
                    self.state_bias_overlap_fractions.tolist()
                ),
                "minimum_identifiable_fraction_observed": (
                    self.minimum_identifiable_fraction_observed
                ),
                "mean_identifiable_fraction": self.mean_identifiable_fraction,
                "maximum_state_bias_overlap": self.maximum_state_bias_overlap,
            },
        }


def identifiability_report_from_bases(
    physical_response: PhysicalResponseBasis,
    identifiable_basis: IdentifiableStateBasis,
    *,
    physical_response_id: str,
    observation_mapping_id: str,
    bias_design_id: str,
    query_id: str,
    minimum_identifiable_fraction_required: float,
    metadata: Mapping[str, Any] | None = None,
) -> IdentifiabilityReportV1:
    """Build a report from the existing physical-response basis contracts."""

    if not isinstance(physical_response, PhysicalResponseBasis):
        raise TypeError("physical_response must be a PhysicalResponseBasis")
    if not isinstance(identifiable_basis, IdentifiableStateBasis):
        raise TypeError("identifiable_basis must be an IdentifiableStateBasis")
    physical_mode_count = physical_response.basis.shape[1]
    if identifiable_basis.coefficient_transform.shape[0] != physical_mode_count:
        raise ValueError(
            "identifiable basis coefficient transform does not match physical modes"
        )
    return IdentifiabilityReportV1(
        physical_response_id=physical_response_id,
        observation_mapping_id=observation_mapping_id,
        bias_design_id=bias_design_id,
        query_id=query_id,
        physical_singular_values_m=physical_response.singular_values_m,
        coefficient_transform=identifiable_basis.coefficient_transform,
        identifiable_fractions=identifiable_basis.identifiable_fractions,
        supported_point_count=physical_response.supported_point_count,
        maximum_response_m=physical_response.maximum_response_m,
        explained_energy_fraction=physical_response.explained_energy_fraction,
        minimum_identifiable_fraction_required=(minimum_identifiable_fraction_required),
        metadata={} if metadata is None else metadata,
    )


__all__ = [
    "IDENTIFIABILITY_REPORT_CLAIM_BOUNDARY",
    "IDENTIFIABILITY_REPORT_SCHEMA",
    "IDENTIFIABILITY_REPORT_SEMANTICS",
    "IDENTIFIABILITY_REPORT_VERSION",
    "IdentifiabilityReportV1",
    "identifiability_report_from_bases",
]
