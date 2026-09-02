"""Interventional identifiability of competing physical-twin error causes.

For an action-stacked local model

    r = sum_c S_c beta_c + N nu + epsilon,

this module tests one registered cause-specific query ``B_c beta_c`` at a time.
All other registered cause signatures are treated as nuisance directions.  The
query is attributable to cause ``c`` exactly when it is identifiable from
``S_c`` after projecting out both the declared nuisance ``N`` and every competing
cause signature.

The implementation is an experimental evidence instrument.  It does not infer a
cause family from unrestricted data and does not prove that the supplied response
signatures are physically correct.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Final

import numpy as np

from bayesian_phystwin.query_identifiability_certificate_v2 import (
    QueryIdentifiabilityCertificateV2,
    QueryIdentifiabilityStatus,
)

INTERVENTIONAL_CAUSE_IDENTIFIABILITY_SCHEMA: Final = (
    "bayesian_phystwin.interventional_cause_identifiability"
)
INTERVENTIONAL_CAUSE_IDENTIFIABILITY_VERSION: Final = 1
INTERVENTIONAL_CAUSE_IDENTIFIABILITY_SEMANTICS: Final = (
    "cause-query-identifiability-after-projecting-competing-causes-v1"
)
INTERVENTIONAL_CAUSE_IDENTIFIABILITY_CLAIM_BOUNDARY: Final = (
    "A passing result establishes local linear identifiability of one registered "
    "cause-specific query under the exact supplied intervention-response "
    "signatures, declared nuisance design, whitening, coordinates, and numerical "
    "tolerances. It does not prove that the cause family is complete, that a "
    "selected cause is the unique data-generating mechanism, that the response "
    "models are correct, or that the result transfers to unseen objects, actions, "
    "providers, nonlinear regimes, control deployment, or safety-critical use."
)


class CauseAttributionStatus(str, Enum):
    """Interpretation of one cause-specific query certificate."""

    IDENTIFIABLE = "identifiable"
    PARTIALLY_IDENTIFIABLE = "partially_identifiable"
    CONFOUNDED = "confounded"
    TRIVIAL_QUERY = "trivial_query"


def _literal_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty literal string")
    return value


def _digest(value: object, *, name: str) -> str:
    result = _literal_string(value, name=name)
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{name} must be a 64-character lowercase hexadecimal digest")
    return result


def _matrix(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _immutable(value: np.ndarray) -> np.ndarray:
    canonical = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(canonical.tobytes(order="C"), dtype=np.float64).reshape(
        canonical.shape
    )


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative real number")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _frozen_finite_json(value: object, *, name: str) -> object:
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
            output[key] = _frozen_finite_json(item, name=name)
        return MappingProxyType(output)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_frozen_finite_json(item, name=name) for item in value)
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
    canonical = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _orthonormal_basis(
    design: np.ndarray,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> np.ndarray:
    if design.shape[1] == 0:
        return np.empty((design.shape[0], 0), dtype=np.float64)
    left, singular_values, _ = np.linalg.svd(design, full_matrices=False)
    scale = float(singular_values[0]) if len(singular_values) else 0.0
    tolerance = max(absolute_tolerance, relative_tolerance * scale)
    rank = int(np.count_nonzero(singular_values > tolerance))
    return left[:, :rank]


def _project_out(
    design: np.ndarray,
    nuisance: np.ndarray,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> np.ndarray:
    basis = _orthonormal_basis(
        nuisance,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    if basis.shape[1] == 0:
        return design.copy()
    return design - basis @ (basis.T @ design)


@dataclass(frozen=True, slots=True)
class InterventionResponseBlockV1:
    """One cause response signature under one registered intervention."""

    intervention_id: str
    response_signature_id: str
    whitened_response_signature: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        intervention_id = _literal_string(
            self.intervention_id,
            name="intervention_id",
        )
        response_signature_id = _digest(
            self.response_signature_id,
            name="response_signature_id",
        )
        response = _matrix(
            self.whitened_response_signature,
            name="whitened_response_signature",
        )
        if response.shape[0] == 0 or response.shape[1] == 0:
            raise ValueError("whitened_response_signature must have nonzero dimensions")
        metadata = _frozen_finite_json(self.metadata, name="response block metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("response block metadata must be a mapping")
        object.__setattr__(self, "intervention_id", intervention_id)
        object.__setattr__(self, "response_signature_id", response_signature_id)
        object.__setattr__(self, "whitened_response_signature", _immutable(response))
        object.__setattr__(self, "metadata", metadata)
        expected = _content_id(self.descriptor())
        if self.artifact_id is not None:
            supplied = _digest(self.artifact_id, name="artifact_id")
            if supplied != expected:
                raise ValueError("response block artifact_id does not match content")
        object.__setattr__(self, "artifact_id", expected)

    @property
    def observation_dimension(self) -> int:
        return int(self.whitened_response_signature.shape[0])

    @property
    def latent_dimension(self) -> int:
        return int(self.whitened_response_signature.shape[1])

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": INTERVENTIONAL_CAUSE_IDENTIFIABILITY_SCHEMA,
            "schema_version": INTERVENTIONAL_CAUSE_IDENTIFIABILITY_VERSION,
            "artifact_kind": "InterventionResponseBlockV1",
            "semantics": INTERVENTIONAL_CAUSE_IDENTIFIABILITY_SEMANTICS,
            "intervention_id": self.intervention_id,
            "response_signature_id": self.response_signature_id,
            "whitened_response_signature": _array_record(
                self.whitened_response_signature
            ),
            "metadata": _plain_json(self.metadata),
            "claim_boundary": INTERVENTIONAL_CAUSE_IDENTIFIABILITY_CLAIM_BOUNDARY,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}


@dataclass(frozen=True, slots=True)
class CauseResponseSignatureV1:
    """Action-stacked response model and registered query for one cause."""

    cause_id: str
    latent_coordinates_id: str
    cause_query_id: str
    intervention_blocks: Sequence[InterventionResponseBlockV1]
    cause_query_map: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    stacked_response_signature: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        cause_id = _literal_string(self.cause_id, name="cause_id")
        latent_coordinates_id = _digest(
            self.latent_coordinates_id,
            name="latent_coordinates_id",
        )
        cause_query_id = _digest(self.cause_query_id, name="cause_query_id")
        if isinstance(self.intervention_blocks, (str, bytes)):
            raise TypeError("intervention_blocks must be a sequence")
        blocks = tuple(self.intervention_blocks)
        if len(blocks) < 2:
            raise ValueError("at least two intervention blocks are required")
        if any(not isinstance(block, InterventionResponseBlockV1) for block in blocks):
            raise TypeError(
                "intervention_blocks must contain InterventionResponseBlockV1 values"
            )
        intervention_ids = tuple(block.intervention_id for block in blocks)
        if intervention_ids != tuple(sorted(intervention_ids)):
            raise ValueError("intervention_blocks must be sorted by intervention_id")
        if len(intervention_ids) != len(set(intervention_ids)):
            raise ValueError(
                "intervention_blocks must have unique intervention_id values"
            )
        latent_dimension = blocks[0].latent_dimension
        if any(block.latent_dimension != latent_dimension for block in blocks):
            raise ValueError("all intervention blocks must share the latent dimension")
        query = _matrix(self.cause_query_map, name="cause_query_map")
        if query.shape[0] == 0 or query.shape[1] != latent_dimension:
            raise ValueError(
                "cause_query_map must have nonzero rows and one column per cause "
                "coordinate"
            )
        stacked = np.vstack([block.whitened_response_signature for block in blocks])
        metadata = _frozen_finite_json(self.metadata, name="cause signature metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("cause signature metadata must be a mapping")
        object.__setattr__(self, "cause_id", cause_id)
        object.__setattr__(self, "latent_coordinates_id", latent_coordinates_id)
        object.__setattr__(self, "cause_query_id", cause_query_id)
        object.__setattr__(self, "intervention_blocks", blocks)
        object.__setattr__(self, "cause_query_map", _immutable(query))
        object.__setattr__(self, "stacked_response_signature", _immutable(stacked))
        object.__setattr__(self, "metadata", metadata)
        expected = _content_id(self.descriptor())
        if self.artifact_id is not None:
            supplied = _digest(self.artifact_id, name="artifact_id")
            if supplied != expected:
                raise ValueError("cause signature artifact_id does not match content")
        object.__setattr__(self, "artifact_id", expected)

    @property
    def intervention_ids(self) -> tuple[str, ...]:
        return tuple(block.intervention_id for block in self.intervention_blocks)

    @property
    def observation_dimensions(self) -> tuple[int, ...]:
        return tuple(block.observation_dimension for block in self.intervention_blocks)

    @property
    def latent_dimension(self) -> int:
        return int(self.cause_query_map.shape[1])

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": INTERVENTIONAL_CAUSE_IDENTIFIABILITY_SCHEMA,
            "schema_version": INTERVENTIONAL_CAUSE_IDENTIFIABILITY_VERSION,
            "artifact_kind": "CauseResponseSignatureV1",
            "semantics": INTERVENTIONAL_CAUSE_IDENTIFIABILITY_SEMANTICS,
            "cause_id": self.cause_id,
            "latent_coordinates_id": self.latent_coordinates_id,
            "cause_query_id": self.cause_query_id,
            "intervention_block_ids": [
                block.artifact_id for block in self.intervention_blocks
            ],
            "cause_query_map": _array_record(self.cause_query_map),
            "stacked_response_signature": _array_record(
                self.stacked_response_signature
            ),
            "metadata": _plain_json(self.metadata),
            "claim_boundary": INTERVENTIONAL_CAUSE_IDENTIFIABILITY_CLAIM_BOUNDARY,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}


@dataclass(frozen=True, slots=True)
class InterventionContributionV1:
    """Effect of removing one intervention from a cause certificate."""

    intervention_id: str
    without_intervention_status: QueryIdentifiabilityStatus
    without_intervention_energy_fraction: float
    energy_fraction_loss: float

    def to_record(self) -> dict[str, object]:
        return {
            "intervention_id": self.intervention_id,
            "without_intervention_status": self.without_intervention_status.value,
            "without_intervention_energy_fraction": (
                self.without_intervention_energy_fraction
            ),
            "energy_fraction_loss": self.energy_fraction_loss,
        }


@dataclass(frozen=True, slots=True)
class PairwiseCauseCoherenceV1:
    """Subspace overlap after removing only the globally declared nuisance."""

    left_cause_id: str
    right_cause_id: str
    maximum_canonical_correlation: float
    minimum_principal_angle_degrees: float
    left_residualized_rank: int
    right_residualized_rank: int

    def to_record(self) -> dict[str, object]:
        return {
            "left_cause_id": self.left_cause_id,
            "right_cause_id": self.right_cause_id,
            "maximum_canonical_correlation": self.maximum_canonical_correlation,
            "minimum_principal_angle_degrees": self.minimum_principal_angle_degrees,
            "left_residualized_rank": self.left_residualized_rank,
            "right_residualized_rank": self.right_residualized_rank,
        }


@dataclass(frozen=True, slots=True)
class CauseAttributionResultV1:
    """Derived attribution diagnostics for one registered cause query."""

    cause_id: str
    status: CauseAttributionStatus
    query_certificate_id: str
    identifiable_query_energy_fraction: float
    normalized_factorization_residual: float
    residualized_cause_rank: int
    cause_latent_dimension: int
    full_cause_identifiable: bool
    requires_multiple_interventions: bool
    minimum_identifying_intervention_count: int | None
    minimal_identifying_intervention_sets: tuple[tuple[str, ...], ...]
    single_intervention_statuses: tuple[tuple[str, QueryIdentifiabilityStatus], ...]
    intervention_contributions: tuple[InterventionContributionV1, ...]
    maximum_competing_coherence: float
    minimum_competing_principal_angle_degrees: float

    def to_record(self) -> dict[str, object]:
        return {
            "cause_id": self.cause_id,
            "status": self.status.value,
            "query_certificate_id": self.query_certificate_id,
            "identifiable_query_energy_fraction": (
                self.identifiable_query_energy_fraction
            ),
            "normalized_factorization_residual": (
                self.normalized_factorization_residual
            ),
            "residualized_cause_rank": self.residualized_cause_rank,
            "cause_latent_dimension": self.cause_latent_dimension,
            "full_cause_identifiable": self.full_cause_identifiable,
            "requires_multiple_interventions": self.requires_multiple_interventions,
            "minimum_identifying_intervention_count": (
                self.minimum_identifying_intervention_count
            ),
            "minimal_identifying_intervention_sets": [
                list(values) for values in self.minimal_identifying_intervention_sets
            ],
            "single_intervention_statuses": [
                [intervention_id, status.value]
                for intervention_id, status in self.single_intervention_statuses
            ],
            "intervention_contributions": [
                item.to_record() for item in self.intervention_contributions
            ],
            "maximum_competing_coherence": self.maximum_competing_coherence,
            "minimum_competing_principal_angle_degrees": (
                self.minimum_competing_principal_angle_degrees
            ),
        }


@dataclass(frozen=True, slots=True)
class InterventionalCauseIdentifiabilityCertificateV1:
    """Test every registered cause query against all competing cause signatures."""

    observation_whitening_id: str
    declared_nuisance_id: str
    cause_family_id: str
    cause_signatures: Sequence[CauseResponseSignatureV1]
    joint_whitened_nuisance_design: np.ndarray
    relative_rank_tolerance: float = 1e-10
    absolute_rank_tolerance: float = 1e-12
    identifiability_tolerance: float = 1e-8
    maximum_exact_subset_actions: int = 10
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    intervention_ids: tuple[str, ...] = field(init=False)
    cause_results: tuple[CauseAttributionResultV1, ...] = field(init=False)
    pairwise_coherences: tuple[PairwiseCauseCoherenceV1, ...] = field(init=False)

    def __post_init__(self) -> None:
        observation_whitening_id = _digest(
            self.observation_whitening_id,
            name="observation_whitening_id",
        )
        declared_nuisance_id = _digest(
            self.declared_nuisance_id,
            name="declared_nuisance_id",
        )
        cause_family_id = _digest(self.cause_family_id, name="cause_family_id")
        if isinstance(self.cause_signatures, (str, bytes)):
            raise TypeError("cause_signatures must be a sequence")
        causes = tuple(self.cause_signatures)
        if len(causes) < 2:
            raise ValueError("at least two competing cause signatures are required")
        if any(not isinstance(cause, CauseResponseSignatureV1) for cause in causes):
            raise TypeError(
                "cause_signatures must contain CauseResponseSignatureV1 values"
            )
        cause_ids = tuple(cause.cause_id for cause in causes)
        if cause_ids != tuple(sorted(cause_ids)):
            raise ValueError("cause_signatures must be sorted by cause_id")
        if len(cause_ids) != len(set(cause_ids)):
            raise ValueError("cause_signatures must have unique cause_id values")
        intervention_ids = causes[0].intervention_ids
        observation_dimensions = causes[0].observation_dimensions
        for cause in causes[1:]:
            if cause.intervention_ids != intervention_ids:
                raise ValueError("all causes must share the intervention roster")
            if cause.observation_dimensions != observation_dimensions:
                raise ValueError(
                    "all causes must share each intervention observation dimension"
                )
        nuisance = _matrix(
            self.joint_whitened_nuisance_design,
            name="joint_whitened_nuisance_design",
        )
        total_rows = sum(observation_dimensions)
        if nuisance.shape[0] != total_rows:
            raise ValueError(
                "joint_whitened_nuisance_design must share the stacked row count"
            )
        relative = _finite_nonnegative(
            self.relative_rank_tolerance,
            name="relative_rank_tolerance",
        )
        absolute = _finite_nonnegative(
            self.absolute_rank_tolerance,
            name="absolute_rank_tolerance",
        )
        identifiability = _finite_nonnegative(
            self.identifiability_tolerance,
            name="identifiability_tolerance",
        )
        if relative == absolute == 0.0:
            raise ValueError("at least one rank tolerance must be positive")
        if identifiability == 0.0:
            raise ValueError("identifiability_tolerance must be positive")
        maximum_subset = _positive_integer(
            self.maximum_exact_subset_actions,
            name="maximum_exact_subset_actions",
        )
        metadata = _frozen_finite_json(self.metadata, name="cause certificate metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("cause certificate metadata must be a mapping")

        row_slices: dict[str, slice] = {}
        start = 0
        for intervention_id, row_count in zip(
            intervention_ids,
            observation_dimensions,
            strict=True,
        ):
            row_slices[intervention_id] = slice(start, start + row_count)
            start += row_count

        pairwise = self._pairwise_coherences(
            causes,
            nuisance,
            relative=relative,
            absolute=absolute,
        )
        coherence_by_cause: dict[str, list[PairwiseCauseCoherenceV1]] = {
            cause_id: [] for cause_id in cause_ids
        }
        for item in pairwise:
            coherence_by_cause[item.left_cause_id].append(item)
            coherence_by_cause[item.right_cause_id].append(item)

        results: list[CauseAttributionResultV1] = []
        for cause in causes:
            full = self._query_certificate(
                cause,
                causes,
                nuisance,
                intervention_ids,
                row_slices,
                observation_whitening_id=observation_whitening_id,
                declared_nuisance_id=declared_nuisance_id,
                relative=relative,
                absolute=absolute,
                identifiability=identifiability,
            )
            single_statuses: list[tuple[str, QueryIdentifiabilityStatus]] = []
            for intervention_id in intervention_ids:
                single = self._query_certificate(
                    cause,
                    causes,
                    nuisance,
                    (intervention_id,),
                    row_slices,
                    observation_whitening_id=observation_whitening_id,
                    declared_nuisance_id=declared_nuisance_id,
                    relative=relative,
                    absolute=absolute,
                    identifiability=identifiability,
                )
                single_statuses.append((intervention_id, single.status))

            contributions: list[InterventionContributionV1] = []
            for intervention_id in intervention_ids:
                remaining = tuple(
                    item for item in intervention_ids if item != intervention_id
                )
                without = self._query_certificate(
                    cause,
                    causes,
                    nuisance,
                    remaining,
                    row_slices,
                    observation_whitening_id=observation_whitening_id,
                    declared_nuisance_id=declared_nuisance_id,
                    relative=relative,
                    absolute=absolute,
                    identifiability=identifiability,
                )
                contributions.append(
                    InterventionContributionV1(
                        intervention_id=intervention_id,
                        without_intervention_status=without.status,
                        without_intervention_energy_fraction=(
                            without.identifiable_query_energy_fraction
                        ),
                        energy_fraction_loss=max(
                            0.0,
                            full.identifiable_query_energy_fraction
                            - without.identifiable_query_energy_fraction,
                        ),
                    )
                )

            minimal_sets = self._minimal_identifying_sets(
                cause,
                causes,
                nuisance,
                intervention_ids,
                row_slices,
                observation_whitening_id=observation_whitening_id,
                declared_nuisance_id=declared_nuisance_id,
                relative=relative,
                absolute=absolute,
                identifiability=identifiability,
                maximum_subset=maximum_subset,
            )
            minimum_count = len(minimal_sets[0]) if minimal_sets else None
            status = self._attribution_status(full)
            cause_pairwise = coherence_by_cause[cause.cause_id]
            maximum_coherence = max(
                (item.maximum_canonical_correlation for item in cause_pairwise),
                default=0.0,
            )
            minimum_angle = min(
                (item.minimum_principal_angle_degrees for item in cause_pairwise),
                default=90.0,
            )
            full_cause_identifiable = bool(
                full.status is QueryIdentifiabilityStatus.IDENTIFIABLE
                and full.query_rank == cause.latent_dimension
                and full.physical_rank == cause.latent_dimension
            )
            results.append(
                CauseAttributionResultV1(
                    cause_id=cause.cause_id,
                    status=status,
                    query_certificate_id=str(full.artifact_id),
                    identifiable_query_energy_fraction=(
                        full.identifiable_query_energy_fraction
                    ),
                    normalized_factorization_residual=(
                        full.normalized_factorization_residual
                    ),
                    residualized_cause_rank=full.physical_rank,
                    cause_latent_dimension=cause.latent_dimension,
                    full_cause_identifiable=full_cause_identifiable,
                    requires_multiple_interventions=(
                        minimum_count is not None and minimum_count > 1
                    ),
                    minimum_identifying_intervention_count=minimum_count,
                    minimal_identifying_intervention_sets=minimal_sets,
                    single_intervention_statuses=tuple(single_statuses),
                    intervention_contributions=tuple(contributions),
                    maximum_competing_coherence=maximum_coherence,
                    minimum_competing_principal_angle_degrees=minimum_angle,
                )
            )

        object.__setattr__(
            self,
            "observation_whitening_id",
            observation_whitening_id,
        )
        object.__setattr__(self, "declared_nuisance_id", declared_nuisance_id)
        object.__setattr__(self, "cause_family_id", cause_family_id)
        object.__setattr__(self, "cause_signatures", causes)
        object.__setattr__(
            self,
            "joint_whitened_nuisance_design",
            _immutable(nuisance),
        )
        object.__setattr__(self, "relative_rank_tolerance", relative)
        object.__setattr__(self, "absolute_rank_tolerance", absolute)
        object.__setattr__(self, "identifiability_tolerance", identifiability)
        object.__setattr__(self, "maximum_exact_subset_actions", maximum_subset)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "intervention_ids", intervention_ids)
        object.__setattr__(self, "cause_results", tuple(results))
        object.__setattr__(self, "pairwise_coherences", pairwise)
        expected = _content_id(self.descriptor())
        if self.artifact_id is not None:
            supplied = _digest(self.artifact_id, name="artifact_id")
            if supplied != expected:
                raise ValueError(
                    "interventional cause certificate artifact_id does not "
                    "match content"
                )
        object.__setattr__(self, "artifact_id", expected)

    @staticmethod
    def _row_mask(
        interventions: Sequence[str],
        row_slices: Mapping[str, slice],
        row_count: int,
    ) -> np.ndarray:
        mask = np.zeros(row_count, dtype=bool)
        for intervention_id in interventions:
            mask[row_slices[intervention_id]] = True
        return mask

    @classmethod
    def _query_certificate(
        cls,
        cause: CauseResponseSignatureV1,
        causes: Sequence[CauseResponseSignatureV1],
        nuisance: np.ndarray,
        interventions: Sequence[str],
        row_slices: Mapping[str, slice],
        *,
        observation_whitening_id: str,
        declared_nuisance_id: str,
        relative: float,
        absolute: float,
        identifiability: float,
    ) -> QueryIdentifiabilityCertificateV2:
        mask = cls._row_mask(interventions, row_slices, nuisance.shape[0])
        physical = cause.stacked_response_signature[mask]
        competitors = [
            candidate.stacked_response_signature[mask]
            for candidate in causes
            if candidate.cause_id != cause.cause_id
        ]
        nuisance_columns = [nuisance[mask], *competitors]
        combined = np.hstack(nuisance_columns)
        physical_response_id = _content_id(
            {
                "kind": "cause-response-signature-v1",
                "cause_signature_id": cause.artifact_id,
                "intervention_ids": list(interventions),
            }
        )
        observation_mapping_id = _content_id(
            {
                "kind": "cause-observation-whitening-v1",
                "observation_whitening_id": observation_whitening_id,
                "intervention_ids": list(interventions),
            }
        )
        nuisance_design_id = _content_id(
            {
                "kind": "competing-cause-nuisance-v1",
                "declared_nuisance_id": declared_nuisance_id,
                "selected_cause_id": cause.cause_id,
                "competitor_ids": [
                    candidate.artifact_id
                    for candidate in causes
                    if candidate.cause_id != cause.cause_id
                ],
                "intervention_ids": list(interventions),
            }
        )
        return QueryIdentifiabilityCertificateV2(
            physical_response_id=physical_response_id,
            observation_mapping_id=observation_mapping_id,
            nuisance_design_id=nuisance_design_id,
            query_id=cause.cause_query_id,
            whitened_physical_design=physical,
            whitened_nuisance_design=combined,
            query_map=cause.cause_query_map,
            relative_rank_tolerance=relative,
            absolute_rank_tolerance=absolute,
            identifiability_tolerance=identifiability,
            metadata={
                "cause_id": cause.cause_id,
                "intervention_ids": list(interventions),
                "competitor_cause_ids": [
                    candidate.cause_id
                    for candidate in causes
                    if candidate.cause_id != cause.cause_id
                ],
                "semantics": INTERVENTIONAL_CAUSE_IDENTIFIABILITY_SEMANTICS,
            },
        )

    @classmethod
    def _minimal_identifying_sets(
        cls,
        cause: CauseResponseSignatureV1,
        causes: Sequence[CauseResponseSignatureV1],
        nuisance: np.ndarray,
        intervention_ids: tuple[str, ...],
        row_slices: Mapping[str, slice],
        *,
        observation_whitening_id: str,
        declared_nuisance_id: str,
        relative: float,
        absolute: float,
        identifiability: float,
        maximum_subset: int,
    ) -> tuple[tuple[str, ...], ...]:
        if len(intervention_ids) > maximum_subset:
            return ()
        for count in range(1, len(intervention_ids) + 1):
            identifying: list[tuple[str, ...]] = []
            for subset in itertools.combinations(intervention_ids, count):
                certificate = cls._query_certificate(
                    cause,
                    causes,
                    nuisance,
                    subset,
                    row_slices,
                    observation_whitening_id=observation_whitening_id,
                    declared_nuisance_id=declared_nuisance_id,
                    relative=relative,
                    absolute=absolute,
                    identifiability=identifiability,
                )
                if certificate.status is QueryIdentifiabilityStatus.IDENTIFIABLE:
                    identifying.append(tuple(subset))
            if identifying:
                return tuple(identifying)
        return ()

    @staticmethod
    def _attribution_status(
        certificate: QueryIdentifiabilityCertificateV2,
    ) -> CauseAttributionStatus:
        if certificate.status is QueryIdentifiabilityStatus.TRIVIAL_QUERY:
            return CauseAttributionStatus.TRIVIAL_QUERY
        if certificate.status is QueryIdentifiabilityStatus.IDENTIFIABLE:
            return CauseAttributionStatus.IDENTIFIABLE
        if certificate.identifiable_query_energy_fraction > 0.0:
            return CauseAttributionStatus.PARTIALLY_IDENTIFIABLE
        return CauseAttributionStatus.CONFOUNDED

    @staticmethod
    def _pairwise_coherences(
        causes: Sequence[CauseResponseSignatureV1],
        nuisance: np.ndarray,
        *,
        relative: float,
        absolute: float,
    ) -> tuple[PairwiseCauseCoherenceV1, ...]:
        residualized = {
            cause.cause_id: _project_out(
                cause.stacked_response_signature,
                nuisance,
                relative_tolerance=relative,
                absolute_tolerance=absolute,
            )
            for cause in causes
        }
        bases = {
            cause_id: _orthonormal_basis(
                design,
                relative_tolerance=relative,
                absolute_tolerance=absolute,
            )
            for cause_id, design in residualized.items()
        }
        output: list[PairwiseCauseCoherenceV1] = []
        for left, right in itertools.combinations(causes, 2):
            left_basis = bases[left.cause_id]
            right_basis = bases[right.cause_id]
            if left_basis.shape[1] and right_basis.shape[1]:
                singular_values = np.linalg.svd(
                    left_basis.T @ right_basis,
                    compute_uv=False,
                )
                coherence = float(np.clip(singular_values[0], 0.0, 1.0))
            else:
                coherence = 0.0
            angle = float(np.degrees(np.arccos(coherence)))
            output.append(
                PairwiseCauseCoherenceV1(
                    left_cause_id=left.cause_id,
                    right_cause_id=right.cause_id,
                    maximum_canonical_correlation=coherence,
                    minimum_principal_angle_degrees=angle,
                    left_residualized_rank=left_basis.shape[1],
                    right_residualized_rank=right_basis.shape[1],
                )
            )
        return tuple(output)

    @property
    def any_nontrivially_identifiable(self) -> bool:
        return any(
            result.status is CauseAttributionStatus.IDENTIFIABLE
            for result in self.cause_results
        )

    @property
    def all_nontrivially_identifiable(self) -> bool:
        return all(
            result.status is CauseAttributionStatus.IDENTIFIABLE
            for result in self.cause_results
        )

    def result_for(self, cause_id: str) -> CauseAttributionResultV1:
        matches = [
            result for result in self.cause_results if result.cause_id == cause_id
        ]
        if len(matches) != 1:
            raise KeyError(cause_id)
        return matches[0]

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": INTERVENTIONAL_CAUSE_IDENTIFIABILITY_SCHEMA,
            "schema_version": INTERVENTIONAL_CAUSE_IDENTIFIABILITY_VERSION,
            "artifact_kind": "InterventionalCauseIdentifiabilityCertificateV1",
            "semantics": INTERVENTIONAL_CAUSE_IDENTIFIABILITY_SEMANTICS,
            "observation_whitening_id": self.observation_whitening_id,
            "declared_nuisance_id": self.declared_nuisance_id,
            "cause_family_id": self.cause_family_id,
            "cause_signature_ids": [
                cause.artifact_id for cause in self.cause_signatures
            ],
            "joint_whitened_nuisance_design": _array_record(
                self.joint_whitened_nuisance_design
            ),
            "intervention_ids": list(self.intervention_ids),
            "relative_rank_tolerance": self.relative_rank_tolerance,
            "absolute_rank_tolerance": self.absolute_rank_tolerance,
            "identifiability_tolerance": self.identifiability_tolerance,
            "maximum_exact_subset_actions": self.maximum_exact_subset_actions,
            "cause_results": [result.to_record() for result in self.cause_results],
            "pairwise_coherences": [
                item.to_record() for item in self.pairwise_coherences
            ],
            "metadata": _plain_json(self.metadata),
            "claim_boundary": INTERVENTIONAL_CAUSE_IDENTIFIABILITY_CLAIM_BOUNDARY,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}

    def summary(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "intervention_ids": list(self.intervention_ids),
            "any_nontrivially_identifiable": self.any_nontrivially_identifiable,
            "all_nontrivially_identifiable": self.all_nontrivially_identifiable,
            "cause_results": [result.to_record() for result in self.cause_results],
            "claim_boundary": INTERVENTIONAL_CAUSE_IDENTIFIABILITY_CLAIM_BOUNDARY,
        }
