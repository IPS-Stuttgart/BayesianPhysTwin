"""Transport only what every adequate cause explanation agrees on.

A registered cause family may explain the observed residual while retaining an
affine coefficient ambiguity. This module asks whether a held-intervention
physical query is invariant over that ambiguity set. It never converts one
minimum-norm coefficient representative into a unique physical-cause claim.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import Any, Final

import numpy as np

from .interventional_cause_adequacy_v1 import (
    CauseFamilyAdequacyStatus,
    InterventionalCauseFamilyAdequacyV1,
)

TRANSPORT_QUOTIENT_SCHEMA: Final = "bayesian_phystwin.interventional_transport_quotient"
TRANSPORT_QUOTIENT_VERSION: Final = 1
TRANSPORT_QUOTIENT_SEMANTICS: Final = (
    "held-intervention-query-invariance-over-cause-affine-set-v1"
)
TRANSPORT_QUOTIENT_CLAIM_BOUNDARY: Final = (
    "A passing target record establishes only local linear invariance of the "
    "registered held-intervention query over the affine coefficient solution set "
    "of one adequate registered cause family. A partial record identifies only "
    "the explicitly projected target-output subspace. It does not identify a "
    "unique physical cause, prove the signature or target map correct, establish "
    "nonlinear closure, held-out empirical transport, unseen-object transfer, "
    "deployment safety, or state of the art."
)


class TransportQuotientStatus(str, Enum):
    """Identifiability of one target query over the cause ambiguity set."""

    FAMILY_INADEQUATE = "family_inadequate"
    NO_DETECTABLE_ERROR = "no_detectable_error"
    FULLY_IDENTIFIABLE = "fully_identifiable"
    PARTIALLY_IDENTIFIABLE = "partially_identifiable"
    NONIDENTIFIABLE = "nonidentifiable"


def _digest(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character lowercase hex digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a 64-character lowercase hex digest")
    return value


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative real number")
    return result


def _matrix(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    matrix = np.ascontiguousarray(raw, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    if float(np.linalg.norm(matrix, ord="fro")) == 0.0:
        raise ValueError(f"{name} must contain a nontrivial target query")
    return matrix


def _immutable(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float64).reshape(
        contiguous.shape
    )


def _array_record(value: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def _canonical_id(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rank_tolerance(
    singular_values: np.ndarray,
    *,
    relative: float,
    absolute: float,
) -> float:
    scale = float(singular_values[0]) if singular_values.size else 0.0
    return max(absolute, relative * scale)


def _stable_pseudoinverse(
    matrix: np.ndarray,
    *,
    rank: int,
) -> np.ndarray:
    left, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
    if rank == 0:
        return np.zeros((matrix.shape[1], matrix.shape[0]), dtype=np.float64)
    return right[:rank, :].T @ np.diag(1.0 / singular_values[:rank]) @ left[:, :rank].T


@dataclass(frozen=True, slots=True)
class TargetTransportQuotientV1:
    """One held-intervention query projected onto its identifiable output space."""

    target_id: str
    target_transport_id: str
    status: TransportQuotientStatus
    target_dimension: int
    identifiable_dimension: int
    ambiguity_dimension: int
    identifiable_energy_fraction: float
    ambiguity_spectral_norm: float
    representative_invariance_residual: float
    stability_gain: float
    noise_error_bound: float
    full_transport_permitted: bool
    partial_transport_available: bool
    representative_effect: np.ndarray
    identifiable_effect: np.ndarray
    ambiguity_map: np.ndarray
    identifiable_projector: np.ndarray
    ambiguity_projector: np.ndarray
    transport_operator: np.ndarray

    def to_record(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "target_transport_id": self.target_transport_id,
            "status": self.status.value,
            "target_dimension": self.target_dimension,
            "identifiable_dimension": self.identifiable_dimension,
            "ambiguity_dimension": self.ambiguity_dimension,
            "identifiable_energy_fraction": self.identifiable_energy_fraction,
            "ambiguity_spectral_norm": self.ambiguity_spectral_norm,
            "representative_invariance_residual": (
                self.representative_invariance_residual
            ),
            "stability_gain": self.stability_gain,
            "noise_error_bound": self.noise_error_bound,
            "full_transport_permitted": self.full_transport_permitted,
            "partial_transport_available": self.partial_transport_available,
            "representative_effect": _array_record(self.representative_effect),
            "identifiable_effect": _array_record(self.identifiable_effect),
            "ambiguity_map": _array_record(self.ambiguity_map),
            "identifiable_projector": _array_record(self.identifiable_projector),
            "ambiguity_projector": _array_record(self.ambiguity_projector),
            "transport_operator": _array_record(self.transport_operator),
        }


@dataclass(frozen=True, slots=True)
class InterventionalTransportQuotientV1:
    """Held-intervention query certificate over one cause-attribution set."""

    adequacy_certificate: InterventionalCauseFamilyAdequacyV1
    target_intervention_roster_id: str
    target_transport_ids: Mapping[str, str]
    target_maps: Mapping[str, np.ndarray]
    relative_rank_tolerance: float = 1e-10
    absolute_rank_tolerance: float = 1e-12
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    target_order: tuple[str, ...] = field(init=False)
    target_records: tuple[TargetTransportQuotientV1, ...] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(
            self.adequacy_certificate,
            InterventionalCauseFamilyAdequacyV1,
        ):
            raise TypeError(
                "adequacy_certificate must be an InterventionalCauseFamilyAdequacyV1"
            )
        _digest(
            self.adequacy_certificate.artifact_id,
            name="adequacy_certificate.artifact_id",
        )
        target_roster = _digest(
            self.target_intervention_roster_id,
            name="target_intervention_roster_id",
        )
        if not isinstance(self.target_maps, Mapping):
            raise TypeError("target_maps must be a mapping")
        if not isinstance(self.target_transport_ids, Mapping):
            raise TypeError("target_transport_ids must be a mapping")
        targets = tuple(sorted(self.target_maps))
        if not targets or any(
            type(target) is not str or not target for target in targets
        ):
            raise ValueError("target_maps must use nonempty literal string keys")
        if set(self.target_transport_ids) != set(targets):
            raise ValueError("target_transport_ids must cover exactly the targets")

        relative = _finite_nonnegative(
            self.relative_rank_tolerance,
            name="relative_rank_tolerance",
        )
        absolute = _finite_nonnegative(
            self.absolute_rank_tolerance,
            name="absolute_rank_tolerance",
        )
        if relative == 0.0 and absolute == 0.0:
            raise ValueError("at least one rank tolerance must be positive")

        target_maps: dict[str, np.ndarray] = {}
        target_ids: dict[str, str] = {}
        for target in targets:
            matrix = _matrix(self.target_maps[target], name=f"target_maps[{target!r}]")
            if matrix.shape[1] != self.adequacy_certificate.coefficient_dimension:
                raise ValueError(
                    "every target map must have one column per cause coefficient"
                )
            target_maps[target] = matrix
            target_ids[target] = _digest(
                self.target_transport_ids[target],
                name=f"target_transport_ids[{target!r}]",
            )

        total = self.adequacy_certificate.total_design
        coefficients = self.adequacy_certificate.minimum_norm_coefficients
        nullspace = self.adequacy_certificate.coefficient_nullspace
        pseudoinverse = _stable_pseudoinverse(
            total,
            rank=self.adequacy_certificate.design_rank,
        )
        records: list[TargetTransportQuotientV1] = []
        immutable_maps: dict[str, np.ndarray] = {}
        for target in targets:
            target_map = target_maps[target]
            target_dimension = target_map.shape[0]
            representative = target_map @ coefficients

            if self.adequacy_certificate.status is (
                CauseFamilyAdequacyStatus.NO_DETECTABLE_ERROR
            ):
                status = TransportQuotientStatus.NO_DETECTABLE_ERROR
                ambiguity_rank = target_dimension
                identifiable_projector = np.zeros(
                    (target_dimension, target_dimension),
                    dtype=np.float64,
                )
                ambiguity_projector = np.eye(target_dimension)
                ambiguity_map = target_map @ nullspace
            elif not self.adequacy_certificate.family_adequate:
                status = TransportQuotientStatus.FAMILY_INADEQUATE
                ambiguity_rank = target_dimension
                identifiable_projector = np.zeros(
                    (target_dimension, target_dimension),
                    dtype=np.float64,
                )
                ambiguity_projector = np.eye(target_dimension)
                ambiguity_map = target_map @ nullspace
            else:
                ambiguity_map = target_map @ nullspace
                left, singular_values, _ = np.linalg.svd(
                    ambiguity_map,
                    full_matrices=False,
                )
                tolerance = _rank_tolerance(
                    singular_values,
                    relative=relative,
                    absolute=absolute,
                )
                ambiguity_rank = int(np.count_nonzero(singular_values > tolerance))
                ambiguity_basis = left[:, :ambiguity_rank]
                ambiguity_projector = ambiguity_basis @ ambiguity_basis.T
                identifiable_projector = np.eye(target_dimension) - ambiguity_projector
                if ambiguity_rank == 0:
                    status = TransportQuotientStatus.FULLY_IDENTIFIABLE
                elif ambiguity_rank < target_dimension:
                    status = TransportQuotientStatus.PARTIALLY_IDENTIFIABLE
                else:
                    status = TransportQuotientStatus.NONIDENTIFIABLE

            identifiable_map = identifiable_projector @ target_map
            identifiable_effect = identifiable_projector @ representative
            transport_operator = identifiable_map @ pseudoinverse
            invariance_residual = float(
                np.linalg.norm(identifiable_map @ nullspace, ord="fro")
            )
            ambiguity_singular = np.linalg.svd(
                ambiguity_map,
                compute_uv=False,
            )
            ambiguity_spectral = (
                float(ambiguity_singular[0]) if ambiguity_singular.size else 0.0
            )
            stability_gain = float(np.linalg.norm(transport_operator, ord=2))
            noise_error_bound = stability_gain * self.adequacy_certificate.noise_radius
            target_energy = float(np.linalg.norm(target_map, ord="fro") ** 2)
            identifiable_energy = float(
                np.linalg.norm(identifiable_map, ord="fro") ** 2
            )
            identifiable_fraction = float(
                np.clip(identifiable_energy / target_energy, 0.0, 1.0)
            )
            identifiable_dimension = target_dimension - ambiguity_rank
            full_permitted = status is TransportQuotientStatus.FULLY_IDENTIFIABLE
            partial_available = status in {
                TransportQuotientStatus.FULLY_IDENTIFIABLE,
                TransportQuotientStatus.PARTIALLY_IDENTIFIABLE,
            }
            records.append(
                TargetTransportQuotientV1(
                    target_id=target,
                    target_transport_id=target_ids[target],
                    status=status,
                    target_dimension=target_dimension,
                    identifiable_dimension=identifiable_dimension,
                    ambiguity_dimension=ambiguity_rank,
                    identifiable_energy_fraction=identifiable_fraction,
                    ambiguity_spectral_norm=ambiguity_spectral,
                    representative_invariance_residual=invariance_residual,
                    stability_gain=stability_gain,
                    noise_error_bound=noise_error_bound,
                    full_transport_permitted=full_permitted,
                    partial_transport_available=partial_available,
                    representative_effect=_immutable(representative),
                    identifiable_effect=_immutable(identifiable_effect),
                    ambiguity_map=_immutable(ambiguity_map),
                    identifiable_projector=_immutable(identifiable_projector),
                    ambiguity_projector=_immutable(ambiguity_projector),
                    transport_operator=_immutable(transport_operator),
                )
            )
            immutable_maps[target] = _immutable(target_map)

        metadata = json.loads(
            json.dumps(self.metadata, sort_keys=True, allow_nan=False)
        )
        object.__setattr__(self, "target_intervention_roster_id", target_roster)
        object.__setattr__(self, "target_transport_ids", target_ids)
        object.__setattr__(self, "target_maps", immutable_maps)
        object.__setattr__(self, "relative_rank_tolerance", relative)
        object.__setattr__(self, "absolute_rank_tolerance", absolute)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "target_order", targets)
        object.__setattr__(self, "target_records", tuple(records))

        expected = _canonical_id(self.descriptor())
        supplied = self.artifact_id
        if supplied is not None:
            supplied = _digest(supplied, name="artifact_id")
            if supplied != expected:
                raise ValueError("artifact_id does not match certificate content")
        object.__setattr__(self, "artifact_id", expected)

    def record_for(self, target_id: str) -> TargetTransportQuotientV1:
        for record in self.target_records:
            if record.target_id == target_id:
                return record
        raise KeyError(target_id)

    def arrays(self) -> Mapping[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for target, target_map in self.target_maps.items():
            result[f"target_map::{target}"] = target_map
        for record in self.target_records:
            prefix = record.target_id
            result[f"representative_effect::{prefix}"] = record.representative_effect
            result[f"identifiable_effect::{prefix}"] = record.identifiable_effect
            result[f"ambiguity_map::{prefix}"] = record.ambiguity_map
            result[f"identifiable_projector::{prefix}"] = record.identifiable_projector
            result[f"ambiguity_projector::{prefix}"] = record.ambiguity_projector
            result[f"transport_operator::{prefix}"] = record.transport_operator
        return result

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": TRANSPORT_QUOTIENT_SCHEMA,
            "schema_version": TRANSPORT_QUOTIENT_VERSION,
            "semantics": TRANSPORT_QUOTIENT_SEMANTICS,
            "adequacy_certificate_id": self.adequacy_certificate.artifact_id,
            "target_intervention_roster_id": (self.target_intervention_roster_id),
            "target_order": list(self.target_order),
            "target_transport_ids": dict(self.target_transport_ids),
            "target_maps": {
                target: _array_record(self.target_maps[target])
                for target in self.target_order
            },
            "relative_rank_tolerance": self.relative_rank_tolerance,
            "absolute_rank_tolerance": self.absolute_rank_tolerance,
            "target_records": [record.to_record() for record in self.target_records],
            "metadata": self.metadata,
            "claim_boundary": TRANSPORT_QUOTIENT_CLAIM_BOUNDARY,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}
