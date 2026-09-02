"""Fail-closed adequacy and set-valued attribution for registered causes.

Interventional cause identifiability is meaningful only when the registered
cause family can explain the observed residual. This module therefore checks
``distance(residual, span(signatures))`` before any cause interpretation is
allowed. When the family is adequate but the coefficients are not unique, the
result retains the complete affine solution set instead of forcing one cause.
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

CAUSE_FAMILY_ADEQUACY_SCHEMA: Final = (
    "bayesian_phystwin.interventional_cause_family_adequacy"
)
CAUSE_FAMILY_ADEQUACY_VERSION: Final = 1
CAUSE_FAMILY_ADEQUACY_SEMANTICS: Final = (
    "distance-to-registered-cause-span-with-set-valued-attribution-v1"
)
CAUSE_FAMILY_ADEQUACY_CLAIM_BOUNDARY: Final = (
    "A passing certificate establishes only that the supplied whitened residual "
    "is within the registered deterministic noise radius of the span of the exact "
    "registered intervention-response signatures. It reports the complete local "
    "linear solution set and cause-specific identifiable dimensions. It does not "
    "prove that the cause family is physically complete, that one cause generated "
    "the data, that the supplied signatures are correct, or that the result "
    "transfers to unseen objects, interventions, nonlinear regimes, or deployment."
)


class CauseFamilyAdequacyStatus(str, Enum):
    """Outcome of the residual-span and solution-uniqueness checks."""

    NO_DETECTABLE_ERROR = "no_detectable_error"
    UNMODELED_CAUSE = "unmodeled_cause"
    ADEQUATE_UNIQUE = "adequate_unique"
    ADEQUATE_SET_VALUED = "adequate_set_valued"


class CauseBlockStatus(str, Enum):
    """Identifiability of one coefficient block inside the complete solution set."""

    IDENTIFIABLE = "identifiable"
    PARTIALLY_IDENTIFIABLE = "partially_identifiable"
    CONFOUNDED = "confounded"


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative real number")
    return result


def _digest(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character lowercase hex digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a 64-character lowercase hex digest")
    return value


def _matrix(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    matrix = np.ascontiguousarray(raw, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    return matrix


def _vector(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    vector = np.ascontiguousarray(raw, dtype=np.float64)
    if vector.ndim == 2 and vector.shape[1] == 1:
        vector = vector[:, 0]
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    return vector


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


def _rank(
    matrix: np.ndarray,
    *,
    relative: float,
    absolute: float,
) -> tuple[int, np.ndarray, float]:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    tolerance = _rank_tolerance(
        singular_values,
        relative=relative,
        absolute=absolute,
    )
    return (
        int(np.count_nonzero(singular_values > tolerance)),
        singular_values,
        tolerance,
    )


def _orthogonal_projector(
    design: np.ndarray,
    *,
    relative: float,
    absolute: float,
) -> np.ndarray:
    if design.shape[1] == 0:
        return np.zeros((design.shape[0], design.shape[0]), dtype=np.float64)
    left, singular_values, _ = np.linalg.svd(design, full_matrices=False)
    tolerance = _rank_tolerance(
        singular_values,
        relative=relative,
        absolute=absolute,
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    basis = left[:, :rank]
    return basis @ basis.T


@dataclass(frozen=True, slots=True)
class CauseBlockAdequacyV1:
    """Cause-specific ambiguity retained from the complete affine solution set."""

    cause_id: str
    coefficient_start: int
    coefficient_stop: int
    dimension: int
    identifiable_dimension: int
    unresolved_dimension: int
    residualized_rank: int
    separation_margin: float
    coefficient_estimate: tuple[float, ...]
    status: CauseBlockStatus

    def to_record(self) -> dict[str, object]:
        return {
            "cause_id": self.cause_id,
            "coefficient_start": self.coefficient_start,
            "coefficient_stop": self.coefficient_stop,
            "dimension": self.dimension,
            "identifiable_dimension": self.identifiable_dimension,
            "unresolved_dimension": self.unresolved_dimension,
            "residualized_rank": self.residualized_rank,
            "separation_margin": self.separation_margin,
            "coefficient_estimate": list(self.coefficient_estimate),
            "status": self.status.value,
        }


@dataclass(frozen=True, slots=True)
class InterventionalCauseFamilyAdequacyV1:
    """Audit family adequacy before any interventional cause label is permitted."""

    residual_id: str
    intervention_roster_id: str
    whitening_id: str
    cause_signature_ids: Mapping[str, str]
    cause_signatures: Mapping[str, np.ndarray]
    whitened_residual: np.ndarray
    noise_radius: float
    relative_rank_tolerance: float = 1e-10
    absolute_rank_tolerance: float = 1e-12
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    cause_order: tuple[str, ...] = field(init=False)
    total_design: np.ndarray = field(init=False, repr=False)
    minimum_norm_coefficients: np.ndarray = field(init=False, repr=False)
    fitted_residual: np.ndarray = field(init=False, repr=False)
    unexplained_residual: np.ndarray = field(init=False, repr=False)
    coefficient_nullspace: np.ndarray = field(init=False, repr=False)
    singular_values: np.ndarray = field(init=False, repr=False)
    design_rank: int = field(init=False)
    coefficient_dimension: int = field(init=False)
    solution_nullity: int = field(init=False)
    residual_norm: float = field(init=False)
    unexplained_norm: float = field(init=False)
    explained_energy_fraction: float = field(init=False)
    smallest_nonzero_singular_value: float | None = field(init=False)
    identifiable_component_error_bound: float | None = field(init=False)
    status: CauseFamilyAdequacyStatus = field(init=False)
    cause_blocks: tuple[CauseBlockAdequacyV1, ...] = field(init=False)

    def __post_init__(self) -> None:
        for name in ("residual_id", "intervention_roster_id", "whitening_id"):
            object.__setattr__(
                self,
                name,
                _digest(getattr(self, name), name=name),
            )
        if not isinstance(self.cause_signatures, Mapping):
            raise TypeError("cause_signatures must be a mapping")
        if not isinstance(self.cause_signature_ids, Mapping):
            raise TypeError("cause_signature_ids must be a mapping")
        causes = tuple(sorted(self.cause_signatures))
        if not causes or any(type(cause) is not str or not cause for cause in causes):
            raise ValueError("cause_signatures must use nonempty literal string keys")
        if set(self.cause_signature_ids) != set(causes):
            raise ValueError("cause_signature_ids must cover exactly the cause family")

        signatures: dict[str, np.ndarray] = {}
        row_count: int | None = None
        for cause in causes:
            signature = _matrix(
                self.cause_signatures[cause],
                name=f"cause_signatures[{cause!r}]",
            )
            if row_count is None:
                row_count = signature.shape[0]
            elif signature.shape[0] != row_count:
                raise ValueError("all cause signatures must share the observation rows")
            signatures[cause] = signature
            _digest(
                self.cause_signature_ids[cause],
                name=f"cause_signature_ids[{cause!r}]",
            )
        residual = _vector(self.whitened_residual, name="whitened_residual")
        if row_count != residual.size:
            raise ValueError("whitened_residual must share the signature row count")

        noise_radius = _finite_nonnegative(self.noise_radius, name="noise_radius")
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

        total = np.hstack([signatures[cause] for cause in causes])
        left, singular_values, right = np.linalg.svd(total, full_matrices=True)
        tolerance = _rank_tolerance(
            singular_values,
            relative=relative,
            absolute=absolute,
        )
        rank = int(np.count_nonzero(singular_values > tolerance))
        coefficient_dimension = int(total.shape[1])
        if rank:
            coefficients = right[:rank, :].T @ (
                (left[:, :rank].T @ residual) / singular_values[:rank]
            )
            smallest_nonzero = float(singular_values[rank - 1])
            error_bound = noise_radius / smallest_nonzero
        else:
            coefficients = np.zeros(coefficient_dimension, dtype=np.float64)
            smallest_nonzero = None
            error_bound = None
        fitted = total @ coefficients
        unexplained = residual - fitted
        coefficient_nullspace = right[rank:, :].T
        residual_norm = float(np.linalg.norm(residual))
        unexplained_norm = float(np.linalg.norm(unexplained))
        explained_fraction = float(
            np.clip(
                1.0
                - unexplained_norm**2
                / max(residual_norm**2, float(np.finfo(np.float64).tiny)),
                0.0,
                1.0,
            )
        )
        if residual_norm <= noise_radius:
            status = CauseFamilyAdequacyStatus.NO_DETECTABLE_ERROR
        elif unexplained_norm > noise_radius:
            status = CauseFamilyAdequacyStatus.UNMODELED_CAUSE
        elif coefficient_dimension == rank:
            status = CauseFamilyAdequacyStatus.ADEQUATE_UNIQUE
        else:
            status = CauseFamilyAdequacyStatus.ADEQUATE_SET_VALUED

        blocks: list[CauseBlockAdequacyV1] = []
        start = 0
        for cause in causes:
            signature = signatures[cause]
            stop = start + signature.shape[1]
            if coefficient_nullspace.shape[1]:
                ambiguity = coefficient_nullspace[start:stop, :]
                unresolved, _, _ = _rank(
                    ambiguity,
                    relative=relative,
                    absolute=absolute,
                )
            else:
                unresolved = 0
            identifiable = signature.shape[1] - unresolved

            if len(causes) > 1:
                other = np.hstack(
                    [signatures[item] for item in causes if item != cause]
                )
            else:
                other = np.empty((residual.size, 0))
            projector = _orthogonal_projector(
                other,
                relative=relative,
                absolute=absolute,
            )
            residualized = signature - projector @ signature
            residualized_rank, residualized_singular, residualized_tol = _rank(
                residualized,
                relative=relative,
                absolute=absolute,
            )
            if residualized_rank == signature.shape[1]:
                separation = float(residualized_singular[residualized_rank - 1])
            else:
                separation = 0.0
            if identifiable == signature.shape[1]:
                block_status = CauseBlockStatus.IDENTIFIABLE
            elif identifiable:
                block_status = CauseBlockStatus.PARTIALLY_IDENTIFIABLE
            else:
                block_status = CauseBlockStatus.CONFOUNDED
            if (
                separation <= residualized_tol
                and block_status is CauseBlockStatus.IDENTIFIABLE
            ):
                raise RuntimeError("inconsistent cause-block rank diagnostics")
            blocks.append(
                CauseBlockAdequacyV1(
                    cause_id=cause,
                    coefficient_start=start,
                    coefficient_stop=stop,
                    dimension=signature.shape[1],
                    identifiable_dimension=identifiable,
                    unresolved_dimension=unresolved,
                    residualized_rank=residualized_rank,
                    separation_margin=separation,
                    coefficient_estimate=tuple(
                        float(value) for value in coefficients[start:stop]
                    ),
                    status=block_status,
                )
            )
            start = stop

        metadata = json.loads(
            json.dumps(self.metadata, sort_keys=True, allow_nan=False)
        )
        immutable_signatures = {
            cause: _immutable(signatures[cause]) for cause in causes
        }
        object.__setattr__(self, "cause_order", causes)
        object.__setattr__(self, "cause_signatures", immutable_signatures)
        object.__setattr__(
            self,
            "cause_signature_ids",
            {cause: self.cause_signature_ids[cause] for cause in causes},
        )
        object.__setattr__(self, "whitened_residual", _immutable(residual))
        object.__setattr__(self, "noise_radius", noise_radius)
        object.__setattr__(self, "relative_rank_tolerance", relative)
        object.__setattr__(self, "absolute_rank_tolerance", absolute)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "total_design", _immutable(total))
        object.__setattr__(
            self,
            "minimum_norm_coefficients",
            _immutable(coefficients),
        )
        object.__setattr__(self, "fitted_residual", _immutable(fitted))
        object.__setattr__(self, "unexplained_residual", _immutable(unexplained))
        object.__setattr__(
            self,
            "coefficient_nullspace",
            _immutable(coefficient_nullspace),
        )
        object.__setattr__(self, "singular_values", _immutable(singular_values))
        object.__setattr__(self, "design_rank", rank)
        object.__setattr__(self, "coefficient_dimension", coefficient_dimension)
        object.__setattr__(self, "solution_nullity", coefficient_dimension - rank)
        object.__setattr__(self, "residual_norm", residual_norm)
        object.__setattr__(self, "unexplained_norm", unexplained_norm)
        object.__setattr__(self, "explained_energy_fraction", explained_fraction)
        object.__setattr__(
            self,
            "smallest_nonzero_singular_value",
            smallest_nonzero,
        )
        object.__setattr__(
            self,
            "identifiable_component_error_bound",
            error_bound,
        )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "cause_blocks", tuple(blocks))

        expected = _canonical_id(self.descriptor())
        supplied = self.artifact_id
        if supplied is not None:
            supplied = _digest(supplied, name="artifact_id")
            if supplied != expected:
                raise ValueError("artifact_id does not match certificate content")
        object.__setattr__(self, "artifact_id", expected)

    @property
    def family_adequate(self) -> bool:
        return self.status in {
            CauseFamilyAdequacyStatus.ADEQUATE_UNIQUE,
            CauseFamilyAdequacyStatus.ADEQUATE_SET_VALUED,
        }

    @property
    def unique_coefficients(self) -> bool:
        return self.status is CauseFamilyAdequacyStatus.ADEQUATE_UNIQUE

    @property
    def attribution_permitted(self) -> bool:
        """Whether any registered-cause interpretation may be consumed downstream."""
        return self.family_adequate

    def arrays(self) -> Mapping[str, np.ndarray]:
        result: dict[str, np.ndarray] = {
            "whitened_residual": self.whitened_residual,
            "total_design": self.total_design,
            "minimum_norm_coefficients": self.minimum_norm_coefficients,
            "fitted_residual": self.fitted_residual,
            "unexplained_residual": self.unexplained_residual,
            "coefficient_nullspace": self.coefficient_nullspace,
            "singular_values": self.singular_values,
        }
        result.update(
            {
                f"cause_signature::{cause}": self.cause_signatures[cause]
                for cause in self.cause_order
            }
        )
        return result

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": CAUSE_FAMILY_ADEQUACY_SCHEMA,
            "schema_version": CAUSE_FAMILY_ADEQUACY_VERSION,
            "semantics": CAUSE_FAMILY_ADEQUACY_SEMANTICS,
            "residual_id": self.residual_id,
            "intervention_roster_id": self.intervention_roster_id,
            "whitening_id": self.whitening_id,
            "cause_order": list(self.cause_order),
            "cause_signature_ids": dict(self.cause_signature_ids),
            "cause_signatures": {
                cause: _array_record(self.cause_signatures[cause])
                for cause in self.cause_order
            },
            "whitened_residual": _array_record(self.whitened_residual),
            "noise_radius": self.noise_radius,
            "relative_rank_tolerance": self.relative_rank_tolerance,
            "absolute_rank_tolerance": self.absolute_rank_tolerance,
            "total_design": _array_record(self.total_design),
            "minimum_norm_coefficients": _array_record(self.minimum_norm_coefficients),
            "fitted_residual": _array_record(self.fitted_residual),
            "unexplained_residual": _array_record(self.unexplained_residual),
            "coefficient_nullspace": _array_record(self.coefficient_nullspace),
            "singular_values": _array_record(self.singular_values),
            "design_rank": self.design_rank,
            "coefficient_dimension": self.coefficient_dimension,
            "solution_nullity": self.solution_nullity,
            "residual_norm": self.residual_norm,
            "unexplained_norm": self.unexplained_norm,
            "explained_energy_fraction": self.explained_energy_fraction,
            "smallest_nonzero_singular_value": (self.smallest_nonzero_singular_value),
            "identifiable_component_error_bound": (
                self.identifiable_component_error_bound
            ),
            "status": self.status.value,
            "cause_blocks": [block.to_record() for block in self.cause_blocks],
            "metadata": self.metadata,
            "claim_boundary": CAUSE_FAMILY_ADEQUACY_CLAIM_BOUNDARY,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}
