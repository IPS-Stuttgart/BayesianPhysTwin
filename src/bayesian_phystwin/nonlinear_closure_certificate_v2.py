"""Worst-case nonlinear-closure evidence over frozen queries and perturbations.

The historical :class:`~bayesian_phystwin.physical_linearization.NonlinearClosureV1`
records one aggregate comparison. This additive certificate binds the exact
physical linearization, perturbation/query identities, replay arrays, per-query
tolerances, and worst per-query/per-horizon closure ratio. It is an experimental
direct-import surface and does not alter the V1 artifact or any frozen study.
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
    immutable_integer_array,
    literal_lower_hex,
    plain_json,
)
from ._portable_contracts import content_id
from .physical_linearization import (
    PHYSICAL_LINEARIZATION_SCHEMA,
    PHYSICAL_LINEARIZATION_VERSION,
    PhysicalLinearizationV1,
)

NONLINEAR_CLOSURE_CERTIFICATE_SCHEMA: Final = "bayesian_phystwin.nonlinear_closure"
NONLINEAR_CLOSURE_CERTIFICATE_VERSION: Final = 2
NONLINEAR_CLOSURE_CERTIFICATE_SEMANTICS: Final = (
    "worst-case-per-query-per-horizon-nonlinear-closure-v2"
)
NONLINEAR_CLOSURE_CERTIFICATE_CLAIM_BOUNDARY: Final = (
    "Local nonlinear replay agreement over the exact bound physical "
    "linearization, perturbation set, query set, horizons, replay arrays, and "
    "source-frozen tolerances only. Passing does not establish simulator or "
    "Jacobian correctness, global nonlinear validity, uncertainty calibration, "
    "provider competence, unseen-object transfer, intervention transport, "
    "deployment safety, or Causal4D benefit."
)


class NonlinearClosureStatus(str, Enum):
    """Worst-case decision under the exact frozen closure policy."""

    LOCALLY_CLOSED = "locally_closed"
    CLOSURE_VIOLATION = "closure_violation"


def _real_float64_array(
    value: object,
    *,
    name: str,
    ndim: int,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if result.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _immutable_float64(value: object) -> np.ndarray:
    return cast(np.ndarray, immutable_array(value, dtype=np.float64))


def _immutable_int64(value: object, *, name: str) -> np.ndarray:
    return cast(np.ndarray, immutable_integer_array(value, name=name))


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative real number")
    return result


def _finite_positive(value: object, *, name: str) -> float:
    result = _finite_nonnegative(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _array_record(value: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(value)
    return {
        "shape": list(contiguous.shape),
        "dtype": contiguous.dtype.str,
        "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }


def _require_unique_nonnegative(values: np.ndarray, *, name: str) -> None:
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    if np.any(values < 0):
        raise ValueError(f"{name} must be nonnegative")
    if np.unique(values).size != values.size:
        raise ValueError(f"{name} must be unique")


@dataclass(frozen=True, slots=True)
class NonlinearClosureCertificateV2:
    """Content-addressed worst-case nonlinear-closure certificate.

    ``linearized_query_m`` and ``nonlinear_query_m`` have shape ``(P, Q, 3)``
    for ``P`` frozen perturbations and ``Q`` registered 3-D queries. Admission is
    conjunctive across every perturbation/query pair and therefore cannot be
    rescued by a large, well-modelled query dominating one aggregate norm.
    """

    linearization: PhysicalLinearizationV1 = field(repr=False)
    perturbation_set_id: str
    query_set_id: str
    perturbation_indices: np.ndarray = field(repr=False)
    query_indices: np.ndarray = field(repr=False)
    horizon_indices: np.ndarray = field(repr=False)
    baseline_query_m: np.ndarray = field(repr=False)
    linearized_query_m: np.ndarray = field(repr=False)
    nonlinear_query_m: np.ndarray = field(repr=False)
    absolute_tolerance_m: np.ndarray = field(repr=False)
    relative_tolerance: np.ndarray = field(repr=False)
    prediction_floor_m: float = 1e-12
    closure_ratio_limit: float = 1.0
    comparison_tolerance: float = 1e-12
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    predicted_change_norm_m: np.ndarray = field(init=False, repr=False)
    nonlinear_remainder_norm_m: np.ndarray = field(init=False, repr=False)
    allowed_remainder_norm_m: np.ndarray = field(init=False, repr=False)
    closure_ratio: np.ndarray = field(init=False, repr=False)
    per_query_maximum_closure_ratio: np.ndarray = field(init=False, repr=False)
    unique_horizon_indices: np.ndarray = field(init=False, repr=False)
    per_horizon_maximum_closure_ratio: np.ndarray = field(init=False, repr=False)
    maximum_closure_ratio: float = field(init=False)
    maximum_absolute_remainder_m: float = field(init=False)
    maximum_relative_remainder: float = field(init=False)
    closure_ratio_margin: float = field(init=False)
    admission_bound: float = field(init=False)
    worst_perturbation_index: int = field(init=False)
    worst_query_index: int = field(init=False)
    worst_horizon_index: int = field(init=False)
    worst_predicted_change_m: float = field(init=False)
    worst_nonlinear_remainder_m: float = field(init=False)
    worst_allowed_remainder_m: float = field(init=False)
    status: NonlinearClosureStatus = field(init=False)

    def __post_init__(self) -> None:
        linearization = self.linearization
        if not isinstance(linearization, PhysicalLinearizationV1):
            raise ValueError("linearization must be a PhysicalLinearizationV1")
        perturbation_set_id = cast(
            str,
            literal_lower_hex(
                self.perturbation_set_id,
                name="perturbation_set_id",
                lengths={64},
            ),
        )
        query_set_id = cast(
            str,
            literal_lower_hex(
                self.query_set_id,
                name="query_set_id",
                lengths={64},
            ),
        )

        perturbation_indices = _immutable_int64(
            self.perturbation_indices,
            name="perturbation_indices",
        )
        query_indices = _immutable_int64(
            self.query_indices,
            name="query_indices",
        )
        horizon_indices = _immutable_int64(
            self.horizon_indices,
            name="horizon_indices",
        )
        _require_unique_nonnegative(
            perturbation_indices,
            name="perturbation_indices",
        )
        _require_unique_nonnegative(query_indices, name="query_indices")
        if horizon_indices.ndim != 1 or horizon_indices.size == 0:
            raise ValueError("horizon_indices must be a nonempty vector")
        if np.any(horizon_indices < 0):
            raise ValueError("horizon_indices must be nonnegative")

        baseline = _real_float64_array(
            self.baseline_query_m,
            name="baseline_query_m",
            ndim=2,
        )
        linearized = _real_float64_array(
            self.linearized_query_m,
            name="linearized_query_m",
            ndim=3,
        )
        nonlinear = _real_float64_array(
            self.nonlinear_query_m,
            name="nonlinear_query_m",
            ndim=3,
        )
        absolute_tolerance = _real_float64_array(
            self.absolute_tolerance_m,
            name="absolute_tolerance_m",
            ndim=1,
        )
        relative_tolerance = _real_float64_array(
            self.relative_tolerance,
            name="relative_tolerance",
            ndim=1,
        )

        perturbation_count = perturbation_indices.size
        query_count = query_indices.size
        if query_count != linearization.query_state_jacobian.shape[0]:
            raise ValueError(
                "query_indices must identify every query row in the bound "
                "physical linearization"
            )
        if horizon_indices.shape != (query_count,):
            raise ValueError("horizon_indices must have one value per query")
        if baseline.shape != (query_count, 3):
            raise ValueError("baseline_query_m must have shape (Q, 3)")
        expected_replay_shape = (perturbation_count, query_count, 3)
        if linearized.shape != expected_replay_shape:
            raise ValueError("linearized_query_m must have shape (P, Q, 3)")
        if nonlinear.shape != expected_replay_shape:
            raise ValueError("nonlinear_query_m must have shape (P, Q, 3)")
        if absolute_tolerance.shape != (query_count,):
            raise ValueError("absolute_tolerance_m must have shape (Q,)")
        if relative_tolerance.shape != (query_count,):
            raise ValueError("relative_tolerance must have shape (Q,)")
        if np.any(absolute_tolerance < 0.0) or np.any(relative_tolerance < 0.0):
            raise ValueError("closure tolerances must be nonnegative")
        if np.any((absolute_tolerance == 0.0) & (relative_tolerance == 0.0)):
            raise ValueError(
                "each query must have a positive absolute or relative tolerance"
            )

        prediction_floor = _finite_positive(
            self.prediction_floor_m,
            name="prediction_floor_m",
        )
        ratio_limit = _finite_nonnegative(
            self.closure_ratio_limit,
            name="closure_ratio_limit",
        )
        comparison_tolerance = _finite_nonnegative(
            self.comparison_tolerance,
            name="comparison_tolerance",
        )

        predicted_change = linearized - baseline[None, :, :]
        nonlinear_remainder = nonlinear - linearized
        predicted_change_norm = np.linalg.norm(predicted_change, axis=2)
        nonlinear_remainder_norm = np.linalg.norm(nonlinear_remainder, axis=2)
        relative_denominator = np.maximum(predicted_change_norm, prediction_floor)
        allowed_remainder = (
            absolute_tolerance[None, :]
            + relative_tolerance[None, :] * relative_denominator
        )
        closure_ratio = nonlinear_remainder_norm / allowed_remainder
        per_query_maximum = np.max(closure_ratio, axis=0)
        unique_horizons = np.unique(horizon_indices)
        per_horizon_maximum = np.asarray(
            [
                float(np.max(closure_ratio[:, horizon_indices == horizon]))
                for horizon in unique_horizons
            ],
            dtype=np.float64,
        )

        flat_worst = int(np.argmax(closure_ratio))
        worst_row, worst_column = np.unravel_index(flat_worst, closure_ratio.shape)
        maximum_ratio = float(closure_ratio[worst_row, worst_column])
        maximum_absolute = float(np.max(nonlinear_remainder_norm))
        maximum_relative = float(
            np.max(nonlinear_remainder_norm / relative_denominator)
        )
        admission_bound = ratio_limit + comparison_tolerance
        status = (
            NonlinearClosureStatus.LOCALLY_CLOSED
            if maximum_ratio <= admission_bound
            else NonlinearClosureStatus.CLOSURE_VIOLATION
        )

        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="nonlinear closure certificate metadata",
        )
        arrays = {
            "perturbation_indices": perturbation_indices,
            "query_indices": query_indices,
            "horizon_indices": horizon_indices,
            "baseline_query_m": baseline,
            "linearized_query_m": linearized,
            "nonlinear_query_m": nonlinear,
            "absolute_tolerance_m": absolute_tolerance,
            "relative_tolerance": relative_tolerance,
            "predicted_change_norm_m": predicted_change_norm,
            "nonlinear_remainder_norm_m": nonlinear_remainder_norm,
            "allowed_remainder_norm_m": allowed_remainder,
            "closure_ratio": closure_ratio,
            "per_query_maximum_closure_ratio": per_query_maximum,
            "unique_horizon_indices": unique_horizons,
            "per_horizon_maximum_closure_ratio": per_horizon_maximum,
        }
        for name, value in arrays.items():
            if name in {
                "perturbation_indices",
                "query_indices",
                "horizon_indices",
                "unique_horizon_indices",
            }:
                frozen = _immutable_int64(value, name=name)
            else:
                frozen = _immutable_float64(value)
            object.__setattr__(self, name, frozen)

        for name, value in (
            ("prediction_floor_m", prediction_floor),
            ("closure_ratio_limit", ratio_limit),
            ("comparison_tolerance", comparison_tolerance),
            ("maximum_closure_ratio", maximum_ratio),
            ("maximum_absolute_remainder_m", maximum_absolute),
            ("maximum_relative_remainder", maximum_relative),
            ("closure_ratio_margin", ratio_limit - maximum_ratio),
            ("admission_bound", admission_bound),
            (
                "worst_perturbation_index",
                int(perturbation_indices[worst_row]),
            ),
            ("worst_query_index", int(query_indices[worst_column])),
            ("worst_horizon_index", int(horizon_indices[worst_column])),
            (
                "worst_predicted_change_m",
                float(predicted_change_norm[worst_row, worst_column]),
            ),
            (
                "worst_nonlinear_remainder_m",
                float(nonlinear_remainder_norm[worst_row, worst_column]),
            ),
            (
                "worst_allowed_remainder_m",
                float(allowed_remainder[worst_row, worst_column]),
            ),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "perturbation_set_id", perturbation_set_id)
        object.__setattr__(self, "query_set_id", query_set_id)
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
                    "nonlinear closure certificate artifact_id does not match content"
                )
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def locally_closed(self) -> bool:
        return self.status is NonlinearClosureStatus.LOCALLY_CLOSED

    @property
    def passes_closure_gate(self) -> bool:
        return self.locally_closed

    @property
    def perturbation_count(self) -> int:
        return int(self.perturbation_indices.size)

    @property
    def query_count(self) -> int:
        return int(self.query_indices.size)

    @property
    def horizon_count(self) -> int:
        return int(self.unique_horizon_indices.size)

    def arrays(self) -> Mapping[str, np.ndarray]:
        """Return immutable input and derived arrays bound by the certificate."""

        return {
            "perturbation_indices": self.perturbation_indices,
            "query_indices": self.query_indices,
            "horizon_indices": self.horizon_indices,
            "baseline_query_m": self.baseline_query_m,
            "linearized_query_m": self.linearized_query_m,
            "nonlinear_query_m": self.nonlinear_query_m,
            "absolute_tolerance_m": self.absolute_tolerance_m,
            "relative_tolerance": self.relative_tolerance,
            "predicted_change_norm_m": self.predicted_change_norm_m,
            "nonlinear_remainder_norm_m": self.nonlinear_remainder_norm_m,
            "allowed_remainder_norm_m": self.allowed_remainder_norm_m,
            "closure_ratio": self.closure_ratio,
            "per_query_maximum_closure_ratio": (self.per_query_maximum_closure_ratio),
            "unique_horizon_indices": self.unique_horizon_indices,
            "per_horizon_maximum_closure_ratio": (
                self.per_horizon_maximum_closure_ratio
            ),
        }

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": NONLINEAR_CLOSURE_CERTIFICATE_SCHEMA,
            "schema_version": NONLINEAR_CLOSURE_CERTIFICATE_VERSION,
            "semantics": NONLINEAR_CLOSURE_CERTIFICATE_SEMANTICS,
            "physical_linearization_id": self.linearization.artifact_id,
            "physical_linearization_schema": PHYSICAL_LINEARIZATION_SCHEMA,
            "physical_linearization_schema_version": PHYSICAL_LINEARIZATION_VERSION,
            "observation_artifact_id": self.linearization.observation_artifact_id,
            "baseline_belief_id": self.linearization.baseline_belief_id,
            "action_prefix_id": self.linearization.action_prefix_id,
            "simulator_revision": self.linearization.simulator_revision,
            "perturbation_set_id": self.perturbation_set_id,
            "query_set_id": self.query_set_id,
            **{name: _array_record(value) for name, value in self.arrays().items()},
            "prediction_floor_m": self.prediction_floor_m,
            "closure_ratio_limit": self.closure_ratio_limit,
            "comparison_tolerance": self.comparison_tolerance,
            "maximum_closure_ratio": self.maximum_closure_ratio,
            "maximum_absolute_remainder_m": self.maximum_absolute_remainder_m,
            "maximum_relative_remainder": self.maximum_relative_remainder,
            "closure_ratio_margin": self.closure_ratio_margin,
            "admission_bound": self.admission_bound,
            "worst_perturbation_index": self.worst_perturbation_index,
            "worst_query_index": self.worst_query_index,
            "worst_horizon_index": self.worst_horizon_index,
            "worst_predicted_change_m": self.worst_predicted_change_m,
            "worst_nonlinear_remainder_m": self.worst_nonlinear_remainder_m,
            "worst_allowed_remainder_m": self.worst_allowed_remainder_m,
            "status": self.status.value,
            "metadata": plain_json(self.metadata),
            "claim_boundary": NONLINEAR_CLOSURE_CERTIFICATE_CLAIM_BOUNDARY,
        }

    def summary(self) -> dict[str, object]:
        return {
            "schema": NONLINEAR_CLOSURE_CERTIFICATE_SCHEMA,
            "schema_version": NONLINEAR_CLOSURE_CERTIFICATE_VERSION,
            "artifact_id": self.artifact_id,
            "physical_linearization_id": self.linearization.artifact_id,
            "perturbation_set_id": self.perturbation_set_id,
            "query_set_id": self.query_set_id,
            "status": self.status.value,
            "locally_closed": self.locally_closed,
            "passes_closure_gate": self.passes_closure_gate,
            "perturbation_count": self.perturbation_count,
            "query_count": self.query_count,
            "horizon_count": self.horizon_count,
            "maximum_closure_ratio": self.maximum_closure_ratio,
            "closure_ratio_limit": self.closure_ratio_limit,
            "closure_ratio_margin": self.closure_ratio_margin,
            "worst_perturbation_index": self.worst_perturbation_index,
            "worst_query_index": self.worst_query_index,
            "worst_horizon_index": self.worst_horizon_index,
            "claim_boundary": NONLINEAR_CLOSURE_CERTIFICATE_CLAIM_BOUNDARY,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}


__all__ = [
    "NONLINEAR_CLOSURE_CERTIFICATE_CLAIM_BOUNDARY",
    "NONLINEAR_CLOSURE_CERTIFICATE_SCHEMA",
    "NONLINEAR_CLOSURE_CERTIFICATE_SEMANTICS",
    "NONLINEAR_CLOSURE_CERTIFICATE_VERSION",
    "NonlinearClosureCertificateV2",
    "NonlinearClosureStatus",
]
