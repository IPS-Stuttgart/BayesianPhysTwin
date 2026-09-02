"""Robust, open-set interventional attribution of physical-twin error.

A nominal interventional certificate can distinguish registered cause signatures
when those signatures are treated as exact.  Real signatures are estimated.  This
module turns deterministic bounds on signature error, coefficient magnitude, and
observation noise into:

* a finite error bound for every identifiable cause query;
* an explicit ``unregistered_cause`` result when the registered family cannot
  explain the stacked intervention response within its declared uncertainty; and
* minimum-cardinality and minimum-cost intervention sets computed before outcome
  access.

The result is local and finite-family.  Passing the family-closure test means only
that the registered family was not falsified at the supplied uncertainty radius.
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

ROBUST_ATTRIBUTION_SCHEMA: Final = (
    "bayesian_phystwin.robust_interventional_cause_attribution"
)
ROBUST_ATTRIBUTION_VERSION: Final = 1
ROBUST_ATTRIBUTION_SEMANTICS: Final = (
    "pre-outcome-bounded-signature-attribution-with-open-set-closure-v1"
)
ROBUST_ATTRIBUTION_CLAIM_BOUNDARY: Final = (
    "A passing decision bounds one registered cause query under the exact nominal "
    "intervention signatures, deterministic signature-error budgets, coefficient "
    "norm bounds, observation-noise radii, whitening, and query tolerance. A "
    "failed closure test rejects the registered family. A passed closure test "
    "does not prove family completeness, unique physical causation, nonlinear "
    "validity, unseen-object transfer, safe control, or deployment safety."
)


class RobustAttributionStatus(str, Enum):
    ROBUSTLY_ATTRIBUTABLE = "robustly_attributable"
    IDENTIFIABLE_BUT_UNSTABLE = "identifiable_but_unstable"
    CONFOUNDED = "confounded"
    TRIVIAL_QUERY = "trivial_query"
    UNREGISTERED_CAUSE = "unregistered_cause"


def _literal(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty literal string")
    return value


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative real number")
    return result


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _matrix(value: object, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be real numeric")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if result.ndim != 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite matrix")
    return result


def _vector(value: object, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be real numeric")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector")
    return result


def _immutable(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(array.shape)


def _freeze_json(value: object, name: str) -> object:
    if value is None or type(value) in {str, bool, int}:
        return value
    if isinstance(value, Real):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError(f"{name} must contain finite JSON values")
        return result
    if isinstance(value, Mapping):
        output: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"{name} keys must be strings")
            output[key] = _freeze_json(item, name)
        return MappingProxyType(output)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze_json(item, name) for item in value)
    raise ValueError(f"{name} must contain finite JSON values")


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _array_record(value: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def _content_id(value: object) -> str:
    data = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _basis(design: np.ndarray, relative: float, absolute: float) -> np.ndarray:
    if design.shape[1] == 0:
        return np.empty((design.shape[0], 0), dtype=np.float64)
    left, singular, _ = np.linalg.svd(design, full_matrices=False)
    tolerance = max(absolute, relative * (float(singular[0]) if len(singular) else 0.0))
    return left[:, singular > tolerance]


def _complement(design: np.ndarray, relative: float, absolute: float) -> np.ndarray:
    basis = _basis(design, relative, absolute)
    return np.eye(design.shape[0]) - basis @ basis.T


def _pinv(design: np.ndarray, relative: float, absolute: float) -> np.ndarray:
    left, singular, right_t = np.linalg.svd(design, full_matrices=False)
    tolerance = max(absolute, relative * (float(singular[0]) if len(singular) else 0.0))
    inverse = np.zeros_like(singular)
    retained = singular > tolerance
    inverse[retained] = 1.0 / singular[retained]
    return (right_t.T * inverse) @ left.T


def _stack_bound(values: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(values, dtype=np.float64)))


@dataclass(frozen=True, slots=True)
class RobustCauseModelV1:
    cause_id: str
    intervention_ids: Sequence[str]
    response_blocks: Sequence[np.ndarray]
    query_map: np.ndarray
    signature_error_bounds: Sequence[float]
    coefficient_norm_bound: float
    query_error_tolerance: float
    minimum_effect_norm: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cause_id = _literal(self.cause_id, "cause_id")
        ids = tuple(_literal(value, "intervention_id") for value in self.intervention_ids)
        if len(ids) < 2 or ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise ValueError("intervention_ids must be sorted, unique, and contain at least two values")
        blocks = tuple(_matrix(value, "response_block") for value in self.response_blocks)
        if len(blocks) != len(ids) or any(block.shape[0] == 0 or block.shape[1] == 0 for block in blocks):
            raise ValueError("one nonempty response block is required per intervention")
        latent = blocks[0].shape[1]
        if any(block.shape[1] != latent for block in blocks):
            raise ValueError("response blocks must share the latent dimension")
        query = _matrix(self.query_map, "query_map")
        if query.shape[0] == 0 or query.shape[1] != latent:
            raise ValueError("query_map must have one column per cause coordinate")
        errors = tuple(_finite_nonnegative(value, "signature_error_bound") for value in self.signature_error_bounds)
        if len(errors) != len(ids):
            raise ValueError("one signature error bound is required per intervention")
        metadata = _freeze_json(self.metadata, "metadata")
        if not isinstance(metadata, Mapping):
            raise ValueEror("metadata must be a mapping")
        object.__setattr__(self, "cause_id", cause_id)
        object.__setattr__(self, "intervention_ids", ids)
        object.__setattr__(self, "response_blocks", tuple(_immutable(value) for value in blocks))
        object.__setattr__(self, "query_map", _immutable(query))
        object.__setattr__(self, "signature_error_bounds", errors)
        object.__setattr__(self, "coefficient_norm_bound", _finite_nonnegative(self.coefficient_norm_bound, "coefficient_norm_bound"))
        object.__setattr__(self, "query_error_tolerance", _finite_nonnegative(self.query_error_tolerance, "query_error_tolerance"))
        object.__setattr__(self, "minimum_effect_norm", _finite_nonnegative(self.minimum_effect_norm, "minimum_effect_norm")
        object.__setattr__(self, "metadata", metadata)

    @property
    def observation_dimensions(self) -> tuple[int, ...]:
        return tuple(int(block.shape[0]) for block in self.response_blocks)

    def descriptor(self) -> dict[str, object]:
        return {
            "cause_id": self.cause_id,
            "intervention_ids": list(self.intervention_ids),
            "response_blocks": [_array_record(value) for value in self.response_blocks],
            "query_map": _array_record(self.query_map),
            "signature_error_bounds": list(self.signature_error_bounds),
            "coefficient_norm_bound": self.coefficient_norm_bound,
            "query_error_tolerance": self.query_error_tolerance,
            "minimum_effect_norm": self.minimum_effect_norm,
            "metadata": _plain(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RobustObservationDesignV1