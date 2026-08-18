"""Typed, content-addressed dependence metadata for physical anchors."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._canonical_contracts import (
    canonical_string_tuple,
    frozen_finite_json_mapping,
    immutable_array,
    plain_json,
)
from .._portable_contracts import content_id
from .._validation import lowercase_sha256

ANCHOR_DEPENDENCE_SCHEMA = "bayesian_phystwin.anchor_dependence"
ANCHOR_DEPENDENCE_VERSION = 1


def _finite_vector(value: object, *, name: str, count: int) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain real numeric values") from error
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    result = immutable_array(raw, dtype=np.dtype("<f8"))
    if result.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_array(value: object, *, name: str, ndim: int) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain real numeric values") from error
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    result = immutable_array(raw, dtype=np.dtype("<f8"))
    if result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _array_descriptor(value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.dtype("<f8")))
    return {
        "dtype": "<f8",
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


@dataclass(frozen=True, slots=True)
class AnchorDependenceV1:
    """Validated dependence and latent-bias assumptions for anchor rows.

    The row order is the order of ``anchor_innovation_m`` supplied to the
    guarded inference call. Repeated correlation-group identifiers declare
    dependent anchor evidence. Optional bias terms describe a shared Gaussian
    nuisance with Jacobian shape ``(A, 3, B)`` and covariance shape ``(B, B)``.
    """

    correlation_group_ids: tuple[str, ...]
    prior_reliability: np.ndarray
    prior_nominal_probability: np.ndarray
    composite_weight: np.ndarray
    bias_jacobian: np.ndarray | None = None
    bias_prior_covariance: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.correlation_group_ids) is not tuple:
            raise TypeError("correlation_group_ids must be a tuple of exact strings")
        if any(type(group) is not str for group in self.correlation_group_ids):
            raise TypeError("correlation_group_ids must be a tuple of exact strings")
        groups = canonical_string_tuple(
            self.correlation_group_ids,
            name="correlation_group_ids",
            allow_empty=False,
        )
        if any(group.strip() != group for group in groups):
            raise ValueError(
                "correlation_group_ids must not contain surrounding whitespace"
            )
        count = len(groups)
        reliability = _finite_vector(
            self.prior_reliability,
            name="prior_reliability",
            count=count,
        )
        nominal = _finite_vector(
            self.prior_nominal_probability,
            name="prior_nominal_probability",
            count=count,
        )
        weight = _finite_vector(
            self.composite_weight,
            name="composite_weight",
            count=count,
        )
        if np.any((reliability < 0.0) | (reliability > 1.0)):
            raise ValueError("prior_reliability must lie in [0, 1]")
        if np.any((nominal < 0.0) | (nominal > 1.0)):
            raise ValueError("prior_nominal_probability must lie in [0, 1]")
        if np.any((weight <= 0.0) | (weight > 1.0)):
            raise ValueError("composite_weight must lie in (0, 1]")

        has_bias_jacobian = self.bias_jacobian is not None
        has_bias_covariance = self.bias_prior_covariance is not None
        if has_bias_jacobian != has_bias_covariance:
            raise ValueError(
                "bias_jacobian and bias_prior_covariance must be supplied together"
            )
        bias_jacobian: np.ndarray | None = None
        bias_covariance: np.ndarray | None = None
        if has_bias_jacobian:
            bias_jacobian = _finite_array(
                self.bias_jacobian,
                name="bias_jacobian",
                ndim=3,
            )
            if bias_jacobian.shape[:2] != (count, 3):
                raise ValueError("bias_jacobian must have shape (A, 3, B)")
            bias_count = bias_jacobian.shape[2]
            if bias_count == 0:
                raise ValueError("bias_jacobian must contain at least one bias mode")
            bias_covariance = _finite_array(
                self.bias_prior_covariance,
                name="bias_prior_covariance",
                ndim=2,
            )
            if bias_covariance.shape != (bias_count, bias_count):
                raise ValueError("bias_prior_covariance must have shape (B, B)")
            if not np.allclose(
                bias_covariance,
                bias_covariance.T,
                atol=1e-12,
                rtol=1e-10,
            ):
                raise ValueError("bias_prior_covariance must be symmetric")
            symmetric = 0.5 * (bias_covariance + bias_covariance.T)
            if np.min(np.linalg.eigvalsh(symmetric)) < -1e-12:
                raise ValueError("bias_prior_covariance must be positive semidefinite")

        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="anchor dependence metadata",
        )
        object.__setattr__(self, "correlation_group_ids", groups)
        object.__setattr__(self, "prior_reliability", reliability)
        object.__setattr__(self, "prior_nominal_probability", nominal)
        object.__setattr__(self, "composite_weight", weight)
        object.__setattr__(self, "bias_jacobian", bias_jacobian)
        object.__setattr__(self, "bias_prior_covariance", bias_covariance)
        object.__setattr__(self, "metadata", metadata)

        expected_id = content_id(self.descriptor())
        supplied_id = self.artifact_id
        if supplied_id is not None:
            supplied_id = lowercase_sha256(supplied_id, name="artifact_id")
            if supplied_id != expected_id:
                raise ValueError("anchor dependence artifact_id does not match content")
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def anchor_count(self) -> int:
        return len(self.correlation_group_ids)

    @property
    def bias_dimension(self) -> int:
        return 0 if self.bias_jacobian is None else self.bias_jacobian.shape[2]

    def require_anchor_count(self, count: int) -> None:
        if isinstance(count, (bool, np.bool_)) or not isinstance(
            count, (int, np.integer)
        ):
            raise TypeError("anchor count must be an integer")
        if int(count) != self.anchor_count:
            raise ValueError(
                "anchor dependence row count does not match anchor_innovation_m"
            )

    def arrays(self) -> dict[str, np.ndarray]:
        result = {
            "prior_reliability": self.prior_reliability,
            "prior_nominal_probability": self.prior_nominal_probability,
            "composite_weight": self.composite_weight,
        }
        if self.bias_jacobian is not None:
            result["bias_jacobian"] = self.bias_jacobian
        if self.bias_prior_covariance is not None:
            result["bias_prior_covariance"] = self.bias_prior_covariance
        return result

    def descriptor(self) -> dict[str, object]:
        arrays = self.arrays()
        return {
            "schema": ANCHOR_DEPENDENCE_SCHEMA,
            "schema_version": ANCHOR_DEPENDENCE_VERSION,
            "anchor_count": self.anchor_count,
            "bias_dimension": self.bias_dimension,
            "correlation_group_ids": list(self.correlation_group_ids),
            "arrays": {
                name: _array_descriptor(value)
                for name, value in sorted(arrays.items())
            },
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}

    def inference_kwargs(self) -> dict[str, object]:
        """Return exact internal keywords consumed by the frozen solver path."""

        return {
            "anchor_correlation_group_ids": self.correlation_group_ids,
            "anchor_prior_reliability": self.prior_reliability,
            "anchor_prior_nominal_probability": self.prior_nominal_probability,
            "anchor_composite_weight": self.composite_weight,
            "anchor_bias_jacobian": self.bias_jacobian,
            "anchor_bias_prior_covariance": self.bias_prior_covariance,
        }


__all__ = [
    "ANCHOR_DEPENDENCE_SCHEMA",
    "ANCHOR_DEPENDENCE_VERSION",
    "AnchorDependenceV1",
]
