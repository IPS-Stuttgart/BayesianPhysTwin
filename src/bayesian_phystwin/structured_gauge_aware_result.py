"""Versioned covariance-bearing results for scalable gauge-aware inference.

The historical :class:`GaugeAwareBeliefResult` stores one complete dense posterior
covariance.  That remains the compatibility surface for existing callers.  This
module adds an explicit structured result whose rejected-update covariance can
remain precision-backed until a caller deliberately requests materialization.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    immutable_array,
    plain_json,
)
from ._gauge_aware_contracts import GaugeAwareBeliefResult

STRUCTURED_GAUGE_AWARE_RESULT_SCHEMA = (
    "bayesian_phystwin.structured_gauge_aware_result"
)
STRUCTURED_GAUGE_AWARE_RESULT_VERSION = 1
DENSE_COVARIANCE_REPRESENTATION = "dense-covariance-v1"
PRECISION_BACKED_COVARIANCE_REPRESENTATION = (
    "block-diagonal-state-covariance-plus-nuisance-precision-v1"
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _readonly_array(value: object, *, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    _require(array.ndim == ndim, f"{name} must have {ndim} dimensions")
    _require(np.all(np.isfinite(array)), f"{name} must be finite")
    return immutable_array(array, dtype=np.dtype(np.float64))


def _symmetric_matrix(
    value: object,
    *,
    name: str,
    positive_semidefinite: bool,
    positive_definite: bool = False,
) -> np.ndarray:
    matrix = _readonly_array(value, name=name, ndim=2)
    _require(matrix.shape[0] == matrix.shape[1], f"{name} must be square")
    _require(
        np.allclose(matrix, matrix.T, atol=1e-10, rtol=1e-10),
        f"{name} must be symmetric",
    )
    if len(matrix):
        eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
        if positive_definite:
            _require(np.min(eigenvalues) > 0.0, f"{name} must be positive definite")
        elif positive_semidefinite:
            _require(
                np.min(eigenvalues) >= -1e-9,
                f"{name} must be positive semidefinite",
            )
    return matrix


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _canonical_id(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            plain_json(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _validate_materialization_budget(
    maximum_bytes: int | None,
    *,
    required_bytes: int,
) -> None:
    if maximum_bytes is None:
        return
    if type(maximum_bytes) is not int or maximum_bytes < 0:
        raise ValueError("maximum_bytes must be a nonnegative integer or None")
    if required_bytes > maximum_bytes:
        raise MemoryError(
            "dense covariance materialization requires "
            f"{required_bytes} bytes, exceeding the {maximum_bytes}-byte limit"
        )


@dataclass(frozen=True, slots=True)
class DenseCovarianceV1:
    """An already materialized complete covariance matrix."""

    covariance: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "covariance",
            _symmetric_matrix(
                self.covariance,
                name="covariance",
                positive_semidefinite=True,
            ),
        )

    @property
    def representation(self) -> str:
        return DENSE_COVARIANCE_REPRESENTATION

    @property
    def dimension(self) -> int:
        return len(self.covariance)

    @property
    def estimated_dense_bytes(self) -> int:
        return int(self.covariance.nbytes)

    @property
    def stored_nbytes(self) -> int:
        return int(self.covariance.nbytes)

    @property
    def dense_materialized(self) -> bool:
        return True

    def descriptor(self) -> Mapping[str, Any]:
        return frozen_finite_json_mapping(
            {
                "schema": "bayesian_phystwin.gauge_aware_covariance",
                "schema_version": 1,
                "representation": self.representation,
                "dimension": self.dimension,
                "estimated_dense_bytes": self.estimated_dense_bytes,
                "stored_nbytes": self.stored_nbytes,
                "covariance_sha256": _array_sha256(self.covariance),
            },
            name="dense covariance descriptor",
        )

    def materialize(self, *, maximum_bytes: int | None = None) -> np.ndarray:
        _validate_materialization_budget(
            maximum_bytes,
            required_bytes=self.estimated_dense_bytes,
        )
        return self.covariance

    def __array__(self, dtype: Any | None = None) -> np.ndarray:
        return (
            self.covariance
            if dtype is None
            else np.asarray(self.covariance, dtype=dtype)
        )


@dataclass(frozen=True, slots=True)
class PrecisionBackedCovarianceV1:
    """Exact block prior with nuisance uncertainty retained as precision.

    The state block is stored as covariance because state priors are normally
    small.  Gauge and other nuisance variables are stored as a joint precision.
    An optional nuisance covariance is retained only for legacy dense-prior
    inputs; tree-sparse claim-bearing paths leave it absent.
    """

    state_covariance: np.ndarray
    nuisance_precision: np.ndarray
    nuisance_covariance: np.ndarray | None = None

    def __post_init__(self) -> None:
        state = _symmetric_matrix(
            self.state_covariance,
            name="state_covariance",
            positive_semidefinite=True,
        )
        precision = _symmetric_matrix(
            self.nuisance_precision,
            name="nuisance_precision",
            positive_semidefinite=False,
            positive_definite=True,
        )
        covariance: np.ndarray | None = None
        if self.nuisance_covariance is not None:
            covariance = _symmetric_matrix(
                self.nuisance_covariance,
                name="nuisance_covariance",
                positive_semidefinite=True,
            )
            _require(
                covariance.shape == precision.shape,
                "nuisance covariance and precision shapes differ",
            )
        object.__setattr__(self, "state_covariance", state)
        object.__setattr__(self, "nuisance_precision", precision)
        object.__setattr__(self, "nuisance_covariance", covariance)

    @property
    def representation(self) -> str:
        return PRECISION_BACKED_COVARIANCE_REPRESENTATION

    @property
    def dimension(self) -> int:
        return len(self.state_covariance) + len(self.nuisance_precision)

    @property
    def estimated_dense_bytes(self) -> int:
        return self.dimension**2 * np.dtype(np.float64).itemsize

    @property
    def stored_nbytes(self) -> int:
        covariance_bytes = (
            0 if self.nuisance_covariance is None else self.nuisance_covariance.nbytes
        )
        return int(
            self.state_covariance.nbytes
            + self.nuisance_precision.nbytes
            + covariance_bytes
        )

    @property
    def dense_materialized(self) -> bool:
        return False

    def descriptor(self) -> Mapping[str, Any]:
        return frozen_finite_json_mapping(
            {
                "schema": "bayesian_phystwin.gauge_aware_covariance",
                "schema_version": 1,
                "representation": self.representation,
                "dimension": self.dimension,
                "state_dimension": len(self.state_covariance),
                "nuisance_dimension": len(self.nuisance_precision),
                "estimated_dense_bytes": self.estimated_dense_bytes,
                "stored_nbytes": self.stored_nbytes,
                "state_covariance_sha256": _array_sha256(self.state_covariance),
                "nuisance_precision_sha256": _array_sha256(
                    self.nuisance_precision
                ),
                "nuisance_covariance_sha256": (
                    None
                    if self.nuisance_covariance is None
                    else _array_sha256(self.nuisance_covariance)
                ),
            },
            name="precision-backed covariance descriptor",
        )

    def materialize(self, *, maximum_bytes: int | None = None) -> np.ndarray:
        _validate_materialization_budget(
            maximum_bytes,
            required_bytes=self.estimated_dense_bytes,
        )
        nuisance_covariance = self.nuisance_covariance
        if nuisance_covariance is None:
            identity = np.eye(len(self.nuisance_precision), dtype=np.float64)
            try:
                nuisance_covariance = np.linalg.solve(
                    self.nuisance_precision,
                    identity,
                )
            except np.linalg.LinAlgError as error:
                raise ValueError(
                    "nuisance precision could not be materialized"
                ) from error
            nuisance_covariance = 0.5 * (
                nuisance_covariance + nuisance_covariance.T
            )
        state_count = len(self.state_covariance)
        result = np.zeros((self.dimension, self.dimension), dtype=np.float64)
        result[:state_count, :state_count] = self.state_covariance
        result[state_count:, state_count:] = nuisance_covariance
        return immutable_array(result, dtype=np.dtype(np.float64))

    def __array__(self, dtype: Any | None = None) -> np.ndarray:
        materialized = self.materialize()
        return (
            materialized
            if dtype is None
            else np.asarray(materialized, dtype=dtype)
        )


GaugeAwareCovarianceV1: TypeAlias = DenseCovarianceV1 | PrecisionBackedCovarianceV1


@dataclass(frozen=True, slots=True)
class StructuredGaugeAwareBeliefResultV1:
    """Gauge-aware result with an explicit covariance representation."""

    inference_admissible: bool
    reason: str
    state_coefficients: np.ndarray
    gauge_delta: np.ndarray
    shared_bias_coefficients: np.ndarray
    view_bias_coefficients: np.ndarray
    anchor_bias_coefficients: np.ndarray
    covariance: GaugeAwareCovarianceV1
    identifiable_state_transform: np.ndarray
    identifiable_fractions: np.ndarray
    query_sensitivity_fractions: np.ndarray
    robust_weights: np.ndarray
    anchor_robust_weights: np.ndarray
    diagnostics: Mapping[str, Any]
    input_lineage: Mapping[str, Any] = field(default_factory=dict)
    _result_id: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.inference_admissible) is not bool:
            raise TypeError("inference_admissible must be a bool")
        if type(self.reason) is not str or not self.reason:
            raise ValueError("reason must be a nonempty string")
        if not isinstance(
            self.covariance,
            (DenseCovarianceV1, PrecisionBackedCovarianceV1),
        ):
            raise TypeError("covariance has an unsupported representation")

        vectors: dict[str, np.ndarray] = {}
        for name in (
            "state_coefficients",
            "gauge_delta",
            "shared_bias_coefficients",
            "view_bias_coefficients",
            "anchor_bias_coefficients",
            "identifiable_fractions",
            "query_sensitivity_fractions",
            "robust_weights",
            "anchor_robust_weights",
        ):
            vectors[name] = _readonly_array(
                getattr(self, name),
                name=name,
                ndim=1,
            )
        transform = _readonly_array(
            self.identifiable_state_transform,
            name="identifiable_state_transform",
            ndim=2,
        )
        state_count = len(vectors["state_coefficients"])
        _require(
            transform.shape[0] == state_count,
            "identifiable state transform has changed shape",
        )
        _require(
            vectors["identifiable_fractions"].shape
            == vectors["query_sensitivity_fractions"].shape
            == (transform.shape[1],),
            "identifiability diagnostics have changed shape",
        )
        dimension = sum(
            len(vectors[name])
            for name in (
                "state_coefficients",
                "gauge_delta",
                "shared_bias_coefficients",
                "view_bias_coefficients",
                "anchor_bias_coefficients",
            )
        )
        _require(
            self.covariance.dimension == dimension,
            "covariance dimension differs from the coefficient layout",
        )
        if self.inference_admissible:
            _require(
                isinstance(self.covariance, DenseCovarianceV1),
                "accepted version-1 structured results require dense covariance",
            )
        else:
            for name in (
                "state_coefficients",
                "gauge_delta",
                "shared_bias_coefficients",
                "view_bias_coefficients",
                "anchor_bias_coefficients",
            ):
                _require(
                    np.count_nonzero(vectors[name]) == 0,
                    "rejected results must preserve zero candidate coefficients",
                )

        for name, value in vectors.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "identifiable_state_transform", transform)
        object.__setattr__(
            self,
            "diagnostics",
            frozen_finite_json_mapping(
                self.diagnostics,
                name="diagnostics",
            ),
        )
        object.__setattr__(
            self,
            "input_lineage",
            frozen_finite_json_mapping(
                self.input_lineage,
                name="input_lineage",
            ),
        )
        object.__setattr__(self, "_result_id", _canonical_id(self.descriptor()))

    @property
    def accepted(self) -> bool:
        return self.inference_admissible

    @property
    def covariance_representation(self) -> str:
        return self.covariance.representation

    @property
    def dense_covariance_materialized(self) -> bool:
        return self.covariance.dense_materialized

    @property
    def estimated_dense_covariance_bytes(self) -> int:
        return self.covariance.estimated_dense_bytes

    @property
    def stored_covariance_bytes(self) -> int:
        return self.covariance.stored_nbytes

    @property
    def result_id(self) -> str:
        return self._result_id

    def descriptor(self) -> Mapping[str, Any]:
        arrays = {
            name: _array_sha256(getattr(self, name))
            for name in (
                "state_coefficients",
                "gauge_delta",
                "shared_bias_coefficients",
                "view_bias_coefficients",
                "anchor_bias_coefficients",
                "identifiable_state_transform",
                "identifiable_fractions",
                "query_sensitivity_fractions",
                "robust_weights",
                "anchor_robust_weights",
            )
        }
        return frozen_finite_json_mapping(
            {
                "schema": STRUCTURED_GAUGE_AWARE_RESULT_SCHEMA,
                "schema_version": STRUCTURED_GAUGE_AWARE_RESULT_VERSION,
                "inference_admissible": self.inference_admissible,
                "reason": self.reason,
                "arrays": arrays,
                "covariance": dict(self.covariance.descriptor()),
                "diagnostics": plain_json(self.diagnostics),
                "input_lineage": plain_json(self.input_lineage),
            },
            name="structured gauge-aware result descriptor",
        )

    def materialize_posterior_covariance(
        self,
        *,
        maximum_bytes: int | None = None,
    ) -> np.ndarray:
        return self.covariance.materialize(maximum_bytes=maximum_bytes)

    def to_legacy(
        self,
        *,
        maximum_covariance_bytes: int | None = None,
    ) -> GaugeAwareBeliefResult:
        return GaugeAwareBeliefResult(
            inference_admissible=self.inference_admissible,
            reason=self.reason,
            state_coefficients=self.state_coefficients,
            gauge_delta=self.gauge_delta,
            shared_bias_coefficients=self.shared_bias_coefficients,
            view_bias_coefficients=self.view_bias_coefficients,
            anchor_bias_coefficients=self.anchor_bias_coefficients,
            posterior_covariance=self.materialize_posterior_covariance(
                maximum_bytes=maximum_covariance_bytes
            ),
            identifiable_state_transform=self.identifiable_state_transform,
            identifiable_fractions=self.identifiable_fractions,
            query_sensitivity_fractions=self.query_sensitivity_fractions,
            robust_weights=self.robust_weights,
            anchor_robust_weights=self.anchor_robust_weights,
            diagnostics=self.diagnostics,
            input_lineage=self.input_lineage,
        )

    @classmethod
    def from_legacy(
        cls,
        result: GaugeAwareBeliefResult,
    ) -> StructuredGaugeAwareBeliefResultV1:
        if not isinstance(result, GaugeAwareBeliefResult):
            raise TypeError("result must be a GaugeAwareBeliefResult")
        return cls(
            inference_admissible=result.inference_admissible,
            reason=result.reason,
            state_coefficients=result.state_coefficients,
            gauge_delta=result.gauge_delta,
            shared_bias_coefficients=result.shared_bias_coefficients,
            view_bias_coefficients=result.view_bias_coefficients,
            anchor_bias_coefficients=result.anchor_bias_coefficients,
            covariance=DenseCovarianceV1(result.posterior_covariance),
            identifiable_state_transform=result.identifiable_state_transform,
            identifiable_fractions=result.identifiable_fractions,
            query_sensitivity_fractions=result.query_sensitivity_fractions,
            robust_weights=result.robust_weights,
            anchor_robust_weights=result.anchor_robust_weights,
            diagnostics=result.diagnostics,
            input_lineage=result.input_lineage,
        )


__all__ = [
    "DENSE_COVARIANCE_REPRESENTATION",
    "PRECISION_BACKED_COVARIANCE_REPRESENTATION",
    "STRUCTURED_GAUGE_AWARE_RESULT_SCHEMA",
    "STRUCTURED_GAUGE_AWARE_RESULT_VERSION",
    "DenseCovarianceV1",
    "GaugeAwareCovarianceV1",
    "PrecisionBackedCovarianceV1",
    "StructuredGaugeAwareBeliefResultV1",
]
