"""Block-local plus labeled low-rank covariance for correlated 3-D points.

Per-point ``(3, 3)`` covariance blocks cannot represent coherent translation,
bending, camera-bias, gauge, process, or between-model uncertainty shared by
several points. This module provides a compact additive representation

``Sigma = blockdiag(D_1, ..., D_N) + sum_c U_c U_c.T``

and propagates it into registered physical queries without materializing the
full ``(3N, 3N)`` covariance. The local blocks are explicitly conditional on,
and therefore exclude, every retained shared component. Distinct named
components are additive; correlated shared modes must be placed in the same
factor root rather than split across component names.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final, Literal, cast

import numpy as np

from ._canonical_contracts import (
    canonical_string_tuple,
    frozen_finite_json_mapping,
    genuine_integer,
    literal_lower_hex,
    plain_json,
)
from ._portable_contracts import content_id

STRUCTURED_POINT_COVARIANCE_SCHEMA: Final = (
    "bayesian_phystwin.structured_point_covariance"
)
STRUCTURED_POINT_COVARIANCE_VERSION: Final = 1
STRUCTURED_POINT_COVARIANCE_SEMANTICS: Final = (
    "conditional-local-blocks-plus-independent-labeled-low-rank-roots-v1"
)

SharedCovarianceComponent = Literal[
    "discrepancy",
    "camera_bias",
    "gauge",
    "between_model",
    "process",
]
SHARED_COVARIANCE_COMPONENTS: tuple[SharedCovarianceComponent, ...] = (
    "discrepancy",
    "camera_bias",
    "gauge",
    "between_model",
    "process",
)

_SYMMETRY_ABSOLUTE_TOLERANCE = 1e-12
_SYMMETRY_RELATIVE_TOLERANCE = 1e-10
_PSD_ABSOLUTE_TOLERANCE = 1e-12
_PSD_RELATIVE_TOLERANCE = 1e-10


def _literal_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(
            f"{name} must be a nonempty literal string without surrounding whitespace"
        )
    return value


def _component_name(value: object) -> SharedCovarianceComponent:
    if type(value) is not str or value not in SHARED_COVARIANCE_COMPONENTS:
        raise ValueError(
            "shared covariance component must be one of "
            f"{list(SHARED_COVARIANCE_COMPONENTS)}"
        )
    return cast(SharedCovarianceComponent, value)


def _real_float64_array(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    array = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _immutable_float64(value: object) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    payload = array.tobytes(order="C")
    return np.frombuffer(payload, dtype=np.float64).reshape(array.shape)


def _array_record(value: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def _validated_local_covariance(value: object, *, point_count: int) -> np.ndarray:
    local = _real_float64_array(value, name="local_covariance_m2")
    if local.shape != (point_count, 3, 3):
        raise ValueError("local_covariance_m2 must have shape (point, 3, 3)")
    if not np.allclose(
        local,
        np.swapaxes(local, 1, 2),
        atol=_SYMMETRY_ABSOLUTE_TOLERANCE,
        rtol=_SYMMETRY_RELATIVE_TOLERANCE,
    ):
        raise ValueError("local_covariance_m2 must be symmetric")
    symmetric = 0.5 * (local + np.swapaxes(local, 1, 2))
    eigenvalues = np.linalg.eigvalsh(symmetric)
    scale = np.maximum(np.max(np.abs(eigenvalues), axis=1), 1.0)
    tolerance = _PSD_ABSOLUTE_TOLERANCE + _PSD_RELATIVE_TOLERANCE * scale
    if np.any(eigenvalues[:, 0] < -tolerance):
        raise ValueError("local_covariance_m2 must be positive semidefinite")
    return symmetric


def _validated_factors(
    value: Mapping[str, np.ndarray],
    *,
    point_count: int,
) -> Mapping[str, np.ndarray]:
    if not isinstance(value, Mapping):
        raise ValueError("shared_factors_m must be a mapping")
    factors: dict[str, np.ndarray] = {}
    for raw_name, raw_factor in value.items():
        component = _component_name(raw_name)
        factor = _real_float64_array(
            raw_factor,
            name=f"shared_factors_m[{component}]",
        )
        if factor.ndim != 3 or factor.shape[:2] != (point_count, 3):
            raise ValueError(
                f"shared_factors_m[{component}] must have shape (point, 3, rank)"
            )
        if factor.shape[2] == 0:
            raise ValueError(
                f"shared_factors_m[{component}] must have positive retained rank"
            )
        factors[component] = _immutable_float64(factor)
    return MappingProxyType({name: factors[name] for name in sorted(factors)})


def _validated_query_jacobian(
    value: object,
    *,
    point_count: int,
) -> np.ndarray:
    jacobian = _real_float64_array(value, name="query_jacobian")
    if jacobian.ndim == 2:
        if jacobian.shape[1] != 3 * point_count:
            raise ValueError(
                "flat query_jacobian must have shape (query, 3 * point_count)"
            )
        jacobian = jacobian.reshape((jacobian.shape[0], point_count, 3))
    elif jacobian.ndim == 3:
        if jacobian.shape[1:] != (point_count, 3):
            raise ValueError("query_jacobian must have shape (query, point_count, 3)")
    else:
        raise ValueError(
            "query_jacobian must have shape (query, point_count, 3) or "
            "(query, 3 * point_count)"
        )
    if jacobian.shape[0] == 0:
        raise ValueError("query_jacobian must contain at least one query row")
    return jacobian


@dataclass(frozen=True, slots=True)
class ProjectedStructuredPointCovarianceV1:
    """Exact query-space decomposition produced by structured projection."""

    local_covariance_m2: np.ndarray
    shared_component_factors_m: Mapping[str, np.ndarray]
    shared_component_covariances_m2: Mapping[str, np.ndarray]
    total_covariance_m2: np.ndarray

    @property
    def query_dimension(self) -> int:
        return int(self.total_covariance_m2.shape[0])

    @property
    def shared_component_names(self) -> tuple[str, ...]:
        return tuple(self.shared_component_covariances_m2)


@dataclass(frozen=True, slots=True)
class StructuredPointCovarianceV1:
    """Content-addressed covariance over identified metric 3-D points."""

    point_ids: tuple[str, ...]
    local_covariance_m2: np.ndarray
    shared_factors_m: Mapping[str, np.ndarray]
    coordinate_frame: str
    source_artifact_id: str
    calibration_artifact_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        point_ids = canonical_string_tuple(
            self.point_ids,
            name="point_ids",
            allow_empty=False,
        )
        if len(set(point_ids)) != len(point_ids):
            raise ValueError("point_ids must be unique")
        coordinate_frame = _literal_string(
            self.coordinate_frame,
            name="coordinate_frame",
        )
        source_artifact_id = literal_lower_hex(
            self.source_artifact_id,
            name="source_artifact_id",
            lengths={64},
        )
        calibration_artifact_id = self.calibration_artifact_id
        if calibration_artifact_id is not None:
            calibration_artifact_id = literal_lower_hex(
                calibration_artifact_id,
                name="calibration_artifact_id",
                lengths={64},
            )
        local_covariance = _validated_local_covariance(
            self.local_covariance_m2,
            point_count=len(point_ids),
        )
        shared_factors = _validated_factors(
            self.shared_factors_m,
            point_count=len(point_ids),
        )
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="structured point covariance metadata",
        )

        object.__setattr__(self, "point_ids", point_ids)
        object.__setattr__(self, "coordinate_frame", coordinate_frame)
        object.__setattr__(self, "source_artifact_id", source_artifact_id)
        object.__setattr__(
            self,
            "calibration_artifact_id",
            calibration_artifact_id,
        )
        object.__setattr__(
            self,
            "local_covariance_m2",
            _immutable_float64(local_covariance),
        )
        object.__setattr__(self, "shared_factors_m", shared_factors)
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
                    "structured point covariance artifact_id does not match content"
                )
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def point_count(self) -> int:
        return len(self.point_ids)

    @property
    def state_dimension(self) -> int:
        return 3 * self.point_count

    @property
    def shared_component_names(self) -> tuple[str, ...]:
        return tuple(self.shared_factors_m)

    @property
    def shared_rank(self) -> int:
        return sum(factor.shape[2] for factor in self.shared_factors_m.values())

    def descriptor(self) -> dict[str, object]:
        """Return the content-addressed descriptor without dense covariance bytes."""

        return {
            "schema": STRUCTURED_POINT_COVARIANCE_SCHEMA,
            "schema_version": STRUCTURED_POINT_COVARIANCE_VERSION,
            "semantics": STRUCTURED_POINT_COVARIANCE_SEMANTICS,
            "point_ids": list(self.point_ids),
            "coordinate_frame": self.coordinate_frame,
            "source_artifact_id": self.source_artifact_id,
            "calibration_artifact_id": self.calibration_artifact_id,
            "local_covariance_m2": _array_record(self.local_covariance_m2),
            "shared_factors_m": {
                name: _array_record(factor)
                for name, factor in self.shared_factors_m.items()
            },
            "metadata": plain_json(self.metadata),
        }

    def summary(self) -> dict[str, object]:
        return {
            "schema": STRUCTURED_POINT_COVARIANCE_SCHEMA,
            "schema_version": STRUCTURED_POINT_COVARIANCE_VERSION,
            "artifact_id": self.artifact_id,
            "point_count": self.point_count,
            "state_dimension": self.state_dimension,
            "shared_rank": self.shared_rank,
            "shared_component_ranks": {
                name: int(factor.shape[2])
                for name, factor in self.shared_factors_m.items()
            },
            "coordinate_frame": self.coordinate_frame,
            "source_artifact_id": self.source_artifact_id,
            "calibration_artifact_id": self.calibration_artifact_id,
            "local_blocks_exclude_shared_components": True,
        }

    def marginal_covariance_m2(self) -> np.ndarray:
        """Return per-point marginal blocks including every shared component."""

        marginal = np.array(self.local_covariance_m2, copy=True)
        for factor in self.shared_factors_m.values():
            marginal += np.einsum("nir,njr->nij", factor, factor, optimize=True)
        return _immutable_float64(marginal)

    def cross_covariance_m2(
        self,
        first_point_id: str,
        second_point_id: str,
    ) -> np.ndarray:
        """Return the shared cross-covariance between two identified points."""

        first = _literal_string(first_point_id, name="first_point_id")
        second = _literal_string(second_point_id, name="second_point_id")
        try:
            first_index = self.point_ids.index(first)
            second_index = self.point_ids.index(second)
        except ValueError as error:
            raise ValueError("cross-covariance point ID is not present") from error
        covariance = np.zeros((3, 3), dtype=np.float64)
        if first_index == second_index:
            covariance += self.local_covariance_m2[first_index]
        for factor in self.shared_factors_m.values():
            covariance += factor[first_index] @ factor[second_index].T
        return _immutable_float64(covariance)

    def dense_covariance_m2(self, *, maximum_dimension: int = 4096) -> np.ndarray:
        """Materialize the full covariance for bounded diagnostics only."""

        limit = genuine_integer(
            maximum_dimension,
            name="maximum_dimension",
            minimum=1,
        )
        if self.state_dimension > limit:
            raise ValueError(
                "structured covariance exceeds the requested dense dimension limit"
            )
        dense = np.zeros(
            (self.state_dimension, self.state_dimension),
            dtype=np.float64,
        )
        for index, block in enumerate(self.local_covariance_m2):
            start = 3 * index
            dense[start : start + 3, start : start + 3] = block
        for factor in self.shared_factors_m.values():
            flat_factor = factor.reshape((self.state_dimension, factor.shape[2]))
            dense += flat_factor @ flat_factor.T
        dense = 0.5 * (dense + dense.T)
        return _immutable_float64(dense)

    def project_query_covariance(
        self,
        query_jacobian: np.ndarray,
    ) -> ProjectedStructuredPointCovarianceV1:
        """Propagate local and shared covariance into one linearized query."""

        jacobian = _validated_query_jacobian(
            query_jacobian,
            point_count=self.point_count,
        )
        local = np.einsum(
            "qni,nij,rnj->qr",
            jacobian,
            self.local_covariance_m2,
            jacobian,
            optimize=True,
        )
        local = 0.5 * (local + local.T)
        projected_factors: dict[str, np.ndarray] = {}
        component_covariances: dict[str, np.ndarray] = {}
        total = np.array(local, copy=True)
        for name, factor in self.shared_factors_m.items():
            projected = np.einsum(
                "qni,nir->qr",
                jacobian,
                factor,
                optimize=True,
            )
            component = projected @ projected.T
            component = 0.5 * (component + component.T)
            projected_factors[name] = _immutable_float64(projected)
            component_covariances[name] = _immutable_float64(component)
            total += component
        total = 0.5 * (total + total.T)
        return ProjectedStructuredPointCovarianceV1(
            local_covariance_m2=_immutable_float64(local),
            shared_component_factors_m=MappingProxyType(projected_factors),
            shared_component_covariances_m2=MappingProxyType(component_covariances),
            total_covariance_m2=_immutable_float64(total),
        )


__all__ = [
    "ProjectedStructuredPointCovarianceV1",
    "SHARED_COVARIANCE_COMPONENTS",
    "STRUCTURED_POINT_COVARIANCE_SCHEMA",
    "STRUCTURED_POINT_COVARIANCE_SEMANTICS",
    "STRUCTURED_POINT_COVARIANCE_VERSION",
    "SharedCovarianceComponent",
    "StructuredPointCovarianceV1",
]
