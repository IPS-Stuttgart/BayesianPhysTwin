"""Content-addressed binding of a physical-query Jacobian to observation rows."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
from numpy.typing import NDArray

from ._canonical_contracts import frozen_finite_json_mapping, plain_json
from ._portable_contracts import (
    content_id,
    load_strict_json_object,
    require_exact_fields,
    sha256_digest,
    write_atomic_json,
)

FloatArray = NDArray[np.float64]

QUERY_JACOBIAN_BINDING_SCHEMA: Final = "bayesian_phystwin.query_jacobian_binding"
QUERY_JACOBIAN_BINDING_VERSION: Final = 1
OBSERVATION_ROW_BINDING_SCHEMA: Final = "phys4d.observation-row-binding"
OBSERVATION_ROW_BINDING_VERSION: Final = 1
QUERY_JACOBIAN_BINDING_CLAIM_BOUNDARY: Final = (
    "Target-blind query-lineage infrastructure only. This binding identifies the "
    "exact query-Jacobian bytes and ordered observation rows used by a covariance "
    "projection. It does not establish provider competence, calibrated uncertainty, "
    "physical-query benefit, Causal4D intervention benefit, deployment safety, or "
    "state of the art."
)

_BINDING_FIELDS: Final = frozenset(
    {
        "artifact_id",
        "schema",
        "schema_version",
        "query_name",
        "component_order",
        "physical_unit",
        "coordinate_frame",
        "source_observation_artifact_id",
        "provider_manifest_id",
        "causal_frame_stop",
        "query_jacobian",
        "observation_rows",
        "target_outcomes_used",
        "future_frames_used",
        "claim_boundary",
        "metadata",
    }
)
_ARRAY_DESCRIPTOR_FIELDS: Final = frozenset({"dtype", "shape", "sha256"})
_ROW_BINDING_FIELDS: Final = frozenset(
    {"schema", "schema_version", "count", "sha256"}
)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a JSON object with literal string keys")
    return cast(Mapping[str, Any], value)


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be canonical nonempty text")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _component_order(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("component_order must be a JSON array")
    result = tuple(_text(item, name="component_order entry") for item in value)
    if not result:
        raise ValueError("component_order must not be empty")
    if len(result) != len(set(result)):
        raise ValueError("component_order must not contain duplicates")
    return result


def _canonical_jacobian(value: object) -> FloatArray:
    raw = np.asarray(value, dtype=np.float64)
    if raw.ndim == 2:
        if raw.shape[1:] != (3,):
            raise ValueError("scalar query_jacobian must have shape (N, 3)")
        raw = raw[None, ...]
    elif raw.ndim != 3 or raw.shape[2] != 3:
        raise ValueError("query_jacobian must have shape (Q, N, 3) or (N, 3)")
    if raw.shape[0] < 1 or raw.shape[1] < 1:
        raise ValueError("query_jacobian must contain at least one query and one row")
    if not np.all(np.isfinite(raw)):
        raise ValueError("query_jacobian must be finite")
    result = np.ascontiguousarray(raw, dtype=np.dtype("<f8"))
    result.setflags(write=False)
    return cast(FloatArray, result)


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _row_ids(value: object, *, expected_count: int) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("row_ids must be a sequence of canonical strings")
    result = tuple(_text(item, name="row_ids entry") for item in value)
    if len(result) != expected_count:
        raise ValueError("row_ids length must equal the Jacobian observation count")
    if len(result) != len(set(result)):
        raise ValueError("row_ids must be unique")
    return result


def _row_ids_sha256(value: Sequence[str]) -> str:
    return content_id({"row_ids": list(value)})


def _validated_array_descriptor(value: object) -> tuple[tuple[int, int, int], str]:
    source = _mapping(value, name="query_jacobian descriptor")
    require_exact_fields(
        source,
        expected=_ARRAY_DESCRIPTOR_FIELDS,
        name="query_jacobian descriptor",
    )
    if source["dtype"] != "<f8":
        raise ValueError("query_jacobian dtype must be <f8")
    shape_raw = source["shape"]
    if isinstance(shape_raw, (str, bytes)) or not isinstance(shape_raw, Sequence):
        raise ValueError("query_jacobian shape must be a JSON array")
    shape = tuple(
        _integer(item, name=f"query_jacobian shape[{index}]", minimum=1)
        for index, item in enumerate(shape_raw)
    )
    if len(shape) != 3 or shape[2] != 3:
        raise ValueError("query_jacobian shape must be [Q, N, 3]")
    return cast(tuple[int, int, int], shape), sha256_digest(
        source["sha256"],
        name="query_jacobian sha256",
    )


def _validated_row_binding(value: object) -> tuple[int, str]:
    source = _mapping(value, name="observation_rows")
    require_exact_fields(
        source,
        expected=_ROW_BINDING_FIELDS,
        name="observation_rows",
    )
    if source["schema"] != OBSERVATION_ROW_BINDING_SCHEMA:
        raise ValueError("observation-row binding schema changed")
    if (
        _integer(
            source["schema_version"],
            name="observation-row binding schema_version",
            minimum=1,
        )
        != OBSERVATION_ROW_BINDING_VERSION
    ):
        raise ValueError("observation-row binding version changed")
    return (
        _integer(source["count"], name="observation row count", minimum=1),
        sha256_digest(source["sha256"], name="observation rows sha256"),
    )


@dataclass(frozen=True, slots=True)
class QueryJacobianBindingV1:
    """Immutable identity for one exact query Jacobian and ordered row roster."""

    query_name: str
    component_order: tuple[str, ...]
    physical_unit: str
    coordinate_frame: str
    source_observation_artifact_id: str
    provider_manifest_id: str
    causal_frame_stop: int
    query_jacobian_shape: tuple[int, int, int]
    query_jacobian_sha256: str
    observation_count: int
    row_ids_sha256: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        query_name = _text(self.query_name, name="query_name")
        components = _component_order(self.component_order)
        physical_unit = _text(self.physical_unit, name="physical_unit")
        coordinate_frame = _text(self.coordinate_frame, name="coordinate_frame")
        source_id = sha256_digest(
            self.source_observation_artifact_id,
            name="source_observation_artifact_id",
        )
        provider_id = sha256_digest(
            self.provider_manifest_id,
            name="provider_manifest_id",
        )
        causal_stop = _integer(
            self.causal_frame_stop,
            name="causal_frame_stop",
            minimum=1,
        )
        if type(self.query_jacobian_shape) is not tuple:
            raise ValueError("query_jacobian_shape must be a tuple")
        shape = tuple(
            _integer(item, name=f"query_jacobian_shape[{index}]", minimum=1)
            for index, item in enumerate(self.query_jacobian_shape)
        )
        if len(shape) != 3 or shape[2] != 3:
            raise ValueError("query_jacobian_shape must be (Q, N, 3)")
        query_dimension, observation_count, _ = shape
        if query_dimension != len(components):
            raise ValueError(
                "component_order length must equal query Jacobian dimension"
            )
        declared_count = _integer(
            self.observation_count,
            name="observation_count",
            minimum=1,
        )
        if declared_count != observation_count:
            raise ValueError("observation_count must equal query_jacobian_shape[1]")
        jacobian_sha = sha256_digest(
            self.query_jacobian_sha256,
            name="query_jacobian_sha256",
        )
        row_sha = sha256_digest(self.row_ids_sha256, name="row_ids_sha256")
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="query Jacobian binding metadata",
        )

        object.__setattr__(self, "query_name", query_name)
        object.__setattr__(self, "component_order", components)
        object.__setattr__(self, "physical_unit", physical_unit)
        object.__setattr__(self, "coordinate_frame", coordinate_frame)
        object.__setattr__(self, "source_observation_artifact_id", source_id)
        object.__setattr__(self, "provider_manifest_id", provider_id)
        object.__setattr__(self, "causal_frame_stop", causal_stop)
        object.__setattr__(
            self,
            "query_jacobian_shape",
            cast(tuple[int, int, int], shape),
        )
        object.__setattr__(self, "query_jacobian_sha256", jacobian_sha)
        object.__setattr__(self, "observation_count", declared_count)
        object.__setattr__(self, "row_ids_sha256", row_sha)
        object.__setattr__(self, "metadata", metadata)

        expected_id = content_id(self.descriptor())
        if self.artifact_id is not None:
            supplied_id = sha256_digest(self.artifact_id, name="artifact_id")
            if supplied_id != expected_id:
                raise ValueError("artifact_id does not match query Jacobian binding")
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def query_dimension(self) -> int:
        return self.query_jacobian_shape[0]

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": QUERY_JACOBIAN_BINDING_SCHEMA,
            "schema_version": QUERY_JACOBIAN_BINDING_VERSION,
            "query_name": self.query_name,
            "component_order": list(self.component_order),
            "physical_unit": self.physical_unit,
            "coordinate_frame": self.coordinate_frame,
            "source_observation_artifact_id": self.source_observation_artifact_id,
            "provider_manifest_id": self.provider_manifest_id,
            "causal_frame_stop": self.causal_frame_stop,
            "query_jacobian": {
                "dtype": "<f8",
                "shape": list(self.query_jacobian_shape),
                "sha256": self.query_jacobian_sha256,
            },
            "observation_rows": {
                "schema": OBSERVATION_ROW_BINDING_SCHEMA,
                "schema_version": OBSERVATION_ROW_BINDING_VERSION,
                "count": self.observation_count,
                "sha256": self.row_ids_sha256,
            },
            "target_outcomes_used": False,
            "future_frames_used": False,
            "claim_boundary": QUERY_JACOBIAN_BINDING_CLAIM_BOUNDARY,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {"artifact_id": self.artifact_id, **self.descriptor()}

    def validate_payload(self, query_jacobian: object, row_ids: object) -> FloatArray:
        """Validate actual Jacobian bytes and ordered rows against this binding."""

        jacobian = _canonical_jacobian(query_jacobian)
        if tuple(jacobian.shape) != self.query_jacobian_shape:
            raise ValueError("query_jacobian shape differs from its binding")
        if _array_sha256(jacobian) != self.query_jacobian_sha256:
            raise ValueError("query_jacobian bytes differ from their binding")
        rows = _row_ids(row_ids, expected_count=self.observation_count)
        if _row_ids_sha256(rows) != self.row_ids_sha256:
            raise ValueError("row_ids differ from their binding")
        return jacobian

    @classmethod
    def from_mapping(cls, value: object) -> QueryJacobianBindingV1:
        source = _mapping(value, name="query Jacobian binding")
        require_exact_fields(
            source,
            expected=_BINDING_FIELDS,
            name="query Jacobian binding",
        )
        if source["schema"] != QUERY_JACOBIAN_BINDING_SCHEMA:
            raise ValueError("query Jacobian binding schema changed")
        if (
            _integer(
                source["schema_version"],
                name="query Jacobian binding schema_version",
                minimum=1,
            )
            != QUERY_JACOBIAN_BINDING_VERSION
        ):
            raise ValueError("query Jacobian binding version changed")
        if source["target_outcomes_used"] is not False:
            raise ValueError("query Jacobian binding must be target blind")
        if source["future_frames_used"] is not False:
            raise ValueError("query Jacobian binding must be causal-prefix only")
        if source["claim_boundary"] != QUERY_JACOBIAN_BINDING_CLAIM_BOUNDARY:
            raise ValueError("query Jacobian binding claim boundary changed")
        shape, jacobian_sha = _validated_array_descriptor(source["query_jacobian"])
        row_count, row_sha = _validated_row_binding(source["observation_rows"])
        return cls(
            query_name=cast(str, source["query_name"]),
            component_order=_component_order(source["component_order"]),
            physical_unit=cast(str, source["physical_unit"]),
            coordinate_frame=cast(str, source["coordinate_frame"]),
            source_observation_artifact_id=cast(
                str,
                source["source_observation_artifact_id"],
            ),
            provider_manifest_id=cast(str, source["provider_manifest_id"]),
            causal_frame_stop=cast(int, source["causal_frame_stop"]),
            query_jacobian_shape=shape,
            query_jacobian_sha256=jacobian_sha,
            observation_count=row_count,
            row_ids_sha256=row_sha,
            metadata=_mapping(source["metadata"], name="metadata"),
            artifact_id=cast(str, source["artifact_id"]),
        )


def build_query_jacobian_binding(
    *,
    query_name: str,
    component_order: Sequence[str],
    physical_unit: str,
    coordinate_frame: str,
    source_observation_artifact_id: str,
    provider_manifest_id: str,
    causal_frame_stop: int,
    query_jacobian: object,
    row_ids: Sequence[str],
    metadata: Mapping[str, Any] | None = None,
) -> QueryJacobianBindingV1:
    """Build a target-blind content identity from actual Jacobian and row bytes."""

    jacobian = _canonical_jacobian(query_jacobian)
    rows = _row_ids(row_ids, expected_count=int(jacobian.shape[1]))
    return QueryJacobianBindingV1(
        query_name=query_name,
        component_order=tuple(component_order),
        physical_unit=physical_unit,
        coordinate_frame=coordinate_frame,
        source_observation_artifact_id=source_observation_artifact_id,
        provider_manifest_id=provider_manifest_id,
        causal_frame_stop=causal_frame_stop,
        query_jacobian_shape=cast(tuple[int, int, int], tuple(jacobian.shape)),
        query_jacobian_sha256=_array_sha256(jacobian),
        observation_count=int(jacobian.shape[1]),
        row_ids_sha256=_row_ids_sha256(rows),
        metadata={} if metadata is None else metadata,
    )


def load_query_jacobian_binding(path: str | Path) -> QueryJacobianBindingV1:
    return QueryJacobianBindingV1.from_mapping(
        load_strict_json_object(path, label="query Jacobian binding")
    )


def write_query_jacobian_binding(
    binding: QueryJacobianBindingV1,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(binding, QueryJacobianBindingV1):
        raise TypeError("binding must be a QueryJacobianBindingV1")
    write_atomic_json(binding.to_record(), path, overwrite=overwrite)


__all__ = [
    "OBSERVATION_ROW_BINDING_SCHEMA",
    "OBSERVATION_ROW_BINDING_VERSION",
    "QUERY_JACOBIAN_BINDING_CLAIM_BOUNDARY",
    "QUERY_JACOBIAN_BINDING_SCHEMA",
    "QUERY_JACOBIAN_BINDING_VERSION",
    "QueryJacobianBindingV1",
    "build_query_jacobian_binding",
    "load_query_jacobian_binding",
    "write_query_jacobian_binding",
]
