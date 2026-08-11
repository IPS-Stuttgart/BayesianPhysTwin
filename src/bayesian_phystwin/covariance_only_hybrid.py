"""Compose an exact-reference-mean prediction with a donor covariance.

This experimental helper isolates uncertainty value from point-prediction value.
The caller-owned reference mean is returned by object identity; only covariance
is transplanted and optionally scaled by a frozen positive schedule.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    immutable_array,
    literal_lower_hex,
    plain_json,
)
from ._portable_contracts import content_id

COVARIANCE_ONLY_HYBRID_SCHEMA: Final = (
    "bayesian-phystwin-covariance-only-hybrid-record-v1"
)
COVARIANCE_ONLY_HYBRID_VERSION: Final = 1


def _canonical_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    if any(character in value for character in "\x00\r\n"):
        raise ValueError(f"{name} must be a single canonical line")
    return value


def _number(value: object, *, name: str, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _shape(value: Sequence[int], *, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a nonempty integer shape")
    try:
        source = tuple(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a nonempty integer shape") from error
    if not source:
        raise ValueError(f"{name} must be a nonempty integer shape")
    result: list[int] = []
    for index, item in enumerate(source):
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
            raise ValueError(f"{name}[{index}] must be a positive integer")
        dimension = int(item)
        if dimension < 1:
            raise ValueError(f"{name}[{index}] must be a positive integer")
        result.append(dimension)
    return tuple(result)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _reference_mean(value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("reference_mean_m must be a NumPy array to preserve identity")
    if value.dtype != np.dtype(np.float64):
        raise ValueError("reference_mean_m must have dtype float64")
    if value.ndim < 1 or value.shape[-1] < 1:
        raise ValueError("reference_mean_m must have shape (..., dimension)")
    if not value.flags.c_contiguous:
        raise ValueError("reference_mean_m must be C-contiguous")
    if not np.all(np.isfinite(value)):
        raise ValueError("reference_mean_m must be finite")
    return value


def _donor_covariance(value: object, *, mean_shape: tuple[int, ...]) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError("donor_covariance_m2 must contain real numeric values")
    covariance = np.asarray(raw, dtype=np.float64)
    expected = mean_shape + (mean_shape[-1],)
    if covariance.shape != expected:
        raise ValueError(
            "donor_covariance_m2 must have shape "
            f"{expected}, received {covariance.shape}"
        )
    if not np.all(np.isfinite(covariance)):
        raise ValueError("donor_covariance_m2 must be finite")
    return covariance


def _scale_schedule(value: object, *, shape: tuple[int, ...]) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError("covariance_scale must contain real numeric values")
    try:
        broadcast = np.broadcast_to(np.asarray(raw, dtype=np.float64), shape)
    except ValueError as error:
        raise ValueError(f"covariance_scale must broadcast to {shape}") from error
    if not np.all(np.isfinite(broadcast)) or np.any(broadcast <= 0.0):
        raise ValueError("covariance_scale must be finite and strictly positive")
    return np.array(broadcast, dtype=np.float64, copy=True, order="C")


def _symmetric_psd(value: np.ndarray, *, tolerance: float) -> np.ndarray:
    transposed = np.swapaxes(value, -1, -2)
    if not np.allclose(value, transposed, atol=tolerance, rtol=tolerance):
        raise ValueError("donor_covariance_m2 must be symmetric")
    symmetric = 0.5 * (value + transposed)
    if float(np.min(np.linalg.eigvalsh(symmetric), initial=0.0)) < -tolerance:
        raise ValueError("donor_covariance_m2 must be positive semidefinite")
    return symmetric


@dataclass(frozen=True, slots=True)
class CovarianceOnlyHybridRecordV1:
    """Content-addressed proof that only covariance changed."""

    reference_predictor_id: str
    covariance_donor_id: str
    mean_shape: Sequence[int]
    covariance_shape: Sequence[int]
    reference_mean_sha256: str
    donor_covariance_sha256: str
    scale_schedule_sha256: str
    output_covariance_sha256: str
    minimum_covariance_scale: float
    maximum_covariance_scale: float
    mean_object_identity_preserved: bool
    point_prediction_changed: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        mean_shape = _shape(self.mean_shape, name="mean_shape")
        covariance_shape = _shape(self.covariance_shape, name="covariance_shape")
        if covariance_shape != mean_shape + (mean_shape[-1],):
            raise ValueError("covariance_shape is incompatible with mean_shape")
        if self.mean_object_identity_preserved is not True:
            raise ValueError("mean_object_identity_preserved must remain true")
        if self.point_prediction_changed is not False:
            raise ValueError("point_prediction_changed must remain false")
        minimum = _number(
            self.minimum_covariance_scale,
            name="minimum_covariance_scale",
            positive=True,
        )
        maximum = _number(
            self.maximum_covariance_scale,
            name="maximum_covariance_scale",
            positive=True,
        )
        if maximum < minimum:
            raise ValueError(
                "maximum_covariance_scale must not be smaller than minimum"
            )
        object.__setattr__(
            self,
            "reference_predictor_id",
            _canonical_string(
                self.reference_predictor_id,
                name="reference_predictor_id",
            ),
        )
        object.__setattr__(
            self,
            "covariance_donor_id",
            _canonical_string(self.covariance_donor_id, name="covariance_donor_id"),
        )
        object.__setattr__(self, "mean_shape", mean_shape)
        object.__setattr__(self, "covariance_shape", covariance_shape)
        for field_name in (
            "reference_mean_sha256",
            "donor_covariance_sha256",
            "scale_schedule_sha256",
            "output_covariance_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                literal_lower_hex(
                    getattr(self, field_name),
                    name=field_name,
                    lengths={64},
                ),
            )
        object.__setattr__(self, "minimum_covariance_scale", minimum)
        object.__setattr__(self, "maximum_covariance_scale", maximum)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="metadata"),
        )
        expected = content_id(self.descriptor())
        if self.artifact_id is None:
            object.__setattr__(self, "artifact_id", expected)
        elif (
            literal_lower_hex(self.artifact_id, name="artifact_id", lengths={64})
            != expected
        ):
            raise ValueError("artifact_id does not match the hybrid record")

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": COVARIANCE_ONLY_HYBRID_SCHEMA,
            "schema_version": COVARIANCE_ONLY_HYBRID_VERSION,
            "reference_predictor_id": self.reference_predictor_id,
            "covariance_donor_id": self.covariance_donor_id,
            "mean_shape": list(self.mean_shape),
            "covariance_shape": list(self.covariance_shape),
            "reference_mean_sha256": self.reference_mean_sha256,
            "donor_covariance_sha256": self.donor_covariance_sha256,
            "scale_schedule_sha256": self.scale_schedule_sha256,
            "output_covariance_sha256": self.output_covariance_sha256,
            "minimum_covariance_scale": self.minimum_covariance_scale,
            "maximum_covariance_scale": self.maximum_covariance_scale,
            "mean_object_identity_preserved": self.mean_object_identity_preserved,
            "point_prediction_changed": self.point_prediction_changed,
            "metadata": plain_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class CovarianceOnlyHybridPredictionV1:
    """Prediction whose mean is the exact reference object."""

    mean_m: np.ndarray
    covariance_m2: np.ndarray
    record: CovarianceOnlyHybridRecordV1


def compose_covariance_only_hybrid(
    reference_mean_m: np.ndarray,
    donor_covariance_m2: object,
    *,
    reference_predictor_id: str,
    covariance_donor_id: str,
    covariance_scale: object = 1.0,
    covariance_psd_tolerance: float = 1e-10,
    metadata: Mapping[str, Any] | None = None,
) -> CovarianceOnlyHybridPredictionV1:
    """Return the exact reference mean with a scaled donor covariance.

    Observation noise is deliberately not added here; scoring or deployment must
    apply its separately frozen observation model.
    """

    mean = _reference_mean(reference_mean_m)
    tolerance = _number(
        covariance_psd_tolerance,
        name="covariance_psd_tolerance",
        positive=True,
    )
    donor = _symmetric_psd(
        _donor_covariance(donor_covariance_m2, mean_shape=mean.shape),
        tolerance=tolerance,
    )
    schedule = _scale_schedule(covariance_scale, shape=mean.shape[:-1])
    scaled = _symmetric_psd(
        donor * schedule[..., None, None],
        tolerance=tolerance,
    )
    output = immutable_array(scaled, dtype=np.float64)
    record = CovarianceOnlyHybridRecordV1(
        reference_predictor_id=reference_predictor_id,
        covariance_donor_id=covariance_donor_id,
        mean_shape=mean.shape,
        covariance_shape=output.shape,
        reference_mean_sha256=_array_sha256(mean),
        donor_covariance_sha256=_array_sha256(donor),
        scale_schedule_sha256=_array_sha256(schedule),
        output_covariance_sha256=_array_sha256(output),
        minimum_covariance_scale=float(np.min(schedule)),
        maximum_covariance_scale=float(np.max(schedule)),
        mean_object_identity_preserved=True,
        point_prediction_changed=False,
        metadata={} if metadata is None else metadata,
    )
    result = CovarianceOnlyHybridPredictionV1(
        mean_m=mean,
        covariance_m2=output,
        record=record,
    )
    if result.mean_m is not reference_mean_m:
        raise AssertionError("covariance-only composition copied the reference mean")
    return result
