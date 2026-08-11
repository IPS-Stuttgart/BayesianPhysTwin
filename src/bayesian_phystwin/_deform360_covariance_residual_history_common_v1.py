"""Shared policy, camera, and validation contracts for the source dry run."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from ._canonical_contracts import (
    genuine_integer,
    immutable_array,
    immutable_integer_array,
    literal_lower_hex,
)
from ._portable_contracts import content_id

RESIDUAL_HISTORY_POLICY_SCHEMA: Final = (
    "bayesian-phystwin/deform360-covariance-residual-history-policy-v1"
)
CAMERA_PARTITION_SCHEMA: Final = (
    "bayesian-phystwin/deform360-disjoint-camera-partition-v1"
)
RESIDUAL_HISTORY_ADAPTER_SCHEMA: Final = (
    "bayesian-phystwin/deform360-residual-history-adapter-v1"
)
RESIDUAL_HISTORY_DECISION_SCHEMA: Final = (
    "bayesian-phystwin/deform360-residual-history-dry-run-decision-v1"
)
SCHEMA_VERSION: Final = 1
CAMERA_PARTITION_NAMESPACE: Final = (
    "deform360-provider-scoring-camera-family-v1"
)
HORIZON_LABELS: Final = ("early", "middle", "late")
TARGET_QUARANTINE_ROOT: Final = Path(
    "/mnt/lexar4tb/datasets/deform360/unopened-candidate-target/"
    "covariance-only-v1/payload"
)
RESIDUAL_STORAGE_SEMANTICS: Final = "zero-outside-validity-mask-no-fill-v1"
REFERENCE_MEAN_SEMANTICS: Final = (
    "exact-final-prefix-residual-on-same-material-identity-v1"
)
FALLBACK_SEMANTICS: Final = (
    "exact-caller-owned-physical-future-mean-and-covariance-v1"
)
CLAIM_BOUNDARY: Final = (
    "This source-only dry run validates residual-history shape, validity, material "
    "identity, horizon, covariance, disjoint-camera, and exact-fallback contracts. "
    "It opens no fresh-target payload or outcome and does not establish fresh-object "
    "calibration, point accuracy, official Deform360 benchmark parity, Prob4D or "
    "Causal4D benefit, deployment safety, or state of the art."
)
_CAMERA_RE = re.compile(r"^(?P<family>.+)_cam(?P<index>[0-9]+)$")


def _canonical_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    if any(character in value for character in "\x00\r\n"):
        raise ValueError(f"{name} must be a single canonical line")
    return value


def _required_sha256(value: str | None, *, name: str) -> str:
    if value is None:
        raise AssertionError(f"{name} was not materialized")
    return literal_lower_hex(value, name=name, lengths={64})


def _finite_real(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if strictly_positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _camera_tuple(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of camera IDs")
    result = tuple(_canonical_string(value, name=name) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    if result != tuple(sorted(result)):
        raise ValueError(f"{name} must be sorted")
    for camera_id in result:
        camera_hardware_family(camera_id)
    return result


def camera_hardware_family(camera_id: str) -> str:
    """Return the physical recorder family for one released camera stream."""

    value = _canonical_string(camera_id, name="camera_id")
    match = _CAMERA_RE.fullmatch(value)
    if match is None:
        raise ValueError("camera_id must end in '_cam<nonnegative integer>'")
    return match.group("family")


def _rank_family(family: str) -> str:
    return hashlib.sha256(
        CAMERA_PARTITION_NAMESPACE.encode("utf-8")
        + b"\0"
        + family.encode("utf-8")
    ).hexdigest()


def _readonly_float_array(
    value: object,
    *,
    name: str,
    ndim: int,
    finite: bool = True,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if finite and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return immutable_array(array, dtype=np.float64)


def _boolean_array(value: object, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype != np.dtype(bool):
        raise ValueError(f"{name} must have Boolean dtype")
    if raw.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return immutable_array(raw, dtype=bool)


def _integer_vector(value: object, *, name: str) -> np.ndarray:
    result = immutable_integer_array(value, name=name)
    if result.ndim != 1 or len(result) == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    return result


def _validate_covariance(
    value: object,
    *,
    name: str,
    expected_shape: tuple[int, ...],
    preserve_identity: bool,
) -> np.ndarray:
    if preserve_identity and not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array to preserve identity")
    raw = np.asarray(value)
    if raw.dtype != np.dtype(np.float64):
        raise ValueError(f"{name} must have dtype float64")
    if raw.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")
    if preserve_identity and not raw.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} must be finite")
    transposed = np.swapaxes(raw, -1, -2)
    if not np.allclose(raw, transposed, atol=1e-10, rtol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    symmetric = 0.5 * (raw + transposed)
    if float(np.min(np.linalg.eigvalsh(symmetric), initial=0.0)) < -1e-10:
        raise ValueError(f"{name} must be positive semidefinite")
    return raw if preserve_identity else immutable_array(symmetric, dtype=np.float64)


def assert_outside_target_quarantine(
    path: str | Path,
    *,
    target_root: str | Path = TARGET_QUARANTINE_ROOT,
    name: str = "path",
) -> Path:
    """Reject any read or write path inside the unopened target quarantine."""

    candidate = Path(path).expanduser().resolve()
    root = Path(target_root).expanduser().resolve()
    if candidate == root or candidate.is_relative_to(root):
        raise ValueError(f"{name} must remain outside the unopened target quarantine")
    return candidate


@dataclass(frozen=True, slots=True)
class ResidualHistoryDryRunPolicyV1:
    """Frozen technical support policy for the opened-source dry run."""

    minimum_prefix_frames: int
    minimum_final_observed_count: int
    minimum_final_observed_fraction: float
    minimum_cameras_per_role: int
    minimum_camera_families_per_role: int
    covariance_scales: tuple[float, float, float]
    metric_unit: str = "metre"
    covariance_unit: str = "metre_squared"
    residual_storage_semantics: str = RESIDUAL_STORAGE_SEMANTICS
    reference_mean_semantics: str = REFERENCE_MEAN_SEMANTICS
    fallback_semantics: str = FALLBACK_SEMANTICS
    policy_id: str = field(init=False)

    def __post_init__(self) -> None:
        minimum_prefix_frames = genuine_integer(
            self.minimum_prefix_frames,
            name="minimum_prefix_frames",
            minimum=2,
        )
        minimum_final_observed_count = genuine_integer(
            self.minimum_final_observed_count,
            name="minimum_final_observed_count",
            minimum=1,
        )
        minimum_cameras_per_role = genuine_integer(
            self.minimum_cameras_per_role,
            name="minimum_cameras_per_role",
            minimum=2,
        )
        minimum_families_per_role = genuine_integer(
            self.minimum_camera_families_per_role,
            name="minimum_camera_families_per_role",
            minimum=2,
        )
        fraction = _finite_real(
            self.minimum_final_observed_fraction,
            name="minimum_final_observed_fraction",
            strictly_positive=True,
        )
        if fraction > 1.0:
            raise ValueError("minimum_final_observed_fraction must not exceed one")
        if (
            type(self.covariance_scales) is not tuple
            or len(self.covariance_scales) != 3
        ):
            raise ValueError(
                "covariance_scales must be a canonical early/middle/late tuple"
            )
        scales = tuple(
            _finite_real(
                value,
                name=f"covariance_scales[{index}]",
                strictly_positive=True,
            )
            for index, value in enumerate(self.covariance_scales)
        )
        if self.metric_unit != "metre" or self.covariance_unit != "metre_squared":
            raise ValueError("residual and covariance units must remain metric")
        if self.residual_storage_semantics != RESIDUAL_STORAGE_SEMANTICS:
            raise ValueError("residual storage semantics changed")
        if self.reference_mean_semantics != REFERENCE_MEAN_SEMANTICS:
            raise ValueError("reference mean semantics changed")
        if self.fallback_semantics != FALLBACK_SEMANTICS:
            raise ValueError("fallback semantics changed")
        object.__setattr__(self, "minimum_prefix_frames", minimum_prefix_frames)
        object.__setattr__(
            self,
            "minimum_final_observed_count",
            minimum_final_observed_count,
        )
        object.__setattr__(self, "minimum_final_observed_fraction", fraction)
        object.__setattr__(self, "minimum_cameras_per_role", minimum_cameras_per_role)
        object.__setattr__(
            self,
            "minimum_camera_families_per_role",
            minimum_families_per_role,
        )
        object.__setattr__(self, "covariance_scales", scales)
        object.__setattr__(self, "policy_id", content_id(self.descriptor()))

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": RESIDUAL_HISTORY_POLICY_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "minimum_prefix_frames": self.minimum_prefix_frames,
            "minimum_final_observed_count": self.minimum_final_observed_count,
            "minimum_final_observed_fraction": self.minimum_final_observed_fraction,
            "minimum_cameras_per_role": self.minimum_cameras_per_role,
            "minimum_camera_families_per_role": (
                self.minimum_camera_families_per_role
            ),
            "horizon_labels": list(HORIZON_LABELS),
            "covariance_scales": list(self.covariance_scales),
            "metric_unit": self.metric_unit,
            "covariance_unit": self.covariance_unit,
            "residual_storage_semantics": self.residual_storage_semantics,
            "reference_mean_semantics": self.reference_mean_semantics,
            "fallback_semantics": self.fallback_semantics,
        }


@dataclass(frozen=True, slots=True)
class DisjointCameraPartitionV1:
    """One deterministic provider/scoring split over physical camera families."""

    all_camera_ids: tuple[str, ...]
    provider_camera_ids: tuple[str, ...]
    scoring_camera_ids: tuple[str, ...]
    provider_family_ids: tuple[str, ...]
    scoring_family_ids: tuple[str, ...]
    minimum_cameras_per_role: int
    minimum_families_per_role: int
    namespace: str = CAMERA_PARTITION_NAMESPACE
    partition_id: str | None = None

    def __post_init__(self) -> None:
        all_cameras = _camera_tuple(self.all_camera_ids, name="all_camera_ids")
        provider = _camera_tuple(
            self.provider_camera_ids,
            name="provider_camera_ids",
        )
        scoring = _camera_tuple(self.scoring_camera_ids, name="scoring_camera_ids")
        provider_families = tuple(
            _canonical_string(value, name="provider_family_ids")
            for value in self.provider_family_ids
        )
        scoring_families = tuple(
            _canonical_string(value, name="scoring_family_ids")
            for value in self.scoring_family_ids
        )
        if provider_families != tuple(sorted(set(provider_families))):
            raise ValueError("provider_family_ids must be sorted and unique")
        if scoring_families != tuple(sorted(set(scoring_families))):
            raise ValueError("scoring_family_ids must be sorted and unique")
        if set(provider) & set(scoring):
            raise ValueError("provider and scoring camera sets must be disjoint")
        if set(provider) | set(scoring) != set(all_cameras):
            raise ValueError("provider and scoring cameras must exhaust all cameras")
        expected_provider_families = {
            camera_hardware_family(camera) for camera in provider
        }
        expected_scoring_families = {
            camera_hardware_family(camera) for camera in scoring
        }
        if expected_provider_families != set(provider_families):
            raise ValueError("provider family roster differs from provider cameras")
        if expected_scoring_families != set(scoring_families):
            raise ValueError("scoring family roster differs from scoring cameras")
        if expected_provider_families & expected_scoring_families:
            raise ValueError(
                "one physical camera family crosses provider/scoring roles"
            )
        minimum_cameras = genuine_integer(
            self.minimum_cameras_per_role,
            name="minimum_cameras_per_role",
            minimum=2,
        )
        minimum_families = genuine_integer(
            self.minimum_families_per_role,
            name="minimum_families_per_role",
            minimum=2,
        )
        if len(provider) < minimum_cameras or len(scoring) < minimum_cameras:
            raise ValueError("one camera role is below minimum camera support")
        if (
            len(provider_families) < minimum_families
            or len(scoring_families) < minimum_families
        ):
            raise ValueError("one camera role is below minimum family support")
        if self.namespace != CAMERA_PARTITION_NAMESPACE:
            raise ValueError("camera partition namespace changed")
        object.__setattr__(self, "all_camera_ids", all_cameras)
        object.__setattr__(self, "provider_camera_ids", provider)
        object.__setattr__(self, "scoring_camera_ids", scoring)
        object.__setattr__(self, "provider_family_ids", provider_families)
        object.__setattr__(self, "scoring_family_ids", scoring_families)
        object.__setattr__(self, "minimum_cameras_per_role", minimum_cameras)
        object.__setattr__(self, "minimum_families_per_role", minimum_families)
        expected = content_id(self.descriptor())
        if self.partition_id is None:
            object.__setattr__(self, "partition_id", expected)
        elif literal_lower_hex(
            self.partition_id,
            name="partition_id",
            lengths={64},
        ) != expected:
            raise ValueError("partition_id does not match the camera partition")

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": CAMERA_PARTITION_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "namespace": self.namespace,
            "all_camera_ids": list(self.all_camera_ids),
            "provider_camera_ids": list(self.provider_camera_ids),
            "scoring_camera_ids": list(self.scoring_camera_ids),
            "provider_family_ids": list(self.provider_family_ids),
            "scoring_family_ids": list(self.scoring_family_ids),
            "minimum_cameras_per_role": self.minimum_cameras_per_role,
            "minimum_families_per_role": self.minimum_families_per_role,
        }


def deterministic_disjoint_camera_partition(
    camera_ids: Sequence[str],
    *,
    policy: ResidualHistoryDryRunPolicyV1,
) -> DisjointCameraPartitionV1:
    """Split whole recorder families into balanced, disjoint camera roles."""

    if not isinstance(policy, ResidualHistoryDryRunPolicyV1):
        raise TypeError("policy must be ResidualHistoryDryRunPolicyV1")
    if isinstance(camera_ids, (str, bytes)):
        raise ValueError("camera_ids must be a sequence")
    canonical = tuple(
        sorted(_canonical_string(value, name="camera_ids") for value in camera_ids)
    )
    if not canonical or len(set(canonical)) != len(canonical):
        raise ValueError("camera_ids must be nonempty and unique")
    families: dict[str, list[str]] = {}
    for camera in canonical:
        families.setdefault(camera_hardware_family(camera), []).append(camera)
    ranked = sorted(families, key=lambda family: (_rank_family(family), family))
    provider_families: list[str] = []
    scoring_families: list[str] = []
    provider_cameras: list[str] = []
    scoring_cameras: list[str] = []
    for family in ranked:
        cameras = sorted(families[family])
        provider_cost = (len(provider_cameras), len(provider_families), 0)
        scoring_cost = (len(scoring_cameras), len(scoring_families), 1)
        if provider_cost <= scoring_cost:
            provider_families.append(family)
            provider_cameras.extend(cameras)
        else:
            scoring_families.append(family)
            scoring_cameras.extend(cameras)
    return DisjointCameraPartitionV1(
        all_camera_ids=canonical,
        provider_camera_ids=tuple(sorted(provider_cameras)),
        scoring_camera_ids=tuple(sorted(scoring_cameras)),
        provider_family_ids=tuple(sorted(provider_families)),
        scoring_family_ids=tuple(sorted(scoring_families)),
        minimum_cameras_per_role=policy.minimum_cameras_per_role,
        minimum_families_per_role=policy.minimum_camera_families_per_role,
    )


__all__ = [
    "CAMERA_PARTITION_NAMESPACE",
    "CLAIM_BOUNDARY",
    "DisjointCameraPartitionV1",
    "FALLBACK_SEMANTICS",
    "HORIZON_LABELS",
    "REFERENCE_MEAN_SEMANTICS",
    "RESIDUAL_STORAGE_SEMANTICS",
    "ResidualHistoryDryRunPolicyV1",
    "TARGET_QUARANTINE_ROOT",
    "assert_outside_target_quarantine",
    "camera_hardware_family",
    "deterministic_disjoint_camera_partition",
]
