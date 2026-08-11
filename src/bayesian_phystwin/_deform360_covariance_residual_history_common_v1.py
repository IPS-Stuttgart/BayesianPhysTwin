"""Shared policy, camera, and validation contracts for the source dry run."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from ._canonical_contracts import (
    genuine_integer,
    immutable_array,
    immutable_integer_array,
    literal_lower_hex,
)
from ._portable_contracts import canonical_json_bytes, content_id

SCHEMA_VERSION: Final = 1
RESIDUAL_HISTORY_POLICY_SCHEMA: Final = (
    "bayesian-phystwin.deform360-covariance-residual-history-policy-v1"
)
DISJOINT_CAMERA_PARTITION_SCHEMA: Final = (
    "bayesian-phystwin.deform360-disjoint-camera-partition-v1"
)
RESIDUAL_HISTORY_ADAPTER_SCHEMA: Final = (
    "bayesian-phystwin.deform360-covariance-residual-history-adapter-v1"
)
RESIDUAL_HISTORY_DECISION_SCHEMA: Final = (
    "bayesian-phystwin.deform360-covariance-residual-history-decision-v1"
)
RESIDUAL_HISTORY_RECEIPT_SCHEMA: Final = (
    "bayesian-phystwin.deform360-covariance-residual-history-receipt-v1"
)
CAMERA_PARTITION_NAMESPACE: Final = "deform360-provider-scoring-camera-family-v1"
HORIZON_LABELS: Final = ("early", "middle", "late")
RESIDUAL_STORAGE_SEMANTICS: Final = (
    "provider-observation-minus-physical-baseline-m; invalid rows stored as zero only"
)
REFERENCE_MEAN_SEMANTICS: Final = (
    "physical-future-plus-final-causal-same-material-residual"
)
FALLBACK_SEMANTICS: Final = (
    "exact-caller-owned-physical-future-mean-and-covariance-objects"
)
CLAIM_BOUNDARY: Final = (
    "source-only adapter contract; no fresh-target payload, prediction, or outcome"
)
TARGET_QUARANTINE_ROOT: Final = Path(
    "/mnt/lexar4tb/datasets/deform360/unopened-candidate-target/"
    "covariance-only-v1/payload"
)
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_ORIENTATION_SUFFIXES = frozenset(
    {
        "left",
        "right",
        "front",
        "back",
        "rear",
        "top",
        "bottom",
        "upper",
        "lower",
    }
)
_CHANNEL_WORDS = frozenset({"cam", "camera", "view", "stream", "sensor"})
_CHANNEL_TOKEN = re.compile(r"^(?:cam|camera|view|stream|sensor)\d+$")
_DIGITS = re.compile(r"^\d+$")


def _canonical_string(value: object, *, name: str) -> str:
    if type(value) is not str or value == "" or value != value.strip():
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def _finite_real(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
) -> float:
    if type(value) not in {int, float} or type(value) is bool:
        raise ValueError(f"{name} must be a finite real number")
    result = float(np.asarray(value).item())
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _required_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _HEX_64.fullmatch(value) is None:
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _readonly_float_array(
    value: object,
    *,
    name: str,
    ndim: int,
    finite: bool = True,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    result = immutable_array(value, dtype=np.float64)
    if finite and not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _integer_vector(value: object, *, name: str) -> np.ndarray:
    array = immutable_integer_array(value, name=name)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must be nonempty")
    return array


def _boolean_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(bool) or array.shape != shape:
        raise ValueError(f"{name} must be a Boolean array with shape {shape}")
    result = np.array(array, dtype=bool, copy=True, order="C")
    result.setflags(write=False)
    return result


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    descriptor = {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "payload_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }
    return hashlib.sha256(canonical_json_bytes(descriptor)).hexdigest()


def _validate_covariance(
    value: object,
    *,
    name: str,
    expected_shape: tuple[int, ...],
    preserve_identity: bool = False,
    numerical_tolerance: float = 1e-10,
) -> np.ndarray:
    if preserve_identity:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array to preserve identity")
        if value.dtype != np.dtype(np.float64):
            raise ValueError(f"{name} must have dtype float64")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
        array = value
    else:
        array = _readonly_float_array(
            value,
            name=name,
            ndim=len(expected_shape),
        )
    if array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    if not np.allclose(
        array,
        np.swapaxes(array, -1, -2),
        atol=numerical_tolerance,
        rtol=0.0,
    ):
        raise ValueError(f"{name} must be symmetric")
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(array)))
    if minimum_eigenvalue < -numerical_tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    return array


def assert_outside_target_quarantine(
    path: Path | str,
    *,
    name: str = "path",
) -> Path:
    label = _canonical_string(name, name="name")
    resolved = Path(path).expanduser().resolve(strict=False)
    target = TARGET_QUARANTINE_ROOT.resolve(strict=False)
    if resolved == target or target in resolved.parents:
        raise ValueError(f"{label} is inside the unopened target quarantine")
    return resolved


def camera_hardware_family(camera_id: str) -> str:
    """Return the physical recorder identity while removing channel suffixes."""

    camera = _canonical_string(camera_id, name="camera_id")
    normalized = re.sub(r"[^a-z0-9]+", "-", camera.lower()).strip("-")
    tokens = [token for token in normalized.split("-") if token]
    if not tokens:
        raise ValueError("camera_id has no canonical hardware family")

    while tokens:
        tail = tokens[-1]
        if tail in _ORIENTATION_SUFFIXES:
            tokens.pop()
            continue
        if _CHANNEL_TOKEN.fullmatch(tail) is not None:
            tokens.pop()
            continue
        if (
            _DIGITS.fullmatch(tail) is not None
            and len(tokens) >= 2
            and tokens[-2] in _CHANNEL_WORDS
        ):
            tokens.pop()
            tokens.pop()
            continue
        if tail in _CHANNEL_WORDS:
            tokens.pop()
            continue
        break

    if not tokens:
        raise ValueError("camera_id has no stable recorder-family prefix")
    return "-".join(tokens)


@dataclass(frozen=True, slots=True)
class ResidualHistoryDryRunPolicyV1:
    """Frozen source-only admission and covariance policy."""

    minimum_prefix_frames: int = 3
    minimum_final_observed_count: int = 3
    minimum_final_observed_fraction: float = 0.75
    minimum_cameras_per_role: int = 8
    minimum_camera_families_per_role: int = 4
    covariance_scales: tuple[float, float, float] = (8.0, 16.0, 16.0)
    policy_id: str | None = None

    def __post_init__(self) -> None:
        integer_fields = (
            "minimum_prefix_frames",
            "minimum_final_observed_count",
            "minimum_cameras_per_role",
            "minimum_camera_families_per_role",
        )
        for name in integer_fields:
            value = genuine_integer(getattr(self, name), name=name, minimum=1)
            object.__setattr__(self, name, value)
        fraction = _finite_real(
            self.minimum_final_observed_fraction,
            name="minimum_final_observed_fraction",
            minimum=0.0,
        )
        if fraction > 1.0:
            raise ValueError("minimum_final_observed_fraction must not exceed one")
        scales = tuple(
            _finite_real(value, name="covariance_scales", minimum=1.0)
            for value in self.covariance_scales
        )
        if len(scales) != len(HORIZON_LABELS):
            raise ValueError("covariance_scales must bind early, middle, and late")
        object.__setattr__(self, "minimum_final_observed_fraction", fraction)
        object.__setattr__(self, "covariance_scales", scales)
        expected = content_id(self.descriptor())
        if self.policy_id is None:
            object.__setattr__(self, "policy_id", expected)
        elif (
            literal_lower_hex(
                self.policy_id,
                name="policy_id",
                lengths={64},
            )
            != expected
        ):
            raise ValueError("policy_id does not match the policy descriptor")

    @property
    def minimum_camera_count_per_role(self) -> int:
        """Compatibility alias for the original implementation spelling."""

        return self.minimum_cameras_per_role

    @property
    def minimum_camera_family_count_per_role(self) -> int:
        """Compatibility alias for the original implementation spelling."""

        return self.minimum_camera_families_per_role

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": RESIDUAL_HISTORY_POLICY_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "minimum_prefix_frames": self.minimum_prefix_frames,
            "minimum_final_observed_count": self.minimum_final_observed_count,
            "minimum_final_observed_fraction": self.minimum_final_observed_fraction,
            "minimum_cameras_per_role": self.minimum_cameras_per_role,
            "minimum_camera_families_per_role": (self.minimum_camera_families_per_role),
            "covariance_scales": list(self.covariance_scales),
            "horizon_labels": list(HORIZON_LABELS),
            "residual_storage_semantics": RESIDUAL_STORAGE_SEMANTICS,
            "reference_mean_semantics": REFERENCE_MEAN_SEMANTICS,
            "fallback_semantics": FALLBACK_SEMANTICS,
            "claim_boundary": CLAIM_BOUNDARY,
        }


@dataclass(frozen=True, slots=True)
class DisjointCameraPartitionV1:
    """Whole-recorder partition with no physical family in both roles."""

    provider_camera_ids: tuple[str, ...]
    scoring_camera_ids: tuple[str, ...]
    provider_family_ids: tuple[str, ...]
    scoring_family_ids: tuple[str, ...]
    namespace: str = CAMERA_PARTITION_NAMESPACE
    partition_id: str | None = None

    def __post_init__(self) -> None:
        namespace = _canonical_string(self.namespace, name="namespace")
        provider_ids = tuple(
            sorted(
                _canonical_string(value, name="provider_camera_ids")
                for value in self.provider_camera_ids
            )
        )
        scoring_ids = tuple(
            sorted(
                _canonical_string(value, name="scoring_camera_ids")
                for value in self.scoring_camera_ids
            )
        )
        provider_families = tuple(
            sorted(
                _canonical_string(value, name="provider_family_ids")
                for value in self.provider_family_ids
            )
        )
        scoring_families = tuple(
            sorted(
                _canonical_string(value, name="scoring_family_ids")
                for value in self.scoring_family_ids
            )
        )
        if (
            provider_ids != tuple(sorted(set(provider_ids)))
            or scoring_ids != tuple(sorted(set(scoring_ids)))
            or provider_families != tuple(sorted(set(provider_families)))
            or scoring_families != tuple(sorted(set(scoring_families)))
        ):
            raise ValueError("camera partition fields must be sorted and unique")
        if set(provider_ids) & set(scoring_ids):
            raise ValueError("provider and scoring cameras must be disjoint")
        if set(provider_families) & set(scoring_families):
            raise ValueError("provider and scoring recorder families must be disjoint")
        if not provider_ids or not scoring_ids:
            raise ValueError("camera partition requires both roles")
        expected_provider_families = tuple(
            sorted({camera_hardware_family(value) for value in provider_ids})
        )
        expected_scoring_families = tuple(
            sorted({camera_hardware_family(value) for value in scoring_ids})
        )
        if provider_families != expected_provider_families:
            raise ValueError("provider families do not bind provider camera IDs")
        if scoring_families != expected_scoring_families:
            raise ValueError("scoring families do not bind scoring camera IDs")
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "provider_camera_ids", provider_ids)
        object.__setattr__(self, "scoring_camera_ids", scoring_ids)
        object.__setattr__(self, "provider_family_ids", provider_families)
        object.__setattr__(self, "scoring_family_ids", scoring_families)
        expected = content_id(self.descriptor())
        if self.partition_id is None:
            object.__setattr__(self, "partition_id", expected)
        elif (
            literal_lower_hex(
                self.partition_id,
                name="partition_id",
                lengths={64},
            )
            != expected
        ):
            raise ValueError("partition_id does not match the camera partition")

    @property
    def provider_camera_families(self) -> tuple[str, ...]:
        """Compatibility alias for the original implementation spelling."""

        return self.provider_family_ids

    @property
    def scoring_camera_families(self) -> tuple[str, ...]:
        """Compatibility alias for the original implementation spelling."""

        return self.scoring_family_ids

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": DISJOINT_CAMERA_PARTITION_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "namespace": self.namespace,
            "provider_camera_ids": list(self.provider_camera_ids),
            "scoring_camera_ids": list(self.scoring_camera_ids),
            "provider_family_ids": list(self.provider_family_ids),
            "scoring_family_ids": list(self.scoring_family_ids),
        }


def _ranked_camera_families(
    family_to_ids: dict[str, list[str]],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            family_to_ids,
            key=lambda family: (
                hashlib.sha256(
                    f"{CAMERA_PARTITION_NAMESPACE}:{family}".encode()
                ).hexdigest(),
                family,
            ),
        )
    )


def deterministic_disjoint_camera_partition(
    camera_ids: Sequence[str],
    *,
    policy: ResidualHistoryDryRunPolicyV1,
) -> DisjointCameraPartitionV1:
    """Assign complete recorder families to deterministic balanced roles."""

    if not isinstance(policy, ResidualHistoryDryRunPolicyV1):
        raise TypeError("policy must be ResidualHistoryDryRunPolicyV1")
    ids = tuple(
        sorted(_canonical_string(value, name="camera_ids") for value in camera_ids)
    )
    if ids != tuple(sorted(set(ids))):
        raise ValueError("camera_ids must be unique")
    family_to_ids: dict[str, list[str]] = {}
    for camera_id in ids:
        family = camera_hardware_family(camera_id)
        family_to_ids.setdefault(family, []).append(camera_id)

    provider_families: list[str] = []
    scoring_families: list[str] = []
    provider_count = 0
    scoring_count = 0
    for family in _ranked_camera_families(family_to_ids):
        camera_count = len(family_to_ids[family])
        provider_state = (
            provider_count,
            len(provider_families),
            0,
        )
        scoring_state = (
            scoring_count,
            len(scoring_families),
            1,
        )
        if provider_state <= scoring_state:
            provider_families.append(family)
            provider_count += camera_count
        else:
            scoring_families.append(family)
            scoring_count += camera_count

    provider_ids = tuple(
        sorted(
            camera for family in provider_families for camera in family_to_ids[family]
        )
    )
    scoring_ids = tuple(
        sorted(
            camera for family in scoring_families for camera in family_to_ids[family]
        )
    )
    if (
        len(provider_ids) < policy.minimum_cameras_per_role
        or len(scoring_ids) < policy.minimum_cameras_per_role
        or len(provider_families) < policy.minimum_camera_families_per_role
        or len(scoring_families) < policy.minimum_camera_families_per_role
    ):
        raise ValueError("camera roster does not meet minimum camera support per role")
    return DisjointCameraPartitionV1(
        provider_camera_ids=provider_ids,
        scoring_camera_ids=scoring_ids,
        provider_family_ids=tuple(sorted(provider_families)),
        scoring_family_ids=tuple(sorted(scoring_families)),
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
