"""Shared policy, provenance, and validation contracts for the source dry run."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
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
CAMERA_RECORDER_FAMILY_MAP_SCHEMA: Final = (
    "bayesian-phystwin.deform360-camera-recorder-family-map-v1"
)
DISJOINT_CAMERA_PARTITION_SCHEMA: Final = (
    "bayesian-phystwin.deform360-disjoint-camera-partition-v1"
)
RECONSTRUCTION_MANIFEST_SCHEMA: Final = (
    "bayesian-phystwin.deform360-reconstruction-manifest-v1"
)
RESIDUAL_HISTORY_ADAPTER_SCHEMA: Final = (
    "bayesian-phystwin.deform360-covariance-residual-history-adapter-v1"
)
RESIDUAL_HISTORY_DECISION_SCHEMA: Final = (
    "bayesian-phystwin.deform360-covariance-residual-history-decision-v1"
)
CAMERA_PARTITION_NAMESPACE: Final = "deform360-provider-scoring-recorder-family-v1"
HORIZON_LABELS: Final = ("early", "middle", "late")
REGISTERED_REFERENCE_PREDICTOR_ID: Final = "last_residual"
REGISTERED_COVARIANCE_DONOR_ID: Final = "independent_endpoint_v1"
REGISTERED_COVARIANCE_SCALES: Final = (8.0, 16.0, 16.0)
REGISTERED_MINIMUM_VALID_OBSERVATIONS_PER_MATERIAL: Final = 2
RESIDUAL_STORAGE_SEMANTICS: Final = (
    "provider-observation-minus-physical-baseline-m; invalid rows stored as zero only"
)
REFERENCE_MEAN_SEMANTICS: Final = (
    "exact-caller-owned-last-residual-mean-verified-against-causal-history"
)
FALLBACK_SEMANTICS: Final = (
    "exact-caller-owned-physical-future-mean-and-covariance-objects"
)
CLAIM_BOUNDARY: Final = (
    "source-only adapter contract; no target payload, prediction, score, or claim"
)


def _canonical_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    if any(character in value for character in "\x00\r\n"):
        raise ValueError(f"{name} must be a single canonical line")
    return value


def _finite_real(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
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
    return result


def _required_sha256(value: object, *, name: str) -> str:
    return literal_lower_hex(value, name=name, lengths={64})


def _canonical_string_tuple(
    value: object,
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of strings")
    result = tuple(sorted(_canonical_string(item, name=name) for item in value))
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


def _sha256_tuple(
    value: object,
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of SHA-256 identities")
    result = tuple(sorted(_required_sha256(item, name=name) for item in value))
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


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


@dataclass(frozen=True, slots=True)
class ResidualHistoryDryRunPolicyV1:
    """Frozen source-only support and covariance policy."""

    minimum_prefix_frames: int = 3
    minimum_valid_observations_per_material: int = (
        REGISTERED_MINIMUM_VALID_OBSERVATIONS_PER_MATERIAL
    )
    minimum_cameras_per_role: int = 8
    minimum_camera_families_per_role: int = 4
    covariance_scales: tuple[float, float, float] = REGISTERED_COVARIANCE_SCALES
    policy_id: str | None = None

    def __post_init__(self) -> None:
        minimum_prefix_frames = genuine_integer(
            self.minimum_prefix_frames,
            name="minimum_prefix_frames",
            minimum=2,
        )
        minimum_support = genuine_integer(
            self.minimum_valid_observations_per_material,
            name="minimum_valid_observations_per_material",
            minimum=1,
        )
        if minimum_support != REGISTERED_MINIMUM_VALID_OBSERVATIONS_PER_MATERIAL:
            raise ValueError(
                "minimum_valid_observations_per_material changed from the registered value"
            )
        minimum_cameras = genuine_integer(
            self.minimum_cameras_per_role,
            name="minimum_cameras_per_role",
            minimum=2,
        )
        minimum_families = genuine_integer(
            self.minimum_camera_families_per_role,
            name="minimum_camera_families_per_role",
            minimum=2,
        )
        if type(self.covariance_scales) is not tuple:
            raise ValueError("covariance_scales must be a canonical tuple")
        scales = tuple(
            _finite_real(value, name="covariance_scales", minimum=0.0)
            for value in self.covariance_scales
        )
        if scales != REGISTERED_COVARIANCE_SCALES:
            raise ValueError("covariance_scales changed from the registered schedule")
        object.__setattr__(self, "minimum_prefix_frames", minimum_prefix_frames)
        object.__setattr__(
            self,
            "minimum_valid_observations_per_material",
            minimum_support,
        )
        object.__setattr__(self, "minimum_cameras_per_role", minimum_cameras)
        object.__setattr__(
            self,
            "minimum_camera_families_per_role",
            minimum_families,
        )
        object.__setattr__(self, "covariance_scales", scales)
        expected = content_id(self.descriptor())
        if self.policy_id is None:
            object.__setattr__(self, "policy_id", expected)
        elif _required_sha256(self.policy_id, name="policy_id") != expected:
            raise ValueError("policy_id does not match the policy descriptor")

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": RESIDUAL_HISTORY_POLICY_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "minimum_prefix_frames": self.minimum_prefix_frames,
            "minimum_valid_observations_per_material": (
                self.minimum_valid_observations_per_material
            ),
            "minimum_cameras_per_role": self.minimum_cameras_per_role,
            "minimum_camera_families_per_role": (
                self.minimum_camera_families_per_role
            ),
            "covariance_scales": list(self.covariance_scales),
            "horizon_labels": list(HORIZON_LABELS),
            "reference_predictor_id": REGISTERED_REFERENCE_PREDICTOR_ID,
            "covariance_donor_id": REGISTERED_COVARIANCE_DONOR_ID,
            "residual_storage_semantics": RESIDUAL_STORAGE_SEMANTICS,
            "reference_mean_semantics": REFERENCE_MEAN_SEMANTICS,
            "fallback_semantics": FALLBACK_SEMANTICS,
            "claim_boundary": CLAIM_BOUNDARY,
        }


@dataclass(frozen=True, slots=True)
class CameraRecorderFamilyMapV1:
    """Content-addressed camera-to-physical-recorder identities from inventory."""

    source_inventory_id: str
    bindings: tuple[tuple[str, str], ...]
    map_id: str | None = None

    def __post_init__(self) -> None:
        inventory_id = _required_sha256(
            self.source_inventory_id,
            name="source_inventory_id",
        )
        if isinstance(self.bindings, (str, bytes)) or not isinstance(
            self.bindings,
            Sequence,
        ):
            raise ValueError("bindings must be a sequence of camera/family pairs")
        rows: list[tuple[str, str]] = []
        for index, raw in enumerate(self.bindings):
            if type(raw) is not tuple or len(raw) != 2:
                raise ValueError(f"bindings[{index}] must be a two-string tuple")
            camera_id = _canonical_string(raw[0], name=f"bindings[{index}].camera_id")
            family_id = _canonical_string(
                raw[1],
                name=f"bindings[{index}].recorder_family_id",
            )
            rows.append((camera_id, family_id))
        bindings = tuple(sorted(rows))
        if not bindings:
            raise ValueError("bindings must not be empty")
        cameras = tuple(camera for camera, _family in bindings)
        if len(set(cameras)) != len(cameras):
            raise ValueError("each camera must have exactly one recorder-family identity")
        object.__setattr__(self, "source_inventory_id", inventory_id)
        object.__setattr__(self, "bindings", bindings)
        expected = content_id(self.descriptor())
        if self.map_id is None:
            object.__setattr__(self, "map_id", expected)
        elif _required_sha256(self.map_id, name="map_id") != expected:
            raise ValueError("map_id does not match the recorder-family map")

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return tuple(camera for camera, _family in self.bindings)

    @property
    def family_ids(self) -> tuple[str, ...]:
        return tuple(sorted({family for _camera, family in self.bindings}))

    def family_for_camera(self, camera_id: str) -> str:
        query = _canonical_string(camera_id, name="camera_id")
        for camera, family in self.bindings:
            if camera == query:
                return family
        raise KeyError(query)

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": CAMERA_RECORDER_FAMILY_MAP_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "source_inventory_id": self.source_inventory_id,
            "bindings": [
                {"camera_id": camera, "recorder_family_id": family}
                for camera, family in self.bindings
            ],
        }


@dataclass(frozen=True, slots=True)
class DisjointCameraPartitionV1:
    """Deterministic whole-recorder provider/scoring split."""

    family_map: CameraRecorderFamilyMapV1
    provider_camera_ids: tuple[str, ...]
    scoring_camera_ids: tuple[str, ...]
    provider_family_ids: tuple[str, ...]
    scoring_family_ids: tuple[str, ...]
    namespace: str = CAMERA_PARTITION_NAMESPACE
    partition_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.family_map, CameraRecorderFamilyMapV1):
            raise TypeError("family_map must be CameraRecorderFamilyMapV1")
        namespace = _canonical_string(self.namespace, name="namespace")
        if namespace != CAMERA_PARTITION_NAMESPACE:
            raise ValueError("camera partition namespace changed")
        provider = _canonical_string_tuple(
            self.provider_camera_ids,
            name="provider_camera_ids",
        )
        scoring = _canonical_string_tuple(
            self.scoring_camera_ids,
            name="scoring_camera_ids",
        )
        provider_families = _canonical_string_tuple(
            self.provider_family_ids,
            name="provider_family_ids",
        )
        scoring_families = _canonical_string_tuple(
            self.scoring_family_ids,
            name="scoring_family_ids",
        )
        if set(provider) & set(scoring):
            raise ValueError("provider and scoring cameras must be disjoint")
        if set(provider) | set(scoring) != set(self.family_map.camera_ids):
            raise ValueError("provider and scoring cameras must exhaust the frozen map")
        expected_provider_families = {
            self.family_map.family_for_camera(camera) for camera in provider
        }
        expected_scoring_families = {
            self.family_map.family_for_camera(camera) for camera in scoring
        }
        if set(provider_families) != expected_provider_families:
            raise ValueError("provider families do not match the frozen camera map")
        if set(scoring_families) != expected_scoring_families:
            raise ValueError("scoring families do not match the frozen camera map")
        if expected_provider_families & expected_scoring_families:
            raise ValueError("one physical recorder family crosses camera roles")
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "provider_camera_ids", provider)
        object.__setattr__(self, "scoring_camera_ids", scoring)
        object.__setattr__(self, "provider_family_ids", provider_families)
        object.__setattr__(self, "scoring_family_ids", scoring_families)
        expected = content_id(self.descriptor())
        if self.partition_id is None:
            object.__setattr__(self, "partition_id", expected)
        elif _required_sha256(self.partition_id, name="partition_id") != expected:
            raise ValueError("partition_id does not match the camera partition")

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": DISJOINT_CAMERA_PARTITION_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "namespace": self.namespace,
            "family_map_id": _required_sha256(
                self.family_map.map_id,
                name="family_map.map_id",
            ),
            "source_inventory_id": self.family_map.source_inventory_id,
            "provider_camera_ids": list(self.provider_camera_ids),
            "scoring_camera_ids": list(self.scoring_camera_ids),
            "provider_family_ids": list(self.provider_family_ids),
            "scoring_family_ids": list(self.scoring_family_ids),
        }


def deterministic_disjoint_camera_partition(
    family_map: CameraRecorderFamilyMapV1,
    *,
    policy: ResidualHistoryDryRunPolicyV1,
) -> DisjointCameraPartitionV1:
    """Hash-rank and balance explicit physical recorder families."""

    if not isinstance(family_map, CameraRecorderFamilyMapV1):
        raise TypeError("family_map must be CameraRecorderFamilyMapV1")
    if not isinstance(policy, ResidualHistoryDryRunPolicyV1):
        raise TypeError("policy must be ResidualHistoryDryRunPolicyV1")
    family_to_cameras: dict[str, list[str]] = {}
    for camera, family in family_map.bindings:
        family_to_cameras.setdefault(family, []).append(camera)
    map_id = _required_sha256(family_map.map_id, name="family_map.map_id")
    ranked = sorted(
        family_to_cameras,
        key=lambda family: (
            hashlib.sha256(
                f"{CAMERA_PARTITION_NAMESPACE}:{map_id}:{family}".encode()
            ).hexdigest(),
            family,
        ),
    )
    provider_families: list[str] = []
    scoring_families: list[str] = []
    provider_count = 0
    scoring_count = 0
    for family in ranked:
        camera_count = len(family_to_cameras[family])
        provider_state = (provider_count, len(provider_families), 0)
        scoring_state = (scoring_count, len(scoring_families), 1)
        if provider_state <= scoring_state:
            provider_families.append(family)
            provider_count += camera_count
        else:
            scoring_families.append(family)
            scoring_count += camera_count
    provider = tuple(
        sorted(
            camera
            for family in provider_families
            for camera in family_to_cameras[family]
        )
    )
    scoring = tuple(
        sorted(
            camera
            for family in scoring_families
            for camera in family_to_cameras[family]
        )
    )
    if (
        len(provider) < policy.minimum_cameras_per_role
        or len(scoring) < policy.minimum_cameras_per_role
        or len(provider_families) < policy.minimum_camera_families_per_role
        or len(scoring_families) < policy.minimum_camera_families_per_role
    ):
        raise ValueError("camera map does not meet minimum support in both roles")
    return DisjointCameraPartitionV1(
        family_map=family_map,
        provider_camera_ids=provider,
        scoring_camera_ids=scoring,
        provider_family_ids=tuple(provider_families),
        scoring_family_ids=tuple(scoring_families),
    )


@dataclass(frozen=True, slots=True)
class ReconstructionManifestV1:
    """Content-addressed reconstruction inputs and lineage for one camera role."""

    role: str
    source_inventory_id: str
    reconstruction_artifact_id: str
    implementation_revision: str
    configuration_id: str
    input_camera_ids: tuple[str, ...]
    input_source_artifact_ids: tuple[str, ...]
    parent_reconstruction_artifact_ids: tuple[str, ...] = ()
    manifest_id: str | None = None

    def __post_init__(self) -> None:
        role = _canonical_string(self.role, name="role")
        if role not in {"provider", "scoring"}:
            raise ValueError("role must be provider or scoring")
        inventory_id = _required_sha256(
            self.source_inventory_id,
            name="source_inventory_id",
        )
        reconstruction_id = _required_sha256(
            self.reconstruction_artifact_id,
            name="reconstruction_artifact_id",
        )
        implementation = literal_lower_hex(
            self.implementation_revision,
            name="implementation_revision",
            lengths={40},
        )
        configuration_id = _required_sha256(
            self.configuration_id,
            name="configuration_id",
        )
        cameras = _canonical_string_tuple(
            self.input_camera_ids,
            name="input_camera_ids",
        )
        source_artifacts = _sha256_tuple(
            self.input_source_artifact_ids,
            name="input_source_artifact_ids",
        )
        parents = _sha256_tuple(
            self.parent_reconstruction_artifact_ids,
            name="parent_reconstruction_artifact_ids",
            allow_empty=True,
        )
        if reconstruction_id in parents:
            raise ValueError("reconstruction artifact cannot be its own parent")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "source_inventory_id", inventory_id)
        object.__setattr__(self, "reconstruction_artifact_id", reconstruction_id)
        object.__setattr__(self, "implementation_revision", implementation)
        object.__setattr__(self, "configuration_id", configuration_id)
        object.__setattr__(self, "input_camera_ids", cameras)
        object.__setattr__(self, "input_source_artifact_ids", source_artifacts)
        object.__setattr__(self, "parent_reconstruction_artifact_ids", parents)
        expected = content_id(self.descriptor())
        if self.manifest_id is None:
            object.__setattr__(self, "manifest_id", expected)
        elif _required_sha256(self.manifest_id, name="manifest_id") != expected:
            raise ValueError("manifest_id does not match reconstruction lineage")

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": RECONSTRUCTION_MANIFEST_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "role": self.role,
            "source_inventory_id": self.source_inventory_id,
            "reconstruction_artifact_id": self.reconstruction_artifact_id,
            "implementation_revision": self.implementation_revision,
            "configuration_id": self.configuration_id,
            "input_camera_ids": list(self.input_camera_ids),
            "input_source_artifact_ids": list(self.input_source_artifact_ids),
            "parent_reconstruction_artifact_ids": list(
                self.parent_reconstruction_artifact_ids
            ),
        }


def validate_reconstruction_separation(
    partition: DisjointCameraPartitionV1,
    provider: ReconstructionManifestV1,
    scoring: ReconstructionManifestV1,
) -> None:
    """Require exact camera roles and disjoint source bytes and lineage."""

    if not isinstance(partition, DisjointCameraPartitionV1):
        raise TypeError("partition must be DisjointCameraPartitionV1")
    if not isinstance(provider, ReconstructionManifestV1):
        raise TypeError("provider manifest must be ReconstructionManifestV1")
    if not isinstance(scoring, ReconstructionManifestV1):
        raise TypeError("scoring manifest must be ReconstructionManifestV1")
    if provider.role != "provider" or scoring.role != "scoring":
        raise ValueError("reconstruction manifest roles changed")
    inventory_id = partition.family_map.source_inventory_id
    if provider.source_inventory_id != inventory_id:
        raise ValueError("provider manifest uses a different source inventory")
    if scoring.source_inventory_id != inventory_id:
        raise ValueError("scoring manifest uses a different source inventory")
    if provider.input_camera_ids != partition.provider_camera_ids:
        raise ValueError("provider manifest camera set differs from the partition")
    if scoring.input_camera_ids != partition.scoring_camera_ids:
        raise ValueError("scoring manifest camera set differs from the partition")
    if set(provider.input_source_artifact_ids) & set(
        scoring.input_source_artifact_ids
    ):
        raise ValueError("provider and scoring reconstructions share source bytes")
    provider_lineage = {
        provider.reconstruction_artifact_id,
        *provider.parent_reconstruction_artifact_ids,
    }
    scoring_lineage = {
        scoring.reconstruction_artifact_id,
        *scoring.parent_reconstruction_artifact_ids,
    }
    if provider_lineage & scoring_lineage:
        raise ValueError("provider and scoring reconstruction lineages overlap")


__all__ = [
    "CAMERA_PARTITION_NAMESPACE",
    "CLAIM_BOUNDARY",
    "CameraRecorderFamilyMapV1",
    "DisjointCameraPartitionV1",
    "FALLBACK_SEMANTICS",
    "HORIZON_LABELS",
    "REFERENCE_MEAN_SEMANTICS",
    "REGISTERED_COVARIANCE_DONOR_ID",
    "REGISTERED_COVARIANCE_SCALES",
    "REGISTERED_MINIMUM_VALID_OBSERVATIONS_PER_MATERIAL",
    "REGISTERED_REFERENCE_PREDICTOR_ID",
    "RESIDUAL_STORAGE_SEMANTICS",
    "ReconstructionManifestV1",
    "ResidualHistoryDryRunPolicyV1",
    "deterministic_disjoint_camera_partition",
    "validate_reconstruction_separation",
]
