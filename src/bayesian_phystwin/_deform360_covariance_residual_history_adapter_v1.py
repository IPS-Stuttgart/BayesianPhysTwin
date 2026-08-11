"""Causal no-fill residual history for one opened source unit."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._canonical_contracts import frozen_finite_json_mapping, plain_json
from ._deform360_covariance_residual_history_common_v1 import (
    RESIDUAL_HISTORY_ADAPTER_SCHEMA,
    RESIDUAL_STORAGE_SEMANTICS,
    SCHEMA_VERSION,
    CameraRecorderFamilyMapV1,
    DisjointCameraPartitionV1,
    ReconstructionManifestV1,
    ResidualHistoryDryRunPolicyV1,
    _array_sha256,
    _boolean_array,
    _canonical_string,
    _integer_vector,
    _readonly_float_array,
    _required_sha256,
    deterministic_disjoint_camera_partition,
    validate_reconstruction_separation,
)
from ._portable_contracts import content_id


@dataclass(frozen=True, slots=True)
class ResidualHistoryAdapterV1:
    """Content-addressed causal residual history with explicit missingness."""

    source_unit_id: str
    frame_indices: np.ndarray
    material_ids: np.ndarray
    residual_history_m: np.ndarray
    observed_validity: np.ndarray
    partition: DisjointCameraPartitionV1
    provider_reconstruction_manifest: ReconstructionManifestV1
    scoring_reconstruction_manifest: ReconstructionManifestV1
    baseline_prefix_sha256: str
    observation_prefix_sha256: str
    policy: ResidualHistoryDryRunPolicyV1
    metadata: Mapping[str, Any] = field(default_factory=dict)
    adapter_id: str | None = None
    residual_history_sha256: str = field(init=False)
    observed_validity_sha256: str = field(init=False)
    valid_observation_count_by_material: tuple[int, ...] = field(init=False)
    supported_material_mask: np.ndarray = field(init=False)
    supported_material_mask_sha256: str = field(init=False)
    prefix_frame_count: int = field(init=False)
    material_count: int = field(init=False)
    observed_count: int = field(init=False)
    supported_material_count: int = field(init=False)
    unsupported_material_count: int = field(init=False)

    def __post_init__(self) -> None:
        source_unit_id = _canonical_string(self.source_unit_id, name="source_unit_id")
        frame_indices = _integer_vector(self.frame_indices, name="frame_indices")
        if not np.all(np.diff(frame_indices) > 0):
            raise ValueError("frame_indices must be strictly increasing")
        material_ids = _integer_vector(self.material_ids, name="material_ids")
        if len(np.unique(material_ids)) != len(material_ids):
            raise ValueError("material_ids must be unique")
        residual = _readonly_float_array(
            self.residual_history_m,
            name="residual_history_m",
            ndim=3,
        )
        expected_shape = (len(frame_indices), len(material_ids), 3)
        if residual.shape != expected_shape:
            raise ValueError(f"residual_history_m must have shape {expected_shape}")
        validity = _boolean_array(
            self.observed_validity,
            name="observed_validity",
            shape=expected_shape[:2],
        )
        if not np.array_equal(residual[~validity], np.zeros_like(residual[~validity])):
            raise ValueError("invalid residual rows must use zero storage")
        if not isinstance(self.partition, DisjointCameraPartitionV1):
            raise TypeError("partition must be DisjointCameraPartitionV1")
        if not isinstance(
            self.provider_reconstruction_manifest,
            ReconstructionManifestV1,
        ):
            raise TypeError(
                "provider_reconstruction_manifest must be ReconstructionManifestV1"
            )
        if not isinstance(
            self.scoring_reconstruction_manifest,
            ReconstructionManifestV1,
        ):
            raise TypeError(
                "scoring_reconstruction_manifest must be ReconstructionManifestV1"
            )
        if not isinstance(self.policy, ResidualHistoryDryRunPolicyV1):
            raise TypeError("policy must be ResidualHistoryDryRunPolicyV1")
        validate_reconstruction_separation(
            self.partition,
            self.provider_reconstruction_manifest,
            self.scoring_reconstruction_manifest,
        )
        baseline_sha = _required_sha256(
            self.baseline_prefix_sha256,
            name="baseline_prefix_sha256",
        )
        observation_sha = _required_sha256(
            self.observation_prefix_sha256,
            name="observation_prefix_sha256",
        )
        metadata = frozen_finite_json_mapping(self.metadata, name="metadata")
        count_array = np.count_nonzero(validity, axis=0)
        counts = tuple(int(value) for value in count_array)
        supported = _boolean_array(
            count_array >= self.policy.minimum_valid_observations_per_material,
            name="supported_material_mask",
            shape=(len(material_ids),),
        )
        supported_count = int(np.count_nonzero(supported))
        object.__setattr__(self, "source_unit_id", source_unit_id)
        object.__setattr__(self, "frame_indices", frame_indices)
        object.__setattr__(self, "material_ids", material_ids)
        object.__setattr__(self, "residual_history_m", residual)
        object.__setattr__(self, "observed_validity", validity)
        object.__setattr__(self, "baseline_prefix_sha256", baseline_sha)
        object.__setattr__(self, "observation_prefix_sha256", observation_sha)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "residual_history_sha256", _array_sha256(residual))
        object.__setattr__(self, "observed_validity_sha256", _array_sha256(validity))
        object.__setattr__(self, "valid_observation_count_by_material", counts)
        object.__setattr__(self, "supported_material_mask", supported)
        object.__setattr__(
            self,
            "supported_material_mask_sha256",
            _array_sha256(supported),
        )
        object.__setattr__(self, "prefix_frame_count", len(frame_indices))
        object.__setattr__(self, "material_count", len(material_ids))
        object.__setattr__(self, "observed_count", int(np.count_nonzero(validity)))
        object.__setattr__(self, "supported_material_count", supported_count)
        object.__setattr__(
            self,
            "unsupported_material_count",
            len(material_ids) - supported_count,
        )
        expected = content_id(self.descriptor())
        if self.adapter_id is None:
            object.__setattr__(self, "adapter_id", expected)
        elif _required_sha256(self.adapter_id, name="adapter_id") != expected:
            raise ValueError("adapter_id does not match the residual history")

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": RESIDUAL_HISTORY_ADAPTER_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "source_unit_id": self.source_unit_id,
            "frame_indices": self.frame_indices.tolist(),
            "material_ids": self.material_ids.tolist(),
            "residual_history_shape": list(self.residual_history_m.shape),
            "residual_history_sha256": self.residual_history_sha256,
            "observed_validity_shape": list(self.observed_validity.shape),
            "observed_validity_sha256": self.observed_validity_sha256,
            "valid_observation_count_by_material": list(
                self.valid_observation_count_by_material
            ),
            "supported_material_mask_sha256": (
                self.supported_material_mask_sha256
            ),
            "prefix_frame_count": self.prefix_frame_count,
            "material_count": self.material_count,
            "observed_count": self.observed_count,
            "supported_material_count": self.supported_material_count,
            "unsupported_material_count": self.unsupported_material_count,
            "residual_storage_semantics": RESIDUAL_STORAGE_SEMANTICS,
            "family_map_id": _required_sha256(
                self.partition.family_map.map_id,
                name="family_map.map_id",
            ),
            "partition_id": _required_sha256(
                self.partition.partition_id,
                name="partition_id",
            ),
            "provider_reconstruction_manifest_id": _required_sha256(
                self.provider_reconstruction_manifest.manifest_id,
                name="provider_reconstruction_manifest.manifest_id",
            ),
            "scoring_reconstruction_manifest_id": _required_sha256(
                self.scoring_reconstruction_manifest.manifest_id,
                name="scoring_reconstruction_manifest.manifest_id",
            ),
            "provider_reconstruction_artifact_id": (
                self.provider_reconstruction_manifest.reconstruction_artifact_id
            ),
            "scoring_reconstruction_artifact_id": (
                self.scoring_reconstruction_manifest.reconstruction_artifact_id
            ),
            "baseline_prefix_sha256": self.baseline_prefix_sha256,
            "observation_prefix_sha256": self.observation_prefix_sha256,
            "policy_id": _required_sha256(self.policy.policy_id, name="policy.policy_id"),
            "metadata": plain_json(self.metadata),
        }


def build_residual_history_adapter(
    physical_prefix_m: object,
    provider_observation_prefix_m: object,
    observed_validity: object,
    *,
    frame_indices: object,
    material_ids: object,
    camera_recorder_family_map: CameraRecorderFamilyMapV1,
    provider_reconstruction_manifest: ReconstructionManifestV1,
    scoring_reconstruction_manifest: ReconstructionManifestV1,
    source_unit_id: str,
    policy: ResidualHistoryDryRunPolicyV1,
    metadata: Mapping[str, Any] | None = None,
) -> ResidualHistoryAdapterV1:
    """Build one provenance-bound no-fill causal residual history."""

    if not isinstance(policy, ResidualHistoryDryRunPolicyV1):
        raise TypeError("policy must be ResidualHistoryDryRunPolicyV1")
    if not isinstance(camera_recorder_family_map, CameraRecorderFamilyMapV1):
        raise TypeError("camera_recorder_family_map must be CameraRecorderFamilyMapV1")
    baseline = _readonly_float_array(
        physical_prefix_m,
        name="physical_prefix_m",
        ndim=3,
    )
    if baseline.shape[-1] != 3 or baseline.shape[0] < policy.minimum_prefix_frames:
        raise ValueError("physical_prefix_m must have supported shape (T, N, 3)")
    observation = _readonly_float_array(
        provider_observation_prefix_m,
        name="provider_observation_prefix_m",
        ndim=3,
        finite=False,
    )
    if observation.shape != baseline.shape:
        raise ValueError("provider observation shape differs from the physical prefix")
    validity = _boolean_array(
        observed_validity,
        name="observed_validity",
        shape=baseline.shape[:2],
    )
    if not np.all(np.isfinite(observation[validity])):
        raise ValueError("valid provider observations must be finite")
    frames = _integer_vector(frame_indices, name="frame_indices")
    materials = _integer_vector(material_ids, name="material_ids")
    if len(frames) != baseline.shape[0]:
        raise ValueError("frame_indices length differs from the prefix")
    if len(materials) != baseline.shape[1]:
        raise ValueError("material_ids length differs from the prefix")
    canonical_observation = np.zeros_like(baseline)
    canonical_observation[validity] = observation[validity]
    residual = np.zeros_like(baseline)
    residual[validity] = canonical_observation[validity] - baseline[validity]
    partition = deterministic_disjoint_camera_partition(
        camera_recorder_family_map,
        policy=policy,
    )
    validate_reconstruction_separation(
        partition,
        provider_reconstruction_manifest,
        scoring_reconstruction_manifest,
    )
    return ResidualHistoryAdapterV1(
        source_unit_id=source_unit_id,
        frame_indices=frames,
        material_ids=materials,
        residual_history_m=residual,
        observed_validity=validity,
        partition=partition,
        provider_reconstruction_manifest=provider_reconstruction_manifest,
        scoring_reconstruction_manifest=scoring_reconstruction_manifest,
        baseline_prefix_sha256=_array_sha256(baseline),
        observation_prefix_sha256=_array_sha256(canonical_observation),
        policy=policy,
        metadata={} if metadata is None else metadata,
    )


__all__ = [
    "ResidualHistoryAdapterV1",
    "build_residual_history_adapter",
]
