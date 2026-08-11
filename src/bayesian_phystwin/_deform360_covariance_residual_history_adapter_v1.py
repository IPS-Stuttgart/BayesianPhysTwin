"""Causal no-fill residual-history adapter for one opened source unit."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    literal_lower_hex,
    plain_json,
)
from ._deform360_covariance_residual_history_common_v1 import (
    RESIDUAL_HISTORY_ADAPTER_SCHEMA,
    RESIDUAL_STORAGE_SEMANTICS,
    SCHEMA_VERSION,
    DisjointCameraPartitionV1,
    ResidualHistoryDryRunPolicyV1,
    _array_sha256,
    _boolean_array,
    _canonical_string,
    _integer_vector,
    _readonly_float_array,
    _required_sha256,
    deterministic_disjoint_camera_partition,
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
    provider_reconstruction_artifact_id: str
    scoring_reconstruction_artifact_id: str
    baseline_prefix_sha256: str
    observation_prefix_sha256: str
    policy_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    adapter_id: str | None = None
    residual_history_sha256: str = field(init=False)
    observed_validity_sha256: str = field(init=False)
    prefix_frame_count: int = field(init=False)
    material_count: int = field(init=False)
    observed_count: int = field(init=False)
    final_observed_count: int = field(init=False)
    final_observed_fraction: float = field(init=False)

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
        provider_artifact = literal_lower_hex(
            self.provider_reconstruction_artifact_id,
            name="provider_reconstruction_artifact_id",
            lengths={64},
        )
        scoring_artifact = literal_lower_hex(
            self.scoring_reconstruction_artifact_id,
            name="scoring_reconstruction_artifact_id",
            lengths={64},
        )
        if provider_artifact == scoring_artifact:
            raise ValueError(
                "provider and scoring reconstruction artifacts must differ"
            )
        baseline_sha = literal_lower_hex(
            self.baseline_prefix_sha256,
            name="baseline_prefix_sha256",
            lengths={64},
        )
        observation_sha = literal_lower_hex(
            self.observation_prefix_sha256,
            name="observation_prefix_sha256",
            lengths={64},
        )
        policy_id = literal_lower_hex(self.policy_id, name="policy_id", lengths={64})
        metadata = frozen_finite_json_mapping(self.metadata, name="metadata")
        observed_count = int(np.count_nonzero(validity))
        final_count = int(np.count_nonzero(validity[-1]))
        final_fraction = final_count / len(material_ids)
        object.__setattr__(self, "source_unit_id", source_unit_id)
        object.__setattr__(self, "frame_indices", frame_indices)
        object.__setattr__(self, "material_ids", material_ids)
        object.__setattr__(self, "residual_history_m", residual)
        object.__setattr__(self, "observed_validity", validity)
        object.__setattr__(
            self,
            "provider_reconstruction_artifact_id",
            provider_artifact,
        )
        object.__setattr__(
            self,
            "scoring_reconstruction_artifact_id",
            scoring_artifact,
        )
        object.__setattr__(self, "baseline_prefix_sha256", baseline_sha)
        object.__setattr__(self, "observation_prefix_sha256", observation_sha)
        object.__setattr__(self, "policy_id", policy_id)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "residual_history_sha256", _array_sha256(residual))
        object.__setattr__(self, "observed_validity_sha256", _array_sha256(validity))
        object.__setattr__(self, "prefix_frame_count", len(frame_indices))
        object.__setattr__(self, "material_count", len(material_ids))
        object.__setattr__(self, "observed_count", observed_count)
        object.__setattr__(self, "final_observed_count", final_count)
        object.__setattr__(self, "final_observed_fraction", final_fraction)
        expected = content_id(self.descriptor())
        if self.adapter_id is None:
            object.__setattr__(self, "adapter_id", expected)
        elif (
            literal_lower_hex(
                self.adapter_id,
                name="adapter_id",
                lengths={64},
            )
            != expected
        ):
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
            "prefix_frame_count": self.prefix_frame_count,
            "material_count": self.material_count,
            "observed_count": self.observed_count,
            "final_observed_count": self.final_observed_count,
            "final_observed_fraction": self.final_observed_fraction,
            "residual_storage_semantics": RESIDUAL_STORAGE_SEMANTICS,
            "partition_id": _required_sha256(
                self.partition.partition_id,
                name="partition_id",
            ),
            "provider_reconstruction_artifact_id": (
                self.provider_reconstruction_artifact_id
            ),
            "scoring_reconstruction_artifact_id": (
                self.scoring_reconstruction_artifact_id
            ),
            "baseline_prefix_sha256": self.baseline_prefix_sha256,
            "observation_prefix_sha256": self.observation_prefix_sha256,
            "policy_id": self.policy_id,
            "metadata": plain_json(self.metadata),
        }


def build_residual_history_adapter(
    physical_prefix_m: object,
    provider_observation_prefix_m: object,
    observed_validity: object,
    *,
    frame_indices: object,
    material_ids: object,
    camera_ids: Sequence[str],
    provider_camera_ids: Sequence[str],
    scoring_camera_ids: Sequence[str],
    provider_reconstruction_artifact_id: str,
    scoring_reconstruction_artifact_id: str,
    source_unit_id: str,
    policy: ResidualHistoryDryRunPolicyV1,
    metadata: Mapping[str, Any] | None = None,
) -> ResidualHistoryAdapterV1:
    """Build a no-fill residual history from one causal opened-source prefix."""

    if not isinstance(policy, ResidualHistoryDryRunPolicyV1):
        raise TypeError("policy must be ResidualHistoryDryRunPolicyV1")
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
    residual = np.zeros_like(baseline)
    residual[validity] = observation[validity] - baseline[validity]
    partition = deterministic_disjoint_camera_partition(camera_ids, policy=policy)
    declared_provider = tuple(sorted(provider_camera_ids))
    declared_scoring = tuple(sorted(scoring_camera_ids))
    if declared_provider != partition.provider_camera_ids:
        raise ValueError("declared provider cameras differ from the frozen partition")
    if declared_scoring != partition.scoring_camera_ids:
        raise ValueError("declared scoring cameras differ from the frozen partition")
    return ResidualHistoryAdapterV1(
        source_unit_id=source_unit_id,
        frame_indices=frames,
        material_ids=materials,
        residual_history_m=residual,
        observed_validity=validity,
        partition=partition,
        provider_reconstruction_artifact_id=provider_reconstruction_artifact_id,
        scoring_reconstruction_artifact_id=scoring_reconstruction_artifact_id,
        baseline_prefix_sha256=_array_sha256(baseline),
        observation_prefix_sha256=_array_sha256(observation),
        policy_id=_required_sha256(policy.policy_id, name="policy_id"),
        metadata={} if metadata is None else metadata,
    )


__all__ = [
    "ResidualHistoryAdapterV1",
    "build_residual_history_adapter",
]
