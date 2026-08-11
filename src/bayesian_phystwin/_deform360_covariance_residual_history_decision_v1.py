"""Admission records and exact fallback for the source-only contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_integer,
    plain_json,
)
from ._deform360_covariance_residual_history_adapter_v1 import (
    ResidualHistoryAdapterV1,
)
from ._deform360_covariance_residual_history_common_v1 import (
    CLAIM_BOUNDARY,
    FALLBACK_SEMANTICS,
    HORIZON_LABELS,
    REFERENCE_MEAN_SEMANTICS,
    REGISTERED_COVARIANCE_DONOR_ID,
    REGISTERED_MINIMUM_VALID_OBSERVATIONS_PER_MATERIAL,
    REGISTERED_REFERENCE_PREDICTOR_ID,
    RESIDUAL_HISTORY_DECISION_SCHEMA,
    SCHEMA_VERSION,
    _array_sha256,
    _canonical_string,
    _integer_vector,
    _required_sha256,
)
from ._portable_contracts import content_id
from .covariance_only_hybrid import CovarianceOnlyHybridPredictionV1
from .endpoint_model_average import MODEL_AVERAGED_ENDPOINT_CONTRACT_VERSION


def _count_tuple(value: object, *, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a nonempty integer sequence")
    result = tuple(
        genuine_integer(item, name=f"{name}[{index}]", minimum=0)
        for index, item in enumerate(value)
    )
    if not result:
        raise ValueError(f"{name} must be nonempty")
    return result


def _digest_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a nonempty SHA-256 sequence")
    result = tuple(_required_sha256(item, name=name) for item in value)
    if not result:
        raise ValueError(f"{name} must be nonempty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


@dataclass(frozen=True, slots=True)
class ResidualHistoryDryRunDecisionV1:
    """Content-addressed admission or exact whole-case fallback decision."""

    source_unit_id: str
    adapter_id: str
    policy_id: str
    family_map_id: str
    partition_id: str
    provider_reconstruction_manifest_id: str
    scoring_reconstruction_manifest_id: str
    registered_mean_sha256: str
    endpoint_contract_version: int
    endpoint_config_id: str
    endpoint_posterior_id: str
    endpoint_prediction_ids: tuple[str, ...]
    future_frame_indices_sha256: str
    future_horizon_steps_sha256: str
    unscaled_donor_covariance_sha256: str
    accepted: bool
    fallback_reasons: tuple[str, ...]
    valid_observation_count_by_material: tuple[int, ...]
    supported_material_count: int
    unsupported_material_count: int
    future_horizon_bins_sha256: str
    physical_future_mean_sha256: str
    physical_fallback_covariance_sha256: str
    deployed_mean_sha256: str
    deployed_covariance_sha256: str
    hybrid_artifact_id: str | None
    hybrid_registered_mean_identity_preserved: bool
    exact_physical_fallback_mean_identity_preserved: bool
    exact_physical_fallback_covariance_identity_preserved: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)
    decision_id: str | None = None

    def __post_init__(self) -> None:
        source_unit_id = _canonical_string(self.source_unit_id, name="source_unit_id")
        digest_fields = (
            "adapter_id",
            "policy_id",
            "family_map_id",
            "partition_id",
            "provider_reconstruction_manifest_id",
            "scoring_reconstruction_manifest_id",
            "registered_mean_sha256",
            "endpoint_config_id",
            "endpoint_posterior_id",
            "future_frame_indices_sha256",
            "future_horizon_steps_sha256",
            "unscaled_donor_covariance_sha256",
            "future_horizon_bins_sha256",
            "physical_future_mean_sha256",
            "physical_fallback_covariance_sha256",
            "deployed_mean_sha256",
            "deployed_covariance_sha256",
        )
        digests = {
            name: _required_sha256(getattr(self, name), name=name)
            for name in digest_fields
        }
        endpoint_version = genuine_integer(
            self.endpoint_contract_version,
            name="endpoint_contract_version",
            minimum=1,
        )
        if endpoint_version != MODEL_AVERAGED_ENDPOINT_CONTRACT_VERSION:
            raise ValueError("endpoint model contract version changed")
        prediction_ids = _digest_tuple(
            self.endpoint_prediction_ids,
            name="endpoint_prediction_ids",
        )
        if type(self.accepted) is not bool:
            raise ValueError("accepted must be a Boolean")
        if type(self.fallback_reasons) is not tuple:
            raise ValueError("fallback_reasons must be a canonical tuple")
        reasons = tuple(
            _canonical_string(value, name="fallback_reasons")
            for value in self.fallback_reasons
        )
        if reasons != tuple(sorted(set(reasons))):
            raise ValueError("fallback_reasons must be sorted and unique")
        counts = _count_tuple(
            self.valid_observation_count_by_material,
            name="valid_observation_count_by_material",
        )
        supported = genuine_integer(
            self.supported_material_count,
            name="supported_material_count",
            minimum=0,
        )
        unsupported = genuine_integer(
            self.unsupported_material_count,
            name="unsupported_material_count",
            minimum=0,
        )
        if supported + unsupported != len(counts):
            raise ValueError("material support counts do not exhaust the material roster")
        expected_supported = sum(
            count >= REGISTERED_MINIMUM_VALID_OBSERVATIONS_PER_MATERIAL
            for count in counts
        )
        if supported != expected_supported:
            raise ValueError("supported_material_count differs from the causal support")
        hybrid_id = self.hybrid_artifact_id
        if hybrid_id is not None:
            hybrid_id = _required_sha256(hybrid_id, name="hybrid_artifact_id")
        identity_fields = (
            "hybrid_registered_mean_identity_preserved",
            "exact_physical_fallback_mean_identity_preserved",
            "exact_physical_fallback_covariance_identity_preserved",
        )
        if any(type(getattr(self, name)) is not bool for name in identity_fields):
            raise ValueError("identity-preservation fields must be Booleans")
        if self.accepted:
            if reasons or hybrid_id is None or unsupported != 0:
                raise ValueError(
                    "accepted decisions require full support, one hybrid, and no fallback"
                )
            if not self.hybrid_registered_mean_identity_preserved:
                raise ValueError("accepted hybrid must preserve the registered mean object")
            if (
                self.exact_physical_fallback_mean_identity_preserved
                or self.exact_physical_fallback_covariance_identity_preserved
            ):
                raise ValueError("accepted decision must not claim fallback identity")
        else:
            if not reasons or hybrid_id is not None:
                raise ValueError("fallback decisions require reasons and no hybrid")
            if self.hybrid_registered_mean_identity_preserved:
                raise ValueError("fallback decision has no hybrid registered mean")
            if not (
                self.exact_physical_fallback_mean_identity_preserved
                and self.exact_physical_fallback_covariance_identity_preserved
            ):
                raise ValueError(
                    "fallback must preserve both physical objects by identity"
                )
        metadata = frozen_finite_json_mapping(self.metadata, name="metadata")
        object.__setattr__(self, "source_unit_id", source_unit_id)
        for name, value in digests.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "endpoint_contract_version", endpoint_version)
        object.__setattr__(self, "endpoint_prediction_ids", prediction_ids)
        object.__setattr__(self, "fallback_reasons", reasons)
        object.__setattr__(self, "valid_observation_count_by_material", counts)
        object.__setattr__(self, "supported_material_count", supported)
        object.__setattr__(self, "unsupported_material_count", unsupported)
        object.__setattr__(self, "hybrid_artifact_id", hybrid_id)
        object.__setattr__(self, "metadata", metadata)
        expected = content_id(self.descriptor())
        if self.decision_id is None:
            object.__setattr__(self, "decision_id", expected)
        elif _required_sha256(self.decision_id, name="decision_id") != expected:
            raise ValueError("decision_id does not match the source-only decision")

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": RESIDUAL_HISTORY_DECISION_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "source_unit_id": self.source_unit_id,
            "adapter_id": self.adapter_id,
            "policy_id": self.policy_id,
            "family_map_id": self.family_map_id,
            "partition_id": self.partition_id,
            "provider_reconstruction_manifest_id": (
                self.provider_reconstruction_manifest_id
            ),
            "scoring_reconstruction_manifest_id": (
                self.scoring_reconstruction_manifest_id
            ),
            "registered_mean_sha256": self.registered_mean_sha256,
            "endpoint_contract_version": self.endpoint_contract_version,
            "endpoint_config_id": self.endpoint_config_id,
            "endpoint_posterior_id": self.endpoint_posterior_id,
            "endpoint_prediction_ids": list(self.endpoint_prediction_ids),
            "future_frame_indices_sha256": self.future_frame_indices_sha256,
            "future_horizon_steps_sha256": self.future_horizon_steps_sha256,
            "unscaled_donor_covariance_sha256": (
                self.unscaled_donor_covariance_sha256
            ),
            "accepted": self.accepted,
            "fallback_reasons": list(self.fallback_reasons),
            "valid_observation_count_by_material": list(
                self.valid_observation_count_by_material
            ),
            "supported_material_count": self.supported_material_count,
            "unsupported_material_count": self.unsupported_material_count,
            "future_horizon_bins_sha256": self.future_horizon_bins_sha256,
            "physical_future_mean_sha256": self.physical_future_mean_sha256,
            "physical_fallback_covariance_sha256": (
                self.physical_fallback_covariance_sha256
            ),
            "deployed_mean_sha256": self.deployed_mean_sha256,
            "deployed_covariance_sha256": self.deployed_covariance_sha256,
            "hybrid_artifact_id": self.hybrid_artifact_id,
            "hybrid_registered_mean_identity_preserved": (
                self.hybrid_registered_mean_identity_preserved
            ),
            "exact_physical_fallback_mean_identity_preserved": (
                self.exact_physical_fallback_mean_identity_preserved
            ),
            "exact_physical_fallback_covariance_identity_preserved": (
                self.exact_physical_fallback_covariance_identity_preserved
            ),
            "reference_predictor_id": REGISTERED_REFERENCE_PREDICTOR_ID,
            "covariance_donor_id": REGISTERED_COVARIANCE_DONOR_ID,
            "reference_mean_semantics": REFERENCE_MEAN_SEMANTICS,
            "fallback_semantics": FALLBACK_SEMANTICS,
            "metadata": plain_json(self.metadata),
            "claim_boundary": CLAIM_BOUNDARY,
        }


@dataclass(frozen=True, slots=True)
class ResidualHistoryDryRunResultV1:
    """One source-only result and its runtime object-identity proofs."""

    mean_m: np.ndarray
    covariance_m2: np.ndarray
    adapter: ResidualHistoryAdapterV1
    decision: ResidualHistoryDryRunDecisionV1
    hybrid: CovarianceOnlyHybridPredictionV1 | None

    @property
    def accepted(self) -> bool:
        return self.decision.accepted


def _physical_future_mean(value: object, *, material_count: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("physical_future_m must be a NumPy array to preserve identity")
    if value.dtype != np.dtype(np.float64):
        raise ValueError("physical_future_m must have dtype float64")
    if value.ndim != 3 or value.shape[1:] != (material_count, 3) or value.shape[0] < 1:
        raise ValueError("physical_future_m must have shape (H, N, 3)")
    if not value.flags.c_contiguous:
        raise ValueError("physical_future_m must be C-contiguous")
    if not np.all(np.isfinite(value)):
        raise ValueError("physical_future_m must be finite")
    return value


def _registered_last_residual_mean(
    value: object,
    *,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(
            "registered_last_residual_mean_m must be a NumPy array to preserve identity"
        )
    if value.dtype != np.dtype(np.float64):
        raise ValueError("registered_last_residual_mean_m must have dtype float64")
    if value.shape != expected_shape:
        raise ValueError(
            f"registered_last_residual_mean_m must have shape {expected_shape}"
        )
    if not value.flags.c_contiguous:
        raise ValueError("registered_last_residual_mean_m must be C-contiguous")
    if not np.all(np.isfinite(value)):
        raise ValueError("registered_last_residual_mean_m must be finite")
    return value


def _horizon_bins(value: object, *, future_count: int) -> np.ndarray:
    bins = _integer_vector(value, name="future_horizon_bins")
    if bins.shape != (future_count,):
        raise ValueError("future_horizon_bins must have one entry per future frame")
    if np.any((bins < 0) | (bins >= len(HORIZON_LABELS))):
        raise ValueError(
            "future_horizon_bins must use early/middle/late indices 0, 1, 2"
        )
    return bins


def _future_frame_contract(
    value: object,
    *,
    causal_frame_indices: np.ndarray,
    future_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not np.all(np.diff(causal_frame_indices) == 1):
        raise ValueError("registered endpoint inference requires contiguous causal frames")
    future = _integer_vector(value, name="future_frame_indices")
    if future.shape != (future_count,):
        raise ValueError("future_frame_indices must have one entry per future frame")
    if not np.all(np.diff(future) > 0):
        raise ValueError("future_frame_indices must be strictly increasing")
    causal_last = int(causal_frame_indices[-1])
    if np.any(future <= causal_last):
        raise ValueError("future_frame_indices must follow the causal prefix")
    steps = np.asarray(future - causal_last, dtype=np.int64)
    steps.setflags(write=False)
    return future, steps


def _fallback_result(
    *,
    physical_future: np.ndarray,
    physical_covariance: np.ndarray,
    registered_mean: np.ndarray,
    adapter: ResidualHistoryAdapterV1,
    horizon_bins: np.ndarray,
    future_frame_indices: np.ndarray,
    future_horizon_steps: np.ndarray,
    endpoint_config_id: str,
    endpoint_posterior_id: str,
    endpoint_prediction_ids: tuple[str, ...],
    unscaled_donor_covariance: np.ndarray,
    reasons: Sequence[str],
    metadata: Mapping[str, Any] | None,
) -> ResidualHistoryDryRunResultV1:
    canonical_reasons = tuple(sorted(set(reasons)))
    decision = ResidualHistoryDryRunDecisionV1(
        source_unit_id=adapter.source_unit_id,
        adapter_id=_required_sha256(adapter.adapter_id, name="adapter_id"),
        policy_id=_required_sha256(adapter.policy.policy_id, name="policy_id"),
        family_map_id=_required_sha256(
            adapter.partition.family_map.map_id,
            name="family_map_id",
        ),
        partition_id=_required_sha256(
            adapter.partition.partition_id,
            name="partition_id",
        ),
        provider_reconstruction_manifest_id=_required_sha256(
            adapter.provider_reconstruction_manifest.manifest_id,
            name="provider_reconstruction_manifest_id",
        ),
        scoring_reconstruction_manifest_id=_required_sha256(
            adapter.scoring_reconstruction_manifest.manifest_id,
            name="scoring_reconstruction_manifest_id",
        ),
        registered_mean_sha256=_array_sha256(registered_mean),
        endpoint_contract_version=MODEL_AVERAGED_ENDPOINT_CONTRACT_VERSION,
        endpoint_config_id=endpoint_config_id,
        endpoint_posterior_id=endpoint_posterior_id,
        endpoint_prediction_ids=endpoint_prediction_ids,
        future_frame_indices_sha256=_array_sha256(future_frame_indices),
        future_horizon_steps_sha256=_array_sha256(future_horizon_steps),
        unscaled_donor_covariance_sha256=_array_sha256(
            unscaled_donor_covariance
        ),
        accepted=False,
        fallback_reasons=canonical_reasons,
        valid_observation_count_by_material=(
            adapter.valid_observation_count_by_material
        ),
        supported_material_count=adapter.supported_material_count,
        unsupported_material_count=adapter.unsupported_material_count,
        future_horizon_bins_sha256=_array_sha256(horizon_bins),
        physical_future_mean_sha256=_array_sha256(physical_future),
        physical_fallback_covariance_sha256=_array_sha256(physical_covariance),
        deployed_mean_sha256=_array_sha256(physical_future),
        deployed_covariance_sha256=_array_sha256(physical_covariance),
        hybrid_artifact_id=None,
        hybrid_registered_mean_identity_preserved=False,
        exact_physical_fallback_mean_identity_preserved=True,
        exact_physical_fallback_covariance_identity_preserved=True,
        metadata={} if metadata is None else metadata,
    )
    result = ResidualHistoryDryRunResultV1(
        mean_m=physical_future,
        covariance_m2=physical_covariance,
        adapter=adapter,
        decision=decision,
        hybrid=None,
    )
    if (
        result.mean_m is not physical_future
        or result.covariance_m2 is not physical_covariance
    ):
        raise AssertionError("exact physical fallback object identity was not preserved")
    return result


__all__ = [
    "ResidualHistoryDryRunDecisionV1",
    "ResidualHistoryDryRunResultV1",
]
