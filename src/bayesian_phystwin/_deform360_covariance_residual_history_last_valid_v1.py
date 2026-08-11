"""Registered exact-mean covariance composition from a causal residual history."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ._canonical_contracts import plain_json
from ._deform360_covariance_residual_history_adapter_v1 import (
    ResidualHistoryAdapterV1,
    build_residual_history_adapter,
)
from ._deform360_covariance_residual_history_common_v1 import (
    HORIZON_LABELS,
    REGISTERED_COVARIANCE_DONOR_ID,
    REGISTERED_REFERENCE_PREDICTOR_ID,
    CameraRecorderFamilyMapV1,
    ReconstructionManifestV1,
    ResidualHistoryDryRunPolicyV1,
    _array_sha256,
    _required_sha256,
    _validate_covariance,
)
from ._deform360_covariance_residual_history_decision_v1 import (
    ResidualHistoryDryRunDecisionV1,
    ResidualHistoryDryRunResultV1,
    _fallback_result,
    _horizon_bins,
    _physical_future_mean,
    _registered_last_residual_mean,
)
from .covariance_only_hybrid import compose_covariance_only_hybrid


def _last_valid_residual(adapter: ResidualHistoryAdapterV1) -> np.ndarray:
    """Return each material's last valid causal residual without filling history."""

    result = np.zeros((adapter.material_count, 3), dtype=np.float64)
    for material_index in range(adapter.material_count):
        support = np.flatnonzero(adapter.observed_validity[:, material_index])
        if len(support):
            result[material_index] = adapter.residual_history_m[
                support[-1], material_index
            ]
    return result


def _verify_registered_mean(
    registered_mean: np.ndarray,
    *,
    physical_future: np.ndarray,
    adapter: ResidualHistoryAdapterV1,
) -> None:
    expected = np.array(physical_future, dtype=np.float64, copy=True, order="C")
    expected += _last_valid_residual(adapter)[None, ...]
    if registered_mean.tobytes(order="C") != expected.tobytes(order="C"):
        raise ValueError(
            "registered_last_residual_mean_m differs from the causal last-valid mean"
        )


def run_source_only_residual_history_dry_run(
    physical_prefix_m: object,
    provider_observation_prefix_m: object,
    observed_validity: object,
    physical_future_m: np.ndarray,
    physical_fallback_covariance_m2: np.ndarray,
    registered_last_residual_mean_m: np.ndarray,
    donor_covariance_m2: object,
    *,
    frame_indices: object,
    material_ids: object,
    future_horizon_bins: object,
    camera_recorder_family_map: CameraRecorderFamilyMapV1,
    provider_reconstruction_manifest: ReconstructionManifestV1,
    scoring_reconstruction_manifest: ReconstructionManifestV1,
    source_unit_id: str,
    policy: ResidualHistoryDryRunPolicyV1,
    metadata: Mapping[str, Any] | None = None,
) -> ResidualHistoryDryRunResultV1:
    """Use the exact registered comparator mean or return exact physical fallback."""

    adapter = build_residual_history_adapter(
        physical_prefix_m,
        provider_observation_prefix_m,
        observed_validity,
        frame_indices=frame_indices,
        material_ids=material_ids,
        camera_recorder_family_map=camera_recorder_family_map,
        provider_reconstruction_manifest=provider_reconstruction_manifest,
        scoring_reconstruction_manifest=scoring_reconstruction_manifest,
        source_unit_id=source_unit_id,
        policy=policy,
        metadata=metadata,
    )
    physical_future = _physical_future_mean(
        physical_future_m,
        material_count=adapter.material_count,
    )
    future_shape = physical_future.shape
    physical_covariance = _validate_covariance(
        physical_fallback_covariance_m2,
        name="physical_fallback_covariance_m2",
        expected_shape=future_shape + (3,),
        preserve_identity=True,
    )
    registered_mean = _registered_last_residual_mean(
        registered_last_residual_mean_m,
        expected_shape=future_shape,
    )
    _verify_registered_mean(
        registered_mean,
        physical_future=physical_future,
        adapter=adapter,
    )
    bins = _horizon_bins(future_horizon_bins, future_count=future_shape[0])
    if adapter.unsupported_material_count:
        return _fallback_result(
            physical_future=physical_future,
            physical_covariance=physical_covariance,
            registered_mean=registered_mean,
            adapter=adapter,
            horizon_bins=bins,
            reasons=("insufficient-per-material-support",),
            metadata=metadata,
        )

    scales = np.asarray(policy.covariance_scales, dtype=np.float64)[bins]
    try:
        hybrid = compose_covariance_only_hybrid(
            registered_mean,
            donor_covariance_m2,
            reference_predictor_id=REGISTERED_REFERENCE_PREDICTOR_ID,
            covariance_donor_id=REGISTERED_COVARIANCE_DONOR_ID,
            covariance_scale=scales[:, None],
            metadata={
                "source_unit_id": adapter.source_unit_id,
                "adapter_id": _required_sha256(
                    adapter.adapter_id,
                    name="adapter_id",
                ),
                "policy_id": _required_sha256(
                    adapter.policy.policy_id,
                    name="policy_id",
                ),
                "family_map_id": _required_sha256(
                    adapter.partition.family_map.map_id,
                    name="family_map_id",
                ),
                "partition_id": _required_sha256(
                    adapter.partition.partition_id,
                    name="partition_id",
                ),
                "provider_reconstruction_manifest_id": _required_sha256(
                    adapter.provider_reconstruction_manifest.manifest_id,
                    name="provider_reconstruction_manifest_id",
                ),
                "scoring_reconstruction_manifest_id": _required_sha256(
                    adapter.scoring_reconstruction_manifest.manifest_id,
                    name="scoring_reconstruction_manifest_id",
                ),
                "future_horizon_labels": [HORIZON_LABELS[int(value)] for value in bins],
            },
        )
    except (TypeError, ValueError) as error:
        fallback_metadata: dict[str, Any] = {
            "covariance_rejection_type": type(error).__name__,
            "covariance_rejection_message": str(error),
        }
        if metadata is not None:
            fallback_metadata["source_metadata"] = plain_json(metadata)
        return _fallback_result(
            physical_future=physical_future,
            physical_covariance=physical_covariance,
            registered_mean=registered_mean,
            adapter=adapter,
            horizon_bins=bins,
            reasons=("covariance-contract-rejection",),
            metadata=fallback_metadata,
        )
    if hybrid.mean_m is not registered_mean:
        raise AssertionError("covariance-only helper copied the registered mean")

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
        accepted=True,
        fallback_reasons=(),
        valid_observation_count_by_material=(
            adapter.valid_observation_count_by_material
        ),
        supported_material_count=adapter.supported_material_count,
        unsupported_material_count=adapter.unsupported_material_count,
        future_horizon_bins_sha256=_array_sha256(bins),
        physical_future_mean_sha256=_array_sha256(physical_future),
        physical_fallback_covariance_sha256=_array_sha256(physical_covariance),
        deployed_mean_sha256=_array_sha256(hybrid.mean_m),
        deployed_covariance_sha256=_array_sha256(hybrid.covariance_m2),
        hybrid_artifact_id=_required_sha256(
            hybrid.record.artifact_id,
            name="hybrid_artifact_id",
        ),
        hybrid_registered_mean_identity_preserved=True,
        exact_physical_fallback_mean_identity_preserved=False,
        exact_physical_fallback_covariance_identity_preserved=False,
        metadata={} if metadata is None else metadata,
    )
    return ResidualHistoryDryRunResultV1(
        mean_m=hybrid.mean_m,
        covariance_m2=hybrid.covariance_m2,
        adapter=adapter,
        decision=decision,
        hybrid=hybrid,
    )


__all__ = ["run_source_only_residual_history_dry_run"]
