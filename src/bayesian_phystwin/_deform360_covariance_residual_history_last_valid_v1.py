"""Exact last-valid residual mean for the covariance-only Deform360 dry run."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ._canonical_contracts import plain_json
from ._deform360_covariance_residual_history_adapter_v1 import (
    ResidualHistoryAdapterV1,
    build_residual_history_adapter,
)
from ._deform360_covariance_residual_history_common_v1 import (
    HORIZON_LABELS,
    ResidualHistoryDryRunPolicyV1,
    _array_sha256,
    _canonical_string,
    _required_sha256,
    _validate_covariance,
)
from ._deform360_covariance_residual_history_decision_v1 import (
    ResidualHistoryDryRunDecisionV1,
    ResidualHistoryDryRunResultV1,
    _fallback_result,
    _horizon_bins,
    _physical_future_mean,
)
from .covariance_only_hybrid import compose_covariance_only_hybrid


def _last_valid_residual(adapter: ResidualHistoryAdapterV1) -> np.ndarray:
    """Return each material's last valid causal residual without filling history."""

    result: np.ndarray = np.zeros((adapter.material_count, 3), dtype=np.float64)
    for material_index in range(adapter.material_count):
        support = np.flatnonzero(adapter.observed_validity[:, material_index])
        if len(support):
            result[material_index] = adapter.residual_history_m[
                support[-1], material_index
            ]
    return result


def run_source_only_residual_history_dry_run(
    physical_prefix_m: object,
    provider_observation_prefix_m: object,
    observed_validity: object,
    physical_future_m: np.ndarray,
    physical_fallback_covariance_m2: np.ndarray,
    donor_covariance_m2: object,
    *,
    frame_indices: object,
    material_ids: object,
    future_horizon_bins: object,
    camera_ids: Sequence[str],
    provider_camera_ids: Sequence[str],
    scoring_camera_ids: Sequence[str],
    provider_reconstruction_artifact_id: str,
    scoring_reconstruction_artifact_id: str,
    source_unit_id: str,
    reference_predictor_id: str,
    covariance_donor_id: str,
    policy: ResidualHistoryDryRunPolicyV1,
    metadata: Mapping[str, Any] | None = None,
) -> ResidualHistoryDryRunResultV1:
    """Deploy the exact last-valid mean with covariance only, or exact fallback."""

    adapter = build_residual_history_adapter(
        physical_prefix_m,
        provider_observation_prefix_m,
        observed_validity,
        frame_indices=frame_indices,
        material_ids=material_ids,
        camera_ids=camera_ids,
        provider_camera_ids=provider_camera_ids,
        scoring_camera_ids=scoring_camera_ids,
        provider_reconstruction_artifact_id=provider_reconstruction_artifact_id,
        scoring_reconstruction_artifact_id=scoring_reconstruction_artifact_id,
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
    bins = _horizon_bins(future_horizon_bins, future_count=future_shape[0])
    reasons: list[str] = []
    if adapter.final_observed_count < policy.minimum_final_observed_count:
        reasons.append("minimum-final-observed-count")
    if adapter.final_observed_fraction < policy.minimum_final_observed_fraction:
        reasons.append("minimum-final-observed-fraction")
    if reasons:
        return _fallback_result(
            physical_future=physical_future,
            physical_covariance=physical_covariance,
            adapter=adapter,
            policy=policy,
            horizon_bins=bins,
            reasons=reasons,
            metadata=metadata,
        )

    reference_mean = np.array(physical_future, dtype=np.float64, copy=True, order="C")
    reference_mean += _last_valid_residual(adapter)[None, ...]
    scales = np.asarray(policy.covariance_scales, dtype=np.float64)[bins]
    try:
        hybrid = compose_covariance_only_hybrid(
            reference_mean,
            donor_covariance_m2,
            reference_predictor_id=_canonical_string(
                reference_predictor_id,
                name="reference_predictor_id",
            ),
            covariance_donor_id=_canonical_string(
                covariance_donor_id,
                name="covariance_donor_id",
            ),
            covariance_scale=scales[:, None],
            metadata={
                "source_unit_id": adapter.source_unit_id,
                "adapter_id": _required_sha256(
                    adapter.adapter_id,
                    name="adapter_id",
                ),
                "policy_id": policy.policy_id,
                "partition_id": _required_sha256(
                    adapter.partition.partition_id,
                    name="partition_id",
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
            fallback_metadata["dry_run_metadata"] = plain_json(metadata)
        return _fallback_result(
            physical_future=physical_future,
            physical_covariance=physical_covariance,
            adapter=adapter,
            policy=policy,
            horizon_bins=bins,
            reasons=("covariance-contract-rejection",),
            metadata=fallback_metadata,
        )
    if hybrid.mean_m is not reference_mean:
        raise AssertionError("covariance-only helper copied the last-residual mean")

    adapter_id = _required_sha256(adapter.adapter_id, name="adapter_id")
    partition_id = _required_sha256(
        adapter.partition.partition_id,
        name="partition_id",
    )
    hybrid_id = _required_sha256(
        hybrid.record.artifact_id,
        name="hybrid_artifact_id",
    )
    decision = ResidualHistoryDryRunDecisionV1(
        source_unit_id=adapter.source_unit_id,
        adapter_id=adapter_id,
        policy_id=_required_sha256(policy.policy_id, name="policy_id"),
        partition_id=partition_id,
        accepted=True,
        fallback_reasons=(),
        final_observed_count=adapter.final_observed_count,
        final_observed_fraction=adapter.final_observed_fraction,
        future_horizon_bins_sha256=_array_sha256(bins),
        physical_future_mean_sha256=_array_sha256(physical_future),
        physical_fallback_covariance_sha256=_array_sha256(physical_covariance),
        deployed_mean_sha256=_array_sha256(hybrid.mean_m),
        deployed_covariance_sha256=_array_sha256(hybrid.covariance_m2),
        hybrid_artifact_id=hybrid_id,
        hybrid_reference_mean_identity_preserved=True,
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
