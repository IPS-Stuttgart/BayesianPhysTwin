"""Registered exact-mean covariance composition from a causal residual history."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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
    _future_frame_contract,
    _horizon_bins,
    _physical_future_mean,
    _registered_last_residual_mean,
)
from ._portable_contracts import content_id
from .contracts.fixed_anchor import FIXED_BAYESIAN_ANCHOR_CONTRACT_VERSION
from .covariance_only_hybrid import compose_covariance_only_hybrid
from .endpoint_model_average import (
    DEFAULT_MODEL_AVERAGED_ENDPOINT_CONFIG_V1,
    MODEL_AVERAGED_ENDPOINT_CONTRACT_VERSION,
    ModelAveragedEndpointConfigV1,
    ModelAveragedEndpointPosteriorV1,
    ModelAveragedEndpointPredictionV1,
    infer_model_averaged_endpoint,
    predict_model_averaged_endpoint,
)


@dataclass(frozen=True, slots=True)
class _EndpointDonorArtifacts:
    covariance_m2: np.ndarray
    config_id: str
    posterior_id: str
    prediction_ids: tuple[str, ...]


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


def _endpoint_config_descriptor(
    config: ModelAveragedEndpointConfigV1,
) -> dict[str, Any]:
    return {
        "schema": "bayesian-phystwin.endpoint-model-average-config-identity-v1",
        "schema_version": 1,
        "endpoint_contract_version": MODEL_AVERAGED_ENDPOINT_CONTRACT_VERSION,
        "fixed_anchor_contract_version": FIXED_BAYESIAN_ANCHOR_CONTRACT_VERSION,
        "components": [
            {
                "process_std_m": component.process_std_m,
                "observation_std_m": component.observation_std_m,
                "initial_std_m": component.initial_std_m,
                "inlier_prior": component.inlier_prior,
                "outlier_variance_multiplier": (
                    component.outlier_variance_multiplier
                ),
            }
            for component in config.components
        ],
        "component_prior_probability": list(
            config.component_prior_probability or ()
        ),
    }


def _posterior_id(
    posterior: ModelAveragedEndpointPosteriorV1,
    *,
    config_id: str,
) -> str:
    return content_id(
        {
            "schema": "bayesian-phystwin.endpoint-model-average-posterior-identity-v1",
            "schema_version": 1,
            "endpoint_contract_version": MODEL_AVERAGED_ENDPOINT_CONTRACT_VERSION,
            "config_id": config_id,
            "end_frame": posterior.end_frame,
            "mean_sha256": _array_sha256(posterior.mean_m),
            "covariance_sha256": _array_sha256(posterior.covariance_m2),
            "final_nominal_probability_sha256": _array_sha256(
                posterior.final_nominal_probability
            ),
            "update_count_sha256": _array_sha256(posterior.update_count),
            "component_weights_sha256": _array_sha256(
                posterior.component_weights
            ),
            "component_log_evidence_sha256": _array_sha256(
                posterior.component_log_evidence
            ),
            "component_mean_sha256": _array_sha256(
                posterior.component_mean_m
            ),
            "component_variance_sha256": _array_sha256(
                posterior.component_variance_m2
            ),
            "component_process_variance_sha256": _array_sha256(
                posterior.component_process_variance_m2
            ),
        }
    )


def _prediction_id(
    prediction: ModelAveragedEndpointPredictionV1,
    *,
    config_id: str,
    posterior_id: str,
    future_frame_index: int,
) -> str:
    return content_id(
        {
            "schema": "bayesian-phystwin.endpoint-model-average-prediction-identity-v1",
            "schema_version": 1,
            "endpoint_contract_version": MODEL_AVERAGED_ENDPOINT_CONTRACT_VERSION,
            "config_id": config_id,
            "posterior_id": posterior_id,
            "future_frame_index": int(future_frame_index),
            "horizon_steps": prediction.horizon_steps,
            "mean_sha256": _array_sha256(prediction.mean_m),
            "covariance_sha256": _array_sha256(prediction.covariance_m2),
            "component_weights_sha256": _array_sha256(
                prediction.component_weights
            ),
        }
    )


def _reproduce_endpoint_donor(
    adapter: ResidualHistoryAdapterV1,
    *,
    future_frame_indices: np.ndarray,
    future_horizon_steps: np.ndarray,
) -> _EndpointDonorArtifacts:
    config = DEFAULT_MODEL_AVERAGED_ENDPOINT_CONFIG_V1
    config_id = content_id(_endpoint_config_descriptor(config))
    posterior = infer_model_averaged_endpoint(
        adapter.residual_history_m,
        adapter.observed_validity,
        end_frame=adapter.prefix_frame_count,
        config=config,
    )
    expected_count = np.asarray(
        adapter.valid_observation_count_by_material,
        dtype=np.int64,
    )
    if not np.array_equal(posterior.update_count, expected_count):
        raise AssertionError("endpoint posterior used a different validity history")
    posterior_id = _posterior_id(posterior, config_id=config_id)
    predictions = tuple(
        predict_model_averaged_endpoint(
            posterior,
            horizon_steps=int(step),
        )
        for step in future_horizon_steps
    )
    covariance = np.stack(
        [prediction.covariance_m2 for prediction in predictions]
    )
    covariance.setflags(write=False)
    prediction_ids = tuple(
        _prediction_id(
            prediction,
            config_id=config_id,
            posterior_id=posterior_id,
            future_frame_index=int(frame),
        )
        for frame, prediction in zip(
            future_frame_indices,
            predictions,
            strict=True,
        )
    )
    return _EndpointDonorArtifacts(
        covariance_m2=covariance,
        config_id=config_id,
        posterior_id=posterior_id,
        prediction_ids=prediction_ids,
    )


def run_source_only_residual_history_dry_run(
    physical_prefix_m: object,
    provider_observation_prefix_m: object,
    observed_validity: object,
    physical_future_m: np.ndarray,
    physical_fallback_covariance_m2: np.ndarray,
    registered_last_residual_mean_m: np.ndarray,
    *,
    frame_indices: object,
    material_ids: object,
    future_frame_indices: object,
    future_horizon_bins: object,
    camera_recorder_family_map: CameraRecorderFamilyMapV1,
    provider_reconstruction_manifest: ReconstructionManifestV1,
    scoring_reconstruction_manifest: ReconstructionManifestV1,
    source_unit_id: str,
    policy: ResidualHistoryDryRunPolicyV1,
    metadata: Mapping[str, Any] | None = None,
) -> ResidualHistoryDryRunResultV1:
    """Reproduce the registered donor or return exact physical fallback."""

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
    future_frames, horizon_steps = _future_frame_contract(
        future_frame_indices,
        causal_frame_indices=adapter.frame_indices,
        future_count=future_shape[0],
    )
    donor = _reproduce_endpoint_donor(
        adapter,
        future_frame_indices=future_frames,
        future_horizon_steps=horizon_steps,
    )
    if adapter.unsupported_material_count:
        return _fallback_result(
            physical_future=physical_future,
            physical_covariance=physical_covariance,
            registered_mean=registered_mean,
            adapter=adapter,
            horizon_bins=bins,
            future_frame_indices=future_frames,
            future_horizon_steps=horizon_steps,
            endpoint_config_id=donor.config_id,
            endpoint_posterior_id=donor.posterior_id,
            endpoint_prediction_ids=donor.prediction_ids,
            unscaled_donor_covariance=donor.covariance_m2,
            reasons=("insufficient-per-material-support",),
            metadata=metadata,
        )

    scales = np.asarray(policy.covariance_scales, dtype=np.float64)[bins]
    try:
        hybrid = compose_covariance_only_hybrid(
            registered_mean,
            donor.covariance_m2,
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
                "endpoint_contract_version": (
                    MODEL_AVERAGED_ENDPOINT_CONTRACT_VERSION
                ),
                "endpoint_config_id": donor.config_id,
                "endpoint_posterior_id": donor.posterior_id,
                "endpoint_prediction_ids": list(donor.prediction_ids),
                "future_frame_indices": future_frames.tolist(),
                "future_horizon_steps": horizon_steps.tolist(),
                "future_horizon_labels": [
                    HORIZON_LABELS[int(value)] for value in bins
                ],
                "unscaled_donor_covariance_sha256": _array_sha256(
                    donor.covariance_m2
                ),
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
            future_frame_indices=future_frames,
            future_horizon_steps=horizon_steps,
            endpoint_config_id=donor.config_id,
            endpoint_posterior_id=donor.posterior_id,
            endpoint_prediction_ids=donor.prediction_ids,
            unscaled_donor_covariance=donor.covariance_m2,
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
        endpoint_contract_version=MODEL_AVERAGED_ENDPOINT_CONTRACT_VERSION,
        endpoint_config_id=donor.config_id,
        endpoint_posterior_id=donor.posterior_id,
        endpoint_prediction_ids=donor.prediction_ids,
        future_frame_indices_sha256=_array_sha256(future_frames),
        future_horizon_steps_sha256=_array_sha256(horizon_steps),
        unscaled_donor_covariance_sha256=_array_sha256(
            donor.covariance_m2
        ),
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
