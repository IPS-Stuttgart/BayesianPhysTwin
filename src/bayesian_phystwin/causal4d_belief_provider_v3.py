"""NumPy-only dynamic endpoint and recursive belief surface for Causal4D.

Provider v2 remains unchanged for frozen model-averaged random-walk experiments.
This additive provider retains its recursive Prob4D stream surface while adding
an exact persistence comparator and robust damped-trend endpoint components.
"""

from __future__ import annotations

import os

import numpy as np

from .causal4d_belief_provider_v2 import (
    CAUSAL4D_BELIEF_PROVIDER_V2_ARTIFACT_SCHEMA_VERSIONS,
    CAUSAL4D_BELIEF_PROVIDER_V2_CAPABILITIES,
    CAUSAL4D_BELIEF_PROVIDER_V2_PACKAGE_VERSION,
    ClaimBearingProb4DStreamRunV1,
    ClaimBearingProb4DStreamStepV1,
    HorizonConditionedEndpointPredictionV1,
    HorizonDiscrepancyCalibrationV1,
    ModelAveragedEndpointConfigV1,
    ModelAveragedEndpointPosteriorV1,
    ModelAveragedEndpointPredictionV1,
    PosteriorCovarianceSemanticsV1,
    Prob4DObservationFactorStreamV1,
    Prob4DStreamObservationBindingV1,
    RecursiveNuisancePolicyV1,
    apply_claim_bearing_prob4d_stream_update,
    bind_prob4d_stream_observation,
    load_claim_bearing_prob4d_stream_run,
    load_prob4d_observation_factor_stream,
    predict_horizon_conditioned_endpoint,
    predict_model_averaged_endpoint,
    start_claim_bearing_prob4d_stream_run,
    working_irls_covariance_semantics,
    write_claim_bearing_prob4d_stream_run,
)
from .contracts.provider import (
    installed_distribution_revision,
    installed_distribution_version,
)
from .dynamic_endpoint_model_average import (
    DEFAULT_DYNAMIC_ENDPOINT_MODEL_AVERAGE_CONFIG_V2,
    DYNAMIC_ENDPOINT_MODEL_AVERAGE_CONTRACT_VERSION,
    DampedTrendEndpointComponentV2,
    DynamicEndpointModelAverageConfigV2,
    DynamicEndpointNumericalError,
    DynamicEndpointPosteriorV2,
    DynamicEndpointPredictionV2,
    PersistenceEndpointComponentV2,
    infer_dynamic_endpoint_model_average,
    predict_dynamic_endpoint_model_average,
)

CAUSAL4D_BELIEF_PROVIDER_V3_API_VERSION = 3
CAUSAL4D_BELIEF_PROVIDER_V3_PACKAGE_VERSION = (
    CAUSAL4D_BELIEF_PROVIDER_V2_PACKAGE_VERSION
)
_NEW_CAPABILITIES = (
    "exact_last_residual_component",
    "robust_local_level_components",
    "robust_damped_trend_components",
    "horizon_dependent_predictive_mean",
    "fail_closed_dynamic_covariance",
    "per_track_or_object_pooled_component_evidence",
)
CAUSAL4D_BELIEF_PROVIDER_V3_CAPABILITIES = tuple(
    dict.fromkeys((*CAUSAL4D_BELIEF_PROVIDER_V2_CAPABILITIES, *_NEW_CAPABILITIES))
)
CAUSAL4D_BELIEF_PROVIDER_V3_ARTIFACT_SCHEMA_VERSIONS = {
    **CAUSAL4D_BELIEF_PROVIDER_V2_ARTIFACT_SCHEMA_VERSIONS,
    "DynamicEndpointModelAverageConfig": (
        DYNAMIC_ENDPOINT_MODEL_AVERAGE_CONTRACT_VERSION
    ),
    "DynamicEndpointPosterior": DYNAMIC_ENDPOINT_MODEL_AVERAGE_CONTRACT_VERSION,
    "DynamicEndpointPrediction": DYNAMIC_ENDPOINT_MODEL_AVERAGE_CONTRACT_VERSION,
}


def infer_dynamic_bayesian_anchor_endpoint(
    residual_m: np.ndarray,
    valid: np.ndarray,
    *,
    end_frame: int,
    config: DynamicEndpointModelAverageConfigV2 | None = None,
) -> DynamicEndpointPosteriorV2:
    """Infer a dynamic robust endpoint from an exclusive causal prefix."""

    settings = (
        DEFAULT_DYNAMIC_ENDPOINT_MODEL_AVERAGE_CONFIG_V2
        if config is None
        else config
    )
    if not isinstance(settings, DynamicEndpointModelAverageConfigV2):
        raise TypeError("config must be a DynamicEndpointModelAverageConfigV2")
    return infer_dynamic_endpoint_model_average(
        residual_m,
        valid,
        end_frame=end_frame,
        config=settings,
    )


def causal4d_belief_provider_v3_manifest(
    *,
    provider_revision: str | None = None,
) -> dict[str, object]:
    """Return the dynamic endpoint and recursive-belief capability descriptor."""

    revision = (
        provider_revision
        or os.environ.get("BAYESIAN_PHYSTWIN_REVISION")
        or installed_distribution_revision("bayesian-phystwin")
        or "unversioned-install"
    )
    return {
        "provider_name": "bayesian-phystwin",
        "provider_version": installed_distribution_version(
            "bayesian-phystwin",
            fallback=CAUSAL4D_BELIEF_PROVIDER_V3_PACKAGE_VERSION,
        ),
        "provider_revision": revision,
        "schema_version": CAUSAL4D_BELIEF_PROVIDER_V3_API_VERSION,
        "capabilities": list(CAUSAL4D_BELIEF_PROVIDER_V3_CAPABILITIES),
        "artifact_schema_versions": dict(
            CAUSAL4D_BELIEF_PROVIDER_V3_ARTIFACT_SCHEMA_VERSIONS
        ),
        "metadata": {
            "provider_api": "bayesian_phystwin.causal4d_belief_provider_v3",
            "provider_api_version": CAUSAL4D_BELIEF_PROVIDER_V3_API_VERSION,
            "inference_role": (
                "evidence-weighted persistent, local-level, and damped-trend "
                "readout-discrepancy endpoint"
            ),
            "compatibility": (
                "additive provider; causal4d_belief_provider_v2 and frozen "
                "provider-v1 experiments are unchanged"
            ),
            "component_prior": (
                "equal prior mass per dynamics family by default; explicit "
                "component probabilities override family balancing"
            ),
            "evidence_pooling": (
                "per-track by default; object-pooled weights are an explicit "
                "source-frozen option"
            ),
            "raw_covariance_claim": (
                "model-based predictive covariance including within-component "
                "uncertainty and between-component disagreement; frequentist "
                "coverage requires independent calibration"
            ),
            "recursive_stream_claim": (
                "the provider-v2 Prob4D recursive stream surface is retained "
                "without changing its contracts or exact fallback behavior"
            ),
        },
    }


__all__ = [
    "CAUSAL4D_BELIEF_PROVIDER_V3_API_VERSION",
    "CAUSAL4D_BELIEF_PROVIDER_V3_ARTIFACT_SCHEMA_VERSIONS",
    "CAUSAL4D_BELIEF_PROVIDER_V3_CAPABILITIES",
    "CAUSAL4D_BELIEF_PROVIDER_V3_PACKAGE_VERSION",
    "DEFAULT_DYNAMIC_ENDPOINT_MODEL_AVERAGE_CONFIG_V2",
    "ClaimBearingProb4DStreamRunV1",
    "ClaimBearingProb4DStreamStepV1",
    "DampedTrendEndpointComponentV2",
    "DynamicEndpointModelAverageConfigV2",
    "DynamicEndpointNumericalError",
    "DynamicEndpointPosteriorV2",
    "DynamicEndpointPredictionV2",
    "HorizonConditionedEndpointPredictionV1",
    "HorizonDiscrepancyCalibrationV1",
    "ModelAveragedEndpointConfigV1",
    "ModelAveragedEndpointPosteriorV1",
    "ModelAveragedEndpointPredictionV1",
    "PersistenceEndpointComponentV2",
    "PosteriorCovarianceSemanticsV1",
    "Prob4DObservationFactorStreamV1",
    "Prob4DStreamObservationBindingV1",
    "RecursiveNuisancePolicyV1",
    "apply_claim_bearing_prob4d_stream_update",
    "bind_prob4d_stream_observation",
    "causal4d_belief_provider_v3_manifest",
    "infer_dynamic_bayesian_anchor_endpoint",
    "load_claim_bearing_prob4d_stream_run",
    "load_prob4d_observation_factor_stream",
    "predict_dynamic_endpoint_model_average",
    "predict_horizon_conditioned_endpoint",
    "predict_model_averaged_endpoint",
    "start_claim_bearing_prob4d_stream_run",
    "working_irls_covariance_semantics",
    "write_claim_bearing_prob4d_stream_run",
]
