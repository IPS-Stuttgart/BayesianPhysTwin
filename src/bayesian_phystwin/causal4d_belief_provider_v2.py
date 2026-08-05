"""NumPy-only model-averaged discrepancy endpoint for Causal4D consumers."""

from __future__ import annotations

import os

import numpy as np

from .contracts.provider import (
    installed_distribution_revision,
    installed_distribution_version,
)
from .endpoint_model_average import (
    DEFAULT_MODEL_AVERAGED_ENDPOINT_CONFIG_V1,
    MODEL_AVERAGED_ENDPOINT_CONTRACT_VERSION,
    ModelAveragedEndpointConfigV1,
    ModelAveragedEndpointPosteriorV1,
    ModelAveragedEndpointPredictionV1,
    infer_model_averaged_endpoint,
    predict_model_averaged_endpoint,
)
from .horizon_conditioned_discrepancy import (
    HORIZON_DISCREPANCY_CALIBRATION_VERSION,
    HorizonConditionedEndpointPredictionV1,
    HorizonDiscrepancyCalibrationV1,
    predict_horizon_conditioned_endpoint,
)

CAUSAL4D_BELIEF_PROVIDER_V2_API_VERSION = 2
CAUSAL4D_BELIEF_PROVIDER_V2_PACKAGE_VERSION = "0.4.0"
CAUSAL4D_BELIEF_PROVIDER_V2_CAPABILITIES = (
    "causal_prefix_endpoint_inference",
    "evidence_weighted_endpoint_model_average",
    "horizon_dependent_predictive_covariance",
    "source_calibrated_horizon_discrepancy",
    "mean_reverting_discrepancy_prediction",
    "immutable_endpoint_posterior",
    "numpy_only_endpoint_inference",
    "per_track_component_evidence",
    "residual_finite_preflight",
)
CAUSAL4D_BELIEF_PROVIDER_V2_ARTIFACT_SCHEMA_VERSIONS = {
    "ModelAveragedEndpointConfig": MODEL_AVERAGED_ENDPOINT_CONTRACT_VERSION,
    "ModelAveragedEndpointPosterior": MODEL_AVERAGED_ENDPOINT_CONTRACT_VERSION,
    "ModelAveragedEndpointPrediction": MODEL_AVERAGED_ENDPOINT_CONTRACT_VERSION,
    "HorizonDiscrepancyCalibration": HORIZON_DISCREPANCY_CALIBRATION_VERSION,
    "HorizonConditionedEndpointPrediction": (
        HORIZON_DISCREPANCY_CALIBRATION_VERSION
    ),
}


def infer_model_averaged_bayesian_anchor_endpoint(
    residual_m: np.ndarray,
    valid: np.ndarray,
    *,
    end_frame: int,
    config: ModelAveragedEndpointConfigV1 | None = None,
) -> ModelAveragedEndpointPosteriorV1:
    """Infer an evidence-weighted robust endpoint from a causal prefix."""

    settings = DEFAULT_MODEL_AVERAGED_ENDPOINT_CONFIG_V1 if config is None else config
    if not isinstance(settings, ModelAveragedEndpointConfigV1):
        raise TypeError("config must be a ModelAveragedEndpointConfigV1")
    return infer_model_averaged_endpoint(
        residual_m,
        valid,
        end_frame=end_frame,
        config=settings,
    )


def causal4d_belief_provider_v2_manifest(
    *,
    provider_revision: str | None = None,
) -> dict[str, object]:
    """Return the additive model-averaged capability descriptor."""

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
            fallback=CAUSAL4D_BELIEF_PROVIDER_V2_PACKAGE_VERSION,
        ),
        "provider_revision": revision,
        "schema_version": CAUSAL4D_BELIEF_PROVIDER_V2_API_VERSION,
        "capabilities": list(CAUSAL4D_BELIEF_PROVIDER_V2_CAPABILITIES),
        "artifact_schema_versions": dict(
            CAUSAL4D_BELIEF_PROVIDER_V2_ARTIFACT_SCHEMA_VERSIONS
        ),
        "metadata": {
            "provider_api": "bayesian_phystwin.causal4d_belief_provider_v2",
            "provider_api_version": CAUSAL4D_BELIEF_PROVIDER_V2_API_VERSION,
            "inference_role": ("model-averaged robust readout-discrepancy endpoint"),
            "compatibility": (
                "additive provider; causal4d_belief_provider_v1 is unchanged"
            ),
            "raw_covariance_claim": (
                "model-based predictive covariance; source-calibrated horizon "
                "dynamics and interval calibration remain separate gates"
            ),
        },
    }


__all__ = [
    "CAUSAL4D_BELIEF_PROVIDER_V2_API_VERSION",
    "CAUSAL4D_BELIEF_PROVIDER_V2_ARTIFACT_SCHEMA_VERSIONS",
    "CAUSAL4D_BELIEF_PROVIDER_V2_CAPABILITIES",
    "CAUSAL4D_BELIEF_PROVIDER_V2_PACKAGE_VERSION",
    "DEFAULT_MODEL_AVERAGED_ENDPOINT_CONFIG_V1",
    "ModelAveragedEndpointConfigV1",
    "ModelAveragedEndpointPosteriorV1",
    "HorizonConditionedEndpointPredictionV1",
    "HorizonDiscrepancyCalibrationV1",
    "ModelAveragedEndpointPredictionV1",
    "causal4d_belief_provider_v2_manifest",
    "infer_model_averaged_bayesian_anchor_endpoint",
    "predict_horizon_conditioned_endpoint",
    "predict_model_averaged_endpoint",
]
