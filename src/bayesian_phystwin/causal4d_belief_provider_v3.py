"""Causal4D provider boundary for source-tempered endpoint beliefs."""

from __future__ import annotations

import os

import numpy as np

from .causal4d_belief_provider_v2 import (
    CAUSAL4D_BELIEF_PROVIDER_V2_PACKAGE_VERSION,
)
from .contracts.provider import (
    installed_distribution_revision,
    installed_distribution_version,
)
from .endpoint_model_average import ModelAveragedEndpointConfigV1
from .tempered_endpoint_belief import (
    DEFAULT_TEMPERED_ENDPOINT_CONFIG_V2,
    TEMPERED_ENDPOINT_CONTRACT_VERSION,
    EndpointGroupedCalibrationV1,
    EndpointRegretGuardV1,
    SourceComponentPriorV1,
    TemperedEndpointConfigV2,
    TemperedEndpointPosteriorV2,
    TemperedEndpointPredictionV2,
    infer_tempered_endpoint,
    predict_tempered_endpoint,
)

CAUSAL4D_BELIEF_PROVIDER_V3_API_VERSION = 3
CAUSAL4D_BELIEF_PROVIDER_V3_PACKAGE_VERSION = (
    CAUSAL4D_BELIEF_PROVIDER_V2_PACKAGE_VERSION
)
CAUSAL4D_BELIEF_PROVIDER_V3_CAPABILITIES = (
    "causal_prefix_endpoint_inference",
    "effective_evidence_tempering",
    "source_group_component_prior",
    "candidate_specific_regret_guard",
    "grouped_conformal_radius_calibration",
    "exact_fallback_guard",
    "horizon_dependent_predictive_covariance",
    "immutable_endpoint_posterior",
    "numpy_only_endpoint_inference",
)
CAUSAL4D_BELIEF_PROVIDER_V3_ARTIFACT_SCHEMA_VERSIONS = {
    "TemperedEndpointConfig": TEMPERED_ENDPOINT_CONTRACT_VERSION,
    "TemperedEndpointPosterior": TEMPERED_ENDPOINT_CONTRACT_VERSION,
    "TemperedEndpointPrediction": TEMPERED_ENDPOINT_CONTRACT_VERSION,
    "SourceComponentPrior": 1,
    "EndpointRegretGuard": 1,
    "EndpointGroupedCalibration": 1,
}


def infer_tempered_bayesian_anchor_endpoint(
    residual_m: np.ndarray,
    valid: np.ndarray,
    *,
    end_frame: int,
    base_config: ModelAveragedEndpointConfigV1 | None = None,
    config: TemperedEndpointConfigV2 | None = None,
) -> TemperedEndpointPosteriorV2:
    """Infer a source-tempered robust endpoint from a causal prefix."""

    settings = (
        DEFAULT_TEMPERED_ENDPOINT_CONFIG_V2 if config is None else config
    )
    if not isinstance(settings, TemperedEndpointConfigV2):
        raise TypeError("config must be a TemperedEndpointConfigV2")
    return infer_tempered_endpoint(
        residual_m,
        valid,
        end_frame=end_frame,
        base_config=base_config,
        config=settings,
    )


def causal4d_belief_provider_v3_manifest(
    *,
    provider_revision: str | None = None,
) -> dict[str, object]:
    """Return the additive source-tempered endpoint capability descriptor."""

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
            "provider_api": (
                "bayesian_phystwin.causal4d_belief_provider_v3"
            ),
            "provider_api_version": CAUSAL4D_BELIEF_PROVIDER_V3_API_VERSION,
            "inference_role": (
                "source-tempered robust readout-discrepancy endpoint"
            ),
            "compatibility": (
                "additive provider; provider v1 and provider v2 are unchanged"
            ),
            "selection_boundary": (
                "temperature, component prior, covariance inflation, and guard "
                "must be frozen from source groups before target outcomes"
            ),
            "calibration_boundary": (
                "raw covariance remains model-based; deployment coverage "
                "requires a separate independent-group calibration artifact"
            ),
        },
    }


__all__ = [
    "CAUSAL4D_BELIEF_PROVIDER_V3_API_VERSION",
    "CAUSAL4D_BELIEF_PROVIDER_V3_ARTIFACT_SCHEMA_VERSIONS",
    "CAUSAL4D_BELIEF_PROVIDER_V3_CAPABILITIES",
    "CAUSAL4D_BELIEF_PROVIDER_V3_PACKAGE_VERSION",
    "DEFAULT_TEMPERED_ENDPOINT_CONFIG_V2",
    "EndpointGroupedCalibrationV1",
    "EndpointRegretGuardV1",
    "SourceComponentPriorV1",
    "TemperedEndpointConfigV2",
    "TemperedEndpointPosteriorV2",
    "TemperedEndpointPredictionV2",
    "causal4d_belief_provider_v3_manifest",
    "infer_tempered_bayesian_anchor_endpoint",
    "predict_tempered_endpoint",
]
