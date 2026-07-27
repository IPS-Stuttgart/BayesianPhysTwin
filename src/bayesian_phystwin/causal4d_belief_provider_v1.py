"""NumPy-only fixed Bayesian endpoint surface for Causal4D consumers."""

from __future__ import annotations

import os

import numpy as np

from .contracts.fixed_anchor import (
    DEFAULT_FIXED_BAYESIAN_ANCHOR_CONFIG_V1,
    FIXED_BAYESIAN_ANCHOR_CONTRACT_VERSION,
    FixedBayesianAnchorConfigV1,
    RobustEndpointPosteriorV1,
)
from .contracts.provider import (
    installed_distribution_revision,
    installed_distribution_version,
)
from .phystwin_bayesian_anchor import robust_random_walk_endpoint

CAUSAL4D_BELIEF_PROVIDER_API_VERSION = 1
CAUSAL4D_BELIEF_PROVIDER_PACKAGE_VERSION = "0.4.0"
CAUSAL4D_BELIEF_PROVIDER_CAPABILITIES = (
    "causal_prefix_endpoint_inference",
    "fixed_bayesian_anchor_endpoint",
    "immutable_endpoint_posterior",
    "numpy_only_endpoint_inference",
    "residual_finite_preflight",
)
CAUSAL4D_BELIEF_ARTIFACT_SCHEMA_VERSIONS = {
    "FixedBayesianAnchorConfig": FIXED_BAYESIAN_ANCHOR_CONTRACT_VERSION,
    "RobustEndpointPosterior": FIXED_BAYESIAN_ANCHOR_CONTRACT_VERSION,
}


def _validated_inputs(
    residual_m: np.ndarray,
    valid: np.ndarray,
    *,
    end_frame: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    residual = np.asarray(residual_m, dtype=np.float64)
    validity = np.asarray(valid, dtype=bool)
    if residual.ndim != 3 or residual.shape[2:] != (3,) or residual.shape[1] < 1:
        raise ValueError("residual_m must have shape (T, N>=1, 3)")
    if validity.shape != residual.shape[:2]:
        raise ValueError("valid must match the residual frame and track dimensions")
    if not np.all(np.isfinite(residual)):
        raise ValueError("residual_m must contain only finite values")
    frame_stop = int(end_frame)
    if isinstance(end_frame, bool) or frame_stop != end_frame:
        raise ValueError("end_frame must be an integer")
    if not 0 < frame_stop <= len(residual):
        raise ValueError("end_frame must lie inside the residual sequence")
    return residual, validity, frame_stop


def infer_fixed_bayesian_anchor_endpoint(
    residual_m: np.ndarray,
    valid: np.ndarray,
    *,
    end_frame: int,
    config: FixedBayesianAnchorConfigV1 | None = None,
) -> RobustEndpointPosteriorV1:
    """Infer one robust endpoint using only frames before ``end_frame``."""

    residual, validity, frame_stop = _validated_inputs(
        residual_m,
        valid,
        end_frame=end_frame,
    )
    settings = DEFAULT_FIXED_BAYESIAN_ANCHOR_CONFIG_V1 if config is None else config
    if not isinstance(settings, FixedBayesianAnchorConfigV1):
        raise TypeError("config must be a FixedBayesianAnchorConfigV1")
    posterior = robust_random_walk_endpoint(
        residual,
        validity,
        end_frame=frame_stop,
        process_variance=settings.process_std_m**2,
        observation_variance=settings.observation_std_m**2,
        initial_variance=settings.initial_std_m**2,
        inlier_prior=settings.inlier_prior,
        outlier_variance_multiplier=settings.outlier_variance_multiplier,
    )
    return RobustEndpointPosteriorV1(
        mean_m=posterior.mean,
        variance_m2=posterior.variance,
        final_nominal_probability=posterior.final_inlier_probability,
        update_count=posterior.update_count,
    )


def causal4d_belief_provider_manifest(
    *,
    provider_revision: str | None = None,
) -> dict[str, object]:
    """Return the fixed-anchor capability descriptor consumed by Causal4D."""

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
            fallback=CAUSAL4D_BELIEF_PROVIDER_PACKAGE_VERSION,
        ),
        "provider_revision": revision,
        "schema_version": CAUSAL4D_BELIEF_PROVIDER_API_VERSION,
        "capabilities": list(CAUSAL4D_BELIEF_PROVIDER_CAPABILITIES),
        "artifact_schema_versions": dict(CAUSAL4D_BELIEF_ARTIFACT_SCHEMA_VERSIONS),
        "metadata": {
            "provider_api": "bayesian_phystwin.causal4d_belief_provider_v1",
            "provider_api_version": CAUSAL4D_BELIEF_PROVIDER_API_VERSION,
            "inference_role": "fixed robust readout-discrepancy endpoint",
        },
    }


__all__ = [
    "CAUSAL4D_BELIEF_ARTIFACT_SCHEMA_VERSIONS",
    "CAUSAL4D_BELIEF_PROVIDER_API_VERSION",
    "CAUSAL4D_BELIEF_PROVIDER_CAPABILITIES",
    "CAUSAL4D_BELIEF_PROVIDER_PACKAGE_VERSION",
    "DEFAULT_FIXED_BAYESIAN_ANCHOR_CONFIG_V1",
    "FixedBayesianAnchorConfigV1",
    "RobustEndpointPosteriorV1",
    "causal4d_belief_provider_manifest",
    "infer_fixed_bayesian_anchor_endpoint",
]
