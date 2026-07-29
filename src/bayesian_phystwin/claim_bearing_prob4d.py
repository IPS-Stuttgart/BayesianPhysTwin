"""Strict Prob4D admission before a Bayesian-PhysTwin innovation is formed."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np

from .observation_belief import ObservationBeliefV1
from .observation_belief_gauge_adapter import (
    ObservationBeliefGaugeAdapterResult,
    build_gauge_aware_batch_from_observation_belief,
)
from .physical_linearization import (
    PhysicalLinearizationV1,
    build_gauge_aware_batch_from_artifacts,
)
from .prob4d_causal_lineage import (
    PROB4D_CAUSAL_STREAM_CONTRACT_VERSION,
    validate_claim_bearing_prob4d_observation_belief,
)


def _required_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _claim_bearing_metadata(
    adapted: ObservationBeliefGaugeAdapterResult,
    validation: Mapping[str, object],
) -> ObservationBeliefGaugeAdapterResult:
    provider = _required_mapping(
        validation.get("provider_attestation"),
        name="validated Prob4D provider attestation",
    )
    calibration = _required_mapping(
        validation.get("claim_bearing_covariance_calibration"),
        name="validated Prob4D covariance calibration",
    )
    metadata = dict(adapted.batch.metadata or {})
    metadata.update(
        {
            "prob4d_claim_bearing_provider_v2_validated": True,
            "prob4d_claim_bearing_stream_contract_version": (
                PROB4D_CAUSAL_STREAM_CONTRACT_VERSION
            ),
            "prob4d_claim_bearing_provider_manifest_id": provider.get(
                "provider_manifest_id"
            ),
            "prob4d_claim_bearing_calibration_artifact_ids": calibration.get(
                "calibration_artifact_ids"
            ),
            "prob4d_claim_bearing_runtime_revision_source": provider.get(
                "runtime_revision_source"
            ),
            "prob4d_claim_bearing_runtime_revision_independently_verified": provider.get(
                "runtime_revision_independently_verified"
            ),
            "prob4d_claim_bearing_validation": dict(validation),
        }
    )
    return replace(adapted, batch=replace(adapted.batch, metadata=metadata))


def build_claim_bearing_gauge_aware_batch_from_observation_belief(
    belief: ObservationBeliefV1,
    *,
    physical_prediction_xyz_m: np.ndarray,
    state_jacobian: np.ndarray,
    query_state_jacobian: np.ndarray,
    physical_response_scale_m: float,
    shared_bias_jacobian: np.ndarray | None = None,
    view_bias_jacobian: np.ndarray | None = None,
    state_prior_covariance_m2: np.ndarray | None = None,
    anchor_innovation_m: np.ndarray | None = None,
    anchor_covariance_m2: np.ndarray | None = None,
    anchor_state_jacobian: np.ndarray | None = None,
    anchor_correlation_group_ids: tuple[str, ...] | None = None,
    anchor_prior_reliability: np.ndarray | None = None,
    anchor_prior_nominal_probability: np.ndarray | None = None,
    anchor_composite_weight: np.ndarray | None = None,
    anchor_bias_jacobian: np.ndarray | None = None,
    anchor_bias_prior_covariance: np.ndarray | None = None,
) -> ObservationBeliefGaugeAdapterResult:
    """Validate provider-v2 evidence before forming the observation innovation.

    The compatibility adapter remains available for frozen provider-v1 and labelled
    exploratory artifacts. This entry point is the admission boundary for new
    prospective Prob4D-to-Bayesian-PhysTwin evidence.
    """

    validation = validate_claim_bearing_prob4d_observation_belief(belief)
    adapted = build_gauge_aware_batch_from_observation_belief(
        belief,
        physical_prediction_xyz_m=physical_prediction_xyz_m,
        state_jacobian=state_jacobian,
        query_state_jacobian=query_state_jacobian,
        physical_response_scale_m=physical_response_scale_m,
        shared_bias_jacobian=shared_bias_jacobian,
        view_bias_jacobian=view_bias_jacobian,
        state_prior_covariance_m2=state_prior_covariance_m2,
        anchor_innovation_m=anchor_innovation_m,
        anchor_covariance_m2=anchor_covariance_m2,
        anchor_state_jacobian=anchor_state_jacobian,
        anchor_correlation_group_ids=anchor_correlation_group_ids,
        anchor_prior_reliability=anchor_prior_reliability,
        anchor_prior_nominal_probability=anchor_prior_nominal_probability,
        anchor_composite_weight=anchor_composite_weight,
        anchor_bias_jacobian=anchor_bias_jacobian,
        anchor_bias_prior_covariance=anchor_bias_prior_covariance,
    )
    return _claim_bearing_metadata(adapted, validation)


def build_claim_bearing_gauge_aware_batch_from_artifacts(
    observation_belief: ObservationBeliefV1,
    linearization: PhysicalLinearizationV1,
    *,
    physical_prediction_xyz_m: np.ndarray,
    shared_bias_jacobian: np.ndarray | None = None,
    view_bias_jacobian: np.ndarray | None = None,
    state_prior_covariance_m2: np.ndarray | None = None,
    anchor_innovation_m: np.ndarray | None = None,
    anchor_covariance_m2: np.ndarray | None = None,
    anchor_state_jacobian: np.ndarray | None = None,
    **anchor_dependence: Any,
) -> ObservationBeliefGaugeAdapterResult:
    """Require claim-bearing Prob4D semantics before using a bound linearization."""

    validation = validate_claim_bearing_prob4d_observation_belief(
        observation_belief
    )
    adapted = build_gauge_aware_batch_from_artifacts(
        observation_belief,
        linearization,
        physical_prediction_xyz_m=physical_prediction_xyz_m,
        shared_bias_jacobian=shared_bias_jacobian,
        view_bias_jacobian=view_bias_jacobian,
        state_prior_covariance_m2=state_prior_covariance_m2,
        anchor_innovation_m=anchor_innovation_m,
        anchor_covariance_m2=anchor_covariance_m2,
        anchor_state_jacobian=anchor_state_jacobian,
        **anchor_dependence,
    )
    return _claim_bearing_metadata(adapted, validation)


__all__ = [
    "build_claim_bearing_gauge_aware_batch_from_artifacts",
    "build_claim_bearing_gauge_aware_batch_from_observation_belief",
]
