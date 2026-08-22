"""Bind provider-to-physical mapping evidence before Prob4D inference.

The composition is deliberately additive. It leaves the historical claim-bearing
Prob4D candidate unchanged, rejects inadmissible or identity-mismatched mappings
before the solver is called, and publishes a content-addressed wrapper that binds
the accepted mapping audit to the resulting candidate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from ._canonical_contracts import frozen_finite_json_mapping, plain_json
from ._portable_contracts import content_id, sha256_digest
from .observation_belief import ObservationBeliefV1
from .physical_linearization import PhysicalLinearizationV1
from .posterior_covariance_semantics import PosteriorCovarianceSemanticsV1
from .prior_aware_gauge_belief import PriorAwareGaugeConfigV1
from .prospective_prob4d_update import (
    ClaimBearingProb4DCandidateV1,
    infer_claim_bearing_prob4d_candidate_from_artifacts,
)
from .provider_physical_mapping_audit_v1 import ProviderPhysicalMappingAuditV1

MAPPED_PROB4D_CANDIDATE_SCHEMA: Final = (
    "bayesian_phystwin.mapped_claim_bearing_prob4d_candidate"
)
MAPPED_PROB4D_CANDIDATE_VERSION: Final = 1
MAPPED_PROB4D_CANDIDATE_IDENTITY_VERSION: Final = 1
MAPPED_PROB4D_CANDIDATE_CLAIM_BOUNDARY: Final = (
    "Software lineage and fail-closed mapping composition only. A mapped "
    "candidate does not establish provider competence, covariance calibration, "
    "fresh-object physical benefit, Causal4D intervention benefit, deployment "
    "safety, or state of the art."
)


def _validated_audit_id(audit: ProviderPhysicalMappingAuditV1) -> str:
    return sha256_digest(audit.audit_id, name="mapping_audit.audit_id")


def _validated_observation_artifact_id(
    observation_belief: ObservationBeliefV1,
) -> str:
    return sha256_digest(
        observation_belief.artifact_id,
        name="observation_belief.artifact_id",
    )


def _require_mapping_preflight(
    *,
    mapping_audit: ProviderPhysicalMappingAuditV1,
    observation_artifact_id: str,
    physical_query_id: str,
) -> None:
    """Reject a mapping before any physical candidate inference can run."""

    if not isinstance(mapping_audit, ProviderPhysicalMappingAuditV1):
        raise TypeError("mapping_audit must be ProviderPhysicalMappingAuditV1")
    expected_observation_id = sha256_digest(
        observation_artifact_id,
        name="observation_artifact_id",
    )
    expected_query_id = sha256_digest(
        physical_query_id,
        name="physical_query_id",
    )
    _validated_audit_id(mapping_audit)
    if mapping_audit.provider_artifact_id != expected_observation_id:
        raise ValueError(
            "mapping audit provider_artifact_id does not match the observation artifact"
        )
    if mapping_audit.physical_query_id != expected_query_id:
        raise ValueError(
            "mapping audit physical_query_id does not match the requested query"
        )
    if not mapping_audit.mapping_admissible:
        reasons = ", ".join(mapping_audit.rejection_reasons)
        raise ValueError(f"provider-to-physical mapping is inadmissible: {reasons}")


@dataclass(frozen=True, slots=True)
class MappedClaimBearingProb4DCandidateV1:
    """One claim-bearing candidate bound to one passing mapping audit."""

    candidate: ClaimBearingProb4DCandidateV1
    mapping_audit: ProviderPhysicalMappingAuditV1
    physical_query_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, ClaimBearingProb4DCandidateV1):
            raise TypeError("candidate must be ClaimBearingProb4DCandidateV1")
        if not isinstance(self.mapping_audit, ProviderPhysicalMappingAuditV1):
            raise TypeError("mapping_audit must be ProviderPhysicalMappingAuditV1")
        query_id = sha256_digest(self.physical_query_id, name="physical_query_id")
        _require_mapping_preflight(
            mapping_audit=self.mapping_audit,
            observation_artifact_id=(self.candidate.update_v1.observation_artifact_id),
            physical_query_id=query_id,
        )
        object.__setattr__(self, "physical_query_id", query_id)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="mapped Prob4D candidate metadata",
            ),
        )
        expected = content_id(self.descriptor())
        if self.artifact_id is not None:
            supplied = sha256_digest(self.artifact_id, name="artifact_id")
            if supplied != expected:
                raise ValueError(
                    "artifact_id does not match mapped Prob4D candidate contents"
                )
        object.__setattr__(self, "artifact_id", expected)

    @property
    def inference_admissible(self) -> bool:
        return self.candidate.inference_admissible

    @property
    def reason(self) -> str:
        return self.candidate.reason

    @property
    def covariance_semantics(self) -> PosteriorCovarianceSemanticsV1:
        return self.candidate.covariance_semantics

    @property
    def mapping_audit_id(self) -> str:
        return _validated_audit_id(self.mapping_audit)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": MAPPED_PROB4D_CANDIDATE_SCHEMA,
            "schema_version": MAPPED_PROB4D_CANDIDATE_VERSION,
            "identity_version": MAPPED_PROB4D_CANDIDATE_IDENTITY_VERSION,
            "candidate_id": self.candidate.candidate_id,
            "v1_update_id": self.candidate.v1_update_id,
            "mapping_audit_id": self.mapping_audit_id,
            "mapping_protocol_id": self.mapping_audit.mapping_protocol_id,
            "provider_artifact_id": self.mapping_audit.provider_artifact_id,
            "physical_query_id": self.physical_query_id,
            "covariance_semantics_id": self.covariance_semantics.artifact_id,
            "inference_admissible": self.inference_admissible,
            "reason": self.reason,
            "claim_boundary": MAPPED_PROB4D_CANDIDATE_CLAIM_BOUNDARY,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {
            **self.descriptor(),
            "candidate": self.candidate.to_record(),
            "mapping_audit": self.mapping_audit.to_dict(),
            "artifact_id": self.artifact_id,
        }


def bind_provider_mapping_to_prob4d_candidate(
    candidate: ClaimBearingProb4DCandidateV1,
    mapping_audit: ProviderPhysicalMappingAuditV1,
    *,
    physical_query_id: str,
    metadata: Mapping[str, Any] | None = None,
) -> MappedClaimBearingProb4DCandidateV1:
    """Bind one passing audit to an already inferred claim-bearing candidate."""

    return MappedClaimBearingProb4DCandidateV1(
        candidate=candidate,
        mapping_audit=mapping_audit,
        physical_query_id=physical_query_id,
        metadata={} if metadata is None else metadata,
    )


def infer_mapped_claim_bearing_prob4d_candidate_from_artifacts(
    observation_belief: ObservationBeliefV1,
    linearization: PhysicalLinearizationV1,
    *,
    mapping_audit: ProviderPhysicalMappingAuditV1,
    physical_query_id: str,
    physical_prediction_xyz_m: np.ndarray,
    shared_bias_jacobian: np.ndarray | None = None,
    view_bias_jacobian: np.ndarray | None = None,
    state_prior_covariance_m2: np.ndarray | None = None,
    anchor_innovation_m: np.ndarray | None = None,
    anchor_covariance_m2: np.ndarray | None = None,
    anchor_state_jacobian: np.ndarray | None = None,
    config: PriorAwareGaugeConfigV1 | None = None,
    covariance_semantics: PosteriorCovarianceSemanticsV1 | None = None,
    metadata: Mapping[str, Any] | None = None,
    **anchor_dependence: Any,
) -> MappedClaimBearingProb4DCandidateV1:
    """Require a passing bound mapping before invoking Prob4D inference."""

    observation_artifact_id = _validated_observation_artifact_id(observation_belief)
    query_id = sha256_digest(physical_query_id, name="physical_query_id")
    _require_mapping_preflight(
        mapping_audit=mapping_audit,
        observation_artifact_id=observation_artifact_id,
        physical_query_id=query_id,
    )
    candidate = infer_claim_bearing_prob4d_candidate_from_artifacts(
        observation_belief,
        linearization,
        physical_prediction_xyz_m=physical_prediction_xyz_m,
        shared_bias_jacobian=shared_bias_jacobian,
        view_bias_jacobian=view_bias_jacobian,
        state_prior_covariance_m2=state_prior_covariance_m2,
        anchor_innovation_m=anchor_innovation_m,
        anchor_covariance_m2=anchor_covariance_m2,
        anchor_state_jacobian=anchor_state_jacobian,
        config=config,
        covariance_semantics=covariance_semantics,
        **anchor_dependence,
    )
    return bind_provider_mapping_to_prob4d_candidate(
        candidate,
        mapping_audit,
        physical_query_id=query_id,
        metadata={} if metadata is None else metadata,
    )


__all__ = [
    "MAPPED_PROB4D_CANDIDATE_CLAIM_BOUNDARY",
    "MAPPED_PROB4D_CANDIDATE_IDENTITY_VERSION",
    "MAPPED_PROB4D_CANDIDATE_SCHEMA",
    "MAPPED_PROB4D_CANDIDATE_VERSION",
    "MappedClaimBearingProb4DCandidateV1",
    "bind_provider_mapping_to_prob4d_candidate",
    "infer_mapped_claim_bearing_prob4d_candidate_from_artifacts",
]
