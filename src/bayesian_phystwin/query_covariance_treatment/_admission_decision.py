"""Content-addressed covariance-only treatment decision."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from .._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    plain_json,
)
from .._portable_contracts import content_id, sha256_digest
from ._admission_policy import (
    CovarianceOnlyTreatmentPolicyV1,
    treatment_reasons,
)
from ._common import canonical_string, finite_float

COVARIANCE_ONLY_TREATMENT_DECISION_SCHEMA: Final = (
    "bayesian_phystwin.covariance_only_treatment_decision"
)
COVARIANCE_ONLY_TREATMENT_DECISION_VERSION: Final = 1


@dataclass(frozen=True, slots=True)
class CovarianceOnlyTreatmentDecisionV1:
    """Bind one covariance-only candidate to every admission dependency."""

    baseline_belief_id: str
    candidate_belief_id: str
    common_domain_id: str
    query_id: str
    calibration_partition_id: str
    statistical_unit: str
    hybrid_record_id: str
    candidate_covariance_artifact_id: str
    calibration_application_id: str
    calibration_certificate_id: str
    harm_risk_certificate_id: str
    query_relevance_certificate_id: str
    evaluation_decision_id: str
    candidate_inference_admissible: bool
    calibration_input_identity_verified: bool
    calibration_applied: bool
    harm_risk_certified: bool
    query_evidence_admissible: bool
    query_mode_supported: bool
    evaluation_evidence_admissible: bool
    proper_score_supported: bool
    proper_score_observed_value: float
    proper_score_threshold_value: float | None
    simultaneous_coverage: float
    mean_full_width_ratio: float
    treatment_frozen_before_target_outcomes: bool
    target_outcomes_used_for_treatment_selection: bool
    independent_statistical_units: bool
    policy: CovarianceOnlyTreatmentPolicyV1
    treatment_authorized: bool
    reasons: Sequence[str]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "baseline_belief_id",
            "candidate_belief_id",
            "common_domain_id",
            "query_id",
            "calibration_partition_id",
            "hybrid_record_id",
            "candidate_covariance_artifact_id",
            "calibration_application_id",
            "calibration_certificate_id",
            "harm_risk_certificate_id",
            "query_relevance_certificate_id",
            "evaluation_decision_id",
        ):
            object.__setattr__(
                self,
                name,
                sha256_digest(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "statistical_unit",
            canonical_string(self.statistical_unit, name="statistical_unit"),
        )
        for name in (
            "candidate_inference_admissible",
            "calibration_input_identity_verified",
            "calibration_applied",
            "harm_risk_certified",
            "query_evidence_admissible",
            "query_mode_supported",
            "evaluation_evidence_admissible",
            "proper_score_supported",
            "treatment_frozen_before_target_outcomes",
            "target_outcomes_used_for_treatment_selection",
            "independent_statistical_units",
            "treatment_authorized",
        ):
            object.__setattr__(
                self,
                name,
                genuine_boolean(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "proper_score_observed_value",
            finite_float(
                self.proper_score_observed_value,
                name="proper_score_observed_value",
            ),
        )
        if self.proper_score_threshold_value is not None:
            object.__setattr__(
                self,
                "proper_score_threshold_value",
                finite_float(
                    self.proper_score_threshold_value,
                    name="proper_score_threshold_value",
                ),
            )
        object.__setattr__(
            self,
            "simultaneous_coverage",
            finite_float(
                self.simultaneous_coverage,
                name="simultaneous_coverage",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "mean_full_width_ratio",
            finite_float(
                self.mean_full_width_ratio,
                name="mean_full_width_ratio",
                minimum=0.0,
            ),
        )
        if self.mean_full_width_ratio <= 0.0:
            raise ValueError("mean_full_width_ratio must be positive")
        if not isinstance(self.policy, CovarianceOnlyTreatmentPolicyV1):
            raise TypeError("policy must be CovarianceOnlyTreatmentPolicyV1")
        expected_reasons = treatment_reasons(
            candidate_inference_admissible=self.candidate_inference_admissible,
            calibration_input_identity_verified=(
                self.calibration_input_identity_verified
            ),
            calibration_applied=self.calibration_applied,
            harm_risk_certified=self.harm_risk_certified,
            query_evidence_admissible=self.query_evidence_admissible,
            query_mode_supported=self.query_mode_supported,
            evaluation_evidence_admissible=self.evaluation_evidence_admissible,
            proper_score_supported=self.proper_score_supported,
            simultaneous_coverage=self.simultaneous_coverage,
            mean_full_width_ratio=self.mean_full_width_ratio,
            treatment_frozen_before_target_outcomes=(
                self.treatment_frozen_before_target_outcomes
            ),
            target_outcomes_used_for_treatment_selection=(
                self.target_outcomes_used_for_treatment_selection
            ),
            independent_statistical_units=self.independent_statistical_units,
            policy=self.policy,
        )
        supplied_reasons = tuple(
            sorted(
                canonical_string(item, name=f"reasons[{index}]")
                for index, item in enumerate(tuple(self.reasons))
            )
        )
        if len(set(supplied_reasons)) != len(supplied_reasons):
            raise ValueError("reasons must not contain duplicates")
        if supplied_reasons != tuple(sorted(expected_reasons)):
            raise ValueError("reasons do not match covariance treatment gates")
        expected_authorized = expected_reasons == ("covariance-treatment-authorized",)
        if self.treatment_authorized != expected_authorized:
            raise ValueError("treatment_authorized does not match admission gates")
        object.__setattr__(self, "reasons", tuple(sorted(expected_reasons)))
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="covariance-only treatment metadata",
            ),
        )
        expected_id = content_id(self.descriptor())
        if self.artifact_id is not None:
            supplied_id = sha256_digest(self.artifact_id, name="artifact_id")
            if supplied_id != expected_id:
                raise ValueError("artifact_id does not match treatment decision")
        object.__setattr__(self, "artifact_id", expected_id)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": COVARIANCE_ONLY_TREATMENT_DECISION_SCHEMA,
            "schema_version": COVARIANCE_ONLY_TREATMENT_DECISION_VERSION,
            "baseline_belief_id": self.baseline_belief_id,
            "candidate_belief_id": self.candidate_belief_id,
            "common_domain_id": self.common_domain_id,
            "query_id": self.query_id,
            "calibration_partition_id": self.calibration_partition_id,
            "statistical_unit": self.statistical_unit,
            "hybrid_record_id": self.hybrid_record_id,
            "candidate_covariance_artifact_id": (
                self.candidate_covariance_artifact_id
            ),
            "calibration_application_id": self.calibration_application_id,
            "calibration_certificate_id": self.calibration_certificate_id,
            "harm_risk_certificate_id": self.harm_risk_certificate_id,
            "query_relevance_certificate_id": self.query_relevance_certificate_id,
            "evaluation_decision_id": self.evaluation_decision_id,
            "candidate_inference_admissible": (
                self.candidate_inference_admissible
            ),
            "calibration_input_identity_verified": (
                self.calibration_input_identity_verified
            ),
            "calibration_applied": self.calibration_applied,
            "harm_risk_certified": self.harm_risk_certified,
            "query_evidence_admissible": self.query_evidence_admissible,
            "query_mode_supported": self.query_mode_supported,
            "evaluation_evidence_admissible": (
                self.evaluation_evidence_admissible
            ),
            "proper_score_supported": self.proper_score_supported,
            "proper_score_observed_value": self.proper_score_observed_value,
            "proper_score_threshold_value": self.proper_score_threshold_value,
            "simultaneous_coverage": self.simultaneous_coverage,
            "mean_full_width_ratio": self.mean_full_width_ratio,
            "treatment_frozen_before_target_outcomes": (
                self.treatment_frozen_before_target_outcomes
            ),
            "target_outcomes_used_for_treatment_selection": (
                self.target_outcomes_used_for_treatment_selection
            ),
            "independent_statistical_units": self.independent_statistical_units,
            "policy": self.policy.descriptor(),
            "treatment_authorized": self.treatment_authorized,
            "reasons": list(self.reasons),
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}
