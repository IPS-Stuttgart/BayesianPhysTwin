"""Frozen policy and deterministic reasons for covariance treatment admission."""

from __future__ import annotations

from dataclasses import dataclass

from .._canonical_contracts import genuine_boolean, genuine_integer
from ._common import COVARIANCE_MODES, canonical_string, finite_float


@dataclass(frozen=True, slots=True)
class CovarianceOnlyTreatmentPolicyV1:
    """Frozen proper-score, coverage, width, and covariance-mode policy."""

    covariance_mode: str
    evaluation_claim_id: str
    evaluation_protocol_id: str
    proper_score_metric_name: str
    proper_score_comparison: str
    proper_score_rule: str
    maximum_proper_score_value: float
    minimum_simultaneous_coverage: float
    maximum_mean_full_width_ratio: float
    minimum_evidence_level: int = 2
    require_claim_authorized_decision: bool = False
    numerical_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        mode = canonical_string(self.covariance_mode, name="covariance_mode")
        if mode not in COVARIANCE_MODES:
            raise ValueError(
                f"covariance_mode must be one of {sorted(COVARIANCE_MODES)}"
            )
        object.__setattr__(self, "covariance_mode", mode)
        for name in (
            "evaluation_claim_id",
            "evaluation_protocol_id",
            "proper_score_metric_name",
            "proper_score_comparison",
            "proper_score_rule",
        ):
            object.__setattr__(
                self,
                name,
                canonical_string(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "maximum_proper_score_value",
            finite_float(
                self.maximum_proper_score_value,
                name="maximum_proper_score_value",
            ),
        )
        object.__setattr__(
            self,
            "minimum_simultaneous_coverage",
            finite_float(
                self.minimum_simultaneous_coverage,
                name="minimum_simultaneous_coverage",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "maximum_mean_full_width_ratio",
            finite_float(
                self.maximum_mean_full_width_ratio,
                name="maximum_mean_full_width_ratio",
                minimum=0.0,
            ),
        )
        if self.maximum_mean_full_width_ratio <= 0.0:
            raise ValueError("maximum_mean_full_width_ratio must be positive")
        level = genuine_integer(
            self.minimum_evidence_level,
            name="minimum_evidence_level",
            minimum=1,
        )
        if level > 3:
            raise ValueError("minimum_evidence_level must be at most three")
        object.__setattr__(self, "minimum_evidence_level", level)
        object.__setattr__(
            self,
            "require_claim_authorized_decision",
            genuine_boolean(
                self.require_claim_authorized_decision,
                name="require_claim_authorized_decision",
            ),
        )
        object.__setattr__(
            self,
            "numerical_tolerance",
            finite_float(
                self.numerical_tolerance,
                name="numerical_tolerance",
                minimum=0.0,
            ),
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "covariance_mode": self.covariance_mode,
            "evaluation_claim_id": self.evaluation_claim_id,
            "evaluation_protocol_id": self.evaluation_protocol_id,
            "proper_score_metric_name": self.proper_score_metric_name,
            "proper_score_comparison": self.proper_score_comparison,
            "proper_score_rule": self.proper_score_rule,
            "maximum_proper_score_value": self.maximum_proper_score_value,
            "minimum_simultaneous_coverage": self.minimum_simultaneous_coverage,
            "maximum_mean_full_width_ratio": self.maximum_mean_full_width_ratio,
            "minimum_evidence_level": self.minimum_evidence_level,
            "require_claim_authorized_decision": (
                self.require_claim_authorized_decision
            ),
            "numerical_tolerance": self.numerical_tolerance,
        }


def treatment_reasons(
    *,
    candidate_inference_admissible: bool,
    calibration_input_identity_verified: bool,
    calibration_applied: bool,
    harm_risk_certified: bool,
    query_evidence_admissible: bool,
    query_mode_supported: bool,
    evaluation_evidence_admissible: bool,
    proper_score_supported: bool,
    simultaneous_coverage: float,
    mean_full_width_ratio: float,
    treatment_frozen_before_target_outcomes: bool,
    target_outcomes_used_for_treatment_selection: bool,
    independent_statistical_units: bool,
    policy: CovarianceOnlyTreatmentPolicyV1,
) -> tuple[str, ...]:
    gates = (
        (candidate_inference_admissible, "candidate-inference-rejected"),
        (
            calibration_input_identity_verified,
            "hybrid-calibration-input-identity-mismatch",
        ),
        (calibration_applied, "covariance-calibration-not-applied"),
        (harm_risk_certified, "harm-risk-certificate-rejected"),
        (
            query_evidence_admissible,
            "query-relevance-evidence-not-admissible",
        ),
        (query_mode_supported, "query-covariance-mode-mismatch"),
        (
            evaluation_evidence_admissible,
            "evaluation-evidence-decision-rejected",
        ),
        (proper_score_supported, "proper-score-gate-rejected"),
        (
            treatment_frozen_before_target_outcomes,
            "treatment-not-frozen-before-target",
        ),
        (
            not target_outcomes_used_for_treatment_selection,
            "target-outcomes-used-for-treatment-selection",
        ),
        (independent_statistical_units, "treatment-groups-not-independent"),
    )
    reasons = [reason for passed, reason in gates if not passed]
    if simultaneous_coverage + policy.numerical_tolerance < (
        policy.minimum_simultaneous_coverage
    ):
        reasons.append("simultaneous-coverage-below-threshold")
    if mean_full_width_ratio > (
        policy.maximum_mean_full_width_ratio + policy.numerical_tolerance
    ):
        reasons.append("mean-full-width-ratio-exceeds-limit")
    return tuple(reasons or ["covariance-treatment-authorized"])
