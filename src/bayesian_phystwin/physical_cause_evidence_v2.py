"""Claim-facing evidence certificates for physical-cause attribution.

The v1 physical-cause selector is an operational router. This module adds a
non-breaking evidence layer for stronger scientific attribution. It binds
simultaneous baseline-relative regret, paired cause comparisons, independent
physical source groups, nonlinear closure, and (for physical causes) held-out
transport evidence. It never changes a selected complete belief.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    literal_lower_hex,
    plain_json,
)
from ._portable_contracts import content_id
from .physical_cause_selection_v1 import PhysicalCause

PHYSICAL_CAUSE_EVIDENCE_VERSION = 2
PHYSICAL_CAUSE_REGRET_CERTIFICATE_SCHEMA = (
    "bayesian_phystwin.physical_cause_regret_certificate"
)
PHYSICAL_CAUSE_PAIRWISE_CERTIFICATE_SCHEMA = (
    "bayesian_phystwin.physical_cause_pairwise_certificate"
)
PHYSICAL_CAUSE_ATTRIBUTION_DECISION_SCHEMA = (
    "bayesian_phystwin.physical_cause_attribution_decision"
)
PHYSICAL_CAUSE_ATTRIBUTION_CLAIM_BOUNDARY = (
    "Source-group evidence can certify a registered selective-regret and "
    "cause-separation statement only. Physical state or parameter attribution "
    "also requires separately bound held-out transport evidence. The artifact "
    "does not establish a unique data-generating cause, provider competence, "
    "deployment safety, unseen-object transfer, or downstream Causal4D benefit."
)


def _sha256(value: object, *, name: str) -> str:
    return cast(str, literal_lower_hex(value, name=name, lengths={64}))


def _finite_real(value: object, *, name: str) -> float:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if raw.shape != () or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real number")
    result = float(raw.item())
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _canonical_group_ids(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of SHA-256 identities")
    result = tuple(_sha256(value, name=f"{name} item") for value in values)
    if not result:
        raise ValueError(f"{name} must be nonempty")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique identities")
    return tuple(sorted(result))


def _canonical_strata(values: Mapping[str, object]) -> Mapping[str, float]:
    if not isinstance(values, Mapping):
        raise TypeError("stratum_upper_regrets must be a mapping")
    normalized: dict[str, float] = {}
    for key, value in values.items():
        if type(key) is not str or not key:
            raise ValueError("stratum names must be nonempty literal strings")
        normalized[key] = _finite_real(
            value,
            name=f"stratum_upper_regrets[{key!r}]",
        )
    return cast(
        Mapping[str, float],
        frozen_finite_json_mapping(
            normalized,
            name="stratum upper-regret bounds",
        ),
    )


@dataclass(frozen=True, slots=True)
class PhysicalCauseRegretCertificateV2:
    """Simultaneous source-group evidence for one cause versus the baseline."""

    cause: PhysicalCause
    baseline_belief_id: str
    candidate_belief_id: str
    candidate_construction_id: str
    common_domain_id: str
    registered_query_id: str
    source_evidence_id: str
    proper_score_id: str
    grouping_rule_id: str
    candidate_universe_id: str
    source_group_ids: tuple[str, ...]
    simultaneous_upper_regret: float
    harm_margin: float
    harm_probability_upper: float
    confidence_level: float
    stratum_upper_regrets: Mapping[str, float] = field(default_factory=dict)
    bounds_simultaneous: bool = True
    thresholds_frozen_before_source_scores: bool = True
    candidate_universe_frozen_before_source_scores: bool = True
    target_outcomes_used: bool = False
    source_groups_independent: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.cause, PhysicalCause)
            or self.cause is PhysicalCause.BASELINE
        ):
            raise ValueError("cause must be a nonbaseline PhysicalCause")
        for name in (
            "baseline_belief_id",
            "candidate_belief_id",
            "candidate_construction_id",
            "common_domain_id",
            "registered_query_id",
            "source_evidence_id",
            "proper_score_id",
            "grouping_rule_id",
            "candidate_universe_id",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name=name))
        if self.candidate_belief_id == self.baseline_belief_id:
            raise ValueError("candidate belief must differ from baseline")
        object.__setattr__(
            self,
            "source_group_ids",
            _canonical_group_ids(self.source_group_ids, name="source_group_ids"),
        )
        object.__setattr__(
            self,
            "simultaneous_upper_regret",
            _finite_real(
                self.simultaneous_upper_regret,
                name="simultaneous_upper_regret",
            ),
        )
        harm_margin = _finite_real(self.harm_margin, name="harm_margin")
        if harm_margin < 0.0:
            raise ValueError("harm_margin must be nonnegative")
        object.__setattr__(self, "harm_margin", harm_margin)
        harm_probability = _finite_real(
            self.harm_probability_upper,
            name="harm_probability_upper",
        )
        if not 0.0 <= harm_probability <= 1.0:
            raise ValueError("harm_probability_upper must lie in [0, 1]")
        object.__setattr__(self, "harm_probability_upper", harm_probability)
        confidence = _finite_real(self.confidence_level, name="confidence_level")
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must lie in (0, 1)")
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(
            self,
            "stratum_upper_regrets",
            _canonical_strata(self.stratum_upper_regrets),
        )
        for name, expected in (
            ("bounds_simultaneous", True),
            ("thresholds_frozen_before_source_scores", True),
            ("candidate_universe_frozen_before_source_scores", True),
            ("target_outcomes_used", False),
            ("source_groups_independent", True),
        ):
            value = genuine_boolean(getattr(self, name), name=name)
            if value is not expected:
                raise ValueError(f"{name} must be {str(expected).lower()}")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="regret-certificate metadata",
            ),
        )

    @property
    def source_group_count(self) -> int:
        return len(self.source_group_ids)

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": PHYSICAL_CAUSE_REGRET_CERTIFICATE_SCHEMA,
            "schema_version": PHYSICAL_CAUSE_EVIDENCE_VERSION,
            "cause": self.cause.value,
            "baseline_belief_id": self.baseline_belief_id,
            "candidate_belief_id": self.candidate_belief_id,
            "candidate_construction_id": self.candidate_construction_id,
            "common_domain_id": self.common_domain_id,
            "registered_query_id": self.registered_query_id,
            "source_evidence_id": self.source_evidence_id,
            "proper_score_id": self.proper_score_id,
            "grouping_rule_id": self.grouping_rule_id,
            "candidate_universe_id": self.candidate_universe_id,
            "source_group_ids": list(self.source_group_ids),
            "simultaneous_upper_regret": self.simultaneous_upper_regret,
            "harm_margin": self.harm_margin,
            "harm_probability_upper": self.harm_probability_upper,
            "confidence_level": self.confidence_level,
            "stratum_upper_regrets": plain_json(self.stratum_upper_regrets),
            "bounds_simultaneous": self.bounds_simultaneous,
            "thresholds_frozen_before_source_scores": (
                self.thresholds_frozen_before_source_scores
            ),
            "candidate_universe_frozen_before_source_scores": (
                self.candidate_universe_frozen_before_source_scores
            ),
            "target_outcomes_used": self.target_outcomes_used,
            "source_groups_independent": self.source_groups_independent,
            "metadata": plain_json(self.metadata),
        }

    @property
    def certificate_id(self) -> str:
        return cast(str, content_id(self.descriptor()))


@dataclass(frozen=True, slots=True)
class PhysicalCausePairwiseCertificateV2:
    """Paired interval for left-cause regret minus right-cause regret."""

    left_cause: PhysicalCause
    right_cause: PhysicalCause
    left_candidate_id: str
    right_candidate_id: str
    baseline_belief_id: str
    common_domain_id: str
    registered_query_id: str
    source_evidence_id: str
    proper_score_id: str
    grouping_rule_id: str
    source_group_ids: tuple[str, ...]
    candidate_universe_id: str
    lower_regret_difference: float
    upper_regret_difference: float
    confidence_level: float
    bounds_simultaneous: bool = True
    pairwise_procedure_frozen_before_source_scores: bool = True
    candidate_universe_frozen_before_source_scores: bool = True
    target_outcomes_used: bool = False
    source_groups_independent: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for cause, name in (
            (self.left_cause, "left_cause"),
            (self.right_cause, "right_cause"),
        ):
            if not isinstance(cause, PhysicalCause) or cause is PhysicalCause.BASELINE:
                raise ValueError(f"{name} must be a nonbaseline PhysicalCause")
        if self.left_cause.value >= self.right_cause.value:
            raise ValueError("pairwise causes must use canonical lexical order")
        for name in (
            "left_candidate_id",
            "right_candidate_id",
            "baseline_belief_id",
            "common_domain_id",
            "registered_query_id",
            "source_evidence_id",
            "proper_score_id",
            "grouping_rule_id",
            "candidate_universe_id",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "source_group_ids",
            _canonical_group_ids(self.source_group_ids, name="source_group_ids"),
        )
        lower = _finite_real(
            self.lower_regret_difference,
            name="lower_regret_difference",
        )
        upper = _finite_real(
            self.upper_regret_difference,
            name="upper_regret_difference",
        )
        if lower > upper:
            raise ValueError("pairwise lower bound must not exceed upper bound")
        object.__setattr__(self, "lower_regret_difference", lower)
        object.__setattr__(self, "upper_regret_difference", upper)
        confidence = _finite_real(self.confidence_level, name="confidence_level")
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must lie in (0, 1)")
        object.__setattr__(self, "confidence_level", confidence)
        for name, expected in (
            ("bounds_simultaneous", True),
            ("pairwise_procedure_frozen_before_source_scores", True),
            ("candidate_universe_frozen_before_source_scores", True),
            ("target_outcomes_used", False),
            ("source_groups_independent", True),
        ):
            value = genuine_boolean(getattr(self, name), name=name)
            if value is not expected:
                raise ValueError(f"{name} must be {str(expected).lower()}")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="pairwise-certificate metadata",
            ),
        )

    @property
    def cause_key(self) -> tuple[PhysicalCause, PhysicalCause]:
        return self.left_cause, self.right_cause

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": PHYSICAL_CAUSE_PAIRWISE_CERTIFICATE_SCHEMA,
            "schema_version": PHYSICAL_CAUSE_EVIDENCE_VERSION,
            "left_cause": self.left_cause.value,
            "right_cause": self.right_cause.value,
            "left_candidate_id": self.left_candidate_id,
            "right_candidate_id": self.right_candidate_id,
            "baseline_belief_id": self.baseline_belief_id,
            "common_domain_id": self.common_domain_id,
            "registered_query_id": self.registered_query_id,
            "source_evidence_id": self.source_evidence_id,
            "proper_score_id": self.proper_score_id,
            "grouping_rule_id": self.grouping_rule_id,
            "source_group_ids": list(self.source_group_ids),
            "candidate_universe_id": self.candidate_universe_id,
            "lower_regret_difference": self.lower_regret_difference,
            "upper_regret_difference": self.upper_regret_difference,
            "confidence_level": self.confidence_level,
            "bounds_simultaneous": self.bounds_simultaneous,
            "pairwise_procedure_frozen_before_source_scores": (
                self.pairwise_procedure_frozen_before_source_scores
            ),
            "candidate_universe_frozen_before_source_scores": (
                self.candidate_universe_frozen_before_source_scores
            ),
            "target_outcomes_used": self.target_outcomes_used,
            "source_groups_independent": self.source_groups_independent,
            "metadata": plain_json(self.metadata),
        }

    @property
    def certificate_id(self) -> str:
        return cast(str, content_id(self.descriptor()))


def _candidate_id(certificate: PhysicalCauseRegretCertificateV2) -> str:
    return cast(
        str,
        content_id(
            {
                "cause": certificate.cause.value,
                "belief_id": certificate.candidate_belief_id,
                "construction_id": certificate.candidate_construction_id,
            }
        ),
    )


def _pair_key(
    a: PhysicalCause,
    b: PhysicalCause,
) -> tuple[PhysicalCause, PhysicalCause]:
    return (a, b) if a.value < b.value else (b, a)


@dataclass(frozen=True, slots=True)
class PhysicalCauseAttributionDecisionV2:
    """Claim-facing decision without changing the operational complete belief."""

    operational_decision_id: str
    baseline_belief_id: str
    selected_cause: PhysicalCause
    selected_belief_id: str
    certificates: tuple[PhysicalCauseRegretCertificateV2, ...]
    pairwise_certificates: tuple[PhysicalCausePairwiseCertificateV2, ...]
    minimum_improvement: float
    maximum_harm_probability: float
    maximum_stratum_regret: float
    required_strata: tuple[str, ...] = ()
    minimum_source_group_count: int = 1
    pairwise_advantage: float = 0.0
    nonlinear_closure_id: str | None = None
    transport_evidence_id: str | None = None
    decision_thresholds_frozen_before_source_scores: bool = True
    target_outcomes_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operational_decision_id",
            _sha256(self.operational_decision_id, name="operational_decision_id"),
        )
        object.__setattr__(
            self,
            "baseline_belief_id",
            _sha256(self.baseline_belief_id, name="baseline_belief_id"),
        )
        if not isinstance(self.selected_cause, PhysicalCause):
            raise TypeError("selected_cause must be a PhysicalCause")
        object.__setattr__(
            self,
            "selected_belief_id",
            _sha256(self.selected_belief_id, name="selected_belief_id"),
        )
        certs = tuple(sorted(self.certificates, key=lambda item: item.cause.value))
        if len({item.cause for item in certs}) != len(certs):
            raise ValueError("at most one regret certificate per cause is permitted")
        object.__setattr__(self, "certificates", certs)
        pairs = tuple(
            sorted(
                self.pairwise_certificates,
                key=lambda item: (item.left_cause.value, item.right_cause.value),
            )
        )
        if len({item.cause_key for item in pairs}) != len(pairs):
            raise ValueError(
                "at most one pairwise certificate per cause pair is permitted"
            )
        object.__setattr__(self, "pairwise_certificates", pairs)
        for name in (
            "minimum_improvement",
            "maximum_harm_probability",
            "maximum_stratum_regret",
            "pairwise_advantage",
        ):
            value = _finite_real(getattr(self, name), name=name)
            if name != "maximum_stratum_regret" and value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
            if name == "maximum_harm_probability" and value > 1.0:
                raise ValueError("maximum_harm_probability must lie in [0, 1]")
            object.__setattr__(self, name, value)
        if (
            type(self.minimum_source_group_count) is not int
            or self.minimum_source_group_count < 1
        ):
            raise ValueError("minimum_source_group_count must be a positive integer")
        required = tuple(sorted(set(self.required_strata)))
        if any(type(item) is not str or not item for item in required):
            raise ValueError("required_strata must contain nonempty strings")
        object.__setattr__(self, "required_strata", required)
        for name in ("nonlinear_closure_id", "transport_evidence_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _sha256(value, name=name))
        for name, expected in (
            ("decision_thresholds_frozen_before_source_scores", True),
            ("target_outcomes_used", False),
        ):
            value = genuine_boolean(getattr(self, name), name=name)
            if value is not expected:
                raise ValueError(f"{name} must be {str(expected).lower()}")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="attribution-decision metadata",
            ),
        )
        self._validate_common_evidence_domain()
        self._validate_selected_belief_binding()

    def _validate_common_evidence_domain(self) -> None:
        if not self.certificates:
            if self.selected_cause is not PhysicalCause.BASELINE:
                raise ValueError("nonbaseline selection requires regret certificates")
            return
        first = self.certificates[0]
        fields = (
            "baseline_belief_id",
            "common_domain_id",
            "registered_query_id",
            "source_evidence_id",
            "proper_score_id",
            "grouping_rule_id",
            "candidate_universe_id",
            "source_group_ids",
            "confidence_level",
            "harm_margin",
        )
        for certificate in self.certificates[1:]:
            for name in fields:
                if getattr(certificate, name) != getattr(first, name):
                    raise ValueError(f"regret certificate {name} differs across causes")
        if first.baseline_belief_id != self.baseline_belief_id:
            raise ValueError("decision baseline differs from regret certificates")
        candidate_ids = {item.cause: _candidate_id(item) for item in self.certificates}
        pair_fields = (
            "baseline_belief_id",
            "common_domain_id",
            "registered_query_id",
            "source_evidence_id",
            "proper_score_id",
            "grouping_rule_id",
            "candidate_universe_id",
            "source_group_ids",
            "confidence_level",
        )
        for pair in self.pairwise_certificates:
            for name in pair_fields:
                if getattr(pair, name) != getattr(first, name):
                    raise ValueError(
                        f"pairwise certificate {name} differs from regret certificates"
                    )
            if (
                pair.left_cause not in candidate_ids
                or pair.right_cause not in candidate_ids
            ):
                raise ValueError(
                    "pairwise certificate references an unregistered cause"
                )
            if (
                pair.left_candidate_id != candidate_ids[pair.left_cause]
                or pair.right_candidate_id != candidate_ids[pair.right_cause]
            ):
                raise ValueError(
                    "pairwise certificate does not bind registered candidates"
                )

    def _validate_selected_belief_binding(self) -> None:
        if self.selected_cause is PhysicalCause.BASELINE:
            if self.selected_belief_id != self.baseline_belief_id:
                raise ValueError(
                    "baseline selection must bind the exact baseline belief"
                )
            return
        selected = next(
            (item for item in self.certificates if item.cause is self.selected_cause),
            None,
        )
        if selected is None:
            raise ValueError("selected cause has no regret certificate")
        if self.selected_belief_id != selected.candidate_belief_id:
            raise ValueError("selected belief does not match the selected cause")

    @property
    def eligible_causes(self) -> tuple[PhysicalCause, ...]:
        result = []
        required = set(self.required_strata)
        for certificate in self.certificates:
            if certificate.source_group_count < self.minimum_source_group_count:
                continue
            if certificate.simultaneous_upper_regret >= -self.minimum_improvement:
                continue
            if certificate.harm_probability_upper > self.maximum_harm_probability:
                continue
            if set(certificate.stratum_upper_regrets) != required:
                continue
            if any(
                value > self.maximum_stratum_regret
                for value in certificate.stratum_upper_regrets.values()
            ):
                continue
            result.append(certificate.cause)
        return tuple(result)

    @property
    def paired_attribution_resolved(self) -> bool:
        if self.selected_cause is PhysicalCause.BASELINE:
            return False
        eligible = self.eligible_causes
        if self.selected_cause not in eligible:
            return False
        pair_map = {item.cause_key: item for item in self.pairwise_certificates}
        margin = self.pairwise_advantage
        for other in eligible:
            if other is self.selected_cause:
                continue
            pair = pair_map.get(_pair_key(self.selected_cause, other))
            if pair is None:
                return False
            if self.selected_cause is pair.left_cause:
                if pair.upper_regret_difference >= -margin:
                    return False
            elif pair.lower_regret_difference <= margin:
                return False
        return True

    @property
    def selected_physical_attribution_claim_ready(self) -> bool:
        if not self.paired_attribution_resolved:
            return False
        if self.selected_cause not in {
            PhysicalCause.PHYSICAL_STATE,
            PhysicalCause.PHYSICAL_PARAMETER,
        }:
            return True
        return (
            self.nonlinear_closure_id is not None
            and self.transport_evidence_id is not None
        )

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": PHYSICAL_CAUSE_ATTRIBUTION_DECISION_SCHEMA,
            "schema_version": PHYSICAL_CAUSE_EVIDENCE_VERSION,
            "operational_decision_id": self.operational_decision_id,
            "baseline_belief_id": self.baseline_belief_id,
            "selected_cause": self.selected_cause.value,
            "selected_belief_id": self.selected_belief_id,
            "certificate_ids": [item.certificate_id for item in self.certificates],
            "pairwise_certificate_ids": [
                item.certificate_id for item in self.pairwise_certificates
            ],
            "minimum_improvement": self.minimum_improvement,
            "maximum_harm_probability": self.maximum_harm_probability,
            "maximum_stratum_regret": self.maximum_stratum_regret,
            "required_strata": list(self.required_strata),
            "minimum_source_group_count": self.minimum_source_group_count,
            "pairwise_advantage": self.pairwise_advantage,
            "decision_thresholds_frozen_before_source_scores": (
                self.decision_thresholds_frozen_before_source_scores
            ),
            "target_outcomes_used": self.target_outcomes_used,
            "eligible_causes": [item.value for item in self.eligible_causes],
            "paired_attribution_resolved": self.paired_attribution_resolved,
            "nonlinear_closure_id": self.nonlinear_closure_id,
            "transport_evidence_id": self.transport_evidence_id,
            "selected_physical_attribution_claim_ready": (
                self.selected_physical_attribution_claim_ready
            ),
            "metadata": plain_json(self.metadata),
            "claim_boundary": PHYSICAL_CAUSE_ATTRIBUTION_CLAIM_BOUNDARY,
        }

    @property
    def decision_id(self) -> str:
        return cast(str, content_id(self.descriptor()))


__all__ = [
    "PHYSICAL_CAUSE_ATTRIBUTION_CLAIM_BOUNDARY",
    "PhysicalCauseAttributionDecisionV2",
    "PhysicalCausePairwiseCertificateV2",
    "PhysicalCauseRegretCertificateV2",
]
