from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any, cast

import pytest

from bayesian_phystwin.physical_cause_selection_v1 import (
    PhysicalCause,
    PhysicalCauseAmbiguityFallback,
    PhysicalCauseCandidateV1,
)
from bayesian_phystwin.physical_cause_selection_v2 import (
    PHYSICAL_CAUSE_DECISION_V2_CLAIM_BOUNDARY,
    PHYSICAL_CAUSE_DECISION_V2_SCHEMA,
    PHYSICAL_CAUSE_DECISION_V2_VERSION,
    PhysicalCauseCandidateEvidenceV2,
    PhysicalCauseDecisionPolicyV2,
    PhysicalCauseDecisionV2,
    PhysicalCauseEvidenceSetV2,
    select_physical_cause_v2,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DummyBelief:
    artifact_id: str


def _candidate(
    cause: PhysicalCause,
    label: str,
    *,
    upper_regret: float,
    inference_admissible: bool = True,
) -> tuple[DummyBelief, PhysicalCauseCandidateV1]:
    belief = DummyBelief(_digest(f"belief-{label}"))
    keywords: dict[str, str | None] = {
        "physical_response_id": None,
        "identifiability_report_id": None,
        "parameter_sensitivity_id": None,
        "bias_design_id": None,
        "discrepancy_model_id": None,
    }
    if cause is PhysicalCause.OBSERVATION_BIAS:
        keywords["bias_design_id"] = _digest(f"bias-{label}")
    elif cause is PhysicalCause.READOUT_DISCREPANCY:
        keywords["discrepancy_model_id"] = _digest(f"discrepancy-{label}")
    elif cause is PhysicalCause.PHYSICAL_PARAMETER:
        keywords["parameter_sensitivity_id"] = _digest(f"parameter-{label}")
        keywords["identifiability_report_id"] = _digest(f"identifiable-{label}")
    elif cause is PhysicalCause.PHYSICAL_STATE:
        keywords["physical_response_id"] = _digest(f"response-{label}")
        keywords["identifiability_report_id"] = _digest(f"identifiable-{label}")
    else:
        raise AssertionError("helper does not create baseline candidates")
    candidate = PhysicalCauseCandidateV1(
        cause=cause,
        belief_id=belief.artifact_id,
        construction_id=_digest(f"construction-{label}"),
        upper_regret=upper_regret,
        inference_admissible=inference_admissible,
        reason="source-crossfit",
        metadata={"label": label},
        **keywords,
    )
    return belief, candidate


def _candidate_evidence(
    candidate: PhysicalCauseCandidateV1,
    *,
    group_count: int = 12,
    score_suffix: str = "",
) -> PhysicalCauseCandidateEvidenceV2:
    return PhysicalCauseCandidateEvidenceV2(
        candidate_id=candidate.candidate_id,
        cause=candidate.cause,
        belief_id=candidate.belief_id,
        construction_id=candidate.construction_id,
        candidate_score_id=_digest(f"score-{candidate.cause.value}-{score_suffix}"),
        upper_regret=candidate.upper_regret,
        inference_admissible=candidate.inference_admissible,
        evaluated_group_count=group_count,
        simultaneous_bound=True,
        candidate_frozen_before_scores=True,
        target_outcomes_used=False,
        metadata={"unit": "object-session"},
    )


def _evidence_set(
    candidates: list[PhysicalCauseCandidateV1],
    *,
    group_count: int = 12,
) -> PhysicalCauseEvidenceSetV2:
    return PhysicalCauseEvidenceSetV2(
        common_domain_id=_digest("domain"),
        registered_query_id=_digest("query"),
        query_jacobian_id=_digest("jacobian"),
        grouping_rule_id=_digest("grouping"),
        source_roster_id=_digest("roster"),
        score_definition_id=_digest("score-definition"),
        source_score_table_id=_digest("score-table"),
        interval_method_id=_digest("max-t-bootstrap"),
        simultaneous_interval_id=_digest("simultaneous-interval"),
        confidence_level=0.95,
        source_group_count=group_count,
        registered_candidate_causes=tuple(candidate.cause for candidate in candidates),
        candidate_evidence=tuple(
            _candidate_evidence(candidate, group_count=group_count)
            for candidate in candidates
        ),
        metadata={"split": "source-only"},
    )


def _policy(
    baseline: DummyBelief,
    evidence: PhysicalCauseEvidenceSetV2,
    *,
    minimum_groups: int = 10,
    minimum_improvement: float = 0.05,
    tie_tolerance: float = 0.01,
    ambiguity_fallback: PhysicalCauseAmbiguityFallback = (
        PhysicalCauseAmbiguityFallback.BASELINE
    ),
) -> PhysicalCauseDecisionPolicyV2:
    return PhysicalCauseDecisionPolicyV2(
        baseline_belief_id=baseline.artifact_id,
        common_domain_id=evidence.common_domain_id,
        registered_query_id=evidence.registered_query_id,
        query_jacobian_id=evidence.query_jacobian_id,
        grouping_rule_id=evidence.grouping_rule_id,
        source_roster_id=evidence.source_roster_id,
        score_definition_id=evidence.score_definition_id,
        source_score_table_id=evidence.source_score_table_id,
        interval_method_id=evidence.interval_method_id,
        simultaneous_interval_id=evidence.simultaneous_interval_id,
        source_evidence_set_id=evidence.evidence_set_id,
        confidence_level=evidence.confidence_level,
        minimum_source_groups=minimum_groups,
        registered_candidate_causes=evidence.registered_candidate_causes,
        minimum_improvement=minimum_improvement,
        tie_tolerance=tie_tolerance,
        ambiguity_fallback=ambiguity_fallback,
        metadata={"target-policy": "frozen"},
    )


def test_v2_selects_exact_candidate_with_bound_evidence() -> None:
    baseline = DummyBelief(_digest("baseline"))
    discrepancy_pair = _candidate(
        PhysicalCause.READOUT_DISCREPANCY,
        "discrepancy",
        upper_regret=-0.12,
    )
    state_pair = _candidate(
        PhysicalCause.PHYSICAL_STATE,
        "state",
        upper_regret=-0.25,
    )
    pairs = [discrepancy_pair, state_pair]
    evidence = _evidence_set([pair[1] for pair in pairs])

    selected, decision = select_physical_cause_v2(
        baseline,
        pairs,
        _policy(baseline, evidence),
        evidence,
        metadata={"study": "cross-action"},
    )

    assert selected is state_pair[0]
    assert decision.selected_cause is PhysicalCause.PHYSICAL_STATE
    assert decision.exact_baseline_fallback is False
    assert decision.to_record()["schema"] == PHYSICAL_CAUSE_DECISION_V2_SCHEMA
    assert decision.to_record()["schema_version"] == PHYSICAL_CAUSE_DECISION_V2_VERSION
    assert (
        decision.to_record()["claim_boundary"]
        == PHYSICAL_CAUSE_DECISION_V2_CLAIM_BOUNDARY
    )
    assert len(decision.decision_id) == 64
    with pytest.raises(TypeError, match="immutable"):
        decision.metadata["study"] = "tampered"  # type: ignore[index]


def test_candidate_order_does_not_change_evidence_or_decision_identity() -> None:
    baseline = DummyBelief(_digest("baseline"))
    state_pair = _candidate(
        PhysicalCause.PHYSICAL_STATE,
        "state",
        upper_regret=-0.25,
    )
    bias_pair = _candidate(
        PhysicalCause.OBSERVATION_BIAS,
        "bias",
        upper_regret=-0.10,
    )
    candidates = [state_pair[1], bias_pair[1]]
    first_evidence = _evidence_set(candidates)
    second_evidence = _evidence_set(list(reversed(candidates)))
    assert first_evidence.evidence_set_id == second_evidence.evidence_set_id

    _, first = select_physical_cause_v2(
        baseline,
        [state_pair, bias_pair],
        _policy(baseline, first_evidence),
        first_evidence,
    )
    _, second = select_physical_cause_v2(
        baseline,
        [bias_pair, state_pair],
        _policy(baseline, second_evidence),
        second_evidence,
    )
    assert first.decision_id == second.decision_id


def test_unbound_regret_scalar_is_rejected() -> None:
    baseline = DummyBelief(_digest("baseline"))
    pair = _candidate(
        PhysicalCause.READOUT_DISCREPANCY,
        "discrepancy",
        upper_regret=-0.20,
    )
    evidence = _evidence_set([pair[1]])
    forged_candidate = replace(pair[1], upper_regret=-1.0)

    with pytest.raises(ValueError, match="does not bind"):
        select_physical_cause_v2(
            baseline,
            [(pair[0], forged_candidate)],
            _policy(baseline, evidence),
            evidence,
        )


def test_candidate_identity_substitution_is_rejected() -> None:
    baseline = DummyBelief(_digest("baseline"))
    pair = _candidate(
        PhysicalCause.PHYSICAL_STATE,
        "state",
        upper_regret=-0.20,
    )
    evidence = _evidence_set([pair[1]])
    bound = evidence.candidate_evidence[0]
    forged = replace(bound, candidate_id=_digest("other-candidate"))
    substituted = replace(evidence, candidate_evidence=(forged,))

    with pytest.raises(ValueError, match="does not bind"):
        select_physical_cause_v2(
            baseline,
            [pair],
            _policy(baseline, substituted),
            substituted,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("common_domain_id", _digest("other-domain")),
        ("registered_query_id", _digest("other-query")),
        ("query_jacobian_id", _digest("other-jacobian")),
        ("grouping_rule_id", _digest("other-grouping")),
        ("source_roster_id", _digest("other-roster")),
        ("score_definition_id", _digest("other-score")),
        ("source_score_table_id", _digest("other-table")),
        ("interval_method_id", _digest("other-method")),
        ("simultaneous_interval_id", _digest("other-interval")),
        ("confidence_level", 0.90),
    ],
)
def test_policy_must_match_every_source_evidence_binding(
    field: str,
    replacement: object,
) -> None:
    baseline = DummyBelief(_digest("baseline"))
    pair = _candidate(
        PhysicalCause.READOUT_DISCREPANCY,
        "discrepancy",
        upper_regret=-0.20,
    )
    evidence = _evidence_set([pair[1]])
    policy = replace(_policy(baseline, evidence), **{field: replacement})

    with pytest.raises(ValueError, match=field):
        select_physical_cause_v2(baseline, [pair], policy, evidence)


def test_policy_binds_the_exact_evidence_set() -> None:
    baseline = DummyBelief(_digest("baseline"))
    pair = _candidate(
        PhysicalCause.READOUT_DISCREPANCY,
        "discrepancy",
        upper_regret=-0.20,
    )
    evidence = _evidence_set([pair[1]])
    policy = replace(
        _policy(baseline, evidence),
        source_evidence_set_id=_digest("other-evidence-set"),
    )

    with pytest.raises(ValueError, match="does not bind"):
        select_physical_cause_v2(baseline, [pair], policy, evidence)


def test_incomplete_candidate_family_is_rejected() -> None:
    baseline = DummyBelief(_digest("baseline"))
    discrepancy_pair = _candidate(
        PhysicalCause.READOUT_DISCREPANCY,
        "discrepancy",
        upper_regret=-0.20,
    )
    state_pair = _candidate(
        PhysicalCause.PHYSICAL_STATE,
        "state",
        upper_regret=-0.25,
    )
    evidence = _evidence_set([discrepancy_pair[1], state_pair[1]])

    with pytest.raises(ValueError, match="registered candidate family"):
        select_physical_cause_v2(
            baseline,
            [state_pair],
            _policy(baseline, evidence),
            evidence,
        )


def test_every_candidate_must_use_the_complete_source_roster() -> None:
    pair = _candidate(
        PhysicalCause.READOUT_DISCREPANCY,
        "discrepancy",
        upper_regret=-0.20,
    )
    bound = _candidate_evidence(pair[1], group_count=11)

    with pytest.raises(ValueError, match="complete source group roster"):
        PhysicalCauseEvidenceSetV2(
            common_domain_id=_digest("domain"),
            registered_query_id=_digest("query"),
            query_jacobian_id=_digest("jacobian"),
            grouping_rule_id=_digest("grouping"),
            source_roster_id=_digest("roster"),
            score_definition_id=_digest("score-definition"),
            source_score_table_id=_digest("score-table"),
            interval_method_id=_digest("interval-method"),
            simultaneous_interval_id=_digest("simultaneous-interval"),
            confidence_level=0.95,
            source_group_count=12,
            registered_candidate_causes=(pair[1].cause,),
            candidate_evidence=(bound,),
        )


def test_candidate_evidence_rejects_post_score_or_target_use() -> None:
    pair = _candidate(
        PhysicalCause.PHYSICAL_STATE,
        "state",
        upper_regret=-0.20,
    )
    bound = _candidate_evidence(pair[1])
    with pytest.raises(ValueError, match="simultaneous"):
        replace(bound, simultaneous_bound=False)
    with pytest.raises(ValueError, match="frozen before"):
        replace(bound, candidate_frozen_before_scores=False)
    with pytest.raises(ValueError, match="target outcomes"):
        replace(bound, target_outcomes_used=True)


def test_source_group_minimum_is_enforced() -> None:
    baseline = DummyBelief(_digest("baseline"))
    pair = _candidate(
        PhysicalCause.READOUT_DISCREPANCY,
        "discrepancy",
        upper_regret=-0.20,
    )
    evidence = _evidence_set([pair[1]], group_count=8)

    with pytest.raises(ValueError, match="fewer groups"):
        select_physical_cause_v2(
            baseline,
            [pair],
            _policy(baseline, evidence, minimum_groups=10),
            evidence,
        )


def test_ambiguous_family_can_select_bound_discrepancy_fallback() -> None:
    baseline = DummyBelief(_digest("baseline"))
    state_pair = _candidate(
        PhysicalCause.PHYSICAL_STATE,
        "state",
        upper_regret=-0.20,
    )
    discrepancy_pair = _candidate(
        PhysicalCause.READOUT_DISCREPANCY,
        "discrepancy",
        upper_regret=-0.19,
    )
    pairs = [state_pair, discrepancy_pair]
    evidence = _evidence_set([pair[1] for pair in pairs])

    selected, decision = select_physical_cause_v2(
        baseline,
        pairs,
        _policy(
            baseline,
            evidence,
            tie_tolerance=0.02,
            ambiguity_fallback=(
                PhysicalCauseAmbiguityFallback.READOUT_DISCREPANCY
            ),
        ),
        evidence,
    )

    assert selected is discrepancy_pair[0]
    assert decision.selected_cause is PhysicalCause.READOUT_DISCREPANCY
    assert decision.ambiguity_detected


def test_insufficient_bound_returns_exact_baseline() -> None:
    baseline = DummyBelief(_digest("baseline"))
    pair = _candidate(
        PhysicalCause.PHYSICAL_STATE,
        "state",
        upper_regret=-0.01,
    )
    evidence = _evidence_set([pair[1]])

    selected, decision = select_physical_cause_v2(
        baseline,
        [pair],
        _policy(baseline, evidence, minimum_improvement=0.05),
        evidence,
    )

    assert selected is baseline
    assert decision.exact_baseline_fallback
    assert decision.reason == "no-source-supported-candidate"


def test_decision_rejects_a_routed_policy_from_another_evidence_set() -> None:
    baseline = DummyBelief(_digest("baseline"))
    pair = _candidate(
        PhysicalCause.READOUT_DISCREPANCY,
        "discrepancy",
        upper_regret=-0.20,
    )
    evidence = _evidence_set([pair[1]])
    _, decision = select_physical_cause_v2(
        baseline,
        [pair],
        _policy(baseline, evidence),
        evidence,
    )
    other_evidence = replace(
        evidence,
        source_score_table_id=_digest("other-score-table"),
    )
    other_policy = _policy(baseline, other_evidence)

    with pytest.raises(ValueError):
        PhysicalCauseDecisionV2(
            policy=other_policy,
            source_evidence=other_evidence,
            routed_decision=decision.routed_decision,
        )


def test_invalid_contract_values_fail_closed() -> None:
    baseline = DummyBelief(_digest("baseline"))
    pair = _candidate(
        PhysicalCause.READOUT_DISCREPANCY,
        "discrepancy",
        upper_regret=-0.20,
    )
    evidence = _evidence_set([pair[1]])
    bound = evidence.candidate_evidence[0]

    with pytest.raises(ValueError, match="strictly between"):
        replace(evidence, confidence_level=1.0)
    with pytest.raises(ValueError, match="integer"):
        replace(bound, evaluated_group_count=cast(Any, True))
    with pytest.raises(ValueError, match="lowercase hexadecimal"):
        replace(bound, candidate_score_id="bad")
    with pytest.raises(TypeError, match="PhysicalCause"):
        replace(bound, cause=cast(Any, "physical_state"))
    with pytest.raises(ValueError, match="finite JSON"):
        replace(_policy(baseline, evidence), metadata={"bad": float("inf")})
