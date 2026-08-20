from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np
import pytest

from bayesian_phystwin.physical_cause_selection_v1 import (
    PHYSICAL_CAUSE_DECISION_CLAIM_BOUNDARY,
    PHYSICAL_CAUSE_DECISION_SCHEMA,
    PHYSICAL_CAUSE_DECISION_VERSION,
    PhysicalCause,
    PhysicalCauseAmbiguityFallback,
    PhysicalCauseCandidateV1,
    PhysicalCauseDecisionPolicyV1,
    PhysicalCauseDecisionV1,
    select_physical_cause,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DummyBelief:
    artifact_id: str


def _policy(
    baseline: DummyBelief,
    *,
    minimum_improvement: float = 0.0,
    tie_tolerance: float = 0.0,
    ambiguity_fallback: PhysicalCauseAmbiguityFallback = (
        PhysicalCauseAmbiguityFallback.BASELINE
    ),
) -> PhysicalCauseDecisionPolicyV1:
    return PhysicalCauseDecisionPolicyV1(
        baseline_belief_id=baseline.artifact_id,
        common_domain_id=_digest("domain"),
        registered_query_id=_digest("query"),
        source_evidence_id=_digest("source"),
        minimum_improvement=minimum_improvement,
        tie_tolerance=tie_tolerance,
        ambiguity_fallback=ambiguity_fallback,
        metadata={"split": "source-only"},
    )


def _candidate(
    cause: PhysicalCause,
    label: str,
    *,
    upper_regret: float,
    inference_admissible: bool = True,
) -> tuple[DummyBelief, PhysicalCauseCandidateV1]:
    belief = DummyBelief(_digest(f"belief-{label}"))
    keyword: dict[str, str | None] = {
        "physical_response_id": None,
        "identifiability_report_id": None,
        "parameter_sensitivity_id": None,
        "bias_design_id": None,
        "discrepancy_model_id": None,
    }
    if cause is PhysicalCause.OBSERVATION_BIAS:
        keyword["bias_design_id"] = _digest(f"bias-{label}")
    elif cause is PhysicalCause.READOUT_DISCREPANCY:
        keyword["discrepancy_model_id"] = _digest(f"discrepancy-{label}")
        keyword["bias_design_id"] = _digest(f"optional-bias-{label}")
    elif cause is PhysicalCause.PHYSICAL_PARAMETER:
        keyword["identifiability_report_id"] = _digest(f"identifiable-{label}")
        keyword["parameter_sensitivity_id"] = _digest(f"sensitivity-{label}")
        keyword["bias_design_id"] = _digest(f"optional-bias-{label}")
    elif cause is PhysicalCause.PHYSICAL_STATE:
        keyword["physical_response_id"] = _digest(f"response-{label}")
        keyword["identifiability_report_id"] = _digest(f"identifiable-{label}")
        keyword["bias_design_id"] = _digest(f"optional-bias-{label}")
    else:
        raise AssertionError("helper does not construct baseline candidates")
    candidate = PhysicalCauseCandidateV1(
        cause=cause,
        belief_id=belief.artifact_id,
        construction_id=_digest(f"construction-{label}"),
        upper_regret=upper_regret,
        inference_admissible=inference_admissible,
        reason="source-cross-fit",
        metadata={"label": label},
        **keyword,
    )
    return belief, candidate


def test_constants_and_records_are_content_addressed() -> None:
    baseline = DummyBelief(_digest("baseline"))
    belief, candidate = _candidate(
        PhysicalCause.READOUT_DISCREPANCY,
        "discrepancy",
        upper_regret=-0.2,
    )
    selected, decision = select_physical_cause(
        baseline,
        [(belief, candidate)],
        _policy(baseline),
        metadata={"study": "controlled"},
    )

    assert selected is belief
    assert PHYSICAL_CAUSE_DECISION_SCHEMA in decision.to_record()["schema"]
    assert decision.to_record()["schema_version"] == PHYSICAL_CAUSE_DECISION_VERSION
    assert (
        decision.to_record()["claim_boundary"] == PHYSICAL_CAUSE_DECISION_CLAIM_BOUNDARY
    )
    assert len(candidate.candidate_id) == 64
    assert len(decision.policy_id) == 64
    assert len(decision.decision_id) == 64
    assert decision.decision_id == decision.decision_id
    with pytest.raises(TypeError, match="immutable"):
        decision.metadata["study"] = "tampered"  # type: ignore[index]


def test_candidate_semantics_require_cause_specific_evidence() -> None:
    common: dict[str, object] = {
        "belief_id": _digest("belief"),
        "construction_id": _digest("construction"),
        "upper_regret": -0.1,
        "inference_admissible": True,
        "reason": "test",
    }
    with pytest.raises(ValueError, match="owned by the decision policy"):
        PhysicalCauseCandidateV1(cause=PhysicalCause.BASELINE, **common)
    with pytest.raises(ValueError, match="bias_design_id"):
        PhysicalCauseCandidateV1(cause=PhysicalCause.OBSERVATION_BIAS, **common)
    with pytest.raises(ValueError, match="discrepancy_model_id"):
        PhysicalCauseCandidateV1(cause=PhysicalCause.READOUT_DISCREPANCY, **common)
    with pytest.raises(ValueError, match="parameter sensitivity"):
        PhysicalCauseCandidateV1(cause=PhysicalCause.PHYSICAL_PARAMETER, **common)
    with pytest.raises(ValueError, match="physical response"):
        PhysicalCauseCandidateV1(cause=PhysicalCause.PHYSICAL_STATE, **common)


def test_candidate_semantics_reject_cross_cause_evidence() -> None:
    common: dict[str, object] = {
        "belief_id": _digest("belief"),
        "construction_id": _digest("construction"),
        "upper_regret": -0.1,
        "inference_admissible": True,
        "reason": "test",
    }
    with pytest.raises(ValueError, match="cannot claim physical"):
        PhysicalCauseCandidateV1(
            cause=PhysicalCause.OBSERVATION_BIAS,
            bias_design_id=_digest("bias"),
            physical_response_id=_digest("response"),
            **common,
        )
    with pytest.raises(ValueError, match="cannot claim state"):
        PhysicalCauseCandidateV1(
            cause=PhysicalCause.READOUT_DISCREPANCY,
            discrepancy_model_id=_digest("discrepancy"),
            identifiability_report_id=_digest("identifiable"),
            **common,
        )
    with pytest.raises(ValueError, match="cannot claim state-response"):
        PhysicalCauseCandidateV1(
            cause=PhysicalCause.PHYSICAL_PARAMETER,
            parameter_sensitivity_id=_digest("sensitivity"),
            identifiability_report_id=_digest("identifiable"),
            physical_response_id=_digest("response"),
            **common,
        )
    with pytest.raises(ValueError, match="cannot claim parameter"):
        PhysicalCauseCandidateV1(
            cause=PhysicalCause.PHYSICAL_STATE,
            physical_response_id=_digest("response"),
            identifiability_report_id=_digest("identifiable"),
            parameter_sensitivity_id=_digest("sensitivity"),
            **common,
        )


def test_candidate_and_policy_fail_closed_on_invalid_values() -> None:
    baseline = DummyBelief(_digest("baseline"))
    _, candidate = _candidate(
        PhysicalCause.PHYSICAL_STATE,
        "state",
        upper_regret=-0.1,
    )
    with pytest.raises(TypeError, match="PhysicalCause"):
        replace(candidate, cause=cast(Any, "physical_state"))
    with pytest.raises(ValueError, match="finite real"):
        replace(candidate, upper_regret=np.nan)
    with pytest.raises(ValueError, match="finite real"):
        replace(candidate, upper_regret=True)
    with pytest.raises(ValueError, match="boolean"):
        replace(candidate, inference_admissible=cast(Any, 1))
    with pytest.raises(ValueError, match="nonempty literal"):
        replace(candidate, reason="")
    with pytest.raises(ValueError, match="lowercase hexadecimal"):
        replace(candidate, belief_id="bad")
    with pytest.raises(ValueError, match="finite JSON"):
        replace(candidate, metadata={"bad": float("inf")})

    policy = _policy(baseline)
    with pytest.raises(ValueError, match="at least 0.0"):
        replace(policy, minimum_improvement=-0.1)
    with pytest.raises(ValueError, match="at least 0.0"):
        replace(policy, tie_tolerance=-0.1)
    with pytest.raises(TypeError, match="AmbiguityFallback"):
        replace(policy, ambiguity_fallback=cast(Any, "baseline"))


def test_unique_source_supported_state_candidate_is_selected_exactly() -> None:
    baseline = DummyBelief(_digest("baseline"))
    state_belief, state = _candidate(
        PhysicalCause.PHYSICAL_STATE,
        "state",
        upper_regret=-0.25,
    )
    discrepancy_belief, discrepancy = _candidate(
        PhysicalCause.READOUT_DISCREPANCY,
        "discrepancy",
        upper_regret=-0.10,
    )

    selected, decision = select_physical_cause(
        baseline,
        [(discrepancy_belief, discrepancy), (state_belief, state)],
        _policy(baseline, minimum_improvement=0.05, tie_tolerance=0.01),
    )

    assert selected is state_belief
    assert decision.selected_cause is PhysicalCause.PHYSICAL_STATE
    assert decision.selected_candidate_id == state.candidate_id
    assert decision.exact_baseline_fallback is False
    assert decision.ambiguity_detected is False
    assert decision.reason == "source-supported-unique-cause"


def test_inadmissible_or_insufficient_candidates_return_exact_baseline() -> None:
    baseline = DummyBelief(_digest("baseline"))
    weak_belief, weak = _candidate(
        PhysicalCause.READOUT_DISCREPANCY,
        "weak",
        upper_regret=-0.01,
    )
    rejected_belief, rejected = _candidate(
        PhysicalCause.PHYSICAL_STATE,
        "rejected",
        upper_regret=-1.0,
        inference_admissible=False,
    )

    selected, decision = select_physical_cause(
        baseline,
        [(weak_belief, weak), (rejected_belief, rejected)],
        _policy(baseline, minimum_improvement=0.05),
    )

    assert selected is baseline
    assert decision.selected_cause is PhysicalCause.BASELINE
    assert decision.selected_candidate_id is None
    assert decision.exact_baseline_fallback is True
    assert decision.ambiguity_detected is False
    assert decision.reason == "no-source-supported-candidate"


def test_exact_baseline_score_tie_does_not_advance() -> None:
    baseline = DummyBelief(_digest("baseline"))
    belief, candidate = _candidate(
        PhysicalCause.READOUT_DISCREPANCY,
        "tie",
        upper_regret=0.0,
    )

    selected, decision = select_physical_cause(
        baseline,
        [(belief, candidate)],
        _policy(baseline, minimum_improvement=0.0),
    )

    assert selected is baseline
    assert decision.reason == "no-source-supported-candidate"


def test_empty_candidate_set_returns_exact_baseline() -> None:
    baseline = DummyBelief(_digest("baseline"))
    selected, decision = select_physical_cause(baseline, [], _policy(baseline))

    assert selected is baseline
    assert decision.candidates == ()
    assert decision.exact_baseline_fallback is True


def test_ambiguous_physical_causes_fall_back_to_exact_baseline() -> None:
    baseline = DummyBelief(_digest("baseline"))
    state_belief, state = _candidate(
        PhysicalCause.PHYSICAL_STATE,
        "state",
        upper_regret=-0.20,
    )
    parameter_belief, parameter = _candidate(
        PhysicalCause.PHYSICAL_PARAMETER,
        "parameter",
        upper_regret=-0.19,
    )

    selected, decision = select_physical_cause(
        baseline,
        [(state_belief, state), (parameter_belief, parameter)],
        _policy(baseline, tie_tolerance=0.02),
    )

    assert selected is baseline
    assert decision.selected_cause is PhysicalCause.BASELINE
    assert decision.ambiguity_detected is True
    assert decision.reason == "ambiguous-exact-baseline-fallback"


def test_ambiguous_attribution_can_choose_near_best_discrepancy() -> None:
    baseline = DummyBelief(_digest("baseline"))
    state_belief, state = _candidate(
        PhysicalCause.PHYSICAL_STATE,
        "state",
        upper_regret=-0.20,
    )
    discrepancy_belief, discrepancy = _candidate(
        PhysicalCause.READOUT_DISCREPANCY,
        "discrepancy",
        upper_regret=-0.19,
    )

    selected, decision = select_physical_cause(
        baseline,
        [(state_belief, state), (discrepancy_belief, discrepancy)],
        _policy(
            baseline,
            tie_tolerance=0.02,
            ambiguity_fallback=(PhysicalCauseAmbiguityFallback.READOUT_DISCREPANCY),
        ),
    )

    assert selected is discrepancy_belief
    assert decision.selected_cause is PhysicalCause.READOUT_DISCREPANCY
    assert decision.ambiguity_detected is True
    assert decision.reason == "ambiguous-select-readout-discrepancy"


def test_discrepancy_fallback_policy_does_not_select_a_non_near_discrepancy() -> None:
    baseline = DummyBelief(_digest("baseline"))
    state_belief, state = _candidate(
        PhysicalCause.PHYSICAL_STATE,
        "state",
        upper_regret=-0.30,
    )
    parameter_belief, parameter = _candidate(
        PhysicalCause.PHYSICAL_PARAMETER,
        "parameter",
        upper_regret=-0.29,
    )
    discrepancy_belief, discrepancy = _candidate(
        PhysicalCause.READOUT_DISCREPANCY,
        "discrepancy",
        upper_regret=-0.10,
    )

    selected, decision = select_physical_cause(
        baseline,
        [
            (state_belief, state),
            (parameter_belief, parameter),
            (discrepancy_belief, discrepancy),
        ],
        _policy(
            baseline,
            tie_tolerance=0.02,
            ambiguity_fallback=(PhysicalCauseAmbiguityFallback.READOUT_DISCREPANCY),
        ),
    )

    assert selected is baseline
    assert decision.ambiguity_detected is True


def test_candidate_order_does_not_change_decision_identity() -> None:
    baseline = DummyBelief(_digest("baseline"))
    state_pair = _candidate(
        PhysicalCause.PHYSICAL_STATE,
        "state",
        upper_regret=-0.20,
    )
    bias_pair = _candidate(
        PhysicalCause.OBSERVATION_BIAS,
        "bias",
        upper_regret=-0.10,
    )
    policy = _policy(baseline)

    _, first = select_physical_cause(baseline, [state_pair, bias_pair], policy)
    _, second = select_physical_cause(baseline, [bias_pair, state_pair], policy)

    assert first.candidates == second.candidates
    assert first.decision_id == second.decision_id
    assert [candidate.cause for candidate in first.candidates] == [
        PhysicalCause.OBSERVATION_BIAS,
        PhysicalCause.PHYSICAL_STATE,
    ]


def test_duplicate_causes_and_beliefs_fail_closed() -> None:
    baseline = DummyBelief(_digest("baseline"))
    first_belief, first = _candidate(
        PhysicalCause.PHYSICAL_STATE,
        "first",
        upper_regret=-0.2,
    )
    second_belief, second = _candidate(
        PhysicalCause.PHYSICAL_STATE,
        "second",
        upper_regret=-0.3,
    )
    with pytest.raises(ValueError, match="one candidate per physical cause"):
        select_physical_cause(
            baseline,
            [(first_belief, first), (second_belief, second)],
            _policy(baseline),
        )

    duplicate_belief_candidate = replace(
        second,
        cause=PhysicalCause.PHYSICAL_PARAMETER,
        belief_id=first_belief.artifact_id,
        physical_response_id=None,
        parameter_sensitivity_id=_digest("sensitivity"),
    )
    with pytest.raises(ValueError, match="belief does not match candidate"):
        select_physical_cause(
            baseline,
            [(first_belief, first), (second_belief, duplicate_belief_candidate)],
            _policy(baseline),
        )


def test_selection_rejects_unbound_or_malformed_beliefs() -> None:
    baseline = DummyBelief(_digest("baseline"))
    belief, candidate = _candidate(
        PhysicalCause.READOUT_DISCREPANCY,
        "discrepancy",
        upper_regret=-0.2,
    )
    with pytest.raises(ValueError, match="does not bind the baseline"):
        select_physical_cause(
            baseline,
            [(belief, candidate)],
            replace(_policy(baseline), baseline_belief_id=_digest("other")),
        )
    with pytest.raises(TypeError, match="must expose artifact_id"):
        select_physical_cause(
            cast(Any, object()),
            [(belief, candidate)],
            _policy(baseline),
        )
    with pytest.raises(TypeError, match="belief/candidate pair"):
        select_physical_cause(
            baseline,
            cast(Any, [candidate]),
            _policy(baseline),
        )
    with pytest.raises(TypeError, match="PhysicalCauseCandidateV1"):
        select_physical_cause(
            baseline,
            cast(Any, [(belief, object())]),
            _policy(baseline),
        )
    with pytest.raises(ValueError, match="belief does not match candidate"):
        select_physical_cause(
            baseline,
            [(DummyBelief(_digest("other")), candidate)],
            _policy(baseline),
        )
    with pytest.raises(TypeError, match="policy"):
        select_physical_cause(
            baseline,
            [(belief, candidate)],
            cast(Any, object()),
        )


def test_candidate_cannot_reuse_baseline_identity() -> None:
    baseline = DummyBelief(_digest("baseline"))
    _, candidate = _candidate(
        PhysicalCause.READOUT_DISCREPANCY,
        "discrepancy",
        upper_regret=-0.2,
    )
    reused = replace(candidate, belief_id=baseline.artifact_id)
    with pytest.raises(ValueError, match="distinct from the baseline"):
        select_physical_cause(
            baseline,
            [(baseline, reused)],
            _policy(baseline),
        )


def test_decision_constructor_rejects_tampered_routing() -> None:
    baseline = DummyBelief(_digest("baseline"))
    belief, candidate = _candidate(
        PhysicalCause.READOUT_DISCREPANCY,
        "discrepancy",
        upper_regret=-0.2,
    )
    _, decision = select_physical_cause(
        baseline,
        [(belief, candidate)],
        _policy(baseline),
    )

    with pytest.raises(ValueError, match="contradicts the frozen policy"):
        replace(decision, selected_cause=PhysicalCause.BASELINE)
    with pytest.raises(ValueError, match="contradicts the frozen policy"):
        replace(decision, selected_belief_id=baseline.artifact_id)
    with pytest.raises(ValueError, match="contradicts the frozen policy"):
        replace(decision, exact_baseline_fallback=True)
    with pytest.raises(TypeError, match="policy"):
        replace(decision, policy=cast(Any, object()))
    with pytest.raises(TypeError, match="selected_cause"):
        replace(decision, selected_cause=cast(Any, "readout_discrepancy"))


def test_decision_constructor_rejects_duplicate_or_baseline_candidates() -> None:
    baseline = DummyBelief(_digest("baseline"))
    belief, candidate = _candidate(
        PhysicalCause.READOUT_DISCREPANCY,
        "discrepancy",
        upper_regret=-0.2,
    )
    _, decision = select_physical_cause(
        baseline,
        [(belief, candidate)],
        _policy(baseline),
    )
    with pytest.raises(ValueError, match="one candidate per physical cause"):
        replace(decision, candidates=(candidate, candidate))
    reused = replace(candidate, belief_id=baseline.artifact_id)
    with pytest.raises(ValueError, match="distinct from the baseline"):
        replace(
            decision,
            candidates=(reused,),
            selected_belief_id=baseline.artifact_id,
        )


def test_numpy_scalars_are_canonicalized_without_bool_coercion() -> None:
    baseline = DummyBelief(_digest("baseline"))
    belief, candidate = _candidate(
        PhysicalCause.OBSERVATION_BIAS,
        "bias",
        upper_regret=np.float64(-0.2),
        inference_admissible=np.bool_(True),
    )
    policy = PhysicalCauseDecisionPolicyV1(
        baseline_belief_id=baseline.artifact_id,
        common_domain_id=_digest("domain"),
        registered_query_id=_digest("query"),
        source_evidence_id=_digest("source"),
        minimum_improvement=np.float64(0.1),
        tie_tolerance=np.float64(0.0),
    )

    selected, decision = select_physical_cause(
        baseline,
        [(belief, candidate)],
        policy,
    )

    assert selected is belief
    assert type(candidate.upper_regret) is float
    assert type(candidate.inference_admissible) is bool
    assert type(policy.minimum_improvement) is float
    assert type(decision.exact_baseline_fallback) is bool


def test_public_types_are_importable_and_named() -> None:
    assert PhysicalCauseDecisionV1.__name__ == "PhysicalCauseDecisionV1"
    assert PhysicalCauseCandidateV1.__name__ == "PhysicalCauseCandidateV1"
    assert PhysicalCauseDecisionPolicyV1.__name__ == "PhysicalCauseDecisionPolicyV1"
