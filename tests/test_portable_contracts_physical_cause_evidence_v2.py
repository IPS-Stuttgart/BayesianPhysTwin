from __future__ import annotations

from dataclasses import replace

import pytest

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.physical_cause_evidence_v2 import (
    PhysicalCauseAttributionDecisionV2,
    PhysicalCausePairwiseCertificateV2,
    PhysicalCauseRegretCertificateV2,
)
from bayesian_phystwin.physical_cause_selection_v1 import PhysicalCause


def _id(value: int) -> str:
    return f"{value:064x}"


_GROUPS = tuple(_id(100 + index) for index in range(12))


def _certificate(
    cause: PhysicalCause,
    *,
    belief: int,
    construction: int,
    upper_regret: float = -0.05,
    harm_probability: float = 0.05,
    harm_margin: float = 0.0,
    groups: tuple[str, ...] = _GROUPS,
    strata: dict[str, float] | None = None,
) -> PhysicalCauseRegretCertificateV2:
    return PhysicalCauseRegretCertificateV2(
        cause=cause,
        baseline_belief_id=_id(1),
        candidate_belief_id=_id(belief),
        candidate_construction_id=_id(construction),
        common_domain_id=_id(2),
        registered_query_id=_id(3),
        source_evidence_id=_id(4),
        proper_score_id=_id(5),
        grouping_rule_id=_id(6),
        candidate_universe_id=_id(7),
        source_group_ids=groups,
        simultaneous_upper_regret=upper_regret,
        harm_margin=harm_margin,
        harm_probability_upper=harm_probability,
        confidence_level=0.95,
        stratum_upper_regrets=(
            {"sheet": -0.02, "volumetric": -0.01} if strata is None else strata
        ),
    )


def _candidate_id(certificate: PhysicalCauseRegretCertificateV2) -> str:
    return content_id(
        {
            "cause": certificate.cause.value,
            "belief_id": certificate.candidate_belief_id,
            "construction_id": certificate.candidate_construction_id,
        }
    )


def _pair(
    left: PhysicalCauseRegretCertificateV2,
    right: PhysicalCauseRegretCertificateV2,
    *,
    lower: float,
    upper: float,
) -> PhysicalCausePairwiseCertificateV2:
    assert left.cause.value < right.cause.value
    return PhysicalCausePairwiseCertificateV2(
        left_cause=left.cause,
        right_cause=right.cause,
        left_candidate_id=_candidate_id(left),
        right_candidate_id=_candidate_id(right),
        baseline_belief_id=left.baseline_belief_id,
        common_domain_id=left.common_domain_id,
        registered_query_id=left.registered_query_id,
        source_evidence_id=left.source_evidence_id,
        proper_score_id=left.proper_score_id,
        grouping_rule_id=left.grouping_rule_id,
        source_group_ids=left.source_group_ids,
        candidate_universe_id=left.candidate_universe_id,
        lower_regret_difference=lower,
        upper_regret_difference=upper,
        confidence_level=left.confidence_level,
    )


def _decision(
    state: PhysicalCauseRegretCertificateV2,
    discrepancy: PhysicalCauseRegretCertificateV2,
    *,
    pairs: tuple[PhysicalCausePairwiseCertificateV2, ...],
    closure: str | None = _id(30),
    transport: str | None = _id(31),
) -> PhysicalCauseAttributionDecisionV2:
    return PhysicalCauseAttributionDecisionV2(
        operational_decision_id=_id(20),
        baseline_belief_id=state.baseline_belief_id,
        selected_cause=PhysicalCause.PHYSICAL_STATE,
        selected_belief_id=state.candidate_belief_id,
        certificates=(discrepancy, state),
        pairwise_certificates=pairs,
        minimum_improvement=0.01,
        maximum_harm_probability=0.20,
        maximum_stratum_regret=0.0,
        required_strata=("volumetric", "sheet"),
        minimum_source_group_count=10,
        pairwise_advantage=0.005,
        nonlinear_closure_id=closure,
        transport_evidence_id=transport,
    )


def test_physical_claim_requires_paired_dominance_closure_and_transport() -> None:
    state = _certificate(PhysicalCause.PHYSICAL_STATE, belief=10, construction=11)
    discrepancy = _certificate(
        PhysicalCause.READOUT_DISCREPANCY,
        belief=12,
        construction=13,
    )
    pair = _pair(state, discrepancy, lower=-0.08, upper=-0.02)

    decision = _decision(state, discrepancy, pairs=(pair,))

    assert decision.eligible_causes == (
        PhysicalCause.PHYSICAL_STATE,
        PhysicalCause.READOUT_DISCREPANCY,
    )
    assert decision.paired_attribution_resolved
    assert decision.selected_physical_attribution_claim_ready
    assert len(decision.decision_id) == 64
    assert decision.descriptor()["claim_boundary"]

    no_transport = _decision(state, discrepancy, pairs=(pair,), transport=None)
    assert no_transport.paired_attribution_resolved
    assert not no_transport.selected_physical_attribution_claim_ready

    no_closure = _decision(state, discrepancy, pairs=(pair,), closure=None)
    assert not no_closure.selected_physical_attribution_claim_ready


def test_missing_or_unresolved_pairwise_evidence_keeps_attribution_unresolved() -> None:
    state = _certificate(PhysicalCause.PHYSICAL_STATE, belief=10, construction=11)
    discrepancy = _certificate(
        PhysicalCause.READOUT_DISCREPANCY,
        belief=12,
        construction=13,
    )
    missing = _decision(state, discrepancy, pairs=())
    assert not missing.paired_attribution_resolved
    assert not missing.selected_physical_attribution_claim_ready

    overlap = _pair(state, discrepancy, lower=-0.02, upper=0.01)
    unresolved = _decision(state, discrepancy, pairs=(overlap,))
    assert not unresolved.paired_attribution_resolved


def test_pairwise_direction_can_certify_right_candidate() -> None:
    parameter = _certificate(
        PhysicalCause.PHYSICAL_PARAMETER,
        belief=14,
        construction=15,
    )
    state = _certificate(PhysicalCause.PHYSICAL_STATE, belief=10, construction=11)
    pair = _pair(parameter, state, lower=0.02, upper=0.08)
    decision = PhysicalCauseAttributionDecisionV2(
        operational_decision_id=_id(20),
        baseline_belief_id=state.baseline_belief_id,
        selected_cause=PhysicalCause.PHYSICAL_STATE,
        selected_belief_id=state.candidate_belief_id,
        certificates=(state, parameter),
        pairwise_certificates=(pair,),
        minimum_improvement=0.01,
        maximum_harm_probability=0.20,
        maximum_stratum_regret=0.0,
        required_strata=("sheet", "volumetric"),
        minimum_source_group_count=10,
        pairwise_advantage=0.005,
        nonlinear_closure_id=_id(30),
        transport_evidence_id=_id(31),
    )
    assert decision.paired_attribution_resolved


def test_candidate_must_pass_every_registered_group_gate() -> None:
    good = _certificate(PhysicalCause.PHYSICAL_STATE, belief=10, construction=11)
    weak = _certificate(
        PhysicalCause.READOUT_DISCREPANCY,
        belief=12,
        construction=13,
        upper_regret=-0.005,
    )
    harmful = _certificate(
        PhysicalCause.OBSERVATION_BIAS,
        belief=14,
        construction=15,
        harm_probability=0.30,
    )
    bad_stratum = _certificate(
        PhysicalCause.PHYSICAL_PARAMETER,
        belief=16,
        construction=17,
        strata={"sheet": 0.01, "volumetric": -0.02},
    )
    too_few = _certificate(
        PhysicalCause.READOUT_DISCREPANCY,
        belief=18,
        construction=19,
        groups=_GROUPS[:5],
    )

    decision = PhysicalCauseAttributionDecisionV2(
        operational_decision_id=_id(20),
        baseline_belief_id=good.baseline_belief_id,
        selected_cause=PhysicalCause.PHYSICAL_STATE,
        selected_belief_id=good.candidate_belief_id,
        certificates=(good, weak, harmful, bad_stratum),
        pairwise_certificates=(),
        minimum_improvement=0.01,
        maximum_harm_probability=0.20,
        maximum_stratum_regret=0.0,
        required_strata=("sheet", "volumetric"),
        minimum_source_group_count=10,
        nonlinear_closure_id=_id(30),
        transport_evidence_id=_id(31),
    )
    assert decision.eligible_causes == (PhysicalCause.PHYSICAL_STATE,)
    assert decision.paired_attribution_resolved

    few = PhysicalCauseAttributionDecisionV2(
        operational_decision_id=_id(20),
        baseline_belief_id=too_few.baseline_belief_id,
        selected_cause=PhysicalCause.READOUT_DISCREPANCY,
        selected_belief_id=too_few.candidate_belief_id,
        certificates=(too_few,),
        pairwise_certificates=(),
        minimum_improvement=0.01,
        maximum_harm_probability=0.20,
        maximum_stratum_regret=0.0,
        required_strata=("sheet", "volumetric"),
        minimum_source_group_count=10,
    )
    assert few.eligible_causes == ()
    assert not few.paired_attribution_resolved


def test_nonphysical_unique_eligible_candidate_can_be_claim_ready() -> None:
    discrepancy = _certificate(
        PhysicalCause.READOUT_DISCREPANCY,
        belief=12,
        construction=13,
    )
    decision = PhysicalCauseAttributionDecisionV2(
        operational_decision_id=_id(20),
        baseline_belief_id=discrepancy.baseline_belief_id,
        selected_cause=PhysicalCause.READOUT_DISCREPANCY,
        selected_belief_id=discrepancy.candidate_belief_id,
        certificates=(discrepancy,),
        pairwise_certificates=(),
        minimum_improvement=0.01,
        maximum_harm_probability=0.20,
        maximum_stratum_regret=0.0,
        required_strata=("sheet", "volumetric"),
        minimum_source_group_count=10,
    )
    assert decision.paired_attribution_resolved
    assert decision.selected_physical_attribution_claim_ready


def test_baseline_selection_without_certificates_is_not_attribution_claim() -> None:
    decision = PhysicalCauseAttributionDecisionV2(
        operational_decision_id=_id(20),
        baseline_belief_id=_id(1),
        selected_cause=PhysicalCause.BASELINE,
        selected_belief_id=_id(1),
        certificates=(),
        pairwise_certificates=(),
        minimum_improvement=0.0,
        maximum_harm_probability=1.0,
        maximum_stratum_regret=0.0,
    )
    assert decision.eligible_causes == ()
    assert not decision.paired_attribution_resolved
    assert not decision.selected_physical_attribution_claim_ready


def test_regret_certificate_is_content_addressed_and_order_canonical() -> None:
    certificate = _certificate(
        PhysicalCause.READOUT_DISCREPANCY,
        belief=12,
        construction=13,
    )
    reordered = _certificate(
        PhysicalCause.READOUT_DISCREPANCY,
        belief=12,
        construction=13,
        groups=tuple(reversed(_GROUPS)),
    )
    assert certificate.source_group_ids == reordered.source_group_ids
    assert certificate.certificate_id == reordered.certificate_id
    assert certificate.source_group_count == 12


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"cause": PhysicalCause.BASELINE}, "nonbaseline"),
        ({"candidate_belief_id": _id(1)}, "differ from baseline"),
        ({"source_group_ids": (_id(100), _id(100))}, "unique"),
        ({"harm_margin": -0.1}, "nonnegative"),
        ({"harm_probability_upper": 1.1}, "[0, 1]"),
        ({"confidence_level": 1.0}, "(0, 1)"),
        ({"bounds_simultaneous": False}, "must be true"),
        ({"thresholds_frozen_before_source_scores": False}, "must be true"),
        ({"candidate_universe_frozen_before_source_scores": False}, "must be true"),
        ({"target_outcomes_used": True}, "must be false"),
        ({"source_groups_independent": False}, "must be true"),
    ],
)
def test_regret_certificate_rejects_invalid_claim_inputs(
    changes: dict[str, object],
    message: str,
) -> None:
    certificate = _certificate(
        PhysicalCause.READOUT_DISCREPANCY,
        belief=12,
        construction=13,
    )
    with pytest.raises((TypeError, ValueError), match=message):
        replace(certificate, **changes)


def test_pairwise_certificate_rejects_bad_order_and_interval() -> None:
    state = _certificate(PhysicalCause.PHYSICAL_STATE, belief=10, construction=11)
    discrepancy = _certificate(
        PhysicalCause.READOUT_DISCREPANCY,
        belief=12,
        construction=13,
    )
    pair = _pair(state, discrepancy, lower=-0.08, upper=-0.02)
    with pytest.raises(ValueError, match="canonical lexical order"):
        replace(
            pair,
            left_cause=discrepancy.cause,
            right_cause=state.cause,
            left_candidate_id=_candidate_id(discrepancy),
            right_candidate_id=_candidate_id(state),
        )
    with pytest.raises(ValueError, match="lower bound"):
        replace(pair, lower_regret_difference=0.1, upper_regret_difference=0.0)
    with pytest.raises(ValueError, match="must be true"):
        replace(pair, bounds_simultaneous=False)
    with pytest.raises(ValueError, match="must be true"):
        replace(pair, pairwise_procedure_frozen_before_source_scores=False)
    with pytest.raises(ValueError, match="must be true"):
        replace(pair, candidate_universe_frozen_before_source_scores=False)
    with pytest.raises(ValueError, match="must be false"):
        replace(pair, target_outcomes_used=True)
    with pytest.raises(ValueError, match="must be true"):
        replace(pair, source_groups_independent=False)


def test_decision_rejects_cross_domain_or_wrong_pair_binding() -> None:
    state = _certificate(PhysicalCause.PHYSICAL_STATE, belief=10, construction=11)
    discrepancy = _certificate(
        PhysicalCause.READOUT_DISCREPANCY,
        belief=12,
        construction=13,
    )
    mismatched = replace(discrepancy, registered_query_id=_id(99))
    with pytest.raises(ValueError, match="registered_query_id differs"):
        _decision(state, mismatched, pairs=())

    pair = _pair(state, discrepancy, lower=-0.08, upper=-0.02)
    wrong_pair = replace(pair, left_candidate_id=_id(99))
    with pytest.raises(ValueError, match="does not bind registered candidates"):
        _decision(state, discrepancy, pairs=(wrong_pair,))


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("baseline_belief_id", "baseline_belief_id differs"),
        ("common_domain_id", "common_domain_id differs"),
        ("registered_query_id", "registered_query_id differs"),
        ("source_evidence_id", "source_evidence_id differs"),
        ("proper_score_id", "proper_score_id differs"),
        ("grouping_rule_id", "grouping_rule_id differs"),
        ("candidate_universe_id", "candidate_universe_id differs"),
    ],
)
def test_pairwise_certificate_must_bind_exact_regret_context(
    field: str,
    message: str,
) -> None:
    state = _certificate(PhysicalCause.PHYSICAL_STATE, belief=10, construction=11)
    discrepancy = _certificate(
        PhysicalCause.READOUT_DISCREPANCY,
        belief=12,
        construction=13,
    )
    pair = _pair(state, discrepancy, lower=-0.08, upper=-0.02)
    mismatched = replace(pair, **{field: _id(99)})
    with pytest.raises(ValueError, match=message):
        _decision(state, discrepancy, pairs=(mismatched,))


def test_decision_rejects_selected_belief_or_baseline_substitution() -> None:
    state = _certificate(PhysicalCause.PHYSICAL_STATE, belief=10, construction=11)
    discrepancy = _certificate(
        PhysicalCause.READOUT_DISCREPANCY,
        belief=12,
        construction=13,
    )
    pair = _pair(state, discrepancy, lower=-0.08, upper=-0.02)
    decision = _decision(state, discrepancy, pairs=(pair,))

    with pytest.raises(ValueError, match="selected belief does not match"):
        replace(decision, selected_belief_id=discrepancy.candidate_belief_id)
    with pytest.raises(ValueError, match="decision baseline differs"):
        replace(decision, baseline_belief_id=_id(99))

    baseline = PhysicalCauseAttributionDecisionV2(
        operational_decision_id=_id(20),
        baseline_belief_id=_id(1),
        selected_cause=PhysicalCause.BASELINE,
        selected_belief_id=_id(1),
        certificates=(),
        pairwise_certificates=(),
        minimum_improvement=0.0,
        maximum_harm_probability=1.0,
        maximum_stratum_regret=0.0,
    )
    with pytest.raises(ValueError, match="exact baseline belief"):
        replace(baseline, selected_belief_id=_id(99))


def test_decision_rejects_inconsistent_harm_definition() -> None:
    state = _certificate(PhysicalCause.PHYSICAL_STATE, belief=10, construction=11)
    discrepancy = _certificate(
        PhysicalCause.READOUT_DISCREPANCY,
        belief=12,
        construction=13,
        harm_margin=0.01,
    )
    with pytest.raises(ValueError, match="harm_margin differs"):
        _decision(state, discrepancy, pairs=())


def test_decision_requires_frozen_thresholds_and_no_target_use() -> None:
    decision = PhysicalCauseAttributionDecisionV2(
        operational_decision_id=_id(20),
        baseline_belief_id=_id(1),
        selected_cause=PhysicalCause.BASELINE,
        selected_belief_id=_id(1),
        certificates=(),
        pairwise_certificates=(),
        minimum_improvement=0.0,
        maximum_harm_probability=1.0,
        maximum_stratum_regret=0.0,
    )
    with pytest.raises(ValueError, match="must be true"):
        replace(decision, decision_thresholds_frozen_before_source_scores=False)
    with pytest.raises(ValueError, match="must be false"):
        replace(decision, target_outcomes_used=True)


def test_decision_rejects_duplicate_cause_and_invalid_thresholds() -> None:
    state = _certificate(PhysicalCause.PHYSICAL_STATE, belief=10, construction=11)
    duplicate = replace(state, candidate_belief_id=_id(40))
    with pytest.raises(ValueError, match="one regret certificate per cause"):
        PhysicalCauseAttributionDecisionV2(
            operational_decision_id=_id(20),
            baseline_belief_id=state.baseline_belief_id,
            selected_cause=PhysicalCause.PHYSICAL_STATE,
            selected_belief_id=state.candidate_belief_id,
            certificates=(state, duplicate),
            pairwise_certificates=(),
            minimum_improvement=0.0,
            maximum_harm_probability=1.0,
            maximum_stratum_regret=0.0,
        )
    with pytest.raises(ValueError, match="minimum_source_group_count"):
        PhysicalCauseAttributionDecisionV2(
            operational_decision_id=_id(20),
            baseline_belief_id=_id(1),
            selected_cause=PhysicalCause.BASELINE,
            selected_belief_id=_id(1),
            certificates=(),
            pairwise_certificates=(),
            minimum_improvement=0.0,
            maximum_harm_probability=1.0,
            maximum_stratum_regret=0.0,
            minimum_source_group_count=0,
        )
