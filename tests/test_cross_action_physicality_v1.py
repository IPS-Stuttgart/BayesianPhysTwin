from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from bayesian_phystwin_experiments.cross_action_physicality_v1 import (
    REQUIRED_PLACEBO_POLICIES,
    BrokenMechanismPolicy,
    CrossActionPhysicalityProtocolV1,
    CrossActionPhysicalityResultV1,
    PhysicalityDecision,
    PlaceboConstructionV1,
    PlaceboScoreRowV1,
    SealedPlaceboPredictionV1,
)
from bayesian_phystwin_experiments.cross_action_transport_contracts_v1 import (
    PredictionDisposition,
    TransportArm,
)
from bayesian_phystwin_experiments.cross_action_transport_v2 import (
    CAUSAL4D_SLOTH_MULTI_ACTION_V1_DESIGN_SHA256,
    ChronologicalSessionPairV2,
    CrossActionProtocolV2,
    CrossActionTransportResultV2,
    SealedTransportPredictionV2,
    SparseTransportDecision,
    TransportScoreRowV2,
)


def _id(seed: int) -> str:
    return f"{seed:064x}"


def _stable(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def _pairs(count: int) -> tuple[ChronologicalSessionPairV2, ...]:
    return tuple(
        ChronologicalSessionPairV2(
            object_session_id=f"s{index:02d}",
            source_execution_id=f"s{index:02d}-e1",
            target_execution_id=f"s{index:02d}-e2",
            source_action_id="a" if index % 2 == 0 else "b",
            target_action_id="b" if index % 2 == 0 else "a",
            contact_id=f"contact-{index % 3}",
            stratum_id=f"stratum-{index % 6}",
        )
        for index in range(count)
    )


def _parent_protocol(
    *,
    sessions: int = 14,
    minimum_sessions: int | None = None,
    minimum_accepted: int | None = None,
    maximum_harm: float = 0.20,
) -> CrossActionProtocolV2:
    return CrossActionProtocolV2(
        causal4d_design_id=CAUSAL4D_SLOTH_MULTI_ACTION_V1_DESIGN_SHA256,
        development_roster_id=_id(1),
        calibration_roster_id=_id(2),
        target_roster_id=_id(3),
        query_id=_id(4),
        query_jacobian_id=_id(5),
        score_definition_id=_id(6),
        grouping_rule_id=_id(7),
        interval_method_id=_id(8),
        harm_interval_method_id=_id(9),
        target_access_policy_id=_id(10),
        technical_failure_policy_id=_id(11),
        model_stack_id=_id(12),
        numerical_environment_id=_id(13),
        candidate_family_id=_id(14),
        support_policy_id=_id(15),
        identifiability_policy_id=_id(16),
        multi_action_identifiability_policy_id=_id(17),
        estimability_policy_id=_id(18),
        guard_policy_id=_id(19),
        session_pairs=_pairs(sessions),
        registered_arms=(
            TransportArm.GUARDED_PHYSICAL,
            TransportArm.DISCREPANCY_ONLY,
            TransportArm.PHYSICAL_FALLBACK,
            TransportArm.LAST_RESIDUAL,
        ),
        physical_transport_arm=TransportArm.GUARDED_PHYSICAL,
        discrepancy_reference_arm=TransportArm.DISCREPANCY_ONLY,
        matched_comparator_arm=TransportArm.LAST_RESIDUAL,
        minimum_sessions=minimum_sessions or sessions,
        minimum_accepted_physical_sessions=minimum_accepted or sessions,
        bootstrap_replicates=500,
        bootstrap_seed=23,
        confidence_level=0.95,
        minimum_transport_gain=1.0,
        minimum_discrepancy_contrast=0.5,
        minimum_comparator_contrast=0.25,
        maximum_harmful_accepted_fraction=maximum_harm,
    )


def _parent_prediction(
    protocol: CrossActionProtocolV2,
    pair: ChronologicalSessionPairV2,
    arm: TransportArm,
    *,
    disposition: PredictionDisposition | None = None,
) -> SealedTransportPredictionV2:
    baseline = _stable("base", pair.object_session_id)
    if arm is TransportArm.PHYSICAL_FALLBACK:
        candidate = None
        selected = baseline
        resolved = PredictionDisposition.BASELINE_REFERENCE
    else:
        candidate = _stable("candidate", pair.object_session_id, arm.value)
        resolved = disposition or PredictionDisposition.CANDIDATE_SELECTED
        selected = (
            baseline
            if resolved is PredictionDisposition.EXACT_FALLBACK
            else candidate
        )
    return SealedTransportPredictionV2(
        protocol_id=protocol.protocol_id,
        information_order_id=pair.information_order_id,
        object_session_id=pair.object_session_id,
        source_execution_id=pair.source_execution_id,
        target_execution_id=pair.target_execution_id,
        source_action_id=pair.source_action_id,
        target_action_id=pair.target_action_id,
        arm=arm,
        baseline_belief_id=baseline,
        candidate_belief_id=candidate,
        selected_belief_id=selected,
        disposition=resolved,
        prediction_artifact_id=_stable(
            "prediction", pair.object_session_id, arm.value
        ),
        source_evidence_id=_stable(
            "source-evidence", pair.object_session_id
        ),
        admission_evidence_id=_id(31),
        prediction_batch_id=_id(40),
        commit_id="a" * 40,
        prediction_sealed_before_target=True,
    )


def _parent_rows(
    protocol: CrossActionProtocolV2,
    *,
    physical_gain: float = 4.0,
    physical_fallback_session: int | None = None,
) -> tuple[TransportScoreRowV2, ...]:
    rows = []
    for index, pair in enumerate(protocol.session_pairs):
        for arm in protocol.registered_arms:
            disposition = None
            gain = 0.0
            if arm is TransportArm.GUARDED_PHYSICAL:
                if index == physical_fallback_session:
                    disposition = PredictionDisposition.EXACT_FALLBACK
                else:
                    gain = physical_gain
            elif arm is TransportArm.DISCREPANCY_ONLY:
                gain = 1.0
            elif arm is TransportArm.LAST_RESIDUAL:
                gain = 1.5
            prediction = _parent_prediction(
                protocol,
                pair,
                arm,
                disposition=disposition,
            )
            rows.append(
                TransportScoreRowV2(
                    prediction=prediction,
                    target_outcome_id=_stable("target", pair.object_session_id),
                    target_access_attestation_id=_id(50),
                    scorer_id=_id(51),
                    proper_score=10.0 - gain,
                )
            )
    return tuple(rows)


def _parent_result(
    protocol: CrossActionProtocolV2,
    *,
    physical_gain: float = 4.0,
    physical_fallback_session: int | None = None,
    technical: tuple[str, ...] = (),
) -> CrossActionTransportResultV2:
    rows = tuple(
        row
        for row in _parent_rows(
            protocol,
            physical_gain=physical_gain,
            physical_fallback_session=physical_fallback_session,
        )
        if row.prediction.object_session_id not in technical
    )
    return CrossActionTransportResultV2(
        protocol=protocol,
        score_rows=rows,
        target_accounting_id=_id(52),
        technical_failure_session_ids=technical,
    )


def _physicality_protocol(
    parent: CrossActionTransportResultV2,
    *,
    minimum_sessions: int | None = None,
) -> CrossActionPhysicalityProtocolV1:
    return CrossActionPhysicalityProtocolV1(
        parent_protocol=parent.protocol,
        wrong_source_action_policy_id=_id(61),
        wrong_object_session_policy_id=_id(62),
        phase_shifted_source_policy_id=_id(63),
        identity_permuted_policy_id=_id(64),
        prediction_batch_id=_id(40),
        commit_id="a" * 40,
        scorer_id=_id(51),
        minimum_sessions=minimum_sessions or len(parent.protocol.session_pairs),
        bootstrap_replicates=500,
        bootstrap_seed=71,
        confidence_level=0.95,
        minimum_placebo_separation_margin=0.25,
    )


def _parent_physical_rows(
    parent: CrossActionTransportResultV2,
) -> tuple[TransportScoreRowV2, ...]:
    return tuple(
        row
        for row in parent.score_rows
        if row.prediction.arm is parent.protocol.physical_transport_arm
    )


def _donor_pair(
    protocol: CrossActionPhysicalityProtocolV1,
    pair: ChronologicalSessionPairV2,
    *,
    preserve_action: bool,
) -> ChronologicalSessionPairV2:
    pairs = protocol.parent_protocol.session_pairs
    index = pairs.index(pair)
    for offset in range(1, len(pairs)):
        candidate = pairs[(index + offset) % len(pairs)]
        same_action = candidate.source_action_id == pair.source_action_id
        if same_action == preserve_action:
            return candidate
    raise AssertionError("the frozen test roster lacks a valid donor")


def _construction(
    protocol: CrossActionPhysicalityProtocolV1,
    parent_row: TransportScoreRowV2,
    policy: BrokenMechanismPolicy,
) -> PlaceboConstructionV1:
    pair = protocol.parent_protocol.pair_by_session[
        parent_row.prediction.object_session_id
    ]
    policy_fields: dict[str, object] = {}
    evidence_session_id = pair.object_session_id
    if policy is BrokenMechanismPolicy.WRONG_SOURCE_ACTION:
        donor = _donor_pair(protocol, pair, preserve_action=False)
        evidence_session_id = donor.object_session_id
        policy_fields.update(
            donor_object_session_id=donor.object_session_id,
            donor_source_execution_id=donor.source_execution_id,
            donor_action_id=donor.source_action_id,
        )
    elif policy is BrokenMechanismPolicy.WRONG_OBJECT_SESSION:
        donor = _donor_pair(protocol, pair, preserve_action=True)
        evidence_session_id = donor.object_session_id
        policy_fields.update(
            donor_object_session_id=donor.object_session_id,
            donor_source_execution_id=donor.source_execution_id,
            donor_action_id=donor.source_action_id,
        )
    elif policy is BrokenMechanismPolicy.PHASE_SHIFTED_SOURCE:
        policy_fields.update(phase_shift_steps=2, phase_period_steps=7)
    else:
        policy_fields.update(
            identity_permutation_id=_stable(
                "permutation", pair.object_session_id
            ),
            identity_permutation_size=10,
            identity_permutation_moved_count=8,
        )
    return PlaceboConstructionV1(
        protocol_id=protocol.protocol_id,
        information_order_id=pair.information_order_id,
        object_session_id=pair.object_session_id,
        source_execution_id=pair.source_execution_id,
        target_execution_id=pair.target_execution_id,
        source_action_id=pair.source_action_id,
        target_action_id=pair.target_action_id,
        policy=policy,
        policy_implementation_id=protocol.policy_implementation_id(policy),
        parent_prediction_id=parent_row.prediction.prediction_id,
        parent_selected_belief_id=parent_row.prediction.selected_belief_id,
        source_evidence_id=_stable(
            "source-evidence", evidence_session_id
        ),
        construction_artifact_id=_stable(
            "construction", pair.object_session_id, policy.value
        ),
        **policy_fields,
    )


def _placebo_rows(
    protocol: CrossActionPhysicalityProtocolV1,
    parent: CrossActionTransportResultV2,
    *,
    contrasts: dict[BrokenMechanismPolicy, float] | None = None,
) -> tuple[PlaceboScoreRowV1, ...]:
    if contrasts is None:
        contrasts = {policy: 2.0 for policy in REQUIRED_PLACEBO_POLICIES}
    rows = []
    for parent_row in _parent_physical_rows(parent):
        for policy in REQUIRED_PLACEBO_POLICIES:
            construction = _construction(protocol, parent_row, policy)
            candidate = _stable("placebo-candidate", construction.construction_id)
            inherited_fallback = (
                parent_row.prediction.disposition
                is PredictionDisposition.EXACT_FALLBACK
            )
            prediction = SealedPlaceboPredictionV1(
                construction=construction,
                baseline_belief_id=parent_row.prediction.baseline_belief_id,
                candidate_belief_id=candidate,
                selected_belief_id=(
                    parent_row.prediction.selected_belief_id
                    if inherited_fallback
                    else candidate
                ),
                disposition=(
                    PredictionDisposition.EXACT_FALLBACK
                    if inherited_fallback
                    else PredictionDisposition.CANDIDATE_SELECTED
                ),
                prediction_artifact_id=(
                    parent_row.prediction.prediction_artifact_id
                    if inherited_fallback
                    else _stable(
                        "placebo-prediction", construction.construction_id
                    )
                ),
                prediction_batch_id=protocol.prediction_batch_id,
                commit_id=protocol.commit_id,
                prediction_sealed_before_target=True,
            )
            rows.append(
                PlaceboScoreRowV1(
                    prediction=prediction,
                    target_outcome_id=parent_row.target_outcome_id,
                    target_access_attestation_id=(
                        parent_row.target_access_attestation_id
                    ),
                    scorer_id=parent_row.scorer_id,
                    proper_score=(
                        parent_row.proper_score
                        if inherited_fallback
                        else parent_row.proper_score + contrasts[policy]
                    ),
                )
            )
    return tuple(rows)


def test_positive_physicality_and_input_order_invariance() -> None:
    parent = _parent_result(_parent_protocol())
    protocol = _physicality_protocol(parent)
    rows = _placebo_rows(protocol, parent)
    forward = CrossActionPhysicalityResultV1(protocol, parent, rows)
    reverse = CrossActionPhysicalityResultV1(
        protocol,
        parent,
        tuple(reversed(rows)),
    )
    assert parent.decision is SparseTransportDecision.SUPPORTED
    assert forward.decision is PhysicalityDecision.SUPPORTED
    assert forward.supports_physicality
    assert forward.result_id == reverse.result_id
    assert all(
        summary.simultaneous_lower_bound > 0.25
        for summary in forward.placebo_summaries
    )


def test_one_failed_placebo_makes_the_familywise_result_negative() -> None:
    parent = _parent_result(_parent_protocol())
    protocol = _physicality_protocol(parent)
    contrasts = {policy: 2.0 for policy in REQUIRED_PLACEBO_POLICIES}
    contrasts[BrokenMechanismPolicy.IDENTITY_PERMUTED] = 0.0
    result = CrossActionPhysicalityResultV1(
        protocol,
        parent,
        _placebo_rows(protocol, parent, contrasts=contrasts),
    )
    assert result.decision is PhysicalityDecision.NOT_SUPPORTED


def test_parent_failure_and_insufficiency_cannot_be_rescued() -> None:
    negative_parent = _parent_result(
        _parent_protocol(),
        physical_gain=0.5,
    )
    negative_protocol = _physicality_protocol(negative_parent)
    negative = CrossActionPhysicalityResultV1(
        negative_protocol,
        negative_parent,
        (),
    )
    assert negative_parent.decision is SparseTransportDecision.NOT_SUPPORTED
    assert negative.decision is PhysicalityDecision.PARENT_NOT_SUPPORTED

    insufficient_parent = _parent_result(
        _parent_protocol(),
        technical=("s13",),
    )
    insufficient_protocol = _physicality_protocol(insufficient_parent)
    insufficient = CrossActionPhysicalityResultV1(
        insufficient_protocol,
        insufficient_parent,
        (),
    )
    assert (
        insufficient_parent.decision
        is SparseTransportDecision.INSUFFICIENT_SESSIONS
    )
    assert insufficient.decision is PhysicalityDecision.INSUFFICIENT


def test_complete_four_policy_table_is_required() -> None:
    parent = _parent_result(_parent_protocol())
    protocol = _physicality_protocol(parent)
    rows = tuple(
        row
        for row in _placebo_rows(protocol, parent)
        if not (
            row.prediction.construction.object_session_id == "s00"
            and row.prediction.construction.policy
            is BrokenMechanismPolicy.IDENTITY_PERMUTED
        )
    )
    with pytest.raises(ValueError, match="all four placebos"):
        CrossActionPhysicalityResultV1(protocol, parent, rows)


def test_parent_fallback_is_inherited_with_zero_contrast() -> None:
    parent = _parent_result(
        _parent_protocol(minimum_accepted=13, maximum_harm=0.25),
        physical_fallback_session=0,
    )
    protocol = _physicality_protocol(parent)
    rows = _placebo_rows(protocol, parent)
    result = CrossActionPhysicalityResultV1(protocol, parent, rows)
    assert parent.decision is SparseTransportDecision.SUPPORTED
    assert result.inherited_fallback_session_count == 1
    assert result.accepted_physical_session_count == 13
    assert all(
        summary.inherited_fallback_sessions == 1
        for summary in result.placebo_summaries
    )

    bad_rows = list(rows)
    fallback_index = next(
        index
        for index, row in enumerate(bad_rows)
        if row.prediction.construction.object_session_id == "s00"
    )
    bad_rows[fallback_index] = replace(
        bad_rows[fallback_index],
        proper_score=bad_rows[fallback_index].proper_score + 0.1,
    )
    with pytest.raises(ValueError, match="identical score"):
        CrossActionPhysicalityResultV1(
            protocol,
            parent,
            tuple(bad_rows),
        )

    bad_rows = list(rows)
    fallback_index = next(
        index
        for index, row in enumerate(bad_rows)
        if row.prediction.construction.object_session_id == "s00"
    )
    fallback_row = bad_rows[fallback_index]
    bad_rows[fallback_index] = replace(
        fallback_row,
        prediction=replace(
            fallback_row.prediction,
            prediction_artifact_id=_id(994),
        ),
    )
    with pytest.raises(ValueError, match="reuse the parent artifact"):
        CrossActionPhysicalityResultV1(
            protocol,
            parent,
            tuple(bad_rows),
        )


def test_parent_acceptance_requires_a_selected_placebo_candidate() -> None:
    parent = _parent_result(_parent_protocol())
    protocol = _physicality_protocol(parent)
    rows = list(_placebo_rows(protocol, parent))
    row = rows[0]
    rows[0] = replace(
        row,
        prediction=replace(
            row.prediction,
            disposition=PredictionDisposition.EXACT_FALLBACK,
            selected_belief_id=row.prediction.baseline_belief_id,
        ),
    )
    with pytest.raises(ValueError, match="complete selected placebo candidate"):
        CrossActionPhysicalityResultV1(protocol, parent, tuple(rows))


def test_policy_specific_constructions_fail_closed() -> None:
    parent = _parent_result(_parent_protocol())
    protocol = _physicality_protocol(parent)
    parent_row = _parent_physical_rows(parent)[0]

    shifted = _construction(
        protocol,
        parent_row,
        BrokenMechanismPolicy.PHASE_SHIFTED_SOURCE,
    )
    with pytest.raises(ValueError, match="nontrivial modulo"):
        replace(shifted, phase_shift_steps=7)
    with pytest.raises(ValueError, match="sealed before target"):
        replace(shifted, target_outcomes_used=True)

    permuted = _construction(
        protocol,
        parent_row,
        BrokenMechanismPolicy.IDENTITY_PERMUTED,
    )
    with pytest.raises(ValueError, match="cannot exceed"):
        replace(permuted, identity_permutation_moved_count=11)

    wrong_session = _construction(
        protocol,
        parent_row,
        BrokenMechanismPolicy.WRONG_OBJECT_SESSION,
    )
    with pytest.raises(ValueError, match="another physical session"):
        replace(
            wrong_session,
            donor_object_session_id=wrong_session.object_session_id,
        )


def test_donor_semantics_and_lineage_are_verified_against_the_roster() -> None:
    parent = _parent_result(_parent_protocol())
    protocol = _physicality_protocol(parent)
    rows = list(_placebo_rows(protocol, parent))
    index = next(
        index
        for index, row in enumerate(rows)
        if row.prediction.construction.policy
        is BrokenMechanismPolicy.WRONG_OBJECT_SESSION
    )
    row = rows[index]
    construction = row.prediction.construction
    wrong_action_donor = _donor_pair(
        protocol,
        protocol.parent_protocol.pair_by_session[construction.object_session_id],
        preserve_action=False,
    )
    rows[index] = replace(
        row,
        prediction=replace(
            row.prediction,
            construction=replace(
                construction,
                donor_object_session_id=wrong_action_donor.object_session_id,
                donor_source_execution_id=wrong_action_donor.source_execution_id,
                donor_action_id=wrong_action_donor.source_action_id,
            ),
        ),
    )
    with pytest.raises(ValueError, match="preserve the source action profile"):
        CrossActionPhysicalityResultV1(protocol, parent, tuple(rows))


def test_policy_batch_scorer_and_target_lineage_mismatches_fail_closed() -> None:
    parent = _parent_result(_parent_protocol())
    protocol = _physicality_protocol(parent)
    rows = list(_placebo_rows(protocol, parent))
    row = rows[0]
    rows[0] = replace(
        row,
        prediction=replace(
            row.prediction,
            construction=replace(
                row.prediction.construction,
                policy_implementation_id=_id(999),
            ),
        ),
    )
    with pytest.raises(ValueError, match="implementation identity"):
        CrossActionPhysicalityResultV1(protocol, parent, tuple(rows))

    rows = list(_placebo_rows(protocol, parent))
    rows[0] = replace(rows[0], scorer_id=_id(998))
    with pytest.raises(ValueError, match="scorer"):
        CrossActionPhysicalityResultV1(protocol, parent, tuple(rows))

    rows = list(_placebo_rows(protocol, parent))
    row = rows[0]
    rows[0] = replace(
        row,
        prediction=replace(row.prediction, prediction_batch_id=_id(997)),
    )
    with pytest.raises(ValueError, match="prediction batch"):
        CrossActionPhysicalityResultV1(protocol, parent, tuple(rows))

    rows = list(_placebo_rows(protocol, parent))
    rows[0] = replace(rows[0], target_outcome_id=_id(996))
    with pytest.raises(ValueError, match="same target outcome"):
        CrossActionPhysicalityResultV1(protocol, parent, tuple(rows))


def test_protocol_rejects_ambiguous_or_unconstructable_placebos() -> None:
    parent = _parent_result(_parent_protocol())
    protocol = _physicality_protocol(parent)
    with pytest.raises(ValueError, match="distinct implementation"):
        replace(
            protocol,
            wrong_object_session_policy_id=(
                protocol.wrong_source_action_policy_id
            ),
        )

    two_session_parent = _parent_result(
        _parent_protocol(sessions=2, maximum_harm=1.0)
    )
    with pytest.raises(ValueError, match="two sessions per source action"):
        _physicality_protocol(two_session_parent)


def test_source_evidence_and_construction_artifacts_are_bound() -> None:
    parent = _parent_result(_parent_protocol())
    protocol = _physicality_protocol(parent)
    rows = list(_placebo_rows(protocol, parent))
    row = rows[0]
    rows[0] = replace(
        row,
        prediction=replace(
            row.prediction,
            construction=replace(
                row.prediction.construction,
                source_evidence_id=_id(995),
            ),
        ),
    )
    with pytest.raises(ValueError, match="registered source lineage"):
        CrossActionPhysicalityResultV1(protocol, parent, tuple(rows))

    rows = list(_placebo_rows(protocol, parent))
    first = rows[0]
    second = rows[1]
    rows[1] = replace(
        second,
        prediction=replace(
            second.prediction,
            construction=replace(
                second.prediction.construction,
                construction_artifact_id=(
                    first.prediction.construction.construction_artifact_id
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="construction artifacts"):
        CrossActionPhysicalityResultV1(protocol, parent, tuple(rows))


def test_checked_in_roster_supports_both_donor_controls() -> None:
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "protocols"
        / "cross_action_transport"
        / "causal4d_sloth_multi_action_v1_sparse_pairs.json"
    )
    payload = json.loads(path.read_text())
    sessions = payload["session_pairs"]
    source_actions = [entry["source_action_id"] for entry in sessions]
    assert len(set(source_actions)) >= 2
    assert all(source_actions.count(action) >= 2 for action in source_actions)
    for entry in sessions:
        assert any(
            candidate["object_session_id"] != entry["object_session_id"]
            and candidate["source_action_id"] == entry["source_action_id"]
            for candidate in sessions
        )
        assert any(
            candidate["source_action_id"] != entry["source_action_id"]
            for candidate in sessions
        )
