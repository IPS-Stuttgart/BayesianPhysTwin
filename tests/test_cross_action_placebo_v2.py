from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin_experiments.cross_action_placebo_v2 import (
    ALL_PLACEBO_CERTIFICATE_ARMS_V2,
    CROSS_ACTION_PLACEBO_V2_FAMILYWISE_METHOD,
    PLACEBO_ARMS_V2,
    ChronologicalPlaceboConstructionV2,
    CrossActionPlaceboProtocolV2,
    CrossActionPlaceboResultV2,
    CrossActionPlaceboScoreRowV2,
    PlaceboArmV2,
    PlaceboDecisionV2,
    SealedCrossActionPlaceboPredictionV2,
)
from bayesian_phystwin_experiments.cross_action_transport_v2 import (
    CAUSAL4D_SLOTH_MULTI_ACTION_V1_DESIGN_SHA256,
    ChronologicalSessionPairV2,
    CrossActionProtocolV2,
    CrossActionTransportResultV2,
    PredictionDisposition,
    SealedTransportPredictionV2,
    SparseTransportDecision,
    TransportArm,
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
        maximum_harmful_accepted_fraction=0.20,
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
            baseline if resolved is PredictionDisposition.EXACT_FALLBACK else candidate
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
        prediction_artifact_id=_stable("prediction", pair.object_session_id, arm.value),
        source_evidence_id=_id(30),
        admission_evidence_id=_id(31),
        prediction_batch_id=_id(40),
        commit_id="a" * 40,
        prediction_sealed_before_target=True,
    )


def _parent_rows(
    protocol: CrossActionProtocolV2,
    *,
    physical_gain: float = 4.0,
    physical_dispositions: tuple[PredictionDisposition, ...] | None = None,
    physical_gains: tuple[float, ...] | None = None,
) -> tuple[TransportScoreRowV2, ...]:
    if physical_dispositions is None:
        physical_dispositions = tuple(
            PredictionDisposition.CANDIDATE_SELECTED for _ in protocol.session_pairs
        )
    if physical_gains is None:
        physical_gains = tuple(physical_gain for _ in protocol.session_pairs)
    rows: list[TransportScoreRowV2] = []
    for index, pair in enumerate(protocol.session_pairs):
        for arm in protocol.registered_arms:
            disposition = None
            gain = 0.0
            if arm is TransportArm.GUARDED_PHYSICAL:
                disposition = physical_dispositions[index]
                gain = physical_gains[index]
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
    physical_dispositions: tuple[PredictionDisposition, ...] | None = None,
    physical_gains: tuple[float, ...] | None = None,
    technical: tuple[str, ...] = (),
) -> CrossActionTransportResultV2:
    rows = _parent_rows(
        protocol,
        physical_gain=physical_gain,
        physical_dispositions=physical_dispositions,
        physical_gains=physical_gains,
    )
    if technical:
        rows = tuple(
            row
            for row in rows
            if row.prediction.object_session_id not in set(technical)
        )
    return CrossActionTransportResultV2(
        protocol=protocol,
        score_rows=rows,
        target_accounting_id=_id(52),
        technical_failure_session_ids=technical,
    )


def _policy_ids() -> dict[str, str]:
    return {arm.value: _stable("policy", arm.value) for arm in PLACEBO_ARMS_V2}


def _construction(
    protocol: CrossActionProtocolV2,
    pair: ChronologicalSessionPairV2,
    arm: PlaceboArmV2,
    *,
    index: int,
    policy_ids: dict[str, str],
) -> ChronologicalPlaceboConstructionV2:
    kwargs: dict[str, object] = {}
    if arm is PlaceboArmV2.WRONG_SOURCE_ACTION:
        donor = next(
            candidate
            for candidate in protocol.session_pairs
            if candidate.source_action_id != pair.source_action_id
        )
        kwargs.update(
            donor_object_session_id=donor.object_session_id,
            donor_source_execution_id=donor.source_execution_id,
            donor_source_action_id=donor.source_action_id,
        )
    elif arm is PlaceboArmV2.WRONG_OBJECT_SESSION:
        donor = next(
            candidate
            for candidate in protocol.session_pairs
            if candidate.object_session_id != pair.object_session_id
            and candidate.source_action_id == pair.source_action_id
        )
        kwargs.update(
            donor_object_session_id=donor.object_session_id,
            donor_source_execution_id=donor.source_execution_id,
            donor_source_action_id=donor.source_action_id,
        )
    elif arm is PlaceboArmV2.PHASE_SHIFTED_SOURCE:
        kwargs["phase_shift_steps"] = 2
    else:
        kwargs.update(
            permutation_artifact_id=_stable("permutation", pair.object_session_id),
            permutation_size=8,
            permutation_fixed_point_count=0,
        )
    return ChronologicalPlaceboConstructionV2(
        object_session_id=pair.object_session_id,
        information_order_id=pair.information_order_id,
        source_execution_id=pair.source_execution_id,
        target_execution_id=pair.target_execution_id,
        source_action_id=pair.source_action_id,
        target_action_id=pair.target_action_id,
        arm=arm,
        policy_id=policy_ids[arm.value],
        source_prefix_artifact_id=_stable("prefix", pair.object_session_id),
        construction_artifact_id=_stable(
            "construction",
            pair.object_session_id,
            arm.value,
        ),
        constructor_commit_id="a" * 40,
        **kwargs,
    )


def _placebo_protocol(
    parent_protocol: CrossActionProtocolV2,
) -> CrossActionPlaceboProtocolV2:
    policies = _policy_ids()
    constructions = tuple(
        _construction(
            parent_protocol,
            pair,
            arm,
            index=index,
            policy_ids=policies,
        )
        for index, pair in enumerate(parent_protocol.session_pairs)
        for arm in PLACEBO_ARMS_V2
    )
    return CrossActionPlaceboProtocolV2(
        parent_transport_protocol=parent_protocol,
        placebo_policy_ids=policies,
        constructions=constructions,
        minimum_sessions=parent_protocol.minimum_sessions,
        bootstrap_replicates=1000,
        bootstrap_seed=20260828,
        familywise_confidence_level=0.95,
        minimum_placebo_separation=0.1,
    )


def _placebo_rows(
    protocol: CrossActionPlaceboProtocolV2,
    parent_result: CrossActionTransportResultV2,
    *,
    placebo_offsets: dict[PlaceboArmV2, float] | None = None,
) -> tuple[CrossActionPlaceboScoreRowV2, ...]:
    offsets = placebo_offsets or {arm: 1.0 for arm in PLACEBO_ARMS_V2}
    construction_by_key = protocol.construction_by_key
    parent_rows = {
        row.prediction.object_session_id: row
        for row in parent_result.score_rows
        if row.prediction.arm is TransportArm.GUARDED_PHYSICAL
    }
    rows: list[CrossActionPlaceboScoreRowV2] = []
    for pair in protocol.parent_transport_protocol.session_pairs:
        parent_row = parent_rows.get(pair.object_session_id)
        if parent_row is None:
            continue
        fallback = (
            parent_row.prediction.disposition is PredictionDisposition.EXACT_FALLBACK
        )
        for arm in ALL_PLACEBO_CERTIFICATE_ARMS_V2:
            if arm is PlaceboArmV2.GUARDED_PHYSICAL:
                construction_id = None
                artifact_id = parent_row.prediction.prediction_artifact_id
                score = parent_row.proper_score
            else:
                construction_id = construction_by_key[
                    (pair.object_session_id, arm)
                ].construction_id
                artifact_id = (
                    parent_row.prediction.prediction_artifact_id
                    if fallback
                    else _stable("placebo", pair.object_session_id, arm.value)
                )
                score = (
                    parent_row.proper_score
                    if fallback
                    else parent_row.proper_score + offsets[arm]
                )
            prediction = SealedCrossActionPlaceboPredictionV2(
                protocol_id=protocol.protocol_id,
                parent_transport_prediction_id=(parent_row.prediction.prediction_id),
                information_order_id=pair.information_order_id,
                object_session_id=pair.object_session_id,
                source_execution_id=pair.source_execution_id,
                target_execution_id=pair.target_execution_id,
                source_action_id=pair.source_action_id,
                target_action_id=pair.target_action_id,
                arm=arm,
                construction_id=construction_id,
                prediction_artifact_id=artifact_id,
                prediction_batch_id=_id(60),
                commit_id=parent_row.prediction.commit_id,
                disposition=parent_row.prediction.disposition,
                prediction_sealed_before_target=True,
            )
            rows.append(
                CrossActionPlaceboScoreRowV2(
                    prediction=prediction,
                    target_outcome_id=parent_row.target_outcome_id,
                    target_access_attestation_id=(
                        parent_row.target_access_attestation_id
                    ),
                    scorer_id=parent_row.scorer_id,
                    proper_score=score,
                )
            )
    return tuple(rows)


def test_positive_certificate_uses_joint_familywise_bounds() -> None:
    parent_protocol = _parent_protocol()
    parent_result = _parent_result(parent_protocol)
    protocol = _placebo_protocol(parent_protocol)
    rows = _placebo_rows(protocol, parent_result)

    forward = CrossActionPlaceboResultV2(protocol, parent_result, rows)
    reverse = CrossActionPlaceboResultV2(
        protocol,
        parent_result,
        tuple(reversed(rows)),
    )

    assert forward.decision is PlaceboDecisionV2.SUPPORTED
    assert forward.supports_physicality
    assert forward.selected_physical_session_count == 14
    assert forward.familywise_critical_value == 0.0
    assert all(
        summary.simultaneous_lower_bound == 1.0
        for summary in forward.contrast_summaries
    )
    assert not forward.session_placebo_contrasts.flags.writeable
    assert forward.result_id == reverse.result_id
    assert forward.to_record()["result_id"] == forward.result_id


def test_one_tied_placebo_blocks_the_conjunctive_claim() -> None:
    parent_protocol = _parent_protocol()
    parent_result = _parent_result(parent_protocol)
    protocol = _placebo_protocol(parent_protocol)
    offsets = {arm: 1.0 for arm in PLACEBO_ARMS_V2}
    offsets[PlaceboArmV2.WRONG_SOURCE_ACTION] = 0.0
    result = CrossActionPlaceboResultV2(
        protocol,
        parent_result,
        _placebo_rows(protocol, parent_result, placebo_offsets=offsets),
    )

    assert result.decision is PlaceboDecisionV2.NOT_SUPPORTED
    summary = next(
        value
        for value in result.contrast_summaries
        if value.arm is PlaceboArmV2.WRONG_SOURCE_ACTION
    )
    assert summary.mean_contrast == 0.0
    assert summary.simultaneous_lower_bound == 0.0


def test_parent_failure_or_insufficiency_cannot_be_rescued() -> None:
    parent_protocol = _parent_protocol()
    negative_parent = _parent_result(parent_protocol, physical_gain=0.5)
    assert negative_parent.decision is SparseTransportDecision.NOT_SUPPORTED
    protocol = _placebo_protocol(parent_protocol)
    negative_result = CrossActionPlaceboResultV2(
        protocol,
        negative_parent,
        _placebo_rows(protocol, negative_parent),
    )
    assert negative_result.decision is PlaceboDecisionV2.PARENT_NOT_SUPPORTED

    technical_parent = _parent_result(parent_protocol, technical=("s13",))
    assert technical_parent.decision is SparseTransportDecision.INSUFFICIENT_SESSIONS
    insufficient_result = CrossActionPlaceboResultV2(
        protocol,
        technical_parent,
        _placebo_rows(protocol, technical_parent),
    )
    assert insufficient_result.decision is PlaceboDecisionV2.PARENT_INSUFFICIENT


def test_exact_fallback_is_identical_and_has_zero_contrast() -> None:
    parent_protocol = _parent_protocol(sessions=18, minimum_accepted=14)
    dispositions = (
        PredictionDisposition.EXACT_FALLBACK,
        *(PredictionDisposition.CANDIDATE_SELECTED for _ in range(17)),
    )
    gains = (0.0, *(4.0 for _ in range(17)))
    parent_result = _parent_result(
        parent_protocol,
        physical_dispositions=dispositions,
        physical_gains=gains,
    )
    assert parent_result.decision is SparseTransportDecision.SUPPORTED
    protocol = _placebo_protocol(parent_protocol)
    rows = list(_placebo_rows(protocol, parent_result))
    result = CrossActionPlaceboResultV2(protocol, parent_result, rows)
    assert np.all(result.session_placebo_contrasts[0] == 0.0)

    index = next(
        index
        for index, row in enumerate(rows)
        if row.prediction.object_session_id == "s00"
        and row.prediction.arm is PlaceboArmV2.WRONG_SOURCE_ACTION
    )
    changed_prediction = replace(
        rows[index].prediction,
        prediction_artifact_id=_id(999),
    )
    with pytest.raises(ValueError, match="artifact- and score-identical"):
        CrossActionPlaceboResultV2(
            protocol,
            parent_result,
            [
                *rows[:index],
                replace(rows[index], prediction=changed_prediction),
                *rows[index + 1 :],
            ],
        )

    changed_score = replace(rows[index], proper_score=rows[index].proper_score + 1.0)
    with pytest.raises(ValueError, match="artifact- and score-identical"):
        CrossActionPlaceboResultV2(
            protocol,
            parent_result,
            [*rows[:index], changed_score, *rows[index + 1 :]],
        )


def test_construction_roster_chronology_and_donor_are_fail_closed() -> None:
    parent_protocol = _parent_protocol()
    protocol = _placebo_protocol(parent_protocol)
    with pytest.raises(ValueError, match="every frozen session"):
        replace(protocol, constructions=protocol.constructions[:-1])

    first = protocol.constructions[0]
    reversed_construction = replace(
        first,
        source_execution_id=first.target_execution_id,
        target_execution_id=first.source_execution_id,
        source_action_id=first.target_action_id,
        target_action_id=first.source_action_id,
    )
    with pytest.raises(ValueError, match="registered chronology"):
        replace(
            protocol,
            constructions=(reversed_construction, *protocol.constructions[1:]),
        )

    donor_index = next(
        index
        for index, construction in enumerate(protocol.constructions)
        if construction.arm is PlaceboArmV2.WRONG_SOURCE_ACTION
    )
    donor_construction = protocol.constructions[donor_index]
    wrong_donor = replace(
        donor_construction,
        donor_source_execution_id="not-the-donor-source",
    )
    changed_constructions = list(protocol.constructions)
    changed_constructions[donor_index] = wrong_donor
    with pytest.raises(ValueError, match="registered source execution"):
        replace(protocol, constructions=tuple(changed_constructions))


def test_prediction_parent_construction_and_scoring_drift_are_rejected() -> None:
    parent_protocol = _parent_protocol()
    parent_result = _parent_result(parent_protocol)
    protocol = _placebo_protocol(parent_protocol)
    rows = list(_placebo_rows(protocol, parent_result))
    placebo_index = next(
        index
        for index, row in enumerate(rows)
        if row.prediction.arm is PlaceboArmV2.WRONG_OBJECT_SESSION
    )

    changed = replace(rows[placebo_index].prediction, construction_id=_id(999))
    with pytest.raises(ValueError, match="frozen construction"):
        CrossActionPlaceboResultV2(
            protocol,
            parent_result,
            [
                *rows[:placebo_index],
                replace(rows[placebo_index], prediction=changed),
                *rows[placebo_index + 1 :],
            ],
        )

    changed = replace(
        rows[placebo_index].prediction,
        parent_transport_prediction_id=_id(998),
    )
    with pytest.raises(ValueError, match="exact guarded parent prediction"):
        CrossActionPlaceboResultV2(
            protocol,
            parent_result,
            [
                *rows[:placebo_index],
                replace(rows[placebo_index], prediction=changed),
                *rows[placebo_index + 1 :],
            ],
        )

    with pytest.raises(ValueError, match="one frozen scorer"):
        CrossActionPlaceboResultV2(
            protocol,
            parent_result,
            [
                *rows[:placebo_index],
                replace(rows[placebo_index], scorer_id=_id(997)),
                *rows[placebo_index + 1 :],
            ],
        )


def test_protocol_requires_all_four_target_free_policies() -> None:
    parent_protocol = _parent_protocol()
    protocol = _placebo_protocol(parent_protocol)
    with pytest.raises(ValueError, match="exactly the four registered controls"):
        replace(
            protocol,
            placebo_policy_ids={
                PlaceboArmV2.WRONG_SOURCE_ACTION.value: _id(1),
            },
        )
    with pytest.raises(ValueError, match="target-selection free"):
        replace(protocol, target_outcomes_used_for_selection=True)
    with pytest.raises(ValueError, match="paired-session bootstrap"):
        replace(protocol, familywise_method="marginal-bootstrap")
    assert protocol.familywise_method == CROSS_ACTION_PLACEBO_V2_FAMILYWISE_METHOD


def test_broken_mechanism_parameters_must_actually_break_identity() -> None:
    parent_protocol = _parent_protocol()
    pair = parent_protocol.session_pairs[0]
    policies = _policy_ids()

    with pytest.raises(ValueError, match="nonzero literal integer"):
        ChronologicalPlaceboConstructionV2(
            object_session_id=pair.object_session_id,
            information_order_id=pair.information_order_id,
            source_execution_id=pair.source_execution_id,
            target_execution_id=pair.target_execution_id,
            source_action_id=pair.source_action_id,
            target_action_id=pair.target_action_id,
            arm=PlaceboArmV2.PHASE_SHIFTED_SOURCE,
            policy_id=policies[PlaceboArmV2.PHASE_SHIFTED_SOURCE.value],
            source_prefix_artifact_id=_id(1),
            construction_artifact_id=_id(2),
            constructor_commit_id="a" * 40,
            phase_shift_steps=0,
        )

    with pytest.raises(ValueError, match="non-identity permutation"):
        ChronologicalPlaceboConstructionV2(
            object_session_id=pair.object_session_id,
            information_order_id=pair.information_order_id,
            source_execution_id=pair.source_execution_id,
            target_execution_id=pair.target_execution_id,
            source_action_id=pair.source_action_id,
            target_action_id=pair.target_action_id,
            arm=PlaceboArmV2.IDENTITY_PERMUTED,
            policy_id=policies[PlaceboArmV2.IDENTITY_PERMUTED.value],
            source_prefix_artifact_id=_id(1),
            construction_artifact_id=_id(2),
            constructor_commit_id="a" * 40,
            permutation_artifact_id=_id(3),
            permutation_size=8,
            permutation_fixed_point_count=8,
        )
