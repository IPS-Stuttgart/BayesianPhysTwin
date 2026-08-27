from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from bayesian_phystwin_experiments.cross_action_transport_v2 import (
    CAUSAL4D_SLOTH_MULTI_ACTION_V1_SHA256,
    ChronologicalSessionPairV2,
    CrossActionProtocolV2,
    CrossActionTransportResultV2,
    PredictionDisposition,
    SealedTransportPredictionV2,
    TransportArm,
    TransportDecision,
    TransportScoreRowV2,
)


def _id(seed: int) -> str:
    return f"{seed:064x}"


def _stable(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def _pair(index: int) -> ChronologicalSessionPairV2:
    return ChronologicalSessionPairV2(
        object_session_id=f"s{index}",
        source_execution_id=f"e{2 * index:02d}",
        target_execution_id=f"e{2 * index + 1:02d}",
        source_action_id=f"a{index % 4}",
        target_action_id=f"a{(index + 1) % 4}",
        contact_stratum_id=_stable("contact", index % 3),
        information_order_id=_stable("chronology", index),
    )


def _protocol(*, sessions: int = 6, minimum: int = 4) -> CrossActionProtocolV2:
    return CrossActionProtocolV2(
        development_roster_id=_id(1),
        calibration_roster_id=_id(2),
        target_roster_id=_id(3),
        source_policy_id=_id(4),
        causal4d_design_sha256=CAUSAL4D_SLOTH_MULTI_ACTION_V1_SHA256,
        query_id=_id(5),
        query_jacobian_id=_id(6),
        score_definition_id=_id(7),
        grouping_rule_id=_id(8),
        interval_method_id=_id(9),
        target_access_policy_id=_id(10),
        model_stack_id=_id(11),
        numerical_environment_id=_id(12),
        technical_failure_policy_id=_id(13),
        candidate_family_id=_id(14),
        support_admission_id=_id(15),
        query_identifiability_id=_id(16),
        multi_action_identifiability_id=_id(17),
        nonlinear_closure_id=_id(18),
        guard_id=_id(19),
        session_pairs=tuple(_pair(i) for i in range(sessions)),
        registered_arms=(
            TransportArm.GUARDED_PHYSICAL,
            TransportArm.DISCREPANCY_ONLY,
            TransportArm.PHYSICAL_FALLBACK,
            TransportArm.LAST_RESIDUAL,
        ),
        minimum_sessions=minimum,
        bootstrap_replicates=500,
        bootstrap_seed=23,
        confidence_level=0.90,
        minimum_gain=0.5,
        minimum_discrepancy_contrast=0.5,
        minimum_comparator_contrast=0.25,
        maximum_harmful_session_fraction=0.50,
        maximum_harmful_selected_fraction=0.50,
    )


def _prediction(
    protocol: CrossActionProtocolV2,
    session: str,
    arm: TransportArm,
    *,
    disposition: PredictionDisposition | None = None,
    reverse: bool = False,
) -> SealedTransportPredictionV2:
    pair = protocol.pair_for_session(session)
    source_execution = pair.target_execution_id if reverse else pair.source_execution_id
    target_execution = pair.source_execution_id if reverse else pair.target_execution_id
    source_action = pair.target_action_id if reverse else pair.source_action_id
    target_action = pair.source_action_id if reverse else pair.target_action_id
    baseline = _stable("base", session)
    if arm is TransportArm.PHYSICAL_FALLBACK:
        candidate = None
        selected = baseline
        resolved = PredictionDisposition.BASELINE_REFERENCE
    else:
        candidate = _stable("candidate", session, arm.value)
        resolved = disposition or PredictionDisposition.CANDIDATE_SELECTED
        selected = (
            baseline if resolved is PredictionDisposition.EXACT_FALLBACK else candidate
        )
    return SealedTransportPredictionV2(
        protocol_id=protocol.protocol_id,
        registered_pair_id=pair.pair_id,
        object_session_id=session,
        source_execution_id=source_execution,
        target_execution_id=target_execution,
        source_action_id=source_action,
        target_action_id=target_action,
        arm=arm,
        baseline_belief_id=baseline,
        candidate_belief_id=candidate,
        selected_belief_id=selected,
        disposition=resolved,
        prediction_artifact_id=_stable("prediction", session, arm.value, reverse),
        source_evidence_id=_stable("source", session),
        prediction_batch_id=_id(40),
        commit_id="a" * 40,
        prediction_sealed_before_target=True,
    )


def _rows(
    protocol: CrossActionProtocolV2,
    *,
    physical_gain: float = 4.0,
    discrepancy_gain: float = 1.0,
    comparator_gain: float = 1.5,
    fallback_physical_session: str | None = None,
) -> tuple[TransportScoreRowV2, ...]:
    rows = []
    for session in protocol.target_session_ids:
        for arm in protocol.registered_arms:
            disposition = None
            gain = {
                TransportArm.GUARDED_PHYSICAL: physical_gain,
                TransportArm.DISCREPANCY_ONLY: discrepancy_gain,
                TransportArm.LAST_RESIDUAL: comparator_gain,
            }.get(arm, 0.0)
            if (
                arm is TransportArm.GUARDED_PHYSICAL
                and session == fallback_physical_session
            ):
                disposition = PredictionDisposition.EXACT_FALLBACK
                gain = 0.0
            prediction = _prediction(
                protocol,
                session,
                arm,
                disposition=disposition,
            )
            rows.append(
                TransportScoreRowV2(
                    prediction=prediction,
                    target_outcome_id=_stable("target", session),
                    target_access_attestation_id=_id(50),
                    scorer_id=_id(51),
                    proper_score=10.0 - gain,
                )
            )
    return tuple(rows)


def _result(
    protocol: CrossActionProtocolV2,
    rows: tuple[TransportScoreRowV2, ...],
    *,
    excluded: tuple[str, ...] = (),
    technical: tuple[str, ...] = (),
) -> CrossActionTransportResultV2:
    return CrossActionTransportResultV2(
        protocol=protocol,
        score_rows=rows,
        target_accounting_id=_id(52),
        excluded_session_ids=excluded,
        technical_failure_session_ids=technical,
    )


def test_positive_chronological_transport_and_order_invariance() -> None:
    protocol = _protocol()
    rows = _rows(protocol, fallback_physical_session="s5")
    forward = _result(protocol, rows)
    reverse_order = _result(protocol, tuple(reversed(rows)))
    assert forward.decision is TransportDecision.SUPPORTED
    assert forward.supports_physical_transport
    physical = next(
        summary
        for summary in forward.arm_summaries
        if summary.arm is TransportArm.GUARDED_PHYSICAL
    )
    assert physical.selected_sessions == 5
    assert physical.fallback_sessions == 1
    assert physical.harmful_selected_sessions == 0
    assert forward.result_id == reverse_order.result_id


def test_reverse_same_session_direction_fails_closed() -> None:
    protocol = _protocol()
    rows = list(_rows(protocol))
    rows[0] = replace(
        rows[0],
        prediction=_prediction(
            protocol,
            rows[0].prediction.object_session_id,
            rows[0].prediction.arm,
            reverse=True,
        ),
    )
    with pytest.raises(ValueError, match="registered chronological"):
        _result(protocol, tuple(rows))


def test_incomplete_sparse_pair_arm_roster_fails_closed() -> None:
    protocol = _protocol()
    rows = tuple(
        row
        for row in _rows(protocol)
        if not (
            row.prediction.object_session_id == "s0"
            and row.prediction.arm is TransportArm.LAST_RESIDUAL
        )
    )
    with pytest.raises(ValueError, match="every registered arm"):
        _result(protocol, rows)


def test_target_accounting_includes_exclusions_and_technical_failures() -> None:
    protocol = _protocol(sessions=6, minimum=5)
    rows = tuple(
        row
        for row in _rows(protocol)
        if row.prediction.object_session_id not in {"s4", "s5"}
    )
    result = _result(protocol, rows, excluded=("s4",), technical=("s5",))
    assert result.independent_session_count == 4
    assert result.decision is TransportDecision.INSUFFICIENT_SESSIONS
    with pytest.raises(ValueError, match="cover the frozen"):
        _result(protocol, rows, excluded=("s4",))


def test_exact_fallback_identity_and_score_are_enforced() -> None:
    protocol = _protocol()
    prediction = _prediction(
        protocol,
        "s0",
        TransportArm.GUARDED_PHYSICAL,
        disposition=PredictionDisposition.EXACT_FALLBACK,
    )
    with pytest.raises(ValueError, match="exact baseline"):
        replace(prediction, selected_belief_id=_id(999))
    rows = list(_rows(protocol, fallback_physical_session="s0"))
    index = next(
        i
        for i, row in enumerate(rows)
        if row.prediction.object_session_id == "s0"
        and row.prediction.arm is TransportArm.GUARDED_PHYSICAL
    )
    rows[index] = replace(rows[index], proper_score=10.25)
    with pytest.raises(ValueError, match="score identically"):
        _result(protocol, tuple(rows))


def test_physical_arm_must_beat_discrepancy_and_matched_comparator() -> None:
    protocol = _protocol()
    result = _result(
        protocol,
        _rows(
            protocol,
            physical_gain=1.1,
            discrepancy_gain=1.0,
            comparator_gain=1.0,
        ),
    )
    assert result.decision is TransportDecision.NOT_SUPPORTED


def test_protocol_is_canonical_and_design_specific() -> None:
    protocol = _protocol()
    reordered = replace(
        protocol,
        session_pairs=tuple(reversed(protocol.session_pairs)),
        registered_arms=tuple(reversed(protocol.registered_arms)),
    )
    assert protocol.protocol_id == reordered.protocol_id
    with pytest.raises(ValueError, match="frozen Causal4D"):
        replace(protocol, causal4d_design_sha256=_id(999))
    with pytest.raises(ValueError, match="exactly physical_fallback"):
        replace(
            protocol,
            registered_arms=protocol.registered_arms + (TransportArm.STATE_ONLY,),
        )


def test_duplicate_execution_or_reverse_roster_is_not_representable() -> None:
    first = _pair(0)
    duplicate_execution = replace(
        _pair(1),
        source_execution_id=first.target_execution_id,
    )
    with pytest.raises(ValueError, match="execution must appear exactly once"):
        replace(
            _protocol(sessions=2, minimum=2),
            session_pairs=(first, duplicate_execution),
        )
    with pytest.raises(ValueError, match="source-to-target order"):
        replace(first, source_precedes_target=False)
