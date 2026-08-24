from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from bayesian_phystwin_experiments.cross_action_transport_v1 import (
    CrossActionProtocolV1,
    CrossActionTransportResultV1,
    PredictionDisposition,
    SealedTransportPredictionV1,
    TransportArm,
    TransportDecision,
    TransportScoreRowV1,
)


def _id(seed: int) -> str:
    return f"{seed:064x}"


def _stable(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def _protocol(*, sessions: int = 3, minimum: int = 3) -> CrossActionProtocolV1:
    return CrossActionProtocolV1(
        development_roster_id=_id(1),
        calibration_roster_id=_id(2),
        target_roster_id=_id(3),
        query_id=_id(4),
        query_jacobian_id=_id(5),
        score_definition_id=_id(6),
        grouping_rule_id=_id(7),
        interval_method_id=_id(8),
        target_access_policy_id=_id(9),
        model_stack_id=_id(10),
        numerical_environment_id=_id(11),
        technical_failure_policy_id=_id(12),
        action_ids=("b", "a"),
        target_session_ids=tuple(f"s{i}" for i in range(sessions)),
        registered_arms=(
            TransportArm.GUARDED_PHYSICAL,
            TransportArm.DISCREPANCY_ONLY,
            TransportArm.PHYSICAL_FALLBACK,
            TransportArm.LAST_RESIDUAL,
        ),
        physical_transport_arm=TransportArm.GUARDED_PHYSICAL,
        discrepancy_reference_arm=TransportArm.DISCREPANCY_ONLY,
        matched_comparator_arm=TransportArm.LAST_RESIDUAL,
        minimum_sessions=minimum,
        bootstrap_replicates=500,
        bootstrap_seed=17,
        confidence_level=0.90,
        minimum_off_diagonal_gain=1.0,
        minimum_discrepancy_contrast=0.5,
        minimum_comparator_contrast=0.2,
        maximum_harmful_session_fraction=0.5,
    )


def _prediction(
    protocol: CrossActionProtocolV1,
    session: str,
    source: str,
    target: str,
    arm: TransportArm,
    *,
    disposition: PredictionDisposition | None = None,
) -> SealedTransportPredictionV1:
    baseline = _stable("base", session, source, target)
    if arm is TransportArm.PHYSICAL_FALLBACK:
        candidate = None
        selected = baseline
        resolved = PredictionDisposition.BASELINE_REFERENCE
    else:
        candidate = _stable("candidate", session, source, target, arm.value)
        resolved = disposition or PredictionDisposition.CANDIDATE_SELECTED
        selected = (
            baseline
            if resolved is PredictionDisposition.EXACT_FALLBACK
            else candidate
        )
    return SealedTransportPredictionV1(
        protocol_id=protocol.protocol_id,
        object_session_id=session,
        source_action_id=source,
        target_action_id=target,
        arm=arm,
        baseline_belief_id=baseline,
        candidate_belief_id=candidate,
        selected_belief_id=selected,
        disposition=resolved,
        prediction_artifact_id=_stable(
            "prediction", session, source, target, arm.value
        ),
        source_evidence_id=_id(30),
        prediction_batch_id=_id(40),
        commit_id="a" * 40,
        prediction_sealed_before_target=True,
    )


def _rows(
    protocol: CrossActionProtocolV1,
    *,
    physical=(4.0, 3.5, 3.0),
    discrepancy=1.0,
    comparator=1.5,
) -> tuple[TransportScoreRowV1, ...]:
    rows = []
    for index, session in enumerate(protocol.target_session_ids):
        for source, target in protocol.action_pairs:
            off_diagonal = source != target
            for arm in protocol.registered_arms:
                prediction = _prediction(protocol, session, source, target, arm)
                gain = 0.0
                if off_diagonal:
                    gain = {
                        TransportArm.GUARDED_PHYSICAL: physical[index],
                        TransportArm.DISCREPANCY_ONLY: discrepancy,
                        TransportArm.LAST_RESIDUAL: comparator,
                    }.get(arm, 0.0)
                rows.append(
                    TransportScoreRowV1(
                        prediction=prediction,
                        target_outcome_id=_stable("target", session, source, target),
                        target_access_attestation_id=_id(50),
                        scorer_id=_id(51),
                        proper_score=10.0 - gain,
                    )
                )
    return tuple(rows)


def _result(protocol, rows, *, excluded=()):
    return CrossActionTransportResultV1(
        protocol=protocol,
        score_rows=rows,
        target_accounting_id=_id(52),
        excluded_session_ids=excluded,
    )


def test_positive_transport_and_order_invariance() -> None:
    protocol = _protocol()
    rows = _rows(protocol)
    forward = _result(protocol, rows)
    reverse = _result(protocol, tuple(reversed(rows)))
    assert forward.decision is TransportDecision.SUPPORTED
    assert forward.supports_physical_transport
    physical = next(
        summary
        for summary in forward.arm_summaries
        if summary.arm is TransportArm.GUARDED_PHYSICAL
    )
    assert physical.harmful_fraction == 0.0
    assert 0.0 < physical.harmful_fraction_interval[1] < 0.5
    assert forward.result_id == reverse.result_id


def test_physical_arm_must_beat_both_references() -> None:
    protocol = _protocol()
    result = _result(
        protocol,
        _rows(protocol, physical=(1.1, 1.1, 1.1), discrepancy=1.0, comparator=1.0),
    )
    assert result.decision is TransportDecision.NOT_SUPPORTED


def test_incomplete_action_matrix_fails_closed() -> None:
    protocol = _protocol()
    rows = tuple(
        row
        for row in _rows(protocol)
        if not (
            row.prediction.object_session_id == "s0"
            and row.prediction.source_action_id == "a"
            and row.prediction.target_action_id == "b"
        )
    )
    with pytest.raises(ValueError, match="complete action matrix"):
        _result(protocol, rows)


def test_exact_fallback_and_target_blindness_are_enforced() -> None:
    protocol = _protocol()
    prediction = _prediction(
        protocol,
        "s0",
        "a",
        "b",
        TransportArm.GUARDED_PHYSICAL,
        disposition=PredictionDisposition.EXACT_FALLBACK,
    )
    with pytest.raises(ValueError, match="exact baseline"):
        replace(prediction, selected_belief_id=_id(999))
    with pytest.raises(ValueError, match="sealed before target"):
        replace(prediction, target_outcomes_used=True)


def test_target_accounting_and_independent_session_gate() -> None:
    protocol = _protocol(sessions=4, minimum=4)
    incomplete = tuple(
        row
        for row in _rows(protocol, physical=(4.0, 3.5, 3.0, 2.5))
        if row.prediction.object_session_id != "s3"
    )
    with pytest.raises(ValueError, match="cover the frozen roster"):
        _result(protocol, incomplete)
    short = _protocol(minimum=4)
    assert (
        _result(short, _rows(short)).decision
        is TransportDecision.INSUFFICIENT_SESSIONS
    )


def test_protocol_canonicalization_and_distinct_references() -> None:
    protocol = _protocol()
    reordered = replace(
        protocol,
        action_ids=tuple(reversed(protocol.action_ids)),
        registered_arms=tuple(reversed(protocol.registered_arms)),
    )
    assert protocol.protocol_id == reordered.protocol_id
    with pytest.raises(ValueError, match="must be distinct"):
        replace(protocol, matched_comparator_arm=TransportArm.DISCREPANCY_ONLY)
