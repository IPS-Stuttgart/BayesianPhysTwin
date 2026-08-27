from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

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


def _protocol(
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


def _prediction(
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


def _rows(
    protocol: CrossActionProtocolV2,
    *,
    physical_gain: float = 4.0,
    discrepancy_gain: float = 1.0,
    comparator_gain: float = 1.5,
    physical_dispositions: tuple[PredictionDisposition, ...] | None = None,
    physical_gains: tuple[float, ...] | None = None,
) -> tuple[TransportScoreRowV2, ...]:
    if physical_dispositions is None:
        physical_dispositions = tuple(
            PredictionDisposition.CANDIDATE_SELECTED for _ in protocol.session_pairs
        )
    if physical_gains is None:
        physical_gains = tuple(physical_gain for _ in protocol.session_pairs)
    rows = []
    for index, pair in enumerate(protocol.session_pairs):
        for arm in protocol.registered_arms:
            disposition = None
            gain = 0.0
            if arm is TransportArm.GUARDED_PHYSICAL:
                disposition = physical_dispositions[index]
                gain = physical_gains[index]
            elif arm is TransportArm.DISCREPANCY_ONLY:
                gain = discrepancy_gain
            elif arm is TransportArm.LAST_RESIDUAL:
                gain = comparator_gain
            prediction = _prediction(
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


def test_positive_sparse_transport_and_order_invariance() -> None:
    protocol = _protocol()
    rows = _rows(protocol)
    forward = _result(protocol, rows)
    reverse = _result(protocol, tuple(reversed(rows)))
    assert forward.decision is SparseTransportDecision.SUPPORTED
    assert forward.supports_physical_transport
    physical = next(
        summary
        for summary in forward.arm_summaries
        if summary.arm is TransportArm.GUARDED_PHYSICAL
    )
    assert physical.selected_sessions == 14
    assert physical.fallback_sessions == 0
    assert physical.harmful_accepted_sessions == 0
    assert physical.harmful_accepted_fraction == 0.0
    assert physical.harmful_accepted_fraction_upper < 0.20
    assert forward.result_id == reverse.result_id


def test_impossible_harm_caps_rejected_before_target_access() -> None:
    with pytest.raises(ValueError, match="harm cap is impossible"):
        _protocol(sessions=13, maximum_harm=0.20)
    with pytest.raises(ValueError, match="harm cap is impossible"):
        _protocol(sessions=18, maximum_harm=0.10)


def test_exact_fallback_does_not_dilute_harm_certificate() -> None:
    protocol = _protocol(minimum_accepted=1)
    dispositions = (
        PredictionDisposition.CANDIDATE_SELECTED,
        *(PredictionDisposition.EXACT_FALLBACK for _ in range(13)),
    )
    gains = (-1.0, *(0.0 for _ in range(13)))
    result = _result(
        protocol,
        _rows(
            protocol,
            physical_dispositions=dispositions,
            physical_gains=gains,
        ),
    )
    physical = next(
        summary
        for summary in result.arm_summaries
        if summary.arm is TransportArm.GUARDED_PHYSICAL
    )
    assert physical.selected_sessions == 1
    assert physical.fallback_sessions == 13
    assert physical.harmful_accepted_sessions == 1
    assert physical.harmful_accepted_fraction == 1.0
    assert physical.harmful_accepted_fraction_upper > 0.90
    assert result.decision is SparseTransportDecision.NOT_SUPPORTED


def test_reverse_same_session_reuse_fails_closed() -> None:
    protocol = _protocol()
    rows = list(_rows(protocol))
    index = next(
        index
        for index, row in enumerate(rows)
        if row.prediction.object_session_id == "s00"
        and row.prediction.arm is TransportArm.GUARDED_PHYSICAL
    )
    prediction = rows[index].prediction
    reversed_prediction = replace(
        prediction,
        source_execution_id=prediction.target_execution_id,
        target_execution_id=prediction.source_execution_id,
        source_action_id=prediction.target_action_id,
        target_action_id=prediction.source_action_id,
    )
    rows[index] = replace(rows[index], prediction=reversed_prediction)
    with pytest.raises(ValueError, match="source->target chronology"):
        _result(protocol, tuple(rows))


def test_sparse_roster_accounting_is_complete_and_session_level() -> None:
    protocol = _protocol(minimum_sessions=14)
    incomplete = tuple(
        row for row in _rows(protocol) if row.prediction.object_session_id != "s13"
    )
    with pytest.raises(ValueError, match="frozen sparse session roster"):
        _result(protocol, incomplete)
    result = _result(protocol, incomplete, technical=("s13",))
    assert result.independent_session_count == 13
    assert result.decision is SparseTransportDecision.INSUFFICIENT_SESSIONS


def test_target_blind_exact_fallback_is_enforced() -> None:
    protocol = _protocol()
    pair = protocol.session_pairs[0]
    prediction = _prediction(
        protocol,
        pair,
        TransportArm.GUARDED_PHYSICAL,
        disposition=PredictionDisposition.EXACT_FALLBACK,
    )
    with pytest.raises(ValueError, match="exact baseline"):
        replace(prediction, selected_belief_id=_id(999))
    with pytest.raises(ValueError, match="sealed before target"):
        replace(prediction, target_outcomes_used=True)


def test_exact_fallback_score_must_match_physical_fallback() -> None:
    protocol = _protocol(minimum_accepted=1)
    dispositions = (
        PredictionDisposition.CANDIDATE_SELECTED,
        PredictionDisposition.EXACT_FALLBACK,
        *(PredictionDisposition.CANDIDATE_SELECTED for _ in range(12)),
    )
    gains = (4.0, 0.0, *(4.0 for _ in range(12)))
    rows = list(
        _rows(
            protocol,
            physical_dispositions=dispositions,
            physical_gains=gains,
        )
    )
    index = next(
        index
        for index, row in enumerate(rows)
        if row.prediction.object_session_id == "s01"
        and row.prediction.arm is TransportArm.GUARDED_PHYSICAL
    )
    rows[index] = replace(rows[index], proper_score=9.5)
    with pytest.raises(ValueError, match="score identically"):
        _result(protocol, tuple(rows))


def test_protocol_roster_is_canonical_and_execution_reuse_fails() -> None:
    protocol = _protocol()
    reordered = replace(protocol, session_pairs=tuple(reversed(protocol.session_pairs)))
    assert protocol.protocol_id == reordered.protocol_id
    first, second, *rest = protocol.session_pairs
    reused = replace(second, source_execution_id=first.source_execution_id)
    with pytest.raises(ValueError, match="each execution"):
        replace(protocol, session_pairs=(first, reused, *rest))


def test_checked_in_causal4d_sparse_pair_roster_matches_design() -> None:
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "protocols"
        / "cross_action_transport"
        / "causal4d_sloth_multi_action_v1_sparse_pairs.json"
    )
    payload = json.loads(path.read_text())
    assert (
        payload["causal4d_design_sha256"]
        == CAUSAL4D_SLOTH_MULTI_ACTION_V1_DESIGN_SHA256
    )
    assert payload["reverse_same_session_reuse_allowed"] is False
    assert payload["independent_unit"] == "physical_grasp_session"
    sessions = payload["session_pairs"]
    assert len(sessions) == 18
    assert len({entry["object_session_id"] for entry in sessions}) == 18
    execution_ids = [
        execution_id
        for entry in sessions
        for execution_id in (
            entry["source_execution_id"],
            entry["target_execution_id"],
        )
    ]
    assert len(execution_ids) == len(set(execution_ids)) == 36
    assert all(entry["source_execution_id"].endswith("-e1") for entry in sessions)
    assert all(entry["target_execution_id"].endswith("-e2") for entry in sessions)
    assert all(
        entry["source_action_id"] != entry["target_action_id"] for entry in sessions
    )
