from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from bayesian_phystwin_experiments.cross_action_placebo_v1 import (
    CrossActionPlaceboScoreRowV1,
    PlaceboArm,
    PlaceboDecision,
    SealedCrossActionPlaceboPredictionV1,
)
from bayesian_phystwin_experiments.cross_action_placebo_v2 import (
    CrossActionPlaceboProtocolV2,
    CrossActionPlaceboResultV2,
)
from bayesian_phystwin_experiments.cross_action_transport_v1 import (
    PredictionDisposition,
    SealedTransportPredictionV1,
    TransportArm,
    TransportDecision,
    TransportScoreRowV1,
)
from bayesian_phystwin_experiments.cross_action_transport_v2 import (
    CrossActionProtocolV2,
    CrossActionTransportResultV2,
    SessionActionSetV2,
)


def _id(seed: int) -> str:
    return f"{seed:064x}"


def _stable(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def _session_sets() -> tuple[SessionActionSetV2, ...]:
    return (
        SessionActionSetV2("s0", ("a", "b")),
        SessionActionSetV2("s1", ("a", "c")),
        SessionActionSetV2("s2", ("b", "c")),
    )


def _transport_protocol() -> CrossActionProtocolV2:
    return CrossActionProtocolV2(
        development_roster_id=_id(1),
        calibration_roster_id=_id(2),
        target_roster_id=_id(3),
        acquisition_binding_id=_id(4),
        query_id=_id(5),
        query_jacobian_id=_id(6),
        identifiability_certificate_id=_id(7),
        nonlinear_closure_certificate_id=_id(8),
        score_definition_id=_id(9),
        grouping_rule_id=_id(10),
        interval_method_id=_id(11),
        target_access_policy_id=_id(12),
        model_stack_id=_id(13),
        numerical_environment_id=_id(14),
        technical_failure_policy_id=_id(15),
        session_action_sets=_session_sets(),
        registered_arms=(
            TransportArm.PHYSICAL_FALLBACK,
            TransportArm.LAST_RESIDUAL,
            TransportArm.DISCREPANCY_ONLY,
            TransportArm.STATE_ONLY,
            TransportArm.STATE_PARAMETER,
            TransportArm.GUARDED_PHYSICAL,
        ),
        physical_transport_arm=TransportArm.GUARDED_PHYSICAL,
        discrepancy_reference_arm=TransportArm.DISCREPANCY_ONLY,
        matched_comparator_arm=TransportArm.LAST_RESIDUAL,
        minimum_sessions=3,
        bootstrap_replicates=200,
        bootstrap_seed=17,
        confidence_level=0.90,
        minimum_off_diagonal_gain=1.0,
        minimum_discrepancy_contrast=0.5,
        minimum_comparator_contrast=0.5,
        maximum_harmful_session_fraction=0.8,
    )


def _transport_prediction(
    protocol: CrossActionProtocolV2,
    session: str,
    source: str,
    target: str,
    arm: TransportArm,
) -> SealedTransportPredictionV1:
    baseline = _stable("base", session, source, target)
    if arm is TransportArm.PHYSICAL_FALLBACK:
        candidate = None
        selected = baseline
        disposition = PredictionDisposition.BASELINE_REFERENCE
    else:
        candidate = _stable("candidate", session, source, target, arm.value)
        selected = candidate
        disposition = PredictionDisposition.CANDIDATE_SELECTED
    return SealedTransportPredictionV1(
        protocol_id=protocol.protocol_id,
        object_session_id=session,
        source_action_id=source,
        target_action_id=target,
        arm=arm,
        baseline_belief_id=baseline,
        candidate_belief_id=candidate,
        selected_belief_id=selected,
        disposition=disposition,
        prediction_artifact_id=_stable(
            "prediction", session, source, target, arm.value
        ),
        source_evidence_id=_id(30),
        prediction_batch_id=_id(31),
        commit_id="a" * 40,
        prediction_sealed_before_target=True,
    )


def _transport_rows(
    protocol: CrossActionProtocolV2,
) -> tuple[TransportScoreRowV1, ...]:
    gain_by_arm = {
        TransportArm.PHYSICAL_FALLBACK: 0.0,
        TransportArm.LAST_RESIDUAL: 1.5,
        TransportArm.DISCREPANCY_ONLY: 1.0,
        TransportArm.STATE_ONLY: 2.0,
        TransportArm.STATE_PARAMETER: 2.5,
        TransportArm.GUARDED_PHYSICAL: 4.0,
    }
    rows = []
    for session in protocol.target_session_ids:
        for source, target in protocol.action_pairs_for_session(session):
            off_diagonal = source != target
            for arm in protocol.registered_arms:
                prediction = _transport_prediction(
                    protocol,
                    session,
                    source,
                    target,
                    arm,
                )
                gain = gain_by_arm[arm] if off_diagonal else 0.0
                rows.append(
                    TransportScoreRowV1(
                        prediction=prediction,
                        target_outcome_id=_stable("target", session, target),
                        target_access_attestation_id=_id(32),
                        scorer_id=_id(33),
                        proper_score=10.0 - gain,
                    )
                )
    return tuple(rows)


def test_session_specific_transport_uses_balanced_incomplete_matrix() -> None:
    protocol = _transport_protocol()
    assert protocol.action_ids == ("a", "b", "c")
    assert protocol.action_pairs_for_session("s0") == (
        ("a", "a"),
        ("a", "b"),
        ("b", "a"),
        ("b", "b"),
    )
    result = CrossActionTransportResultV2(
        protocol=protocol,
        score_rows=_transport_rows(protocol),
        target_accounting_id=_id(34),
    )
    assert result.decision is TransportDecision.SUPPORTED
    assert result.supports_physical_transport
    assert result.independent_session_count == 3


def test_unobserved_global_pair_is_not_required_but_extra_pair_fails() -> None:
    protocol = _transport_protocol()
    rows = _transport_rows(protocol)
    assert not any(
        row.prediction.object_session_id == "s0"
        and row.prediction.source_action_id == "a"
        and row.prediction.target_action_id == "c"
        for row in rows
    )
    result = CrossActionTransportResultV2(
        protocol=protocol,
        score_rows=rows,
        target_accounting_id=_id(34),
    )
    assert result.decision is TransportDecision.SUPPORTED

    template = next(
        row
        for row in rows
        if row.prediction.object_session_id == "s0"
        and row.prediction.arm is TransportArm.PHYSICAL_FALLBACK
    )
    extra_prediction = replace(
        template.prediction,
        source_action_id="a",
        target_action_id="c",
        prediction_artifact_id=_id(900),
    )
    extra = replace(template, prediction=extra_prediction)
    with pytest.raises(ValueError, match="registered action matrix"):
        CrossActionTransportResultV2(
            protocol=protocol,
            score_rows=(*rows, extra),
            target_accounting_id=_id(34),
        )


def _placebo_protocol(
    transport: CrossActionProtocolV2,
) -> CrossActionPlaceboProtocolV2:
    placebos = (
        PlaceboArm.WRONG_ACTION,
        PlaceboArm.WRONG_OBJECT,
        PlaceboArm.PHASE_SHIFTED,
        PlaceboArm.IDENTITY_PERMUTED,
    )
    labels = (TransportArm.GUARDED_PHYSICAL.value, *(arm.value for arm in placebos))
    return CrossActionPlaceboProtocolV2(
        parent_transport_protocol_id=transport.protocol_id,
        target_roster_id=transport.target_roster_id,
        session_action_sets=transport.session_action_sets,
        physical_arm_label=TransportArm.GUARDED_PHYSICAL.value,
        placebo_arms=placebos,
        arm_construction_ids={
            label: _stable("construction", label) for label in labels
        },
        minimum_sessions=3,
        bootstrap_replicates=200,
        bootstrap_seed=211,
        confidence_level=0.90,
        minimum_placebo_contrast=0.5,
    )


def _placebo_rows(
    protocol: CrossActionPlaceboProtocolV2,
) -> tuple[CrossActionPlaceboScoreRowV1, ...]:
    rows = []
    for session in protocol.target_session_ids:
        for source, target in protocol.off_diagonal_action_pairs_for_session(session):
            parent = _stable("parent", session, source, target)
            outcome = _stable("target", session, target)
            for arm in protocol.arm_labels:
                physical = arm == protocol.physical_arm_label
                prediction = SealedCrossActionPlaceboPredictionV1(
                    protocol_id=protocol.protocol_id,
                    object_session_id=session,
                    source_action_id=source,
                    target_action_id=target,
                    arm_label=arm,
                    parent_transport_prediction_id=parent,
                    construction_id=protocol.arm_construction_ids[arm],
                    prediction_artifact_id=_stable(
                        "placebo-prediction", session, source, target, arm
                    ),
                    prediction_batch_id=_id(40),
                    commit_id="a" * 40,
                    candidate_selected=True,
                    exact_fallback=False,
                    prediction_sealed_before_target=True,
                )
                rows.append(
                    CrossActionPlaceboScoreRowV1(
                        prediction=prediction,
                        target_outcome_id=outcome,
                        target_access_attestation_id=_id(32),
                        scorer_id=_id(33),
                        proper_score=6.0 if physical else 9.0,
                    )
                )
    return tuple(rows)


def test_session_specific_placebos_use_only_observed_action_pairs() -> None:
    transport = _transport_protocol()
    protocol = _placebo_protocol(transport)
    result = CrossActionPlaceboResultV2(
        protocol=protocol,
        score_rows=_placebo_rows(protocol),
        target_accounting_id=_id(34),
    )
    assert result.decision is PlaceboDecision.SUPPORTED
    assert result.independent_session_count == 3
    assert result.selected_physical_prediction_count == 6


def test_session_roster_is_canonical_and_immutable() -> None:
    protocol = _transport_protocol()
    reordered = replace(
        protocol,
        session_action_sets=tuple(reversed(protocol.session_action_sets)),
    )
    assert reordered.protocol_id == protocol.protocol_id
    with pytest.raises(ValueError, match="repeat a session"):
        replace(
            protocol,
            session_action_sets=(protocol.session_action_sets[0],) * 3,
        )
