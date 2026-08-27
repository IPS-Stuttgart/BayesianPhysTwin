from __future__ import annotations

import hashlib
import itertools
from copy import deepcopy

import pytest

from bayesian_phystwin_experiments.causal4d_cross_action_registration_v1 import (
    Causal4DJointTransportResultV1,
    JointTransportDecision,
    build_causal4d_cross_action_registration_v1,
    causal4d_protocol_design_sha256,
    extract_causal4d_cross_action_design_v1,
)
from bayesian_phystwin_experiments.cross_action_placebo_v1 import (
    CrossActionPlaceboScoreRowV1,
    SealedCrossActionPlaceboPredictionV1,
)
from bayesian_phystwin_experiments.cross_action_placebo_v2 import (
    CrossActionPlaceboResultV2,
)
from bayesian_phystwin_experiments.cross_action_transport_v1 import (
    PredictionDisposition,
    SealedTransportPredictionV1,
    TransportArm,
    TransportScoreRowV1,
)
from bayesian_phystwin_experiments.cross_action_transport_v2 import (
    CrossActionTransportResultV2,
)

_ACTIONS = ("lateral_low", "lift_high", "lift_low", "lower_high")
_CONTACTS = ("left_forepaw", "right_forepaw", "upper_torso")


def _id(seed: int) -> str:
    return f"{seed:064x}"


def _stable(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def _protocol_document() -> dict[str, object]:
    sessions: list[str] = []
    executions: list[dict[str, object]] = []
    execution_index = 0
    for contact_index, contact in enumerate(_CONTACTS, start=1):
        for pair_index, pair in enumerate(itertools.combinations(_ACTIONS, 2), start=1):
            session = f"sloth-v1-c{contact_index}-s{pair_index}"
            sessions.append(session)
            for pair_order, action in enumerate(pair):
                executions.append(
                    {
                        "acquisition_execution_index": execution_index,
                        "command_profile_id": action,
                        "contact_region_id": contact,
                        "execution_id": f"{session}-e{pair_order + 1}",
                        "pair_order": pair_order,
                        "session_id": session,
                    }
                )
                execution_index += 1
    document: dict[str, object] = {
        "protocol_id": "causal4d-sloth-multi-action-v1",
        "schema_version": 1,
        "analysis_lock": {
            "exclusions_locked_before_target_evaluation": True,
            "no_held_contact_or_profile_in_fold_source_sessions": True,
            "no_session_shared_between_fit_and_calibration": True,
            "split_unit": "grasp_session",
            "target_outcomes_may_not_select_hyperparameters": True,
        },
        "command_profiles": [{"id": action} for action in _ACTIONS],
        "contact_regions": [{"id": contact} for contact in _CONTACTS],
        "acquisition_session_order": sessions,
        "executions": executions,
    }
    document["design_sha256"] = causal4d_protocol_design_sha256(document)
    return document


def _registration(document: dict[str, object]):
    placebo_labels = (
        "guarded_physical",
        "wrong_action",
        "wrong_object",
        "phase_shifted",
        "identity_permuted",
    )
    return build_causal4d_cross_action_registration_v1(
        document,
        causal4d_revision="b" * 40,
        bayesian_phystwin_revision="a" * 40,
        causal4d_amendment_id=_id(1),
        causal4d_method_freeze_id=_id(2),
        causal4d_method_freeze_attestation_id=_id(3),
        causal4d_source_panel_id=_id(4),
        causal4d_readiness_id=_id(5),
        causal4d_primary_analysis_id=_id(6),
        causal4d_target_access_policy_id=_id(7),
        bayesian_phystwin_distribution_id=_id(8),
        prob4d_usage_declaration_id=_id(9),
        prediction_batch_policy_id=_id(10),
        development_roster_id=_id(11),
        calibration_roster_id=_id(12),
        target_roster_id=_id(13),
        query_id=_id(14),
        query_jacobian_id=_id(15),
        identifiability_certificate_id=_id(16),
        nonlinear_closure_certificate_id=_id(17),
        score_definition_id=_id(18),
        grouping_rule_id=_id(19),
        interval_method_id=_id(20),
        model_stack_id=_id(21),
        numerical_environment_id=_id(22),
        technical_failure_policy_id=_id(23),
        placebo_arm_construction_ids={
            label: _stable("construction", label) for label in placebo_labels
        },
        minimum_sessions=12,
        bootstrap_replicates=200,
        bootstrap_seed=20260827,
        confidence_level=0.90,
        minimum_off_diagonal_gain=1.0,
        minimum_discrepancy_contrast=0.5,
        minimum_comparator_contrast=0.5,
        maximum_harmful_session_fraction=0.3,
        minimum_placebo_contrast=0.5,
        expected_design_sha256=str(document["design_sha256"]),
    )


def test_extracts_balanced_k4_action_design() -> None:
    document = _protocol_document()
    design = extract_causal4d_cross_action_design_v1(
        document,
        expected_design_sha256=str(document["design_sha256"]),
    )
    assert len(design.target_session_ids) == 18
    assert len(design.execution_ids) == 36
    assert design.action_ids == _ACTIONS
    observed_pairs = {
        session.action_ids for session in design.session_action_sets
    }
    assert observed_pairs == set(itertools.combinations(_ACTIONS, 2))
    assert all(
        len(session.off_diagonal_action_pairs) == 2
        for session in design.session_action_sets
    )


def test_rejects_pair_imbalance_even_with_recomputed_digest() -> None:
    document = _protocol_document()
    tampered = deepcopy(document)
    executions = tampered["executions"]
    assert isinstance(executions, list)
    first = executions[0]
    second = executions[1]
    assert isinstance(first, dict) and isinstance(second, dict)
    second["command_profile_id"] = first["command_profile_id"]
    tampered["design_sha256"] = causal4d_protocol_design_sha256(tampered)
    with pytest.raises(ValueError, match="two distinct actions"):
        extract_causal4d_cross_action_design_v1(
            tampered,
            expected_design_sha256=str(tampered["design_sha256"]),
        )


def test_registration_binds_all_transport_and_placebo_arms() -> None:
    registration = _registration(_protocol_document())
    assert len(registration.transport_protocol.session_action_sets) == 18
    assert set(registration.transport_protocol.registered_arms) == set(TransportArm)
    assert len(registration.placebo_protocol.placebo_arms) == 4
    assert (
        registration.transport_protocol.acquisition_binding_id
        == registration.design.design_id
    )
    assert (
        registration.placebo_protocol.parent_transport_protocol_id
        == registration.transport_protocol.protocol_id
    )


def _transport_rows(registration) -> tuple[TransportScoreRowV1, ...]:
    protocol = registration.transport_protocol
    gains = {
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
            baseline = _stable("baseline", session, source, target)
            for arm in protocol.registered_arms:
                if arm is TransportArm.PHYSICAL_FALLBACK:
                    candidate = None
                    selected = baseline
                    disposition = PredictionDisposition.BASELINE_REFERENCE
                else:
                    candidate = _stable(
                        "candidate", session, source, target, arm.value
                    )
                    selected = candidate
                    disposition = PredictionDisposition.CANDIDATE_SELECTED
                prediction = SealedTransportPredictionV1(
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
                    commit_id=registration.bayesian_phystwin_revision,
                    prediction_sealed_before_target=True,
                )
                gain = gains[arm] if source != target else 0.0
                rows.append(
                    TransportScoreRowV1(
                        prediction=prediction,
                        target_outcome_id=_stable(
                            "outcome", session, source, target
                        ),
                        target_access_attestation_id=_id(32),
                        scorer_id=_id(33),
                        proper_score=10.0 - gain,
                    )
                )
    return tuple(rows)


def _placebo_rows(
    registration,
    transport_rows: tuple[TransportScoreRowV1, ...],
) -> tuple[CrossActionPlaceboScoreRowV1, ...]:
    protocol = registration.placebo_protocol
    physical_predictions = {
        (
            row.prediction.object_session_id,
            row.prediction.source_action_id,
            row.prediction.target_action_id,
        ): row
        for row in transport_rows
        if row.prediction.arm is TransportArm.GUARDED_PHYSICAL
        and row.prediction.source_action_id != row.prediction.target_action_id
    }
    rows = []
    for key, transport_row in sorted(physical_predictions.items()):
        session, source, target = key
        parent = transport_row.prediction
        for arm in protocol.arm_labels:
            physical = arm == protocol.physical_arm_label
            prediction = SealedCrossActionPlaceboPredictionV1(
                protocol_id=protocol.protocol_id,
                object_session_id=session,
                source_action_id=source,
                target_action_id=target,
                arm_label=arm,
                parent_transport_prediction_id=parent.prediction_id,
                construction_id=protocol.arm_construction_ids[arm],
                prediction_artifact_id=_stable(
                    "placebo", session, source, target, arm
                ),
                prediction_batch_id=_id(40),
                commit_id=registration.bayesian_phystwin_revision,
                candidate_selected=True,
                exact_fallback=False,
                prediction_sealed_before_target=True,
            )
            rows.append(
                CrossActionPlaceboScoreRowV1(
                    prediction=prediction,
                    target_outcome_id=transport_row.target_outcome_id,
                    target_access_attestation_id=_id(32),
                    scorer_id=_id(33),
                    proper_score=6.0 if physical else 9.0,
                )
            )
    return tuple(rows)


def test_joint_result_requires_transport_and_placebo_success() -> None:
    registration = _registration(_protocol_document())
    transport_rows = _transport_rows(registration)
    transport = CrossActionTransportResultV2(
        protocol=registration.transport_protocol,
        score_rows=transport_rows,
        target_accounting_id=_id(41),
    )
    placebo = CrossActionPlaceboResultV2(
        protocol=registration.placebo_protocol,
        score_rows=_placebo_rows(registration, transport_rows),
        target_accounting_id=_id(41),
    )
    result = Causal4DJointTransportResultV1(
        registration=registration,
        transport_result=transport,
        placebo_result=placebo,
    )
    assert result.decision is JointTransportDecision.SUPPORTED
    assert result.supports_physical_transport
    assert result.descriptor()["independent_session_count"] == 18
