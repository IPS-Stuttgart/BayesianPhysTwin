from __future__ import annotations

from dataclasses import replace

import pytest

from bayesian_phystwin.prospective_study_lifecycle_v1 import (
    ProspectiveStudyProtocolV1,
    ProspectiveStudyStateV1,
    advance_prospective_study,
    lock_prospective_study,
    validate_prospective_study_chain,
)


def _digest(character: str) -> str:
    return character * 64


def _protocol() -> ProspectiveStudyProtocolV1:
    return ProspectiveStudyProtocolV1(
        protocol_id="prospective-example-v1",
        method_set_id=_digest("1"),
        decision_rule_id=_digest("2"),
        fallback_identity_id=_digest("3"),
        information_boundary_id=_digest("4"),
        statistical_unit="physical object session",
        development_group_ids=("source-b", "source-a"),
        calibration_group_ids=("calibration-a",),
        target_group_ids=("target-b", "target-a"),
        metadata={"owner": "BayesianPhysTwin"},
    )


def _happy_chain(
    *,
    positive: bool = True,
) -> tuple[
    ProspectiveStudyProtocolV1,
    list[ProspectiveStudyStateV1],
]:
    protocol = _protocol()
    states = [
        lock_prospective_study(
            protocol,
            metadata={"receipt": "design"},
        )
    ]
    terminal = "terminal-positive" if positive else "terminal-negative"
    for stage, character in (
        ("source-predictions-sealed", "5"),
        ("source-scored", "6"),
        ("target-authorized", "7"),
        ("target-predictions-sealed", "8"),
        ("target-scored", "9"),
        (terminal, "a"),
    ):
        states.append(
            advance_prospective_study(
                states[-1],
                next_stage=stage,  # type: ignore[arg-type]
                artifact_id=_digest(character),
                metadata={"stage": stage},
            )
        )
    return protocol, states


def test_protocol_normalizes_and_content_addresses_disjoint_rosters() -> None:
    protocol = _protocol()
    assert protocol.development_group_ids == ("source-a", "source-b")
    assert protocol.target_group_ids == ("target-a", "target-b")
    assert len(protocol.protocol_content_id) == 64
    restored = ProspectiveStudyProtocolV1.from_mapping(protocol.as_dict())
    assert restored == protocol


def test_protocol_rejects_overlapping_or_empty_source_rosters() -> None:
    with pytest.raises(ValueError, match="appears in both"):
        replace(_protocol(), target_group_ids=("source-a",))
    with pytest.raises(
        ValueError,
        match="at least one development or calibration",
    ):
        replace(
            _protocol(),
            development_group_ids=(),
            calibration_group_ids=(),
        )


def test_design_lock_is_target_closed_and_content_addressed() -> None:
    protocol = _protocol()
    state = lock_prospective_study(protocol)
    assert state.stage == "design-locked"
    assert state.sequence_number == 0
    assert state.previous_state_id is None
    assert not state.target_payload_opened
    assert not state.target_outcomes_opened
    assert not state.claim_authorized
    assert ProspectiveStudyStateV1.from_mapping(state.as_dict()) == state


def test_complete_positive_chain_seals_predictions_before_outcomes() -> None:
    protocol, states = _happy_chain()
    assert states[3].stage == "target-authorized"
    assert not states[3].target_payload_opened
    assert states[4].target_payload_opened
    assert not states[4].target_outcomes_opened
    assert states[5].target_outcomes_opened
    assert states[-1].positive_result
    assert not states[-1].claim_authorized
    assert states[-1].terminal
    validate_prospective_study_chain(protocol, states)


def test_complete_negative_chain_never_authorizes_claim() -> None:
    protocol, states = _happy_chain(positive=False)
    assert states[-1].stage == "terminal-negative"
    assert not states[-1].claim_authorized
    validate_prospective_study_chain(protocol, states)


def test_source_negative_can_stop_before_predictions_without_target_access() -> None:
    protocol = _protocol()
    locked = lock_prospective_study(protocol)
    stopped = advance_prospective_study(
        locked,
        next_stage="terminal-source-negative",
        artifact_id=_digest("5"),
        metadata={"reason": "support-negative"},
    )
    assert stopped.terminal
    assert not stopped.target_payload_opened
    assert not stopped.target_outcomes_opened
    assert not stopped.claim_authorized
    validate_prospective_study_chain(protocol, [locked, stopped])


def test_illegal_skip_to_target_authorization_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="illegal prospective-study transition",
    ):
        advance_prospective_study(
            lock_prospective_study(_protocol()),
            next_stage="target-authorized",
            artifact_id=_digest("5"),
        )


def test_terminal_state_cannot_advance() -> None:
    _, states = _happy_chain()
    with pytest.raises(
        ValueError,
        match="illegal prospective-study transition",
    ):
        advance_prospective_study(
            states[-1],
            next_stage="terminal-technical",
            artifact_id=_digest("b"),
        )


def test_technical_terminal_records_early_payload_access_without_claim() -> None:
    protocol = _protocol()
    locked = lock_prospective_study(protocol)
    technical = advance_prospective_study(
        locked,
        next_stage="terminal-technical",
        artifact_id=_digest("5"),
        target_payload_opened=True,
        target_outcomes_opened=False,
        metadata={"reason": "payload-opened-before-authorization"},
    )
    assert technical.target_payload_opened
    assert not technical.target_outcomes_opened
    assert not technical.claim_authorized
    validate_prospective_study_chain(protocol, [locked, technical])


def test_technical_terminal_cannot_reclose_protected_data() -> None:
    _, states = _happy_chain(positive=False)
    scored = states[-2]
    with pytest.raises(ValueError, match="cannot close"):
        advance_prospective_study(
            scored,
            next_stage="terminal-technical",
            artifact_id=_digest("b"),
            target_payload_opened=False,
            target_outcomes_opened=False,
        )


def test_direct_state_construction_rejects_outcomes_without_payload() -> None:
    protocol = _protocol()
    with pytest.raises(ValueError, match="outcomes cannot be open"):
        ProspectiveStudyStateV1(
            protocol_id=protocol.protocol_id,
            protocol_content_id=protocol.protocol_content_id,
            stage="terminal-technical",
            sequence_number=1,
            previous_state_id=_digest("5"),
            terminal_decision_id=_digest("6"),
            target_payload_opened=False,
            target_outcomes_opened=True,
        )


def test_direct_state_construction_rejects_claim_authorization() -> None:
    protocol = _protocol()
    with pytest.raises(ValueError, match="cannot authorize paper claims"):
        ProspectiveStudyStateV1(
            protocol_id=protocol.protocol_id,
            protocol_content_id=protocol.protocol_content_id,
            stage="terminal-technical",
            sequence_number=1,
            previous_state_id=_digest("5"),
            terminal_decision_id=_digest("6"),
            claim_authorized=True,
        )


def test_direct_positive_state_requires_complete_target_lineage() -> None:
    protocol = _protocol()
    with pytest.raises(
        ValueError,
        match="requires source_prediction_bundle_id",
    ):
        ProspectiveStudyStateV1(
            protocol_id=protocol.protocol_id,
            protocol_content_id=protocol.protocol_content_id,
            stage="terminal-positive",
            sequence_number=1,
            previous_state_id=_digest("5"),
            terminal_decision_id=_digest("6"),
            target_payload_opened=True,
            target_outcomes_opened=True,
        )


def test_state_roundtrip_rejects_tampered_content_identity() -> None:
    state = lock_prospective_study(_protocol())
    payload = state.as_dict()
    payload["metadata"] = {"tampered": True}
    with pytest.raises(ValueError, match="content identity mismatch"):
        ProspectiveStudyStateV1.from_mapping(payload)


def test_protocol_roundtrip_rejects_unknown_field() -> None:
    payload = _protocol().as_dict()
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="fields changed"):
        ProspectiveStudyProtocolV1.from_mapping(payload)


def test_schema_version_rejects_boolean_coercion() -> None:
    payload = _protocol().as_dict()
    payload["schema_version"] = True
    with pytest.raises(ValueError, match="schema version"):
        ProspectiveStudyProtocolV1.from_mapping(payload)


def test_metadata_is_defensively_frozen() -> None:
    protocol = _protocol()
    with pytest.raises(TypeError, match="immutable"):
        protocol.metadata["owner"] = "changed"  # type: ignore[index]


def test_chain_validator_rejects_artifact_replacement() -> None:
    protocol, states = _happy_chain()
    forged = replace(
        states[2],
        source_prediction_bundle_id=_digest("b"),
    )
    with pytest.raises(
        ValueError,
        match="changed ancestry, artifacts, or access state",
    ):
        validate_prospective_study_chain(
            protocol,
            [states[0], states[1], forged],
        )


def test_chain_validator_rejects_foreign_protocol() -> None:
    protocol, states = _happy_chain()
    foreign = replace(
        protocol,
        protocol_id="foreign-protocol-v1",
    )
    with pytest.raises(
        ValueError,
        match="does not bind the supplied protocol",
    ):
        validate_prospective_study_chain(foreign, states)


def test_artifact_ids_fail_closed_without_string_coercion() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        advance_prospective_study(
            lock_prospective_study(_protocol()),
            next_stage="source-predictions-sealed",
            artifact_id=123,  # type: ignore[arg-type]
        )
