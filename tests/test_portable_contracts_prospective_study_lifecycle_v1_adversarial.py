from __future__ import annotations

from dataclasses import fields
from typing import Any

import pytest

from bayesian_phystwin.prospective_study_lifecycle_v1 import (
    PROSPECTIVE_STUDY_PROTOCOL_SCHEMA,
    PROSPECTIVE_STUDY_STATE_SCHEMA,
    ProspectiveStudyProtocolV1,
    ProspectiveStudyStateV1,
    advance_prospective_study,
    lock_prospective_study,
    validate_prospective_study_chain,
)


def _digest(character: str) -> str:
    return character * 64


def _protocol_kwargs() -> dict[str, Any]:
    return {
        "protocol_id": "prospective-adversarial-v1",
        "method_set_id": _digest("1"),
        "decision_rule_id": _digest("2"),
        "fallback_identity_id": _digest("3"),
        "information_boundary_id": _digest("4"),
        "statistical_unit": "physical object session",
        "development_group_ids": ("source-a",),
        "calibration_group_ids": ("calibration-a",),
        "target_group_ids": ("target-a",),
        "metadata": {"owner": "BayesianPhysTwin"},
    }


def _protocol() -> ProspectiveStudyProtocolV1:
    return ProspectiveStudyProtocolV1(**_protocol_kwargs())


def _state_kwargs(
    *,
    stage: str,
    sequence_number: int = 1,
    previous_state_id: str | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    protocol = _protocol()
    values: dict[str, Any] = {
        "protocol_id": protocol.protocol_id,
        "protocol_content_id": protocol.protocol_content_id,
        "stage": stage,
        "sequence_number": sequence_number,
        "previous_state_id": previous_state_id or _digest("5"),
    }
    values.update(overrides)
    return values


def _source_scored_state() -> ProspectiveStudyStateV1:
    return ProspectiveStudyStateV1(
        **_state_kwargs(
            stage="source-scored",
            source_prediction_bundle_id=_digest("6"),
            source_score_bundle_id=_digest("7"),
        )
    )


def _target_scored_state() -> ProspectiveStudyStateV1:
    return ProspectiveStudyStateV1(
        **_state_kwargs(
            stage="target-scored",
            source_prediction_bundle_id=_digest("6"),
            source_score_bundle_id=_digest("7"),
            target_authorization_id=_digest("8"),
            target_prediction_bundle_id=_digest("9"),
            target_score_bundle_id=_digest("a"),
            target_payload_opened=True,
            target_outcomes_opened=True,
        )
    )


def _forge(
    state: ProspectiveStudyStateV1,
    **changes: Any,
) -> ProspectiveStudyStateV1:
    forged = object.__new__(ProspectiveStudyStateV1)
    for data_field in fields(ProspectiveStudyStateV1):
        object.__setattr__(forged, data_field.name, getattr(state, data_field.name))
    for name, value in changes.items():
        object.__setattr__(forged, name, value)
    return forged


@pytest.mark.parametrize(
    "field_name,value,match",
    (
        ("protocol_id", " leading", "canonical single-line"),
        ("protocol_id", "line\nbreak", "canonical single-line"),
        ("development_group_ids", "source-a", "must be a JSON array"),
        ("metadata", {1: "invalid"}, "literal string object keys"),
    ),
)
def test_protocol_mapping_rejects_noncanonical_values(
    field_name: str,
    value: object,
    match: str,
) -> None:
    payload = _protocol().as_dict()
    payload[field_name] = value
    with pytest.raises(ValueError, match=match):
        ProspectiveStudyProtocolV1.from_mapping(payload)


def test_protocol_mapping_rejects_wrong_schema_and_content_identity() -> None:
    wrong_schema = _protocol().as_dict()
    wrong_schema["schema_name"] = "foreign-schema"
    with pytest.raises(ValueError, match="unexpected.*protocol schema"):
        ProspectiveStudyProtocolV1.from_mapping(wrong_schema)

    wrong_identity = _protocol().as_dict()
    wrong_identity["protocol_content_id"] = _digest("f")
    with pytest.raises(ValueError, match="content identity mismatch"):
        ProspectiveStudyProtocolV1.from_mapping(wrong_identity)


def test_state_rejects_invalid_boolean_integer_and_stage_types() -> None:
    with pytest.raises(ValueError, match="target_payload_opened must be boolean"):
        ProspectiveStudyStateV1(
            **_state_kwargs(
                stage="terminal-technical",
                terminal_decision_id=_digest("6"),
                target_payload_opened=1,
            )
        )
    with pytest.raises(ValueError, match="nonnegative integer"):
        ProspectiveStudyStateV1(
            **_state_kwargs(
                stage="terminal-technical",
                sequence_number=-1,
                terminal_decision_id=_digest("6"),
            )
        )
    with pytest.raises(ValueError, match="unsupported prospective-study stage"):
        ProspectiveStudyStateV1(
            **_state_kwargs(
                stage="unknown-stage",
                terminal_decision_id=_digest("6"),
            )
        )


def test_noninitial_and_technical_states_require_lineage_and_decision() -> None:
    with pytest.raises(ValueError, match="require a predecessor"):
        ProspectiveStudyStateV1(
            **_state_kwargs(
                stage="source-predictions-sealed",
                sequence_number=0,
                previous_state_id=None,
                source_prediction_bundle_id=_digest("6"),
            )
        )
    with pytest.raises(ValueError, match="requires a terminal decision"):
        ProspectiveStudyStateV1(**_state_kwargs(stage="terminal-technical"))


@pytest.mark.parametrize(
    "overrides,match",
    (
        ({"sequence_number": 1}, "must be the first"),
        ({"source_prediction_bundle_id": _digest("6")}, "execution artifacts"),
        ({"terminal_decision_id": _digest("6")}, "terminal decision"),
        ({"target_payload_opened": True}, "protected target data closed"),
    ),
)
def test_design_lock_rejects_execution_or_access_state(
    overrides: dict[str, Any],
    match: str,
) -> None:
    protocol = _protocol()
    values: dict[str, Any] = {
        "protocol_id": protocol.protocol_id,
        "protocol_content_id": protocol.protocol_content_id,
        "stage": "design-locked",
        "sequence_number": 0,
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=match):
        ProspectiveStudyStateV1(**values)


def test_state_rejects_noncontiguous_or_excess_artifacts() -> None:
    with pytest.raises(ValueError, match="contiguous prefix"):
        ProspectiveStudyStateV1(
            **_state_kwargs(
                stage="target-authorized",
                source_prediction_bundle_id=_digest("6"),
                target_authorization_id=_digest("8"),
            )
        )
    with pytest.raises(ValueError, match="too many execution artifacts"):
        ProspectiveStudyStateV1(
            **_state_kwargs(
                stage="source-predictions-sealed",
                source_prediction_bundle_id=_digest("6"),
                source_score_bundle_id=_digest("7"),
            )
        )


def test_state_rejects_wrong_access_and_terminal_shape() -> None:
    with pytest.raises(ValueError, match="inconsistent target-access state"):
        ProspectiveStudyStateV1(
            **_state_kwargs(
                stage="target-predictions-sealed",
                source_prediction_bundle_id=_digest("6"),
                source_score_bundle_id=_digest("7"),
                target_authorization_id=_digest("8"),
                target_prediction_bundle_id=_digest("9"),
            )
        )
    with pytest.raises(ValueError, match="cannot bind terminal_decision_id"):
        ProspectiveStudyStateV1(
            **_state_kwargs(
                stage="source-predictions-sealed",
                source_prediction_bundle_id=_digest("6"),
                terminal_decision_id=_digest("b"),
            )
        )


def test_source_negative_rejects_missing_decision_artifacts_and_access() -> None:
    with pytest.raises(ValueError, match="requires a terminal decision"):
        ProspectiveStudyStateV1(
            **_state_kwargs(stage="terminal-source-negative")
        )
    with pytest.raises(ValueError, match="cannot bind target artifacts"):
        ProspectiveStudyStateV1(
            **_state_kwargs(
                stage="terminal-source-negative",
                source_prediction_bundle_id=_digest("6"),
                source_score_bundle_id=_digest("7"),
                target_authorization_id=_digest("8"),
                terminal_decision_id=_digest("b"),
            )
        )
    with pytest.raises(ValueError, match="protected target data closed"):
        ProspectiveStudyStateV1(
            **_state_kwargs(
                stage="terminal-source-negative",
                terminal_decision_id=_digest("b"),
                target_payload_opened=True,
            )
        )


def test_state_mapping_rejects_wrong_schema_and_content_identity() -> None:
    locked = lock_prospective_study(_protocol())
    wrong_schema = locked.as_dict()
    wrong_schema["schema_name"] = "foreign-schema"
    with pytest.raises(ValueError, match="unexpected.*state schema"):
        ProspectiveStudyStateV1.from_mapping(wrong_schema)

    wrong_identity = locked.as_dict()
    wrong_identity["state_id"] = _digest("f")
    with pytest.raises(ValueError, match="content identity mismatch"):
        ProspectiveStudyStateV1.from_mapping(wrong_identity)


def test_lock_and_advance_reject_wrong_contract_types() -> None:
    with pytest.raises(TypeError, match="ProspectiveStudyProtocolV1"):
        lock_prospective_study("not-a-protocol")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ProspectiveStudyStateV1"):
        advance_prospective_study(  # type: ignore[arg-type]
            "not-a-state",
            next_stage="terminal-technical",
            artifact_id=_digest("6"),
        )


def test_advance_rejects_artifact_replacement_in_corrupted_state() -> None:
    scored = _source_scored_state()
    corrupted = _forge(scored, target_authorization_id=_digest("8"))
    with pytest.raises(ValueError, match="replace existing target_authorization_id"):
        advance_prospective_study(
            corrupted,
            next_stage="target-authorized",
            artifact_id=_digest("9"),
        )


def test_nontechnical_transition_rejects_explicit_access_flags() -> None:
    with pytest.raises(ValueError, match="require terminal-technical"):
        advance_prospective_study(
            lock_prospective_study(_protocol()),
            next_stage="source-predictions-sealed",
            artifact_id=_digest("6"),
            target_payload_opened=False,
        )


def test_technical_transition_inherits_access_and_validates_flags() -> None:
    scored = _target_scored_state()
    technical = advance_prospective_study(
        scored,
        next_stage="terminal-technical",
        artifact_id=_digest("b"),
    )
    assert technical.target_payload_opened
    assert technical.target_outcomes_opened

    with pytest.raises(ValueError, match="target_payload_opened must be boolean"):
        advance_prospective_study(
            lock_prospective_study(_protocol()),
            next_stage="terminal-technical",
            artifact_id=_digest("6"),
            target_payload_opened=1,
        )


def test_chain_validator_rejects_invalid_arguments_and_missing_artifact() -> None:
    protocol = _protocol()
    locked = lock_prospective_study(protocol)
    with pytest.raises(TypeError, match="ProspectiveStudyProtocolV1"):
        validate_prospective_study_chain(  # type: ignore[arg-type]
            "not-a-protocol",
            [locked],
        )
    for invalid in ([], "not-a-sequence"):
        with pytest.raises(ValueError, match="nonempty sequence"):
            validate_prospective_study_chain(protocol, invalid)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must contain"):
        validate_prospective_study_chain(
            protocol,
            [locked, "not-a-state"],  # type: ignore[list-item]
        )

    source_scored = _source_scored_state()
    missing_score = _forge(source_scored, source_score_bundle_id=None)
    with pytest.raises(ValueError, match="lacks source_score_bundle_id"):
        validate_prospective_study_chain(protocol, [locked, missing_score])


def test_exported_schema_names_remain_exact() -> None:
    assert PROSPECTIVE_STUDY_PROTOCOL_SCHEMA.endswith("protocol-v1")
    assert PROSPECTIVE_STUDY_STATE_SCHEMA.endswith("state-v1")
