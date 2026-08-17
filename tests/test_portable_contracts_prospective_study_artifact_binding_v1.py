from __future__ import annotations

from dataclasses import replace

import pytest

from bayesian_phystwin.prospective_study_artifact_binding_v1 import (
    ProspectiveStudyArtifactBindingV1,
    advance_role_bound_prospective_study,
    artifact_role_for_stage,
    bind_prospective_artifact,
    validate_role_bound_prospective_study_chain,
)
from bayesian_phystwin.prospective_study_lifecycle_v1 import (
    ProspectiveStudyProtocolV1,
    ProspectiveStudyStateV1,
    advance_prospective_study,
    lock_prospective_study,
)


def _digest(character: str) -> str:
    return character * 64


def _protocol() -> ProspectiveStudyProtocolV1:
    return ProspectiveStudyProtocolV1(
        protocol_id="prospective-binding-example-v1",
        method_set_id=_digest("1"),
        decision_rule_id=_digest("2"),
        fallback_identity_id=_digest("3"),
        information_boundary_id=_digest("4"),
        statistical_unit="physical object session",
        development_group_ids=("source-a",),
        calibration_group_ids=("calibration-a",),
        target_group_ids=("target-a",),
    )


def _bound_chain() -> tuple[
    ProspectiveStudyProtocolV1,
    list[ProspectiveStudyStateV1],
    list[ProspectiveStudyArtifactBindingV1],
]:
    protocol = _protocol()
    states = [lock_prospective_study(protocol)]
    bindings: list[ProspectiveStudyArtifactBindingV1] = []
    for stage, character, schema_name in (
        ("source-predictions-sealed", "5", "example.source-predictions-v1"),
        ("source-scored", "6", "example.source-scores-v1"),
        ("target-authorized", "7", "example.target-authorization-v1"),
        ("target-predictions-sealed", "8", "example.target-predictions-v1"),
        ("target-scored", "9", "example.target-scores-v1"),
        ("terminal-positive", "a", "example.terminal-decision-v1"),
    ):
        state, binding = advance_role_bound_prospective_study(
            states[-1],
            next_stage=stage,  # type: ignore[arg-type]
            artifact_content_id=_digest(character),
            artifact_schema_name=schema_name,
            artifact_schema_version=1,
            binding_metadata={"stage": stage},
            state_metadata={"binding_id": "external-recorded"},
        )
        states.append(state)
        bindings.append(binding)
    return protocol, states, bindings


def test_binding_identity_is_domain_separated_by_stage_and_role() -> None:
    protocol = _protocol()
    source = bind_prospective_artifact(
        protocol,
        stage="source-predictions-sealed",
        artifact_content_id=_digest("5"),
        artifact_schema_name="example.bundle-v1",
        artifact_schema_version=1,
    )
    score = bind_prospective_artifact(
        protocol,
        stage="source-scored",
        artifact_content_id=_digest("5"),
        artifact_schema_name="example.bundle-v1",
        artifact_schema_version=1,
    )
    assert source.artifact_content_id == score.artifact_content_id
    assert source.binding_id != score.binding_id
    assert source.artifact_role == "source-prediction-bundle"
    assert score.artifact_role == "source-score-bundle"


def test_binding_roundtrip_recomputes_identity() -> None:
    binding = bind_prospective_artifact(
        _protocol(),
        stage="target-authorized",
        artifact_content_id=_digest("5"),
        artifact_schema_name="example.authorization-v1",
        artifact_schema_version=3,
        metadata={"approved_by": "independent-verifier"},
    )
    restored = ProspectiveStudyArtifactBindingV1.from_mapping(binding.as_dict())
    assert restored == binding
    assert restored.binding_id == binding.binding_id


def test_binding_rejects_role_that_does_not_match_stage() -> None:
    protocol = _protocol()
    with pytest.raises(ValueError, match="requires artifact role"):
        ProspectiveStudyArtifactBindingV1(
            protocol_id=protocol.protocol_id,
            protocol_content_id=protocol.protocol_content_id,
            stage="target-authorized",
            artifact_role="source-score-bundle",
            artifact_content_id=_digest("5"),
            artifact_schema_name="example.authorization-v1",
            artifact_schema_version=1,
        )


def test_binding_rejects_design_stage_and_boolean_schema_version() -> None:
    protocol = _protocol()
    with pytest.raises(ValueError, match="unsupported role-bound"):
        bind_prospective_artifact(
            protocol,
            stage="design-locked",
            artifact_content_id=_digest("5"),
            artifact_schema_name="example.design-v1",
            artifact_schema_version=1,
        )
    with pytest.raises(ValueError, match="positive integer"):
        bind_prospective_artifact(
            protocol,
            stage="source-predictions-sealed",
            artifact_content_id=_digest("5"),
            artifact_schema_name="example.source-v1",
            artifact_schema_version=True,  # type: ignore[arg-type]
        )


def test_binding_rejects_tampered_content_identity() -> None:
    binding = bind_prospective_artifact(
        _protocol(),
        stage="source-predictions-sealed",
        artifact_content_id=_digest("5"),
        artifact_schema_name="example.source-v1",
        artifact_schema_version=1,
    )
    payload = binding.as_dict()
    payload["metadata"] = {"tampered": True}
    with pytest.raises(ValueError, match="identity mismatch"):
        ProspectiveStudyArtifactBindingV1.from_mapping(payload)


def test_binding_rejects_unknown_fields_and_noncanonical_text() -> None:
    binding = bind_prospective_artifact(
        _protocol(),
        stage="source-predictions-sealed",
        artifact_content_id=_digest("5"),
        artifact_schema_name="example.source-v1",
        artifact_schema_version=1,
    )
    payload = binding.as_dict()
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="fields changed"):
        ProspectiveStudyArtifactBindingV1.from_mapping(payload)
    with pytest.raises(ValueError, match="canonical single-line"):
        replace(binding, artifact_schema_name=" example.source-v1")


def test_complete_role_bound_chain_validates() -> None:
    protocol, states, bindings = _bound_chain()
    validate_role_bound_prospective_study_chain(
        protocol,
        states,
        bindings,
    )
    for state, binding in zip(states[1:], bindings, strict=True):
        field_name = {
            "source-predictions-sealed": "source_prediction_bundle_id",
            "source-scored": "source_score_bundle_id",
            "target-authorized": "target_authorization_id",
            "target-predictions-sealed": "target_prediction_bundle_id",
            "target-scored": "target_score_bundle_id",
            "terminal-positive": "terminal_decision_id",
        }[state.stage]
        assert getattr(state, field_name) == binding.binding_id


def test_role_bound_validator_rejects_raw_content_id_in_state() -> None:
    protocol = _protocol()
    locked = lock_prospective_study(protocol)
    binding = bind_prospective_artifact(
        protocol,
        stage="source-predictions-sealed",
        artifact_content_id=_digest("5"),
        artifact_schema_name="example.source-v1",
        artifact_schema_version=1,
    )
    raw_state = advance_prospective_study(
        locked,
        next_stage="source-predictions-sealed",
        artifact_id=binding.artifact_content_id,
    )
    with pytest.raises(ValueError, match="role-binding identity"):
        validate_role_bound_prospective_study_chain(
            protocol,
            [locked, raw_state],
            [binding],
        )


def test_role_bound_validator_rejects_binding_order_and_count() -> None:
    protocol, states, bindings = _bound_chain()
    with pytest.raises(ValueError, match="exactly one binding"):
        validate_role_bound_prospective_study_chain(
            protocol,
            states,
            bindings[:-1],
        )
    with pytest.raises(ValueError, match="stage does not match"):
        validate_role_bound_prospective_study_chain(
            protocol,
            states,
            [bindings[1], bindings[0], *bindings[2:]],
        )


def test_role_bound_validator_rejects_foreign_protocol_binding() -> None:
    protocol, states, bindings = _bound_chain()
    foreign_protocol = replace(protocol, protocol_id="foreign-protocol-v1")
    foreign = replace(
        bindings[0],
        protocol_id=foreign_protocol.protocol_id,
        protocol_content_id=foreign_protocol.protocol_content_id,
    )
    with pytest.raises(ValueError, match="does not bind the supplied protocol"):
        validate_role_bound_prospective_study_chain(
            protocol,
            states,
            [foreign, *bindings[1:]],
        )


def test_role_bound_validator_rejects_raw_content_reuse_across_roles() -> None:
    protocol = _protocol()
    states = [lock_prospective_study(protocol)]
    first_state, first_binding = advance_role_bound_prospective_study(
        states[-1],
        next_stage="source-predictions-sealed",
        artifact_content_id=_digest("5"),
        artifact_schema_name="example.source-v1",
        artifact_schema_version=1,
    )
    states.append(first_state)
    second_state, second_binding = advance_role_bound_prospective_study(
        states[-1],
        next_stage="source-scored",
        artifact_content_id=_digest("5"),
        artifact_schema_name="example.score-v1",
        artifact_schema_version=1,
    )
    states.append(second_state)
    assert first_binding.binding_id != second_binding.binding_id
    with pytest.raises(ValueError, match="raw artifact content identity was reused"):
        validate_role_bound_prospective_study_chain(
            protocol,
            states,
            [first_binding, second_binding],
        )


def test_technical_terminal_preserves_explicit_access_flags() -> None:
    protocol = _protocol()
    locked = lock_prospective_study(protocol)
    technical, binding = advance_role_bound_prospective_study(
        locked,
        next_stage="terminal-technical",
        artifact_content_id=_digest("5"),
        artifact_schema_name="example.technical-decision-v1",
        artifact_schema_version=1,
        target_payload_opened=True,
        target_outcomes_opened=False,
    )
    assert technical.target_payload_opened
    assert not technical.target_outcomes_opened
    assert binding.artifact_role == "terminal-decision"
    validate_role_bound_prospective_study_chain(
        protocol,
        [locked, technical],
        [binding],
    )


def test_stage_role_registry_is_explicit_and_fail_closed() -> None:
    assert artifact_role_for_stage("target-scored") == "target-score-bundle"
    with pytest.raises(ValueError, match="unsupported role-bound"):
        artifact_role_for_stage("unknown-stage")  # type: ignore[arg-type]


def test_binding_from_mapping_rejects_contract_header_changes() -> None:
    binding = bind_prospective_artifact(
        _protocol(),
        stage="source-predictions-sealed",
        artifact_content_id=_digest("5"),
        artifact_schema_name="example.source-v1",
        artifact_schema_version=1,
    )
    for field, value, message in (
        ("binding_domain", "other-domain", "unexpected.*domain"),
        ("schema_name", "other-schema", "unexpected.*schema"),
        ("schema_version", 2, "schema version"),
    ):
        payload = binding.as_dict()
        payload[field] = value
        with pytest.raises(ValueError, match=message):
            ProspectiveStudyArtifactBindingV1.from_mapping(payload)


def test_binding_from_mapping_rejects_invalid_role_and_metadata() -> None:
    binding = bind_prospective_artifact(
        _protocol(),
        stage="source-predictions-sealed",
        artifact_content_id=_digest("5"),
        artifact_schema_name="example.source-v1",
        artifact_schema_version=1,
    )
    payload = binding.as_dict()
    payload["artifact_role"] = "unknown-role"
    with pytest.raises(ValueError, match="unsupported.*role"):
        ProspectiveStudyArtifactBindingV1.from_mapping(payload)
    payload = binding.as_dict()
    payload["metadata"] = []
    with pytest.raises(ValueError, match="literal string object keys"):
        ProspectiveStudyArtifactBindingV1.from_mapping(payload)


def test_binding_helpers_reject_wrong_object_types() -> None:
    with pytest.raises(TypeError, match="protocol must be"):
        bind_prospective_artifact(
            object(),  # type: ignore[arg-type]
            stage="source-predictions-sealed",
            artifact_content_id=_digest("5"),
            artifact_schema_name="example.source-v1",
            artifact_schema_version=1,
        )
    with pytest.raises(TypeError, match="state must be"):
        advance_role_bound_prospective_study(
            object(),  # type: ignore[arg-type]
            next_stage="source-predictions-sealed",
            artifact_content_id=_digest("5"),
            artifact_schema_name="example.source-v1",
            artifact_schema_version=1,
        )


def test_role_bound_validator_rejects_invalid_binding_sequences() -> None:
    protocol = _protocol()
    locked = lock_prospective_study(protocol)
    with pytest.raises(ValueError, match="bindings must be a sequence"):
        validate_role_bound_prospective_study_chain(
            protocol,
            [locked],
            "not-bindings",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="must contain"):
        validate_role_bound_prospective_study_chain(
            protocol,
            [locked],
            [object()],  # type: ignore[list-item]
        )
