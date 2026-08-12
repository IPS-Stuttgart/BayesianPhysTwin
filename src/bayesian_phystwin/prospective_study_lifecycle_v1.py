"""Content-addressed, target-closed lifecycle for prospective studies.

The contract records information access and immutable artifact handoffs. It does
not implement a scientific method, choose a candidate, score an outcome, or
authorize a paper claim by itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Final, Literal, cast

from ._canonical_contracts import frozen_finite_json_mapping, plain_json
from ._portable_contracts import (
    canonical_sorted_strings,
    content_id,
    nonempty_string,
    require_exact_fields,
    sha256_digest,
)

PROSPECTIVE_STUDY_PROTOCOL_SCHEMA: Final = (
    "bayesian-phystwin.prospective-study-protocol-v1"
)
PROSPECTIVE_STUDY_STATE_SCHEMA: Final = "bayesian-phystwin.prospective-study-state-v1"
PROSPECTIVE_STUDY_SCHEMA_VERSION: Final = 1

StudyStageV1 = Literal[
    "design-locked",
    "source-predictions-sealed",
    "source-scored",
    "target-authorized",
    "target-predictions-sealed",
    "target-scored",
    "terminal-source-negative",
    "terminal-positive",
    "terminal-negative",
    "terminal-technical",
]

_DESIGN: Final[StudyStageV1] = "design-locked"
_SOURCE_PREDICTIONS: Final[StudyStageV1] = "source-predictions-sealed"
_SOURCE_SCORED: Final[StudyStageV1] = "source-scored"
_TARGET_AUTHORIZED: Final[StudyStageV1] = "target-authorized"
_TARGET_PREDICTIONS: Final[StudyStageV1] = "target-predictions-sealed"
_TARGET_SCORED: Final[StudyStageV1] = "target-scored"
_SOURCE_NEGATIVE: Final[StudyStageV1] = "terminal-source-negative"
_POSITIVE: Final[StudyStageV1] = "terminal-positive"
_NEGATIVE: Final[StudyStageV1] = "terminal-negative"
_TECHNICAL: Final[StudyStageV1] = "terminal-technical"

_STAGES: Final = frozenset(
    {
        _DESIGN,
        _SOURCE_PREDICTIONS,
        _SOURCE_SCORED,
        _TARGET_AUTHORIZED,
        _TARGET_PREDICTIONS,
        _TARGET_SCORED,
        _SOURCE_NEGATIVE,
        _POSITIVE,
        _NEGATIVE,
        _TECHNICAL,
    }
)
_TERMINAL_STAGES: Final = frozenset(
    {_SOURCE_NEGATIVE, _POSITIVE, _NEGATIVE, _TECHNICAL}
)
_TRANSITIONS: Final[Mapping[StudyStageV1, frozenset[StudyStageV1]]] = {
    _DESIGN: frozenset({_SOURCE_PREDICTIONS, _SOURCE_NEGATIVE, _TECHNICAL}),
    _SOURCE_PREDICTIONS: frozenset({_SOURCE_SCORED, _SOURCE_NEGATIVE, _TECHNICAL}),
    _SOURCE_SCORED: frozenset({_TARGET_AUTHORIZED, _SOURCE_NEGATIVE, _TECHNICAL}),
    _TARGET_AUTHORIZED: frozenset({_TARGET_PREDICTIONS, _TECHNICAL}),
    _TARGET_PREDICTIONS: frozenset({_TARGET_SCORED, _TECHNICAL}),
    _TARGET_SCORED: frozenset({_POSITIVE, _NEGATIVE, _TECHNICAL}),
    _SOURCE_NEGATIVE: frozenset(),
    _POSITIVE: frozenset(),
    _NEGATIVE: frozenset(),
    _TECHNICAL: frozenset(),
}
_ARTIFACT_FIELDS: Final = (
    "source_prediction_bundle_id",
    "source_score_bundle_id",
    "target_authorization_id",
    "target_prediction_bundle_id",
    "target_score_bundle_id",
)
_STAGE_ARTIFACT: Final[Mapping[StudyStageV1, str]] = {
    _SOURCE_PREDICTIONS: "source_prediction_bundle_id",
    _SOURCE_SCORED: "source_score_bundle_id",
    _TARGET_AUTHORIZED: "target_authorization_id",
    _TARGET_PREDICTIONS: "target_prediction_bundle_id",
    _TARGET_SCORED: "target_score_bundle_id",
    _SOURCE_NEGATIVE: "terminal_decision_id",
    _POSITIVE: "terminal_decision_id",
    _NEGATIVE: "terminal_decision_id",
    _TECHNICAL: "terminal_decision_id",
}
# artifact prefix length, payload opened, outcomes opened, terminal decision required
_STANDARD_SHAPE: Final[Mapping[StudyStageV1, tuple[int, bool, bool, bool]]] = {
    _SOURCE_PREDICTIONS: (1, False, False, False),
    _SOURCE_SCORED: (2, False, False, False),
    _TARGET_AUTHORIZED: (3, False, False, False),
    _TARGET_PREDICTIONS: (4, True, False, False),
    _TARGET_SCORED: (5, True, True, False),
    _POSITIVE: (5, True, True, True),
    _NEGATIVE: (5, True, True, True),
}
_PROTOCOL_FIELDS: Final = frozenset(
    {
        "protocol_content_id",
        "schema_name",
        "schema_version",
        "protocol_id",
        "method_set_id",
        "decision_rule_id",
        "fallback_identity_id",
        "information_boundary_id",
        "statistical_unit",
        "development_group_ids",
        "calibration_group_ids",
        "target_group_ids",
        "metadata",
    }
)
_STATE_FIELDS: Final = frozenset(
    {
        "state_id",
        "schema_name",
        "schema_version",
        "protocol_id",
        "protocol_content_id",
        "stage",
        "sequence_number",
        "previous_state_id",
        *_ARTIFACT_FIELDS,
        "terminal_decision_id",
        "target_payload_opened",
        "target_outcomes_opened",
        "claim_authorized",
        "metadata",
    }
)


def _text(value: object, *, name: str) -> str:
    result = nonempty_string(value, name=name)
    if result.strip() != result or any(c in result for c in "\x00\r\n"):
        raise ValueError(f"{name} must be canonical single-line text")
    return result


def _boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean")
    return value


def _integer(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _version(value: object, *, name: str) -> int:
    if type(value) is not int or value != PROSPECTIVE_STUDY_SCHEMA_VERSION:
        raise ValueError(f"unexpected {name} schema version")
    return value


def _optional_digest(value: object, *, name: str) -> str | None:
    return None if value is None else sha256_digest(value, name=name)


def _stage(value: object) -> StudyStageV1:
    if type(value) is not str or value not in _STAGES:
        raise ValueError("unsupported prospective-study stage")
    return cast(StudyStageV1, value)


def _strings(value: object, *, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return tuple(_text(item, name=f"{name}[{i}]") for i, item in enumerate(value))


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must use literal string object keys")
    return cast(Mapping[str, Any], value)


@dataclass(frozen=True)
class ProspectiveStudyProtocolV1:
    """An immutable study design and pairwise-disjoint group roster."""

    protocol_id: str
    method_set_id: str
    decision_rule_id: str
    fallback_identity_id: str
    information_boundary_id: str
    statistical_unit: str
    development_group_ids: tuple[str, ...]
    target_group_ids: tuple[str, ...]
    calibration_group_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "protocol_id",
            _text(self.protocol_id, name="protocol_id"),
        )
        for name in (
            "method_set_id",
            "decision_rule_id",
            "fallback_identity_id",
            "information_boundary_id",
        ):
            object.__setattr__(
                self,
                name,
                sha256_digest(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "statistical_unit",
            _text(self.statistical_unit, name="statistical_unit"),
        )
        rosters = {
            "development_group_ids": canonical_sorted_strings(
                self.development_group_ids,
                name="development_group_ids",
                allow_empty=True,
            ),
            "calibration_group_ids": canonical_sorted_strings(
                self.calibration_group_ids,
                name="calibration_group_ids",
                allow_empty=True,
            ),
            "target_group_ids": canonical_sorted_strings(
                self.target_group_ids,
                name="target_group_ids",
            ),
        }
        if (
            not rosters["development_group_ids"]
            and not rosters["calibration_group_ids"]
        ):
            raise ValueError(
                "at least one development or calibration group is required"
            )
        seen: dict[str, str] = {}
        for roster_name, group_ids in rosters.items():
            for group_id in group_ids:
                if group_id in seen:
                    raise ValueError(
                        f"group {group_id!r} appears in both "
                        f"{seen[group_id]} and {roster_name}"
                    )
                seen[group_id] = roster_name
            object.__setattr__(self, roster_name, group_ids)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="protocol metadata"),
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema_name": PROSPECTIVE_STUDY_PROTOCOL_SCHEMA,
            "schema_version": PROSPECTIVE_STUDY_SCHEMA_VERSION,
            "protocol_id": self.protocol_id,
            "method_set_id": self.method_set_id,
            "decision_rule_id": self.decision_rule_id,
            "fallback_identity_id": self.fallback_identity_id,
            "information_boundary_id": self.information_boundary_id,
            "statistical_unit": self.statistical_unit,
            "development_group_ids": list(self.development_group_ids),
            "calibration_group_ids": list(self.calibration_group_ids),
            "target_group_ids": list(self.target_group_ids),
            "metadata": plain_json(self.metadata),
        }

    @property
    def protocol_content_id(self) -> str:
        return content_id(self.descriptor())

    def as_dict(self) -> dict[str, object]:
        return {"protocol_content_id": self.protocol_content_id, **self.descriptor()}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ProspectiveStudyProtocolV1:
        require_exact_fields(value, expected=_PROTOCOL_FIELDS, name="study protocol")
        if (
            _text(value["schema_name"], name="schema_name")
            != PROSPECTIVE_STUDY_PROTOCOL_SCHEMA
        ):
            raise ValueError("unexpected prospective-study protocol schema")
        _version(value["schema_version"], name="protocol")
        result = cls(
            protocol_id=_text(value["protocol_id"], name="protocol_id"),
            method_set_id=sha256_digest(value["method_set_id"], name="method_set_id"),
            decision_rule_id=sha256_digest(
                value["decision_rule_id"], name="decision_rule_id"
            ),
            fallback_identity_id=sha256_digest(
                value["fallback_identity_id"], name="fallback_identity_id"
            ),
            information_boundary_id=sha256_digest(
                value["information_boundary_id"], name="information_boundary_id"
            ),
            statistical_unit=_text(value["statistical_unit"], name="statistical_unit"),
            development_group_ids=_strings(
                value["development_group_ids"], name="development_group_ids"
            ),
            calibration_group_ids=_strings(
                value["calibration_group_ids"], name="calibration_group_ids"
            ),
            target_group_ids=_strings(
                value["target_group_ids"], name="target_group_ids"
            ),
            metadata=_mapping(value["metadata"], name="protocol metadata"),
        )
        supplied = sha256_digest(
            value["protocol_content_id"], name="protocol_content_id"
        )
        if supplied != result.protocol_content_id:
            raise ValueError("prospective-study protocol content identity mismatch")
        return result


@dataclass(frozen=True)
class ProspectiveStudyStateV1:
    """One immutable lifecycle state linked to its exact predecessor."""

    protocol_id: str
    protocol_content_id: str
    stage: StudyStageV1
    sequence_number: int
    previous_state_id: str | None = None
    source_prediction_bundle_id: str | None = None
    source_score_bundle_id: str | None = None
    target_authorization_id: str | None = None
    target_prediction_bundle_id: str | None = None
    target_score_bundle_id: str | None = None
    terminal_decision_id: str | None = None
    target_payload_opened: bool = False
    target_outcomes_opened: bool = False
    claim_authorized: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "protocol_id",
            _text(self.protocol_id, name="protocol_id"),
        )
        object.__setattr__(
            self,
            "protocol_content_id",
            sha256_digest(self.protocol_content_id, name="protocol_content_id"),
        )
        object.__setattr__(self, "stage", _stage(self.stage))
        object.__setattr__(
            self,
            "sequence_number",
            _integer(self.sequence_number, name="sequence_number"),
        )
        object.__setattr__(
            self,
            "previous_state_id",
            _optional_digest(self.previous_state_id, name="previous_state_id"),
        )
        for name in (*_ARTIFACT_FIELDS, "terminal_decision_id"):
            object.__setattr__(
                self,
                name,
                _optional_digest(getattr(self, name), name=name),
            )
        for name in (
            "target_payload_opened",
            "target_outcomes_opened",
            "claim_authorized",
        ):
            object.__setattr__(self, name, _boolean(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="state metadata"),
        )
        self._validate_shape()

    def _validate_shape(self) -> None:
        if self.target_outcomes_opened and not self.target_payload_opened:
            raise ValueError(
                "target outcomes cannot be open while target payload is closed"
            )
        if self.claim_authorized:
            raise ValueError("study lifecycle states cannot authorize paper claims")
        if self.stage == _DESIGN:
            self._validate_design()
            return
        if self.sequence_number == 0 or self.previous_state_id is None:
            raise ValueError("noninitial lifecycle states require a predecessor")
        self._validate_contiguous_artifacts()
        if self.stage in _STANDARD_SHAPE:
            prefix, payload, outcomes, terminal = _STANDARD_SHAPE[self.stage]
            self._require_shape(prefix, payload, outcomes, terminal)
        elif self.stage == _SOURCE_NEGATIVE:
            self._validate_source_negative()
        elif self.stage == _TECHNICAL and self.terminal_decision_id is None:
            raise ValueError("terminal-technical requires a terminal decision")

    def _validate_design(self) -> None:
        if self.sequence_number != 0 or self.previous_state_id is not None:
            raise ValueError("design-locked must be the first lifecycle state")
        if any(getattr(self, name) is not None for name in _ARTIFACT_FIELDS):
            raise ValueError("design-locked cannot bind execution artifacts")
        if self.terminal_decision_id is not None:
            raise ValueError("design-locked cannot bind a terminal decision")
        if self.target_payload_opened or self.target_outcomes_opened:
            raise ValueError("design-locked must keep protected target data closed")

    def _validate_contiguous_artifacts(self) -> None:
        present = [getattr(self, name) is not None for name in _ARTIFACT_FIELDS]
        expected = [True] * sum(present) + [False] * (len(present) - sum(present))
        if present != expected:
            raise ValueError("study artifacts must form one contiguous prefix")

    def _require_shape(
        self,
        prefix: int,
        payload: bool,
        outcomes: bool,
        terminal: bool,
    ) -> None:
        present = sum(getattr(self, name) is not None for name in _ARTIFACT_FIELDS)
        if present != prefix:
            missing = _ARTIFACT_FIELDS[min(present, len(_ARTIFACT_FIELDS) - 1)]
            if present < prefix:
                raise ValueError(f"{self.stage} requires {missing}")
            raise ValueError(f"{self.stage} binds too many execution artifacts")
        if (
            self.target_payload_opened != payload
            or self.target_outcomes_opened != outcomes
        ):
            raise ValueError(f"{self.stage} has inconsistent target-access state")
        if terminal != (self.terminal_decision_id is not None):
            verb = "requires" if terminal else "cannot bind"
            raise ValueError(f"{self.stage} {verb} terminal_decision_id")

    def _validate_source_negative(self) -> None:
        if self.terminal_decision_id is None:
            raise ValueError("terminal-source-negative requires a terminal decision")
        if self.target_authorization_id is not None:
            raise ValueError("source-negative state cannot bind target artifacts")
        if self.target_payload_opened or self.target_outcomes_opened:
            raise ValueError(
                "source-negative state must keep protected target data closed"
            )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema_name": PROSPECTIVE_STUDY_STATE_SCHEMA,
            "schema_version": PROSPECTIVE_STUDY_SCHEMA_VERSION,
            "protocol_id": self.protocol_id,
            "protocol_content_id": self.protocol_content_id,
            "stage": self.stage,
            "sequence_number": self.sequence_number,
            "previous_state_id": self.previous_state_id,
            "source_prediction_bundle_id": self.source_prediction_bundle_id,
            "source_score_bundle_id": self.source_score_bundle_id,
            "target_authorization_id": self.target_authorization_id,
            "target_prediction_bundle_id": self.target_prediction_bundle_id,
            "target_score_bundle_id": self.target_score_bundle_id,
            "terminal_decision_id": self.terminal_decision_id,
            "target_payload_opened": self.target_payload_opened,
            "target_outcomes_opened": self.target_outcomes_opened,
            "claim_authorized": self.claim_authorized,
            "metadata": plain_json(self.metadata),
        }

    @property
    def state_id(self) -> str:
        return content_id(self.descriptor())

    @property
    def terminal(self) -> bool:
        return self.stage in _TERMINAL_STAGES

    @property
    def positive_result(self) -> bool:
        return self.stage == _POSITIVE

    def as_dict(self) -> dict[str, object]:
        return {"state_id": self.state_id, **self.descriptor()}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ProspectiveStudyStateV1:
        require_exact_fields(value, expected=_STATE_FIELDS, name="study state")
        if (
            _text(value["schema_name"], name="schema_name")
            != PROSPECTIVE_STUDY_STATE_SCHEMA
        ):
            raise ValueError("unexpected prospective-study state schema")
        _version(value["schema_version"], name="state")
        result = cls(
            protocol_id=_text(value["protocol_id"], name="protocol_id"),
            protocol_content_id=sha256_digest(
                value["protocol_content_id"], name="protocol_content_id"
            ),
            stage=_stage(value["stage"]),
            sequence_number=_integer(value["sequence_number"], name="sequence_number"),
            previous_state_id=_optional_digest(
                value["previous_state_id"], name="previous_state_id"
            ),
            source_prediction_bundle_id=_optional_digest(
                value["source_prediction_bundle_id"], name="source_prediction_bundle_id"
            ),
            source_score_bundle_id=_optional_digest(
                value["source_score_bundle_id"], name="source_score_bundle_id"
            ),
            target_authorization_id=_optional_digest(
                value["target_authorization_id"], name="target_authorization_id"
            ),
            target_prediction_bundle_id=_optional_digest(
                value["target_prediction_bundle_id"], name="target_prediction_bundle_id"
            ),
            target_score_bundle_id=_optional_digest(
                value["target_score_bundle_id"], name="target_score_bundle_id"
            ),
            terminal_decision_id=_optional_digest(
                value["terminal_decision_id"], name="terminal_decision_id"
            ),
            target_payload_opened=_boolean(
                value["target_payload_opened"], name="target_payload_opened"
            ),
            target_outcomes_opened=_boolean(
                value["target_outcomes_opened"], name="target_outcomes_opened"
            ),
            claim_authorized=_boolean(
                value["claim_authorized"], name="claim_authorized"
            ),
            metadata=_mapping(value["metadata"], name="state metadata"),
        )
        supplied = sha256_digest(value["state_id"], name="state_id")
        if supplied != result.state_id:
            raise ValueError("prospective-study state content identity mismatch")
        return result


def lock_prospective_study(
    protocol: ProspectiveStudyProtocolV1,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> ProspectiveStudyStateV1:
    """Create the first target-closed state for one immutable protocol."""

    if not isinstance(protocol, ProspectiveStudyProtocolV1):
        raise TypeError("protocol must be a ProspectiveStudyProtocolV1")
    return ProspectiveStudyStateV1(
        protocol_id=protocol.protocol_id,
        protocol_content_id=protocol.protocol_content_id,
        stage=_DESIGN,
        sequence_number=0,
        metadata={} if metadata is None else metadata,
    )


def advance_prospective_study(
    state: ProspectiveStudyStateV1,
    *,
    next_stage: StudyStageV1,
    artifact_id: str,
    metadata: Mapping[str, Any] | None = None,
    target_payload_opened: bool | None = None,
    target_outcomes_opened: bool | None = None,
) -> ProspectiveStudyStateV1:
    """Advance exactly one legal transition without replacing artifacts."""

    if not isinstance(state, ProspectiveStudyStateV1):
        raise TypeError("state must be a ProspectiveStudyStateV1")
    destination = _stage(next_stage)
    if destination not in _TRANSITIONS[state.stage]:
        raise ValueError(
            f"illegal prospective-study transition: {state.stage} -> {destination}"
        )
    field_name = _STAGE_ARTIFACT[destination]
    if getattr(state, field_name) is not None:
        raise ValueError(f"transition would replace existing {field_name}")
    digest = sha256_digest(artifact_id, name="artifact_id")
    payload, outcomes = _access_for_transition(
        state,
        destination=destination,
        target_payload_opened=target_payload_opened,
        target_outcomes_opened=target_outcomes_opened,
    )
    common: dict[str, Any] = {
        "stage": destination,
        "sequence_number": state.sequence_number + 1,
        "previous_state_id": state.state_id,
        "target_payload_opened": payload,
        "target_outcomes_opened": outcomes,
        "claim_authorized": False,
        "metadata": {} if metadata is None else metadata,
    }
    return _advance_with_artifact(
        state,
        field_name=field_name,
        artifact_id=digest,
        common=common,
    )


def _advance_with_artifact(
    state: ProspectiveStudyStateV1,
    *,
    field_name: str,
    artifact_id: str,
    common: Mapping[str, Any],
) -> ProspectiveStudyStateV1:
    changes = dict(common)
    changes[field_name] = artifact_id
    return replace(state, **changes)


def _access_for_transition(
    state: ProspectiveStudyStateV1,
    *,
    destination: StudyStageV1,
    target_payload_opened: bool | None,
    target_outcomes_opened: bool | None,
) -> tuple[bool, bool]:
    if destination != _TECHNICAL:
        if target_payload_opened is not None or target_outcomes_opened is not None:
            raise ValueError("explicit target-access flags require terminal-technical")
        shape = _STANDARD_SHAPE.get(destination)
        return (False, False) if shape is None else (shape[1], shape[2])
    payload = (
        state.target_payload_opened
        if target_payload_opened is None
        else _boolean(target_payload_opened, name="target_payload_opened")
    )
    outcomes = (
        state.target_outcomes_opened
        if target_outcomes_opened is None
        else _boolean(target_outcomes_opened, name="target_outcomes_opened")
    )
    if state.target_payload_opened and not payload:
        raise ValueError("technical terminal cannot close an open target payload")
    if state.target_outcomes_opened and not outcomes:
        raise ValueError("technical terminal cannot close open target outcomes")
    return payload, outcomes


def validate_prospective_study_chain(
    protocol: ProspectiveStudyProtocolV1,
    states: Sequence[ProspectiveStudyStateV1],
) -> None:
    """Recompute every transition and reject broken ancestry or replacement."""

    if not isinstance(protocol, ProspectiveStudyProtocolV1):
        raise TypeError("protocol must be a ProspectiveStudyProtocolV1")
    if (
        isinstance(states, (str, bytes))
        or not isinstance(states, Sequence)
        or not states
    ):
        raise ValueError("states must be a nonempty sequence")
    if any(not isinstance(state, ProspectiveStudyStateV1) for state in states):
        raise ValueError("states must contain ProspectiveStudyStateV1 values")
    first = states[0]
    _require_protocol(protocol, first)
    expected = lock_prospective_study(protocol, metadata=plain_json(first.metadata))
    if first.state_id != expected.state_id:
        raise ValueError(
            "lifecycle chain does not begin with the canonical design lock"
        )
    previous = first
    for current in states[1:]:
        _require_protocol(protocol, current)
        field_name = _STAGE_ARTIFACT[current.stage]
        artifact_id = getattr(current, field_name)
        if artifact_id is None:
            raise ValueError(f"lifecycle transition lacks {field_name}")
        expected = advance_prospective_study(
            previous,
            next_stage=current.stage,
            artifact_id=artifact_id,
            metadata=plain_json(current.metadata),
            target_payload_opened=(
                current.target_payload_opened if current.stage == _TECHNICAL else None
            ),
            target_outcomes_opened=(
                current.target_outcomes_opened if current.stage == _TECHNICAL else None
            ),
        )
        if current.state_id != expected.state_id:
            raise ValueError(
                "lifecycle transition changed ancestry, artifacts, or access state"
            )
        previous = current


def _require_protocol(
    protocol: ProspectiveStudyProtocolV1,
    state: ProspectiveStudyStateV1,
) -> None:
    if (
        state.protocol_id != protocol.protocol_id
        or state.protocol_content_id != protocol.protocol_content_id
    ):
        raise ValueError("lifecycle state does not bind the supplied protocol")


__all__ = [
    "PROSPECTIVE_STUDY_PROTOCOL_SCHEMA",
    "PROSPECTIVE_STUDY_SCHEMA_VERSION",
    "PROSPECTIVE_STUDY_STATE_SCHEMA",
    "ProspectiveStudyProtocolV1",
    "ProspectiveStudyStateV1",
    "StudyStageV1",
    "advance_prospective_study",
    "lock_prospective_study",
    "validate_prospective_study_chain",
]
