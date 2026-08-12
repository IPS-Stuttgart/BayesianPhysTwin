"""Role-bound artifact identities for prospective-study lifecycle transitions.

The lifecycle v1 contract deliberately accepts opaque SHA-256 artifact identities.
This additive contract preserves the raw content digest while deriving a separate,
domain-separated binding identity from the study protocol, lifecycle stage,
artifact role, artifact schema, and optional finite metadata.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal, cast

from ._canonical_contracts import frozen_finite_json_mapping, plain_json
from ._portable_contracts import (
    content_id,
    nonempty_string,
    require_exact_fields,
    sha256_digest,
)
from .prospective_study_lifecycle_v1 import (
    ProspectiveStudyProtocolV1,
    ProspectiveStudyStateV1,
    StudyStageV1,
    advance_prospective_study,
    validate_prospective_study_chain,
)

PROSPECTIVE_STUDY_ARTIFACT_BINDING_SCHEMA: Final = (
    "bayesian-phystwin.prospective-study-artifact-binding-v1"
)
PROSPECTIVE_STUDY_ARTIFACT_BINDING_DOMAIN: Final = (
    "bayesian-phystwin/prospective-study-artifact-binding/v1"
)
PROSPECTIVE_STUDY_ARTIFACT_BINDING_VERSION: Final = 1

ArtifactRoleV1 = Literal[
    "source-prediction-bundle",
    "source-score-bundle",
    "target-authorization",
    "target-prediction-bundle",
    "target-score-bundle",
    "terminal-decision",
]

_STAGE_ROLE: Final[Mapping[StudyStageV1, ArtifactRoleV1]] = {
    "source-predictions-sealed": "source-prediction-bundle",
    "source-scored": "source-score-bundle",
    "target-authorized": "target-authorization",
    "target-predictions-sealed": "target-prediction-bundle",
    "target-scored": "target-score-bundle",
    "terminal-source-negative": "terminal-decision",
    "terminal-positive": "terminal-decision",
    "terminal-negative": "terminal-decision",
    "terminal-technical": "terminal-decision",
}
_STAGE_ARTIFACT_FIELD: Final[Mapping[StudyStageV1, str]] = {
    "source-predictions-sealed": "source_prediction_bundle_id",
    "source-scored": "source_score_bundle_id",
    "target-authorized": "target_authorization_id",
    "target-predictions-sealed": "target_prediction_bundle_id",
    "target-scored": "target_score_bundle_id",
    "terminal-source-negative": "terminal_decision_id",
    "terminal-positive": "terminal_decision_id",
    "terminal-negative": "terminal_decision_id",
    "terminal-technical": "terminal_decision_id",
}
_ROLES: Final = frozenset(_STAGE_ROLE.values())
_BINDING_FIELDS: Final = frozenset(
    {
        "binding_id",
        "binding_domain",
        "schema_name",
        "schema_version",
        "protocol_id",
        "protocol_content_id",
        "stage",
        "artifact_role",
        "artifact_content_id",
        "artifact_schema_name",
        "artifact_schema_version",
        "metadata",
    }
)


def _text(value: object, *, name: str) -> str:
    result = nonempty_string(value, name=name)
    if result.strip() != result or any(character in result for character in "\x00\r\n"):
        raise ValueError(f"{name} must be canonical single-line text")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must use literal string object keys")
    return cast(Mapping[str, Any], value)


def _binding_stage(value: object) -> StudyStageV1:
    if type(value) is not str or value not in _STAGE_ROLE:
        raise ValueError("unsupported role-bound prospective-study stage")
    return cast(StudyStageV1, value)


def _role(value: object) -> ArtifactRoleV1:
    if type(value) is not str or value not in _ROLES:
        raise ValueError("unsupported prospective-study artifact role")
    return cast(ArtifactRoleV1, value)


def artifact_role_for_stage(stage: StudyStageV1) -> ArtifactRoleV1:
    """Return the only artifact role permitted for one lifecycle transition."""

    destination = _binding_stage(stage)
    return _STAGE_ROLE[destination]


@dataclass(frozen=True)
class ProspectiveStudyArtifactBindingV1:
    """Bind raw artifact content to one protocol, stage, role, and schema."""

    protocol_id: str
    protocol_content_id: str
    stage: StudyStageV1
    artifact_role: ArtifactRoleV1
    artifact_content_id: str
    artifact_schema_name: str
    artifact_schema_version: int
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
        stage = _binding_stage(self.stage)
        object.__setattr__(self, "stage", stage)
        role = _role(self.artifact_role)
        object.__setattr__(self, "artifact_role", role)
        expected_role = _STAGE_ROLE[stage]
        if role != expected_role:
            raise ValueError(
                f"{stage} requires artifact role {expected_role!r}, not {role!r}"
            )
        object.__setattr__(
            self,
            "artifact_content_id",
            sha256_digest(self.artifact_content_id, name="artifact_content_id"),
        )
        object.__setattr__(
            self,
            "artifact_schema_name",
            _text(self.artifact_schema_name, name="artifact_schema_name"),
        )
        object.__setattr__(
            self,
            "artifact_schema_version",
            _positive_integer(
                self.artifact_schema_version,
                name="artifact_schema_version",
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="artifact binding metadata"),
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "binding_domain": PROSPECTIVE_STUDY_ARTIFACT_BINDING_DOMAIN,
            "schema_name": PROSPECTIVE_STUDY_ARTIFACT_BINDING_SCHEMA,
            "schema_version": PROSPECTIVE_STUDY_ARTIFACT_BINDING_VERSION,
            "protocol_id": self.protocol_id,
            "protocol_content_id": self.protocol_content_id,
            "stage": self.stage,
            "artifact_role": self.artifact_role,
            "artifact_content_id": self.artifact_content_id,
            "artifact_schema_name": self.artifact_schema_name,
            "artifact_schema_version": self.artifact_schema_version,
            "metadata": plain_json(self.metadata),
        }

    @property
    def binding_id(self) -> str:
        """Return the domain-separated identity stored in the lifecycle state."""

        return content_id(self.descriptor())

    def as_dict(self) -> dict[str, object]:
        return {"binding_id": self.binding_id, **self.descriptor()}

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> ProspectiveStudyArtifactBindingV1:
        require_exact_fields(
            value,
            expected=_BINDING_FIELDS,
            name="prospective-study artifact binding",
        )
        if (
            _text(value["binding_domain"], name="binding_domain")
            != PROSPECTIVE_STUDY_ARTIFACT_BINDING_DOMAIN
        ):
            raise ValueError("unexpected prospective-study artifact-binding domain")
        if (
            _text(value["schema_name"], name="schema_name")
            != PROSPECTIVE_STUDY_ARTIFACT_BINDING_SCHEMA
        ):
            raise ValueError("unexpected prospective-study artifact-binding schema")
        if (
            type(value["schema_version"]) is not int
            or value["schema_version"] != PROSPECTIVE_STUDY_ARTIFACT_BINDING_VERSION
        ):
            raise ValueError("unexpected artifact-binding schema version")
        result = cls(
            protocol_id=_text(value["protocol_id"], name="protocol_id"),
            protocol_content_id=sha256_digest(
                value["protocol_content_id"],
                name="protocol_content_id",
            ),
            stage=_binding_stage(value["stage"]),
            artifact_role=_role(value["artifact_role"]),
            artifact_content_id=sha256_digest(
                value["artifact_content_id"],
                name="artifact_content_id",
            ),
            artifact_schema_name=_text(
                value["artifact_schema_name"],
                name="artifact_schema_name",
            ),
            artifact_schema_version=_positive_integer(
                value["artifact_schema_version"],
                name="artifact_schema_version",
            ),
            metadata=_mapping(value["metadata"], name="artifact binding metadata"),
        )
        supplied = sha256_digest(value["binding_id"], name="binding_id")
        if supplied != result.binding_id:
            raise ValueError("prospective-study artifact-binding identity mismatch")
        return result


def bind_prospective_artifact(
    protocol: ProspectiveStudyProtocolV1,
    *,
    stage: StudyStageV1,
    artifact_content_id: str,
    artifact_schema_name: str,
    artifact_schema_version: int,
    metadata: Mapping[str, Any] | None = None,
) -> ProspectiveStudyArtifactBindingV1:
    """Create the canonical role binding for one immutable protocol."""

    if not isinstance(protocol, ProspectiveStudyProtocolV1):
        raise TypeError("protocol must be a ProspectiveStudyProtocolV1")
    destination = _binding_stage(stage)
    return ProspectiveStudyArtifactBindingV1(
        protocol_id=protocol.protocol_id,
        protocol_content_id=protocol.protocol_content_id,
        stage=destination,
        artifact_role=_STAGE_ROLE[destination],
        artifact_content_id=artifact_content_id,
        artifact_schema_name=artifact_schema_name,
        artifact_schema_version=artifact_schema_version,
        metadata={} if metadata is None else metadata,
    )


def advance_role_bound_prospective_study(
    state: ProspectiveStudyStateV1,
    *,
    next_stage: StudyStageV1,
    artifact_content_id: str,
    artifact_schema_name: str,
    artifact_schema_version: int,
    binding_metadata: Mapping[str, Any] | None = None,
    state_metadata: Mapping[str, Any] | None = None,
    target_payload_opened: bool | None = None,
    target_outcomes_opened: bool | None = None,
) -> tuple[ProspectiveStudyStateV1, ProspectiveStudyArtifactBindingV1]:
    """Advance one lifecycle transition using its role-bound identity."""

    if not isinstance(state, ProspectiveStudyStateV1):
        raise TypeError("state must be a ProspectiveStudyStateV1")
    destination = _binding_stage(next_stage)
    binding = ProspectiveStudyArtifactBindingV1(
        protocol_id=state.protocol_id,
        protocol_content_id=state.protocol_content_id,
        stage=destination,
        artifact_role=_STAGE_ROLE[destination],
        artifact_content_id=artifact_content_id,
        artifact_schema_name=artifact_schema_name,
        artifact_schema_version=artifact_schema_version,
        metadata={} if binding_metadata is None else binding_metadata,
    )
    next_state = advance_prospective_study(
        state,
        next_stage=destination,
        artifact_id=binding.binding_id,
        metadata={} if state_metadata is None else state_metadata,
        target_payload_opened=target_payload_opened,
        target_outcomes_opened=target_outcomes_opened,
    )
    return next_state, binding


def validate_role_bound_prospective_study_chain(
    protocol: ProspectiveStudyProtocolV1,
    states: Sequence[ProspectiveStudyStateV1],
    bindings: Sequence[ProspectiveStudyArtifactBindingV1],
) -> None:
    """Validate lifecycle ancestry and every external role-binding record."""

    validate_prospective_study_chain(protocol, states)
    if isinstance(bindings, (str, bytes)) or not isinstance(bindings, Sequence):
        raise ValueError("bindings must be a sequence")
    if any(
        not isinstance(binding, ProspectiveStudyArtifactBindingV1)
        for binding in bindings
    ):
        raise ValueError(
            "bindings must contain ProspectiveStudyArtifactBindingV1 values"
        )
    expected_count = len(states) - 1
    if len(bindings) != expected_count:
        raise ValueError(
            "role-bound lifecycle requires exactly one binding per transition"
        )
    seen_content_ids: set[str] = set()
    for state, binding in zip(states[1:], bindings, strict=True):
        if (
            binding.protocol_id != protocol.protocol_id
            or binding.protocol_content_id != protocol.protocol_content_id
        ):
            raise ValueError("artifact binding does not bind the supplied protocol")
        if binding.stage != state.stage:
            raise ValueError("artifact binding stage does not match lifecycle state")
        field_name = _STAGE_ARTIFACT_FIELD[state.stage]
        state_artifact_id = getattr(state, field_name)
        if state_artifact_id != binding.binding_id:
            raise ValueError(
                "lifecycle state does not store the artifact role-binding identity"
            )
        if binding.artifact_content_id in seen_content_ids:
            raise ValueError(
                "raw artifact content identity was reused across lifecycle roles"
            )
        seen_content_ids.add(binding.artifact_content_id)


__all__ = [
    "PROSPECTIVE_STUDY_ARTIFACT_BINDING_DOMAIN",
    "PROSPECTIVE_STUDY_ARTIFACT_BINDING_SCHEMA",
    "PROSPECTIVE_STUDY_ARTIFACT_BINDING_VERSION",
    "ArtifactRoleV1",
    "ProspectiveStudyArtifactBindingV1",
    "advance_role_bound_prospective_study",
    "artifact_role_for_stage",
    "bind_prospective_artifact",
    "validate_role_bound_prospective_study_chain",
]
