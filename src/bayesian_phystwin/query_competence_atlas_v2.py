"""Staged, query-scoped simulator competence atlas with exact fallback."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    plain_json,
)
from ._portable_contracts import (
    content_id,
    load_strict_json_object,
    require_exact_fields,
    sha256_digest,
    write_atomic_json,
)
from .complete_belief_selection import ArtifactBelief
from .query_competence_certificate_v1 import SimulatorQueryScopeV1

QUERY_COMPETENCE_STAGE_SCHEMA = "bayesian_phystwin.query_competence_stage"
QUERY_COMPETENCE_STAGE_VERSION = 2
QUERY_COMPETENCE_ATLAS_SCHEMA = "bayesian_phystwin.query_competence_atlas"
QUERY_COMPETENCE_ATLAS_VERSION = 2

StageStatus = Literal["passed", "failed", "not_evaluated"]
EvidenceRole = Literal["source_screen", "prospective_certificate"]
Decision = Literal["certified", "rejected", "not_promoted"]
BeliefT = TypeVar("BeliefT", bound=ArtifactBelief)

STAGE_NAMES = (
    "native_qualification",
    "action_headroom",
    "source_transfer",
    "prospective_risk",
)
_STAGE_VALUES = frozenset({"passed", "failed", "not_evaluated"})
_EVIDENCE_ROLES = frozenset({"source_screen", "prospective_certificate"})


def _canonical_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def _stage_status(value: object, *, name: str) -> StageStatus:
    if type(value) is not str or value not in _STAGE_VALUES:
        raise ValueError(f"{name} must be a registered stage status")
    return cast(StageStatus, value)


def _evidence_role(value: object) -> EvidenceRole:
    if type(value) is not str or value not in _EVIDENCE_ROLES:
        raise ValueError("evidence_role must be registered")
    return cast(EvidenceRole, value)


@dataclass(frozen=True, slots=True)
class QueryCompetenceStageV2:
    """Bind one exact query to the furthest competence stage it earned."""

    query_scope: SimulatorQueryScopeV1
    evidence_role: EvidenceRole
    evidence_artifact_id: str
    evidence_file_sha256: str
    independent_group_count: int
    native_qualification: StageStatus
    action_headroom: StageStatus
    source_transfer: StageStatus
    prospective_risk: StageStatus
    exact_fallback_retained: bool
    protocol_frozen_before_outcomes: bool
    outcomes_used_for_selection: bool
    protected_data_read: bool
    terminal_reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.query_scope, SimulatorQueryScopeV1):
            raise TypeError("query_scope must be a SimulatorQueryScopeV1")
        if self.query_scope.query_id is None:
            raise ValueError("query_scope must have a content identity")
        object.__setattr__(self, "evidence_role", _evidence_role(self.evidence_role))
        for name in ("evidence_artifact_id", "evidence_file_sha256"):
            object.__setattr__(
                self,
                name,
                sha256_digest(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "independent_group_count",
            genuine_integer(
                self.independent_group_count,
                name="independent_group_count",
                minimum=1,
            ),
        )
        for name in STAGE_NAMES:
            object.__setattr__(
                self,
                name,
                _stage_status(getattr(self, name), name=name),
            )
        for name in (
            "exact_fallback_retained",
            "protocol_frozen_before_outcomes",
            "outcomes_used_for_selection",
            "protected_data_read",
        ):
            object.__setattr__(
                self,
                name,
                genuine_boolean(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "terminal_reason",
            _canonical_string(self.terminal_reason, name="terminal_reason"),
        )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="stage metadata"),
        )
        self._validate_stage_order()
        expected_id = content_id(self.descriptor())
        supplied_id = self.artifact_id
        if supplied_id is not None:
            supplied_id = sha256_digest(supplied_id, name="artifact_id")
            if supplied_id != expected_id:
                raise ValueError("artifact_id does not match stage content")
        object.__setattr__(self, "artifact_id", expected_id)

    def _validate_stage_order(self) -> None:
        statuses = [cast(StageStatus, getattr(self, name)) for name in STAGE_NAMES]
        for index, status in enumerate(statuses):
            if status == "passed" and any(
                previous != "passed" for previous in statuses[:index]
            ):
                raise ValueError("a stage cannot pass after an earlier non-pass")
        if self.evidence_role == "source_screen" and self.prospective_risk != (
            "not_evaluated"
        ):
            raise ValueError("source evidence cannot decide prospective risk")

    @property
    def decision(self) -> Decision:
        statuses = tuple(cast(StageStatus, getattr(self, name)) for name in STAGE_NAMES)
        custody = (
            self.exact_fallback_retained
            and self.protocol_frozen_before_outcomes
            and not self.outcomes_used_for_selection
            and not self.protected_data_read
        )
        if custody and all(status == "passed" for status in statuses):
            return "certified"
        if any(status == "failed" for status in statuses) or not custody:
            return "rejected"
        return "not_promoted"

    @property
    def first_failed_stage(self) -> str | None:
        for name in STAGE_NAMES:
            if getattr(self, name) == "failed":
                return name
        return None

    @property
    def furthest_evaluated_stage(self) -> str:
        evaluated = [
            name for name in STAGE_NAMES if getattr(self, name) != "not_evaluated"
        ]
        if not evaluated:
            raise ValueError("at least one competence stage must be evaluated")
        return evaluated[-1]

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": QUERY_COMPETENCE_STAGE_SCHEMA,
            "schema_version": QUERY_COMPETENCE_STAGE_VERSION,
            "query_scope": self.query_scope.to_record(),
            "evidence_role": self.evidence_role,
            "evidence_artifact_id": self.evidence_artifact_id,
            "evidence_file_sha256": self.evidence_file_sha256,
            "independent_group_count": self.independent_group_count,
            "stages": {name: getattr(self, name) for name in STAGE_NAMES},
            "exact_fallback_retained": self.exact_fallback_retained,
            "protocol_frozen_before_outcomes": self.protocol_frozen_before_outcomes,
            "outcomes_used_for_selection": self.outcomes_used_for_selection,
            "protected_data_read": self.protected_data_read,
            "terminal_reason": self.terminal_reason,
            "decision": self.decision,
            "first_failed_stage": self.first_failed_stage,
            "furthest_evaluated_stage": self.furthest_evaluated_stage,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        name: str = "query competence stage",
    ) -> QueryCompetenceStageV2:
        require_exact_fields(
            value,
            expected=frozenset(
                {
                    "schema",
                    "schema_version",
                    "query_scope",
                    "evidence_role",
                    "evidence_artifact_id",
                    "evidence_file_sha256",
                    "independent_group_count",
                    "stages",
                    "exact_fallback_retained",
                    "protocol_frozen_before_outcomes",
                    "outcomes_used_for_selection",
                    "protected_data_read",
                    "terminal_reason",
                    "decision",
                    "first_failed_stage",
                    "furthest_evaluated_stage",
                    "metadata",
                    "artifact_id",
                }
            ),
            name=name,
        )
        if value["schema"] != QUERY_COMPETENCE_STAGE_SCHEMA:
            raise ValueError(f"{name} schema changed")
        if value["schema_version"] != QUERY_COMPETENCE_STAGE_VERSION:
            raise ValueError(f"{name} schema version changed")
        raw_stages = cast(Mapping[str, object], value["stages"])
        require_exact_fields(
            raw_stages,
            expected=frozenset(STAGE_NAMES),
            name=f"{name} stages",
        )
        result = cls(
            query_scope=SimulatorQueryScopeV1.from_mapping(
                cast(Mapping[str, object], value["query_scope"]),
                name=f"{name} query scope",
            ),
            evidence_role=cast(EvidenceRole, value["evidence_role"]),
            evidence_artifact_id=cast(str, value["evidence_artifact_id"]),
            evidence_file_sha256=cast(str, value["evidence_file_sha256"]),
            independent_group_count=cast(int, value["independent_group_count"]),
            native_qualification=cast(StageStatus, raw_stages["native_qualification"]),
            action_headroom=cast(StageStatus, raw_stages["action_headroom"]),
            source_transfer=cast(StageStatus, raw_stages["source_transfer"]),
            prospective_risk=cast(StageStatus, raw_stages["prospective_risk"]),
            exact_fallback_retained=cast(bool, value["exact_fallback_retained"]),
            protocol_frozen_before_outcomes=cast(
                bool, value["protocol_frozen_before_outcomes"]
            ),
            outcomes_used_for_selection=cast(
                bool, value["outcomes_used_for_selection"]
            ),
            protected_data_read=cast(bool, value["protected_data_read"]),
            terminal_reason=cast(str, value["terminal_reason"]),
            metadata=cast(Mapping[str, Any], value["metadata"]),
            artifact_id=cast(str, value["artifact_id"]),
        )
        for key, expected in (
            ("decision", result.decision),
            ("first_failed_stage", result.first_failed_stage),
            ("furthest_evaluated_stage", result.furthest_evaluated_stage),
        ):
            if value[key] != expected:
                raise ValueError(f"{name} derived field {key!r} changed")
        return result


@dataclass(frozen=True, slots=True)
class QueryCompetenceAtlasV2:
    """A non-pooling registry of staged competence evidence."""

    entries: Sequence[QueryCompetenceStageV2]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        if not entries or any(
            not isinstance(entry, QueryCompetenceStageV2) for entry in entries
        ):
            raise ValueError("atlas requires competence stage entries")
        entries = tuple(
            sorted(entries, key=lambda entry: str(entry.query_scope.query_id))
        )
        query_ids = tuple(str(entry.query_scope.query_id) for entry in entries)
        if len(set(query_ids)) != len(query_ids):
            raise ValueError("atlas query scopes must be unique")
        object.__setattr__(self, "entries", entries)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="atlas metadata"),
        )
        expected_id = content_id(self.descriptor())
        supplied_id = self.artifact_id
        if supplied_id is not None:
            supplied_id = sha256_digest(supplied_id, name="artifact_id")
            if supplied_id != expected_id:
                raise ValueError("artifact_id does not match atlas content")
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def certified_query_ids(self) -> tuple[str, ...]:
        return tuple(
            str(entry.query_scope.query_id)
            for entry in self.entries
            if entry.decision == "certified"
        )

    @property
    def rejected_query_ids(self) -> tuple[str, ...]:
        return tuple(
            str(entry.query_scope.query_id)
            for entry in self.entries
            if entry.decision == "rejected"
        )

    def by_query_id(self, query_id: str) -> QueryCompetenceStageV2 | None:
        checked = sha256_digest(query_id, name="query_id")
        return next(
            (entry for entry in self.entries if entry.query_scope.query_id == checked),
            None,
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": QUERY_COMPETENCE_ATLAS_SCHEMA,
            "schema_version": QUERY_COMPETENCE_ATLAS_VERSION,
            "entries": [entry.to_record() for entry in self.entries],
            "certified_query_ids": list(self.certified_query_ids),
            "rejected_query_ids": list(self.rejected_query_ids),
            "backend_wide_competence_claim": False,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        name: str = "query competence atlas",
    ) -> QueryCompetenceAtlasV2:
        require_exact_fields(
            value,
            expected=frozenset(
                {
                    "schema",
                    "schema_version",
                    "entries",
                    "certified_query_ids",
                    "rejected_query_ids",
                    "backend_wide_competence_claim",
                    "metadata",
                    "artifact_id",
                }
            ),
            name=name,
        )
        if value["schema"] != QUERY_COMPETENCE_ATLAS_SCHEMA:
            raise ValueError(f"{name} schema changed")
        if value["schema_version"] != QUERY_COMPETENCE_ATLAS_VERSION:
            raise ValueError(f"{name} schema version changed")
        if value["backend_wide_competence_claim"] is not False:
            raise ValueError(f"{name} cannot assert backend-wide competence")
        raw_entries = value["entries"]
        if isinstance(raw_entries, (str, bytes)) or not isinstance(
            raw_entries, Sequence
        ):
            raise ValueError(f"{name} entries must be a sequence")
        result = cls(
            entries=tuple(
                QueryCompetenceStageV2.from_mapping(
                    cast(Mapping[str, object], entry),
                    name=f"{name} entry {index}",
                )
                for index, entry in enumerate(raw_entries)
            ),
            metadata=cast(Mapping[str, Any], value["metadata"]),
            artifact_id=cast(str, value["artifact_id"]),
        )
        if value["certified_query_ids"] != list(result.certified_query_ids):
            raise ValueError(f"{name} certified query roster changed")
        if value["rejected_query_ids"] != list(result.rejected_query_ids):
            raise ValueError(f"{name} rejected query roster changed")
        return result


def save_query_competence_atlas(path: Path, atlas: QueryCompetenceAtlasV2) -> None:
    write_atomic_json(atlas.to_record(), path, overwrite=False)


def load_query_competence_atlas(path: Path) -> QueryCompetenceAtlasV2:
    return QueryCompetenceAtlasV2.from_mapping(
        load_strict_json_object(path, label="query competence atlas")
    )


def select_atlas_candidate(
    baseline: BeliefT,
    candidate: BeliefT,
    atlas: QueryCompetenceAtlasV2,
    *,
    query_id: str,
    inference_admissible: bool,
) -> tuple[BeliefT, dict[str, object]]:
    """Select only a fully certified exact query; otherwise reuse baseline."""

    if not isinstance(atlas, QueryCompetenceAtlasV2):
        raise TypeError("atlas must be a QueryCompetenceAtlasV2")
    checked_query_id = sha256_digest(query_id, name="query_id")
    admissible = genuine_boolean(inference_admissible, name="inference_admissible")
    baseline_id = sha256_digest(baseline.artifact_id, name="baseline artifact_id")
    candidate_id = sha256_digest(candidate.artifact_id, name="candidate artifact_id")
    entry = atlas.by_query_id(checked_query_id)
    if entry is None:
        reason = "unknown-query"
    elif entry.decision != "certified":
        reason = "query-stage-rejected"
    elif not admissible:
        reason = "inference-rejected"
    else:
        reason = "query-certified"
    selected_candidate = reason == "query-certified"
    selected = candidate if selected_candidate else baseline
    descriptor: dict[str, object] = {
        "schema": "bayesian_phystwin.query_competence_atlas_selection",
        "schema_version": 2,
        "atlas_id": atlas.artifact_id,
        "query_id": checked_query_id,
        "baseline_belief_id": baseline_id,
        "candidate_belief_id": candidate_id,
        "selected_belief_id": candidate_id if selected_candidate else baseline_id,
        "selected_candidate": selected_candidate,
        "inference_admissible": admissible,
        "reason": reason,
        "stage_artifact_id": None if entry is None else entry.artifact_id,
    }
    return selected, {**descriptor, "artifact_id": content_id(descriptor)}
