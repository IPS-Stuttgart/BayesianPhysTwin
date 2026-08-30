"""Comparator-bound simulator validation atlas with exact fallback.

The atlas separates runtime availability, local numerical checks, complete
horizon qualification, task headroom, source value, and prospective value.
Each assessment is bound to an exact simulator query and immutable evidence;
no backend-wide conclusion is inferred from one entry.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
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

SIMULATOR_VALIDATION_EVIDENCE_SCHEMA = (
    "bayesian_phystwin.simulator_validation_evidence_reference"
)
SIMULATOR_VALIDATION_STAGE_SCHEMA = (
    "bayesian_phystwin.simulator_validation_stage_assessment"
)
SIMULATOR_VALIDATION_ENTRY_SCHEMA = "bayesian_phystwin.simulator_validation_entry"
SIMULATOR_VALIDATION_ATLAS_SCHEMA = "bayesian_phystwin.simulator_validation_atlas"
SIMULATOR_VALIDATION_ATLAS_VERSION = 1

StageStatus = Literal["passed", "failed", "not_evaluated", "not_applicable"]
ValidationDecision = Literal[
    "prospective_certified",
    "source_supported",
    "full_horizon_qualified",
    "native_qualified",
    "runtime_available",
    "rejected",
    "not_evaluated",
]
BeliefT = TypeVar("BeliefT", bound=ArtifactBelief)

STAGE_NAMES = (
    "runtime_execution",
    "native_qualification",
    "full_horizon_qualification",
    "decision_headroom",
    "source_value",
    "prospective_value",
)
_STAGE_STATUSES = frozenset({"passed", "failed", "not_evaluated", "not_applicable"})
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def _canonical_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} cannot contain control characters")
    return value


def _repository_path(value: object) -> str:
    checked = _canonical_string(value, name="evidence path")
    if "\\" in checked:
        raise ValueError("evidence path must use POSIX separators")
    path = PurePosixPath(checked)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("evidence path must be canonical and repository-relative")
    return path.as_posix()


def _stage_status(value: object, *, name: str) -> StageStatus:
    if type(value) is not str or value not in _STAGE_STATUSES:
        raise ValueError(f"{name} must be a registered stage status")
    return cast(StageStatus, value)


@dataclass(frozen=True, slots=True)
class ValidationEvidenceReferenceV1:
    """Locate one immutable evidence file at its originating revision."""

    repository: str
    commit: str
    path: str
    file_sha256: str
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "repository",
            _canonical_string(self.repository, name="evidence repository"),
        )
        if type(self.commit) is not str or _COMMIT_RE.fullmatch(self.commit) is None:
            raise ValueError("evidence commit must be a lowercase 40-hex Git revision")
        object.__setattr__(self, "path", _repository_path(self.path))
        object.__setattr__(
            self,
            "file_sha256",
            sha256_digest(self.file_sha256, name="evidence file_sha256"),
        )
        if self.artifact_id is not None:
            object.__setattr__(
                self,
                "artifact_id",
                sha256_digest(self.artifact_id, name="evidence artifact_id"),
            )

    def to_record(self) -> dict[str, object]:
        return {
            "schema": SIMULATOR_VALIDATION_EVIDENCE_SCHEMA,
            "schema_version": SIMULATOR_VALIDATION_ATLAS_VERSION,
            "repository": self.repository,
            "commit": self.commit,
            "path": self.path,
            "file_sha256": self.file_sha256,
            "artifact_id": self.artifact_id,
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        name: str = "validation evidence reference",
    ) -> ValidationEvidenceReferenceV1:
        require_exact_fields(
            value,
            expected=frozenset(
                {
                    "schema",
                    "schema_version",
                    "repository",
                    "commit",
                    "path",
                    "file_sha256",
                    "artifact_id",
                }
            ),
            name=name,
        )
        if value["schema"] != SIMULATOR_VALIDATION_EVIDENCE_SCHEMA:
            raise ValueError(f"{name} schema changed")
        if value["schema_version"] != SIMULATOR_VALIDATION_ATLAS_VERSION:
            raise ValueError(f"{name} schema version changed")
        return cls(
            repository=cast(str, value["repository"]),
            commit=cast(str, value["commit"]),
            path=cast(str, value["path"]),
            file_sha256=cast(str, value["file_sha256"]),
            artifact_id=cast(str | None, value["artifact_id"]),
        )


@dataclass(frozen=True, slots=True)
class ValidationStageAssessmentV1:
    """One stage decision and the evidence that supports exactly that decision."""

    status: StageStatus
    reason: str
    evidence: Sequence[ValidationEvidenceReferenceV1] = field(default_factory=tuple)
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _stage_status(self.status, name="status"))
        object.__setattr__(
            self,
            "reason",
            _canonical_string(self.reason, name="stage reason"),
        )
        evidence = tuple(self.evidence)
        if any(
            not isinstance(item, ValidationEvidenceReferenceV1) for item in evidence
        ):
            raise TypeError("stage evidence must contain evidence references")
        if self.status in {"passed", "failed"} and not evidence:
            raise ValueError("an evaluated stage requires immutable evidence")
        if self.status in {"not_evaluated", "not_applicable"} and evidence:
            raise ValueError("an unevaluated stage cannot carry outcome evidence")
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(
            self,
            "metrics",
            frozen_finite_json_mapping(self.metrics, name="stage metrics"),
        )
        if self.status in {"not_evaluated", "not_applicable"} and self.metrics:
            raise ValueError("an unevaluated stage cannot carry metrics")

    def to_record(self) -> dict[str, object]:
        return {
            "schema": SIMULATOR_VALIDATION_STAGE_SCHEMA,
            "schema_version": SIMULATOR_VALIDATION_ATLAS_VERSION,
            "status": self.status,
            "reason": self.reason,
            "evidence": [item.to_record() for item in self.evidence],
            "metrics": plain_json(self.metrics),
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        name: str = "validation stage assessment",
    ) -> ValidationStageAssessmentV1:
        require_exact_fields(
            value,
            expected=frozenset(
                {"schema", "schema_version", "status", "reason", "evidence", "metrics"}
            ),
            name=name,
        )
        if value["schema"] != SIMULATOR_VALIDATION_STAGE_SCHEMA:
            raise ValueError(f"{name} schema changed")
        if value["schema_version"] != SIMULATOR_VALIDATION_ATLAS_VERSION:
            raise ValueError(f"{name} schema version changed")
        raw_evidence = value["evidence"]
        if isinstance(raw_evidence, (str, bytes)) or not isinstance(
            raw_evidence, Sequence
        ):
            raise ValueError(f"{name} evidence must be a sequence")
        return cls(
            status=cast(StageStatus, value["status"]),
            reason=cast(str, value["reason"]),
            evidence=tuple(
                ValidationEvidenceReferenceV1.from_mapping(
                    cast(Mapping[str, object], item),
                    name=f"{name} evidence {index}",
                )
                for index, item in enumerate(raw_evidence)
            ),
            metrics=cast(Mapping[str, Any], value["metrics"]),
        )


@dataclass(frozen=True, slots=True)
class SimulatorValidationEntryV1:
    """Validation ladder for one exact simulator/query/comparator scope."""

    backend_key: str
    display_name: str
    dataset: str
    query_scope: SimulatorQueryScopeV1
    independent_group_count: int
    stages: Mapping[str, ValidationStageAssessmentV1]
    exact_fallback_retained: bool
    protocol_frozen_before_outcomes: bool
    protected_target_data_read: bool
    new_recording_used: bool
    terminal_reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    entry_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("backend_key", "display_name", "dataset"):
            object.__setattr__(
                self,
                name,
                _canonical_string(getattr(self, name), name=name),
            )
        if not isinstance(self.query_scope, SimulatorQueryScopeV1):
            raise TypeError("query_scope must be a SimulatorQueryScopeV1")
        if self.query_scope.query_id is None:
            raise ValueError("query_scope must have a content identity")
        object.__setattr__(
            self,
            "independent_group_count",
            genuine_integer(
                self.independent_group_count,
                name="independent_group_count",
                minimum=1,
            ),
        )
        if set(self.stages) != set(STAGE_NAMES):
            raise ValueError("entry must contain exactly the registered stage roster")
        stages: dict[str, ValidationStageAssessmentV1] = {}
        for name in STAGE_NAMES:
            stage = self.stages[name]
            if not isinstance(stage, ValidationStageAssessmentV1):
                raise TypeError(f"stage {name!r} must be a stage assessment")
            stages[name] = stage
        object.__setattr__(self, "stages", MappingProxyType(stages))
        for name in (
            "exact_fallback_retained",
            "protocol_frozen_before_outcomes",
            "protected_target_data_read",
            "new_recording_used",
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
            frozen_finite_json_mapping(self.metadata, name="entry metadata"),
        )
        self._validate_stage_order()
        expected_id = content_id(self.descriptor())
        if self.entry_id is not None:
            supplied = sha256_digest(self.entry_id, name="entry_id")
            if supplied != expected_id:
                raise ValueError("entry_id does not match entry content")
        object.__setattr__(self, "entry_id", expected_id)

    def _validate_stage_order(self) -> None:
        prior_blocker = False
        for name in STAGE_NAMES:
            status = self.stages[name].status
            if status in {"failed", "not_evaluated"}:
                prior_blocker = True
            elif status == "passed" and prior_blocker:
                raise ValueError("a stage cannot pass after an unmet prerequisite")

    @property
    def decision(self) -> ValidationDecision:
        statuses = {name: self.stages[name].status for name in STAGE_NAMES}
        custody = (
            self.exact_fallback_retained
            and self.protocol_frozen_before_outcomes
            and not self.protected_target_data_read
            and not self.new_recording_used
        )
        if not custody or "failed" in statuses.values():
            return "rejected"
        if statuses["prospective_value"] == "passed":
            return "prospective_certified"
        if statuses["source_value"] == "passed":
            return "source_supported"
        if statuses["full_horizon_qualification"] == "passed":
            return "full_horizon_qualified"
        if statuses["native_qualification"] == "passed":
            return "native_qualified"
        if statuses["runtime_execution"] == "passed":
            return "runtime_available"
        return "not_evaluated"

    @property
    def first_failed_stage(self) -> str | None:
        return next(
            (name for name in STAGE_NAMES if self.stages[name].status == "failed"),
            None,
        )

    @property
    def furthest_evaluated_stage(self) -> str:
        evaluated = [
            name
            for name in STAGE_NAMES
            if self.stages[name].status in {"passed", "failed"}
        ]
        if not evaluated:
            raise ValueError("at least one validation stage must be evaluated")
        return evaluated[-1]

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": SIMULATOR_VALIDATION_ENTRY_SCHEMA,
            "schema_version": SIMULATOR_VALIDATION_ATLAS_VERSION,
            "backend_key": self.backend_key,
            "display_name": self.display_name,
            "dataset": self.dataset,
            "query_scope": self.query_scope.to_record(),
            "independent_group_count": self.independent_group_count,
            "stages": {name: self.stages[name].to_record() for name in STAGE_NAMES},
            "exact_fallback_retained": self.exact_fallback_retained,
            "protocol_frozen_before_outcomes": self.protocol_frozen_before_outcomes,
            "protected_target_data_read": self.protected_target_data_read,
            "new_recording_used": self.new_recording_used,
            "terminal_reason": self.terminal_reason,
            "decision": self.decision,
            "first_failed_stage": self.first_failed_stage,
            "furthest_evaluated_stage": self.furthest_evaluated_stage,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "entry_id": self.entry_id}

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        name: str = "simulator validation entry",
    ) -> SimulatorValidationEntryV1:
        require_exact_fields(
            value,
            expected=frozenset(
                {
                    "schema",
                    "schema_version",
                    "backend_key",
                    "display_name",
                    "dataset",
                    "query_scope",
                    "independent_group_count",
                    "stages",
                    "exact_fallback_retained",
                    "protocol_frozen_before_outcomes",
                    "protected_target_data_read",
                    "new_recording_used",
                    "terminal_reason",
                    "decision",
                    "first_failed_stage",
                    "furthest_evaluated_stage",
                    "metadata",
                    "entry_id",
                }
            ),
            name=name,
        )
        if value["schema"] != SIMULATOR_VALIDATION_ENTRY_SCHEMA:
            raise ValueError(f"{name} schema changed")
        if value["schema_version"] != SIMULATOR_VALIDATION_ATLAS_VERSION:
            raise ValueError(f"{name} schema version changed")
        raw_stages = cast(Mapping[str, object], value["stages"])
        require_exact_fields(
            raw_stages,
            expected=frozenset(STAGE_NAMES),
            name=f"{name} stages",
        )
        result = cls(
            backend_key=cast(str, value["backend_key"]),
            display_name=cast(str, value["display_name"]),
            dataset=cast(str, value["dataset"]),
            query_scope=SimulatorQueryScopeV1.from_mapping(
                cast(Mapping[str, object], value["query_scope"]),
                name=f"{name} query scope",
            ),
            independent_group_count=cast(int, value["independent_group_count"]),
            stages={
                stage_name: ValidationStageAssessmentV1.from_mapping(
                    cast(Mapping[str, object], raw_stages[stage_name]),
                    name=f"{name} stage {stage_name}",
                )
                for stage_name in STAGE_NAMES
            },
            exact_fallback_retained=cast(bool, value["exact_fallback_retained"]),
            protocol_frozen_before_outcomes=cast(
                bool, value["protocol_frozen_before_outcomes"]
            ),
            protected_target_data_read=cast(bool, value["protected_target_data_read"]),
            new_recording_used=cast(bool, value["new_recording_used"]),
            terminal_reason=cast(str, value["terminal_reason"]),
            metadata=cast(Mapping[str, Any], value["metadata"]),
            entry_id=cast(str, value["entry_id"]),
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
class SimulatorValidationAtlasV1:
    """Non-pooling validation map over exact public simulator queries."""

    entries: Sequence[SimulatorValidationEntryV1]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        if not entries or any(
            not isinstance(entry, SimulatorValidationEntryV1) for entry in entries
        ):
            raise ValueError("atlas requires simulator validation entries")
        entries = tuple(sorted(entries, key=lambda item: str(item.entry_id)))
        entry_ids = tuple(str(entry.entry_id) for entry in entries)
        query_ids = tuple(str(entry.query_scope.query_id) for entry in entries)
        if len(set(entry_ids)) != len(entry_ids):
            raise ValueError("atlas entry identities must be unique")
        if len(set(query_ids)) != len(query_ids):
            raise ValueError("atlas query scopes must be unique")
        object.__setattr__(self, "entries", entries)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="atlas metadata"),
        )
        expected_id = content_id(self.descriptor())
        if self.artifact_id is not None:
            supplied = sha256_digest(self.artifact_id, name="artifact_id")
            if supplied != expected_id:
                raise ValueError("artifact_id does not match atlas content")
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def decision_counts(self) -> dict[str, int]:
        counts = Counter(entry.decision for entry in self.entries)
        return {key: counts[key] for key in sorted(counts)}

    @property
    def stage_counts(self) -> dict[str, dict[str, int]]:
        return {
            name: {
                status: sum(
                    entry.stages[name].status == status for entry in self.entries
                )
                for status in sorted(_STAGE_STATUSES)
            }
            for name in STAGE_NAMES
        }

    @property
    def prospectively_certified_query_ids(self) -> tuple[str, ...]:
        return tuple(
            str(entry.query_scope.query_id)
            for entry in self.entries
            if entry.decision == "prospective_certified"
        )

    def by_query_id(self, query_id: str) -> SimulatorValidationEntryV1 | None:
        checked = sha256_digest(query_id, name="query_id")
        return next(
            (entry for entry in self.entries if entry.query_scope.query_id == checked),
            None,
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": SIMULATOR_VALIDATION_ATLAS_SCHEMA,
            "schema_version": SIMULATOR_VALIDATION_ATLAS_VERSION,
            "entries": [entry.to_record() for entry in self.entries],
            "entry_count": len(self.entries),
            "backend_count": len({entry.backend_key for entry in self.entries}),
            "dataset_count": len({entry.dataset for entry in self.entries}),
            "decision_counts": self.decision_counts,
            "stage_counts": self.stage_counts,
            "prospectively_certified_query_ids": list(
                self.prospectively_certified_query_ids
            ),
            "backend_wide_competence_claim": False,
            "cross_backend_ranking_claim": False,
            "official_benchmark_claim": False,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        name: str = "simulator validation atlas",
    ) -> SimulatorValidationAtlasV1:
        require_exact_fields(
            value,
            expected=frozenset(
                {
                    "schema",
                    "schema_version",
                    "entries",
                    "entry_count",
                    "backend_count",
                    "dataset_count",
                    "decision_counts",
                    "stage_counts",
                    "prospectively_certified_query_ids",
                    "backend_wide_competence_claim",
                    "cross_backend_ranking_claim",
                    "official_benchmark_claim",
                    "metadata",
                    "artifact_id",
                }
            ),
            name=name,
        )
        if value["schema"] != SIMULATOR_VALIDATION_ATLAS_SCHEMA:
            raise ValueError(f"{name} schema changed")
        if value["schema_version"] != SIMULATOR_VALIDATION_ATLAS_VERSION:
            raise ValueError(f"{name} schema version changed")
        for key in (
            "backend_wide_competence_claim",
            "cross_backend_ranking_claim",
            "official_benchmark_claim",
        ):
            if value[key] is not False:
                raise ValueError(f"{name} cannot assert {key}")
        raw_entries = value["entries"]
        if isinstance(raw_entries, (str, bytes)) or not isinstance(
            raw_entries, Sequence
        ):
            raise ValueError(f"{name} entries must be a sequence")
        result = cls(
            entries=tuple(
                SimulatorValidationEntryV1.from_mapping(
                    cast(Mapping[str, object], entry),
                    name=f"{name} entry {index}",
                )
                for index, entry in enumerate(raw_entries)
            ),
            metadata=cast(Mapping[str, Any], value["metadata"]),
            artifact_id=cast(str, value["artifact_id"]),
        )
        expected_fields: tuple[tuple[str, object], ...] = (
            ("entry_count", len(result.entries)),
            ("backend_count", len({entry.backend_key for entry in result.entries})),
            ("dataset_count", len({entry.dataset for entry in result.entries})),
            ("decision_counts", result.decision_counts),
            ("stage_counts", result.stage_counts),
            (
                "prospectively_certified_query_ids",
                list(result.prospectively_certified_query_ids),
            ),
        )
        for key, expected in expected_fields:
            if value[key] != expected:
                raise ValueError(f"{name} derived field {key!r} changed")
        return result


def save_simulator_validation_atlas(
    path: Path, atlas: SimulatorValidationAtlasV1
) -> None:
    write_atomic_json(atlas.to_record(), path, overwrite=False)


def load_simulator_validation_atlas(path: Path) -> SimulatorValidationAtlasV1:
    return SimulatorValidationAtlasV1.from_mapping(
        load_strict_json_object(path, label="simulator validation atlas")
    )


def select_prospectively_validated_candidate(
    baseline: BeliefT,
    candidate: BeliefT,
    atlas: SimulatorValidationAtlasV1,
    *,
    query_id: str,
    inference_admissible: bool,
) -> tuple[BeliefT, dict[str, object]]:
    """Select only an exact prospective certificate; otherwise return baseline."""

    if not isinstance(atlas, SimulatorValidationAtlasV1):
        raise TypeError("atlas must be a SimulatorValidationAtlasV1")
    checked_query_id = sha256_digest(query_id, name="query_id")
    admissible = genuine_boolean(inference_admissible, name="inference_admissible")
    baseline_id = sha256_digest(baseline.artifact_id, name="baseline artifact_id")
    candidate_id = sha256_digest(candidate.artifact_id, name="candidate artifact_id")
    entry = atlas.by_query_id(checked_query_id)
    if entry is None:
        reason = "unknown-query"
    elif entry.decision != "prospective_certified":
        reason = "query-not-prospectively-certified"
    elif not admissible:
        reason = "inference-rejected"
    else:
        reason = "prospective-query-certified"
    selected_candidate = reason == "prospective-query-certified"
    selected = candidate if selected_candidate else baseline
    descriptor: dict[str, object] = {
        "schema": "bayesian_phystwin.simulator_validation_atlas_selection",
        "schema_version": SIMULATOR_VALIDATION_ATLAS_VERSION,
        "atlas_id": atlas.artifact_id,
        "query_id": checked_query_id,
        "baseline_belief_id": baseline_id,
        "candidate_belief_id": candidate_id,
        "selected_belief_id": candidate_id if selected_candidate else baseline_id,
        "selected_candidate": selected_candidate,
        "inference_admissible": admissible,
        "reason": reason,
        "validation_entry_id": None if entry is None else entry.entry_id,
    }
    return selected, {**descriptor, "artifact_id": content_id(descriptor)}
