"""Sealed Deform360 calibration choices before confirmation payload access.

Stage 0 fixes the official dataset revision and fresh object/episode split from
names and metadata only. This module defines the next portable boundary: every
calibration-derived mapping, covariance, bias, reliability, evidence-scaling,
physical-response, guard, and interval choice must be content-addressed before a
confirmation camera, tactile, robot, geometry, or outcome payload is opened.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
)
from ._portable_contracts import (
    canonical_sorted_strings,
    content_id,
    exact_revision,
    load_strict_json_object,
    nonempty_string,
    require_exact_fields,
    sha256_digest,
    source_artifact_mapping,
    write_atomic_json,
)

DEFORM360_CALIBRATION_UNIT_SCHEMA = "bayesian-phystwin.deform360-cohort-unit"
DEFORM360_CALIBRATION_ARTIFACT_SCHEMA = (
    "bayesian-phystwin.deform360-calibration-artifact-ref"
)
DEFORM360_CALIBRATION_BUNDLE_SCHEMA = "bayesian-phystwin.deform360-calibration-bundle"
DEFORM360_CALIBRATION_VERSION = 1
DEFORM360_CALIBRATION_SEMANTICS = (
    "calibration-only-choices-sealed-before-confirmation-payload-v1"
)
DEFORM360_CALIBRATION_STATUS = "sealed-before-confirmation-payload-access"
DEFORM360_CALIBRATION_PROTOCOL_ID = "deform360-official-hub-visuotactile-v1"
DEFORM360_DATASET_REPOSITORY = "brownu/deform360"
DEFORM360_PROCESSING_REPOSITORY = "lhy0807/deform360"
DEFORM360_CALIBRATION_OBJECTS_PER_STRATUM = 5
DEFORM360_CONFIRMATION_OBJECTS_PER_STRATUM = 6
DEFORM360_CALIBRATION_CLAIM_BOUNDARY = (
    "Calibration and information-boundary evidence only. A valid bundle does not "
    "establish Deform360 accuracy, tactile benefit, calibrated raw covariance, "
    "material-parameter identification, Causal4D benefit, or state of the art."
)

Deform360Stratum = Literal["sheet", "volumetric"]
Deform360CalibrationRole = Literal[
    "contact_feature_and_grouping",
    "contact_linearization_and_covariance",
    "anchor_bias_prior",
    "visual_reliability_and_gauge",
    "normalized_evidence",
    "physical_response_and_closure",
    "regret_guard",
    "conformal_interval",
]
DEFORM360_CALIBRATION_ROLES: tuple[Deform360CalibrationRole, ...] = (
    "contact_feature_and_grouping",
    "contact_linearization_and_covariance",
    "anchor_bias_prior",
    "visual_reliability_and_gauge",
    "normalized_evidence",
    "physical_response_and_closure",
    "regret_guard",
    "conformal_interval",
)
_ROLE_ORDER = {role: index for index, role in enumerate(DEFORM360_CALIBRATION_ROLES)}

_UNIT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "unit_id",
        "object_id",
        "episode_id",
        "stratum",
        "metadata_path",
        "metadata_sha256",
    }
)
_ARTIFACT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "ref_id",
        "role",
        "artifact_id",
        "implementation_revision",
        "selection_evidence_id",
        "selected_candidate_id",
        "candidate_count",
        "calibration_group_ids",
        "source_artifacts",
        "target_outcomes_used",
        "confirmation_payload_used",
        "metadata",
    }
)
_BUNDLE_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "bundle_id",
        "confirmation_opening_token",
        "protocol_id",
        "selection_artifact_sha256",
        "content_selection_sha256",
        "dataset_repository",
        "dataset_revision",
        "processing_repository",
        "processing_revision",
        "implementation_revision",
        "calibration_units",
        "confirmation_units",
        "calibration_artifacts",
        "evidence_use_ledger_id",
        "source_artifacts",
        "status",
        "confirmation_payload_opened",
        "target_outcomes_used",
        "replacement_allowed",
        "metadata",
        "claim_boundary",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _stratum(value: object) -> Deform360Stratum:
    if value not in {"sheet", "volumetric"} or type(value) is not str:
        raise ValueError("stratum must be sheet or volumetric")
    return cast(Deform360Stratum, value)


def _role(value: object) -> Deform360CalibrationRole:
    if type(value) is not str or value not in DEFORM360_CALIBRATION_ROLES:
        raise ValueError("calibration role is unsupported")
    return cast(Deform360CalibrationRole, value)


def _metadata_path(value: object, *, object_id: str) -> str:
    path_text = nonempty_string(value, name="metadata_path")
    path = PurePosixPath(path_text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("metadata_path must be a confined relative path")
    expected = PurePosixPath("raw") / object_id / "metadata.json"
    if path != expected:
        raise ValueError("metadata_path must identify raw/<object>/metadata.json")
    return path.as_posix()


@dataclass(frozen=True, order=True)
class Deform360CohortUnitV1:
    """One exact official-Hub object/episode selected before raw payload access."""

    object_id: str
    episode_id: int
    stratum: Deform360Stratum
    metadata_path: str
    metadata_sha256: str

    def __post_init__(self) -> None:
        object_id = nonempty_string(self.object_id, name="object_id")
        episode_id = genuine_integer(self.episode_id, name="episode_id", minimum=0)
        stratum = _stratum(self.stratum)
        metadata_path = _metadata_path(self.metadata_path, object_id=object_id)
        metadata_sha256 = sha256_digest(
            self.metadata_sha256,
            name="metadata_sha256",
        )
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "episode_id", episode_id)
        object.__setattr__(self, "stratum", stratum)
        object.__setattr__(self, "metadata_path", metadata_path)
        object.__setattr__(self, "metadata_sha256", metadata_sha256)

    @property
    def unit_id(self) -> str:
        """Return the immutable object/episode selection identity."""

        return content_id(self._descriptor())

    def _descriptor(self) -> dict[str, object]:
        return {
            "schema": DEFORM360_CALIBRATION_UNIT_SCHEMA,
            "schema_version": DEFORM360_CALIBRATION_VERSION,
            "object_id": self.object_id,
            "episode_id": self.episode_id,
            "stratum": self.stratum,
            "metadata_path": self.metadata_path,
            "metadata_sha256": self.metadata_sha256,
        }

    def to_record(self) -> dict[str, object]:
        return {**self._descriptor(), "unit_id": self.unit_id}

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        name: str = "Deform360 cohort unit",
    ) -> Deform360CohortUnitV1:
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a JSON object")
        require_exact_fields(value, expected=_UNIT_FIELDS, name=name)
        if value["schema"] != DEFORM360_CALIBRATION_UNIT_SCHEMA:
            raise ValueError(f"{name} schema changed")
        if (
            genuine_integer(
                value["schema_version"],
                name=f"{name} schema_version",
                minimum=1,
            )
            != DEFORM360_CALIBRATION_VERSION
        ):
            raise ValueError(f"{name} schema_version changed")
        result = cls(
            object_id=value["object_id"],
            episode_id=value["episode_id"],
            stratum=value["stratum"],
            metadata_path=value["metadata_path"],
            metadata_sha256=value["metadata_sha256"],
        )
        supplied_id = sha256_digest(value["unit_id"], name=f"{name} unit_id")
        if supplied_id != result.unit_id:
            raise ValueError(f"{name} unit_id does not match its content")
        return result


@dataclass(frozen=True)
class Deform360CalibrationArtifactRefV1:
    """One calibration-only selected artifact with replayable selection lineage."""

    role: Deform360CalibrationRole
    artifact_id: str
    implementation_revision: str
    selection_evidence_id: str
    selected_candidate_id: str
    candidate_count: int
    calibration_group_ids: Sequence[str]
    source_artifacts: Mapping[str, str]
    target_outcomes_used: bool = False
    confirmation_payload_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        role = _role(self.role)
        artifact_id = sha256_digest(self.artifact_id, name="artifact_id")
        implementation_revision = exact_revision(
            self.implementation_revision,
            name="implementation_revision",
        )
        selection_evidence_id = sha256_digest(
            self.selection_evidence_id,
            name="selection_evidence_id",
        )
        selected_candidate_id = nonempty_string(
            self.selected_candidate_id,
            name="selected_candidate_id",
        )
        candidate_count = genuine_integer(
            self.candidate_count,
            name="candidate_count",
            minimum=1,
        )
        groups = canonical_sorted_strings(
            self.calibration_group_ids,
            name="calibration_group_ids",
        )
        source_artifacts = source_artifact_mapping(
            self.source_artifacts,
            name="source_artifacts",
        )
        target_outcomes_used = genuine_boolean(
            self.target_outcomes_used,
            name="target_outcomes_used",
        )
        confirmation_payload_used = genuine_boolean(
            self.confirmation_payload_used,
            name="confirmation_payload_used",
        )
        _require(
            not target_outcomes_used,
            "calibration artifact must not use target outcomes",
        )
        _require(
            not confirmation_payload_used,
            "calibration artifact must not use confirmation payloads",
        )
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="Deform360 calibration artifact metadata",
        )

        object.__setattr__(self, "role", role)
        object.__setattr__(self, "artifact_id", artifact_id)
        object.__setattr__(self, "implementation_revision", implementation_revision)
        object.__setattr__(self, "selection_evidence_id", selection_evidence_id)
        object.__setattr__(self, "selected_candidate_id", selected_candidate_id)
        object.__setattr__(self, "candidate_count", candidate_count)
        object.__setattr__(self, "calibration_group_ids", groups)
        object.__setattr__(self, "source_artifacts", source_artifacts)
        object.__setattr__(self, "target_outcomes_used", target_outcomes_used)
        object.__setattr__(
            self,
            "confirmation_payload_used",
            confirmation_payload_used,
        )
        object.__setattr__(self, "metadata", metadata)

    @property
    def ref_id(self) -> str:
        """Return the immutable selected-calibration identity."""

        return content_id(self._descriptor())

    def _descriptor(self) -> dict[str, object]:
        return {
            "schema": DEFORM360_CALIBRATION_ARTIFACT_SCHEMA,
            "schema_version": DEFORM360_CALIBRATION_VERSION,
            "semantics": DEFORM360_CALIBRATION_SEMANTICS,
            "role": self.role,
            "artifact_id": self.artifact_id,
            "implementation_revision": self.implementation_revision,
            "selection_evidence_id": self.selection_evidence_id,
            "selected_candidate_id": self.selected_candidate_id,
            "candidate_count": self.candidate_count,
            "calibration_group_ids": self.calibration_group_ids,
            "source_artifacts": self.source_artifacts,
            "target_outcomes_used": self.target_outcomes_used,
            "confirmation_payload_used": self.confirmation_payload_used,
            "metadata": self.metadata,
        }

    def to_record(self) -> dict[str, object]:
        return {**self._descriptor(), "ref_id": self.ref_id}

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        name: str = "Deform360 calibration artifact",
    ) -> Deform360CalibrationArtifactRefV1:
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a JSON object")
        require_exact_fields(value, expected=_ARTIFACT_FIELDS, name=name)
        if value["schema"] != DEFORM360_CALIBRATION_ARTIFACT_SCHEMA:
            raise ValueError(f"{name} schema changed")
        if (
            genuine_integer(
                value["schema_version"],
                name=f"{name} schema_version",
                minimum=1,
            )
            != DEFORM360_CALIBRATION_VERSION
        ):
            raise ValueError(f"{name} schema_version changed")
        if value["semantics"] != DEFORM360_CALIBRATION_SEMANTICS:
            raise ValueError(f"{name} semantics changed")
        result = cls(
            role=value["role"],
            artifact_id=value["artifact_id"],
            implementation_revision=value["implementation_revision"],
            selection_evidence_id=value["selection_evidence_id"],
            selected_candidate_id=value["selected_candidate_id"],
            candidate_count=value["candidate_count"],
            calibration_group_ids=value["calibration_group_ids"],
            source_artifacts=value["source_artifacts"],
            target_outcomes_used=value["target_outcomes_used"],
            confirmation_payload_used=value["confirmation_payload_used"],
            metadata=value["metadata"],
        )
        supplied_id = sha256_digest(value["ref_id"], name=f"{name} ref_id")
        if supplied_id != result.ref_id:
            raise ValueError(f"{name} ref_id does not match its content")
        return result


@dataclass(frozen=True)
class Deform360CalibrationBundleV1:
    """Complete pre-confirmation seal for the official-Hub visuotactile study."""

    selection_artifact_sha256: str
    content_selection_sha256: str
    dataset_revision: str
    processing_revision: str
    implementation_revision: str
    calibration_units: Sequence[Deform360CohortUnitV1]
    confirmation_units: Sequence[Deform360CohortUnitV1]
    calibration_artifacts: Sequence[Deform360CalibrationArtifactRefV1]
    evidence_use_ledger_id: str
    source_artifacts: Mapping[str, str]
    protocol_id: str = DEFORM360_CALIBRATION_PROTOCOL_ID
    status: str = DEFORM360_CALIBRATION_STATUS
    confirmation_payload_opened: bool = False
    target_outcomes_used: bool = False
    replacement_allowed: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        protocol_id = nonempty_string(self.protocol_id, name="protocol_id")
        _require(
            protocol_id == DEFORM360_CALIBRATION_PROTOCOL_ID,
            "protocol_id changed from the locked official-Hub study",
        )
        selection_artifact_sha256 = sha256_digest(
            self.selection_artifact_sha256,
            name="selection_artifact_sha256",
        )
        content_selection_sha256 = sha256_digest(
            self.content_selection_sha256,
            name="content_selection_sha256",
        )
        dataset_revision = exact_revision(
            self.dataset_revision,
            name="dataset_revision",
        )
        processing_revision = exact_revision(
            self.processing_revision,
            name="processing_revision",
        )
        implementation_revision = exact_revision(
            self.implementation_revision,
            name="implementation_revision",
        )
        calibration_units = self._units(
            self.calibration_units,
            name="calibration_units",
        )
        confirmation_units = self._units(
            self.confirmation_units,
            name="confirmation_units",
        )
        calibration_ids = {unit.object_id for unit in calibration_units}
        confirmation_ids = {unit.object_id for unit in confirmation_units}
        _require(
            calibration_ids.isdisjoint(confirmation_ids),
            "calibration and confirmation objects overlap",
        )
        self._validate_stratum_counts(
            calibration_units,
            expected=DEFORM360_CALIBRATION_OBJECTS_PER_STRATUM,
            name="calibration",
        )
        self._validate_stratum_counts(
            confirmation_units,
            expected=DEFORM360_CONFIRMATION_OBJECTS_PER_STRATUM,
            name="confirmation",
        )

        if isinstance(self.calibration_artifacts, (str, bytes)):
            raise ValueError("calibration_artifacts must be a sequence")
        calibration_artifacts = tuple(self.calibration_artifacts)
        if any(
            not isinstance(artifact, Deform360CalibrationArtifactRefV1)
            for artifact in calibration_artifacts
        ):
            raise ValueError(
                "calibration_artifacts must contain "
                "Deform360CalibrationArtifactRefV1 objects"
            )
        roles = [artifact.role for artifact in calibration_artifacts]
        _require(len(set(roles)) == len(roles), "duplicate calibration role")
        _require(
            set(roles) == set(DEFORM360_CALIBRATION_ROLES),
            "calibration roles are incomplete",
        )
        calibration_artifacts = tuple(
            sorted(calibration_artifacts, key=lambda item: _ROLE_ORDER[item.role])
        )
        expected_groups = tuple(sorted(calibration_ids))
        for artifact in calibration_artifacts:
            _require(
                artifact.calibration_group_ids == expected_groups,
                f"calibration artifact {artifact.role} does not retain every "
                "calibration object",
            )

        evidence_use_ledger_id = sha256_digest(
            self.evidence_use_ledger_id,
            name="evidence_use_ledger_id",
        )
        source_artifacts = source_artifact_mapping(
            self.source_artifacts,
            name="source_artifacts",
        )
        status = nonempty_string(self.status, name="status")
        _require(status == DEFORM360_CALIBRATION_STATUS, "calibration status changed")
        confirmation_payload_opened = genuine_boolean(
            self.confirmation_payload_opened,
            name="confirmation_payload_opened",
        )
        target_outcomes_used = genuine_boolean(
            self.target_outcomes_used,
            name="target_outcomes_used",
        )
        replacement_allowed = genuine_boolean(
            self.replacement_allowed,
            name="replacement_allowed",
        )
        _require(
            not confirmation_payload_opened,
            "confirmation payload was opened before the calibration seal",
        )
        _require(
            not target_outcomes_used,
            "target outcomes were used before the calibration seal",
        )
        _require(
            not replacement_allowed,
            "selected units may not be replaced after metadata selection",
        )
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="Deform360 calibration bundle metadata",
        )

        object.__setattr__(self, "protocol_id", protocol_id)
        object.__setattr__(
            self,
            "selection_artifact_sha256",
            selection_artifact_sha256,
        )
        object.__setattr__(
            self,
            "content_selection_sha256",
            content_selection_sha256,
        )
        object.__setattr__(self, "dataset_revision", dataset_revision)
        object.__setattr__(self, "processing_revision", processing_revision)
        object.__setattr__(self, "implementation_revision", implementation_revision)
        object.__setattr__(self, "calibration_units", calibration_units)
        object.__setattr__(self, "confirmation_units", confirmation_units)
        object.__setattr__(self, "calibration_artifacts", calibration_artifacts)
        object.__setattr__(self, "evidence_use_ledger_id", evidence_use_ledger_id)
        object.__setattr__(self, "source_artifacts", source_artifacts)
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "confirmation_payload_opened",
            confirmation_payload_opened,
        )
        object.__setattr__(self, "target_outcomes_used", target_outcomes_used)
        object.__setattr__(self, "replacement_allowed", replacement_allowed)
        object.__setattr__(self, "metadata", metadata)

    @staticmethod
    def _units(
        values: Sequence[Deform360CohortUnitV1],
        *,
        name: str,
    ) -> tuple[Deform360CohortUnitV1, ...]:
        if isinstance(values, (str, bytes)):
            raise ValueError(f"{name} must be a sequence")
        units = tuple(values)
        invalid = any(not isinstance(unit, Deform360CohortUnitV1) for unit in units)
        if not units or invalid:
            raise ValueError(f"{name} must contain Deform360CohortUnitV1 objects")
        units = tuple(sorted(units, key=lambda unit: (unit.stratum, unit.object_id)))
        object_ids = [unit.object_id for unit in units]
        unit_ids = [unit.unit_id for unit in units]
        _require(len(set(object_ids)) == len(object_ids), f"{name} repeats an object")
        _require(len(set(unit_ids)) == len(unit_ids), f"{name} repeats a unit")
        return units

    @staticmethod
    def _validate_stratum_counts(
        units: Sequence[Deform360CohortUnitV1],
        *,
        expected: int,
        name: str,
    ) -> None:
        for stratum in ("sheet", "volumetric"):
            count = sum(unit.stratum == stratum for unit in units)
            _require(
                count == expected,
                f"{name} stratum {stratum} must contain exactly {expected} objects",
            )

    @property
    def bundle_id(self) -> str:
        """Return the immutable pre-confirmation calibration seal identity."""

        return content_id(self._descriptor())

    @property
    def confirmation_opening_token(self) -> str:
        """Return a token bound to this exact valid seal and confirmation cohort."""

        return content_id(
            {
                "schema": "bayesian-phystwin.deform360-confirmation-opening-token",
                "schema_version": DEFORM360_CALIBRATION_VERSION,
                "bundle_id": self.bundle_id,
                "confirmation_unit_ids": [
                    unit.unit_id for unit in self.confirmation_units
                ],
                "status": self.status,
            }
        )

    def _descriptor(self) -> dict[str, object]:
        return {
            "schema": DEFORM360_CALIBRATION_BUNDLE_SCHEMA,
            "schema_version": DEFORM360_CALIBRATION_VERSION,
            "semantics": DEFORM360_CALIBRATION_SEMANTICS,
            "protocol_id": self.protocol_id,
            "selection_artifact_sha256": self.selection_artifact_sha256,
            "content_selection_sha256": self.content_selection_sha256,
            "dataset_repository": DEFORM360_DATASET_REPOSITORY,
            "dataset_revision": self.dataset_revision,
            "processing_repository": DEFORM360_PROCESSING_REPOSITORY,
            "processing_revision": self.processing_revision,
            "implementation_revision": self.implementation_revision,
            "calibration_units": [unit.to_record() for unit in self.calibration_units],
            "confirmation_units": [
                unit.to_record() for unit in self.confirmation_units
            ],
            "calibration_artifacts": [
                artifact.to_record() for artifact in self.calibration_artifacts
            ],
            "evidence_use_ledger_id": self.evidence_use_ledger_id,
            "source_artifacts": self.source_artifacts,
            "status": self.status,
            "confirmation_payload_opened": self.confirmation_payload_opened,
            "target_outcomes_used": self.target_outcomes_used,
            "replacement_allowed": self.replacement_allowed,
            "metadata": self.metadata,
            "claim_boundary": DEFORM360_CALIBRATION_CLAIM_BOUNDARY,
        }

    def to_record(self) -> dict[str, object]:
        """Return the complete portable calibration bundle."""

        return {
            **self._descriptor(),
            "bundle_id": self.bundle_id,
            "confirmation_opening_token": self.confirmation_opening_token,
        }

    def summary(self) -> dict[str, object]:
        """Return compact information-boundary and cohort lineage."""

        return {
            "schema": DEFORM360_CALIBRATION_BUNDLE_SCHEMA,
            "schema_version": DEFORM360_CALIBRATION_VERSION,
            "semantics": DEFORM360_CALIBRATION_SEMANTICS,
            "bundle_id": self.bundle_id,
            "confirmation_opening_token": self.confirmation_opening_token,
            "protocol_id": self.protocol_id,
            "dataset_revision": self.dataset_revision,
            "processing_revision": self.processing_revision,
            "calibration_object_count": len(self.calibration_units),
            "confirmation_object_count": len(self.confirmation_units),
            "calibration_roles": [
                artifact.role for artifact in self.calibration_artifacts
            ],
            "evidence_use_ledger_id": self.evidence_use_ledger_id,
            "status": self.status,
            "confirmation_payload_opened": False,
            "target_outcomes_used": False,
            "replacement_allowed": False,
            "claim_boundary": DEFORM360_CALIBRATION_CLAIM_BOUNDARY,
        }

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        name: str = "Deform360 calibration bundle",
    ) -> Deform360CalibrationBundleV1:
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a JSON object")
        require_exact_fields(value, expected=_BUNDLE_FIELDS, name=name)
        if value["schema"] != DEFORM360_CALIBRATION_BUNDLE_SCHEMA:
            raise ValueError(f"{name} schema changed")
        if (
            genuine_integer(
                value["schema_version"],
                name=f"{name} schema_version",
                minimum=1,
            )
            != DEFORM360_CALIBRATION_VERSION
        ):
            raise ValueError(f"{name} schema_version changed")
        if value["semantics"] != DEFORM360_CALIBRATION_SEMANTICS:
            raise ValueError(f"{name} semantics changed")
        if value["dataset_repository"] != DEFORM360_DATASET_REPOSITORY:
            raise ValueError(f"{name} dataset repository changed")
        if value["processing_repository"] != DEFORM360_PROCESSING_REPOSITORY:
            raise ValueError(f"{name} processing repository changed")
        if value["claim_boundary"] != DEFORM360_CALIBRATION_CLAIM_BOUNDARY:
            raise ValueError(f"{name} claim boundary changed")
        calibration_units_raw = value["calibration_units"]
        confirmation_units_raw = value["confirmation_units"]
        artifacts_raw = value["calibration_artifacts"]
        if not isinstance(calibration_units_raw, list):
            raise ValueError(f"{name} calibration_units must be a JSON array")
        if not isinstance(confirmation_units_raw, list):
            raise ValueError(f"{name} confirmation_units must be a JSON array")
        if not isinstance(artifacts_raw, list):
            raise ValueError(f"{name} calibration_artifacts must be a JSON array")
        result = cls(
            protocol_id=value["protocol_id"],
            selection_artifact_sha256=value["selection_artifact_sha256"],
            content_selection_sha256=value["content_selection_sha256"],
            dataset_revision=value["dataset_revision"],
            processing_revision=value["processing_revision"],
            implementation_revision=value["implementation_revision"],
            calibration_units=tuple(
                Deform360CohortUnitV1.from_mapping(
                    unit,
                    name=f"{name} calibration unit {index}",
                )
                for index, unit in enumerate(calibration_units_raw)
            ),
            confirmation_units=tuple(
                Deform360CohortUnitV1.from_mapping(
                    unit,
                    name=f"{name} confirmation unit {index}",
                )
                for index, unit in enumerate(confirmation_units_raw)
            ),
            calibration_artifacts=tuple(
                Deform360CalibrationArtifactRefV1.from_mapping(
                    artifact,
                    name=f"{name} calibration artifact {index}",
                )
                for index, artifact in enumerate(artifacts_raw)
            ),
            evidence_use_ledger_id=value["evidence_use_ledger_id"],
            source_artifacts=value["source_artifacts"],
            status=value["status"],
            confirmation_payload_opened=value["confirmation_payload_opened"],
            target_outcomes_used=value["target_outcomes_used"],
            replacement_allowed=value["replacement_allowed"],
            metadata=value["metadata"],
        )
        supplied_id = sha256_digest(value["bundle_id"], name=f"{name} bundle_id")
        if supplied_id != result.bundle_id:
            raise ValueError(f"{name} bundle_id does not match its content")
        supplied_token = sha256_digest(
            value["confirmation_opening_token"],
            name=f"{name} confirmation_opening_token",
        )
        if supplied_token != result.confirmation_opening_token:
            raise ValueError(
                f"{name} confirmation_opening_token does not match its content"
            )
        return result


def verify_deform360_confirmation_gate(
    bundle: Deform360CalibrationBundleV1,
    *,
    expected_bundle_id: str,
    expected_selection_artifact_sha256: str,
    expected_evidence_use_ledger_id: str,
) -> str:
    """Verify exact reviewed identities before any confirmation payload is opened."""

    if not isinstance(bundle, Deform360CalibrationBundleV1):
        raise TypeError("bundle must be a Deform360CalibrationBundleV1")
    expected_id = sha256_digest(expected_bundle_id, name="expected_bundle_id")
    expected_selection = sha256_digest(
        expected_selection_artifact_sha256,
        name="expected_selection_artifact_sha256",
    )
    expected_ledger = sha256_digest(
        expected_evidence_use_ledger_id,
        name="expected_evidence_use_ledger_id",
    )
    _require(bundle.bundle_id == expected_id, "calibration bundle identity changed")
    _require(
        bundle.selection_artifact_sha256 == expected_selection,
        "Stage-0 selection artifact changed",
    )
    _require(
        bundle.evidence_use_ledger_id == expected_ledger,
        "calibration evidence-use ledger changed",
    )
    return bundle.confirmation_opening_token


def save_deform360_calibration_bundle(
    bundle: Deform360CalibrationBundleV1,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically persist one validated pre-confirmation seal."""

    if not isinstance(bundle, Deform360CalibrationBundleV1):
        raise TypeError("bundle must be a Deform360CalibrationBundleV1")
    write_atomic_json(bundle.to_record(), path, overwrite=overwrite)


def load_deform360_calibration_bundle(
    path: str | Path,
) -> Deform360CalibrationBundleV1:
    """Load and independently revalidate one persisted calibration seal."""

    return Deform360CalibrationBundleV1.from_mapping(
        load_strict_json_object(path, label="Deform360 calibration bundle")
    )


__all__ = [
    "DEFORM360_CALIBRATION_ARTIFACT_SCHEMA",
    "DEFORM360_CALIBRATION_BUNDLE_SCHEMA",
    "DEFORM360_CALIBRATION_CLAIM_BOUNDARY",
    "DEFORM360_CALIBRATION_OBJECTS_PER_STRATUM",
    "DEFORM360_CALIBRATION_PROTOCOL_ID",
    "DEFORM360_CALIBRATION_ROLES",
    "DEFORM360_CALIBRATION_SEMANTICS",
    "DEFORM360_CALIBRATION_STATUS",
    "DEFORM360_CALIBRATION_UNIT_SCHEMA",
    "DEFORM360_CALIBRATION_VERSION",
    "DEFORM360_CONFIRMATION_OBJECTS_PER_STRATUM",
    "DEFORM360_DATASET_REPOSITORY",
    "DEFORM360_PROCESSING_REPOSITORY",
    "Deform360CalibrationArtifactRefV1",
    "Deform360CalibrationBundleV1",
    "Deform360CalibrationRole",
    "Deform360CohortUnitV1",
    "Deform360Stratum",
    "load_deform360_calibration_bundle",
    "save_deform360_calibration_bundle",
    "verify_deform360_confirmation_gate",
]
