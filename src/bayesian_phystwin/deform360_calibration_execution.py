"""Fail-closed assembly of the Deform360 pre-confirmation calibration seal.

The official-Hub protocol has three independently reviewable boundaries:

1. a metadata-only Stage-0 cohort selection;
2. a target-blind visual-provider lock before calibration payload access; and
3. a complete calibration seal before confirmation payload access.

This module verifies those boundaries together. It does not download data, fit a
model, inspect confirmation payloads, or score an empirical outcome.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

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
from .deform360_calibration_bundle import (
    DEFORM360_CALIBRATION_OBJECTS_PER_STRATUM,
    DEFORM360_CALIBRATION_PROTOCOL_ID,
    DEFORM360_CALIBRATION_ROLES,
    DEFORM360_CALIBRATION_STATUS,
    DEFORM360_CONFIRMATION_OBJECTS_PER_STRATUM,
    Deform360CalibrationArtifactRefV1,
    Deform360CalibrationBundleV1,
    Deform360CohortUnitV1,
)
from .deform360_visual_provider_lock import (
    DEFORM360_FINITE_GROUP_CALIBRATION_DESIGN_ID,
    DEFORM360_FINITE_GROUP_CALIBRATION_GROUP_COUNT,
    DEFORM360_FINITE_GROUP_CONFORMAL_RANK,
    Deform360VisualCalibrationLockV1,
    Deform360VisualProviderLockV1,
)
from .evidence_use_ledger import EvidenceUseLedgerV1

DEFORM360_STAGE0_SELECTION_SCHEMA = (
    "bayesian-phystwin/deform360-official-hub-selection-v1"
)
DEFORM360_STAGE0_SNAPSHOT_SCHEMA = (
    "bayesian-phystwin.deform360-stage0-selection-snapshot"
)
DEFORM360_STAGE0_SNAPSHOT_VERSION = 1
DEFORM360_CALIBRATION_EXECUTION_SCHEMA = (
    "bayesian-phystwin.deform360-calibration-execution-seal"
)
DEFORM360_CALIBRATION_EXECUTION_VERSION = 1
DEFORM360_CALIBRATION_EXECUTION_SEMANTICS = (
    "calibration-open-confirmation-closed-complete-seal-v1"
)
DEFORM360_CALIBRATION_LEDGER_CASE_ID = "deform360-official-hub-calibration-cohort-v1"
DEFORM360_CALIBRATION_EXECUTION_CLAIM_BOUNDARY = (
    "Calibration execution and information-boundary evidence only. A valid seal "
    "does not establish Deform360 accuracy, tactile benefit, provider competence, "
    "calibrated deployment uncertainty, material identification, Causal4D "
    "intervention benefit, safety, or state of the art."
)

_STAGE0_FIELDS = frozenset(
    {
        "available_raw_object_count",
        "cache_preflight",
        "content_selection_sha256",
        "dataset",
        "excluded_object_count",
        "implementation_revision",
        "information_boundary",
        "next_gate",
        "official_processing",
        "prior_protocols",
        "protocol_id",
        "protocol_sha256",
        "replacement_allowed_after_payload_access",
        "schema",
        "schema_version",
        "selection",
        "selection_artifact_sha256",
        "selection_sha256",
    }
)
_STAGE0_UNIT_FIELDS = frozenset(
    {
        "episode_id",
        "metadata_path",
        "metadata_sha256",
        "object_id",
        "stratum",
    }
)
_EXECUTION_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "seal_id",
        "protocol_id",
        "status",
        "implementation_revision",
        "stage0_snapshot_id",
        "stage0_source_sha256",
        "selection_sha256",
        "visual_provider_lock_id",
        "visual_calibration_lock_id",
        "calibration_bundle_id",
        "confirmation_opening_token",
        "evidence_use_ledger_id",
        "calibration_object_ids",
        "confirmation_object_ids",
        "source_artifacts",
        "calibration_payloads_opened",
        "confirmation_payloads_opened",
        "target_outcomes_used",
        "metadata",
        "claim_boundary",
    }
)
_COMPONENT_ROLES: Mapping[str, tuple[str, ...]] = {
    "visual": (
        "visual_reliability_and_gauge",
        "normalized_evidence",
    ),
    "contact_anchor": (
        "contact_feature_and_grouping",
        "contact_linearization_and_covariance",
        "anchor_bias_prior",
    ),
    "guard": (
        "physical_response_and_closure",
        "regret_guard",
    ),
    "interval": ("conformal_interval",),
}
_REQUIRED_SOURCE_KEYS = frozenset(
    {
        "sources/stage0/selection.json",
        "sources/locks/visual-provider-lock.json",
        "sources/calibration/evidence-use-ledger.json",
        *(
            f"sources/calibration/artifacts/{role}.json"
            for role in DEFORM360_CALIBRATION_ROLES
        ),
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 of one ordinary file without following a symlink."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"source must be an ordinary file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_units(
    values: Sequence[Deform360CohortUnitV1],
    *,
    expected_per_stratum: int,
    name: str,
) -> tuple[Deform360CohortUnitV1, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    units = tuple(values)
    if not units or any(not isinstance(unit, Deform360CohortUnitV1) for unit in units):
        raise ValueError(f"{name} must contain Deform360CohortUnitV1 objects")
    units = tuple(sorted(units, key=lambda item: (item.stratum, item.object_id)))
    object_ids = [unit.object_id for unit in units]
    _require(len(set(object_ids)) == len(object_ids), f"{name} repeats an object")
    for stratum in ("sheet", "volumetric"):
        count = sum(unit.stratum == stratum for unit in units)
        _require(
            count == expected_per_stratum,
            f"{name} stratum {stratum} must contain exactly "
            f"{expected_per_stratum} objects",
        )
    return units


@dataclass(frozen=True)
class Deform360Stage0SelectionV1:
    """Exact Stage-0 cohort and source-byte identity."""

    source_sha256: str
    selection_artifact_sha256: str
    selection_sha256: str
    content_selection_sha256: str
    protocol_sha256: str
    dataset_revision: str
    processing_revision: str
    implementation_revision: str
    calibration_units: Sequence[Deform360CohortUnitV1]
    confirmation_units: Sequence[Deform360CohortUnitV1]
    protocol_id: str = DEFORM360_CALIBRATION_PROTOCOL_ID

    def __post_init__(self) -> None:
        protocol_id = nonempty_string(self.protocol_id, name="protocol_id")
        _require(
            protocol_id == DEFORM360_CALIBRATION_PROTOCOL_ID,
            "Stage-0 protocol_id changed",
        )
        digests = {
            name: sha256_digest(value, name=name)
            for name, value in (
                ("source_sha256", self.source_sha256),
                (
                    "selection_artifact_sha256",
                    self.selection_artifact_sha256,
                ),
                ("selection_sha256", self.selection_sha256),
                (
                    "content_selection_sha256",
                    self.content_selection_sha256,
                ),
                ("protocol_sha256", self.protocol_sha256),
            )
        }
        revisions = {
            name: exact_revision(value, name=name)
            for name, value in (
                ("dataset_revision", self.dataset_revision),
                ("processing_revision", self.processing_revision),
                ("implementation_revision", self.implementation_revision),
            )
        }
        calibration_units = _validated_units(
            self.calibration_units,
            expected_per_stratum=DEFORM360_CALIBRATION_OBJECTS_PER_STRATUM,
            name="calibration_units",
        )
        confirmation_units = _validated_units(
            self.confirmation_units,
            expected_per_stratum=DEFORM360_CONFIRMATION_OBJECTS_PER_STRATUM,
            name="confirmation_units",
        )
        calibration_ids = {unit.object_id for unit in calibration_units}
        confirmation_ids = {unit.object_id for unit in confirmation_units}
        _require(
            calibration_ids.isdisjoint(confirmation_ids),
            "Stage-0 calibration and confirmation objects overlap",
        )

        object.__setattr__(self, "protocol_id", protocol_id)
        for name, value in {**digests, **revisions}.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "calibration_units", calibration_units)
        object.__setattr__(self, "confirmation_units", confirmation_units)

    @property
    def snapshot_id(self) -> str:
        return content_id(
            {
                "schema": DEFORM360_STAGE0_SNAPSHOT_SCHEMA,
                "schema_version": DEFORM360_STAGE0_SNAPSHOT_VERSION,
                "protocol_id": self.protocol_id,
                "source_sha256": self.source_sha256,
                "selection_artifact_sha256": (self.selection_artifact_sha256),
                "selection_sha256": self.selection_sha256,
                "content_selection_sha256": self.content_selection_sha256,
                "protocol_sha256": self.protocol_sha256,
                "dataset_revision": self.dataset_revision,
                "processing_revision": self.processing_revision,
                "implementation_revision": self.implementation_revision,
                "calibration_units": [
                    unit.to_record() for unit in self.calibration_units
                ],
                "confirmation_units": [
                    unit.to_record() for unit in self.confirmation_units
                ],
            }
        )


def _stage0_unit(value: object, *, name: str) -> Deform360CohortUnitV1:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    require_exact_fields(
        value,
        expected=_STAGE0_UNIT_FIELDS,
        name=name,
    )
    return Deform360CohortUnitV1(
        object_id=cast(str, value["object_id"]),
        episode_id=cast(int, value["episode_id"]),
        stratum=cast(Literal["sheet", "volumetric"], value["stratum"]),
        metadata_path=cast(str, value["metadata_path"]),
        metadata_sha256=cast(str, value["metadata_sha256"]),
    )


def load_deform360_stage0_selection(
    path: str | Path,
) -> Deform360Stage0SelectionV1:
    """Load and independently validate the committed metadata-only Stage 0."""

    source = Path(path)
    value = load_strict_json_object(source, label="Deform360 Stage-0 selection")
    require_exact_fields(
        value,
        expected=_STAGE0_FIELDS,
        name="Deform360 Stage-0 selection",
    )
    if value["schema"] != DEFORM360_STAGE0_SELECTION_SCHEMA:
        raise ValueError("Deform360 Stage-0 selection schema changed")
    if (
        genuine_integer(
            value["schema_version"],
            name="Stage-0 schema_version",
            minimum=1,
        )
        != 1
    ):
        raise ValueError("Deform360 Stage-0 selection version changed")
    if value["protocol_id"] != DEFORM360_CALIBRATION_PROTOCOL_ID:
        raise ValueError("Deform360 Stage-0 protocol changed")

    boundary = value["information_boundary"]
    if not isinstance(boundary, Mapping):
        raise ValueError("Stage-0 information_boundary must be an object")
    required_boundary = {
        "camera_media_opened": False,
        "geometry_annotations_opened": False,
        "object_directory_names_opened": True,
        "object_metadata_json_opened": True,
        "robot_arrays_opened": False,
        "tactile_arrays_opened": False,
        "target_outcomes_opened": False,
    }
    for key, expected in required_boundary.items():
        observed = genuine_boolean(
            boundary.get(key),
            name=f"Stage-0 boundary {key}",
        )
        if observed != expected:
            raise ValueError(f"Stage-0 information boundary changed: {key}")
    replacement = genuine_boolean(
        value["replacement_allowed_after_payload_access"],
        name="replacement_allowed_after_payload_access",
    )
    if replacement:
        raise ValueError("Stage-0 replacement boundary changed")

    dataset = value["dataset"]
    if not isinstance(dataset, Mapping):
        raise ValueError("Stage-0 dataset must be an object")
    if dataset.get("repo_id") != "brownu/deform360":
        raise ValueError("Stage-0 dataset repository changed")
    if dataset.get("raw_prefix") != "raw":
        raise ValueError("Stage-0 raw prefix changed")
    dataset_revision = exact_revision(
        dataset.get("resolved_revision"),
        name="Stage-0 dataset revision",
    )

    processing = value["official_processing"]
    if not isinstance(processing, Mapping):
        raise ValueError("Stage-0 official_processing must be an object")
    if processing.get("repository") != "lhy0807/deform360":
        raise ValueError("Stage-0 processing repository changed")
    processing_revision = exact_revision(
        processing.get("revision"),
        name="Stage-0 processing revision",
    )

    selection = value["selection"]
    if not isinstance(selection, Mapping) or set(selection) != {
        "calibration",
        "confirmation",
    }:
        raise ValueError("Stage-0 selection roles changed")
    calibration_raw = selection["calibration"]
    confirmation_raw = selection["confirmation"]
    if not isinstance(calibration_raw, list):
        raise ValueError("Stage-0 calibration selection must be an array")
    if not isinstance(confirmation_raw, list):
        raise ValueError("Stage-0 confirmation selection must be an array")

    return Deform360Stage0SelectionV1(
        protocol_id=cast(str, value["protocol_id"]),
        source_sha256=file_sha256(source),
        selection_artifact_sha256=cast(
            str,
            value["selection_artifact_sha256"],
        ),
        selection_sha256=cast(str, value["selection_sha256"]),
        content_selection_sha256=cast(
            str,
            value["content_selection_sha256"],
        ),
        protocol_sha256=cast(str, value["protocol_sha256"]),
        dataset_revision=dataset_revision,
        processing_revision=processing_revision,
        implementation_revision=cast(
            str,
            value["implementation_revision"],
        ),
        calibration_units=tuple(
            _stage0_unit(item, name=f"Stage-0 calibration unit {index}")
            for index, item in enumerate(calibration_raw)
        ),
        confirmation_units=tuple(
            _stage0_unit(item, name=f"Stage-0 confirmation unit {index}")
            for index, item in enumerate(confirmation_raw)
        ),
    )


def load_deform360_calibration_artifact_ref(
    path: str | Path,
) -> Deform360CalibrationArtifactRefV1:
    """Strictly load one selected calibration artifact reference."""

    return Deform360CalibrationArtifactRefV1.from_mapping(
        load_strict_json_object(path, label="Deform360 calibration artifact")
    )


def deform360_calibration_component_ids(
    artifacts: Sequence[Deform360CalibrationArtifactRefV1],
) -> dict[str, str]:
    """Compress the eight complete calibration roles into four Stage-1 IDs."""

    if isinstance(artifacts, (str, bytes)):
        raise ValueError("calibration artifacts must be a sequence")
    selected = tuple(artifacts)
    if any(
        not isinstance(item, Deform360CalibrationArtifactRefV1) for item in selected
    ):
        raise ValueError(
            "calibration artifacts must contain "
            "Deform360CalibrationArtifactRefV1 objects"
        )
    by_role = {item.role: item for item in selected}
    if len(by_role) != len(selected):
        raise ValueError("duplicate calibration role")
    if set(by_role) != set(DEFORM360_CALIBRATION_ROLES):
        raise ValueError("calibration roles are incomplete")

    result: dict[str, str] = {}
    for component, roles in _COMPONENT_ROLES.items():
        result[component] = content_id(
            {
                "schema": ("bayesian-phystwin.deform360-calibration-component"),
                "schema_version": 1,
                "component": component,
                "artifacts": [
                    {
                        "role": role,
                        "ref_id": by_role[role].ref_id,
                    }
                    for role in roles
                ],
            }
        )
    return result


def _entry_object_ids(metadata: Mapping[str, Any]) -> set[str]:
    single = metadata.get("object_id")
    multiple = metadata.get("object_ids")
    if single is not None and multiple is not None:
        raise ValueError(
            "calibration evidence metadata cannot contain object_id and object_ids"
        )
    if single is not None:
        return {nonempty_string(single, name="calibration evidence object_id")}
    if multiple is None or isinstance(multiple, (str, bytes)):
        raise ValueError(
            "calibration evidence metadata must identify object_id or object_ids"
        )
    if not isinstance(multiple, Sequence):
        raise ValueError("calibration evidence object_ids must be a sequence")
    object_ids = canonical_sorted_strings(
        cast(Sequence[str], multiple),
        name="calibration evidence object_ids",
    )
    return set(object_ids)


def _validate_calibration_ledger(
    ledger: EvidenceUseLedgerV1,
    stage0: Deform360Stage0SelectionV1,
) -> None:
    if not isinstance(ledger, EvidenceUseLedgerV1):
        raise TypeError("ledger must be an EvidenceUseLedgerV1")
    if ledger.protocol_id != DEFORM360_CALIBRATION_PROTOCOL_ID:
        raise ValueError("calibration ledger protocol changed")
    if ledger.case_id != DEFORM360_CALIBRATION_LEDGER_CASE_ID:
        raise ValueError("calibration ledger case_id changed")
    if not ledger.entries:
        raise ValueError("calibration ledger must contain evidence entries")

    calibration_ids = {unit.object_id for unit in stage0.calibration_units}
    confirmation_ids = {unit.object_id for unit in stage0.confirmation_units}
    covered: set[str] = set()
    for entry in ledger.entries:
        if entry.inference_role != "calibration_only":
            raise ValueError("calibration ledger entries must use calibration_only")
        entry_ids = _entry_object_ids(entry.metadata)
        unexpected = entry_ids - calibration_ids
        if unexpected & confirmation_ids:
            raise ValueError("calibration ledger contains confirmation-object evidence")
        if unexpected:
            raise ValueError(
                "calibration ledger contains objects outside the Stage-0 "
                f"calibration cohort: {sorted(unexpected)}"
            )
        covered.update(entry_ids)
    missing = sorted(calibration_ids - covered)
    if missing:
        raise ValueError(
            f"calibration ledger does not cover every calibration object: {missing}"
        )


@dataclass(frozen=True)
class Deform360CalibrationExecutionSealV1:
    """Portable identity linking all pre-confirmation calibration artifacts."""

    implementation_revision: str
    stage0_snapshot_id: str
    stage0_source_sha256: str
    selection_sha256: str
    visual_provider_lock_id: str
    visual_calibration_lock_id: str
    calibration_bundle_id: str
    confirmation_opening_token: str
    evidence_use_ledger_id: str
    calibration_object_ids: Sequence[str]
    confirmation_object_ids: Sequence[str]
    source_artifacts: Mapping[str, str]
    calibration_payloads_opened: bool = True
    confirmation_payloads_opened: bool = False
    target_outcomes_used: bool = False
    status: str = DEFORM360_CALIBRATION_STATUS
    metadata: Mapping[str, Any] = field(default_factory=dict)
    protocol_id: str = DEFORM360_CALIBRATION_PROTOCOL_ID

    def __post_init__(self) -> None:
        protocol_id = nonempty_string(self.protocol_id, name="protocol_id")
        _require(
            protocol_id == DEFORM360_CALIBRATION_PROTOCOL_ID,
            "calibration execution protocol changed",
        )
        status = nonempty_string(self.status, name="status")
        _require(
            status == DEFORM360_CALIBRATION_STATUS,
            "calibration execution status changed",
        )
        revision = exact_revision(
            self.implementation_revision,
            name="implementation_revision",
        )
        digests = {
            name: sha256_digest(value, name=name)
            for name, value in (
                ("stage0_snapshot_id", self.stage0_snapshot_id),
                ("stage0_source_sha256", self.stage0_source_sha256),
                ("selection_sha256", self.selection_sha256),
                (
                    "visual_provider_lock_id",
                    self.visual_provider_lock_id,
                ),
                (
                    "visual_calibration_lock_id",
                    self.visual_calibration_lock_id,
                ),
                ("calibration_bundle_id", self.calibration_bundle_id),
                (
                    "confirmation_opening_token",
                    self.confirmation_opening_token,
                ),
                ("evidence_use_ledger_id", self.evidence_use_ledger_id),
            )
        }
        calibration_ids = canonical_sorted_strings(
            self.calibration_object_ids,
            name="calibration_object_ids",
        )
        confirmation_ids = canonical_sorted_strings(
            self.confirmation_object_ids,
            name="confirmation_object_ids",
        )
        if len(calibration_ids) != DEFORM360_FINITE_GROUP_CALIBRATION_GROUP_COUNT:
            raise ValueError(
                "calibration_object_ids must contain the registered "
                "10 independent objects"
            )
        if len(confirmation_ids) != 12:
            raise ValueError("confirmation_object_ids must contain 12 unique objects")
        _require(
            set(calibration_ids).isdisjoint(confirmation_ids),
            "calibration and confirmation object IDs overlap",
        )
        sources = source_artifact_mapping(
            self.source_artifacts,
            name="source_artifacts",
        )
        missing_sources = sorted(_REQUIRED_SOURCE_KEYS - set(sources))
        _require(
            not missing_sources,
            f"calibration execution source artifacts are incomplete: {missing_sources}",
        )
        _require(
            sources["sources/stage0/selection.json"] == digests["stage0_source_sha256"],
            "Stage-0 source bytes differ from the sealed digest",
        )
        calibration_opened = genuine_boolean(
            self.calibration_payloads_opened,
            name="calibration_payloads_opened",
        )
        confirmation_opened = genuine_boolean(
            self.confirmation_payloads_opened,
            name="confirmation_payloads_opened",
        )
        target_used = genuine_boolean(
            self.target_outcomes_used,
            name="target_outcomes_used",
        )
        _require(
            calibration_opened,
            "calibration execution must acknowledge opened calibration payloads",
        )
        _require(
            not confirmation_opened,
            "confirmation payloads were opened before the calibration seal",
        )
        _require(
            not target_used,
            "target outcomes were used before the calibration seal",
        )
        object.__setattr__(self, "protocol_id", protocol_id)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "implementation_revision", revision)
        for name, value in digests.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "calibration_object_ids",
            calibration_ids,
        )
        object.__setattr__(
            self,
            "confirmation_object_ids",
            confirmation_ids,
        )
        object.__setattr__(self, "source_artifacts", sources)
        object.__setattr__(
            self,
            "calibration_payloads_opened",
            calibration_opened,
        )
        object.__setattr__(
            self,
            "confirmation_payloads_opened",
            confirmation_opened,
        )
        object.__setattr__(self, "target_outcomes_used", target_used)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="calibration execution metadata",
            ),
        )

    def _descriptor(self) -> dict[str, object]:
        return {
            "schema": DEFORM360_CALIBRATION_EXECUTION_SCHEMA,
            "schema_version": DEFORM360_CALIBRATION_EXECUTION_VERSION,
            "semantics": DEFORM360_CALIBRATION_EXECUTION_SEMANTICS,
            "protocol_id": self.protocol_id,
            "status": self.status,
            "implementation_revision": self.implementation_revision,
            "stage0_snapshot_id": self.stage0_snapshot_id,
            "stage0_source_sha256": self.stage0_source_sha256,
            "selection_sha256": self.selection_sha256,
            "visual_provider_lock_id": self.visual_provider_lock_id,
            "visual_calibration_lock_id": self.visual_calibration_lock_id,
            "calibration_bundle_id": self.calibration_bundle_id,
            "confirmation_opening_token": self.confirmation_opening_token,
            "evidence_use_ledger_id": self.evidence_use_ledger_id,
            "calibration_object_ids": self.calibration_object_ids,
            "confirmation_object_ids": self.confirmation_object_ids,
            "source_artifacts": self.source_artifacts,
            "calibration_payloads_opened": self.calibration_payloads_opened,
            "confirmation_payloads_opened": (self.confirmation_payloads_opened),
            "target_outcomes_used": self.target_outcomes_used,
            "metadata": self.metadata,
            "claim_boundary": DEFORM360_CALIBRATION_EXECUTION_CLAIM_BOUNDARY,
        }

    @property
    def seal_id(self) -> str:
        return content_id(self._descriptor())

    def to_record(self) -> dict[str, object]:
        return {**self._descriptor(), "seal_id": self.seal_id}

    def summary(self) -> dict[str, object]:
        return {
            "seal_id": self.seal_id,
            "protocol_id": self.protocol_id,
            "status": self.status,
            "implementation_revision": self.implementation_revision,
            "stage0_snapshot_id": self.stage0_snapshot_id,
            "visual_provider_lock_id": self.visual_provider_lock_id,
            "visual_calibration_lock_id": self.visual_calibration_lock_id,
            "calibration_bundle_id": self.calibration_bundle_id,
            "confirmation_opening_token": self.confirmation_opening_token,
            "evidence_use_ledger_id": self.evidence_use_ledger_id,
            "calibration_object_count": len(self.calibration_object_ids),
            "confirmation_object_count": len(self.confirmation_object_ids),
            "calibration_payloads_opened": True,
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
            "claim_boundary": DEFORM360_CALIBRATION_EXECUTION_CLAIM_BOUNDARY,
        }

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        name: str = "Deform360 calibration execution seal",
    ) -> Deform360CalibrationExecutionSealV1:
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a JSON object")
        require_exact_fields(
            value,
            expected=_EXECUTION_FIELDS,
            name=name,
        )
        if value["schema"] != DEFORM360_CALIBRATION_EXECUTION_SCHEMA:
            raise ValueError(f"{name} schema changed")
        if (
            genuine_integer(
                value["schema_version"],
                name=f"{name} schema_version",
                minimum=1,
            )
            != DEFORM360_CALIBRATION_EXECUTION_VERSION
        ):
            raise ValueError(f"{name} schema_version changed")
        if value["semantics"] != DEFORM360_CALIBRATION_EXECUTION_SEMANTICS:
            raise ValueError(f"{name} semantics changed")
        if value["claim_boundary"] != DEFORM360_CALIBRATION_EXECUTION_CLAIM_BOUNDARY:
            raise ValueError(f"{name} claim boundary changed")
        result = cls(
            protocol_id=cast(str, value["protocol_id"]),
            status=cast(str, value["status"]),
            implementation_revision=cast(
                str,
                value["implementation_revision"],
            ),
            stage0_snapshot_id=cast(str, value["stage0_snapshot_id"]),
            stage0_source_sha256=cast(
                str,
                value["stage0_source_sha256"],
            ),
            selection_sha256=cast(str, value["selection_sha256"]),
            visual_provider_lock_id=cast(
                str,
                value["visual_provider_lock_id"],
            ),
            visual_calibration_lock_id=cast(
                str,
                value["visual_calibration_lock_id"],
            ),
            calibration_bundle_id=cast(
                str,
                value["calibration_bundle_id"],
            ),
            confirmation_opening_token=cast(
                str,
                value["confirmation_opening_token"],
            ),
            evidence_use_ledger_id=cast(
                str,
                value["evidence_use_ledger_id"],
            ),
            calibration_object_ids=cast(
                Sequence[str],
                value["calibration_object_ids"],
            ),
            confirmation_object_ids=cast(
                Sequence[str],
                value["confirmation_object_ids"],
            ),
            source_artifacts=cast(
                Mapping[str, str],
                value["source_artifacts"],
            ),
            calibration_payloads_opened=cast(
                bool,
                value["calibration_payloads_opened"],
            ),
            confirmation_payloads_opened=cast(
                bool,
                value["confirmation_payloads_opened"],
            ),
            target_outcomes_used=cast(bool, value["target_outcomes_used"]),
            metadata=cast(Mapping[str, Any], value["metadata"]),
        )
        declared_id = sha256_digest(
            value["seal_id"],
            name=f"{name} seal_id",
        )
        if declared_id != result.seal_id:
            raise ValueError(f"{name} seal_id does not match content")
        return result


class Deform360CalibrationExecutionArtifactsV1(NamedTuple):
    """All products emitted by one validated calibration execution."""

    visual_calibration_lock: Deform360VisualCalibrationLockV1
    calibration_bundle: Deform360CalibrationBundleV1
    execution_seal: Deform360CalibrationExecutionSealV1


def build_deform360_calibration_execution_seal(
    *,
    stage0_selection: Deform360Stage0SelectionV1,
    visual_provider_lock: Deform360VisualProviderLockV1,
    evidence_use_ledger: EvidenceUseLedgerV1,
    calibration_artifacts: Sequence[Deform360CalibrationArtifactRefV1],
    implementation_revision: str,
    source_artifacts: Mapping[str, str],
    metadata: Mapping[str, Any] | None = None,
) -> Deform360CalibrationExecutionArtifactsV1:
    """Build all Stage-1 artifacts from one complete calibration execution."""

    if not isinstance(stage0_selection, Deform360Stage0SelectionV1):
        raise TypeError("stage0_selection must be a Deform360Stage0SelectionV1")
    if not isinstance(
        visual_provider_lock,
        Deform360VisualProviderLockV1,
    ):
        raise TypeError("visual_provider_lock must be a Deform360VisualProviderLockV1")
    _validate_calibration_ledger(evidence_use_ledger, stage0_selection)
    revision = exact_revision(
        implementation_revision,
        name="implementation_revision",
    )
    artifacts = tuple(calibration_artifacts)
    components = deform360_calibration_component_ids(artifacts)
    calibration_ids = tuple(
        sorted(unit.object_id for unit in stage0_selection.calibration_units)
    )
    calibration_lock = Deform360VisualCalibrationLockV1(
        visual_provider_lock_id=visual_provider_lock.artifact_id,
        selection_lock_id=stage0_selection.selection_sha256,
        calibration_object_ids=calibration_ids,
        visual_calibration_id=components["visual"],
        contact_anchor_calibration_id=components["contact_anchor"],
        guard_calibration_id=components["guard"],
        interval_calibration_id=components["interval"],
        calibration_design_id=DEFORM360_FINITE_GROUP_CALIBRATION_DESIGN_ID,
        calibration_group_count=len(calibration_ids),
        conformal_rank=DEFORM360_FINITE_GROUP_CONFORMAL_RANK,
        metadata={
            "calibration_artifact_ref_ids": {
                artifact.role: artifact.ref_id for artifact in artifacts
            },
            "evidence_use_ledger_id": evidence_use_ledger.ledger_id,
        },
    )
    bundle = Deform360CalibrationBundleV1(
        selection_artifact_sha256=(stage0_selection.selection_artifact_sha256),
        content_selection_sha256=(stage0_selection.content_selection_sha256),
        dataset_revision=stage0_selection.dataset_revision,
        processing_revision=stage0_selection.processing_revision,
        implementation_revision=revision,
        calibration_units=stage0_selection.calibration_units,
        confirmation_units=stage0_selection.confirmation_units,
        calibration_artifacts=artifacts,
        evidence_use_ledger_id=evidence_use_ledger.ledger_id,
        source_artifacts=source_artifacts,
        metadata={
            "visual_provider_lock_id": visual_provider_lock.artifact_id,
            "visual_calibration_lock_id": calibration_lock.artifact_id,
        },
    )
    seal = Deform360CalibrationExecutionSealV1(
        implementation_revision=revision,
        stage0_snapshot_id=stage0_selection.snapshot_id,
        stage0_source_sha256=stage0_selection.source_sha256,
        selection_sha256=stage0_selection.selection_sha256,
        visual_provider_lock_id=visual_provider_lock.artifact_id,
        visual_calibration_lock_id=calibration_lock.artifact_id,
        calibration_bundle_id=bundle.bundle_id,
        confirmation_opening_token=bundle.confirmation_opening_token,
        evidence_use_ledger_id=evidence_use_ledger.ledger_id,
        calibration_object_ids=calibration_ids,
        confirmation_object_ids=tuple(
            unit.object_id for unit in stage0_selection.confirmation_units
        ),
        source_artifacts=source_artifacts,
        metadata=metadata or {},
    )
    products = Deform360CalibrationExecutionArtifactsV1(
        visual_calibration_lock=calibration_lock,
        calibration_bundle=bundle,
        execution_seal=seal,
    )
    verify_deform360_calibration_execution_artifacts(
        products,
        stage0_selection=stage0_selection,
        visual_provider_lock=visual_provider_lock,
        evidence_use_ledger=evidence_use_ledger,
    )
    return products


def verify_deform360_calibration_execution_artifacts(
    products: Deform360CalibrationExecutionArtifactsV1,
    *,
    stage0_selection: Deform360Stage0SelectionV1,
    visual_provider_lock: Deform360VisualProviderLockV1,
    evidence_use_ledger: EvidenceUseLedgerV1,
) -> None:
    """Independently verify all cross-artifact identities before confirmation."""

    if not isinstance(
        products,
        Deform360CalibrationExecutionArtifactsV1,
    ):
        raise TypeError("products must be a Deform360CalibrationExecutionArtifactsV1")
    _validate_calibration_ledger(evidence_use_ledger, stage0_selection)
    calibration = products.visual_calibration_lock
    bundle = products.calibration_bundle
    seal = products.execution_seal
    _require(
        calibration.visual_provider_lock_id == visual_provider_lock.artifact_id,
        "visual calibration lock provider identity changed",
    )
    _require(
        calibration.selection_lock_id == stage0_selection.selection_sha256,
        "visual calibration lock Stage-0 identity changed",
    )
    _require(
        bundle.selection_artifact_sha256 == stage0_selection.selection_artifact_sha256,
        "calibration bundle Stage-0 artifact changed",
    )
    _require(
        bundle.content_selection_sha256 == stage0_selection.content_selection_sha256,
        "calibration bundle content selection changed",
    )
    _require(
        bundle.dataset_revision == stage0_selection.dataset_revision,
        "calibration bundle dataset revision changed",
    )
    _require(
        bundle.processing_revision == stage0_selection.processing_revision,
        "calibration bundle processing revision changed",
    )
    _require(
        bundle.calibration_units == stage0_selection.calibration_units,
        "calibration bundle calibration cohort changed",
    )
    _require(
        bundle.confirmation_units == stage0_selection.confirmation_units,
        "calibration bundle confirmation cohort changed",
    )
    _require(
        bundle.evidence_use_ledger_id == evidence_use_ledger.ledger_id,
        "calibration bundle evidence ledger changed",
    )
    components = deform360_calibration_component_ids(bundle.calibration_artifacts)
    _require(
        calibration.visual_calibration_id == components["visual"],
        "visual calibration component ID changed",
    )
    _require(
        calibration.contact_anchor_calibration_id == components["contact_anchor"],
        "contact-anchor calibration component ID changed",
    )
    _require(
        calibration.guard_calibration_id == components["guard"],
        "guard calibration component ID changed",
    )
    _require(
        calibration.interval_calibration_id == components["interval"],
        "interval calibration component ID changed",
    )
    _require(
        seal.stage0_snapshot_id == stage0_selection.snapshot_id,
        "execution seal Stage-0 snapshot changed",
    )
    _require(
        seal.visual_provider_lock_id == visual_provider_lock.artifact_id,
        "execution seal provider lock changed",
    )
    _require(
        seal.visual_calibration_lock_id == calibration.artifact_id,
        "execution seal calibration lock changed",
    )
    _require(
        seal.calibration_bundle_id == bundle.bundle_id,
        "execution seal calibration bundle changed",
    )
    _require(
        seal.confirmation_opening_token == bundle.confirmation_opening_token,
        "execution seal confirmation token changed",
    )
    _require(
        seal.evidence_use_ledger_id == evidence_use_ledger.ledger_id,
        "execution seal evidence ledger changed",
    )
    _require(
        dict(bundle.source_artifacts) == dict(seal.source_artifacts),
        "execution seal source artifacts changed",
    )


def save_deform360_calibration_execution_seal(
    seal: Deform360CalibrationExecutionSealV1,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically persist one complete pre-confirmation execution seal."""

    if not isinstance(seal, Deform360CalibrationExecutionSealV1):
        raise TypeError("seal must be a Deform360CalibrationExecutionSealV1")
    write_atomic_json(seal.to_record(), path, overwrite=overwrite)


def load_deform360_calibration_execution_seal(
    path: str | Path,
) -> Deform360CalibrationExecutionSealV1:
    """Strictly load and independently revalidate an execution seal."""

    return Deform360CalibrationExecutionSealV1.from_mapping(
        load_strict_json_object(
            path,
            label="Deform360 calibration execution seal",
        )
    )


__all__ = [
    "DEFORM360_CALIBRATION_EXECUTION_CLAIM_BOUNDARY",
    "DEFORM360_CALIBRATION_EXECUTION_SCHEMA",
    "DEFORM360_CALIBRATION_EXECUTION_SEMANTICS",
    "DEFORM360_CALIBRATION_EXECUTION_VERSION",
    "DEFORM360_CALIBRATION_LEDGER_CASE_ID",
    "DEFORM360_STAGE0_SELECTION_SCHEMA",
    "DEFORM360_STAGE0_SNAPSHOT_SCHEMA",
    "DEFORM360_STAGE0_SNAPSHOT_VERSION",
    "Deform360CalibrationExecutionArtifactsV1",
    "Deform360CalibrationExecutionSealV1",
    "Deform360Stage0SelectionV1",
    "build_deform360_calibration_execution_seal",
    "deform360_calibration_component_ids",
    "file_sha256",
    "load_deform360_calibration_artifact_ref",
    "load_deform360_calibration_execution_seal",
    "load_deform360_stage0_selection",
    "save_deform360_calibration_execution_seal",
    "verify_deform360_calibration_execution_artifacts",
]
