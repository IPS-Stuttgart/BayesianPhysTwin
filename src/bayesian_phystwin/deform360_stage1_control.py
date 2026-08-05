"""Fail-closed control plane for the official-Hub Deform360 Stage-1 gate.

The Stage-0 selection, visual-provider lock, finite-group calibration design,
and calibration bundle are intentionally separate contracts. This module binds
those contracts into two explicit transitions:

1. a target-blind plan that authorizes calibration-payload access only; and
2. a calibration seal whose exact identity authorizes confirmation access.

Neither transition reads Deform360 camera, tactile, robot, geometry, or target
payloads. The caller remains responsible for executing the frozen calibration
method and for retaining every technical failure without replacement.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
)
from ._portable_contracts import (
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
    DEFORM360_CALIBRATION_ROLES,
    DEFORM360_CONFIRMATION_OBJECTS_PER_STRATUM,
    DEFORM360_DATASET_REPOSITORY,
    DEFORM360_PROCESSING_REPOSITORY,
    Deform360CalibrationArtifactRefV1,
    Deform360CalibrationBundleV1,
    Deform360CohortUnitV1,
    verify_deform360_confirmation_gate,
)
from .deform360_visual_provider_lock import (
    DEFORM360_FINITE_GROUP_CALIBRATION_DESIGN_ID,
    DEFORM360_FINITE_GROUP_CALIBRATION_GROUP_COUNT,
    DEFORM360_FINITE_GROUP_CONFORMAL_RANK,
    DEFORM360_VISUAL_PROVIDER_AMENDMENT_ID,
    DEFORM360_VISUOTACTILE_PROTOCOL_ID,
    Deform360VisualCalibrationLockV1,
    Deform360VisualProviderLockV1,
)
from .prob4d_provider_attestation import validate_prob4d_provider_attestation

DEFORM360_STAGE1_PLAN_SCHEMA = "bayesian-phystwin.deform360-stage1-plan"
DEFORM360_STAGE1_PLAN_VERSION = 1
DEFORM360_STAGE1_PLAN_SEMANTICS = (
    "target-blind-provider-lock-authorizes-calibration-payload-only-v1"
)
DEFORM360_STAGE1_PLAN_STATUS = "authorized-for-calibration-payload-access"
DEFORM360_STAGE1_SELECTION_SCHEMA = (
    "bayesian-phystwin/deform360-official-hub-selection-v1"
)
DEFORM360_STAGE1_CALIBRATION_DESIGN_SCHEMA = (
    "bayesian-phystwin/deform360-finite-group-calibration-design"
)
DEFORM360_STAGE1_CLAIM_BOUNDARY = (
    "Stage-1 information-order and provenance evidence only. A valid plan or "
    "seal does not establish observation competence, physical-query benefit, "
    "calibrated deployment uncertainty, Causal4D benefit, safety, or state of "
    "the art."
)

_PLAN_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "plan_id",
        "calibration_access_token",
        "protocol_id",
        "status",
        "selection_artifact_sha256",
        "canonical_selection_sha256",
        "content_selection_sha256",
        "visual_provider_lock_id",
        "calibration_design_id",
        "dataset_revision",
        "processing_revision",
        "calibration_units",
        "confirmation_units",
        "source_artifacts",
        "calibration_payloads_opened",
        "confirmation_payloads_opened",
        "target_outcomes_used",
        "replacement_allowed",
        "metadata",
        "claim_boundary",
    }
)
_SELECTION_UNIT_FIELDS = frozenset(
    {
        "object_id",
        "episode_id",
        "stratum",
        "metadata_path",
        "metadata_sha256",
    }
)

_VISUAL_GROUP_ROLES = (
    "visual_reliability_and_gauge",
    "normalized_evidence",
)
_CONTACT_GROUP_ROLES = (
    "contact_feature_and_grouping",
    "contact_linearization_and_covariance",
    "anchor_bias_prior",
)
_GUARD_GROUP_ROLES = (
    "physical_response_and_closure",
    "regret_guard",
)
_INTERVAL_GROUP_ROLES = ("conformal_interval",)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: str | Path, *, label: str) -> str:
    source = Path(path)
    if source.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link: {source}")
    if not source.is_file():
        raise ValueError(f"{label} is not a regular file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cohort_units(
    values: Sequence[Deform360CohortUnitV1],
    *,
    name: str,
    expected_per_stratum: int,
) -> tuple[Deform360CohortUnitV1, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    units = tuple(values)
    if not units or any(not isinstance(unit, Deform360CohortUnitV1) for unit in units):
        raise ValueError(f"{name} must contain Deform360CohortUnitV1 objects")
    units = tuple(sorted(units, key=lambda unit: (unit.stratum, unit.object_id)))
    object_ids = [unit.object_id for unit in units]
    unit_ids = [unit.unit_id for unit in units]
    _require(len(set(object_ids)) == len(object_ids), f"{name} repeats an object")
    _require(len(set(unit_ids)) == len(unit_ids), f"{name} repeats a unit")
    for stratum in ("sheet", "volumetric"):
        count = sum(unit.stratum == stratum for unit in units)
        _require(
            count == expected_per_stratum,
            f"{name} stratum {stratum} must contain exactly "
            f"{expected_per_stratum} objects",
        )
    return units


def _unit_from_selection(value: object, *, name: str) -> Deform360CohortUnitV1:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    require_exact_fields(value, expected=_SELECTION_UNIT_FIELDS, name=name)
    return Deform360CohortUnitV1(
        object_id=value["object_id"],
        episode_id=value["episode_id"],
        stratum=value["stratum"],
        metadata_path=value["metadata_path"],
        metadata_sha256=value["metadata_sha256"],
    )


def _selection_units(
    selection: Mapping[str, Any],
    *,
    role: str,
) -> tuple[Deform360CohortUnitV1, ...]:
    raw = selection.get(role)
    if not isinstance(raw, list):
        raise ValueError(f"Stage-0 selection {role} must be a JSON array")
    expected = (
        DEFORM360_CALIBRATION_OBJECTS_PER_STRATUM
        if role == "calibration"
        else DEFORM360_CONFIRMATION_OBJECTS_PER_STRATUM
    )
    return _cohort_units(
        tuple(
            _unit_from_selection(value, name=f"Stage-0 {role} unit {index}")
            for index, value in enumerate(raw)
        ),
        name=f"Stage-0 {role} units",
        expected_per_stratum=expected,
    )


def _plain_boolean(value: object, *, name: str) -> bool:
    return genuine_boolean(value, name=name)


def _validate_stage0_selection(
    path: str | Path,
) -> tuple[
    str,
    str,
    str,
    str,
    str,
    str,
    tuple[Deform360CohortUnitV1, ...],
    tuple[Deform360CohortUnitV1, ...],
]:
    selection = load_strict_json_object(path, label="Deform360 Stage-0 selection")
    _require(
        selection.get("schema") == DEFORM360_STAGE1_SELECTION_SCHEMA,
        "Stage-0 selection schema changed",
    )
    _require(
        genuine_integer(
            selection.get("schema_version"),
            name="Stage-0 selection schema_version",
            minimum=1,
        )
        == 1,
        "Stage-0 selection schema_version changed",
    )
    _require(
        selection.get("protocol_id") == DEFORM360_VISUOTACTILE_PROTOCOL_ID,
        "Stage-0 selection protocol changed",
    )
    selection_artifact_sha256 = sha256_digest(
        selection.get("selection_artifact_sha256"),
        name="Stage-0 selection artifact SHA-256",
    )
    artifact_descriptor = dict(selection)
    artifact_descriptor.pop("selection_artifact_sha256", None)
    _require(
        content_id(artifact_descriptor) == selection_artifact_sha256,
        "Stage-0 selection artifact identity does not match its descriptor",
    )
    selection_file_sha256 = _sha256_file(path, label="Stage-0 selection")
    canonical_selection_sha256 = sha256_digest(
        selection.get("selection_sha256"),
        name="Stage-0 canonical selection SHA-256",
    )
    selected_payload = selection.get("selection")
    if not isinstance(selected_payload, Mapping):
        raise ValueError("Stage-0 selection cohorts must be a JSON object")
    _require(
        content_id(selected_payload) == canonical_selection_sha256,
        "Stage-0 canonical selection identity does not match its cohort",
    )
    content_selection_sha256 = sha256_digest(
        selection.get("content_selection_sha256"),
        name="Stage-0 content selection SHA-256",
    )
    content_descriptor = dict(selection)
    for field_name in (
        "content_selection_sha256",
        "implementation_revision",
        "selection_artifact_sha256",
    ):
        content_descriptor.pop(field_name, None)
    _require(
        content_id(content_descriptor) == content_selection_sha256,
        "Stage-0 content selection identity does not match its descriptor",
    )

    dataset = selection.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("Stage-0 dataset must be a JSON object")
    _require(
        dataset.get("repo_id") == DEFORM360_DATASET_REPOSITORY,
        "Stage-0 dataset repository changed",
    )
    dataset_revision = exact_revision(
        dataset.get("resolved_revision"),
        name="Stage-0 dataset revision",
    )

    processing = selection.get("official_processing")
    if not isinstance(processing, Mapping):
        raise ValueError("Stage-0 official_processing must be a JSON object")
    _require(
        processing.get("repository") == DEFORM360_PROCESSING_REPOSITORY,
        "Stage-0 processing repository changed",
    )
    processing_revision = exact_revision(
        processing.get("revision"),
        name="Stage-0 processing revision",
    )

    boundary = selection.get("information_boundary")
    if not isinstance(boundary, Mapping):
        raise ValueError("Stage-0 information_boundary must be a JSON object")
    _require(
        _plain_boolean(
            boundary.get("object_directory_names_opened"),
            name="object_directory_names_opened",
        ),
        "Stage-0 object-directory inventory was not opened",
    )
    _require(
        _plain_boolean(
            boundary.get("object_metadata_json_opened"),
            name="object_metadata_json_opened",
        ),
        "Stage-0 object metadata was not opened",
    )
    for field_name in (
        "camera_media_opened",
        "geometry_annotations_opened",
        "robot_arrays_opened",
        "tactile_arrays_opened",
        "target_outcomes_opened",
    ):
        _require(
            not _plain_boolean(boundary.get(field_name), name=field_name),
            f"Stage-0 information boundary opened {field_name}",
        )
    _require(
        not _plain_boolean(
            selection.get("replacement_allowed_after_payload_access"),
            name="replacement_allowed_after_payload_access",
        ),
        "Stage-0 selection permits replacement after payload access",
    )

    selected = selection.get("selection")
    if not isinstance(selected, Mapping):
        raise ValueError("Stage-0 selection cohorts must be a JSON object")
    calibration_units = _selection_units(selected, role="calibration")
    confirmation_units = _selection_units(selected, role="confirmation")
    _require(
        {unit.object_id for unit in calibration_units}.isdisjoint(
            unit.object_id for unit in confirmation_units
        ),
        "Stage-0 calibration and confirmation objects overlap",
    )
    return (
        selection_artifact_sha256,
        selection_file_sha256,
        canonical_selection_sha256,
        content_selection_sha256,
        dataset_revision,
        processing_revision,
        calibration_units,
        confirmation_units,
    )


def _validate_amendment(
    path: str | Path,
    *,
    selection_artifact_sha256: str,
    canonical_selection_sha256: str,
    content_selection_sha256: str,
    dataset_revision: str,
) -> None:
    amendment = load_strict_json_object(
        path,
        label="Deform360 visual-provider amendment",
    )
    _require(
        amendment.get("amendment_id") == DEFORM360_VISUAL_PROVIDER_AMENDMENT_ID,
        "Deform360 visual-provider amendment ID changed",
    )
    _require(
        amendment.get("status")
        == "locked-before-selected-calibration-payload-access",
        "Deform360 visual-provider amendment status changed",
    )
    parent = amendment.get("parent_protocol")
    if not isinstance(parent, Mapping):
        raise ValueError("visual-provider amendment parent_protocol is missing")
    _require(
        parent.get("id") == DEFORM360_VISUOTACTILE_PROTOCOL_ID,
        "visual-provider amendment parent protocol changed",
    )
    selection_lock = amendment.get("selection_lock")
    if not isinstance(selection_lock, Mapping):
        raise ValueError("visual-provider amendment selection_lock is missing")
    expected_selection = {
        "complete_artifact_sha256": selection_artifact_sha256,
        "canonical_selection_sha256": canonical_selection_sha256,
        "content_selection_sha256": content_selection_sha256,
        "dataset_revision": dataset_revision,
    }
    for field_name, expected in expected_selection.items():
        _require(
            selection_lock.get(field_name) == expected,
            f"visual-provider amendment changed {field_name}",
        )
    design = amendment.get("finite_group_calibration_design")
    if not isinstance(design, Mapping):
        raise ValueError("visual-provider amendment calibration design is missing")
    _require(
        design.get("artifact_id")
        == DEFORM360_FINITE_GROUP_CALIBRATION_DESIGN_ID,
        "visual-provider amendment calibration design changed",
    )
    _require(
        design.get("calibration_group_count")
        == DEFORM360_FINITE_GROUP_CALIBRATION_GROUP_COUNT,
        "visual-provider amendment calibration group count changed",
    )
    _require(
        design.get("conformal_rank") == DEFORM360_FINITE_GROUP_CONFORMAL_RANK,
        "visual-provider amendment conformal rank changed",
    )
    _require(
        design.get("policy_selection_uses_calibration_outcomes") is False,
        "visual-provider amendment permits adaptive calibration reuse",
    )
    boundary = amendment.get("information_boundary")
    if not isinstance(boundary, Mapping):
        raise ValueError("visual-provider amendment information_boundary is missing")
    for field_name, raw in boundary.items():
        _require(
            not _plain_boolean(raw, name=f"amendment {field_name}"),
            f"visual-provider amendment opened {field_name}",
        )


def _validate_calibration_design(
    path: str | Path,
    *,
    canonical_selection_sha256: str,
) -> None:
    design = load_strict_json_object(
        path,
        label="Deform360 finite-group calibration design",
    )
    _require(
        design.get("schema") == DEFORM360_STAGE1_CALIBRATION_DESIGN_SCHEMA,
        "finite-group calibration design schema changed",
    )
    _require(
        genuine_integer(
            design.get("schema_version"),
            name="finite-group calibration design schema_version",
            minimum=1,
        )
        == 1,
        "finite-group calibration design schema_version changed",
    )
    _require(
        design.get("artifact_id")
        == DEFORM360_FINITE_GROUP_CALIBRATION_DESIGN_ID,
        "finite-group calibration design artifact changed",
    )
    _require(
        design.get("protocol_id") == DEFORM360_VISUOTACTILE_PROTOCOL_ID,
        "finite-group calibration design protocol changed",
    )
    _require(
        design.get("selection_sha256") == canonical_selection_sha256,
        "finite-group calibration design selection identity changed",
    )
    primary = design.get("primary_interval")
    if not isinstance(primary, Mapping):
        raise ValueError("finite-group primary_interval is missing")
    _require(
        primary.get("calibration_group_count")
        == DEFORM360_FINITE_GROUP_CALIBRATION_GROUP_COUNT,
        "finite-group calibration count changed",
    )
    _require(
        primary.get("finite_sample_rank") == DEFORM360_FINITE_GROUP_CONFORMAL_RANK,
        "finite-group conformal rank changed",
    )
    _require(
        primary.get("nominal_coverage") == 0.9,
        "finite-group nominal coverage changed",
    )
    _require(primary.get("pooling") == "pooled", "finite-group pooling changed")
    order = design.get("information_order")
    if not isinstance(order, Mapping):
        raise ValueError("finite-group information_order is missing")
    _require(
        order.get("predictor_score_guard_grouping_and_endpoints_frozen_before_scores")
        is True,
        "finite-group predictor was not frozen before scores",
    )
    _require(
        order.get("calibration_outcomes_used_for_policy_selection") is False,
        "finite-group design permits adaptive policy selection",
    )
    boundary = design.get("access_boundary")
    if not isinstance(boundary, Mapping):
        raise ValueError("finite-group access_boundary is missing")
    for field_name, raw in boundary.items():
        _require(
            not _plain_boolean(raw, name=f"calibration design {field_name}"),
            f"finite-group design opened {field_name}",
        )


@dataclass(frozen=True)
class Deform360Stage1PlanV1:
    """Exact pre-calibration authorization for the selected official-Hub cohort."""

    selection_artifact_sha256: str
    canonical_selection_sha256: str
    content_selection_sha256: str
    visual_provider_lock_id: str
    calibration_design_id: str
    dataset_revision: str
    processing_revision: str
    calibration_units: Sequence[Deform360CohortUnitV1]
    confirmation_units: Sequence[Deform360CohortUnitV1]
    source_artifacts: Mapping[str, str]
    protocol_id: str = DEFORM360_VISUOTACTILE_PROTOCOL_ID
    status: str = DEFORM360_STAGE1_PLAN_STATUS
    calibration_payloads_opened: bool = False
    confirmation_payloads_opened: bool = False
    target_outcomes_used: bool = False
    replacement_allowed: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        protocol_id = nonempty_string(self.protocol_id, name="protocol_id")
        _require(
            protocol_id == DEFORM360_VISUOTACTILE_PROTOCOL_ID,
            "Stage-1 plan protocol changed",
        )
        status = nonempty_string(self.status, name="status")
        _require(status == DEFORM360_STAGE1_PLAN_STATUS, "Stage-1 status changed")
        selection_artifact_sha256 = sha256_digest(
            self.selection_artifact_sha256,
            name="selection_artifact_sha256",
        )
        canonical_selection_sha256 = sha256_digest(
            self.canonical_selection_sha256,
            name="canonical_selection_sha256",
        )
        content_selection_sha256 = sha256_digest(
            self.content_selection_sha256,
            name="content_selection_sha256",
        )
        visual_provider_lock_id = sha256_digest(
            self.visual_provider_lock_id,
            name="visual_provider_lock_id",
        )
        calibration_design_id = sha256_digest(
            self.calibration_design_id,
            name="calibration_design_id",
        )
        _require(
            calibration_design_id == DEFORM360_FINITE_GROUP_CALIBRATION_DESIGN_ID,
            "Stage-1 calibration design changed",
        )
        dataset_revision = exact_revision(
            self.dataset_revision,
            name="dataset_revision",
        )
        processing_revision = exact_revision(
            self.processing_revision,
            name="processing_revision",
        )
        calibration_units = _cohort_units(
            self.calibration_units,
            name="calibration_units",
            expected_per_stratum=DEFORM360_CALIBRATION_OBJECTS_PER_STRATUM,
        )
        confirmation_units = _cohort_units(
            self.confirmation_units,
            name="confirmation_units",
            expected_per_stratum=DEFORM360_CONFIRMATION_OBJECTS_PER_STRATUM,
        )
        _require(
            {unit.object_id for unit in calibration_units}.isdisjoint(
                unit.object_id for unit in confirmation_units
            ),
            "Stage-1 calibration and confirmation objects overlap",
        )
        source_artifacts = source_artifact_mapping(
            self.source_artifacts,
            name="Stage-1 source artifacts",
        )
        calibration_payloads_opened = genuine_boolean(
            self.calibration_payloads_opened,
            name="calibration_payloads_opened",
        )
        confirmation_payloads_opened = genuine_boolean(
            self.confirmation_payloads_opened,
            name="confirmation_payloads_opened",
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
            not calibration_payloads_opened,
            "Stage-1 plan must be sealed before calibration payload access",
        )
        _require(
            not confirmation_payloads_opened,
            "Stage-1 plan must precede confirmation payload access",
        )
        _require(not target_outcomes_used, "Stage-1 plan must be target blind")
        _require(
            not replacement_allowed,
            "Stage-1 plan may not replace selected objects",
        )
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="Stage-1 plan metadata",
        )

        object.__setattr__(self, "protocol_id", protocol_id)
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "selection_artifact_sha256",
            selection_artifact_sha256,
        )
        object.__setattr__(
            self,
            "canonical_selection_sha256",
            canonical_selection_sha256,
        )
        object.__setattr__(
            self,
            "content_selection_sha256",
            content_selection_sha256,
        )
        object.__setattr__(
            self,
            "visual_provider_lock_id",
            visual_provider_lock_id,
        )
        object.__setattr__(self, "calibration_design_id", calibration_design_id)
        object.__setattr__(self, "dataset_revision", dataset_revision)
        object.__setattr__(self, "processing_revision", processing_revision)
        object.__setattr__(self, "calibration_units", calibration_units)
        object.__setattr__(self, "confirmation_units", confirmation_units)
        object.__setattr__(self, "source_artifacts", source_artifacts)
        object.__setattr__(
            self,
            "calibration_payloads_opened",
            calibration_payloads_opened,
        )
        object.__setattr__(
            self,
            "confirmation_payloads_opened",
            confirmation_payloads_opened,
        )
        object.__setattr__(self, "target_outcomes_used", target_outcomes_used)
        object.__setattr__(self, "replacement_allowed", replacement_allowed)
        object.__setattr__(self, "metadata", metadata)

    def _descriptor(self) -> dict[str, object]:
        return {
            "schema": DEFORM360_STAGE1_PLAN_SCHEMA,
            "schema_version": DEFORM360_STAGE1_PLAN_VERSION,
            "semantics": DEFORM360_STAGE1_PLAN_SEMANTICS,
            "protocol_id": self.protocol_id,
            "status": self.status,
            "selection_artifact_sha256": self.selection_artifact_sha256,
            "canonical_selection_sha256": self.canonical_selection_sha256,
            "content_selection_sha256": self.content_selection_sha256,
            "visual_provider_lock_id": self.visual_provider_lock_id,
            "calibration_design_id": self.calibration_design_id,
            "dataset_revision": self.dataset_revision,
            "processing_revision": self.processing_revision,
            "calibration_units": [unit.to_record() for unit in self.calibration_units],
            "confirmation_units": [
                unit.to_record() for unit in self.confirmation_units
            ],
            "source_artifacts": self.source_artifacts,
            "calibration_payloads_opened": self.calibration_payloads_opened,
            "confirmation_payloads_opened": self.confirmation_payloads_opened,
            "target_outcomes_used": self.target_outcomes_used,
            "replacement_allowed": self.replacement_allowed,
            "metadata": self.metadata,
            "claim_boundary": DEFORM360_STAGE1_CLAIM_BOUNDARY,
        }

    @property
    def plan_id(self) -> str:
        return content_id(self._descriptor())

    @property
    def calibration_access_token(self) -> str:
        return content_id(
            {
                "schema": "bayesian-phystwin.deform360-calibration-access-token",
                "schema_version": DEFORM360_STAGE1_PLAN_VERSION,
                "plan_id": self.plan_id,
                "visual_provider_lock_id": self.visual_provider_lock_id,
                "calibration_unit_ids": [
                    unit.unit_id for unit in self.calibration_units
                ],
                "status": self.status,
            }
        )

    def to_record(self) -> dict[str, object]:
        return {
            **self._descriptor(),
            "plan_id": self.plan_id,
            "calibration_access_token": self.calibration_access_token,
        }

    def summary(self) -> dict[str, object]:
        return {
            "schema": DEFORM360_STAGE1_PLAN_SCHEMA,
            "schema_version": DEFORM360_STAGE1_PLAN_VERSION,
            "plan_id": self.plan_id,
            "calibration_access_token": self.calibration_access_token,
            "protocol_id": self.protocol_id,
            "status": self.status,
            "selection_artifact_sha256": self.selection_artifact_sha256,
            "visual_provider_lock_id": self.visual_provider_lock_id,
            "calibration_design_id": self.calibration_design_id,
            "calibration_object_count": len(self.calibration_units),
            "confirmation_object_count": len(self.confirmation_units),
            "calibration_payloads_opened": False,
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
            "replacement_allowed": False,
            "claim_boundary": DEFORM360_STAGE1_CLAIM_BOUNDARY,
        }

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        name: str = "Deform360 Stage-1 plan",
    ) -> Deform360Stage1PlanV1:
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a JSON object")
        require_exact_fields(value, expected=_PLAN_FIELDS, name=name)
        _require(
            value["schema"] == DEFORM360_STAGE1_PLAN_SCHEMA,
            f"{name} schema changed",
        )
        _require(
            genuine_integer(
                value["schema_version"],
                name=f"{name} schema_version",
                minimum=1,
            )
            == DEFORM360_STAGE1_PLAN_VERSION,
            f"{name} schema_version changed",
        )
        _require(
            value["semantics"] == DEFORM360_STAGE1_PLAN_SEMANTICS,
            f"{name} semantics changed",
        )
        _require(
            value["claim_boundary"] == DEFORM360_STAGE1_CLAIM_BOUNDARY,
            f"{name} claim boundary changed",
        )
        calibration_raw = value["calibration_units"]
        confirmation_raw = value["confirmation_units"]
        if not isinstance(calibration_raw, list):
            raise ValueError(f"{name} calibration_units must be a JSON array")
        if not isinstance(confirmation_raw, list):
            raise ValueError(f"{name} confirmation_units must be a JSON array")
        result = cls(
            protocol_id=cast(str, value["protocol_id"]),
            status=cast(str, value["status"]),
            selection_artifact_sha256=cast(
                str,
                value["selection_artifact_sha256"],
            ),
            canonical_selection_sha256=cast(
                str,
                value["canonical_selection_sha256"],
            ),
            content_selection_sha256=cast(
                str,
                value["content_selection_sha256"],
            ),
            visual_provider_lock_id=cast(str, value["visual_provider_lock_id"]),
            calibration_design_id=cast(str, value["calibration_design_id"]),
            dataset_revision=cast(str, value["dataset_revision"]),
            processing_revision=cast(str, value["processing_revision"]),
            calibration_units=tuple(
                Deform360CohortUnitV1.from_mapping(
                    unit,
                    name=f"{name} calibration unit {index}",
                )
                for index, unit in enumerate(calibration_raw)
            ),
            confirmation_units=tuple(
                Deform360CohortUnitV1.from_mapping(
                    unit,
                    name=f"{name} confirmation unit {index}",
                )
                for index, unit in enumerate(confirmation_raw)
            ),
            source_artifacts=cast(Mapping[str, str], value["source_artifacts"]),
            calibration_payloads_opened=cast(
                bool,
                value["calibration_payloads_opened"],
            ),
            confirmation_payloads_opened=cast(
                bool,
                value["confirmation_payloads_opened"],
            ),
            target_outcomes_used=cast(bool, value["target_outcomes_used"]),
            replacement_allowed=cast(bool, value["replacement_allowed"]),
            metadata=cast(Mapping[str, Any], value["metadata"]),
        )
        supplied_id = sha256_digest(value["plan_id"], name=f"{name} plan_id")
        _require(
            supplied_id == result.plan_id,
            f"{name} plan_id does not match content",
        )
        supplied_token = sha256_digest(
            value["calibration_access_token"],
            name=f"{name} calibration_access_token",
        )
        _require(
            supplied_token == result.calibration_access_token,
            f"{name} calibration_access_token does not match content",
        )
        return result


def create_deform360_visual_provider_lock(
    *,
    provider_attestation_path: str | Path,
    motioncrafter_revision: str,
    model_set_id: str,
    root_seed: int,
    seed_policy: str,
    window_size: int,
    overlap: int,
    height: int,
    width: int,
    storage_dtype: str,
    initial_metric_frame_prior_id: str,
    additional_metric_anchor_policy: str,
    max_gauge_rank: int | None,
    minimum_retained_gauge_trace: float,
    metadata: Mapping[str, Any] | None = None,
) -> Deform360VisualProviderLockV1:
    """Build a provider lock from an independently validated Prob4D attestation."""

    attestation = load_strict_json_object(
        provider_attestation_path,
        label="Prob4D provider attestation",
    )
    source_revision = exact_revision(
        attestation.get("provider_revision"),
        name="Prob4D provider attestation revision",
    )
    validated = validate_prob4d_provider_attestation(
        attestation,
        source_revision=source_revision,
        require_claim_bearing=True,
    )
    attestation_sha256 = _sha256_file(
        provider_attestation_path,
        label="Prob4D provider attestation",
    )
    provider_metadata: dict[str, Any] = {
        "provider_attestation": {
            "export_mode": validated["export_mode"],
            "claim_bearing": validated["claim_bearing"],
            "calibration_artifact_ids": validated["calibration_artifact_ids"],
            "covariance_root_mode": validated["covariance_root_mode"],
            "composition_jacobian_mode": validated["composition_jacobian_mode"],
            "runtime_revision": validated["runtime_revision"],
        }
    }
    if metadata:
        provider_metadata["operator_metadata"] = dict(metadata)
    return Deform360VisualProviderLockV1(
        provider_revision=cast(str, validated["provider_revision"]),
        provider_manifest_id=cast(str, validated["provider_manifest_id"]),
        provider_attestation_sha256=attestation_sha256,
        motioncrafter_revision=motioncrafter_revision,
        model_set_id=model_set_id,
        root_seed=root_seed,
        seed_policy=seed_policy,
        window_size=window_size,
        overlap=overlap,
        height=height,
        width=width,
        storage_dtype=cast(Literal["float32", "float64"], storage_dtype),
        initial_metric_frame_prior_id=initial_metric_frame_prior_id,
        additional_metric_anchor_policy=cast(
            Literal["none", "independent_sparse"],
            additional_metric_anchor_policy,
        ),
        max_gauge_rank=max_gauge_rank,
        minimum_retained_gauge_trace=minimum_retained_gauge_trace,
        metadata=provider_metadata,
    )


def build_deform360_stage1_plan(
    *,
    selection_path: str | Path,
    provider_lock_path: str | Path,
    amendment_path: str | Path,
    calibration_design_path: str | Path,
    metadata: Mapping[str, Any] | None = None,
) -> Deform360Stage1PlanV1:
    """Bind the exact Stage-0, provider, and calibration-design identities."""

    (
        selection_artifact_sha256,
        selection_file_sha256,
        canonical_selection_sha256,
        content_selection_sha256,
        dataset_revision,
        processing_revision,
        calibration_units,
        confirmation_units,
    ) = _validate_stage0_selection(selection_path)
    _validate_amendment(
        amendment_path,
        selection_artifact_sha256=selection_artifact_sha256,
        canonical_selection_sha256=canonical_selection_sha256,
        content_selection_sha256=content_selection_sha256,
        dataset_revision=dataset_revision,
    )
    _validate_calibration_design(
        calibration_design_path,
        canonical_selection_sha256=canonical_selection_sha256,
    )
    provider_lock = Deform360VisualProviderLockV1.from_mapping(
        load_strict_json_object(
            provider_lock_path,
            label="Deform360 visual-provider lock",
        )
    )
    source_artifacts = {
        (
            "protocols/locks/"
            "deform360_official_hub_visuotactile_v1_selection.json"
        ): selection_file_sha256,
        (
            "protocols/amendments/"
            "deform360_official_hub_visuotactile_v1_visual_provider_lock.json"
        ): _sha256_file(
            amendment_path,
            label="visual-provider amendment",
        ),
        (
            "protocols/amendments/"
            "deform360_official_hub_visuotactile_v1_calibration_separation.json"
        ): _sha256_file(
            calibration_design_path,
            label="finite-group calibration design",
        ),
        (
            "protocols/locks/"
            "deform360_official_hub_visuotactile_v1_visual_provider.json"
        ): _sha256_file(
            provider_lock_path,
            label="visual-provider lock",
        ),
    }
    plan_metadata: dict[str, Any] = {
        "provider_revision": provider_lock.provider_revision,
        "provider_manifest_id": provider_lock.provider_manifest_id,
        "provider_attestation_sha256": provider_lock.provider_attestation_sha256,
        "motioncrafter_revision": provider_lock.motioncrafter_revision,
        "model_set_id": provider_lock.model_set_id,
        "seed_policy": provider_lock.seed_policy,
        "window": {
            "size": provider_lock.window_size,
            "overlap": provider_lock.overlap,
            "height": provider_lock.height,
            "width": provider_lock.width,
            "storage_dtype": provider_lock.storage_dtype,
        },
        "additional_metric_anchor_policy": (
            provider_lock.additional_metric_anchor_policy
        ),
        "maximum_gauge_rank": provider_lock.max_gauge_rank,
        "minimum_retained_gauge_trace": (
            provider_lock.minimum_retained_gauge_trace
        ),
    }
    if metadata:
        plan_metadata["operator_metadata"] = dict(metadata)
    return Deform360Stage1PlanV1(
        selection_artifact_sha256=selection_artifact_sha256,
        canonical_selection_sha256=canonical_selection_sha256,
        content_selection_sha256=content_selection_sha256,
        visual_provider_lock_id=provider_lock.artifact_id,
        calibration_design_id=DEFORM360_FINITE_GROUP_CALIBRATION_DESIGN_ID,
        dataset_revision=dataset_revision,
        processing_revision=processing_revision,
        calibration_units=calibration_units,
        confirmation_units=confirmation_units,
        source_artifacts=source_artifacts,
        metadata=plan_metadata,
    )


def verify_deform360_calibration_access(
    plan: Deform360Stage1PlanV1,
    *,
    expected_plan_id: str,
    expected_provider_lock_id: str,
    expected_selection_artifact_sha256: str,
) -> str:
    """Verify reviewed identities before any selected calibration payload opens."""

    if not isinstance(plan, Deform360Stage1PlanV1):
        raise TypeError("plan must be a Deform360Stage1PlanV1")
    expected_plan = sha256_digest(expected_plan_id, name="expected_plan_id")
    expected_provider = sha256_digest(
        expected_provider_lock_id,
        name="expected_provider_lock_id",
    )
    expected_selection = sha256_digest(
        expected_selection_artifact_sha256,
        name="expected_selection_artifact_sha256",
    )
    _require(plan.plan_id == expected_plan, "Stage-1 plan identity changed")
    _require(
        plan.visual_provider_lock_id == expected_provider,
        "Stage-1 visual-provider lock changed",
    )
    _require(
        plan.selection_artifact_sha256 == expected_selection,
        "Stage-1 selection artifact changed",
    )
    return plan.calibration_access_token


def _artifact_map(
    bundle: Deform360CalibrationBundleV1,
) -> dict[str, Deform360CalibrationArtifactRefV1]:
    result = {artifact.role: artifact for artifact in bundle.calibration_artifacts}
    _require(
        tuple(sorted(result)) == tuple(sorted(DEFORM360_CALIBRATION_ROLES)),
        "Deform360 calibration bundle roles changed",
    )
    return result


def _group_id(
    name: str,
    roles: Sequence[str],
    artifacts: Mapping[str, Deform360CalibrationArtifactRefV1],
) -> str:
    return content_id(
        {
            "schema": f"bayesian-phystwin.deform360-{name}-calibration-group",
            "schema_version": 1,
            "roles": [
                {"role": role, "ref_id": artifacts[role].ref_id} for role in roles
            ],
        }
    )


def derive_deform360_visual_calibration_lock(
    *,
    plan: Deform360Stage1PlanV1,
    provider_lock: Deform360VisualProviderLockV1,
    bundle: Deform360CalibrationBundleV1,
) -> Deform360VisualCalibrationLockV1:
    """Bridge the complete calibration bundle into the visual lock contract."""

    if not isinstance(plan, Deform360Stage1PlanV1):
        raise TypeError("plan must be a Deform360Stage1PlanV1")
    if not isinstance(provider_lock, Deform360VisualProviderLockV1):
        raise TypeError("provider_lock must be a Deform360VisualProviderLockV1")
    if not isinstance(bundle, Deform360CalibrationBundleV1):
        raise TypeError("bundle must be a Deform360CalibrationBundleV1")
    _require(
        provider_lock.artifact_id == plan.visual_provider_lock_id,
        "visual-provider lock does not match the Stage-1 plan",
    )
    _require(
        bundle.selection_artifact_sha256 == plan.selection_artifact_sha256,
        "calibration bundle selection artifact changed",
    )
    _require(
        bundle.content_selection_sha256 == plan.content_selection_sha256,
        "calibration bundle content selection changed",
    )
    _require(
        bundle.dataset_revision == plan.dataset_revision,
        "calibration bundle dataset revision changed",
    )
    _require(
        bundle.processing_revision == plan.processing_revision,
        "calibration bundle processing revision changed",
    )
    _require(
        tuple(bundle.calibration_units) == tuple(plan.calibration_units),
        "calibration bundle changed the calibration cohort",
    )
    _require(
        tuple(bundle.confirmation_units) == tuple(plan.confirmation_units),
        "calibration bundle changed the confirmation cohort",
    )

    artifacts = _artifact_map(bundle)
    visual_calibration_id = _group_id(
        "visual",
        _VISUAL_GROUP_ROLES,
        artifacts,
    )
    contact_anchor_calibration_id = _group_id(
        "contact-anchor",
        _CONTACT_GROUP_ROLES,
        artifacts,
    )
    guard_calibration_id = _group_id(
        "guard",
        _GUARD_GROUP_ROLES,
        artifacts,
    )
    interval_calibration_id = _group_id(
        "interval",
        _INTERVAL_GROUP_ROLES,
        artifacts,
    )
    return Deform360VisualCalibrationLockV1(
        visual_provider_lock_id=provider_lock.artifact_id,
        selection_lock_id=plan.selection_artifact_sha256,
        calibration_object_ids=tuple(
            unit.object_id for unit in bundle.calibration_units
        ),
        visual_calibration_id=visual_calibration_id,
        contact_anchor_calibration_id=contact_anchor_calibration_id,
        guard_calibration_id=guard_calibration_id,
        interval_calibration_id=interval_calibration_id,
        calibration_design_id=DEFORM360_FINITE_GROUP_CALIBRATION_DESIGN_ID,
        calibration_group_count=DEFORM360_FINITE_GROUP_CALIBRATION_GROUP_COUNT,
        conformal_rank=DEFORM360_FINITE_GROUP_CONFORMAL_RANK,
        metadata={
            "stage1_plan_id": plan.plan_id,
            "calibration_access_token": plan.calibration_access_token,
            "calibration_bundle_id": bundle.bundle_id,
            "confirmation_opening_token": bundle.confirmation_opening_token,
            "calibration_artifact_ref_ids": {
                role: artifacts[role].ref_id for role in sorted(artifacts)
            },
        },
    )


def verify_deform360_stage1_seal(
    *,
    plan: Deform360Stage1PlanV1,
    provider_lock: Deform360VisualProviderLockV1,
    bundle: Deform360CalibrationBundleV1,
    calibration_lock: Deform360VisualCalibrationLockV1,
) -> dict[str, object]:
    """Verify every reviewed Stage-1 identity before confirmation access."""

    expected = derive_deform360_visual_calibration_lock(
        plan=plan,
        provider_lock=provider_lock,
        bundle=bundle,
    )
    _require(
        calibration_lock.artifact_id == expected.artifact_id,
        "visual calibration lock does not match the complete calibration bundle",
    )
    token = verify_deform360_confirmation_gate(
        bundle,
        expected_bundle_id=bundle.bundle_id,
        expected_selection_artifact_sha256=plan.selection_artifact_sha256,
        expected_evidence_use_ledger_id=bundle.evidence_use_ledger_id,
    )
    return {
        "schema": "bayesian-phystwin.deform360-stage1-seal-summary",
        "schema_version": 1,
        "stage1_plan_id": plan.plan_id,
        "calibration_access_token": plan.calibration_access_token,
        "visual_provider_lock_id": provider_lock.artifact_id,
        "calibration_bundle_id": bundle.bundle_id,
        "visual_calibration_lock_id": calibration_lock.artifact_id,
        "confirmation_opening_token": token,
        "confirmation_payloads_opened": False,
        "target_outcomes_used": False,
        "replacement_allowed": False,
        "claim_boundary": DEFORM360_STAGE1_CLAIM_BOUNDARY,
    }


def verify_deform360_confirmation_access(
    *,
    plan: Deform360Stage1PlanV1,
    provider_lock: Deform360VisualProviderLockV1,
    bundle: Deform360CalibrationBundleV1,
    calibration_lock: Deform360VisualCalibrationLockV1,
    expected_plan_id: str,
    expected_provider_lock_id: str,
    expected_bundle_id: str,
    expected_calibration_lock_id: str,
    expected_selection_artifact_sha256: str,
    expected_evidence_use_ledger_id: str,
) -> dict[str, object]:
    """Require every reviewed identity before confirmation payload access."""

    summary = verify_deform360_stage1_seal(
        plan=plan,
        provider_lock=provider_lock,
        bundle=bundle,
        calibration_lock=calibration_lock,
    )
    reviewed = {
        "stage1 plan": (plan.plan_id, expected_plan_id),
        "visual-provider lock": (
            provider_lock.artifact_id,
            expected_provider_lock_id,
        ),
        "calibration bundle": (bundle.bundle_id, expected_bundle_id),
        "visual calibration lock": (
            calibration_lock.artifact_id,
            expected_calibration_lock_id,
        ),
        "Stage-0 selection artifact": (
            plan.selection_artifact_sha256,
            expected_selection_artifact_sha256,
        ),
        "evidence-use ledger": (
            bundle.evidence_use_ledger_id,
            expected_evidence_use_ledger_id,
        ),
    }
    for label, (observed, expected_raw) in reviewed.items():
        expected = sha256_digest(expected_raw, name=f"expected {label} ID")
        _require(observed == expected, f"reviewed {label} identity changed")
    token = verify_deform360_confirmation_gate(
        bundle,
        expected_bundle_id=expected_bundle_id,
        expected_selection_artifact_sha256=(
            expected_selection_artifact_sha256
        ),
        expected_evidence_use_ledger_id=expected_evidence_use_ledger_id,
    )
    _require(
        token == summary["confirmation_opening_token"],
        "confirmation-opening token changed after reviewed-identity validation",
    )
    return {
        **summary,
        "reviewed_identity_gate_passed": True,
    }


def save_deform360_stage1_plan(
    plan: Deform360Stage1PlanV1,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(plan, Deform360Stage1PlanV1):
        raise TypeError("plan must be a Deform360Stage1PlanV1")
    write_atomic_json(plan.to_record(), path, overwrite=overwrite)


def load_deform360_stage1_plan(path: str | Path) -> Deform360Stage1PlanV1:
    return Deform360Stage1PlanV1.from_mapping(
        load_strict_json_object(path, label="Deform360 Stage-1 plan")
    )


def save_deform360_visual_provider_lock_atomic(
    lock: Deform360VisualProviderLockV1,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(lock, Deform360VisualProviderLockV1):
        raise TypeError("lock must be a Deform360VisualProviderLockV1")
    write_atomic_json(lock.to_record(), path, overwrite=overwrite)


def save_deform360_visual_calibration_lock_atomic(
    lock: Deform360VisualCalibrationLockV1,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(lock, Deform360VisualCalibrationLockV1):
        raise TypeError("lock must be a Deform360VisualCalibrationLockV1")
    write_atomic_json(lock.to_record(), path, overwrite=overwrite)


__all__ = [
    "DEFORM360_STAGE1_CLAIM_BOUNDARY",
    "DEFORM360_STAGE1_PLAN_SCHEMA",
    "DEFORM360_STAGE1_PLAN_SEMANTICS",
    "DEFORM360_STAGE1_PLAN_STATUS",
    "DEFORM360_STAGE1_PLAN_VERSION",
    "Deform360Stage1PlanV1",
    "build_deform360_stage1_plan",
    "create_deform360_visual_provider_lock",
    "derive_deform360_visual_calibration_lock",
    "load_deform360_stage1_plan",
    "save_deform360_stage1_plan",
    "save_deform360_visual_calibration_lock_atomic",
    "save_deform360_visual_provider_lock_atomic",
    "verify_deform360_calibration_access",
    "verify_deform360_confirmation_access",
    "verify_deform360_stage1_seal",
]
