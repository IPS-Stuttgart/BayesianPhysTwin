"""Claim-bearing Stage-1 binding for Deform360 observability evidence.

The historical calibration-execution builder remains a low-level composition
primitive. A claim-bearing confirmation opening additionally requires one
successful calibration-source run, one supported object-balanced observability
report, exact retained bytes for both artifacts, and calibration selections that
cite the report. This module binds those facts into one content-addressed
confirmation-opening authorization without opening confirmation payloads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    plain_json,
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
    Deform360CalibrationArtifactRefV1,
    Deform360CalibrationRole,
)
from .deform360_calibration_execution import (
    Deform360CalibrationExecutionArtifactsV1,
    Deform360Stage0SelectionV1,
    build_deform360_calibration_execution_seal,
    verify_deform360_calibration_execution_artifacts,
)
from .deform360_calibration_observability_report import (
    Deform360CalibrationObservabilityReportV1,
)
from .deform360_calibration_source_run_record import (
    validate_deform360_calibration_source_run_record,
)
from .deform360_visual_provider_lock import (
    Deform360VisualProviderLockV1,
)
from .evidence_use_ledger import EvidenceUseLedgerV1

DEFORM360_CALIBRATION_SOURCE_RUN_RECORD_SOURCE_KEY = (
    "sources/additional/calibration-source-run-record.json"
)
DEFORM360_CALIBRATION_OBSERVABILITY_SOURCE_KEY = (
    "sources/additional/calibration-observability-report.json"
)
DEFORM360_OBSERVABILITY_REPORT_RUN_RECORD_SOURCE_KEY = (
    "sources/calibration-source/execution-manifest.json"
)
DEFORM360_OBSERVABILITY_BOUND_ROLES: tuple[
    Deform360CalibrationRole, ...
] = (
    "contact_linearization_and_covariance",
    "anchor_bias_prior",
    "physical_response_and_closure",
)

DEFORM360_CONFIRMATION_AUTHORIZATION_SCHEMA = (
    "bayesian-phystwin.deform360-confirmation-opening-authorization"
)
DEFORM360_CONFIRMATION_AUTHORIZATION_VERSION = 1
DEFORM360_CONFIRMATION_AUTHORIZATION_SEMANTICS = (
    "successful-calibration-and-supported-observability-bound-before-"
    "confirmation-access-v1"
)
DEFORM360_CONFIRMATION_AUTHORIZATION_STATUS = (
    "authorized-before-confirmation-payload-access"
)
DEFORM360_CONFIRMATION_AUTHORIZATION_CLAIM_BOUNDARY = (
    "Pre-confirmation information-order and artifact-binding evidence only. A "
    "valid authorization does not establish provider competence, physical-query "
    "accuracy, tactile benefit, uncertainty calibration, Causal4D benefit, "
    "deployment safety, or state of the art."
)

_BINDING_METADATA_FIELDS = frozenset(
    {
        "calibration_source_run_record_sha256",
        "calibration_source_run_record_file_sha256",
        "calibration_source_revision",
        "calibration_observability_report_id",
        "calibration_observability_report_file_sha256",
        "calibration_observability_physical_query_id",
        "calibration_observability_implementation_revision",
        "calibration_observability_support_passed",
    }
)
_AUTHORIZATION_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "authorization_id",
        "status",
        "execution_seal_id",
        "calibration_bundle_id",
        "confirmation_opening_token",
        "stage0_selection_artifact_sha256",
        "visual_provider_lock_id",
        "evidence_use_ledger_id",
        "calibration_source_run_record_sha256",
        "calibration_source_run_record_file_sha256",
        "calibration_source_revision",
        "calibration_observability_report_id",
        "calibration_observability_report_file_sha256",
        "calibration_observability_physical_query_id",
        "calibration_observability_implementation_revision",
        "source_artifacts",
        "confirmation_payloads_opened",
        "target_outcomes_used",
        "metadata",
        "claim_boundary",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _report_id(report: Deform360CalibrationObservabilityReportV1) -> str:
    value = report.report_id
    if value is None:
        raise ValueError("calibration observability report lacks a report_id")
    return sha256_digest(value, name="calibration observability report_id")


def _validated_run_record(value: Mapping[str, Any]) -> Mapping[str, Any]:
    validated = validate_deform360_calibration_source_run_record(value)
    if validated.get("status") != "succeeded" or validated.get("exit_code") != 0:
        raise ValueError("calibration source run did not succeed")
    if validated.get("confirmation_boundary_verified") is not True:
        raise ValueError("calibration source confirmation boundary is unverified")
    if validated.get("confirmation_payloads_opened") is not False:
        raise ValueError("calibration source reports confirmation payload access")
    gate = validated.get("support_gate")
    if not isinstance(gate, Mapping) or gate.get("support_passed") is not True:
        raise ValueError("calibration source support gate did not pass")
    return validated


def _source_digest(
    source_artifacts: Mapping[str, str],
    *,
    key: str,
    name: str,
) -> str:
    if key not in source_artifacts:
        raise ValueError(f"{name} bytes are not retained at {key}")
    return sha256_digest(source_artifacts[key], name=f"{name} file SHA-256")


def _binding_metadata(
    report: Deform360CalibrationObservabilityReportV1,
    run_record: Mapping[str, Any],
    *,
    run_record_file_sha256: str,
    report_file_sha256: str,
) -> dict[str, object]:
    return {
        "calibration_source_run_record_sha256": sha256_digest(
            run_record.get("record_sha256"),
            name="calibration source run record_sha256",
        ),
        "calibration_source_run_record_file_sha256": run_record_file_sha256,
        "calibration_source_revision": exact_revision(
            run_record.get("source_revision"),
            name="calibration source revision",
        ),
        "calibration_observability_report_id": _report_id(report),
        "calibration_observability_report_file_sha256": report_file_sha256,
        "calibration_observability_physical_query_id": report.physical_query_id,
        "calibration_observability_implementation_revision": (
            report.implementation_revision
        ),
        "calibration_observability_support_passed": True,
    }


def _augmented_metadata(
    metadata: Mapping[str, Any] | None,
    report: Deform360CalibrationObservabilityReportV1,
    run_record: Mapping[str, Any],
    *,
    run_record_file_sha256: str,
    report_file_sha256: str,
) -> dict[str, object]:
    caller = plain_json({} if metadata is None else metadata)
    if not isinstance(caller, dict):
        raise ValueError("calibration execution metadata must be a mapping")
    overlap = _BINDING_METADATA_FIELDS.intersection(caller)
    if overlap:
        raise ValueError(
            "calibration execution metadata reserves observability fields "
            f"{sorted(overlap)}"
        )
    return {
        **caller,
        **_binding_metadata(
            report,
            run_record,
            run_record_file_sha256=run_record_file_sha256,
            report_file_sha256=report_file_sha256,
        ),
    }


def validate_deform360_calibration_observability_binding(
    report: Deform360CalibrationObservabilityReportV1,
    calibration_source_run_record: Mapping[str, Any],
    *,
    stage0_selection: Deform360Stage0SelectionV1,
    visual_provider_lock: Deform360VisualProviderLockV1,
    calibration_artifacts: Sequence[Deform360CalibrationArtifactRefV1],
    source_artifacts: Mapping[str, str],
    calibration_source_run_record_file_sha256: str,
    calibration_observability_report_file_sha256: str,
) -> None:
    """Validate the exact run, report, cohort, role bindings, and file bytes."""

    if not isinstance(report, Deform360CalibrationObservabilityReportV1):
        raise TypeError(
            "calibration_observability_report must be a "
            "Deform360CalibrationObservabilityReportV1"
        )
    if not isinstance(stage0_selection, Deform360Stage0SelectionV1):
        raise TypeError("stage0_selection must be a Deform360Stage0SelectionV1")
    if not isinstance(visual_provider_lock, Deform360VisualProviderLockV1):
        raise TypeError(
            "visual_provider_lock must be a Deform360VisualProviderLockV1"
        )
    if not isinstance(calibration_source_run_record, Mapping):
        raise TypeError("calibration_source_run_record must be a mapping")
    run_record = _validated_run_record(calibration_source_run_record)
    report_id = _report_id(report)
    run_record_id = sha256_digest(
        run_record.get("record_sha256"),
        name="calibration source run record_sha256",
    )
    run_source_revision = exact_revision(
        run_record.get("source_revision"),
        name="calibration source revision",
    )
    run_file_sha256 = sha256_digest(
        calibration_source_run_record_file_sha256,
        name="calibration_source_run_record_file_sha256",
    )
    report_file_sha256 = sha256_digest(
        calibration_observability_report_file_sha256,
        name="calibration_observability_report_file_sha256",
    )

    _require(
        report.status == "completed-supported-calibration-observability",
        "calibration observability report did not complete with support",
    )
    _require(
        report.support_gate.get("support_passed") is True,
        "calibration observability support gate did not pass",
    )
    _require(
        run_record.get("selection_artifact_sha256")
        == stage0_selection.selection_artifact_sha256,
        "calibration source Stage-0 selection changed",
    )
    _require(
        run_record.get("visual_provider_lock_id")
        == visual_provider_lock.artifact_id,
        "calibration source visual-provider lock changed",
    )
    _require(
        report.selection_artifact_sha256
        == stage0_selection.selection_artifact_sha256,
        "calibration observability Stage-0 selection changed",
    )
    _require(
        report.visual_provider_lock_id == visual_provider_lock.artifact_id,
        "calibration observability visual-provider lock changed",
    )
    _require(
        report.calibration_source_run_record_sha256 == run_record_id,
        "calibration observability report cites another source run",
    )
    _require(
        report.calibration_source_revision == run_source_revision,
        "calibration observability source revision changed",
    )

    expected_units = {
        (unit.object_id, unit.episode_id, unit.stratum)
        for unit in stage0_selection.calibration_units
    }
    observed_units = {
        (case.object_id, case.episode_id, case.stratum) for case in report.cases
    }
    _require(
        observed_units == expected_units and len(report.cases) == len(expected_units),
        "calibration observability cohort differs from Stage 0",
    )

    retained = source_artifact_mapping(
        source_artifacts,
        name="Stage-1 source_artifacts",
    )
    retained_run = _source_digest(
        retained,
        key=DEFORM360_CALIBRATION_SOURCE_RUN_RECORD_SOURCE_KEY,
        name="calibration source run record",
    )
    retained_report = _source_digest(
        retained,
        key=DEFORM360_CALIBRATION_OBSERVABILITY_SOURCE_KEY,
        name="calibration observability report",
    )
    _require(
        retained_run == run_file_sha256,
        "retained calibration source run-record bytes changed",
    )
    _require(
        retained_report == report_file_sha256,
        "retained calibration observability-report bytes changed",
    )
    report_run_file = _source_digest(
        report.source_artifacts,
        key=DEFORM360_OBSERVABILITY_REPORT_RUN_RECORD_SOURCE_KEY,
        name="observability report calibration source run record",
    )
    _require(
        report_run_file == run_file_sha256,
        "observability report was built from different run-record bytes",
    )

    if isinstance(calibration_artifacts, (str, bytes)):
        raise ValueError("calibration_artifacts must be a sequence")
    artifacts = tuple(calibration_artifacts)
    if any(
        not isinstance(value, Deform360CalibrationArtifactRefV1)
        for value in artifacts
    ):
        raise ValueError("calibration_artifacts contain an unsupported value")
    by_role = {artifact.role: artifact for artifact in artifacts}
    if len(by_role) != len(artifacts):
        raise ValueError("calibration_artifacts repeat a role")
    missing = sorted(set(DEFORM360_OBSERVABILITY_BOUND_ROLES) - set(by_role))
    if missing:
        raise ValueError(f"observability-bound calibration roles are missing: {missing}")
    for role in DEFORM360_OBSERVABILITY_BOUND_ROLES:
        artifact = by_role[role]
        _require(
            artifact.selection_evidence_id == report_id,
            f"{role} does not bind the calibration observability report",
        )


def build_deform360_calibration_execution_seal_with_observability(
    *,
    stage0_selection: Deform360Stage0SelectionV1,
    visual_provider_lock: Deform360VisualProviderLockV1,
    evidence_use_ledger: EvidenceUseLedgerV1,
    calibration_artifacts: Sequence[Deform360CalibrationArtifactRefV1],
    calibration_source_run_record: Mapping[str, Any],
    calibration_observability_report: Deform360CalibrationObservabilityReportV1,
    calibration_source_run_record_file_sha256: str,
    calibration_observability_report_file_sha256: str,
    implementation_revision: str,
    source_artifacts: Mapping[str, str],
    metadata: Mapping[str, Any] | None = None,
) -> Deform360CalibrationExecutionArtifactsV1:
    """Build a Stage-1 seal that is claim-bound to run and observability evidence."""

    validate_deform360_calibration_observability_binding(
        calibration_observability_report,
        calibration_source_run_record,
        stage0_selection=stage0_selection,
        visual_provider_lock=visual_provider_lock,
        calibration_artifacts=calibration_artifacts,
        source_artifacts=source_artifacts,
        calibration_source_run_record_file_sha256=(
            calibration_source_run_record_file_sha256
        ),
        calibration_observability_report_file_sha256=(
            calibration_observability_report_file_sha256
        ),
    )
    validated_run = _validated_run_record(calibration_source_run_record)
    products = build_deform360_calibration_execution_seal(
        stage0_selection=stage0_selection,
        visual_provider_lock=visual_provider_lock,
        evidence_use_ledger=evidence_use_ledger,
        calibration_artifacts=calibration_artifacts,
        implementation_revision=implementation_revision,
        source_artifacts=source_artifacts,
        metadata=_augmented_metadata(
            metadata,
            calibration_observability_report,
            validated_run,
            run_record_file_sha256=sha256_digest(
                calibration_source_run_record_file_sha256,
                name="calibration_source_run_record_file_sha256",
            ),
            report_file_sha256=sha256_digest(
                calibration_observability_report_file_sha256,
                name="calibration_observability_report_file_sha256",
            ),
        ),
    )
    verify_deform360_calibration_execution_observability_binding(
        products,
        calibration_source_run_record=validated_run,
        calibration_observability_report=calibration_observability_report,
        calibration_source_run_record_file_sha256=(
            calibration_source_run_record_file_sha256
        ),
        calibration_observability_report_file_sha256=(
            calibration_observability_report_file_sha256
        ),
        stage0_selection=stage0_selection,
        visual_provider_lock=visual_provider_lock,
        evidence_use_ledger=evidence_use_ledger,
    )
    return products


def verify_deform360_calibration_execution_observability_binding(
    products: Deform360CalibrationExecutionArtifactsV1,
    *,
    calibration_source_run_record: Mapping[str, Any],
    calibration_observability_report: Deform360CalibrationObservabilityReportV1,
    calibration_source_run_record_file_sha256: str,
    calibration_observability_report_file_sha256: str,
    stage0_selection: Deform360Stage0SelectionV1,
    visual_provider_lock: Deform360VisualProviderLockV1,
    evidence_use_ledger: EvidenceUseLedgerV1,
) -> None:
    """Revalidate the ordinary seal and every report-dependent identity."""

    verify_deform360_calibration_execution_artifacts(
        products,
        stage0_selection=stage0_selection,
        visual_provider_lock=visual_provider_lock,
        evidence_use_ledger=evidence_use_ledger,
    )
    validated_run = _validated_run_record(calibration_source_run_record)
    validate_deform360_calibration_observability_binding(
        calibration_observability_report,
        validated_run,
        stage0_selection=stage0_selection,
        visual_provider_lock=visual_provider_lock,
        calibration_artifacts=products.calibration_bundle.calibration_artifacts,
        source_artifacts=products.calibration_bundle.source_artifacts,
        calibration_source_run_record_file_sha256=(
            calibration_source_run_record_file_sha256
        ),
        calibration_observability_report_file_sha256=(
            calibration_observability_report_file_sha256
        ),
    )
    expected_metadata = _binding_metadata(
        calibration_observability_report,
        validated_run,
        run_record_file_sha256=sha256_digest(
            calibration_source_run_record_file_sha256,
            name="calibration_source_run_record_file_sha256",
        ),
        report_file_sha256=sha256_digest(
            calibration_observability_report_file_sha256,
            name="calibration_observability_report_file_sha256",
        ),
    )
    seal_metadata = dict(products.execution_seal.metadata)
    for key, value in expected_metadata.items():
        _require(
            seal_metadata.get(key) == value,
            f"execution seal observability metadata changed: {key}",
        )
    _require(
        dict(products.calibration_bundle.source_artifacts)
        == dict(products.execution_seal.source_artifacts),
        "observability source bytes differ between bundle and execution seal",
    )


@dataclass(frozen=True)
class Deform360ConfirmationOpeningAuthorizationV1:
    """One exact authorization to open the frozen confirmation cohort once."""

    execution_seal_id: str
    calibration_bundle_id: str
    confirmation_opening_token: str
    stage0_selection_artifact_sha256: str
    visual_provider_lock_id: str
    evidence_use_ledger_id: str
    calibration_source_run_record_sha256: str
    calibration_source_run_record_file_sha256: str
    calibration_source_revision: str
    calibration_observability_report_id: str
    calibration_observability_report_file_sha256: str
    calibration_observability_physical_query_id: str
    calibration_observability_implementation_revision: str
    source_artifacts: Mapping[str, str]
    confirmation_payloads_opened: bool = False
    target_outcomes_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    status: str = DEFORM360_CONFIRMATION_AUTHORIZATION_STATUS
    authorization_id: str | None = None

    def __post_init__(self) -> None:
        digests = {
            name: sha256_digest(value, name=name)
            for name, value in (
                ("execution_seal_id", self.execution_seal_id),
                ("calibration_bundle_id", self.calibration_bundle_id),
                ("confirmation_opening_token", self.confirmation_opening_token),
                (
                    "stage0_selection_artifact_sha256",
                    self.stage0_selection_artifact_sha256,
                ),
                ("visual_provider_lock_id", self.visual_provider_lock_id),
                ("evidence_use_ledger_id", self.evidence_use_ledger_id),
                (
                    "calibration_source_run_record_sha256",
                    self.calibration_source_run_record_sha256,
                ),
                (
                    "calibration_source_run_record_file_sha256",
                    self.calibration_source_run_record_file_sha256,
                ),
                (
                    "calibration_observability_report_id",
                    self.calibration_observability_report_id,
                ),
                (
                    "calibration_observability_report_file_sha256",
                    self.calibration_observability_report_file_sha256,
                ),
                (
                    "calibration_observability_physical_query_id",
                    self.calibration_observability_physical_query_id,
                ),
            )
        }
        revisions = {
            "calibration_source_revision": exact_revision(
                self.calibration_source_revision,
                name="calibration_source_revision",
            ),
            "calibration_observability_implementation_revision": exact_revision(
                self.calibration_observability_implementation_revision,
                name="calibration_observability_implementation_revision",
            ),
        }
        status = nonempty_string(self.status, name="status")
        _require(
            status == DEFORM360_CONFIRMATION_AUTHORIZATION_STATUS,
            "confirmation authorization status changed",
        )
        sources = source_artifact_mapping(
            self.source_artifacts,
            name="confirmation authorization source_artifacts",
        )
        required_sources = {
            DEFORM360_CALIBRATION_SOURCE_RUN_RECORD_SOURCE_KEY: digests[
                "calibration_source_run_record_file_sha256"
            ],
            DEFORM360_CALIBRATION_OBSERVABILITY_SOURCE_KEY: digests[
                "calibration_observability_report_file_sha256"
            ],
        }
        for key, expected in required_sources.items():
            _require(
                sources.get(key) == expected,
                f"confirmation authorization source bytes changed: {key}",
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
            not confirmation_opened,
            "confirmation payloads were opened before authorization",
        )
        _require(
            not target_used,
            "target outcomes were used before authorization",
        )
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="confirmation authorization metadata",
        )

        for name, value in {**digests, **revisions}.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "source_artifacts", sources)
        object.__setattr__(
            self,
            "confirmation_payloads_opened",
            confirmation_opened,
        )
        object.__setattr__(self, "target_outcomes_used", target_used)
        object.__setattr__(self, "metadata", metadata)

        expected_id = content_id(self.identity_record())
        if self.authorization_id is not None:
            supplied_id = sha256_digest(
                self.authorization_id,
                name="authorization_id",
            )
            _require(
                supplied_id == expected_id,
                "confirmation authorization_id does not match content",
            )
        object.__setattr__(self, "authorization_id", expected_id)

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": DEFORM360_CONFIRMATION_AUTHORIZATION_SCHEMA,
            "schema_version": DEFORM360_CONFIRMATION_AUTHORIZATION_VERSION,
            "semantics": DEFORM360_CONFIRMATION_AUTHORIZATION_SEMANTICS,
            "status": self.status,
            "execution_seal_id": self.execution_seal_id,
            "calibration_bundle_id": self.calibration_bundle_id,
            "confirmation_opening_token": self.confirmation_opening_token,
            "stage0_selection_artifact_sha256": (
                self.stage0_selection_artifact_sha256
            ),
            "visual_provider_lock_id": self.visual_provider_lock_id,
            "evidence_use_ledger_id": self.evidence_use_ledger_id,
            "calibration_source_run_record_sha256": (
                self.calibration_source_run_record_sha256
            ),
            "calibration_source_run_record_file_sha256": (
                self.calibration_source_run_record_file_sha256
            ),
            "calibration_source_revision": self.calibration_source_revision,
            "calibration_observability_report_id": (
                self.calibration_observability_report_id
            ),
            "calibration_observability_report_file_sha256": (
                self.calibration_observability_report_file_sha256
            ),
            "calibration_observability_physical_query_id": (
                self.calibration_observability_physical_query_id
            ),
            "calibration_observability_implementation_revision": (
                self.calibration_observability_implementation_revision
            ),
            "source_artifacts": plain_json(self.source_artifacts),
            "confirmation_payloads_opened": self.confirmation_payloads_opened,
            "target_outcomes_used": self.target_outcomes_used,
            "metadata": plain_json(self.metadata),
            "claim_boundary": DEFORM360_CONFIRMATION_AUTHORIZATION_CLAIM_BOUNDARY,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.identity_record(), "authorization_id": self.authorization_id}

    def summary(self) -> dict[str, object]:
        return {
            "authorization_id": self.authorization_id,
            "status": self.status,
            "execution_seal_id": self.execution_seal_id,
            "calibration_bundle_id": self.calibration_bundle_id,
            "confirmation_opening_token": self.confirmation_opening_token,
            "calibration_source_run_record_sha256": (
                self.calibration_source_run_record_sha256
            ),
            "calibration_observability_report_id": (
                self.calibration_observability_report_id
            ),
            "calibration_observability_physical_query_id": (
                self.calibration_observability_physical_query_id
            ),
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
            "claim_boundary": DEFORM360_CONFIRMATION_AUTHORIZATION_CLAIM_BOUNDARY,
        }

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        name: str = "Deform360 confirmation opening authorization",
    ) -> Deform360ConfirmationOpeningAuthorizationV1:
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a JSON object")
        require_exact_fields(value, expected=_AUTHORIZATION_FIELDS, name=name)
        if value["schema"] != DEFORM360_CONFIRMATION_AUTHORIZATION_SCHEMA:
            raise ValueError(f"{name} schema changed")
        version = genuine_integer(
            value["schema_version"],
            name=f"{name} schema_version",
            minimum=1,
        )
        if version != DEFORM360_CONFIRMATION_AUTHORIZATION_VERSION:
            raise ValueError(f"{name} schema_version changed")
        if value["semantics"] != DEFORM360_CONFIRMATION_AUTHORIZATION_SEMANTICS:
            raise ValueError(f"{name} semantics changed")
        if value["claim_boundary"] != (
            DEFORM360_CONFIRMATION_AUTHORIZATION_CLAIM_BOUNDARY
        ):
            raise ValueError(f"{name} claim boundary changed")
        sources = value["source_artifacts"]
        metadata = value["metadata"]
        if not isinstance(sources, Mapping):
            raise ValueError(f"{name} source_artifacts must be a JSON object")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{name} metadata must be a JSON object")
        return cls(
            execution_seal_id=value["execution_seal_id"],
            calibration_bundle_id=value["calibration_bundle_id"],
            confirmation_opening_token=value["confirmation_opening_token"],
            stage0_selection_artifact_sha256=value[
                "stage0_selection_artifact_sha256"
            ],
            visual_provider_lock_id=value["visual_provider_lock_id"],
            evidence_use_ledger_id=value["evidence_use_ledger_id"],
            calibration_source_run_record_sha256=value[
                "calibration_source_run_record_sha256"
            ],
            calibration_source_run_record_file_sha256=value[
                "calibration_source_run_record_file_sha256"
            ],
            calibration_source_revision=value["calibration_source_revision"],
            calibration_observability_report_id=value[
                "calibration_observability_report_id"
            ],
            calibration_observability_report_file_sha256=value[
                "calibration_observability_report_file_sha256"
            ],
            calibration_observability_physical_query_id=value[
                "calibration_observability_physical_query_id"
            ],
            calibration_observability_implementation_revision=value[
                "calibration_observability_implementation_revision"
            ],
            source_artifacts=value["source_artifacts"],
            confirmation_payloads_opened=value["confirmation_payloads_opened"],
            target_outcomes_used=value["target_outcomes_used"],
            metadata=value["metadata"],
            status=value["status"],
            authorization_id=value["authorization_id"],
        )


def build_deform360_confirmation_opening_authorization(
    products: Deform360CalibrationExecutionArtifactsV1,
    *,
    calibration_source_run_record: Mapping[str, Any],
    calibration_observability_report: Deform360CalibrationObservabilityReportV1,
    calibration_source_run_record_file_sha256: str,
    calibration_observability_report_file_sha256: str,
    stage0_selection: Deform360Stage0SelectionV1,
    visual_provider_lock: Deform360VisualProviderLockV1,
    evidence_use_ledger: EvidenceUseLedgerV1,
    metadata: Mapping[str, Any] | None = None,
) -> Deform360ConfirmationOpeningAuthorizationV1:
    """Verify the complete Stage-1 chain and emit its opening authorization."""

    verify_deform360_calibration_execution_observability_binding(
        products,
        calibration_source_run_record=calibration_source_run_record,
        calibration_observability_report=calibration_observability_report,
        calibration_source_run_record_file_sha256=(
            calibration_source_run_record_file_sha256
        ),
        calibration_observability_report_file_sha256=(
            calibration_observability_report_file_sha256
        ),
        stage0_selection=stage0_selection,
        visual_provider_lock=visual_provider_lock,
        evidence_use_ledger=evidence_use_ledger,
    )
    run_record = _validated_run_record(calibration_source_run_record)
    return Deform360ConfirmationOpeningAuthorizationV1(
        execution_seal_id=products.execution_seal.seal_id,
        calibration_bundle_id=products.calibration_bundle.bundle_id,
        confirmation_opening_token=(
            products.calibration_bundle.confirmation_opening_token
        ),
        stage0_selection_artifact_sha256=(
            stage0_selection.selection_artifact_sha256
        ),
        visual_provider_lock_id=visual_provider_lock.artifact_id,
        evidence_use_ledger_id=evidence_use_ledger.ledger_id,
        calibration_source_run_record_sha256=run_record["record_sha256"],
        calibration_source_run_record_file_sha256=(
            calibration_source_run_record_file_sha256
        ),
        calibration_source_revision=run_record["source_revision"],
        calibration_observability_report_id=_report_id(
            calibration_observability_report
        ),
        calibration_observability_report_file_sha256=(
            calibration_observability_report_file_sha256
        ),
        calibration_observability_physical_query_id=(
            calibration_observability_report.physical_query_id
        ),
        calibration_observability_implementation_revision=(
            calibration_observability_report.implementation_revision
        ),
        source_artifacts={
            DEFORM360_CALIBRATION_SOURCE_RUN_RECORD_SOURCE_KEY: sha256_digest(
                calibration_source_run_record_file_sha256,
                name="calibration_source_run_record_file_sha256",
            ),
            DEFORM360_CALIBRATION_OBSERVABILITY_SOURCE_KEY: sha256_digest(
                calibration_observability_report_file_sha256,
                name="calibration_observability_report_file_sha256",
            ),
        },
        metadata={} if metadata is None else metadata,
    )


def save_deform360_confirmation_opening_authorization(
    value: Deform360ConfirmationOpeningAuthorizationV1,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically persist one validated confirmation-opening authorization."""

    if not isinstance(value, Deform360ConfirmationOpeningAuthorizationV1):
        raise TypeError(
            "value must be a Deform360ConfirmationOpeningAuthorizationV1"
        )
    write_atomic_json(value.to_record(), path, overwrite=overwrite)


def load_deform360_confirmation_opening_authorization(
    path: str | Path,
) -> Deform360ConfirmationOpeningAuthorizationV1:
    """Strictly load and independently revalidate one authorization."""

    return Deform360ConfirmationOpeningAuthorizationV1.from_mapping(
        load_strict_json_object(
            path,
            label="Deform360 confirmation opening authorization",
        )
    )


__all__ = [
    "DEFORM360_CALIBRATION_OBSERVABILITY_SOURCE_KEY",
    "DEFORM360_CALIBRATION_SOURCE_RUN_RECORD_SOURCE_KEY",
    "DEFORM360_CONFIRMATION_AUTHORIZATION_CLAIM_BOUNDARY",
    "DEFORM360_CONFIRMATION_AUTHORIZATION_SCHEMA",
    "DEFORM360_CONFIRMATION_AUTHORIZATION_SEMANTICS",
    "DEFORM360_CONFIRMATION_AUTHORIZATION_STATUS",
    "DEFORM360_CONFIRMATION_AUTHORIZATION_VERSION",
    "DEFORM360_OBSERVABILITY_BOUND_ROLES",
    "Deform360ConfirmationOpeningAuthorizationV1",
    "build_deform360_calibration_execution_seal_with_observability",
    "build_deform360_confirmation_opening_authorization",
    "load_deform360_confirmation_opening_authorization",
    "save_deform360_confirmation_opening_authorization",
    "validate_deform360_calibration_observability_binding",
    "verify_deform360_calibration_execution_observability_binding",
]
