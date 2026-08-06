"""Claim-bearing Stage-1 binding for Deform360 observability evidence.

The historical execution builder remains a low-level composition primitive. The
actual confirmation-opening path must additionally validate one supported,
object-balanced calibration observability report, retain its exact bytes in the
sealed source tree, and require the contact/physical-response selections to cite
that report as their selection evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from ._canonical_contracts import plain_json
from ._portable_contracts import sha256_digest
from .deform360_calibration_bundle import (
    Deform360CalibrationArtifactRefV1,
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
from .deform360_visual_provider_lock import (
    Deform360VisualProviderLockV1,
)
from .evidence_use_ledger import EvidenceUseLedgerV1

DEFORM360_CALIBRATION_OBSERVABILITY_SOURCE_KEY = (
    "sources/calibration/observability-report.json"
)
DEFORM360_OBSERVABILITY_BOUND_ROLES = (
    "contact_linearization_and_covariance",
    "anchor_bias_prior",
    "physical_response_and_closure",
)

_BINDING_METADATA_FIELDS = frozenset(
    {
        "calibration_observability_report_id",
        "calibration_observability_physical_query_id",
        "calibration_observability_source_revision",
        "calibration_observability_implementation_revision",
        "calibration_observability_support_passed",
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


def _binding_metadata(
    report: Deform360CalibrationObservabilityReportV1,
) -> dict[str, object]:
    return {
        "calibration_observability_report_id": _report_id(report),
        "calibration_observability_physical_query_id": report.physical_query_id,
        "calibration_observability_source_revision": (
            report.calibration_source_revision
        ),
        "calibration_observability_implementation_revision": (
            report.implementation_revision
        ),
        "calibration_observability_support_passed": True,
    }


def _augmented_metadata(
    metadata: Mapping[str, Any] | None,
    report: Deform360CalibrationObservabilityReportV1,
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
    return {**caller, **_binding_metadata(report)}


def validate_deform360_calibration_observability_binding(
    report: Deform360CalibrationObservabilityReportV1,
    *,
    stage0_selection: Deform360Stage0SelectionV1,
    visual_provider_lock: Deform360VisualProviderLockV1,
    calibration_artifacts: Sequence[Deform360CalibrationArtifactRefV1],
    source_artifacts: Mapping[str, str],
) -> None:
    """Validate the report, exact cohort, role bindings, and retained bytes."""

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
    report_id = _report_id(report)
    _require(
        report.status == "completed-supported-calibration-observability",
        "calibration observability report did not complete with support",
    )
    _require(
        report.support_gate.get("support_passed") is True,
        "calibration observability support gate did not pass",
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

    if not isinstance(source_artifacts, Mapping):
        raise ValueError("source_artifacts must be a mapping")
    if DEFORM360_CALIBRATION_OBSERVABILITY_SOURCE_KEY not in source_artifacts:
        raise ValueError("calibration observability report bytes are not retained")
    sha256_digest(
        source_artifacts[DEFORM360_CALIBRATION_OBSERVABILITY_SOURCE_KEY],
        name="calibration observability report file SHA-256",
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
        artifact = by_role[cast(Any, role)]
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
    calibration_observability_report: Deform360CalibrationObservabilityReportV1,
    implementation_revision: str,
    source_artifacts: Mapping[str, str],
    metadata: Mapping[str, Any] | None = None,
) -> Deform360CalibrationExecutionArtifactsV1:
    """Build the claim-bearing Stage-1 seal with observability evidence."""

    validate_deform360_calibration_observability_binding(
        calibration_observability_report,
        stage0_selection=stage0_selection,
        visual_provider_lock=visual_provider_lock,
        calibration_artifacts=calibration_artifacts,
        source_artifacts=source_artifacts,
    )
    products = build_deform360_calibration_execution_seal(
        stage0_selection=stage0_selection,
        visual_provider_lock=visual_provider_lock,
        evidence_use_ledger=evidence_use_ledger,
        calibration_artifacts=calibration_artifacts,
        implementation_revision=implementation_revision,
        source_artifacts=source_artifacts,
        metadata=_augmented_metadata(metadata, calibration_observability_report),
    )
    verify_deform360_calibration_execution_observability_binding(
        products,
        calibration_observability_report=calibration_observability_report,
        stage0_selection=stage0_selection,
        visual_provider_lock=visual_provider_lock,
        evidence_use_ledger=evidence_use_ledger,
    )
    return products


def verify_deform360_calibration_execution_observability_binding(
    products: Deform360CalibrationExecutionArtifactsV1,
    *,
    calibration_observability_report: Deform360CalibrationObservabilityReportV1,
    stage0_selection: Deform360Stage0SelectionV1,
    visual_provider_lock: Deform360VisualProviderLockV1,
    evidence_use_ledger: EvidenceUseLedgerV1,
) -> None:
    """Revalidate the ordinary seal and its report-dependent identities."""

    verify_deform360_calibration_execution_artifacts(
        products,
        stage0_selection=stage0_selection,
        visual_provider_lock=visual_provider_lock,
        evidence_use_ledger=evidence_use_ledger,
    )
    validate_deform360_calibration_observability_binding(
        calibration_observability_report,
        stage0_selection=stage0_selection,
        visual_provider_lock=visual_provider_lock,
        calibration_artifacts=products.calibration_bundle.calibration_artifacts,
        source_artifacts=products.calibration_bundle.source_artifacts,
    )
    expected_metadata = _binding_metadata(calibration_observability_report)
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


__all__ = [
    "DEFORM360_CALIBRATION_OBSERVABILITY_SOURCE_KEY",
    "DEFORM360_OBSERVABILITY_BOUND_ROLES",
    "build_deform360_calibration_execution_seal_with_observability",
    "validate_deform360_calibration_observability_binding",
    "verify_deform360_calibration_execution_observability_binding",
]
