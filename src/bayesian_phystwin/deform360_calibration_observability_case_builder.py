"""Build strict per-object Deform360 calibration observability cases.

The object-balanced report contract intentionally accepts only complete case
artifacts. This module supplies the missing claim-bearing producer: it binds one
case to the exact data-free locks, successful calibration-source terminal record,
validated plan/download/result chain, and ordinary source files used to construct
the nuisance-marginalized comparison.

No confirmation payload or target outcome is admitted here.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from ._deform360_calibration_artifact_chain import (
    download_summary,
    plan_summary,
    result_summary,
    source_lock_summary,
)
from ._deform360_calibration_run_common import load_json_object
from ._portable_contracts import content_id, exact_revision, nonempty_string
from .deform360_calibration_observability_report import (
    Deform360CalibrationObservabilityCaseV1,
)
from .deform360_calibration_source_run_record import (
    validate_deform360_calibration_source_run_record,
)

DEFORM360_OBSERVABILITY_CASE_INPUT_SCHEMA = (
    "bayesian-phystwin.deform360-calibration-observability-case-input"
)
DEFORM360_OBSERVABILITY_CASE_INPUT_VERSION = 1

_SourceRole = Literal[
    "reference-marginal-precision",
    "candidate-marginal-precision",
    "physical-query-jacobian",
    "contact-anchor",
]
_Stratum = Literal["sheet", "volumetric"]

_SOURCE_LOCK_KEYS = (
    "source_locks_available",
    "source_locks_valid",
    "source_locks_error",
    "source_protocol_file_sha256",
    "source_protocol_sha256",
    "stage0_protocol_file_sha256",
    "stage0_protocol_sha256",
    "selection_lock_file_sha256",
    "selection_artifact_sha256",
    "content_selection_sha256",
    "visual_provider_lock_file_sha256",
    "visual_provider_lock_id",
)
_PLAN_KEYS = (
    "plan_available",
    "plan_valid",
    "plan_error",
    "plan_file_sha256",
    "plan_sha256",
    "plan_support_gate",
)
_DOWNLOAD_KEYS = (
    "download_available",
    "download_valid",
    "download_error",
    "download_file_sha256",
    "download_sha256",
)
_RESULT_KEYS = (
    "result_available",
    "result_valid",
    "result_error",
    "result_file_sha256",
    "result_sha256",
    "support_gate",
)


@dataclass(frozen=True)
class _CaseContext:
    object_id: str
    episode_id: int
    stratum: _Stratum
    result_row: Mapping[str, Any]
    selection_artifact_sha256: str
    visual_provider_lock_id: str
    run_record_sha256: str
    source_artifacts: Mapping[str, str]


def _ordinary_file(path: str | Path, *, name: str) -> Path:
    absolute = Path(path).absolute()
    candidates = (absolute, *absolute.parents)
    if any(candidate.is_symlink() for candidate in candidates):
        raise ValueError(f"{name} path must not contain symlinks: {path}")
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{name} does not exist: {path}") from error
    if not resolved.is_file():
        raise ValueError(f"{name} must be an ordinary file: {path}")
    return resolved


def _read_ordinary_bytes(
    path: str | Path,
    *,
    name: str,
) -> tuple[Path, bytes, str]:
    ordinary = _ordinary_file(path, name=name)
    try:
        data = ordinary.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {name}: {path}") from error
    if not data:
        raise ValueError(f"{name} must not be empty")
    return ordinary, data, hashlib.sha256(data).hexdigest()


def _role_artifact_id(role: _SourceRole, source_sha256: str) -> str:
    return content_id(
        {
            "schema": DEFORM360_OBSERVABILITY_CASE_INPUT_SCHEMA,
            "schema_version": DEFORM360_OBSERVABILITY_CASE_INPUT_VERSION,
            "role": role,
            "source_sha256": source_sha256,
        }
    )


def _load_npy_matrix(
    path: str | Path,
    *,
    name: str,
) -> tuple[np.ndarray, str]:
    ordinary, data, digest = _read_ordinary_bytes(path, name=name)
    if ordinary.suffix.lower() != ".npy":
        raise ValueError(f"{name} must be an ordinary .npy file")
    try:
        loaded = np.load(io.BytesIO(data), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot load {name}: {path}") from error
    if not isinstance(loaded, np.ndarray):
        raise ValueError(f"{name} must contain one NumPy array")
    if loaded.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    matrix = np.array(loaded, dtype=np.dtype("<f8"), copy=True, order="C")
    if matrix.ndim != 2 or not matrix.size or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite nonempty matrix")
    matrix.setflags(write=False)
    return matrix, digest


def physical_query_id_from_path(path: str | Path) -> str:
    """Return the role-aware content identity of one query Jacobian file."""

    _, digest = _load_npy_matrix(path, name="physical query Jacobian")
    return _role_artifact_id("physical-query-jacobian", digest)


def _require_successful_run(record: Mapping[str, Any]) -> None:
    if record.get("status") != "succeeded" or record.get("exit_code") != 0:
        raise ValueError("calibration-source terminal record did not succeed")
    if record.get("confirmation_boundary_verified") is not True:
        raise ValueError("calibration-source confirmation boundary is unverified")
    if record.get("confirmation_payloads_opened") is not False:
        raise ValueError("calibration-source record reports confirmation access")
    for key in (
        "source_locks_valid",
        "plan_valid",
        "download_valid",
        "result_valid",
    ):
        if record.get(key) is not True:
            raise ValueError(f"calibration-source record has invalid {key}")
    gate = record.get("support_gate")
    if not isinstance(gate, Mapping) or gate.get("support_passed") is not True:
        raise ValueError("calibration-source support gate did not pass")


def _require_equal_summary(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    keys: tuple[str, ...],
    name: str,
) -> None:
    for key in keys:
        if observed.get(key) != expected.get(key):
            raise ValueError(f"{name} differs from terminal record: {key}")


def _required_digest(
    value: Mapping[str, Any],
    key: str,
    *,
    name: str,
) -> str:
    digest = value.get(key)
    if type(digest) is not str or len(digest) != 64:
        raise ValueError(f"{name} lacks {key}")
    try:
        int(digest, 16)
    except ValueError as error:
        raise ValueError(f"{name} has an invalid {key}") from error
    return digest


def _result_row(
    value: Mapping[str, Any],
    *,
    object_id: str,
    episode_id: int,
    stratum: str,
) -> Mapping[str, Any]:
    rows = value.get("objects")
    if not isinstance(rows, list):
        raise ValueError("calibration-source result rows are missing")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("object_id") == object_id
    ]
    if len(matches) != 1:
        raise ValueError("calibration-source result does not contain one object row")
    row = matches[0]
    if row.get("episode_id") != episode_id or row.get("stratum") != stratum:
        raise ValueError("calibration-source result row changed object identity")
    return row


def _load_context(
    *,
    source_protocol_path: str | Path,
    stage0_protocol_path: str | Path,
    selection_lock_path: str | Path,
    visual_provider_lock_path: str | Path,
    calibration_source_plan_path: str | Path,
    calibration_source_download_path: str | Path,
    calibration_source_run_record_path: str | Path,
    calibration_source_result_path: str | Path,
    object_id: str,
) -> _CaseContext:
    object_name = nonempty_string(object_id, name="object_id")
    if object_name != object_name.strip():
        raise ValueError("object_id must not contain surrounding whitespace")

    source_protocol = _ordinary_file(
        source_protocol_path,
        name="calibration-source protocol",
    )
    stage0_protocol = _ordinary_file(
        stage0_protocol_path,
        name="Stage-0 protocol",
    )
    selection_lock = _ordinary_file(
        selection_lock_path,
        name="Stage-0 selection lock",
    )
    visual_lock = _ordinary_file(
        visual_provider_lock_path,
        name="visual-provider lock",
    )
    plan_path = _ordinary_file(
        calibration_source_plan_path,
        name="calibration-source plan",
    )
    download_path = _ordinary_file(
        calibration_source_download_path,
        name="calibration-source download manifest",
    )
    run_path = _ordinary_file(
        calibration_source_run_record_path,
        name="calibration-source terminal record",
    )
    result_path = _ordinary_file(
        calibration_source_result_path,
        name="calibration-source result",
    )

    raw_run_record, run_file_sha256 = load_json_object(run_path)
    run_record = validate_deform360_calibration_source_run_record(raw_run_record)
    _require_successful_run(run_record)
    processing_revision = exact_revision(
        run_record.get("processing_revision"),
        name="processing_revision",
    )
    source_locks, expected_units, confirmation_ids = source_lock_summary(
        source_protocol_json=source_protocol,
        stage0_protocol_json=stage0_protocol,
        selection_lock=selection_lock,
        visual_provider_lock=visual_lock,
        processing_revision=processing_revision,
    )
    if source_locks.get("source_locks_valid") is not True:
        raise ValueError("supplied calibration source locks are invalid")
    _require_equal_summary(
        source_locks,
        run_record,
        keys=_SOURCE_LOCK_KEYS,
        name="source-lock summary",
    )

    if object_name in confirmation_ids:
        raise ValueError("confirmation objects are forbidden")
    unit = expected_units.get(object_name)
    if unit is None:
        raise ValueError("object_id is not in the frozen calibration cohort")
    episode_id, stratum, _, _ = unit

    plan, expected_identities, planned_ids = plan_summary(
        plan_path,
        processing_revision=processing_revision,
        source_locks=source_locks,
        expected_units=expected_units,
        confirmation_ids=confirmation_ids,
    )
    if plan.get("plan_valid") is not True:
        raise ValueError("supplied calibration-source plan is invalid")
    _require_equal_summary(
        plan,
        run_record,
        keys=_PLAN_KEYS,
        name="plan summary",
    )

    download = download_summary(
        download_path,
        plan_sha256=cast(str, plan.get("plan_sha256")),
        planned_ids=planned_ids,
        confirmation_ids=confirmation_ids,
    )
    if download.get("download_valid") is not True:
        raise ValueError("supplied calibration-source download is invalid")
    _require_equal_summary(
        download,
        run_record,
        keys=_DOWNLOAD_KEYS,
        name="download summary",
    )

    result = result_summary(
        result_path,
        processing_revision=processing_revision,
        plan_sha256=cast(str, plan.get("plan_sha256")),
        download_sha256=cast(str, download.get("download_sha256")),
        expected_identities=expected_identities,
        planned_ids=planned_ids,
    )
    if result.get("result_valid") is not True:
        raise ValueError("supplied calibration-source result is invalid")
    _require_equal_summary(
        result,
        run_record,
        keys=_RESULT_KEYS,
        name="result summary",
    )
    result_value, result_file_sha256 = load_json_object(result_path)
    if result_file_sha256 != result.get("result_file_sha256"):
        raise ValueError("calibration-source result changed after validation")
    row = _result_row(
        result_value,
        object_id=object_name,
        episode_id=episode_id,
        stratum=stratum,
    )

    source_artifacts = {
        "sources/calibration-source/protocol.json": _required_digest(
            source_locks,
            "source_protocol_file_sha256",
            name="source-lock summary",
        ),
        "sources/stage0/protocol.json": _required_digest(
            source_locks,
            "stage0_protocol_file_sha256",
            name="source-lock summary",
        ),
        "sources/stage0/selection.json": _required_digest(
            source_locks,
            "selection_lock_file_sha256",
            name="source-lock summary",
        ),
        "sources/locks/visual-provider-lock.json": _required_digest(
            source_locks,
            "visual_provider_lock_file_sha256",
            name="source-lock summary",
        ),
        "sources/calibration-source/plan.json": _required_digest(
            plan,
            "plan_file_sha256",
            name="plan summary",
        ),
        "sources/calibration-source/download.json": _required_digest(
            download,
            "download_file_sha256",
            name="download summary",
        ),
        "sources/calibration-source/execution-manifest.json": run_file_sha256,
        "sources/calibration-source/result.json": _required_digest(
            result,
            "result_file_sha256",
            name="result summary",
        ),
    }
    return _CaseContext(
        object_id=object_name,
        episode_id=episode_id,
        stratum=cast(_Stratum, stratum),
        result_row=row,
        selection_artifact_sha256=_required_digest(
            source_locks,
            "selection_artifact_sha256",
            name="source-lock summary",
        ),
        visual_provider_lock_id=_required_digest(
            source_locks,
            "visual_provider_lock_id",
            name="source-lock summary",
        ),
        run_record_sha256=_required_digest(
            run_record,
            "record_sha256",
            name="terminal record",
        ),
        source_artifacts=source_artifacts,
    )


def build_evaluated_case_from_paths(
    *,
    source_protocol_path: str | Path,
    stage0_protocol_path: str | Path,
    selection_lock_path: str | Path,
    visual_provider_lock_path: str | Path,
    calibration_source_plan_path: str | Path,
    calibration_source_download_path: str | Path,
    calibration_source_run_record_path: str | Path,
    calibration_source_result_path: str | Path,
    object_id: str,
    implementation_revision: str,
    reference_marginal_precision_path: str | Path,
    candidate_marginal_precision_path: str | Path,
    query_jacobian_path: str | Path,
    contact_anchor_artifact_path: str | Path,
) -> Deform360CalibrationObservabilityCaseV1:
    """Build one evaluated case from exact ordinary source files."""

    implementation = exact_revision(
        implementation_revision,
        name="implementation_revision",
    )
    context = _load_context(
        source_protocol_path=source_protocol_path,
        stage0_protocol_path=stage0_protocol_path,
        selection_lock_path=selection_lock_path,
        visual_provider_lock_path=visual_provider_lock_path,
        calibration_source_plan_path=calibration_source_plan_path,
        calibration_source_download_path=calibration_source_download_path,
        calibration_source_run_record_path=calibration_source_run_record_path,
        calibration_source_result_path=calibration_source_result_path,
        object_id=object_id,
    )
    if context.result_row.get("status") != "source_prepared":
        raise ValueError("evaluated case requires a source_prepared object")

    reference, reference_sha = _load_npy_matrix(
        reference_marginal_precision_path,
        name="reference marginal precision",
    )
    candidate, candidate_sha = _load_npy_matrix(
        candidate_marginal_precision_path,
        name="candidate marginal precision",
    )
    query, query_sha = _load_npy_matrix(
        query_jacobian_path,
        name="physical query Jacobian",
    )
    _, _, anchor_sha = _read_ordinary_bytes(
        contact_anchor_artifact_path,
        name="contact-anchor artifact",
    )

    logical_root = f"sources/observability/{context.object_id}"
    sources = {
        **context.source_artifacts,
        f"{logical_root}/reference-marginal-precision.npy": reference_sha,
        f"{logical_root}/candidate-marginal-precision.npy": candidate_sha,
        f"{logical_root}/physical-query-jacobian.npy": query_sha,
        f"{logical_root}/contact-anchor.artifact": anchor_sha,
    }
    return Deform360CalibrationObservabilityCaseV1(
        selection_artifact_sha256=context.selection_artifact_sha256,
        visual_provider_lock_id=context.visual_provider_lock_id,
        calibration_source_run_record_sha256=context.run_record_sha256,
        implementation_revision=implementation,
        object_id=context.object_id,
        episode_id=context.episode_id,
        stratum=context.stratum,
        physical_query_id=_role_artifact_id(
            "physical-query-jacobian",
            query_sha,
        ),
        status="evaluated",
        reference_state_artifact_id=_role_artifact_id(
            "reference-marginal-precision",
            reference_sha,
        ),
        candidate_state_artifact_id=_role_artifact_id(
            "candidate-marginal-precision",
            candidate_sha,
        ),
        contact_anchor_artifact_id=_role_artifact_id(
            "contact-anchor",
            anchor_sha,
        ),
        reference_marginal_precision=reference,
        candidate_marginal_precision=candidate,
        query_jacobian=query,
        source_artifacts=sources,
        information_boundary={
            "calibration_payloads_opened": True,
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
            "replacement_allowed": False,
        },
    )


def build_technical_failure_case_from_paths(
    *,
    source_protocol_path: str | Path,
    stage0_protocol_path: str | Path,
    selection_lock_path: str | Path,
    visual_provider_lock_path: str | Path,
    calibration_source_plan_path: str | Path,
    calibration_source_download_path: str | Path,
    calibration_source_run_record_path: str | Path,
    calibration_source_result_path: str | Path,
    object_id: str,
    implementation_revision: str,
    query_jacobian_path: str | Path,
    failure_evidence_path: str | Path,
    failure_reason: str,
) -> Deform360CalibrationObservabilityCaseV1:
    """Build one retained technical-failure case without numerical results."""

    implementation = exact_revision(
        implementation_revision,
        name="implementation_revision",
    )
    reason = nonempty_string(failure_reason, name="failure_reason")
    if reason != reason.strip():
        raise ValueError("failure_reason must not contain surrounding whitespace")
    context = _load_context(
        source_protocol_path=source_protocol_path,
        stage0_protocol_path=stage0_protocol_path,
        selection_lock_path=selection_lock_path,
        visual_provider_lock_path=visual_provider_lock_path,
        calibration_source_plan_path=calibration_source_plan_path,
        calibration_source_download_path=calibration_source_download_path,
        calibration_source_run_record_path=calibration_source_run_record_path,
        calibration_source_result_path=calibration_source_result_path,
        object_id=object_id,
    )
    _, query_sha = _load_npy_matrix(
        query_jacobian_path,
        name="physical query Jacobian",
    )
    _, _, evidence_sha = _read_ordinary_bytes(
        failure_evidence_path,
        name="technical-failure evidence",
    )
    logical_root = f"sources/observability/{context.object_id}"
    sources = {
        **context.source_artifacts,
        f"{logical_root}/physical-query-jacobian.npy": query_sha,
        f"{logical_root}/technical-failure-evidence.artifact": evidence_sha,
    }
    calibration_opened = (
        context.result_row.get("status") != "unsupported_without_replacement"
    )
    return Deform360CalibrationObservabilityCaseV1(
        selection_artifact_sha256=context.selection_artifact_sha256,
        visual_provider_lock_id=context.visual_provider_lock_id,
        calibration_source_run_record_sha256=context.run_record_sha256,
        implementation_revision=implementation,
        object_id=context.object_id,
        episode_id=context.episode_id,
        stratum=context.stratum,
        physical_query_id=_role_artifact_id(
            "physical-query-jacobian",
            query_sha,
        ),
        status="technical_failure_without_replacement",
        source_artifacts=sources,
        information_boundary={
            "calibration_payloads_opened": calibration_opened,
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
            "replacement_allowed": False,
        },
        failure_reason=reason,
    )


__all__ = [
    "DEFORM360_OBSERVABILITY_CASE_INPUT_SCHEMA",
    "DEFORM360_OBSERVABILITY_CASE_INPUT_VERSION",
    "build_evaluated_case_from_paths",
    "build_technical_failure_case_from_paths",
    "physical_query_id_from_path",
]
