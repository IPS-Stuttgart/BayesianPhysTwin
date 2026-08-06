"""Validation of the Deform360 plan, download, and result chain."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ._deform360_calibration_run_common import (
    ALLOWED_RESULT_OBJECT_STATUSES,
    DEFORM360_CALIBRATION_DOWNLOAD_SCHEMA,
    DEFORM360_CALIBRATION_SOURCE_PLAN_SCHEMA,
    DEFORM360_CALIBRATION_SOURCE_PROTOCOL_ID,
    DEFORM360_CALIBRATION_SOURCE_RESULT_SCHEMA,
    DEFORM360_DATASET_REVISION,
    EXPECTED_DOWNLOAD_INFORMATION_BOUNDARY,
    EXPECTED_PLAN_INFORMATION_BOUNDARY,
    EXPECTED_RESULT_INFORMATION_BOUNDARY,
    InvalidJsonError,
    ObjectIdentitySet,
    canonical_sha256,
    load_json_object,
    object_support_counts,
    sha256,
    validated_support_gate,
)


def _invalid_plan(
    *,
    available: bool,
    error: str,
    file_sha256: str | None,
) -> dict[str, Any]:
    return {
        "plan_available": available,
        "plan_valid": False,
        "plan_error": error,
        "plan_file_sha256": file_sha256,
        "plan_sha256": None,
        "plan_support_gate": None,
    }


def _invalid_download(
    *,
    available: bool,
    error: str,
    file_sha256: str | None,
) -> dict[str, Any]:
    return {
        "download_available": available,
        "download_valid": False,
        "download_error": error,
        "download_file_sha256": file_sha256,
        "download_sha256": None,
    }


def _invalid_result(
    *,
    available: bool,
    error: str,
    file_sha256: str | None,
) -> dict[str, Any]:
    return {
        "result_available": available,
        "result_valid": False,
        "result_error": error,
        "result_file_sha256": file_sha256,
        "result_sha256": None,
        "support_gate": None,
    }


def plan_summary(
    path: Path,
    *,
    processing_revision: str,
) -> tuple[dict[str, Any], ObjectIdentitySet, frozenset[str]]:
    """Validate the names-only plan without publishing object identities."""

    if path.is_symlink() or not path.is_file():
        return (
            _invalid_plan(
                available=False,
                error="missing",
                file_sha256=None,
            ),
            frozenset(),
            frozenset(),
        )
    try:
        value, file_sha256 = load_json_object(path)
    except OSError:
        return (
            _invalid_plan(
                available=True,
                error="unreadable",
                file_sha256=None,
            ),
            frozenset(),
            frozenset(),
        )
    except InvalidJsonError as error:
        return (
            _invalid_plan(
                available=True,
                error="invalid-json",
                file_sha256=error.file_sha256,
            ),
            frozenset(),
            frozenset(),
        )
    try:
        if value.get("schema") != DEFORM360_CALIBRATION_SOURCE_PLAN_SCHEMA:
            raise ValueError("plan schema changed")
        if value.get("schema_version") != 1:
            raise ValueError("plan schema version changed")
        if value.get("protocol_id") != DEFORM360_CALIBRATION_SOURCE_PROTOCOL_ID:
            raise ValueError("plan protocol changed")
        if value.get("dataset_revision") != DEFORM360_DATASET_REVISION:
            raise ValueError("plan dataset revision changed")
        if value.get("processing_revision") != processing_revision:
            raise ValueError("plan processing revision changed")
        plan_sha256 = sha256(value.get("plan_sha256"), name="plan_sha256")
        if plan_sha256 != canonical_sha256(value, digest_key="plan_sha256"):
            raise ValueError("plan digest does not match its content")
        information_boundary = value.get("information_boundary")
        if not isinstance(information_boundary, Mapping) or dict(
            information_boundary
        ) != dict(EXPECTED_PLAN_INFORMATION_BOUNDARY):
            raise ValueError("plan information boundary changed")
        identities, planned_ids, supported, by_stratum = object_support_counts(
            value,
            artifact="plan",
            allowed_statuses=frozenset(
                {"planned", "unsupported_without_replacement"}
            ),
            supported_status="planned",
        )
        support_gate = validated_support_gate(
            value,
            artifact="plan",
            object_supported=supported,
            object_supported_by_stratum=by_stratum,
        )
    except ValueError:
        return (
            _invalid_plan(
                available=True,
                error="invalid-contract",
                file_sha256=file_sha256,
            ),
            frozenset(),
            frozenset(),
        )
    return (
        {
            "plan_available": True,
            "plan_valid": True,
            "plan_error": None,
            "plan_file_sha256": file_sha256,
            "plan_sha256": plan_sha256,
            "plan_support_gate": support_gate,
        },
        identities,
        planned_ids,
    )


def download_summary(
    path: Path,
    *,
    plan_sha256: str | None,
    planned_ids: frozenset[str],
) -> dict[str, Any]:
    """Validate that the download manifest binds to the actual plan."""

    if path.is_symlink() or not path.is_file():
        return _invalid_download(
            available=False,
            error="missing",
            file_sha256=None,
        )
    try:
        value, file_sha256 = load_json_object(path)
    except OSError:
        return _invalid_download(
            available=True,
            error="unreadable",
            file_sha256=None,
        )
    except InvalidJsonError as error:
        return _invalid_download(
            available=True,
            error="invalid-json",
            file_sha256=error.file_sha256,
        )
    try:
        if value.get("schema") != DEFORM360_CALIBRATION_DOWNLOAD_SCHEMA:
            raise ValueError("download schema changed")
        if value.get("schema_version") != 1:
            raise ValueError("download schema version changed")
        if value.get("protocol_id") != DEFORM360_CALIBRATION_SOURCE_PROTOCOL_ID:
            raise ValueError("download protocol changed")
        if value.get("dataset_revision") != DEFORM360_DATASET_REVISION:
            raise ValueError("download dataset revision changed")
        if plan_sha256 is None or value.get("plan_sha256") != plan_sha256:
            raise ValueError("download plan binding changed")
        download_sha256 = sha256(
            value.get("download_sha256"),
            name="download_sha256",
        )
        if download_sha256 != canonical_sha256(
            value,
            digest_key="download_sha256",
        ):
            raise ValueError("download digest does not match its content")
        information_boundary = value.get("information_boundary")
        if not isinstance(information_boundary, Mapping) or dict(
            information_boundary
        ) != dict(EXPECTED_DOWNLOAD_INFORMATION_BOUNDARY):
            raise ValueError("download information boundary changed")
        object_ids = value.get("object_ids")
        if (
            not isinstance(object_ids, list)
            or any(type(item) is not str for item in object_ids)
            or object_ids != sorted(planned_ids)
        ):
            raise ValueError("download object identities changed")
    except ValueError:
        return _invalid_download(
            available=True,
            error="invalid-contract",
            file_sha256=file_sha256,
        )
    return {
        "download_available": True,
        "download_valid": True,
        "download_error": None,
        "download_file_sha256": file_sha256,
        "download_sha256": download_sha256,
    }


def result_summary(
    path: Path,
    *,
    processing_revision: str,
    plan_sha256: str | None,
    download_sha256: str | None,
    expected_identities: ObjectIdentitySet,
) -> dict[str, Any]:
    """Validate that the prepared-source result closes the artifact chain."""

    if path.is_symlink() or not path.is_file():
        return _invalid_result(
            available=False,
            error="missing",
            file_sha256=None,
        )
    try:
        value, result_file_sha256 = load_json_object(path)
    except OSError:
        return _invalid_result(
            available=True,
            error="unreadable",
            file_sha256=None,
        )
    except InvalidJsonError as error:
        return _invalid_result(
            available=True,
            error="invalid-json",
            file_sha256=error.file_sha256,
        )
    try:
        if value.get("schema") != DEFORM360_CALIBRATION_SOURCE_RESULT_SCHEMA:
            raise ValueError("result schema changed")
        if value.get("schema_version") != 1:
            raise ValueError("result schema version changed")
        if value.get("protocol_id") != DEFORM360_CALIBRATION_SOURCE_PROTOCOL_ID:
            raise ValueError("result protocol changed")
        if value.get("dataset_revision") != DEFORM360_DATASET_REVISION:
            raise ValueError("result dataset revision changed")
        if value.get("processing_revision") != processing_revision:
            raise ValueError("result processing revision changed")
        if plan_sha256 is None or value.get("plan_sha256") != plan_sha256:
            raise ValueError("result plan binding changed")
        if (
            download_sha256 is None
            or value.get("download_sha256") != download_sha256
        ):
            raise ValueError("result download binding changed")
        result_sha256 = sha256(
            value.get("result_sha256"),
            name="result_sha256",
        )
        if result_sha256 != canonical_sha256(
            value,
            digest_key="result_sha256",
        ):
            raise ValueError("result digest does not match its content")
        information_boundary = value.get("information_boundary")
        if not isinstance(information_boundary, Mapping) or dict(
            information_boundary
        ) != dict(EXPECTED_RESULT_INFORMATION_BOUNDARY):
            raise ValueError("result information boundary changed")
        identities, _, supported, by_stratum = object_support_counts(
            value,
            artifact="result",
            allowed_statuses=ALLOWED_RESULT_OBJECT_STATUSES,
            supported_status="source_prepared",
        )
        if identities != expected_identities:
            raise ValueError("result cohort identity changed")
        support_gate = validated_support_gate(
            value,
            artifact="result",
            object_supported=supported,
            object_supported_by_stratum=by_stratum,
        )
    except ValueError:
        return _invalid_result(
            available=True,
            error="invalid-contract",
            file_sha256=result_file_sha256,
        )
    return {
        "result_available": True,
        "result_valid": True,
        "result_error": None,
        "result_file_sha256": result_file_sha256,
        "result_sha256": result_sha256,
        "support_gate": support_gate,
    }
