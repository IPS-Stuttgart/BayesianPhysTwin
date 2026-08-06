"""Validation of the Deform360 source locks and execution artifact chain."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ._deform360_calibration_run_common import (
    ALLOWED_RESULT_OBJECT_STATUSES,
    DEFORM360_CALIBRATION_DOWNLOAD_SCHEMA,
    DEFORM360_CALIBRATION_SOURCE_PLAN_SCHEMA,
    DEFORM360_CALIBRATION_SOURCE_PROTOCOL_ID,
    DEFORM360_CALIBRATION_SOURCE_PROTOCOL_SCHEMA,
    DEFORM360_CALIBRATION_SOURCE_RESULT_SCHEMA,
    DEFORM360_DATASET_REPOSITORY,
    DEFORM360_DATASET_REVISION,
    DEFORM360_EXPECTED_TACTILE_BASELINE_POLICY,
    DEFORM360_PARENT_PROTOCOL_ID,
    DEFORM360_PROCESSING_REPOSITORY,
    DEFORM360_STAGE0_SELECTION_SCHEMA,
    DEFORM360_VISUAL_PROVIDER_LOCK_SCHEMA,
    DEFORM360_VISUAL_PROVIDER_LOCK_SEMANTICS,
    EXPECTED_CONFIRMATION_OBJECT_COUNT,
    EXPECTED_CONFIRMATION_OBJECTS_PER_STRATUM,
    EXPECTED_DOWNLOAD_INFORMATION_BOUNDARY,
    EXPECTED_OBJECT_COUNT,
    EXPECTED_OBJECTS_PER_STRATUM,
    EXPECTED_PLAN_INFORMATION_BOUNDARY,
    EXPECTED_RESULT_INFORMATION_BOUNDARY,
    EXPECTED_STRATA,
    MINIMUM_ALIGNED_FRAMES,
    MINIMUM_CAMERA_STREAMS,
    ExpectedUnitMap,
    InvalidJsonError,
    ObjectIdentitySet,
    canonical_sha256,
    content_sha256,
    integer_field,
    load_json_object,
    object_support_counts,
    raw_object_path,
    sha256,
    string_sequence,
    validated_support_gate,
)


def _invalid_source_locks(
    *,
    available: bool,
    error: str,
) -> dict[str, Any]:
    return {
        "source_locks_available": available,
        "source_locks_valid": False,
        "source_locks_error": error,
        "source_protocol_file_sha256": None,
        "source_protocol_sha256": None,
        "stage0_protocol_file_sha256": None,
        "stage0_protocol_sha256": None,
        "selection_lock_file_sha256": None,
        "selection_artifact_sha256": None,
        "content_selection_sha256": None,
        "visual_provider_lock_file_sha256": None,
        "visual_provider_lock_id": None,
    }


def _load_required_json(path: Path) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    return load_json_object(path)


def _selection_units(
    rows: object,
    *,
    expected_count: int,
    expected_per_stratum: int,
    name: str,
) -> ExpectedUnitMap:
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError(f"{name} has the wrong object count")
    result: ExpectedUnitMap = {}
    by_stratum = dict.fromkeys(EXPECTED_STRATA, 0)
    expected_fields = {
        "object_id",
        "episode_id",
        "stratum",
        "metadata_path",
        "metadata_sha256",
    }
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != expected_fields:
            raise ValueError(f"{name} unit fields changed")
        object_id = row.get("object_id")
        if type(object_id) is not str or not object_id or object_id in result:
            raise ValueError(f"{name} object identity changed")
        episode_id = row.get("episode_id")
        if type(episode_id) is not int or episode_id < 0:
            raise ValueError(f"{name} episode identity changed")
        stratum = row.get("stratum")
        if stratum not in EXPECTED_STRATA:
            raise ValueError(f"{name} stratum changed")
        metadata_path, metadata_object = raw_object_path(
            row.get("metadata_path"),
            name=f"{name} metadata_path",
        )
        if metadata_object != object_id or not metadata_path.endswith(
            "/metadata.json"
        ):
            raise ValueError(f"{name} metadata path changed")
        metadata_sha256 = sha256(
            row.get("metadata_sha256"),
            name=f"{name} metadata_sha256",
        )
        result[object_id] = (
            episode_id,
            stratum,
            metadata_path,
            metadata_sha256,
        )
        by_stratum[stratum] += 1
    if any(count != expected_per_stratum for count in by_stratum.values()):
        raise ValueError(f"{name} stratum counts changed")
    return result


def source_lock_summary(
    *,
    source_protocol_json: Path,
    stage0_protocol_json: Path,
    selection_lock: Path,
    visual_provider_lock: Path,
    processing_revision: str,
) -> tuple[dict[str, Any], ExpectedUnitMap, frozenset[str]]:
    """Validate all data-free source locks and recover the exact cohort."""

    paths = (
        source_protocol_json,
        stage0_protocol_json,
        selection_lock,
        visual_provider_lock,
    )
    if any(path.is_symlink() or not path.is_file() for path in paths):
        return _invalid_source_locks(available=False, error="missing"), {}, frozenset()
    try:
        source_protocol, source_protocol_file = _load_required_json(
            source_protocol_json
        )
        stage0_protocol, stage0_protocol_file = _load_required_json(
            stage0_protocol_json
        )
        selection, selection_file = _load_required_json(selection_lock)
        provider, provider_file = _load_required_json(visual_provider_lock)
    except InvalidJsonError:
        return (
            _invalid_source_locks(available=True, error="invalid-json"),
            {},
            frozenset(),
        )
    except OSError:
        return (
            _invalid_source_locks(available=True, error="unreadable"),
            {},
            frozenset(),
        )

    try:
        if source_protocol.get("schema") != (
            DEFORM360_CALIBRATION_SOURCE_PROTOCOL_SCHEMA
        ):
            raise ValueError("source protocol schema changed")
        if source_protocol.get("schema_version") != 1:
            raise ValueError("source protocol version changed")
        if source_protocol.get("protocol_id") != (
            DEFORM360_CALIBRATION_SOURCE_PROTOCOL_ID
        ):
            raise ValueError("source protocol identity changed")
        if source_protocol.get("parent_protocol_id") != DEFORM360_PARENT_PROTOCOL_ID:
            raise ValueError("source parent protocol changed")
        source_protocol_sha = sha256(
            source_protocol.get("protocol_sha256"),
            name="source protocol SHA-256",
        )
        if source_protocol_sha != canonical_sha256(
            source_protocol,
            digest_key="protocol_sha256",
        ):
            raise ValueError("source protocol digest changed")
        dataset = source_protocol.get("dataset")
        processing = source_protocol.get("processing")
        if not isinstance(dataset, Mapping) or not isinstance(processing, Mapping):
            raise ValueError("source protocol repositories are missing")
        if (
            dataset.get("repository") != DEFORM360_DATASET_REPOSITORY
            or dataset.get("revision") != DEFORM360_DATASET_REVISION
            or processing.get("repository") != DEFORM360_PROCESSING_REPOSITORY
            or processing.get("revision") != processing_revision
        ):
            raise ValueError("source protocol revisions changed")

        stage0_protocol_sha = content_sha256(stage0_protocol)
        if stage0_protocol.get("protocol_id") != DEFORM360_PARENT_PROTOCOL_ID:
            raise ValueError("Stage-0 protocol identity changed")

        if selection.get("schema") != DEFORM360_STAGE0_SELECTION_SCHEMA:
            raise ValueError("selection schema changed")
        if selection.get("schema_version") != 1:
            raise ValueError("selection version changed")
        if selection.get("protocol_id") != DEFORM360_PARENT_PROTOCOL_ID:
            raise ValueError("selection protocol changed")
        if selection.get("protocol_sha256") != stage0_protocol_sha:
            raise ValueError("selection no longer binds the Stage-0 protocol")
        selection_payload = selection.get("selection")
        if not isinstance(selection_payload, Mapping) or set(selection_payload) != {
            "calibration",
            "confirmation",
        }:
            raise ValueError("selection roles changed")
        declared_selection = sha256(
            selection.get("selection_sha256"),
            name="selection_sha256",
        )
        if declared_selection != content_sha256(selection_payload):
            raise ValueError("selection digest changed")
        content_payload = dict(selection)
        content_payload.pop("content_selection_sha256", None)
        content_payload.pop("implementation_revision", None)
        content_payload.pop("selection_artifact_sha256", None)
        content_selection = sha256(
            selection.get("content_selection_sha256"),
            name="content_selection_sha256",
        )
        if content_selection != content_sha256(content_payload):
            raise ValueError("selection content digest changed")
        artifact_payload = dict(selection)
        artifact_payload.pop("selection_artifact_sha256", None)
        selection_artifact = sha256(
            selection.get("selection_artifact_sha256"),
            name="selection_artifact_sha256",
        )
        if selection_artifact != content_sha256(artifact_payload):
            raise ValueError("selection artifact digest changed")
        selection_dataset = selection.get("dataset")
        selection_processing = selection.get("official_processing")
        if not isinstance(selection_dataset, Mapping) or not isinstance(
            selection_processing,
            Mapping,
        ):
            raise ValueError("selection revisions are missing")
        if (
            selection_dataset.get("repo_id") != DEFORM360_DATASET_REPOSITORY
            or selection_dataset.get("resolved_revision")
            != DEFORM360_DATASET_REVISION
            or selection_processing.get("repository")
            != DEFORM360_PROCESSING_REPOSITORY
            or selection_processing.get("revision") != processing_revision
            or selection.get("replacement_allowed_after_payload_access") is not False
        ):
            raise ValueError("selection boundary changed")
        calibration = _selection_units(
            selection_payload.get("calibration"),
            expected_count=EXPECTED_OBJECT_COUNT,
            expected_per_stratum=EXPECTED_OBJECTS_PER_STRATUM,
            name="calibration selection",
        )
        confirmation = _selection_units(
            selection_payload.get("confirmation"),
            expected_count=EXPECTED_CONFIRMATION_OBJECT_COUNT,
            expected_per_stratum=EXPECTED_CONFIRMATION_OBJECTS_PER_STRATUM,
            name="confirmation selection",
        )
        if set(calibration) & set(confirmation):
            raise ValueError("calibration and confirmation objects overlap")

        if provider.get("schema") != DEFORM360_VISUAL_PROVIDER_LOCK_SCHEMA:
            raise ValueError("provider lock schema changed")
        if provider.get("schema_version") != 1:
            raise ValueError("provider lock version changed")
        if provider.get("semantics") != DEFORM360_VISUAL_PROVIDER_LOCK_SEMANTICS:
            raise ValueError("provider lock semantics changed")
        if provider.get("protocol_id") != DEFORM360_PARENT_PROTOCOL_ID:
            raise ValueError("provider lock protocol changed")
        provider_id = sha256(provider.get("artifact_id"), name="provider lock ID")
        provider_payload = dict(provider)
        provider_payload.pop("artifact_id", None)
        if provider_id != content_sha256(provider_payload):
            raise ValueError("provider lock artifact digest changed")
        if (
            provider.get("selected_raw_payloads_opened") is not False
            or provider.get("target_outcomes_used") is not False
        ):
            raise ValueError("provider lock information boundary changed")
        locks = source_protocol.get("locks")
        if not isinstance(locks, Mapping) or locks.get(
            "visual_provider_lock_id"
        ) != provider_id:
            raise ValueError("source protocol provider binding changed")
    except ValueError:
        return (
            _invalid_source_locks(available=True, error="invalid-contract"),
            {},
            frozenset(),
        )

    return (
        {
            "source_locks_available": True,
            "source_locks_valid": True,
            "source_locks_error": None,
            "source_protocol_file_sha256": source_protocol_file,
            "source_protocol_sha256": source_protocol_sha,
            "stage0_protocol_file_sha256": stage0_protocol_file,
            "stage0_protocol_sha256": stage0_protocol_sha,
            "selection_lock_file_sha256": selection_file,
            "selection_artifact_sha256": selection_artifact,
            "content_selection_sha256": content_selection,
            "visual_provider_lock_file_sha256": provider_file,
            "visual_provider_lock_id": provider_id,
        },
        calibration,
        frozenset(confirmation),
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


def _validate_plan_rows(
    value: Mapping[str, Any],
    *,
    expected_units: ExpectedUnitMap,
    confirmation_ids: frozenset[str],
) -> None:
    rows = value.get("objects")
    if not isinstance(rows, list):
        raise ValueError("plan object rows are missing")
    selected_paths: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("plan row is malformed")
        object_id = row.get("object_id")
        if type(object_id) is not str or object_id not in expected_units:
            raise ValueError("plan substituted the frozen cohort")
        episode, stratum, metadata_path, metadata_digest = expected_units[object_id]
        if (
            row.get("episode_id") != episode
            or row.get("stratum") != stratum
            or row.get("metadata_path") != metadata_path
            or row.get("metadata_sha256") != metadata_digest
        ):
            raise ValueError("plan unit differs from the frozen selection")
        status = row.get("status")
        errors = row.get("errors")
        cameras = string_sequence(
            row.get("camera_streams"),
            name="plan camera_streams",
        )
        tactile = string_sequence(
            row.get("tactile_streams"),
            name="plan tactile_streams",
        )
        if status == "planned":
            if errors != [] or len(cameras) < MINIMUM_CAMERA_STREAMS or not tactile:
                raise ValueError("planned row lacks admitted source support")
        elif status == "unsupported_without_replacement":
            if not isinstance(errors, list) or not errors or any(
                type(item) is not str or not item for item in errors
            ):
                raise ValueError("unsupported row lacks retained failure evidence")
        files = row.get("selected_files")
        if not isinstance(files, list) or not files:
            raise ValueError("plan selected_files are missing")
        row_paths: set[str] = set()
        for record in files:
            if not isinstance(record, Mapping):
                raise ValueError("plan selected file record is malformed")
            path, path_object = raw_object_path(
                record.get("path"),
                name="plan selected path",
            )
            if (
                path_object != object_id
                or path_object in confirmation_ids
                or path in selected_paths
            ):
                raise ValueError("plan selected path escaped its frozen object")
            selected_paths.add(path)
            row_paths.add(path)
            size = record.get("size")
            if size is not None and (type(size) is not int or size < 0):
                raise ValueError("plan selected file size is invalid")
            blob_id = record.get("blob_id")
            if blob_id is not None and type(blob_id) is not str:
                raise ValueError("plan selected blob identity is invalid")
            lfs_sha = record.get("lfs_sha256")
            if lfs_sha is not None:
                sha256(lfs_sha, name="plan selected LFS SHA-256")
        if metadata_path not in row_paths:
            raise ValueError("plan omitted the frozen metadata file")


def plan_summary(
    path: Path,
    *,
    processing_revision: str,
    source_locks: Mapping[str, Any],
    expected_units: ExpectedUnitMap,
    confirmation_ids: frozenset[str],
) -> tuple[dict[str, Any], ObjectIdentitySet, frozenset[str]]:
    """Validate the names-only plan against the exact data-free locks."""

    if path.is_symlink() or not path.is_file():
        return (
            _invalid_plan(available=False, error="missing", file_sha256=None),
            frozenset(),
            frozenset(),
        )
    try:
        value, file_sha256 = load_json_object(path)
    except OSError:
        return (
            _invalid_plan(available=True, error="unreadable", file_sha256=None),
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
        if value.get("parent_protocol_id") != DEFORM360_PARENT_PROTOCOL_ID:
            raise ValueError("plan parent protocol changed")
        if value.get("protocol_sha256") != source_locks.get(
            "source_protocol_sha256"
        ):
            raise ValueError("plan source protocol binding changed")
        if value.get("selection_source_sha256") != source_locks.get(
            "selection_lock_file_sha256"
        ):
            raise ValueError("plan selection binding changed")
        if value.get("visual_provider_lock_id") != source_locks.get(
            "visual_provider_lock_id"
        ):
            raise ValueError("plan provider identity changed")
        if value.get("visual_provider_source_sha256") != source_locks.get(
            "visual_provider_lock_file_sha256"
        ):
            raise ValueError("plan provider source changed")
        if (
            value.get("dataset_repository") != DEFORM360_DATASET_REPOSITORY
            or value.get("dataset_revision") != DEFORM360_DATASET_REVISION
            or value.get("processing_repository")
            != DEFORM360_PROCESSING_REPOSITORY
            or value.get("processing_revision") != processing_revision
        ):
            raise ValueError("plan revisions changed")
        if value.get("tactile_baseline_policy") != dict(
            DEFORM360_EXPECTED_TACTILE_BASELINE_POLICY
        ):
            raise ValueError("plan tactile baseline policy changed")
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
        expected_identities = frozenset(
            (object_id, unit[0], unit[1])
            for object_id, unit in expected_units.items()
        )
        if identities != expected_identities:
            raise ValueError("plan cohort differs from the frozen selection")
        _validate_plan_rows(
            value,
            expected_units=expected_units,
            confirmation_ids=confirmation_ids,
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
    confirmation_ids: frozenset[str],
) -> dict[str, Any]:
    """Validate that the download manifest contains only planned object files."""

    if path.is_symlink() or not path.is_file():
        return _invalid_download(available=False, error="missing", file_sha256=None)
    try:
        value, file_sha256 = load_json_object(path)
    except OSError:
        return _invalid_download(available=True, error="unreadable", file_sha256=None)
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
        if value.get("dataset_repository") != DEFORM360_DATASET_REPOSITORY:
            raise ValueError("download repository changed")
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
        files = value.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("download file records are missing")
        seen_paths: set[str] = set()
        seen_objects: set[str] = set()
        expected_fields = {
            "path",
            "size",
            "blob_id",
            "lfs_sha256",
            "downloaded_size",
            "downloaded_sha256",
        }
        for record in files:
            if not isinstance(record, Mapping) or set(record) != expected_fields:
                raise ValueError("download file fields changed")
            relative, object_id = raw_object_path(
                record.get("path"),
                name="download file path",
            )
            if (
                relative in seen_paths
                or object_id not in planned_ids
                or object_id in confirmation_ids
            ):
                raise ValueError("download admitted an unplanned object path")
            seen_paths.add(relative)
            seen_objects.add(object_id)
            downloaded_size = integer_field(
                record.get("downloaded_size"),
                name="downloaded_size",
            )
            declared_size = record.get("size")
            if declared_size is not None and declared_size != downloaded_size:
                raise ValueError("downloaded size differs from the plan")
            if declared_size is not None and type(declared_size) is not int:
                raise ValueError("declared download size is invalid")
            blob_id = record.get("blob_id")
            if blob_id is not None and type(blob_id) is not str:
                raise ValueError("download blob identity is invalid")
            downloaded_sha = sha256(
                record.get("downloaded_sha256"),
                name="downloaded_sha256",
            )
            lfs_sha = record.get("lfs_sha256")
            if lfs_sha is not None and sha256(
                lfs_sha,
                name="download LFS SHA-256",
            ) != downloaded_sha:
                raise ValueError("downloaded bytes differ from LFS identity")
        if seen_objects != set(planned_ids):
            raise ValueError("download omitted a planned object")
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


def _validate_prepared_row(row: Mapping[str, Any]) -> None:
    if row.get("completed_stage") != "action-window-selection":
        raise ValueError("prepared row did not complete action-window selection")
    if row.get("synthetic_episode_index") != 0:
        raise ValueError("prepared row synthetic episode changed")
    if type(row.get("bimanual")) is not bool:
        raise ValueError("prepared row bimanual identity changed")
    camera_count = integer_field(row.get("camera_count"), name="camera_count")
    cameras = string_sequence(
        row.get("cameras"),
        name="prepared cameras",
        minimum_length=MINIMUM_CAMERA_STREAMS,
    )
    if camera_count != len(cameras):
        raise ValueError("prepared camera count disagrees")
    frame_count = integer_field(
        row.get("aligned_frame_count"),
        name="aligned_frame_count",
    )
    if frame_count < MINIMUM_ALIGNED_FRAMES:
        raise ValueError("prepared row is shorter than the frozen window")
    tactile_count = integer_field(
        row.get("tactile_sensor_count"),
        name="tactile_sensor_count",
    )
    tactile = string_sequence(
        row.get("tactile_sensors"),
        name="prepared tactile sensors",
        minimum_length=1,
    )
    if tactile_count != len(tactile):
        raise ValueError("prepared tactile count disagrees")
    if not isinstance(row.get("action_window"), Mapping):
        raise ValueError("prepared action window is missing")
    outputs = row.get("outputs_sha256")
    if not isinstance(outputs, Mapping):
        raise ValueError("prepared output identities are missing")
    for key in ("alignment", "undistorted_intrinsics", "extrinsics", "robot"):
        sha256(outputs.get(key), name=f"prepared {key} SHA-256")
    tactile_outputs = outputs.get("tactile")
    if not isinstance(tactile_outputs, Mapping) or set(tactile_outputs) != set(tactile):
        raise ValueError("prepared tactile output identities changed")
    for sensor, digest in tactile_outputs.items():
        sha256(digest, name=f"prepared tactile {sensor} SHA-256")
    if "error" in row:
        raise ValueError("prepared row retains a failure error")


def result_summary(
    path: Path,
    *,
    processing_revision: str,
    plan_sha256: str | None,
    download_sha256: str | None,
    expected_identities: ObjectIdentitySet,
    planned_ids: frozenset[str],
) -> dict[str, Any]:
    """Validate that the prepared-source result closes the exact chain."""

    if path.is_symlink() or not path.is_file():
        return _invalid_result(available=False, error="missing", file_sha256=None)
    try:
        value, result_file_sha256 = load_json_object(path)
    except OSError:
        return _invalid_result(available=True, error="unreadable", file_sha256=None)
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
        if download_sha256 is None or value.get("download_sha256") != download_sha256:
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
        rows = value.get("objects")
        if not isinstance(rows, list):
            raise ValueError("result object rows are missing")
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("result row is malformed")
            object_id = row.get("object_id")
            status = row.get("status")
            if status == "source_prepared":
                if object_id not in planned_ids:
                    raise ValueError("unplanned object was reported as prepared")
                _validate_prepared_row(row)
            elif status == "technical_failure_without_replacement":
                if object_id not in planned_ids:
                    raise ValueError("unplanned object became a technical failure")
                if type(row.get("error")) is not str or not row.get("error"):
                    raise ValueError("technical failure lacks retained evidence")
            elif status == "unsupported_without_replacement":
                if object_id in planned_ids:
                    raise ValueError("planned object was reclassified as unsupported")
                errors = row.get("errors")
                if not isinstance(errors, list) or not errors or any(
                    type(item) is not str or not item for item in errors
                ):
                    raise ValueError("unsupported result lacks plan evidence")
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
