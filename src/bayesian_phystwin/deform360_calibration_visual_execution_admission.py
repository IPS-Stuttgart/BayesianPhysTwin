"""Bind target-blind visual jobs to exact retained calibration source bytes.

The prepared-source inventory proves byte custody for the frozen ten-object
calibration cohort. The visual-production plan fixes cameras, frame windows,
seeds, dependence groups, and output locations. This module joins those two
metadata artifacts before any MotionCrafter or Prob4D call is made.

Only the content-addressed JSON metadata is opened here. Retained media,
confirmation payloads, and target outcomes remain outside this boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

from ._canonical_contracts import plain_json
from ._portable_contracts import (
    content_id,
    exact_revision,
    require_exact_fields,
    sha256_digest,
    write_atomic_json,
)
from .deform360_calibration_visual_production_plan import (
    DEFORM360_CALIBRATION_OBJECT_COUNT,
    DEFORM360_CALIBRATION_PREDICTION_FRAME_COUNT,
    DEFORM360_CALIBRATION_PREFIX_FRAME_COUNT,
    DEFORM360_CALIBRATION_SELECTED_FRAME_COUNT,
    validate_deform360_calibration_visual_production_plan,
)

DEFORM360_CALIBRATION_VISUAL_EXECUTION_ADMISSION_SCHEMA: Final = (
    "bayesian-phystwin.deform360-calibration-visual-execution-admission"
)
DEFORM360_CALIBRATION_VISUAL_EXECUTION_ADMISSION_VERSION: Final = 1
DEFORM360_CALIBRATION_VISUAL_EXECUTION_ADMISSION_SEMANTICS: Final = (
    "inventory-bound-target-blind-calibration-visual-execution-v1"
)
DEFORM360_PREPARED_SOURCE_INVENTORY_SCHEMA: Final = (
    "bayesian-phystwin.deform360-calibration-prepared-source-inventory"
)
DEFORM360_PREPARED_SOURCE_INVENTORY_VERSION: Final = 1
DEFORM360_PREPARED_SOURCE_INVENTORY_SEMANTICS: Final = (
    "exact-retained-calibration-rgb-tactile-robot-inventory-v1"
)
DEFORM360_PREPARED_SOURCE_INVENTORY_STATUS: Final = (
    "complete-calibration-only-prepared-source"
)
DEFORM360_PREPARED_SOURCE_INVENTORY_CLAIM_BOUNDARY: Final = (
    "Calibration-only retained-source custody and portable array/media contracts. "
    "A valid inventory does not establish visual-provider competence, contact "
    "calibration, physical-query observability, uncertainty calibration, "
    "confirmation accuracy, Causal4D benefit, deployment safety, or state of the art."
)
DEFORM360_CALIBRATION_VISUAL_EXECUTION_ADMISSION_CLAIM_BOUNDARY: Final = (
    "Calibration-only execution admission over exact plan and retained-byte "
    "inventory metadata. A valid admission does not execute MotionCrafter or "
    "Prob4D, establish provider competence or physical-query benefit, authorize "
    "confirmation access, establish uncertainty calibration, or support a "
    "deployment-safety or state-of-the-art claim."
)

_MAX_METADATA_BYTES = 64 * 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024

_INVENTORY_INFORMATION_BOUNDARY = {
    "calibration_camera_payloads_opened": True,
    "calibration_tactile_payloads_opened": True,
    "calibration_robot_state_opened": True,
    "calibration_target_metrics_computed": False,
    "confirmation_payloads_opened": False,
    "target_outcomes_used": False,
    "replacement_allowed": False,
}
_ADMISSION_INFORMATION_BOUNDARY = {
    "plan_metadata_opened": True,
    "inventory_metadata_opened": True,
    "retained_calibration_payloads_opened": False,
    "confirmation_payloads_opened": False,
    "target_outcomes_used": False,
    "replacement_allowed": False,
}

_INVENTORY_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "status",
        "implementation_revision",
        "calibration_source_revision",
        "processing_revision",
        "selection_artifact_sha256",
        "visual_provider_lock_id",
        "calibration_source_run_record_sha256",
        "object_count",
        "objects",
        "source_artifacts",
        "information_boundary",
        "claim_boundary",
        "inventory_id",
    }
)
_INVENTORY_OBJECT_FIELDS = frozenset(
    {
        "object_id",
        "episode_id",
        "stratum",
        "synthetic_episode_index",
        "aligned_frame_count",
        "action_window",
        "episode_files",
        "cameras",
        "tactile",
    }
)
_INVENTORY_CAMERA_FIELDS = frozenset(
    {
        "camera",
        "video",
        "preview",
        "timestamps",
        "alignment",
        "metadata",
        "frame_count",
        "width",
        "height",
        "fps",
        "timeline_sha256",
    }
)
_FILE_RECORD_FIELDS = frozenset({"path", "sha256", "byte_count"})
_ADMISSION_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "admission_id",
        "implementation_revision",
        "plan_id",
        "plan_file_sha256",
        "plan_file_byte_count",
        "inventory_id",
        "inventory_file_sha256",
        "inventory_file_byte_count",
        "protocol_id",
        "selection_artifact_sha256",
        "visual_provider_lock_id",
        "calibration_source_run_record_sha256",
        "calibration_source_result_sha256",
        "plan_implementation_revision",
        "inventory_implementation_revision",
        "calibration_source_revision",
        "processing_revision",
        "provider_revision",
        "motioncrafter_revision",
        "model_set_id",
        "object_count",
        "camera_view_count",
        "jobs",
        "information_boundary",
        "claim_boundary",
    }
)
_JOB_FIELDS = frozenset(
    {
        "job_id",
        "object_id",
        "episode_id",
        "stratum",
        "object_root_seed",
        "camera_id",
        "view_root_seed",
        "call_namespace",
        "selected_source_frame_range_half_open",
        "prediction_source_frame_range_half_open",
        "prefix_source_frame_range_half_open",
        "aligned_frame_count",
        "source_video",
        "source_timestamps",
        "camera_width",
        "camera_height",
        "camera_fps",
        "camera_timeline_sha256",
        "output_relative_directory",
        "dependence_group_ids",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return cast(Sequence[Any], value)


def _literal_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(
            f"{name} must be a nonempty literal string without surrounding whitespace"
        )
    return value


def _literal_integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _positive_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    result: float = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _safe_relative_path(value: object, *, name: str) -> str:
    text = _literal_string(value, name=name)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or "\\" in text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{name} must be a safe POSIX relative path")
    return path.as_posix()


def _frame_range(value: object, *, name: str, expected_count: int) -> list[int]:
    items = _sequence(value, name=name)
    if len(items) != 2:
        raise ValueError(f"{name} must contain two bounds")
    start = _literal_integer(items[0], name=f"{name}[0]")
    stop = _literal_integer(items[1], name=f"{name}[1]")
    if stop - start != expected_count:
        raise ValueError(f"{name} must contain exactly {expected_count} frames")
    return [start, stop]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not permitted: {value}")


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    for candidate in (absolute, *absolute.parents):
        if candidate.is_symlink():
            raise ValueError(f"metadata path contains a symbolic link: {candidate}")


def _load_stable_json_object(
    path: str | Path,
    *,
    label: str,
) -> tuple[Mapping[str, Any], str, int]:
    source = Path(path)
    _reject_symlink_components(source)
    flags = (
        os.O_RDONLY
        | int(getattr(os, "O_BINARY", 0))
        | int(getattr(os, "O_CLOEXEC", 0))
        | int(getattr(os, "O_NOFOLLOW", 0))
    )
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise ValueError(f"cannot open {label}: {source}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        if before.st_size > _MAX_METADATA_BYTES:
            raise ValueError(f"{label} exceeds {_MAX_METADATA_BYTES} bytes")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        byte_count = 0
        while True:
            block = os.read(descriptor, _READ_CHUNK_BYTES)
            if not block:
                break
            chunks.append(block)
            digest.update(block)
            byte_count += len(block)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ValueError(f"cannot read {label}: {source}") from error
    finally:
        os.close(descriptor)
    if _file_identity(before) != _file_identity(after) or byte_count != after.st_size:
        raise ValueError(f"{label} changed while being read")
    payload = b"".join(chunks)
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} root must be a JSON object")
    return cast(Mapping[str, Any], value), digest.hexdigest(), byte_count


def _json_copy(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    try:
        copied = json.loads(
            json.dumps(plain_json(value), sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite JSON") from error
    if not isinstance(copied, dict):
        raise ValueError(f"{name} must be a JSON object")
    return cast(dict[str, Any], copied)


def _validate_file_record(value: object, *, name: str) -> dict[str, object]:
    record = _mapping(value, name=name)
    require_exact_fields(record, expected=_FILE_RECORD_FIELDS, name=name)
    return {
        "path": _safe_relative_path(record["path"], name=f"{name}.path"),
        "sha256": sha256_digest(record["sha256"], name=f"{name}.sha256"),
        "byte_count": _literal_integer(
            record["byte_count"],
            name=f"{name}.byte_count",
            minimum=1,
        ),
    }


def _validate_inventory_camera(
    value: object,
    *,
    object_id: str,
    aligned_frame_count: int,
) -> dict[str, Any]:
    camera = _mapping(value, name=f"{object_id} inventory camera")
    require_exact_fields(
        camera,
        expected=_INVENTORY_CAMERA_FIELDS,
        name=f"{object_id} inventory camera",
    )
    camera_id = _literal_string(camera["camera"], name="inventory camera")
    frame_count = _literal_integer(
        camera["frame_count"],
        name=f"{object_id}/{camera_id} frame_count",
        minimum=1,
    )
    if frame_count != aligned_frame_count:
        raise ValueError(
            f"inventory camera frame count changed: {object_id}/{camera_id}"
        )
    normalized = {
        "camera": camera_id,
        "video": _validate_file_record(
            camera["video"],
            name=f"{object_id}/{camera_id} video",
        ),
        "preview": _validate_file_record(
            camera["preview"],
            name=f"{object_id}/{camera_id} preview",
        ),
        "timestamps": _validate_file_record(
            camera["timestamps"],
            name=f"{object_id}/{camera_id} timestamps",
        ),
        "alignment": _validate_file_record(
            camera["alignment"],
            name=f"{object_id}/{camera_id} alignment",
        ),
        "metadata": _validate_file_record(
            camera["metadata"],
            name=f"{object_id}/{camera_id} metadata",
        ),
        "frame_count": frame_count,
        "width": _literal_integer(
            camera["width"],
            name=f"{object_id}/{camera_id} width",
            minimum=1,
        ),
        "height": _literal_integer(
            camera["height"],
            name=f"{object_id}/{camera_id} height",
            minimum=1,
        ),
        "fps": _positive_number(
            camera["fps"],
            name=f"{object_id}/{camera_id} fps",
        ),
        "timeline_sha256": sha256_digest(
            camera["timeline_sha256"],
            name=f"{object_id}/{camera_id} timeline_sha256",
        ),
    }
    return normalized


def validate_deform360_prepared_source_inventory(
    value: object,
) -> dict[str, Any]:
    """Validate the complete metadata-only prepared-source inventory contract."""

    inventory = _json_copy(
        _mapping(value, name="prepared-source inventory"),
        name="prepared-source inventory",
    )
    require_exact_fields(
        inventory,
        expected=_INVENTORY_FIELDS,
        name="prepared-source inventory",
    )
    if inventory["schema"] != DEFORM360_PREPARED_SOURCE_INVENTORY_SCHEMA:
        raise ValueError("prepared-source inventory schema changed")
    if (
        type(inventory["schema_version"]) is not int
        or inventory["schema_version"] != DEFORM360_PREPARED_SOURCE_INVENTORY_VERSION
    ):
        raise ValueError("prepared-source inventory version changed")
    if inventory["semantics"] != DEFORM360_PREPARED_SOURCE_INVENTORY_SEMANTICS:
        raise ValueError("prepared-source inventory semantics changed")
    if inventory["status"] != DEFORM360_PREPARED_SOURCE_INVENTORY_STATUS:
        raise ValueError("prepared-source inventory is incomplete")
    exact_revision(
        inventory["implementation_revision"],
        name="inventory implementation_revision",
    )
    exact_revision(
        inventory["calibration_source_revision"],
        name="calibration_source_revision",
    )
    exact_revision(inventory["processing_revision"], name="processing_revision")
    for field in (
        "selection_artifact_sha256",
        "visual_provider_lock_id",
        "calibration_source_run_record_sha256",
    ):
        sha256_digest(inventory[field], name=field)
    if inventory["information_boundary"] != _INVENTORY_INFORMATION_BOUNDARY:
        raise ValueError("prepared-source inventory information boundary changed")
    if (
        inventory["claim_boundary"]
        != DEFORM360_PREPARED_SOURCE_INVENTORY_CLAIM_BOUNDARY
    ):
        raise ValueError("prepared-source inventory claim boundary changed")

    source_artifacts = _mapping(
        inventory["source_artifacts"],
        name="inventory source_artifacts",
    )
    if not source_artifacts:
        raise ValueError("inventory source_artifacts must not be empty")
    for path, digest in source_artifacts.items():
        _safe_relative_path(path, name="inventory source-artifact path")
        sha256_digest(digest, name=f"inventory source artifact {path}")
    required_result_path = "sources/calibration-source/result.json"
    if required_result_path not in source_artifacts:
        raise ValueError("inventory does not bind the calibration-source result")

    objects = _sequence(inventory["objects"], name="inventory objects")
    object_count = _literal_integer(
        inventory["object_count"],
        name="inventory object_count",
        minimum=1,
    )
    if (
        object_count != DEFORM360_CALIBRATION_OBJECT_COUNT
        or len(objects) != object_count
    ):
        raise ValueError("inventory must contain exactly ten calibration objects")

    object_ids: list[str] = []
    strata = {"sheet": 0, "volumetric": 0}
    for raw_object in objects:
        item = _mapping(raw_object, name="inventory object")
        require_exact_fields(
            item,
            expected=_INVENTORY_OBJECT_FIELDS,
            name="inventory object",
        )
        object_id = _literal_string(item["object_id"], name="inventory object_id")
        object_ids.append(object_id)
        _literal_integer(item["episode_id"], name=f"{object_id} episode_id")
        if item["synthetic_episode_index"] != 0:
            raise ValueError(f"inventory episode index changed: {object_id}")
        stratum = _literal_string(item["stratum"], name=f"{object_id} stratum")
        if stratum not in strata:
            raise ValueError(f"inventory stratum changed: {object_id}")
        strata[stratum] += 1
        aligned_frame_count = _literal_integer(
            item["aligned_frame_count"],
            name=f"{object_id} aligned_frame_count",
            minimum=1,
        )
        action_window = _mapping(
            item["action_window"],
            name=f"{object_id} action_window",
        )
        selected = _frame_range(
            action_window.get("selected_raw_frame_range_half_open"),
            name=f"{object_id} selected frame range",
            expected_count=DEFORM360_CALIBRATION_SELECTED_FRAME_COUNT,
        )
        prediction = _frame_range(
            action_window.get("prediction_raw_frame_range_half_open"),
            name=f"{object_id} prediction frame range",
            expected_count=DEFORM360_CALIBRATION_PREDICTION_FRAME_COUNT,
        )
        prefix = _frame_range(
            action_window.get("prefix_raw_frame_range_half_open"),
            name=f"{object_id} prefix frame range",
            expected_count=DEFORM360_CALIBRATION_PREFIX_FRAME_COUNT,
        )
        if not (
            selected[0] == prediction[0] == prefix[0]
            and prefix[1] <= prediction[1] <= selected[1] <= aligned_frame_count
        ):
            raise ValueError(f"inventory frame ranges are not nested: {object_id}")
        if not isinstance(item["episode_files"], Mapping):
            raise ValueError(f"inventory episode_files changed: {object_id}")
        if isinstance(item["tactile"], (str, bytes)) or not isinstance(
            item["tactile"], Sequence
        ):
            raise ValueError(f"inventory tactile records changed: {object_id}")
        cameras = _sequence(item["cameras"], name=f"{object_id} cameras")
        if not cameras:
            raise ValueError(f"inventory camera roster is empty: {object_id}")
        normalized_cameras = [
            _validate_inventory_camera(
                camera,
                object_id=object_id,
                aligned_frame_count=aligned_frame_count,
            )
            for camera in cameras
        ]
        camera_ids = [camera["camera"] for camera in normalized_cameras]
        if camera_ids != sorted(camera_ids) or len(set(camera_ids)) != len(camera_ids):
            raise ValueError(f"inventory camera roster is not canonical: {object_id}")
    if object_ids != sorted(object_ids) or len(set(object_ids)) != len(object_ids):
        raise ValueError("inventory object roster is not canonical")
    if strata != {"sheet": 5, "volumetric": 5}:
        raise ValueError("inventory must retain five objects per stratum")

    declared = sha256_digest(inventory["inventory_id"], name="inventory_id")
    identity = {key: item for key, item in inventory.items() if key != "inventory_id"}
    if declared != content_id(identity):
        raise ValueError("inventory_id does not match inventory content")
    return inventory


def _inventory_objects_by_id(
    inventory: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    return {
        cast(str, item["object_id"]): cast(Mapping[str, Any], item)
        for item in cast(list[dict[str, Any]], inventory["objects"])
    }


def _inventory_cameras_by_id(
    item: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    return {
        cast(str, camera["camera"]): cast(Mapping[str, Any], camera)
        for camera in cast(list[dict[str, Any]], item["cameras"])
    }


def _build_jobs(
    plan: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> list[dict[str, Any]]:
    inventory_objects = _inventory_objects_by_id(inventory)
    plan_objects = cast(list[dict[str, Any]], plan["objects"])
    if {item["object_id"] for item in plan_objects} != set(inventory_objects):
        raise ValueError("plan and inventory object cohorts differ")

    jobs: list[dict[str, Any]] = []
    for plan_object in plan_objects:
        object_id = cast(str, plan_object["object_id"])
        inventory_object = inventory_objects[object_id]
        for field in ("episode_id", "stratum"):
            if plan_object[field] != inventory_object[field]:
                raise ValueError(f"plan and inventory differ for {object_id}: {field}")
        action_window = cast(Mapping[str, Any], inventory_object["action_window"])
        range_pairs = (
            (
                "selected_source_frame_range_half_open",
                "selected_raw_frame_range_half_open",
            ),
            (
                "prediction_source_frame_range_half_open",
                "prediction_raw_frame_range_half_open",
            ),
            (
                "prefix_source_frame_range_half_open",
                "prefix_raw_frame_range_half_open",
            ),
        )
        for plan_field, inventory_field in range_pairs:
            if plan_object[plan_field] != action_window.get(inventory_field):
                raise ValueError(
                    f"plan and inventory frame ranges differ for {object_id}: "
                    f"{plan_field}"
                )
        inventory_cameras = _inventory_cameras_by_id(inventory_object)
        plan_cameras = cast(list[dict[str, Any]], plan_object["cameras"])
        plan_camera_ids = [camera["camera_id"] for camera in plan_cameras]
        if plan_camera_ids != list(inventory_cameras):
            raise ValueError(f"plan and inventory camera rosters differ: {object_id}")
        for plan_camera in plan_cameras:
            camera_id = cast(str, plan_camera["camera_id"])
            inventory_camera = inventory_cameras[camera_id]
            video = _validate_file_record(
                inventory_camera["video"],
                name=f"{object_id}/{camera_id} video",
            )
            timestamps = _validate_file_record(
                inventory_camera["timestamps"],
                name=f"{object_id}/{camera_id} timestamps",
            )
            if plan_camera["source_video_relative_path"] != video["path"]:
                raise ValueError(
                    f"plan video path differs from inventory: {object_id}/{camera_id}"
                )
            if plan_camera["source_timestamps_relative_path"] != timestamps["path"]:
                raise ValueError(
                    f"plan timestamp path differs from inventory: {object_id}/{camera_id}"
                )
            identity: dict[str, Any] = {
                "object_id": object_id,
                "episode_id": plan_object["episode_id"],
                "stratum": plan_object["stratum"],
                "object_root_seed": plan_object["object_root_seed"],
                "camera_id": camera_id,
                "view_root_seed": plan_camera["view_root_seed"],
                "call_namespace": plan_camera["call_namespace"],
                "selected_source_frame_range_half_open": plan_object[
                    "selected_source_frame_range_half_open"
                ],
                "prediction_source_frame_range_half_open": plan_object[
                    "prediction_source_frame_range_half_open"
                ],
                "prefix_source_frame_range_half_open": plan_object[
                    "prefix_source_frame_range_half_open"
                ],
                "aligned_frame_count": inventory_object["aligned_frame_count"],
                "source_video": video,
                "source_timestamps": timestamps,
                "camera_width": inventory_camera["width"],
                "camera_height": inventory_camera["height"],
                "camera_fps": inventory_camera["fps"],
                "camera_timeline_sha256": inventory_camera["timeline_sha256"],
                "output_relative_directory": plan_camera["output_relative_directory"],
                "dependence_group_ids": plan_camera["dependence_group_ids"],
            }
            jobs.append({**identity, "job_id": content_id(identity)})
    return jobs


def _validate_job(value: object) -> dict[str, Any]:
    job = _json_copy(_mapping(value, name="visual execution job"), name="job")
    require_exact_fields(job, expected=_JOB_FIELDS, name="visual execution job")
    _literal_string(job["object_id"], name="job object_id")
    _literal_integer(job["episode_id"], name="job episode_id")
    stratum = _literal_string(job["stratum"], name="job stratum")
    if stratum not in {"sheet", "volumetric"}:
        raise ValueError("job stratum changed")
    _literal_integer(job["object_root_seed"], name="job object_root_seed")
    _literal_string(job["camera_id"], name="job camera_id")
    _literal_integer(job["view_root_seed"], name="job view_root_seed")
    _literal_string(job["call_namespace"], name="job call_namespace")
    selected = _frame_range(
        job["selected_source_frame_range_half_open"],
        name="job selected frame range",
        expected_count=DEFORM360_CALIBRATION_SELECTED_FRAME_COUNT,
    )
    prediction = _frame_range(
        job["prediction_source_frame_range_half_open"],
        name="job prediction frame range",
        expected_count=DEFORM360_CALIBRATION_PREDICTION_FRAME_COUNT,
    )
    prefix = _frame_range(
        job["prefix_source_frame_range_half_open"],
        name="job prefix frame range",
        expected_count=DEFORM360_CALIBRATION_PREFIX_FRAME_COUNT,
    )
    aligned = _literal_integer(
        job["aligned_frame_count"],
        name="job aligned_frame_count",
        minimum=1,
    )
    if not (
        selected[0] == prediction[0] == prefix[0]
        and prefix[1] <= prediction[1] <= selected[1] <= aligned
    ):
        raise ValueError("job frame ranges are not nested")
    _validate_file_record(job["source_video"], name="job source_video")
    _validate_file_record(job["source_timestamps"], name="job source_timestamps")
    _literal_integer(job["camera_width"], name="job camera_width", minimum=1)
    _literal_integer(job["camera_height"], name="job camera_height", minimum=1)
    _positive_number(job["camera_fps"], name="job camera_fps")
    sha256_digest(job["camera_timeline_sha256"], name="job camera_timeline_sha256")
    _safe_relative_path(
        job["output_relative_directory"],
        name="job output_relative_directory",
    )
    groups = _sequence(job["dependence_group_ids"], name="job dependence_group_ids")
    if len(groups) != 2:
        raise ValueError("job must bind exactly two dependence groups")
    normalized_groups = [
        sha256_digest(group, name="job dependence group") for group in groups
    ]
    if len(set(normalized_groups)) != len(normalized_groups):
        raise ValueError("job dependence groups must be distinct")
    declared = sha256_digest(job["job_id"], name="job_id")
    identity = {key: item for key, item in job.items() if key != "job_id"}
    if declared != content_id(identity):
        raise ValueError("job_id does not match job content")
    return job


def validate_deform360_calibration_visual_execution_admission(
    value: object,
) -> dict[str, Any]:
    """Validate one complete metadata-only visual execution admission."""

    admission = _json_copy(
        _mapping(value, name="visual execution admission"),
        name="visual execution admission",
    )
    require_exact_fields(
        admission,
        expected=_ADMISSION_FIELDS,
        name="visual execution admission",
    )
    if admission["schema"] != DEFORM360_CALIBRATION_VISUAL_EXECUTION_ADMISSION_SCHEMA:
        raise ValueError("visual execution admission schema changed")
    if (
        type(admission["schema_version"]) is not int
        or admission["schema_version"]
        != DEFORM360_CALIBRATION_VISUAL_EXECUTION_ADMISSION_VERSION
    ):
        raise ValueError("visual execution admission version changed")
    if (
        admission["semantics"]
        != DEFORM360_CALIBRATION_VISUAL_EXECUTION_ADMISSION_SEMANTICS
    ):
        raise ValueError("visual execution admission semantics changed")
    for field in (
        "implementation_revision",
        "plan_implementation_revision",
        "inventory_implementation_revision",
        "calibration_source_revision",
        "processing_revision",
        "provider_revision",
        "motioncrafter_revision",
    ):
        exact_revision(admission[field], name=field)
    for field in (
        "plan_id",
        "plan_file_sha256",
        "inventory_id",
        "inventory_file_sha256",
        "selection_artifact_sha256",
        "visual_provider_lock_id",
        "calibration_source_run_record_sha256",
        "calibration_source_result_sha256",
        "model_set_id",
    ):
        sha256_digest(admission[field], name=field)
    for field in ("plan_file_byte_count", "inventory_file_byte_count"):
        _literal_integer(admission[field], name=field, minimum=1)
    _literal_string(admission["protocol_id"], name="protocol_id")
    if admission["information_boundary"] != _ADMISSION_INFORMATION_BOUNDARY:
        raise ValueError("visual execution admission information boundary changed")
    if (
        admission["claim_boundary"]
        != DEFORM360_CALIBRATION_VISUAL_EXECUTION_ADMISSION_CLAIM_BOUNDARY
    ):
        raise ValueError("visual execution admission claim boundary changed")

    jobs = [
        _validate_job(job)
        for job in _sequence(admission["jobs"], name="admission jobs")
    ]
    camera_view_count = _literal_integer(
        admission["camera_view_count"],
        name="camera_view_count",
        minimum=1,
    )
    if camera_view_count != len(jobs):
        raise ValueError("camera_view_count differs from admission jobs")
    object_count = _literal_integer(
        admission["object_count"],
        name="object_count",
        minimum=1,
    )
    object_ids = [cast(str, job["object_id"]) for job in jobs]
    unique_objects = sorted(set(object_ids))
    if (
        object_count != len(unique_objects)
        or object_count != DEFORM360_CALIBRATION_OBJECT_COUNT
    ):
        raise ValueError("admission must contain exactly ten physical objects")
    ordering = [
        (cast(str, job["object_id"]), cast(str, job["camera_id"])) for job in jobs
    ]
    if ordering != sorted(ordering) or len(set(ordering)) != len(ordering):
        raise ValueError("admission jobs must be sorted and unique")
    outputs = [cast(str, job["output_relative_directory"]) for job in jobs]
    namespaces = [cast(str, job["call_namespace"]) for job in jobs]
    view_seeds = [cast(int, job["view_root_seed"]) for job in jobs]
    if len(set(outputs)) != len(outputs):
        raise ValueError("admission output path collision detected")
    if len(set(namespaces)) != len(namespaces):
        raise ValueError("admission call namespace collision detected")
    if len(set(view_seeds)) != len(view_seeds):
        raise ValueError("admission view seed collision detected")
    object_strata: dict[str, str] = {}
    object_contracts: dict[str, tuple[object, ...]] = {}
    for job in jobs:
        object_id = cast(str, job["object_id"])
        stratum = cast(str, job["stratum"])
        previous_stratum = object_strata.setdefault(object_id, stratum)
        if previous_stratum != stratum:
            raise ValueError(f"admission stratum changes within object {object_id}")
        contract = (
            job["episode_id"],
            job["object_root_seed"],
            tuple(job["selected_source_frame_range_half_open"]),
            tuple(job["prediction_source_frame_range_half_open"]),
            tuple(job["prefix_source_frame_range_half_open"]),
            job["aligned_frame_count"],
        )
        previous_contract = object_contracts.setdefault(object_id, contract)
        if previous_contract != contract:
            raise ValueError(f"admission object contract changes within {object_id}")
    if sorted(object_strata.values()).count("sheet") != 5:
        raise ValueError("admission must retain five sheet objects")
    if sorted(object_strata.values()).count("volumetric") != 5:
        raise ValueError("admission must retain five volumetric objects")

    declared = sha256_digest(admission["admission_id"], name="admission_id")
    identity = {key: item for key, item in admission.items() if key != "admission_id"}
    if declared != content_id(identity):
        raise ValueError("admission_id does not match admission content")
    return admission


def build_deform360_calibration_visual_execution_admission(
    *,
    visual_production_plan_path: str | Path,
    prepared_source_inventory_path: str | Path,
    implementation_revision: str,
) -> dict[str, Any]:
    """Join the exact plan and inventory into one executable metadata contract."""

    plan_value, plan_file_sha256, plan_file_byte_count = _load_stable_json_object(
        visual_production_plan_path,
        label="visual production plan",
    )
    plan = validate_deform360_calibration_visual_production_plan(plan_value)
    inventory_value, inventory_file_sha256, inventory_file_byte_count = (
        _load_stable_json_object(
            prepared_source_inventory_path,
            label="prepared-source inventory",
        )
    )
    inventory = validate_deform360_prepared_source_inventory(inventory_value)

    for field in (
        "selection_artifact_sha256",
        "visual_provider_lock_id",
        "calibration_source_run_record_sha256",
    ):
        if plan[field] != inventory[field]:
            raise ValueError(f"plan and inventory differ: {field}")
    inventory_sources = cast(Mapping[str, Any], inventory["source_artifacts"])
    result_sha256 = sha256_digest(
        inventory_sources["sources/calibration-source/result.json"],
        name="inventory calibration-source result",
    )
    if plan["calibration_source_result_sha256"] != result_sha256:
        raise ValueError("plan and inventory differ: calibration-source result")

    jobs = _build_jobs(plan, inventory)
    provider = cast(Mapping[str, Any], plan["provider"])
    identity: dict[str, Any] = {
        "schema": DEFORM360_CALIBRATION_VISUAL_EXECUTION_ADMISSION_SCHEMA,
        "schema_version": DEFORM360_CALIBRATION_VISUAL_EXECUTION_ADMISSION_VERSION,
        "semantics": DEFORM360_CALIBRATION_VISUAL_EXECUTION_ADMISSION_SEMANTICS,
        "implementation_revision": exact_revision(
            implementation_revision,
            name="implementation_revision",
        ),
        "plan_id": plan["plan_id"],
        "plan_file_sha256": plan_file_sha256,
        "plan_file_byte_count": plan_file_byte_count,
        "inventory_id": inventory["inventory_id"],
        "inventory_file_sha256": inventory_file_sha256,
        "inventory_file_byte_count": inventory_file_byte_count,
        "protocol_id": plan["protocol_id"],
        "selection_artifact_sha256": plan["selection_artifact_sha256"],
        "visual_provider_lock_id": plan["visual_provider_lock_id"],
        "calibration_source_run_record_sha256": plan[
            "calibration_source_run_record_sha256"
        ],
        "calibration_source_result_sha256": result_sha256,
        "plan_implementation_revision": plan["implementation_revision"],
        "inventory_implementation_revision": inventory["implementation_revision"],
        "calibration_source_revision": inventory["calibration_source_revision"],
        "processing_revision": inventory["processing_revision"],
        "provider_revision": provider["revision"],
        "motioncrafter_revision": provider["motioncrafter_revision"],
        "model_set_id": provider["model_set_id"],
        "object_count": plan["object_count"],
        "camera_view_count": plan["camera_view_count"],
        "jobs": jobs,
        "information_boundary": dict(_ADMISSION_INFORMATION_BOUNDARY),
        "claim_boundary": (
            DEFORM360_CALIBRATION_VISUAL_EXECUTION_ADMISSION_CLAIM_BOUNDARY
        ),
    }
    return validate_deform360_calibration_visual_execution_admission(
        {**identity, "admission_id": content_id(identity)}
    )


def save_deform360_calibration_visual_execution_admission(
    path: str | Path,
    value: Mapping[str, Any],
) -> None:
    """Validate and atomically publish one admission without replacement."""

    validated = validate_deform360_calibration_visual_execution_admission(value)
    write_atomic_json(validated, path, overwrite=False)


def load_deform360_calibration_visual_execution_admission(
    path: str | Path,
) -> dict[str, Any]:
    """Load one descriptor-stable admission and revalidate all identities."""

    value, _digest, _byte_count = _load_stable_json_object(
        path,
        label="visual execution admission",
    )
    return validate_deform360_calibration_visual_execution_admission(value)


__all__ = [
    "DEFORM360_CALIBRATION_VISUAL_EXECUTION_ADMISSION_CLAIM_BOUNDARY",
    "DEFORM360_CALIBRATION_VISUAL_EXECUTION_ADMISSION_SCHEMA",
    "DEFORM360_CALIBRATION_VISUAL_EXECUTION_ADMISSION_SEMANTICS",
    "DEFORM360_CALIBRATION_VISUAL_EXECUTION_ADMISSION_VERSION",
    "DEFORM360_PREPARED_SOURCE_INVENTORY_CLAIM_BOUNDARY",
    "build_deform360_calibration_visual_execution_admission",
    "load_deform360_calibration_visual_execution_admission",
    "save_deform360_calibration_visual_execution_admission",
    "validate_deform360_calibration_visual_execution_admission",
    "validate_deform360_prepared_source_inventory",
]
