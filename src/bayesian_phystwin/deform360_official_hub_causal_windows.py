"""Causal-window manifests for the official-Hub Deform360 calibration cohort.

This module consumes only the already-acquired calibration view. It derives the
frozen tactile event window and pose-only camera panel while keeping prediction
scores, provider comparisons, and confirmation payloads outside the boundary.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from ._portable_contracts import (
    content_id,
    load_strict_json_object,
    write_atomic_json,
)
from .deform360_visual_provider_recovery_lock import (
    DEFORM360_FUTURE_FRAMES,
    DEFORM360_OBSERVED_CONTACT_FRAMES,
    DEFORM360_OBSERVED_HISTORY_FRAMES,
    DEFORM360_PROVIDER_OVERLAP,
    DEFORM360_PROVIDER_WINDOW_COUNT,
    DEFORM360_PROVIDER_WINDOW_SIZE,
    Deform360CausalWindowV1,
    Deform360VisualProviderRecoveryLockV1,
    derive_deform360_causal_window,
    first_deform360_contact_frame,
    select_deform360_camera_panel,
)

DEFORM360_CAUSAL_WINDOW_MANIFEST_SCHEMA = (
    "bayesian-phystwin.deform360-official-hub-causal-window-manifest"
)
DEFORM360_CAUSAL_WINDOW_MANIFEST_VERSION = 1
DEFORM360_CAUSAL_WINDOW_MANIFEST_SCHEMA_V2 = (
    "bayesian-phystwin.deform360-official-hub-causal-window-manifest-v2"
)
DEFORM360_STAGE1_PROCESSING_REPORT_SCHEMA = (
    "bayesian-phystwin/deform360-official-hub-stage1-processing-report-v1"
)
DEFORM360_VISUAL_EXECUTION_LOCK_ID = (
    "87b1efe7dc7e9a8f1fd4163e0a4164b5ba45e6bcc47d66e2fba2a2526c6f51e9"
)
DEFORM360_VISUAL_PROVIDER_RECOVERY_LOCK_ID = (
    "02d90e58f4f21d052073e098469ff8a7cd991f48b895d7496cebdb84dd10cb3d"
)
DEFORM360_CAMERA_PANEL_POLICY_ID = (
    "ec8d7f56bb59731b6c5eec03fd627f79ef5864a447257f198a5c8d3b4869ffb5"
)
DEFORM360_CAUSAL_SCHEDULE_RECOVERY_LOCK_ID = (
    "56a6ebc0ac65e19c098ccfa83cda1be9990c579b396d46cfdd13c52bc5f0530e"
)
DEFORM360_V1_CAUSAL_WINDOW_MANIFEST_ID = (
    "9a23701c8181b89ba5b09cb545ca750513fb1e3f9a8d360099d00582d8b9f406"
)
DEFORM360_V1_CAUSAL_WINDOW_MANIFEST_FILE_SHA256 = (
    "3367947f2c6d78c67d5c0c2691f302a4a1838d9b069ffd30773da5d9c8e04a8b"
)

CameraCalibrationLoader = Callable[
    [Path], tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]]
]


class Deform360CustodyError(ValueError):
    """Raised when an acquired file no longer matches the Stage 1 inventory."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_safe_name(value: object, *, name: str) -> str:
    _require(type(value) is str and bool(value), f"{name} must be a string")
    result = str(value)
    path = PurePosixPath(result)
    _require(
        result == path.as_posix()
        and not path.is_absolute()
        and len(path.parts) == 1
        and path.parts[0] not in {".", ".."},
        f"unsafe {name}: {result}",
    )
    return result


def _validated_content_address(
    value: Mapping[str, Any],
    *,
    id_field: str,
    expected_id: str | None,
    name: str,
) -> str:
    declared = value.get(id_field)
    _require(type(declared) is str, f"{name} {id_field} is missing")
    canonical = dict(value)
    canonical.pop(id_field)
    computed = content_id(canonical)
    _require(declared == computed, f"{name} content identity changed")
    if expected_id is not None:
        _require(declared == expected_id, f"unexpected {name} identity")
    return computed


def validate_deform360_visual_execution_lock(
    value: Mapping[str, Any],
) -> str:
    """Validate the transitive provider and camera-panel execution lock."""

    lock_id = _validated_content_address(
        value,
        id_field="artifact_id",
        expected_id=DEFORM360_VISUAL_EXECUTION_LOCK_ID,
        name="visual execution lock",
    )
    _require(
        value.get("schema") == "bayesian-phystwin.deform360-visual-execution-lock"
        and value.get("schema_version") == 1,
        "unsupported visual execution lock",
    )
    _require(
        value.get("status") == "locked-post-payload-pre-score",
        "visual execution lock has the wrong information boundary",
    )
    provider = value.get("visual_provider_recovery_lock")
    panel = value.get("camera_panel_policy")
    _require(isinstance(provider, Mapping), "provider binding is missing")
    _require(isinstance(panel, Mapping), "camera-panel binding is missing")
    _require(
        provider.get("artifact_id") == DEFORM360_VISUAL_PROVIDER_RECOVERY_LOCK_ID,
        "visual execution lock changed provider recovery lock",
    )
    _require(
        panel.get("artifact_id") == DEFORM360_CAMERA_PANEL_POLICY_ID,
        "visual execution lock changed camera-panel policy",
    )
    boundary = value.get("information_boundary")
    _require(isinstance(boundary, Mapping), "execution information boundary is missing")
    for field in (
        "calibration_scores_opened",
        "camera_image_values_used_to_choose_panel",
        "confirmation_payloads_opened",
        "target_outcomes_used",
    ):
        _require(boundary.get(field) is False, f"execution lock opened {field}")
    return lock_id


def validate_deform360_causal_schedule_recovery_lock(
    value: Mapping[str, Any],
) -> str:
    """Validate the post-v1-feasibility, pre-score v2 schedule lock."""

    lock_id = _validated_content_address(
        value,
        id_field="artifact_id",
        expected_id=DEFORM360_CAUSAL_SCHEDULE_RECOVERY_LOCK_ID,
        name="causal schedule recovery lock",
    )
    _require(
        value.get("schema")
        == "bayesian-phystwin.deform360-causal-schedule-recovery-lock"
        and value.get("schema_version") == 1,
        "unsupported causal schedule recovery lock",
    )
    _require(
        value.get("status") == "locked-post-v1-feasibility-pre-provider-score",
        "causal schedule lock has the wrong information boundary",
    )
    parent_execution = value.get("parent_visual_execution_lock")
    parent_provider = value.get("parent_provider_recovery_lock")
    source = value.get("source_feasibility")
    schedule = value.get("schedule")
    _require(isinstance(parent_execution, Mapping), "parent execution lock is missing")
    _require(isinstance(parent_provider, Mapping), "parent provider lock is missing")
    _require(isinstance(source, Mapping), "v1 feasibility binding is missing")
    _require(isinstance(schedule, Mapping), "v2 schedule is missing")
    _require(
        parent_execution.get("artifact_id") == DEFORM360_VISUAL_EXECUTION_LOCK_ID,
        "v2 schedule changed the visual execution lock",
    )
    _require(
        parent_provider.get("artifact_id")
        == DEFORM360_VISUAL_PROVIDER_RECOVERY_LOCK_ID,
        "v2 schedule changed the provider recovery lock",
    )
    _require(
        source.get("manifest_id") == DEFORM360_V1_CAUSAL_WINDOW_MANIFEST_ID
        and source.get("file_sha256") == DEFORM360_V1_CAUSAL_WINDOW_MANIFEST_FILE_SHA256
        and source.get("supported_objects") == 4
        and source.get("required_supported_objects") == 8
        and source.get("retained_technical_failures") == 6
        and source.get("provider_inference_run") is False
        and source.get("calibration_scores_opened") is False,
        "v2 schedule changed the v1 feasibility evidence",
    )
    expected_schedule = {
        "version": "earliest-fully-observed-two-window-v2",
        "event_clock": "official-processed-tactile-first-contact-v1",
        "observed_history_frames": 42,
        "minimum_post_contact_frames": 6,
        "causal_cutoff_rule": "max(first_contact_frame + 6, 42)",
        "source_start_rule": "causal_cutoff_frame - 42",
        "future_frames": 24,
        "window_size": 25,
        "overlap": 8,
        "window_count": 2,
        "causal_cutoff_convention": "exclusive",
        "future_use_for_prediction": "forbidden",
    }
    _require(dict(schedule) == expected_schedule, "v2 schedule changed")
    boundary = value.get("information_boundary")
    _require(isinstance(boundary, Mapping), "v2 information boundary is missing")
    _require(
        boundary.get("v1_schedule_feasibility_opened") is True
        and boundary.get("calibration_tactile_schedule_values_used_for_v2_design")
        is True
        and boundary.get("calibration_camera_images_opened_for_provider_inference")
        is False
        and boundary.get("calibration_provider_outputs_opened") is False
        and boundary.get("calibration_scores_opened") is False
        and boundary.get("calibration_policy_fit") is False
        and boundary.get("confirmation_payloads_opened") is False
        and boundary.get("target_outcomes_used") is False,
        "v2 schedule crossed its information boundary",
    )
    return lock_id


def derive_deform360_causal_window_v2(
    tactile_by_sensor: Mapping[str, np.ndarray],
    *,
    total_episode_frames: int,
) -> Deform360CausalWindowV1:
    """Derive the earliest full two-window prefix with six contact frames."""

    contact_start = first_deform360_contact_frame(
        tactile_by_sensor,
        total_episode_frames=total_episode_frames,
    )
    cutoff = max(
        contact_start + DEFORM360_OBSERVED_CONTACT_FRAMES,
        DEFORM360_OBSERVED_HISTORY_FRAMES,
    )
    source_start = cutoff - DEFORM360_OBSERVED_HISTORY_FRAMES
    future_stop = cutoff + DEFORM360_FUTURE_FRAMES
    if future_stop > total_episode_frames:
        raise ValueError("insufficient untouched future for the v2 evaluation")
    return Deform360CausalWindowV1(
        contact_start_frame=contact_start,
        source_start_frame=source_start,
        causal_cutoff_frame=cutoff,
        future_stop_frame=future_stop,
        total_episode_frames=total_episode_frames,
    )


def validate_deform360_stage1_processing_report(
    value: Mapping[str, Any],
) -> str:
    """Validate the complete ten-object calibration processing report."""

    report_id = _validated_content_address(
        value,
        id_field="processing_report_sha256",
        expected_id=None,
        name="Stage 1 processing report",
    )
    _require(
        value.get("schema") == DEFORM360_STAGE1_PROCESSING_REPORT_SCHEMA
        and value.get("schema_version") == 1,
        "unsupported Stage 1 processing report",
    )
    _require(value.get("role") == "calibration", "report is not calibration-only")
    _require(value.get("status") == "complete", "Stage 1 processing is incomplete")
    _require(value.get("object_count") == 10, "Stage 1 object count changed")
    _require(value.get("success_count") == 10, "Stage 1 success count changed")
    _require(
        value.get("retained_technical_failure_count") == 0,
        "Stage 1 unexpectedly contains processing failures",
    )
    boundary = value.get("information_boundary")
    _require(isinstance(boundary, Mapping), "processing boundary is missing")
    _require(
        boundary.get("confirmation_payload_opened") is False
        and boundary.get("future_target_opened") is False
        and boundary.get("replacement_performed") is False,
        "Stage 1 processing crossed the frozen boundary",
    )
    objects = value.get("objects")
    _require(isinstance(objects, list), "Stage 1 object records are missing")
    _require(len(objects) == 10, "Stage 1 object records changed")
    return report_id


def _inventory(row: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records = row.get("output_files")
    _require(isinstance(records, list), "Stage 1 output inventory is missing")
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        _require(isinstance(record, Mapping), "invalid Stage 1 file record")
        relative = record.get("path")
        _require(type(relative) is str and bool(relative), "file path is missing")
        relative_path = PurePosixPath(str(relative))
        _require(
            str(relative) == relative_path.as_posix()
            and not relative_path.is_absolute()
            and ".." not in relative_path.parts,
            f"unsafe Stage 1 file path: {relative}",
        )
        _require(str(relative) not in result, f"duplicate Stage 1 file: {relative}")
        result[str(relative)] = record
    return result


def _verified_file_record(
    *,
    object_id: str,
    object_root: Path,
    relative: str,
    inventory: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    record = inventory.get(relative)
    if record is None:
        raise Deform360CustodyError(f"unbound Stage 1 file: {object_id}/{relative}")
    path = object_root / relative
    if not path.is_file():
        raise Deform360CustodyError(f"missing Stage 1 file: {object_id}/{relative}")
    actual_size = path.stat().st_size
    expected_size = record.get("size")
    expected_sha256 = record.get("sha256")
    if type(expected_size) is not int or expected_size < 0:
        raise Deform360CustodyError(f"invalid bound size: {object_id}/{relative}")
    if (
        type(expected_sha256) is not str
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise Deform360CustodyError(f"invalid bound digest: {object_id}/{relative}")
    actual_sha256 = _file_sha256(path)
    if actual_size != expected_size or actual_sha256 != expected_sha256:
        raise Deform360CustodyError(f"Stage 1 file drift: {object_id}/{relative}")
    return {
        "path": f"{object_id}/{relative}",
        "size": actual_size,
        "sha256": actual_sha256,
    }


def _validate_camera_calibration(
    intrinsics: Mapping[str, np.ndarray],
    extrinsics: Mapping[str, np.ndarray],
) -> None:
    _require(set(intrinsics) == set(extrinsics), "camera calibration keys differ")
    _require(len(intrinsics) >= 3, "fewer than three calibrated cameras")
    for camera, matrix in intrinsics.items():
        _require(type(camera) is str and bool(camera), "invalid camera name")
        value = np.asarray(matrix, dtype=np.float64)
        _require(
            value.shape == (3, 3) and np.isfinite(value).all(),
            f"invalid intrinsics for {camera}",
        )


def _provider_windows(source_start: int) -> list[dict[str, int]]:
    step = DEFORM360_PROVIDER_WINDOW_SIZE - DEFORM360_PROVIDER_OVERLAP
    return [
        {
            "window_index": index,
            "frame_start": source_start + index * step,
            "frame_stop_exclusive": (
                source_start + index * step + DEFORM360_PROVIDER_WINDOW_SIZE
            ),
        }
        for index in range(DEFORM360_PROVIDER_WINDOW_COUNT)
    ]


def build_deform360_official_hub_causal_window_manifest(
    *,
    processing_report: Mapping[str, Any],
    processed_root: str | Path,
    provider_lock: Deform360VisualProviderRecoveryLockV1,
    execution_lock: Mapping[str, Any],
    implementation_revision: str,
    camera_calibration_loader: CameraCalibrationLoader,
    schedule_recovery_lock: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the target-free calibration window and camera-panel manifest."""

    processing_report_id = validate_deform360_stage1_processing_report(
        processing_report
    )
    execution_lock_id = validate_deform360_visual_execution_lock(execution_lock)
    if schedule_recovery_lock is None:
        manifest_schema = DEFORM360_CAUSAL_WINDOW_MANIFEST_SCHEMA
        window_deriver = derive_deform360_causal_window
        schedule_recovery_lock_id = None
    else:
        manifest_schema = DEFORM360_CAUSAL_WINDOW_MANIFEST_SCHEMA_V2
        window_deriver = derive_deform360_causal_window_v2
        schedule_recovery_lock_id = validate_deform360_causal_schedule_recovery_lock(
            schedule_recovery_lock
        )
    _require(
        provider_lock.artifact_id == DEFORM360_VISUAL_PROVIDER_RECOVERY_LOCK_ID,
        "unexpected visual-provider recovery lock",
    )
    _require(
        type(implementation_revision) is str
        and len(implementation_revision) == 40
        and all(
            character in "0123456789abcdef" for character in implementation_revision
        ),
        "implementation revision must be an exact Git commit",
    )

    root = Path(processed_root).resolve()
    _require(root.is_dir(), "processed calibration root is missing")
    cases: list[dict[str, Any]] = []
    seen_objects: set[str] = set()
    for raw_row in processing_report["objects"]:
        _require(isinstance(raw_row, Mapping), "invalid Stage 1 object record")
        object_id = _require_safe_name(raw_row.get("object_id"), name="object_id")
        _require(object_id not in seen_objects, f"duplicate object: {object_id}")
        seen_objects.add(object_id)
        _require(raw_row.get("status") == "success", "non-success Stage 1 object")
        _require(
            raw_row.get("processing_episode_index") == 0,
            "processing episode index changed",
        )
        frame_count = raw_row.get("frame_count")
        _require(
            type(frame_count) is int and frame_count > 0,
            f"invalid frame count for {object_id}",
        )
        inventory = _inventory(raw_row)
        object_root = root / object_id
        episode_root = object_root / "episode_0000"
        tactile_outputs = raw_row.get("tactile_outputs")
        _require(
            isinstance(tactile_outputs, list) and len(tactile_outputs) == 4,
            f"invalid tactile streams for {object_id}",
        )

        bound_files: list[dict[str, object]] = []
        tactile_paths: dict[str, Path] = {}
        for raw_sensor in sorted(tactile_outputs):
            sensor = _require_safe_name(raw_sensor, name="tactile sensor")
            relative = f"episode_0000/{sensor}/synced_tactile.npy"
            bound_files.append(
                _verified_file_record(
                    object_id=object_id,
                    object_root=object_root,
                    relative=relative,
                    inventory=inventory,
                )
            )
            tactile_paths[sensor] = object_root / relative
        for relative in (
            "episode_0000/extrinsics.npy",
            "episode_0000/undistorted_intrinsics.npy",
        ):
            bound_files.append(
                _verified_file_record(
                    object_id=object_id,
                    object_root=object_root,
                    relative=relative,
                    inventory=inventory,
                )
            )

        case: dict[str, Any] = {
            "object_id": object_id,
            "stratum": raw_row.get("stratum"),
            "source_episode_id": raw_row.get("source_episode_id"),
            "processing_episode_index": 0,
            "action": raw_row.get("action"),
            "source_output_tree_sha256": raw_row.get("output_tree_sha256"),
        }
        try:
            tactile_by_sensor = {
                sensor: np.load(path, allow_pickle=False)
                for sensor, path in tactile_paths.items()
            }
            intrinsics, extrinsics = camera_calibration_loader(episode_root)
            _validate_camera_calibration(intrinsics, extrinsics)
            panel = select_deform360_camera_panel(extrinsics)
            for camera in panel:
                for filename in (
                    "aligned_timestamps.txt",
                    "alignment.json",
                    "metadata.json",
                    "undistorted.mp4",
                ):
                    relative = f"episode_0000/{camera}/{filename}"
                    bound_files.append(
                        _verified_file_record(
                            object_id=object_id,
                            object_root=object_root,
                            relative=relative,
                            inventory=inventory,
                        )
                    )
            window = window_deriver(
                tactile_by_sensor,
                total_episode_frames=frame_count,
            )
            case.update(
                {
                    "status": "success",
                    "camera_panel": list(panel),
                    "reference_camera": panel[0],
                    "causal_window": window.to_record(),
                    "provider_windows": _provider_windows(window.source_start_frame),
                    "untouched_future": {
                        "frame_start": window.causal_cutoff_frame,
                        "frame_stop_exclusive": window.future_stop_frame,
                    },
                }
            )
        except Deform360CustodyError:
            raise
        except Exception as error:
            case.update(
                {
                    "status": "retained_technical_failure",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
        case["bound_input_files"] = sorted(
            bound_files,
            key=lambda record: str(record["path"]),
        )
        case["bound_input_files_sha256"] = content_id(
            {"files": case["bound_input_files"]}
        )
        cases.append(case)

    cases.sort(key=lambda row: row["object_id"])
    success_count = sum(case["status"] == "success" for case in cases)
    descriptor: dict[str, Any] = {
        "schema": manifest_schema,
        "schema_version": DEFORM360_CAUSAL_WINDOW_MANIFEST_VERSION,
        "protocol_id": provider_lock.protocol_id,
        "role": "calibration",
        "status": (
            "complete"
            if success_count == len(cases)
            else "complete_with_retained_technical_failures"
        ),
        "implementation_revision": implementation_revision,
        "processing_report_sha256": processing_report_id,
        "visual_execution_lock_id": execution_lock_id,
        "visual_provider_recovery_lock_id": provider_lock.artifact_id,
        "camera_panel_policy_id": DEFORM360_CAMERA_PANEL_POLICY_ID,
        "object_count": len(cases),
        "success_count": success_count,
        "retained_technical_failure_count": len(cases) - success_count,
        "replacement_performed": False,
        "cases": cases,
        "information_boundary": {
            "calibration_tactile_values_used_for_causal_window": True,
            "calibration_camera_pose_values_used_for_panel": True,
            "calibration_camera_image_values_used_for_panel": False,
            "calibration_scores_opened": False,
            "calibration_policy_fit": False,
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
            "future_frames_used_for_prediction": False,
        },
        "claim_boundary": (
            "Target-free calibration scheduling and input custody only. This artifact "
            "does not establish provider competence, predictive improvement, "
            "calibration, confirmation performance, or state-of-the-art performance."
        ),
    }
    if schedule_recovery_lock_id is not None:
        descriptor.update(
            {
                "causal_schedule_recovery_lock_id": schedule_recovery_lock_id,
                "parent_v1_causal_window_manifest_id": (
                    DEFORM360_V1_CAUSAL_WINDOW_MANIFEST_ID
                ),
            }
        )
        descriptor["information_boundary"].update(
            {
                "v1_schedule_feasibility_opened": True,
                "calibration_tactile_schedule_values_used_for_v2_design": True,
            }
        )
    return {"manifest_sha256": content_id(descriptor), **descriptor}


def _validate_deform360_official_hub_causal_window_manifest(
    value: Mapping[str, Any],
    *,
    schema: str,
    schedule_recovery_lock_id: str | None,
) -> str:
    manifest_id = _validated_content_address(
        value,
        id_field="manifest_sha256",
        expected_id=None,
        name="causal-window manifest",
    )
    _require(
        value.get("schema") == schema
        and value.get("schema_version") == DEFORM360_CAUSAL_WINDOW_MANIFEST_VERSION,
        "unsupported causal-window manifest",
    )
    _require(value.get("role") == "calibration", "manifest role changed")
    _require(
        value.get("protocol_id") == "deform360-official-hub-visuotactile-v1",
        "manifest protocol changed",
    )
    _require(
        value.get("visual_execution_lock_id") == DEFORM360_VISUAL_EXECUTION_LOCK_ID,
        "manifest execution lock changed",
    )
    _require(
        value.get("visual_provider_recovery_lock_id")
        == DEFORM360_VISUAL_PROVIDER_RECOVERY_LOCK_ID,
        "manifest provider lock changed",
    )
    _require(
        value.get("camera_panel_policy_id") == DEFORM360_CAMERA_PANEL_POLICY_ID,
        "manifest camera-panel policy changed",
    )
    if schedule_recovery_lock_id is None:
        _require(
            "causal_schedule_recovery_lock_id" not in value
            and "parent_v1_causal_window_manifest_id" not in value,
            "v1 manifest unexpectedly binds a later schedule",
        )
    else:
        _require(
            value.get("causal_schedule_recovery_lock_id") == schedule_recovery_lock_id,
            "manifest schedule recovery lock changed",
        )
        _require(
            value.get("parent_v1_causal_window_manifest_id")
            == DEFORM360_V1_CAUSAL_WINDOW_MANIFEST_ID,
            "manifest v1 feasibility parent changed",
        )
    _require(value.get("object_count") == 10, "manifest object count changed")
    cases = value.get("cases")
    _require(isinstance(cases, list) and len(cases) == 10, "manifest cases changed")
    object_ids: list[str] = []
    for case in cases:
        _require(isinstance(case, Mapping), "manifest case is not an object")
        object_ids.append(_require_safe_name(case.get("object_id"), name="object_id"))
        files = case.get("bound_input_files")
        _require(isinstance(files, list), "bound input files are missing")
        _require(
            case.get("bound_input_files_sha256") == content_id({"files": files}),
            "bound input-file identity changed",
        )
        if case.get("status") == "success":
            panel = case.get("camera_panel")
            _require(
                isinstance(panel, list)
                and len(panel) == 3
                and len(set(panel)) == 3
                and panel == sorted(panel),
                "successful case camera panel changed",
            )
            _require(
                case.get("reference_camera") == panel[0],
                "successful case reference camera changed",
            )
            causal_window = case.get("causal_window")
            _require(isinstance(causal_window, Mapping), "causal window is missing")
            contact_start = causal_window.get("contact_start_frame")
            source_start = causal_window.get("source_start_frame")
            cutoff = causal_window.get("causal_cutoff_frame")
            future_stop = causal_window.get("future_stop_frame")
            _require(
                all(
                    type(item) is int
                    for item in (contact_start, source_start, cutoff, future_stop)
                ),
                "causal window indices changed type",
            )
            expected_cutoff = int(contact_start) + DEFORM360_OBSERVED_CONTACT_FRAMES
            if schedule_recovery_lock_id is not None:
                expected_cutoff = max(
                    expected_cutoff,
                    DEFORM360_OBSERVED_HISTORY_FRAMES,
                )
            _require(
                causal_window.get("observed_frame_count") == 42
                and causal_window.get("future_frame_count") == 24
                and causal_window.get("processing_frame_count") == 66
                and cutoff == expected_cutoff
                and source_start == int(cutoff) - 42
                and future_stop == int(cutoff) + 24,
                "causal window dimensions changed",
            )
            _require(
                case.get("provider_windows") == _provider_windows(int(source_start)),
                "provider windows changed",
            )
            _require(
                case.get("untouched_future")
                == {
                    "frame_start": cutoff,
                    "frame_stop_exclusive": future_stop,
                },
                "untouched future changed",
            )
        else:
            _require(
                case.get("status") == "retained_technical_failure",
                "unknown case status",
            )
    _require(object_ids == sorted(set(object_ids)), "manifest cases are not canonical")
    success_count = sum(
        isinstance(case, Mapping) and case.get("status") == "success" for case in cases
    )
    _require(value.get("success_count") == success_count, "success count changed")
    _require(
        value.get("retained_technical_failure_count") == len(cases) - success_count,
        "failure count changed",
    )
    expected_status = (
        "complete"
        if success_count == len(cases)
        else "complete_with_retained_technical_failures"
    )
    _require(value.get("status") == expected_status, "manifest status changed")
    _require(value.get("replacement_performed") is False, "replacement was performed")
    boundary = value.get("information_boundary")
    _require(isinstance(boundary, Mapping), "manifest boundary is missing")
    _require(
        boundary.get("calibration_tactile_values_used_for_causal_window") is True
        and boundary.get("calibration_camera_pose_values_used_for_panel") is True
        and boundary.get("calibration_camera_image_values_used_for_panel") is False
        and boundary.get("calibration_scores_opened") is False
        and boundary.get("calibration_policy_fit") is False
        and boundary.get("confirmation_payloads_opened") is False
        and boundary.get("target_outcomes_used") is False
        and boundary.get("future_frames_used_for_prediction") is False,
        "manifest information boundary changed",
    )
    if schedule_recovery_lock_id is not None:
        _require(
            boundary.get("v1_schedule_feasibility_opened") is True
            and boundary.get("calibration_tactile_schedule_values_used_for_v2_design")
            is True,
            "v2 manifest omitted its schedule-development boundary",
        )
    return manifest_id


def validate_deform360_official_hub_causal_window_manifest(
    value: Mapping[str, Any],
) -> str:
    """Validate a completed v1 causal-window manifest and return its identity."""

    return _validate_deform360_official_hub_causal_window_manifest(
        value,
        schema=DEFORM360_CAUSAL_WINDOW_MANIFEST_SCHEMA,
        schedule_recovery_lock_id=None,
    )


def validate_deform360_official_hub_causal_window_manifest_v2(
    value: Mapping[str, Any],
) -> str:
    """Validate a completed v2 causal-window manifest and return its identity."""

    return _validate_deform360_official_hub_causal_window_manifest(
        value,
        schema=DEFORM360_CAUSAL_WINDOW_MANIFEST_SCHEMA_V2,
        schedule_recovery_lock_id=DEFORM360_CAUSAL_SCHEDULE_RECOVERY_LOCK_ID,
    )


def load_deform360_official_hub_causal_window_manifest(
    path: str | Path,
) -> Mapping[str, Any]:
    """Load and validate a strict causal-window manifest."""

    value = load_strict_json_object(path, label="causal-window manifest")
    validate_deform360_official_hub_causal_window_manifest(value)
    return value


def load_deform360_official_hub_causal_window_manifest_v2(
    path: str | Path,
) -> Mapping[str, Any]:
    """Load and validate a strict v2 causal-window manifest."""

    value = load_strict_json_object(path, label="v2 causal-window manifest")
    validate_deform360_official_hub_causal_window_manifest_v2(value)
    return value


def save_deform360_official_hub_causal_window_manifest(
    path: str | Path,
    value: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    """Validate and atomically save one causal-window manifest."""

    validate_deform360_official_hub_causal_window_manifest(value)
    write_atomic_json(value, path, overwrite=overwrite)


def save_deform360_official_hub_causal_window_manifest_v2(
    path: str | Path,
    value: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    """Validate and atomically save one v2 causal-window manifest."""

    validate_deform360_official_hub_causal_window_manifest_v2(value)
    write_atomic_json(value, path, overwrite=overwrite)


__all__ = [
    "DEFORM360_CAMERA_PANEL_POLICY_ID",
    "DEFORM360_CAUSAL_SCHEDULE_RECOVERY_LOCK_ID",
    "DEFORM360_CAUSAL_WINDOW_MANIFEST_SCHEMA",
    "DEFORM360_CAUSAL_WINDOW_MANIFEST_SCHEMA_V2",
    "DEFORM360_CAUSAL_WINDOW_MANIFEST_VERSION",
    "DEFORM360_VISUAL_EXECUTION_LOCK_ID",
    "DEFORM360_VISUAL_PROVIDER_RECOVERY_LOCK_ID",
    "Deform360CustodyError",
    "build_deform360_official_hub_causal_window_manifest",
    "derive_deform360_causal_window_v2",
    "load_deform360_official_hub_causal_window_manifest",
    "load_deform360_official_hub_causal_window_manifest_v2",
    "save_deform360_official_hub_causal_window_manifest",
    "save_deform360_official_hub_causal_window_manifest_v2",
    "validate_deform360_causal_schedule_recovery_lock",
    "validate_deform360_official_hub_causal_window_manifest",
    "validate_deform360_official_hub_causal_window_manifest_v2",
    "validate_deform360_stage1_processing_report",
    "validate_deform360_visual_execution_lock",
]
