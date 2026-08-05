"""Calibration-only RGB, tactile, robot, and action-window preparation."""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bayesian_phystwin.deform360_bias_aware_prospective_staging import (
    STAGING_FRAME_COUNT,
    select_action_only_window,
)

from .contracts import (
    MINIMUM_CAMERA_STREAMS,
    PROCESSING_REVISION,
    PROTOCOL_ID,
    RESULT_SCHEMA,
    canonical_sha256,
    file_sha256,
    load_json,
    require,
    require_clean_revision,
    summary_gate,
    write_json,
)
from .download import verify_download
from .planning import verify_plan


def _hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        require(
            destination.is_file() and not destination.is_symlink(),
            f"staged path is unsafe: {destination}",
        )
        require(
            file_sha256(destination) == file_sha256(source),
            f"staged bytes changed: {destination}",
        )
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _stage_raw_object(
    *,
    row: Mapping[str, Any],
    data_root: Path,
    staged_root: Path,
) -> Path:
    object_id = str(row["object_id"])
    object_root = staged_root / "raw" / object_id
    selected_files = row.get("selected_files")
    require(isinstance(selected_files, list), "selected file list is malformed")
    prefix = f"raw/{object_id}/"
    for record in selected_files:
        require(
            isinstance(record, Mapping),
            "selected file record is malformed",
        )
        relative = str(record["path"])
        require(relative.startswith(prefix), "selected file escaped its object")
        object_relative = relative[len(prefix) :]
        _hardlink_or_copy(
            data_root / relative,
            object_root / object_relative,
        )
    marker = object_root / "selected-original-episode.json"
    marker_value = {
        "object_id": object_id,
        "original_episode_id": int(row["episode_id"]),
        "synthetic_raw_episode_index": 0,
        "selected_file_count": len(selected_files),
    }
    if marker.exists():
        require(
            load_json(marker) == marker_value,
            "staged episode marker changed",
        )
    else:
        write_json(marker, marker_value)
    return object_root


def _metadata_bimanual(path: Path, episode_id: int) -> bool:
    value = load_json(path)
    sequences = value.get("sequences")
    require(isinstance(sequences, Mapping), "metadata sequences are missing")
    episode = sequences.get(str(episode_id))
    require(
        isinstance(episode, Mapping),
        "selected metadata episode is missing",
    )
    bimanual = episode.get("bimanual")
    require(bimanual in {"yes", "no"}, "bimanual metadata changed")
    return bimanual == "yes"


def prepare_one(
    *,
    row: Mapping[str, Any],
    data_root: Path,
    staged_root: Path,
    processed_root: Path,
) -> dict[str, Any]:
    object_id = str(row["object_id"])
    episode_id = int(row["episode_id"])
    result: dict[str, Any] = {
        "object_id": object_id,
        "episode_id": episode_id,
        "stratum": row["stratum"],
        "status": "technical_failure_without_replacement",
        "completed_stage": None,
    }
    try:
        from deform360 import undistort
        from deform360.processing import robot_stage
        from deform360.robot import load_robot_state
        from deform360.tactile import process_tactile_episode

        staged_object = _stage_raw_object(
            row=row,
            data_root=data_root,
            staged_root=staged_root,
        )
        processed_object = processed_root / object_id
        result["completed_stage"] = "download-staging"
        episode_dir = undistort.undistort_episode(
            staged_object,
            processed_object,
            0,
            overwrite=False,
            rebuild_timeline=False,
        )
        result["completed_stage"] = "undistort-and-align"
        tactile_outputs = process_tactile_episode(
            object_dir=staged_object,
            aligned_dir=processed_object,
            episode_index=0,
            overwrite=False,
        )
        result["completed_stage"] = "tactile-align"
        bimanual = _metadata_bimanual(
            staged_object / "metadata.json",
            episode_id,
        )
        robot_path = robot_stage.process_robot_episode(
            processed_object,
            0,
            bimanual=bimanual,
            seed=0,
            overwrite=False,
            plot=False,
        )
        result["completed_stage"] = "robot"
        robot = load_robot_state(robot_path)
        selection = select_action_only_window(robot.actions, robot.openings)
        alignment_path = episode_dir / "alignment.json"
        alignment = load_json(alignment_path)
        cameras = alignment.get("cameras")
        frame_count = alignment.get("frame_count")
        require(
            isinstance(cameras, list)
            and len(cameras) >= MINIMUM_CAMERA_STREAMS,
            "fewer than eight aligned cameras",
        )
        require(
            isinstance(frame_count, int)
            and frame_count >= STAGING_FRAME_COUNT,
            "aligned episode is too short",
        )
        require(len(tactile_outputs) >= 1, "no tactile stream was aligned")
        result.update(
            {
                "status": "source_prepared",
                "completed_stage": "action-window-selection",
                "synthetic_episode_index": 0,
                "bimanual": bimanual,
                "camera_count": len(cameras),
                "cameras": cameras,
                "aligned_frame_count": frame_count,
                "tactile_sensor_count": len(tactile_outputs),
                "tactile_sensors": sorted(tactile_outputs),
                "action_window": selection,
                "outputs_sha256": {
                    "alignment": file_sha256(alignment_path),
                    "undistorted_intrinsics": file_sha256(
                        episode_dir / "undistorted_intrinsics.npy"
                    ),
                    "extrinsics": file_sha256(
                        episode_dir / "extrinsics.npy"
                    ),
                    "robot": file_sha256(robot_path),
                    "tactile": {
                        sensor: file_sha256(path)
                        for sensor, path in sorted(tactile_outputs.items())
                    },
                },
            }
        )
    except Exception as error:  # noqa: BLE001
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def prepare_sources(
    *,
    plan_path: Path,
    download_path: Path,
    protocol_path: Path,
    selection_path: Path,
    provider_path: Path,
    data_root: Path,
    staged_root: Path,
    processed_root: Path,
    processing_repository: Path,
    output_path: Path,
) -> dict[str, Any]:
    plan, confirmations = verify_plan(
        plan_path,
        protocol_path=protocol_path,
        selection_path=selection_path,
        provider_path=provider_path,
    )
    download = verify_download(
        download_path,
        plan_path=plan_path,
        protocol_path=protocol_path,
        selection_path=selection_path,
        provider_path=provider_path,
        data_root=data_root,
    )
    require_clean_revision(
        processing_repository.resolve(),
        PROCESSING_REVISION,
    )
    raw_root = data_root.resolve() / "raw"
    present = {path.name for path in raw_root.iterdir() if path.is_dir()}
    require(
        not present & set(confirmations),
        "confirmation payload exists in calibration root",
    )
    imported = __import__("deform360")
    package_root = Path(imported.__file__).resolve().parents[1]
    require(
        package_root == processing_repository.resolve(),
        "imported Deform360 source changed",
    )

    rows = [
        prepare_one(
            row=row,
            data_root=data_root.resolve(),
            staged_root=staged_root.resolve(),
            processed_root=processed_root.resolve(),
        )
        for row in plan["objects"]
        if row.get("status") == "planned"
    ]
    for row in plan["objects"]:
        if row.get("status") != "planned":
            rows.append(
                {
                    "object_id": row["object_id"],
                    "episode_id": row["episode_id"],
                    "stratum": row["stratum"],
                    "status": "unsupported_without_replacement",
                    "errors": row["errors"],
                }
            )
    rows.sort(key=lambda item: str(item["object_id"]))
    gate = summary_gate(rows, status="source_prepared")
    payload: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "plan_sha256": plan["plan_sha256"],
        "download_sha256": download["download_sha256"],
        "dataset_revision": download["dataset_revision"],
        "processing_revision": PROCESSING_REVISION,
        "objects": rows,
        "gate": gate,
        "next_stage": (
            "stage the sealed 81-frame calibration windows, then run the "
            "frozen MotionCrafter/Prob4D/contact/physical calibration candidates"
        ),
        "information_boundary": {
            "calibration_camera_payloads_opened": True,
            "calibration_tactile_payloads_opened": True,
            "calibration_robot_state_derived": True,
            "calibration_target_metrics_computed": False,
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
            "replacement_allowed": False,
        },
    }
    payload["result_sha256"] = canonical_sha256(
        payload,
        digest_key="result_sha256",
    )
    write_json(output_path, payload)
    return payload
