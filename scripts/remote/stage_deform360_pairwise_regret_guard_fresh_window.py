#!/usr/bin/env python3
"""Select and materialize one locked fresh Deform360 source window."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from deform360.robot import RobotState, load_robot_state, save_robot_state

from bayesian_phystwin.deform360_exact_video_cadence import (
    decoded_frame_count,
    trim_video_exact_30hz,
)
from bayesian_phystwin.deform360_pairwise_regret_guard_fresh_processing import (
    PREPARATION_KIND,
    RAW_FRAME_COUNT,
    WINDOW_SELECTION_KIND,
    WINDOW_STAGE_KIND,
    fresh_processing_case,
    seal_case_artifact,
    select_fresh_source_window,
    validate_case_artifact,
    validate_fresh_processing_sources,
    write_json_artifact,
)
from bayesian_phystwin.deform360_pairwise_regret_guard_fresh_protocol import (
    file_sha256,
)

PREPARATION_FILENAME = "fresh_pairwise_preparation.json"
SELECTION_FILENAME = "fresh_pairwise_window_selection.json"
STAGE_FILENAME = "fresh_pairwise_window_stage.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_repository(repository: Path, expected: str) -> str:
    revision = _git_revision(repository)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _require(not status.strip(), f"repository has uncommitted files: {repository}")
    _require(revision == expected, "processing implementation revision changed")
    return revision


def _trim_timestamps(source: Path, destination: Path, start: int) -> None:
    lines = source.read_text(encoding="utf-8").splitlines()
    selected = lines[start : start + RAW_FRAME_COUNT]
    _require(len(selected) == RAW_FRAME_COUNT, "timestamp stream is too short")
    destination.write_text("\n".join(selected) + "\n", encoding="utf-8")


def _subset_calibration(
    source: Path, destination: Path, cameras: tuple[str, ...]
) -> None:
    values = np.load(source, allow_pickle=True).item()
    _require(set(cameras) <= set(values), f"calibration lacks cameras: {source}")
    np.save(destination, {camera: values[camera] for camera in cameras})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--technical-lock", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--aligned-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("ffmpeg"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    protocol, lock, _, _ = validate_fresh_processing_sources(
        args.protocol.resolve(),
        args.technical_lock.resolve(),
        args.source_plan.resolve(),
        args.download_manifest.resolve(),
    )
    case = fresh_processing_case(lock, args.object_id, args.episode_id)
    code_revision = _require_clean_repository(
        args.repo.resolve(), str(protocol["implementation_commit"])
    )
    source_episode = (
        args.aligned_root.resolve() / args.object_id / f"episode_{args.episode_id:04d}"
    )
    preparation_path = source_episode / PREPARATION_FILENAME
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    validate_case_artifact(
        preparation,
        artifact_kind=PREPARATION_KIND,
        protocol=protocol,
        case=case,
    )
    _require(preparation.get("status") == "prepared", "source is not prepared")
    robot_path = source_episode / "robot" / "robot.npz"
    _require(
        file_sha256(robot_path) == preparation["outputs_sha256"]["robot"],
        "prepared robot state changed",
    )
    robot = load_robot_state(robot_path)
    selection = select_fresh_source_window(robot.actions, robot.openings)
    start, stop = selection["selected_raw_frame_range_half_open"]
    _require(stop - start == RAW_FRAME_COUNT, "selected frame count changed")
    selection_artifact = seal_case_artifact(
        WINDOW_SELECTION_KIND,
        protocol=protocol,
        case=case,
        payload={
            "code_revision": code_revision,
            "selection": selection,
            "source_robot_sha256": file_sha256(robot_path),
            "source_preparation_sha256": file_sha256(preparation_path),
            "information_boundary": {
                "known_future_action_read": True,
                "object_geometry_read": False,
                "object_tracks_read": False,
                "object_response_read": False,
                "tactile_read": False,
                "target_metric_read": False,
                "selection_sealed_before_object_processing": True,
                "held_v8_runtime_or_target_artifact_access": False,
            },
        },
    )
    cameras = tuple(protocol["dataset"]["camera_panel"])
    _require(
        set(cameras) <= set(preparation["cameras"]),
        "prepared source lacks frozen cameras",
    )
    destination = (
        args.output_root.resolve() / args.object_id / f"episode_{args.episode_id:04d}"
    )
    _require(not destination.exists(), "fresh source window already exists")
    scratch = destination.with_name(f".{destination.name}.incomplete-{os.getpid()}")
    _require(not scratch.exists(), "fresh source window scratch already exists")
    scratch.mkdir(parents=True)
    try:
        _subset_calibration(
            source_episode / "undistorted_intrinsics.npy",
            scratch / "undistorted_intrinsics.npy",
            cameras,
        )
        _subset_calibration(
            source_episode / "extrinsics.npy",
            scratch / "extrinsics.npy",
            cameras,
        )
        staged_robot = RobotState(
            actions=robot.actions[start:stop],
            T_worlds=robot.T_worlds[start:stop],
            openings=robot.openings[start:stop],
            bimanual=robot.bimanual,
        )
        staged_robot_path = save_robot_state(
            scratch / "robot" / "robot.npz", staged_robot
        )
        camera_rows: list[dict[str, Any]] = []
        for camera in cameras:
            source_camera = source_episode / camera
            output_camera = scratch / camera
            output_camera.mkdir()
            output_video = output_camera / "undistorted.mp4"
            trim_video_exact_30hz(
                args.ffmpeg,
                source_camera / "undistorted.mp4",
                output_video,
                start,
                RAW_FRAME_COUNT,
            )
            output_timestamps = output_camera / "aligned_timestamps.txt"
            _trim_timestamps(
                source_camera / "aligned_timestamps.txt", output_timestamps, start
            )
            metadata = source_camera / "metadata.json"
            if metadata.is_file():
                shutil.copy2(metadata, output_camera / "metadata.json")
            camera_rows.append(
                {
                    "camera": camera,
                    "decoded_frame_count": decoded_frame_count(output_video),
                    "video_sha256": file_sha256(output_video),
                    "timestamps_sha256": file_sha256(output_timestamps),
                    "metadata_sha256": (
                        file_sha256(output_camera / "metadata.json")
                        if (output_camera / "metadata.json").is_file()
                        else None
                    ),
                }
            )
        selection_path = scratch / SELECTION_FILENAME
        write_json_artifact(selection_artifact, selection_path)
        stage = seal_case_artifact(
            WINDOW_STAGE_KIND,
            protocol=protocol,
            case=case,
            payload={
                "status": "staged",
                "code_revision": code_revision,
                "source_preparation_sha256": file_sha256(preparation_path),
                "selection_result_sha256": selection_artifact["result_sha256"],
                "selection_file_sha256": file_sha256(selection_path),
                "selected_raw_frame_range_half_open": [start, stop],
                "staged_frame_count": RAW_FRAME_COUNT,
                "camera_count": len(camera_rows),
                "camera_records": camera_rows,
                "outputs_sha256": {
                    "intrinsics": file_sha256(scratch / "undistorted_intrinsics.npy"),
                    "extrinsics": file_sha256(scratch / "extrinsics.npy"),
                    "robot": file_sha256(staged_robot_path),
                },
                "information_boundary": {
                    "known_future_action_read": True,
                    "rgb_materialized_after_selection_seal": True,
                    "object_geometry_read": False,
                    "object_tracks_read": False,
                    "object_response_used_for_window_selection": False,
                    "tactile_read": False,
                    "target_metric_read": False,
                    "held_v8_runtime_or_target_artifact_access": False,
                },
            },
        )
        write_json_artifact(stage, scratch / STAGE_FILENAME)
        destination.parent.mkdir(parents=True, exist_ok=True)
        scratch.rename(destination)
    except BaseException:
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    print(json.dumps(stage, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
