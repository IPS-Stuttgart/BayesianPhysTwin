#!/usr/bin/env python3
"""Materialize one exact queue-bound Deform360 source window."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping

import numpy as np

from bayesian_phystwin.deform360_exact_video_cadence import (
    decoded_frame_count,
    trim_video_exact_30hz,
)
from bayesian_phystwin.deform360_fresh_source_window import (
    FROZEN_CAMERA_PANEL,
    RAW_FRAME_COUNT,
    canonical_sha256,
    file_sha256,
    fresh_source_case,
    seal_fresh_source_window_selection,
    select_fresh_source_window,
    validate_window_sources,
)
from deform360.robot import RobotState, load_robot_state, save_robot_state


SOURCE_PREPARATION_FILENAME = "fresh_source_preparation_manifest.json"
WINDOW_SELECTION_FILENAME = "fresh_source_window_selection.json"
WINDOW_STAGE_FILENAME = "fresh_source_window_stage.json"


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


def _require_clean_repository(repository: Path) -> str:
    revision = _git_revision(repository)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _require(not status.strip(), f"repository has uncommitted files: {repository}")
    return revision


def _trim_timestamps(
    source: Path,
    destination: Path,
    start: int,
    count: int,
) -> None:
    lines = source.read_text(encoding="utf-8").splitlines()
    selected = lines[start : start + count]
    _require(len(selected) == count, f"timestamp stream is too short: {source}")
    destination.write_text("\n".join(selected) + "\n", encoding="utf-8")


def _subset_calibration(
    source: Path,
    destination: Path,
    cameras: tuple[str, ...],
) -> None:
    values = np.load(source, allow_pickle=True).item()
    _require(set(cameras) <= set(values), f"calibration lacks cameras: {source}")
    np.save(destination, {camera: values[camera] for camera in cameras})


def _validate_source_preparation(
    source_episode: Path,
    *,
    protocol: Mapping[str, Any],
    case: Mapping[str, Any],
    code_revision: str,
) -> tuple[dict[str, Any], Path]:
    path = source_episode / SOURCE_PREPARATION_FILENAME
    _require(path.is_file(), "fresh source preparation manifest is missing")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _require(
        manifest.get("artifact_kind") == "Deform360FreshSourcePreparation"
        and manifest.get("protocol_config_sha256") == protocol["config_sha256"]
        and manifest.get("code_revision") == code_revision
        and manifest.get("result_sha256")
        == canonical_sha256(manifest, digest_key="result_sha256")
        and all(manifest.get(key) == value for key, value in case.items()),
        "fresh source preparation is incompatible",
    )
    outputs = manifest.get("outputs_sha256")
    _require(isinstance(outputs, Mapping), "source output hashes are missing")
    fixed = {
        "undistorted_intrinsics": source_episode / "undistorted_intrinsics.npy",
        "extrinsics": source_episode / "extrinsics.npy",
        "robot": source_episode / "robot" / "robot.npz",
    }
    _require(
        all(
            path.is_file() and file_sha256(path) == outputs.get(name)
            for name, path in fixed.items()
        ),
        "fresh source preparation outputs changed",
    )
    return manifest, path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--source-aligned-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("ffmpeg"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    protocol, queue, _ = validate_window_sources(
        args.protocol.resolve(),
        args.queue.resolve(),
        args.download_manifest.resolve(),
    )
    case = fresh_source_case(queue, args.object_id, args.episode_id)
    code_revision = _require_clean_repository(args.repo.resolve())
    source_episode = (
        args.source_aligned_root.resolve()
        / args.object_id
        / f"episode_{args.episode_id:04d}"
    )
    _require(source_episode.is_dir(), "aligned source episode is missing")
    source_preparation, source_preparation_path = _validate_source_preparation(
        source_episode,
        protocol=protocol,
        case=case,
        code_revision=code_revision,
    )
    robot_path = source_episode / "robot" / "robot.npz"
    robot = load_robot_state(robot_path)
    selection = select_fresh_source_window(robot.actions, robot.openings)
    start, stop = selection["selected_raw_frame_range_half_open"]
    _require(stop - start == RAW_FRAME_COUNT, "selected frame count changed")
    selection_seal = seal_fresh_source_window_selection(
        protocol=protocol,
        case=case,
        selection=selection,
        source_robot_sha256=file_sha256(robot_path),
        source_preparation_sha256=file_sha256(source_preparation_path),
        code_revision=code_revision,
    )

    source_cameras = set(source_preparation["cameras"])
    missing = sorted(set(FROZEN_CAMERA_PANEL) - source_cameras)
    _require(not missing, f"frozen source cameras are missing: {missing}")
    for camera in FROZEN_CAMERA_PANEL:
        _require(
            (source_episode / camera / "undistorted.mp4").is_file()
            and (source_episode / camera / "aligned_timestamps.txt").is_file(),
            f"aligned source camera is incomplete: {camera}",
        )

    destination = (
        args.output_root.resolve()
        / args.object_id
        / f"episode_{args.episode_id:04d}"
    )
    _require(not destination.exists(), f"fresh source window exists: {destination}")
    scratch = destination.with_name(f".{destination.name}.incomplete-{os.getpid()}")
    _require(not scratch.exists(), f"fresh source scratch exists: {scratch}")
    scratch.mkdir(parents=True)
    try:
        _subset_calibration(
            source_episode / "undistorted_intrinsics.npy",
            scratch / "undistorted_intrinsics.npy",
            FROZEN_CAMERA_PANEL,
        )
        _subset_calibration(
            source_episode / "extrinsics.npy",
            scratch / "extrinsics.npy",
            FROZEN_CAMERA_PANEL,
        )
        staged_robot = RobotState(
            actions=robot.actions[start:stop],
            T_worlds=robot.T_worlds[start:stop],
            openings=robot.openings[start:stop],
            bimanual=robot.bimanual,
        )
        staged_robot_path = save_robot_state(
            scratch / "robot" / "robot.npz",
            staged_robot,
        )
        camera_rows: list[dict[str, Any]] = []
        for camera in FROZEN_CAMERA_PANEL:
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
                source_camera / "aligned_timestamps.txt",
                output_timestamps,
                start,
                RAW_FRAME_COUNT,
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
        selection_path = scratch / WINDOW_SELECTION_FILENAME
        selection_path.write_text(
            json.dumps(
                selection_seal,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        stage: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": "Deform360FreshSourceWindowStage",
            "protocol_id": protocol["protocol_id"],
            "protocol_config_sha256": protocol["config_sha256"],
            **case,
            "code_revision": code_revision,
            "source_preparation_sha256": file_sha256(source_preparation_path),
            "selection_result_sha256": selection_seal["result_sha256"],
            "selection_file_sha256": file_sha256(selection_path),
            "selected_raw_frame_range_half_open": [start, stop],
            "staged_frame_count": RAW_FRAME_COUNT,
            "camera_count": len(camera_rows),
            "camera_records": camera_rows,
            "outputs_sha256": {
                "intrinsics": file_sha256(
                    scratch / "undistorted_intrinsics.npy"
                ),
                "extrinsics": file_sha256(scratch / "extrinsics.npy"),
                "robot": file_sha256(staged_robot_path),
            },
            "information_boundary": {
                "known_future_action_read": True,
                "object_rgb_materialized_after_selection_seal_built": True,
                "object_geometry_read": False,
                "object_tracks_read": False,
                "object_response_used_for_window_selection": False,
                "tactile_read": False,
                "target_metric_read": False,
            },
        }
        stage["result_sha256"] = canonical_sha256(
            stage, digest_key="result_sha256"
        )
        (scratch / WINDOW_STAGE_FILENAME).write_text(
            json.dumps(stage, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        scratch.rename(destination)
    except BaseException:
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    print(json.dumps(stage, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
