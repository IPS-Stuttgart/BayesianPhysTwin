#!/usr/bin/env python3
"""Stage one action-only V14 source window with aligned tactile support."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.deform360_causal_response_direct_depth_cohort import (
    validate_v14_staging_queue,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_preflight import (
    deform360_v14_case_hash,
)
from bayesian_phystwin.deform360_causal_response_preflight import (
    REGISTERED_CAMERA_IDS,
    deform360_object_hash,
)
from bayesian_phystwin.deform360_exact_video_cadence import (
    decoded_frame_count,
    trim_video_exact_30hz,
)
from bayesian_phystwin.deform360_fresh_source_window import (
    PREDICTION_FRAME_COUNT,
    RAW_FRAME_COUNT,
    select_fresh_source_window,
)
from bayesian_phystwin.deform360_object_exclusion import file_sha256

PREPARATION_KIND = "Deform360CausalDirectDepthSourcePreparationV14"
PREPARATION_CONTRACT = "deform360-causal-response-direct-depth-preparation-v14"
RESULT_KIND = "Deform360CausalDirectDepthWindowStageV14"
RESULT_CONTRACT = "deform360-causal-response-direct-depth-window-v14"
DEFORM360_REVISION = "0fe36f0b7a7a917ba62b5f8cee707299a9a4a317"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_sha256(payload: Mapping[str, Any], *, namespace: bytes) -> str:
    canonical = dict(payload)
    canonical.pop("artifact_sha256", None)
    return hashlib.sha256(
        namespace
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON artifact: {path}") from error
    _require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


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


def _resolve_required_executable(path: Path, *, name: str) -> Path:
    expanded = path.expanduser()
    _require(expanded.is_absolute(), f"{name} path must be absolute")
    resolved = expanded.resolve()
    _require(resolved.is_file(), f"{name} executable is unavailable")
    _require(os.access(resolved, os.X_OK), f"{name} path is not executable")
    try:
        subprocess.run(
            [str(resolved), "-version"],
            check=True,
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f"{name} executable cannot run") from error
    return resolved


def _trim_timestamps(
    source: Path,
    destination: Path,
    *,
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
    *,
    cameras: tuple[str, ...],
) -> None:
    values = np.load(source, allow_pickle=True).item()
    _require(set(cameras).issubset(values), f"calibration lacks cameras: {source}")
    np.save(destination, {camera: values[camera] for camera in cameras})


def _validate_preparation(
    path: Path,
    *,
    protocol: Mapping[str, Any],
    queue: Mapping[str, Any],
    queue_rank: int,
    object_hash: str,
    case_hash: str,
    source_episode: Path,
) -> dict[str, Any]:
    payload = _read_json(path)
    _require(
        payload.get("artifact_kind") == PREPARATION_KIND
        and payload.get("contract") == PREPARATION_CONTRACT
        and payload.get("status") == "prepared"
        and payload.get("protocol_id") == protocol["protocol_id"]
        and payload.get("protocol_config_sha256") == protocol["config_sha256"]
        and payload.get("queue_sha256") == queue["queue_sha256"]
        and payload.get("queue_rank") == queue_rank
        and payload.get("object_hash") == object_hash
        and payload.get("case_hash") == case_hash
        and payload.get("deform360_revision") == DEFORM360_REVISION
        and payload.get("artifact_sha256")
        == _canonical_sha256(
            payload,
            namespace=(
                b"deform360-causal-response-direct-depth-preparation-v14\0"
            ),
        ),
        "V14 source preparation binding changed",
    )
    outputs = payload.get("outputs_sha256")
    _require(isinstance(outputs, Mapping), "V14 preparation outputs are missing")
    fixed = {
        "alignment": source_episode / "alignment.json",
        "undistorted_intrinsics": source_episode / "undistorted_intrinsics.npy",
        "extrinsics": source_episode / "extrinsics.npy",
        "robot": source_episode / "robot" / "robot.npz",
        "robot_metadata": source_episode / "robot" / "robot.meta.json",
    }
    _require(
        all(
            file.is_file() and file_sha256(file) == outputs.get(role)
            for role, file in fixed.items()
        ),
        "V14 preparation fixed outputs changed",
    )
    camera_outputs = outputs.get("camera_metadata")
    tactile_outputs = outputs.get("tactile")
    _require(
        isinstance(camera_outputs, Mapping)
        and set(camera_outputs) == set(REGISTERED_CAMERA_IDS)
        and all(
            file_sha256(source_episode / camera / "metadata.json")
            == camera_outputs[camera]
            for camera in REGISTERED_CAMERA_IDS
        ),
        "V14 preparation camera outputs changed",
    )
    _require(
        isinstance(tactile_outputs, Mapping)
        and bool(tactile_outputs)
        and all(
            file_sha256(source_episode / sensor / "synced_tactile.npy")
            == values["array"]
            and file_sha256(source_episode / sensor / "alignment.json")
            == values["alignment"]
            and file_sha256(source_episode / sensor / "metadata.json")
            == values["metadata"]
            for sensor, values in tactile_outputs.items()
        ),
        "V14 preparation tactile outputs changed",
    )
    return payload


def _write_result(path: Path, payload: dict[str, Any]) -> None:
    _require(not path.exists(), f"refusing to replace V14 stage result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["artifact_sha256"] = _canonical_sha256(
        payload,
        namespace=b"deform360-causal-response-direct-depth-window-v14\0",
    )
    temporary = path.with_name(f".{path.name}.tmp")
    _require(not temporary.exists(), f"temporary V14 stage result exists: {temporary}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--aligned-root", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--candidate-rank", type=int, required=True)
    parser.add_argument("--deform360-repo", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    ffmpeg = _resolve_required_executable(args.ffmpeg, name="ffmpeg")
    repository = args.repo.resolve()
    code_revision = _require_clean_repository(repository)
    protocol_path = args.protocol.resolve()
    protocol = _read_json(protocol_path)
    _require(
        protocol.get("protocol_id")
        == "deform360-causal-response-direct-depth-v14-source",
        "V14 protocol ID changed",
    )
    queue_path = args.queue.resolve()
    queue = validate_v14_staging_queue(queue_path)
    rank = args.candidate_rank
    _require(
        1 <= rank <= len(queue["candidates"]),
        "candidate rank is outside the frozen V14 queue",
    )
    candidate = queue["candidates"][rank - 1]
    object_id = str(candidate["object_id"])
    episode_id = int(candidate["episode_id"])
    object_hash = deform360_object_hash(object_id)
    case_hash = deform360_v14_case_hash(object_id, episode_id)
    source_episode = (
        args.aligned_root.resolve()
        / object_id
        / f"episode_{episode_id:04d}"
    )
    preparation_path = args.preparation.resolve()
    preparation = _validate_preparation(
        preparation_path,
        protocol=protocol,
        queue=queue,
        queue_rank=rank,
        object_hash=object_hash,
        case_hash=case_hash,
        source_episode=source_episode,
    )
    deform360_repository = args.deform360_repo.resolve()
    _require(
        _git_revision(deform360_repository) == DEFORM360_REVISION,
        "Deform360 revision changed",
    )
    sys.path.insert(0, str(deform360_repository))
    from deform360.robot import RobotState, load_robot_state, save_robot_state

    robot_path = source_episode / "robot" / "robot.npz"
    robot = load_robot_state(robot_path)
    selection = select_fresh_source_window(robot.actions, robot.openings)
    start, stop = selection["selected_raw_frame_range_half_open"]
    _require(
        stop - start == RAW_FRAME_COUNT,
        "V14 selected source window length changed",
    )
    output_episode = (
        args.stage_root.resolve()
        / object_id
        / f"episode_{episode_id:04d}"
    )
    result_path = args.result.resolve()
    _require(
        not output_episode.exists() and not result_path.exists(),
        "V14 stage output or result already exists",
    )
    scratch = output_episode.with_name(
        f".{output_episode.name}.incomplete-{os.getpid()}"
    )
    _require(not scratch.exists(), f"V14 stage scratch exists: {scratch}")
    scratch.mkdir(parents=True)

    base: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": RESULT_KIND,
        "contract": RESULT_CONTRACT,
        "protocol_id": protocol["protocol_id"],
        "protocol_config_sha256": protocol["config_sha256"],
        "queue_sha256": queue["queue_sha256"],
        "queue_file_sha256": file_sha256(queue_path),
        "queue_rank": rank,
        "object_hash": object_hash,
        "case_hash": case_hash,
        "category": candidate["category"],
        "repository_revision": code_revision,
        "deform360_revision": DEFORM360_REVISION,
        "preparation_artifact_sha256": preparation["artifact_sha256"],
        "preparation_file_sha256": file_sha256(preparation_path),
        "ffmpeg_path": str(ffmpeg),
        "ffmpeg_sha256": file_sha256(ffmpeg),
        "window_selection": selection,
    }
    try:
        _subset_calibration(
            source_episode / "undistorted_intrinsics.npy",
            scratch / "undistorted_intrinsics.npy",
            cameras=REGISTERED_CAMERA_IDS,
        )
        _subset_calibration(
            source_episode / "extrinsics.npy",
            scratch / "extrinsics.npy",
            cameras=REGISTERED_CAMERA_IDS,
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
        for camera in REGISTERED_CAMERA_IDS:
            source_camera = source_episode / camera
            output_camera = scratch / camera
            output_camera.mkdir()
            output_video = output_camera / "undistorted.mp4"
            trim_video_exact_30hz(
                ffmpeg,
                source_camera / "undistorted.mp4",
                output_video,
                start,
                RAW_FRAME_COUNT,
            )
            output_timestamps = output_camera / "aligned_timestamps.txt"
            _trim_timestamps(
                source_camera / "aligned_timestamps.txt",
                output_timestamps,
                start=start,
                count=RAW_FRAME_COUNT,
            )
            metadata_path = source_camera / "metadata.json"
            if metadata_path.is_file():
                shutil.copy2(metadata_path, output_camera / "metadata.json")
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
        tactile_rows: list[dict[str, Any]] = []
        for sensor in sorted(preparation["outputs_sha256"]["tactile"]):
            source_array = source_episode / sensor / "synced_tactile.npy"
            values = np.load(source_array, allow_pickle=False)
            _require(
                values.ndim == 3 and values.shape[0] >= stop,
                f"aligned tactile stream is too short: {sensor}",
            )
            output_sensor = scratch / sensor
            output_sensor.mkdir()
            output_array = output_sensor / "synced_tactile.npy"
            np.save(output_array, values[start:stop])
            for name in ("metadata.json", "alignment.json"):
                shutil.copy2(source_episode / sensor / name, output_sensor / name)
            tactile_rows.append(
                {
                    "sensor": sensor,
                    "frame_count": int(values[start:stop].shape[0]),
                    "array_sha256": file_sha256(output_array),
                    "source_array_sha256": file_sha256(source_array),
                }
            )
        _require(
            all(row["decoded_frame_count"] == RAW_FRAME_COUNT for row in camera_rows)
            and all(row["frame_count"] == RAW_FRAME_COUNT for row in tactile_rows),
            "V14 staged source streams have inconsistent lengths",
        )
        payload = {
            **base,
            "status": "staged",
            "raw_frame_count": RAW_FRAME_COUNT,
            "prediction_frame_count": PREDICTION_FRAME_COUNT,
            "selected_raw_frame_range_half_open": [start, stop],
            "camera_records": camera_rows,
            "tactile_records": tactile_rows,
            "outputs_sha256": {
                "intrinsics": file_sha256(
                    scratch / "undistorted_intrinsics.npy"
                ),
                "extrinsics": file_sha256(scratch / "extrinsics.npy"),
                "robot": file_sha256(staged_robot_path),
            },
            "information_boundary": {
                "window_selection_used_full_known_robot_action_only": True,
                "object_rgb_read_for_window_selection": False,
                "tactile_read_for_window_selection": False,
                "object_rgb_materialized_after_selection": True,
                "tactile_materialized_after_selection": True,
                "object_mask_or_geometry_created": False,
                "future_identity_or_metric_read": False,
                "target_object_or_outcome_read": False,
                "held_v8_access": False,
            },
        }
        output_episode.parent.mkdir(parents=True, exist_ok=True)
        scratch.rename(output_episode)
    except Exception as error:
        print(
            f"V14 source staging failed with {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        shutil.rmtree(scratch, ignore_errors=True)
        payload = {
            **base,
            "status": "technical_preflight_failure",
            "failure": {
                "type": type(error).__name__,
                "reason": "action-only-window-staging-failed",
            },
            "information_boundary": {
                "window_selection_used_full_known_robot_action_only": True,
                "object_rgb_read_for_window_selection": False,
                "tactile_read_for_window_selection": False,
                "object_mask_or_geometry_created": False,
                "future_identity_or_metric_read": False,
                "target_object_or_outcome_read": False,
                "held_v8_access": False,
            },
        }
    _write_result(result_path, payload)
    print(payload["status"])
    print(payload["artifact_sha256"])
    return 0 if payload["status"] == "staged" else 2


if __name__ == "__main__":
    raise SystemExit(main())
