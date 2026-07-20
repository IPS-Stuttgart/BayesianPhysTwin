#!/usr/bin/env python3
"""Stage RGB prefixes and the separately allowed known action trajectory."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import h5py
import numpy as np

from bayesian_phystwin.deform360_bias_aware_prospective_artifacts import (
    canonical_sha256,
    file_sha256,
    prospective_case_record,
)
from bayesian_phystwin.deform360_bias_aware_prospective_protocol import (
    PROTOCOL_ID,
    load_bias_aware_prospective_protocol,
)
from bayesian_phystwin.deform360_bias_aware_prospective_staging import (
    PREFIX_FRAME_COUNT,
    PREDICTION_FRAME_COUNT,
    STAGING_FRAME_COUNT,
    select_action_only_window,
)
from deform360.robot import RobotState, load_robot_state, save_robot_state


GENERIC_SELECTOR_SHA256 = (
    "79b161fa66489f75b5b078c7ae409387feed74c51a38b86e89800d0aa578b1df"
)
SAM2_REPOSITORY_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SAM2_CHECKPOINT_SHA256 = (
    "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"
)


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


def _load_selector_class(source: Path):
    _require(source.is_file(), "generic SAM2 selector is missing")
    _require(file_sha256(source) == GENERIC_SELECTOR_SHA256, "selector changed")
    name = "causal4d_public.deform360_object_sam2_locked"
    spec = importlib.util.spec_from_file_location(name, source)
    _require(spec is not None and spec.loader is not None, "cannot load selector")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.DeformableObjectSam2VideoPredictor


def _read_rgb_frame(path: Path, frame: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, bgr = capture.read()
        observed = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    finally:
        capture.release()
    _require(ok and observed == frame, f"cannot read exact frame {frame}: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _trim_video(source: Path, destination: Path, start: int, count: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            f"select='between(n,{start},{start + count - 1})',setpts=N/FRAME_RATE/TB",
            "-frames:v",
            str(count),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "12",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            str(destination),
        ],
        check=True,
    )


def _trim_timestamps(source: Path, destination: Path, start: int, count: int) -> None:
    selected = source.read_text(encoding="utf-8").splitlines()[start : start + count]
    _require(len(selected) == count, f"timestamp stream is too short: {source}")
    destination.write_text("\n".join(selected) + "\n", encoding="utf-8")


def _write_mask(path: Path, mask: np.ndarray) -> None:
    with h5py.File(path, "w") as stream:
        stream.create_dataset(
            "data",
            data=np.asarray(mask, dtype=np.uint8)[None],
            dtype=np.uint8,
            compression="gzip",
            compression_opts=4,
        )


def _save_calibration(source: Path, destination: Path, cameras: list[str]) -> None:
    values = np.load(source, allow_pickle=True).item()
    _require(set(cameras) <= set(values), f"calibration lacks cameras: {source}")
    np.save(destination, {camera: values[camera] for camera in cameras})


def _slice_robot(robot: RobotState, start: int, stop: int) -> RobotState:
    return RobotState(
        actions=robot.actions[start:stop],
        T_worlds=robot.T_worlds[start:stop],
        openings=robot.openings[start:stop],
        bimanual=robot.bimanual,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-aligned-root", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--generic-selector-source", type=Path, required=True)
    parser.add_argument("--sam2-repository", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    protocol = load_bias_aware_prospective_protocol(args.protocol)
    record = prospective_case_record(
        args.protocol, object_id=args.object_id, episode_id=args.episode_id
    )
    source_episode = (
        args.source_aligned_root.resolve()
        / args.object_id
        / f"episode_{args.episode_id:04d}"
    )
    _require(source_episode.is_dir(), "aligned source episode is missing")
    robot_path = source_episode / "robot" / "robot.npz"
    robot = load_robot_state(robot_path)
    selection = select_action_only_window(robot.actions, robot.openings)
    start, raw_stop = selection["selected_raw_frame_range_half_open"]
    _require(raw_stop - start == STAGING_FRAME_COUNT, "staging window changed")
    intrinsics = np.load(
        source_episode / "undistorted_intrinsics.npy", allow_pickle=True
    ).item()
    extrinsics = np.load(source_episode / "extrinsics.npy", allow_pickle=True).item()
    candidates = sorted(
        camera
        for camera in set(intrinsics) & set(extrinsics)
        if (source_episode / camera / "undistorted.mp4").is_file()
    )
    _require(len(candidates) >= 8, "fewer than eight calibrated cameras")
    sam2_repository = args.sam2_repository.resolve()
    checkpoint = args.sam2_checkpoint.resolve()
    _require(_git_revision(sam2_repository) == SAM2_REPOSITORY_REVISION, "SAM2 changed")
    _require(file_sha256(checkpoint) == SAM2_CHECKPOINT_SHA256, "SAM2 checkpoint changed")
    selector_class = _load_selector_class(args.generic_selector_source.resolve())
    selector = selector_class(sam2_repository, checkpoint, device=args.device)
    masks: dict[str, np.ndarray] = {}
    diagnostics: dict[str, object] = {}
    try:
        for camera in candidates:
            rgb = _read_rgb_frame(source_episode / camera / "undistorted.mp4", start)
            try:
                mask, diagnostic = selector.select_initial_mask_from_rgb(
                    rgb, camera=camera, video_name=f"source-frame-{start:06d}"
                )
            except ValueError as error:
                diagnostics[camera] = {
                    "accepted": False,
                    "error": f"{type(error).__name__}: {error}",
                }
                continue
            masks[camera] = np.asarray(mask, dtype=bool)
            diagnostics[camera] = {"accepted": True, "selection": diagnostic}
    finally:
        selector.close()
    cameras = sorted(masks)
    _require(len(cameras) >= 8, "fewer than eight cameras passed masking")

    destination = args.output_root.resolve() / str(record["case"])
    _require(not destination.exists(), "prediction prefix already exists")
    scratch = destination.with_name(f".{destination.name}.incomplete-{os.getpid()}")
    _require(not scratch.exists(), "prediction scratch path exists")
    prefix_episode = scratch / "prefix" / "episode_0000"
    frame_zero_episode = scratch / "frame-zero" / "episode_0000"
    prefix_episode.mkdir(parents=True)
    frame_zero_episode.mkdir(parents=True)
    try:
        for root in (prefix_episode, frame_zero_episode):
            _save_calibration(
                source_episode / "undistorted_intrinsics.npy",
                root / "undistorted_intrinsics.npy",
                cameras,
            )
            _save_calibration(
                source_episode / "extrinsics.npy", root / "extrinsics.npy", cameras
            )
        prefix_robot = _slice_robot(robot, start, start + PREFIX_FRAME_COUNT)
        frame_zero_robot = _slice_robot(robot, start, start + 1)
        known_action_robot = _slice_robot(robot, start, start + PREDICTION_FRAME_COUNT)
        prefix_robot_path = prefix_episode / "robot" / "robot.npz"
        frame_zero_robot_path = frame_zero_episode / "robot" / "robot.npz"
        known_action_path = scratch / "known-action" / "robot.npz"
        save_robot_state(prefix_robot_path, prefix_robot)
        save_robot_state(frame_zero_robot_path, frame_zero_robot)
        save_robot_state(known_action_path, known_action_robot)
        camera_records = []
        for camera in cameras:
            source_camera = source_episode / camera
            prefix_camera = prefix_episode / camera
            frame_zero_camera = frame_zero_episode / camera
            prefix_camera.mkdir()
            frame_zero_camera.mkdir()
            _trim_video(
                source_camera / "undistorted.mp4",
                prefix_camera / "undistorted.mp4",
                start,
                PREFIX_FRAME_COUNT,
            )
            _trim_timestamps(
                source_camera / "aligned_timestamps.txt",
                prefix_camera / "aligned_timestamps.txt",
                start,
                PREFIX_FRAME_COUNT,
            )
            _trim_video(
                source_camera / "undistorted.mp4",
                frame_zero_camera / "undistorted.mp4",
                start,
                1,
            )
            _trim_timestamps(
                source_camera / "aligned_timestamps.txt",
                frame_zero_camera / "aligned_timestamps.txt",
                start,
                1,
            )
            for root in (prefix_camera, frame_zero_camera):
                _write_mask(root / "mask_refined.h5", masks[camera])
            camera_records.append(
                {
                    "camera": camera,
                    "prefix_video_sha256": file_sha256(
                        prefix_camera / "undistorted.mp4"
                    ),
                    "frame_zero_video_sha256": file_sha256(
                        frame_zero_camera / "undistorted.mp4"
                    ),
                    "frame_zero_mask_sha256": file_sha256(
                        prefix_camera / "mask_refined.h5"
                    ),
                }
            )
        manifest: dict[str, object] = {
            "schema_version": 1,
            "artifact_kind": "Deform360BiasAwarePredictionPrefix",
            "protocol_id": PROTOCOL_ID,
            "protocol_config_sha256": protocol["config_sha256"],
            **record,
            "action_window": selection,
            "staged_prefix_frame_count": PREFIX_FRAME_COUNT,
            "staged_frame_zero_frame_count": 1,
            "known_action_frame_count": PREDICTION_FRAME_COUNT,
            "camera_count": len(cameras),
            "camera_records": camera_records,
            "mask_diagnostics": diagnostics,
            "staged_robot_sha256": {
                "prefix": file_sha256(prefix_robot_path),
                "frame_zero": file_sha256(frame_zero_robot_path),
                "known_action": file_sha256(known_action_path),
            },
            "inputs_sha256": {
                "protocol": file_sha256(args.protocol.resolve()),
                "source_robot": file_sha256(robot_path),
                "source_intrinsics": file_sha256(
                    source_episode / "undistorted_intrinsics.npy"
                ),
                "source_extrinsics": file_sha256(source_episode / "extrinsics.npy"),
                "generic_selector_source": file_sha256(
                    args.generic_selector_source.resolve()
                ),
                "sam2_checkpoint": file_sha256(checkpoint),
            },
            "implementation_revisions": {"sam2": _git_revision(sam2_repository)},
            "information_boundary": {
                "full_robot_action_read_for_window_selection": True,
                "known_future_action_is_conditioning_input": True,
                "tactile_read": False,
                "object_geometry_used_for_window_selection": False,
                "maximum_source_object_frame_read": int(
                    start + PREFIX_FRAME_COUNT - 1
                ),
                "source_object_frames_after_prefix_read": False,
                "future_dense_reconstruction_read": False,
                "future_particle_tracks_read": False,
                "target_metric_read": False,
            },
        }
        manifest["result_sha256"] = canonical_sha256(
            manifest, digest_key="result_sha256"
        )
        (scratch / "prediction_prefix_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(scratch, destination)
    except Exception:
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
