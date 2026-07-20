#!/usr/bin/env python3
"""Stage only the locked Deform360 prediction-facing RGB prefix."""

from __future__ import annotations

import argparse
import hashlib
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

from bayesian_phystwin.deform360_selective_virtual_sensing_artifacts import (
    selective_case_records,
)
from bayesian_phystwin.deform360_selective_virtual_sensing_protocol import (
    EXPECTED_UPDATE_FRAMES,
    PROTOCOL_ID,
    load_selective_virtual_sensing_protocol,
)
from bayesian_phystwin.deform360_selective_virtual_sensing_staging import (
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("result_sha256", None)
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_selector_class(source: Path):
    _require(source.is_file(), "generic SAM2 selector source is missing")
    _require(_sha256(source) == GENERIC_SELECTOR_SHA256, "generic SAM2 selector changed")
    name = "causal4d_public.deform360_object_sam2_locked"
    spec = importlib.util.spec_from_file_location(name, source)
    _require(spec is not None and spec.loader is not None, "cannot load SAM2 selector")
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
    _require(ok and observed == frame, f"cannot read exact source frame {frame}: {path}")
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
    lines = source.read_text(encoding="utf-8").splitlines()
    selected = lines[start : start + count]
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


def _case_record(protocol: Path, object_id: str, episode_id: int) -> dict[str, object]:
    matches = [
        row
        for row in selective_case_records(protocol)
        if row["object_id"] == object_id and row["episode_id"] == episode_id
    ]
    _require(len(matches) == 1, "case is outside the locked prospective panel")
    return matches[0]


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
    protocol = load_selective_virtual_sensing_protocol(args.protocol)
    record = _case_record(args.protocol, args.object_id, args.episode_id)
    source_episode = (
        args.source_aligned_root.resolve()
        / args.object_id
        / f"episode_{args.episode_id:04d}"
    )
    _require(source_episode.is_dir(), f"aligned source episode is missing: {source_episode}")
    robot_path = source_episode / "robot" / "robot.npz"
    robot = load_robot_state(robot_path)
    selection = select_action_only_window(
        robot.actions,
        robot.openings,
        protocol_path=str(args.protocol),
    )
    start, raw_stop = selection["selected_raw_frame_range_half_open"]
    raw_count = raw_stop - start
    prefix_count = EXPECTED_UPDATE_FRAMES[-1] + 1
    _require(raw_count == 81 and prefix_count == 58, "locked frame contract changed")

    intrinsics = np.load(
        source_episode / "undistorted_intrinsics.npy", allow_pickle=True
    ).item()
    extrinsics = np.load(source_episode / "extrinsics.npy", allow_pickle=True).item()
    candidates = sorted(
        camera
        for camera in set(intrinsics) & set(extrinsics)
        if (source_episode / camera / "undistorted.mp4").is_file()
    )
    _require(len(candidates) >= 8, "fewer than eight calibrated source cameras")
    sam2_repository = args.sam2_repository.resolve()
    sam2_checkpoint = args.sam2_checkpoint.resolve()
    _require(
        _git_revision(sam2_repository) == SAM2_REPOSITORY_REVISION,
        "SAM2 repository revision changed",
    )
    _require(
        _sha256(sam2_checkpoint) == SAM2_CHECKPOINT_SHA256,
        "SAM2 checkpoint changed",
    )
    selector_class = _load_selector_class(args.generic_selector_source.resolve())
    selector = selector_class(
        sam2_repository,
        sam2_checkpoint,
        device=args.device,
    )
    masks: dict[str, np.ndarray] = {}
    diagnostics: dict[str, object] = {}
    try:
        for camera in candidates:
            rgb = _read_rgb_frame(
                source_episode / camera / "undistorted.mp4", int(start)
            )
            try:
                mask, diagnostic = selector.select_initial_mask_from_rgb(
                    rgb,
                    camera=camera,
                    video_name=f"source-frame-{start:06d}",
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
    _require(len(cameras) >= 8, "fewer than eight cameras passed frame-zero masking")

    destination = args.output_root.resolve() / str(record["case"])
    _require(not destination.exists(), f"prediction prefix already exists: {destination}")
    scratch = destination.with_name(f".{destination.name}.incomplete-{os.getpid()}")
    _require(not scratch.exists(), f"prediction scratch path exists: {scratch}")
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
        prefix_robot = RobotState(
            actions=robot.actions[start : start + prefix_count],
            T_worlds=robot.T_worlds[start : start + prefix_count],
            openings=robot.openings[start : start + prefix_count],
            bimanual=robot.bimanual,
        )
        frame_zero_robot = RobotState(
            actions=robot.actions[start : start + 1],
            T_worlds=robot.T_worlds[start : start + 1],
            openings=robot.openings[start : start + 1],
            bimanual=robot.bimanual,
        )
        prefix_robot_path = prefix_episode / "robot" / "robot.npz"
        frame_zero_robot_path = frame_zero_episode / "robot" / "robot.npz"
        save_robot_state(prefix_robot_path, prefix_robot)
        save_robot_state(frame_zero_robot_path, frame_zero_robot)
        output_rows = []
        for camera in cameras:
            source_camera = source_episode / camera
            prefix_camera = prefix_episode / camera
            frame_zero_camera = frame_zero_episode / camera
            prefix_camera.mkdir()
            frame_zero_camera.mkdir()
            _trim_video(
                source_camera / "undistorted.mp4",
                prefix_camera / "undistorted.mp4",
                int(start),
                prefix_count,
            )
            _trim_timestamps(
                source_camera / "aligned_timestamps.txt",
                prefix_camera / "aligned_timestamps.txt",
                int(start),
                prefix_count,
            )
            _trim_video(
                source_camera / "undistorted.mp4",
                frame_zero_camera / "undistorted.mp4",
                int(start),
                1,
            )
            _trim_timestamps(
                source_camera / "aligned_timestamps.txt",
                frame_zero_camera / "aligned_timestamps.txt",
                int(start),
                1,
            )
            for camera_root in (prefix_camera, frame_zero_camera):
                _write_mask(camera_root / "mask_refined.h5", masks[camera])
            output_rows.append(
                {
                    "camera": camera,
                    "prefix_video_sha256": _sha256(
                        prefix_camera / "undistorted.mp4"
                    ),
                    "frame_zero_video_sha256": _sha256(
                        frame_zero_camera / "undistorted.mp4"
                    ),
                    "frame_zero_mask_sha256": _sha256(
                        prefix_camera / "mask_refined.h5"
                    ),
                }
            )
        manifest: dict[str, object] = {
            "schema_version": 1,
            "artifact_kind": "Deform360SelectivePredictionPrefix",
            "protocol_id": PROTOCOL_ID,
            "protocol_config_sha256": protocol["config_sha256"],
            **record,
            "action_window": selection,
            "raw_window_frame_range_half_open": [start, raw_stop],
            "materialized_object_prefix_frame_range_half_open": [
                start,
                start + prefix_count,
            ],
            "staged_prefix_frame_count": prefix_count,
            "staged_frame_zero_frame_count": 1,
            "camera_count": len(cameras),
            "camera_records": output_rows,
            "mask_diagnostics": diagnostics,
            "staged_robot_sha256": {
                "prefix": _sha256(prefix_robot_path),
                "frame_zero": _sha256(frame_zero_robot_path),
            },
            "inputs_sha256": {
                "protocol": _sha256(args.protocol.resolve()),
                "source_robot": _sha256(robot_path),
                "source_intrinsics": _sha256(
                    source_episode / "undistorted_intrinsics.npy"
                ),
                "source_extrinsics": _sha256(source_episode / "extrinsics.npy"),
                "generic_selector_source": _sha256(
                    args.generic_selector_source.resolve()
                ),
                "sam2_checkpoint": _sha256(args.sam2_checkpoint.resolve()),
            },
            "implementation_revisions": {
                "sam2": _git_revision(sam2_repository),
            },
            "information_boundary": {
                "full_robot_action_read_for_window_selection": True,
                "tactile_read": False,
                "object_geometry_used_for_window_selection": False,
                "maximum_source_object_frame_read": int(start + prefix_count - 1),
                "source_object_frames_after_prefix_read": False,
                "future_dense_reconstruction_read": False,
                "future_particle_tracks_read": False,
                "target_metric_read": False,
            },
        }
        manifest["result_sha256"] = _canonical_sha256(manifest)
        (scratch / "prediction_prefix_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(scratch, destination)
    except Exception:
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "passed": True,
                "case": record["case"],
                "window": [start, raw_stop],
                "prefix_frame_count": prefix_count,
                "camera_count": len(cameras),
                "result_sha256": manifest["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
