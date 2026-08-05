#!/usr/bin/env python3
"""Recover a source-only causal robot prefix from all calibrated cameras."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.deform360_causal_robot_prefix import (
    PART_NAMES,
    canonical_camera_order,
    evaluate_causal_robot_prefix_quality,
    load_deform360_causal_robot_prefix_lock,
    run_causal_capture_loop,
    write_causal_robot_prefix_artifact,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean(root: Path, *, name: str) -> str:
    head = _git_head(root)
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError(f"{name} checkout is dirty")
    return head


def _require_ancestor(root: Path, expected: str, *, name: str) -> str:
    head = _require_clean(root, name=name)
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", expected, "HEAD"],
        cwd=root,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"{name} does not contain frozen implementation {expected}")
    return head


def _relative_hashes(episode_dir: Path, cameras: Sequence[str]) -> dict[str, str]:
    paths = [episode_dir / "undistorted_intrinsics.npy", episode_dir / "extrinsics.npy"]
    for camera in cameras:
        paths.extend(
            (
                episode_dir / camera / "undistorted.mp4",
                episode_dir / camera / "aligned_timestamps.txt",
                episode_dir / camera / "metadata.json",
            )
        )
    result: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        result[path.relative_to(episode_dir).as_posix()] = _sha256(path)
    return result


def _assemble_frame_with_support(
    world_observations: Mapping[int, Sequence[tuple[str, np.ndarray]]],
    *,
    gripper_index: int,
    rng: np.random.Generator,
    ransac_average_pose: Any,
    average_transforms: Any,
    markers_per_gripper: int,
    wrist_markers: Sequence[int],
    left_finger_markers: Sequence[int],
    right_finger_markers: Sequence[int],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    base = gripper_index * markers_per_gripper
    pooled: list[np.ndarray] = []
    owners: list[int] = []
    cameras: list[str] = []
    for marker_id in sorted(world_observations):
        if not (base <= marker_id < base + markers_per_gripper):
            continue
        for camera, pose in sorted(world_observations[marker_id], key=lambda item: item[0]):
            pooled.append(np.asarray(pose, dtype=np.float64))
            owners.append(marker_id)
            cameras.append(camera)
    part_counts = np.zeros(len(PART_NAMES), dtype=np.int16)
    marker_counts = np.zeros(markers_per_gripper, dtype=np.int16)
    if not pooled:
        return {}, part_counts, marker_counts

    inlier_indices, _ = ransac_average_pose(pooled, rng=rng)
    by_marker: dict[int, list[np.ndarray]] = {}
    marker_cameras: dict[int, set[str]] = {}
    for index in inlier_indices:
        marker_id = owners[index]
        by_marker.setdefault(marker_id, []).append(pooled[index])
        marker_cameras.setdefault(marker_id, set()).add(cameras[index])
    marker_means = {
        marker_id: average_transforms(poses)
        for marker_id, poses in by_marker.items()
    }
    for marker_id, camera_set in marker_cameras.items():
        marker_counts[marker_id - base] = len(camera_set)

    parts: dict[str, np.ndarray] = {}
    groups = (wrist_markers, left_finger_markers, right_finger_markers)
    for part_index, (part, marker_ids) in enumerate(zip(PART_NAMES, groups, strict=True)):
        present = [
            marker_means[base + marker_id]
            for marker_id in marker_ids
            if base + marker_id in marker_means
        ]
        if not present:
            continue
        parts[part] = average_transforms(present)
        part_counts[part_index] = len(
            {
                camera
                for marker_id in marker_ids
                for camera in marker_cameras.get(base + marker_id, ())
            }
        )
    return parts, part_counts, marker_counts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repository_root = args.repository_root.resolve()
    upstream_root = args.upstream_root.resolve()
    lock = load_deform360_causal_robot_prefix_lock(args.lock)
    estimator = lock["estimator"]
    assert isinstance(estimator, Mapping)
    runtime_revision = _require_ancestor(
        repository_root,
        str(estimator["implementation_revision"]),
        name="Bayesian-PhysTwin",
    )
    upstream_revision = _require_clean(upstream_root, name="Deform360 processing")
    if upstream_revision != estimator["revision"]:
        raise ValueError(
            f"Deform360 processing is at {upstream_revision}, expected {estimator['revision']}"
        )

    sys.path.insert(0, str(upstream_root))
    import cv2  # noqa: PLC0415
    from deform360.processing.episode import (  # noqa: PLC0415
        camera_frame_count,
        episode_cameras,
        load_episode_calibration,
    )
    from deform360.processing.robot_stage import (  # noqa: PLC0415
        LEFT_FINGER_MARKERS,
        MARKERS_PER_GRIPPER,
        RIGHT_FINGER_MARKERS,
        WRIST_MARKERS,
        build_robot_state,
        default_detect_fn,
        dense_part_trajectories,
        marker_gripper_offset,
        ransac_average_pose,
        suppress_velocity_outliers,
    )
    from deform360.robot import (  # noqa: PLC0415
        average_transforms,
        observation_to_world,
    )

    source = lock["source_case"]
    window = lock["causal_window"]
    assert isinstance(source, Mapping)
    assert isinstance(window, Mapping)
    object_id = str(source["object_id"])
    episode_index = int(source["processing_episode_index"])
    episode_dir = (
        args.processed_root.resolve()
        / object_id
        / f"episode_{episode_index:04d}"
    )
    cameras = canonical_camera_order(tuple(source["cameras"]))
    discovered = canonical_camera_order(episode_cameras(episode_dir))
    if discovered != cameras:
        raise ValueError("processed all-camera panel changed")
    intrinsics, extrinsics = load_episode_calibration(episode_dir)
    if set(cameras) - set(intrinsics) or set(cameras) - set(extrinsics):
        raise ValueError("all-camera panel is not fully calibrated")

    start = int(window["source_frame_start"])
    stop = int(window["causal_frame_stop"])
    frame_count = stop - start
    for camera in cameras:
        if camera_frame_count(episode_dir, camera) < stop:
            raise ValueError(f"camera {camera} ends before the causal cutoff")

    video_paths = {camera: episode_dir / camera / "undistorted.mp4" for camera in cameras}
    captures = {camera: cv2.VideoCapture(str(video_paths[camera])) for camera in cameras}
    for camera, capture in captures.items():
        if not capture.isOpened():
            raise ValueError(f"cannot open camera video {camera}")
    detect = default_detect_fn(intrinsics)
    observations: list[dict[int, list[tuple[str, np.ndarray]]]] = [
        {} for _ in range(frame_count)
    ]
    raw_counts = np.zeros((frame_count, len(cameras), 16), dtype=np.int16)
    camera_indices = {camera: index for index, camera in enumerate(cameras)}

    def process_frame(frame_id: int, camera: str, frame_bgr: np.ndarray) -> None:
        local = frame_id - start
        for observation in detect(frame_bgr, camera):
            marker_id = observation.marker_id
            if marker_id is None or not (0 <= int(marker_id) < 16):
                continue
            marker_id = int(marker_id)
            raw_counts[local, camera_indices[camera], marker_id] += 1
            world = observation_to_world(
                observation,
                extrinsics[camera],
                marker_from_target=marker_gripper_offset(marker_id),
            )
            observations[local].setdefault(marker_id, []).append((camera, world))

    try:
        callback_count = run_causal_capture_loop(
            captures,
            source_frame_start=start,
            causal_frame_stop=stop,
            process_frame=process_frame,
        )
    finally:
        for capture in captures.values():
            capture.release()
    if callback_count != frame_count * len(cameras):
        raise RuntimeError("causal decoder callback count changed")

    bimanual = bool(source["bimanual"])
    gripper_count = 2 if bimanual else 1
    per_gripper_frames: list[dict[int, dict[str, np.ndarray]]] = [
        {} for _ in range(gripper_count)
    ]
    part_counts = np.zeros(
        (frame_count, gripper_count, len(PART_NAMES)), dtype=np.int16
    )
    marker_counts = np.zeros(
        (frame_count, gripper_count, MARKERS_PER_GRIPPER), dtype=np.int16
    )
    rng = np.random.default_rng(int(estimator["seed"]))
    for local, frame_observations in enumerate(observations):
        for gripper in range(gripper_count):
            parts, frame_part_counts, frame_marker_counts = (
                _assemble_frame_with_support(
                    frame_observations,
                    gripper_index=gripper,
                    rng=rng,
                    ransac_average_pose=ransac_average_pose,
                    average_transforms=average_transforms,
                    markers_per_gripper=MARKERS_PER_GRIPPER,
                    wrist_markers=WRIST_MARKERS,
                    left_finger_markers=LEFT_FINGER_MARKERS,
                    right_finger_markers=RIGHT_FINGER_MARKERS,
                )
            )
            part_counts[local, gripper] = frame_part_counts
            marker_counts[local, gripper] = frame_marker_counts
            if parts:
                per_gripper_frames[gripper][local] = parts

    dense = []
    for frames in per_gripper_frames:
        filtered = suppress_velocity_outliers(frames)
        dense.append(dense_part_trajectories(filtered, frame_count))
    state = build_robot_state(dense, frame_count, bimanual=bimanual)
    frame_ids = np.arange(start, stop, dtype=np.int64)
    quality_gate = lock["quality_gate"]
    assert isinstance(quality_gate, Mapping)
    quality = evaluate_causal_robot_prefix_quality(
        transforms=state.T_worlds,
        openings_m=state.openings,
        part_inlier_camera_counts=part_counts,
        source_frame_ids=frame_ids,
        bimanual=bimanual,
        quality_gate=quality_gate,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "source_frame_ids": frame_ids,
        "actions": state.actions,
        "T_worlds": state.T_worlds,
        "openings": state.openings,
        "part_inlier_camera_counts": part_counts,
        "marker_inlier_camera_counts": marker_counts,
        "raw_marker_detection_counts": raw_counts,
    }
    manifest = write_causal_robot_prefix_artifact(
        output_npz=output_dir / "causal_robot_prefix.npz",
        output_manifest=output_dir / "causal_robot_prefix.json",
        arrays=arrays,
        lock=lock,
        lock_file_sha256=_sha256(args.lock),
        implementation_revision=runtime_revision,
        source_artifacts=_relative_hashes(episode_dir, cameras),
        quality=quality,
        overwrite=args.overwrite,
    )
    print(
        f"artifact_id={manifest['artifact_id']} admitted={quality.admitted} "
        f"reasons={','.join(quality.reason_codes) or 'none'}"
    )


if __name__ == "__main__":
    main()
