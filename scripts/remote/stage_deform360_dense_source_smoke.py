#!/usr/bin/env python3
"""Stage a short, source-only Deform360 episode slice for the dense pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np

from causal4d_public.deform360_dense_source import (
    require_source_episode,
    sha256_file,
    unpack_sampled_mask,
    write_dense_source_manifest,
)
from causal4d_public.deform360_object_sam2 import (
    DeformableObjectSam2VideoPredictor,
)
from deform360.robot import RobotState, load_robot_state, save_robot_state


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
    if len(selected) != count:
        raise ValueError(f"requested {count} timestamps but found {len(selected)}")
    destination.write_text("\n".join(selected) + "\n", encoding="utf-8")


def _subset_calibration(source: Path, destination: Path, cameras: list[str]) -> None:
    payload = np.load(source, allow_pickle=True).item()
    missing = sorted(set(cameras) - set(payload))
    if missing:
        raise ValueError(f"calibration {source.name} lacks cameras {missing}")
    np.save(destination, {camera: payload[camera] for camera in cameras})


def _write_masks(
    destination: Path,
    masks: list[np.ndarray],
) -> None:
    values = np.asarray(masks, dtype=np.uint8)
    with h5py.File(destination, "w") as stream:
        stream.create_dataset(
            "data",
            data=values,
            dtype=np.uint8,
            compression="gzip",
            compression_opts=4,
        )


def _trim_robot(
    source_episode: Path, output_episode: Path, start: int, count: int
) -> Path:
    source = load_robot_state(source_episode / "robot" / "robot.npz")
    stop = start + count
    if stop > source.num_frames:
        raise ValueError("robot state is shorter than the requested frame slice")
    trimmed = RobotState(
        actions=source.actions[start:stop],
        T_worlds=source.T_worlds[start:stop],
        openings=source.openings[start:stop],
        bimanual=source.bimanual,
    )
    return save_robot_state(output_episode / "robot" / "robot.npz", trimmed)


def _trim_tactile_streams(
    source_episode: Path,
    output_episode: Path,
    start: int,
    count: int,
) -> dict[str, str]:
    stop = start + count
    outputs: dict[str, str] = {}
    for source_dir in sorted(source_episode.glob("*tactile*")):
        source = source_dir / "synced_tactile.npy"
        if not source.exists():
            continue
        values = np.load(source, allow_pickle=False)
        trimmed = values[start:stop]
        if len(trimmed) != count:
            raise ValueError(f"tactile stream {source_dir.name} is too short")
        output_dir = output_episode / source_dir.name
        output_dir.mkdir()
        destination = output_dir / "synced_tactile.npy"
        np.save(destination, trimmed)
        for filename in ("metadata.json", "alignment.json"):
            if (source_dir / filename).exists():
                shutil.copy2(source_dir / filename, output_dir / filename)
        outputs[source_dir.name] = sha256_file(destination)
    if not outputs:
        raise FileNotFoundError(f"no tactile streams found in {source_episode}")
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_aligned_root")
    parser.add_argument("sampled_masks_npz")
    parser.add_argument("output_aligned_root")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--sam2-repository", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    require_source_episode(args.protocol, args.object_id, args.episode)
    source_episode = (
        Path(args.source_aligned_root) / args.object_id / f"episode_{args.episode:04d}"
    )
    output_episode = Path(args.output_aligned_root) / "episode_0000"
    if output_episode.exists():
        shutil.rmtree(output_episode)
    output_episode.mkdir(parents=True)

    with np.load(args.sampled_masks_npz, allow_pickle=False) as archive:
        cameras = [str(value) for value in archive["cameras"]]
        initial_masks = {
            camera: unpack_sampled_mask(archive, camera, args.start_frame)
            for camera in cameras
        }

    _subset_calibration(
        source_episode / "undistorted_intrinsics.npy",
        output_episode / "undistorted_intrinsics.npy",
        cameras,
    )
    _subset_calibration(
        source_episode / "extrinsics.npy",
        output_episode / "extrinsics.npy",
        cameras,
    )
    robot_path = _trim_robot(
        source_episode,
        output_episode,
        args.start_frame,
        args.frame_count,
    )
    tactile_hashes = _trim_tactile_streams(
        source_episode,
        output_episode,
        args.start_frame,
        args.frame_count,
    )

    predictor = DeformableObjectSam2VideoPredictor(
        args.sam2_repository,
        args.checkpoint,
        device=args.device,
    )
    diagnostics: dict[str, object] = {}
    try:
        for camera in cameras:
            source_camera = source_episode / camera
            output_camera = output_episode / camera
            output_camera.mkdir()
            _trim_video(
                source_camera / "undistorted.mp4",
                output_camera / "undistorted.mp4",
                args.start_frame,
                args.frame_count,
            )
            _trim_timestamps(
                source_camera / "aligned_timestamps.txt",
                output_camera / "aligned_timestamps.txt",
                args.start_frame,
                args.frame_count,
            )
            metadata_path = source_camera / "metadata.json"
            if metadata_path.exists():
                shutil.copy2(metadata_path, output_camera / "metadata.json")
            masks = list(
                predictor.segment_from_initial_mask(
                    output_camera / "undistorted.mp4",
                    initial_masks[camera],
                    initialization={
                        "source_archive_sha256": sha256_file(args.sampled_masks_npz),
                        "source_frame_index": args.start_frame,
                    },
                )
            )
            if [index for index, _ in masks] != list(range(args.frame_count)):
                raise ValueError(f"SAM2 returned incomplete frames for {camera}")
            _write_masks(
                output_camera / "mask_refined.h5",
                [mask for _, mask in masks],
            )
            diagnostics[camera] = predictor.diagnostics[-1]
    finally:
        predictor.close()

    diagnostics_path = output_episode / "sam2_source_masks.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_dense_source_manifest(
        output_episode / "dense_source_smoke.manifest.json",
        protocol_path=args.protocol,
        object_id=args.object_id,
        episode_index=args.episode,
        source_episode_dir=source_episode,
        sampled_masks_path=args.sampled_masks_npz,
        start_frame=args.start_frame,
        frame_count=args.frame_count,
        cameras=cameras,
        outputs={
            "episode_dir": str(output_episode.resolve()),
            "sam2_diagnostics_sha256": sha256_file(diagnostics_path),
            "robot_sha256": sha256_file(robot_path),
            "tactile_sha256": tactile_hashes,
        },
    )
    print(
        json.dumps(
            {
                "passed": True,
                "source_only": True,
                "episode_dir": str(output_episode),
                "camera_count": len(cameras),
                "frame_count": args.frame_count,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
