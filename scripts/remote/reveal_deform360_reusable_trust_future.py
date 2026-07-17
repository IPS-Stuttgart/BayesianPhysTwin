#!/usr/bin/env python3
"""Reveal a fresh fit episode's object future only after its grid is sealed."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from causal4d_public.deform360_dense_source import sha256_file
from causal4d_public.deform360_object_sam2 import (
    DeformableObjectSam2VideoPredictor,
)
from causal4d_public.deform360_reusable_physics import (
    validate_reusable_physics_fit_grid_seal,
)
from causal4d_public.deform360_reusable_trust_state import (
    load_reusable_trust_state_addendum,
)


def _result_sha256(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _append_future_to_prediction_frame(
    source: Path,
    prediction_frame: Path,
    destination: Path,
    start: int,
    count: int,
) -> None:
    if count < 2:
        raise ValueError("future reveal requires at least two frames")
    first_segment = destination.with_name("prediction_frame.mp4")
    tail_segment = destination.with_name("future_tail.mp4")
    concat_list = destination.with_name("concat.txt")
    shutil.copy2(prediction_frame, first_segment)
    _trim_video(source, tail_segment, start + 1, count - 1)
    concat_list.write_text(
        f"file '{first_segment}'\nfile '{tail_segment}'\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            str(destination),
        ],
        check=True,
    )
    first_segment.unlink()
    tail_segment.unlink()
    concat_list.unlink()


def _trim_timestamps(
    source: Path, destination: Path, start: int, count: int
) -> None:
    selected = source.read_text(encoding="utf-8").splitlines()[start : start + count]
    if len(selected) != count:
        raise ValueError(f"requested {count} timestamps but found {len(selected)}")
    destination.write_text("\n".join(selected) + "\n", encoding="utf-8")


def _write_masks(destination: Path, masks: list[np.ndarray]) -> None:
    values = np.asarray(masks, dtype=np.uint8)
    with h5py.File(destination, "w") as stream:
        stream.create_dataset(
            "data",
            data=values,
            dtype=np.uint8,
            compression="gzip",
            compression_opts=4,
        )


def _first_decoded_frame(path: Path) -> bytes:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    if not result.stdout:
        raise ValueError(f"video has no decodable frame: {path}")
    return result.stdout


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-lock", type=Path, required=True)
    parser.add_argument("--physics-addendum", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--mask-addendum", type=Path, required=True)
    parser.add_argument("--state-addendum", type=Path, required=True)
    parser.add_argument("--fit-grid-seal", type=Path, required=True)
    parser.add_argument("--aligned-dir", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--sam2-repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    protocol = load_reusable_trust_state_addendum(
        args.parent_lock,
        args.physics_addendum,
        args.execution_lock,
        args.mask_addendum,
        args.state_addendum,
    )
    seal = json.loads(args.fit_grid_seal.read_text(encoding="utf-8"))
    access = validate_reusable_physics_fit_grid_seal(
        seal, protocol=protocol, verify_responses=True
    )
    episode_dir = args.aligned_dir / f"episode_{args.episode:04d}"
    manifest_path = episode_dir / "dense_source_smoke.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("source_only") is not True
        or manifest.get("target_episode_accessed") is not False
        or manifest.get("calibration_episode_accessed") is not False
        or manifest.get("object_id") != access["object_id"]
        or int(manifest.get("episode_index", -1)) != access["episode_id"]
        or int(manifest.get("outputs", {}).get("object_observation_frame_count", -1))
        != 1
    ):
        raise ValueError("prediction-only staging differs from the fit-grid seal")
    start, stop = [int(value) for value in manifest["frame_range"]]
    frame_count = int(manifest["outputs"]["known_robot_action_frame_count"])
    if stop != start + 1 or frame_count != 81:
        raise ValueError("fresh reveal requires a one-frame stage and 81-frame window")
    if sha256_file(args.checkpoint) != protocol["mask_addendum"]["sam2"][
        "checkpoint_sha256"
    ]:
        raise ValueError("SAM2 checkpoint differs from the frozen mask policy")

    source_episode = Path(manifest["inputs"]["source_episode_dir"])
    cameras = [str(value) for value in manifest["cameras"]]
    expected_cameras = protocol["mask_addendum"]["objects"][access["object_id"]][
        "cameras"
    ]
    if cameras != expected_cameras:
        raise ValueError("staged cameras differ from the frozen camera panel")

    output_path = args.output or episode_dir / "future_reveal.meta.json"
    if output_path.exists():
        raise FileExistsError(f"future reveal already exists: {output_path}")
    temporary_root = episode_dir / ".future_reveal_tmp"
    archive_root = episode_dir / "prediction_only_frame_zero"
    if temporary_root.exists() or archive_root.exists():
        raise FileExistsError("future reveal scratch or archive already exists")
    temporary_root.mkdir()

    before: dict[str, dict[str, Any]] = {}
    prepared: dict[str, dict[str, Any]] = {}
    predictor = DeformableObjectSam2VideoPredictor(
        args.sam2_repository,
        args.checkpoint,
        device=args.device,
    )
    try:
        for camera in cameras:
            staged_camera = episode_dir / camera
            source_camera = source_episode / camera
            temporary_camera = temporary_root / camera
            temporary_camera.mkdir()
            staged_video = staged_camera / "undistorted.mp4"
            staged_timestamps = staged_camera / "aligned_timestamps.txt"
            staged_masks = staged_camera / "mask_refined.h5"
            with h5py.File(staged_masks, "r") as stream:
                initial_masks = np.asarray(stream["data"], dtype=np.uint8)
            if initial_masks.shape[0] != 1:
                raise ValueError(f"{camera} is not a prediction-only mask stage")
            initial_mask = initial_masks[0].astype(bool)
            initial_timestamp = staged_timestamps.read_text(
                encoding="utf-8"
            ).splitlines()
            if len(initial_timestamp) != 1:
                raise ValueError(f"{camera} is not a one-frame timestamp stage")
            before[camera] = {
                "video_sha256": sha256_file(staged_video),
                "timestamps_sha256": sha256_file(staged_timestamps),
                "mask_sha256": sha256_file(staged_masks),
                "decoded_frame_zero_sha256": hashlib.sha256(
                    _first_decoded_frame(staged_video)
                ).hexdigest(),
                "mask_frame_zero_sha256": hashlib.sha256(
                    initial_masks[0].tobytes()
                ).hexdigest(),
            }

            future_video = temporary_camera / "undistorted.mp4"
            future_timestamps = temporary_camera / "aligned_timestamps.txt"
            future_masks = temporary_camera / "mask_refined.h5"
            _append_future_to_prediction_frame(
                source_camera / "undistorted.mp4",
                staged_video,
                future_video,
                start,
                frame_count,
            )
            _trim_timestamps(
                source_camera / "aligned_timestamps.txt",
                future_timestamps,
                start,
                frame_count,
            )
            if future_timestamps.read_text(encoding="utf-8").splitlines()[0] != (
                initial_timestamp[0]
            ):
                raise ValueError(f"{camera} frame-zero timestamp changed")
            decoded = _first_decoded_frame(future_video)
            if hashlib.sha256(decoded).hexdigest() != before[camera][
                "decoded_frame_zero_sha256"
            ]:
                raise ValueError(f"{camera} decoded frame zero changed")
            propagated = list(
                predictor.segment_from_initial_mask(
                    future_video,
                    initial_mask,
                    initialization={
                        "policy": "sealed_prediction_frame_mask",
                        "fit_grid_seal_result_sha256": seal["result_sha256"],
                        "object_observation_frames_used_for_initialization": [0],
                        "future_used_for_initialization": False,
                    },
                )
            )
            if [index for index, _ in propagated] != list(range(frame_count)):
                raise ValueError(f"SAM2 returned incomplete future for {camera}")
            masks = [np.asarray(mask, dtype=bool) for _, mask in propagated]
            masks[0] = initial_mask
            _write_masks(future_masks, masks)
            with h5py.File(future_masks, "r") as stream:
                revealed_initial = np.asarray(stream["data"][0], dtype=np.uint8)
            if not np.array_equal(revealed_initial, initial_masks[0]):
                raise ValueError(f"{camera} frame-zero mask changed")
            prepared[camera] = {
                "video_sha256": sha256_file(future_video),
                "timestamps_sha256": sha256_file(future_timestamps),
                "mask_sha256": sha256_file(future_masks),
                "sam2_diagnostics": predictor.diagnostics[-1],
            }
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    finally:
        predictor.close()

    archive_root.mkdir()
    for camera in cameras:
        staged_camera = episode_dir / camera
        temporary_camera = temporary_root / camera
        archived_camera = archive_root / camera
        archived_camera.mkdir()
        for name in ("undistorted.mp4", "aligned_timestamps.txt", "mask_refined.h5"):
            shutil.copy2(staged_camera / name, archived_camera / name)
            (temporary_camera / name).replace(staged_camera / name)
    shutil.rmtree(temporary_root)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360ReusableTrustAuthorizedFutureReveal",
        "protocol_id": protocol["parent"]["protocol_id"],
        "physics_addendum_id": protocol["addendum"]["protocol_id"],
        "mask_addendum_id": protocol["mask_addendum"]["protocol_id"],
        "state_addendum_id": protocol["state_addendum"]["protocol_id"],
        "object_id": access["object_id"],
        "episode_id": access["episode_id"],
        "frame_range_half_open": [start, start + frame_count],
        "frame_count": frame_count,
        "camera_count": len(cameras),
        "fit_grid_seal": {
            "path": str(args.fit_grid_seal.resolve()),
            "file_sha256": sha256_file(args.fit_grid_seal),
            "result_sha256": seal["result_sha256"],
        },
        "input_sha256": {
            "parent_lock": sha256_file(args.parent_lock),
            "physics_addendum": sha256_file(args.physics_addendum),
            "execution_lock": sha256_file(args.execution_lock),
            "mask_addendum": sha256_file(args.mask_addendum),
            "state_addendum": sha256_file(args.state_addendum),
            "manifest": sha256_file(manifest_path),
            "sam2_checkpoint": sha256_file(args.checkpoint),
        },
        "prediction_only_inputs": before,
        "revealed_inputs": prepared,
        "information_boundary": {
            "fit_grid_verified_before_future_open": True,
            "all_18_physical_responses_hashed_before_future_open": True,
            "frame_zero_rgb_preserved_after_decode": True,
            "frame_zero_mask_preserved_exactly": True,
            "future_object_observations_opened": True,
            "future_tactile_read": False,
            "held_outcome_read": False,
        },
        "claim_boundary": (
            "authorized fit-episode outcome reveal after the complete physical "
            "grid seal; no held episode or tactile future was opened"
        ),
    }
    payload["result_sha256"] = _result_sha256(payload)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
