#!/usr/bin/env python3
"""Reveal one selected Deform360 future after the prediction cohort is sealed."""

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
from typing import Any, Mapping

import h5py
import numpy as np

from bayesian_phystwin.deform360_online_belief_evaluation import _sha256
from bayesian_phystwin.deform360_selective_virtual_sensing_artifacts import (
    VIRTUAL_SENSING_ARCHIVE_FILENAME,
    authorize_selective_target_case,
)
from bayesian_phystwin.deform360_selective_virtual_sensing_protocol import (
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
RAW_FRAME_COUNT = 81
PREFIX_FRAME_COUNT = 58


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
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


def _load_selector_class(source: Path):
    _require(source.is_file(), "generic SAM2 selector source is missing")
    _require(_sha256(source) == GENERIC_SELECTOR_SHA256, "generic SAM2 selector changed")
    name = "causal4d_public.deform360_object_sam2_future_locked"
    spec = importlib.util.spec_from_file_location(name, source)
    _require(spec is not None and spec.loader is not None, "cannot load SAM2 selector")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.DeformableObjectSam2VideoPredictor


def _trim_video(source: Path, destination: Path, start: int, count: int) -> None:
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


def _append_tail_to_prefix(
    prefix: Path,
    source: Path,
    destination: Path,
    *,
    source_start: int,
) -> None:
    prefix_segment = destination.with_name("sealed_prefix.mp4")
    tail_segment = destination.with_name("authorized_tail.mp4")
    concat_list = destination.with_name("concat.txt")
    shutil.copy2(prefix, prefix_segment)
    _trim_video(
        source,
        tail_segment,
        source_start + PREFIX_FRAME_COUNT,
        RAW_FRAME_COUNT - PREFIX_FRAME_COUNT,
    )
    concat_list.write_text(
        f"file '{prefix_segment}'\nfile '{tail_segment}'\n", encoding="utf-8"
    )
    try:
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
    finally:
        prefix_segment.unlink(missing_ok=True)
        tail_segment.unlink(missing_ok=True)
        concat_list.unlink(missing_ok=True)


def _decoded_prefix_sha256(path: Path, frame_count: int) -> str:
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-frames:v",
            str(frame_count),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
    )
    _require(process.stdout is not None, "ffmpeg output pipe is unavailable")
    digest = hashlib.sha256()
    for chunk in iter(lambda: process.stdout.read(1024 * 1024), b""):
        digest.update(chunk)
    return_code = process.wait()
    _require(return_code == 0, f"cannot decode video prefix: {path}")
    return digest.hexdigest()


def _write_masks(path: Path, masks: list[np.ndarray]) -> None:
    values = np.asarray(masks, dtype=np.uint8)
    with h5py.File(path, "w") as stream:
        stream.create_dataset(
            "data",
            data=values,
            dtype=np.uint8,
            compression="gzip",
            compression_opts=4,
        )


def _save_calibration(source: Path, destination: Path, cameras: list[str]) -> None:
    values = np.load(source, allow_pickle=True).item()
    _require(set(cameras) <= set(values), f"calibration lacks cameras: {source}")
    np.save(destination, {camera: values[camera] for camera in cameras})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cohort-seal", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--failure-root", type=Path, required=True)
    parser.add_argument("--staged-case-dir", type=Path, required=True)
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
    code_revision = _require_clean_repository(args.repo.resolve())
    protocol_path = args.protocol.resolve()
    protocol = load_selective_virtual_sensing_protocol(protocol_path)
    cohort_path = args.cohort_seal.resolve()
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    record, prediction_seal = authorize_selective_target_case(
        cohort,
        protocol_path=protocol_path,
        prediction_root=args.prediction_root,
        failure_root=args.failure_root,
        object_id=args.object_id,
        episode_id=args.episode_id,
    )
    prediction_dir = args.prediction_root.resolve() / str(record["case"])
    prediction_archive = prediction_dir / VIRTUAL_SENSING_ARCHIVE_FILENAME
    with np.load(prediction_archive, allow_pickle=False) as stored:
        selected_cameras = np.asarray(stored["selected_cameras"]).astype(str).tolist()
    _require(
        len(selected_cameras) == 8 and len(set(selected_cameras)) == 8,
        "sealed prediction does not contain exactly eight cameras",
    )

    staged_case = args.staged_case_dir.resolve()
    _require(staged_case.name == record["case"], "staged case identity changed")
    prefix_manifest_path = staged_case / "prediction_prefix_manifest.json"
    prefix_manifest = json.loads(prefix_manifest_path.read_text(encoding="utf-8"))
    _require(
        prefix_manifest.get("artifact_kind") == "Deform360SelectivePredictionPrefix"
        and prefix_manifest.get("protocol_id") == PROTOCOL_ID
        and prefix_manifest.get("protocol_config_sha256") == protocol["config_sha256"]
        and prefix_manifest.get("result_sha256")
        == _canonical_sha256(prefix_manifest),
        "prediction-prefix manifest changed",
    )
    _require(
        all(prefix_manifest.get(key) == value for key, value in record.items()),
        "prediction-prefix case identity changed",
    )
    available_cameras = {
        str(row["camera"]) for row in prefix_manifest["camera_records"]
    }
    _require(
        set(selected_cameras) <= available_cameras,
        "sealed camera panel is absent from the prediction prefix",
    )

    source_episode = (
        args.source_aligned_root.resolve()
        / args.object_id
        / f"episode_{args.episode_id:04d}"
    )
    source_manifest_path = source_episode / "selective_source_preparation_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    _require(
        source_manifest.get("artifact_kind") == "Deform360SelectiveSourcePreparation"
        and source_manifest.get("result_sha256") == _canonical_sha256(source_manifest)
        and all(source_manifest.get(key) == value for key, value in record.items()),
        "source-preparation manifest changed",
    )
    robot_path = source_episode / "robot" / "robot.npz"
    robot = load_robot_state(robot_path)
    selection = select_action_only_window(
        robot.actions, robot.openings, protocol_path=str(protocol_path)
    )
    _require(
        selection == prefix_manifest["action_window"],
        "authorized action window differs from prediction staging",
    )
    start, stop = selection["selected_raw_frame_range_half_open"]
    _require(stop - start == RAW_FRAME_COUNT, "authorized raw window changed")

    sam2_repository = args.sam2_repository.resolve()
    sam2_checkpoint = args.sam2_checkpoint.resolve()
    selector_source = args.generic_selector_source.resolve()
    _require(
        _git_revision(sam2_repository) == SAM2_REPOSITORY_REVISION,
        "SAM2 repository revision changed",
    )
    _require(
        _sha256(sam2_checkpoint) == SAM2_CHECKPOINT_SHA256,
        "SAM2 checkpoint changed",
    )
    selector_class = _load_selector_class(selector_source)

    destination = args.output_root.resolve() / str(record["case"])
    _require(not destination.exists(), f"authorized future already exists: {destination}")
    scratch = destination.with_name(f".{destination.name}.incomplete-{os.getpid()}")
    _require(not scratch.exists(), f"authorized-future scratch exists: {scratch}")
    episode = scratch / "episode_0000"
    episode.mkdir(parents=True)
    try:
        _save_calibration(
            source_episode / "undistorted_intrinsics.npy",
            episode / "undistorted_intrinsics.npy",
            selected_cameras,
        )
        _save_calibration(
            source_episode / "extrinsics.npy",
            episode / "extrinsics.npy",
            selected_cameras,
        )
        full_robot = RobotState(
            actions=robot.actions[start:stop],
            T_worlds=robot.T_worlds[start:stop],
            openings=robot.openings[start:stop],
            bimanual=robot.bimanual,
        )
        full_robot_path = episode / "robot" / "robot.npz"
        save_robot_state(full_robot_path, full_robot)

        camera_rows = []
        predictor = selector_class(
            sam2_repository, sam2_checkpoint, device=args.device
        )
        try:
            for camera in selected_cameras:
                source_camera = source_episode / camera
                prefix_camera = staged_case / "prefix" / "episode_0000" / camera
                output_camera = episode / camera
                output_camera.mkdir()
                full_video = output_camera / "undistorted.mp4"
                _append_tail_to_prefix(
                    prefix_camera / "undistorted.mp4",
                    source_camera / "undistorted.mp4",
                    full_video,
                    source_start=int(start),
                )
                prefix_video_digest = _decoded_prefix_sha256(
                    prefix_camera / "undistorted.mp4", PREFIX_FRAME_COUNT
                )
                _require(
                    _decoded_prefix_sha256(full_video, PREFIX_FRAME_COUNT)
                    == prefix_video_digest,
                    f"authorized future changed the sealed RGB prefix: {camera}",
                )

                prefix_timestamps = (
                    prefix_camera / "aligned_timestamps.txt"
                ).read_text(encoding="utf-8").splitlines()
                source_timestamps = (
                    source_camera / "aligned_timestamps.txt"
                ).read_text(encoding="utf-8").splitlines()
                _require(
                    len(prefix_timestamps) == PREFIX_FRAME_COUNT,
                    f"prediction timestamp prefix changed: {camera}",
                )
                timestamps = prefix_timestamps + source_timestamps[
                    start + PREFIX_FRAME_COUNT : stop
                ]
                _require(
                    len(timestamps) == RAW_FRAME_COUNT,
                    f"authorized timestamp window is incomplete: {camera}",
                )
                timestamps_path = output_camera / "aligned_timestamps.txt"
                timestamps_path.write_text(
                    "\n".join(timestamps) + "\n", encoding="utf-8"
                )

                with h5py.File(prefix_camera / "mask_refined.h5", "r") as stream:
                    initial_values = np.asarray(stream["data"], dtype=np.uint8)
                _require(
                    initial_values.ndim == 3 and initial_values.shape[0] == 1,
                    f"sealed initial mask changed: {camera}",
                )
                initial_mask = initial_values[0].astype(bool)
                propagated = list(
                    predictor.segment_from_initial_mask(
                        full_video,
                        initial_mask,
                        initialization={
                            "policy": "sealed_prediction_frame_mask",
                            "prediction_cohort_result_sha256": cohort[
                                "result_sha256"
                            ],
                            "future_used_for_initialization": False,
                        },
                    )
                )
                _require(
                    [index for index, _ in propagated]
                    == list(range(RAW_FRAME_COUNT)),
                    f"SAM2 returned an incomplete authorized future: {camera}",
                )
                masks = [np.asarray(mask, dtype=bool) for _, mask in propagated]
                masks[0] = initial_mask
                masks_path = output_camera / "mask_refined.h5"
                _write_masks(masks_path, masks)
                with h5py.File(masks_path, "r") as stream:
                    revealed_initial = np.asarray(stream["data"][0], dtype=np.uint8)
                _require(
                    np.array_equal(revealed_initial, initial_values[0]),
                    f"authorized future changed the sealed initial mask: {camera}",
                )
                camera_rows.append(
                    {
                        "camera": camera,
                        "video_sha256": _sha256(full_video),
                        "decoded_sealed_prefix_sha256": prefix_video_digest,
                        "timestamps_sha256": _sha256(timestamps_path),
                        "masks_sha256": _sha256(masks_path),
                        "sam2_diagnostics": predictor.diagnostics[-1],
                    }
                )
        finally:
            predictor.close()

        frame_zero_manifest_path = staged_case / "frame_zero_reconstruction_manifest.json"
        frame_zero_manifest = json.loads(
            frame_zero_manifest_path.read_text(encoding="utf-8")
        )
        sealed_splat = (
            staged_case
            / "frame-zero"
            / "episode_0000"
            / "splatfacto"
            / "splat_0.ply"
        )
        _require(
            _sha256(sealed_splat)
            == frame_zero_manifest["outputs_sha256"]["frame_zero_splat"],
            "sealed frame-zero splat changed",
        )
        full_splat = episode / "splatfacto" / "splat_0.ply"
        full_splat.parent.mkdir()
        shutil.copy2(sealed_splat, full_splat)
        _require(
            _sha256(full_splat) == _sha256(sealed_splat),
            "frame-zero splat copy changed",
        )

        manifest: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": "Deform360SelectiveAuthorizedFuture",
            "protocol_id": PROTOCOL_ID,
            "protocol_config_sha256": protocol["config_sha256"],
            **record,
            "code_revision": code_revision,
            "raw_frame_range_half_open": [start, stop],
            "frame_count": RAW_FRAME_COUNT,
            "camera_count": len(selected_cameras),
            "selected_cameras": selected_cameras,
            "camera_records": camera_rows,
            "inputs_sha256": {
                "protocol": _sha256(protocol_path),
                "prediction_cohort_seal": _sha256(cohort_path),
                "prediction_seal": _sha256(
                    prediction_dir / "virtual_sensing_prediction_seal.json"
                ),
                "prediction_archive": _sha256(prediction_archive),
                "prediction_prefix_manifest": _sha256(prefix_manifest_path),
                "source_preparation_manifest": _sha256(source_manifest_path),
                "frame_zero_reconstruction_manifest": _sha256(
                    frame_zero_manifest_path
                ),
                "source_robot": _sha256(robot_path),
                "generic_selector_source": _sha256(selector_source),
                "sam2_checkpoint": _sha256(sam2_checkpoint),
            },
            "outputs_sha256": {
                "robot": _sha256(full_robot_path),
                "frame_zero_splat": _sha256(full_splat),
                "intrinsics": _sha256(episode / "undistorted_intrinsics.npy"),
                "extrinsics": _sha256(episode / "extrinsics.npy"),
            },
            "authorization": {
                "prediction_cohort_result_sha256": cohort["result_sha256"],
                "prediction_result_sha256": prediction_seal["result_sha256"],
                "eligible_cohort_verified_before_future_read": True,
            },
            "information_boundary": {
                "future_rgb_read_after_cohort_authorization": True,
                "future_masks_created_after_cohort_authorization": True,
                "future_dense_reconstruction_created": False,
                "future_particle_tracks_created": False,
                "target_metric_computed": False,
                "future_tactile_read": False,
            },
        }
        manifest["result_sha256"] = _canonical_sha256(manifest)
        (scratch / "authorized_future_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(scratch, destination)
    except Exception:
        shutil.rmtree(scratch, ignore_errors=True)
        raise

    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
