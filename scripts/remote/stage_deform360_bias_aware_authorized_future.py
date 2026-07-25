#!/usr/bin/env python3
"""Reveal one future only after the complete prediction cohort was sealed."""

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

import h5py
import numpy as np

from bayesian_phystwin.deform360_bias_aware_prospective_artifacts import (
    PREDICTION_ARCHIVE_FILENAME,
    PREDICTION_REPORT_FILENAME,
    authorize_prospective_outcome_case,
    canonical_sha256,
    file_sha256,
)
from bayesian_phystwin.deform360_bias_aware_prospective_evaluation import (
    AUTHORIZED_FUTURE_ARTIFACT_KIND,
    AUTHORIZED_FUTURE_MANIFEST_FILENAME,
    validate_bias_aware_calibration_gate,
)
from bayesian_phystwin.deform360_bias_aware_prospective_protocol import (
    PROTOCOL_ID,
    load_bias_aware_prospective_protocol,
)
from bayesian_phystwin.deform360_bias_aware_prospective_staging import (
    PREFIX_FRAME_COUNT,
    STAGING_FRAME_COUNT,
    select_action_only_window,
)
from bayesian_phystwin.deform360_raw_camera_observation import (
    MEASUREMENT_FILENAME,
)
from deform360.robot import RobotState, load_robot_state, save_robot_state


GENERIC_SELECTOR_SHA256 = (
    "79b161fa66489f75b5b078c7ae409387feed74c51a38b86e89800d0aa578b1df"
)
SAM2_REPOSITORY_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SAM2_CHECKPOINT_SHA256 = (
    "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"
)
SOURCE_PREPARATION_FILENAME = "bias_aware_source_preparation_manifest.json"


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
    _require(not status.strip(), f"repository is dirty: {repository}")
    return revision


def _load_selector_class(source: Path):
    _require(source.is_file(), "generic SAM2 selector source is missing")
    _require(file_sha256(source) == GENERIC_SELECTOR_SHA256, "selector changed")
    name = "causal4d_public.deform360_object_sam2_bias_aware_future_locked"
    spec = importlib.util.spec_from_file_location(name, source)
    _require(spec is not None and spec.loader is not None, "cannot load selector")
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
        STAGING_FRAME_COUNT - PREFIX_FRAME_COUNT,
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


def _decoded_prefix_sha256(path: Path) -> str:
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-frames:v",
            str(PREFIX_FRAME_COUNT),
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
    _require(process.wait() == 0, f"cannot decode video prefix: {path}")
    return digest.hexdigest()


def _write_masks(path: Path, masks: list[np.ndarray]) -> None:
    with h5py.File(path, "w") as stream:
        stream.create_dataset(
            "data",
            data=np.asarray(masks, dtype=np.uint8),
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
    parser.add_argument("--role", choices=("calibration", "target"), required=True)
    parser.add_argument("--cohort-seal", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--staged-case-dir", type=Path, required=True)
    parser.add_argument("--source-aligned-root", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--generic-selector-source", type=Path, required=True)
    parser.add_argument("--sam2-repository", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-gate", type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    code_revision = _require_clean_repository(args.repo.resolve())
    protocol_path = args.protocol.resolve()
    protocol = load_bias_aware_prospective_protocol(protocol_path)
    cohort_path = args.cohort_seal.resolve()
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    record, prediction_seal = authorize_prospective_outcome_case(
        cohort,
        protocol_path=protocol_path,
        role=args.role,
        artifact_root=args.prediction_root,
        object_id=args.object_id,
        episode_id=args.episode_id,
    )
    gate: dict[str, object] | None = None
    gate_path: Path | None = None
    if args.role == "target":
        _require(
            args.calibration_gate is not None, "target future needs calibration gate"
        )
        gate_path = args.calibration_gate.resolve()
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        validate_bias_aware_calibration_gate(
            gate, protocol_path=protocol_path, require_passed=True
        )
    else:
        _require(
            args.calibration_gate is None, "calibration must not consume target gate"
        )

    prediction_dir = args.prediction_root.resolve() / str(record["case"])
    prediction_archive = prediction_dir / PREDICTION_ARCHIVE_FILENAME
    prediction_report_path = prediction_dir / PREDICTION_REPORT_FILENAME
    prediction_report = json.loads(prediction_report_path.read_text(encoding="utf-8"))
    measurement_dir = args.measurement_root.resolve() / str(record["case"])
    measurement_path = measurement_dir / MEASUREMENT_FILENAME
    _require(
        file_sha256(measurement_path)
        == prediction_report["inputs_sha256"]["measurement_archive"],
        "measurement differs from sealed prediction",
    )
    with np.load(measurement_path, allow_pickle=False) as stored:
        selected_cameras = np.asarray(stored["selected_cameras"]).astype(str).tolist()
    _require(
        len(selected_cameras) == 8 and len(set(selected_cameras)) == 8,
        "sealed camera panel changed",
    )

    staged = args.staged_case_dir.resolve()
    _require(staged.name == record["case"], "staged case identity changed")
    stage_manifest_path = staged / "prediction_prefix_manifest.json"
    stage_manifest = json.loads(stage_manifest_path.read_text(encoding="utf-8"))
    _require(
        stage_manifest.get("artifact_kind") == "Deform360BiasAwarePredictionPrefix"
        and stage_manifest.get("protocol_id") == PROTOCOL_ID
        and stage_manifest.get("protocol_config_sha256") == protocol["config_sha256"]
        and stage_manifest.get("result_sha256")
        == canonical_sha256(stage_manifest, digest_key="result_sha256")
        and all(stage_manifest.get(key) == value for key, value in record.items()),
        "prediction prefix changed",
    )
    if gate is not None:
        _require(
            stage_manifest.get("target_access_authorization", {}).get(
                "calibration_gate_result_sha256"
            )
            == gate["result_sha256"],
            "prediction prefix used another calibration gate",
        )
    else:
        _require(
            stage_manifest.get("target_access_authorization") is None,
            "calibration prefix consumed a target gate",
        )
    available_cameras = {str(row["camera"]) for row in stage_manifest["camera_records"]}
    _require(set(selected_cameras) <= available_cameras, "camera panel left prefix")

    source_episode = (
        args.source_aligned_root.resolve()
        / args.object_id
        / f"episode_{args.episode_id:04d}"
    )
    source_manifest_path = source_episode / SOURCE_PREPARATION_FILENAME
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    _require(
        source_manifest.get("artifact_kind") == "Deform360BiasAwareSourcePreparation"
        and source_manifest.get("result_sha256")
        == canonical_sha256(source_manifest, digest_key="result_sha256")
        and all(source_manifest.get(key) == value for key, value in record.items()),
        "source preparation changed",
    )
    robot_path = source_episode / "robot" / "robot.npz"
    robot = load_robot_state(robot_path)
    selection = select_action_only_window(robot.actions, robot.openings)
    _require(selection == stage_manifest["action_window"], "action window changed")
    start, stop = selection["selected_raw_frame_range_half_open"]
    _require(stop - start == STAGING_FRAME_COUNT, "authorized raw window changed")

    sam2_repository = args.sam2_repository.resolve()
    sam2_checkpoint = args.sam2_checkpoint.resolve()
    selector_source = args.generic_selector_source.resolve()
    _require(
        _git_revision(sam2_repository) == SAM2_REPOSITORY_REVISION,
        "SAM2 repository changed",
    )
    _require(
        file_sha256(sam2_checkpoint) == SAM2_CHECKPOINT_SHA256,
        "SAM2 checkpoint changed",
    )
    selector_class = _load_selector_class(selector_source)

    destination = args.output_root.resolve() / str(record["case"])
    _require(not destination.exists(), f"authorized future exists: {destination}")
    scratch = destination.with_name(f".{destination.name}.incomplete-{os.getpid()}")
    _require(not scratch.exists(), "authorized-future scratch exists")
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
        predictor = selector_class(sam2_repository, sam2_checkpoint, device=args.device)
        try:
            for camera in selected_cameras:
                source_camera = source_episode / camera
                prefix_camera = staged / "prefix" / "episode_0000" / camera
                output_camera = episode / camera
                output_camera.mkdir()
                full_video = output_camera / "undistorted.mp4"
                _append_tail_to_prefix(
                    prefix_camera / "undistorted.mp4",
                    source_camera / "undistorted.mp4",
                    full_video,
                    source_start=int(start),
                )
                prefix_digest = _decoded_prefix_sha256(
                    prefix_camera / "undistorted.mp4"
                )
                _require(
                    _decoded_prefix_sha256(full_video) == prefix_digest,
                    f"authorized future changed sealed RGB: {camera}",
                )
                prefix_timestamps = (
                    (prefix_camera / "aligned_timestamps.txt")
                    .read_text(encoding="utf-8")
                    .splitlines()
                )
                source_timestamps = (
                    (source_camera / "aligned_timestamps.txt")
                    .read_text(encoding="utf-8")
                    .splitlines()
                )
                timestamps = (
                    prefix_timestamps
                    + source_timestamps[start + PREFIX_FRAME_COUNT : stop]
                )
                _require(
                    len(timestamps) == STAGING_FRAME_COUNT,
                    f"authorized timestamps are incomplete: {camera}",
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
                            "prediction_cohort_result_sha256": cohort["result_sha256"],
                            "future_used_for_initialization": False,
                        },
                    )
                )
                _require(
                    [index for index, _ in propagated]
                    == list(range(STAGING_FRAME_COUNT)),
                    f"SAM2 returned incomplete future: {camera}",
                )
                masks = [np.asarray(mask, dtype=bool) for _, mask in propagated]
                masks[0] = initial_mask
                masks_path = output_camera / "mask_refined.h5"
                _write_masks(masks_path, masks)
                camera_rows.append(
                    {
                        "camera": camera,
                        "video_sha256": file_sha256(full_video),
                        "decoded_sealed_prefix_sha256": prefix_digest,
                        "timestamps_sha256": file_sha256(timestamps_path),
                        "masks_sha256": file_sha256(masks_path),
                        "sam2_diagnostics": predictor.diagnostics[-1],
                    }
                )
        finally:
            predictor.close()

        frame_manifest_path = staged / "frame_zero_reconstruction_manifest.json"
        frame_manifest = json.loads(frame_manifest_path.read_text(encoding="utf-8"))
        sealed_splat = (
            staged / "frame-zero" / "episode_0000" / "splatfacto" / "splat_0.ply"
        )
        _require(
            file_sha256(sealed_splat)
            == frame_manifest["outputs_sha256"]["frame_zero_splat"],
            "sealed frame-zero splat changed",
        )
        full_splat = episode / "splatfacto" / "splat_0.ply"
        full_splat.parent.mkdir()
        shutil.copy2(sealed_splat, full_splat)
        _require(
            file_sha256(full_splat) == file_sha256(sealed_splat), "splat copy changed"
        )

        manifest: dict[str, object] = {
            "schema_version": 1,
            "artifact_kind": AUTHORIZED_FUTURE_ARTIFACT_KIND,
            "protocol_id": PROTOCOL_ID,
            "protocol_config_sha256": protocol["config_sha256"],
            **record,
            "code_revision": code_revision,
            "raw_frame_range_half_open": [start, stop],
            "frame_count": STAGING_FRAME_COUNT,
            "selected_cameras": selected_cameras,
            "camera_records": camera_rows,
            "inputs_sha256": {
                "protocol": file_sha256(protocol_path),
                "prediction_cohort_seal": file_sha256(cohort_path),
                "prediction_seal": file_sha256(
                    prediction_dir / "bias_aware_prediction_seal.json"
                ),
                "prediction_archive": file_sha256(prediction_archive),
                "prediction_prefix_manifest": file_sha256(stage_manifest_path),
                "source_preparation_manifest": file_sha256(source_manifest_path),
                "frame_zero_reconstruction_manifest": file_sha256(frame_manifest_path),
                "source_robot": file_sha256(robot_path),
                "measurement_archive": file_sha256(measurement_path),
                "generic_selector_source": file_sha256(selector_source),
                "sam2_checkpoint": file_sha256(sam2_checkpoint),
                "calibration_gate": (
                    None if gate_path is None else file_sha256(gate_path)
                ),
            },
            "outputs_sha256": {
                "robot": file_sha256(full_robot_path),
                "frame_zero_splat": file_sha256(full_splat),
                "intrinsics": file_sha256(episode / "undistorted_intrinsics.npy"),
                "extrinsics": file_sha256(episode / "extrinsics.npy"),
            },
            "authorization": {
                "prediction_cohort_result_sha256": cohort["result_sha256"],
                "prediction_result_sha256": prediction_seal["result_sha256"],
                "calibration_gate_result_sha256": (
                    None if gate is None else gate["result_sha256"]
                ),
                "prediction_cohort_verified_before_future_read": True,
                "target_access_gate_verified": args.role == "target",
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
        manifest["result_sha256"] = canonical_sha256(
            manifest, digest_key="result_sha256"
        )
        (scratch / AUTHORIZED_FUTURE_MANIFEST_FILENAME).write_text(
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
