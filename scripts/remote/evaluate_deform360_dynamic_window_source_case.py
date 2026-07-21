#!/usr/bin/env python3
"""Reveal and score one sealed dynamic-window source case."""

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
from types import ModuleType
from typing import Any, Mapping

import h5py
import numpy as np

from bayesian_phystwin.deform360_online_belief_evaluation import _sha256
from bayesian_phystwin.deform360_raw_camera_observation import (
    MANIFEST_FILENAME,
    MEASUREMENT_FILENAME,
)
from bayesian_phystwin.deform360_selective_virtual_sensing_artifacts import (
    VIRTUAL_SENSING_ARCHIVE_FILENAME,
    VIRTUAL_SENSING_REPORT_FILENAME,
    VIRTUAL_SENSING_SEAL_FILENAME,
    selective_case_records,
    validate_selective_prediction_seal,
)
from bayesian_phystwin.deform360_selective_virtual_sensing_evaluation import (
    ARM_TO_ARCHIVE_KEY,
    SCORED_FRAMES,
    _measurement_target_audit,
    score_selective_virtual_sensing_arrays,
)
from bayesian_phystwin.deform360_selective_virtual_sensing_protocol import (
    PROTOCOL_ID,
    load_selective_virtual_sensing_protocol,
)
from bayesian_phystwin.deform360_selective_virtual_sensing_staging import (
    dynamic_window_source_case,
)
from deform360.robot import RobotState, load_robot_state, save_robot_state


TARGET_ARCHIVE_FILENAME = "target_trajectory.npz"
SOURCE_FUTURE_MANIFEST_FILENAME = "source_future_manifest.json"
SOURCE_EVALUATION_FILENAME = "dynamic_window_source_evaluation.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("config_sha256", None)
    unsigned.pop("result_sha256", None)
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _load_script(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    _require(spec is not None and spec.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


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


def _case(protocol: Path, object_id: str, episode_id: int) -> dict[str, Any]:
    matches = [
        row
        for row in selective_case_records(protocol)
        if row["object_id"] == object_id and row["episode_id"] == episode_id
    ]
    _require(len(matches) == 1, "case is outside the exhausted source panel")
    return matches[0]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--selection-seal", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--staged-case-dir", type=Path, required=True)
    parser.add_argument("--source-aligned-root", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--deform360-repo", type=Path, required=True)
    parser.add_argument("--tracking-checkpoint", type=Path, required=True)
    parser.add_argument("--cotracker-repository", type=Path, required=True)
    parser.add_argument("--generic-selector-source", type=Path, required=True)
    parser.add_argument("--sam2-repository", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _validate_prefix(
    path: Path,
    *,
    protocol: Mapping[str, Any],
    record: Mapping[str, Any],
    source_row: Mapping[str, Any],
    selection_seal: Path,
) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _require(
        manifest.get("artifact_kind") == "Deform360SelectivePredictionPrefix"
        and manifest.get("protocol_id") == PROTOCOL_ID
        and manifest.get("protocol_config_sha256") == protocol["config_sha256"]
        and manifest.get("result_sha256") == _canonical_sha256(manifest),
        "prediction-prefix manifest changed",
    )
    _require(
        all(manifest.get(key) == value for key, value in record.items()),
        "prediction-prefix case identity changed",
    )
    _require(
        manifest.get("action_window") == source_row["translation_contact_v2"],
        "prediction prefix differs from the source-window seal",
    )
    provenance = manifest.get("source_window_selection", {})
    _require(
        provenance.get("file_sha256") == _sha256(selection_seal)
        and provenance.get("result_sha256")
        == json.loads(selection_seal.read_text(encoding="utf-8"))["result_sha256"],
        "prediction prefix references another source-window seal",
    )
    boundary = manifest.get("information_boundary", {})
    _require(
        boundary.get("source_object_frames_after_prefix_read") is False
        and boundary.get("future_dense_reconstruction_read") is False
        and boundary.get("future_particle_tracks_read") is False
        and boundary.get("target_metric_read") is False
        and boundary.get("tactile_exposed_to_prediction_method") is False,
        "prediction prefix crossed its target boundary",
    )
    return manifest


def _validate_source_config(path: Path, selection: Mapping[str, Any]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(
        payload.get("config_sha256") == _canonical_sha256(payload),
        "dynamic-window source config changed",
    )
    _require(
        _sha256(path) == selection.get("source_config_sha256"),
        "selection seal references another source config",
    )
    return payload["config"]


def _stage_source_future(
    *,
    stage: ModuleType,
    destination: Path,
    source_episode: Path,
    staged_case: Path,
    selected_cameras: list[str],
    start: int,
    stop: int,
    record: Mapping[str, Any],
    protocol: Mapping[str, Any],
    prediction_seal: Mapping[str, Any],
    prediction_archive: Path,
    prefix_manifest_path: Path,
    source_manifest_path: Path,
    selection_seal: Path,
    frame_zero_manifest_path: Path,
    sam2_repository: Path,
    sam2_checkpoint: Path,
    generic_selector_source: Path,
    device: str,
) -> dict[str, Any]:
    manifest_path = destination / SOURCE_FUTURE_MANIFEST_FILENAME
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _require(
            manifest.get("result_sha256") == _canonical_sha256(manifest)
            and all(manifest.get(key) == value for key, value in record.items())
            and manifest.get("selected_cameras") == selected_cameras
            and manifest.get("raw_frame_range_half_open") == [start, stop],
            "existing dynamic-window source future changed",
        )
        return manifest
    _require(
        not destination.exists(), f"incomplete source future exists: {destination}"
    )
    scratch = destination.with_name(f".{destination.name}.incomplete-{os.getpid()}")
    _require(not scratch.exists(), f"source-future scratch exists: {scratch}")
    episode = scratch / "episode_0000"
    episode.mkdir(parents=True)
    robot = load_robot_state(source_episode / "robot" / "robot.npz")
    try:
        stage._save_calibration(
            source_episode / "undistorted_intrinsics.npy",
            episode / "undistorted_intrinsics.npy",
            selected_cameras,
        )
        stage._save_calibration(
            source_episode / "extrinsics.npy",
            episode / "extrinsics.npy",
            selected_cameras,
        )
        selected_robot = RobotState(
            actions=robot.actions[start:stop],
            T_worlds=robot.T_worlds[start:stop],
            openings=robot.openings[start:stop],
            bimanual=robot.bimanual,
        )
        robot_path = episode / "robot" / "robot.npz"
        save_robot_state(robot_path, selected_robot)

        selector_class = stage._load_selector_class(generic_selector_source)
        predictor = selector_class(sam2_repository, sam2_checkpoint, device=device)
        camera_rows = []
        try:
            for camera in selected_cameras:
                source_camera = source_episode / camera
                prefix_camera = staged_case / "prefix" / "episode_0000" / camera
                output_camera = episode / camera
                output_camera.mkdir()
                full_video = output_camera / "undistorted.mp4"
                stage._append_tail_to_prefix(
                    prefix_camera / "undistorted.mp4",
                    source_camera / "undistorted.mp4",
                    full_video,
                    source_start=start,
                )
                prefix_digest = stage._decoded_prefix_sha256(
                    prefix_camera / "undistorted.mp4", stage.PREFIX_FRAME_COUNT
                )
                _require(
                    stage._decoded_prefix_sha256(full_video, stage.PREFIX_FRAME_COUNT)
                    == prefix_digest,
                    f"source reveal changed the sealed RGB prefix: {camera}",
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
                    + source_timestamps[start + stage.PREFIX_FRAME_COUNT : stop]
                )
                _require(
                    len(timestamps) == stage.RAW_FRAME_COUNT,
                    f"source timestamp window is incomplete: {camera}",
                )
                timestamps_path = output_camera / "aligned_timestamps.txt"
                timestamps_path.write_text(
                    "\n".join(timestamps) + "\n", encoding="utf-8"
                )
                with h5py.File(prefix_camera / "mask_refined.h5", "r") as stream:
                    initial_values = np.asarray(stream["data"], dtype=np.uint8)
                initial_mask = initial_values[0].astype(bool)
                propagated = list(
                    predictor.segment_from_initial_mask(
                        full_video,
                        initial_mask,
                        initialization={
                            "policy": "sealed_source_prediction_frame_mask",
                            "prediction_result_sha256": prediction_seal[
                                "result_sha256"
                            ],
                            "future_used_for_initialization": False,
                        },
                    )
                )
                _require(
                    [index for index, _ in propagated]
                    == list(range(stage.RAW_FRAME_COUNT)),
                    f"SAM2 returned an incomplete source future: {camera}",
                )
                masks = [np.asarray(mask, dtype=bool) for _, mask in propagated]
                masks[0] = initial_mask
                masks_path = output_camera / "mask_refined.h5"
                stage._write_masks(masks_path, masks)
                camera_rows.append(
                    {
                        "camera": camera,
                        "video_sha256": _sha256(full_video),
                        "decoded_sealed_prefix_sha256": prefix_digest,
                        "timestamps_sha256": _sha256(timestamps_path),
                        "masks_sha256": _sha256(masks_path),
                        "sam2_diagnostics": predictor.diagnostics[-1],
                    }
                )
        finally:
            predictor.close()

        frame_zero_manifest = json.loads(
            frame_zero_manifest_path.read_text(encoding="utf-8")
        )
        sealed_splat = (
            staged_case / "frame-zero" / "episode_0000" / "splatfacto" / "splat_0.ply"
        )
        _require(
            _sha256(sealed_splat)
            == frame_zero_manifest["outputs_sha256"]["frame_zero_splat"],
            "sealed frame-zero splat changed",
        )
        full_splat = episode / "splatfacto" / "splat_0.ply"
        full_splat.parent.mkdir()
        shutil.copy2(sealed_splat, full_splat)
        payload: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": "Deform360DynamicWindowSourceFuture",
            "protocol_id": "deform360-dynamic-window-source-v1",
            "protocol_config_sha256": protocol["config_sha256"],
            **record,
            "raw_frame_range_half_open": [start, stop],
            "frame_count": stage.RAW_FRAME_COUNT,
            "camera_count": len(selected_cameras),
            "selected_cameras": selected_cameras,
            "camera_records": camera_rows,
            "inputs_sha256": {
                "selection_seal": _sha256(selection_seal),
                "prediction_seal": _sha256(
                    prediction_archive.parent / VIRTUAL_SENSING_SEAL_FILENAME
                ),
                "prediction_archive": _sha256(prediction_archive),
                "prediction_prefix_manifest": _sha256(prefix_manifest_path),
                "source_preparation_manifest": _sha256(source_manifest_path),
                "frame_zero_reconstruction_manifest": _sha256(frame_zero_manifest_path),
                "generic_selector_source": _sha256(generic_selector_source),
                "sam2_checkpoint": _sha256(sam2_checkpoint),
            },
            "outputs_sha256": {
                "robot": _sha256(robot_path),
                "frame_zero_splat": _sha256(full_splat),
                "intrinsics": _sha256(episode / "undistorted_intrinsics.npy"),
                "extrinsics": _sha256(episode / "extrinsics.npy"),
            },
            "authorization": {
                "source_window_selection_result_sha256": json.loads(
                    selection_seal.read_text(encoding="utf-8")
                )["result_sha256"],
                "prediction_result_sha256": prediction_seal["result_sha256"],
                "prediction_verified_before_future_read": True,
            },
            "information_boundary": {
                "exhausted_source_case": True,
                "fresh_objects_or_reserved_targets_read": False,
                "future_rgb_read_after_prediction_seal": True,
                "future_tactile_read": False,
                "target_metric_computed": False,
            },
        }
        payload["result_sha256"] = _canonical_sha256(payload)
        (scratch / SOURCE_FUTURE_MANIFEST_FILENAME).write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(scratch, destination)
        return payload
    except Exception:
        shutil.rmtree(scratch, ignore_errors=True)
        raise


def _build_target(
    outcome: ModuleType, case_root: Path, cameras: list[str], checkpoint: Path
) -> np.ndarray:
    original_visual_hull = outcome.reconstruct_stage.visual_hull_points

    def strict_visual_hull(*call_args: object, **call_kwargs: object):
        call_kwargs["min_points"] = outcome.MINIMUM_VISUAL_HULL_POINTS
        return original_visual_hull(*call_args, **call_kwargs)

    outcome.reconstruct_stage.visual_hull_points = strict_visual_hull
    try:
        splats = outcome.reconstruct_stage.process_reconstruction_episode(
            case_root,
            0,
            cameras=cameras,
            first_frame_iterations=outcome.FIRST_FRAME_ITERATIONS,
            warm_start_iterations=outcome.WARM_START_ITERATIONS,
            cube_half_extent_m=outcome.CUBE_HALF_EXTENT_M,
            voxel_resolution=outcome.VOXEL_RESOLUTION,
            overwrite=False,
            keep_scratch=False,
        )
    finally:
        outcome.reconstruct_stage.visual_hull_points = original_visual_hull
    _require(
        set(splats) == set(range(outcome.RAW_FRAME_COUNT)), "reconstruction failed"
    )
    gripper_masks = outcome.urdf_render.process_gripper_masks_episode(
        case_root, 0, cameras=cameras, overwrite=False
    )
    depths = outcome.depth_stage.process_depth_episode(
        case_root, 0, cameras=cameras, overwrite=False, preview=False
    )
    tracks = outcome.tracking_stage.process_tracking_episode(
        case_root, 0, cameras=cameras, checkpoint=checkpoint, overwrite=False
    )
    _require(
        set(gripper_masks) == set(depths) == set(tracks) == set(cameras),
        "target camera stages disagree",
    )
    pcd_dir = outcome.pcd_stage.process_pcd_episode(
        case_root, 0, cameras=cameras, overwrite=False, rng_seed=0
    )
    points = []
    for frame in range(outcome.TARGET_FRAME_COUNT):
        with np.load(pcd_dir / f"{frame:06d}.npz", allow_pickle=False) as stored:
            points.append(np.asarray(stored["pts"], dtype=np.float32))
    target = np.stack(points)
    _require(
        target.ndim == 3
        and target.shape[0] == outcome.TARGET_FRAME_COUNT
        and target.shape[2] == 3
        and np.all(np.isfinite(target)),
        "source target trajectory is invalid",
    )
    return target


def main() -> int:
    args = _parse_args()
    repo = args.repo.resolve()
    code_revision = _require_clean_repository(repo)
    protocol_path = args.protocol.resolve()
    protocol = load_selective_virtual_sensing_protocol(protocol_path)
    record = _case(protocol_path, args.object_id, args.episode_id)
    case_name = str(record["case"])
    selection_path = args.selection_seal.resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    source_config = _validate_source_config(args.source_config.resolve(), selection)
    source_row = dynamic_window_source_case(selection, case_name)
    _require(
        source_config["source_cohort"]["protocol_config_sha256"]
        == protocol["config_sha256"],
        "source config references another base protocol",
    )

    prediction_dir = args.prediction_root.resolve() / case_name
    prediction_seal_path = prediction_dir / VIRTUAL_SENSING_SEAL_FILENAME
    prediction_seal = json.loads(prediction_seal_path.read_text(encoding="utf-8"))
    validate_selective_prediction_seal(
        prediction_seal,
        protocol_path=protocol_path,
        prediction_dir=prediction_dir,
    )
    staged_case = args.staged_case_dir.resolve()
    _require(staged_case.name == case_name, "staged case identity changed")
    prefix_manifest_path = staged_case / "prediction_prefix_manifest.json"
    _validate_prefix(
        prefix_manifest_path,
        protocol=protocol,
        record=record,
        source_row=source_row,
        selection_seal=selection_path,
    )
    prediction_archive = prediction_dir / VIRTUAL_SENSING_ARCHIVE_FILENAME
    prediction_report_path = prediction_dir / VIRTUAL_SENSING_REPORT_FILENAME
    prediction_report = json.loads(prediction_report_path.read_text(encoding="utf-8"))
    with np.load(prediction_archive, allow_pickle=False) as stored:
        selected_cameras = np.asarray(stored["selected_cameras"]).astype(str).tolist()
    _require(
        len(selected_cameras) == 8 and len(set(selected_cameras)) == 8,
        "sealed prediction camera panel changed",
    )
    source_episode = (
        args.source_aligned_root.resolve()
        / args.object_id
        / f"episode_{args.episode_id:04d}"
    )
    source_manifest_path = source_episode / "selective_source_preparation_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    _require(
        source_manifest.get("result_sha256") == _canonical_sha256(source_manifest)
        and all(source_manifest.get(key) == value for key, value in record.items()),
        "source preparation changed",
    )

    stage = _load_script(
        repo / "scripts" / "remote" / "stage_deform360_selective_authorized_future.py",
        "deform360_dynamic_window_frozen_stage_helpers",
    )
    outcome = _load_script(
        repo / "scripts" / "remote" / "build_deform360_selective_authorized_outcome.py",
        "deform360_dynamic_window_frozen_outcome_helpers",
    )
    deform360_repo = args.deform360_repo.resolve()
    _require(
        _git_revision(deform360_repo) == outcome.DEFORM360_REVISION,
        "Deform360 revision changed",
    )
    for name, expected in outcome.SOURCE_SHA256.items():
        path = deform360_repo / "deform360" / "processing" / f"{name}.py"
        _require(_sha256(path) == expected, f"official {name} changed")
    outcome._validate_runtime_constants()
    checkpoint = args.tracking_checkpoint.resolve()
    _require(
        _sha256(checkpoint) == outcome.TRACKING_CHECKPOINT_SHA256,
        "CoTracker checkpoint changed",
    )
    cotracker = args.cotracker_repository.resolve()
    _require(
        _git_revision(cotracker) == outcome.COTRACKER_REVISION
        and subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=cotracker,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        == outcome.COTRACKER_TREE
        and _sha256(cotracker / "cotracker" / "predictor.py")
        == outcome.COTRACKER_PREDICTOR_SHA256,
        "CoTracker source checkout changed",
    )
    sys.path.insert(0, str(cotracker))
    sam2_repo = args.sam2_repository.resolve()
    sam2_checkpoint = args.sam2_checkpoint.resolve()
    _require(
        _git_revision(sam2_repo) == stage.SAM2_REPOSITORY_REVISION
        and _sha256(sam2_checkpoint) == stage.SAM2_CHECKPOINT_SHA256,
        "SAM2 runtime changed",
    )
    start, stop = source_row["translation_contact_v2"][
        "selected_raw_frame_range_half_open"
    ]
    frame_zero_manifest_path = staged_case / "frame_zero_reconstruction_manifest.json"
    work_case = args.work_root.resolve() / case_name
    source_future = _stage_source_future(
        stage=stage,
        destination=work_case,
        source_episode=source_episode,
        staged_case=staged_case,
        selected_cameras=selected_cameras,
        start=int(start),
        stop=int(stop),
        record=record,
        protocol=protocol,
        prediction_seal=prediction_seal,
        prediction_archive=prediction_archive,
        prefix_manifest_path=prefix_manifest_path,
        source_manifest_path=source_manifest_path,
        selection_seal=selection_path,
        frame_zero_manifest_path=frame_zero_manifest_path,
        sam2_repository=sam2_repo,
        sam2_checkpoint=sam2_checkpoint,
        generic_selector_source=args.generic_selector_source.resolve(),
        device=args.device,
    )
    target = _build_target(outcome, work_case, selected_cameras, checkpoint)
    with np.load(prediction_archive, allow_pickle=False) as stored:
        trajectories = {
            arm: np.asarray(stored[key]).copy()
            for arm, key in ARM_TO_ARCHIVE_KEY.items()
        }
        center_ids = np.asarray(stored["center_ids"], dtype=np.int64)
    _require(
        np.array_equal(target[0], trajectories["persistence"][0]),
        "source target material identity changed",
    )
    visibility = np.ones(target.shape[:2], dtype=bool)
    validity = np.ones(target.shape[:2], dtype=bool)
    scores = score_selective_virtual_sensing_arrays(
        trajectories,
        target,
        visibility,
        validity,
        center_ids=center_ids,
    )
    measurement_dir = args.measurement_root.resolve() / case_name
    audit = _measurement_target_audit(
        measurement_dir, prediction_report, target, center_ids
    )
    scored = np.asarray(SCORED_FRAMES, dtype=np.int64)
    displacement = target[scored] - target[0][None]
    target_motion_rmse = float(np.sqrt(np.mean(np.sum(displacement**2, axis=-1))))
    primary_arm = next(iter(ARM_TO_ARCHIVE_KEY))
    identity_key = "post_update_hidden_identity_rmse_m"
    chamfer_key = "post_update_hidden_symmetric_chamfer_m"
    persistence_identity = float(scores["persistence"][identity_key])
    persistence_chamfer = float(scores["persistence"][chamfer_key])
    primary_identity = float(scores[primary_arm][identity_key])
    primary_chamfer = float(scores[primary_arm][chamfer_key])

    output = args.output_root.resolve() / case_name
    _require(not output.exists(), f"source evaluation already exists: {output}")
    scratch = output.with_name(f".{output.name}.incomplete-{os.getpid()}")
    scratch.mkdir(parents=True)
    try:
        target_archive = scratch / TARGET_ARCHIVE_FILENAME
        np.savez_compressed(
            target_archive,
            target_m=target,
            target_visibility=visibility,
            target_validity=validity,
        )
        payload: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": "Deform360DynamicWindowSourceEvaluation",
            "protocol_id": source_config["protocol_id"],
            "protocol_config_sha256": protocol["config_sha256"],
            **record,
            "code_revision": code_revision,
            "raw_frame_range_half_open": [int(start), int(stop)],
            "selected_cameras": selected_cameras,
            "material_point_count": int(target.shape[1]),
            "material_identity_sha256": _array_sha256(target[0]),
            "target_motion_rmse_m": target_motion_rmse,
            "scores": scores,
            "primary_comparison": {
                "arm": primary_arm,
                "persistence_identity_rmse_m": persistence_identity,
                "arm_identity_rmse_m": primary_identity,
                "identity_change_percent": 100.0
                * (primary_identity / persistence_identity - 1.0),
                "persistence_chamfer_m": persistence_chamfer,
                "arm_chamfer_m": primary_chamfer,
                "chamfer_change_percent": 100.0
                * (primary_chamfer / persistence_chamfer - 1.0),
            },
            "raw_measurement_target_open_audit": audit,
            "inputs_sha256": {
                "source_config": _sha256(args.source_config.resolve()),
                "selection_seal": _sha256(selection_path),
                "prediction_seal": _sha256(prediction_seal_path),
                "prediction_archive": _sha256(prediction_archive),
                "prediction_report": _sha256(prediction_report_path),
                "measurement_manifest": _sha256(measurement_dir / MANIFEST_FILENAME),
                "measurement_archive": _sha256(measurement_dir / MEASUREMENT_FILENAME),
                "source_future_manifest": _sha256(
                    work_case / SOURCE_FUTURE_MANIFEST_FILENAME
                ),
            },
            "output": {
                "target_archive": str(output / TARGET_ARCHIVE_FILENAME),
                "target_archive_sha256": _sha256(target_archive),
                "target_array_sha256": _array_sha256(target),
            },
            "authorization": source_future["authorization"],
            "information_boundary": {
                "prediction_verified_before_future_read": True,
                "exhausted_source_case": True,
                "fresh_objects_or_reserved_targets_read": False,
                "future_tactile_exposed_to_prediction_method": False,
                "target_metric_used_for_window_selection": False,
            },
            "claim_boundary": (
                "Exploratory source-window diagnosis after the original cohort was "
                "opened; not a prospective or state-of-the-art claim."
            ),
        }
        payload["result_sha256"] = _canonical_sha256(payload)
        (scratch / SOURCE_EVALUATION_FILENAME).write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(scratch, output)
    except Exception:
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
