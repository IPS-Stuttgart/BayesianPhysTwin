#!/usr/bin/env python3
"""Run one target-free Deform360 action-response source certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.action_response_admission import (
    ActionResponseAdmissionConfig,
    evaluate_action_response_admission,
)
from bayesian_phystwin.deform360_raw_camera_observation import (
    AllTrackerPrefixRuntime,
    RawCameraObservationConfig,
    _array_sha256,
    _load_calibration,
    _projection_matrix,
    _sha256,
    frame_zero_camera_support,
    select_frame_zero_observation_plan,
    triangulate_observation_ransac,
)
from bayesian_phystwin.grouped_multiview_observation import (
    GroupedMultiviewConfig,
    partition_disjoint_camera_groups,
    triangulate_disjoint_camera_groups,
)

PROTOCOL_ID = "deform360-action-response-source-v1"
EXPECTED_CASE = "059-shoe-ep0000"
UPDATE_FRAMES = (19, 38, 57)


def _canonical_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _controller_centroid_m(controller_points_m: np.ndarray) -> np.ndarray:
    controller = np.asarray(controller_points_m, dtype=np.float64)
    if controller.ndim != 3 or controller.shape[2] != 3:
        raise ValueError("controller points must have shape (T, A, 3)")
    finite = np.all(np.isfinite(controller), axis=2)
    centroid: np.ndarray = np.full(
        (len(controller), 3), np.nan, dtype=np.float64
    )
    for frame in range(len(controller)):
        if not np.any(finite[frame]):
            continue
        centroid[frame] = np.median(controller[frame, finite[frame]], axis=0)
    if not np.all(np.isfinite(centroid)):
        raise ValueError("controller trajectory has an empty frame")
    return centroid[:, None, :]


def _validate_physical_manifest(manifest: dict[str, Any], case: str) -> None:
    if manifest.get("artifact_kind") != (
        "Deform360DynamicTAPNextPPPhysicalBackbone"
    ):
        raise ValueError("unsupported physical manifest")
    if manifest.get("case") != case or manifest.get("partition") != "source":
        raise ValueError("physical manifest is outside the frozen source case")
    boundary = manifest.get("information_boundary", {})
    if not (
        boundary.get("future_object_geometry_read") is False
        and boundary.get("future_object_track_read") is False
        and boundary.get("known_future_robot_action_read") is True
        and boundary.get("provider_outcome_or_metric_read") is False
        and boundary.get("held_v8_target_query_score_barrier_or_outcome_access")
        is False
    ):
        raise ValueError("physical manifest crossed the source information boundary")


def _eligible_prefix_calibration(
    processed_dir: Path,
    intrinsics: Mapping[str, Any],
    extrinsics: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...]]:
    required_files = (
        "undistorted.mp4",
        "mask_refined.h5",
        "rendered_depth.h5",
    )
    cameras = tuple(
        sorted(
            camera
            for camera in set(intrinsics) & set(extrinsics)
            if all(
                (processed_dir / camera / filename).is_file()
                for filename in required_files
            )
        )
    )
    if len(cameras) < 2:
        raise ValueError("fewer than two cameras have complete prefix assets")
    return (
        {camera: intrinsics[camera] for camera in cameras},
        {camera: extrinsics[camera] for camera in cameras},
        cameras,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    physical_dir = Path(args.physical_dir).resolve()
    processed_dir = Path(args.processed_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    case = str(args.case)
    if case != EXPECTED_CASE:
        raise ValueError("case differs from the frozen source smoke")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    physical_manifest_path = (
        physical_dir / "sealed_physical" / "dynamic_tapnextpp_physical.json"
    )
    physical_archive_path = (
        physical_dir / "sealed_physical" / "dynamic_tapnextpp_physical.npz"
    )
    prediction_only_path = physical_dir / "prediction_only_input.pkl"
    prediction_only_summary_path = physical_dir / "prediction_only_input.json"
    physical_manifest = json.loads(
        physical_manifest_path.read_text(encoding="utf-8")
    )
    prediction_only_summary = json.loads(
        prediction_only_summary_path.read_text(encoding="utf-8")
    )
    _validate_physical_manifest(physical_manifest, case)
    if prediction_only_summary.get("information_boundary") != {
        "future_object_observations_present": False,
        "future_tactile_used": False,
        "object_observation_frames_used": [0],
    }:
        raise ValueError("prediction-only carrier crossed the object boundary")
    if (
        _sha256(physical_archive_path)
        != physical_manifest["physical_archive"]["file_sha256"]
    ):
        raise ValueError("physical archive digest changed")
    if (
        _sha256(prediction_only_path)
        != physical_manifest["inputs_sha256"]["prediction_only_input"]
    ):
        raise ValueError("prediction-only carrier digest changed")
    with np.load(physical_archive_path, allow_pickle=False) as stored:
        physical = np.asarray(stored["physical_prediction_m"], dtype=np.float64)
        frame_zero = np.asarray(stored["frame_zero_points_m"], dtype=np.float64)
        action_support = np.asarray(stored["action_support"], dtype=np.float64)
    with prediction_only_path.open("rb") as handle:
        prediction_only = pickle.load(handle)
    controller = _controller_centroid_m(prediction_only["controller_points"])
    if len(controller) != len(physical):
        raise ValueError("controller and physical frame counts differ")

    raw_config = RawCameraObservationConfig(
        center_count=16,
        selected_camera_count=8,
        update_frames=UPDATE_FRAMES,
    )
    grouped_config = GroupedMultiviewConfig()
    admission_config = ActionResponseAdmissionConfig()
    runtime = AllTrackerPrefixRuntime(
        args.alltracker_source,
        args.checkpoint,
        device=args.device,
        config=raw_config,
    )
    intrinsics, extrinsics = _load_calibration(processed_dir)
    intrinsics, extrinsics, eligible_cameras = _eligible_prefix_calibration(
        processed_dir,
        intrinsics,
        extrinsics,
    )
    cameras, support, projected = frame_zero_camera_support(
        frame_zero,
        processed_dir,
        intrinsics,
        extrinsics,
        depth_tolerance_m=raw_config.frame_zero_depth_tolerance_m,
    )
    plan = select_frame_zero_observation_plan(
        frame_zero,
        cameras,
        support,
        projected,
        extrinsics,
        config=raw_config,
    )
    centers = np.asarray(plan["center_ids"], dtype=np.int64)
    selected_cameras = tuple(plan["selected_cameras"])
    projection_matrices = {
        camera: _projection_matrix(intrinsics[camera], extrinsics[camera])
        for camera in selected_cameras
    }
    camera_origins = {
        camera: np.asarray(extrinsics[camera], dtype=np.float64)[:3, 3]
        for camera in selected_cameras
    }
    camera_groups = partition_disjoint_camera_groups(
        selected_cameras,
        camera_origins,
        frame_zero[centers],
        config=grouped_config,
    )
    sampled_frames = np.asarray((0, *UPDATE_FRAMES), dtype=np.int64)
    grouped_shape = (
        grouped_config.group_count,
        len(sampled_frames),
        len(centers),
    )
    grouped_points: np.ndarray = np.full(
        (*grouped_shape, 3), np.nan, dtype=np.float64
    )
    grouped_valid: np.ndarray = np.zeros(grouped_shape, dtype=bool)
    grouped_covariance: np.ndarray = np.full(
        (*grouped_shape, 3, 3), np.nan, dtype=np.float64
    )
    grouped_reliability: np.ndarray = np.zeros(
        grouped_shape, dtype=np.float64
    )
    grouped_association: np.ndarray = np.zeros(
        grouped_shape, dtype=np.float64
    )
    grouped_points[:, 0] = frame_zero[centers]
    grouped_valid[:, 0] = True
    grouped_covariance[:, 0] = grouped_config.covariance_floor_m2 * np.eye(3)
    grouped_reliability[:, 0] = 1.0
    grouped_association[:, 0] = 1.0
    tracker_records: list[dict[str, Any]] = []
    group_support: list[list[int]] = []
    for sample_index, update_frame in enumerate(UPDATE_FRAMES, start=1):
        tracks_by_camera: dict[str, dict[int, np.ndarray]] = {}
        update_tracker_records: list[dict[str, Any]] = []
        for camera in selected_cameras:
            query_ids = np.asarray(plan["query_ids"][camera], dtype=np.int64)
            query_pixels = np.asarray(plan["query_pixels"][camera], dtype=np.float64)
            tracks, visible, record = runtime.track_prefix(
                processed_dir / camera / "undistorted.mp4",
                query_pixels,
                update_frame,
            )
            tracks_by_camera[camera] = {
                int(point_id): tracks[index]
                for index, point_id in enumerate(query_ids)
                if visible[index]
            }
            update_tracker_records.append(
                {
                    **record,
                    "camera": camera,
                    "query_ids": query_ids.tolist(),
                }
            )

        def grouped_triangulator(
            observations: Mapping[str, np.ndarray],
            initial_point: np.ndarray,
        ) -> tuple[np.ndarray | None, Mapping[str, Any]]:
            return triangulate_observation_ransac(
                observations,
                projection_matrices,
                camera_origins,
                initial_point,
                config=raw_config,
            )

        grouped = triangulate_disjoint_camera_groups(
            tracks_by_camera,
            centers,
            frame_zero,
            camera_groups,
            projection_matrices,
            grouped_triangulator,
            config=grouped_config,
        )
        grouped_points[:, sample_index] = grouped.points_m
        grouped_valid[:, sample_index] = grouped.valid
        grouped_covariance[:, sample_index] = grouped.covariance_m2
        grouped_reliability[:, sample_index] = grouped.prior_reliability
        grouped_association[:, sample_index] = grouped.association_probability
        group_support.append(
            [int(np.sum(grouped.valid[group])) for group in range(len(camera_groups))]
        )
        tracker_records.append(
            {
                "update_frame": update_frame,
                "tracker": update_tracker_records,
            }
        )
    observation_prefix_id = _array_sha256(grouped_points)
    admission = evaluate_action_response_admission(
        physical[sampled_frames][:, centers],
        grouped_points,
        grouped_valid,
        grouped_covariance,
        grouped_reliability,
        grouped_association,
        controller[sampled_frames],
        tuple(f"disjoint-camera-group-{index}" for index in range(len(camera_groups))),
        tuple(f"node-{int(center)}" for center in centers),
        action_support[centers],
        physical_prefix_id=physical_manifest["physical_archive"]["array_sha256"][
            "physical_prediction_m"
        ],
        observation_prefix_id=observation_prefix_id,
        action_prefix_id=prediction_only_summary["controller_trajectory_sha256"],
        config=admission_config,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    observation_path = output_dir / "grouped_observation.npz"
    np.savez_compressed(
        observation_path,
        sampled_frames=sampled_frames,
        center_ids=centers,
        physical_positions_m=physical[sampled_frames][:, centers],
        observed_positions_m=grouped_points,
        observation_validity=grouped_valid,
        observation_covariance_m2=grouped_covariance,
        prior_reliability=grouped_reliability,
        association_probability=grouped_association,
        actuator_positions_m=controller[sampled_frames],
        action_support=action_support[centers],
        sensor_group_ids=np.asarray(
            [
                f"disjoint-camera-group-{index}"
                for index in range(len(camera_groups))
            ]
        ),
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "case": case,
        "repository_revision": args.repository_revision,
        "configs": {
            "raw_camera": asdict(raw_config),
            "grouped_multiview": asdict(grouped_config),
            "action_response_admission": asdict(admission_config),
        },
        "eligible_prefix_cameras": list(eligible_cameras),
        "camera_groups": [list(group) for group in camera_groups],
        "group_support_by_update": group_support,
        "admission": admission.to_dict(),
        "tracker_records": tracker_records,
        "inputs_sha256": {
            "physical_manifest": _sha256(physical_manifest_path),
            "physical_archive": _sha256(physical_archive_path),
            "prediction_only_carrier": _sha256(prediction_only_path),
            "prediction_only_summary": _sha256(prediction_only_summary_path),
            "alltracker_source": runtime.source_sha256,
            "alltracker_checkpoint": runtime.checkpoint_sha256,
        },
        "output": {
            "grouped_observation": str(observation_path),
            "grouped_observation_sha256": _sha256(observation_path),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "device": runtime.device_name,
        },
        "information_boundary": {
            "future_object_observation_read": False,
            "future_hidden_identity_read": False,
            "future_target_metric_read": False,
            "maximum_rgb_frame_read": max(UPDATE_FRAMES),
            "known_action_read": True,
            "held_v8_artifact_read": False,
            "v1_sealed_target_read": False,
        },
        "claim_boundary": (
            "already-open source admission smoke only; no hidden outcome, "
            "accuracy, calibration, or state-of-the-art claim"
        ),
    }
    report["result_sha256"] = _canonical_sha256(report)
    report_path = output_dir / "source_admission_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default=EXPECTED_CASE)
    parser.add_argument("--physical-dir", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--alltracker-source", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repository-revision", required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report["admission"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
