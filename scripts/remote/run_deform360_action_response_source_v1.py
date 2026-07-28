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
    project_world_points,
    select_frame_zero_observation_plan,
    triangulate_observation_ransac,
)
from bayesian_phystwin.grouped_multiview_observation import (
    GroupedMultiviewConfig,
    partition_disjoint_camera_groups,
    partition_supported_disjoint_camera_groups,
    select_balanced_group_point_ids,
    triangulate_disjoint_camera_groups,
)
from bayesian_phystwin.projected_view_response import (
    ProjectedViewResponseConfig,
    build_projected_view_response,
)

LEGACY_PLANNER = "legacy-global-centers-v1"
BALANCED_PLANNER = "balanced-physical-response-v2"
PROJECTED_VIEW_PLANNER = "projected-view-response-v3"
PROTOCOL_IDS = {
    LEGACY_PLANNER: "deform360-action-response-source-v1",
    BALANCED_PLANNER: "deform360-action-response-source-v2",
    PROJECTED_VIEW_PLANNER: "deform360-action-response-source-v3",
}
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


def _named_arrays_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(np.asarray(arrays[name]))
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


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


def _group_frame_zero_support_counts(
    support: np.ndarray,
    center_ids: np.ndarray,
    cameras: tuple[str, ...],
    camera_groups: tuple[tuple[str, ...], ...],
    *,
    minimum_cameras_per_group: int,
) -> list[int]:
    support_array = np.asarray(support, dtype=bool)
    centers = np.asarray(center_ids, dtype=np.int64)
    camera_index = {camera: index for index, camera in enumerate(cameras)}
    return [
        int(
            np.sum(
                np.sum(
                    support_array[
                        np.ix_(
                            centers,
                            np.asarray(
                                [camera_index[camera] for camera in group],
                                dtype=np.int64,
                            ),
                        )
                    ],
                    axis=1,
                )
                >= minimum_cameras_per_group
            )
        )
        for group in camera_groups
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    physical_dir = Path(args.physical_dir).resolve()
    processed_dir = Path(args.processed_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    case = str(args.case)
    planner = str(args.planner)
    if case != EXPECTED_CASE:
        raise ValueError("case differs from the frozen source smoke")
    if planner not in PROTOCOL_IDS:
        raise ValueError("unknown frame-zero planner")
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
    projected_config = ProjectedViewResponseConfig()
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
    sampled_frames = np.asarray((0, *UPDATE_FRAMES), dtype=np.int64)
    camera_origins = {
        camera: np.asarray(extrinsics[camera], dtype=np.float64)[:3, 3]
        for camera in cameras
    }
    if planner == LEGACY_PLANNER:
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
        selected_camera_origins = {
            camera: camera_origins[camera] for camera in selected_cameras
        }
        planning_camera_groups = partition_disjoint_camera_groups(
            selected_cameras,
            selected_camera_origins,
            frame_zero[centers],
            config=grouped_config,
        )
        camera_groups = planning_camera_groups
        response_support = action_support[centers]
    else:
        maximum_physical_response = np.max(
            np.linalg.norm(
                physical[sampled_frames] - physical[sampled_frames[:1]],
                axis=2,
            ),
            axis=0,
        )
        eligible_response = (
            maximum_physical_response
            >= admission_config.minimum_identifiable_physical_rms_m
        )
        planning_camera_groups = partition_supported_disjoint_camera_groups(
            cameras,
            camera_origins,
            frame_zero,
            support,
            eligible_response,
            config=grouped_config,
        )
        centers = select_balanced_group_point_ids(
            frame_zero,
            cameras,
            support,
            planning_camera_groups,
            eligible_response,
            count=raw_config.center_count,
            minimum_per_group=admission_config.minimum_supported_cluster_count,
            minimum_cameras_per_group=grouped_config.minimum_cameras_per_group,
        )
        camera_index = {
            camera: index for index, camera in enumerate(cameras)
        }
        planned_cameras = tuple(
            camera for group in planning_camera_groups for camera in group
        )
        if planner == PROJECTED_VIEW_PLANNER:
            selected_cameras = tuple(
                camera
                for camera in planned_cameras
                if int(np.sum(support[centers, camera_index[camera]]))
                >= admission_config.minimum_supported_cluster_count
            )
            if (
                len(selected_cameras)
                < admission_config.minimum_independent_group_count
            ):
                raise ValueError(
                    "too few cameras support projected response admission"
                )
            camera_groups = tuple(
                (camera,) for camera in selected_cameras
            )
        else:
            selected_cameras = planned_cameras
            camera_groups = planning_camera_groups
        plan = {
            "query_ids": {
                camera: centers[support[centers, camera_index[camera]]]
                for camera in selected_cameras
            },
            "query_pixels": {
                camera: projected[camera][
                    centers[support[centers, camera_index[camera]]]
                ]
                for camera in selected_cameras
            },
        }
        response_support = np.ones(len(centers), dtype=np.float64)
    projection_matrices = {
        camera: _projection_matrix(intrinsics[camera], extrinsics[camera])
        for camera in selected_cameras
    }
    projected_mode = planner == PROJECTED_VIEW_PLANNER
    grouped_points: np.ndarray | None = None
    grouped_valid: np.ndarray | None = None
    grouped_covariance: np.ndarray | None = None
    grouped_reliability: np.ndarray | None = None
    grouped_association: np.ndarray | None = None
    physical_pixels: np.ndarray | None = None
    observed_pixels: np.ndarray | None = None
    projected_valid: np.ndarray | None = None
    cycle_error: np.ndarray | None = None
    source_confidence: np.ndarray | None = None
    initial_depth: np.ndarray | None = None
    focal_lengths: np.ndarray | None = None
    center_index = {
        int(point_id): index for index, point_id in enumerate(centers)
    }
    if projected_mode:
        projected_shape = (
            len(selected_cameras),
            len(sampled_frames),
            len(centers),
        )
        physical_pixels = np.full(
            (*projected_shape, 2),
            np.nan,
            dtype=np.float64,
        )
        observed_pixels = np.full_like(physical_pixels, np.nan)
        projected_valid = np.zeros(projected_shape, dtype=bool)
        cycle_error = np.full(projected_shape, np.nan, dtype=np.float64)
        source_confidence = np.zeros(projected_shape, dtype=np.float64)
        initial_depth = np.full(
            (len(selected_cameras), len(centers)),
            projected_config.minimum_initial_depth_m,
            dtype=np.float64,
        )
        focal_lengths = np.zeros(
            (len(selected_cameras), 2),
            dtype=np.float64,
        )
        for sensor_index, camera in enumerate(selected_cameras):
            focal_lengths[sensor_index] = (
                float(np.asarray(intrinsics[camera])[0, 0]),
                float(np.asarray(intrinsics[camera])[1, 1]),
            )
            for sample_index, frame in enumerate(sampled_frames):
                pixels, depth = project_world_points(
                    physical[int(frame), centers],
                    np.asarray(intrinsics[camera]),
                    np.asarray(extrinsics[camera]),
                )
                physical_pixels[sensor_index, sample_index] = pixels
                if sample_index == 0:
                    initial_depth[sensor_index] = np.where(
                        np.isfinite(depth),
                        np.maximum(
                            depth,
                            projected_config.minimum_initial_depth_m,
                        ),
                        projected_config.minimum_initial_depth_m,
                    )
            query_ids = np.asarray(
                plan["query_ids"][camera],
                dtype=np.int64,
            )
            query_pixels = np.asarray(
                plan["query_pixels"][camera],
                dtype=np.float64,
            )
            local_ids = np.asarray(
                [center_index[int(point_id)] for point_id in query_ids],
                dtype=np.int64,
            )
            observed_pixels[sensor_index, 0, local_ids] = query_pixels
            projected_valid[sensor_index, 0, local_ids] = True
            cycle_error[sensor_index, 0, local_ids] = 0.0
            source_confidence[sensor_index, 0, local_ids] = 1.0
    else:
        grouped_shape = (
            grouped_config.group_count,
            len(sampled_frames),
            len(centers),
        )
        grouped_points = np.full(
            (*grouped_shape, 3), np.nan, dtype=np.float64
        )
        grouped_valid = np.zeros(grouped_shape, dtype=bool)
        grouped_covariance = np.full(
            (*grouped_shape, 3, 3), np.nan, dtype=np.float64
        )
        grouped_reliability = np.zeros(
            grouped_shape, dtype=np.float64
        )
        grouped_association = np.zeros(
            grouped_shape, dtype=np.float64
        )
        grouped_points[:, 0] = frame_zero[centers]
        grouped_valid[:, 0] = True
        grouped_covariance[:, 0] = (
            grouped_config.covariance_floor_m2 * np.eye(3)
        )
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
            reverse_record: dict[str, Any] | None = None
            if projected_mode:
                if (
                    observed_pixels is None
                    or projected_valid is None
                    or cycle_error is None
                    or source_confidence is None
                ):
                    raise AssertionError(
                        "projected-view observation arrays are incomplete"
                    )
                recovered, reverse_visible, reverse_record = (
                    runtime.track_reversed_prefix(
                        processed_dir / camera / "undistorted.mp4",
                        tracks,
                        update_frame,
                    )
                )
                sensor_index = selected_cameras.index(camera)
                local_ids = np.asarray(
                    [center_index[int(point_id)] for point_id in query_ids],
                    dtype=np.int64,
                )
                height, width = record["original_image_shape"]
                inside = (
                    np.all(np.isfinite(tracks), axis=1)
                    & (tracks[:, 0] >= 0.0)
                    & (tracks[:, 0] < float(width))
                    & (tracks[:, 1] >= 0.0)
                    & (tracks[:, 1] < float(height))
                )
                accepted = visible & reverse_visible & inside
                observed_pixels[
                    sensor_index,
                    sample_index,
                    local_ids,
                ] = tracks
                projected_valid[
                    sensor_index,
                    sample_index,
                    local_ids,
                ] = accepted
                cycle_error[
                    sensor_index,
                    sample_index,
                    local_ids,
                ] = np.linalg.norm(recovered - query_pixels, axis=1)
                source_confidence[
                    sensor_index,
                    sample_index,
                    local_ids,
                ] = accepted.astype(np.float64)
            tracks_by_camera[camera] = {
                int(point_id): tracks[index]
                for index, point_id in enumerate(query_ids)
                if visible[index]
            }
            tracker_payload = {
                **record,
                "camera": camera,
                "query_ids": query_ids.tolist(),
            }
            if reverse_record is not None:
                tracker_payload["reverse_tracker"] = reverse_record
            update_tracker_records.append(tracker_payload)

        if projected_mode:
            if projected_valid is None:
                raise AssertionError("projected validity is missing")
            group_support.append(
                [
                    int(np.sum(projected_valid[sensor, sample_index]))
                    for sensor in range(len(selected_cameras))
                ]
            )
            tracker_records.append(
                {
                    "update_frame": update_frame,
                    "tracker": update_tracker_records,
                }
            )
            continue

        def grouped_triangulator(
            observations: Mapping[str, np.ndarray],
            initial_point: np.ndarray,
        ) -> tuple[np.ndarray | None, Mapping[str, Any]]:
            return triangulate_observation_ransac(
                observations,
                projection_matrices,
                {
                    camera: camera_origins[camera]
                    for camera in selected_cameras
                },
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
        if (
            grouped_points is None
            or grouped_valid is None
            or grouped_covariance is None
            or grouped_reliability is None
            or grouped_association is None
        ):
            raise AssertionError("grouped observation arrays are incomplete")
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
    if projected_mode:
        if (
            physical_pixels is None
            or observed_pixels is None
            or projected_valid is None
            or cycle_error is None
            or source_confidence is None
            or initial_depth is None
            or focal_lengths is None
        ):
            raise AssertionError("projected-view response inputs are incomplete")
        projected_response = build_projected_view_response(
            physical_pixels,
            observed_pixels,
            projected_valid,
            initial_depth,
            focal_lengths,
            cycle_error,
            source_confidence,
            config=projected_config,
        )
        projected_physical_prefix_id = "sha256:" + _canonical_sha256(
            {
                "physical_prediction_array_sha256": physical_manifest[
                    "physical_archive"
                ]["array_sha256"]["physical_prediction_m"],
                "projected_physical_response_sha256": _array_sha256(
                    projected_response.physical_positions_m
                ),
                "intrinsics_sha256": _sha256(
                    processed_dir / "undistorted_intrinsics.npy"
                ),
                "extrinsics_sha256": _sha256(
                    processed_dir / "extrinsics.npy"
                ),
                "sampled_frames": sampled_frames.tolist(),
                "selected_cameras": list(selected_cameras),
            }
        )
        observation_prefix_id = "sha256:" + _named_arrays_sha256(
            {
                "association_probability": (
                    projected_response.association_probability
                ),
                "cycle_error_px": projected_response.cycle_error_px,
                "observation_covariance_m2": (
                    projected_response.observation_covariance_m2
                ),
                "observation_validity": (
                    projected_response.observation_validity
                ),
                "observed_positions_m": (
                    projected_response.observed_positions_m
                ),
                "prior_reliability": (
                    projected_response.prior_reliability
                ),
            }
        )
        admission = evaluate_action_response_admission(
            projected_response.physical_positions_m,
            projected_response.observed_positions_m,
            projected_response.observation_validity,
            projected_response.observation_covariance_m2,
            projected_response.prior_reliability,
            projected_response.association_probability,
            controller[sampled_frames],
            selected_cameras,
            tuple(f"node-{int(center)}" for center in centers),
            response_support,
            physical_prefix_id=projected_physical_prefix_id,
            observation_prefix_id=observation_prefix_id,
            action_prefix_id=prediction_only_summary[
                "controller_trajectory_sha256"
            ],
            config=admission_config,
        )
        output_dir.mkdir(parents=True, exist_ok=False)
        observation_path = output_dir / "projected_view_observation.npz"
        np.savez_compressed(
            observation_path,
            sampled_frames=sampled_frames,
            center_ids=centers,
            physical_pixels_px=physical_pixels,
            observed_pixels_px=observed_pixels,
            physical_positions_m=projected_response.physical_positions_m,
            observed_positions_m=projected_response.observed_positions_m,
            observation_validity=projected_response.observation_validity,
            observation_covariance_m2=(
                projected_response.observation_covariance_m2
            ),
            prior_reliability=projected_response.prior_reliability,
            association_probability=(
                projected_response.association_probability
            ),
            cycle_error_px=projected_response.cycle_error_px,
            actuator_positions_m=controller[sampled_frames],
            action_support=response_support,
            sensor_group_ids=np.asarray(selected_cameras),
        )
    else:
        if (
            grouped_points is None
            or grouped_valid is None
            or grouped_covariance is None
            or grouped_reliability is None
            or grouped_association is None
        ):
            raise AssertionError("grouped observation inputs are incomplete")
        observation_prefix_id = _array_sha256(grouped_points)
        admission = evaluate_action_response_admission(
            physical[sampled_frames][:, centers],
            grouped_points,
            grouped_valid,
            grouped_covariance,
            grouped_reliability,
            grouped_association,
            controller[sampled_frames],
            tuple(
                f"disjoint-camera-group-{index}"
                for index in range(len(camera_groups))
            ),
            tuple(f"node-{int(center)}" for center in centers),
            response_support,
            physical_prefix_id=physical_manifest["physical_archive"][
                "array_sha256"
            ]["physical_prediction_m"],
            observation_prefix_id=observation_prefix_id,
            action_prefix_id=prediction_only_summary[
                "controller_trajectory_sha256"
            ],
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
            action_support=response_support,
            sensor_group_ids=np.asarray(
                [
                    f"disjoint-camera-group-{index}"
                    for index in range(len(camera_groups))
                ]
            ),
        )
    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_IDS[planner],
        "case": case,
        "repository_revision": args.repository_revision,
        "frame_zero_planner": planner,
        "configs": {
            "raw_camera": asdict(raw_config),
            "grouped_multiview": asdict(grouped_config),
            "action_response_admission": asdict(admission_config),
        },
        "eligible_prefix_cameras": list(eligible_cameras),
        "camera_groups": [list(group) for group in camera_groups],
        "frame_zero_group_support": _group_frame_zero_support_counts(
            support,
            centers,
            cameras,
            camera_groups,
            minimum_cameras_per_group=(
                1
                if projected_mode
                else grouped_config.minimum_cameras_per_group
            ),
        ),
        "response_support_definition": (
            "legacy physical-provider action_support"
            if planner == LEGACY_PLANNER
            else (
                "binary selected-node support after sealed physical response "
                f">= {admission_config.minimum_identifiable_physical_rms_m} m"
            )
        ),
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
            "intrinsics": _sha256(
                processed_dir / "undistorted_intrinsics.npy"
            ),
            "extrinsics": _sha256(processed_dir / "extrinsics.npy"),
        },
        "output": (
            {
                "projected_view_observation": str(observation_path),
                "projected_view_observation_sha256": _sha256(
                    observation_path
                ),
            }
            if projected_mode
            else {
                "grouped_observation": str(observation_path),
                "grouped_observation_sha256": _sha256(observation_path),
            }
        ),
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
    if projected_mode:
        report["configs"]["projected_view_response"] = asdict(
            projected_config
        )
        report["planning_camera_groups"] = [
            list(group) for group in planning_camera_groups
        ]
        report["evidence_camera_groups"] = [
            list(group) for group in camera_groups
        ]
        report["projected_evidence_ids"] = {
            "physical_prefix_id": admission.physical_prefix_id,
            "observation_prefix_id": admission.observation_prefix_id,
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
    parser.add_argument(
        "--planner",
        choices=tuple(PROTOCOL_IDS),
        default=LEGACY_PLANNER,
    )
    parser.add_argument("--repository-revision", required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report["admission"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
