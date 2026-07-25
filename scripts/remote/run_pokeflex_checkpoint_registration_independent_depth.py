#!/usr/bin/env python3
"""Evaluate PokeFlex residual arms with causal independent-depth evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


sys.path.insert(0, str(_repository_root() / "src"))

from bayesian_phystwin.pokeflex_bayesian_registration import (  # noqa: E402
    PokeFlexActionGuardConfig,
    PokeFlexBayesianRegistrationConfig,
    pokeflex_action_contact_fields,
    pokeflex_correction_field_variants,
    pokeflex_force_supported_contact_fields,
    register_pokeflex_graph_posterior,
    voxel_cluster_centroids,
)
from bayesian_phystwin.pokeflex_independent_depth import (  # noqa: E402
    PokeFlexDepthCalibration,
    PokeFlexIndependentDepthAnchor,
    anchor_fit_scores_mm,
    build_independent_depth_anchor,
    calibrate_depth_translation,
    crop_points_to_geometry,
    realsense_depth_to_world_points,
    select_points_near_geometry,
)
from bayesian_phystwin.pokeflex_independent_depth_protocol import (  # noqa: E402
    load_pokeflex_independent_depth_protocol,
)
from bayesian_phystwin.pokeflex_registration_protocol import (  # noqa: E402
    load_pokeflex_registration_protocol,
)
from bayesian_phystwin.pokeflex_released_checkpoint import (  # noqa: E402
    PokeFlexReleasedCheckpoint,
)
from run_pokeflex_bayesian_registration_smoke import (  # noqa: E402
    _cd_ul1_mm,
    _load_mesh,
    _surface_sample,
    _template_frame,
    _view_points,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_official_template(
    path: Path, maximum_face_count: int = 19000
) -> tuple[np.ndarray, np.ndarray, dict[str, int | bool]]:
    """Apply the released VTK topology-preserving template simplification."""

    mesh = _load_mesh(path)
    input_vertex_count = len(mesh.vertices)
    input_face_count = len(mesh.faces)
    if input_face_count <= maximum_face_count:
        return (
            np.asarray(mesh.vertices, dtype=np.float64) / 1000.0,
            np.asarray(mesh.faces, dtype=np.int64),
            {
                "decimated": False,
                "input_vertex_count": input_vertex_count,
                "input_face_count": input_face_count,
                "output_vertex_count": input_vertex_count,
                "output_face_count": input_face_count,
            },
        )

    import pyvista as pv

    polydata = pv.read(path)
    reduction = 1.0 - maximum_face_count / polydata.n_cells
    polydata.decimate_pro(
        reduction=reduction,
        preserve_topology=True,
        inplace=True,
    )
    packed_faces = np.asarray(polydata.faces, dtype=np.int64)
    if len(packed_faces) % 4 != 0 or not np.all(packed_faces[::4] == 3):
        raise ValueError("official template simplification produced non-triangle faces")
    faces = packed_faces.reshape(-1, 4)[:, 1:]
    vertices = np.asarray(polydata.points, dtype=np.float64) / 1000.0
    return (
        vertices,
        faces,
        {
            "decimated": True,
            "input_vertex_count": input_vertex_count,
            "input_face_count": input_face_count,
            "output_vertex_count": len(vertices),
            "output_face_count": len(faces),
        },
    )


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_CD_UL1_mm": float(np.mean(array)),
        "median_CD_UL1_mm": float(np.median(array)),
        "p90_CD_UL1_mm": float(np.quantile(array, 0.9)),
    }


def _rms_field(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.square(value), axis=1))))


def _field_cosine(first: np.ndarray, second: np.ndarray) -> float | None:
    flat_first = np.asarray(first, dtype=np.float64).reshape(-1)
    flat_second = np.asarray(second, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(flat_first) * np.linalg.norm(flat_second))
    if denominator <= 1e-12:
        return None
    return float(np.dot(flat_first, flat_second) / denominator)


def _observation_fit_score_mm(
    vertices_m: np.ndarray,
    observation_views_m: tuple[np.ndarray, ...],
) -> tuple[float, ...]:
    """Return robust, equal-view fit scores for online baseline regret."""

    from scipy.spatial import cKDTree

    tree = cKDTree(vertices_m)
    scores = []
    for view in observation_views_m:
        clustered = voxel_cluster_centroids(view, 0.004)
        if len(clustered) > 128:
            indices = np.linspace(0, len(clustered) - 1, 128, dtype=np.int64)
            clustered = clustered[indices]
        distance = np.asarray(tree.query(clustered, k=1)[0], dtype=np.float64)
        cutoff = float(np.quantile(distance, 0.9))
        inliers = distance <= cutoff
        scores.append(float(1000.0 * np.mean(distance[inliers])))
    return tuple(scores)


def _realsense_parameters(root: Path, camera: int) -> tuple[dict[str, object], str]:
    path = root / "realsense" / str(camera) / "camera_parameters.json"
    return json.loads(path.read_text(encoding="utf-8")), _sha256(path)


def _realsense_world_points(
    root: Path,
    frame: int,
    camera: int,
    parameters: dict[str, object],
) -> np.ndarray:
    path = root / "realsense" / str(camera) / "depth" / f"{frame:05d}.png"
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(
            f"missing RealSense depth frame {frame} for camera {camera}"
        )
    extrinsics = parameters.get("extrinsics")
    if not isinstance(extrinsics, dict):
        raise ValueError("RealSense per-frame extrinsics are missing")
    frame_key = f"{frame:05d}"
    if frame_key not in extrinsics:
        raise ValueError(f"RealSense extrinsic is missing for frame {frame}")
    return realsense_depth_to_world_points(
        depth,
        np.asarray(parameters["depth_intrinsics"], dtype=np.float64),
        np.asarray(extrinsics[frame_key], dtype=np.float64),
        stride=2,
    )


def _realsense_anchor_inventory(
    root: Path,
    template_frame: int,
    template_vertices_m: np.ndarray,
    frame_limit: int,
    maximum_template_distance_m: float | None,
) -> tuple[
    dict[int, PokeFlexIndependentDepthAnchor],
    tuple[PokeFlexDepthCalibration, ...],
    tuple[str, ...],
]:
    parameters_and_hashes = tuple(
        _realsense_parameters(root, camera) for camera in (0, 1)
    )
    parameters = tuple(value[0] for value in parameters_and_hashes)
    calibration_hashes = tuple(value[1] for value in parameters_and_hashes)
    calibrations = []
    for camera in (0, 1):
        template_points = _realsense_world_points(
            root,
            template_frame,
            camera,
            parameters[camera],
        )
        calibrations.append(
            calibrate_depth_translation(template_points, template_vertices_m)
        )

    anchors = {}
    for frame in range(1, frame_limit):
        sensor_points = []
        for camera in (0, 1):
            points = _realsense_world_points(
                root,
                frame,
                camera,
                parameters[camera],
            )
            points = points + calibrations[camera].translation_m
            points = crop_points_to_geometry(
                points,
                template_vertices_m,
                padding_m=0.05,
            )
            if maximum_template_distance_m is not None:
                points = select_points_near_geometry(
                    points,
                    template_vertices_m,
                    maximum_distance_m=maximum_template_distance_m,
                )
            sensor_points.append(points)
        anchors[frame] = build_independent_depth_anchor(
            take_id=root.name,
            frame_id=frame,
            causal_cutoff_frame=frame,
            sensor_points_m=sensor_points,
            sensor_names=("realsense0", "realsense1"),
            calibration_sha256=calibration_hashes,
            sensor_variance_m2=(0.004**2, 0.004**2),
            voxel_size_m=0.004,
            maximum_clusters_per_sensor=256,
            metadata={
                "template_frame": template_frame,
                "calibration_median_residual_m": [
                    value.median_residual_m for value in calibrations
                ],
                "calibration_p90_residual_m": [
                    value.p90_residual_m for value in calibrations
                ],
                "maximum_template_distance_m": maximum_template_distance_m,
            },
        )
    return anchors, tuple(calibrations), calibration_hashes


def _candidate_name(field: str, scale: float) -> str:
    prefix = "checkpoint_residual" if field == "raw" else f"checkpoint_{field}_residual"
    return f"{prefix}_scale_{scale:g}"


def _correction_field_variants(
    source_prior: np.ndarray,
    target_prior: np.ndarray,
    correction: np.ndarray,
    names: tuple[str, ...],
    *,
    previous_correction: np.ndarray | None,
    tool_positions: np.ndarray,
    end_effector_positions: np.ndarray,
    force_vectors: np.ndarray,
) -> dict[str, np.ndarray]:
    available = pokeflex_correction_field_variants(
        source_prior,
        target_prior,
        correction,
        previous_correction_m=previous_correction,
    )
    available.update(
        pokeflex_action_contact_fields(
            source_prior,
            target_prior,
            correction,
            tool_positions,
            end_effector_positions,
        )
    )
    available.update(
        pokeflex_force_supported_contact_fields(
            source_prior,
            target_prior,
            correction,
            tool_positions,
            end_effector_positions,
            force_vectors,
        )
    )
    object_radius = float(
        np.max(np.linalg.norm(source_prior - source_prior.mean(axis=0), axis=1))
    )
    for radius_fraction in (0.25, 0.4, 0.55, 0.7):
        relative_fields = pokeflex_action_contact_fields(
            source_prior,
            target_prior,
            correction,
            tool_positions,
            end_effector_positions,
            influence_radius_m=radius_fraction * object_radius,
        )
        available[f"action_local_state_relative_{radius_fraction:g}"] = (
            relative_fields["action_local_state"]
        )
        relative_force_fields = pokeflex_force_supported_contact_fields(
            source_prior,
            target_prior,
            correction,
            tool_positions,
            end_effector_positions,
            force_vectors,
            influence_radius_m=radius_fraction * object_radius,
        )
        for field, value in relative_force_fields.items():
            available[f"{field}_relative_{radius_fraction:g}"] = value
    unknown = set(names) - set(available)
    if unknown:
        raise ValueError(f"unknown correction fields: {sorted(unknown)}")
    return {name: available[name] for name in names}


def run_smoke(
    take_root: Path,
    protocol_path: Path,
    upstream_checkout: Path,
    checkpoint_root: Path,
    *,
    correction_scales: tuple[float, ...],
    correction_fields: tuple[str, ...],
    residual_geometry: str,
    maximum_frame: int | None,
    include_frozen_action_guard: bool,
    record_online_observation_regret: bool,
    record_independent_anchor_regret: bool = False,
    independent_depth_protocol_path: Path | None = None,
    independent_anchor_maximum_template_distance_m: float | None = None,
) -> dict[str, object]:
    protocol = load_pokeflex_registration_protocol(protocol_path)
    development_objects = set(
        protocol["payload"]["cohort"]["development_objects"]
    )
    object_name, separator, take_number = take_root.name.rpartition("_T")
    if not separator or not take_number.isdigit() or object_name not in development_objects:
        raise ValueError(f"take is outside the locked development cohort: {take_root.name}")
    if 0.0 not in correction_scales:
        raise ValueError("correction scales must include exact checkpoint fallback 0")
    if not correction_fields:
        raise ValueError("at least one correction field is required")

    robot_path = take_root / "robot_data.json"
    robot_records = json.loads(robot_path.read_text(encoding="utf-8"))
    robot_by_frame = {int(record["frame"]): record for record in robot_records}
    active = [
        frame
        for frame, record in sorted(robot_by_frame.items())
        if float(record["forces"][1]) > 3.0
    ]
    template_frame = _template_frame(active)
    template_path = take_root / "meshes" / f"mesh-f{template_frame:05d}.obj"
    template_vertices, template_faces, template_preprocessing = _load_official_template(
        template_path
    )
    frame_limit = maximum_frame or max(robot_by_frame)
    valid_targets = sorted(frame for frame in active if 6 <= frame <= frame_limit)
    if not valid_targets:
        raise ValueError("smoke interval contains no causal target frames")

    checkpoint = PokeFlexReleasedCheckpoint.load(
        template_vertices,
        upstream_checkout=upstream_checkout,
        checkpoint_root=checkpoint_root,
    )
    views_by_frame: dict[int, tuple[np.ndarray, ...]] = {}
    features_by_frame: dict[int, object] = {}
    preprocessing_by_frame = {}
    for frame in range(1, frame_limit):
        views = tuple(
            _view_points(take_root, frame, camera, template_vertices)
            for camera in (0, 1)
        )
        feature, preprocessing = checkpoint.encode_frame(views)
        views_by_frame[frame] = views
        features_by_frame[frame] = feature
        preprocessing_by_frame[frame] = preprocessing

    predictions_by_frame = {}
    for frame in range(6, frame_limit + 1):
        history = range(frame - checkpoint.history_frame_count, frame)
        predictions_by_frame[frame] = checkpoint.predict_from_encoded_history(
            [features_by_frame[index] for index in history],
            [preprocessing_by_frame[index] for index in history],
        )

    independent_anchors: dict[int, PokeFlexIndependentDepthAnchor] = {}
    independent_calibrations: tuple[PokeFlexDepthCalibration, ...] = ()
    independent_calibration_hashes: tuple[str, ...] = ()
    independent_protocol = None
    if record_independent_anchor_regret:
        if independent_depth_protocol_path is None:
            raise ValueError("independent-depth protocol is required")
        independent_protocol = load_pokeflex_independent_depth_protocol(
            independent_depth_protocol_path
        )
        if protocol["protocol_sha256"] != independent_protocol["payload"][
            "parent_protocol"
        ]["protocol_sha256"]:
            raise ValueError("independent-depth parent protocol changed")
        (
            independent_anchors,
            independent_calibrations,
            independent_calibration_hashes,
        ) = _realsense_anchor_inventory(
            take_root,
            template_frame,
            template_vertices,
            frame_limit,
            independent_anchor_maximum_template_distance_m,
        )

    config = PokeFlexBayesianRegistrationConfig(residual_geometry=residual_geometry)
    updates_by_frame = {}
    corrections_by_frame: dict[int, np.ndarray] = {}
    for source_frame in range(6, frame_limit):
        source_prior = predictions_by_frame[source_frame].vertices_m
        action_supported = float(robot_by_frame[source_frame]["forces"][1]) > 3.0
        update = register_pokeflex_graph_posterior(
            source_prior,
            views_by_frame[source_frame],
            action_supported=action_supported,
            prior_faces=template_faces,
            config=config,
        )
        updates_by_frame[source_frame] = update
        corrections_by_frame[source_frame] = update.posterior_vertices_m - source_prior

    checkpoint_errors: list[float] = []
    template_errors: list[float] = []
    oracle_persistence: list[float] = []
    corrected_errors: dict[tuple[str, float], list[float]] = {
        (field, scale): []
        for field in correction_fields
        for scale in correction_scales
    }
    guarded_action_errors: list[float] = []
    target_records = []
    update_records = []
    previous_prediction_frame: int | None = None
    previous_candidate_predictions: dict[str, np.ndarray] = {}
    sample_count = int(protocol["payload"]["evaluation"]["sampling"]["surface_points"])
    base_seed = int(protocol["payload"]["evaluation"]["sampling"]["seed"])

    for target_frame in valid_targets:
        target_prior = predictions_by_frame[target_frame].vertices_m
        source_frame = target_frame - 1
        if source_frame in updates_by_frame:
            source_prior = predictions_by_frame[source_frame].vertices_m
            update = updates_by_frame[source_frame]
            action_supported = float(robot_by_frame[source_frame]["forces"][1]) > 3.0
            update_accepted = bool(update.accepted)
            correction = corrections_by_frame[source_frame]
            prior_motion = target_prior - source_prior
            previous_correction = corrections_by_frame.get(source_frame - 1)
            force_y = float(robot_by_frame[source_frame]["forces"][1])
            previous_force_y = float(robot_by_frame[source_frame - 1]["forces"][1])
            update_records.append(
                {
                    "source_frame": source_frame,
                    "target_frame": target_frame,
                    "accepted": update.accepted,
                    "reason": update.reason,
                    "action_supported": action_supported,
                    "rms_update_m": update.diagnostics.get("rms_update_m", 0.0),
                    "maximum_update_m": update.diagnostics.get(
                        "maximum_update_m", 0.0
                    ),
                    "associated_points": update.diagnostics.get(
                        "association_count", 0
                    ),
                    "camera_biases_m": update.camera_biases_m.tolist(),
                    "prior_motion_rms_m": _rms_field(prior_motion),
                    "correction_to_prior_motion_ratio": _rms_field(correction)
                    / max(_rms_field(prior_motion), 1e-12),
                    "correction_prior_motion_cosine": _field_cosine(
                        correction, prior_motion
                    ),
                    "previous_correction_cosine": (
                        _field_cosine(correction, previous_correction)
                        if previous_correction is not None
                        else None
                    ),
                    "force_y": force_y,
                    "force_y_delta": force_y - previous_force_y,
                    "effective_information_mass": update.diagnostics.get(
                        "effective_information_mass", 0.0
                    ),
                    "median_robust_weight": update.diagnostics.get(
                        "median_robust_weight", 0.0
                    ),
                    "downweighted_fraction": update.diagnostics.get(
                        "downweighted_fraction", 0.0
                    ),
                    "assignment_variance_m2_mean": update.diagnostics.get(
                        "assignment_variance_m2_mean", 0.0
                    ),
                    "condition_number": update.diagnostics.get(
                        "condition_number", 0.0
                    ),
                }
            )
        else:
            correction = np.zeros_like(target_prior)
            action_supported = False
            update_accepted = False
            update_records.append(
                {
                    "source_frame": source_frame,
                    "target_frame": target_frame,
                    "accepted": False,
                    "reason": "no-five-frame-source-prior",
                    "action_supported": False,
                    "rms_update_m": 0.0,
                    "maximum_update_m": 0.0,
                    "associated_points": 0,
                    "camera_biases_m": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    "prior_motion_rms_m": 0.0,
                    "correction_to_prior_motion_ratio": 0.0,
                    "correction_prior_motion_cosine": None,
                    "previous_correction_cosine": None,
                    "force_y": float(robot_by_frame[source_frame]["forces"][1]),
                    "force_y_delta": 0.0,
                    "effective_information_mass": 0.0,
                    "median_robust_weight": 0.0,
                    "downweighted_fraction": 0.0,
                    "assignment_variance_m2_mean": 0.0,
                    "condition_number": 0.0,
                }
            )

        correction_variants = _correction_field_variants(
            source_prior if source_frame in predictions_by_frame else target_prior,
            target_prior,
            correction,
            correction_fields,
            previous_correction=corrections_by_frame.get(source_frame - 1),
            tool_positions=np.asarray(
                [
                    robot_by_frame[frame]["T_WT"]
                    for frame in range(max(1, source_frame - 3), source_frame + 1)
                ],
                dtype=np.float64,
            )[:, :3, 3],
            end_effector_positions=np.asarray(
                [
                    robot_by_frame[frame]["T_WE"]
                    for frame in range(max(1, source_frame - 3), source_frame + 1)
                ],
                dtype=np.float64,
            )[:, :3, 3],
            force_vectors=np.asarray(
                [
                    robot_by_frame[frame]["forces"][:3]
                    for frame in range(max(1, source_frame - 3), source_frame + 1)
                ],
                dtype=np.float64,
            ),
        )
        if not update_accepted or not action_supported:
            for field in correction_variants:
                if field.startswith(("action_", "force_")):
                    correction_variants[field] = np.zeros_like(target_prior)
        action_guard = PokeFlexActionGuardConfig()
        guarded_action_scale = action_guard.selected_scale(
            float(robot_by_frame[source_frame]["forces"][1]),
            observation_update_accepted=update_accepted,
            action_supported=bool(action_supported),
        )
        online_observation_regret: dict[str, dict[str, object]] = {}
        independent_anchor_regret: dict[str, dict[str, object]] = {}
        if (
            record_online_observation_regret
            and previous_prediction_frame == source_frame
        ):
            baseline_scores = _observation_fit_score_mm(
                previous_candidate_predictions["released_checkpoint"],
                views_by_frame[source_frame],
            )
            for name, candidate in previous_candidate_predictions.items():
                if name == "released_checkpoint":
                    continue
                candidate_scores = _observation_fit_score_mm(
                    candidate,
                    views_by_frame[source_frame],
                )
                regret = np.asarray(candidate_scores) - np.asarray(baseline_scores)
                online_observation_regret[name] = {
                    "per_view_mm": regret.tolist(),
                    "mean_mm": float(np.mean(regret)),
                    "covariance_intersection_upper_mm": float(np.max(regret)),
                }
        if (
            record_independent_anchor_regret
            and previous_prediction_frame == source_frame
        ):
            anchor = independent_anchors[source_frame]
            baseline_scores = anchor_fit_scores_mm(
                previous_candidate_predictions["released_checkpoint"], anchor
            )
            for name, candidate in previous_candidate_predictions.items():
                if name == "released_checkpoint":
                    continue
                candidate_scores = anchor_fit_scores_mm(candidate, anchor)
                regret = candidate_scores - baseline_scores
                independent_anchor_regret[name] = {
                    "evaluated_prediction_frame": source_frame,
                    "per_sensor_mm": regret.tolist(),
                    "mean_mm": float(np.mean(regret)),
                    "covariance_intersection_upper_mm": float(np.max(regret)),
                }

        target_mesh = _load_mesh(
            take_root / "meshes" / f"mesh-f{target_frame:05d}.obj"
        )
        target_sample = _surface_sample(
            np.asarray(target_mesh.vertices, dtype=np.float64) / 1000.0,
            np.asarray(target_mesh.faces, dtype=np.int64),
            sample_count,
            base_seed + target_frame,
        )
        previous_mesh = _load_mesh(
            take_root / "meshes" / f"mesh-f{source_frame:05d}.obj"
        )
        previous_sample = _surface_sample(
            np.asarray(previous_mesh.vertices, dtype=np.float64) / 1000.0,
            np.asarray(previous_mesh.faces, dtype=np.int64),
            sample_count,
            base_seed + target_frame,
        )
        template_sample = _surface_sample(
            template_vertices,
            template_faces,
            sample_count,
            base_seed + target_frame,
        )
        checkpoint_sample = _surface_sample(
            target_prior,
            template_faces,
            sample_count,
            base_seed + target_frame,
        )
        template_error = _cd_ul1_mm(template_sample, target_sample)
        oracle_error = _cd_ul1_mm(previous_sample, target_sample)
        checkpoint_error = _cd_ul1_mm(checkpoint_sample, target_sample)
        template_errors.append(template_error)
        oracle_persistence.append(oracle_error)
        checkpoint_errors.append(checkpoint_error)

        frame_errors: dict[str, float] = {}
        current_candidate_predictions = {"released_checkpoint": target_prior}
        for field, field_correction in correction_variants.items():
            for scale in correction_scales:
                candidate = target_prior + scale * field_correction
                if scale == 0.0 and not np.array_equal(candidate, target_prior):
                    raise AssertionError(
                        "scale-zero candidate changed released checkpoint bytes"
                    )
                candidate_sample = _surface_sample(
                    candidate,
                    template_faces,
                    sample_count,
                    base_seed + target_frame,
                )
                error = _cd_ul1_mm(candidate_sample, target_sample)
                corrected_errors[(field, scale)].append(error)
                candidate_name = _candidate_name(field, scale)
                frame_errors[candidate_name] = error
                if (
                    record_online_observation_regret
                    or record_independent_anchor_regret
                ) and scale > 0.0:
                    current_candidate_predictions[candidate_name] = candidate
        if include_frozen_action_guard:
            action_field = correction_variants["action_local_state"]
            guarded_candidate = target_prior + guarded_action_scale * action_field
            if guarded_action_scale == 0.0 and not np.array_equal(
                guarded_candidate, target_prior
            ):
                raise AssertionError("rejected action update changed checkpoint bytes")
            guarded_sample = _surface_sample(
                guarded_candidate,
                template_faces,
                sample_count,
                base_seed + target_frame,
            )
            guarded_error = _cd_ul1_mm(guarded_sample, target_sample)
            guarded_action_errors.append(guarded_error)
            frame_errors["checkpoint_action_guarded"] = guarded_error
        target_records.append(
            {
                "target_frame": target_frame,
                "action_guard_scale": guarded_action_scale,
                "online_observation_regret": online_observation_regret,
                **(
                    {"independent_anchor_regret": independent_anchor_regret}
                    if record_independent_anchor_regret
                    else {}
                ),
                "template_CD_UL1_mm": template_error,
                "oracle_previous_mesh_CD_UL1_mm": oracle_error,
                "released_checkpoint_CD_UL1_mm": checkpoint_error,
                **frame_errors,
            }
        )
        if record_online_observation_regret or record_independent_anchor_regret:
            previous_prediction_frame = target_frame
            previous_candidate_predictions = current_candidate_predictions

    aggregates = {
        "template": _summary(template_errors),
        "oracle_previous_mesh": _summary(oracle_persistence),
        "released_checkpoint": _summary(checkpoint_errors),
        **{
            _candidate_name(field, scale): _summary(values)
            for (field, scale), values in corrected_errors.items()
        },
    }
    if include_frozen_action_guard:
        aggregates["checkpoint_action_guarded"] = _summary(guarded_action_errors)
    best_candidate = min(
        (
            name
            for name in aggregates
            if name != "released_checkpoint" and "_residual_scale_" in name
        ),
        key=lambda name: aggregates[name]["mean_CD_UL1_mm"],
    )
    checkpoint_hashes = {
        filename: _sha256(checkpoint_root / filename)
        for filename in (
            "pointcloud_encoder.pth",
            "attention_model.pth",
            "decoder.pth",
        )
    }
    result = {
        "schema_version": 1,
        "artifact_kind": "PokeFlexCheckpointBayesianRegistrationDevelopmentSmoke",
        "protocol": {
            "path": str(protocol_path.resolve()),
            "sha256": protocol["protocol_sha256"],
        },
        "take": {
            "id": take_root.name,
            "robot_sha256": _sha256(robot_path),
            "template_frame": template_frame,
            "template_sha256": _sha256(template_path),
            "template_preprocessing": template_preprocessing,
            "causal_target_frame_count": len(target_records),
            "maximum_frame": frame_limit,
        },
        "upstream": {
            "checkout": str(upstream_checkout.resolve()),
            "git_commit": protocol["payload"]["upstream"]["code_commit"],
            "checkpoint_root": str(checkpoint_root.resolve()),
            "checkpoint_sha256": checkpoint_hashes,
        },
        "registration_config": config.as_dict(),
        "action_guard": (
            PokeFlexActionGuardConfig().as_dict()
            if include_frozen_action_guard
            else None
        ),
        "causal_history": "every target f uses only Kinect frames f-5 through f-1",
        "residual_transfer": "correction fitted at f-1 is transferred once to f by material vertex identity",
        "correction_fields": list(correction_fields),
        "future_observation_used": False,
        "online_observation_regret_recorded": record_online_observation_regret,
        "aggregates": aggregates,
        "best_development_candidate": best_candidate,
        "published_kinect_reference_CD_UL1_mm": 6.498,
        "updates": update_records,
        "targets": target_records,
    }
    if record_independent_anchor_regret:
        assert independent_protocol is not None
        result["independent_depth_anchor"] = {
            "protocol_path": str(independent_depth_protocol_path.resolve()),
            "protocol_sha256": independent_protocol["protocol_sha256"],
            "sensor_family": "eye-in-hand RealSense D405 depth",
            "causal_history": "anchor frame is at most f-1 for prediction f",
            "future_observation_used": False,
            "calibration_sha256": list(independent_calibration_hashes),
            "template_frame": template_frame,
            "translation_m": [
                value.translation_m.tolist() for value in independent_calibrations
            ],
            "median_residual_mm": [
                1000.0 * value.median_residual_m
                for value in independent_calibrations
            ],
            "p90_residual_mm": [
                1000.0 * value.p90_residual_m
                for value in independent_calibrations
            ],
            "unknown_correlation_aggregation": "maximum per-sensor regret",
            "maximum_template_distance_m": (
                independent_anchor_maximum_template_distance_m
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("take_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--upstream-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=(
            _repository_root()
            / "configs"
            / "sota"
            / "pokeflex_bayesian_registration_v1.json"
        ),
    )
    parser.add_argument(
        "--correction-scales",
        type=float,
        nargs="+",
        default=(0.0, 0.25, 0.5, 1.0),
    )
    parser.add_argument(
        "--correction-fields",
        nargs="+",
        choices=(
            "raw",
            "translation_free",
            "translation_only",
            "affine_free",
            "affine_only",
            "motion_parallel",
            "temporal_linear",
            "temporal_mean",
            "temporal_shared",
            "action_velocity",
            "action_local_state",
            "action_augmented",
            "action_local_state_relative_0.25",
            "action_local_state_relative_0.4",
            "action_local_state_relative_0.55",
            "action_local_state_relative_0.7",
            "force_parallel_local_state",
            "action_axis_local_state",
            "force_action_plane_local_state",
            "force_mean_local_state",
            "force_parallel_local_state_relative_0.25",
            "force_parallel_local_state_relative_0.4",
            "force_parallel_local_state_relative_0.55",
            "force_parallel_local_state_relative_0.7",
            "action_axis_local_state_relative_0.25",
            "action_axis_local_state_relative_0.4",
            "action_axis_local_state_relative_0.55",
            "action_axis_local_state_relative_0.7",
            "force_action_plane_local_state_relative_0.25",
            "force_action_plane_local_state_relative_0.4",
            "force_action_plane_local_state_relative_0.55",
            "force_action_plane_local_state_relative_0.7",
            "force_mean_local_state_relative_0.25",
            "force_mean_local_state_relative_0.4",
            "force_mean_local_state_relative_0.55",
            "force_mean_local_state_relative_0.7",
        ),
        default=("raw",),
    )
    parser.add_argument(
        "--residual-geometry",
        choices=("point_to_point", "point_to_plane"),
        default="point_to_point",
    )
    parser.add_argument("--maximum-frame", type=int)
    parser.add_argument("--include-frozen-action-guard", action="store_true")
    parser.add_argument("--record-online-observation-regret", action="store_true")
    parser.add_argument(
        "--record-independent-anchor-regret",
        action="store_true",
    )
    parser.add_argument(
        "--independent-depth-protocol",
        type=Path,
        default=(
            _repository_root()
            / "configs"
            / "sota"
            / "pokeflex_independent_depth_development_v1.json"
        ),
    )
    parser.add_argument(
        "--independent-anchor-maximum-template-distance-mm",
        type=float,
        help=(
            "Optional source-only material-support radius around the allowed "
            "static template"
        ),
    )
    args = parser.parse_args()
    result = run_smoke(
        args.take_root.resolve(),
        args.protocol.resolve(),
        args.upstream_checkout.resolve(),
        args.checkpoint_root.resolve(),
        correction_scales=tuple(args.correction_scales),
        correction_fields=tuple(args.correction_fields),
        residual_geometry=args.residual_geometry,
        maximum_frame=args.maximum_frame,
        include_frozen_action_guard=args.include_frozen_action_guard,
        record_online_observation_regret=args.record_online_observation_regret,
        record_independent_anchor_regret=args.record_independent_anchor_regret,
        independent_depth_protocol_path=args.independent_depth_protocol.resolve(),
        independent_anchor_maximum_template_distance_m=(
            args.independent_anchor_maximum_template_distance_mm / 1000.0
            if args.independent_anchor_maximum_template_distance_mm is not None
            else None
        ),
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"existing smoke artifact differs: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), **result["aggregates"]}, indent=2))


if __name__ == "__main__":
    main()
