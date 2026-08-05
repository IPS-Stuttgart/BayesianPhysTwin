"""Tactile-conditioned, bias-aware metric carriers for Deform360.

Tactile contact geometry is used only to identify the observed object.  It does
not provide an object displacement measurement and it never sees the PhysTwin
state innovation.  A second camera validates the selected carrier, but unknown
cross-camera correlation is represented as shared covariance rather than
independent precision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._portable_contracts import content_id, load_strict_json_object
from .deform360_metric_object_carrier import (
    BlockPointCandidates,
    deterministic_farthest_point_indices,
    mutual_nearest_mapping,
    reduce_masked_point_map,
)
from .deform360_tactile_metric_gauge import SimilarityTransform, apply_similarity

TACTILE_PROMPTED_CARRIER_POLICY = {
    "prompt_offset_m": 0.016,
    "minimum_positive_prompt_hits": 1,
    "maximum_negative_prompt_hits": 0,
    "minimum_mask_area_fraction": 0.002,
    "maximum_mask_area_fraction": 0.35,
    "block_size_px": 8,
    "minimum_mask_pixels_per_block": 8,
    "minimum_valid_fraction_per_block": 0.5,
    "minimum_deform_fraction_for_full_reliability": 0.5,
    "local_covariance_floor_m": 0.005,
    "maximum_cross_view_distance_m": 0.03,
    "maximum_cross_view_percentile_90_m": 0.02,
    "minimum_mutual_block_matches": 5,
    "carrier_node_count": 128,
    "shared_bias_floor_m": 0.005,
    "unsupported_node_floor_m": 0.015,
    "unsupported_node_reliability_scale": 0.5,
    "object_facing_normal": (
        "finger-side-0-minus-gripper-x;finger-side-1-plus-gripper-x"
    ),
    "assignment_policy": "retain-direct-and-swapped-with-exact-fallback-per-branch",
    "cross_view_policy": "validation-and-covariance-inflation-never-precision-gain",
    "representation_policy": "dense-nodes-inherit-fixed-block-information-clusters",
}

DEFORM360_TACTILE_PROMPTED_CARRIER_VALIDATION_LOCK_SCHEMA = (
    "bayesian-phystwin.deform360-tactile-prompted-carrier-validation-lock"
)
DEFORM360_TACTILE_PROMPTED_CARRIER_VALIDATION_LOCK_VERSION = 1

TACTILE_PROMPTED_CARRIER_VALIDATION_INFORMATION_BOUNDARY = {
    "calibration_camera_prefix_allowed": True,
    "calibration_provider_values_allowed_after_lock": True,
    "calibration_scores_opened": False,
    "calibration_tactile_prefix_allowed": True,
    "confirmation_payloads_opened": False,
    "future_camera_frames_used": False,
    "future_tactile_values_used": False,
    "held_v8_accessed": False,
    "physical_state_residual_used_for_reliability": False,
    "target_outcomes_used": False,
}

TACTILE_PROMPTED_CARRIER_VALIDATION_STAGE_GATES = {
    "causal_robot_prefix": {
        "contact_tail_frame_count": 6,
        "maximum_opening_m": 0.112,
        "maximum_rotation_step_deg": 20.0,
        "maximum_translation_step_m": 0.05,
        "minimum_both_fingers_fraction": 0.5,
        "minimum_contact_ready_frames": 4,
        "minimum_direct_wrist_fraction": 0.75,
        "minimum_inlier_cameras_per_part": 2,
        "minimum_opening_m": 0.04,
        "rotation_matrix_tolerance": 0.001,
    },
    "tactile_contact_geometry": {
        "minimum_active_frames": 3,
        "minimum_active_taxels": 6,
        "minimum_assignment_separation_m": 0.05,
    },
    "tactile_metric_gauge": {
        "assignment_admission": "both-direct-and-swapped-must-pass",
        "covariance_floor_m": 0.005,
        "cross_view_correlation": "unknown-equal-weight-covariance-intersection",
        "huber_delta_m": 0.005,
        "maximum_median_held_frame_error_m": 0.005,
        "maximum_percentile_90_held_frame_error_m": 0.015,
        "minimum_admitted_cameras": 3,
    },
    "carrier": TACTILE_PROMPTED_CARRIER_POLICY,
    "failure_policy": (
        "exact-baseline-fallback-per-assignment-with-original-prior-mass"
    ),
}


@dataclass(frozen=True, slots=True)
class TactilePromptAssignment:
    """Object- and robot-side prompt points under one tactile assignment."""

    assignment_index: int
    source_frame_id: int
    positive_world_m: np.ndarray
    negative_world_m: np.ndarray
    gripper_indices: np.ndarray
    finger_side_indices: np.ndarray
    contributing_taxel_counts: np.ndarray


@dataclass(frozen=True, slots=True)
class ProjectedPromptPair:
    """Projected positive and negative prompts with visibility flags."""

    positive_pixel_xy: np.ndarray
    negative_pixel_xy: np.ndarray
    positive_visible: np.ndarray
    negative_visible: np.ndarray


@dataclass(frozen=True, slots=True)
class PromptMaskDiagnostics:
    """Residual-independent evidence for one candidate object mask."""

    eligible: bool
    positive_hits: int
    positive_visible: int
    negative_hits: int
    negative_visible: int
    area_pixels: int
    area_fraction: float
    prior_score: float


@dataclass(frozen=True, slots=True)
class DensePointCandidates:
    """Dense representation rows tied to fixed information clusters."""

    pixel_xy: np.ndarray
    points_world_m: np.ndarray
    covariance_m2: np.ndarray
    prior_reliability: np.ndarray
    information_cluster_id: np.ndarray
    block_candidates: BlockPointCandidates


@dataclass(frozen=True, slots=True)
class PromptedCandidateGeometry:
    """One SAM2 candidate with prompt and metric-geometry diagnostics."""

    candidate_index: int
    predicted_iou: float
    stability_score: float
    prompt: PromptMaskDiagnostics
    dense: DensePointCandidates


@dataclass(frozen=True, slots=True)
class CrossViewCandidatePair:
    """Best source-only candidate pair under one assignment."""

    assignment_index: int
    reference_camera: str
    support_camera: str
    reference: PromptedCandidateGeometry
    support: PromptedCandidateGeometry
    mutual_block_match_count: int
    median_block_distance_m: float
    percentile_90_block_distance_m: float


@dataclass(frozen=True, slots=True)
class BiasAwareMetricCarrier:
    """A dense carrier whose information rank remains block limited."""

    points_world_m: np.ndarray
    local_covariance_m2: np.ndarray
    shared_bias_covariance_m2: np.ndarray
    marginal_covariance_m2: np.ndarray
    prior_reliability: np.ndarray
    reference_pixel_xy: np.ndarray
    information_cluster_id: np.ndarray
    support_indices: np.ndarray
    support_distance_m: np.ndarray
    estimated_cross_view_bias_m: np.ndarray
    mutual_block_match_count: int


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _sha256(value: object, *, name: str) -> str:
    _require(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"invalid {name}",
    )
    return str(value)


def validate_tactile_prompted_carrier_validation_lock(
    value: Mapping[str, Any],
) -> str:
    """Validate the frozen independent calibration-object carrier protocol."""

    artifact_id = _sha256(value.get("artifact_id"), name="artifact_id")
    descriptor = dict(value)
    descriptor.pop("artifact_id")
    _require(content_id(descriptor) == artifact_id, "validation lock identity changed")
    _require(
        value.get("schema")
        == DEFORM360_TACTILE_PROMPTED_CARRIER_VALIDATION_LOCK_SCHEMA
        and value.get("schema_version")
        == DEFORM360_TACTILE_PROMPTED_CARRIER_VALIDATION_LOCK_VERSION,
        "unsupported validation lock",
    )
    _require(
        value.get("status") == "locked-independent-calibration-pre-payload",
        "validation lock has the wrong status",
    )
    _require(
        value.get("information_boundary")
        == TACTILE_PROMPTED_CARRIER_VALIDATION_INFORMATION_BOUNDARY,
        "validation information boundary changed",
    )
    _require(
        value.get("stage_gates") == TACTILE_PROMPTED_CARRIER_VALIDATION_STAGE_GATES,
        "validation stage gates changed",
    )

    source = value.get("source_case")
    selection = value.get("selection")
    bindings = value.get("bindings")
    implementation = value.get("implementation")
    for item, name in (
        (source, "source_case"),
        (selection, "selection"),
        (bindings, "bindings"),
        (implementation, "implementation"),
    ):
        _require(isinstance(item, Mapping), f"missing {name}")
    assert isinstance(source, Mapping)
    assert isinstance(selection, Mapping)
    assert isinstance(bindings, Mapping)
    assert isinstance(implementation, Mapping)

    _require(source.get("object_id") == "036-napkin-cloth", "source object changed")
    _require(source.get("source_episode_id") == 9, "source episode changed")
    _require(source.get("processing_episode_index") == 0, "processing episode changed")
    _require(source.get("bimanual") is True, "validation source is not bimanual")
    _require(source.get("stratum") == "sheet", "source stratum changed")
    _require(
        source.get("camera_panel")
        == [
            "brics-odroid-010_cam0",
            "brics-odroid-019_cam1",
            "brics-odroid-022_cam1",
        ],
        "camera panel changed",
    )
    _require(
        source.get("causal_window")
        == {
            "source_frame_start": 78,
            "contact_start_frame": 114,
            "causal_frame_stop": 120,
            "untouched_future_frame_start": 120,
            "untouched_future_frame_stop_exclusive": 144,
        },
        "causal window changed",
    )
    _sha256(source.get("bound_input_files_sha256"), name="bound input files")
    _sha256(source.get("source_output_tree_sha256"), name="source output tree")

    _require(
        selection.get("rule")
        == "first-unopened-bimanual-calibration-object-after-development-in-stratum",
        "selection rule changed",
    )
    _require(
        selection.get("development_object_id") == "026-sock-cloth",
        "development object changed",
    )
    candidates = selection.get("candidate_audit")
    _require(
        candidates
        == [
            {
                "bimanual": False,
                "metadata_sha256": (
                    "c2c5d30efc85fea22b52d1b5317c3a8084f272ef79154f313441a79c35adaa08"
                ),
                "object_id": "031-cotton-cloth",
            },
            {
                "bimanual": True,
                "metadata_sha256": (
                    "3eca16d7f72ce1d83828d60fdc98fa942843d12cd1aa20846f036a3e882547a6"
                ),
                "object_id": "036-napkin-cloth",
            },
        ],
        "selection audit changed",
    )

    for name in (
        "selection_lock_file_sha256",
        "causal_window_manifest_file_sha256",
        "causal_window_manifest_id",
        "motioncrafter_job_manifest_file_sha256",
        "motioncrafter_job_manifest_id",
        "motioncrafter_stage1_run_report_sha256",
    ):
        _sha256(bindings.get(name), name=name)
    jobs = bindings.get("motioncrafter_jobs")
    _require(isinstance(jobs, list) and len(jobs) == 3, "provider jobs changed")
    for camera, job in zip(source["camera_panel"], jobs, strict=True):
        _require(isinstance(job, Mapping), "invalid provider job")
        assert isinstance(job, Mapping)
        _require(job.get("camera") == camera, "provider camera order changed")
        _sha256(job.get("job_id"), name="provider job ID")
        _require(job.get("source_frame_start") == 78, "provider start changed")
        _require(job.get("source_frame_stop_exclusive") == 120, "provider stop changed")
        _sha256(job.get("video_sha256"), name="provider video")

    revision = implementation.get("revision")
    _require(
        type(revision) is str
        and len(revision) == 40
        and all(character in "0123456789abcdef" for character in revision),
        "invalid implementation revision",
    )
    for name in ("module_source_sha256", "validation_runner_source_sha256"):
        _sha256(implementation.get(name), name=name)
    return artifact_id


def load_tactile_prompted_carrier_validation_lock(
    path: str | Path,
) -> Mapping[str, Any]:
    """Load and validate one independent carrier validation lock."""

    value = load_strict_json_object(path, label="tactile-prompted carrier validation lock")
    validate_tactile_prompted_carrier_validation_lock(value)
    return value


def _nearest(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _require(len(source) > 0 and len(target) > 0, "empty point set")
    indices = np.empty(len(source), dtype=np.int64)
    distances = np.empty(len(source), dtype=np.float64)
    for start in range(0, len(source), 256):
        stop = min(start + 256, len(source))
        squared = np.sum(
            (source[start:stop, None] - target[None]) ** 2,
            axis=2,
        )
        local = np.argmin(squared, axis=1)
        indices[start:stop] = local
        distances[start:stop] = np.sqrt(squared[np.arange(stop - start), local])
    return indices, distances


def _robust_covariance(values: np.ndarray, *, floor_m: float) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float64)
    _require(rows.ndim == 2 and rows.shape[1] == 3 and len(rows) > 0, "invalid rows")
    center = np.median(rows, axis=0)
    mad = 1.4826 * np.median(np.abs(rows - center), axis=0)
    return np.diag(np.maximum(mad**2, floor_m**2))


def object_facing_finger_normal_world(
    finger_side: int,
    world_from_gripper: np.ndarray,
) -> np.ndarray:
    """Return the object-facing taxel normal in world coordinates.

    Deform360's UMI taxel geometry places the two sampled finger planes on
    opposite sides of the gripper-root x axis.  Their inward normals are
    therefore ``-x`` for released side zero and ``+x`` for side one.
    """

    _require(finger_side in {0, 1}, "finger side must be zero or one")
    pose = np.asarray(world_from_gripper, dtype=np.float64)
    _require(pose.shape == (4, 4), "gripper pose must have shape (4,4)")
    _require(np.all(np.isfinite(pose)), "gripper pose is non-finite")
    normal = (-1.0 if finger_side == 0 else 1.0) * pose[:3, 0]
    norm = float(np.linalg.norm(normal))
    _require(norm > 0.0, "gripper x axis is degenerate")
    return normal / norm


def build_tactile_prompt_assignments(
    *,
    tactile_source_frame_ids: np.ndarray,
    tactile_values: np.ndarray,
    finger_side_indices: np.ndarray,
    world_points_hypotheses_m: np.ndarray,
    gripper_indices_hypotheses: np.ndarray,
    robot_source_frame_ids: np.ndarray,
    robot_world_from_gripper: np.ndarray,
    offset_m: float,
) -> tuple[TactilePromptAssignment, ...]:
    """Build assignment-preserving prompts from the latest active tactile frame."""

    frames = np.asarray(tactile_source_frame_ids, dtype=np.int64)
    values = np.asarray(tactile_values, dtype=np.float64)
    sides = np.asarray(finger_side_indices, dtype=np.int8)
    points = np.asarray(world_points_hypotheses_m, dtype=np.float64)
    grippers = np.asarray(gripper_indices_hypotheses, dtype=np.int8)
    robot_frames = np.asarray(robot_source_frame_ids, dtype=np.int64)
    poses = np.asarray(robot_world_from_gripper, dtype=np.float64)
    _require(frames.ndim == values.ndim == sides.ndim == 1, "invalid tactile rows")
    _require(
        len(frames) > 0 and len(values) == len(sides) == len(frames), "row mismatch"
    )
    _require(
        points.ndim == 3 and points.shape[0] == len(frames) and points.shape[2] == 3,
        "point hypotheses changed",
    )
    _require(grippers.shape == points.shape[:2], "gripper hypotheses changed")
    _require(poses.shape == (len(robot_frames), 2, 4, 4), "robot poses changed")
    _require(
        np.all(np.isfinite(values)) and np.all(np.isfinite(points)),
        "non-finite tactile geometry",
    )
    _require(offset_m > 0.0, "prompt offset must be positive")
    prompt_frame = int(np.max(frames))
    robot_rows = np.flatnonzero(robot_frames == prompt_frame)
    _require(len(robot_rows) == 1, "prompt frame is absent from robot prefix")
    robot_row = int(robot_rows[0])
    active = np.flatnonzero(frames == prompt_frame)
    assignments: list[TactilePromptAssignment] = []
    for assignment_index in range(points.shape[1]):
        keys = sorted(
            {(int(grippers[row, assignment_index]), int(sides[row])) for row in active}
        )
        positive: list[np.ndarray] = []
        negative: list[np.ndarray] = []
        output_grippers: list[int] = []
        output_sides: list[int] = []
        counts: list[int] = []
        for gripper, side in keys:
            rows = active[
                (grippers[active, assignment_index] == gripper)
                & (sides[active] == side)
            ]
            weights = np.maximum(values[rows], np.finfo(np.float64).eps)
            weights /= np.sum(weights)
            contact = np.sum(points[rows, assignment_index] * weights[:, None], axis=0)
            normal = object_facing_finger_normal_world(side, poses[robot_row, gripper])
            positive.append(contact + offset_m * normal)
            negative.append(contact - offset_m * normal)
            output_grippers.append(gripper)
            output_sides.append(side)
            counts.append(len(rows))
        _require(positive, "latest tactile frame has no contact groups")
        assignments.append(
            TactilePromptAssignment(
                assignment_index=assignment_index,
                source_frame_id=prompt_frame,
                positive_world_m=np.asarray(positive),
                negative_world_m=np.asarray(negative),
                gripper_indices=np.asarray(output_grippers, dtype=np.int8),
                finger_side_indices=np.asarray(output_sides, dtype=np.int8),
                contributing_taxel_counts=np.asarray(counts, dtype=np.int64),
            )
        )
    return tuple(assignments)


def project_prompt_assignment(
    assignment: TactilePromptAssignment,
    *,
    intrinsics: np.ndarray,
    world_from_camera: np.ndarray,
    image_shape: tuple[int, int],
) -> ProjectedPromptPair:
    """Project one prompt assignment into a calibrated camera."""

    matrix = np.asarray(intrinsics, dtype=np.float64)
    pose = np.asarray(world_from_camera, dtype=np.float64)
    _require(matrix.shape == (3, 3) and pose.shape == (4, 4), "invalid calibration")
    height, width = image_shape
    _require(height > 0 and width > 0, "invalid image shape")

    def project(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        homogeneous = np.column_stack((points, np.ones(len(points))))
        camera = (np.linalg.inv(pose) @ homogeneous.T).T[:, :3]
        depth = camera[:, 2]
        pixel = np.column_stack(
            (
                matrix[0, 0] * camera[:, 0] / depth + matrix[0, 2],
                matrix[1, 1] * camera[:, 1] / depth + matrix[1, 2],
            )
        )
        visible = (
            (depth > 0.0)
            & (pixel[:, 0] >= 0.0)
            & (pixel[:, 0] < width)
            & (pixel[:, 1] >= 0.0)
            & (pixel[:, 1] < height)
            & np.all(np.isfinite(pixel), axis=1)
        )
        return pixel, visible

    positive, positive_visible = project(assignment.positive_world_m)
    negative, negative_visible = project(assignment.negative_world_m)
    return ProjectedPromptPair(
        positive_pixel_xy=positive,
        negative_pixel_xy=negative,
        positive_visible=positive_visible,
        negative_visible=negative_visible,
    )


def evaluate_prompted_mask(
    mask: np.ndarray,
    prompts: ProjectedPromptPair,
    *,
    predicted_iou: float,
    stability_score: float,
    minimum_positive_hits: int,
    maximum_negative_hits: int,
    minimum_area_fraction: float,
    maximum_area_fraction: float,
) -> PromptMaskDiagnostics:
    """Evaluate tactile prompt agreement without a physical-state residual."""

    selected = np.asarray(mask, dtype=bool)
    _require(selected.ndim == 2, "mask must be two-dimensional")
    _require(0.0 <= predicted_iou <= 1.0, "invalid predicted IoU")
    _require(0.0 <= stability_score <= 1.0, "invalid stability score")

    def count_hits(pixel: np.ndarray, visible: np.ndarray) -> int:
        rounded = np.rint(pixel[visible]).astype(np.int64)
        if len(rounded) == 0:
            return 0
        return int(np.count_nonzero(selected[rounded[:, 1], rounded[:, 0]]))

    positive_visible = int(np.count_nonzero(prompts.positive_visible))
    negative_visible = int(np.count_nonzero(prompts.negative_visible))
    positive_hits = count_hits(prompts.positive_pixel_xy, prompts.positive_visible)
    negative_hits = count_hits(prompts.negative_pixel_xy, prompts.negative_visible)
    area_pixels = int(np.count_nonzero(selected))
    area_fraction = float(np.mean(selected))
    eligible = bool(
        positive_visible >= minimum_positive_hits
        and positive_hits >= minimum_positive_hits
        and negative_hits <= maximum_negative_hits
        and minimum_area_fraction <= area_fraction <= maximum_area_fraction
    )
    positive_fraction = positive_hits / max(positive_visible, 1)
    negative_fraction = negative_hits / max(negative_visible, 1)
    quality = float(np.sqrt(predicted_iou * stability_score))
    prior_score = float(positive_fraction - 2.0 * negative_fraction + 0.25 * quality)
    return PromptMaskDiagnostics(
        eligible=eligible,
        positive_hits=positive_hits,
        positive_visible=positive_visible,
        negative_hits=negative_hits,
        negative_visible=negative_visible,
        area_pixels=area_pixels,
        area_fraction=area_fraction,
        prior_score=prior_score,
    )


def build_dense_point_candidates(
    point_map: np.ndarray,
    valid_mask: np.ndarray,
    object_mask: np.ndarray,
    deform_mask: np.ndarray,
    *,
    transform: SimilarityTransform,
    gauge_covariance_m2: np.ndarray,
    block_size_px: int,
    minimum_mask_pixels: int,
    minimum_valid_fraction: float,
    full_reliability_deform_fraction: float,
    covariance_floor_m: float,
) -> DensePointCandidates:
    """Keep dense carrier rows while assigning fixed-block information IDs."""

    points = np.asarray(point_map, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    selected = np.asarray(object_mask, dtype=bool)
    _require(points.ndim == 3 and points.shape[2] == 3, "invalid point map")
    _require(valid.shape == selected.shape == points.shape[:2], "mask shape mismatch")
    blocks = reduce_masked_point_map(
        points,
        valid,
        selected,
        deform_mask,
        transform=transform,
        gauge_covariance_m2=gauge_covariance_m2,
        block_size_px=block_size_px,
        minimum_mask_pixels=minimum_mask_pixels,
        minimum_valid_fraction=minimum_valid_fraction,
        full_reliability_deform_fraction=full_reliability_deform_fraction,
        covariance_floor_m=covariance_floor_m,
    )
    block_lookup = {
        (int(row), int(column)): index
        for index, (row, column) in enumerate(blocks.block_yx)
    }
    rows, columns = np.nonzero(selected & valid & np.all(np.isfinite(points), axis=2))
    cluster = np.asarray(
        [
            block_lookup.get(
                (int(row // block_size_px), int(column // block_size_px)), -1
            )
            for row, column in zip(rows, columns, strict=True)
        ],
        dtype=np.int64,
    )
    admitted = cluster >= 0
    _require(np.any(admitted), "mask has no dense rows in admitted blocks")
    rows = rows[admitted]
    columns = columns[admitted]
    cluster = cluster[admitted]
    world = apply_similarity(transform, points[rows, columns])
    return DensePointCandidates(
        pixel_xy=np.column_stack((columns, rows)).astype(np.float64),
        points_world_m=world,
        covariance_m2=blocks.covariance_m2[cluster],
        prior_reliability=blocks.prior_reliability[cluster],
        information_cluster_id=cluster,
        block_candidates=blocks,
    )


def select_crossview_candidate_pair(
    candidates_by_camera: Mapping[str, Sequence[PromptedCandidateGeometry]],
    *,
    assignment_index: int,
    camera_order: Sequence[str],
    maximum_distance_m: float,
    minimum_mutual_matches: int,
    maximum_percentile_90_m: float,
) -> CrossViewCandidatePair:
    """Select one candidate pair using only prompts and cross-view geometry."""

    cameras = tuple(camera_order)
    _require(len(cameras) >= 2, "at least two cameras are required")
    ranked: list[tuple[tuple[float, ...], CrossViewCandidatePair]] = []
    for first_index, first_camera in enumerate(cameras):
        for second_camera in cameras[first_index + 1 :]:
            for first in candidates_by_camera.get(first_camera, ()):
                for second in candidates_by_camera.get(second_camera, ()):
                    mapping, distance = mutual_nearest_mapping(
                        first.dense.block_candidates.points_world_m,
                        second.dense.block_candidates.points_world_m,
                        maximum_distance_m=maximum_distance_m,
                    )
                    accepted = mapping >= 0
                    match_count = int(np.count_nonzero(accepted))
                    if match_count == 0:
                        continue
                    accepted_distance = distance[accepted]
                    median = float(np.median(accepted_distance))
                    percentile_90 = float(np.quantile(accepted_distance, 0.9))
                    if (
                        match_count < minimum_mutual_matches
                        or percentile_90 > maximum_percentile_90_m
                    ):
                        continue
                    if len(second.dense.block_candidates.points_world_m) > len(
                        first.dense.block_candidates.points_world_m
                    ):
                        reference_camera, support_camera = second_camera, first_camera
                        reference, support = second, first
                    else:
                        reference_camera, support_camera = first_camera, second_camera
                        reference, support = first, second
                    pair = CrossViewCandidatePair(
                        assignment_index=assignment_index,
                        reference_camera=reference_camera,
                        support_camera=support_camera,
                        reference=reference,
                        support=support,
                        mutual_block_match_count=match_count,
                        median_block_distance_m=median,
                        percentile_90_block_distance_m=percentile_90,
                    )
                    prompt_score = first.prompt.prior_score + second.prompt.prior_score
                    quality = (
                        first.predicted_iou
                        + first.stability_score
                        + second.predicted_iou
                        + second.stability_score
                    )
                    key = (
                        float(match_count),
                        -percentile_90,
                        prompt_score,
                        quality,
                        -float(cameras.index(reference_camera)),
                        -float(reference.candidate_index),
                        -float(support.candidate_index),
                    )
                    ranked.append((key, pair))
    _require(ranked, "no prompted camera pair passed cross-view support")
    return max(ranked, key=lambda item: item[0])[1]


def build_bias_aware_metric_carrier(
    pair: CrossViewCandidatePair,
    *,
    node_count: int,
    maximum_distance_m: float,
    shared_bias_floor_m: float,
    unsupported_node_floor_m: float,
    unsupported_reliability_scale: float,
) -> BiasAwareMetricCarrier:
    """Densify a reference mask without increasing independent precision."""

    reference = pair.reference.dense
    support = pair.support.dense
    _require(
        len(reference.points_world_m) >= node_count, "reference carrier is too small"
    )
    _require(0.0 < unsupported_reliability_scale <= 1.0, "invalid fallback reliability")
    selected = deterministic_farthest_point_indices(reference.pixel_xy, node_count)
    selected_points = reference.points_world_m[selected]
    support_indices, support_distance = _nearest(
        selected_points, support.points_world_m
    )
    supported = support_distance <= maximum_distance_m
    output_support = support_indices.copy()
    output_support[~supported] = -1

    block_mapping, _ = mutual_nearest_mapping(
        reference.block_candidates.points_world_m,
        support.block_candidates.points_world_m,
        maximum_distance_m=maximum_distance_m,
    )
    block_supported = block_mapping >= 0
    _require(
        np.count_nonzero(block_supported) >= pair.mutual_block_match_count,
        "cross-view block support changed after selection",
    )
    residual = (
        support.block_candidates.points_world_m[block_mapping[block_supported]]
        - reference.block_candidates.points_world_m[block_supported]
    )
    bias = np.median(residual, axis=0)
    shared_bias = _robust_covariance(
        residual - bias,
        floor_m=shared_bias_floor_m,
    ) + np.outer(bias, bias)

    local = reference.covariance_m2[selected].copy()
    reliability = reference.prior_reliability[selected].copy()
    for output_index in range(node_count):
        if supported[output_index]:
            difference = (
                support.points_world_m[support_indices[output_index]]
                - selected_points[output_index]
                - bias
            )
            local[output_index] += np.outer(difference, difference)
            reliability[output_index] *= float(
                np.exp(-support_distance[output_index] / maximum_distance_m)
            )
        else:
            local[output_index] += np.eye(3) * unsupported_node_floor_m**2
            reliability[output_index] *= unsupported_reliability_scale
    marginal = local + shared_bias[None]
    _require(
        np.all(
            np.linalg.eigvalsh(marginal - reference.covariance_m2[selected]) >= -1e-12
        ),
        "cross-view carrier became more confident than its reference",
    )
    return BiasAwareMetricCarrier(
        points_world_m=selected_points,
        local_covariance_m2=local,
        shared_bias_covariance_m2=shared_bias,
        marginal_covariance_m2=marginal,
        prior_reliability=np.clip(reliability, 0.0, 1.0),
        reference_pixel_xy=reference.pixel_xy[selected],
        information_cluster_id=reference.information_cluster_id[selected],
        support_indices=output_support,
        support_distance_m=np.where(supported, support_distance, np.inf),
        estimated_cross_view_bias_m=bias,
        mutual_block_match_count=int(np.count_nonzero(block_supported)),
    )


__all__ = [
    "BiasAwareMetricCarrier",
    "CrossViewCandidatePair",
    "DEFORM360_TACTILE_PROMPTED_CARRIER_VALIDATION_LOCK_SCHEMA",
    "DEFORM360_TACTILE_PROMPTED_CARRIER_VALIDATION_LOCK_VERSION",
    "DensePointCandidates",
    "ProjectedPromptPair",
    "PromptMaskDiagnostics",
    "PromptedCandidateGeometry",
    "TACTILE_PROMPTED_CARRIER_POLICY",
    "TACTILE_PROMPTED_CARRIER_VALIDATION_INFORMATION_BOUNDARY",
    "TACTILE_PROMPTED_CARRIER_VALIDATION_STAGE_GATES",
    "TactilePromptAssignment",
    "build_bias_aware_metric_carrier",
    "build_dense_point_candidates",
    "build_tactile_prompt_assignments",
    "evaluate_prompted_mask",
    "object_facing_finger_normal_world",
    "project_prompt_assignment",
    "select_crossview_candidate_pair",
    "load_tactile_prompted_carrier_validation_lock",
    "validate_tactile_prompted_carrier_validation_lock",
]
