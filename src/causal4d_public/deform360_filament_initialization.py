"""Silhouette-derived initialization for multiview filament registration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from .deform360_filament_registration import (
    filament_multiview_support_diagnostics,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class FilamentMaskInitializerConfig:
    node_count: int = 21
    minimum_mask_component_pixels: int = 32
    minimum_skeleton_component_pixels: int = 8
    maximum_plain_bridge_distance_px: float = 18.0
    maximum_gripper_bridge_distance_px: float = 120.0
    minimum_gripper_bridge_fraction: float = 0.45
    samples_per_edge_for_scoring: int = 5
    maximum_mask_sample_count_for_scoring: int = 256
    maximum_candidate_length_m: float = 1.5
    minimum_raw_length_fraction_of_target: float = 0.50
    length_scale_penalty_weight: float = 0.15
    minimum_source_camera_reliability: float = 0.25
    minimum_triangulation_baseline_m: float = 0.08
    minimum_path_length_px_for_triangulation: float = 40.0

    def __post_init__(self) -> None:
        _require(self.node_count >= 4, "filament initializer needs four nodes")
        _require(
            self.minimum_mask_component_pixels >= 1,
            "minimum mask component must be positive",
        )
        _require(
            self.minimum_skeleton_component_pixels >= 2,
            "minimum skeleton component is too small",
        )
        _require(
            self.maximum_plain_bridge_distance_px >= 0.0,
            "plain bridge distance must be nonnegative",
        )
        _require(
            self.maximum_gripper_bridge_distance_px
            >= self.maximum_plain_bridge_distance_px,
            "gripper bridge distance must include the plain bridge distance",
        )
        _require(
            0.0 <= self.minimum_gripper_bridge_fraction <= 1.0,
            "gripper bridge fraction must be a probability",
        )
        _require(
            self.samples_per_edge_for_scoring >= 2,
            "initializer scoring needs two samples per edge",
        )
        _require(
            self.maximum_mask_sample_count_for_scoring >= 16,
            "initializer mask sampling budget is too small",
        )
        _require(
            self.maximum_candidate_length_m > 0.0,
            "candidate length limit must be positive",
        )
        _require(
            0.0 < self.minimum_raw_length_fraction_of_target <= 1.0,
            "minimum raw length fraction must lie in (0,1]",
        )
        _require(
            self.length_scale_penalty_weight >= 0.0,
            "length scale penalty must be nonnegative",
        )
        _require(
            0.0 < self.minimum_source_camera_reliability <= 1.0,
            "source camera reliability must lie in (0,1]",
        )
        _require(
            self.minimum_triangulation_baseline_m >= 0.0,
            "triangulation baseline must be nonnegative",
        )
        _require(
            self.minimum_path_length_px_for_triangulation > 0.0,
            "triangulation path length must be positive",
        )


def _remove_small_components(mask: np.ndarray, minimum_pixels: int) -> np.ndarray:
    try:
        from scipy.ndimage import label
    except ImportError as error:  # pragma: no cover - scipy is a graph dependency
        raise RuntimeError("SciPy is required for filament initialization") from error
    labels, count = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.reshape(-1))
    keep = np.flatnonzero(sizes >= minimum_pixels)
    keep = keep[keep != 0]
    return np.isin(labels, keep)


def _morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy.ndimage import binary_dilation, binary_erosion
    except ImportError as error:  # pragma: no cover - scipy is a graph dependency
        raise RuntimeError("SciPy is required for filament initialization") from error
    structure = np.ones((3, 3), dtype=bool)
    eroded = np.asarray(mask, dtype=bool).copy()
    skeleton = np.zeros_like(eroded)
    while np.any(eroded):
        next_eroded = binary_erosion(eroded, structure=structure)
        opened = binary_dilation(next_eroded, structure=structure)
        skeleton |= eroded & ~opened
        eroded = next_eroded
    return skeleton


def _line_pixels(first_rc: np.ndarray, second_rc: np.ndarray) -> np.ndarray:
    count = max(int(np.ceil(np.linalg.norm(second_rc - first_rc))) + 1, 2)
    return np.rint(
        np.linspace(first_rc.astype(float), second_rc.astype(float), count)
    ).astype(np.int64)


def _bridge_skeleton_components(
    skeleton: np.ndarray,
    gripper_mask: np.ndarray,
    config: FilamentMaskInitializerConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    try:
        from scipy.ndimage import label
        from scipy.spatial import cKDTree
    except ImportError as error:  # pragma: no cover - scipy is a graph dependency
        raise RuntimeError("SciPy is required for filament initialization") from error

    result = np.asarray(skeleton, dtype=bool).copy()
    bridges = []
    while True:
        labels, count = label(result, structure=np.ones((3, 3), dtype=np.uint8))
        if count <= 1:
            break
        components = []
        for component in range(1, count + 1):
            coordinates = np.column_stack(np.nonzero(labels == component))
            if len(coordinates) >= config.minimum_skeleton_component_pixels:
                components.append(coordinates)
        if len(components) <= 1:
            if components:
                retained = np.zeros_like(result)
                retained[tuple(components[0].T)] = True
                result = retained
            break

        best = None
        for first_index, first in enumerate(components[:-1]):
            first_tree = cKDTree(first)
            for second_index in range(first_index + 1, len(components)):
                second = components[second_index]
                distance, nearest = first_tree.query(second, k=1)
                second_point_index = int(np.argmin(distance))
                first_point = first[int(nearest[second_point_index])]
                second_point = second[second_point_index]
                bridge_pixels = _line_pixels(first_point, second_point)
                bridge_pixels[:, 0] = np.clip(
                    bridge_pixels[:, 0], 0, result.shape[0] - 1
                )
                bridge_pixels[:, 1] = np.clip(
                    bridge_pixels[:, 1], 0, result.shape[1] - 1
                )
                bridge_distance = float(np.linalg.norm(second_point - first_point))
                gripper_fraction = float(np.mean(gripper_mask[tuple(bridge_pixels.T)]))
                allowed = (
                    bridge_distance <= config.maximum_plain_bridge_distance_px
                    or (
                        bridge_distance <= config.maximum_gripper_bridge_distance_px
                        and gripper_fraction >= config.minimum_gripper_bridge_fraction
                    )
                )
                candidate = {
                    "distance_px": bridge_distance,
                    "gripper_fraction": gripper_fraction,
                    "allowed": allowed,
                    "pixels": bridge_pixels,
                }
                if allowed and (best is None or bridge_distance < best["distance_px"]):
                    best = candidate
        if best is None:
            largest = max(components, key=len)
            retained = np.zeros_like(result)
            retained[tuple(largest.T)] = True
            result = retained
            break
        result[tuple(best.pop("pixels").T)] = True
        bridges.append(best)
    return result, bridges


def _skeleton_diameter_path(skeleton: np.ndarray) -> np.ndarray:
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components, dijkstra
    except ImportError as error:  # pragma: no cover - scipy is a graph dependency
        raise RuntimeError("SciPy is required for filament initialization") from error

    coordinates = np.column_stack(np.nonzero(skeleton)).astype(np.int64)
    _require(len(coordinates) >= 2, "filament skeleton has fewer than two pixels")
    index = np.full(skeleton.shape, -1, dtype=np.int64)
    index[tuple(coordinates.T)] = np.arange(len(coordinates))
    rows = []
    columns = []
    values = []
    for row_offset in (-1, 0, 1):
        for column_offset in (-1, 0, 1):
            if row_offset == column_offset == 0:
                continue
            shifted = coordinates + (row_offset, column_offset)
            valid = (
                (shifted[:, 0] >= 0)
                & (shifted[:, 0] < skeleton.shape[0])
                & (shifted[:, 1] >= 0)
                & (shifted[:, 1] < skeleton.shape[1])
            )
            source = np.flatnonzero(valid)
            target = index[tuple(shifted[valid].T)]
            connected = target >= 0
            rows.extend(source[connected].tolist())
            columns.extend(target[connected].tolist())
            values.extend(
                [float(np.hypot(row_offset, column_offset))]
                * int(np.count_nonzero(connected))
            )
    graph = coo_matrix(
        (values, (rows, columns)), shape=(len(coordinates), len(coordinates))
    ).tocsr()
    component_count, labels = connected_components(graph, directed=False)
    if component_count > 1:
        largest = int(np.argmax(np.bincount(labels)))
        keep = labels == largest
        coordinates = coordinates[keep]
        graph = graph[keep][:, keep]
    degree = np.diff(graph.indptr)
    endpoints = np.flatnonzero(degree == 1)
    starts = endpoints if len(endpoints) else np.asarray([0])
    best_distance = -np.inf
    best_start = 0
    best_stop = 0
    for start in starts:
        distances = dijkstra(graph, directed=False, indices=int(start))
        candidate_stop = int(np.argmax(distances))
        if distances[candidate_stop] > best_distance:
            best_distance = float(distances[candidate_stop])
            best_start = int(start)
            best_stop = candidate_stop
    _, predecessors = dijkstra(
        graph,
        directed=False,
        indices=best_start,
        return_predecessors=True,
    )
    path = [best_stop]
    cursor = best_stop
    while cursor != best_start:
        cursor = int(predecessors[cursor])
        _require(cursor >= 0, "cannot reconstruct filament skeleton diameter")
        path.append(cursor)
    path.reverse()
    return coordinates[np.asarray(path)]


def _resample_polyline(points: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    edge = np.linalg.norm(np.diff(values, axis=0), axis=1)
    keep = np.concatenate(([True], edge > 1e-8))
    values = values[keep]
    _require(len(values) >= 2, "filament polyline has no arc length")
    cumulative = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(values, axis=0), axis=1)))
    )
    target = np.linspace(0.0, cumulative[-1], count)
    return np.column_stack(
        [
            np.interp(target, cumulative, values[:, axis])
            for axis in range(values.shape[1])
        ]
    )


def extract_filament_mask_centerline(
    object_mask: np.ndarray,
    *,
    gripper_mask: np.ndarray | None = None,
    config: FilamentMaskInitializerConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract an ordered 2D medial path from a possibly occluded rope mask."""

    cfg = config or FilamentMaskInitializerConfig()
    mask = np.asarray(object_mask, dtype=bool)
    _require(mask.ndim == 2 and np.any(mask), "filament object mask is empty")
    gripper = (
        np.asarray(gripper_mask, dtype=bool)
        if gripper_mask is not None
        else np.zeros_like(mask)
    )
    _require(gripper.shape == mask.shape, "gripper/object mask shapes differ")
    cleaned = _remove_small_components(mask, cfg.minimum_mask_component_pixels)
    _require(np.any(cleaned), "filament mask has no retained component")
    skeleton = _morphological_skeleton(cleaned)
    skeleton, bridges = _bridge_skeleton_components(skeleton, gripper, cfg)
    path_rc = _skeleton_diameter_path(skeleton)
    path_xy = path_rc[:, ::-1].astype(np.float64)
    resampled = _resample_polyline(path_xy, cfg.node_count)
    return resampled, {
        "input_foreground_pixel_count": int(np.count_nonzero(mask)),
        "retained_foreground_pixel_count": int(np.count_nonzero(cleaned)),
        "skeleton_pixel_count": int(np.count_nonzero(skeleton)),
        "diameter_path_pixel_count": len(path_xy),
        "diameter_path_length_px": float(
            np.linalg.norm(np.diff(path_xy, axis=0), axis=1).sum()
        ),
        "bridge_count": len(bridges),
        "bridges": bridges,
    }


def fit_filament_seed_plane(
    points_world_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_world_m, dtype=np.float64)
    _require(
        points.ndim == 2 and points.shape[1] == 3 and len(points) >= 8,
        "filament seed points must have shape (N,3)",
    )
    point = np.median(points, axis=0)
    _, _, axes = np.linalg.svd(points - point, full_matrices=False)
    normal = axes[-1]
    return point, normal


def backproject_pixels_to_plane(
    pixels_xy: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    plane_point_world_m: np.ndarray,
    plane_normal_world: np.ndarray,
) -> np.ndarray:
    pixels = np.asarray(pixels_xy, dtype=np.float64)
    intrinsic = np.asarray(intrinsics, dtype=np.float64)
    pose = np.asarray(camera_to_world, dtype=np.float64)
    plane_point = np.asarray(plane_point_world_m, dtype=np.float64)
    normal = np.asarray(plane_normal_world, dtype=np.float64)
    _require(pixels.ndim == 2 and pixels.shape[1] == 2, "pixels must have shape (N,2)")
    _require(intrinsic.shape == (3, 3), "camera intrinsics must have shape (3,3)")
    _require(pose.shape == (4, 4), "camera pose must have shape (4,4)")
    _require(plane_point.shape == normal.shape == (3,), "plane vectors must be 3D")
    normal_norm = float(np.linalg.norm(normal))
    _require(normal_norm > 1e-8, "plane normal is degenerate")
    normal = normal / normal_norm
    homogeneous = np.column_stack((pixels, np.ones(len(pixels))))
    ray_camera = homogeneous @ np.linalg.inv(intrinsic).T
    ray_world = ray_camera @ pose[:3, :3].T
    origin = pose[:3, 3]
    denominator = ray_world @ normal
    _require(
        np.all(np.abs(denominator) > 1e-8),
        "camera rays are parallel to the filament seed plane",
    )
    distance = ((plane_point - origin) @ normal) / denominator
    _require(np.all(distance > 0.0), "filament seed plane lies behind a camera")
    return origin + distance[:, None] * ray_world


def triangulate_corresponding_pixels(
    first_pixels_xy: np.ndarray,
    second_pixels_xy: np.ndarray,
    first_intrinsics: np.ndarray,
    first_camera_to_world: np.ndarray,
    second_intrinsics: np.ndarray,
    second_camera_to_world: np.ndarray,
) -> np.ndarray:
    """Triangulate ordered two-view correspondences with linear DLT."""

    first = np.asarray(first_pixels_xy, dtype=np.float64)
    second = np.asarray(second_pixels_xy, dtype=np.float64)
    _require(
        first.shape == second.shape and first.ndim == 2 and first.shape[1] == 2,
        "triangulation pixels must have matching shape (N,2)",
    )

    def projection(intrinsics: np.ndarray, pose: np.ndarray) -> np.ndarray:
        intrinsic = np.asarray(intrinsics, dtype=np.float64)
        camera_to_world = np.asarray(pose, dtype=np.float64)
        _require(intrinsic.shape == (3, 3), "camera intrinsics must be (3,3)")
        _require(camera_to_world.shape == (4, 4), "camera pose must be (4,4)")
        return intrinsic @ np.linalg.inv(camera_to_world)[:3]

    first_projection = projection(first_intrinsics, first_camera_to_world)
    second_projection = projection(second_intrinsics, second_camera_to_world)
    points = []
    for first_pixel, second_pixel in zip(first, second, strict=True):
        system = np.stack(
            (
                first_pixel[0] * first_projection[2] - first_projection[0],
                first_pixel[1] * first_projection[2] - first_projection[1],
                second_pixel[0] * second_projection[2] - second_projection[0],
                second_pixel[1] * second_projection[2] - second_projection[1],
            )
        )
        _, _, right = np.linalg.svd(system, full_matrices=False)
        homogeneous = right[-1]
        _require(
            abs(homogeneous[3]) > 1e-10,
            "triangulated filament point lies at infinity",
        )
        points.append(homogeneous[:3] / homogeneous[3])
    output = np.asarray(points, dtype=np.float64)
    for pose in (first_camera_to_world, second_camera_to_world):
        world_to_camera = np.linalg.inv(np.asarray(pose, dtype=np.float64))
        depth = output @ world_to_camera[2, :3] + world_to_camera[2, 3]
        _require(np.all(depth > 1e-5), "triangulated filament lies behind a camera")
    return output


def initialize_filament_from_multiview_masks(
    seed_points_world_m: np.ndarray,
    masks_by_camera: Mapping[str, np.ndarray],
    intrinsics_by_camera: Mapping[str, np.ndarray],
    camera_to_world_by_camera: Mapping[str, np.ndarray],
    *,
    gripper_masks_by_camera: Mapping[str, np.ndarray] | None = None,
    camera_reliability_by_camera: Mapping[str, float] | None = None,
    target_length_m: float | None = None,
    config: FilamentMaskInitializerConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select a plane-lifted or triangulated medial path by all-view support."""

    cfg = config or FilamentMaskInitializerConfig()
    if target_length_m is not None:
        _require(target_length_m > 0.0, "target filament length must be positive")
    cameras = tuple(sorted(masks_by_camera))
    _require(len(cameras) >= 2, "filament initialization needs at least two views")
    _require(
        all(camera in intrinsics_by_camera for camera in cameras)
        and all(camera in camera_to_world_by_camera for camera in cameras),
        "filament initialization camera calibration is incomplete",
    )
    if camera_reliability_by_camera is not None:
        _require(
            all(camera in camera_reliability_by_camera for camera in cameras),
            "filament initialization is missing camera reliability",
        )
    plane_point, plane_normal = fit_filament_seed_plane(seed_points_world_m)
    paths: dict[str, np.ndarray] = {}
    path_records = []
    for camera in cameras:
        try:
            pixels, diagnostics = extract_filament_mask_centerline(
                masks_by_camera[camera],
                gripper_mask=(
                    None
                    if gripper_masks_by_camera is None
                    else gripper_masks_by_camera[camera]
                ),
                config=cfg,
            )
            paths[camera] = pixels
            path_records.append(
                {"camera": camera, "passed": True, "diagnostics": diagnostics}
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            path_records.append(
                {"camera": camera, "passed": False, "failure": str(error)}
            )
    _require(len(paths) >= 2, "fewer than two filament medial paths passed")

    candidate_source_cameras = tuple(
        camera
        for camera in sorted(paths)
        if camera_reliability_by_camera is None
        or camera_reliability_by_camera[camera] >= cfg.minimum_source_camera_reliability
    )
    _require(
        len(candidate_source_cameras) >= 2,
        "fewer than two reliable filament initializer cameras remain",
    )

    candidates = []

    def add_candidate(
        centerline: np.ndarray,
        *,
        method: str,
        source_cameras: tuple[str, ...],
        orientation: str,
    ) -> None:
        raw_length = float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).sum())
        if not 1e-4 < raw_length <= cfg.maximum_candidate_length_m:
            return
        if (
            target_length_m is not None
            and raw_length < cfg.minimum_raw_length_fraction_of_target * target_length_m
        ):
            return
        scored_centerline = centerline
        if target_length_m is not None:
            centroid = np.mean(centerline, axis=0)
            scored_centerline = centroid + (centerline - centroid) * (
                target_length_m / raw_length
            )
        support = filament_multiview_support_diagnostics(
            scored_centerline,
            masks_by_camera,
            intrinsics_by_camera,
            camera_to_world_by_camera,
            gripper_masks_by_camera=gripper_masks_by_camera,
            observed_centerline_pixels_by_camera=paths,
            camera_reliability_by_camera=camera_reliability_by_camera,
            samples_per_edge=cfg.samples_per_edge_for_scoring,
            maximum_mask_sample_count=cfg.maximum_mask_sample_count_for_scoring,
        )
        use_weighted = camera_reliability_by_camera is not None
        support_summary = (
            support["reliability_weighted_visibility_aware_mask_support"]
            if use_weighted
            else support["visibility_aware_mask_support"]
        )
        coverage_summary = (
            support["reliability_weighted_mask_coverage_p95_px"]
            if use_weighted
            else support["mask_coverage_p95_px"]
        )
        median_support = support_summary["median"]
        lower_support = support_summary["lower_quartile"]
        median_coverage = coverage_summary["median"]
        score = (
            median_support + 0.5 * lower_support - 0.01 * min(median_coverage, 200.0)
        )
        if target_length_m is not None:
            score -= cfg.length_scale_penalty_weight * abs(
                np.log(raw_length / target_length_m)
            )
        candidates.append(
            {
                "method": method,
                "source_cameras": list(source_cameras),
                "orientation": orientation,
                "score": float(score),
                "centerline": scored_centerline,
                "raw_centerline_length_m": raw_length,
                "scored_centerline_length_m": float(
                    np.linalg.norm(np.diff(scored_centerline, axis=0), axis=1).sum()
                ),
                "median_mask_support": median_support,
                "lower_quartile_mask_support": lower_support,
                "median_medial_coverage_p95_px": median_coverage,
                "multiview_support": support,
            }
        )

    for camera in candidate_source_cameras:
        pixels = paths[camera]
        try:
            centerline = backproject_pixels_to_plane(
                pixels,
                intrinsics_by_camera[camera],
                camera_to_world_by_camera[camera],
                plane_point,
                plane_normal,
            )
            add_candidate(
                centerline,
                method="single-view-seed-plane",
                source_cameras=(camera,),
                orientation="as-extracted",
            )
        except (ValueError, np.linalg.LinAlgError):
            continue

    path_cameras = candidate_source_cameras
    for first_index, first_camera in enumerate(path_cameras[:-1]):
        first_pose = np.asarray(camera_to_world_by_camera[first_camera])
        first_path_length = float(
            np.linalg.norm(np.diff(paths[first_camera], axis=0), axis=1).sum()
        )
        if first_path_length < cfg.minimum_path_length_px_for_triangulation:
            continue
        for second_camera in path_cameras[first_index + 1 :]:
            second_pose = np.asarray(camera_to_world_by_camera[second_camera])
            baseline = float(np.linalg.norm(first_pose[:3, 3] - second_pose[:3, 3]))
            second_path_length = float(
                np.linalg.norm(np.diff(paths[second_camera], axis=0), axis=1).sum()
            )
            if (
                baseline < cfg.minimum_triangulation_baseline_m
                or second_path_length < cfg.minimum_path_length_px_for_triangulation
            ):
                continue
            for orientation, second_pixels in (
                ("second-forward", paths[second_camera]),
                ("second-reversed", paths[second_camera][::-1]),
            ):
                try:
                    centerline = triangulate_corresponding_pixels(
                        paths[first_camera],
                        second_pixels,
                        intrinsics_by_camera[first_camera],
                        first_pose,
                        intrinsics_by_camera[second_camera],
                        second_pose,
                    )
                    add_candidate(
                        centerline,
                        method="two-view-normalized-arc-triangulation",
                        source_cameras=(first_camera, second_camera),
                        orientation=orientation,
                    )
                except (ValueError, np.linalg.LinAlgError):
                    continue

    _require(candidates, "no multiview filament initializer candidate passed")
    selected = max(candidates, key=lambda candidate: candidate["score"])
    centerline = np.asarray(selected["centerline"], dtype=np.float64)
    serializable = []
    for candidate in candidates:
        serializable.append(
            {
                key: value
                for key, value in candidate.items()
                if key not in {"centerline", "multiview_support"}
            }
        )
    return centerline, {
        "parameters": asdict(cfg),
        "seed_plane_point_world_m": plane_point.tolist(),
        "seed_plane_normal_world": plane_normal.tolist(),
        "path_records": path_records,
        "candidate_source_cameras": list(candidate_source_cameras),
        "candidate_count": len(candidates),
        "passing_candidate_count": len(candidates),
        "selected_method": selected["method"],
        "selected_source_cameras": selected["source_cameras"],
        "selected_orientation": selected["orientation"],
        "selected_score": selected["score"],
        "target_length_m": target_length_m,
        "selected_raw_centerline_length_m": selected["raw_centerline_length_m"],
        "selected_centerline_length_m": selected["scored_centerline_length_m"],
        "candidate_scoring": (
            "source-reliability-weighted"
            if camera_reliability_by_camera is not None
            else "all-view"
        ),
        "selected_multiview_support": selected["multiview_support"],
        "candidates": serializable,
    }


__all__ = [
    "FilamentMaskInitializerConfig",
    "backproject_pixels_to_plane",
    "extract_filament_mask_centerline",
    "fit_filament_seed_plane",
    "initialize_filament_from_multiview_masks",
    "triangulate_corresponding_pixels",
]
