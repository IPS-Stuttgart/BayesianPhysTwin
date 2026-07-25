"""Reliability-aware graph registration for causal PokeFlex observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _points(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    _require(result.ndim == 2 and result.shape[1] == 3, f"{name} must be Nx3")
    _require(len(result) > 0, f"{name} is empty")
    _require(np.all(np.isfinite(result)), f"{name} contains non-finite values")
    return result


@dataclass(frozen=True)
class PokeFlexBayesianRegistrationConfig:
    """Frozen-shape numerical settings; values are selected on development data."""

    control_node_count: int = 128
    graph_neighbors: int = 8
    interpolation_neighbors: int = 4
    assignment_candidates: int = 4
    voxel_size_m: float = 0.004
    association_radius_m: float = 0.025
    observation_variance_m2: float = 0.005**2
    camera_bias_variance_m2: float = 0.005**2
    control_prior_variance_m2: float = 0.020**2
    laplacian_precision: float = 2.5e3
    effective_samples_per_view: float = 128.0
    maximum_clusters_per_view: int = 512
    huber_scale_m: float = 0.010
    maximum_iterations: int = 6
    maximum_rms_update_m: float = 0.030
    minimum_points_per_view: int = 32
    minimum_independent_view_count: int = 3
    residual_geometry: str = "point_to_point"

    def __post_init__(self) -> None:
        _require(self.control_node_count >= 8, "control graph is too small")
        _require(
            2 <= self.graph_neighbors < self.control_node_count,
            "graph neighbor count is invalid",
        )
        _require(
            1 <= self.interpolation_neighbors <= self.control_node_count,
            "interpolation neighbor count is invalid",
        )
        _require(self.assignment_candidates >= 1, "assignment bank is empty")
        for name in (
            "voxel_size_m",
            "association_radius_m",
            "observation_variance_m2",
            "camera_bias_variance_m2",
            "control_prior_variance_m2",
            "laplacian_precision",
            "effective_samples_per_view",
            "huber_scale_m",
            "maximum_rms_update_m",
        ):
            _require(float(getattr(self, name)) > 0.0, f"{name} must be positive")
        _require(self.maximum_iterations >= 1, "robust iteration count is invalid")
        _require(self.minimum_points_per_view >= 1, "point support is invalid")
        _require(
            self.maximum_clusters_per_view >= self.minimum_points_per_view,
            "cluster cap is smaller than minimum support",
        )
        _require(
            self.minimum_independent_view_count >= 2,
            "independent-view gate is invalid",
        )
        _require(
            self.residual_geometry in {"point_to_point", "point_to_plane"},
            "residual geometry is invalid",
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PokeFlexActionGuardConfig:
    """Prospectively frozen shrinkage policy for the action-supported update."""

    minimum_action_force_n: float = 3.0
    strong_update_force_n: float = 15.0
    weak_scale: float = 0.125
    strong_scale: float = 0.5

    def __post_init__(self) -> None:
        _require(self.minimum_action_force_n >= 0.0, "action force gate is invalid")
        _require(
            self.strong_update_force_n > self.minimum_action_force_n,
            "strong-action threshold must exceed the support threshold",
        )
        _require(0.0 <= self.weak_scale <= self.strong_scale, "guard scales are invalid")

    def selected_scale(
        self,
        force_n: float,
        *,
        observation_update_accepted: bool,
        action_supported: bool,
    ) -> float:
        """Return zero exactly unless both physical and observation support exist."""

        if not observation_update_accepted or not action_supported:
            return 0.0
        if not np.isfinite(force_n) or force_n < self.minimum_action_force_n:
            return 0.0
        if force_n >= self.strong_update_force_n:
            return self.strong_scale
        return self.weak_scale

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class PokeFlexGraphRegistrationResult:
    """Posterior state and diagnostics from one causal graph update."""

    posterior_vertices_m: np.ndarray
    control_displacements_m: np.ndarray
    camera_biases_m: np.ndarray
    control_covariance_m2: np.ndarray
    accepted: bool
    reason: str
    diagnostics: dict[str, object]


def depth_image_to_world_points(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    *,
    depth_scale: float = 1000.0,
    depth_trunc_m: float = 2.5,
    stride: int = 1,
) -> np.ndarray:
    """Back-project one PokeFlex depth image using the upstream convention."""

    image = np.asarray(depth)
    camera = np.asarray(intrinsics, dtype=np.float64)
    world_to_camera = np.asarray(extrinsics, dtype=np.float64)
    _require(image.ndim == 2, "depth image must be two-dimensional")
    _require(camera.shape == (3, 3), "depth intrinsics must be 3x3")
    _require(world_to_camera.shape == (4, 4), "depth extrinsics must be 4x4")
    _require(depth_scale > 0.0, "depth scale must be positive")
    _require(depth_trunc_m > 0.0, "depth truncation must be positive")
    _require(stride >= 1, "depth stride must be positive")

    rows, columns = np.indices(image.shape)
    values = image.astype(np.float64) / depth_scale
    valid = (values > 0.0) & (values < depth_trunc_m)
    if stride > 1:
        valid &= (rows % stride == 0) & (columns % stride == 0)
    z = values[valid]
    x = (columns[valid] - camera[0, 2]) * z / camera[0, 0]
    y = (rows[valid] - camera[1, 2]) * z / camera[1, 1]
    homogeneous = np.column_stack((x, y, z, np.ones_like(z)))
    camera_to_world = np.linalg.inv(world_to_camera)
    return (camera_to_world @ homogeneous.T).T[:, :3]


def crop_points_to_template(
    points_m: np.ndarray,
    template_vertices_m: np.ndarray,
    *,
    scale: float = 1.3,
    minimum_vertical_offset_m: float = 0.010,
) -> np.ndarray:
    """Apply the released PokeFlex axis-aligned template crop deterministically."""

    points = _points(points_m, "point cloud")
    template = _points(template_vertices_m, "template vertices")
    _require(scale >= 1.0, "template crop scale must be at least one")
    center = 0.5 * (template.min(axis=0) + template.max(axis=0))
    half_extent = 0.5 * (template.max(axis=0) - template.min(axis=0)) * scale
    lower = center - half_extent
    upper = center + half_extent
    lower[1] = template[:, 1].min() + minimum_vertical_offset_m
    mask = np.all((points >= lower) & (points <= upper), axis=1)
    return points[mask]


def voxel_cluster_centroids(points_m: np.ndarray, voxel_size_m: float) -> np.ndarray:
    """Collapse correlated pixel blocks so duplicate samples add no confidence."""

    points = _points(points_m, "voxel input")
    _require(voxel_size_m > 0.0, "voxel size must be positive")
    keys = np.floor(points / voxel_size_m).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    ordered_keys = keys[order]
    ordered_points = points[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.any(np.diff(ordered_keys, axis=0), axis=1))]
    counts = np.diff(np.r_[starts, len(ordered_points)])
    sums = np.add.reduceat(ordered_points, starts, axis=0)
    return sums / counts[:, None]


def pokeflex_correction_field_variants(
    source_prior_m: np.ndarray,
    target_prior_m: np.ndarray,
    current_correction_m: np.ndarray,
    *,
    previous_correction_m: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build causal discrepancy predictors without reading the target frame."""

    source_prior = _points(source_prior_m, "source prior")
    target_prior = _points(target_prior_m, "target prior")
    correction = _points(current_correction_m, "current correction")
    _require(
        source_prior.shape == target_prior.shape == correction.shape,
        "correction fields must share the template topology",
    )
    if previous_correction_m is None:
        previous = correction
    else:
        previous = _points(previous_correction_m, "previous correction")
        _require(
            previous.shape == correction.shape,
            "previous correction changed template topology",
        )

    center = source_prior.mean(axis=0)
    affine_design = np.column_stack(
        (np.ones(len(source_prior)), source_prior - center[None, :])
    )
    affine_coefficients = np.linalg.lstsq(affine_design, correction, rcond=None)[0]
    affine = affine_design @ affine_coefficients
    translation = np.broadcast_to(correction.mean(axis=0), correction.shape)

    prior_motion = target_prior - source_prior
    motion_energy = float(np.sum(np.square(prior_motion)))
    if motion_energy <= 1e-12:
        motion_parallel = np.zeros_like(correction)
    else:
        gain = float(np.sum(correction * prior_motion) / motion_energy)
        motion_parallel = np.clip(gain, -2.0, 2.0) * prior_motion

    previous_energy = np.sum(np.square(previous), axis=1)
    shared_gain = np.divide(
        np.sum(correction * previous, axis=1),
        previous_energy,
        out=np.zeros(len(correction), dtype=np.float64),
        where=previous_energy > 1e-12,
    )
    temporal_shared = np.clip(shared_gain, 0.0, 1.0)[:, None] * previous
    return {
        "raw": correction,
        "translation_free": correction - translation,
        "translation_only": translation,
        "affine_free": correction - affine,
        "affine_only": affine,
        "motion_parallel": motion_parallel,
        "temporal_linear": 2.0 * correction - previous,
        "temporal_mean": 0.5 * (correction + previous),
        "temporal_shared": temporal_shared,
    }


def _pokeflex_action_contact_support(
    source_prior_m: np.ndarray,
    target_prior_m: np.ndarray,
    current_correction_m: np.ndarray,
    tool_positions_m: np.ndarray,
    end_effector_positions_m: np.ndarray,
    *,
    influence_radius_m: float = 0.060,
    contact_candidate_count: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return validated state, support, and action mismatch for contact fields."""

    source_prior = _points(source_prior_m, "source prior")
    target_prior = _points(target_prior_m, "target prior")
    correction = _points(current_correction_m, "current correction")
    _require(
        source_prior.shape == target_prior.shape == correction.shape,
        "action fields must share the template topology",
    )
    tool_positions = _points(tool_positions_m, "tool positions")
    end_effector_positions = _points(
        end_effector_positions_m, "end-effector positions"
    )
    _require(
        tool_positions.shape == end_effector_positions.shape,
        "tool and end-effector histories differ",
    )
    _require(len(tool_positions) >= 2, "action prediction needs two history frames")
    _require(influence_radius_m > 0.0, "contact radius must be positive")
    _require(contact_candidate_count >= 1, "contact candidate count is invalid")

    current_state = source_prior + correction
    tool_axis = end_effector_positions[-1] - tool_positions[-1]
    axis_norm = float(np.linalg.norm(tool_axis))
    _require(axis_norm > 1e-8, "tool axis is undefined")
    tool_axis /= axis_norm
    relative = current_state - tool_positions[-1]
    axial_distance = relative @ tool_axis
    perpendicular = relative - axial_distance[:, None] * tool_axis[None, :]
    ray_score = np.sum(np.square(perpendicular), axis=1)
    ray_score += 16.0 * np.square(np.minimum(axial_distance, 0.0))
    candidate_count = min(contact_candidate_count, len(source_prior))
    candidate_indices = np.argpartition(ray_score, candidate_count - 1)[
        :candidate_count
    ]
    contact_center = np.mean(current_state[candidate_indices], axis=0)

    tool_steps = np.diff(tool_positions, axis=0)
    predicted_tool_step = np.median(tool_steps, axis=0)
    distance = np.linalg.norm(current_state - contact_center, axis=1)
    influence = np.exp(-0.5 * np.square(distance / influence_radius_m))
    influence /= max(float(np.max(influence)), 1e-12)
    prior_motion = target_prior - source_prior
    local_weight = influence / max(float(np.sum(influence)), 1e-12)
    baseline_contact_step = np.sum(local_weight[:, None] * prior_motion, axis=0)
    action_mismatch = predicted_tool_step - baseline_contact_step
    return current_state, correction, influence, action_mismatch, predicted_tool_step


def pokeflex_action_contact_fields(
    source_prior_m: np.ndarray,
    target_prior_m: np.ndarray,
    current_correction_m: np.ndarray,
    tool_positions_m: np.ndarray,
    end_effector_positions_m: np.ndarray,
    *,
    influence_radius_m: float = 0.060,
    contact_candidate_count: int = 32,
) -> dict[str, np.ndarray]:
    """Predict a local correction from causal measured-tool motion."""

    _, correction, influence, action_mismatch, _ = _pokeflex_action_contact_support(
        source_prior_m,
        target_prior_m,
        current_correction_m,
        tool_positions_m,
        end_effector_positions_m,
        influence_radius_m=influence_radius_m,
        contact_candidate_count=contact_candidate_count,
    )
    action_velocity = influence[:, None] * action_mismatch[None, :]
    local_state = influence[:, None] * (correction + action_mismatch[None, :])
    return {
        "action_velocity": action_velocity,
        "action_local_state": local_state,
        "action_augmented": correction + action_velocity,
    }


def pokeflex_force_supported_contact_fields(
    source_prior_m: np.ndarray,
    target_prior_m: np.ndarray,
    current_correction_m: np.ndarray,
    tool_positions_m: np.ndarray,
    end_effector_positions_m: np.ndarray,
    force_vectors_n: np.ndarray,
    *,
    influence_radius_m: float = 0.060,
    contact_candidate_count: int = 32,
) -> dict[str, np.ndarray]:
    """Restrict visual state innovations to measured physical directions."""

    _, correction, influence, action_mismatch, tool_step = (
        _pokeflex_action_contact_support(
            source_prior_m,
            target_prior_m,
            current_correction_m,
            tool_positions_m,
            end_effector_positions_m,
            influence_radius_m=influence_radius_m,
            contact_candidate_count=contact_candidate_count,
        )
    )
    forces = _points(force_vectors_n, "force vectors")
    force_on_object = -np.median(forces[-3:], axis=0)
    force_norm = float(np.linalg.norm(force_on_object))
    if force_norm <= 1e-8:
        zero = np.zeros_like(correction)
        return {
            "force_parallel_local_state": zero.copy(),
            "action_axis_local_state": zero.copy(),
            "force_action_plane_local_state": zero.copy(),
            "force_mean_local_state": zero.copy(),
        }
    force_direction = force_on_object / force_norm

    step_norm = float(np.linalg.norm(tool_step))
    if step_norm > 1e-8:
        action_direction = tool_step / step_norm
    else:
        tool = _points(tool_positions_m, "tool positions")
        end_effector = _points(end_effector_positions_m, "end-effector positions")
        action_direction = tool[-1] - end_effector[-1]
        action_direction /= max(float(np.linalg.norm(action_direction)), 1e-12)

    force_projection = (
        correction @ force_direction
    )[:, None] * force_direction[None, :]
    action_projection = (
        correction @ action_direction
    )[:, None] * action_direction[None, :]
    basis = np.stack((force_direction, action_direction), axis=0)
    orthogonal_basis = np.linalg.svd(basis, full_matrices=False)[2]
    rank = int(np.linalg.matrix_rank(basis, tol=1e-6))
    orthogonal_basis = orthogonal_basis[:rank]
    plane_projection = (correction @ orthogonal_basis.T) @ orthogonal_basis

    local_weight = influence / max(float(np.sum(influence)), 1e-12)
    local_mean = np.sum(local_weight[:, None] * correction, axis=0)
    force_mean = float(local_mean @ force_direction) * force_direction
    action_velocity = influence[:, None] * action_mismatch[None, :]
    return {
        "force_parallel_local_state": influence[:, None] * force_projection
        + action_velocity,
        "action_axis_local_state": influence[:, None] * action_projection
        + action_velocity,
        "force_action_plane_local_state": influence[:, None] * plane_projection
        + action_velocity,
        "force_mean_local_state": influence[:, None] * (
            force_mean[None, :] + action_mismatch[None, :]
        ),
    }


def _even_subsample(points: np.ndarray, maximum_count: int) -> np.ndarray:
    if len(points) <= maximum_count:
        return points
    indices = np.linspace(0, len(points) - 1, maximum_count, dtype=np.int64)
    return points[indices]


class _NumpyTree:
    """Small deterministic fallback for environments with no compatible SciPy."""

    def __init__(self, points: np.ndarray) -> None:
        self.points = points

    def query(self, queries: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
        query = np.asarray(queries, dtype=np.float64)
        _require(1 <= k <= len(self.points), "nearest-neighbor count is invalid")
        distance_chunks = []
        index_chunks = []
        for start in range(0, len(query), 512):
            current = query[start : start + 512]
            squared = np.sum(
                np.square(current[:, None, :] - self.points[None, :, :]), axis=2
            )
            if k == 1:
                indices = np.argmin(squared, axis=1)
                distances = np.sqrt(squared[np.arange(len(current)), indices])
            else:
                indices = np.argpartition(squared, kth=k - 1, axis=1)[:, :k]
                selected = np.take_along_axis(squared, indices, axis=1)
                order = np.argsort(selected, axis=1)
                indices = np.take_along_axis(indices, order, axis=1)
                distances = np.sqrt(np.take_along_axis(selected, order, axis=1))
            distance_chunks.append(distances)
            index_chunks.append(indices)
        return np.concatenate(distance_chunks), np.concatenate(index_chunks)


def _tree(points: np.ndarray):
    try:
        from scipy.spatial import cKDTree
    except (ImportError, ValueError):  # pragma: no cover - environment dependent
        return _NumpyTree(points)
    return cKDTree(points)


def _farthest_point_indices(points: np.ndarray, count: int) -> np.ndarray:
    _require(len(points) >= count, "control-node count exceeds vertex count")
    selected = np.empty(count, dtype=np.int64)
    center = points.mean(axis=0)
    selected[0] = int(np.argmax(np.sum(np.square(points - center), axis=1)))
    nearest_squared = np.sum(np.square(points - points[selected[0]]), axis=1)
    for index in range(1, count):
        selected[index] = int(np.argmax(nearest_squared))
        distance = np.sum(np.square(points - points[selected[index]]), axis=1)
        nearest_squared = np.minimum(nearest_squared, distance)
    return selected


def _control_laplacian(control_points: np.ndarray, neighbors: int) -> np.ndarray:
    _, indices = _tree(control_points).query(control_points, k=neighbors + 1)
    adjacency = np.zeros((len(control_points), len(control_points)), dtype=np.float64)
    for source, row in enumerate(np.asarray(indices)[:, 1:]):
        for target in row:
            target = int(target)
            distance = np.linalg.norm(control_points[source] - control_points[target])
            weight = 1.0 / max(distance, 1e-8)
            adjacency[source, target] = max(adjacency[source, target], weight)
            adjacency[target, source] = max(adjacency[target, source], weight)
    degree = adjacency.sum(axis=1)
    normalized = adjacency / np.sqrt(np.maximum(degree[:, None] * degree[None, :], 1e-12))
    return np.eye(len(control_points), dtype=np.float64) - normalized


def _interpolation_matrix(
    vertices: np.ndarray, control_points: np.ndarray, neighbors: int
) -> np.ndarray:
    distances, indices = _tree(control_points).query(vertices, k=neighbors)
    distances = np.asarray(distances, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    weights = 1.0 / np.maximum(distances, 1e-8)
    exact = distances[:, 0] <= 1e-8
    weights /= weights.sum(axis=1, keepdims=True)
    if np.any(exact):
        weights[exact] = 0.0
        weights[exact, 0] = 1.0
    result = np.zeros((len(vertices), len(control_points)), dtype=np.float64)
    rows = np.repeat(np.arange(len(vertices)), neighbors)
    result[rows, indices.reshape(-1)] = weights.reshape(-1)
    return result


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = np.asarray(faces, dtype=np.int64)
    _require(
        triangles.ndim == 2 and triangles.shape[1] == 3,
        "prior faces must be Mx3",
    )
    _require(len(triangles) > 0, "prior faces are empty")
    _require(
        np.all((triangles >= 0) & (triangles < len(vertices))),
        "prior face index is invalid",
    )
    edge_first = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
    edge_second = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
    face_normals = np.cross(edge_first, edge_second)
    normals = np.zeros_like(vertices)
    for column in range(3):
        np.add.at(normals, triangles[:, column], face_normals)
    magnitude = np.linalg.norm(normals, axis=1)
    _require(np.all(magnitude > 1e-12), "prior mesh contains undefined vertex normals")
    return normals / magnitude[:, None]


def _fallback(
    prior_vertices: np.ndarray,
    control_count: int,
    view_count: int,
    reason: str,
    diagnostics: dict[str, object],
) -> PokeFlexGraphRegistrationResult:
    return PokeFlexGraphRegistrationResult(
        posterior_vertices_m=prior_vertices,
        control_displacements_m=np.zeros((control_count, 3), dtype=np.float64),
        camera_biases_m=np.zeros((view_count, 3), dtype=np.float64),
        control_covariance_m2=np.zeros((control_count, control_count), dtype=np.float64),
        accepted=False,
        reason=reason,
        diagnostics=diagnostics,
    )


def register_pokeflex_graph_posterior(
    prior_vertices_m: np.ndarray,
    observation_views_m: Sequence[np.ndarray],
    *,
    action_supported: bool,
    prior_faces: np.ndarray | None = None,
    source_reliabilities: Sequence[float] | None = None,
    config: PokeFlexBayesianRegistrationConfig | None = None,
) -> PokeFlexGraphRegistrationResult:
    """Assimilate causal depth views without treating dense pixels as independent."""

    cfg = config or PokeFlexBayesianRegistrationConfig()
    prior = _points(prior_vertices_m, "prior vertices")
    views = tuple(_points(value, f"observation view {index}") for index, value in enumerate(observation_views_m))
    _require(bool(views), "at least one observation view is required")
    if source_reliabilities is None:
        reliabilities = np.ones(len(views), dtype=np.float64)
    else:
        reliabilities = np.asarray(source_reliabilities, dtype=np.float64)
        _require(reliabilities.shape == (len(views),), "view reliability shape changed")
        _require(
            np.all(np.isfinite(reliabilities))
            and np.all((reliabilities >= 0.0) & (reliabilities <= 1.0)),
            "view reliability is invalid",
        )

    voxelized = tuple(voxel_cluster_centroids(view, cfg.voxel_size_m) for view in views)
    clustered = tuple(
        _even_subsample(view, cfg.maximum_clusters_per_view) for view in voxelized
    )
    supported = tuple(
        index
        for index, points in enumerate(clustered)
        if len(points) >= cfg.minimum_points_per_view and reliabilities[index] > 0.0
    )
    diagnostics: dict[str, object] = {
        "source_reliabilities": reliabilities.tolist(),
        "raw_points_per_view": [len(view) for view in views],
        "voxel_clusters_per_view": [len(view) for view in voxelized],
        "clustered_points_per_view": [len(view) for view in clustered],
        "supported_view_count": len(supported),
        "action_supported": bool(action_supported),
        "correlation_treatment": "voxel clusters within view; covariance intersection across views",
        "innovation_uses_prior_reliability": False,
    }
    if not action_supported and len(supported) < cfg.minimum_independent_view_count:
        return _fallback(
            prior,
            cfg.control_node_count,
            len(views),
            "insufficient-independent-support",
            diagnostics,
        )
    if len(supported) < 2:
        return _fallback(
            prior,
            cfg.control_node_count,
            len(views),
            "insufficient-view-support",
            diagnostics,
        )

    control_indices = _farthest_point_indices(prior, cfg.control_node_count)
    control_points = prior[control_indices]
    interpolation = _interpolation_matrix(
        prior, control_points, cfg.interpolation_neighbors
    )
    laplacian = _control_laplacian(control_points, cfg.graph_neighbors)
    prior_tree = _tree(prior)
    prior_normals = None
    if cfg.residual_geometry == "point_to_plane":
        _require(prior_faces is not None, "point-to-plane update requires prior faces")
        prior_normals = _vertex_normals(prior, prior_faces)
    design_blocks = []
    targets = []
    base_variances = []
    view_ids = []
    accepted_counts = []
    active_view_count = len(supported)
    for view_index in supported:
        points = clustered[view_index]
        candidate_count = min(cfg.assignment_candidates, len(prior))
        distances, assignments = prior_tree.query(points, k=candidate_count)
        distances = np.asarray(distances, dtype=np.float64)
        assignments = np.asarray(assignments, dtype=np.int64)
        if distances.ndim == 1:
            distances = distances[:, None]
            assignments = assignments[:, None]
        associated = distances[:, 0] <= cfg.association_radius_m
        points = points[associated]
        distances = distances[associated]
        assignments = assignments[associated]
        accepted_counts.append(int(np.sum(associated)))
        if len(points) < cfg.minimum_points_per_view:
            continue

        inverse = 1.0 / np.maximum(distances, 1e-8)
        probabilities = inverse / inverse.sum(axis=1, keepdims=True)
        candidate_positions = prior[assignments]
        assignment_mean = np.sum(probabilities[:, :, None] * candidate_positions, axis=1)
        assignment_spread = np.sum(
            probabilities
            * np.sum(np.square(candidate_positions - assignment_mean[:, None, :]), axis=2),
            axis=1,
        )
        base_design = np.zeros(
            (len(points), cfg.control_node_count + len(views)), dtype=np.float64
        )
        candidate_design = interpolation[assignments]
        base_design[:, : cfg.control_node_count] = np.sum(
            probabilities[:, :, None] * candidate_design, axis=1
        )
        base_design[:, cfg.control_node_count + view_index] = 1.0
        if cfg.residual_geometry == "point_to_point":
            design = base_design
            current_target = points - assignment_mean
            current_variance = cfg.observation_variance_m2 + assignment_spread
        else:
            assert prior_normals is not None
            candidate_normals = prior_normals[assignments]
            mixed_normals = np.sum(
                probabilities[:, :, None] * candidate_normals, axis=1
            )
            normal_magnitude = np.linalg.norm(mixed_normals, axis=1)
            valid_normals = normal_magnitude > 0.25
            points = points[valid_normals]
            assignments = assignments[valid_normals]
            probabilities = probabilities[valid_normals]
            candidate_positions = candidate_positions[valid_normals]
            assignment_mean = assignment_mean[valid_normals]
            candidate_normals = candidate_normals[valid_normals]
            mixed_normals = mixed_normals[valid_normals]
            normal_magnitude = normal_magnitude[valid_normals]
            base_design = base_design[valid_normals]
            mixed_normals /= normal_magnitude[:, None]
            design = np.zeros(
                (len(points), 3 * (cfg.control_node_count + len(views))),
                dtype=np.float64,
            )
            for coordinate in range(3):
                design[:, coordinate::3] = (
                    base_design * mixed_normals[:, coordinate, None]
                )
            current_target = np.sum(
                (points - assignment_mean) * mixed_normals, axis=1
            )
            assignment_projection = np.sum(
                (candidate_positions - assignment_mean[:, None, :])
                * mixed_normals[:, None, :],
                axis=2,
            )
            projected_spread = np.sum(
                probabilities * np.square(assignment_projection), axis=1
            )
            current_variance = cfg.observation_variance_m2 + projected_spread
            assignment_spread = projected_spread
        design_blocks.append(design)
        targets.append(current_target)
        base_variances.append(current_variance)
        view_ids.append(np.full(len(points), view_index, dtype=np.int64))

    diagnostics["associated_points_per_supported_view"] = accepted_counts
    if len(design_blocks) < 2:
        return _fallback(
            prior,
            cfg.control_node_count,
            len(views),
            "insufficient-associated-support",
            diagnostics,
        )

    design = np.concatenate(design_blocks, axis=0)
    target = np.concatenate(targets, axis=0)
    variance = np.concatenate(base_variances, axis=0)
    row_views = np.concatenate(view_ids, axis=0)
    ci_weight = np.zeros(len(design), dtype=np.float64)
    for view_index in np.unique(row_views):
        mask = row_views == view_index
        count = int(np.sum(mask))
        information = min(cfg.effective_samples_per_view, float(count))
        ci_weight[mask] = (
            reliabilities[view_index]
            * information
            / count
            / active_view_count
        )

    base_dimension = cfg.control_node_count + len(views)
    base_regularizer = np.zeros((base_dimension, base_dimension), dtype=np.float64)
    base_regularizer[: cfg.control_node_count, : cfg.control_node_count] = (
        np.eye(cfg.control_node_count) / cfg.control_prior_variance_m2
        + cfg.laplacian_precision * (laplacian.T @ laplacian)
    )
    base_regularizer[cfg.control_node_count :, cfg.control_node_count :] = (
        np.eye(len(views)) / cfg.camera_bias_variance_m2
    )
    if cfg.residual_geometry == "point_to_point":
        regularizer = base_regularizer
        solution = np.zeros((base_dimension, 3), dtype=np.float64)
    else:
        regularizer = np.kron(base_regularizer, np.eye(3, dtype=np.float64))
        solution = np.zeros(3 * base_dimension, dtype=np.float64)
    robust = np.ones(len(design), dtype=np.float64)
    normal = regularizer.copy()
    for _ in range(cfg.maximum_iterations):
        weights = ci_weight * robust / variance
        normal = regularizer + design.T @ (weights[:, None] * design)
        if cfg.residual_geometry == "point_to_point":
            right = design.T @ (weights[:, None] * target)
        else:
            right = design.T @ (weights * target)
        solution = np.linalg.solve(normal, right)
        innovation = target - design @ solution
        if cfg.residual_geometry == "point_to_point":
            magnitude = np.linalg.norm(innovation, axis=1)
        else:
            magnitude = np.abs(innovation)
        robust = np.minimum(1.0, cfg.huber_scale_m / np.maximum(magnitude, 1e-12))

    if cfg.residual_geometry == "point_to_plane":
        solution = solution.reshape(base_dimension, 3)
    control_displacement = solution[: cfg.control_node_count]
    posterior = prior + interpolation @ control_displacement
    update_rms = float(np.sqrt(np.mean(np.sum(np.square(posterior - prior), axis=1))))
    diagnostics.update(
        {
            "association_count": len(design),
            "effective_information_mass": float(np.sum(ci_weight)),
            "median_robust_weight": float(np.median(robust)),
            "minimum_robust_weight": float(np.min(robust)),
            "downweighted_fraction": float(np.mean(robust < 1.0)),
            "rms_update_m": update_rms,
            "maximum_update_m": float(
                np.max(np.linalg.norm(posterior - prior, axis=1))
            ),
            "condition_number": float(np.linalg.cond(normal)),
            "assignment_variance_m2_mean": float(np.mean(variance))
            - cfg.observation_variance_m2,
            "residual_geometry": cfg.residual_geometry,
        }
    )
    if not np.all(np.isfinite(posterior)) or update_rms > cfg.maximum_rms_update_m:
        return _fallback(
            prior,
            cfg.control_node_count,
            len(views),
            "implausible-update",
            diagnostics,
        )

    covariance = np.linalg.inv(normal)
    if cfg.residual_geometry == "point_to_point":
        covariance = covariance[: cfg.control_node_count, : cfg.control_node_count]
    else:
        covariance = covariance[
            : 3 * cfg.control_node_count, : 3 * cfg.control_node_count
        ]
    return PokeFlexGraphRegistrationResult(
        posterior_vertices_m=posterior,
        control_displacements_m=control_displacement,
        camera_biases_m=solution[cfg.control_node_count :],
        control_covariance_m2=covariance,
        accepted=True,
        reason="accepted",
        diagnostics=diagnostics,
    )
