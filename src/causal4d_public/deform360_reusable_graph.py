"""Canonical dense PhysTwin graph registration across Deform360 episodes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import permutations, product
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra, minimum_spanning_tree
from scipy.spatial.distance import cdist

from bayesian_phystwin.phystwin_graph import (
    PhysTwinSpringGraph,
    PhysTwinSpringGraphConfig,
    build_phystwin_spring_graph,
)


REUSABLE_GRAPH_SCHEMA_VERSION = 1
EPISODE_REGISTRATION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ReusableGraphRegistrationConfig:
    """Source-only controls for automatic cross-episode material association."""

    canonical_node_count: int = 192
    geometry_sigma_m: float = 0.01
    color_sigma: float = 0.20
    color_cost_weight: float = 0.25
    assignment_temperature: float = 0.5
    measurement_variance_m2: float = 4e-6
    maximum_match_distance_m: float = 0.02
    minimum_match_fraction: float = 0.95
    minimum_effective_reliable_fraction: float = 0.80
    icp_iterations: int = 6
    trim_fraction: float = 0.90
    use_pca_multistart: bool = True

    def validate(self) -> None:
        if self.canonical_node_count < 4:
            raise ValueError("canonical_node_count must be at least four")
        if self.geometry_sigma_m <= 0.0 or self.color_sigma <= 0.0:
            raise ValueError("association scales must be positive")
        if self.color_cost_weight < 0.0:
            raise ValueError("color_cost_weight must be non-negative")
        if self.assignment_temperature <= 0.0:
            raise ValueError("assignment_temperature must be positive")
        if self.measurement_variance_m2 < 0.0:
            raise ValueError("measurement_variance_m2 must be non-negative")
        if self.maximum_match_distance_m <= 0.0:
            raise ValueError("maximum_match_distance_m must be positive")
        for name, value in (
            ("minimum_match_fraction", self.minimum_match_fraction),
            (
                "minimum_effective_reliable_fraction",
                self.minimum_effective_reliable_fraction,
            ),
            ("trim_fraction", self.trim_fraction),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")
        if self.icp_iterations < 1:
            raise ValueError("icp_iterations must be positive")


@dataclass(frozen=True)
class CanonicalDeform360Graph:
    """One immutable object graph, appearance map, and material identity set."""

    vertices: np.ndarray
    colors: np.ndarray
    source_indices: np.ndarray
    springs: np.ndarray
    rest_lengths: np.ndarray
    masses: np.ndarray
    bridge_spring_count: int
    observed_node_count: int
    latent_node_count: int
    contact_anchor_indices: np.ndarray
    contact_chain_spring_count: int
    sha256: str


@dataclass(frozen=True)
class EpisodeGraphRegistration:
    """One-to-one canonical-to-episode association inferred without dynamics."""

    rotation: np.ndarray
    translation: np.ndarray
    target_indices: np.ndarray
    geometric_error_m: np.ndarray
    color_error: np.ndarray
    assignment_probability: np.ndarray
    assignment_entropy: np.ndarray
    prior_reliability: np.ndarray
    observation_covariance_m2: np.ndarray
    matched_fraction: float
    effective_reliable_fraction: float
    passed: bool


def _points(value: np.ndarray, *, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if len(points) < 1 or not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must be non-empty and finite")
    return points


def _colors(value: np.ndarray, *, count: int, name: str) -> np.ndarray:
    colors = np.asarray(value, dtype=np.float64)
    if colors.shape != (count, 3) or not np.all(np.isfinite(colors)):
        raise ValueError(f"{name} must have finite shape ({count}, 3)")
    return colors


def _array_digest(digest: Any, name: str, value: np.ndarray, dtype: Any) -> None:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    digest.update(name.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def reusable_graph_sha256(
    vertices: np.ndarray,
    colors: np.ndarray,
    source_indices: np.ndarray,
    springs: np.ndarray,
    rest_lengths: np.ndarray,
    *,
    contact_anchor_indices: np.ndarray | None = None,
) -> str:
    """Hash geometry, topology, appearance, and source material identities."""

    points = _points(vertices, name="vertices")
    rgb = _colors(colors, count=len(points), name="colors")
    indices = np.asarray(source_indices, dtype=np.int64)
    edges = np.asarray(springs, dtype=np.int32)
    lengths = np.asarray(rest_lengths, dtype=np.float64)
    if indices.shape != (len(points),) or len(np.unique(indices)) != len(indices):
        raise ValueError("source_indices must be unique per canonical vertex")
    if edges.ndim != 2 or edges.shape[1] != 2 or lengths.shape != (len(edges),):
        raise ValueError("springs and rest_lengths must agree")
    if len(edges) < 1 or np.any(edges < 0) or np.any(edges >= len(points)):
        raise ValueError("canonical springs are empty or out of bounds")
    if np.any(lengths <= 0.0) or not np.all(np.isfinite(lengths)):
        raise ValueError("canonical rest lengths must be positive and finite")
    anchors = (
        np.empty(0, dtype=np.int64)
        if contact_anchor_indices is None
        else np.asarray(contact_anchor_indices, dtype=np.int64)
    )
    if anchors.ndim != 1 or np.any(anchors < 0) or np.any(anchors >= len(points)):
        raise ValueError("contact_anchor_indices are out of bounds")
    digest = hashlib.sha256()
    _array_digest(digest, "vertices", points, np.float32)
    _array_digest(digest, "colors", rgb, np.float32)
    _array_digest(digest, "source_indices", indices, np.int64)
    _array_digest(digest, "springs", edges, np.int32)
    _array_digest(digest, "rest_lengths", lengths, np.float32)
    _array_digest(digest, "contact_anchor_indices", anchors, np.int64)
    return digest.hexdigest()


def deterministic_farthest_point_indices(
    points: np.ndarray,
    count: int,
) -> np.ndarray:
    """Select a deterministic, spatially covering subset without randomness."""

    positions = _points(points, name="points")
    if not 1 <= count <= len(positions):
        raise ValueError("count must lie in [1, len(points)]")
    center = np.mean(positions, axis=0)
    selected = np.empty(count, dtype=np.int64)
    selected[0] = int(np.argmax(np.sum((positions - center) ** 2, axis=1)))
    minimum_distance_sq = np.sum(
        (positions - positions[selected[0]]) ** 2,
        axis=1,
    )
    minimum_distance_sq[selected[0]] = -np.inf
    for offset in range(1, count):
        selected[offset] = int(np.argmax(minimum_distance_sq))
        distance_sq = np.sum(
            (positions - positions[selected[offset]]) ** 2,
            axis=1,
        )
        minimum_distance_sq = np.minimum(minimum_distance_sq, distance_sq)
        minimum_distance_sq[selected[: offset + 1]] = -np.inf
    return selected


def build_canonical_deform360_graph(
    reference_points: np.ndarray,
    reference_colors: np.ndarray,
    *,
    registration_config: ReusableGraphRegistrationConfig,
    spring_config: PhysTwinSpringGraphConfig,
    reference_controller_points: np.ndarray | None = None,
    controller_group_size: int = 768,
    contact_clearance_m: float = 0.002,
) -> CanonicalDeform360Graph:
    """Freeze a dense source-reference graph before another episode is scored."""

    registration_config.validate()
    points = _points(reference_points, name="reference_points")
    colors = _colors(
        reference_colors,
        count=len(points),
        name="reference_colors",
    )
    indices = deterministic_farthest_point_indices(
        points,
        registration_config.canonical_node_count,
    )
    vertices = points[indices].astype(np.float32)
    rgb = colors[indices].astype(np.float32)
    graph = build_phystwin_spring_graph(
        vertices,
        None,
        config=spring_config,
    )
    (
        vertices,
        rgb,
        indices,
        springs,
        rest_lengths,
        bridge_count,
        latent_count,
    ) = _connect_material_components(
        vertices,
        rgb,
        indices,
        graph.springs,
        graph.rest_lengths,
        maximum_bridge_segment_length=0.5 * spring_config.object_radius,
    )
    contact_anchors = np.empty(0, dtype=np.int64)
    contact_chain_count = 0
    if reference_controller_points is not None:
        (
            vertices,
            rgb,
            indices,
            springs,
            rest_lengths,
            contact_anchors,
            contact_chain_count,
            contact_latent_count,
        ) = _append_contact_chains(
            vertices,
            rgb,
            indices,
            springs,
            rest_lengths,
            reference_controller_points,
            observed_node_count=registration_config.canonical_node_count,
            controller_group_size=controller_group_size,
            maximum_segment_length=0.5 * spring_config.object_radius,
            contact_clearance_m=contact_clearance_m,
        )
        bridge_count += contact_chain_count
        latent_count += contact_latent_count
    digest = reusable_graph_sha256(
        vertices,
        rgb,
        indices,
        springs,
        rest_lengths,
        contact_anchor_indices=contact_anchors,
    )
    return CanonicalDeform360Graph(
        vertices=vertices,
        colors=rgb,
        source_indices=indices,
        springs=springs,
        rest_lengths=rest_lengths,
        masses=np.ones(len(vertices), dtype=np.float32),
        bridge_spring_count=bridge_count,
        observed_node_count=registration_config.canonical_node_count,
        latent_node_count=latent_count,
        contact_anchor_indices=contact_anchors,
        contact_chain_spring_count=contact_chain_count,
        sha256=digest,
    )


def _append_contact_chains(
    vertices: np.ndarray,
    colors: np.ndarray,
    source_indices: np.ndarray,
    springs: np.ndarray,
    rest_lengths: np.ndarray,
    controller_points: np.ndarray,
    *,
    observed_node_count: int,
    controller_group_size: int,
    maximum_segment_length: float,
    contact_clearance_m: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
]:
    """Complete source grasp occlusions with latent material chains."""

    points = _points(vertices, name="vertices")
    rgb = _colors(colors, count=len(points), name="colors")
    material_indices = np.asarray(source_indices, dtype=np.int64)
    controls = _points(controller_points, name="controller_points")
    if not 0 < observed_node_count <= len(points):
        raise ValueError("observed_node_count is invalid")
    if controller_group_size < 1 or len(controls) % controller_group_size:
        raise ValueError("controller points do not form locked groups")
    if maximum_segment_length <= 0.0:
        raise ValueError("maximum_segment_length must be positive")
    if not 1e-4 < contact_clearance_m < maximum_segment_length:
        raise ValueError(
            "contact_clearance_m must lie between 1e-4 and the segment length"
        )
    expanded_points = [point.copy() for point in points]
    expanded_colors = [color.copy() for color in rgb]
    expanded_indices = [int(index) for index in material_indices]
    next_latent_index = int(np.min(material_indices, initial=0)) - 1
    contact_edges: list[tuple[int, int]] = []
    contact_lengths: list[float] = []
    anchors: list[int] = []
    observed_points = points[:observed_node_count]
    for start in range(0, len(controls), controller_group_size):
        group = controls[start : start + controller_group_size]
        distances = cdist(observed_points, group)
        nearest = np.unravel_index(np.argmin(distances), distances.shape)
        object_index = int(nearest[0])
        controller_position = group[int(nearest[1])]
        controller_to_object = points[object_index] - controller_position
        controller_distance = float(np.linalg.norm(controller_to_object))
        if controller_distance <= contact_clearance_m:
            anchors.append(object_index)
            continue
        contact_position = controller_position + (
            contact_clearance_m * controller_to_object / controller_distance
        )
        displacement = contact_position - points[object_index]
        distance = float(np.linalg.norm(displacement))
        segment_count = max(1, int(np.ceil(distance / maximum_segment_length)))
        chain = [object_index]
        for segment in range(1, segment_count + 1):
            fraction = segment / segment_count
            chain.append(len(expanded_points))
            expanded_points.append(points[object_index] + fraction * displacement)
            expanded_colors.append(rgb[object_index].copy())
            expanded_indices.append(next_latent_index)
            next_latent_index -= 1
        anchors.append(chain[-1])
        segment_length = distance / segment_count
        contact_edges.extend(zip(chain[:-1], chain[1:]))
        contact_lengths.extend([segment_length] * segment_count)
    connected_edges = np.concatenate(
        (springs, np.asarray(contact_edges, dtype=np.int32).reshape(-1, 2)),
        axis=0,
    )
    connected_lengths = np.concatenate(
        (rest_lengths, np.asarray(contact_lengths, dtype=np.float32)),
        axis=0,
    )
    return (
        np.asarray(expanded_points, dtype=np.float32),
        np.asarray(expanded_colors, dtype=np.float32),
        np.asarray(expanded_indices, dtype=np.int64),
        connected_edges,
        connected_lengths,
        np.asarray(anchors, dtype=np.int64),
        len(contact_edges),
        len(expanded_points) - len(points),
    )


def _connect_material_components(
    vertices: np.ndarray,
    colors: np.ndarray,
    source_indices: np.ndarray,
    springs: np.ndarray,
    rest_lengths: np.ndarray,
    *,
    maximum_bridge_segment_length: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
]:
    """Connect occlusion-separated observations with a minimal component tree."""

    points = _points(vertices, name="vertices")
    rgb = _colors(colors, count=len(points), name="colors")
    material_indices = np.asarray(source_indices, dtype=np.int64)
    if material_indices.shape != (len(points),):
        raise ValueError("source_indices must match vertices")
    if maximum_bridge_segment_length <= 0.0:
        raise ValueError("maximum_bridge_segment_length must be positive")
    edges = np.asarray(springs, dtype=np.int32)
    lengths = np.asarray(rest_lengths, dtype=np.float32)
    if edges.ndim != 2 or edges.shape[1] != 2 or lengths.shape != (len(edges),):
        raise ValueError("springs and rest_lengths must agree")
    adjacency = coo_matrix(
        (
            np.ones(2 * len(edges), dtype=np.float64),
            (
                np.concatenate((edges[:, 0], edges[:, 1])),
                np.concatenate((edges[:, 1], edges[:, 0])),
            ),
        ),
        shape=(len(points), len(points)),
    ).tocsr()
    component_count, labels = connected_components(adjacency, directed=False)
    if component_count == 1:
        return points, rgb, material_indices, edges, lengths, 0, 0
    component_cost = np.full((component_count, component_count), np.inf)
    component_pairs: dict[tuple[int, int], tuple[int, int]] = {}
    for first in range(component_count):
        first_indices = np.flatnonzero(labels == first)
        for second in range(first + 1, component_count):
            second_indices = np.flatnonzero(labels == second)
            distances = cdist(points[first_indices], points[second_indices])
            local = np.unravel_index(np.argmin(distances), distances.shape)
            first_vertex = int(first_indices[local[0]])
            second_vertex = int(second_indices[local[1]])
            distance = float(distances[local])
            component_cost[first, second] = distance
            component_cost[second, first] = distance
            component_pairs[first, second] = (first_vertex, second_vertex)
            component_pairs[second, first] = (second_vertex, first_vertex)
    tree = minimum_spanning_tree(component_cost).tocoo()
    component_bridges = np.asarray(
        [component_pairs[int(a), int(b)] for a, b in zip(tree.row, tree.col)],
        dtype=np.int32,
    ).reshape(-1, 2)
    expanded_points = [point.copy() for point in points]
    expanded_colors = [color.copy() for color in rgb]
    expanded_indices = [int(index) for index in material_indices]
    bridge_edges: list[tuple[int, int]] = []
    bridge_lengths: list[float] = []
    next_latent_index = int(np.min(material_indices, initial=0)) - 1
    for first, second in component_bridges:
        displacement = points[second] - points[first]
        distance = float(np.linalg.norm(displacement))
        segment_count = max(
            2,
            int(np.ceil(distance / maximum_bridge_segment_length)),
        )
        chain = [int(first)]
        for segment in range(1, segment_count):
            fraction = segment / segment_count
            chain.append(len(expanded_points))
            expanded_points.append(points[first] + fraction * displacement)
            expanded_colors.append(rgb[first] + fraction * (rgb[second] - rgb[first]))
            expanded_indices.append(next_latent_index)
            next_latent_index -= 1
        chain.append(int(second))
        segment_length = distance / segment_count
        bridge_edges.extend(zip(chain[:-1], chain[1:]))
        bridge_lengths.extend([segment_length] * segment_count)
    connected_edges = np.concatenate(
        (edges, np.asarray(bridge_edges, dtype=np.int32)),
        axis=0,
    )
    connected_lengths = np.concatenate(
        (lengths, np.asarray(bridge_lengths, dtype=np.float32)),
        axis=0,
    )
    connected_points = np.asarray(expanded_points, dtype=np.float32)
    connected_colors = np.asarray(expanded_colors, dtype=np.float32)
    connected_indices = np.asarray(expanded_indices, dtype=np.int64)
    connected_adjacency = coo_matrix(
        (
            np.ones(2 * len(connected_edges), dtype=np.float64),
            (
                np.concatenate((connected_edges[:, 0], connected_edges[:, 1])),
                np.concatenate((connected_edges[:, 1], connected_edges[:, 0])),
            ),
        ),
        shape=(len(connected_points), len(connected_points)),
    )
    if connected_components(connected_adjacency, directed=False)[0] != 1:
        raise ValueError("canonical material graph could not be connected")
    latent_count = len(connected_points) - len(points)
    return (
        connected_points,
        connected_colors,
        connected_indices,
        connected_edges,
        connected_lengths,
        len(bridge_edges),
        latent_count,
    )


def write_canonical_deform360_graph(
    path: str | Path,
    graph: CanonicalDeform360Graph,
) -> dict[str, Any]:
    output = Path(path)
    if output.suffix != ".npz":
        raise ValueError("canonical graph path must end in .npz")
    digest = reusable_graph_sha256(
        graph.vertices,
        graph.colors,
        graph.source_indices,
        graph.springs,
        graph.rest_lengths,
        contact_anchor_indices=graph.contact_anchor_indices,
    )
    if digest != graph.sha256:
        raise ValueError("canonical graph digest changed")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        vertices=np.asarray(graph.vertices, dtype=np.float32),
        colors=np.asarray(graph.colors, dtype=np.float32),
        source_indices=np.asarray(graph.source_indices, dtype=np.int64),
        springs=np.asarray(graph.springs, dtype=np.int32),
        rest_lengths=np.asarray(graph.rest_lengths, dtype=np.float32),
        masses=np.asarray(graph.masses, dtype=np.float32),
        bridge_spring_count=np.asarray(graph.bridge_spring_count, dtype=np.int64),
        observed_node_count=np.asarray(graph.observed_node_count, dtype=np.int64),
        latent_node_count=np.asarray(graph.latent_node_count, dtype=np.int64),
        contact_anchor_indices=np.asarray(
            graph.contact_anchor_indices,
            dtype=np.int64,
        ),
        contact_chain_spring_count=np.asarray(
            graph.contact_chain_spring_count,
            dtype=np.int64,
        ),
        reusable_graph_sha256=np.asarray(digest),
    )
    return {
        "schema_version": REUSABLE_GRAPH_SCHEMA_VERSION,
        "artifact_kind": "Deform360CanonicalReusableGraph",
        "path": str(output.resolve()),
        "reusable_graph_sha256": digest,
        "node_count": len(graph.vertices),
        "object_spring_count": len(graph.springs),
        "bridge_spring_count": graph.bridge_spring_count,
        "observed_node_count": graph.observed_node_count,
        "latent_node_count": graph.latent_node_count,
        "contact_anchor_count": len(graph.contact_anchor_indices),
        "contact_chain_spring_count": graph.contact_chain_spring_count,
    }


def load_canonical_deform360_graph(path: str | Path) -> CanonicalDeform360Graph:
    with np.load(Path(path), allow_pickle=False) as archive:
        required = {
            "vertices",
            "colors",
            "source_indices",
            "springs",
            "rest_lengths",
            "masses",
            "bridge_spring_count",
            "observed_node_count",
            "latent_node_count",
            "contact_anchor_indices",
            "contact_chain_spring_count",
            "reusable_graph_sha256",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(
                "canonical graph is missing: " + ", ".join(sorted(missing))
            )
        values = {name: np.asarray(archive[name]) for name in required}
    digest = reusable_graph_sha256(
        values["vertices"],
        values["colors"],
        values["source_indices"],
        values["springs"],
        values["rest_lengths"],
        contact_anchor_indices=values["contact_anchor_indices"],
    )
    if str(values["reusable_graph_sha256"].item()) != digest:
        raise ValueError("canonical graph SHA-256 mismatch")
    masses = np.asarray(values["masses"], dtype=np.float32)
    if masses.shape != (len(values["vertices"]),) or np.any(masses <= 0.0):
        raise ValueError("canonical graph masses are invalid")
    return CanonicalDeform360Graph(
        vertices=np.asarray(values["vertices"], dtype=np.float32),
        colors=np.asarray(values["colors"], dtype=np.float32),
        source_indices=np.asarray(values["source_indices"], dtype=np.int64),
        springs=np.asarray(values["springs"], dtype=np.int32),
        rest_lengths=np.asarray(values["rest_lengths"], dtype=np.float32),
        masses=masses,
        bridge_spring_count=int(values["bridge_spring_count"].item()),
        observed_node_count=int(values["observed_node_count"].item()),
        latent_node_count=int(values["latent_node_count"].item()),
        contact_anchor_indices=np.asarray(
            values["contact_anchor_indices"],
            dtype=np.int64,
        ),
        contact_chain_spring_count=int(values["contact_chain_spring_count"].item()),
        sha256=digest,
    )


def _rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(source) < 3:
        raise ValueError("rigid fitting needs at least three points")
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    left, _, right = np.linalg.svd(
        (source - source_center).T @ (target - target_center),
        full_matrices=False,
    )
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right[-1] *= -1.0
        rotation = right.T @ left.T
    translation = target_center - source_center @ rotation.T
    return rotation, translation


def _principal_axes(points: np.ndarray) -> np.ndarray:
    centered = points - np.mean(points, axis=0)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    axes = right.T
    for column in range(3):
        dominant = int(np.argmax(np.abs(axes[:, column])))
        if axes[dominant, column] < 0.0:
            axes[:, column] *= -1.0
    if np.linalg.det(axes) < 0.0:
        axes[:, -1] *= -1.0
    return axes


def _initial_transforms(
    source: np.ndarray,
    target: np.ndarray,
    *,
    pca_multistart: bool,
) -> list[tuple[np.ndarray, np.ndarray]]:
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    rotations = [np.eye(3)]
    if pca_multistart:
        source_axes = _principal_axes(source)
        target_axes = _principal_axes(target)
        for order in permutations(range(3)):
            permutation = np.eye(3)[:, order]
            for signs in product((-1.0, 1.0), repeat=3):
                signed = permutation @ np.diag(signs)
                rotation = target_axes @ signed @ source_axes.T
                if np.linalg.det(rotation) > 0.0:
                    rotations.append(rotation)
    unique: list[tuple[np.ndarray, np.ndarray]] = []
    for rotation in rotations:
        if any(np.allclose(rotation, seen, atol=1e-10) for seen, _ in unique):
            continue
        translation = target_center - source_center @ rotation.T
        unique.append((rotation, translation))
    return unique


def _association_cost(
    transformed: np.ndarray,
    source_colors: np.ndarray,
    target: np.ndarray,
    target_colors: np.ndarray,
    config: ReusableGraphRegistrationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geometry_sq = cdist(transformed, target, metric="sqeuclidean")
    color_sq = cdist(source_colors, target_colors, metric="sqeuclidean")
    cost = geometry_sq / (config.geometry_sigma_m**2)
    cost += config.color_cost_weight * color_sq / (config.color_sigma**2)
    return cost, geometry_sq, color_sq


def register_canonical_graph_to_episode(
    canonical: CanonicalDeform360Graph,
    episode_points: np.ndarray,
    episode_colors: np.ndarray,
    *,
    config: ReusableGraphRegistrationConfig,
    candidate_reliability: np.ndarray | None = None,
) -> EpisodeGraphRegistration:
    """Associate one canonical graph using only observed geometry and appearance."""

    config.validate()
    source = _points(canonical.vertices, name="canonical.vertices")
    source_colors = _colors(
        canonical.colors,
        count=len(source),
        name="canonical.colors",
    )
    target = _points(episode_points, name="episode_points")
    target_colors = _colors(
        episode_colors,
        count=len(target),
        name="episode_colors",
    )
    if len(target) < len(source):
        raise ValueError("episode has fewer candidates than canonical nodes")
    if candidate_reliability is None:
        candidate_prior = np.ones(len(target), dtype=np.float64)
    else:
        candidate_prior = np.asarray(candidate_reliability, dtype=np.float64)
        if candidate_prior.shape != (len(target),) or not np.all(
            np.isfinite(candidate_prior)
        ):
            raise ValueError("candidate_reliability must be finite per target point")
        if np.any(candidate_prior < 0.0) or np.any(candidate_prior > 1.0):
            raise ValueError("candidate_reliability must lie in [0, 1]")

    best: tuple[float, np.ndarray, np.ndarray, np.ndarray] | None = None
    for initial_rotation, initial_translation in _initial_transforms(
        source,
        target,
        pca_multistart=config.use_pca_multistart,
    ):
        rotation = initial_rotation
        translation = initial_translation
        assignment = np.arange(len(source), dtype=np.int64)
        objective = np.inf
        for _ in range(config.icp_iterations):
            transformed = source @ rotation.T + translation
            cost, geometry_sq, _ = _association_cost(
                transformed,
                source_colors,
                target,
                target_colors,
                config,
            )
            cost = cost - 2.0 * np.log(np.maximum(candidate_prior, 1e-6))[None]
            row, column = linear_sum_assignment(cost)
            assignment = column[np.argsort(row)].astype(np.int64)
            assigned_distance = np.sqrt(geometry_sq[np.arange(len(source)), assignment])
            cutoff = float(np.quantile(assigned_distance, config.trim_fraction))
            retained = assigned_distance <= max(cutoff, config.geometry_sigma_m)
            if np.count_nonzero(retained) < 3:
                retained[:] = True
            rotation, translation = _rigid_fit(
                source[retained],
                target[assignment[retained]],
            )
            objective = float(np.mean(cost[np.arange(len(source)), assignment]))
        if best is None or objective < best[0]:
            best = (objective, rotation, translation, assignment)
    assert best is not None
    _, rotation, translation, assignment = best
    transformed = source @ rotation.T + translation
    cost, geometry_sq, color_sq = _association_cost(
        transformed,
        source_colors,
        target,
        target_colors,
        config,
    )
    cost = cost - 2.0 * np.log(np.maximum(candidate_prior, 1e-6))[None]
    row, column = linear_sum_assignment(cost)
    assignment = column[np.argsort(row)].astype(np.int64)
    logits = (
        -(cost - np.min(cost, axis=1, keepdims=True)) / config.assignment_temperature
    )
    logits = np.clip(logits, -700.0, 0.0)
    probability = np.exp(logits)
    probability /= np.sum(probability, axis=1, keepdims=True)
    entropy = -np.sum(probability * np.log(np.maximum(probability, 1e-300)), axis=1)
    entropy /= np.log(max(2, len(target)))
    assigned_probability = probability[np.arange(len(source)), assignment]
    geometric_error = np.sqrt(geometry_sq[np.arange(len(source)), assignment])
    color_error = np.sqrt(color_sq[np.arange(len(source)), assignment])

    mixture_mean = probability @ target
    covariance = np.empty((len(source), 3, 3), dtype=np.float64)
    for point in range(len(source)):
        residual = target - mixture_mean[point]
        covariance[point] = (
            residual.T * probability[point]
        ) @ residual + config.measurement_variance_m2 * np.eye(3)

    geometry_reliability = np.exp(
        -0.5 * (geometric_error / config.maximum_match_distance_m) ** 2
    )
    color_reliability = np.exp(-0.5 * (color_error / config.color_sigma) ** 2)
    # Association entropy already contributes its full spatial mixture spread
    # to covariance. It remains a conservative reliability cue here without
    # counting the same ambiguity as a second observation residual.
    ambiguity_reliability = np.clip(1.0 - 0.25 * entropy, 0.0, 1.0)
    prior_reliability = (
        candidate_prior[assignment]
        * geometry_reliability
        * np.sqrt(color_reliability)
        * ambiguity_reliability
    )
    matched = geometric_error <= config.maximum_match_distance_m
    matched_fraction = float(np.mean(matched))
    effective = float(np.mean(prior_reliability * matched))
    passed = bool(
        matched_fraction >= config.minimum_match_fraction
        and effective >= config.minimum_effective_reliable_fraction
    )
    return EpisodeGraphRegistration(
        rotation=rotation,
        translation=translation,
        target_indices=assignment,
        geometric_error_m=geometric_error,
        color_error=color_error,
        assignment_probability=assigned_probability,
        assignment_entropy=entropy,
        prior_reliability=prior_reliability,
        observation_covariance_m2=covariance,
        matched_fraction=matched_fraction,
        effective_reliable_fraction=effective,
        passed=passed,
    )


def canonical_reference_registration(
    canonical: CanonicalDeform360Graph,
    *,
    config: ReusableGraphRegistrationConfig,
    candidate_reliability: np.ndarray,
) -> EpisodeGraphRegistration:
    """Use exact source indices when the episode created the canonical graph."""

    config.validate()
    reliability = np.asarray(candidate_reliability, dtype=np.float64)
    if reliability.ndim != 1 or not np.all(np.isfinite(reliability)):
        raise ValueError("candidate_reliability must be a finite vector")
    if np.any(reliability < 0.0) or np.any(reliability > 1.0):
        raise ValueError("candidate_reliability must lie in [0, 1]")
    indices = np.asarray(canonical.source_indices, dtype=np.int64)
    if np.any(indices < 0):
        raise ValueError("latent canonical nodes require partial state completion")
    if np.any(indices < 0) or np.any(indices >= len(reliability)):
        raise ValueError("canonical source index exceeds reference candidates")
    node_reliability = reliability[indices]
    effective = float(np.mean(node_reliability))
    covariance = np.repeat(
        (config.measurement_variance_m2 * np.eye(3))[None],
        len(indices),
        axis=0,
    )
    return EpisodeGraphRegistration(
        rotation=np.eye(3),
        translation=np.zeros(3),
        target_indices=indices.copy(),
        geometric_error_m=np.zeros(len(indices)),
        color_error=np.zeros(len(indices)),
        assignment_probability=np.ones(len(indices)),
        assignment_entropy=np.zeros(len(indices)),
        prior_reliability=node_reliability,
        observation_covariance_m2=covariance,
        matched_fraction=1.0,
        effective_reliable_fraction=effective,
        passed=bool(effective >= config.minimum_effective_reliable_fraction),
    )


def registered_episode_data(
    data: Mapping[str, Any],
    registration: EpisodeGraphRegistration,
    *,
    canonical_graph_sha256: str,
) -> dict[str, Any]:
    """Reorder episode tracks into canonical material-node order."""

    required = {
        "object_points",
        "object_colors",
        "object_visibilities",
        "object_motions_valid",
        "controller_points",
        "surface_points",
        "interior_points",
    }
    missing = required - set(data)
    if missing:
        raise ValueError("episode data is missing: " + ", ".join(sorted(missing)))
    indices = np.asarray(registration.target_indices, dtype=np.int64)
    points = np.asarray(data["object_points"])
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("object_points must have shape (T, N, 3)")
    if np.any(indices < 0) or np.any(indices >= points.shape[1]):
        raise ValueError("registration target index is out of bounds")
    result = dict(data)
    for name in (
        "object_points",
        "object_colors",
        "object_visibilities",
        "object_motions_valid",
    ):
        values = np.asarray(data[name])
        if values.shape[1] != points.shape[1]:
            raise ValueError(f"{name} does not share the object-point axis")
        result[name] = values[:, indices].copy()
    result["surface_points"] = np.empty((0, 3), dtype=points.dtype)
    result["interior_points"] = np.empty((0, 3), dtype=points.dtype)
    result["reusable_graph_registration"] = {
        "canonical_graph_sha256": canonical_graph_sha256,
        "target_indices": indices.copy(),
        "prior_reliability": registration.prior_reliability.copy(),
        "observation_covariance_m2": registration.observation_covariance_m2.copy(),
        "matched_fraction": registration.matched_fraction,
        "effective_reliable_fraction": registration.effective_reliable_fraction,
        "passed": registration.passed,
    }
    return result


def build_registered_phystwin_graph(
    canonical: CanonicalDeform360Graph,
    episode_initial_points: np.ndarray,
    controller_reference: np.ndarray | None,
    *,
    spring_config: PhysTwinSpringGraphConfig,
    controller_patch_size: int = 1,
) -> PhysTwinSpringGraph:
    """Keep canonical object springs and rebuild only episode contact springs."""

    if controller_patch_size < 1:
        raise ValueError("controller_patch_size must be positive")
    episode_points = _points(
        episode_initial_points,
        name="episode_initial_points",
    ).astype(np.float32)
    if episode_points.shape != canonical.vertices.shape:
        raise ValueError("registered episode does not match the canonical node count")
    if controller_reference is None or not len(canonical.contact_anchor_indices):
        candidate = build_phystwin_spring_graph(
            episode_points,
            controller_reference,
            config=spring_config,
        )
        controller_springs = candidate.springs[candidate.num_object_springs :].copy()
        controller_rest = candidate.rest_lengths[candidate.num_object_springs :].copy()
        controller_vertices = candidate.vertices[len(episode_points) :]
    else:
        controller_vertices = _points(
            controller_reference,
            name="controller_reference",
        ).astype(np.float32)
        anchor_count = len(canonical.contact_anchor_indices)
        if len(controller_vertices) % anchor_count:
            raise ValueError(
                "controller points cannot be divided among canonical contact anchors"
            )
        group_size = len(controller_vertices) // anchor_count
        controller_springs_list: list[tuple[int, int]] = []
        controller_rest_list: list[float] = []
        graph_adjacency = coo_matrix(
            (
                np.concatenate((canonical.rest_lengths, canonical.rest_lengths)),
                (
                    np.concatenate((canonical.springs[:, 0], canonical.springs[:, 1])),
                    np.concatenate((canonical.springs[:, 1], canonical.springs[:, 0])),
                ),
            ),
            shape=(len(episode_points), len(episode_points)),
        ).tocsr()
        for group_index, anchor_index in enumerate(canonical.contact_anchor_indices):
            start = group_index * group_size
            stop = start + group_size
            group = controller_vertices[start:stop]
            controller_distance = cdist(episode_points, group)
            nearest_controller = np.argmin(controller_distance, axis=1)
            nearest_distance = controller_distance[
                np.arange(len(episode_points)), nearest_controller
            ]
            anchor = int(anchor_index)
            if nearest_distance[anchor] > spring_config.controller_radius:
                raise ValueError(
                    "canonical contact anchor is outside the controller radius"
                )
            if controller_patch_size == 1:
                selected_nodes = np.asarray([anchor], dtype=np.int64)
            else:
                graph_distance = np.asarray(
                    dijkstra(graph_adjacency, indices=anchor, directed=False)
                )
                admissible = np.flatnonzero(
                    np.isfinite(graph_distance)
                    & (nearest_distance <= spring_config.controller_radius)
                )
                order = np.lexsort(
                    (
                        admissible,
                        nearest_distance[admissible],
                        graph_distance[admissible],
                    )
                )
                selected_nodes = admissible[order[:controller_patch_size]]
                if anchor not in selected_nodes:
                    selected_nodes = np.concatenate(
                        (np.asarray([anchor]), selected_nodes[:-1])
                    )
            for node_index in selected_nodes:
                local_index = int(nearest_controller[int(node_index)])
                controller_springs_list.append(
                    (
                        len(episode_points) + start + local_index,
                        int(node_index),
                    )
                )
                controller_rest_list.append(float(nearest_distance[int(node_index)]))
        controller_springs = np.asarray(
            controller_springs_list,
            dtype=np.int32,
        )
        controller_rest = np.asarray(controller_rest_list, dtype=np.float32)
    springs = np.concatenate((canonical.springs, controller_springs), axis=0)
    rest_lengths = np.concatenate((canonical.rest_lengths, controller_rest), axis=0)
    controller_count = len(controller_vertices)
    vertices = np.concatenate(
        (episode_points, controller_vertices),
        axis=0,
    )
    return PhysTwinSpringGraph(
        vertices=vertices.astype(np.float32),
        springs=springs.astype(np.int32),
        rest_lengths=rest_lengths.astype(np.float32),
        masses=np.concatenate(
            (
                canonical.masses,
                np.ones(controller_count, dtype=np.float32),
            )
        ),
        num_object_springs=len(canonical.springs),
    )


def episode_registration_summary(
    registration: EpisodeGraphRegistration,
    *,
    canonical_graph_sha256: str,
    information_boundary: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a JSON-safe audit summary without hiding assignment uncertainty."""

    canonical_boundary = dict(information_boundary)
    if canonical_boundary.get("simulator_residual_used") is not False:
        raise ValueError("registration must not use simulator residuals")
    if canonical_boundary.get("future_object_frames_used") is not False:
        raise ValueError("registration must not use future object frames")
    payload = {
        "schema_version": EPISODE_REGISTRATION_SCHEMA_VERSION,
        "artifact_kind": "Deform360ReusableGraphEpisodeRegistration",
        "canonical_graph_sha256": canonical_graph_sha256,
        "rotation": registration.rotation.tolist(),
        "translation_m": registration.translation.tolist(),
        "node_count": len(registration.target_indices),
        "matched_fraction": registration.matched_fraction,
        "effective_reliable_fraction": registration.effective_reliable_fraction,
        "geometric_error_m": {
            "median": float(np.median(registration.geometric_error_m)),
            "p95": float(np.quantile(registration.geometric_error_m, 0.95)),
            "maximum": float(np.max(registration.geometric_error_m)),
        },
        "assignment": {
            "median_probability": float(np.median(registration.assignment_probability)),
            "median_normalized_entropy": float(
                np.median(registration.assignment_entropy)
            ),
            "median_covariance_trace_m2": float(
                np.median(
                    np.trace(registration.observation_covariance_m2, axis1=1, axis2=2)
                )
            ),
        },
        "passed": registration.passed,
        "information_boundary": canonical_boundary,
        "claim_boundary": (
            "automatic material association from observed prefix geometry and "
            "appearance; no simulator-residual or future-frame evidence"
        ),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    payload["result_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


__all__ = [
    "CanonicalDeform360Graph",
    "EpisodeGraphRegistration",
    "ReusableGraphRegistrationConfig",
    "build_canonical_deform360_graph",
    "build_registered_phystwin_graph",
    "canonical_reference_registration",
    "deterministic_farthest_point_indices",
    "episode_registration_summary",
    "load_canonical_deform360_graph",
    "register_canonical_graph_to_episode",
    "registered_episode_data",
    "reusable_graph_sha256",
    "write_canonical_deform360_graph",
]
