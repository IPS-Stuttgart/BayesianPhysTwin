"""Causal Gaussian-to-material association for reusable Deform360 filaments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra, minimum_spanning_tree

from .deform360_rope_predict import select_visual_contact_patch


MATERIAL_ASSOCIATION_SCHEMA_VERSION = 1


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _result_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True)
class FilamentMaterialAssociationConfig:
    """Frozen numerical choices for prefix-only material association."""

    observation_frame_count: int = 6
    neighbor_count: int = 12
    motion_weight: float = 1.0
    material_bandwidth: float = 0.02
    node_count: int = 21
    minimum_robust_support: float = 0.50
    minimum_opacity: float = 0.005
    contact_patch_taxel_count: int = 8
    minimum_node_contributors: int = 6

    def __post_init__(self) -> None:
        _require(self.observation_frame_count >= 2, "association needs two frames")
        _require(self.neighbor_count >= 2, "association graph needs two neighbors")
        _require(self.motion_weight >= 0.0, "motion weight must be nonnegative")
        _require(self.material_bandwidth > 0.0, "material bandwidth must be positive")
        _require(self.node_count >= 4, "filament association needs four nodes")
        _require(
            0.0 <= self.minimum_robust_support <= 1.0,
            "robust support gate must be a probability",
        )
        _require(self.minimum_opacity >= 0.0, "opacity gate must be nonnegative")
        _require(self.contact_patch_taxel_count >= 2, "contact patch is too small")
        _require(self.minimum_node_contributors >= 2, "node support is too small")


@dataclass(frozen=True)
class MaterialContactAnchor:
    robot_axis: int
    selected_taxel_indices: tuple[int, ...]
    anchor_initial_gaussian_id: int
    contact_patch_world_m: np.ndarray
    anchor_offset_m: np.ndarray
    nearest_surface_distance_m: float
    patch_to_gaussian_distance_m: float
    material_coordinate: float
    contact_node_index: int

    def __post_init__(self) -> None:
        patch = np.asarray(self.contact_patch_world_m, dtype=np.float64)
        offset = np.asarray(self.anchor_offset_m, dtype=np.float64)
        _require(self.robot_axis >= 0, "robot axis must be nonnegative")
        _require(len(self.selected_taxel_indices) >= 2, "contact patch is too small")
        _require(
            len(set(self.selected_taxel_indices)) == len(self.selected_taxel_indices),
            "contact patch repeats a taxel",
        )
        _require(self.anchor_initial_gaussian_id >= 0, "Gaussian id is invalid")
        _require(
            patch.shape == (3,) and offset.shape == (3,), "contact vectors must be 3D"
        )
        _require(
            np.all(np.isfinite(patch)) and np.all(np.isfinite(offset)),
            "contact vectors are non-finite",
        )
        _require(
            np.isfinite(self.nearest_surface_distance_m)
            and self.nearest_surface_distance_m >= 0.0,
            "surface distance is invalid",
        )
        _require(
            np.isfinite(self.patch_to_gaussian_distance_m)
            and self.patch_to_gaussian_distance_m >= 0.0,
            "patch distance is invalid",
        )
        _require(
            0.0 <= self.material_coordinate <= 1.0, "material coordinate is invalid"
        )
        _require(self.contact_node_index >= 0, "contact node is invalid")
        patch = patch.copy()
        offset = offset.copy()
        patch.setflags(write=False)
        offset.setflags(write=False)
        object.__setattr__(self, "contact_patch_world_m", patch)
        object.__setattr__(self, "anchor_offset_m", offset)

    def as_dict(self) -> dict[str, Any]:
        return {
            "robot_axis": self.robot_axis,
            "selected_taxel_indices": list(self.selected_taxel_indices),
            "anchor_initial_gaussian_id": self.anchor_initial_gaussian_id,
            "contact_patch_world_m": self.contact_patch_world_m.tolist(),
            "anchor_offset_m": self.anchor_offset_m.tolist(),
            "nearest_surface_distance_m": self.nearest_surface_distance_m,
            "patch_to_gaussian_distance_m": self.patch_to_gaussian_distance_m,
            "material_coordinate": self.material_coordinate,
            "contact_node_index": self.contact_node_index,
        }


@dataclass(frozen=True)
class Deform360MaterialAssociation:
    """One prefix-identified map from Gaussian slots to material coordinates."""

    object_id: str
    episode_id: str
    config: FilamentMaterialAssociationConfig
    selected_initial_ids: np.ndarray
    material_coordinate: np.ndarray
    slice_weights: np.ndarray
    prefix_node_tracks_m: np.ndarray
    node_reliability: np.ndarray
    node_observation_variance_m2: np.ndarray
    node_effective_sample_size: np.ndarray
    contact_anchors: tuple[MaterialContactAnchor, ...]
    graph_diagnostics: dict[str, Any]
    source_sha256: dict[str, str]

    def __post_init__(self) -> None:
        ids = np.asarray(self.selected_initial_ids, dtype=np.int64)
        coordinate = np.asarray(self.material_coordinate, dtype=np.float64)
        weights = np.asarray(self.slice_weights, dtype=np.float64)
        tracks = np.asarray(self.prefix_node_tracks_m, dtype=np.float64)
        reliability = np.asarray(self.node_reliability, dtype=np.float64)
        variance = np.asarray(self.node_observation_variance_m2, dtype=np.float64)
        effective = np.asarray(self.node_effective_sample_size, dtype=np.float64)
        _require(
            bool(self.object_id) and bool(self.episode_id),
            "association identity is missing",
        )
        _require(
            ids.ndim == 1 and len(ids) >= self.config.node_count,
            "too few selected Gaussians",
        )
        _require(
            len(np.unique(ids)) == len(ids) and np.all(ids >= 0),
            "Gaussian ids are invalid",
        )
        _require(
            coordinate.shape == (len(ids),), "material coordinates differ from ids"
        )
        _require(
            np.all(np.isfinite(coordinate))
            and np.all((coordinate >= 0.0) & (coordinate <= 1.0)),
            "material coordinates are invalid",
        )
        _require(
            weights.shape == (self.config.node_count, len(ids)),
            "material weights have the wrong shape",
        )
        _require(
            np.all(np.isfinite(weights))
            and np.all(weights >= 0.0)
            and np.allclose(np.sum(weights, axis=1), 1.0),
            "material weights are not normalized",
        )
        _require(
            tracks.shape
            == (self.config.observation_frame_count, self.config.node_count, 3)
            and np.all(np.isfinite(tracks)),
            "prefix node tracks are invalid",
        )
        _require(
            reliability.shape == (self.config.node_count,)
            and np.all(np.isfinite(reliability))
            and np.all((reliability >= 0.0) & (reliability <= 1.0)),
            "node reliability is invalid",
        )
        _require(
            variance.shape == (self.config.node_count, 3)
            and np.all(np.isfinite(variance))
            and np.all(variance >= 0.0),
            "node observation variance is invalid",
        )
        _require(
            effective.shape == (self.config.node_count,)
            and np.all(np.isfinite(effective))
            and np.all(effective >= 1.0),
            "effective sample size is invalid",
        )
        _require(
            1 <= len(self.contact_anchors) <= 2, "filament needs one or two contacts"
        )
        _require(
            len({anchor.robot_axis for anchor in self.contact_anchors})
            == len(self.contact_anchors),
            "contact axis is repeated",
        )
        _require(
            len({anchor.contact_node_index for anchor in self.contact_anchors})
            == len(self.contact_anchors),
            "contact anchors collapse onto one node",
        )
        _require(
            all(
                anchor.contact_node_index < self.config.node_count
                for anchor in self.contact_anchors
            ),
            "contact node lies outside the graph",
        )
        _require(bool(self.source_sha256), "association has no source provenance")
        _require(
            all(_valid_sha256(value) for value in self.source_sha256.values()),
            "association source checksum is invalid",
        )
        for name, value in (
            ("selected_initial_ids", ids),
            ("material_coordinate", coordinate),
            ("slice_weights", weights),
            ("prefix_node_tracks_m", tracks),
            ("node_reliability", reliability),
            ("node_observation_variance_m2", variance),
            ("node_effective_sample_size", effective),
        ):
            copied = value.copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)


def _persistent_cost(prefix_tracks_m: np.ndarray, motion_weight: float) -> np.ndarray:
    spatial = np.median(
        np.linalg.norm(
            prefix_tracks_m[:, :, None, :] - prefix_tracks_m[:, None, :, :],
            axis=3,
        ),
        axis=0,
    )
    displacement = prefix_tracks_m - prefix_tracks_m[:1]
    motion = np.sqrt(
        np.mean(
            np.sum(
                (displacement[:, :, None, :] - displacement[:, None, :, :]) ** 2,
                axis=3,
            ),
            axis=0,
        )
    )
    cost = spatial + motion_weight * motion
    np.fill_diagonal(cost, np.inf)
    return cost


def _bridge_anchor_components(
    graph,
    cost: np.ndarray,
    labels: np.ndarray,
    anchor_indices: tuple[int, ...],
) -> tuple[Any, list[dict[str, float | int]]]:
    anchor_labels = labels[np.asarray(anchor_indices)]
    if np.all(anchor_labels == anchor_labels[0]):
        return graph, []
    _require(len(anchor_indices) == 2, "component bridging requires two contacts")
    component_count = int(np.max(labels)) + 1
    component_cost = np.full((component_count, component_count), np.inf)
    component_pairs: dict[tuple[int, int], tuple[int, int]] = {}
    for first_label in range(component_count):
        first_indices = np.flatnonzero(labels == first_label)
        for second_label in range(first_label + 1, component_count):
            second_indices = np.flatnonzero(labels == second_label)
            pair_cost = cost[np.ix_(first_indices, second_indices)]
            flat_index = int(np.argmin(pair_cost))
            first_local, second_local = np.unravel_index(flat_index, pair_cost.shape)
            value = float(pair_cost[first_local, second_local])
            component_cost[first_label, second_label] = value
            component_cost[second_label, first_label] = value
            component_pairs[(first_label, second_label)] = (
                int(first_indices[first_local]),
                int(second_indices[second_local]),
            )
    tree = minimum_spanning_tree(component_cost)
    tree = (tree + tree.T).tocsr()
    start_label, end_label = map(int, anchor_labels)
    _, predecessors = dijkstra(
        tree, directed=False, indices=start_label, return_predecessors=True
    )
    label_path = [end_label]
    cursor = end_label
    while cursor != start_label:
        cursor = int(predecessors[cursor])
        _require(cursor >= 0, "cannot bridge contact components")
        label_path.append(cursor)
    label_path.reverse()
    output = graph.tolil()
    bridges = []
    for first_label, second_label in zip(label_path[:-1], label_path[1:]):
        first_index, second_index = component_pairs[
            tuple(sorted((first_label, second_label)))
        ]
        value = float(cost[first_index, second_index])
        output[first_index, second_index] = value
        output[second_index, first_index] = value
        bridges.append(
            {
                "first_index": first_index,
                "second_index": second_index,
                "persistent_cost_m": value,
            }
        )
    return output.tocsr(), bridges


def _anchored_coordinate(
    cost: np.ndarray,
    points_m: np.ndarray,
    anchor_indices: tuple[int, ...],
    *,
    neighbor_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    _require(neighbor_count < len(cost), "neighbor count exceeds selected Gaussians")
    neighbors = np.argpartition(cost, neighbor_count, axis=1)[:, :neighbor_count]
    rows = np.repeat(np.arange(len(cost)), neighbor_count)
    columns = neighbors.reshape(-1)
    graph = coo_matrix((cost[rows, columns], (rows, columns)), shape=cost.shape).tocsr()
    graph = graph.maximum(graph.T)
    _, labels = connected_components(graph, directed=False, return_labels=True)
    graph, bridges = _bridge_anchor_components(graph, cost, labels, anchor_indices)
    component_count, labels = connected_components(
        graph, directed=False, return_labels=True
    )
    anchor_labels = labels[np.asarray(anchor_indices)]
    _require(
        np.all(anchor_labels == anchor_labels[0]), "contact anchors remain disconnected"
    )
    retained = np.flatnonzero(labels == anchor_labels[0])
    local_by_global = {int(value): index for index, value in enumerate(retained)}
    local_anchors = tuple(local_by_global[index] for index in anchor_indices)
    spanning = minimum_spanning_tree(graph[retained][:, retained])
    spanning = (spanning + spanning.T).tocsr()
    seed_distance = dijkstra(spanning, directed=False, indices=local_anchors[0])
    start = int(np.argmax(seed_distance))
    start_distance, predecessors = dijkstra(
        spanning, directed=False, indices=start, return_predecessors=True
    )
    end = int(np.argmax(start_distance))
    path = [end]
    cursor = end
    while cursor != start:
        cursor = int(predecessors[cursor])
        _require(cursor >= 0, "cannot reconstruct material trunk")
        path.append(cursor)
    path.reverse()
    path_array = np.asarray(path, dtype=np.int64)
    distances_to_path = dijkstra(spanning, directed=False, indices=path_array)
    nearest_path_position = np.argmin(distances_to_path, axis=0)
    path_points = points_m[retained][path_array]
    cumulative = np.concatenate(
        (np.zeros(1), np.cumsum(np.linalg.norm(np.diff(path_points, axis=0), axis=1)))
    )
    _require(cumulative[-1] > 0.0, "material trunk has zero length")
    coordinate = np.full(len(cost), np.nan)
    coordinate[retained] = (cumulative / cumulative[-1])[nearest_path_position]
    return coordinate, {
        "component_count": int(component_count),
        "retained_count": int(len(retained)),
        "path_vertex_count": int(len(path_array)),
        "path_length_m": float(cumulative[-1]),
        "path_endpoint_indices": [
            int(retained[path_array[0]]),
            int(retained[path_array[-1]]),
        ],
        "anchor_coordinate": [float(coordinate[index]) for index in anchor_indices],
        "contact_component_bridges": bridges,
    }


def _slice_weights(
    coordinate: np.ndarray,
    anchor_indices: tuple[int, ...],
    endpoint_indices: tuple[int, int],
    config: FilamentMaterialAssociationConfig,
) -> tuple[np.ndarray, tuple[int, ...]]:
    targets = np.linspace(0.0, 1.0, config.node_count)
    distance = np.abs(targets[:, None] - coordinate[None, :])
    weights = np.exp(-0.5 * (distance / config.material_bandwidth) ** 2)
    weights[:, ~np.isfinite(coordinate)] = 0.0
    finite = np.flatnonzero(np.isfinite(coordinate))
    for node_index in range(config.node_count):
        local = distance[node_index, finite] <= 2.5 * config.material_bandwidth
        if np.count_nonzero(local) < config.minimum_node_contributors:
            nearest = finite[
                np.argsort(distance[node_index, finite])[
                    : config.minimum_node_contributors
                ]
            ]
            weights[node_index] = 0.0
            weights[node_index, nearest] = 1.0
    weights /= np.sum(weights, axis=1, keepdims=True)
    weights[0] = 0.0
    weights[0, endpoint_indices[0]] = 1.0
    weights[-1] = 0.0
    weights[-1, endpoint_indices[1]] = 1.0
    anchor_nodes = tuple(
        int(np.argmin(np.abs(targets - coordinate[index]))) for index in anchor_indices
    )
    _require(len(set(anchor_nodes)) == len(anchor_nodes), "contact anchors collapse")
    for node, anchor_index in zip(anchor_nodes, anchor_indices, strict=True):
        weights[node] = 0.0
        weights[node, anchor_index] = 1.0
    return weights, anchor_nodes


def _node_uncertainty(
    selected_tracks_m: np.ndarray,
    coordinate: np.ndarray,
    weights: np.ndarray,
    robust_support: np.ndarray,
    node_tracks_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    node_count = weights.shape[0]
    scaled = np.clip(coordinate, 0.0, 1.0) * (node_count - 1)
    left = np.minimum(np.floor(scaled).astype(np.int64), node_count - 2)
    right = left + 1
    alpha = scaled - left
    interpolated = (1.0 - alpha[None, :, None]) * node_tracks_m[:, left] + alpha[
        None, :, None
    ] * node_tracks_m[:, right]
    relative_offset = selected_tracks_m - interpolated
    drift = relative_offset - relative_offset[:1]
    drift_variance = np.mean(drift**2, axis=0)
    targets = np.linspace(0.0, 1.0, node_count)
    total_length = float(
        np.sum(np.linalg.norm(np.diff(node_tracks_m[-1], axis=0), axis=1))
    )
    coordinate_spread_m2 = (
        coordinate[None, :] - targets[:, None]
    ) ** 2 * total_length**2
    variance = weights @ drift_variance + (weights * coordinate_spread_m2) @ np.ones(
        (len(coordinate), 3)
    )
    reliability = np.clip(weights @ robust_support, 0.0, 1.0)
    effective = 1.0 / np.maximum(np.sum(weights**2, axis=1), 1e-12)
    return reliability, variance, effective


def fit_contact_anchored_material_association(
    prefix_tracks_m: np.ndarray,
    robust_support: np.ndarray,
    median_opacity: np.ndarray,
    track_complete: np.ndarray,
    gripper_taxels_world_m: Sequence[np.ndarray],
    *,
    object_id: str,
    episode_id: str,
    source_sha256: Mapping[str, str],
    config: FilamentMaterialAssociationConfig = FilamentMaterialAssociationConfig(),
) -> Deform360MaterialAssociation:
    """Fit material coordinates from a causal prefix and contact geometry."""

    tracks = np.asarray(prefix_tracks_m, dtype=np.float64)
    support = np.asarray(robust_support, dtype=np.float64)
    opacity = np.asarray(median_opacity, dtype=np.float64)
    complete = np.asarray(track_complete, dtype=bool)
    _require(
        tracks.ndim == 3
        and tracks.shape[0] == config.observation_frame_count
        and tracks.shape[2] == 3,
        "prefix tracks have the wrong shape",
    )
    gaussian_count = tracks.shape[1]
    _require(
        support.shape == opacity.shape == complete.shape == (gaussian_count,),
        "prefix cue shapes differ",
    )
    finite_support = np.isfinite(support)
    _require(
        not np.any(np.isinf(support))
        and np.all(np.isfinite(opacity))
        and np.all((support[finite_support] >= 0.0) & (support[finite_support] <= 1.0)),
        "prefix cues are invalid",
    )
    support = np.nan_to_num(support, nan=0.0)
    _require(
        1 <= len(gripper_taxels_world_m) <= 2, "filament needs one or two grippers"
    )
    frame = config.observation_frame_count - 1
    finite = np.all(np.isfinite(tracks[frame]), axis=1)
    finite_ids = np.flatnonzero(finite)
    _require(len(finite_ids) >= config.node_count, "too few finite prefix Gaussians")
    anchor_ids = []
    contact_inputs = []
    for axis, taxels in enumerate(gripper_taxels_world_m):
        selected, patch, local_node, _, diagnostics = select_visual_contact_patch(
            np.asarray(taxels, dtype=np.float64),
            tracks[frame, finite],
            taxel_count=config.contact_patch_taxel_count,
        )
        anchor_id = int(finite_ids[local_node])
        anchor_ids.append(anchor_id)
        contact_inputs.append((axis, selected, patch, anchor_id, diagnostics))
    _require(len(set(anchor_ids)) == len(anchor_ids), "grippers map to one Gaussian")
    keep = (
        (support >= config.minimum_robust_support)
        & (opacity >= config.minimum_opacity)
        & complete
    )
    keep[np.asarray(anchor_ids)] = True
    selected_ids = np.flatnonzero(keep)
    _require(
        len(selected_ids) > config.neighbor_count
        and len(selected_ids) >= config.minimum_node_contributors,
        "too few reliable prefix Gaussians",
    )
    selected_tracks = tracks[:, selected_ids]
    _require(
        np.all(np.isfinite(selected_tracks)), "selected prefix track is incomplete"
    )
    local_by_id = {int(value): index for index, value in enumerate(selected_ids)}
    local_anchors = tuple(local_by_id[value] for value in anchor_ids)
    cost = _persistent_cost(selected_tracks, config.motion_weight)
    coordinate, graph = _anchored_coordinate(
        cost,
        selected_tracks[frame],
        local_anchors,
        neighbor_count=config.neighbor_count,
    )
    retained = np.isfinite(coordinate)
    selected_ids = selected_ids[retained]
    selected_tracks = selected_tracks[:, retained]
    coordinate = coordinate[retained]
    remap = np.cumsum(retained) - 1
    local_anchors = tuple(int(remap[index]) for index in local_anchors)
    endpoint_indices = tuple(
        int(remap[index]) for index in graph["path_endpoint_indices"]
    )
    weights, contact_nodes = _slice_weights(
        coordinate,
        local_anchors,
        endpoint_indices,
        config,
    )
    node_tracks = np.einsum("nm,fmd->fnd", weights, selected_tracks)
    selected_support = support[selected_ids]
    reliability, variance, effective = _node_uncertainty(
        selected_tracks,
        coordinate,
        weights,
        selected_support,
        node_tracks,
    )
    contacts = []
    for contact_input, local_anchor, contact_node in zip(
        contact_inputs, local_anchors, contact_nodes, strict=True
    ):
        axis, selected_taxels, patch, _, diagnostics = contact_input
        gaussian_id = int(selected_ids[local_anchor])
        contacts.append(
            MaterialContactAnchor(
                robot_axis=axis,
                selected_taxel_indices=tuple(map(int, selected_taxels)),
                anchor_initial_gaussian_id=gaussian_id,
                contact_patch_world_m=patch,
                anchor_offset_m=selected_tracks[frame, local_anchor] - patch,
                nearest_surface_distance_m=float(
                    diagnostics["nearest_surface_distance_m"]
                ),
                patch_to_gaussian_distance_m=float(
                    diagnostics["patch_to_node_distance_m"]
                ),
                material_coordinate=float(coordinate[local_anchor]),
                contact_node_index=int(contact_node),
            )
        )
    graph = dict(graph)
    graph["prefix_endpoint_length_m"] = float(
        np.sum(np.linalg.norm(np.diff(node_tracks[-1], axis=0), axis=1))
    )
    graph["uncertainty_policy"] = (
        "normalized cue reliability; correlated contributors do not accumulate "
        "independently; metric variance includes temporal drift and assignment spread"
    )
    return Deform360MaterialAssociation(
        object_id=object_id,
        episode_id=episode_id,
        config=config,
        selected_initial_ids=selected_ids,
        material_coordinate=coordinate,
        slice_weights=weights,
        prefix_node_tracks_m=node_tracks,
        node_reliability=reliability,
        node_observation_variance_m2=variance,
        node_effective_sample_size=effective,
        contact_anchors=tuple(contacts),
        graph_diagnostics=graph,
        source_sha256=dict(source_sha256),
    )


def write_material_association_artifact(
    metadata_path: str | Path,
    association: Deform360MaterialAssociation,
) -> tuple[Path, Path]:
    metadata = Path(metadata_path)
    archive = metadata.with_suffix(".npz")
    metadata.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        archive,
        selected_initial_ids=association.selected_initial_ids,
        material_coordinate=association.material_coordinate,
        slice_weights=association.slice_weights,
        prefix_node_tracks_m=association.prefix_node_tracks_m,
        node_reliability=association.node_reliability,
        node_observation_variance_m2=association.node_observation_variance_m2,
        node_effective_sample_size=association.node_effective_sample_size,
    )
    payload: dict[str, Any] = {
        "schema_version": MATERIAL_ASSOCIATION_SCHEMA_VERSION,
        "artifact_kind": "Deform360MaterialAssociation",
        "object_id": association.object_id,
        "episode_id": association.episode_id,
        "config": association.config.__dict__,
        "contact_anchors": [anchor.as_dict() for anchor in association.contact_anchors],
        "graph_diagnostics": association.graph_diagnostics,
        "source_sha256": association.source_sha256,
        "archive": {
            "path": archive.name,
            "sha256": _sha256_file(archive),
        },
        "information_boundary": {
            "observation_frame_count": association.config.observation_frame_count,
            "future_splats_read": False,
            "future_masks_read": False,
            "future_outcomes_used_for_association": False,
            "state_innovation_used_for_prior_reliability": False,
        },
    }
    payload["result_sha256"] = _result_sha256(payload)
    metadata.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return metadata, archive


def load_material_association_artifact(
    metadata_path: str | Path,
) -> Deform360MaterialAssociation:
    metadata = Path(metadata_path)
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    _require(
        payload.get("schema_version") == MATERIAL_ASSOCIATION_SCHEMA_VERSION,
        "material-association schema changed",
    )
    _require(
        payload.get("artifact_kind") == "Deform360MaterialAssociation",
        "material-association kind changed",
    )
    _require(
        payload.get("result_sha256") == _result_sha256(payload), "checksum mismatch"
    )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("future_splats_read") is False
        and boundary.get("future_masks_read") is False
        and boundary.get("future_outcomes_used_for_association") is False
        and boundary.get("state_innovation_used_for_prior_reliability") is False,
        "material-association information boundary changed",
    )
    archive = metadata.parent / payload["archive"]["path"]
    _require(
        _sha256_file(archive) == payload["archive"]["sha256"],
        "archive checksum mismatch",
    )
    with np.load(archive, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    contacts = tuple(
        MaterialContactAnchor(
            robot_axis=int(value["robot_axis"]),
            selected_taxel_indices=tuple(map(int, value["selected_taxel_indices"])),
            anchor_initial_gaussian_id=int(value["anchor_initial_gaussian_id"]),
            contact_patch_world_m=np.asarray(value["contact_patch_world_m"]),
            anchor_offset_m=np.asarray(value["anchor_offset_m"]),
            nearest_surface_distance_m=float(value["nearest_surface_distance_m"]),
            patch_to_gaussian_distance_m=float(value["patch_to_gaussian_distance_m"]),
            material_coordinate=float(value["material_coordinate"]),
            contact_node_index=int(value["contact_node_index"]),
        )
        for value in payload["contact_anchors"]
    )
    return Deform360MaterialAssociation(
        object_id=str(payload["object_id"]),
        episode_id=str(payload["episode_id"]),
        config=FilamentMaterialAssociationConfig(**payload["config"]),
        selected_initial_ids=arrays["selected_initial_ids"],
        material_coordinate=arrays["material_coordinate"],
        slice_weights=arrays["slice_weights"],
        prefix_node_tracks_m=arrays["prefix_node_tracks_m"],
        node_reliability=arrays["node_reliability"],
        node_observation_variance_m2=arrays["node_observation_variance_m2"],
        node_effective_sample_size=arrays["node_effective_sample_size"],
        contact_anchors=contacts,
        graph_diagnostics=dict(payload["graph_diagnostics"]),
        source_sha256=dict(payload["source_sha256"]),
    )


__all__ = [
    "Deform360MaterialAssociation",
    "FilamentMaterialAssociationConfig",
    "MaterialContactAnchor",
    "fit_contact_anchored_material_association",
    "load_material_association_artifact",
    "write_material_association_artifact",
]
