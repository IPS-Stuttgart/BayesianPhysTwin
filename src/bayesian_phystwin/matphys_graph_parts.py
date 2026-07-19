"""Deterministic graph-connected semantic parts for causal MatPhys fits."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import json
import pickle
from pathlib import Path
import shutil
from typing import Mapping, Sequence

import numpy as np

from .matphys_causal_bridge import sha256_file


GRAPH_PART_PROXY_CONTRACT = "causal-dino-graph-voronoi-parts-v1"
GRAPH_PART_COMPACT_PROXY_CONTRACT = (
    "causal-dino-graph-parts-compact-unused-edge-semantics-v1"
)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _normalized_features(value: np.ndarray) -> np.ndarray:
    features = np.asarray(value, dtype=np.float64)
    if features.ndim != 2 or len(features) == 0:
        raise ValueError("node features must have shape (N, D)")
    if not np.all(np.isfinite(features)):
        raise ValueError("node features must be finite")
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError("every node needs a nonzero semantic feature")
    return features / norm


def _validated_graph(
    points: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(points, dtype=np.float64)
    links = np.asarray(edges, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if links.ndim != 2 or links.shape[1] != 2 or len(links) == 0:
        raise ValueError("edges must have shape (E, 2)")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("points must be finite")
    if np.any(links < 0) or np.any(links >= len(vertices)):
        raise ValueError("edge endpoint exceeds the node array")
    if np.any(links[:, 0] == links[:, 1]):
        raise ValueError("self edges are not supported")
    return vertices, links


def _adjacency(
    points: np.ndarray,
    edges: np.ndarray,
    features: np.ndarray,
    semantic_edge_weight: float,
) -> list[list[tuple[int, float]]]:
    if not np.isfinite(semantic_edge_weight) or semantic_edge_weight < 0.0:
        raise ValueError("semantic edge weight must be finite and nonnegative")
    lengths = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    positive = lengths[lengths > 1e-12]
    if len(positive) == 0:
        raise ValueError("graph edges have zero geometric length")
    length_scale = float(np.median(positive))
    cosine = np.einsum("ij,ij->i", features[edges[:, 0]], features[edges[:, 1]])
    semantic_distance = np.clip(1.0 - cosine, 0.0, 2.0)
    weights = np.maximum(lengths / length_scale, 1e-6) * (
        1.0 + semantic_edge_weight * semantic_distance
    )
    result: list[list[tuple[int, float]]] = [[] for _ in range(len(points))]
    for (left, right), weight in zip(edges, weights, strict=True):
        result[int(left)].append((int(right), float(weight)))
        result[int(right)].append((int(left), float(weight)))
    for neighbors in result:
        neighbors.sort(key=lambda item: item[0])
    return result


def _components(adjacency: list[list[tuple[int, float]]]) -> list[np.ndarray]:
    labels = np.full(len(adjacency), -1, dtype=np.int64)
    result: list[np.ndarray] = []
    for start in range(len(adjacency)):
        if labels[start] >= 0:
            continue
        component_id = len(result)
        stack = [start]
        labels[start] = component_id
        members: list[int] = []
        while stack:
            node = stack.pop()
            members.append(node)
            for neighbor, _ in adjacency[node]:
                if labels[neighbor] < 0:
                    labels[neighbor] = component_id
                    stack.append(neighbor)
        result.append(np.asarray(sorted(members), dtype=np.int64))
    return result


def _distances(
    adjacency: list[list[tuple[int, float]]],
    seeds: Sequence[int],
) -> np.ndarray:
    distance = np.full(len(adjacency), np.inf, dtype=np.float64)
    queue: list[tuple[float, int]] = []
    for seed in sorted(set(int(value) for value in seeds)):
        distance[seed] = 0.0
        heapq.heappush(queue, (0.0, seed))
    while queue:
        current, node = heapq.heappop(queue)
        if current > distance[node]:
            continue
        for neighbor, weight in adjacency[node]:
            candidate = current + weight
            if candidate + 1e-12 < distance[neighbor]:
                distance[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return distance


def _voronoi_labels(
    adjacency: list[list[tuple[int, float]]],
    seeds: Sequence[int],
) -> np.ndarray:
    distance = np.full(len(adjacency), np.inf, dtype=np.float64)
    labels = np.full(len(adjacency), -1, dtype=np.int64)
    queue: list[tuple[float, int, int]] = []
    for label, seed in enumerate(seeds):
        distance[int(seed)] = 0.0
        labels[int(seed)] = label
        heapq.heappush(queue, (0.0, label, int(seed)))
    while queue:
        current, label, node = heapq.heappop(queue)
        if current > distance[node] + 1e-12:
            continue
        if abs(current - distance[node]) <= 1e-12 and label != labels[node]:
            continue
        for neighbor, weight in adjacency[node]:
            candidate = current + weight
            better = candidate + 1e-12 < distance[neighbor]
            tie = abs(candidate - distance[neighbor]) <= 1e-12
            if better or (tie and (labels[neighbor] < 0 or label < labels[neighbor])):
                distance[neighbor] = candidate
                labels[neighbor] = label
                heapq.heappush(queue, (candidate, label, neighbor))
    if np.any(labels < 0):
        raise RuntimeError("graph Voronoi assignment left nodes unassigned")
    return labels


@dataclass(frozen=True)
class GraphPartPartition:
    assignments: np.ndarray
    seeds: np.ndarray
    part_features: np.ndarray
    part_counts: np.ndarray
    boundary_edge_fraction: float
    connected_component_count: int


def graph_semantic_parts(
    points: np.ndarray,
    edges: np.ndarray,
    node_features: np.ndarray,
    *,
    part_count: int = 5,
    semantic_edge_weight: float = 4.0,
) -> GraphPartPartition:
    """Build deterministic connected parts with semantic graph geodesics.

    Seeds are spread by farthest-point sampling on the semantic-geometric
    spring graph. Multi-source shortest-path assignment then guarantees each
    Voronoi part is connected to its seed. The fixed defaults follow the
    public MatPhys five-part preprocessing recipe and are not target-tuned.
    """

    vertices, links = _validated_graph(points, edges)
    features = _normalized_features(node_features)
    if len(features) != len(vertices):
        raise ValueError("node feature count must match the graph")
    if not 1 <= part_count <= len(vertices):
        raise ValueError("part count must lie in [1, N]")
    adjacency = _adjacency(
        vertices,
        links,
        features,
        semantic_edge_weight,
    )
    components = _components(adjacency)
    if len(components) > part_count:
        raise ValueError("part count is smaller than the graph component count")

    centered = vertices - np.mean(vertices, axis=0, keepdims=True)
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    coordinate = (
        centered @ right[0]
        if singular_values[0] > 1e-12
        else np.arange(len(vertices), dtype=np.float64)
    )
    seeds = [
        int(component[np.argmin(coordinate[component])])
        for component in components
    ]
    while len(seeds) < part_count:
        distance = _distances(adjacency, seeds)
        finite = np.isfinite(distance)
        if not np.all(finite):
            candidate = int(np.flatnonzero(~finite)[0])
        else:
            maximum = float(np.max(distance))
            candidate = int(np.flatnonzero(np.isclose(distance, maximum))[0])
        if candidate in seeds:
            raise RuntimeError("farthest-point part seeding stalled")
        seeds.append(candidate)

    assignments = _voronoi_labels(adjacency, seeds)
    counts = np.bincount(assignments, minlength=part_count).astype(np.int64)
    if np.any(counts == 0):
        raise RuntimeError("graph part construction produced an empty part")
    part_features = np.stack(
        [np.mean(features[assignments == part], axis=0) for part in range(part_count)]
    )
    part_features /= np.maximum(
        np.linalg.norm(part_features, axis=1, keepdims=True),
        1e-12,
    )
    boundary_fraction = float(
        np.mean(assignments[links[:, 0]] != assignments[links[:, 1]])
    )
    return GraphPartPartition(
        assignments=assignments.astype(np.int64),
        seeds=np.asarray(seeds, dtype=np.int64),
        part_features=part_features.astype(np.float32),
        part_counts=counts,
        boundary_edge_fraction=boundary_fraction,
        connected_component_count=len(components),
    )


def prepare_graph_part_proxy(
    data_root: str | Path,
    case_names: Sequence[str],
    source_mapping_path: str | Path,
    output_root: str | Path,
    *,
    node_features_by_case: Mapping[str, np.ndarray],
    graph_edges_by_case: Mapping[str, np.ndarray],
    contributor_count_by_case: Mapping[str, np.ndarray],
    provenance_by_case: Mapping[str, Mapping[str, object]],
    part_count: int = 5,
    semantic_edge_weight: float = 4.0,
    num_materials: int = 10,
) -> dict[str, object]:
    """Write node-aligned MatPhys inputs with auditable graph-connected parts."""

    import torch

    root = Path(data_root).resolve()
    mapping_path = Path(source_mapping_path).resolve()
    destination = Path(output_root).resolve()
    raw_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    case_to_label = raw_mapping.get("case_to_material", raw_mapping)
    class_to_id = raw_mapping.get("class_to_id", {})
    if not class_to_id:
        labels = sorted(set(case_to_label.values()))
        class_to_id = {label: index for index, label in enumerate(labels)}
    selected_mapping: dict[str, str | int] = {}
    case_records: list[dict[str, object]] = []

    for case in case_names:
        missing = [
            name
            for name, source in (
                ("node features", node_features_by_case),
                ("graph edges", graph_edges_by_case),
                ("contributor counts", contributor_count_by_case),
                ("provenance", provenance_by_case),
            )
            if case not in source
        ]
        if missing:
            raise ValueError(f"{case}: missing {', '.join(missing)}")
        if case not in case_to_label:
            raise ValueError(f"MatPhys material mapping omits {case}")
        label = case_to_label[case]
        material_id = int(class_to_id[label]) if isinstance(label, str) else int(label)
        if not 0 <= material_id < num_materials:
            raise ValueError(f"{case}: material id exceeds decoder dimensions")
        with (root / case / "final_data.pkl").open("rb") as handle:
            data = pickle.load(handle)
        structure_points = np.concatenate(
            (
                np.asarray(data["object_points"])[0],
                np.asarray(data["surface_points"]),
                np.asarray(data["interior_points"]),
            ),
            axis=0,
        ).astype(np.float32)
        node_features = np.asarray(node_features_by_case[case], dtype=np.float32)
        contributor_count = np.asarray(
            contributor_count_by_case[case], dtype=np.int32
        ).reshape(-1)
        if node_features.shape[0] != len(structure_points):
            raise ValueError(f"{case}: semantic feature count does not match structure")
        if len(contributor_count) != len(structure_points):
            raise ValueError(f"{case}: contributor count does not match structure")
        edges = np.asarray(graph_edges_by_case[case], dtype=np.int64)
        partition = graph_semantic_parts(
            structure_points,
            edges,
            node_features,
            part_count=part_count,
            semantic_edge_weight=semantic_edge_weight,
        )

        node_sem_path = destination / "semantic_cache" / f"{case}_node_sem.npz"
        node_sem_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            node_sem_path,
            node_sem=node_features,
            contributor_count=contributor_count,
            part_assignments=partition.assignments,
        )
        material_distribution = torch.zeros(
            (part_count, num_materials), dtype=torch.float32
        )
        material_distribution[:, material_id] = 1.0
        train_ready_path = destination / "results" / case / "train" / "train_ready.pt"
        train_ready_path.parent.mkdir(parents=True, exist_ok=True)
        part_probs = torch.nn.functional.one_hot(
            torch.from_numpy(partition.assignments), num_classes=part_count
        ).float()
        torch.save(
            {
                "part_assignments": torch.from_numpy(partition.assignments),
                "part_probs": part_probs,
                "num_parts": part_count,
                "material_distributions": material_distribution,
                "part_features": torch.from_numpy(partition.part_features),
                "part_counts": torch.from_numpy(partition.part_counts),
                "xyz": torch.from_numpy(structure_points),
                "proxy_contract": GRAPH_PART_PROXY_CONTRACT,
            },
            train_ready_path,
        )
        selected_mapping[case] = label
        case_records.append(
            {
                "name": case,
                "material_label": label,
                "material_id": material_id,
                "structure_point_count": len(structure_points),
                "semantic_dimension": int(node_features.shape[1]),
                "direct_semantic_node_count": int(np.sum(contributor_count > 0)),
                "part_count": part_count,
                "part_counts": partition.part_counts.tolist(),
                "seed_node_indices": partition.seeds.tolist(),
                "boundary_edge_fraction": partition.boundary_edge_fraction,
                "connected_component_count": partition.connected_component_count,
                "graph_edges_sha256": _array_sha256(edges),
                "node_features_sha256": _array_sha256(node_features),
                "provenance": dict(provenance_by_case[case]),
                "node_sem": {
                    "path": str(node_sem_path),
                    "sha256": sha256_file(node_sem_path),
                },
                "train_ready": {
                    "path": str(train_ready_path),
                    "sha256": sha256_file(train_ready_path),
                },
            }
        )

    proxy_mapping_path = destination / "case_to_material.json"
    proxy_mapping_path.write_text(
        json.dumps(
            {
                "class_to_id": class_to_id,
                "case_to_material": selected_mapping,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "contract": GRAPH_PART_PROXY_CONTRACT,
        "claim_boundary": (
            "DINO descriptors use only numerically indexed fit-prefix keyframes; "
            "parts are deterministic graph Voronoi regions, not the unpublished "
            "MatPhys Gaussian-part preprocessing."
        ),
        "part_count": part_count,
        "semantic_edge_weight": semantic_edge_weight,
        "source_mapping": {
            "path": str(mapping_path),
            "sha256": sha256_file(mapping_path),
        },
        "mapping": {
            "path": str(proxy_mapping_path),
            "sha256": sha256_file(proxy_mapping_path),
        },
        "semantic_cache_dir": str(destination / "semantic_cache"),
        "results_dir": str(destination / "results"),
        "cases": case_records,
    }
    summary_path = destination / "proxy_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["summary_path"] = str(summary_path)
    return summary


def compact_graph_part_proxy(
    source_summary_path: str | Path,
    output_root: str | Path,
) -> dict[str, object]:
    """Remove edge-semantic bulk unused by the simple part-aware decoder.

    MatPhys's dataset eagerly expands node semantics to every spring even
    though ``train_model_video_material_simple`` never consumes ``z_sem`` or
    ``ctrl_sem``. The learned adapter consumes ``part_features`` from
    ``train_ready.pt`` instead. This derivative proxy preserves those bytes,
    assignments, material distributions, and graph provenance exactly while
    replacing only the dead node-semantic input with one constant channel.
    """

    source_path = Path(source_summary_path).resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("contract") != GRAPH_PART_PROXY_CONTRACT:
        raise ValueError("compact conversion requires a full graph-part proxy")
    destination = Path(output_root).resolve()
    mapping_source = Path(source["mapping"]["path"]).resolve()
    if sha256_file(mapping_source) != source["mapping"]["sha256"]:
        raise ValueError("source graph-part mapping bytes changed")
    mapping_path = destination / "case_to_material.json"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(mapping_source, mapping_path)

    import torch

    records = []
    for raw_record in source.get("cases", []):
        case = str(raw_record["name"])
        source_node_sem = raw_record["node_sem"]
        source_train_ready = raw_record["train_ready"]
        if sha256_file(source_node_sem["path"]) != source_node_sem["sha256"]:
            raise ValueError(f"{case}: source node semantics changed")
        if sha256_file(source_train_ready["path"]) != source_train_ready["sha256"]:
            raise ValueError(f"{case}: source train-ready bytes changed")
        train_ready = torch.load(
            source_train_ready["path"], map_location="cpu", weights_only=False
        )
        point_count = int(train_ready["xyz"].shape[0])
        if point_count != int(raw_record["structure_point_count"]):
            raise ValueError(f"{case}: source point count changed")

        node_sem_path = destination / "semantic_cache" / f"{case}_node_sem.npz"
        node_sem_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            node_sem_path,
            node_sem=np.ones((point_count, 1), dtype=np.float32),
        )
        train_ready_path = destination / "results" / case / "train" / "train_ready.pt"
        train_ready_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_train_ready["path"], train_ready_path)
        records.append(
            {
                **raw_record,
                "source_node_sem": dict(source_node_sem),
                "node_sem": {
                    "path": str(node_sem_path),
                    "sha256": sha256_file(node_sem_path),
                },
                "train_ready": {
                    "path": str(train_ready_path),
                    "sha256": sha256_file(train_ready_path),
                },
                "edge_semantic_dimension": 1,
                "edge_semantics_consumed_by_model": False,
            }
        )
    if not records:
        raise ValueError("source graph-part proxy contains no cases")

    summary = {
        **source,
        "contract": GRAPH_PART_COMPACT_PROXY_CONTRACT,
        "claim_boundary": (
            f"{source.get('claim_boundary', '')} The 1024-D DINO part features "
            "and all part assignments remain unchanged; only node/edge semantic "
            "tensors unused by the simple decoder are represented by one constant "
            "channel."
        ).strip(),
        "source_proxy": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
        },
        "mapping": {"path": str(mapping_path), "sha256": sha256_file(mapping_path)},
        "semantic_cache_dir": str(destination / "semantic_cache"),
        "results_dir": str(destination / "results"),
        "cases": records,
    }
    summary_path = destination / "proxy_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["summary_path"] = str(summary_path)
    return summary
