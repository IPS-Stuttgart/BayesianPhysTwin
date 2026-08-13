"""Physical archive construction from driven and held-controller replays."""

from __future__ import annotations

import hashlib
import heapq
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np

from .deform360_bias_aware_prospective_artifacts import PHYSICAL_ARRAY_NAMES

PHYSTWIN_ACTION_PAIR_CONTRACT: Final = "phystwin-driven-held-controller-pair-v1"
PHYSTWIN_ACTION_SUPPORT_LENGTH_SCALE_M: Final = 0.12


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def phystwin_graph_action_support(
    springs: np.ndarray,
    rest_lengths_m: np.ndarray,
    *,
    object_point_count: int,
    object_spring_count: int,
    length_scale_m: float = PHYSTWIN_ACTION_SUPPORT_LENGTH_SCALE_M,
) -> np.ndarray:
    """Propagate controller attachment support over the object spring graph."""

    edges = np.asarray(springs)
    lengths = np.asarray(rest_lengths_m, dtype=np.float64)
    _require(
        edges.ndim == 2
        and edges.shape[1] == 2
        and np.issubdtype(edges.dtype, np.integer),
        "springs must have shape (S,2) and integer endpoints",
    )
    _require(
        lengths.shape == (len(edges),)
        and np.all(np.isfinite(lengths))
        and np.all(lengths > 0.0),
        "rest lengths must be finite and positive",
    )
    _require(object_point_count > 0, "object_point_count must be positive")
    _require(
        0 < object_spring_count <= len(edges),
        "object_spring_count must select a nonempty spring prefix",
    )
    _require(
        np.isfinite(length_scale_m) and length_scale_m > 0.0,
        "length_scale_m must be finite and positive",
    )

    object_edges = np.asarray(edges[:object_spring_count], dtype=np.int64)
    _require(
        np.all((object_edges >= 0) & (object_edges < object_point_count)),
        "object springs must connect only object points",
    )
    controller_edges = np.asarray(edges[object_spring_count:], dtype=np.int64)
    anchors: set[int] = set()
    for first, second in controller_edges:
        if first < object_point_count <= second:
            anchors.add(int(first))
        elif second < object_point_count <= first:
            anchors.add(int(second))
    _require(bool(anchors), "spring graph has no controller attachment points")

    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(object_point_count)]
    for spring_index, (first, second) in enumerate(object_edges):
        length = float(lengths[spring_index])
        adjacency[int(first)].append((int(second), length))
        adjacency[int(second)].append((int(first), length))

    distances: np.ndarray = np.full(
        object_point_count,
        np.inf,
        dtype=np.float64,
    )
    queue: list[tuple[float, int]] = []
    for anchor in sorted(anchors):
        distances[anchor] = 0.0
        heapq.heappush(queue, (0.0, anchor))
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        for neighbour, edge_length in adjacency[node]:
            candidate = distance + edge_length
            if candidate < distances[neighbour]:
                distances[neighbour] = candidate
                heapq.heappush(queue, (candidate, neighbour))
    _require(
        np.all(np.isfinite(distances)),
        "controller attachments do not reach the complete object graph",
    )
    return np.exp(-distances / length_scale_m).astype(np.float32)


def build_phystwin_action_pair_arrays(
    driven_vertices_m: np.ndarray,
    held_controller_vertices_m: np.ndarray,
    *,
    springs: np.ndarray,
    rest_lengths_m: np.ndarray,
    object_spring_count: int,
) -> dict[str, np.ndarray]:
    """Create the six-array physical contract from two official Warp replays."""

    driven = np.asarray(driven_vertices_m)
    held = np.asarray(held_controller_vertices_m)
    _require(
        driven.ndim == 3
        and driven.shape[0] >= 2
        and driven.shape[1] >= 1
        and driven.shape[2] == 3,
        "driven trajectory must have shape (T,N,3)",
    )
    _require(
        held.shape == driven.shape,
        "held-controller trajectory shape differs from the driven trajectory",
    )
    _require(
        np.issubdtype(driven.dtype, np.floating)
        and np.issubdtype(held.dtype, np.floating)
        and np.all(np.isfinite(driven))
        and np.all(np.isfinite(held)),
        "paired trajectories must be finite floating arrays",
    )
    driven32 = np.asarray(driven, dtype=np.float32)
    held32 = np.asarray(held, dtype=np.float32)
    _require(
        np.array_equal(driven32[0], held32[0]),
        "paired replays do not share the exact frame-zero state",
    )
    frame_zero = driven32[0].copy()
    persistence = np.repeat(frame_zero[None], len(driven32), axis=0)
    support = phystwin_graph_action_support(
        springs,
        rest_lengths_m,
        object_point_count=driven32.shape[1],
        object_spring_count=object_spring_count,
    )
    arrays = {
        "prediction_m": driven32.copy(),
        "persistence_m": persistence,
        "driven_readout_m": driven32.copy(),
        "zero_action_readout_m": held32.copy(),
        "action_support": support,
        "frame_zero_points_m": frame_zero,
    }
    _require(set(arrays) == PHYSICAL_ARRAY_NAMES, "physical array roster changed")
    return arrays


def write_phystwin_action_pair_archive(
    path: str | Path,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Write one immutable action-pair archive and return compact provenance."""

    output = Path(path).absolute()
    _require(not output.exists(), "action-pair archive already exists")
    _require(set(arrays) == PHYSICAL_ARRAY_NAMES, "physical array roster changed")
    output.parent.mkdir(parents=True, exist_ok=True)
    stored = {name: np.asarray(arrays[name]) for name in sorted(PHYSICAL_ARRAY_NAMES)}
    np.savez_compressed(output, **stored)
    return {
        "contract": PHYSTWIN_ACTION_PAIR_CONTRACT,
        "path": str(output.resolve()),
        "sha256": _sha256(output),
        "byte_count": output.stat().st_size,
        "zero_action_semantics": "controller points held at their frame-zero values",
        "action_support": {
            "kind": "object-graph geodesic decay from controller attachments",
            "length_scale_m": PHYSTWIN_ACTION_SUPPORT_LENGTH_SCALE_M,
            "residual_independent": True,
        },
        "arrays": {
            name: {
                "shape": list(stored[name].shape),
                "dtype": str(stored[name].dtype),
                "sha256": _array_sha256(stored[name]),
            }
            for name in sorted(stored)
        },
    }


__all__ = [
    "PHYSTWIN_ACTION_PAIR_CONTRACT",
    "PHYSTWIN_ACTION_SUPPORT_LENGTH_SCALE_M",
    "build_phystwin_action_pair_arrays",
    "phystwin_graph_action_support",
    "write_phystwin_action_pair_archive",
]
