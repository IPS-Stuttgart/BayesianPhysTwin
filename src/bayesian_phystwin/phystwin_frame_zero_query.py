"""Frame-zero-only association of sparse query points with PhysTwin nodes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrameZeroQueryAssociation:
    """Nearest-node association fixed from frame-zero geometry only."""

    node_indices: np.ndarray
    distance_m: np.ndarray


def associate_frame_zero_queries(
    initial_node_positions_m: np.ndarray,
    query_positions_m: np.ndarray,
) -> FrameZeroQueryAssociation:
    """Associate finite frame-zero queries without seeing later observations."""

    nodes = np.asarray(initial_node_positions_m, dtype=float)
    queries = np.asarray(query_positions_m, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 3 or len(nodes) == 0:
        raise ValueError("initial_node_positions_m must have nonempty shape (N, 3)")
    if queries.ndim != 2 or queries.shape[1] != 3:
        raise ValueError("query_positions_m must have shape (Q, 3)")
    if not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(queries)):
        raise ValueError("frame-zero node and query positions must be finite")
    if len(queries) == 0:
        return FrameZeroQueryAssociation(
            node_indices=np.empty(0, dtype=np.int64),
            distance_m=np.empty(0, dtype=float),
        )

    distance = np.linalg.norm(nodes[:, None] - queries[None], axis=2)
    node_indices = np.argmin(distance, axis=0).astype(np.int64)
    query_indices = np.arange(len(queries))
    return FrameZeroQueryAssociation(
        node_indices=node_indices,
        distance_m=distance[node_indices, query_indices],
    )
