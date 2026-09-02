"""Registered physical-action metrics for the Tracking Cloth source audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def object_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def read_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if (
        protocol.get("schema")
        != "bayesian-phystwin.tracking-cloth-action-feasibility.v1"
    ):
        raise ValueError("unexpected action-feasibility protocol schema")
    if protocol.get("schema_version") != 1:
        raise ValueError("unexpected action-feasibility protocol version")
    if protocol["source_repetitions"] != [1, 2]:
        raise ValueError("source repetitions must remain [1, 2]")
    if protocol["reserved_target_repetition"] != 3:
        raise ValueError("rep3 must remain the reserved target")
    boundary = protocol["information_boundary"]
    if boundary["target_rep3_numeric_outcomes_read"] is not False:
        raise ValueError("rep3 numerical access must remain forbidden")
    if boundary["target_protocol_authorized"] is not False:
        raise ValueError("source audit cannot authorize a target protocol")
    return protocol


def cloth_grid_edges() -> np.ndarray:
    edges: list[tuple[int, int]] = []
    for row in range(5):
        for column in range(4):
            index = 4 * row + column
            if row + 1 < 5:
                edges.append((index, index + 4))
            if column + 1 < 4:
                edges.append((index, index + 1))
    return np.asarray(edges, dtype=np.int64)


def nonneighbor_pairs() -> np.ndarray:
    pairs: list[tuple[int, int]] = []
    for left in range(20):
        left_row, left_column = divmod(left, 4)
        for right in range(left + 1, 20):
            right_row, right_column = divmod(right, 4)
            if max(abs(left_row - right_row), abs(left_column - right_column)) <= 1:
                continue
            pairs.append((left, right))
    return np.asarray(pairs, dtype=np.int64)


def pairwise_shape_change(points: np.ndarray, diameter: float) -> float:
    if points.ndim != 3 or points.shape[1:] != (20, 3) or points.shape[0] < 2:
        raise ValueError("points must have shape (time>=2, 20, 3)")
    if not np.isfinite(points).all() or not np.isfinite(diameter) or diameter <= 0.0:
        raise ValueError("points and diameter must be finite")
    first = np.linalg.norm(points[0, :, None] - points[0, None, :], axis=2)
    last = np.linalg.norm(points[-1, :, None] - points[-1, None, :], axis=2)
    upper = np.triu_indices(20, k=1)
    return float(np.sqrt(np.mean(np.square(last[upper] - first[upper]))) / diameter)


def causal_fill_truth(truth: np.ndarray) -> tuple[np.ndarray, float]:
    """Causally carry the last finite coordinate through marker dropouts."""

    values = np.asarray(truth, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (20, 3) or values.shape[0] < 2:
        raise ValueError("truth must have shape (time>=2, 20, 3)")
    missing_fraction = float(np.mean(~np.isfinite(values)))
    output = values.copy()
    if not np.isfinite(output[0]).all():
        raise ValueError("the first scored cloth frame must be complete")
    for index in range(1, output.shape[0]):
        missing = ~np.isfinite(output[index])
        output[index][missing] = output[index - 1][missing]
    if not np.isfinite(output).all():
        raise ValueError("truth remains nonfinite after causal carry-forward")
    return output, missing_fraction


def physical_action_metrics(
    truth: np.ndarray,
    *,
    cutoff: int,
    contact_distance_m: float,
    edge_strain_weight: float,
    edge_strain_quantile: float,
    initial_diameter_m: float,
) -> dict[str, float]:
    if truth.ndim != 3 or truth.shape[1:] != (20, 3):
        raise ValueError("truth must have shape (time, 20, 3)")
    if cutoff < 1 or cutoff >= truth.shape[0] - 1:
        raise ValueError("cutoff does not leave a future trajectory")
    if not np.isfinite(truth).all():
        raise ValueError("truth must be finite")
    if not 0.0 < edge_strain_quantile <= 1.0:
        raise ValueError("edge_strain_quantile must be in (0, 1]")

    future = truth[cutoff + 1 :]
    pairs = nonneighbor_pairs()
    pair_distance = np.linalg.norm(
        future[:, pairs[:, 0]] - future[:, pairs[:, 1]],
        axis=2,
    )
    penetration = np.maximum(contact_distance_m - pair_distance, 0.0)
    contact_fraction = float(np.mean(pair_distance < contact_distance_m))
    contact_depth_rms = float(
        np.sqrt(np.mean(np.square(penetration))) / contact_distance_m
    )

    edges = cloth_grid_edges()
    initial_edges = np.linalg.norm(
        truth[0, edges[:, 0]] - truth[0, edges[:, 1]],
        axis=1,
    )
    if np.any(initial_edges <= 1e-8):
        raise ValueError("initial cloth edge length is degenerate")
    future_edges = np.linalg.norm(
        future[:, edges[:, 0]] - future[:, edges[:, 1]],
        axis=2,
    )
    absolute_strain = np.abs(future_edges / initial_edges[None, :] - 1.0)
    edge_strain_q = float(np.quantile(absolute_strain, edge_strain_quantile))
    edge_strain_peak = float(np.max(absolute_strain))
    task_loss = contact_depth_rms + edge_strain_weight * edge_strain_q

    prefix = truth[: cutoff + 1]
    probe_feature = pairwise_shape_change(prefix, initial_diameter_m)
    return {
        "task_loss": float(task_loss),
        "contact_fraction": contact_fraction,
        "contact_depth_rms": contact_depth_rms,
        "edge_strain_quantile": edge_strain_q,
        "edge_strain_peak": edge_strain_peak,
        "probe_feature": probe_feature,
    }
