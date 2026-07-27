"""Stable Bayesian-PhysTwin graph and controller surface for Causal4D.

This module is deliberately NumPy-only. Causal4D should import graph construction,
graph value types, and released controller-grouping semantics from this versioned
surface rather than from experiment modules.
"""

from __future__ import annotations

import json
import os
from importlib.metadata import PackageNotFoundError, distribution, version

import numpy as np

from .phystwin_graph import (
    PhysTwinSpringGraph,
    PhysTwinSpringGraphConfig,
    build_phystwin_spring_graph,
)

CAUSAL4D_GRAPH_PROVIDER_API_VERSION = 1
CAUSAL4D_GRAPH_PROVIDER_PACKAGE_VERSION = "0.4.0"
CAUSAL4D_GRAPH_PROVIDER_CAPABILITIES = (
    "controller_grouping",
    "phystwin_spring_graph",
)
CAUSAL4D_GRAPH_ARTIFACT_SCHEMA_VERSIONS = {
    "PhysTwinSpringGraph": 1,
}


def _installed_provider_version() -> str:
    try:
        return version("bayesian-phystwin")
    except PackageNotFoundError:
        return CAUSAL4D_GRAPH_PROVIDER_PACKAGE_VERSION


def _installed_provider_revision() -> str | None:
    try:
        direct_url = distribution("bayesian-phystwin").read_text("direct_url.json")
    except PackageNotFoundError:
        return None
    if not direct_url:
        return None
    try:
        payload = json.loads(direct_url)
    except (TypeError, json.JSONDecodeError):
        return None
    commit_id = payload.get("vcs_info", {}).get("commit_id")
    return str(commit_id) if commit_id else None


def causal4d_graph_provider_manifest(
    *,
    provider_revision: str | None = None,
) -> dict[str, object]:
    """Return the versioned graph-provider descriptor consumed by Causal4D."""

    revision = (
        provider_revision
        or os.environ.get("BAYESIAN_PHYSTWIN_REVISION")
        or _installed_provider_revision()
        or "unversioned-install"
    )
    return {
        "provider_name": "bayesian-phystwin",
        "provider_version": _installed_provider_version(),
        "provider_revision": revision,
        "schema_version": CAUSAL4D_GRAPH_PROVIDER_API_VERSION,
        "capabilities": list(CAUSAL4D_GRAPH_PROVIDER_CAPABILITIES),
        "artifact_schema_versions": dict(CAUSAL4D_GRAPH_ARTIFACT_SCHEMA_VERSIONS),
        "metadata": {
            "provider_api": "bayesian_phystwin.causal4d_graph_provider_v1",
            "provider_api_version": CAUSAL4D_GRAPH_PROVIDER_API_VERSION,
        },
    }


def controller_hand_count(case_name: str) -> int:
    """Infer the released one- or two-hand interaction contract.

    This is kept on the lightweight provider surface because graph construction and
    Causal4D contact hypotheses need the same grouping convention without importing
    the controller-sensitivity experiment implementation.
    """

    if not isinstance(case_name, str) or not case_name:
        raise ValueError("case_name must be a nonempty string")
    return (
        2
        if case_name.startswith("double_") or case_name == "rope_double_hand"
        else 1
    )


def infer_controller_groups(
    initial_controller_points: np.ndarray,
    *,
    group_count: int,
) -> np.ndarray:
    """Partition controller points into deterministic spatial hand groups."""

    points = np.asarray(initial_controller_points, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < group_count
        or not np.all(np.isfinite(points))
    ):
        raise ValueError(
            "initial_controller_points must contain finite shape (C>=G, 3)"
        )
    if group_count == 1:
        return np.zeros(len(points), dtype=np.int32)
    if group_count != 2:
        raise ValueError("released controller grouping supports one or two hands")

    squared = np.sum(np.square(points[:, None] - points[None]), axis=2)
    first, second = np.unravel_index(int(np.argmax(squared)), squared.shape)
    centroids = np.stack((points[first], points[second]))
    labels = np.zeros(len(points), dtype=np.int32)
    for _ in range(32):
        distances = np.sum(np.square(points[:, None] - centroids[None]), axis=2)
        updated = np.argmin(distances, axis=1).astype(np.int32)
        if np.all(updated == updated[0]):
            axis = centroids[1] - centroids[0]
            coordinate = points @ axis
            order = np.argsort(coordinate, kind="stable")
            updated[order[len(points) // 2 :]] = 1
        new_centroids = np.stack(
            [np.mean(points[updated == group], axis=0) for group in range(2)]
        )
        if np.array_equal(updated, labels) and np.allclose(new_centroids, centroids):
            labels = updated
            break
        labels = updated
        centroids = new_centroids
    if set(labels.tolist()) != {0, 1}:
        raise RuntimeError("controller hand partition produced an empty group")

    first_centroid = np.mean(points[labels == 0], axis=0)
    second_centroid = np.mean(points[labels == 1], axis=0)
    difference = second_centroid - first_centroid
    dominant = int(np.argmax(np.abs(difference)))
    if difference[dominant] < 0.0:
        labels = 1 - labels
    return labels.astype(np.int32)


__all__ = [
    "CAUSAL4D_GRAPH_ARTIFACT_SCHEMA_VERSIONS",
    "CAUSAL4D_GRAPH_PROVIDER_API_VERSION",
    "CAUSAL4D_GRAPH_PROVIDER_CAPABILITIES",
    "CAUSAL4D_GRAPH_PROVIDER_PACKAGE_VERSION",
    "PhysTwinSpringGraph",
    "PhysTwinSpringGraphConfig",
    "build_phystwin_spring_graph",
    "causal4d_graph_provider_manifest",
    "controller_hand_count",
    "infer_controller_groups",
]
