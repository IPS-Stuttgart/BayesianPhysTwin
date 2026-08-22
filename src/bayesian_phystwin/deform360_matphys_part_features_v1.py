"""Causal frame-zero DINO part features for Deform360 MatPhys runs.

This module converts a registered Deform360 physical graph and frame-zero
multiview evidence into the part artifact consumed by the opt-in MatPhys fold
ensemble.  Only the first RGB, mask, and rendered-depth frame is admissible.
The learned features influence the MatPhys proposal, never the frozen DEFORM
mean or its exact fallback.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

from .matphys_dino_features import transfer_observed_features
from .matphys_graph_parts import GraphPartPartition, graph_semantic_parts

DEFORM360_MATPHYS_PART_SCHEMA: Final = (
    "bayesian-phystwin.deform360-matphys-frame-zero-parts"
)
DEFORM360_MATPHYS_PART_VERSION: Final = 1
DEFORM360_MATPHYS_PART_CONTRACT: Final = (
    "all-calibrated-frame-zero-rgb-mask-depth-dinov2-graph-parts-v1"
)
DEFORM360_MATPHYS_PART_CLAIM_BOUNDARY: Final = (
    "The artifact uses frame-zero Deform360 RGB, mask, rendered depth, and "
    "registered geometry to condition a target-excluded MatPhys proposal. It "
    "does not use future object observations, calibrate uncertainty, alter the "
    "frozen DEFORM mean, or establish target performance."
)

FloatArray = npt.NDArray[np.floating]
IntegerArray = npt.NDArray[np.integer]


def array_sha256(value: np.ndarray) -> str:
    """Hash an array with dtype and shape framing."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def material_distribution_for_stratum(
    stratum: str,
    *,
    part_count: int,
    material_class_count: int = 10,
) -> npt.NDArray[np.float32]:
    """Return the frozen coarse Deform360-to-MatPhys material prior.

    MatPhys's public five labeled classes are sloth, zebra, cloth, rope, and
    dinosaur at indices 0--4.  Deform360 supplies only a source-side
    sheet/volumetric stratum here.  Sheets map to the cloth class; volumetric
    objects remain uniformly uncertain over the three volumetric classes.
    Unused MatPhys output classes receive zero probability.
    """

    if part_count < 1:
        raise ValueError("part_count must be positive")
    if material_class_count < 5:
        raise ValueError("material_class_count must cover MatPhys classes 0--4")
    normalized = stratum.strip().lower()
    row: npt.NDArray[np.float32] = np.zeros(
        material_class_count,
        dtype=np.float32,
    )
    if normalized == "sheet":
        row[2] = 1.0
    elif normalized == "volumetric":
        row[[0, 1, 4]] = np.float32(1.0 / 3.0)
    else:
        raise ValueError("stratum must be 'sheet' or 'volumetric'")
    return np.repeat(row[None, :], part_count, axis=0)


def aggregate_direct_node_features(
    sampled_features_by_camera: Mapping[str, np.ndarray],
    support_by_camera: Mapping[str, np.ndarray],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    """Average normalized DINO observations over independently supported views."""

    camera_names = tuple(sorted(sampled_features_by_camera))
    if not camera_names or set(camera_names) != set(support_by_camera):
        raise ValueError("sampled features and support must cover the same cameras")
    feature_sum: np.ndarray | None = None
    contributor_count: np.ndarray | None = None
    for camera in camera_names:
        sampled = np.asarray(sampled_features_by_camera[camera], dtype=np.float64)
        raw_support = np.asarray(support_by_camera[camera])
        if raw_support.dtype != np.dtype(np.bool_):
            raise ValueError(f"support mask for {camera} must have boolean dtype")
        support = raw_support.astype(np.bool_, copy=False).reshape(-1)
        if sampled.ndim != 2 or sampled.shape[0] != len(support):
            raise ValueError(f"invalid sampled feature shape for {camera}")
        if not np.all(np.isfinite(sampled)):
            raise ValueError(f"non-finite sampled features for {camera}")
        norms = np.linalg.norm(sampled, axis=1, keepdims=True)
        if np.any(norms[support] <= 1e-12):
            raise ValueError(f"zero supported feature for {camera}")
        normalized = sampled / np.maximum(norms, 1e-12)
        if feature_sum is None:
            feature_sum = np.zeros_like(normalized, dtype=np.float64)
            contributor_count = np.zeros(len(support), dtype=np.int32)
        elif sampled.shape != feature_sum.shape:
            raise ValueError("camera feature shapes disagree")
        feature_sum[support] += normalized[support]
        assert contributor_count is not None
        contributor_count[support] += 1
    assert feature_sum is not None and contributor_count is not None
    direct = contributor_count > 0
    if not np.any(direct):
        raise ValueError("no node has a direct frame-zero DINO observation")
    averaged = feature_sum / np.maximum(contributor_count[:, None], 1)
    return averaged.astype(np.float32), contributor_count


@dataclass(frozen=True)
class Deform360MatPhysPartArrays:
    """Arrays and diagnostics written to one registered MatPhys part archive."""

    point_part: npt.NDArray[np.int64]
    part_features: npt.NDArray[np.float32]
    material_distribution: npt.NDArray[np.float32]
    node_features: npt.NDArray[np.float32]
    contributor_count: npt.NDArray[np.int32]
    nearest_direct_node: npt.NDArray[np.int64]
    partition: GraphPartPartition


def build_part_arrays(
    points_m: np.ndarray,
    edges: np.ndarray,
    direct_features: np.ndarray,
    contributor_count: np.ndarray,
    *,
    stratum: str,
    part_count: int = 5,
    semantic_edge_weight: float = 4.0,
) -> Deform360MatPhysPartArrays:
    """Fill unseen graph nodes and construct deterministic semantic parts."""

    points = np.asarray(points_m, dtype=np.float32)
    links = np.asarray(edges, dtype=np.int64)
    features = np.asarray(direct_features, dtype=np.float32)
    counts = np.asarray(contributor_count, dtype=np.int32).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_m must have shape (N,3)")
    if features.ndim != 2 or len(features) != len(points) or len(counts) != len(points):
        raise ValueError("direct feature arrays must cover every graph node")
    node_features, direct_counts, nearest = transfer_observed_features(
        points,
        points,
        features,
        counts,
    )
    partition = graph_semantic_parts(
        points,
        links,
        node_features,
        part_count=part_count,
        semantic_edge_weight=semantic_edge_weight,
    )
    material = material_distribution_for_stratum(
        stratum,
        part_count=part_count,
    )
    return Deform360MatPhysPartArrays(
        point_part=partition.assignments,
        part_features=partition.part_features,
        material_distribution=material,
        node_features=node_features,
        contributor_count=direct_counts,
        nearest_direct_node=nearest,
        partition=partition,
    )


def ordinary_file(path: str | Path, *, name: str) -> Path:
    """Resolve one immutable ordinary-file input."""

    source = Path(path).absolute()
    if (
        not source.is_file()
        or source.is_symlink()
        or any(parent.is_symlink() for parent in source.parents)
    ):
        raise ValueError(f"{name} must be an ordinary non-symlink file")
    return source.resolve(strict=True)
