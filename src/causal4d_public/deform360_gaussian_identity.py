"""Conservative material identity for independently exported 3D Gaussians."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class GaussianIdentityConfig:
    """Residual-independent gates for sparse spatial identity recovery."""

    maximum_distance_m: float = 0.03
    stable_order_distance_m: float = 0.03
    ambiguity_margin_m: float = 0.001
    ambiguity_ratio: float = 1.25
    maximum_neighbors: int = 4

    def __post_init__(self) -> None:
        _require(self.maximum_distance_m > 0.0, "maximum distance must be positive")
        _require(
            0.0 < self.stable_order_distance_m <= self.maximum_distance_m,
            "stable-order distance must lie within the match gate",
        )
        _require(self.ambiguity_margin_m > 0.0, "ambiguity margin must be positive")
        _require(self.ambiguity_ratio > 1.0, "ambiguity ratio must exceed one")
        _require(self.maximum_neighbors >= 2, "at least two neighbors are required")


@dataclass(frozen=True)
class GaussianIdentityResult:
    """Previous-to-current assignment with explicit ambiguity uncertainty."""

    current_index_by_previous: np.ndarray
    distance_m: np.ndarray
    reliability: np.ndarray
    assignment_variance_m2: np.ndarray
    diagnostics: dict[str, Any]


def _separation_reliability(
    distance_m: float,
    alternative_distance_m: float,
    config: GaussianIdentityConfig,
) -> float:
    if not np.isfinite(distance_m) or distance_m > config.maximum_distance_m:
        return 0.0
    if np.isfinite(alternative_distance_m):
        margin_score = np.clip(
            (alternative_distance_m - distance_m) / config.ambiguity_margin_m,
            0.0,
            1.0,
        )
        ratio = alternative_distance_m / max(distance_m, 1e-12)
        ratio_score = np.clip((ratio - 1.0) / (config.ambiguity_ratio - 1.0), 0.0, 1.0)
        ambiguity_score = float(max(margin_score, ratio_score))
    else:
        ambiguity_score = 1.0
    distance_score = float(np.exp(-((distance_m / config.maximum_distance_m) ** 2)))
    return distance_score * ambiguity_score


def _neighbor_query(
    previous: np.ndarray,
    current: np.ndarray,
    maximum_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.spatial import cKDTree
    except ImportError as error:  # pragma: no cover - optional GPU-host runtime
        raise RuntimeError(
            "scipy is required for Deform360 Gaussian identity recovery"
        ) from error
    count = min(maximum_neighbors, len(current))
    distances, indices = cKDTree(current).query(previous, k=count)
    if count == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    return np.asarray(distances, dtype=np.float64), np.asarray(indices, dtype=np.int64)


def match_gaussian_identities(
    previous_means_m: np.ndarray,
    current_means_m: np.ndarray,
    config: GaussianIdentityConfig | None = None,
) -> GaussianIdentityResult:
    """Recover identities without treating equal cardinality as stable ordering.

    Same-index pairs inside the frozen motion gate are retained as a warm-start
    hypothesis. Outliers are spatially rematched with a sparse one-to-one
    assignment. Ambiguous assignments remain usable but receive low reliability
    and an explicit variance term; they do not make the whole episode invalid.
    """

    cfg = config or GaussianIdentityConfig()
    previous = np.asarray(previous_means_m, dtype=np.float64)
    current = np.asarray(current_means_m, dtype=np.float64)
    _require(
        previous.ndim == 2 and previous.shape[1] == 3,
        "previous Gaussian means must be Nx3",
    )
    _require(
        current.ndim == 2 and current.shape[1] == 3,
        "current Gaussian means must be Mx3",
    )
    _require(len(previous) > 0 and len(current) > 0, "Gaussian sets must be nonempty")
    _require(
        np.isfinite(previous).all() and np.isfinite(current).all(),
        "Gaussian means contain non-finite values",
    )

    mapping = np.full(len(previous), -1, dtype=np.int64)
    distances = np.full(len(previous), np.inf, dtype=np.float64)
    reliability = np.zeros(len(previous), dtype=np.float64)
    variance = np.full(len(previous), 4.0 * cfg.maximum_distance_m**2, dtype=np.float64)
    reserved_current: set[int] = set()
    stable_count = 0

    neighbor_distance, neighbor_index = _neighbor_query(
        previous, current, cfg.maximum_neighbors
    )
    if len(previous) == len(current):
        stable_distance = np.linalg.norm(current - previous, axis=1)
        stable_indices = np.flatnonzero(stable_distance <= cfg.stable_order_distance_m)
        for previous_index in stable_indices:
            current_index = int(previous_index)
            distance = float(stable_distance[previous_index])
            confidence = float(np.exp(-((distance / cfg.maximum_distance_m) ** 2)))
            mapping[previous_index] = current_index
            distances[previous_index] = distance
            reliability[previous_index] = confidence
            variance[previous_index] = (
                distance**2 + (1.0 - confidence) * cfg.maximum_distance_m**2
            )
            reserved_current.add(current_index)
            stable_count += 1

    unmatched_previous = np.flatnonzero(mapping < 0)
    edges: list[tuple[float, int, int, int]] = []
    for previous_index in unmatched_previous:
        for neighbor_rank, (distance, current_index) in enumerate(
            zip(
                neighbor_distance[previous_index],
                neighbor_index[previous_index],
                strict=True,
            )
        ):
            if distance > cfg.maximum_distance_m:
                continue
            current_index = int(current_index)
            if current_index in reserved_current:
                continue
            edges.append(
                (float(distance), int(previous_index), current_index, neighbor_rank)
            )
    edges.sort(key=lambda item: (item[0], item[1], item[2]))
    used_previous: set[int] = set()
    rematched_count = 0
    for distance, previous_index, current_index, neighbor_rank in edges:
        if previous_index in used_previous or current_index in reserved_current:
            continue
        alternatives = []
        for other_distance, other_index in zip(
            neighbor_distance[previous_index],
            neighbor_index[previous_index],
            strict=True,
        ):
            other_index = int(other_index)
            if other_index == current_index or other_index in reserved_current:
                continue
            alternatives.append(float(other_distance))
        alternative = min(alternatives, default=np.inf)
        confidence = _separation_reliability(distance, alternative, cfg)
        mapping[previous_index] = current_index
        distances[previous_index] = distance
        reliability[previous_index] = confidence
        variance[previous_index] = (
            distance**2 + (1.0 - confidence) * cfg.maximum_distance_m**2
        )
        used_previous.add(previous_index)
        reserved_current.add(current_index)
        rematched_count += 1

    matched = mapping >= 0
    matched_distances = distances[matched]
    diagnostics = {
        "policy": "stable-order-with-sparse-rematch-v1",
        "previous_count": int(len(previous)),
        "current_count": int(len(current)),
        "stable_order_match_count": stable_count,
        "spatial_rematch_count": rematched_count,
        "unmatched_previous_count": int(np.count_nonzero(~matched)),
        "unused_current_count": int(len(current) - len(reserved_current)),
        "match_fraction": float(np.mean(matched)),
        "effective_reliable_match_count": float(np.sum(reliability)),
        "median_match_distance_m": (
            float(np.median(matched_distances)) if len(matched_distances) else None
        ),
        "maximum_match_distance_m": (
            float(np.max(matched_distances)) if len(matched_distances) else None
        ),
        "ambiguous_match_count": int(np.count_nonzero(matched & (reliability < 0.5))),
    }
    return GaussianIdentityResult(
        current_index_by_previous=mapping,
        distance_m=distances,
        reliability=reliability,
        assignment_variance_m2=variance,
        diagnostics=diagnostics,
    )


__all__ = [
    "GaussianIdentityConfig",
    "GaussianIdentityResult",
    "match_gaussian_identities",
]
