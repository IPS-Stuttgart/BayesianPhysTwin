"""Outcome-free pairwise-consensus gate for sparse point correspondences.

The online belief field assumes that each sparse observation retains its
material identity.  A tracker identity swap violates that assumption and can
turn one bad centre into a dense RBF correction.  This module detects such
failures using only the current source/observation geometry: two proposed
correspondences are compatible when their pairwise distance changes by no
more than a fixed absolute-plus-relative strain envelope.

For the small centre sets used by PhysTwin (16 points), the maximum compatible
subset is found exactly.  The detector is invariant to a common rigid motion,
does not inspect a future target, and deliberately returns only a decision and
an inlier mask.  The caller owns the exact-backbone fallback contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairwiseCorrespondenceGateConfig:
    """Frozen geometric tolerances for maximum-consensus correspondences.

    The 30 mm absolute envelope is deliberately conservative for the planned
    5 mm coordinate-noise stress: it exceeds four standard deviations of a
    difference of two independent scalar range errors.  The relative term
    allows modest material strain on longer baselines.  Requiring at least 70
    percent consensus gives the gate a stated operating regime below 30
    percent correspondence contamination; a 50 percent corruption should
    normally abstain rather than pretend to identify the correct half.
    """

    absolute_pair_strain_m: float = 0.030
    relative_pair_strain: float = 0.10
    minimum_inlier_count: int = 9
    minimum_inlier_fraction: float = 0.70
    maximum_exact_center_count: int = 24

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.absolute_pair_strain_m)
            or self.absolute_pair_strain_m <= 0.0
        ):
            raise ValueError("absolute_pair_strain_m must be positive")
        if (
            not np.isfinite(self.relative_pair_strain)
            or self.relative_pair_strain < 0.0
        ):
            raise ValueError("relative_pair_strain must be nonnegative")
        if self.minimum_inlier_count < 2:
            raise ValueError("minimum_inlier_count must be at least two")
        if not np.isfinite(self.minimum_inlier_fraction) or not (
            0.0 < self.minimum_inlier_fraction <= 1.0
        ):
            raise ValueError("minimum_inlier_fraction must lie in (0, 1]")
        if self.maximum_exact_center_count < self.minimum_inlier_count:
            raise ValueError(
                "maximum_exact_center_count must cover minimum_inlier_count"
            )


@dataclass(frozen=True)
class PairwiseCorrespondenceGateResult:
    """Immutable maximum-consensus result on the caller's full centre axis."""

    inlier_mask: np.ndarray
    accepted: bool
    decision: str
    available_count: int
    inlier_count: int
    inlier_fraction: float
    pair_count: int
    compatible_pair_fraction: float
    median_inlier_normalized_strain: float | None
    maximum_inlier_normalized_strain: float | None

    def __post_init__(self) -> None:
        mask = np.asarray(self.inlier_mask, dtype=bool).copy()
        if mask.ndim != 1:
            raise ValueError("inlier_mask must be a vector")
        if self.available_count < 0 or self.available_count > len(mask):
            raise ValueError("available_count is inconsistent with inlier_mask")
        if self.inlier_count != int(np.sum(mask)):
            raise ValueError("inlier_count is inconsistent with inlier_mask")
        if self.inlier_count > self.available_count:
            raise ValueError("inlier_count exceeds available_count")
        if not 0.0 <= self.inlier_fraction <= 1.0:
            raise ValueError("inlier_fraction must lie in [0, 1]")
        if self.pair_count < 0:
            raise ValueError("pair_count must be nonnegative")
        if not 0.0 <= self.compatible_pair_fraction <= 1.0:
            raise ValueError("compatible_pair_fraction must lie in [0, 1]")
        if not self.decision:
            raise ValueError("decision must be nonempty")
        mask.setflags(write=False)
        object.__setattr__(self, "inlier_mask", mask)


def pairwise_distance_strain_m(
    source_positions_m: np.ndarray,
    observed_positions_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return absolute pair-distance changes and source distances."""

    source = np.asarray(source_positions_m, dtype=float)
    observed = np.asarray(observed_positions_m, dtype=float)
    if source.ndim != 2 or source.shape[1] != 3 or observed.shape != source.shape:
        raise ValueError("source and observed positions must share shape (K, 3)")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(observed)):
        raise ValueError("source and observed positions must be finite")
    source_distance = np.linalg.norm(source[:, None] - source[None, :], axis=2)
    observed_distance = np.linalg.norm(observed[:, None] - observed[None, :], axis=2)
    return np.abs(observed_distance - source_distance), source_distance


def _maximum_compatible_subset(
    compatibility: np.ndarray,
    normalized_strain: np.ndarray,
    material_ids: np.ndarray,
) -> np.ndarray:
    """Find an exact maximum clique with deterministic physical tie breaks."""

    graph = np.asarray(compatibility, dtype=bool)
    score = np.asarray(normalized_strain, dtype=float)
    ids = np.asarray(material_ids, dtype=np.int64)
    count = len(graph)
    if graph.shape != (count, count) or score.shape != graph.shape:
        raise ValueError("compatibility and strain must be square matrices")
    if ids.shape != (count,) or len(np.unique(ids)) != count:
        raise ValueError("material_ids must contain one unique ID per centre")
    if not np.array_equal(graph, graph.T) or not np.all(np.diag(graph)):
        raise ValueError("compatibility must be symmetric with a true diagonal")
    if count == 0:
        return np.empty(0, dtype=np.int64)

    order = np.argsort(ids, kind="mergesort").tolist()
    best: tuple[int, ...] = ()
    best_key: tuple[float, tuple[int, ...]] | None = None

    def consider(chosen: list[int]) -> None:
        nonlocal best, best_key
        if len(chosen) < len(best):
            return
        subset = np.asarray(chosen, dtype=np.int64)
        if len(chosen) < 2:
            mean_strain = 0.0
        else:
            upper = np.triu_indices(len(chosen), 1)
            mean_strain = float(np.mean(score[np.ix_(subset, subset)][upper]))
        key = (mean_strain, tuple(int(ids[index]) for index in chosen))
        if len(chosen) > len(best) or best_key is None or key < best_key:
            best = tuple(chosen)
            best_key = key

    def search(chosen: list[int], candidates: list[int]) -> None:
        if len(chosen) + len(candidates) < len(best):
            return
        if not candidates:
            consider(chosen)
            return
        remaining = candidates
        while remaining:
            if len(chosen) + len(remaining) < len(best):
                return
            vertex = remaining[0]
            rest = remaining[1:]
            search(
                chosen + [vertex],
                [candidate for candidate in rest if graph[vertex, candidate]],
            )
            remaining = rest

    search([], order)
    if not best:
        raise AssertionError("a nonempty compatibility graph must contain a clique")
    result = np.asarray(best, dtype=np.int64)
    result.setflags(write=False)
    return result


def detect_pairwise_consensus_correspondences(
    source_positions_m: np.ndarray,
    observed_positions_m: np.ndarray,
    available: np.ndarray,
    *,
    material_ids: np.ndarray | None = None,
    config: PairwiseCorrespondenceGateConfig | None = None,
) -> PairwiseCorrespondenceGateResult:
    """Select a rigid-invariant maximum-consensus correspondence subset.

    Nonfinite rows are treated as unavailable even if the caller marks them
    available.  The returned mask always uses the original full centre axis.
    """

    cfg = config or PairwiseCorrespondenceGateConfig()
    source = np.asarray(source_positions_m, dtype=float)
    observed = np.asarray(observed_positions_m, dtype=float)
    mask = np.asarray(available, dtype=bool)
    if source.ndim != 2 or source.shape[1] != 3 or observed.shape != source.shape:
        raise ValueError("source and observed positions must share shape (K, 3)")
    if mask.shape != (len(source),):
        raise ValueError("available must have shape (K,)")
    if material_ids is None:
        ids = np.arange(len(source), dtype=np.int64)
    else:
        ids = np.asarray(material_ids, dtype=np.int64)
        if ids.shape != (len(source),) or len(np.unique(ids)) != len(ids):
            raise ValueError("material_ids must contain one unique ID per centre")

    finite = np.all(np.isfinite(source), axis=1) & np.all(np.isfinite(observed), axis=1)
    effective = mask & finite
    available_indices = np.flatnonzero(effective)
    available_count = len(available_indices)
    output_mask = np.zeros(len(source), dtype=bool)
    if available_count < cfg.minimum_inlier_count:
        return PairwiseCorrespondenceGateResult(
            inlier_mask=output_mask,
            accepted=False,
            decision="insufficient_available_support",
            available_count=available_count,
            inlier_count=0,
            inlier_fraction=0.0,
            pair_count=available_count * (available_count - 1) // 2,
            compatible_pair_fraction=0.0,
            median_inlier_normalized_strain=None,
            maximum_inlier_normalized_strain=None,
        )
    if available_count > cfg.maximum_exact_center_count:
        raise ValueError(
            "available centre count exceeds the frozen exact-consensus limit"
        )

    local_source = source[available_indices]
    local_observed = observed[available_indices]
    strain, source_distance = pairwise_distance_strain_m(local_source, local_observed)
    tolerance = np.maximum(
        cfg.absolute_pair_strain_m,
        cfg.relative_pair_strain * source_distance,
    )
    normalized = strain / tolerance
    compatibility = normalized <= 1.0
    np.fill_diagonal(compatibility, True)
    local_inliers = _maximum_compatible_subset(
        compatibility,
        normalized,
        ids[available_indices],
    )
    output_mask[available_indices[local_inliers]] = True
    inlier_count = len(local_inliers)
    inlier_fraction = inlier_count / available_count
    accepted = (
        inlier_count >= cfg.minimum_inlier_count
        and inlier_fraction >= cfg.minimum_inlier_fraction
    )
    if inlier_count < cfg.minimum_inlier_count:
        decision = "insufficient_consensus_count"
    elif inlier_fraction < cfg.minimum_inlier_fraction:
        decision = "insufficient_consensus_fraction"
    else:
        decision = "accepted"
    upper = np.triu_indices(available_count, 1)
    compatible_pair_fraction = float(np.mean(compatibility[upper]))
    if inlier_count < 2:
        median_inlier_strain = None
        maximum_inlier_strain = None
    else:
        inlier_upper = np.triu_indices(inlier_count, 1)
        inlier_values = normalized[np.ix_(local_inliers, local_inliers)][inlier_upper]
        median_inlier_strain = float(np.median(inlier_values))
        maximum_inlier_strain = float(np.max(inlier_values))
    return PairwiseCorrespondenceGateResult(
        inlier_mask=output_mask,
        accepted=accepted,
        decision=decision,
        available_count=available_count,
        inlier_count=inlier_count,
        inlier_fraction=inlier_fraction,
        pair_count=len(upper[0]),
        compatible_pair_fraction=compatible_pair_fraction,
        median_inlier_normalized_strain=median_inlier_strain,
        maximum_inlier_normalized_strain=maximum_inlier_strain,
    )


__all__ = [
    "PairwiseCorrespondenceGateConfig",
    "PairwiseCorrespondenceGateResult",
    "detect_pairwise_consensus_correspondences",
    "pairwise_distance_strain_m",
]
