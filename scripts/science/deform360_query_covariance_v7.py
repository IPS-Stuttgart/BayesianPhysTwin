"""Object-grouped same-mean covariance utilities for Deform360 queries."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from statistics import NormalDist
from typing import Any

import numpy as np


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def nearest_psd(covariance: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    value = np.asarray(covariance, dtype=np.float64)
    if (
        value.ndim != 2
        or value.shape[0] != value.shape[1]
        or not np.all(np.isfinite(value))
    ):
        raise ValueError("covariance must be a finite square matrix")
    symmetric = 0.5 * (value + value.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    eigenvalues = np.maximum(eigenvalues, floor * scale)
    result = (eigenvectors * eigenvalues[None, :]) @ eigenvectors.T
    return 0.5 * (result + result.T)


def correlation_shrinkage(covariance: np.ndarray, weight: float) -> np.ndarray:
    covariance = nearest_psd(covariance)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("correlation weight must lie in [0,1]")
    diagonal = np.diag(covariance).copy()
    inverse = 1.0 / np.sqrt(diagonal)
    correlation = covariance * inverse[:, None] * inverse[None, :]
    correlation = weight * correlation + (1.0 - weight) * np.eye(
        covariance.shape[0]
    )
    standard = np.sqrt(diagonal)
    result = correlation * standard[:, None] * standard[None, :]
    np.fill_diagonal(result, diagonal)
    return nearest_psd(result)


def permuted_correlation(covariance: np.ndarray, order: Sequence[int]) -> np.ndarray:
    covariance = nearest_psd(covariance)
    order_array = np.asarray(order, dtype=np.int64)
    if order_array.shape != (covariance.shape[0],) or set(order_array.tolist()) != set(
        range(covariance.shape[0])
    ):
        raise ValueError("invalid dependence permutation")
    diagonal = np.diag(covariance).copy()
    inverse = 1.0 / np.sqrt(diagonal)
    correlation = covariance * inverse[:, None] * inverse[None, :]
    correlation = correlation[np.ix_(order_array, order_array)]
    standard = np.sqrt(diagonal)
    result = correlation * standard[:, None] * standard[None, :]
    np.fill_diagonal(result, diagonal)
    return nearest_psd(result)


def cosine_query_matrix(dimension: int, maximum: int) -> np.ndarray:
    if dimension < 1 or maximum < 1:
        raise ValueError("dimension and maximum query count must be positive")
    count = min(dimension, maximum)
    coordinate = np.arange(dimension, dtype=np.float64)
    vectors = []
    for frequency in range(count):
        vector = np.cos(
            np.pi * frequency * (coordinate + 0.5) / float(dimension)
        )
        vector /= np.linalg.norm(vector)
        vectors.append(vector)
    return np.stack(vectors)


def project_compact_covariance(
    diagonal: np.ndarray,
    factor: np.ndarray,
    multiplier: float,
    query: np.ndarray,
) -> np.ndarray:
    diagonal = np.asarray(diagonal, dtype=np.float64).reshape(-1)
    factor = np.asarray(factor, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    if factor.ndim != 2 or factor.shape[0] != diagonal.size:
        raise ValueError("compact covariance factor and diagonal do not align")
    if query.ndim != 2 or query.shape[1] != diagonal.size:
        raise ValueError("query matrix and covariance dimension do not align")
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("covariance multiplier must be finite and positive")
    effective_diagonal = multiplier * diagonal
    effective_factor = math.sqrt(multiplier) * factor
    projected = (query * effective_diagonal[None, :]) @ query.T
    if effective_factor.shape[1]:
        query_factor = query @ effective_factor
        projected += query_factor @ query_factor.T
    return nearest_psd(projected)


def gaussian_terms(
    residual: np.ndarray, covariance: np.ndarray
) -> tuple[float, float, float]:
    covariance = nearest_psd(covariance)
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise RuntimeError("nonpositive covariance determinant")
    distance = float(residual @ np.linalg.solve(covariance, residual))
    nll = 0.5 * (residual.size * math.log(2.0 * math.pi) + logdet + distance)
    return nll, distance, float(logdet)


def metrics(
    residuals: np.ndarray,
    covariances: np.ndarray,
    groups: np.ndarray,
    probability: float = 0.9,
) -> dict[str, Any]:
    dimension = residuals.shape[1]
    if groups.shape != (residuals.shape[0],):
        raise ValueError("group identifiers do not align with residuals")
    marginal_z = NormalDist().inv_cdf(0.5 + probability / 2.0)
    chi = (
        dimension
        * (
            1.0
            - 2.0 / (9.0 * dimension)
            + NormalDist().inv_cdf(probability)
            * math.sqrt(2.0 / (9.0 * dimension))
        )
        ** 3
    )
    rows: dict[str, dict[str, list[float] | int]] = {}
    for group in sorted(set(groups.tolist())):
        rows[group] = {
            "nll": [],
            "distance": [],
            "logdet": [],
            "width": [],
            "marginal_hits": 0,
            "marginal_total": 0,
            "ellipsoid_hits": 0,
            "case_count": 0,
        }
    for residual, covariance, group in zip(
        residuals, covariances, groups, strict=True
    ):
        nll, distance, logdet = gaussian_terms(residual, covariance)
        standard = np.sqrt(np.diag(nearest_psd(covariance)))
        record = rows[str(group)]
        record["nll"].append(nll / dimension)
        record["distance"].append(distance / dimension)
        record["logdet"].append(logdet)
        record["width"].extend((2.0 * marginal_z * standard).tolist())
        record["ellipsoid_hits"] += int(distance <= chi)
        record["marginal_hits"] += int(
            np.count_nonzero(np.abs(residual) <= marginal_z * standard)
        )
        record["marginal_total"] += dimension
        record["case_count"] += 1
    group_metrics: list[dict[str, float]] = []
    for record in rows.values():
        case_count = int(record["case_count"])
        marginal_total = int(record["marginal_total"])
        group_metrics.append(
            {
                "nll_per_dimension": float(np.mean(record["nll"])),
                "normalized_anees": float(np.mean(record["distance"])),
                "ellipsoid_coverage": float(
                    int(record["ellipsoid_hits"]) / case_count
                ),
                "marginal_coverage": float(
                    int(record["marginal_hits"]) / marginal_total
                ),
                "mean_marginal_width": float(np.mean(record["width"])),
                "mean_log_determinant": float(np.mean(record["logdet"])),
            }
        )
    names = tuple(group_metrics[0])
    return {
        "n_cases": int(residuals.shape[0]),
        "n_groups": len(group_metrics),
        "query_dimension": int(dimension),
        "weighting": "equal-object-after-within-object-average",
        **{
            name: float(np.mean([record[name] for record in group_metrics]))
            for name in names
        },
    }


def transform(covariances: np.ndarray, weight: float, scale: float) -> np.ndarray:
    return np.stack(
        [
            scale * correlation_shrinkage(covariance, weight)
            for covariance in covariances
        ]
    )


def fit(
    residuals: np.ndarray,
    covariances: np.ndarray,
    groups: np.ndarray,
    weights: Iterable[float],
) -> dict[str, float]:
    best: dict[str, float] | None = None
    dimension = residuals.shape[1]
    for weight in weights:
        base = transform(covariances, float(weight), 1.0)
        distances = [
            gaussian_terms(error, covariance)[1]
            for error, covariance in zip(residuals, base, strict=True)
        ]
        group_distances = [
            float(np.mean(np.asarray(distances)[groups == group] / dimension))
            for group in sorted(set(groups.tolist()))
        ]
        scale = float(np.clip(np.mean(group_distances), 1e-4, 1e4))
        score = metrics(
            residuals,
            transform(covariances, float(weight), scale),
            groups,
        )["nll_per_dimension"]
        candidate = {
            "correlation_weight": float(weight),
            "scale": scale,
            "source_nll_per_dimension": float(score),
        }
        if best is None or (
            candidate["source_nll_per_dimension"],
            -candidate["correlation_weight"],
        ) < (
            best["source_nll_per_dimension"],
            -best["correlation_weight"],
        ):
            best = candidate
    if best is None:
        raise ValueError("empty correlation-weight grid")
    return best


def fold_for(group: str, fold_count: int) -> int:
    digest = hashlib.sha256(
        ("deform360-exact-same-mean-v7\0" + group).encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % fold_count


def study(
    residuals: np.ndarray,
    covariances: np.ndarray,
    groups: np.ndarray,
    fold_count: int,
) -> dict[str, Any]:
    unique = sorted(set(groups.tolist()))
    if len(unique) < fold_count:
        raise ValueError("not enough independent object groups")
    group_fold = {group: fold_for(group, fold_count) for group in unique}
    if set(group_fold.values()) != set(range(fold_count)):
        raise ValueError("deterministic folds are not all populated")
    rng = np.random.default_rng(20260901)
    order = rng.permutation(residuals.shape[1])
    if np.array_equal(order, np.arange(residuals.shape[1])):
        order = np.roll(order, 1)
    names = ("hybrid", "full", "diagonal", "permuted", "uncalibrated")
    arm_residuals: dict[str, list[np.ndarray]] = {name: [] for name in names}
    arm_covariances: dict[str, list[np.ndarray]] = {name: [] for name in names}
    arm_groups: dict[str, list[np.ndarray]] = {name: [] for name in names}
    folds: list[dict[str, Any]] = []
    for index in range(fold_count):
        target_mask = np.asarray([group_fold[group] == index for group in groups])
        source_mask = ~target_mask
        source_residuals = residuals[source_mask]
        target_residuals = residuals[target_mask]
        source_groups = groups[source_mask]
        target_groups = groups[target_mask]
        source_covariance = covariances[source_mask]
        target_covariance = covariances[target_mask]
        source_permuted = np.stack(
            [permuted_correlation(value, order) for value in source_covariance]
        )
        target_permuted = np.stack(
            [permuted_correlation(value, order) for value in target_covariance]
        )
        fits = {
            "hybrid": fit(
                source_residuals,
                source_covariance,
                source_groups,
                np.linspace(0.0, 1.0, 11),
            ),
            "full": fit(
                source_residuals, source_covariance, source_groups, (1.0,)
            ),
            "diagonal": fit(
                source_residuals, source_covariance, source_groups, (0.0,)
            ),
            "permuted": fit(
                source_residuals, source_permuted, source_groups, (1.0,)
            ),
        }
        transformed = {
            "hybrid": transform(
                target_covariance,
                fits["hybrid"]["correlation_weight"],
                fits["hybrid"]["scale"],
            ),
            "full": transform(target_covariance, 1.0, fits["full"]["scale"]),
            "diagonal": transform(
                target_covariance, 0.0, fits["diagonal"]["scale"]
            ),
            "permuted": transform(
                target_permuted, 1.0, fits["permuted"]["scale"]
            ),
            "uncalibrated": target_covariance,
        }
        for name, covariance in transformed.items():
            arm_residuals[name].append(target_residuals)
            arm_covariances[name].append(covariance)
            arm_groups[name].append(target_groups)
        folds.append(
            {
                "fold": index,
                "source_groups": len(set(groups[source_mask].tolist())),
                "target_groups": len(set(groups[target_mask].tolist())),
                "target_cases": int(np.count_nonzero(target_mask)),
                "fits": fits,
            }
        )
    arm_metrics = {
        name: metrics(
            np.concatenate(arm_residuals[name]),
            np.concatenate(arm_covariances[name]),
            np.concatenate(arm_groups[name]),
        )
        for name in names
    }
    primary = arm_metrics["hybrid"]
    diagonal = arm_metrics["diagonal"]
    permuted = arm_metrics["permuted"]
    gates = {
        "nll_better_than_diagonal_by_0p02": primary["nll_per_dimension"]
        <= diagonal["nll_per_dimension"] - 0.02,
        "nll_better_than_permuted_by_0p02": primary["nll_per_dimension"]
        <= permuted["nll_per_dimension"] - 0.02,
        "normalized_anees_between_0p8_and_1p2": 0.8
        <= primary["normalized_anees"]
        <= 1.2,
        "marginal_coverage_between_0p87_and_0p93": 0.87
        <= primary["marginal_coverage"]
        <= 0.93,
    }
    return {
        "independent_group_count": len(unique),
        "fold_count": fold_count,
        "case_count": int(residuals.shape[0]),
        "query_dimension": int(residuals.shape[1]),
        "dependence_permutation": order.astype(int).tolist(),
        "folds": folds,
        "metrics": arm_metrics,
        "contrasts": {
            "hybrid_minus_diagonal_nll_per_dimension": primary[
                "nll_per_dimension"
            ]
            - diagonal["nll_per_dimension"],
            "hybrid_minus_permuted_nll_per_dimension": primary[
                "nll_per_dimension"
            ]
            - permuted["nll_per_dimension"],
        },
        "gates": gates,
        "superior_target_passed": all(gates.values()),
    }


def self_test() -> None:
    rng = np.random.default_rng(76103)
    cases = 600
    dimension = 24
    rank = 3
    query = cosine_query_matrix(dimension, 10)
    factor = rng.normal(size=(dimension, rank)) * 0.18
    diagonal = np.linspace(0.04, 0.12, dimension)
    multiplier = 1.4
    full = multiplier * (np.diag(diagonal) + factor @ factor.T)
    projected = project_compact_covariance(diagonal, factor, multiplier, query)
    assert np.allclose(projected, query @ full @ query.T, atol=1e-10)
    field_residuals = rng.multivariate_normal(np.zeros(dimension), full, size=cases)
    residuals = field_residuals @ query.T
    covariances = np.broadcast_to(
        projected, (cases, projected.shape[0], projected.shape[1])
    ).copy()
    groups = np.asarray([f"object-{index % 60:02d}" for index in range(cases)])
    result = study(residuals, covariances, groups, 5)
    assert 0.8 <= result["metrics"]["hybrid"]["normalized_anees"] <= 1.2
    assert (
        result["metrics"]["hybrid"]["nll_per_dimension"]
        < result["metrics"]["diagonal"]["nll_per_dimension"]
    )
    print("Deform360 query covariance v7 self-test passed")
