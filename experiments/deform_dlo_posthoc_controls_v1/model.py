"""Numerical helpers for the retrospective DEFORM residual controls."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

FEATURE_COUNT: Final = 92
INTERNAL: Final = slice(2, -2)


@dataclass(frozen=True)
class LinearResidualModel:
    locations: FloatArray
    scales: FloatArray
    coefficients: FloatArray
    feature_indices: IntArray


def feature_layout() -> dict[str, slice]:
    """Return the exact group layout emitted by the frozen feature builder."""

    widths = (
        ("time", 4),
        ("arc", 2),
        ("initial_position", 3),
        ("initial_velocity", 3),
        ("baseline_position", 3),
        ("baseline_velocity", 3),
        ("baseline_acceleration", 3),
        ("curvature", 3),
        ("action_position", 12),
        ("action_velocity", 12),
        ("action_acceleration", 12),
        ("relative_left", 3),
        ("relative_right", 3),
        ("scalar_norms", 5),
        ("dynamic_baseline_position", 3),
        ("dynamic_baseline_velocity", 3),
        ("dynamic_curvature", 3),
        ("dynamic_action_velocity", 12),
    )
    result: dict[str, slice] = {}
    start = 0
    for name, width in widths:
        result[name] = slice(start, start + width)
        start += width
    if start != FEATURE_COUNT:
        raise RuntimeError(f"feature layout changed: {start} != {FEATURE_COUNT}")
    return result


def feature_indices(arm: str) -> IntArray:
    """Select a frozen structural ablation without inspecting target outcomes."""

    layout = feature_layout()
    excluded: set[str]
    if arm == "full":
        excluded = set()
    elif arm == "time_only_ridge":
        excluded = set(layout) - {"time"}
    elif arm == "no_explicit_action_features":
        excluded = {
            "action_position",
            "action_velocity",
            "action_acceleration",
            "relative_left",
            "relative_right",
            "scalar_norms",
            "dynamic_action_velocity",
        }
    elif arm == "no_baseline_dynamics_features":
        excluded = {
            "baseline_velocity",
            "baseline_acceleration",
            "curvature",
            "scalar_norms",
            "dynamic_baseline_velocity",
            "dynamic_curvature",
        }
    else:
        raise ValueError(f"unknown feature arm: {arm}")
    indices = [
        index
        for name, span in layout.items()
        if name not in excluded
        for index in range(span.start, span.stop)
    ]
    if not indices:
        raise ValueError(f"feature arm is empty: {arm}")
    return np.asarray(indices, dtype=np.int64)


def deterministic_subset_indices(
    names: Sequence[str],
    *,
    dlo: str,
    repeat: int,
    count: int,
    domain: str,
) -> IntArray:
    """Return a target-independent nested source subset."""

    normalized = tuple(str(name) for name in names)
    if (
        not normalized
        or len(set(normalized)) != len(normalized)
        or repeat < 0
        or not 1 <= count <= len(normalized)
        or not domain
    ):
        raise ValueError("invalid deterministic source-subset request")

    def key(index: int) -> tuple[bytes, str]:
        name = normalized[index]
        payload = f"{domain}\0{dlo}\0{repeat}\0{name}".encode()
        return hashlib.sha256(payload).digest(), name

    order = sorted(range(len(normalized)), key=key)
    return np.asarray(order[:count], dtype=np.int64)


def fit_linear_residual(
    features: FloatArray,
    residual_canonical: FloatArray,
    trajectory_indices: IntArray,
    *,
    selected_features: IntArray,
    ridge: float,
) -> LinearResidualModel:
    """Fit the point-mean part of the frozen per-node Bayesian ridge model."""

    x_all = np.asarray(features, dtype=np.float64)
    y_all = np.asarray(residual_canonical, dtype=np.float64)
    indices = np.asarray(trajectory_indices, dtype=np.int64)
    chosen = np.asarray(selected_features, dtype=np.int64)
    if (
        x_all.ndim != 4
        or y_all.shape != (*x_all.shape[:3], 3)
        or indices.ndim != 1
        or indices.size == 0
        or chosen.ndim != 1
        or chosen.size == 0
        or np.any(indices < 0)
        or np.any(indices >= x_all.shape[0])
        or np.any(chosen < 0)
        or np.any(chosen >= x_all.shape[3])
        or not np.isfinite(x_all).all()
        or not np.isfinite(y_all).all()
        or not np.isfinite(ridge)
        or ridge <= 0.0
    ):
        raise ValueError("invalid linear-residual fit arrays")
    internal_count = x_all.shape[2]
    feature_count = chosen.size
    locations = np.zeros((internal_count, feature_count), dtype=np.float64)
    scales = np.ones_like(locations)
    coefficients = np.zeros((internal_count, feature_count + 1, 3), dtype=np.float64)
    penalty = np.eye(feature_count + 1, dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    for node in range(internal_count):
        raw = x_all[indices, :, node][:, :, chosen]
        location = np.mean(raw, axis=(0, 1))
        scale = np.std(raw, axis=(0, 1))
        scale = np.where(scale > 1e-10, scale, 1.0)
        standardized = (raw - location) / scale
        design = np.concatenate(
            (
                np.ones((*standardized.shape[:2], 1), dtype=np.float64),
                standardized,
            ),
            axis=2,
        ).reshape(-1, feature_count + 1)
        response = y_all[indices, :, node].reshape(-1, 3)
        normal = design.T @ design + penalty
        right = design.T @ response
        coefficients[node] = np.linalg.solve(normal, right)
        locations[node] = location
        scales[node] = scale
    return LinearResidualModel(
        locations=locations,
        scales=scales,
        coefficients=coefficients,
        feature_indices=chosen.copy(),
    )


def predict_linear_residual(
    model: LinearResidualModel,
    features: FloatArray,
) -> FloatArray:
    """Predict canonical internal-node residuals."""

    x_all = np.asarray(features, dtype=np.float64)
    if x_all.ndim != 4 or not np.isfinite(x_all).all():
        raise ValueError("invalid linear-residual query features")
    chosen = model.feature_indices
    internal_count = x_all.shape[2]
    if (
        model.locations.shape != (internal_count, chosen.size)
        or model.scales.shape != model.locations.shape
        or model.coefficients.shape != (internal_count, chosen.size + 1, 3)
    ):
        raise ValueError("linear-residual model shape differs")
    means = []
    for node in range(internal_count):
        raw = x_all[:, :, node][:, :, chosen]
        standardized = (raw - model.locations[node]) / model.scales[node]
        design = np.concatenate(
            (
                np.ones((*standardized.shape[:2], 1), dtype=np.float64),
                standardized,
            ),
            axis=2,
        )
        means.append(np.einsum("ntd,dc->ntc", design, model.coefficients[node]))
    return np.stack(means, axis=2)


def candidate_from_canonical(
    baseline: FloatArray,
    correction_canonical: FloatArray,
    frames: FloatArray,
    *,
    shrinkage: float,
) -> FloatArray:
    """Apply a canonical correction while retaining clamped nodes exactly."""

    base = np.asarray(baseline, dtype=np.float64)
    correction = np.asarray(correction_canonical, dtype=np.float64)
    rotations = np.asarray(frames, dtype=np.float64)
    if (
        base.ndim != 4
        or correction.shape != (base.shape[0], base.shape[1], base.shape[2] - 4, 3)
        or rotations.shape != (base.shape[0], 3, 3)
        or not np.isfinite(base).all()
        or not np.isfinite(correction).all()
        or not np.isfinite(rotations).all()
        or not np.isfinite(shrinkage)
        or not 0.0 <= shrinkage <= 1.0
    ):
        raise ValueError("invalid canonical correction")
    correction_global = np.einsum("ntvj,nij->ntvi", correction, rotations)
    candidate = base.copy()
    candidate[:, :, INTERNAL] += shrinkage * correction_global
    if not np.array_equal(candidate[:, :, :2], base[:, :, :2]) or not np.array_equal(
        candidate[:, :, -2:], base[:, :, -2:]
    ):
        raise RuntimeError("clamped nodes changed")
    return candidate


def node_constant_bias(
    train_residual_canonical: FloatArray,
    target_count: int,
    target_horizon: int,
) -> FloatArray:
    """Use one source mean per internal node and coordinate."""

    residual = np.asarray(train_residual_canonical, dtype=np.float64)
    mean = np.mean(residual, axis=(0, 1), keepdims=True)
    return np.broadcast_to(
        mean,
        (target_count, target_horizon, residual.shape[2], 3),
    ).copy()


def time_node_mean_residual(
    train_residual_canonical: FloatArray,
    target_count: int,
) -> FloatArray:
    """Use the source mean trajectory indexed by time and internal node."""

    residual = np.asarray(train_residual_canonical, dtype=np.float64)
    mean = np.mean(residual, axis=0, keepdims=True)
    return np.broadcast_to(mean, (target_count, *mean.shape[1:])).copy()


def score_arm(
    candidate: FloatArray,
    baseline: FloatArray,
    target: FloatArray,
    names: Sequence[str],
) -> dict[str, Any]:
    """Score complete trajectories without pooling coordinates as units."""

    cand = np.asarray(candidate, dtype=np.float64)
    base = np.asarray(baseline, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    normalized_names = tuple(str(name) for name in names)
    if (
        cand.shape != base.shape
        or cand.shape != truth.shape
        or cand.ndim != 4
        or cand.shape[0] != len(normalized_names)
        or len(set(normalized_names)) != len(normalized_names)
        or not np.isfinite(cand).all()
        or not np.isfinite(base).all()
        or not np.isfinite(truth).all()
    ):
        raise ValueError("score arrays do not align")

    def errors(values: FloatArray, nodes: slice) -> FloatArray:
        return np.mean(np.abs(values[:, :, nodes] - truth[:, :, nodes]), axis=(1, 2, 3))

    candidate_case = errors(cand, slice(None))
    baseline_case = errors(base, slice(None))
    candidate_free = errors(cand, INTERNAL)
    baseline_free = errors(base, INTERNAL)
    if np.any(baseline_case <= 0.0) or np.any(baseline_free <= 0.0):
        raise ValueError("baseline errors must be positive")
    ratios = candidate_case / baseline_case
    free_ratios = candidate_free / baseline_free
    return {
        "metric": "mean-coordinate-l1-all-nodes-m",
        "candidate_mean_l1_m": float(np.mean(candidate_case)),
        "baseline_mean_l1_m": float(np.mean(baseline_case)),
        "relative_improvement": float(1.0 - np.mean(candidate_case) / np.mean(baseline_case)),
        "wins": int(np.count_nonzero(candidate_case < baseline_case)),
        "ties": int(np.count_nonzero(candidate_case == baseline_case)),
        "losses": int(np.count_nonzero(candidate_case > baseline_case)),
        "worst_candidate_to_baseline_ratio": float(np.max(ratios)),
        "case_names": list(normalized_names),
        "candidate_case_l1_m": candidate_case.tolist(),
        "baseline_case_l1_m": baseline_case.tolist(),
        "case_ratios": ratios.tolist(),
        "free_node_diagnostic": {
            "candidate_mean_l1_m": float(np.mean(candidate_free)),
            "baseline_mean_l1_m": float(np.mean(baseline_free)),
            "relative_improvement": float(
                1.0 - np.mean(candidate_free) / np.mean(baseline_free)
            ),
            "wins": int(np.count_nonzero(candidate_free < baseline_free)),
            "ties": int(np.count_nonzero(candidate_free == baseline_free)),
            "losses": int(np.count_nonzero(candidate_free > baseline_free)),
            "worst_candidate_to_baseline_ratio": float(np.max(free_ratios)),
        },
    }


def trajectory_rows(
    summary: Mapping[str, Any],
    *,
    dlo: str,
    arm: str,
    repeat: int | None = None,
    source_count: int | None = None,
) -> list[dict[str, Any]]:
    """Expand a score summary into one row per complete trajectory."""

    names = list(summary["case_names"])
    candidate = list(summary["candidate_case_l1_m"])
    baseline = list(summary["baseline_case_l1_m"])
    ratios = list(summary["case_ratios"])
    rows = []
    for name, candidate_error, baseline_error, ratio in zip(
        names, candidate, baseline, ratios, strict=True
    ):
        rows.append(
            {
                "dlo": dlo,
                "arm": arm,
                "repeat": "" if repeat is None else repeat,
                "source_count": "" if source_count is None else source_count,
                "trajectory": name,
                "candidate_l1_mm": 1000.0 * float(candidate_error),
                "physical_l1_mm": 1000.0 * float(baseline_error),
                "candidate_to_physical_ratio": float(ratio),
            }
        )
    return rows


def summarize_repeated_curve(
    rows: Sequence[Mapping[str, Any]],
    sizes: Sequence[int],
) -> dict[str, Any]:
    """Summarize target-independent repeated source subsets by size."""

    result: dict[str, Any] = {}
    for size in sizes:
        selected = [row for row in rows if int(row["source_count"]) == int(size)]
        if not selected:
            raise ValueError(f"source-data curve omits size {size}")
        errors = np.asarray([row["candidate_mean_l1_m"] for row in selected], dtype=np.float64)
        improvements = np.asarray([row["relative_improvement"] for row in selected], dtype=np.float64)
        result[str(size)] = {
            "repeat_count": len(selected),
            "mean_candidate_l1_m": float(np.mean(errors)),
            "median_candidate_l1_m": float(np.median(errors)),
            "minimum_candidate_l1_m": float(np.min(errors)),
            "maximum_candidate_l1_m": float(np.max(errors)),
            "mean_relative_improvement": float(np.mean(improvements)),
            "median_relative_improvement": float(np.median(improvements)),
            "improving_repeats": int(np.count_nonzero(improvements > 0.0)),
        }
    return result


def equal_dlo_bootstrap(
    candidate_by_dlo: Mapping[str, Sequence[float]],
    baseline_by_dlo: Mapping[str, Sequence[float]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    """Bootstrap complete trajectories within each DLO and weight DLOs equally."""

    if set(candidate_by_dlo) != set(baseline_by_dlo) or not candidate_by_dlo:
        raise ValueError("bootstrap DLO sets differ")
    if replicates < 100 or seed < 0:
        raise ValueError("invalid bootstrap settings")
    rng = np.random.default_rng(seed)
    observed_candidate = []
    observed_baseline = []
    samples = np.empty(replicates, dtype=np.float64)
    arrays: dict[str, tuple[FloatArray, FloatArray]] = {}
    for dlo in sorted(candidate_by_dlo):
        candidate = np.asarray(candidate_by_dlo[dlo], dtype=np.float64)
        baseline = np.asarray(baseline_by_dlo[dlo], dtype=np.float64)
        if candidate.ndim != 1 or candidate.shape != baseline.shape or candidate.size < 2:
            raise ValueError("invalid bootstrap trajectory errors")
        arrays[dlo] = candidate, baseline
        observed_candidate.append(float(np.mean(candidate)))
        observed_baseline.append(float(np.mean(baseline)))
    for index in range(replicates):
        candidate_means = []
        baseline_means = []
        for candidate, baseline in arrays.values():
            draw = rng.integers(0, candidate.size, size=candidate.size)
            candidate_means.append(float(np.mean(candidate[draw])))
            baseline_means.append(float(np.mean(baseline[draw])))
        samples[index] = 1.0 - np.mean(candidate_means) / np.mean(baseline_means)
    observed = 1.0 - np.mean(observed_candidate) / np.mean(observed_baseline)
    return {
        "relative_improvement": float(observed),
        "bootstrap_low": float(np.quantile(samples, 0.025)),
        "bootstrap_high": float(np.quantile(samples, 0.975)),
    }
