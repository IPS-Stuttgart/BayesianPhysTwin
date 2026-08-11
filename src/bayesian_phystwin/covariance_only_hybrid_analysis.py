"""Cross-fitted scoring helpers for covariance-only hybrid experiments."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

DONORS: Final = ("independent_endpoint_v1", "dynamic_endpoint_v2")
HORIZONS: Final = ("early", "middle", "late")
LOG_2PI: Final = math.log(2.0 * math.pi)


@dataclass(frozen=True, slots=True)
class FoldSelection:
    """One leave-one-unit-out donor and scale decision."""

    held_case_id: str
    selected_donor: str
    selected_scales: tuple[float, ...]
    donor_scales: Mapping[str, tuple[float, ...]]
    donor_training_scores: Mapping[str, float]


def _eigen_projection(
    error_m: np.ndarray,
    covariance_m2: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    error = np.asarray(error_m, dtype=np.float64)
    covariance = np.asarray(covariance_m2, dtype=np.float64)
    validity = np.asarray(valid)
    if (
        error.ndim != 3
        or error.shape[-1] != 3
        or covariance.shape != error.shape + (3,)
        or validity.dtype.kind != "b"
        or validity.shape != error.shape[:2]
    ):
        raise ValueError("future error, covariance, and validity shapes differ")
    selected_error = error[validity]
    selected_covariance = covariance[validity]
    if len(selected_error) < 1:
        raise ValueError("horizon has no valid scoring events")
    symmetric = 0.5 * (selected_covariance + np.swapaxes(selected_covariance, -1, -2))
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if float(np.min(eigenvalues, initial=0.0)) < -1e-10:
        raise ValueError("donor covariance is not positive semidefinite")
    projected = np.einsum("eji,ej->ei", eigenvectors, selected_error)
    return eigenvalues, np.square(projected)


def score_scale_grid(
    error_m: np.ndarray,
    covariance_m2: np.ndarray,
    valid: np.ndarray,
    *,
    scales: Sequence[float],
    observation_std_m: float,
    eigenvalue_floor_m2: float,
    marginal_coverage_z: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean NLL, marginal coverage, and width for each scale."""

    observation_variance = observation_std_m**2
    eigenvalues, projected_square = _eigen_projection(error_m, covariance_m2, valid)
    output_nll = np.empty(len(scales), dtype=np.float64)
    output_coverage = np.empty(len(scales), dtype=np.float64)
    output_width = np.empty(len(scales), dtype=np.float64)
    validity = np.asarray(valid)
    error = np.asarray(error_m, dtype=np.float64)[validity]
    covariance = np.asarray(covariance_m2, dtype=np.float64)[validity]
    diagonal = np.diagonal(covariance, axis1=-2, axis2=-1)
    for index, scale in enumerate(scales):
        total_eigenvalues = np.maximum(
            float(scale) * eigenvalues + observation_variance,
            eigenvalue_floor_m2,
        )
        values = 0.5 * (
            3.0 * LOG_2PI
            + np.sum(
                np.log(total_eigenvalues) + projected_square / total_eigenvalues,
                axis=1,
            )
        )
        total_diagonal = np.maximum(
            float(scale) * diagonal + observation_variance,
            eigenvalue_floor_m2,
        )
        half_width = marginal_coverage_z * np.sqrt(total_diagonal)
        output_nll[index] = float(np.mean(values))
        output_coverage[index] = float(np.mean(np.abs(error) <= half_width))
        output_width[index] = float(np.mean(2.0 * half_width))
    return output_nll, output_coverage, output_width


def score_zero_covariance(
    error_m: np.ndarray,
    valid: np.ndarray,
    *,
    observation_std_m: float,
    eigenvalue_floor_m2: float,
    marginal_coverage_z: float,
) -> tuple[float, float, float]:
    """Score the unchanged mean with only the common observation variance."""

    error = np.asarray(error_m, dtype=np.float64)[np.asarray(valid)]
    if len(error) < 1:
        raise ValueError("horizon has no valid scoring events")
    variance = max(observation_std_m**2, eigenvalue_floor_m2)
    nll = 0.5 * (
        3.0 * LOG_2PI
        + 3.0 * math.log(variance)
        + np.sum(np.square(error), axis=1) / variance
    )
    half_width = marginal_coverage_z * math.sqrt(variance)
    return (
        float(np.mean(nll)),
        float(np.mean(np.abs(error) <= half_width)),
        float(2.0 * half_width),
    )


def _select_scale(
    training_scores: np.ndarray,
    scales: Sequence[float],
) -> tuple[int, float]:
    means = np.mean(training_scores, axis=0)
    index = min(
        range(len(scales)),
        key=lambda position: (
            float(means[position]),
            abs(math.log(float(scales[position]))),
            float(scales[position]),
        ),
    )
    return index, float(means[index])


def crossfit_select(
    case_ids: Sequence[str],
    nll_grid: np.ndarray,
    scales: Sequence[float],
) -> tuple[tuple[FoldSelection, ...], dict[str, object]]:
    """Select one donor and horizon scales without using each held case."""

    cases = tuple(case_ids)
    expected_shape = (len(cases), len(DONORS), len(HORIZONS), len(scales))
    if nll_grid.shape != expected_shape or not np.all(np.isfinite(nll_grid)):
        raise ValueError(f"nll_grid must be finite with shape {expected_shape}")

    def fit(
        training: np.ndarray,
    ) -> tuple[
        str,
        dict[str, tuple[float, ...]],
        dict[str, float],
    ]:
        donor_scales: dict[str, tuple[float, ...]] = {}
        donor_scores: dict[str, float] = {}
        for donor_index, donor in enumerate(DONORS):
            selected_indices: list[int] = []
            selected_scales: list[float] = []
            for horizon_index in range(len(HORIZONS)):
                index, _ = _select_scale(
                    nll_grid[training, donor_index, horizon_index, :],
                    scales,
                )
                selected_indices.append(index)
                selected_scales.append(float(scales[index]))
            donor_scales[donor] = tuple(selected_scales)
            selected = np.column_stack(
                [
                    nll_grid[
                        training,
                        donor_index,
                        horizon_index,
                        selected_indices[horizon_index],
                    ]
                    for horizon_index in range(len(HORIZONS))
                ]
            )
            donor_scores[donor] = float(np.mean(selected))
        donor = min(
            DONORS,
            key=lambda value: (donor_scores[value], DONORS.index(value)),
        )
        return donor, donor_scales, donor_scores

    folds: list[FoldSelection] = []
    for held_index, held_case_id in enumerate(cases):
        training = np.ones(len(cases), dtype=bool)
        training[held_index] = False
        donor, donor_scales, donor_scores = fit(training)
        folds.append(
            FoldSelection(
                held_case_id=held_case_id,
                selected_donor=donor,
                selected_scales=donor_scales[donor],
                donor_scales=donor_scales,
                donor_training_scores=donor_scores,
            )
        )
    donor, donor_scales, donor_scores = fit(np.ones(len(cases), dtype=bool))
    return tuple(folds), {
        "selected_donor": donor,
        "selected_scales": list(donor_scales[donor]),
        "donor_scales": {name: list(donor_scales[name]) for name in DONORS},
        "donor_training_scores": donor_scores,
    }


def effect_matrices(
    reference_nll: np.ndarray,
    nll_grid: np.ndarray,
    scales: Sequence[float],
    folds: Sequence[FoldSelection],
) -> dict[str, np.ndarray]:
    """Build raw, donor-specific, and selected cross-fitted effects."""

    scale_index = {float(scale): index for index, scale in enumerate(scales)}
    raw_index = scale_index[1.0]
    arms: dict[str, np.ndarray] = {
        "independent_raw_covariance": nll_grid[:, 0, :, raw_index] - reference_nll,
        "dynamic_raw_covariance": nll_grid[:, 1, :, raw_index] - reference_nll,
    }
    for donor_index, donor in enumerate(DONORS):
        matrix = np.empty_like(reference_nll)
        for case_index, fold in enumerate(folds):
            for horizon_index, scale in enumerate(fold.donor_scales[donor]):
                matrix[case_index, horizon_index] = (
                    nll_grid[
                        case_index,
                        donor_index,
                        horizon_index,
                        scale_index[float(scale)],
                    ]
                    - reference_nll[case_index, horizon_index]
                )
        arms[f"{donor}_crossfit_scaled"] = matrix
    selected = np.empty_like(reference_nll)
    for case_index, fold in enumerate(folds):
        donor_index = DONORS.index(fold.selected_donor)
        for horizon_index, scale in enumerate(fold.selected_scales):
            selected[case_index, horizon_index] = (
                nll_grid[
                    case_index,
                    donor_index,
                    horizon_index,
                    scale_index[float(scale)],
                ]
                - reference_nll[case_index, horizon_index]
            )
    arms["crossfit_selected_scaled_covariance"] = selected
    return arms


def metric_for_fold(
    grid: np.ndarray,
    folds: Sequence[FoldSelection],
    scales: Sequence[float],
) -> np.ndarray:
    """Read one donor/scale metric for every outer held case."""

    scale_index = {float(scale): index for index, scale in enumerate(scales)}
    output = np.empty((len(folds), len(HORIZONS)), dtype=np.float64)
    for case_index, fold in enumerate(folds):
        donor_index = DONORS.index(fold.selected_donor)
        for horizon_index, scale in enumerate(fold.selected_scales):
            output[case_index, horizon_index] = grid[
                case_index,
                donor_index,
                horizon_index,
                scale_index[float(scale)],
            ]
    return output


def _sign_test_pvalue(values: np.ndarray) -> float:
    nonzero = np.asarray(values, dtype=np.float64)
    nonzero = nonzero[nonzero != 0.0]
    count = len(nonzero)
    if count == 0:
        return 1.0
    smaller = min(int(np.sum(nonzero < 0.0)), int(np.sum(nonzero > 0.0)))
    tail = sum(math.comb(count, index) for index in range(smaller + 1))
    return min(1.0, 2.0 * tail / (2**count))


def bootstrap_family(
    matrices: Mapping[str, np.ndarray],
    *,
    arm_order: Sequence[str],
    replicates: int,
    seed: int,
    confidence: float,
) -> list[dict[str, object]]:
    """Return case-clustered max-t intervals across arms and time bins."""

    if replicates < 1000 or not 0.5 < confidence < 1.0:
        raise ValueError("invalid bootstrap configuration")
    columns: list[np.ndarray] = []
    keys: list[tuple[str, str]] = []
    case_count: int | None = None
    for arm in arm_order:
        matrix = np.asarray(matrices[arm], dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(HORIZONS):
            raise ValueError(f"effect matrix changed shape for {arm}")
        if case_count is None:
            case_count = len(matrix)
        elif len(matrix) != case_count:
            raise ValueError("effect matrices use different case counts")
        vectors = np.column_stack((np.mean(matrix, axis=1), matrix))
        for index, aggregation in enumerate(("overall", *HORIZONS)):
            keys.append((arm, aggregation))
            columns.append(vectors[:, index])
    observations = np.column_stack(columns)
    count = len(observations)
    estimates = np.mean(observations, axis=0)
    standard_deviation = np.std(observations, axis=0, ddof=1)
    standard_errors = standard_deviation / math.sqrt(count)
    bootstrap_means = np.empty((replicates, observations.shape[1]), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for start in range(0, replicates, min(5000, replicates)):
        stop = min(start + min(5000, replicates), replicates)
        indices = rng.integers(0, count, size=(stop - start, count))
        bootstrap_means[start:stop] = np.mean(observations[indices], axis=1)
    denominator = np.where(standard_errors > 0.0, standard_errors, np.inf)
    max_t = np.max(np.abs((bootstrap_means - estimates) / denominator), axis=1)
    critical = float(np.quantile(max_t, confidence))
    rows: list[dict[str, object]] = []
    for index, (arm, aggregation) in enumerate(keys):
        values = observations[:, index]
        lower = float(estimates[index] - critical * standard_errors[index])
        upper = float(estimates[index] + critical * standard_errors[index])
        decision = (
            "hybrid_better"
            if upper < 0.0
            else "hybrid_worse"
            if lower > 0.0
            else "inconclusive"
        )
        rows.append(
            {
                "arm": arm,
                "aggregation": aggregation,
                "mean_nll_difference": float(estimates[index]),
                "median_nll_difference": float(np.median(values)),
                "standard_error": float(standard_errors[index]),
                "simultaneous_interval_lower": lower,
                "simultaneous_interval_upper": upper,
                "familywise_decision": decision,
                "hybrid_better_case_count": int(np.sum(values < 0.0)),
                "exact_tie_case_count": int(np.sum(values == 0.0)),
                "hybrid_worse_case_count": int(np.sum(values > 0.0)),
                "sign_test_pvalue": _sign_test_pvalue(values),
                "bootstrap_probability_hybrid_better": float(
                    np.mean(bootstrap_means[:, index] < 0.0)
                ),
                "familywise_critical_value": critical,
            }
        )
    return rows
