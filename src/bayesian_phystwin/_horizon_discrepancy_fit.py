"""Equal-group fitting for horizon discrepancy dynamics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ._canonical_contracts import canonical_string_tuple, plain_json
from ._horizon_discrepancy_common import (
    finite_real,
    horizon_vector,
    probability,
    source_summary_id,
)
from ._horizon_discrepancy_contract import HorizonDiscrepancyCalibrationV1


def _retention(
    half_life: float | None, floor: float, horizons: np.ndarray
) -> np.ndarray:
    if half_life is None:
        return np.ones(len(horizons), dtype=np.float64)
    return floor + (1.0 - floor) * np.power(
        2.0, -horizons.astype(np.float64) / half_life
    )


def _fit_nonnegative_two_term(
    design: np.ndarray, target: np.ndarray, *, process_floor: float
) -> np.ndarray:
    candidates: list[np.ndarray] = []
    unconstrained = np.linalg.lstsq(design, target, rcond=None)[0]
    if unconstrained[0] >= 0 and unconstrained[1] >= process_floor:
        candidates.append(unconstrained)
    first, second = design[:, 0], design[:, 1]
    stationary = max(
        0.0,
        float(first @ (target - process_floor * second)) / float(first @ first)
        if float(first @ first) > 0
        else 0.0,
    )
    candidates.append(np.asarray([stationary, process_floor]))
    process = max(
        process_floor,
        float(second @ target) / float(second @ second)
        if float(second @ second) > 0
        else process_floor,
    )
    candidates.append(np.asarray([0.0, process]))
    return min(
        candidates,
        key=lambda value: float(np.sum(np.square(design @ value - target))),
    )


def fit_horizon_discrepancy_calibration(
    source_group_ids: Sequence[str],
    endpoint_mean_m: np.ndarray,
    future_mean_m: np.ndarray,
    horizon_steps: Sequence[int],
    *,
    half_life_candidates: Sequence[float | None] = (
        None,
        4.0,
        8.0,
        16.0,
        32.0,
        64.0,
        128.0,
    ),
    minimum_retention_candidates: Sequence[float] = (0.0, 0.25, 0.5, 0.75),
    component_process_variance_scale: float = 1.0,
    minimum_process_std_m_per_sqrt_step: float = 1e-6,
    metadata: Mapping[str, Any] | None = None,
) -> HorizonDiscrepancyCalibrationV1:
    """Fit compact dynamics with equal weight per independent source group."""

    groups = canonical_string_tuple(
        source_group_ids, name="source_group_ids", allow_empty=False
    )
    if len(groups) < 2 or len(set(groups)) != len(groups):
        raise ValueError("source_group_ids must contain unique independent groups")
    endpoint = np.asarray(endpoint_mean_m, dtype=np.float64)
    future = np.asarray(future_mean_m, dtype=np.float64)
    horizons = horizon_vector(horizon_steps, allow_zero=False)
    if endpoint.shape != (len(groups), 3):
        raise ValueError("endpoint_mean_m must have shape (group, 3)")
    if future.shape != (len(groups), len(horizons), 3):
        raise ValueError("future_mean_m must have shape (group, horizon, 3)")
    if not np.all(np.isfinite(endpoint)) or not np.all(np.isfinite(future)):
        raise ValueError("source discrepancy summaries must be finite")

    order = np.argsort(np.asarray(groups), kind="stable")
    groups = tuple(groups[int(index)] for index in order)
    endpoint = np.ascontiguousarray(endpoint[order])
    future = np.ascontiguousarray(future[order])
    component_scale = finite_real(
        component_process_variance_scale,
        name="component_process_variance_scale",
    )
    process_std_floor = finite_real(
        minimum_process_std_m_per_sqrt_step,
        name="minimum_process_std_m_per_sqrt_step",
        minimum=np.finfo(np.float64).tiny,
    )

    half_lives = []
    for value in half_life_candidates:
        half_lives.append(
            None
            if value is None
            else finite_real(
                value,
                name="half_life_candidates entry",
                minimum=np.finfo(np.float64).tiny,
            )
        )
    if not half_lives:
        raise ValueError("at least one half-life candidate is required")
    half_lives = sorted(
        set(half_lives),
        key=lambda value: (value is None, np.inf if value is None else value),
    )
    floors = sorted(
        {
            probability(value, name="minimum_retention_candidates entry")
            for value in minimum_retention_candidates
        }
    )
    if not floors:
        raise ValueError("at least one minimum-retention candidate is required")

    candidates: list[tuple[float | None, float, np.ndarray, float]] = []
    for half_life in half_lives:
        for floor in (1.0,) if half_life is None else floors:
            if half_life is not None and floor >= 1.0:
                continue
            retention = _retention(half_life, floor, horizons)
            error = future - retention[None, :, None] * endpoint[:, None, :]
            score = float(np.mean(np.mean(np.linalg.norm(error, axis=2), axis=1)))
            candidates.append((half_life, floor, retention, score))
    if not candidates:
        raise ValueError("horizon candidate grid is empty")
    selected_index = min(range(len(candidates)), key=lambda index: candidates[index][3])
    half_life, floor, retention, _ = candidates[selected_index]

    residual = future - retention[None, :, None] * endpoint[:, None, :]
    second_moment = np.mean(np.square(residual), axis=0)
    design = np.column_stack((1.0 - retention**2, horizons.astype(np.float64)))
    stationary_variance = np.empty(3)
    process_variance = np.empty(3)
    process_floor = process_std_floor**2
    for axis in range(3):
        coefficients = _fit_nonnegative_two_term(
            design, second_moment[:, axis], process_floor=process_floor
        )
        stationary_variance[axis], process_variance[axis] = coefficients

    fit_metadata: dict[str, Any] = {
        "selection_unit": "independent source group",
        "selection_loss": "equal-group horizon-mean Euclidean error",
        "candidate_scores": [
            {
                "half_life_steps": candidate[0],
                "minimum_retention": candidate[1],
                "mean_group_loss_m": candidate[3],
                "selected": index == selected_index,
            }
            for index, candidate in enumerate(candidates)
        ],
        "minimum_process_std_m_per_sqrt_step": process_std_floor,
    }
    if metadata is not None:
        fit_metadata["user"] = plain_json(metadata)

    return HorizonDiscrepancyCalibrationV1(
        source_group_ids=groups,
        source_summary_sha256=source_summary_id(groups, horizons, endpoint, future),
        horizon_steps=horizons,
        mean_reversion_half_life_steps=half_life,
        minimum_mean_retention=floor,
        stationary_std_m=np.sqrt(stationary_variance),
        additional_process_std_m_per_sqrt_step=np.sqrt(process_variance),
        component_process_variance_scale=component_scale,
        metadata=fit_metadata,
    )
