"""Ridge regret models and exact-fallback routing for Tracking Cloth v2."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

TARGET_ORDER = ("bayesian_physics", "last_residual")
LOSS_FIELD = {
    "persistence": "fallback_loss_mm",
    "bayesian_physics": "candidate_loss_mm",
    "last_residual": "last_residual_loss_mm",
}
TARGET_FIELD = {
    "bayesian_physics": "candidate_regret_mm",
    "last_residual": "last_residual_regret_mm",
}
FEATURE_FIELDS = ("motion_query_horizon", "speed", "grasp", "size")


@dataclass(frozen=True)
class RidgeState:
    """Dense one-hot ridge state with an unpenalized intercept."""

    categories: dict[str, tuple[str, ...]]
    x_mean: np.ndarray
    y_mean: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    alpha: float


def _context(row: dict[str, Any]) -> str:
    horizon = f"{float(row['horizon_seconds']):g}"
    return f"{row['motion']}|{row['query']}|{horizon}"


def prepare_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add target-blind context fields and explicit arm regrets."""

    prepared: list[dict[str, Any]] = []
    for row_id, source in enumerate(rows):
        row = dict(source)
        row["row_id"] = row_id
        row["motion_query_horizon"] = _context(row)
        row["candidate_regret_mm"] = float(row["candidate_loss_mm"]) - float(
            row["fallback_loss_mm"]
        )
        row["last_residual_regret_mm"] = float(row["last_residual_loss_mm"]) - float(
            row["fallback_loss_mm"]
        )
        for field in (
            "candidate_regret_mm",
            "last_residual_regret_mm",
            "fallback_loss_mm",
            "practical_harm_margin_mm",
        ):
            if not math.isfinite(float(row[field])):
                raise ValueError(f"Non-finite {field} in row {row_id}")
        prepared.append(row)
    if len({int(row["row_id"]) for row in prepared}) != len(prepared):
        raise ValueError("Row identities are not unique")
    return prepared


def _fit_categories(rows: Sequence[dict[str, Any]]) -> dict[str, tuple[str, ...]]:
    if not rows:
        raise ValueError("Cannot fit feature schema on no rows")
    return {
        field: tuple(sorted({str(row[field]) for row in rows}))
        for field in FEATURE_FIELDS
    }


def _feature_matrix(
    rows: Sequence[dict[str, Any]],
    categories: dict[str, tuple[str, ...]],
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for field in FEATURE_FIELDS:
        values = np.asarray([str(row[field]) for row in rows], dtype=object)
        for category in categories[field]:
            columns.append((values == category).astype(float))
    if not columns:
        raise ValueError("Feature matrix has no columns")
    return np.column_stack(columns)


def _target_matrix(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            [
                float(row[TARGET_FIELD["bayesian_physics"]]),
                float(row[TARGET_FIELD["last_residual"]]),
            ]
            for row in rows
        ],
        dtype=float,
    )


def fit_ridge(
    rows: Sequence[dict[str, Any]],
    alpha: float,
) -> RidgeState:
    """Fit the two arm-regret models with deterministic dense linear algebra."""

    if alpha <= 0 or not math.isfinite(alpha):
        raise ValueError("Ridge alpha must be finite and positive")
    categories = _fit_categories(rows)
    x = _feature_matrix(rows, categories)
    y = _target_matrix(rows)
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    x_centered = x - x_mean
    y_centered = y - y_mean
    gram = x_centered.T @ x_centered
    rhs = x_centered.T @ y_centered
    coefficients = np.linalg.solve(
        gram + float(alpha) * np.eye(gram.shape[0]),
        rhs,
    )
    intercept = y_mean - x_mean @ coefficients
    return RidgeState(
        categories=categories,
        x_mean=x_mean,
        y_mean=y_mean,
        coefficients=coefficients,
        intercept=intercept,
        alpha=float(alpha),
    )


def predict_ridge(
    state: RidgeState,
    rows: Sequence[dict[str, Any]],
) -> np.ndarray:
    x = _feature_matrix(rows, state.categories)
    prediction = x @ state.coefficients + state.intercept
    if prediction.shape != (len(rows), len(TARGET_ORDER)):
        raise ValueError("Unexpected ridge prediction shape")
    if not np.isfinite(prediction).all():
        raise ValueError("Ridge prediction contains non-finite values")
    return prediction


def _arm_index(arm: str) -> int:
    try:
        return TARGET_ORDER.index(arm)
    except ValueError as exc:
        raise ValueError(f"Arm has no regret model: {arm}") from exc


def route_rows(
    rows: Sequence[dict[str, Any]],
    predictions: np.ndarray,
    allowed_arms: Sequence[str],
    threshold_mm: float,
    *,
    policy: str,
    heldout_material: str,
    alpha: float,
    inner_feasible: bool,
) -> list[dict[str, Any]]:
    """Apply one fitted expert router with byte-exact persistence fallback."""

    if not allowed_arms:
        raise ValueError("At least one non-fallback arm is required")
    if predictions.shape != (len(rows), len(TARGET_ORDER)):
        raise ValueError("Prediction matrix does not match rows")
    indices = [_arm_index(arm) for arm in allowed_arms]
    candidate_predictions = predictions[:, indices]
    selected_indices = np.argmin(candidate_predictions, axis=1)
    best_prediction = candidate_predictions[np.arange(len(rows)), selected_indices]
    selected_arms = np.asarray(allowed_arms, dtype=object)[selected_indices]
    accepted = best_prediction < float(threshold_mm)

    routed: list[dict[str, Any]] = []
    for position, source in enumerate(rows):
        row = dict(source)
        arm = str(selected_arms[position]) if accepted[position] else "persistence"
        fallback = float(row["fallback_loss_mm"])
        selected_loss = float(row[LOSS_FIELD[arm]])
        regret = selected_loss - fallback
        practical_harm = regret > float(row["practical_harm_margin_mm"])
        row.update(
            {
                "policy": policy,
                "outer_heldout_material": heldout_material,
                "selected_arm": arm,
                "accepted": bool(accepted[position]),
                "selected_loss_mm": selected_loss,
                "selected_minus_fallback_mm": regret,
                "selected_practical_harm": bool(practical_harm),
                "selected_strict_regression": bool(regret > 0.0),
                "predicted_best_regret_mm": float(best_prediction[position]),
                "ridge_alpha": float(alpha),
                "admission_threshold_mm": float(threshold_mm),
                "inner_feasible": bool(inner_feasible),
                "exact_fallback": bool(accepted[position] or selected_loss == fallback),
            }
        )
        routed.append(row)
    return routed
