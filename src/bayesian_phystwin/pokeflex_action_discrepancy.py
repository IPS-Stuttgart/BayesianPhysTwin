"""Low-capacity action-conditioned discrepancy models for PokeFlex diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def robust_nearest_translation_m(
    prediction_points_m: np.ndarray,
    target_points_m: np.ndarray,
    *,
    retained_fraction: float = 0.9,
) -> np.ndarray:
    """Estimate a global translation from robust prediction-to-target matches."""

    from scipy.spatial import cKDTree

    prediction = np.asarray(prediction_points_m, dtype=np.float64)
    target = np.asarray(target_points_m, dtype=np.float64)
    _require(
        prediction.ndim == target.ndim == 2
        and prediction.shape[1:] == target.shape[1:] == (3,)
        and len(prediction) > 0
        and len(target) > 0,
        "point samples must have shape (N, 3) and be nonempty",
    )
    _require(
        np.all(np.isfinite(prediction)) and np.all(np.isfinite(target)),
        "point samples must be finite",
    )
    _require(
        np.isfinite(retained_fraction) and 0.0 < retained_fraction <= 1.0,
        "retained_fraction must be in (0, 1]",
    )
    distance, index = cKDTree(target).query(prediction, k=1)
    residual = target[np.asarray(index, dtype=np.int64)] - prediction
    if retained_fraction < 1.0:
        cutoff = float(np.quantile(distance, retained_fraction))
        residual = residual[np.asarray(distance) <= cutoff]
    _require(len(residual) > 0, "robust nearest-neighbour trimming removed every row")
    result = np.median(residual, axis=0)
    result.setflags(write=False)
    return result


def causal_action_features(
    history_records: list[dict[str, Any]],
    *,
    template_vertices_m: np.ndarray,
    predicted_vertices_m: np.ndarray,
) -> np.ndarray:
    """Build residual-independent features from the permitted five-frame history."""

    _require(len(history_records) == 5, "exactly five causal records are required")
    template = np.asarray(template_vertices_m, dtype=np.float64)
    predicted = np.asarray(predicted_vertices_m, dtype=np.float64)
    _require(
        template.ndim == predicted.ndim == 2
        and template.shape == predicted.shape
        and template.shape[1] == 3,
        "template and prediction must share shape (N, 3)",
    )
    force = np.asarray(
        [record["forces"][:3] for record in history_records],
        dtype=np.float64,
    )
    tool = np.asarray(
        [record["T_WT"] for record in history_records],
        dtype=np.float64,
    )[:, :3, 3]
    end_effector = np.asarray(
        [record["T_WE"] for record in history_records],
        dtype=np.float64,
    )[:, :3, 3]
    _require(
        force.shape == tool.shape == end_effector.shape == (5, 3),
        "causal robot record shapes changed",
    )
    deformation = predicted - template
    features = np.concatenate(
        (
            force[-1],
            np.mean(force, axis=0),
            force[-1] - force[-2],
            tool[-1] - tool[0],
            tool[-1] - tool[-2],
            end_effector[-1] - end_effector[0],
            np.mean(deformation, axis=0),
            np.asarray(
                [
                    np.sqrt(np.mean(np.sum(np.square(deformation), axis=1))),
                    np.linalg.norm(tool[-1] - tool[0]),
                    np.linalg.norm(end_effector[-1] - end_effector[0]),
                ],
                dtype=np.float64,
            ),
        )
    )
    _require(
        features.shape == (24,) and np.all(np.isfinite(features)),
        "causal action features are invalid",
    )
    features.setflags(write=False)
    return features


def _equal_group_weights(groups: np.ndarray) -> np.ndarray:
    value = np.asarray(groups)
    _require(value.ndim == 1 and len(value) > 0, "groups must be a nonempty vector")
    _, inverse, counts = np.unique(value, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    weights *= len(weights) / np.sum(weights)
    return weights


@dataclass(frozen=True)
class TranslationRidgeModel:
    """Weighted standardized ridge model with a bounded metric output."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    maximum_translation_m: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        value = np.asarray(features, dtype=np.float64)
        _require(
            value.ndim == 2
            and value.shape[1] == len(self.feature_mean)
            and np.all(np.isfinite(value)),
            "prediction features have the wrong shape or contain nonfinite values",
        )
        standardized = (value - self.feature_mean) / self.feature_scale
        design = np.column_stack((np.ones(len(value)), standardized))
        result = design @ self.coefficients
        norm = np.linalg.norm(result, axis=1, keepdims=True)
        result *= np.minimum(
            1.0,
            self.maximum_translation_m / np.maximum(norm, 1e-15),
        )
        result.setflags(write=False)
        return result


def fit_translation_ridge(
    features: np.ndarray,
    target_translation_m: np.ndarray,
    groups: np.ndarray,
    *,
    ridge_penalty: float = 10.0,
    maximum_translation_m: float = 0.01,
) -> TranslationRidgeModel:
    """Fit an object-balanced causal translation model."""

    feature = np.asarray(features, dtype=np.float64)
    target = np.asarray(target_translation_m, dtype=np.float64)
    group = np.asarray(groups)
    _require(
        feature.ndim == 2
        and target.shape == (len(feature), 3)
        and group.shape == (len(feature),)
        and len(feature) > 0,
        "ridge inputs have incompatible shapes",
    )
    _require(
        np.all(np.isfinite(feature)) and np.all(np.isfinite(target)),
        "ridge inputs must be finite",
    )
    _require(
        np.isfinite(ridge_penalty) and ridge_penalty >= 0.0,
        "ridge_penalty must be finite and nonnegative",
    )
    _require(
        np.isfinite(maximum_translation_m) and maximum_translation_m > 0.0,
        "maximum_translation_m must be positive and finite",
    )
    weight = _equal_group_weights(group)
    feature_mean = np.average(feature, axis=0, weights=weight)
    centered = feature - feature_mean
    feature_scale = np.sqrt(np.average(np.square(centered), axis=0, weights=weight))
    feature_scale = np.maximum(feature_scale, 1e-8)
    standardized = centered / feature_scale
    design = np.column_stack((np.ones(len(feature)), standardized))
    weighted_design = design * np.sqrt(weight[:, None])
    weighted_target = target * np.sqrt(weight[:, None])
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge_penalty
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ weighted_target,
    )
    for value in (feature_mean, feature_scale, coefficients):
        value.setflags(write=False)
    return TranslationRidgeModel(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        coefficients=coefficients,
        maximum_translation_m=maximum_translation_m,
    )


def apply_bounded_translation(
    vertices_or_points_m: np.ndarray,
    translation_m: np.ndarray,
    *,
    scale: float,
) -> np.ndarray:
    """Apply one translation per frame while preserving exact scale-zero fallback."""

    value = np.asarray(vertices_or_points_m)
    translation = np.asarray(translation_m, dtype=value.dtype)
    _require(
        value.ndim == 3
        and value.shape[2] == 3
        and translation.shape == (len(value), 3),
        "values must have shape (T, N, 3) and translations shape (T, 3)",
    )
    _require(
        np.all(np.isfinite(value))
        and np.all(np.isfinite(translation))
        and np.isfinite(scale)
        and scale >= 0.0,
        "translation application inputs are invalid",
    )
    if scale == 0.0:
        return value.copy()
    return value + np.asarray(scale, dtype=value.dtype) * translation[:, None, :]
