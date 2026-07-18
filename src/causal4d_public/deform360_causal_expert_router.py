"""Competence-gated routing for Deform360 causal transport experts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROUTER_SCHEMA_VERSION = 1
PERSISTENCE_LABEL = "persistence"
PARAMETER_FEATURE_NAMES = (
    "candidate_log_base_support_scale_m",
    "candidate_support_growth_per_travel",
    "candidate_initial_contact_gain",
    "candidate_acquired_contact_gain",
    "candidate_is_se3",
    "candidate_maximum_transport_weight",
    "candidate_mean_contact_fraction",
    "candidate_minimum_controller_distance_m",
    "candidate_initial_contact_fraction",
    "candidate_missing_contact_fraction",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _conformal_quantile(values: np.ndarray, level: float) -> float:
    scores = np.sort(np.asarray(values, dtype=np.float64))
    _require(scores.ndim == 1 and len(scores) >= 1, "calibration scores are empty")
    _require(np.all(np.isfinite(scores)), "calibration scores are not finite")
    _require(0.0 < level <= 1.0, "calibration level must lie in (0, 1]")
    rank = min(int(np.ceil((len(scores) + 1) * level)), len(scores))
    return float(scores[rank - 1])


def build_causal_expert_features(record: Mapping[str, Any]) -> dict[str, float]:
    """Build outcome-independent features for one transport candidate."""

    trust = record.get("trust_features")
    diagnostics = record.get("diagnostics")
    _require(isinstance(trust, Mapping), "candidate lacks trust features")
    _require(isinstance(diagnostics, Mapping), "candidate lacks diagnostics")
    values = {str(name): float(value) for name, value in trust.items()}
    scales = float(record["base_support_scale_m"])
    contact_fraction = np.asarray(
        diagnostics.get("contact_fraction_by_group", ()), dtype=np.float64
    )
    distance = np.asarray(
        diagnostics.get("minimum_controller_to_initial_object_distance_m_by_group", ()),
        dtype=np.float64,
    )
    onsets = diagnostics.get("onset_frames", ())
    _require(
        scales > 0.0
        and contact_fraction.ndim == distance.ndim == 1
        and len(contact_fraction) == len(distance) == len(onsets)
        and len(onsets) >= 1,
        "candidate diagnostics are incompatible",
    )
    values.update(
        {
            "candidate_log_base_support_scale_m": float(np.log(scales)),
            "candidate_support_growth_per_travel": float(
                record["support_growth_per_travel"]
            ),
            "candidate_initial_contact_gain": float(record["initial_contact_gain"]),
            "candidate_acquired_contact_gain": float(record["acquired_contact_gain"]),
            "candidate_is_se3": float(record["transform_mode"] == "se3"),
            "candidate_maximum_transport_weight": float(
                diagnostics["maximum_transport_weight"]
            ),
            "candidate_mean_contact_fraction": float(np.mean(contact_fraction)),
            "candidate_minimum_controller_distance_m": float(np.min(distance)),
            "candidate_initial_contact_fraction": float(
                np.mean([value == 0 for value in onsets])
            ),
            "candidate_missing_contact_fraction": float(
                np.mean([value is None for value in onsets])
            ),
        }
    )
    _require(
        all(np.isfinite(value) for value in values.values()),
        "candidate features are not finite",
    )
    return values


def normalized_candidate_score(
    record: Mapping[str, Any], persistence_metrics: Mapping[str, Any]
) -> float:
    full = record["metrics"]["ranges"]["full"]
    baseline = persistence_metrics["ranges"]["full"]
    track_ratio = float(full["track_error_m"]) / float(baseline["track_error_m"])
    chamfer_ratio = float(full["chamfer_m"]) / float(baseline["chamfer_m"])
    score = 0.5 * (track_ratio + chamfer_ratio)
    _require(np.isfinite(score) and score > 0.0, "candidate score is invalid")
    return float(score)


@dataclass(frozen=True)
class CausalExpertEpisode:
    object_id: str
    episode_id: int
    labels: tuple[str, ...]
    features: np.ndarray
    normalized_scores: np.ndarray

    def __post_init__(self) -> None:
        feature = np.asarray(self.features, dtype=np.float64)
        score = np.asarray(self.normalized_scores, dtype=np.float64)
        _require(
            len(self.labels) >= 2
            and self.labels[0] == PERSISTENCE_LABEL
            and len(set(self.labels)) == len(self.labels),
            "episode candidate labels are invalid",
        )
        _require(
            feature.ndim == 2
            and feature.shape[0] == len(self.labels) - 1
            and score.shape == (len(self.labels),)
            and np.all(np.isfinite(feature))
            and np.all(np.isfinite(score))
            and np.all(score > 0.0)
            and score[0] == 1.0,
            "episode features or scores are invalid",
        )
        feature = feature.copy()
        score = score.copy()
        feature.setflags(write=False)
        score.setflags(write=False)
        object.__setattr__(self, "features", feature)
        object.__setattr__(self, "normalized_scores", score)


@dataclass(frozen=True)
class CausalExpertRouterDecision:
    selected_label: str
    selected_index: int
    predicted_log_score: float
    upper_log_score: float
    accepted: bool


@dataclass(frozen=True)
class CausalExpertRouterModel:
    feature_names: tuple[str, ...]
    candidate_labels: tuple[str, ...]
    coefficients: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    ridge: float
    selected_residual_quantile: float
    minimum_improvement_fraction: float
    calibration_level: float
    result_sha256: str = ""

    def __post_init__(self) -> None:
        count = len(self.feature_names)
        _require(
            self.candidate_labels[0] == PERSISTENCE_LABEL
            and len(self.candidate_labels) >= 2,
            "router candidate labels are invalid",
        )
        _require(
            self.coefficients.shape == (count + 1,)
            and self.feature_mean.shape == self.feature_scale.shape == (count,)
            and np.all(np.isfinite(self.coefficients))
            and np.all(np.isfinite(self.feature_mean))
            and np.all(np.isfinite(self.feature_scale))
            and np.all(self.feature_scale > 0.0),
            "router parameter shapes are invalid",
        )
        _require(
            self.ridge > 0.0
            and np.isfinite(self.selected_residual_quantile)
            and 0.0 <= self.minimum_improvement_fraction < 1.0
            and 0.0 < self.calibration_level <= 1.0,
            "router policy parameters are invalid",
        )

    def predict_log_scores(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        _require(
            values.ndim == 2
            and values.shape[1] == len(self.feature_names)
            and np.all(np.isfinite(values)),
            "router features are incompatible",
        )
        normalized = (values - self.feature_mean) / self.feature_scale
        return self.coefficients[0] + normalized @ self.coefficients[1:]

    def decide(self, episode: CausalExpertEpisode) -> CausalExpertRouterDecision:
        _require(
            episode.labels == self.candidate_labels,
            "router candidate order changed",
        )
        predicted = self.predict_log_scores(episode.features)
        chosen_offset = int(np.argmin(predicted))
        predicted_log_score = float(predicted[chosen_offset])
        upper_log_score = predicted_log_score + self.selected_residual_quantile
        acceptance_limit = float(np.log1p(-self.minimum_improvement_fraction))
        accepted = upper_log_score < acceptance_limit
        if not accepted:
            return CausalExpertRouterDecision(
                selected_label=PERSISTENCE_LABEL,
                selected_index=0,
                predicted_log_score=predicted_log_score,
                upper_log_score=upper_log_score,
                accepted=False,
            )
        selected_index = chosen_offset + 1
        return CausalExpertRouterDecision(
            selected_label=episode.labels[selected_index],
            selected_index=selected_index,
            predicted_log_score=predicted_log_score,
            upper_log_score=upper_log_score,
            accepted=True,
        )

    def to_payload(self, *, source: Mapping[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": ROUTER_SCHEMA_VERSION,
            "artifact_kind": "Deform360CausalExpertRouter",
            "feature_names": list(self.feature_names),
            "candidate_labels": list(self.candidate_labels),
            "coefficients": self.coefficients.tolist(),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "ridge": self.ridge,
            "selected_residual_quantile": self.selected_residual_quantile,
            "minimum_improvement_fraction": self.minimum_improvement_fraction,
            "calibration_level": self.calibration_level,
            "source": dict(source),
            "information_boundary": {
                "decision_uses_outcome": False,
                "training_uses_exhausted_source_outcomes": True,
                "exact_persistence_fallback": True,
            },
        }
        payload["result_sha256"] = _canonical_sha256(payload)
        return payload


def _fit_ridge(
    episodes: Sequence[CausalExpertEpisode], ridge: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.concatenate([episode.features for episode in episodes], axis=0)
    targets = np.concatenate(
        [np.log(episode.normalized_scores[1:]) for episode in episodes]
    )
    mean = np.mean(features, axis=0)
    scale = np.maximum(np.std(features, axis=0), 1e-8)
    design = np.column_stack((np.ones(len(features)), (features - mean) / scale))
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + ridge * penalty,
        design.T @ targets,
    )
    return coefficients, mean, scale


def _make_model(
    episodes: Sequence[CausalExpertEpisode],
    *,
    feature_names: tuple[str, ...],
    ridge: float,
    residual_quantile: float,
    minimum_improvement_fraction: float,
    calibration_level: float,
) -> CausalExpertRouterModel:
    coefficients, mean, scale = _fit_ridge(episodes, ridge)
    return CausalExpertRouterModel(
        feature_names=feature_names,
        candidate_labels=episodes[0].labels,
        coefficients=coefficients,
        feature_mean=mean,
        feature_scale=scale,
        ridge=ridge,
        selected_residual_quantile=residual_quantile,
        minimum_improvement_fraction=minimum_improvement_fraction,
        calibration_level=calibration_level,
    )


def _inner_predictions(
    episodes: Sequence[CausalExpertEpisode],
    *,
    feature_names: tuple[str, ...],
    ridge: float,
) -> list[dict[str, Any]]:
    object_ids = sorted({episode.object_id for episode in episodes})
    _require(len(object_ids) >= 2, "router cross-fit needs at least two objects")
    rows: list[dict[str, Any]] = []
    for held_object in object_ids:
        train = [episode for episode in episodes if episode.object_id != held_object]
        held = [episode for episode in episodes if episode.object_id == held_object]
        model = _make_model(
            train,
            feature_names=feature_names,
            ridge=ridge,
            residual_quantile=0.0,
            minimum_improvement_fraction=0.0,
            calibration_level=1.0,
        )
        for episode in held:
            predicted = model.predict_log_scores(episode.features)
            offset = int(np.argmin(predicted))
            rows.append(
                {
                    "object_id": episode.object_id,
                    "episode_id": episode.episode_id,
                    "candidate_index": offset + 1,
                    "candidate_label": episode.labels[offset + 1],
                    "predicted_log_score": float(predicted[offset]),
                    "actual_log_score": float(
                        np.log(episode.normalized_scores[offset + 1])
                    ),
                }
            )
    return rows


def fit_causal_expert_router(
    episodes: Sequence[CausalExpertEpisode],
    *,
    feature_names: Sequence[str],
    ridge_grid: Sequence[float] = (0.1, 1.0, 10.0, 100.0),
    calibration_level: float = 0.9,
    minimum_improvement_fraction: float = 0.0,
    maximum_cross_fitted_degradation_fraction: float = 0.1,
) -> tuple[CausalExpertRouterModel, dict[str, Any]]:
    """Fit a router with object-held-out tuning and selected-action calibration."""

    values = list(episodes)
    names = tuple(str(name) for name in feature_names)
    _require(len(values) >= 2 and len(names) >= 1, "router training data are empty")
    labels = values[0].labels
    width = values[0].features.shape[1]
    _require(
        all(
            episode.labels == labels and episode.features.shape[1] == width
            for episode in values
        )
        and width == len(names),
        "router episodes use different candidate or feature contracts",
    )
    candidates = []
    for ridge in ridge_grid:
        _require(np.isfinite(ridge) and ridge > 0.0, "ridge grid is invalid")
        rows = _inner_predictions(values, feature_names=names, ridge=float(ridge))
        residual = np.asarray(
            [row["actual_log_score"] - row["predicted_log_score"] for row in rows]
        )
        quantile = _conformal_quantile(residual, calibration_level)
        limit = float(np.log1p(-minimum_improvement_fraction))
        selected_scores = []
        for row in rows:
            accepted = row["predicted_log_score"] + quantile < limit
            selected_scores.append(
                float(np.exp(row["actual_log_score"])) if accepted else 1.0
            )
            row["accepted"] = bool(accepted)
            row["selected_normalized_score"] = selected_scores[-1]
        mean_score = float(np.mean(selected_scores))
        maximum_score = float(np.max(selected_scores))
        candidates.append(
            {
                "ridge": float(ridge),
                "selected_residual_quantile": quantile,
                "mean_normalized_score": mean_score,
                "maximum_normalized_score": maximum_score,
                "win_fraction": float(np.mean(np.asarray(selected_scores) < 1.0)),
                "accepted_fraction": float(
                    np.mean([bool(row["accepted"]) for row in rows])
                ),
                "safety_passed": maximum_score
                <= 1.0 + maximum_cross_fitted_degradation_fraction,
                "rows": rows,
            }
        )
    safe = [candidate for candidate in candidates if candidate["safety_passed"]]
    if not safe:
        selected = min(
            candidates,
            key=lambda item: (
                item["maximum_normalized_score"],
                item["mean_normalized_score"],
                item["ridge"],
            ),
        )
        minimum_improvement_fraction = 1.0 - np.finfo(np.float64).eps
    else:
        selected = min(
            safe,
            key=lambda item: (
                item["mean_normalized_score"],
                item["maximum_normalized_score"],
                -item["win_fraction"],
                item["ridge"],
            ),
        )
    model = _make_model(
        values,
        feature_names=names,
        ridge=float(selected["ridge"]),
        residual_quantile=float(selected["selected_residual_quantile"]),
        minimum_improvement_fraction=float(minimum_improvement_fraction),
        calibration_level=calibration_level,
    )
    report = {
        "ridge_candidates": candidates,
        "selected_ridge": model.ridge,
        "selected_residual_quantile": model.selected_residual_quantile,
        "minimum_improvement_fraction": model.minimum_improvement_fraction,
        "calibration_level": model.calibration_level,
        "maximum_cross_fitted_degradation_fraction": (
            maximum_cross_fitted_degradation_fraction
        ),
        "selected_cross_fitted_mean_normalized_score": selected[
            "mean_normalized_score"
        ],
        "selected_cross_fitted_maximum_normalized_score": selected[
            "maximum_normalized_score"
        ],
        "selected_cross_fitted_win_fraction": selected["win_fraction"],
        "selected_cross_fitted_accepted_fraction": selected["accepted_fraction"],
        "safety_passed": bool(selected["safety_passed"]),
    }
    return model, report


def cross_fit_causal_expert_router(
    episodes: Sequence[CausalExpertEpisode],
    *,
    feature_names: Sequence[str],
    ridge_grid: Sequence[float] = (0.1, 1.0, 10.0, 100.0),
    calibration_level: float = 0.9,
    minimum_improvement_fraction: float = 0.0,
    maximum_cross_fitted_degradation_fraction: float = 0.1,
) -> dict[str, Any]:
    """Evaluate the complete router with a leave-one-object-out outer loop."""

    values = list(episodes)
    object_ids = sorted({episode.object_id for episode in values})
    _require(len(object_ids) >= 3, "outer router audit needs at least three objects")
    rows = []
    folds = []
    for held_object in object_ids:
        train = [episode for episode in values if episode.object_id != held_object]
        held = [episode for episode in values if episode.object_id == held_object]
        model, inner = fit_causal_expert_router(
            train,
            feature_names=feature_names,
            ridge_grid=ridge_grid,
            calibration_level=calibration_level,
            minimum_improvement_fraction=minimum_improvement_fraction,
            maximum_cross_fitted_degradation_fraction=(
                maximum_cross_fitted_degradation_fraction
            ),
        )
        folds.append(
            {
                "held_object_id": held_object,
                "fit_object_ids": sorted({episode.object_id for episode in train}),
                "selected_ridge": model.ridge,
                "selected_residual_quantile": model.selected_residual_quantile,
                "inner_safety_passed": inner["safety_passed"],
            }
        )
        for episode in held:
            decision = model.decide(episode)
            score = float(episode.normalized_scores[decision.selected_index])
            rows.append(
                {
                    "object_id": episode.object_id,
                    "episode_id": episode.episode_id,
                    "selected_label": decision.selected_label,
                    "accepted": decision.accepted,
                    "predicted_log_score": decision.predicted_log_score,
                    "upper_log_score": decision.upper_log_score,
                    "normalized_score": score,
                }
            )
    scores = np.asarray([row["normalized_score"] for row in rows])
    return {
        "outer_unit": "object",
        "folds": folds,
        "rows": rows,
        "mean_normalized_score": float(np.mean(scores)),
        "maximum_normalized_score": float(np.max(scores)),
        "improvement_fraction": float(1.0 - np.mean(scores)),
        "win_fraction": float(np.mean(scores < 1.0)),
        "accepted_fraction": float(np.mean([row["accepted"] for row in rows])),
        "object_mean_normalized_scores": {
            object_id: float(
                np.mean(
                    [
                        row["normalized_score"]
                        for row in rows
                        if row["object_id"] == object_id
                    ]
                )
            )
            for object_id in object_ids
        },
    }


def load_causal_expert_router(path: str | Path) -> CausalExpertRouterModel:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(
        isinstance(payload, dict)
        and payload.get("schema_version") == ROUTER_SCHEMA_VERSION
        and payload.get("artifact_kind") == "Deform360CausalExpertRouter"
        and payload.get("result_sha256") == _canonical_sha256(payload),
        "causal expert router artifact is incompatible",
    )
    return CausalExpertRouterModel(
        feature_names=tuple(str(value) for value in payload["feature_names"]),
        candidate_labels=tuple(str(value) for value in payload["candidate_labels"]),
        coefficients=np.asarray(payload["coefficients"], dtype=np.float64),
        feature_mean=np.asarray(payload["feature_mean"], dtype=np.float64),
        feature_scale=np.asarray(payload["feature_scale"], dtype=np.float64),
        ridge=float(payload["ridge"]),
        selected_residual_quantile=float(payload["selected_residual_quantile"]),
        minimum_improvement_fraction=float(payload["minimum_improvement_fraction"]),
        calibration_level=float(payload["calibration_level"]),
        result_sha256=str(payload["result_sha256"]),
    )


__all__ = [
    "CausalExpertEpisode",
    "CausalExpertRouterDecision",
    "CausalExpertRouterModel",
    "PARAMETER_FEATURE_NAMES",
    "PERSISTENCE_LABEL",
    "build_causal_expert_features",
    "cross_fit_causal_expert_router",
    "fit_causal_expert_router",
    "load_causal_expert_router",
    "normalized_candidate_score",
]
