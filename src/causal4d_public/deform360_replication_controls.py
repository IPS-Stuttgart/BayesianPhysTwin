"""Locked pooling and causal contact-transition controls for Deform360."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize


POOLING_CONTROL_SCHEMA_VERSION = 1


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class PoolingControlSelection:
    pooled_candidate_index: int
    single_source_candidate_indices: tuple[int, ...]
    unique_single_source_candidate_indices: tuple[int, ...]
    source_count: int
    candidate_count: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_pooling_controls(
    source_chamfer_by_candidate: np.ndarray,
) -> PoolingControlSelection:
    """Select pooled and matched single-source candidates without target data."""

    scores = np.asarray(source_chamfer_by_candidate, dtype=np.float64)
    _require(scores.ndim == 2, "source scores must have shape (candidate, source)")
    candidate_count, source_count = scores.shape
    _require(candidate_count >= 1 and source_count >= 2, "too few source scores")
    pooled_valid = np.all(np.isfinite(scores), axis=1)
    _require(np.any(pooled_valid), "no candidate is valid on every source")
    pooled_valid_indices = np.flatnonzero(pooled_valid)
    pooled = int(
        min(
            pooled_valid_indices,
            key=lambda index: (float(np.mean(scores[index])), int(index)),
        )
    )
    single_values = []
    for source in range(source_count):
        valid_indices = np.flatnonzero(np.isfinite(scores[:, source]))
        _require(len(valid_indices) > 0, f"source {source} has no valid candidate")
        single_values.append(
            int(
                min(
                    valid_indices,
                    key=lambda index: (float(scores[index, source]), int(index)),
                )
            )
        )
    single = tuple(single_values)
    return PoolingControlSelection(
        pooled_candidate_index=pooled,
        single_source_candidate_indices=single,
        unique_single_source_candidate_indices=tuple(sorted(set(single))),
        source_count=source_count,
        candidate_count=candidate_count,
    )


def evaluate_pooling_control(
    selection: PoolingControlSelection,
    target_chamfer_by_candidate: Mapping[int, float] | np.ndarray,
) -> dict[str, Any]:
    """Evaluate already-sealed candidates after target outcomes are opened."""

    required = (
        selection.pooled_candidate_index,
        *selection.single_source_candidate_indices,
    )
    if isinstance(target_chamfer_by_candidate, Mapping):
        _require(
            all(index in target_chamfer_by_candidate for index in required),
            "target metric does not cover every sealed candidate",
        )
        target = {
            index: float(target_chamfer_by_candidate[index]) for index in required
        }
        required_values = np.asarray([target[index] for index in required])
        pooled_value = target[selection.pooled_candidate_index]
        single_values = np.asarray(
            [target[index] for index in selection.single_source_candidate_indices]
        )
    else:
        target_array = np.asarray(target_chamfer_by_candidate, dtype=np.float64)
        _require(
            target_array.shape == (selection.candidate_count,),
            "target metric does not cover every candidate",
        )
        required_values = target_array[list(required)]
        pooled_value = float(target_array[selection.pooled_candidate_index])
        single_values = target_array[list(selection.single_source_candidate_indices)]
    _require(
        np.all(np.isfinite(required_values)),
        "a sealed pooling-control target metric is nonfinite",
    )
    single_median = float(np.median(single_values))
    return {
        "pooled_candidate_index": selection.pooled_candidate_index,
        "pooled_target_chamfer_m": pooled_value,
        "single_source_candidate_indices": list(
            selection.single_source_candidate_indices
        ),
        "single_source_target_chamfer_m": single_values.tolist(),
        "single_source_median_target_chamfer_m": single_median,
        "pooled_relative_improvement_over_single_source_median": (
            (single_median - pooled_value) / single_median
        ),
        "pooled_better_than_single_source_median": bool(pooled_value < single_median),
    }


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def pooling_control_artifact_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def build_pooling_control_selection_artifact(
    source_fit: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal pooled and single-source identities from a source fit artifact."""

    rows = source_fit.get("candidate_scores")
    _require(isinstance(rows, list) and rows, "source fit has no candidate table")
    first_per_episode = rows[0].get("per_episode")
    _require(
        isinstance(first_per_episode, list) and len(first_per_episode) >= 2,
        "source fit has too few source episodes",
    )
    episode_ids = [str(record["episode_id"]) for record in first_per_episode]
    scores = np.full((len(rows), len(episode_ids)), np.inf, dtype=np.float64)
    parameters = {}
    for expected_index, row in enumerate(rows):
        candidate_index = int(row["candidate_index"])
        _require(candidate_index == expected_index, "candidate table is not contiguous")
        per_episode = row.get("per_episode")
        _require(
            [str(record["episode_id"]) for record in per_episode] == episode_ids,
            "candidate source episode ordering changed",
        )
        for source_index, record in enumerate(per_episode):
            value = record.get("chamfer_distance_m")
            scores[candidate_index, source_index] = (
                float(value) if value is not None else np.inf
            )
        parameters[candidate_index] = dict(row["parameters"])
    selection = select_pooling_controls(scores)
    required_indices = tuple(
        sorted(
            {
                selection.pooled_candidate_index,
                *selection.single_source_candidate_indices,
            }
        )
    )
    payload: dict[str, Any] = {
        "schema_version": POOLING_CONTROL_SCHEMA_VERSION,
        "artifact_kind": "Deform360PoolingControlSelection",
        "source_fit_artifact_kind": source_fit.get("artifact_kind"),
        "source_fit_result_sha256": source_fit.get("result_sha256"),
        "source_episode_ids": episode_ids,
        "selection": selection.as_dict(),
        "sealed_candidate_indices": list(required_indices),
        "sealed_candidate_parameters": {
            str(index): parameters[index] for index in required_indices
        },
        "information_boundary": {
            "source_candidate_scores_read": True,
            "target_prefix_read": False,
            "target_future_geometry_read": False,
            "target_metrics_computed": False,
        },
        "target_summary_rule": (
            "median target metric over every source-selected candidate, with "
            "source multiplicity preserved"
        ),
    }
    payload["result_sha256"] = pooling_control_artifact_sha256(payload)
    return payload


def validate_pooling_control_selection_artifact(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        payload.get("schema_version") == POOLING_CONTROL_SCHEMA_VERSION,
        "unsupported pooling-control schema",
    )
    _require(
        payload.get("artifact_kind") == "Deform360PoolingControlSelection",
        "unexpected pooling-control artifact kind",
    )
    _require(
        payload.get("result_sha256") == pooling_control_artifact_sha256(payload),
        "pooling-control checksum mismatch",
    )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("target_future_geometry_read") is False
        and boundary.get("target_metrics_computed") is False,
        "pooling control was not selected source-only",
    )
    return {
        "passed": True,
        "result_sha256": payload["result_sha256"],
        "sealed_candidate_count": len(payload["sealed_candidate_indices"]),
    }


def write_pooling_control_selection_artifact(
    path: str | Path, payload: Mapping[str, Any]
) -> Path:
    validate_pooling_control_selection_artifact(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


@dataclass(frozen=True)
class ContactTransitionEpisode:
    episode_id: str
    openings_m: np.ndarray
    controller_positions_m: np.ndarray
    predicted_object_positions_m: np.ndarray
    contact_active: np.ndarray
    dt_seconds: float

    def __post_init__(self) -> None:
        openings = np.asarray(self.openings_m, dtype=np.float64)
        controllers = np.asarray(self.controller_positions_m, dtype=np.float64)
        objects = np.asarray(self.predicted_object_positions_m, dtype=np.float64)
        active = np.asarray(self.contact_active, dtype=bool)
        _require(
            openings.ndim == 2 and len(openings) >= 3,
            "openings must have shape (T,C)",
        )
        _require(
            controllers.shape == (*openings.shape, 3),
            "controllers must have shape (T,C,3)",
        )
        _require(
            objects.ndim == 3
            and objects.shape[0] == len(openings)
            and objects.shape[2] == 3,
            "object predictions must have shape (T,N,3)",
        )
        _require(active.shape == openings.shape, "contact labels must have shape (T,C)")
        _require(
            np.all(np.isfinite(openings))
            and np.all(np.isfinite(controllers))
            and np.all(np.isfinite(objects)),
            "transition episode contains nonfinite values",
        )
        _require(self.dt_seconds > 0.0, "transition episode dt must be positive")
        for name, values in (
            ("openings_m", openings),
            ("controller_positions_m", controllers),
            ("predicted_object_positions_m", objects),
            ("contact_active", active),
        ):
            copied = values.copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)


@dataclass(frozen=True)
class ContactTransitionModel:
    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    onset_coefficients: tuple[float, ...]
    release_coefficients: tuple[float, ...]
    ridge_strength: float
    transition_threshold: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContactTransitionFit:
    model: ContactTransitionModel
    source_metrics: dict[str, float]
    calibration_metrics: dict[str, float]
    candidate_table: tuple[dict[str, Any], ...]


def contact_transition_geometry_features(
    openings_m: np.ndarray,
    controller_positions_m: np.ndarray,
    predicted_object_positions_m: np.ndarray,
    *,
    dt_seconds: float,
) -> np.ndarray:
    """Return opening, proximity, and causal relative-closing-speed features."""

    openings = np.asarray(openings_m, dtype=np.float64)
    controllers = np.asarray(controller_positions_m, dtype=np.float64)
    objects = np.asarray(predicted_object_positions_m, dtype=np.float64)
    _require(openings.ndim == 2 and len(openings) >= 2, "openings must be (T,C)")
    _require(
        controllers.shape == (*openings.shape, 3),
        "controllers must be (T,C,3)",
    )
    _require(
        objects.ndim == 3
        and objects.shape[0] == len(openings)
        and objects.shape[2] == 3,
        "predicted objects must be (T,N,3)",
    )
    _require(dt_seconds > 0.0, "dt must be positive")
    difference = controllers[:, :, None, :] - objects[:, None, :, :]
    proximity = np.min(np.linalg.norm(difference, axis=3), axis=2)
    closing_speed = np.zeros_like(proximity)
    closing_speed[1:] = (proximity[:-1] - proximity[1:]) / dt_seconds
    return np.stack((openings, proximity, closing_speed), axis=2)


def _episode_features(episode: ContactTransitionEpisode) -> np.ndarray:
    return contact_transition_geometry_features(
        episode.openings_m,
        episode.controller_positions_m,
        episode.predicted_object_positions_m,
        dt_seconds=episode.dt_seconds,
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    positive = values >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _fit_binary_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    ridge_strength: float,
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    _require(x.ndim == 2 and len(x) == len(y), "logistic rows disagree")
    _require(len(x) >= 2, "too few transition rows")
    design = np.column_stack((np.ones(len(x)), x))
    penalty = np.ones(design.shape[1], dtype=np.float64)
    penalty[0] = 1e-4

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        logits = design @ coefficients
        value = float(
            np.sum(np.logaddexp(0.0, logits) - y * logits)
            + 0.5 * ridge_strength * np.sum(penalty * coefficients**2)
        )
        gradient = (
            design.T @ (_sigmoid(logits) - y) + ridge_strength * penalty * coefficients
        )
        return value, gradient

    initial_probability = (float(np.sum(y)) + 0.5) / (len(y) + 1.0)
    initial = np.zeros(design.shape[1], dtype=np.float64)
    initial[0] = np.log(initial_probability / (1.0 - initial_probability))
    result = minimize(
        lambda coefficients: objective(coefficients)[0],
        initial,
        jac=lambda coefficients: objective(coefficients)[1],
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-12},
    )
    _require(result.success, f"contact-transition fit failed: {result.message}")
    _require(np.all(np.isfinite(result.x)), "contact-transition fit is nonfinite")
    return np.asarray(result.x, dtype=np.float64)


def _training_rows(
    episodes: Sequence[ContactTransitionEpisode],
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    onset_x = []
    onset_y = []
    release_x = []
    release_y = []
    for episode in episodes:
        normalized = (_episode_features(episode) - feature_mean) / feature_scale
        previous = episode.contact_active[:-1].reshape(-1)
        current = episode.contact_active[1:].reshape(-1)
        rows = normalized[1:].reshape(-1, normalized.shape[2])
        onset_mask = ~previous
        release_mask = previous
        onset_x.append(rows[onset_mask])
        onset_y.append(current[onset_mask].astype(np.float64))
        release_x.append(rows[release_mask])
        release_y.append((~current[release_mask]).astype(np.float64))
    return (
        np.concatenate(onset_x),
        np.concatenate(onset_y),
        np.concatenate(release_x),
        np.concatenate(release_y),
    )


def predict_causal_contact_transition(
    model: ContactTransitionModel,
    openings_m: np.ndarray,
    controller_positions_m: np.ndarray,
    predicted_object_positions_m: np.ndarray,
    *,
    dt_seconds: float,
    initial_contact_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Recursively predict contact without future tactile labels."""

    features = contact_transition_geometry_features(
        openings_m,
        controller_positions_m,
        predicted_object_positions_m,
        dt_seconds=dt_seconds,
    )
    mean = np.asarray(model.feature_mean, dtype=np.float64)
    scale = np.asarray(model.feature_scale, dtype=np.float64)
    _require(
        mean.shape == scale.shape == (features.shape[2],),
        "contact-transition feature scaling changed",
    )
    normalized = (features - mean) / scale
    onset = np.asarray(model.onset_coefficients, dtype=np.float64)
    release = np.asarray(model.release_coefficients, dtype=np.float64)
    _require(
        onset.shape == release.shape == (features.shape[2] + 1,),
        "contact-transition coefficients changed",
    )
    state = np.asarray(initial_contact_state, dtype=bool).copy()
    _require(state.shape == (features.shape[1],), "initial contact state changed")
    states = np.empty(features.shape[:2], dtype=bool)
    probabilities = np.empty(features.shape[:2], dtype=np.float64)
    states[0] = state
    probabilities[0] = state.astype(np.float64)
    for frame in range(1, len(features)):
        design = np.column_stack((np.ones(features.shape[1]), normalized[frame]))
        onset_probability = _sigmoid(design @ onset)
        release_probability = _sigmoid(design @ release)
        active_probability = np.where(
            state, 1.0 - release_probability, onset_probability
        )
        transition_probability = np.where(state, release_probability, onset_probability)
        transition = transition_probability >= model.transition_threshold
        state = np.logical_xor(state, transition)
        probabilities[frame] = active_probability
        states[frame] = state
    return probabilities, states


def _contact_metrics(
    episodes: Sequence[ContactTransitionEpisode],
    model: ContactTransitionModel,
) -> dict[str, float]:
    probabilities = []
    predictions = []
    references = []
    onset_errors = []
    for episode in episodes:
        probability, state = predict_causal_contact_transition(
            model,
            episode.openings_m,
            episode.controller_positions_m,
            episode.predicted_object_positions_m,
            dt_seconds=episode.dt_seconds,
            initial_contact_state=episode.contact_active[0],
        )
        probabilities.append(probability[1:].reshape(-1))
        predictions.append(state[1:].reshape(-1))
        references.append(episode.contact_active[1:].reshape(-1))
        for controller in range(episode.contact_active.shape[1]):
            true_onset = np.flatnonzero(episode.contact_active[:, controller])
            pred_onset = np.flatnonzero(state[:, controller])
            if len(true_onset) and len(pred_onset):
                onset_errors.append(abs(int(pred_onset[0]) - int(true_onset[0])))
    probability_values = np.concatenate(probabilities)
    prediction_values = np.concatenate(predictions)
    reference_values = np.concatenate(references)
    positive_recall = (
        float(np.mean(prediction_values[reference_values]))
        if np.any(reference_values)
        else 1.0
    )
    negative_recall = (
        float(np.mean(~prediction_values[~reference_values]))
        if np.any(~reference_values)
        else 1.0
    )
    return {
        "brier_score": float(
            np.mean((probability_values - reference_values.astype(float)) ** 2)
        ),
        "accuracy": float(np.mean(prediction_values == reference_values)),
        "balanced_accuracy": 0.5 * (positive_recall + negative_recall),
        "mean_onset_absolute_error_frames": (
            float(np.mean(onset_errors)) if onset_errors else float("inf")
        ),
    }


def fit_causal_contact_transition(
    source_episodes: Sequence[ContactTransitionEpisode],
    calibration_episodes: Sequence[ContactTransitionEpisode],
    *,
    ridge_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    threshold_grid: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5),
) -> ContactTransitionFit:
    """Fit source hazards and choose ridge/threshold on calibration only."""

    _require(len(source_episodes) >= 2, "two source episodes are required")
    _require(len(calibration_episodes) >= 1, "a calibration episode is required")
    ridge_values = tuple(sorted(set(map(float, ridge_grid))))
    threshold_values = tuple(sorted(set(map(float, threshold_grid))))
    _require(
        ridge_values
        and all(np.isfinite(value) and value > 0.0 for value in ridge_values),
        "ridge grid must be finite and positive",
    )
    _require(
        threshold_values and all(0.0 < value <= 0.5 for value in threshold_values),
        "transition thresholds must lie in (0, 0.5]",
    )
    source_features = np.concatenate(
        [_episode_features(episode).reshape(-1, 3) for episode in source_episodes]
    )
    feature_mean = np.mean(source_features, axis=0)
    feature_scale = np.std(source_features, axis=0)
    feature_scale = np.maximum(feature_scale, 1e-8)
    onset_x, onset_y, release_x, release_y = _training_rows(
        source_episodes, feature_mean, feature_scale
    )
    candidate_table = []
    fitted: dict[tuple[float, float], ContactTransitionModel] = {}
    for ridge in ridge_values:
        onset = _fit_binary_logistic(onset_x, onset_y, ridge_strength=ridge)
        release = _fit_binary_logistic(release_x, release_y, ridge_strength=ridge)
        for threshold in threshold_values:
            model = ContactTransitionModel(
                feature_names=(
                    "gripper_openness_m",
                    "gripper_to_predicted_object_proximity_m",
                    "relative_closing_speed_m_s",
                ),
                feature_mean=tuple(map(float, feature_mean)),
                feature_scale=tuple(map(float, feature_scale)),
                onset_coefficients=tuple(map(float, onset)),
                release_coefficients=tuple(map(float, release)),
                ridge_strength=ridge,
                transition_threshold=threshold,
            )
            metrics = _contact_metrics(calibration_episodes, model)
            fitted[(ridge, threshold)] = model
            candidate_table.append(
                {
                    "ridge_strength": ridge,
                    "transition_threshold": threshold,
                    "calibration_metrics": metrics,
                }
            )
    selected = min(
        candidate_table,
        key=lambda row: (
            row["calibration_metrics"]["brier_score"],
            -row["calibration_metrics"]["balanced_accuracy"],
            row["calibration_metrics"]["mean_onset_absolute_error_frames"],
            row["ridge_strength"],
            row["transition_threshold"],
        ),
    )
    model = fitted[
        (
            float(selected["ridge_strength"]),
            float(selected["transition_threshold"]),
        )
    ]
    return ContactTransitionFit(
        model=model,
        source_metrics=_contact_metrics(source_episodes, model),
        calibration_metrics=_contact_metrics(calibration_episodes, model),
        candidate_table=tuple(candidate_table),
    )


__all__ = [
    "ContactTransitionEpisode",
    "ContactTransitionFit",
    "ContactTransitionModel",
    "PoolingControlSelection",
    "build_pooling_control_selection_artifact",
    "contact_transition_geometry_features",
    "evaluate_pooling_control",
    "fit_causal_contact_transition",
    "predict_causal_contact_transition",
    "select_pooling_controls",
    "validate_pooling_control_selection_artifact",
    "write_pooling_control_selection_artifact",
]
