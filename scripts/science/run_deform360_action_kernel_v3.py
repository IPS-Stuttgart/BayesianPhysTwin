#!/usr/bin/env python3
"""Nonlinear action-conditioned Deform360 tactile forecasting.

This v3 candidate was frozen before the predecessor ridge-v2 target result became
available.  It combines the registered action-ridge family with a local
state-and-future-action kernel family, using leave-one-source-episode-out scores
for every model weight, guard decision, bias correction, and covariance fit.
The target episode is selected from metadata and carrier identities before its
tactile payload is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "bayesian-phystwin/deform360-action-kernel-result-v3"
METHODS = (
    "persistence",
    "last_trend",
    "state_ridge",
    "state_kernel",
    "action_ridge",
    "action_kernel",
    "bayesian_action_ensemble",
    "shuffled_action_control",
    "guarded_action_ensemble",
)


def load_base() -> Any:
    path = Path(__file__).with_name("run_deform360_action_conditioned_tactile_v2.py")
    spec = importlib.util.spec_from_file_location(
        "deform360_action_conditioned_tactile_v2", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import predecessor evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise base.EvaluationError(f"expected JSON object: {path}")
    return value


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class KernelSpec:
    neighbors: int
    action_scale: float

    @property
    def name(self) -> str:
        return f"kernel_k{self.neighbors}_action{self.action_scale:g}"


@dataclass(frozen=True)
class KernelFit:
    state_mean: np.ndarray
    state_scale: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    train_state: np.ndarray
    train_action: np.ndarray
    train_target: np.ndarray
    spec: KernelSpec


@dataclass
class UnifiedCandidate:
    name: str
    family: str
    cv_objective: float
    per_episode_active_rmse: dict[int, float]
    cv_predictions: dict[int, np.ndarray]
    cv_truths: dict[int, np.ndarray]
    cv_currents: dict[int, np.ndarray]
    cv_masks: dict[int, np.ndarray]
    ridge_index: int | None = None
    kernel_spec: KernelSpec | None = None


def episode_rows(
    episode: Any,
    transform: Any,
    base_protocol: dict[str, Any],
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state, target, current, truth, active = base.design_for_episode(
        episode, transform, base_protocol, horizon, "state"
    )
    model = base_protocol["model"]
    starts = base.starts_for(
        len(episode.tactile),
        horizon,
        int(model["trend_lag_frames"]),
        int(model["window_stride_frames"]),
    )
    action = base.action_features(
        episode.robot_actions,
        starts,
        horizon,
        episode.descriptor.action,
    )
    if len(state) != len(action):
        raise base.EvaluationError("kernel state/action window counts disagree")
    return state, action, target, current, truth, active


def fit_kernel(
    state: np.ndarray,
    action: np.ndarray,
    target: np.ndarray,
    spec: KernelSpec,
) -> KernelFit:
    state_mean = state.mean(axis=0)
    state_scale = np.maximum(state.std(axis=0), 1e-6)
    action_mean = action.mean(axis=0)
    action_std = np.maximum(action.std(axis=0), 1e-6)
    return KernelFit(
        state_mean=state_mean,
        state_scale=state_scale,
        action_mean=action_mean,
        action_std=action_std,
        train_state=(state - state_mean) / state_scale,
        train_action=(action - action_mean) / action_std,
        train_target=np.asarray(target, dtype=np.float64),
        spec=spec,
    )


def kernel_predict(
    model: KernelFit,
    state: np.ndarray,
    action: np.ndarray,
) -> np.ndarray:
    query_state = (state - model.state_mean) / model.state_scale
    query_action = (action - model.action_mean) / model.action_std
    state_distance = np.mean(
        np.square(query_state[:, None, :] - model.train_state[None, :, :]),
        axis=2,
    )
    if model.spec.action_scale > 0.0:
        action_distance = np.mean(
            np.square(query_action[:, None, :] - model.train_action[None, :, :]),
            axis=2,
        )
        distance = state_distance + model.spec.action_scale * action_distance
    else:
        distance = state_distance
    neighbor_count = min(model.spec.neighbors, model.train_target.shape[0])
    if neighbor_count < 1:
        raise base.EvaluationError("kernel model has no training window")
    indices = np.argpartition(distance, neighbor_count - 1, axis=1)[:, :neighbor_count]
    selected_distance = np.take_along_axis(distance, indices, axis=1)
    selected_target = model.train_target[indices]
    positive = np.where(selected_distance > 0.0, selected_distance, np.nan)
    with np.errstate(invalid="ignore"):
        bandwidth = np.nanmedian(positive, axis=1)
    fallback = np.maximum(np.mean(selected_distance, axis=1), 1e-12)
    bandwidth = np.where(np.isfinite(bandwidth), bandwidth, fallback)
    bandwidth = np.maximum(bandwidth, 1e-12)
    logits = (
        -(selected_distance - np.min(selected_distance, axis=1, keepdims=True))
        / bandwidth[:, None]
    )
    weights = np.exp(logits)
    weights /= np.sum(weights, axis=1, keepdims=True)
    return np.einsum("qk,qkd->qd", weights, selected_target)


def kernel_cv_candidates(
    source: list[Any],
    transform: Any,
    base_protocol: dict[str, Any],
    horizon: int,
    specs: list[KernelSpec],
) -> tuple[list[UnifiedCandidate], dict[int, tuple[np.ndarray, ...]]]:
    rows = {
        episode.descriptor.episode_id: episode_rows(
            episode, transform, base_protocol, horizon
        )
        for episode in source
    }
    candidates: list[UnifiedCandidate] = []
    clip = float(base_protocol["model"]["normalized_feature_clip"])
    for spec in specs:
        predictions: dict[int, np.ndarray] = {}
        truths: dict[int, np.ndarray] = {}
        currents: dict[int, np.ndarray] = {}
        masks: dict[int, np.ndarray] = {}
        per_episode: dict[int, float] = {}
        squared: list[float] = []
        for held in source:
            held_id = held.descriptor.episode_id
            train_rows = [
                row for episode_id, row in rows.items() if episode_id != held_id
            ]
            train_state = np.concatenate([row[0] for row in train_rows])
            train_action = np.concatenate([row[1] for row in train_rows])
            train_target = np.concatenate([row[2] for row in train_rows])
            fit = fit_kernel(train_state, train_action, train_target, spec)
            state, action, _, current, truth, active = rows[held_id]
            latent = kernel_predict(fit, state, action)
            prediction = base.decode_prediction(latent, current, transform, clip)
            active_rmse = base.rmse(prediction, truth, active)[1]
            predictions[held_id] = prediction
            truths[held_id] = truth
            currents[held_id] = current
            masks[held_id] = active
            per_episode[held_id] = active_rmse
            squared.append(active_rmse * active_rmse)
        candidates.append(
            UnifiedCandidate(
                name=spec.name,
                family="state_kernel" if spec.action_scale == 0.0 else "action_kernel",
                cv_objective=float(np.mean(squared)),
                per_episode_active_rmse=per_episode,
                cv_predictions=predictions,
                cv_truths=truths,
                cv_currents=currents,
                cv_masks=masks,
                kernel_spec=spec,
            )
        )
    return candidates, rows


def ridge_candidates(
    source: list[Any],
    transform: Any,
    base_protocol: dict[str, Any],
    horizon: int,
) -> tuple[list[UnifiedCandidate], list[Any], list[Any]]:
    raw = base.candidate_cv(source, transform, base_protocol, horizon)
    unified: list[UnifiedCandidate] = []
    for index, candidate in enumerate(raw):
        unified.append(
            UnifiedCandidate(
                name=candidate.name,
                family=(
                    "action_ridge" if candidate.variant == "action" else "state_ridge"
                ),
                cv_objective=float(candidate.cv_objective),
                per_episode_active_rmse=dict(candidate.per_episode_active_rmse),
                cv_predictions=dict(candidate.cv_predictions),
                cv_truths=dict(candidate.cv_truths),
                cv_currents=dict(candidate.cv_currents),
                cv_masks=dict(candidate.cv_masks),
                ridge_index=index,
            )
        )
    fitted = base.fit_candidates_all_source(
        source, transform, base_protocol, horizon, raw
    )
    return unified, raw, fitted


def generalized_bayes_weights(
    candidates: list[UnifiedCandidate], floor_fraction: float
) -> tuple[np.ndarray, float]:
    losses = np.asarray(
        [candidate.cv_objective for candidate in candidates], dtype=np.float64
    )
    minimum = float(np.min(losses))
    spread = float(np.median(np.abs(losses - np.median(losses))))
    temperature = max(spread, minimum * floor_fraction, 1e-12)
    logits = -(losses - minimum) / temperature
    logits -= np.max(logits)
    weights = np.exp(logits)
    weights /= np.sum(weights)
    return weights, temperature


def ensemble_source_residuals(
    source: list[Any],
    candidates: list[UnifiedCandidate],
    weights: np.ndarray,
) -> tuple[np.ndarray, dict[int, float], np.ndarray]:
    raw_residuals: dict[int, np.ndarray] = {}
    ensemble_predictions: dict[int, np.ndarray] = {}
    for episode in source:
        episode_id = episode.descriptor.episode_id
        stacked = np.stack(
            [candidate.cv_predictions[episode_id] for candidate in candidates]
        )
        prediction = np.einsum("k,kwd->wd", weights, stacked)
        truth = candidates[0].cv_truths[episode_id]
        ensemble_predictions[episode_id] = prediction
        raw_residuals[episode_id] = (truth - prediction).reshape(len(truth), -1)
    global_bias = np.concatenate(list(raw_residuals.values())).mean(axis=0)
    corrected_residuals: list[np.ndarray] = []
    per_episode: dict[int, float] = {}
    for episode in source:
        episode_id = episode.descriptor.episode_id
        donor = np.concatenate(
            [
                residual
                for other_id, residual in raw_residuals.items()
                if other_id != episode_id
            ]
        )
        donor_bias = donor.mean(axis=0)
        corrected_residuals.append(raw_residuals[episode_id] - donor_bias[None, :])
        truth = candidates[0].cv_truths[episode_id]
        active = candidates[0].cv_masks[episode_id]
        prediction = ensemble_predictions[episode_id] + donor_bias[None, :]
        per_episode[episode_id] = base.rmse(prediction, truth, active)[1]
    return np.concatenate(corrected_residuals), per_episode, global_bias


def fit_all_kernel_models(
    source_rows: dict[int, tuple[np.ndarray, ...]],
    candidates: list[UnifiedCandidate],
) -> dict[str, KernelFit]:
    state = np.concatenate([row[0] for row in source_rows.values()])
    action = np.concatenate([row[1] for row in source_rows.values()])
    target = np.concatenate([row[2] for row in source_rows.values()])
    result: dict[str, KernelFit] = {}
    for candidate in candidates:
        if candidate.kernel_spec is not None:
            result[candidate.name] = fit_kernel(
                state, action, target, candidate.kernel_spec
            )
    return result


def source_guard(
    source: list[Any],
    base_protocol: dict[str, Any],
    horizon: int,
    transform: Any,
    state_ridge: UnifiedCandidate,
    state_kernel: UnifiedCandidate,
    ensemble_per_episode: dict[int, float],
) -> tuple[bool, str, dict[str, float]]:
    baselines = base.source_baseline_metrics(source, transform, base_protocol, horizon)
    means = {
        "persistence": float(np.mean(list(baselines["persistence"].values()))),
        "last_trend": float(np.mean(list(baselines["last_trend"].values()))),
        "state_ridge": float(
            np.mean(list(state_ridge.per_episode_active_rmse.values()))
        ),
        "state_kernel": float(
            np.mean(list(state_kernel.per_episode_active_rmse.values()))
        ),
        "bayesian_action_ensemble": float(np.mean(list(ensemble_per_episode.values()))),
    }
    fallback = min(
        ("persistence", "last_trend", "state_ridge", "state_kernel"),
        key=lambda name: means[name],
    )
    guard = base_protocol["guard"]
    accepts = bool(
        means["bayesian_action_ensemble"]
        < (1.0 - float(guard["minimum_relative_gain"])) * means[fallback]
        and all(
            ensemble_per_episode[episode.descriptor.episode_id]
            <= (1.0 + float(guard["maximum_episode_regret_fraction"]))
            * min(
                baselines["persistence"][episode.descriptor.episode_id],
                baselines["last_trend"][episode.descriptor.episode_id],
                state_ridge.per_episode_active_rmse[episode.descriptor.episode_id],
                state_kernel.per_episode_active_rmse[episode.descriptor.episode_id],
            )
            for episode in source
        )
    )
    return accepts, fallback, means


def target_candidate_prediction(
    candidate: UnifiedCandidate,
    ridge_fits: list[Any],
    kernel_fits: dict[str, KernelFit],
    target_rows: tuple[np.ndarray, ...],
    target_action_design: np.ndarray,
    current: np.ndarray,
    transform: Any,
    clip: float,
) -> np.ndarray:
    state, action = target_rows[:2]
    if candidate.ridge_index is not None:
        latent = base.predict_ridge(
            ridge_fits[candidate.ridge_index], target_action_design
        )
    elif candidate.kernel_spec is not None:
        latent = kernel_predict(kernel_fits[candidate.name], state, action)
    else:
        raise base.EvaluationError("candidate has no fitted implementation")
    return base.decode_prediction(latent, current, transform, clip)


def shuffled_candidate_prediction(
    candidate: UnifiedCandidate,
    ridge_fits: list[Any],
    kernel_fits: dict[str, KernelFit],
    target_rows: tuple[np.ndarray, ...],
    target_action_design: np.ndarray,
    state_width: int,
    permutation: np.ndarray,
    current: np.ndarray,
    transform: Any,
    clip: float,
) -> np.ndarray:
    state, action = target_rows[:2]
    if candidate.ridge_index is not None:
        design = target_action_design.copy()
        design[:, state_width:] = design[permutation, state_width:]
        latent = base.predict_ridge(ridge_fits[candidate.ridge_index], design)
    elif candidate.kernel_spec is not None:
        latent = kernel_predict(kernel_fits[candidate.name], state, action[permutation])
    else:
        raise base.EvaluationError("candidate has no fitted implementation")
    return base.decode_prediction(latent, current, transform, clip)


def evaluate_object(
    descriptors: list[Any],
    protocol: dict[str, Any],
    base_protocol: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    target_descriptor = max(descriptors, key=lambda item: item.episode_id)
    source_descriptors = [
        descriptor for descriptor in descriptors if descriptor is not target_descriptor
    ]
    source = [base.load_episode(descriptor) for descriptor in source_descriptors]
    horizon = int(protocol["shared_preprocessing"]["forecast_horizon_frames"])
    transform = base.build_transform(source, base_protocol, horizon)

    ridge_unified, _, ridge_fits = ridge_candidates(
        source, transform, base_protocol, horizon
    )
    kernel_config = protocol["kernel_candidates"]
    kernel_specs = [
        KernelSpec(int(neighbors), 0.0)
        for neighbors in kernel_config["state_only_control_neighbor_counts"]
    ] + [
        KernelSpec(int(neighbors), float(action_scale))
        for neighbors in kernel_config["neighbor_counts"]
        for action_scale in kernel_config["action_distance_scales"]
    ]
    kernel_unified, source_rows = kernel_cv_candidates(
        source, transform, base_protocol, horizon, kernel_specs
    )
    kernel_fits = fit_all_kernel_models(source_rows, kernel_unified)

    state_ridge = min(
        (candidate for candidate in ridge_unified if candidate.family == "state_ridge"),
        key=lambda candidate: candidate.cv_objective,
    )
    state_kernel = min(
        (
            candidate
            for candidate in kernel_unified
            if candidate.family == "state_kernel"
        ),
        key=lambda candidate: candidate.cv_objective,
    )
    action_ridge = min(
        (
            candidate
            for candidate in ridge_unified
            if candidate.family == "action_ridge"
        ),
        key=lambda candidate: candidate.cv_objective,
    )
    action_kernel = min(
        (
            candidate
            for candidate in kernel_unified
            if candidate.family == "action_kernel"
        ),
        key=lambda candidate: candidate.cv_objective,
    )
    action_candidates = [
        candidate
        for candidate in ridge_unified + kernel_unified
        if candidate.family in {"action_ridge", "action_kernel"}
    ]
    weights, temperature = generalized_bayes_weights(
        action_candidates,
        float(protocol["model_averaging"]["temperature_floor_fraction"]),
    )
    residuals, ensemble_source, source_bias = ensemble_source_residuals(
        source, action_candidates, weights
    )
    guard_accepts, fallback, source_means = source_guard(
        source,
        base_protocol,
        horizon,
        transform,
        state_ridge,
        state_kernel,
        ensemble_source,
    )
    covariance = base.fit_covariance(
        residuals,
        int(protocol["uncertainty"]["maximum_low_rank"]),
        float(protocol["uncertainty"]["coverage_probability"]),
        mean_error=source_bias,
    )
    source_fit_id = canonical_digest(
        {
            "object_id": target_descriptor.object_id,
            "source_episode_ids": [item.episode_id for item in source_descriptors],
            "target_episode_id": target_descriptor.episode_id,
            "candidate_names": [candidate.name for candidate in action_candidates],
            "candidate_weights": weights.tolist(),
            "candidate_objectives": [
                candidate.cv_objective for candidate in action_candidates
            ],
            "state_ridge": state_ridge.name,
            "state_kernel": state_kernel.name,
            "action_ridge": action_ridge.name,
            "action_kernel": action_kernel.name,
            "guard_accepts": guard_accepts,
            "fallback": fallback,
            "feature_scale": transform.feature_scale.tolist(),
            "state_mean": transform.state_mean.tolist(),
            "state_basis": transform.state_basis.tolist(),
            "delta_mean": transform.delta_mean.tolist(),
            "delta_basis": transform.delta_basis.tolist(),
            "covariance_mean_error": covariance.mean_error.tolist(),
            "covariance_diagonal": covariance.diagonal.tolist(),
            "covariance_factor": covariance.factor.tolist(),
            "covariance_multiplier": covariance.multiplier,
            "marginal_z": covariance.marginal_z,
        }
    )

    target = base.load_episode(target_descriptor)
    target_rows = episode_rows(target, transform, base_protocol, horizon)
    state, _, _, current, truth, active = target_rows
    target_action_design = base.design_for_episode(
        target, transform, base_protocol, horizon, "action"
    )[0]
    clip = float(base_protocol["model"]["normalized_feature_clip"])
    values = base.normalize_tactile(target.tactile, transform.feature_scale, clip)
    starts = base.starts_for(
        len(values),
        horizon,
        int(base_protocol["model"]["trend_lag_frames"]),
        int(base_protocol["model"]["window_stride_frames"]),
    )
    predictions: dict[str, np.ndarray] = {
        "persistence": base.baseline_prediction(
            values,
            starts,
            horizon,
            int(base_protocol["model"]["trend_lag_frames"]),
            clip,
            "persistence",
        ),
        "last_trend": base.baseline_prediction(
            values,
            starts,
            horizon,
            int(base_protocol["model"]["trend_lag_frames"]),
            clip,
            "last_trend",
        ),
    }
    state_target_design = base.design_for_episode(
        target, transform, base_protocol, horizon, "state"
    )[0]
    state_ridge_latent = base.predict_ridge(
        ridge_fits[state_ridge.ridge_index], state_target_design
    )
    predictions["state_ridge"] = base.decode_prediction(
        state_ridge_latent, current, transform, clip
    )
    state_kernel_latent = kernel_predict(
        kernel_fits[state_kernel.name], state, target_rows[1]
    )
    predictions["state_kernel"] = base.decode_prediction(
        state_kernel_latent, current, transform, clip
    )
    predictions["action_ridge"] = target_candidate_prediction(
        action_ridge,
        ridge_fits,
        kernel_fits,
        target_rows,
        target_action_design,
        current,
        transform,
        clip,
    )
    predictions["action_kernel"] = target_candidate_prediction(
        action_kernel,
        ridge_fits,
        kernel_fits,
        target_rows,
        target_action_design,
        current,
        transform,
        clip,
    )
    candidate_predictions = np.stack(
        [
            target_candidate_prediction(
                candidate,
                ridge_fits,
                kernel_fits,
                target_rows,
                target_action_design,
                current,
                transform,
                clip,
            )
            for candidate in action_candidates
        ]
    )
    ensemble = np.einsum("k,kwd->wd", weights, candidate_predictions)
    ensemble = np.clip(ensemble + covariance.mean_error[None, :], 0.0, clip)
    predictions["bayesian_action_ensemble"] = ensemble
    permutation = rng.permutation(len(current))
    state_width = state_target_design.shape[1]
    shuffled_candidates = np.stack(
        [
            shuffled_candidate_prediction(
                candidate,
                ridge_fits,
                kernel_fits,
                target_rows,
                target_action_design,
                state_width,
                permutation,
                current,
                transform,
                clip,
            )
            for candidate in action_candidates
        ]
    )
    predictions["shuffled_action_control"] = np.clip(
        np.einsum("k,kwd->wd", weights, shuffled_candidates)
        + covariance.mean_error[None, :],
        0.0,
        clip,
    )
    predictions["guarded_action_ensemble"] = (
        ensemble if guard_accepts else predictions[fallback]
    )

    metrics: dict[str, dict[str, float]] = {}
    for method in METHODS:
        all_rmse, active_rmse = base.rmse(predictions[method], truth, active)
        metrics[method] = {
            "field_rmse": all_rmse,
            "active_field_rmse": active_rmse,
            "field_mae": float(np.mean(np.abs(predictions[method] - truth))),
        }
    uncertainty = base.probabilistic_metrics(
        (truth - ensemble).reshape(len(truth), -1),
        covariance,
        float(protocol["uncertainty"]["coverage_probability"]),
        rng,
        int(protocol["uncertainty"]["energy_score_samples"]),
    )
    return {
        "object_id": target_descriptor.object_id,
        "source_episode_ids": [item.episode_id for item in source_descriptors],
        "source_actions": [item.action for item in source_descriptors],
        "target_episode_id": target_descriptor.episode_id,
        "target_action": target_descriptor.action,
        "target_action_family": base.action_family(target_descriptor.action),
        "target_bimanual": bool(target.bimanual),
        "source_fit_id": source_fit_id,
        "source_fit_frozen_before_target_tactile_open": True,
        "predecessor_target_result_available_at_protocol_freeze": False,
        "forecast_horizon_frames": horizon,
        "forecast_window_count": int(len(current)),
        "pooled_field_dimension": int(truth.shape[1]),
        "candidate_temperature": temperature,
        "candidate_names": [candidate.name for candidate in action_candidates],
        "candidate_families": [candidate.family for candidate in action_candidates],
        "candidate_weights": [float(value) for value in weights],
        "selected_state_ridge": state_ridge.name,
        "selected_state_kernel": state_kernel.name,
        "selected_action_ridge": action_ridge.name,
        "selected_action_kernel": action_kernel.name,
        "source_cv_active_rmse": source_means,
        "guard_accepts": guard_accepts,
        "fallback_method": fallback,
        "metrics": metrics,
        "uncertainty": uncertainty,
        "source_uncertainty_calibration": {
            "marginal_coverage": covariance.source_marginal_coverage,
            "joint_nanees": covariance.source_joint_nanees,
            "multiplier": covariance.multiplier,
            "rank": int(covariance.factor.shape[1]),
        },
        "target_fingerprint": target.fingerprints,
        "source_fingerprints": [episode.fingerprints for episode in source],
    }


def bootstrap_interval(values: np.ndarray, repetitions: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def aggregate(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    metric_names = ("field_rmse", "active_field_rmse", "field_mae")
    methods = {
        method: {
            metric: float(np.mean([row["metrics"][method][metric] for row in rows]))
            for metric in metric_names
        }
        for method in METHODS
    }
    primary = "active_field_rmse"
    comparisons: dict[str, Any] = {}
    for comparator in (
        "persistence",
        "last_trend",
        "state_ridge",
        "state_kernel",
        "action_ridge",
        "action_kernel",
        "shuffled_action_control",
    ):
        differences = np.asarray(
            [
                row["metrics"]["bayesian_action_ensemble"][primary]
                - row["metrics"][comparator][primary]
                for row in rows
            ]
        )
        denominator = float(
            np.mean([row["metrics"][comparator][primary] for row in rows])
        )
        comparisons[comparator] = {
            "ensemble_minus_comparator": float(np.mean(differences)),
            "relative_change": float(np.mean(differences) / denominator),
            "object_bootstrap_95_interval": bootstrap_interval(
                differences,
                int(protocol["statistics"]["bootstrap_repetitions"]),
                int(protocol["statistics"]["random_seed"]),
            ),
            "object_wins": int(np.sum(differences < 0.0)),
            "object_ties": int(np.sum(differences == 0.0)),
            "object_losses": int(np.sum(differences > 0.0)),
            "worst_object_regret": float(np.max(differences)),
        }
    uncertainty = {
        key: float(np.mean([row["uncertainty"][key] for row in rows]))
        for key in rows[0]["uncertainty"]
    }
    return {
        "object_count": len(rows),
        "primary_metric": primary,
        "primary_horizon_frames": int(
            protocol["shared_preprocessing"]["forecast_horizon_frames"]
        ),
        "methods": methods,
        "comparisons": comparisons,
        "guard_acceptance_fraction": float(
            np.mean([bool(row["guard_accepts"]) for row in rows])
        ),
        "target_action_families": sorted({row["target_action_family"] for row in rows}),
        "uncertainty": uncertainty,
    }


def report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# Deform360 nonlinear action-kernel development v3",
        "",
        f"- Status: **{result['status']}**",
        f"- Objects: **{summary['object_count']}**",
        f"- Horizon: **{summary['primary_horizon_frames']} frames**",
        f"- Guard acceptance: **{summary['guard_acceptance_fraction']:.1%}**",
        "- Protocol frozen before predecessor ridge-v2 target result: **yes**",
        "",
        "## Object-balanced results",
        "",
        "| Method | Active RMSE | All-field RMSE | MAE |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        value = summary["methods"][method]
        lines.append(
            f"| `{method}` | {value['active_field_rmse']:.8g} | "
            f"{value['field_rmse']:.8g} | {value['field_mae']:.8g} |"
        )
    lines.extend(
        [
            "",
            "## Paired object-level contrasts",
            "",
            "Negative values favor `bayesian_action_ensemble`.",
            "",
            "| Comparator | Difference | Relative | 95% bootstrap | W/T/L | Worst regret |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for comparator, value in summary["comparisons"].items():
        interval = value["object_bootstrap_95_interval"]
        lines.append(
            f"| `{comparator}` | {value['ensemble_minus_comparator']:.8g} | "
            f"{value['relative_change']:+.2%} | "
            f"[{interval[0]:.8g}, {interval[1]:.8g}] | "
            f"{value['object_wins']}/{value['object_ties']}/{value['object_losses']} | "
            f"{value['worst_object_regret']:.8g} |"
        )
    lines.extend(
        [
            "",
            "## Probabilistic diagnostics",
            "",
            "| Diagnostic | Value |",
            "|---|---:|",
        ]
    )
    for key, value in summary["uncertainty"].items():
        lines.append(f"| `{key}` | {value:.8g} |")
    lines.extend(
        [
            "",
            "Targets were selected before target tactile loading. Every model, kernel",
            "scale, model weight, guard, bias correction, and covariance parameter was",
            "fit from other episodes of the same object. Future robot motion is an allowed",
            "intervention input; future tactile response is used only for scoring.",
            "Reserved objects and camera/geometry payloads remain closed.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_protocols(
    protocol: dict[str, Any],
    base_protocol: dict[str, Any],
    root: Path,
) -> None:
    if (
        protocol.get("schema")
        != "bayesian-phystwin/deform360-action-kernel-protocol-v3"
    ):
        raise base.EvaluationError("unexpected v3 protocol schema")
    if Path(str(protocol["dataset_root"])) != root:
        raise base.EvaluationError("v3 dataset root changed")
    if Path(str(base_protocol["dataset_root"])) != root:
        raise base.EvaluationError("base dataset root changed")
    if list(protocol["development_object_ids"]) != list(
        base_protocol["development_object_ids"]
    ):
        raise base.EvaluationError("v3 development roster changed")
    if set(protocol["reserved_object_ids"]) != set(
        base_protocol["reserved_object_ids"]
    ):
        raise base.EvaluationError("v3 reserved roster changed")
    if set(protocol["development_object_ids"]) & set(protocol["reserved_object_ids"]):
        raise base.EvaluationError("development and reserved rosters overlap")
    if (
        protocol["information_boundary"].get(
            "protocol_frozen_before_predecessor_target_result_available"
        )
        is not True
    ):
        raise base.EvaluationError("v3 pre-result freeze was not declared")
    if protocol.get("paper_claim_authorized") is not False:
        raise base.EvaluationError("v3 protocol self-authorized a claim")


def run(protocol_path: Path, root: Path) -> dict[str, Any]:
    protocol = read_json(protocol_path)
    base_protocol_path = Path(protocol["shared_preprocessing"]["base_protocol_path"])
    base_protocol = read_json(base_protocol_path)
    root = root.resolve(strict=True)
    validate_protocols(protocol, base_protocol, root)
    descriptors_by_object: dict[str, list[Any]] = {}
    for object_id in protocol["development_object_ids"]:
        descriptors = base.discover_object(
            root,
            str(object_id),
            int(protocol["selection"]["minimum_episodes"]),
        )
        if descriptors:
            descriptors_by_object[str(object_id)] = descriptors
    if len(descriptors_by_object) < int(protocol["selection"]["minimum_objects"]):
        raise base.EvaluationError("too few objects expose the registered carriers")
    rng = np.random.default_rng(int(protocol["statistics"]["random_seed"]))
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for object_id in protocol["development_object_ids"]:
        descriptors = descriptors_by_object.get(str(object_id))
        if descriptors is None:
            failures.append(
                {
                    "object_id": str(object_id),
                    "reason": "registered-carriers-unavailable",
                }
            )
            continue
        try:
            rows.append(evaluate_object(descriptors, protocol, base_protocol, rng))
        except (
            base.EvaluationError,
            OSError,
            ValueError,
            np.linalg.LinAlgError,
        ) as error:
            failures.append(
                {
                    "object_id": str(object_id),
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
    if len(rows) < int(protocol["selection"]["minimum_objects"]):
        raise base.EvaluationError(
            f"only {len(rows)} objects completed; failures={failures}"
        )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 3,
        "status": "complete",
        "protocol_id": protocol["protocol_id"],
        "dataset_root": str(root),
        "github_sha": os.environ.get("GITHUB_SHA"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "information_boundary": {
            "protocol_frozen_before_predecessor_target_result_available": True,
            "development_robot_trajectories_opened": True,
            "development_tactile_responses_opened": True,
            "future_robot_trajectory_is_intervention_input": True,
            "target_tactile_opened_after_source_fit": True,
            "camera_pixels_opened": False,
            "geometry_or_point_cloud_opened": False,
            "reserved_object_payloads_opened": False,
            "fresh_confirmation_authorized": False,
            "paper_claim_authorized": False,
        },
        "summary": aggregate(rows, protocol),
        "objects": rows,
        "failures": failures,
        "protocol": protocol,
    }
    result["result_sha256"] = canonical_digest(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.protocol, args.data_root)
    write_json(args.output_json, result)
    args.output_report.write_text(report(result), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
