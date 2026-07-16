"""Source-only Bayesian model averaging for reusable Deform360 twins."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from causal4d_public.deform360_phystwin_trust import (
    CausalTrustEpisode,
    CausalTrustWeights,
    causal_control_variate_prediction,
    score_causal_trust_interval,
)


REUSABLE_ENSEMBLE_SCHEMA_VERSION = 1
REUSABLE_ENSEMBLE_PROTOCOL_ID = "deform360-reusable-ensemble-081-v1"
CANONICAL_REUSABLE_ENSEMBLE_CONFIG_SHA256 = (
    "7a5c41a37a710d6daf65296a26d3e160e32b582047498f38362925c6c5e4f50c"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def reusable_ensemble_result_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def reusable_ensemble_config_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_reusable_ensemble_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable post-calibration source-development boundary."""

    _require(
        payload.get("schema_version") == REUSABLE_ENSEMBLE_SCHEMA_VERSION,
        "unsupported reusable-ensemble config schema",
    )
    observed = reusable_ensemble_config_sha256(payload)
    _require(
        payload.get("config_sha256") == observed,
        "reusable-ensemble config checksum mismatch",
    )
    _require(
        observed == CANONICAL_REUSABLE_ENSEMBLE_CONFIG_SHA256,
        "reusable-ensemble config differs from the canonical lock",
    )
    config = payload.get("config", {})
    _require(
        config.get("protocol_id") == REUSABLE_ENSEMBLE_PROTOCOL_ID,
        "reusable-ensemble protocol id changed",
    )
    source = config.get("source_fit", {})
    _require(
        source.get("episode_ids") == [1, 4, 6]
        and source.get("frame_range_half_open") == [1, 60]
        and source.get("temperature_grid") == [0.01, 0.03, 0.1, 0.3, 1.0]
        and source.get("minimum_effective_candidate_count") == 2.0
        and source.get("source_untouched_tails_used_for_fit") is False,
        "reusable-ensemble source fit changed",
    )
    boundary = config.get("information_boundary", {})
    _require(
        boundary.get("parent_calibration_outcomes_known_before_method_design")
        is True
        and boundary.get("parent_calibration_outcomes_allowed_for_numerical_fit")
        is False
        and boundary.get("sealed_target_media_allowed") is False,
        "reusable-ensemble information boundary changed",
    )
    target = config.get("sealed_target", {})
    _require(
        target.get("episode_id") == 5
        and target.get("may_open_under_this_protocol") is False,
        "reusable-ensemble target boundary changed",
    )
    return {
        "passed": True,
        "protocol_id": REUSABLE_ENSEMBLE_PROTOCOL_ID,
        "config_sha256": observed,
        "source_episode_ids": [1, 4, 6],
        "sealed_target_episode_id": 5,
    }


def load_reusable_ensemble_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_reusable_ensemble_config(payload)
    return payload


def supported_controller_count(
    *,
    controller_count: int,
    controller_spring_count: int,
) -> int:
    """Count only controller groups that can directly transmit an action."""

    _require(controller_count >= 1, "controller count must be positive")
    _require(controller_spring_count >= 1, "rollout has no controller support")
    return min(controller_count, controller_spring_count)


def gibbs_weights(losses: Mapping[str, float], temperature: float) -> dict[str, float]:
    """Convert finite source losses to a stable Gibbs posterior."""

    _require(np.isfinite(temperature) and temperature > 0.0, "invalid temperature")
    labels = sorted(losses)
    _require(bool(labels), "Gibbs posterior has no candidates")
    values = np.asarray([float(losses[label]) for label in labels], dtype=np.float64)
    _require(np.all(np.isfinite(values)), "Gibbs loss is non-finite")
    logits = -(values - np.min(values)) / temperature
    unnormalized = np.exp(logits)
    weights = unnormalized / np.sum(unnormalized)
    return {label: float(weight) for label, weight in zip(labels, weights, strict=True)}


def _posterior_diagnostics(weights: Mapping[str, float]) -> dict[str, float]:
    values = np.asarray(list(weights.values()), dtype=np.float64)
    _require(np.all(values >= 0.0), "posterior weight is negative")
    _require(np.isclose(values.sum(), 1.0), "posterior weights do not sum to one")
    positive = values[values > 0.0]
    entropy = float(-np.sum(positive * np.log(positive)))
    return {
        "entropy_nats": entropy,
        "normalized_entropy": float(entropy / np.log(len(values))),
        "effective_candidate_count": float(1.0 / np.sum(values**2)),
        "maximum_weight": float(np.max(values)),
    }


def trusted_candidate_prediction(
    episode: CausalTrustEpisode,
    *,
    base_action_response: float,
    autonomous_drift: float,
    controller_spring_count: int,
) -> np.ndarray:
    """Apply fixed trust using only controller groups with graph support."""

    effective_count = supported_controller_count(
        controller_count=episode.controller_count,
        controller_spring_count=controller_spring_count,
    )
    weights = CausalTrustWeights(
        action_response=base_action_response / float(effective_count),
        autonomous_drift=autonomous_drift,
    )
    return causal_control_variate_prediction(
        episode.target_m[:1], episode.driven_m, episode.zero_action_m, weights
    )


def ensemble_prediction(
    predictions: Mapping[str, np.ndarray],
    posterior_weights: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return posterior mean and coordinate variance over twin predictions."""

    labels = sorted(posterior_weights)
    _require(
        set(labels) == set(predictions), "posterior and prediction support differ"
    )
    values = [np.asarray(predictions[label], dtype=np.float64) for label in labels]
    shape = values[0].shape
    _require(all(value.shape == shape for value in values), "candidate shapes differ")
    weights = np.asarray(
        [posterior_weights[label] for label in labels], dtype=np.float64
    )
    _require(np.all(weights >= 0.0), "posterior weight is negative")
    _require(np.isclose(weights.sum(), 1.0), "posterior weights do not sum to one")
    stack = np.stack(values)
    mean = np.tensordot(weights, stack, axes=(0, 0))
    variance = np.tensordot(weights, (stack - mean[None]) ** 2, axes=(0, 0))
    return mean, variance


def _validate_candidate_bank(
    candidates: Mapping[str, Mapping[str, CausalTrustEpisode]],
    physical_parameters: Mapping[str, Mapping[str, float]],
    controller_springs: Mapping[str, int],
) -> tuple[list[str], list[str]]:
    labels = sorted(candidates)
    _require(len(labels) >= 2, "ensemble needs at least two physical candidates")
    _require(
        set(labels) == set(physical_parameters),
        "physical candidate metadata differs",
    )
    episode_ids = sorted(next(iter(candidates.values())))
    _require(len(episode_ids) >= 3, "ensemble needs at least three source actions")
    _require(set(episode_ids) == set(controller_springs), "support metadata differs")
    for label in labels:
        _require(
            sorted(candidates[label]) == episode_ids,
            "candidate source episode support differs",
        )
        for episode_id in episode_ids:
            episode = candidates[label][episode_id]
            _require(
                episode.episode_id == episode_id,
                "candidate episode identity changed",
            )
            supported_controller_count(
                controller_count=episode.controller_count,
                controller_spring_count=int(controller_springs[episode_id]),
            )
    return labels, episode_ids


def fit_source_gibbs_ensemble(
    candidates: Mapping[str, Mapping[str, CausalTrustEpisode]],
    *,
    physical_parameters: Mapping[str, Mapping[str, float]],
    controller_springs: Mapping[str, int],
    base_action_response: float,
    autonomous_drift: float,
    frame_range: tuple[int, int],
    temperature_grid: Sequence[float],
    minimum_effective_candidate_count: float = 2.0,
) -> dict[str, Any]:
    """Fit a Gibbs twin posterior with leave-one-source-action-out temperature."""

    labels, episode_ids = _validate_candidate_bank(
        candidates, physical_parameters, controller_springs
    )
    start, stop = (int(value) for value in frame_range)
    _require(start >= 1 and stop > start, "ensemble frame range is invalid")
    temperatures = tuple(float(value) for value in temperature_grid)
    _require(bool(temperatures), "temperature grid is empty")
    _require(
        len(set(temperatures)) == len(temperatures)
        and all(np.isfinite(value) and value > 0.0 for value in temperatures),
        "temperature grid is invalid",
    )
    _require(
        1.0 < minimum_effective_candidate_count <= len(labels),
        "minimum effective candidate count is invalid",
    )

    predictions: dict[str, dict[str, np.ndarray]] = {}
    metrics: dict[str, dict[str, dict[str, float | int]]] = {}
    for label in labels:
        predictions[label] = {}
        metrics[label] = {}
        for episode_id in episode_ids:
            episode = candidates[label][episode_id]
            prediction = trusted_candidate_prediction(
                episode,
                base_action_response=base_action_response,
                autonomous_drift=autonomous_drift,
                controller_spring_count=int(controller_springs[episode_id]),
            )
            predictions[label][episode_id] = prediction
            metrics[label][episode_id] = score_causal_trust_interval(
                episode, prediction, start, stop
            )

    pooled_losses = {
        label: float(
            np.mean(
                [
                    metrics[label][episode_id]["relative_score_vs_persistence"]
                    for episode_id in episode_ids
                ]
            )
        )
        for label in labels
    }
    temperature_table = []
    for temperature in temperatures:
        folds = []
        for held_out in episode_ids:
            fit_ids = [value for value in episode_ids if value != held_out]
            losses = {
                label: float(
                    np.mean(
                        [
                            metrics[label][episode_id][
                                "relative_score_vs_persistence"
                            ]
                            for episode_id in fit_ids
                        ]
                    )
                )
                for label in labels
            }
            weights = gibbs_weights(losses, temperature)
            point_map_label = min(labels, key=lambda label: (losses[label], label))
            held_out_mean, _ = ensemble_prediction(
                {
                    label: predictions[label][held_out]
                    for label in labels
                },
                weights,
            )
            held_out_metrics = score_causal_trust_interval(
                candidates[labels[0]][held_out], held_out_mean, start, stop
            )
            point_map_metrics = metrics[point_map_label][held_out]
            folds.append(
                {
                    "held_out_episode_id": held_out,
                    "fit_episode_ids": fit_ids,
                    "posterior_diagnostics": _posterior_diagnostics(weights),
                    "metrics": held_out_metrics,
                    "point_map_candidate_label": point_map_label,
                    "point_map_metrics": point_map_metrics,
                }
            )
        pooled_temperature_weights = gibbs_weights(pooled_losses, temperature)
        pooled_temperature_diagnostics = _posterior_diagnostics(
            pooled_temperature_weights
        )
        temperature_table.append(
            {
                "temperature": temperature,
                "execution_balanced_relative_score_vs_persistence": float(
                    np.mean(
                        [
                            fold["metrics"]["relative_score_vs_persistence"]
                            for fold in folds
                        ]
                    )
                ),
                "point_map_execution_balanced_relative_score_vs_persistence": float(
                    np.mean(
                        [
                            fold["point_map_metrics"][
                                "relative_score_vs_persistence"
                            ]
                            for fold in folds
                        ]
                    )
                ),
                "pooled_posterior_diagnostics": pooled_temperature_diagnostics,
                "effective_support_gate": (
                    pooled_temperature_diagnostics["effective_candidate_count"]
                    >= minimum_effective_candidate_count
                ),
                "folds": folds,
            }
        )
    eligible_temperatures = [
        row for row in temperature_table if row["effective_support_gate"]
    ]
    _require(
        bool(eligible_temperatures),
        "no source temperature preserves the minimum posterior support",
    )
    selected = min(
        eligible_temperatures,
        key=lambda row: (
            row["execution_balanced_relative_score_vs_persistence"],
            -row["temperature"],
        ),
    )
    selected_temperature = float(selected["temperature"])
    posterior = gibbs_weights(pooled_losses, selected_temperature)
    single_source_posteriors = {
        episode_id: gibbs_weights(
            {
                label: float(
                    metrics[label][episode_id]["relative_score_vs_persistence"]
                )
                for label in labels
            },
            selected_temperature,
        )
        for episode_id in episode_ids
    }
    result: dict[str, Any] = {
        "schema_version": REUSABLE_ENSEMBLE_SCHEMA_VERSION,
        "artifact_kind": "Deform360ReusableTwinSourceGibbsPosterior",
        "candidate_count": len(labels),
        "source_episode_ids": episode_ids,
        "frame_range_half_open": [start, stop],
        "fixed_action_trust": {
            "base_action_response": float(base_action_response),
            "autonomous_drift": float(autonomous_drift),
            "normalization": "supported_controller_count",
        },
        "controller_spring_count_by_episode": {
            key: int(value) for key, value in controller_springs.items()
        },
        "temperature_grid": list(temperatures),
        "minimum_effective_candidate_count": float(
            minimum_effective_candidate_count
        ),
        "temperature_selection": "leave-one-source-action-out",
        "temperature_table": temperature_table,
        "selected_temperature": selected_temperature,
        "pooled_source_losses": pooled_losses,
        "posterior_weights": posterior,
        "posterior_diagnostics": _posterior_diagnostics(posterior),
        "single_source_posterior_weights": single_source_posteriors,
        "physical_parameters": {
            label: dict(physical_parameters[label]) for label in labels
        },
        "source_gate": {
            "leave_one_action_out_beats_persistence": (
                selected["execution_balanced_relative_score_vs_persistence"] < 1.0
            ),
            "leave_one_action_out_matches_or_beats_point_map": (
                selected["execution_balanced_relative_score_vs_persistence"]
                <= selected[
                    "point_map_execution_balanced_relative_score_vs_persistence"
                ]
            ),
            "minimum_effective_candidate_count": (
                _posterior_diagnostics(posterior)["effective_candidate_count"]
                >= minimum_effective_candidate_count
            ),
        },
        "candidate_input_sha256": {
            label: {
                episode_id: {
                    "source_data": candidates[label][episode_id].source_data_sha256,
                    "driven_trajectory": candidates[label][
                        episode_id
                    ].driven_trajectory_sha256,
                    "zero_trajectory": candidates[label][
                        episode_id
                    ].zero_action_trajectory_sha256,
                }
                for episode_id in episode_ids
            }
            for label in labels
        },
        "information_boundary": {
            "source_train_frames_read": True,
            "source_untouched_tails_used_for_fit": False,
            "previous_calibration_outcomes_used_for_numerical_fit": False,
            "sealed_target_episode_read": False,
        },
        "claim_boundary": (
            "source-only Gibbs posterior; requires a fresh independent panel "
            "for reusable-twin or state-of-the-art claims"
        ),
    }
    result["source_gate"]["passed"] = all(result["source_gate"].values())
    result["result_sha256"] = reusable_ensemble_result_sha256(result)
    return result


def validate_source_gibbs_ensemble_artifact(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the source-only evidence boundary of a Gibbs posterior artifact."""

    _require(
        payload.get("schema_version") == REUSABLE_ENSEMBLE_SCHEMA_VERSION,
        "unsupported reusable-ensemble schema",
    )
    _require(
        payload.get("artifact_kind")
        == "Deform360ReusableTwinSourceGibbsPosterior",
        "unexpected reusable-ensemble artifact",
    )
    _require(
        payload.get("result_sha256") == reusable_ensemble_result_sha256(payload),
        "reusable-ensemble checksum mismatch",
    )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("source_untouched_tails_used_for_fit") is False
        and boundary.get("previous_calibration_outcomes_used_for_numerical_fit")
        is False
        and boundary.get("sealed_target_episode_read") is False,
        "reusable-ensemble information boundary changed",
    )
    weights = payload.get("posterior_weights", {})
    _require(
        len(weights) == payload.get("candidate_count")
        and np.isclose(sum(float(value) for value in weights.values()), 1.0),
        "reusable-ensemble posterior is invalid",
    )
    return {
        "passed": True,
        "result_sha256": payload["result_sha256"],
        "candidate_count": payload["candidate_count"],
        "selected_temperature": payload["selected_temperature"],
    }


def derive_source_trusted_point_map_control(
    gibbs_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze the source point-MAP control already evaluated by Gibbs LOO."""

    validate_source_gibbs_ensemble_artifact(gibbs_payload)
    selected_temperature = float(gibbs_payload["selected_temperature"])
    rows = [
        row
        for row in gibbs_payload["temperature_table"]
        if float(row["temperature"]) == selected_temperature
    ]
    _require(len(rows) == 1, "selected source temperature row is ambiguous")
    selected_row = rows[0]
    pooled_losses = gibbs_payload["pooled_source_losses"]
    pooled_label = min(
        pooled_losses,
        key=lambda label: (float(pooled_losses[label]), label),
    )
    single_labels = {
        episode_id: min(weights, key=lambda label: (-float(weights[label]), label))
        for episode_id, weights in gibbs_payload[
            "single_source_posterior_weights"
        ].items()
    }
    folds = selected_row["folds"]
    joint_wins = 0
    maximum_metric_degradation = -np.inf
    fold_records = []
    for fold in folds:
        metrics = fold["point_map_metrics"]
        track_ratio = float(metrics["track_rmse_m"]) / float(
            metrics["persistence_track_rmse_m"]
        )
        chamfer_ratio = float(metrics["chamfer_m"]) / float(
            metrics["persistence_chamfer_m"]
        )
        joint_win = track_ratio < 1.0 and chamfer_ratio < 1.0
        joint_wins += int(joint_win)
        maximum_metric_degradation = max(
            maximum_metric_degradation,
            track_ratio - 1.0,
            chamfer_ratio - 1.0,
        )
        fold_records.append(
            {
                "held_out_episode_id": fold["held_out_episode_id"],
                "fit_episode_ids": fold["fit_episode_ids"],
                "selected_candidate_label": fold["point_map_candidate_label"],
                "metrics": metrics,
                "joint_win_vs_persistence": joint_win,
            }
        )
    loo_labels = [record["selected_candidate_label"] for record in fold_records]
    result: dict[str, Any] = {
        "schema_version": REUSABLE_ENSEMBLE_SCHEMA_VERSION,
        "artifact_kind": "Deform360ReusableTwinSourceTrustedPointMapControl",
        "protocol_id": gibbs_payload.get("protocol_id"),
        "config_sha256": gibbs_payload.get("config_sha256"),
        "gibbs_result_sha256": gibbs_payload["result_sha256"],
        "source_episode_ids": gibbs_payload["source_episode_ids"],
        "frame_range_half_open": gibbs_payload["frame_range_half_open"],
        "fixed_action_trust": gibbs_payload["fixed_action_trust"],
        "selected_pooled_candidate_label": pooled_label,
        "selected_pooled_physical_parameters": gibbs_payload[
            "physical_parameters"
        ][pooled_label],
        "selected_single_source_candidate_labels": single_labels,
        "selected_single_source_physical_parameters": {
            episode_id: gibbs_payload["physical_parameters"][label]
            for episode_id, label in single_labels.items()
        },
        "leave_one_action_out": fold_records,
        "source_diagnostics": {
            "execution_balanced_relative_score_vs_persistence": selected_row[
                "point_map_execution_balanced_relative_score_vs_persistence"
            ],
            "joint_win_episode_count": joint_wins,
            "maximum_metric_degradation_fraction": float(
                maximum_metric_degradation
            ),
            "same_candidate_selected_in_every_outer_fold": (
                len(set(loo_labels)) == 1 and loo_labels[0] == pooled_label
            ),
        },
        "information_boundary": {
            "derived_only_from_source_gibbs_control": True,
            "source_untouched_tails_used_for_selection": False,
            "previous_calibration_outcomes_used_for_numerical_selection": False,
            "sealed_target_episode_read": False,
        },
        "claim_boundary": (
            "source-only trust-aligned point-MAP diagnostic; current calibration "
            "is exploratory and a fresh panel is required for claims"
        ),
    }
    result["result_sha256"] = reusable_ensemble_result_sha256(result)
    return result


__all__ = [
    "CANONICAL_REUSABLE_ENSEMBLE_CONFIG_SHA256",
    "REUSABLE_ENSEMBLE_SCHEMA_VERSION",
    "REUSABLE_ENSEMBLE_PROTOCOL_ID",
    "ensemble_prediction",
    "derive_source_trusted_point_map_control",
    "fit_source_gibbs_ensemble",
    "gibbs_weights",
    "load_reusable_ensemble_config",
    "reusable_ensemble_config_sha256",
    "reusable_ensemble_result_sha256",
    "supported_controller_count",
    "trusted_candidate_prediction",
    "validate_source_gibbs_ensemble_artifact",
    "validate_reusable_ensemble_config",
]
