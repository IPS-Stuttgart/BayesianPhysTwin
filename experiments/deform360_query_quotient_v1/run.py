#!/usr/bin/env python3
"""Source-only query-quotient diagnostics on public Deform360 point clouds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.query_quotient_belief_v1 import (
    aggregate_to_query_quotient,
    minimum_information_query_lift,
    query_ambiguity_envelope,
    query_quotient_information_decomposition,
)
from experiments.deform360_real_v1.run import (
    Carrier,
    Profile,
    canonical_bytes,
    load_pcd_tar_sequence,
    write_json,
)


@dataclass(frozen=True, slots=True)
class EpisodeData:
    """Persistent public point positions for one permitted source episode."""

    episode_id: int
    action: str
    positions_m: np.ndarray
    archive_metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class TrainingPrior:
    """Equal-episode prior over the registered damping grid."""

    rho_grid: np.ndarray
    weights: np.ndarray
    residual_variance_m2: float
    training_episode_ids: tuple[int, ...]
    episode_rho_estimates: tuple[float, ...]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _content_id(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("result_sha256", None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def load_protocol(path: Path) -> dict[str, Any]:
    """Load and validate the frozen source-only public-data contract."""

    value = json.loads(path.read_text(encoding="utf-8"))
    _require(
        value.get("schema")
        == "bayesian-phystwin/deform360-query-quotient-real-v1",
        "unexpected protocol schema",
    )
    _require(value.get("schema_version") == 1, "unexpected protocol version")
    source = tuple(value.get("source_episode_ids", ()))
    forbidden = tuple(value.get("forbidden_episode_ids", ()))
    _require(
        len(source) >= 3
        and len(set(source)) == len(source)
        and all(type(item) is int and item >= 0 for item in source),
        "source episode roster is invalid",
    )
    _require(
        len(set(forbidden)) == len(forbidden)
        and all(type(item) is int and item >= 0 for item in forbidden),
        "forbidden episode roster is invalid",
    )
    _require(not set(source) & set(forbidden), "source and forbidden rosters overlap")
    actions = value.get("source_episode_actions")
    _require(isinstance(actions, dict), "source episode actions are missing")
    _require(
        set(actions) == {str(item) for item in source}
        and all(isinstance(item, str) and item for item in actions.values()),
        "source episode actions do not match the source roster",
    )
    query = value.get("query")
    _require(isinstance(query, dict), "query contract is missing")
    thresholds = tuple(float(item) for item in query.get("transport_factor_thresholds", ()))
    labels = tuple(query.get("class_labels", ()))
    _require(
        thresholds
        and tuple(sorted(set(thresholds))) == thresholds
        and all(np.isfinite(item) for item in thresholds),
        "query thresholds must be finite, sorted, and unique",
    )
    _require(
        len(labels) == len(thresholds) + 1
        and len(set(labels)) == len(labels)
        and all(isinstance(item, str) and item for item in labels),
        "query class labels are invalid",
    )
    grid = value.get("rho_grid")
    _require(isinstance(grid, dict), "rho grid is missing")
    grid_minimum = float(grid.get("minimum", math.nan))
    grid_maximum = float(grid.get("maximum", math.nan))
    grid_count = grid.get("count")
    _require(
        np.isfinite(grid_minimum)
        and np.isfinite(grid_maximum)
        and grid_minimum < grid_maximum
        and type(grid_count) is int
        and grid_count >= 3,
        "rho grid is invalid",
    )
    for key in (
        "horizon_frames",
        "reset_count",
        "minimum_prefix_frames",
        "maximum_frames",
        "maximum_points",
        "likelihood_information_cap",
        "analysis",
        "information_boundary",
    ):
        _require(key in value, f"protocol field is missing: {key}")
    for key in (
        "horizon_frames",
        "reset_count",
        "minimum_prefix_frames",
        "maximum_frames",
        "maximum_points",
        "likelihood_information_cap",
    ):
        _require(
            type(value[key]) is int and value[key] >= 1,
            f"protocol field must be a positive integer: {key}",
        )
    _require(
        value["maximum_frames"]
        >= value["minimum_prefix_frames"] + value["horizon_frames"] + 2,
        "maximum frame budget cannot support the registered query",
    )
    for key in (
        "prior_variance_floor",
        "latent_rho_decision_threshold",
        "minimum_velocity_energy_m2",
    ):
        item = value.get(key)
        _require(
            type(item) in {int, float}
            and type(item) is not bool
            and np.isfinite(item),
            f"protocol field must be finite: {key}",
        )
    _require(value["prior_variance_floor"] > 0.0, "prior variance floor must be positive")
    _require(
        value["minimum_velocity_energy_m2"] > 0.0,
        "minimum velocity energy must be positive",
    )
    analysis = value["analysis"]
    _require(isinstance(analysis, dict), "analysis contract is missing")
    _require(
        type(analysis.get("bootstrap_repetitions")) is int
        and analysis["bootstrap_repetitions"] >= 1,
        "bootstrap repetition count is invalid",
    )
    _require(
        type(analysis.get("bootstrap_seed")) is int,
        "bootstrap seed is invalid",
    )
    boundary = value["information_boundary"]
    _require(isinstance(boundary, dict), "information boundary is missing")
    required_true = (
        "public_real_measurements",
        "source_only",
        "leave_one_source_episode_out",
        "held_episode_prefix_used_for_adaptation",
        "held_episode_future_used_for_query_scoring_only",
    )
    required_false = (
        "forbidden_episode_payloads_opened",
        "official_velocity_arrays_used",
        "dataset_modified",
        "raw_payload_uploaded",
        "fresh_confirmation_authorized",
        "paper_claim_authorized",
    )
    _require(
        all(boundary.get(key) is True for key in required_true)
        and all(boundary.get(key) is False for key in required_false),
        "protocol widens a closed information boundary",
    )
    return value


def _rho_grid(protocol: Mapping[str, Any]) -> np.ndarray:
    grid = protocol["rho_grid"]
    return np.linspace(
        float(grid["minimum"]),
        float(grid["maximum"]),
        int(grid["count"]),
        dtype=np.float64,
    )


def _profile(protocol: Mapping[str, Any]) -> Profile:
    return Profile(
        max_cases=1,
        max_frames=int(protocol["maximum_frames"]),
        max_points=int(protocol["maximum_points"]),
        max_tactile_channels=1,
        max_candidate_archives=1,
    )


def load_episode(
    dataset_root: Path,
    protocol: Mapping[str, Any],
    episode_id: int,
) -> EpisodeData:
    """Load one explicitly permitted official point-cloud archive read-only."""

    source_ids = set(map(int, protocol["source_episode_ids"]))
    _require(episode_id in source_ids, "attempted to load a non-source episode")
    processed = dataset_root / str(protocol["processed_root_suffix"])
    archive = (
        processed
        / str(protocol["object_id"])
        / f"episode_{episode_id}"
        / "pcd_clean.tar"
    )
    _require(archive.is_file() and not archive.is_symlink(), f"archive is missing: {archive}")
    carrier = Carrier(
        kind="pcd_clean_tar",
        object_id=str(protocol["object_id"]),
        path=archive,
    )
    data = load_pcd_tar_sequence(carrier, _profile(protocol), dataset_root)
    _require(data.clouds is not None, "point-cloud archive returned no clouds")
    shapes = {cloud.shape for cloud in data.clouds}
    _require(len(shapes) == 1, "persistent point shape changed within the episode")
    positions = np.stack(data.clouds).astype(np.float64, copy=False)
    _require(
        positions.ndim == 3
        and positions.shape[2] == 3
        and np.all(np.isfinite(positions)),
        "point positions are malformed",
    )
    action = str(protocol["source_episode_actions"][str(episode_id)])
    return EpisodeData(
        episode_id=episode_id,
        action=action,
        positions_m=positions,
        archive_metadata=data.metadata,
    )


def _episode_rho_and_residual(sequence: EpisodeData) -> tuple[float, float]:
    velocity = np.diff(sequence.positions_m, axis=0)
    _require(len(velocity) >= 2, "episode has too few velocity transitions")
    previous = velocity[:-1]
    following = velocity[1:]
    denominator = float(np.sum(previous * previous))
    rho = 0.0 if denominator <= np.finfo(np.float64).tiny else float(
        np.sum(previous * following) / denominator
    )
    residual = following - rho * previous
    return rho, float(np.mean(residual * residual))


def fit_training_prior(
    sequences: Sequence[EpisodeData],
    protocol: Mapping[str, Any],
) -> TrainingPrior:
    """Fit an equal-episode Gaussian grid prior from other opened episodes."""

    _require(len(sequences) >= 2, "at least two training episodes are required")
    estimates = [_episode_rho_and_residual(sequence) for sequence in sequences]
    raw_rhos = np.asarray([item[0] for item in estimates], dtype=np.float64)
    grid = _rho_grid(protocol)
    clipped = np.clip(raw_rhos, grid[0], grid[-1])
    variance = max(
        float(np.var(clipped, ddof=1)),
        float(protocol["prior_variance_floor"]),
    )
    mean = float(np.mean(clipped))
    log_weights = -0.5 * np.square(grid - mean) / variance
    log_weights -= float(np.max(log_weights))
    weights = np.exp(log_weights)
    weights /= float(np.sum(weights))
    residual_variance = max(
        float(np.median([item[1] for item in estimates])),
        np.finfo(np.float64).eps,
    )
    return TrainingPrior(
        rho_grid=grid,
        weights=weights,
        residual_variance_m2=residual_variance,
        training_episode_ids=tuple(sequence.episode_id for sequence in sequences),
        episode_rho_estimates=tuple(float(item) for item in raw_rhos),
    )


def posterior_weights(
    sequence: EpisodeData,
    reset_position: int,
    prior: TrainingPrior,
    protocol: Mapping[str, Any],
) -> np.ndarray:
    """Update the rho grid using only the held episode prefix."""

    prefix = sequence.positions_m[: reset_position + 1]
    velocity = np.diff(prefix, axis=0)
    _require(len(velocity) >= 2, "held prefix has too few velocity transitions")
    previous = velocity[:-1]
    following = velocity[1:]
    residual = following[None] - prior.rho_grid[:, None, None, None] * previous[None]
    mean_square = np.mean(residual * residual, axis=(1, 2, 3))
    effective_count = min(
        len(previous),
        int(protocol["likelihood_information_cap"]),
    )
    log_weights = np.log(np.maximum(prior.weights, np.finfo(np.float64).tiny))
    log_weights -= 0.5 * effective_count * mean_square / prior.residual_variance_m2
    log_weights -= float(np.max(log_weights))
    weights = np.exp(log_weights)
    total = float(np.sum(weights))
    _require(np.isfinite(total) and total > 0.0, "posterior normalization failed")
    return weights / total


def _transport_factor(rho: np.ndarray, horizon: int) -> np.ndarray:
    powers = np.arange(horizon, dtype=np.int64)
    return np.sum(rho[:, None] ** powers[None], axis=1)


def query_class_index(
    rho_grid: np.ndarray,
    protocol: Mapping[str, Any],
) -> np.ndarray:
    """Return the outcome-independent class map for the registered query."""

    factors = _transport_factor(rho_grid, int(protocol["horizon_frames"]))
    thresholds = np.asarray(
        protocol["query"]["transport_factor_thresholds"],
        dtype=np.float64,
    )
    classes = np.searchsorted(thresholds, factors, side="right").astype(np.int64)
    expected = np.arange(len(thresholds) + 1, dtype=np.int64)
    _require(np.array_equal(np.unique(classes), expected), "query classes are not all represented")
    return classes


def _reset_positions(
    sequence: EpisodeData,
    protocol: Mapping[str, Any],
) -> tuple[int, ...]:
    earliest = int(protocol["minimum_prefix_frames"])
    latest = len(sequence.positions_m) - int(protocol["horizon_frames"]) - 1
    count = int(protocol["reset_count"])
    _require(latest >= earliest + count - 1, "episode cannot support the reset roster")
    positions = tuple(
        int(item)
        for item in np.linspace(earliest, latest, count, dtype=np.int64)
    )
    _require(len(set(positions)) == count, "reset positions are not unique")
    return positions


def _observed_transport_factor(
    sequence: EpisodeData,
    reset_position: int,
    protocol: Mapping[str, Any],
) -> float:
    horizon = int(protocol["horizon_frames"])
    current = sequence.positions_m[reset_position]
    velocity = current - sequence.positions_m[reset_position - 1]
    displacement = sequence.positions_m[reset_position + horizon] - current
    denominator = float(np.sum(velocity * velocity))
    _require(
        denominator >= float(protocol["minimum_velocity_energy_m2"]),
        "current persistent-point velocity has insufficient energy",
    )
    return float(np.sum(velocity * displacement) / denominator)


def _classify_factor(value: float, protocol: Mapping[str, Any]) -> int:
    thresholds = np.asarray(
        protocol["query"]["transport_factor_thresholds"],
        dtype=np.float64,
    )
    return int(np.searchsorted(thresholds, value, side="right"))


def _brier(probabilities: np.ndarray, actual_class: int) -> float:
    target = np.zeros_like(probabilities)
    target[actual_class] = 1.0
    return float(np.sum(np.square(probabilities - target)))


def _log_score(probabilities: np.ndarray, actual_class: int) -> float:
    return -math.log(max(float(probabilities[actual_class]), 1e-12))


def _comparison_lift(
    prior: np.ndarray,
    classes: np.ndarray,
    quotient: np.ndarray,
    mode: str,
) -> np.ndarray:
    result = np.zeros_like(prior)
    for class_id, class_mass in enumerate(quotient):
        members = np.flatnonzero(classes == class_id)
        _require(len(members) >= 1, "query class has no members")
        if mode == "uniform":
            result[members] = class_mass / len(members)
        elif mode == "prior_map":
            result[members[int(np.argmax(prior[members]))]] = class_mass
        elif mode == "reverse_prior":
            result[members[int(np.argmin(prior[members]))]] = class_mass
        else:
            raise ValueError(f"unsupported comparison lift: {mode}")
    return result


def _lift_summary(
    name: str,
    weights: np.ndarray,
    prior: np.ndarray,
    classes: np.ndarray,
    rho_grid: np.ndarray,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    information = query_quotient_information_decomposition(prior, weights, classes)
    rho_mean = float(np.sum(weights * rho_grid))
    return {
        "name": name,
        "rho_mean": rho_mean,
        "latent_decision": (
            "persistent"
            if rho_mean >= float(protocol["latent_rho_decision_threshold"])
            else "damped"
        ),
        "total_information_nats": information.total_information_nats,
        "quotient_information_nats": information.quotient_information_nats,
        "unsupported_specificity_nats": information.unsupported_specificity_nats,
        "quotient_weights": information.posterior_quotient_weights.tolist(),
    }


def analyze_reset(
    sequence: EpisodeData,
    reset_position: int,
    prior: TrainingPrior,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate one prefix/held-future query without changing the dataset."""

    classes = query_class_index(prior.rho_grid, protocol)
    full_posterior = posterior_weights(sequence, reset_position, prior, protocol)
    prior_quotient = aggregate_to_query_quotient(prior.weights, classes)
    quotient = aggregate_to_query_quotient(full_posterior, classes)
    jeffrey = minimum_information_query_lift(
        prior.weights,
        classes,
        quotient,
    )
    lifts = [
        _lift_summary(
            "jeffrey_i_projection",
            jeffrey.lifted_weights,
            prior.weights,
            classes,
            prior.rho_grid,
            protocol,
        ),
        _lift_summary(
            "full_prefix_posterior",
            full_posterior,
            prior.weights,
            classes,
            prior.rho_grid,
            protocol,
        ),
    ]
    for mode in ("uniform", "prior_map", "reverse_prior"):
        lifts.append(
            _lift_summary(
                mode,
                _comparison_lift(prior.weights, classes, quotient, mode),
                prior.weights,
                classes,
                prior.rho_grid,
                protocol,
            )
        )
    reference_quotient = np.asarray(lifts[0]["quotient_weights"], dtype=np.float64)
    _require(
        all(
            np.allclose(
                np.asarray(item["quotient_weights"], dtype=np.float64),
                reference_quotient,
                rtol=0.0,
                atol=1e-12,
            )
            for item in lifts
        ),
        "comparison lifts do not preserve the quotient posterior",
    )
    _require(
        abs(lifts[0]["unsupported_specificity_nats"]) <= 1e-10,
        "Jeffrey lift added unsupported specificity",
    )
    observed_factor = _observed_transport_factor(sequence, reset_position, protocol)
    actual_class = _classify_factor(observed_factor, protocol)
    envelope = query_ambiguity_envelope(
        quotient,
        classes,
        prior.rho_grid,
    )
    lower = float(envelope.lower[0])
    upper = float(envelope.upper[0])
    threshold = float(protocol["latent_rho_decision_threshold"])
    actions = {str(item["latent_decision"]) for item in lifts}
    return {
        "episode_id": sequence.episode_id,
        "action": sequence.action,
        "reset_position": reset_position,
        "evaluation_position": reset_position + int(protocol["horizon_frames"]),
        "observed_transport_factor": observed_factor,
        "actual_query_class": actual_class,
        "actual_query_label": protocol["query"]["class_labels"][actual_class],
        "prior_quotient_weights": prior_quotient.tolist(),
        "posterior_quotient_weights": quotient.tolist(),
        "prior_brier_score": _brier(prior_quotient, actual_class),
        "posterior_brier_score": _brier(quotient, actual_class),
        "prior_log_score": _log_score(prior_quotient, actual_class),
        "posterior_log_score": _log_score(quotient, actual_class),
        "posterior_query_prediction": int(np.argmax(quotient)),
        "lifts": lifts,
        "latent_rho_ambiguity_lower": lower,
        "latent_rho_ambiguity_upper": upper,
        "latent_rho_ambiguity_width": upper - lower,
        "latent_decision_threshold": threshold,
        "latent_decision_ambiguous": lower < threshold < upper,
        "complete_lift_decisions_disagree": len(actions) > 1,
    }


def _bootstrap_interval(
    episode_values: np.ndarray,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.integers(
        0,
        len(episode_values),
        size=(repetitions, len(episode_values)),
    )
    means = np.mean(episode_values[samples], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def analyze_sequences(
    sequences: Sequence[EpisodeData],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Run leave-one-source-episode-out query-quotient evaluation."""

    source_ids = tuple(map(int, protocol["source_episode_ids"]))
    by_id = {sequence.episode_id: sequence for sequence in sequences}
    _require(set(by_id) == set(source_ids), "loaded source episode roster changed")
    reset_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for episode_id in source_ids:
        held = by_id[episode_id]
        training = [by_id[item] for item in source_ids if item != episode_id]
        prior = fit_training_prior(training, protocol)
        local = [
            analyze_reset(held, reset, prior, protocol)
            for reset in _reset_positions(held, protocol)
        ]
        reset_rows.extend(local)
        prior_brier = float(np.mean([item["prior_brier_score"] for item in local]))
        posterior_brier = float(
            np.mean([item["posterior_brier_score"] for item in local])
        )
        prior_log = float(np.mean([item["prior_log_score"] for item in local]))
        posterior_log = float(np.mean([item["posterior_log_score"] for item in local]))
        episode_rows.append(
            {
                "episode_id": episode_id,
                "action": held.action,
                "reset_count": len(local),
                "prior_brier_score": prior_brier,
                "posterior_brier_score": posterior_brier,
                "posterior_minus_prior_brier": posterior_brier - prior_brier,
                "prior_log_score": prior_log,
                "posterior_log_score": posterior_log,
                "posterior_minus_prior_log_score": posterior_log - prior_log,
                "posterior_brier_win": posterior_brier < prior_brier,
                "mean_full_posterior_unsupported_specificity_nats": float(
                    np.mean(
                        [
                            next(
                                lift["unsupported_specificity_nats"]
                                for lift in item["lifts"]
                                if lift["name"] == "full_prefix_posterior"
                            )
                            for item in local
                        ]
                    )
                ),
                "latent_decision_ambiguity_fraction": float(
                    np.mean([item["latent_decision_ambiguous"] for item in local])
                ),
                "complete_lift_decision_disagreement_fraction": float(
                    np.mean(
                        [item["complete_lift_decisions_disagree"] for item in local]
                    )
                ),
            }
        )
    brier_deltas = np.asarray(
        [item["posterior_minus_prior_brier"] for item in episode_rows],
        dtype=np.float64,
    )
    log_deltas = np.asarray(
        [item["posterior_minus_prior_log_score"] for item in episode_rows],
        dtype=np.float64,
    )
    analysis = protocol["analysis"]
    brier_interval = _bootstrap_interval(
        brier_deltas,
        int(analysis["bootstrap_repetitions"]),
        int(analysis["bootstrap_seed"]),
    )
    log_interval = _bootstrap_interval(
        log_deltas,
        int(analysis["bootstrap_repetitions"]),
        int(analysis["bootstrap_seed"]) + 1,
    )
    mean_brier_delta = float(np.mean(brier_deltas))
    episode_win_fraction = float(
        np.mean([item["posterior_brier_win"] for item in episode_rows])
    )
    classification = (
        "source-only-real-query-quotient-positive-pilot"
        if mean_brier_delta < 0.0 and episode_win_fraction >= 2.0 / 3.0
        else "source-only-real-query-quotient-nonpositive-pilot"
    )
    return {
        "classification": classification,
        "episode_count": len(episode_rows),
        "reset_count": len(reset_rows),
        "query_class_count": len(protocol["query"]["class_labels"]),
        "episode_brier_win_fraction": episode_win_fraction,
        "mean_prior_brier_score": float(
            np.mean([item["prior_brier_score"] for item in episode_rows])
        ),
        "mean_posterior_brier_score": float(
            np.mean([item["posterior_brier_score"] for item in episode_rows])
        ),
        "mean_posterior_minus_prior_brier": mean_brier_delta,
        "brier_delta_episode_bootstrap_95": list(brier_interval),
        "mean_prior_log_score": float(
            np.mean([item["prior_log_score"] for item in episode_rows])
        ),
        "mean_posterior_log_score": float(
            np.mean([item["posterior_log_score"] for item in episode_rows])
        ),
        "mean_posterior_minus_prior_log_score": float(np.mean(log_deltas)),
        "log_delta_episode_bootstrap_95": list(log_interval),
        "mean_full_posterior_unsupported_specificity_nats": float(
            np.mean(
                [
                    item["mean_full_posterior_unsupported_specificity_nats"]
                    for item in episode_rows
                ]
            )
        ),
        "jeffrey_unsupported_specificity_nats": 0.0,
        "latent_decision_ambiguity_fraction": float(
            np.mean([item["latent_decision_ambiguous"] for item in reset_rows])
        ),
        "complete_lift_decision_disagreement_fraction": float(
            np.mean(
                [item["complete_lift_decisions_disagree"] for item in reset_rows]
            )
        ),
        "episode_records": episode_rows,
        "reset_records": reset_rows,
    }


def validate_result(result: Mapping[str, Any]) -> None:
    """Fail closed if a retained result violates its registered boundary."""

    _require(result.get("artifact_kind") == "Deform360QueryQuotientRealPilotV1", "result kind changed")
    _require(result.get("schema_version") == 1, "result schema changed")
    boundary = result.get("information_boundary")
    _require(isinstance(boundary, Mapping), "result boundary is missing")
    _require(boundary.get("source_only") is True, "result is not source-only")
    for key in (
        "forbidden_episode_payloads_opened",
        "official_velocity_arrays_used",
        "dataset_modified",
        "raw_payload_uploaded",
        "fresh_confirmation_authorized",
        "paper_claim_authorized",
    ):
        _require(boundary.get(key) is False, f"closed result boundary widened: {key}")
    summary = result.get("summary")
    _require(isinstance(summary, Mapping), "result summary is missing")
    _require(summary.get("episode_count") >= 3, "too few source episodes")
    _require(summary.get("reset_count") >= summary["episode_count"], "too few resets")
    _require(
        abs(float(summary.get("jeffrey_unsupported_specificity_nats", math.nan)))
        <= 1e-10,
        "Jeffrey lift added unsupported specificity",
    )
    _require(result.get("paper_claim_authorized") is False, "result self-authorized a paper claim")
    _require(result.get("result_sha256") == _content_id(result), "result digest mismatch")


def _write_reset_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = (
        "episode_id",
        "action",
        "reset_position",
        "evaluation_position",
        "observed_transport_factor",
        "actual_query_class",
        "actual_query_label",
        "prior_brier_score",
        "posterior_brier_score",
        "prior_log_score",
        "posterior_log_score",
        "latent_rho_ambiguity_lower",
        "latent_rho_ambiguity_upper",
        "latent_rho_ambiguity_width",
        "latent_decision_ambiguous",
        "complete_lift_decisions_disagree",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def _report(result: Mapping[str, Any]) -> str:
    summary = result["summary"]
    low, high = summary["brier_delta_episode_bootstrap_95"]
    return "\n".join(
        (
            "## Deform360 source-only query-quotient pilot",
            "",
            f"- Classification: `{summary['classification']}`",
            f"- Public object: `{result['object_id']}`",
            f"- Source episodes: `{summary['episode_count']}`",
            f"- Nested prefix resets: `{summary['reset_count']}`",
            f"- Prior quotient Brier: `{summary['mean_prior_brier_score']:.6f}`",
            f"- Posterior quotient Brier: `{summary['mean_posterior_brier_score']:.6f}`",
            "- Posterior-minus-prior Brier: "
            f"`{summary['mean_posterior_minus_prior_brier']:+.6f}` "
            f"(episode bootstrap 95% `[{low:+.6f}, {high:+.6f}]`)",
            "- Episode Brier win fraction: "
            f"`{summary['episode_brier_win_fraction']:.3f}`",
            "- Mean full-posterior unsupported specificity: "
            f"`{summary['mean_full_posterior_unsupported_specificity_nats']:.6f}` nats",
            "- Jeffrey-lift unsupported specificity: `0.000000` nats",
            "- Latent-decision ambiguity fraction: "
            f"`{summary['latent_decision_ambiguity_fraction']:.3f}`",
            "- Complete-lift decision-disagreement fraction: "
            f"`{summary['complete_lift_decision_disagreement_fraction']:.3f}`",
            "",
            "This is a same-object, leave-one-source-episode-out diagnostic on public ",
            "reconstructed point clouds. It opens no forbidden episode, authorizes no ",
            "fresh confirmation, and is not by itself a paper claim.",
            "",
        )
    )


def run(
    dataset_root: Path,
    protocol_path: Path,
    output_dir: Path,
    revision: str | None,
) -> dict[str, Any]:
    """Execute and retain the compact source-only real-data diagnostic."""

    dataset_root = dataset_root.resolve()
    protocol_path = protocol_path.resolve()
    output_dir = output_dir.resolve()
    _require(dataset_root.is_dir(), "dataset root is missing")
    _require(protocol_path.is_file(), "protocol is missing")
    protocol = load_protocol(protocol_path)
    _require(
        str(dataset_root) == str(protocol["dataset_root"]),
        "dataset root differs from the registered path",
    )
    sequences = [
        load_episode(dataset_root, protocol, episode_id)
        for episode_id in protocol["source_episode_ids"]
    ]
    summary = analyze_sequences(sequences, protocol)
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol_copy = output_dir / "protocol.json"
    write_json(protocol_copy, protocol)
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360QueryQuotientRealPilotV1",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "repository_revision": revision,
        "dataset_root": str(dataset_root),
        "object_id": protocol["object_id"],
        "source_episode_ids": list(protocol["source_episode_ids"]),
        "source_episode_actions": protocol["source_episode_actions"],
        "forbidden_episode_ids": list(protocol["forbidden_episode_ids"]),
        "source_archive_records": [
            {
                "episode_id": sequence.episode_id,
                "action": sequence.action,
                "metadata": sequence.archive_metadata,
            }
            for sequence in sequences
        ],
        "summary": summary,
        "information_boundary": dict(protocol["information_boundary"]),
        "paper_claim_authorized": False,
        "interpretation": (
            "A favorable result supports only source-prefix prediction of a "
            "registered displacement-continuation class on one public object. "
            "It does not establish a correct quotient, unique physical cause, "
            "fresh-object transfer, calibrated physical uncertainty, or safety."
        ),
    }
    result["result_sha256"] = _content_id(result)
    validate_result(result)
    write_json(output_dir / "result.json", result)
    _write_reset_csv(output_dir / "reset_metrics.csv", summary["reset_records"])
    (output_dir / "report.md").write_text(_report(result), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).with_name("protocol.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision", default=os.environ.get("GITHUB_SHA"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(
        args.data_root,
        args.protocol,
        args.output_dir,
        args.revision,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
