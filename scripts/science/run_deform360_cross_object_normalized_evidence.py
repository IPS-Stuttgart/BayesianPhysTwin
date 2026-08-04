#!/usr/bin/env python3
"""Cross-fitted normalized-evidence diagnostic on locked Deform360 hull paths.

The six Deform360 objects used here have all appeared in prior repository work.
This evaluator therefore performs a retrospective leave-one-object-out mechanism
study, not fresh independent validation.  For each target object, evidence-scale
selection uses only the other five objects.  The same object can be a source in
other folds, so there is no globally sealed target cohort.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.endpoint_model_average import infer_model_averaged_endpoint

SCHEMA = "bayesian-phystwin/deform360-cross-object-normalized-evidence-result-v1"
PROTOCOL_SCHEMA = "bayesian-phystwin/deform360-cross-object-normalized-evidence-v1"
CHI_SQUARE_3D_90 = 6.251388631170325
METHODS = (
    "persistence",
    "last_residual",
    "cumulative_model_average",
    "normalized_kappa_1",
    "cross_fitted_normalized",
)
UNCERTAINTY_METHODS = (
    "cumulative_model_average",
    "normalized_kappa_1",
    "cross_fitted_normalized",
)


@dataclass(frozen=True, slots=True)
class EpisodeSpec:
    object_id: str
    relative_path: str
    expected_frame_count: int


@dataclass(frozen=True, slots=True)
class RollingEvent:
    object_id: str
    episode_path: str
    step_index: int
    target_delta_m: np.ndarray
    last_delta_m: np.ndarray
    current_hull_m: np.ndarray
    target_hull_m: np.ndarray
    component_log_evidence: np.ndarray
    component_mean_m: np.ndarray
    component_variance_m2: np.ndarray
    cumulative_weights: np.ndarray
    prior_probability: np.ndarray
    update_count: int


@dataclass(frozen=True, slots=True)
class LoadedEpisode:
    object_id: str
    relative_path: str
    file_sha256: str
    frame_indices: np.ndarray
    centroids_m: np.ndarray
    hulls_m: tuple[np.ndarray, ...]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    """Load and validate the locked non-fresh cross-fitting protocol."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unexpected Deform360 cross-object protocol schema")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported Deform360 cross-object protocol version")
    if payload.get("status") != "retrospective-non-fresh-cross-fitted-diagnostic":
        raise ValueError("protocol must retain its retrospective non-fresh status")
    boundary = payload.get("information_boundary")
    if not isinstance(boundary, Mapping):
        raise ValueError("protocol information boundary is missing")
    required_boundary = {
        "all_objects_previously_used_by_repository": True,
        "array_payloads_opened_when_protocol_locked": False,
        "fold_local_target_exclusion": True,
        "globally_sealed_target_cohort": False,
        "npz_headers_previously_inspected": True,
        "official_deform360_task": False,
    }
    if dict(boundary) != required_boundary:
        raise ValueError("protocol information boundary changed")
    objects = payload.get("objects")
    if not isinstance(objects, list) or len(objects) < 3:
        raise ValueError("protocol requires at least three object groups")
    object_ids = tuple(objects)
    if any(not isinstance(value, str) or not value for value in object_ids):
        raise ValueError("protocol object IDs must be nonempty strings")
    if tuple(sorted(set(object_ids))) != object_ids:
        raise ValueError("protocol object IDs must be unique and sorted")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("protocol requires explicit episode paths")
    allowed_root = str(payload["provenance_locks"]["allowed_relative_root"]).rstrip("/")
    seen_paths: set[str] = set()
    seen_objects: set[str] = set()
    for index, raw in enumerate(episodes):
        if not isinstance(raw, Mapping):
            raise ValueError(f"episodes[{index}] must be an object")
        object_id = raw.get("object_id")
        relative_path = raw.get("path")
        frame_count = raw.get("expected_frame_count")
        if object_id not in object_ids:
            raise ValueError(f"episodes[{index}] uses an undeclared object")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"episodes[{index}].path must be a nonempty string")
        if relative_path in seen_paths:
            raise ValueError("protocol episode paths must be unique")
        seen_paths.add(relative_path)
        seen_objects.add(str(object_id))
        expected_prefix = f"{allowed_root}/{object_id}/"
        if not relative_path.startswith(expected_prefix):
            raise ValueError("episode path escaped its declared object/root")
        if Path(relative_path).name != "sampled_hulls.npz":
            raise ValueError("only locked sampled_hulls.npz paths are supported")
        if isinstance(frame_count, bool) or not isinstance(frame_count, int):
            raise ValueError("expected frame count must be an integer")
        if frame_count < int(payload["minimum_prefix_displacements"]) + 2:
            raise ValueError("episode is too short for the locked rolling prefix")
    if seen_objects != set(object_ids):
        raise ValueError("every declared object must have at least one episode")
    candidates = payload["normalized_evidence"]["kappa_candidates"]
    if not isinstance(candidates, list) or len(candidates) < 3:
        raise ValueError("kappa candidate grid is too small")
    parsed = tuple(float(value) for value in candidates)
    if any(not math.isfinite(value) or value <= 0.0 for value in parsed):
        raise ValueError("kappa candidates must be finite and positive")
    if tuple(sorted(set(parsed))) != parsed or 1.0 not in parsed:
        raise ValueError("kappa candidates must be unique, sorted, and contain one")
    if tuple(payload.get("methods", ())) != METHODS:
        raise ValueError("protocol method ordering changed")
    return payload, _canonical_sha256(payload)


def _episode_specs(protocol: Mapping[str, Any]) -> tuple[EpisodeSpec, ...]:
    return tuple(
        EpisodeSpec(
            object_id=str(raw["object_id"]),
            relative_path=str(raw["path"]),
            expected_frame_count=int(raw["expected_frame_count"]),
        )
        for raw in protocol["episodes"]
    )


def _subsample(points: np.ndarray, limit: int) -> np.ndarray:
    if len(points) <= limit:
        result = np.asarray(points, dtype=np.float64).copy()
    else:
        indices = np.linspace(0, len(points) - 1, limit, dtype=np.int64)
        result = np.asarray(points[indices], dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def load_episode(
    data_root: Path,
    spec: EpisodeSpec,
    *,
    max_points: int,
) -> LoadedEpisode:
    """Load one explicitly locked packed-hull sequence."""

    root = data_root.expanduser().resolve()
    path = (root / spec.relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("episode path escaped the dataset root") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as stored:
        required = {"frame_indices", "point_offsets", "points_world_m"}
        if not required.issubset(stored.files):
            raise ValueError(f"packed hull contract missing in {spec.relative_path}")
        frames = np.asarray(stored["frame_indices"], dtype=np.int64)
        offsets = np.asarray(stored["point_offsets"], dtype=np.int64)
        points = np.asarray(stored["points_world_m"], dtype=np.float64)
    if frames.shape != (spec.expected_frame_count,):
        raise ValueError(f"frame count changed for {spec.relative_path}")
    if offsets.shape != (len(frames) + 1,) or offsets[0] != 0:
        raise ValueError(f"point offsets changed for {spec.relative_path}")
    if offsets[-1] != len(points) or np.any(np.diff(offsets) <= 0):
        raise ValueError(f"packed point offsets are invalid for {spec.relative_path}")
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError(f"packed hull points are invalid for {spec.relative_path}")
    if len(frames) > 1 and np.any(np.diff(frames) <= 0):
        raise ValueError(
            f"frame indices must be strictly increasing in {spec.relative_path}"
        )
    hulls: list[np.ndarray] = []
    centroids = np.empty((len(frames), 3), dtype=np.float64)
    for index in range(len(frames)):
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        hull = points[start:stop]
        centroids[index] = np.mean(hull, axis=0)
        hulls.append(_subsample(hull, max_points))
    frames = np.array(frames, copy=True, order="C")
    centroids.setflags(write=False)
    frames.setflags(write=False)
    return LoadedEpisode(
        object_id=spec.object_id,
        relative_path=spec.relative_path,
        file_sha256=_file_sha256(path),
        frame_indices=frames,
        centroids_m=centroids,
        hulls_m=tuple(hulls),
    )


def _build_events(
    episode: LoadedEpisode,
    *,
    minimum_prefix: int,
) -> tuple[RollingEvent, ...]:
    residual = np.diff(episode.centroids_m, axis=0)
    valid = np.ones((len(residual), 1), dtype=bool)
    values = residual[:, None, :]
    events: list[RollingEvent] = []
    for current_frame in range(minimum_prefix, len(episode.centroids_m) - 1):
        posterior = infer_model_averaged_endpoint(
            values,
            valid,
            end_frame=current_frame,
        )
        component_variance = (
            posterior.component_variance_m2[:, 0]
            + posterior.component_process_variance_m2
        )
        fields = (
            residual[current_frame],
            residual[current_frame - 1],
            posterior.component_log_evidence[0],
            posterior.component_mean_m[:, 0],
            component_variance,
            posterior.component_weights[0],
            np.asarray(posterior.config.component_prior_probability),
        )
        copied = [
            np.array(value, dtype=np.float64, copy=True, order="C") for value in fields
        ]
        for value in copied:
            value.setflags(write=False)
        events.append(
            RollingEvent(
                object_id=episode.object_id,
                episode_path=episode.relative_path,
                step_index=current_frame,
                target_delta_m=copied[0],
                last_delta_m=copied[1],
                current_hull_m=episode.hulls_m[current_frame],
                target_hull_m=episode.hulls_m[current_frame + 1],
                component_log_evidence=copied[2],
                component_mean_m=copied[3],
                component_variance_m2=copied[4],
                cumulative_weights=copied[5],
                prior_probability=copied[6],
                update_count=int(posterior.update_count[0]),
            )
        )
    if not events:
        raise ValueError(f"episode produced no rolling events: {episode.relative_path}")
    return tuple(events)


def _episode_job(
    job: tuple[Path, EpisodeSpec, int, int],
) -> tuple[LoadedEpisode, tuple[RollingEvent, ...]]:
    root, spec, max_points, minimum_prefix = job
    episode = load_episode(root, spec, max_points=max_points)
    return episode, _build_events(episode, minimum_prefix=minimum_prefix)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


def normalized_weights(event: RollingEvent, kappa: float) -> np.ndarray:
    """Return update-count-normalized component weights for one rolling event."""

    if not math.isfinite(kappa) or kappa <= 0.0:
        raise ValueError("kappa must be finite and positive")
    if event.update_count < 1:
        raise ValueError("rolling event must contain at least one update")
    logits = np.log(event.prior_probability) + (
        kappa * event.component_log_evidence / event.update_count
    )
    weights = _softmax(logits)
    weights.setflags(write=False)
    return weights


def _mixture_moments(
    event: RollingEvent,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    normalized = np.asarray(weights, dtype=np.float64)
    if normalized.shape != event.component_log_evidence.shape:
        raise ValueError("component weight shape changed")
    if np.any(normalized < 0.0) or not np.isclose(np.sum(normalized), 1.0):
        raise ValueError("component weights must be normalized")
    mean = np.einsum("k,kc->c", normalized, event.component_mean_m)
    centered = event.component_mean_m - mean[None, :]
    covariance = np.einsum(
        "k,kij->ij",
        normalized,
        event.component_variance_m2[:, None, None] * np.eye(3)
        + centered[:, :, None] * centered[:, None, :],
    )
    covariance = 0.5 * (covariance + covariance.T)
    return mean, covariance


def _gaussian_stats(
    error: np.ndarray, covariance: np.ndarray
) -> tuple[float, float, bool]:
    regularized = np.asarray(covariance, dtype=np.float64) + np.eye(3) * 1e-12
    sign, logdet = np.linalg.slogdet(regularized)
    if sign <= 0:
        raise ValueError("predictive covariance must be positive definite")
    solved = np.linalg.solve(regularized, error)
    nees = float(error @ solved)
    nll = 0.5 * (3.0 * math.log(2.0 * math.pi) + float(logdet) + nees)
    return nll, nees, nees <= CHI_SQUARE_3D_90


def _effective_components(weights: np.ndarray) -> float:
    entropy = -float(np.sum(weights * np.log(np.maximum(weights, 1e-300))))
    return math.exp(entropy)


def _chamfer_rmse(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if len(left) == 0 or len(right) == 0:
        raise ValueError("Chamfer inputs must be nonempty")

    def directed(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        minimum = np.full(len(source), np.inf, dtype=np.float64)
        for start in range(0, len(target), 64):
            block = target[start : start + 64]
            squared = np.sum(
                np.square(source[:, None, :] - block[None, :, :]),
                axis=2,
            )
            minimum = np.minimum(minimum, np.min(squared, axis=1))
        return minimum

    mean_squared = 0.5 * (
        float(np.mean(directed(left, right))) + float(np.mean(directed(right, left)))
    )
    return math.sqrt(max(mean_squared, 0.0))


def _method_prediction(
    event: RollingEvent,
    method: str,
    *,
    selected_kappa: float,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    if method == "persistence":
        return np.zeros(3), None, None
    if method == "last_residual":
        return event.last_delta_m, None, None
    if method == "cumulative_model_average":
        mean, covariance = _mixture_moments(event, event.cumulative_weights)
        return mean, covariance, event.cumulative_weights
    kappa = 1.0 if method == "normalized_kappa_1" else selected_kappa
    weights = normalized_weights(event, kappa)
    mean, covariance = _mixture_moments(event, weights)
    return mean, covariance, weights


def _object_selection_score(
    events: Sequence[RollingEvent],
    kappa: float,
) -> tuple[float, float]:
    nll: list[float] = []
    squared_error: list[float] = []
    for event in events:
        weights = normalized_weights(event, kappa)
        mean, covariance = _mixture_moments(event, weights)
        error = event.target_delta_m - mean
        event_nll, _, _ = _gaussian_stats(error, covariance)
        nll.append(event_nll)
        squared_error.append(float(error @ error))
    return float(np.mean(nll)), math.sqrt(float(np.mean(squared_error)))


def select_kappa(
    events_by_object: Mapping[str, Sequence[RollingEvent]],
    *,
    target_object: str,
    candidates: Sequence[float],
) -> dict[str, Any]:
    """Select kappa using equal source-object weighting, excluding target."""

    source_objects = tuple(sorted(set(events_by_object) - {target_object}))
    if len(source_objects) < 2:
        raise ValueError("cross-fitted selection requires at least two source objects")
    table: list[dict[str, Any]] = []
    for kappa in candidates:
        object_scores = {
            object_id: _object_selection_score(
                events_by_object[object_id], float(kappa)
            )
            for object_id in source_objects
        }
        mean_nll = float(np.mean([score[0] for score in object_scores.values()]))
        mean_rmse = float(np.mean([score[1] for score in object_scores.values()]))
        table.append(
            {
                "kappa": float(kappa),
                "equal_source_object_mean_nll": mean_nll,
                "equal_source_object_mean_centroid_rmse_m": mean_rmse,
                "per_source_object": {
                    object_id: {
                        "mean_nll": object_scores[object_id][0],
                        "centroid_rmse_m": object_scores[object_id][1],
                    }
                    for object_id in source_objects
                },
            }
        )
    selected = min(
        table,
        key=lambda record: (
            record["equal_source_object_mean_nll"],
            record["equal_source_object_mean_centroid_rmse_m"],
            abs(math.log2(record["kappa"])),
            record["kappa"],
        ),
    )
    selected_kappa = float(selected["kappa"])
    return {
        "target_object": target_object,
        "source_objects": list(source_objects),
        "selected_kappa": selected_kappa,
        "selected_at_grid_boundary": selected_kappa
        in {float(candidates[0]), float(candidates[-1])},
        "candidate_table": table,
    }


def _aggregate_target_object(
    events: Sequence[RollingEvent],
    *,
    selected_kappa: float,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    by_episode: dict[str, list[RollingEvent]] = defaultdict(list)
    for event in events:
        by_episode[event.episode_path].append(event)
    episode_results: dict[str, dict[str, Any]] = {}
    for episode_path, episode_events in sorted(by_episode.items()):
        method_results: dict[str, Any] = {}
        for method in METHODS:
            squared_error: list[float] = []
            chamfer: list[float] = []
            nll: list[float] = []
            coverage: list[bool] = []
            effective: list[float] = []
            for event in episode_events:
                mean, covariance, weights = _method_prediction(
                    event,
                    method,
                    selected_kappa=selected_kappa,
                )
                error = event.target_delta_m - mean
                squared_error.append(float(error @ error))
                chamfer.append(
                    _chamfer_rmse(event.current_hull_m + mean, event.target_hull_m)
                )
                if covariance is not None and weights is not None:
                    event_nll, _, covered = _gaussian_stats(error, covariance)
                    nll.append(event_nll)
                    coverage.append(covered)
                    effective.append(_effective_components(weights))
            result: dict[str, Any] = {
                "event_count": len(episode_events),
                "centroid_rmse_m": math.sqrt(float(np.mean(squared_error))),
                "translated_hull_chamfer_rmse_m": float(np.mean(chamfer)),
            }
            if nll:
                result.update(
                    {
                        "mean_gaussian_nll": float(np.mean(nll)),
                        "raw_coverage_90": float(np.mean(coverage)),
                        "effective_component_count": float(np.mean(effective)),
                    }
                )
            method_results[method] = result
        episode_results[episode_path] = method_results

    object_methods: dict[str, Any] = {}
    for method in METHODS:
        records = [episode_results[path][method] for path in sorted(episode_results)]
        result = {
            "episode_count": len(records),
            "event_count": int(sum(record["event_count"] for record in records)),
            "centroid_rmse_m": float(
                np.mean([record["centroid_rmse_m"] for record in records])
            ),
            "translated_hull_chamfer_rmse_m": float(
                np.mean(
                    [record["translated_hull_chamfer_rmse_m"] for record in records]
                )
            ),
        }
        if method in UNCERTAINTY_METHODS:
            result.update(
                {
                    "mean_gaussian_nll": float(
                        np.mean([record["mean_gaussian_nll"] for record in records])
                    ),
                    "raw_coverage_90": float(
                        np.mean([record["raw_coverage_90"] for record in records])
                    ),
                    "effective_component_count": float(
                        np.mean(
                            [record["effective_component_count"] for record in records]
                        )
                    ),
                }
            )
        object_methods[method] = result
    return object_methods, episode_results


def _paired_bootstrap(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(candidate, dtype=np.float64)
    baseline = np.asarray(reference, dtype=np.float64)
    if values.shape != baseline.shape or values.ndim != 1 or len(values) < 2:
        raise ValueError("paired object bootstrap requires aligned nontrivial vectors")
    delta = values - baseline
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(samples, len(delta)))
    boot = np.mean(delta[indices], axis=1)
    return {
        "object_count": len(delta),
        "mean_delta_m": float(np.mean(delta)),
        "median_delta_m": float(np.median(delta)),
        "lower_95_delta_m": float(np.quantile(boot, 0.025)),
        "upper_95_delta_m": float(np.quantile(boot, 0.975)),
        "bootstrap_probability_mean_improvement": float(np.mean(boot < 0.0)),
        "candidate_win_count": int(np.sum(delta < 0.0)),
        "tie_count": int(np.sum(delta == 0.0)),
    }


def _write_csv(path: Path, folds: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "target_object",
                "selected_kappa",
                "selected_at_grid_boundary",
                "method",
                "centroid_rmse_m",
                "translated_hull_chamfer_rmse_m",
                "raw_coverage_90",
                "mean_gaussian_nll",
                "effective_component_count",
            ]
        )
        for fold in folds:
            for method in METHODS:
                record = fold["target_metrics"][method]
                writer.writerow(
                    [
                        fold["target_object"],
                        fold["selection"]["selected_kappa"],
                        fold["selection"]["selected_at_grid_boundary"],
                        method,
                        record["centroid_rmse_m"],
                        record["translated_hull_chamfer_rmse_m"],
                        record.get("raw_coverage_90"),
                        record.get("mean_gaussian_nll"),
                        record.get("effective_component_count"),
                    ]
                )


def run_experiment(
    data_root: Path,
    protocol_path: Path,
    output_dir: Path,
    *,
    workers: int,
    bootstrap_samples: int | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(protocol_path)
    if isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    root = data_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    max_points = int(protocol["max_points_per_hull"])
    minimum_prefix = int(protocol["minimum_prefix_displacements"])
    jobs = [
        (root, spec, max_points, minimum_prefix) for spec in _episode_specs(protocol)
    ]
    if workers == 1:
        loaded = list(map(_episode_job, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            loaded = list(executor.map(_episode_job, jobs))
    events_by_object: dict[str, list[RollingEvent]] = defaultdict(list)
    input_archives: list[dict[str, Any]] = []
    for episode, events in loaded:
        events_by_object[episode.object_id].extend(events)
        input_archives.append(
            {
                "object_id": episode.object_id,
                "path": episode.relative_path,
                "sha256": episode.file_sha256,
                "frame_count": len(episode.frame_indices),
                "event_count": len(events),
            }
        )
    expected_objects = set(map(str, protocol["objects"]))
    if set(events_by_object) != expected_objects:
        raise ValueError("loaded object set disagrees with the locked protocol")
    candidates = tuple(
        float(value) for value in protocol["normalized_evidence"]["kappa_candidates"]
    )
    folds: list[dict[str, Any]] = []
    for target_object in sorted(events_by_object):
        selection = select_kappa(
            events_by_object,
            target_object=target_object,
            candidates=candidates,
        )
        target_metrics, episode_metrics = _aggregate_target_object(
            events_by_object[target_object],
            selected_kappa=float(selection["selected_kappa"]),
        )
        folds.append(
            {
                "target_object": target_object,
                "selection": selection,
                "target_metrics": target_metrics,
                "target_episode_metrics": episode_metrics,
            }
        )

    aggregate_methods: dict[str, Any] = {}
    for method in METHODS:
        records = [fold["target_metrics"][method] for fold in folds]
        result: dict[str, Any] = {
            "object_count": len(records),
            "centroid_rmse_m": float(
                np.mean([record["centroid_rmse_m"] for record in records])
            ),
            "translated_hull_chamfer_rmse_m": float(
                np.mean(
                    [record["translated_hull_chamfer_rmse_m"] for record in records]
                )
            ),
        }
        if method in UNCERTAINTY_METHODS:
            result.update(
                {
                    "raw_coverage_90": float(
                        np.mean([record["raw_coverage_90"] for record in records])
                    ),
                    "mean_gaussian_nll": float(
                        np.mean([record["mean_gaussian_nll"] for record in records])
                    ),
                    "effective_component_count": float(
                        np.mean(
                            [record["effective_component_count"] for record in records]
                        )
                    ),
                }
            )
        aggregate_methods[method] = result

    sample_count = (
        int(protocol["bootstrap"]["samples"])
        if bootstrap_samples is None
        else int(bootstrap_samples)
    )
    seed = int(protocol["bootstrap"]["seed"])
    comparisons: dict[str, Any] = {}
    for metric in ("centroid_rmse_m", "translated_hull_chamfer_rmse_m"):
        candidate = np.asarray(
            [
                fold["target_metrics"]["cross_fitted_normalized"][metric]
                for fold in folds
            ]
        )
        reference = np.asarray(
            [fold["target_metrics"]["last_residual"][metric] for fold in folds]
        )
        comparisons[metric] = _paired_bootstrap(
            candidate,
            reference,
            samples=sample_count,
            seed=seed,
        )

    selected_kappas = [float(fold["selection"]["selected_kappa"]) for fold in folds]
    cumulative = aggregate_methods["cumulative_model_average"]
    selected = aggregate_methods["cross_fitted_normalized"]
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "repository_revision": revision,
        "protocol_sha256": protocol_sha,
        "claim_boundary": protocol["claim_boundary"],
        "information_boundary": protocol["information_boundary"],
        "input_archives": sorted(input_archives, key=lambda record: record["path"]),
        "folds": folds,
        "aggregate": {
            "methods": aggregate_methods,
            "cross_fitted_normalized_vs_last_residual": comparisons,
            "selected_kappas": selected_kappas,
            "boundary_selection_count": int(
                sum(fold["selection"]["selected_at_grid_boundary"] for fold in folds)
            ),
        },
        "diagnosis": {
            "normalized_coverage_closer_to_90_than_cumulative": abs(
                selected["raw_coverage_90"] - 0.9
            )
            < abs(cumulative["raw_coverage_90"] - 0.9),
            "normalized_effective_component_ratio": selected[
                "effective_component_count"
            ]
            / cumulative["effective_component_count"],
            "all_fold_selections_interior": all(
                not fold["selection"]["selected_at_grid_boundary"] for fold in folds
            ),
            "no_automatic_claim_promotion": True,
        },
    }
    result["result_sha256"] = _canonical_sha256(result)

    output = output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    readout_path = output / "readout.json"
    csv_path = output / "per_object.csv"
    _write_json(summary_path, result)
    readout = {
        "schema": "bayesian-phystwin/deform360-cross-object-readout-v1",
        "schema_version": 1,
        "repository_revision": revision,
        "protocol_sha256": protocol_sha,
        "result_sha256": result["result_sha256"],
        "claim_boundary": result["claim_boundary"],
        "aggregate": result["aggregate"],
        "diagnosis": result["diagnosis"],
    }
    _write_json(readout_path, readout)
    _write_csv(csv_path, folds)
    manifest = {
        "schema": "bayesian-phystwin/deform360-cross-object-artifact-manifest-v1",
        "schema_version": 1,
        "protocol": {
            "path": str(protocol_path.resolve()),
            "canonical_sha256": protocol_sha,
            "file_sha256": _file_sha256(protocol_path),
        },
        "inputs": result["input_archives"],
        "outputs": {
            "summary.json": _file_sha256(summary_path),
            "readout.json": _file_sha256(readout_path),
            "per_object.csv": _file_sha256(csv_path),
        },
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    _write_json(output / "artifact_manifest.json", manifest)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--bootstrap-samples", type=int)
    parser.add_argument("--revision", default=os.environ.get("GITHUB_SHA"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = run_experiment(
        args.data_root,
        args.protocol,
        args.output_dir,
        workers=args.workers,
        bootstrap_samples=args.bootstrap_samples,
        revision=args.revision,
    )
    compact = {
        "protocol_sha256": result["protocol_sha256"],
        "result_sha256": result["result_sha256"],
        "selected_kappas": result["aggregate"]["selected_kappas"],
        "boundary_selection_count": result["aggregate"]["boundary_selection_count"],
        "methods": result["aggregate"]["methods"],
        "comparisons": result["aggregate"]["cross_fitted_normalized_vs_last_residual"],
        "diagnosis": result["diagnosis"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
