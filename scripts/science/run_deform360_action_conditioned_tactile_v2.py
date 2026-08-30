#!/usr/bin/env python3
"""Action-conditioned multistep tactile forecasting on public Deform360.

For each registered development object, the highest available episode ID is
selected from metadata and file identities before any target tactile payload is
opened. The remaining episodes train a same-object action-conditioned residual
model. Known future robot trajectories are intervention inputs; future tactile
measurements are used only for scoring.

This is retrospective development evidence, not fresh confirmation or a paper
claim. Reserved objects, camera pixels, geometry, and point clouds are not read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

SCHEMA = "bayesian-phystwin/deform360-action-conditioned-tactile-result-v2"
METHODS = (
    "persistence",
    "last_trend",
    "state_ridge",
    "action_ridge",
    "bayesian_action_ensemble",
    "shuffled_action_control",
    "guarded_action_ensemble",
)
TACTILE_RE = re.compile(r"tactile", re.IGNORECASE)


class EvaluationError(RuntimeError):
    """Raised when the frozen development protocol cannot be executed."""


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvaluationError(f"expected JSON object: {path}")
    return value


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def episode_records(metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = metadata.get("sequences", metadata.get("episodes", metadata.get("takes")))
    if isinstance(raw, Mapping):
        items = sorted(
            raw.items(),
            key=lambda item: (
                0 if str(item[0]).isdigit() else 1,
                int(item[0]) if str(item[0]).isdigit() else str(item[0]),
            ),
        )
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        items = list(enumerate(raw))
    else:
        return []
    result: list[dict[str, Any]] = []
    for raw_id, record in items:
        if not isinstance(record, Mapping):
            continue
        action = record.get("action")
        result.append(
            {
                "episode_id": int(raw_id) if str(raw_id).isdigit() else len(result),
                "action": action.strip() if isinstance(action, str) else None,
                "bimanual_metadata": record.get("bimanual"),
                "nonprehensile": record.get("nonprehensile"),
            }
        )
    return result


def action_family(action: str | None) -> str:
    token = "" if action is None else action.strip().lower()
    if any(word in token for word in ("lift", "raise")):
        return "lift"
    if any(word in token for word in ("move", "drag", "push", "pull")):
        return "translate"
    if any(word in token for word in ("fold", "curl", "curve", "twist")):
        return "shape"
    if any(word in token for word in ("squeeze", "press", "compress")):
        return "compress"
    if any(word in token for word in ("wave", "shake")):
        return "dynamic"
    return "other"


def action_one_hot(action: str | None) -> np.ndarray:
    vocabulary = ("lift", "translate", "shape", "compress", "dynamic", "other")
    result = np.zeros(len(vocabulary), dtype=np.float64)
    result[vocabulary.index(action_family(action))] = 1.0
    return result


@dataclass(frozen=True)
class EpisodeDescriptor:
    object_id: str
    episode_id: int
    action: str | None
    robot_path: Path
    tactile_paths: tuple[Path, ...]
    median_paths: tuple[Path | None, ...]


@dataclass
class EpisodeData:
    descriptor: EpisodeDescriptor
    tactile: np.ndarray
    robot_actions: np.ndarray
    bimanual: bool
    fingerprints: dict[str, Any]


@dataclass(frozen=True)
class Transform:
    feature_scale: np.ndarray
    state_mean: np.ndarray
    state_basis: np.ndarray
    delta_mean: np.ndarray
    delta_basis: np.ndarray


@dataclass(frozen=True)
class RidgeModel:
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    coefficients: np.ndarray
    alpha: float


@dataclass
class Candidate:
    name: str
    alpha: float
    variant: str
    cv_mse: float
    cv_objective: float
    per_episode_active_rmse: dict[int, float]
    cv_predictions: dict[int, np.ndarray]
    cv_truths: dict[int, np.ndarray]
    cv_currents: dict[int, np.ndarray]
    cv_masks: dict[int, np.ndarray]


@dataclass(frozen=True)
class CovarianceModel:
    mean_error: np.ndarray
    diagonal: np.ndarray
    factor: np.ndarray
    multiplier: float
    marginal_z: float
    source_marginal_coverage: float
    source_joint_nanees: float


def sampled_fingerprint(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as stream:
        digest.update(stream.read(1024 * 1024))
        if size > 1024 * 1024:
            stream.seek(max(size - 1024 * 1024, 0))
            digest.update(stream.read(1024 * 1024))
    return {
        "path": str(path),
        "size_bytes": int(size),
        "sampled_sha256": digest.hexdigest(),
        "rule": "sha256(size || first_1MiB || last_1MiB)",
    }


def median_path_for(path: Path) -> Path | None:
    stamp = path.stem.rsplit("_", 1)[-1]
    exact = path.parent / f"median_{stamp}.npy"
    if exact.is_file():
        return exact
    candidates = sorted(path.parent.glob("median_*.npy"), key=lambda item: item.name)
    return candidates[0] if len(candidates) == 1 else None


def episode_directory(processed_object: Path, episode_id: int) -> Path | None:
    for name in (
        f"episode_{episode_id}",
        f"episode_{episode_id:04d}",
        f"episode-{episode_id}",
    ):
        candidate = processed_object / name
        if candidate.is_dir():
            return candidate
    return None


def discover_object(
    root: Path, object_id: str, minimum_episodes: int
) -> list[EpisodeDescriptor]:
    raw_object = root / "raw-repository" / "raw" / object_id
    processed_object = root / "processed-repository" / "processed" / object_id
    metadata_path = raw_object / "metadata.json"
    if not metadata_path.is_file():
        return []
    episodes = episode_records(read_json(metadata_path))
    tactile_groups: list[list[Path]] = []
    for directory in sorted(
        (path for path in raw_object.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    ):
        if not TACTILE_RE.search(directory.name):
            continue
        files = sorted(
            (
                path
                for path in directory.glob("*.npy")
                if not path.name.lower().startswith("median_")
                and path.stat().st_size > 0
            ),
            key=lambda path: path.name,
        )
        if len(files) == len(episodes):
            tactile_groups.append(files)
    if len(tactile_groups) < 2:
        return []
    descriptors: list[EpisodeDescriptor] = []
    for episode in episodes:
        episode_id = int(episode["episode_id"])
        directory = episode_directory(processed_object, episode_id)
        if directory is None:
            continue
        robot_candidates = [
            directory / "robot" / "robot.npy",
            directory / "robot" / "robot.npz",
        ]
        robot_path = next((path for path in robot_candidates if path.is_file()), None)
        if robot_path is None:
            continue
        tactile_paths = tuple(group[episode_id] for group in tactile_groups)
        descriptors.append(
            EpisodeDescriptor(
                object_id=object_id,
                episode_id=episode_id,
                action=episode["action"],
                robot_path=robot_path,
                tactile_paths=tactile_paths,
                median_paths=tuple(median_path_for(path) for path in tactile_paths),
            )
        )
    return descriptors if len(descriptors) >= minimum_episodes else []


def load_robot(path: Path) -> tuple[np.ndarray, bool]:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            actions = np.asarray(archive["actions"], dtype=np.float64)
            raw_bimanual = np.asarray(archive["bimanual"])
            bimanual = bool(raw_bimanual.item())
    else:
        loaded = np.load(path, allow_pickle=True)
        if loaded.shape != () or loaded.dtype != np.dtype(object):
            raise EvaluationError("legacy robot.npy is not a scalar object mapping")
        payload = loaded.item()
        if not isinstance(payload, Mapping):
            raise EvaluationError("legacy robot.npy does not contain a mapping")
        actions = np.asarray(payload["actions"], dtype=np.float64)
        bimanual = bool(payload.get("bimanual", actions.ndim == 4))
    expected_tail = (2, 5, 3) if bimanual else (5, 3)
    if actions.ndim != len(expected_tail) + 1 or actions.shape[1:] != expected_tail:
        raise EvaluationError(f"unexpected robot action shape {actions.shape}")
    if not np.all(np.isfinite(actions)) or len(actions) < 32:
        raise EvaluationError("invalid or too-short robot action carrier")
    if bimanual:
        padded = actions
    else:
        padded = np.zeros((len(actions), 2, 5, 3), dtype=np.float64)
        padded[:, 0] = actions
    return padded, bimanual


def load_raw_tactile(path: Path) -> np.ndarray:
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        array = np.asarray(value)
        if array.ndim != 3 or array.shape[1:] != (16, 32):
            raise ValueError
        result = np.asarray(array, dtype=np.float64)
    except (OSError, ValueError) as error:
        frame_size = 16 * 32
        flat = np.fromfile(path, dtype=np.float32)
        if flat.size == 0 or flat.size % frame_size:
            raise EvaluationError(f"unsupported tactile carrier: {path}") from error
        result = np.asarray(flat.reshape(-1, 16, 32), dtype=np.float64)
    if not np.all(np.isfinite(result)) or len(result) < 32:
        raise EvaluationError(f"invalid tactile carrier: {path}")
    return result


def load_median(path: Path | None) -> np.ndarray:
    if path is None:
        return np.zeros((16, 32), dtype=np.float64)
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        value = np.fromfile(path, dtype=np.float32)
    result = np.asarray(value, dtype=np.float64)
    if result.size != 16 * 32:
        raise EvaluationError(f"invalid tactile median: {path}")
    result = result.reshape(16, 32)
    if not np.all(np.isfinite(result)):
        raise EvaluationError(f"nonfinite tactile median: {path}")
    return result


def pool_tactile(grid: np.ndarray) -> np.ndarray:
    selected = np.maximum(grid[:, :12, :], 0.0).copy()
    selected[:, :, -1] = 0.0
    pooled = selected.reshape(len(selected), 6, 2, 16, 2).mean(axis=(2, 4))
    return pooled.reshape(len(selected), -1)


def load_episode(descriptor: EpisodeDescriptor) -> EpisodeData:
    robot, bimanual = load_robot(descriptor.robot_path)
    tactile_fields = []
    fingerprints = [sampled_fingerprint(descriptor.robot_path)]
    for path, median_path in zip(
        descriptor.tactile_paths, descriptor.median_paths, strict=True
    ):
        raw = load_raw_tactile(path)
        median = load_median(median_path)
        tactile_fields.append(pool_tactile(raw - median[None, :, :]))
        fingerprints.append(sampled_fingerprint(path))
        if median_path is not None:
            fingerprints.append(sampled_fingerprint(median_path))
    tactile_length = min(len(field) for field in tactile_fields)
    tactile = np.concatenate(
        [field[:tactile_length] for field in tactile_fields], axis=1
    )
    robot_indices = np.rint(
        np.linspace(0, len(robot) - 1, tactile_length, dtype=np.float64)
    ).astype(int)
    robot_aligned = robot[robot_indices]
    if not np.all(np.isfinite(tactile)):
        raise EvaluationError("tactile preprocessing produced nonfinite values")
    return EpisodeData(
        descriptor=descriptor,
        tactile=tactile,
        robot_actions=robot_aligned,
        bimanual=bimanual,
        fingerprints={"files": fingerprints, "file_count": len(fingerprints)},
    )


def feature_scale(source: Sequence[EpisodeData], quantile: float) -> np.ndarray:
    frames = np.concatenate([episode.tactile for episode in source], axis=0)
    scale = np.quantile(frames, quantile, axis=0)
    positive = scale[scale > 0.0]
    global_scale = float(np.median(positive)) if len(positive) else 1.0
    scale = np.maximum(scale, max(global_scale * 0.02, 1e-9))
    return scale


def normalize_tactile(values: np.ndarray, scale: np.ndarray, clip: float) -> np.ndarray:
    result = values / scale[None, :]
    return np.clip(result, 0.0, clip)


def truncated_basis(matrix: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=np.float64)
    mean = values.mean(axis=0)
    centered = values - mean
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    retained = right[: min(rank, len(right))]
    return mean, retained


def action_features(
    robot: np.ndarray, starts: np.ndarray, horizon: int, action: str | None
) -> np.ndarray:
    rows = []
    static = action_one_hot(action)
    for start in starts:
        current = robot[start]
        segment = robot[start : start + horizon + 1]
        future = segment[-1]
        final_delta = (future - current).reshape(-1)
        mean_delta = (segment.mean(axis=0) - current).reshape(-1)
        translation = segment[:, :, 0, :]
        translation_step = np.linalg.norm(np.diff(translation, axis=0), axis=2)
        path_length = translation_step.sum(axis=0)
        max_speed = translation_step.max(axis=0)
        opening = segment[:, :, 4, 0]
        opening_path = np.abs(np.diff(opening, axis=0)).sum(axis=0)
        active_gripper = (
            np.max(np.abs(segment - segment[0:1]), axis=(0, 2, 3)) > 1e-10
        ).astype(np.float64)
        rows.append(
            np.concatenate(
                (
                    final_delta,
                    mean_delta,
                    path_length,
                    max_speed,
                    opening_path,
                    active_gripper,
                    static,
                )
            )
        )
    return np.asarray(rows, dtype=np.float64)


def simple_tactile_features(values: np.ndarray, sensor_count: int) -> np.ndarray:
    per_sensor = values.reshape(len(values), sensor_count, -1)
    return np.concatenate(
        (
            per_sensor.mean(axis=2),
            np.mean(per_sensor > 0.05, axis=2),
            per_sensor.max(axis=2),
        ),
        axis=1,
    )


def starts_for(length: int, horizon: int, lag: int, stride: int) -> np.ndarray:
    return np.arange(lag, length - horizon, stride, dtype=int)


def build_transform(
    source: Sequence[EpisodeData], protocol: Mapping[str, Any], horizon: int
) -> Transform:
    model = protocol["model"]
    scale = feature_scale(source, float(model["source_feature_scale_quantile"]))
    clip = float(model["normalized_feature_clip"])
    lag = int(model["trend_lag_frames"])
    stride = int(model["window_stride_frames"])
    states = []
    deltas = []
    for episode in source:
        normalized = normalize_tactile(episode.tactile, scale, clip)
        starts = starts_for(len(normalized), horizon, lag, stride)
        if len(starts) == 0:
            continue
        states.append(normalized[starts])
        deltas.append(normalized[starts + horizon] - normalized[starts])
    if not states:
        raise EvaluationError("source episodes provide no training windows")
    state_mean, state_basis = truncated_basis(
        np.concatenate(states), int(model["state_rank"])
    )
    delta_mean, delta_basis = truncated_basis(
        np.concatenate(deltas), int(model["delta_rank"])
    )
    return Transform(scale, state_mean, state_basis, delta_mean, delta_basis)


def design_for_episode(
    episode: EpisodeData,
    transform: Transform,
    protocol: Mapping[str, Any],
    horizon: int,
    variant: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model = protocol["model"]
    lag = int(model["trend_lag_frames"])
    stride = int(model["window_stride_frames"])
    clip = float(model["normalized_feature_clip"])
    values = normalize_tactile(episode.tactile, transform.feature_scale, clip)
    starts = starts_for(len(values), horizon, lag, stride)
    if len(starts) == 0:
        raise EvaluationError(
            f"episode {episode.descriptor.episode_id} has no windows for horizon {horizon}"
        )
    current = values[starts]
    truth = values[starts + horizon]
    state = (current - transform.state_mean) @ transform.state_basis.T
    trends = []
    for trend_lag in model["input_trend_lags_frames"]:
        trend_lag = int(trend_lag)
        previous_indices = np.maximum(starts - trend_lag, 0)
        trend = (current - values[previous_indices]) @ transform.state_basis.T
        trends.append(trend / np.maximum(starts - previous_indices, 1)[:, None])
    tactile_summary = simple_tactile_features(
        current, len(episode.descriptor.tactile_paths)
    )
    state_input = np.concatenate((state, *trends, tactile_summary), axis=1)
    action = action_features(
        episode.robot_actions, starts, horizon, episode.descriptor.action
    )
    if variant == "state":
        design = state_input
    elif variant == "action":
        design = np.concatenate((state_input, action), axis=1)
    else:
        raise EvaluationError(f"unknown design variant: {variant}")
    delta = truth - current
    target = (delta - transform.delta_mean) @ transform.delta_basis.T
    active = (truth > float(model["active_threshold"])) | (
        current > float(model["active_threshold"])
    )
    return design, target, current, truth, active


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> RidgeModel:
    x_mean = x.mean(axis=0)
    x_scale = np.maximum(x.std(axis=0), 1e-6)
    y_mean = y.mean(axis=0)
    standardized_x = (x - x_mean) / x_scale
    centered_y = y - y_mean
    gram = standardized_x.T @ standardized_x
    regularizer = alpha * np.eye(gram.shape[0], dtype=np.float64)
    coefficients = np.linalg.solve(gram + regularizer, standardized_x.T @ centered_y)
    return RidgeModel(x_mean, x_scale, y_mean, coefficients, float(alpha))


def predict_ridge(model: RidgeModel, x: np.ndarray) -> np.ndarray:
    return model.y_mean + ((x - model.x_mean) / model.x_scale) @ model.coefficients


def decode_prediction(
    latent: np.ndarray,
    current: np.ndarray,
    transform: Transform,
    clip: float,
) -> np.ndarray:
    delta = transform.delta_mean + latent @ transform.delta_basis
    return np.clip(current + delta, 0.0, clip)


def rmse(
    prediction: np.ndarray, truth: np.ndarray, active: np.ndarray
) -> tuple[float, float]:
    error = prediction - truth
    all_value = float(np.sqrt(np.mean(error * error)))
    if np.any(active):
        active_value = float(np.sqrt(np.mean(np.square(error[active]))))
    else:
        active_value = all_value
    return all_value, active_value


def baseline_prediction(
    values: np.ndarray,
    starts: np.ndarray,
    horizon: int,
    lag: int,
    clip: float,
    method: str,
) -> np.ndarray:
    current = values[starts]
    if method == "persistence":
        return current.copy()
    if method == "last_trend":
        previous = values[starts - lag]
        velocity = (current - previous) / float(lag)
        return np.clip(current + horizon * velocity, 0.0, clip)
    raise EvaluationError(f"unknown baseline: {method}")


def candidate_cv(
    source: Sequence[EpisodeData],
    transform: Transform,
    protocol: Mapping[str, Any],
    horizon: int,
) -> list[Candidate]:
    model = protocol["model"]
    clip = float(model["normalized_feature_clip"])
    designs: dict[str, list[tuple[np.ndarray, ...]]] = {"state": [], "action": []}
    for episode in source:
        for variant in designs:
            designs[variant].append(
                design_for_episode(episode, transform, protocol, horizon, variant)
            )
    candidates: list[Candidate] = []
    for variant in ("state", "action"):
        for alpha in model["ridge_alphas"]:
            alpha = float(alpha)
            predictions: dict[int, np.ndarray] = {}
            truths: dict[int, np.ndarray] = {}
            currents: dict[int, np.ndarray] = {}
            masks: dict[int, np.ndarray] = {}
            per_episode: dict[int, float] = {}
            squared = []
            active_squared = []
            for held, episode in enumerate(source):
                train_x = np.concatenate(
                    [
                        row[0]
                        for index, row in enumerate(designs[variant])
                        if index != held
                    ]
                )
                train_y = np.concatenate(
                    [
                        row[1]
                        for index, row in enumerate(designs[variant])
                        if index != held
                    ]
                )
                model_fit = fit_ridge(train_x, train_y, alpha)
                x, _, current, truth, active = designs[variant][held]
                latent = predict_ridge(model_fit, x)
                prediction = decode_prediction(latent, current, transform, clip)
                all_rmse, active_rmse = rmse(prediction, truth, active)
                episode_id = episode.descriptor.episode_id
                predictions[episode_id] = prediction
                truths[episode_id] = truth
                currents[episode_id] = current
                masks[episode_id] = active
                per_episode[episode_id] = active_rmse
                squared.append(all_rmse * all_rmse)
                active_squared.append(active_rmse * active_rmse)
            candidates.append(
                Candidate(
                    name=f"{variant}_ridge_alpha_{alpha:g}",
                    alpha=alpha,
                    variant=variant,
                    cv_mse=float(np.mean(squared)),
                    cv_objective=float(np.mean(active_squared)),
                    per_episode_active_rmse=per_episode,
                    cv_predictions=predictions,
                    cv_truths=truths,
                    cv_currents=currents,
                    cv_masks=masks,
                )
            )
    return candidates


def candidate_weights(
    candidates: Sequence[Candidate], temperature_floor: float
) -> tuple[np.ndarray, float]:
    losses = np.asarray([candidate.cv_objective for candidate in candidates])
    minimum = float(np.min(losses))
    spread = float(np.median(np.abs(losses - np.median(losses))))
    temperature = max(spread, minimum * temperature_floor, 1e-12)
    logits = -(losses - minimum) / temperature
    logits -= np.max(logits)
    weights = np.exp(logits)
    weights /= weights.sum()
    return weights, temperature


def fit_candidates_all_source(
    source: Sequence[EpisodeData],
    transform: Transform,
    protocol: Mapping[str, Any],
    horizon: int,
    candidates: Sequence[Candidate],
) -> list[RidgeModel]:
    fitted = []
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for variant in ("state", "action"):
        rows = [
            design_for_episode(episode, transform, protocol, horizon, variant)
            for episode in source
        ]
        cache[variant] = (
            np.concatenate([row[0] for row in rows]),
            np.concatenate([row[1] for row in rows]),
        )
    for candidate in candidates:
        x, y = cache[candidate.variant]
        fitted.append(fit_ridge(x, y, candidate.alpha))
    return fitted


def source_baseline_metrics(
    source: Sequence[EpisodeData],
    transform: Transform,
    protocol: Mapping[str, Any],
    horizon: int,
) -> dict[str, dict[int, float]]:
    model = protocol["model"]
    result = {"persistence": {}, "last_trend": {}}
    for episode in source:
        values = normalize_tactile(
            episode.tactile,
            transform.feature_scale,
            float(model["normalized_feature_clip"]),
        )
        starts = starts_for(
            len(values),
            horizon,
            int(model["trend_lag_frames"]),
            int(model["window_stride_frames"]),
        )
        truth = values[starts + horizon]
        active = (truth > float(model["active_threshold"])) | (
            values[starts] > float(model["active_threshold"])
        )
        for method in result:
            prediction = baseline_prediction(
                values,
                starts,
                horizon,
                int(model["trend_lag_frames"]),
                float(model["normalized_feature_clip"]),
                method,
            )
            result[method][episode.descriptor.episode_id] = rmse(
                prediction, truth, active
            )[1]
    return result


def ensemble_cv_residuals(
    source: Sequence[EpisodeData],
    candidates: Sequence[Candidate],
    weights: np.ndarray,
) -> tuple[np.ndarray, dict[int, float], np.ndarray]:
    raw_by_episode: dict[int, np.ndarray] = {}
    prediction_by_episode: dict[int, np.ndarray] = {}
    for episode in source:
        episode_id = episode.descriptor.episode_id
        predictions = np.stack(
            [candidate.cv_predictions[episode_id] for candidate in candidates]
        )
        prediction = np.einsum("k,kwd->wd", weights, predictions)
        truth = candidates[0].cv_truths[episode_id]
        prediction_by_episode[episode_id] = prediction
        raw_by_episode[episode_id] = (truth - prediction).reshape(len(truth), -1)
    global_bias = np.concatenate(list(raw_by_episode.values())).mean(axis=0)
    corrected_residuals = []
    per_episode = {}
    for episode in source:
        episode_id = episode.descriptor.episode_id
        donor = np.concatenate(
            [
                residual
                for other_id, residual in raw_by_episode.items()
                if other_id != episode_id
            ]
        )
        bias = donor.mean(axis=0)
        raw = raw_by_episode[episode_id]
        corrected = raw - bias[None, :]
        corrected_residuals.append(corrected)
        truth = candidates[0].cv_truths[episode_id]
        active = candidates[0].cv_masks[episode_id]
        corrected_prediction = prediction_by_episode[episode_id] + bias[None, :]
        per_episode[episode_id] = rmse(corrected_prediction, truth, active)[1]
    return np.concatenate(corrected_residuals), per_episode, global_bias


def fit_covariance(
    residuals: np.ndarray,
    rank: int,
    probability: float,
    mean_error: np.ndarray | None = None,
) -> CovarianceModel:
    centered = residuals - residuals.mean(axis=0)
    count, dimension = centered.shape
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    retained = min(rank, len(singular), max(count - 1, 0))
    if retained:
        factor = (
            right[:retained].T
            * (singular[:retained] / math.sqrt(max(count - 1, 1)))[None, :]
        )
    else:
        factor = np.empty((dimension, 0), dtype=np.float64)
    variance = np.mean(centered * centered, axis=0)
    diagonal = variance - np.sum(factor * factor, axis=1)
    positive = variance[variance > 0.0]
    floor = max(
        (float(np.median(positive)) if len(positive) else 1.0) * 1e-3,
        1e-8,
    )
    diagonal = np.maximum(diagonal, floor)
    stored_mean = residuals.mean(axis=0) if mean_error is None else mean_error
    provisional = CovarianceModel(stored_mean, diagonal, factor, 1.0, 1.0, 0.0, 0.0)
    nees = np.asarray(
        [woodbury_quadratic(error, provisional) / dimension for error in centered]
    )
    multiplier = max(float(np.mean(nees)), 1e-6)
    marginal_variance = multiplier * (diagonal + np.sum(factor * factor, axis=1))
    ratios = np.abs(centered) / np.sqrt(marginal_variance)[None, :]
    marginal_z = float(np.quantile(ratios, probability))
    marginal_coverage = float(np.mean(ratios <= marginal_z))
    return CovarianceModel(
        mean_error=stored_mean,
        diagonal=diagonal,
        factor=factor,
        multiplier=multiplier,
        marginal_z=marginal_z,
        source_marginal_coverage=marginal_coverage,
        source_joint_nanees=float(np.mean(nees / multiplier)),
    )


def woodbury_quadratic(error: np.ndarray, model: CovarianceModel) -> float:
    diagonal = model.multiplier * model.diagonal
    factor = math.sqrt(model.multiplier) * model.factor
    inverse = 1.0 / diagonal
    if factor.shape[1] == 0:
        return float(np.sum(error * error * inverse))
    weighted = inverse[:, None] * factor
    core = np.eye(factor.shape[1]) + factor.T @ weighted
    projected = factor.T @ (inverse * error)
    value = np.sum(error * error * inverse) - projected @ np.linalg.solve(
        core, projected
    )
    return float(max(value, 0.0))


def covariance_logdet(model: CovarianceModel) -> float:
    diagonal = model.multiplier * model.diagonal
    factor = math.sqrt(model.multiplier) * model.factor
    logdet = float(np.sum(np.log(diagonal)))
    if factor.shape[1]:
        core = np.eye(factor.shape[1]) + factor.T @ (factor / diagonal[:, None])
        sign, value = np.linalg.slogdet(core)
        if sign <= 0:
            raise EvaluationError("covariance core is not positive definite")
        logdet += float(value)
    return logdet


def probabilistic_metrics(
    errors: np.ndarray,
    covariance: CovarianceModel,
    probability: float,
    rng: np.random.Generator,
    sample_count: int,
) -> dict[str, float]:
    dimension = errors.shape[1]
    normal = NormalDist().inv_cdf(0.5 + probability / 2.0)
    chi_square = (
        dimension
        * (1.0 - 2.0 / (9.0 * dimension) + normal * math.sqrt(2.0 / (9.0 * dimension)))
        ** 3
    )
    marginal_variance = covariance.multiplier * (
        covariance.diagonal + np.sum(covariance.factor * covariance.factor, axis=1)
    )
    marginal_radius = covariance.marginal_z * np.sqrt(marginal_variance)
    logdet = covariance_logdet(covariance)
    quadratics = np.asarray([woodbury_quadratic(error, covariance) for error in errors])
    nll = 0.5 * (dimension * math.log(2.0 * math.pi) + logdet + quadratics) / dimension
    energy_values = []
    diagonal_std = np.sqrt(covariance.multiplier * covariance.diagonal)
    factor = math.sqrt(covariance.multiplier) * covariance.factor
    for error in errors:
        draws = rng.standard_normal((sample_count, dimension)) * diagonal_std[None, :]
        if factor.shape[1]:
            draws += rng.standard_normal((sample_count, factor.shape[1])) @ factor.T
        first = np.mean(np.linalg.norm(draws + error[None, :], axis=1))
        paired = draws[::2] - draws[1::2]
        second = 0.5 * np.mean(np.linalg.norm(paired, axis=1))
        energy_values.append(float((first - second) / math.sqrt(dimension)))
    return {
        "joint_nanees": float(np.mean(quadratics) / dimension),
        "joint_90_ellipsoid_coverage": float(np.mean(quadratics <= chi_square)),
        "marginal_90_coverage": float(
            np.mean(np.abs(errors) <= marginal_radius[None, :])
        ),
        "mean_marginal_90_width": float(2.0 * np.mean(marginal_radius)),
        "nll_per_dimension": float(np.mean(nll)),
        "energy_score": float(np.mean(energy_values)),
    }


def bootstrap_interval(values: np.ndarray, repetitions: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def evaluate_object(
    descriptors: Sequence[EpisodeDescriptor],
    protocol: Mapping[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    target_descriptor = max(descriptors, key=lambda item: item.episode_id)
    source_descriptors = [
        descriptor for descriptor in descriptors if descriptor is not target_descriptor
    ]
    source = [load_episode(descriptor) for descriptor in source_descriptors]
    horizon = int(protocol["model"]["primary_horizon_frames"])
    transform = build_transform(source, protocol, horizon)
    candidates = candidate_cv(source, transform, protocol, horizon)
    action_indices = [
        index
        for index, candidate in enumerate(candidates)
        if candidate.variant == "action"
    ]
    state_indices = [
        index
        for index, candidate in enumerate(candidates)
        if candidate.variant == "state"
    ]
    action_candidates = [candidates[index] for index in action_indices]
    weights, temperature = candidate_weights(
        action_candidates,
        float(protocol["model"]["ensemble_temperature_floor_fraction"]),
    )
    fitted = fit_candidates_all_source(source, transform, protocol, horizon, candidates)
    residuals, ensemble_source, source_bias = ensemble_cv_residuals(
        source, action_candidates, weights
    )
    baselines = source_baseline_metrics(source, transform, protocol, horizon)
    persistence_source = float(np.mean(list(baselines["persistence"].values())))
    last_source = float(np.mean(list(baselines["last_trend"].values())))
    state_best_index = min(
        state_indices, key=lambda index: candidates[index].cv_objective
    )
    state_source_by_episode = candidates[state_best_index].per_episode_active_rmse
    state_source = float(np.mean(list(state_source_by_episode.values())))
    ensemble_source_mean = float(np.mean(list(ensemble_source.values())))
    source_reference = min(persistence_source, last_source, state_source)
    guard = protocol["guard"]
    guard_accepts = bool(
        ensemble_source_mean
        < (1.0 - float(guard["minimum_relative_gain"])) * source_reference
        and all(
            ensemble_source[episode_id]
            <= (1.0 + float(guard["maximum_episode_regret_fraction"]))
            * min(
                baselines["persistence"][episode_id],
                baselines["last_trend"][episode_id],
                state_source_by_episode[episode_id],
            )
            for episode_id in ensemble_source
        )
    )
    source_method_values = {
        "persistence": persistence_source,
        "last_trend": last_source,
        "state_ridge": state_source,
    }
    fallback = min(source_method_values, key=source_method_values.get)
    covariance = fit_covariance(
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
            "candidate_weights": weights.tolist(),
            "candidate_names": [candidate.name for candidate in action_candidates],
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

    target = load_episode(target_descriptor)
    model = protocol["model"]
    variants: dict[str, tuple[np.ndarray, ...]] = {}
    for variant in ("state", "action"):
        variants[variant] = design_for_episode(
            target, transform, protocol, horizon, variant
        )
    _, _, current, truth, active = variants["state"]
    values = normalize_tactile(
        target.tactile,
        transform.feature_scale,
        float(model["normalized_feature_clip"]),
    )
    starts = starts_for(
        len(values),
        horizon,
        int(model["trend_lag_frames"]),
        int(model["window_stride_frames"]),
    )
    predictions: dict[str, np.ndarray] = {
        "persistence": baseline_prediction(
            values,
            starts,
            horizon,
            int(model["trend_lag_frames"]),
            float(model["normalized_feature_clip"]),
            "persistence",
        ),
        "last_trend": baseline_prediction(
            values,
            starts,
            horizon,
            int(model["trend_lag_frames"]),
            float(model["normalized_feature_clip"]),
            "last_trend",
        ),
    }
    state_best = state_best_index
    action_best = min(action_indices, key=lambda index: candidates[index].cv_objective)
    for name, index in (("state_ridge", state_best), ("action_ridge", action_best)):
        candidate = candidates[index]
        x = variants[candidate.variant][0]
        latent = predict_ridge(fitted[index], x)
        predictions[name] = decode_prediction(
            latent,
            current,
            transform,
            float(model["normalized_feature_clip"]),
        )
    all_candidate_predictions = []
    for index in action_indices:
        x = variants["action"][0]
        latent = predict_ridge(fitted[index], x)
        all_candidate_predictions.append(
            decode_prediction(
                latent,
                current,
                transform,
                float(model["normalized_feature_clip"]),
            )
        )
    stacked = np.stack(all_candidate_predictions)
    ensemble = np.einsum("k,kwd->wd", weights, stacked)
    ensemble = np.clip(
        ensemble + covariance.mean_error[None, :],
        0.0,
        float(model["normalized_feature_clip"]),
    )
    predictions["bayesian_action_ensemble"] = ensemble

    shuffled_rows = rng.permutation(len(starts))
    shuffled_predictions = []
    state_width = variants["state"][0].shape[1]
    for index in action_indices:
        x = variants["action"][0].copy()
        x[:, state_width:] = x[shuffled_rows, state_width:]
        latent = predict_ridge(fitted[index], x)
        shuffled_predictions.append(
            decode_prediction(
                latent,
                current,
                transform,
                float(model["normalized_feature_clip"]),
            )
        )
    shuffled = np.einsum("k,kwd->wd", weights, np.stack(shuffled_predictions))
    predictions["shuffled_action_control"] = np.clip(
        shuffled + covariance.mean_error[None, :],
        0.0,
        float(model["normalized_feature_clip"]),
    )
    predictions["guarded_action_ensemble"] = (
        ensemble if guard_accepts else predictions[fallback]
    )

    metrics = {}
    for method in METHODS:
        all_rmse, active_rmse = rmse(predictions[method], truth, active)
        metrics[method] = {
            "field_rmse": all_rmse,
            "active_field_rmse": active_rmse,
            "field_mae": float(np.mean(np.abs(predictions[method] - truth))),
        }
    uncertainty_errors = (truth - ensemble).reshape(len(truth), -1)
    uncertainty = probabilistic_metrics(
        uncertainty_errors,
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
        "target_action_family": action_family(target_descriptor.action),
        "target_bimanual": bool(target.bimanual),
        "source_fit_id": source_fit_id,
        "source_fit_frozen_before_target_tactile_open": True,
        "known_future_robot_trajectory_used": True,
        "forecast_horizon_frames": horizon,
        "forecast_window_count": int(len(starts)),
        "pooled_field_dimension": int(truth.shape[1]),
        "candidate_temperature": temperature,
        "candidate_names": [candidate.name for candidate in action_candidates],
        "candidate_weights": [float(value) for value in weights],
        "source_cv_active_rmse": {
            "persistence": persistence_source,
            "last_trend": last_source,
            "state_ridge": state_source,
            "bayesian_action_ensemble": ensemble_source_mean,
        },
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


def aggregate(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    metric_names = ("field_rmse", "active_field_rmse", "field_mae")
    methods = {
        method: {
            metric: float(np.mean([row["metrics"][method][metric] for row in rows]))
            for metric in metric_names
        }
        for method in METHODS
    }
    primary = "active_field_rmse"
    comparisons = {}
    for comparator in (
        "persistence",
        "last_trend",
        "state_ridge",
        "shuffled_action_control",
    ):
        differences = np.asarray(
            [
                row["metrics"]["bayesian_action_ensemble"][primary]
                - row["metrics"][comparator][primary]
                for row in rows
            ],
            dtype=np.float64,
        )
        comparisons[comparator] = {
            "ensemble_minus_comparator": float(np.mean(differences)),
            "relative_change": float(
                np.mean(differences)
                / np.mean([row["metrics"][comparator][primary] for row in rows])
            ),
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
    uncertainty_keys = rows[0]["uncertainty"].keys()
    uncertainty = {
        key: float(np.mean([row["uncertainty"][key] for row in rows]))
        for key in uncertainty_keys
    }
    return {
        "object_count": len(rows),
        "primary_metric": primary,
        "primary_horizon_frames": int(protocol["model"]["primary_horizon_frames"]),
        "methods": methods,
        "comparisons": comparisons,
        "guard_acceptance_fraction": float(
            np.mean([bool(row["guard_accepts"]) for row in rows])
        ),
        "target_action_families": sorted(
            {str(row["target_action_family"]) for row in rows}
        ),
        "uncertainty": uncertainty,
    }


def report(result: Mapping[str, Any]) -> str:
    summary = result["summary"]
    primary = summary["primary_metric"]
    lines = [
        "# Deform360 action-conditioned tactile development v2",
        "",
        f"- Status: **{result['status']}**",
        f"- Dataset: `{result['dataset_root']}`",
        f"- Objects: **{summary['object_count']}**",
        f"- Horizon: **{summary['primary_horizon_frames']} frames**",
        f"- Primary metric: **{primary}**",
        f"- Guard acceptance: **{summary['guard_acceptance_fraction']:.1%}**",
        "",
        "## Object-balanced results",
        "",
        "| Method | Active RMSE | All-field RMSE | MAE |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        values = summary["methods"][method]
        lines.append(
            f"| `{method}` | {values['active_field_rmse']:.8g} | "
            f"{values['field_rmse']:.8g} | {values['field_mae']:.8g} |"
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
    for comparator, values in summary["comparisons"].items():
        interval = values["object_bootstrap_95_interval"]
        wtl = (
            f"{values['object_wins']}/{values['object_ties']}/{values['object_losses']}"
        )
        lines.append(
            f"| `{comparator}` | {values['ensemble_minus_comparator']:.8g} | "
            f"{values['relative_change']:+.2%} | "
            f"[{interval[0]:.8g}, {interval[1]:.8g}] | {wtl} | "
            f"{values['worst_object_regret']:.8g} |"
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
            "The target episode for each object was selected before target tactile loading.",
            "All model, guard, normalization, dimensionality reduction, and covariance",
            "parameters were frozen from other episodes of that same object. The future",
            "robot trajectory is an allowed intervention input; future tactile response is",
            "used only for scoring. Reserved objects and camera/geometry payloads remain closed.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_protocol(protocol: Mapping[str, Any], root: Path) -> None:
    if (
        protocol.get("schema")
        != "bayesian-phystwin/deform360-action-conditioned-tactile-protocol-v2"
    ):
        raise EvaluationError("unexpected protocol schema")
    if Path(str(protocol["dataset_root"])) != root:
        raise EvaluationError("dataset root changed")
    development = list(map(str, protocol["development_object_ids"]))
    reserved = set(map(str, protocol["reserved_object_ids"]))
    if not development or len(development) != len(set(development)):
        raise EvaluationError("development roster is invalid")
    if set(development) & reserved:
        raise EvaluationError("development and reserved rosters overlap")
    if protocol.get("paper_claim_authorized") is not False:
        raise EvaluationError("development protocol self-authorized a paper claim")


def run(protocol_path: Path, root: Path) -> dict[str, Any]:
    protocol = read_json(protocol_path)
    root = root.resolve(strict=True)
    validate_protocol(protocol, root)
    descriptors_by_object = {}
    for object_id in protocol["development_object_ids"]:
        descriptors = discover_object(
            root, str(object_id), int(protocol["selection"]["minimum_episodes"])
        )
        if descriptors:
            descriptors_by_object[str(object_id)] = descriptors
    if len(descriptors_by_object) < int(protocol["selection"]["minimum_objects"]):
        raise EvaluationError(
            f"only {len(descriptors_by_object)} objects expose the registered carriers"
        )
    rng = np.random.default_rng(int(protocol["statistics"]["random_seed"]))
    rows = []
    failures = []
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
            rows.append(evaluate_object(descriptors, protocol, rng))
        except (EvaluationError, OSError, ValueError, np.linalg.LinAlgError) as error:
            failures.append(
                {
                    "object_id": str(object_id),
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
    if len(rows) < int(protocol["selection"]["minimum_objects"]):
        raise EvaluationError(
            f"only {len(rows)} objects completed; failures={failures}"
        )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 2,
        "status": "complete",
        "protocol_id": protocol["protocol_id"],
        "dataset_root": str(root),
        "github_sha": os.environ.get("GITHUB_SHA"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "information_boundary": {
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
