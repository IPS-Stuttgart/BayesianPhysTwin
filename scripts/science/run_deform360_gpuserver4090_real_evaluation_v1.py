#!/usr/bin/env python3
"""Read-only Deform360 real-data development evaluation.

The evaluator prefers released processed 3-D point-cloud/trajectory carriers.  If
none are materialized below the user-supplied Deform360 root, it evaluates the
released raw tactile streams instead.  Target recordings are selected from path
names and metadata before any target numeric array is opened.  Source-only model
selection and covariance calibration are frozen before target loading.

This is retrospective development evidence.  It does not authorize a paper,
state-of-the-art, safety, prospective-confirmation, or strict counterfactual
claim.  Raw arrays are never written to the output bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

OBJECT_RE = re.compile(r"^\d{3}-.+")
METHODS = (
    "persistence",
    "last_residual",
    "map_motion",
    "bayesian_motion",
    "guarded_bayesian_motion",
)


class EvaluationError(RuntimeError):
    """Raised when the frozen evaluation contract cannot be completed."""


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvaluationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_json_pairs)
    if not isinstance(value, dict):
        raise EvaluationError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def object_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sampled_fingerprint(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as stream:
        digest.update(stream.read(1024 * 1024))
        if size > 1024 * 1024:
            stream.seek(max(0, size - 1024 * 1024))
            digest.update(stream.read(1024 * 1024))
    return {
        "relative_size_bytes": int(size),
        "sampled_sha256": digest.hexdigest(),
        "fingerprint_rule": "sha256(size || first_1MiB || last_1MiB)",
    }


def finite(value: object, *, name: str, ndim: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if ndim is not None and result.ndim != ndim:
        raise EvaluationError(f"{name} must have {ndim} dimensions")
    if not np.all(np.isfinite(result)):
        raise EvaluationError(f"{name} contains nonfinite values")
    return result


def softmax(logits: np.ndarray) -> np.ndarray:
    values = finite(logits, name="logits", ndim=1)
    values = values - float(np.max(values))
    weights = np.exp(values)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise EvaluationError("invalid model weights")
    return weights / total


def bootstrap_interval(
    values: np.ndarray, *, repetitions: int, rng: np.random.Generator
) -> list[float]:
    values = finite(values, name="bootstrap values", ndim=1)
    if values.size == 0:
        raise EvaluationError("cannot bootstrap an empty vector")
    if values.size == 1:
        return [float(values[0]), float(values[0])]
    indices = rng.integers(0, values.size, size=(repetitions, values.size))
    samples = values[indices].mean(axis=1)
    return [float(number) for number in np.quantile(samples, [0.025, 0.975])]


@dataclass(frozen=True)
class Descriptor:
    modality: str
    object_id: str
    group_id: str
    recording_id: str
    paths: tuple[Path, ...]
    action: str | None


@dataclass
class LoadedSequence:
    descriptor: Descriptor
    features: np.ndarray
    field: object
    fingerprint: dict[str, Any]
    frame_stride: int


@dataclass
class SourceFit:
    gains: np.ndarray
    weights: np.ndarray
    map_index: int
    location: np.ndarray
    scale: np.ndarray
    covariance_diagonal: np.ndarray
    covariance_factor: np.ndarray
    covariance_multiplier: float
    guard_accepts: bool
    fallback_method: str
    source_cv: dict[str, Any]
    source_recordings: list[str]
    source_fingerprints: list[dict[str, Any]]
    feature_dimension: int
    horizon: int


def validate_protocol(protocol: Mapping[str, Any], dataset_root: Path) -> None:
    if protocol.get("schema") != (
        "bayesian-phystwin/deform360-gpuserver4090-real-evaluation-protocol-v1"
    ):
        raise EvaluationError("unexpected protocol schema")
    expected = Path(str(protocol["dataset"]["root"]))
    if dataset_root != expected:
        raise EvaluationError(
            f"dataset root must equal the frozen path {expected}, got {dataset_root}"
        )
    development = tuple(str(value) for value in protocol["development_object_ids"])
    reserved = tuple(str(value) for value in protocol["forbidden_reserved_object_ids"])
    if not development or len(development) != len(set(development)):
        raise EvaluationError("development object roster must be nonempty and unique")
    if len(reserved) != len(set(reserved)) or set(development) & set(reserved):
        raise EvaluationError("development and reserved object rosters overlap")
    if protocol.get("replacement_allowed") is not False:
        raise EvaluationError("replacement must remain disabled")
    boundary = protocol["information_boundary"]
    required_false = (
        "reserved_objects_opened",
        "dataset_mutation_allowed",
        "raw_data_upload",
    )
    if any(boundary.get(key) is not False for key in required_false):
        raise EvaluationError("protocol information boundary changed")
    if protocol.get("paper_claim_authorized") is not False:
        raise EvaluationError("development evaluation cannot authorize a paper claim")


def _metadata_actions(object_dir: Path) -> list[str | None]:
    candidates = [
        object_dir / "metadata.json",
        object_dir / "meta.json",
        object_dir / "object_metadata.json",
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return []
    try:
        metadata = read_json(path)
    except (OSError, ValueError, EvaluationError):
        return []
    raw = metadata.get("sequences", metadata.get("episodes", metadata.get("takes", [])))
    if not isinstance(raw, list):
        return []
    actions: list[str | None] = []
    for entry in raw:
        if not isinstance(entry, dict):
            actions.append(None)
            continue
        value: object | None = None
        for key in (
            "action",
            "action_name",
            "manipulation",
            "primitive",
            "description",
            "instruction",
        ):
            if entry.get(key) not in (None, ""):
                value = entry[key]
                break
        actions.append(None if value is None else str(value))
    return actions


def _candidate_object_dirs(root: Path, object_id: str) -> list[Path]:
    candidates: list[Path] = []
    direct_roots = (
        root / "raw-repository" / "raw",
        root / "raw-repository",
        root / "raw",
        root / "processed-repository" / "processed",
        root / "processed-repository",
        root / "processed",
    )
    for parent in direct_roots:
        candidate = parent / object_id
        if candidate.is_dir():
            candidates.append(candidate.resolve())
    if not candidates:
        for candidate in root.rglob(object_id):
            if candidate.is_dir():
                candidates.append(candidate.resolve())
    return sorted(set(candidates), key=str)


def discover_geometry(
    root: Path, protocol: Mapping[str, Any]
) -> dict[tuple[str, str, str], list[Descriptor]]:
    groups: dict[tuple[str, str, str], list[Descriptor]] = {}
    for object_id in protocol["development_object_ids"]:
        for object_dir in _candidate_object_dirs(root, str(object_id)):
            actions = _metadata_actions(object_dir)
            pcd_dirs = sorted(
                {
                    path.resolve()
                    for path in object_dir.rglob("pcd_clean")
                    if path.is_dir()
                },
                key=str,
            )
            for index, pcd_dir in enumerate(pcd_dirs):
                frames = tuple(
                    sorted(pcd_dir.glob("*.npz"), key=lambda path: path.name)
                )
                if len(frames) < int(
                    protocol["limits"]["minimum_frames_per_recording"]
                ):
                    continue
                episode = pcd_dir.parent.name or f"episode_{index:04d}"
                descriptor = Descriptor(
                    "geometry_3d_point_cloud",
                    str(object_id),
                    "pcd_clean",
                    episode,
                    frames,
                    actions[index] if index < len(actions) else None,
                )
                groups.setdefault(
                    (descriptor.modality, descriptor.object_id, descriptor.group_id), []
                ).append(descriptor)

            # Fixed-identity trajectory archives are admitted by path identity first;
            # numeric shape qualification happens only after source/target selection.
            generic = []
            for path in object_dir.rglob("*.npz"):
                lower = "/".join(part.lower() for part in path.parts)
                if any(
                    token in lower
                    for token in (
                        "/pcd_clean/",
                        "tactile",
                        "/robot/",
                        "calibration",
                        "camera",
                        "intrinsic",
                        "extrinsic",
                    )
                ):
                    continue
                if path.stat().st_size <= 0:
                    continue
                generic.append(path.resolve())
            for index, path in enumerate(sorted(set(generic), key=str)):
                descriptor = Descriptor(
                    "geometry_3d_fixed_identity",
                    str(object_id),
                    "trajectory_npz",
                    path.stem,
                    (path,),
                    actions[index] if index < len(actions) else None,
                )
                groups.setdefault(
                    (descriptor.modality, descriptor.object_id, descriptor.group_id), []
                ).append(descriptor)
    return groups


def discover_tactile(
    root: Path, protocol: Mapping[str, Any]
) -> dict[tuple[str, str, str], list[Descriptor]]:
    groups: dict[tuple[str, str, str], list[Descriptor]] = {}
    for object_id in protocol["development_object_ids"]:
        for object_dir in _candidate_object_dirs(root, str(object_id)):
            # Do not reinterpret processed directories as raw tactile recordings.
            if "processed-repository" in object_dir.parts:
                continue
            actions = _metadata_actions(object_dir)
            by_parent: dict[Path, list[Path]] = {}
            for path in object_dir.rglob("*.npy"):
                lower = "/".join(part.lower() for part in path.parts)
                if "tactile" not in lower or path.name.lower().startswith("median_"):
                    continue
                if path.stat().st_size <= 0:
                    continue
                by_parent.setdefault(path.parent.resolve(), []).append(path.resolve())
            for parent, paths in sorted(
                by_parent.items(), key=lambda item: str(item[0])
            ):
                paths = sorted(set(paths), key=lambda path: path.name)
                group_id = str(parent.relative_to(object_dir)).replace(os.sep, "/")
                for index, path in enumerate(paths):
                    descriptor = Descriptor(
                        "tactile_response",
                        str(object_id),
                        group_id,
                        path.stem,
                        (path,),
                        actions[index] if index < len(actions) else None,
                    )
                    groups.setdefault(
                        (
                            descriptor.modality,
                            descriptor.object_id,
                            descriptor.group_id,
                        ),
                        [],
                    ).append(descriptor)
    return groups


def _load_npz_array(path: Path, keys: Sequence[str]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        for key in keys:
            if key in archive:
                return np.asarray(archive[key])
        arrays = [np.asarray(archive[key]) for key in archive.files]
    candidates = [array for array in arrays if np.issubdtype(array.dtype, np.number)]
    if len(candidates) == 1:
        return candidates[0]
    raise EvaluationError(f"no unambiguous numeric array in {path.name}")


def _subsample_indices(count: int, maximum: int) -> tuple[np.ndarray, int]:
    if count <= maximum:
        return np.arange(count, dtype=int), 1
    stride = int(math.ceil(count / maximum))
    indices = np.arange(0, count, stride, dtype=int)[:maximum]
    return indices, stride


def load_geometry(
    descriptor: Descriptor, protocol: Mapping[str, Any], root: Path
) -> LoadedSequence:
    maximum_frames = int(protocol["limits"]["maximum_frames_per_recording"])
    maximum_points = int(protocol["limits"]["maximum_points_per_cloud"])
    minimum_frames = int(protocol["limits"]["minimum_frames_per_recording"])
    fingerprints = []
    if descriptor.modality == "geometry_3d_point_cloud":
        indices, frame_stride = _subsample_indices(
            len(descriptor.paths), maximum_frames
        )
        clouds: list[np.ndarray] = []
        selected_paths = [descriptor.paths[int(index)] for index in indices]
        for path in selected_paths:
            array = _load_npz_array(path, ("pts", "points", "xyz", "pcd"))
            points = finite(array, name=f"points in {path.name}", ndim=2)
            if points.shape[1] != 3 or points.shape[0] < 8:
                raise EvaluationError(f"unsupported point cloud shape in {path.name}")
            if points.shape[0] > maximum_points:
                point_indices = np.linspace(
                    0, points.shape[0] - 1, maximum_points, dtype=int
                )
                points = points[point_indices]
            clouds.append(points.copy())
            fingerprints.append(
                {
                    "path": str(path.relative_to(root)),
                    **sampled_fingerprint(path),
                }
            )
        if len(clouds) < minimum_frames:
            raise EvaluationError(
                "point-cloud sequence is too short after qualification"
            )
        features = np.stack([cloud.mean(axis=0) for cloud in clouds])
        return LoadedSequence(
            descriptor,
            features,
            clouds,
            {"files": fingerprints, "count": len(fingerprints)},
            frame_stride,
        )

    path = descriptor.paths[0]
    array = _load_npz_array(
        path,
        (
            "positions",
            "points",
            "trajectory",
            "trajectories",
            "world_points",
            "pts_world",
            "points_world_m",
        ),
    )
    trajectory = finite(array, name=f"trajectory in {path.name}", ndim=3)
    if trajectory.shape[-1] != 3 or trajectory.shape[0] < minimum_frames:
        raise EvaluationError(
            f"unsupported fixed-identity trajectory shape in {path.name}"
        )
    indices, frame_stride = _subsample_indices(trajectory.shape[0], maximum_frames)
    trajectory = trajectory[indices]
    if trajectory.shape[1] > maximum_points:
        point_indices = np.linspace(
            0, trajectory.shape[1] - 1, maximum_points, dtype=int
        )
        trajectory = trajectory[:, point_indices]
    features = trajectory.mean(axis=1)
    return LoadedSequence(
        descriptor,
        features,
        trajectory,
        {
            "files": [
                {"path": str(path.relative_to(root)), **sampled_fingerprint(path)}
            ],
            "count": 1,
        },
        frame_stride,
    )


def _load_tactile_payload(path: Path) -> np.ndarray:
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        value = np.fromfile(path, dtype=np.float32)
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[-2:] == (16, 32):
        return finite(array, name=f"tactile payload {path.name}", ndim=3)
    flat = np.asarray(array, dtype=np.float64).reshape(-1)
    taxels = 16 * 32
    if flat.size == 0 or flat.size % taxels:
        raise EvaluationError(f"unsupported tactile payload size in {path.name}")
    return finite(flat.reshape(-1, 16, 32), name=f"tactile payload {path.name}", ndim=3)


def _median_candidate(path: Path) -> Path | None:
    stamp = path.stem.rsplit("_", 1)[-1]
    exact = path.parent / f"median_{stamp}.npy"
    if exact.is_file():
        return exact.resolve()
    candidates = sorted(path.parent.glob("median_*.npy"), key=lambda value: value.name)
    if len(candidates) == 1:
        return candidates[0].resolve()
    return None


def _load_median(path: Path | None) -> np.ndarray:
    if path is None:
        return np.zeros((16, 32), dtype=np.float64)
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        raw = np.fromfile(path, dtype=np.float32)
        value = raw.reshape(16, 32) if raw.size == 16 * 32 else raw
    median = finite(value, name=f"tactile median {path.name}")
    if median.shape != (16, 32):
        raise EvaluationError(f"unsupported tactile median shape in {path.name}")
    return median


def _activity_crop(grid: np.ndarray, minimum_frames: int) -> np.ndarray:
    active = np.max(grid, axis=(1, 2)) > 0.0
    indices = np.flatnonzero(active)
    if indices.size:
        start = max(0, int(indices[0]) - 4)
        stop = min(len(grid), int(indices[-1]) + 5)
        if stop - start >= minimum_frames:
            return grid[start:stop]
    return grid


def tactile_features(grid: np.ndarray) -> np.ndarray:
    rows = np.linspace(-1.0, 1.0, grid.shape[1], dtype=np.float64)[None, :, None]
    cols = np.linspace(-1.0, 1.0, grid.shape[2], dtype=np.float64)[None, None, :]
    mass_sum = grid.sum(axis=(1, 2))
    denominator = np.maximum(mass_sum, 1e-12)
    row_center = (grid * rows).sum(axis=(1, 2)) / denominator
    col_center = (grid * cols).sum(axis=(1, 2)) / denominator
    row_spread = np.sqrt(
        np.maximum(
            (grid * (rows - row_center[:, None, None]) ** 2).sum(axis=(1, 2))
            / denominator,
            0.0,
        )
    )
    col_spread = np.sqrt(
        np.maximum(
            (grid * (cols - col_center[:, None, None]) ** 2).sum(axis=(1, 2))
            / denominator,
            0.0,
        )
    )
    empty = mass_sum <= 1e-12
    row_center[empty] = 0.0
    col_center[empty] = 0.0
    row_spread[empty] = 0.0
    col_spread[empty] = 0.0
    features = np.column_stack(
        (
            grid.mean(axis=(1, 2)),
            np.mean(grid > 0.0, axis=(1, 2)),
            grid.max(axis=(1, 2)),
            row_center,
            col_center,
            row_spread,
            col_spread,
            grid[:, :, : grid.shape[2] // 2].mean(axis=(1, 2)),
            grid[:, :, grid.shape[2] // 2 :].mean(axis=(1, 2)),
        )
    )
    return finite(features, name="tactile features", ndim=2)


def load_tactile(
    descriptor: Descriptor, protocol: Mapping[str, Any], root: Path
) -> LoadedSequence:
    path = descriptor.paths[0]
    median_path = _median_candidate(path)
    raw = _load_tactile_payload(path)
    median = _load_median(median_path)
    grid = raw - median[None, :, :]
    grid = np.maximum(grid, 0.0)
    positive_max = float(np.max(grid))
    if positive_max > 0.0:
        grid /= positive_max
    grid[grid < 0.3] = 0.0
    grid[:, :, -1] = 0.0
    grid = grid[:, :12, :]
    minimum_frames = int(protocol["limits"]["minimum_frames_per_recording"])
    grid = _activity_crop(grid, minimum_frames)
    indices, frame_stride = _subsample_indices(
        len(grid), int(protocol["limits"]["maximum_frames_per_recording"])
    )
    grid = grid[indices]
    if len(grid) < minimum_frames:
        raise EvaluationError(f"tactile recording {path.name} is too short")
    features = tactile_features(grid)
    files = [{"path": str(path.relative_to(root)), **sampled_fingerprint(path)}]
    if median_path is not None:
        files.append(
            {
                "path": str(median_path.relative_to(root)),
                **sampled_fingerprint(median_path),
            }
        )
    return LoadedSequence(
        descriptor,
        features,
        grid,
        {"files": files, "count": len(files)},
        frame_stride,
    )


def load_sequence(
    descriptor: Descriptor, protocol: Mapping[str, Any], root: Path
) -> LoadedSequence:
    if descriptor.modality.startswith("geometry_3d"):
        return load_geometry(descriptor, protocol, root)
    if descriptor.modality == "tactile_response":
        return load_tactile(descriptor, protocol, root)
    raise EvaluationError(f"unknown modality: {descriptor.modality}")


def robust_standardizer(
    sequences: Sequence[LoadedSequence],
) -> tuple[np.ndarray, np.ndarray]:
    if not sequences:
        raise EvaluationError("at least one source recording is required")
    values = np.concatenate([sequence.features for sequence in sequences], axis=0)
    location = np.median(values, axis=0)
    mad = np.median(np.abs(values - location), axis=0)
    scale = 1.4826 * mad
    fallback = np.std(values, axis=0)
    scale = np.where(scale > 1e-8, scale, fallback)
    scale = np.maximum(scale, 1e-4)
    return finite(location, name="source location", ndim=1), finite(
        scale, name="source scale", ndim=1
    )


def standardized(
    sequence: LoadedSequence, location: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    if sequence.features.shape[1] != location.size or scale.shape != location.shape:
        raise EvaluationError("feature standardizer dimension mismatch")
    return finite(
        (sequence.features - location[None, :]) / scale[None, :],
        name="standardized sequence",
        ndim=2,
    )


def windows(
    values: np.ndarray, gains: np.ndarray, horizon: int, stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = finite(values, name="sequence values", ndim=2)
    if len(values) <= horizon + 2:
        raise EvaluationError("sequence is shorter than the registered horizon")
    starts = np.arange(1, len(values) - horizon, stride, dtype=int)
    if starts.size == 0:
        raise EvaluationError("sequence contains no forecast windows")
    steps = np.arange(1, horizon + 1, dtype=np.float64)[None, None, :, None]
    current = values[starts]
    velocity = current - values[starts - 1]
    bank = current[:, None, None, :] + (
        gains[None, :, None, None] * steps * velocity[:, None, None, :]
    )
    truth = np.stack([values[start + 1 : start + horizon + 1] for start in starts])
    return bank, truth, starts


def _weights_from_losses(losses: np.ndarray, floor: float) -> tuple[np.ndarray, float]:
    losses = finite(losses, name="source losses", ndim=2)
    temperature = max(float(np.median(np.min(losses, axis=1))), floor * floor)
    weights = softmax(-losses.sum(axis=0) / (2.0 * temperature))
    return weights, temperature


def _method_prediction(bank: np.ndarray, fit: SourceFit, method: str) -> np.ndarray:
    if method == "persistence":
        index = int(np.argmin(np.abs(fit.gains - 0.0)))
        return bank[:, index]
    if method == "last_residual":
        index = int(np.argmin(np.abs(fit.gains - 1.0)))
        return bank[:, index]
    if method == "map_motion":
        return bank[:, fit.map_index]
    if method == "bayesian_motion":
        return np.einsum("k,wkhd->whd", fit.weights, bank)
    if method == "guarded_bayesian_motion":
        chosen = "bayesian_motion" if fit.guard_accepts else fit.fallback_method
        return _method_prediction(bank, fit, chosen)
    raise EvaluationError(f"unknown method: {method}")


def _fit_low_rank(
    second_moment: np.ndarray, rank: int, floor: float
) -> tuple[np.ndarray, np.ndarray]:
    matrix = finite(second_moment, name="residual second moment", ndim=2)
    if matrix.shape[0] != matrix.shape[1]:
        raise EvaluationError("residual second moment must be square")
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    positive = np.flatnonzero(eigenvalues > floor)
    selected = (
        positive[-min(rank, len(positive)) :]
        if len(positive)
        else np.array([], dtype=int)
    )
    factor = (
        eigenvectors[:, selected] * np.sqrt(eigenvalues[selected])[None, :]
        if selected.size
        else np.empty((matrix.shape[0], 0), dtype=np.float64)
    )
    unexplained = np.diag(matrix) - np.sum(factor * factor, axis=1)
    diagonal = np.maximum(unexplained, floor)
    return diagonal, factor


def _quadratic_and_logdet(
    error: np.ndarray, covariance: np.ndarray
) -> tuple[float, float, np.ndarray]:
    covariance = finite(covariance, name="predictive covariance", ndim=2)
    covariance = 0.5 * (covariance + covariance.T)
    jitter = max(1e-10, 1e-10 * float(np.max(np.diag(covariance))))
    for _ in range(8):
        try:
            factor = np.linalg.cholesky(covariance + jitter * np.eye(len(covariance)))
            solved = np.linalg.solve(factor, error)
            quadratic = float(np.dot(solved, solved))
            logdet = float(2.0 * np.log(np.diag(factor)).sum())
            return quadratic, logdet, factor
        except np.linalg.LinAlgError:
            jitter *= 10.0
    raise EvaluationError("predictive covariance is not numerically positive definite")


def _window_covariance(bank: np.ndarray, fit: SourceFit) -> np.ndarray:
    flattened = bank.reshape(bank.shape[0], -1)
    mean = np.einsum("k,km->m", fit.weights, flattened)
    centered = flattened - mean[None, :]
    spread = np.einsum("k,ki,kj->ij", fit.weights, centered, centered)
    base = (
        np.diag(fit.covariance_diagonal)
        + fit.covariance_factor @ fit.covariance_factor.T
    )
    return fit.covariance_multiplier * (base + spread)


def fit_sources(
    source: Sequence[LoadedSequence], protocol: Mapping[str, Any]
) -> SourceFit:
    gains = finite(
        protocol["finite_motion_bank"]["residual_gains"],
        name="residual gains",
        ndim=1,
    )
    if (
        gains.size < 3
        or not np.any(np.isclose(gains, 0.0))
        or not np.any(np.isclose(gains, 1.0))
    ):
        raise EvaluationError(
            "motion bank must include persistence and last-residual gains"
        )
    horizon = int(protocol["limits"]["forecast_horizon_frames"])
    stride = int(protocol["limits"]["window_stride"])
    floor = float(protocol["finite_motion_bank"]["measurement_floor_standardized"])
    location, scale = robust_standardizer(source)
    banks: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    losses: list[np.ndarray] = []
    for sequence in source:
        bank, truth, _ = windows(
            standardized(sequence, location, scale), gains, horizon, stride
        )
        banks.append(bank)
        truths.append(truth)
        losses.append(np.mean((bank - truth[:, None, :, :]) ** 2, axis=(0, 2, 3)))
    loss_matrix = np.stack(losses)
    weights, temperature = _weights_from_losses(loss_matrix, floor)
    map_index = int(np.argmax(weights))

    cv_rows = []
    for held in range(len(source)):
        training = np.delete(loss_matrix, held, axis=0)
        held_weights, _ = _weights_from_losses(
            training if len(training) else np.ones((1, gains.size)), floor
        )
        held_bank = banks[held]
        held_truth = truths[held]
        mixture = np.einsum("k,wkhd->whd", held_weights, held_bank)
        persistence = held_bank[:, int(np.argmin(np.abs(gains - 0.0)))]
        last = held_bank[:, int(np.argmin(np.abs(gains - 1.0)))]
        cv_rows.append(
            {
                "recording": source[held].descriptor.recording_id,
                "persistence_mse": float(np.mean((persistence - held_truth) ** 2)),
                "last_residual_mse": float(np.mean((last - held_truth) ** 2)),
                "bayesian_mse": float(np.mean((mixture - held_truth) ** 2)),
            }
        )
    cv_persistence = float(np.mean([row["persistence_mse"] for row in cv_rows]))
    cv_last = float(np.mean([row["last_residual_mse"] for row in cv_rows]))
    cv_bayesian = float(np.mean([row["bayesian_mse"] for row in cv_rows]))
    reference = min(cv_persistence, cv_last)
    minimum_gain = float(protocol["finite_motion_bank"]["minimum_guard_relative_gain"])
    maximum_regret = float(
        protocol["finite_motion_bank"]["maximum_guard_recording_regret_fraction"]
    )
    guard_accepts = bool(
        cv_bayesian < (1.0 - minimum_gain) * reference
        and all(
            row["bayesian_mse"]
            <= (1.0 + maximum_regret)
            * min(row["persistence_mse"], row["last_residual_mse"])
            for row in cv_rows
        )
    )
    fallback = "persistence" if cv_persistence <= cv_last else "last_residual"

    error_moments = []
    source_errors: list[tuple[np.ndarray, np.ndarray]] = []
    for bank, truth in zip(banks, truths, strict=True):
        mean = np.einsum("k,wkhd->whd", weights, bank)
        errors = (truth - mean).reshape(len(mean), -1)
        error_moments.append(errors.T @ errors / len(errors))
        source_errors.extend(
            (errors[index], bank[index]) for index in range(len(errors))
        )
    second_moment = np.mean(error_moments, axis=0)
    rank = min(
        int(protocol["joint_uncertainty"]["maximum_low_rank"]),
        second_moment.shape[0] - 1,
    )
    diagonal, factor = _fit_low_rank(second_moment, rank, floor * floor)

    normalized_nees = []
    provisional = SourceFit(
        gains,
        weights,
        map_index,
        location,
        scale,
        diagonal,
        factor,
        1.0,
        guard_accepts,
        fallback,
        {},
        [sequence.descriptor.recording_id for sequence in source],
        [sequence.fingerprint for sequence in source],
        int(location.size),
        horizon,
    )
    dimension = second_moment.shape[0]
    for error, bank in source_errors:
        covariance = _window_covariance(bank, provisional)
        quadratic, _, _ = _quadratic_and_logdet(error, covariance)
        normalized_nees.append(quadratic / dimension)
    multiplier = float(np.mean(normalized_nees))
    multiplier = float(
        np.clip(
            multiplier,
            protocol["joint_uncertainty"]["source_calibration_multiplier_minimum"],
            protocol["joint_uncertainty"]["source_calibration_multiplier_maximum"],
        )
    )
    total_variance = float(np.trace(second_moment))
    low_rank_variance = float(np.sum(factor * factor))
    source_cv = {
        "temperature": temperature,
        "recordings": cv_rows,
        "mean_mse": {
            "persistence": cv_persistence,
            "last_residual": cv_last,
            "bayesian_motion": cv_bayesian,
        },
        "guard_accepts": guard_accepts,
        "fallback_method": fallback,
        "low_rank_residual_energy_fraction": (
            low_rank_variance / total_variance if total_variance > 0.0 else 0.0
        ),
        "uncalibrated_source_joint_nanees": float(np.mean(normalized_nees)),
        "source_covariance_multiplier": multiplier,
    }
    return SourceFit(
        gains,
        weights,
        map_index,
        location,
        scale,
        diagonal,
        factor,
        multiplier,
        guard_accepts,
        fallback,
        source_cv,
        [sequence.descriptor.recording_id for sequence in source],
        [sequence.fingerprint for sequence in source],
        int(location.size),
        horizon,
    )


def _probabilistic_metrics(
    bank: np.ndarray,
    truth: np.ndarray,
    fit: SourceFit,
    protocol: Mapping[str, Any],
    rng: np.random.Generator,
) -> dict[str, float]:
    mean = np.einsum("k,wkhd->whd", fit.weights, bank)
    errors = (truth - mean).reshape(len(mean), -1)
    dimension = errors.shape[1]
    marginal_probability = float(
        protocol["joint_uncertainty"]["marginal_coverage_probability"]
    )
    marginal_z = NormalDist().inv_cdf(0.5 + 0.5 * marginal_probability)
    joint_q = (
        dimension
        * (
            1.0
            - 2.0 / (9.0 * dimension)
            + marginal_z * math.sqrt(2.0 / (9.0 * dimension))
        )
        ** 3
    )
    nll = []
    nees = []
    marginal_hits = []
    widths = []
    joint_hits = []
    energy = []
    sample_count = int(protocol["joint_uncertainty"]["energy_score_samples"])
    for index, error in enumerate(errors):
        covariance = _window_covariance(bank[index], fit)
        quadratic, logdet, cholesky = _quadratic_and_logdet(error, covariance)
        nll.append(
            0.5 * (dimension * math.log(2.0 * math.pi) + logdet + quadratic) / dimension
        )
        nees.append(quadratic / dimension)
        standard = np.sqrt(np.maximum(np.diag(covariance), 1e-15))
        marginal_hits.append(float(np.mean(np.abs(error) <= marginal_z * standard)))
        widths.append(float(np.mean(2.0 * marginal_z * standard)))
        joint_hits.append(float(quadratic <= joint_q))
        draws = rng.standard_normal((sample_count, dimension)) @ cholesky.T
        # Draws are errors around the predictive mean.  The observed outcome is
        # represented by the negative prediction error in that coordinate frame.
        observed = -error
        first = np.mean(np.linalg.norm(draws - observed[None, :], axis=1))
        paired = draws[::2] - draws[1::2]
        second = 0.5 * np.mean(np.linalg.norm(paired, axis=1))
        energy.append(float((first - second) / math.sqrt(dimension)))
    return {
        "coordinate_nll_standardized": float(np.mean(nll)),
        "joint_nanees": float(np.mean(nees)),
        "marginal_90_coverage": float(np.mean(marginal_hits)),
        "mean_marginal_90_width_standardized": float(np.mean(widths)),
        "joint_90_ellipsoid_coverage": float(np.mean(joint_hits)),
        "energy_score_standardized": float(np.mean(energy)),
    }


def _method_feature_metrics(
    bank: np.ndarray,
    truth: np.ndarray,
    fit: SourceFit,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    raw_truth = truth * fit.scale[None, None, :] + fit.location[None, None, :]
    for method in METHODS:
        prediction = _method_prediction(bank, fit, method)
        error = prediction - truth
        raw_prediction = (
            prediction * fit.scale[None, None, :] + fit.location[None, None, :]
        )
        raw_error = raw_prediction - raw_truth
        metrics = {
            "rmse_standardized": float(np.sqrt(np.mean(error * error))),
            "mae_standardized": float(np.mean(np.abs(error))),
            "raw_feature_rmse": float(np.sqrt(np.mean(raw_error * raw_error))),
        }
        if fit.feature_dimension == 9:
            metrics.update(
                {
                    "total_contact_mae": float(np.mean(np.abs(raw_error[..., 0]))),
                    "active_fraction_mae": float(np.mean(np.abs(raw_error[..., 1]))),
                    "tactile_centroid_mae": float(
                        np.mean(np.linalg.norm(raw_error[..., 3:5], axis=-1))
                    ),
                }
            )
        elif fit.feature_dimension == 3:
            metrics["centroid_rmse_mm"] = 1000.0 * float(
                np.sqrt(np.mean(np.sum(raw_error * raw_error, axis=-1)))
            )
        result[method] = metrics
    return result


def _field_method_gain(fit: SourceFit, method: str) -> float:
    if method == "persistence":
        return 0.0
    if method == "last_residual":
        return 1.0
    if method == "map_motion":
        return float(fit.gains[fit.map_index])
    if method == "bayesian_motion":
        return float(np.dot(fit.weights, fit.gains))
    if method == "guarded_bayesian_motion":
        chosen = "bayesian_motion" if fit.guard_accepts else fit.fallback_method
        return _field_method_gain(fit, chosen)
    raise EvaluationError(f"unknown method: {method}")


def _tactile_field_metrics(
    sequence: LoadedSequence, fit: SourceFit, starts: np.ndarray
) -> dict[str, dict[str, float]]:
    grid = finite(sequence.field, name="tactile field", ndim=3)
    horizon = fit.horizon
    steps = np.arange(1, horizon + 1, dtype=np.float64)[:, None, None]
    result = {}
    for method in METHODS:
        gain = _field_method_gain(fit, method)
        squared = []
        absolute = []
        for start in starts:
            current = grid[start]
            velocity = current - grid[start - 1]
            prediction = current[None, :, :] + gain * steps * velocity[None, :, :]
            truth = grid[start + 1 : start + horizon + 1]
            error = prediction - truth
            squared.append(float(np.mean(error * error)))
            absolute.append(float(np.mean(np.abs(error))))
        result[method] = {
            "tactile_field_rmse": float(math.sqrt(np.mean(squared))),
            "tactile_field_mae": float(np.mean(absolute)),
        }
    return result


def _symmetric_chamfer_mm(first: np.ndarray, second: np.ndarray) -> float:
    first = finite(first, name="predicted cloud", ndim=2)
    second = finite(second, name="target cloud", ndim=2)
    if first.shape[1] != 3 or second.shape[1] != 3:
        raise EvaluationError("point clouds must contain XYZ coordinates")
    # Limit the quadratic calculation independently of the retained carrier cap.
    first = first[np.linspace(0, len(first) - 1, min(len(first), 256), dtype=int)]
    second = second[np.linspace(0, len(second) - 1, min(len(second), 256), dtype=int)]
    distance2 = np.sum((first[:, None, :] - second[None, :, :]) ** 2, axis=2)
    distance = 0.5 * (
        np.sqrt(np.min(distance2, axis=1)).mean()
        + np.sqrt(np.min(distance2, axis=0)).mean()
    )
    return 1000.0 * float(distance)


def _geometry_field_metrics(
    sequence: LoadedSequence, fit: SourceFit, starts: np.ndarray
) -> dict[str, dict[str, float]]:
    horizon = fit.horizon
    chosen_starts = starts[:: max(1, len(starts) // 24)]
    if isinstance(sequence.field, np.ndarray):
        clouds = [sequence.field[index] for index in range(len(sequence.field))]
    else:
        clouds = list(sequence.field)
    result = {}
    for method in METHODS:
        gain = _field_method_gain(fit, method)
        distances = []
        for start in chosen_starts:
            current = clouds[int(start)]
            current_centroid = current.mean(axis=0)
            previous_centroid = clouds[int(start) - 1].mean(axis=0)
            velocity = current_centroid - previous_centroid
            for step in range(1, horizon + 1):
                prediction = current + gain * step * velocity[None, :]
                distances.append(
                    _symmetric_chamfer_mm(prediction, clouds[int(start) + step])
                )
        result[method] = {"symmetric_chamfer_mm": float(np.mean(distances))}
    return result


def evaluate_target(
    target: LoadedSequence,
    fit: SourceFit,
    protocol: Mapping[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    values = standardized(target, fit.location, fit.scale)
    bank, truth, starts = windows(
        values,
        fit.gains,
        fit.horizon,
        int(protocol["limits"]["window_stride"]),
    )
    method_metrics = _method_feature_metrics(bank, truth, fit)
    field_metrics = (
        _tactile_field_metrics(target, fit, starts)
        if target.descriptor.modality == "tactile_response"
        else _geometry_field_metrics(target, fit, starts)
    )
    for method in METHODS:
        method_metrics[method].update(field_metrics[method])
    probabilities = _probabilistic_metrics(bank, truth, fit, protocol, rng)
    return {
        "object_id": target.descriptor.object_id,
        "modality": target.descriptor.modality,
        "group_id": target.descriptor.group_id,
        "target_recording": target.descriptor.recording_id,
        "target_action": target.descriptor.action,
        "target_frame_count": int(len(target.features)),
        "target_frame_stride": int(target.frame_stride),
        "forecast_window_count": int(len(starts)),
        "source_recordings": fit.source_recordings,
        "source_recording_count": len(fit.source_recordings),
        "source_actions": [],
        "source_fit_frozen_before_target_open": True,
        "target_future_used_for_scoring_only": True,
        "target_fingerprint": target.fingerprint,
        "model_weights": [float(value) for value in fit.weights],
        "model_gains": [float(value) for value in fit.gains],
        "map_gain": float(fit.gains[fit.map_index]),
        "posterior_mean_gain": float(np.dot(fit.weights, fit.gains)),
        "guard_accepts": bool(fit.guard_accepts),
        "fallback_method": fit.fallback_method,
        "source_cv": fit.source_cv,
        "methods": method_metrics,
        "bayesian_uncertainty": probabilities,
    }


def _qualified_groups(
    groups: Mapping[tuple[str, str, str], Sequence[Descriptor]],
    protocol: Mapping[str, Any],
) -> list[tuple[tuple[str, str, str], list[Descriptor]]]:
    maximum_objects = int(protocol["limits"]["maximum_objects"])
    maximum_groups = int(protocol["limits"]["maximum_sensor_groups_per_object"])
    result = []
    object_counts: dict[tuple[str, str], int] = {}
    selected_objects: list[str] = []
    for key in sorted(groups):
        modality, object_id, _ = key
        descriptors = sorted(groups[key], key=lambda value: value.recording_id)
        if len(descriptors) < 3:
            continue
        if object_id not in selected_objects:
            if len(selected_objects) >= maximum_objects:
                continue
            selected_objects.append(object_id)
        count_key = (modality, object_id)
        count = object_counts.get(count_key, 0)
        if count >= maximum_groups:
            continue
        object_counts[count_key] = count + 1
        result.append((key, descriptors))
    return result


def _select_primary(rows: Sequence[Mapping[str, Any]]) -> str:
    objects_by_modality: dict[str, set[str]] = {}
    for row in rows:
        objects_by_modality.setdefault(str(row["modality"]), set()).add(
            str(row["object_id"])
        )
    geometry = sum(
        len(objects)
        for modality, objects in objects_by_modality.items()
        if modality.startswith("geometry_3d")
    )
    tactile = len(objects_by_modality.get("tactile_response", set()))
    if geometry >= 2:
        return "geometry_3d"
    if tactile:
        return "tactile_response"
    if geometry:
        return "geometry_3d"
    raise EvaluationError("no completed modality is available")


def _row_matches_primary(row: Mapping[str, Any], primary: str) -> bool:
    modality = str(row["modality"])
    return (
        modality.startswith("geometry_3d")
        if primary == "geometry_3d"
        else modality == primary
    )


def aggregate(
    rows: Sequence[Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    if not rows:
        raise EvaluationError("no real-data target group completed")
    primary = _select_primary(rows)
    selected = [row for row in rows if _row_matches_primary(row, primary)]
    object_ids = sorted({str(row["object_id"]) for row in selected})
    object_metrics: dict[str, dict[str, Any]] = {}
    for object_id in object_ids:
        subset = [row for row in selected if row["object_id"] == object_id]
        methods = {}
        for method in METHODS:
            keys = sorted(
                set.intersection(*(set(row["methods"][method]) for row in subset))
            )
            methods[method] = {
                key: float(np.mean([row["methods"][method][key] for row in subset]))
                for key in keys
            }
        uncertainty_keys = sorted(
            set.intersection(*(set(row["bayesian_uncertainty"]) for row in subset))
        )
        object_metrics[object_id] = {
            "group_count": len(subset),
            "guard_acceptance_fraction": float(
                np.mean([bool(row["guard_accepts"]) for row in subset])
            ),
            "methods": methods,
            "bayesian_uncertainty": {
                key: float(
                    np.mean([row["bayesian_uncertainty"][key] for row in subset])
                )
                for key in uncertainty_keys
            },
        }

    common_method_keys = sorted(
        set.intersection(
            *(
                set(object_metrics[object_id]["methods"][METHODS[0]])
                for object_id in object_ids
            )
        )
    )
    method_summary = {
        method: {
            key: float(
                np.mean(
                    [
                        object_metrics[object_id]["methods"][method][key]
                        for object_id in object_ids
                    ]
                )
            )
            for key in common_method_keys
        }
        for method in METHODS
    }
    uncertainty_keys = sorted(
        set.intersection(
            *(
                set(object_metrics[object_id]["bayesian_uncertainty"])
                for object_id in object_ids
            )
        )
    )
    uncertainty = {
        key: float(
            np.mean(
                [
                    object_metrics[object_id]["bayesian_uncertainty"][key]
                    for object_id in object_ids
                ]
            )
        )
        for key in uncertainty_keys
    }

    point_key = "centroid_rmse_mm" if primary == "geometry_3d" else "tactile_field_rmse"
    rng = np.random.default_rng(int(protocol["joint_uncertainty"]["random_seed"]))
    contrasts = {}
    for comparator in ("persistence", "last_residual", "map_motion"):
        differences = np.array(
            [
                object_metrics[object_id]["methods"]["guarded_bayesian_motion"][
                    point_key
                ]
                - object_metrics[object_id]["methods"][comparator][point_key]
                for object_id in object_ids
            ],
            dtype=np.float64,
        )
        contrasts[comparator] = {
            "guarded_minus_comparator": float(differences.mean()),
            "object_bootstrap_95_interval": bootstrap_interval(
                differences,
                repetitions=int(protocol["limits"]["bootstrap_repetitions"]),
                rng=rng,
            ),
            "object_wins": int(np.sum(differences < 0.0)),
            "object_ties": int(np.sum(differences == 0.0)),
            "object_losses": int(np.sum(differences > 0.0)),
            "worst_object_regret": float(np.max(differences)),
        }
    modality_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        entry = modality_counts.setdefault(
            str(row["modality"]), {"groups": 0, "objects": 0}
        )
        entry["groups"] += 1
    for modality in modality_counts:
        modality_counts[modality]["objects"] = len(
            {str(row["object_id"]) for row in rows if row["modality"] == modality}
        )
    return {
        "status": "complete",
        "primary_modality": primary,
        "primary_point_metric": point_key,
        "primary_object_count": len(object_ids),
        "completed_group_count": len(rows),
        "rejected_group_count": len(rejected),
        "modality_counts": modality_counts,
        "guard_acceptance_fraction": float(
            np.mean([bool(row["guard_accepts"]) for row in selected])
        ),
        "methods": method_summary,
        "bayesian_uncertainty": uncertainty,
        "contrasts": contrasts,
        "object_metrics": object_metrics,
        "aggregation": protocol["aggregation"],
    }


def make_report(result: Mapping[str, Any]) -> str:
    summary = result["summary"]
    primary = str(summary["primary_modality"])
    point_key = str(summary["primary_point_metric"])
    lines = [
        "# Deform360 real-data development evaluation",
        "",
        f"- Status: **{result['status']}**",
        f"- Dataset root: `{result['dataset_root']}`",
        f"- Git revision: `{result.get('github_sha')}`",
        f"- Runner: `{result.get('runner_name')}` with required label `gpuserver4090`",
        f"- Primary released carrier: **{primary}**",
        f"- Physical objects in the primary aggregate: **{summary['primary_object_count']}**",
        f"- Completed source/target groups: **{summary['completed_group_count']}**",
        f"- Rejected groups retained in the audit: **{summary['rejected_group_count']}**",
        "",
        "## Object-balanced point prediction",
        "",
        f"Primary metric: `{point_key}`. Lower is better.",
        "",
        "| Method | Value |",
        "|---|---:|",
    ]
    for method in METHODS:
        value = summary["methods"][method][point_key]
        lines.append(f"| `{method}` | {value:.8g} |")
    uncertainty = summary["bayesian_uncertainty"]
    lines.extend(
        [
            "",
            "## Joint uncertainty diagnostics",
            "",
            "| Diagnostic | Value |",
            "|---|---:|",
        ]
    )
    for key, value in uncertainty.items():
        lines.append(f"| `{key}` | {value:.8g} |")
    lines.extend(
        [
            "",
            "## Paired object-level contrasts",
            "",
            "Negative guarded-minus-comparator values favor the guarded Bayesian method.",
            "",
            "| Comparator | Mean difference | 95% object bootstrap | W/T/L | Worst regret |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for comparator, values in summary["contrasts"].items():
        interval = values["object_bootstrap_95_interval"]
        record = (
            f"{values['object_wins']}/{values['object_ties']}/{values['object_losses']}"
        )
        lines.append(
            f"| `{comparator}` | {values['guarded_minus_comparator']:.8g} | "
            f"[{interval[0]:.8g}, {interval[1]:.8g}] | {record} | "
            f"{values['worst_object_regret']:.8g} |"
        )
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "This is a retrospective public-real-data **development evaluation**. The",
            "target recording in every group was selected from path identity and metadata",
            "before target numeric loading; source model selection and covariance calibration",
            "were frozen first. Reserved confirmation objects were never numerically opened.",
            "Frames and sensor streams are averaged within physical objects and are not",
            "treated as independent inferential units.",
            "",
            "A tactile primary result validates real contact-response forecasting and joint",
            "uncertainty, but it does not by itself validate dense 4-D geometry or a strict",
            "counterfactual intervention claim. A geometry primary result uses only released",
            "processed carriers discovered beneath the frozen dataset root.",
            "",
            "No raw recording, point cloud, tactile tensor, or private trajectory is retained",
            "in this evidence bundle. `paper_claim_authorized` and",
            "`fresh_confirmation_authorized` remain false.",
            "",
        ]
    )
    return "\n".join(lines)


def run(protocol_path: Path, dataset_root: Path) -> dict[str, Any]:
    protocol = read_json(protocol_path)
    dataset_root = dataset_root.resolve(strict=True)
    validate_protocol(protocol, dataset_root)
    reserved = set(protocol["forbidden_reserved_object_ids"])
    if any((dataset_root / name).is_file() for name in reserved):
        raise EvaluationError("reserved roster unexpectedly resolves as files")

    geometry_groups = discover_geometry(dataset_root, protocol)
    tactile_groups = discover_tactile(dataset_root, protocol)
    carrier_inventory = {
        "geometry_descriptor_groups": len(geometry_groups),
        "tactile_descriptor_groups": len(tactile_groups),
        "geometry_descriptor_recordings": int(
            sum(len(value) for value in geometry_groups.values())
        ),
        "tactile_descriptor_recordings": int(
            sum(len(value) for value in tactile_groups.values())
        ),
        "numeric_payloads_opened_during_inventory": False,
        "reserved_object_payloads_opened": False,
    }

    qualified_geometry = _qualified_groups(geometry_groups, protocol)
    qualified_tactile = _qualified_groups(tactile_groups, protocol)
    # Run every qualified geometry group and retain tactile as an independent
    # fallback/supplement.  Primary-modality selection is object-count based.
    roster = qualified_geometry + qualified_tactile
    if not roster:
        raise EvaluationError(
            "no development object exposes at least three released recordings in a supported carrier"
        )

    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(protocol["joint_uncertainty"]["random_seed"]))
    maximum_sources = int(protocol["limits"]["maximum_source_recordings_per_group"])
    for key, descriptors in roster:
        modality, object_id, group_id = key
        if object_id in reserved:
            raise EvaluationError("reserved object entered the execution roster")
        descriptors = sorted(descriptors, key=lambda value: value.recording_id)
        target_descriptor = descriptors[-1]
        source_descriptors = descriptors[:-1][:maximum_sources]
        try:
            # Target payload is deliberately unopened until this source fit is complete.
            source_sequences = [
                load_sequence(descriptor, protocol, dataset_root)
                for descriptor in source_descriptors
            ]
            fit = fit_sources(source_sequences, protocol)
            source_actions = [descriptor.action for descriptor in source_descriptors]
            fit_fingerprint = object_digest(
                {
                    "gains": fit.gains.tolist(),
                    "weights": fit.weights.tolist(),
                    "location": fit.location.tolist(),
                    "scale": fit.scale.tolist(),
                    "covariance_diagonal": fit.covariance_diagonal.tolist(),
                    "covariance_factor": fit.covariance_factor.tolist(),
                    "covariance_multiplier": fit.covariance_multiplier,
                    "guard_accepts": fit.guard_accepts,
                    "fallback_method": fit.fallback_method,
                    "source_recordings": fit.source_recordings,
                }
            )
            target = load_sequence(target_descriptor, protocol, dataset_root)
            row = evaluate_target(target, fit, protocol, rng)
            row["source_actions"] = source_actions
            row["source_fit_id"] = fit_fingerprint
            row["target_opened_after_source_fit_id_created"] = True
            rows.append(row)
        except (EvaluationError, OSError, ValueError, np.linalg.LinAlgError) as error:
            rejected.append(
                {
                    "modality": modality,
                    "object_id": object_id,
                    "group_id": group_id,
                    "source_recordings": [
                        value.recording_id for value in source_descriptors
                    ],
                    "target_recording": target_descriptor.recording_id,
                    "reason": f"{type(error).__name__}: {error}",
                    "replacement_attempted": False,
                }
            )

    summary = aggregate(rows, rejected, protocol)
    result = {
        "schema": "bayesian-phystwin/deform360-gpuserver4090-real-evaluation-result-v1",
        "status": "complete",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": file_digest(protocol_path),
        "dataset_root": str(dataset_root),
        "dataset_mutated": False,
        "runner_name": os.environ.get("RUNNER_NAME"),
        "runner_labels_expected": protocol["runner"]["required_labels"],
        "github_repository": os.environ.get("GITHUB_REPOSITORY"),
        "github_sha": os.environ.get("GITHUB_SHA"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "carrier_inventory": carrier_inventory,
        "completed_groups": rows,
        "rejected_groups": rejected,
        "summary": summary,
        "evidence_class": protocol["evidence_class"],
        "fresh_confirmation_authorized": False,
        "paper_claim_authorized": False,
        "raw_data_uploaded": False,
        "strict_counterfactual_claim_authorized": False,
    }
    result["result_id"] = object_digest(result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args.protocol, args.dataset_root)
        write_json(args.output_json, result)
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(make_report(result), encoding="utf-8")
    except Exception as error:  # fail closed with a compact, non-payload receipt
        failure = {
            "schema": "bayesian-phystwin/deform360-gpuserver4090-real-evaluation-failure-v1",
            "status": "incomplete",
            "error_type": type(error).__name__,
            "error": str(error),
            "dataset_root": str(args.dataset_root),
            "github_sha": os.environ.get("GITHUB_SHA"),
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "paper_claim_authorized": False,
            "fresh_confirmation_authorized": False,
        }
        write_json(args.output_json, failure)
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(
            "# Incomplete Deform360 evaluation\n\n"
            f"`{type(error).__name__}: {error}`\n\n"
            "No scientific conclusion or paper claim is authorized.\n",
            encoding="utf-8",
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
