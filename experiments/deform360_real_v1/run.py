#!/usr/bin/env python3
"""Read-only, causal-prefix evaluation on runner-resident Deform360 carriers.

The evaluator deliberately supports several released carrier forms.  It prefers
3-D trajectories and cleaned point-cloud sequences.  When the mounted release
contains raw data only, it falls back to synchronized/raw tactile fields and
reports that boundary explicitly.  No result produced by this module authorizes
a fresh-confirmation or paper claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

NORMAL_90 = 1.6448536269514722
OBJECT_PATTERN = re.compile(r"^\d{3}-.+")
FRAME_PATTERN = re.compile(r"^(\d+)\.npz$")
TRAJECTORY_HINTS = (
    "positions_world_m",
    "points_world_m",
    "positions_m",
    "control_points",
    "particle_tracks",
    "particles",
    "tracks",
    "trajectory",
    "positions",
    "points",
)
VALID_HINTS = ("valid_mask", "track_valid", "visibility", "valid")
PATH_HINTS = ("track", "control", "particle", "trajectory", "position", "hull")
SKIP_NPZ_NAMES = {
    "robot.npz",
    "intrinsics.npz",
    "extrinsics.npz",
    "calibration.npz",
}


@dataclass(frozen=True, slots=True)
class Profile:
    max_cases: int
    max_frames: int
    max_points: int
    max_tactile_channels: int
    max_candidate_archives: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Profile:
        fields = {
            "max_cases",
            "max_frames",
            "max_points",
            "max_tactile_channels",
            "max_candidate_archives",
        }
        if set(value) != fields:
            raise ValueError("profile keys differ from the registered contract")
        numbers = {name: int(value[name]) for name in fields}
        if any(number < 1 for number in numbers.values()):
            raise ValueError("profile limits must be positive")
        return cls(**numbers)


@dataclass(frozen=True, slots=True)
class Carrier:
    kind: str
    object_id: str
    path: Path
    members: tuple[Path, ...] = ()

    @property
    def identity_path(self) -> Path:
        return self.path


@dataclass(frozen=True, slots=True)
class SequenceData:
    values: np.ndarray
    valid: np.ndarray
    representation: str
    unit: str
    primary_metric: str
    metadata: Mapping[str, Any]
    clouds: tuple[np.ndarray, ...] | None = None


@dataclass(frozen=True, slots=True)
class PredictionStep:
    frame: int
    persistence: np.ndarray
    last_residual: np.ndarray
    bayesian: np.ndarray
    diagonal: np.ndarray
    factors: np.ndarray
    weights: np.ndarray


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def content_id(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("result_sha256", None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_protocol(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "bayesian-phystwin/deform360-real-evaluation-protocol-v1":
        raise ValueError("unexpected protocol schema")
    if value.get("schema_version") != 1:
        raise ValueError("unexpected protocol schema version")
    boundary = value.get("information_boundary")
    if not isinstance(boundary, dict):
        raise ValueError("protocol information boundary is missing")
    required_false = (
        "reserved_objects_opened",
        "method_parameters_fit_to_evaluated_future",
        "raw_payload_uploaded",
        "fresh_confirmation_authorized",
        "paper_claim_authorized",
    )
    if any(boundary.get(key) is not False for key in required_false):
        raise ValueError("protocol attempts to widen a closed information boundary")
    if boundary.get("rolling_prefix_only_prediction") is not True:
        raise ValueError("causal-prefix prediction must remain enabled")
    reserved = tuple(map(str, value.get("reserved_objects", ())))
    development = tuple(map(str, value.get("development_objects", ())))
    if not development or len(set(development)) != len(development):
        raise ValueError("development object roster must be nonempty and unique")
    if set(reserved) & set(development):
        raise ValueError("development and reserved object rosters overlap")
    return value


def _object_id(path: Path, root: Path) -> str:
    try:
        parts = path.resolve().relative_to(root.resolve()).parts
    except ValueError:
        parts = path.parts
    for part in parts:
        if OBJECT_PATTERN.match(part):
            return part
    return "unknown"


def _is_regular_file(path: Path) -> bool:
    try:
        return path.is_file() and not path.is_symlink()
    except OSError:
        return False


def _ordered_objects(protocol: Mapping[str, Any], root: Path) -> list[str]:
    registered = list(map(str, protocol["development_objects"]))
    reserved = set(map(str, protocol["reserved_objects"]))
    raw = root / "raw-repository" / "raw"
    present = []
    if raw.is_dir():
        present = sorted(
            path.name
            for path in raw.iterdir()
            if path.is_dir() and OBJECT_PATTERN.match(path.name)
        )
    return [name for name in registered if name in present] + [
        name for name in present if name not in registered and name not in reserved
    ]


def discover_carriers(
    root: Path,
    protocol: Mapping[str, Any],
    profile: Profile,
) -> tuple[list[Carrier], dict[str, Any]]:
    """Select carriers from names only; values and achieved scores are unopened."""

    reserved = set(map(str, protocol["reserved_objects"]))
    development_rank = {
        object_id: index
        for index, object_id in enumerate(map(str, protocol["development_objects"]))
    }
    pcd: list[Carrier] = []
    fixed: list[Carrier] = []
    tactile: list[Carrier] = []
    named_npz = 0
    named_pcd_dirs = 0
    named_tactile = 0

    for directory, names, files in os.walk(root):
        names[:] = sorted(
            name for name in names if name not in {".git", "__pycache__", "node_modules"}
        )
        base = Path(directory)
        object_id = _object_id(base, root)
        if object_id in reserved:
            names[:] = []
            continue
        if base.name == "pcd_clean":
            frame_paths = tuple(
                base / name
                for name in sorted(files)
                if FRAME_PATTERN.fullmatch(name) and _is_regular_file(base / name)
            )
            if len(frame_paths) >= int(protocol["model"]["minimum_prefix_steps"]) + 2:
                pcd.append(Carrier("pcd_clean", object_id, base, frame_paths))
                named_pcd_dirs += 1
            names[:] = []
            continue
        tactile_directory = "tactile" in base.name.lower()
        for name in sorted(files):
            path = base / name
            if not _is_regular_file(path):
                continue
            lowered = name.lower()
            if lowered.endswith(".npz"):
                named_npz += 1
                if (
                    lowered not in SKIP_NPZ_NAMES
                    and base.name != "pcd_clean"
                    and any(hint in path.as_posix().lower() for hint in PATH_HINTS)
                ):
                    fixed.append(Carrier("trajectory_npz", object_id, path))
            elif (
                tactile_directory
                and lowered.endswith(".npy")
                and not lowered.startswith("median_")
            ):
                named_tactile += 1
                tactile.append(Carrier("tactile", object_id, path))

    def rank(carrier: Carrier) -> tuple[int, int, str]:
        return (
            0 if carrier.object_id in development_rank else 1,
            development_rank.get(carrier.object_id, 10**9),
            carrier.path.relative_to(root).as_posix(),
        )

    pcd.sort(key=rank)
    fixed.sort(key=rank)
    tactile.sort(key=rank)
    fixed = fixed[: profile.max_candidate_archives]

    # Preserve representation diversity without selecting from numerical outcomes.
    selected: list[Carrier] = []
    per_object: dict[tuple[str, str], int] = {}
    for pool in (pcd, fixed, tactile):
        for carrier in pool:
            key = (carrier.kind, carrier.object_id)
            limit = 2 if carrier.kind == "tactile" else 3
            if per_object.get(key, 0) >= limit:
                continue
            selected.append(carrier)
            per_object[key] = per_object.get(key, 0) + 1
            if len(selected) >= profile.max_cases:
                break
        if len(selected) >= profile.max_cases:
            break

    inventory = {
        "selection_uses_payload_values": False,
        "selection_uses_achieved_scores": False,
        "reserved_objects": sorted(reserved),
        "named_npz_files": named_npz,
        "named_pcd_clean_directories": named_pcd_dirs,
        "named_tactile_data_files": named_tactile,
        "candidate_counts": {
            "pcd_clean": len(pcd),
            "trajectory_npz": len(fixed),
            "tactile": len(tactile),
        },
        "selected": [
            {
                "kind": carrier.kind,
                "object_id": carrier.object_id,
                "path": carrier.path.relative_to(root).as_posix(),
                "member_count": len(carrier.members),
            }
            for carrier in selected
        ],
    }
    return selected, inventory


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path, *, full_hash_limit: int = 64 * 1024 * 1024) -> dict[str, Any]:
    stat = path.stat()
    result: dict[str, Any] = {
        "path": path.name,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if stat.st_size <= full_hash_limit:
        result["digest_mode"] = "full_sha256"
        result["sha256"] = _stream_sha256(path)
        return result
    sample = hashlib.sha256()
    sample.update(str(stat.st_size).encode("ascii"))
    block_size = 1024 * 1024
    with path.open("rb") as handle:
        for offset in (0, max(0, stat.st_size // 2 - block_size // 2), max(0, stat.st_size - block_size)):
            handle.seek(offset)
            block = handle.read(block_size)
            sample.update(offset.to_bytes(8, "little", signed=False))
            sample.update(block)
    result["digest_mode"] = "size_plus_three_1MiB_samples_sha256"
    result["sampled_sha256"] = sample.hexdigest()
    return result


def _indices(count: int, limit: int) -> np.ndarray:
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, limit, dtype=np.int64)


def _scale_to_meters(key: str, points: np.ndarray) -> tuple[float, str]:
    lowered = key.lower()
    if lowered.endswith("_mm") or "millimet" in lowered:
        return 1e-3, "declared_mm"
    if lowered.endswith("_m") or "world_m" in lowered:
        return 1.0, "declared_m"
    norms = np.linalg.norm(points.reshape(-1, 3), axis=1)
    finite = norms[np.isfinite(norms)]
    median = float(np.median(finite)) if len(finite) else 0.0
    if median > 20.0:
        return 1e-3, "heuristic_mm"
    return 1.0, "heuristic_m"


def _trajectory_candidate(stored: Any) -> tuple[str, np.ndarray] | None:
    candidates: list[tuple[int, str, np.ndarray]] = []
    for key in stored.files:
        lowered = key.lower()
        matches = [index for index, hint in enumerate(TRAJECTORY_HINTS) if hint in lowered]
        if not matches:
            continue
        value = np.asarray(stored[key])
        if (
            value.dtype.kind in "iuf"
            and value.ndim == 3
            and value.shape[-1] == 3
            and value.shape[0] >= 4
        ):
            candidates.append((min(matches), key, value))
    if not candidates:
        return None
    _, key, value = min(candidates, key=lambda item: (item[0], item[1]))
    return key, value


def _validity(stored: Any, shape: tuple[int, int]) -> np.ndarray:
    for key in stored.files:
        if not any(hint in key.lower() for hint in VALID_HINTS):
            continue
        value = np.asarray(stored[key])
        if value.shape == shape:
            return np.asarray(value, dtype=bool)
    return np.ones(shape, dtype=bool)


def load_trajectory_npz(carrier: Carrier, profile: Profile, root: Path) -> SequenceData:
    with np.load(carrier.path, allow_pickle=False) as stored:
        candidate = _trajectory_candidate(stored)
        if candidate is None:
            raise ValueError("archive contains no registered (T,N,3) trajectory")
        key, raw = candidate
        frame_indices = _indices(raw.shape[0], profile.max_frames)
        point_indices = _indices(raw.shape[1], profile.max_points)
        values = np.asarray(raw[np.ix_(frame_indices, point_indices, np.arange(3))], dtype=np.float64)
        valid = _validity(stored, raw.shape[:2])[np.ix_(frame_indices, point_indices)]
    scale, unit_source = _scale_to_meters(key, values)
    values *= scale
    return SequenceData(
        values=values,
        valid=valid & np.all(np.isfinite(values), axis=2),
        representation="fixed_identity_3d",
        unit="m",
        primary_metric="point_rmse_mm",
        metadata={
            "relative_path": carrier.path.relative_to(root).as_posix(),
            "array_key": key,
            "source_frame_count": int(raw.shape[0]),
            "source_point_count": int(raw.shape[1]),
            "frame_indices": frame_indices.tolist(),
            "point_indices": point_indices.tolist(),
            "unit_source": unit_source,
            "files": [file_identity(carrier.path)],
        },
    )


def _load_cloud(path: Path, max_points: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as stored:
        if "pts" in stored.files:
            value = np.asarray(stored["pts"], dtype=np.float64)
        else:
            candidate = next(
                (
                    np.asarray(stored[key], dtype=np.float64)
                    for key in stored.files
                    if np.asarray(stored[key]).ndim == 2
                    and np.asarray(stored[key]).shape[1] == 3
                    and np.asarray(stored[key]).dtype.kind in "iuf"
                ),
                None,
            )
            if candidate is None:
                raise ValueError("point-cloud frame has no numeric (N,3) array")
            value = candidate
    value = value[np.all(np.isfinite(value), axis=1)]
    if len(value) < 4:
        raise ValueError("point-cloud frame has fewer than four finite points")
    return value[_indices(len(value), max_points)]


def load_pcd_sequence(carrier: Carrier, profile: Profile, root: Path) -> SequenceData:
    frame_paths = carrier.members[: profile.max_frames]
    clouds = tuple(_load_cloud(path, profile.max_points) for path in frame_paths)
    centroids = np.asarray([np.mean(cloud, axis=0) for cloud in clouds], dtype=np.float64)
    identities = [file_identity(path) for path in frame_paths[:8]]
    return SequenceData(
        values=centroids[:, None, :],
        valid=np.ones((len(centroids), 1), dtype=bool),
        representation="pcd_clean_centroid_3d",
        unit="m",
        primary_metric="centroid_error_mm",
        metadata={
            "relative_path": carrier.path.relative_to(root).as_posix(),
            "source_frame_count": len(carrier.members),
            "evaluated_frame_count": len(frame_paths),
            "points_per_frame": [len(cloud) for cloud in clouds],
            "files": identities,
            "identity_scope": "first eight evaluated frames; remaining frame names retained in carrier inventory",
        },
        clouds=clouds,
    )


def _headered_npy(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(6) == b"\x93NUMPY"


def _load_tactile_array(path: Path) -> np.ndarray:
    if _headered_npy(path):
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if value.ndim != 3 or value.shape[1:] != (16, 32):
            raise ValueError(f"headered tactile shape is unsupported: {value.shape}")
        if value.dtype.kind not in "iuf":
            raise ValueError("tactile array must be real numeric")
        return value
    frame_bytes = 16 * 32 * np.dtype(np.float32).itemsize
    size = path.stat().st_size
    if size < 4 * frame_bytes or size % frame_bytes:
        raise ValueError("headerless tactile bytes do not form (T,16,32) float32")
    return np.memmap(path, dtype=np.float32, mode="r", shape=(size // frame_bytes, 16, 32))


def load_tactile(carrier: Carrier, profile: Profile, root: Path) -> SequenceData:
    raw = _load_tactile_array(carrier.path)
    if raw.shape[0] < 12:
        raise ValueError("tactile stream is too short")
    channel_indices = _indices(16 * 32, profile.max_tactile_channels)
    probe_frames = _indices(raw.shape[0], min(raw.shape[0], 512))
    probe = np.asarray(raw[probe_frames], dtype=np.float64).reshape(len(probe_frames), -1)[:, channel_indices]
    baseline = np.median(probe[: min(8, len(probe))], axis=0)
    activity = np.mean(np.abs(probe - baseline), axis=1)
    median = float(np.median(activity))
    mad = float(np.median(np.abs(activity - median)))
    threshold = median + 5.0 * max(mad, np.finfo(np.float64).eps)
    active = np.flatnonzero(activity > threshold)
    start_fraction = 0.0 if not len(active) else float(active[0]) / max(len(probe) - 1, 1)
    start = int(round(start_fraction * max(raw.shape[0] - 1, 0)))
    start = max(0, start - 5)
    end = min(raw.shape[0], start + profile.max_frames)
    if end - start < 12:
        start = max(0, raw.shape[0] - profile.max_frames)
        end = raw.shape[0]
    values = np.asarray(raw[start:end], dtype=np.float64).reshape(end - start, -1)[:, channel_indices]
    local_baseline = np.median(values[: min(8, len(values))], axis=0)
    values -= local_baseline
    scale = float(np.quantile(np.abs(values), 0.99))
    if not np.isfinite(scale) or scale <= np.finfo(np.float64).eps:
        scale = 1.0
    values /= scale
    values = values[:, :, None]
    return SequenceData(
        values=values,
        valid=np.all(np.isfinite(values), axis=2),
        representation="raw_tactile_field",
        unit="normalized_tactile",
        primary_metric="field_rmse",
        metadata={
            "relative_path": carrier.path.relative_to(root).as_posix(),
            "source_frame_count": int(raw.shape[0]),
            "evaluated_range_half_open": [start, end],
            "selected_channel_count": len(channel_indices),
            "channel_indices": channel_indices.tolist(),
            "normalization": "subtract first-window channel median and divide by window 99th absolute percentile",
            "files": [file_identity(carrier.path)],
        },
    )


def load_carrier(carrier: Carrier, profile: Profile, root: Path) -> SequenceData:
    if carrier.kind == "trajectory_npz":
        return load_trajectory_npz(carrier, profile, root)
    if carrier.kind == "pcd_clean":
        return load_pcd_sequence(carrier, profile, root)
    if carrier.kind == "tactile":
        return load_tactile(carrier, profile, root)
    raise ValueError(f"unsupported carrier kind: {carrier.kind}")


def _forward_fill(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.ndim != 3 or valid.shape != values.shape[:2]:
        raise ValueError("sequence values/valid shapes disagree")
    keep = valid[0] & np.all(np.isfinite(values[0]), axis=1)
    if not np.any(keep):
        raise ValueError("no point/channel is finite at the first evaluated frame")
    result = np.asarray(values[:, keep], dtype=np.float64).copy()
    validity = np.asarray(valid[:, keep], dtype=bool).copy()
    for frame in range(1, len(result)):
        missing = ~validity[frame] | ~np.all(np.isfinite(result[frame]), axis=1)
        result[frame, missing] = result[frame - 1, missing]
    if not np.all(np.isfinite(result)):
        raise ValueError("causal forward fill did not produce finite values")
    return result, validity


def _candidate_increment(values: np.ndarray, frame: int, lag: int) -> np.ndarray:
    return (values[frame] - values[frame - lag]) / float(lag)


def _loss_for_lag(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    frame: int,
    lag: int,
) -> float:
    losses: list[float] = []
    for current in range(lag, frame):
        mask = valid[current - lag] & valid[current] & valid[current + 1]
        if not np.any(mask):
            continue
        estimate = _candidate_increment(values, current, lag)
        target = values[current + 1] - values[current]
        error = estimate[mask] - target[mask]
        losses.append(float(np.mean(np.square(error))))
    return float(np.mean(losses)) if losses else math.inf


def _softmax_negative(losses: np.ndarray, floor_fraction: float) -> np.ndarray:
    finite = np.isfinite(losses)
    if not np.any(finite):
        raise ValueError("no finite retrospective model loss")
    typical = float(np.median(losses[finite]))
    floor = max(abs(typical) * floor_fraction, np.finfo(np.float64).eps)
    temperature = max(typical, floor)
    logits = np.full_like(losses, -np.inf)
    logits[finite] = -losses[finite] / temperature
    logits -= np.max(logits[finite])
    weights = np.zeros_like(losses)
    weights[finite] = np.exp(logits[finite])
    weights /= np.sum(weights)
    return weights


def _residual_diagonal(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    frame: int,
    lags: Sequence[int],
    weights: np.ndarray,
    floor_fraction: float,
) -> np.ndarray:
    residuals: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for current in range(max(lags), frame):
        available = [index for index, lag in enumerate(lags) if lag <= current]
        local_weights = weights[available]
        local_weights = local_weights / np.sum(local_weights)
        candidates = np.stack(
            [_candidate_increment(values, current, lags[index]) for index in available]
        )
        estimate = np.einsum("k,knd->nd", local_weights, candidates)
        target = values[current + 1] - values[current]
        mask = valid[current + 1] & valid[current]
        residuals.append(target - estimate)
        masks.append(mask)
    delta = np.diff(values[: frame + 1], axis=0)
    scale2 = float(np.mean(np.square(delta))) if delta.size else 0.0
    floor = max(scale2 * floor_fraction, np.finfo(np.float64).eps)
    shape = values.shape[1:]
    if not residuals:
        return np.full(shape, floor, dtype=np.float64)
    stack = np.stack(residuals)
    mask_stack = np.stack(masks)
    expanded = np.repeat(mask_stack[:, :, None], shape[1], axis=2)
    squared = np.where(expanded, np.square(stack), np.nan)
    with np.errstate(invalid="ignore"):
        diagonal = np.nanmean(squared, axis=0)
    diagonal = np.where(np.isfinite(diagonal), diagonal, floor)
    return np.maximum(diagonal, floor)


def prediction_for_step(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    frame: int,
    lags: Sequence[int],
    floor_fraction: float,
    temperature_floor_fraction: float,
) -> PredictionStep:
    """Produce one prediction using bytes no later than ``frame``."""

    available_lags = tuple(lag for lag in lags if lag <= frame)
    if not available_lags:
        raise ValueError("no registered lag is available")
    losses = np.asarray(
        [
            _loss_for_lag(values, valid, frame=frame, lag=lag)
            for lag in available_lags
        ],
        dtype=np.float64,
    )
    weights = _softmax_negative(losses, temperature_floor_fraction)
    candidates = np.stack(
        [_candidate_increment(values, frame, lag) for lag in available_lags]
    )
    mean_increment = np.einsum("k,knd->nd", weights, candidates)
    diagonal = _residual_diagonal(
        values,
        valid,
        frame=frame,
        lags=available_lags,
        weights=weights,
        floor_fraction=floor_fraction,
    )
    deviation = candidates - mean_increment[None]
    factors = (
        deviation.reshape(len(available_lags), -1).T * np.sqrt(weights)[None, :]
    )
    return PredictionStep(
        frame=frame,
        persistence=values[frame].copy(),
        last_residual=values[frame] + values[frame] - values[frame - 1],
        bayesian=values[frame] + mean_increment,
        diagonal=diagonal.reshape(-1),
        factors=factors,
        weights=weights,
    )


def _low_rank_metrics(
    error: np.ndarray,
    diagonal: np.ndarray,
    factors: np.ndarray,
) -> dict[str, float]:
    if error.ndim != 1 or diagonal.shape != error.shape or factors.shape[0] != len(error):
        raise ValueError("joint covariance dimensions disagree")
    diagonal = np.maximum(diagonal, np.finfo(np.float64).eps)
    inverse = 1.0 / diagonal
    weighted_factors = inverse[:, None] * factors
    core = np.eye(factors.shape[1], dtype=np.float64) + factors.T @ weighted_factors
    projected = factors.T @ (inverse * error)
    solved = np.linalg.solve(core, projected)
    quadratic = float(error @ (inverse * error) - projected @ solved)
    sign, core_logdet = np.linalg.slogdet(core)
    if sign <= 0 or quadratic < -1e-8:
        raise ValueError("diagonal-plus-low-rank covariance is not positive definite")
    quadratic = max(quadratic, 0.0)
    logdet = float(np.sum(np.log(diagonal)) + core_logdet)
    dimension = len(error)
    marginal_variance = diagonal + np.sum(np.square(factors), axis=1)
    radius = NORMAL_90 * np.sqrt(marginal_variance)
    return {
        "nll_per_dimension": 0.5
        * (math.log(2.0 * math.pi) + logdet / dimension + quadratic / dimension),
        "joint_nees_normalized": quadratic / dimension,
        "marginal_90_coverage": float(np.mean(np.abs(error) <= radius)),
        "mean_full_90_width": float(np.mean(2.0 * radius)),
    }


def _point_rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.square(error), axis=1))))


def _chamfer_rmse(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    left = left[_indices(len(left), 256)]
    right = right[_indices(len(right), 256)]

    def directed(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        minimum = np.full(len(source), np.inf, dtype=np.float64)
        for start in range(0, len(target), 64):
            block = target[start : start + 64]
            squared = np.sum(
                np.square(source[:, None, :] - block[None, :, :]), axis=2
            )
            minimum = np.minimum(minimum, np.min(squared, axis=1))
        return minimum

    value = 0.5 * (
        float(np.mean(directed(left, right)))
        + float(np.mean(directed(right, left)))
    )
    return float(np.sqrt(max(value, 0.0)))


def evaluate_sequence(
    data: SequenceData,
    model: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[PredictionStep]]:
    values, valid = _forward_fill(data.values, data.valid)
    lags = tuple(int(value) for value in model["velocity_lags"])
    minimum_prefix = int(model["minimum_prefix_steps"])
    if len(values) < minimum_prefix + 2:
        raise ValueError("carrier is shorter than the registered causal prefix")
    if min(lags) < 1 or tuple(sorted(set(lags))) != lags:
        raise ValueError("velocity lag grid must be sorted and unique")
    steps: list[dict[str, Any]] = []
    predictions: list[PredictionStep] = []
    for frame in range(minimum_prefix, len(values) - 1):
        target_mask = valid[frame] & valid[frame + 1]
        if not np.any(target_mask):
            continue
        prediction = prediction_for_step(
            values,
            valid,
            frame=frame,
            lags=lags,
            floor_fraction=float(model["variance_floor_fraction"]),
            temperature_floor_fraction=float(
                model["gibbs_temperature_floor_fraction"]
            ),
        )
        target = values[frame + 1]
        selected = np.repeat(target_mask[:, None], values.shape[2], axis=1)
        method_errors = {
            "persistence": prediction.persistence - target,
            "last_residual": prediction.last_residual - target,
            "bayesian": prediction.bayesian - target,
        }
        row: dict[str, Any] = {
            "frame": frame,
            "valid_point_count": int(np.sum(target_mask)),
            "effective_model_count": float(
                np.exp(
                    -np.sum(
                        prediction.weights
                        * np.log(np.maximum(prediction.weights, 1e-300))
                    )
                )
            ),
        }
        for name, error in method_errors.items():
            row[f"{name}_rmse"] = _point_rmse(error[target_mask])
        bayesian_error = method_errors["bayesian"].reshape(-1)[selected.reshape(-1)]
        covariance = _low_rank_metrics(
            bayesian_error,
            prediction.diagonal[selected.reshape(-1)],
            prediction.factors[selected.reshape(-1)],
        )
        row.update({f"bayesian_{key}": value for key, value in covariance.items()})
        steps.append(row)
        predictions.append(prediction)
    if not steps:
        raise ValueError("carrier has no evaluable future step")

    def mean(key: str) -> float:
        return float(np.mean([float(row[key]) for row in steps]))

    factor = 1000.0 if data.unit == "m" else 1.0
    suffix = "_mm" if data.unit == "m" else ""
    metrics: dict[str, Any] = {
        f"persistence_rmse{suffix}": factor * mean("persistence_rmse"),
        f"last_residual_rmse{suffix}": factor * mean("last_residual_rmse"),
        f"bayesian_rmse{suffix}": factor * mean("bayesian_rmse"),
        "bayesian_nll_per_dimension": mean("bayesian_nll_per_dimension"),
        "bayesian_joint_nees_normalized": mean("bayesian_joint_nees_normalized"),
        "bayesian_marginal_90_coverage": mean("bayesian_marginal_90_coverage"),
        f"bayesian_mean_full_90_width{suffix}": factor
        * mean("bayesian_mean_full_90_width"),
        "mean_effective_model_count": mean("effective_model_count"),
    }
    return metrics, steps, predictions


def _add_pcd_chamfer(
    data: SequenceData,
    steps: list[dict[str, Any]],
    predictions: list[PredictionStep],
    minimum_prefix: int,
) -> dict[str, float]:
    if data.clouds is None:
        return {}
    values = data.values
    rows: list[dict[str, float]] = []
    for row, prediction in zip(steps, predictions, strict=True):
        frame = int(row["frame"])
        current = data.clouds[frame]
        target = data.clouds[frame + 1]
        translations = {
            "persistence": np.zeros(3),
            "last_residual": (values[frame, 0] - values[frame - 1, 0]),
            "bayesian": prediction.bayesian[0] - values[frame, 0],
        }
        chamfer = {
            name: 1000.0 * _chamfer_rmse(current + shift, target)
            for name, shift in translations.items()
        }
        row.update({f"{name}_chamfer_mm": value for name, value in chamfer.items()})
        rows.append(chamfer)
    return {
        f"{name}_chamfer_mm": float(np.mean([row[name] for row in rows]))
        for name in ("persistence", "last_residual", "bayesian")
    }


def _case_primary_values(case: Mapping[str, Any]) -> tuple[float, float, float]:
    metric = str(case["primary_metric"])
    values = case["metrics"]
    if metric in {"point_rmse_mm", "centroid_error_mm"}:
        return (
            float(values["persistence_rmse_mm"]),
            float(values["last_residual_rmse_mm"]),
            float(values["bayesian_rmse_mm"]),
        )
    return (
        float(values["persistence_rmse"]),
        float(values["last_residual_rmse"]),
        float(values["bayesian_rmse"]),
    )


def _mean(values: Iterable[float]) -> float | None:
    array = np.asarray(tuple(values), dtype=np.float64)
    return None if len(array) == 0 else float(np.mean(array))


def _bootstrap_interval(values: np.ndarray, repetitions: int, seed: int) -> list[float] | None:
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    sampled = np.mean(values[indices], axis=1)
    return np.quantile(sampled, [0.025, 0.975]).astype(float).tolist()


def aggregate(
    cases: Sequence[Mapping[str, Any]],
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for representation in sorted({str(case["representation"]) for case in cases}):
        selected = [case for case in cases if case["representation"] == representation]
        primary = np.asarray([_case_primary_values(case) for case in selected])
        differences = primary[:, 2] - np.minimum(primary[:, 0], primary[:, 1])
        object_rows: dict[str, list[np.ndarray]] = {}
        for case, values in zip(selected, primary, strict=True):
            object_rows.setdefault(str(case["object_id"]), []).append(values)
        object_primary = np.asarray(
            [np.mean(rows, axis=0) for _, rows in sorted(object_rows.items())]
        )
        object_differences = object_primary[:, 2] - np.minimum(
            object_primary[:, 0], object_primary[:, 1]
        )
        result[representation] = {
            "case_count": len(selected),
            "object_count": len(object_rows),
            "primary_metric": selected[0]["primary_metric"],
            "mean_primary_error": {
                "persistence": float(np.mean(primary[:, 0])),
                "last_residual": float(np.mean(primary[:, 1])),
                "bayesian": float(np.mean(primary[:, 2])),
            },
            "bayesian_minus_best_baseline": float(np.mean(differences)),
            "case_bootstrap_95_interval": _bootstrap_interval(
                differences,
                int(analysis["bootstrap_repetitions"]),
                int(analysis["bootstrap_seed"]),
            ),
            "object_balanced_bayesian_minus_best_baseline": float(
                np.mean(object_differences)
            ),
            "object_bootstrap_95_interval": _bootstrap_interval(
                object_differences,
                int(analysis["bootstrap_repetitions"]),
                int(analysis["bootstrap_seed"]) + 1,
            ),
            "bayesian_calibration": {
                "joint_nees_normalized": _mean(
                    float(case["metrics"]["bayesian_joint_nees_normalized"])
                    for case in selected
                ),
                "marginal_90_coverage": _mean(
                    float(case["metrics"]["bayesian_marginal_90_coverage"])
                    for case in selected
                ),
                "nll_per_dimension": _mean(
                    float(case["metrics"]["bayesian_nll_per_dimension"])
                    for case in selected
                ),
                "mean_effective_model_count": _mean(
                    float(case["metrics"]["mean_effective_model_count"])
                    for case in selected
                ),
            },
        }
    return result


def save_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def report_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Deform360 real-data evaluation",
        "",
        f"Protocol: `{result['protocol_id']}`",
        f"Revision: `{result.get('revision')}`",
        f"Dataset root: `{result['data_root']}`",
        "",
        "This is a retrospective, non-confirmatory real-data diagnostic. Reserved",
        "objects were excluded, rolling predictions used causal prefixes, and raw",
        "dataset payloads are not included in the artifact.",
        "",
        "## Results",
        "",
        "| Representation | Cases | Objects | Metric | Persistence | Last residual | Bayesian | Bayesian − best baseline | Joint nNEES | Marginal 90% coverage |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, summary in sorted(result["summary"].items()):
        errors = summary["mean_primary_error"]
        calibration = summary["bayesian_calibration"]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(summary["case_count"]),
                    str(summary["object_count"]),
                    str(summary["primary_metric"]),
                    f"{errors['persistence']:.6g}",
                    f"{errors['last_residual']:.6g}",
                    f"{errors['bayesian']:.6g}",
                    f"{summary['bayesian_minus_best_baseline']:.6g}",
                    f"{calibration['joint_nees_normalized']:.4f}",
                    f"{calibration['marginal_90_coverage']:.4f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A 3-D carrier result evaluates rolling real-geometry dynamics. A raw",
            "tactile-field result evaluates measured interaction-field dynamics only",
            "and is not a 4-D geometry or causal-intervention validation. This run",
            "does not establish official Deform360 benchmark parity, fresh confirmation,",
            "state of the art, calibrated deployment risk, or a paper claim.",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    *,
    data_root: Path,
    protocol_path: Path,
    output_dir: Path,
    profile_name: str,
    revision: str | None,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    expected_root = Path(protocol["dataset_root"]).resolve()
    root = data_root.expanduser().resolve(strict=True)
    if root != expected_root:
        raise ValueError(f"dataset root changed: {root} != {expected_root}")
    if profile_name not in protocol["profiles"]:
        raise ValueError("unknown evaluation profile")
    profile = Profile.from_mapping(protocol["profiles"][profile_name])
    output = output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("output and dataset roots must be disjoint")
    output.mkdir(parents=True)
    shutil.copy2(protocol_path, output / "protocol.json")

    carriers, inventory = discover_carriers(root, protocol, profile)
    write_json(output / "carrier_inventory.json", inventory)
    cases: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for index, carrier in enumerate(carriers):
        try:
            data = load_carrier(carrier, profile, root)
            metrics, steps, predictions = evaluate_sequence(data, protocol["model"])
            metrics.update(
                _add_pcd_chamfer(
                    data,
                    steps,
                    predictions,
                    int(protocol["model"]["minimum_prefix_steps"]),
                )
            )
            case = {
                "case_id": f"case-{index:03d}",
                "kind": carrier.kind,
                "object_id": carrier.object_id,
                "representation": data.representation,
                "unit": data.unit,
                "primary_metric": data.primary_metric,
                "frame_count": int(data.values.shape[0]),
                "point_or_channel_count": int(data.values.shape[1]),
                "step_count": len(steps),
                "metrics": metrics,
                "provenance": dict(data.metadata),
            }
            cases.append(case)
            case_rows.append(
                {
                    "case_id": case["case_id"],
                    "kind": case["kind"],
                    "object_id": case["object_id"],
                    "representation": case["representation"],
                    "unit": case["unit"],
                    **metrics,
                }
            )
            for row in steps:
                step_rows.append(
                    {
                        "case_id": case["case_id"],
                        "object_id": case["object_id"],
                        "representation": case["representation"],
                        **row,
                    }
                )
        except (OSError, ValueError, TypeError, KeyError, np.linalg.LinAlgError) as error:
            failures.append(
                {
                    "kind": carrier.kind,
                    "object_id": carrier.object_id,
                    "path": carrier.path.relative_to(root).as_posix(),
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    if not cases:
        write_json(output / "failures.json", failures)
        raise RuntimeError(
            "no registered Deform360 carrier could be evaluated; "
            "carrier_inventory.json and failures.json retain the diagnosis"
        )

    result: dict[str, Any] = {
        "schema": "bayesian-phystwin/deform360-real-evaluation-result-v1",
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "profile": profile_name,
        "revision": revision,
        "data_root": str(root),
        "official_processing_revision": protocol["official_processing_revision"],
        "information_boundary": protocol["information_boundary"],
        "selection": {
            "selected_count": len(carriers),
            "evaluated_count": len(cases),
            "failure_count": len(failures),
            "reserved_object_overlap": sorted(
                {case["object_id"] for case in cases}
                & set(map(str, protocol["reserved_objects"]))
            ),
        },
        "method": {
            "baselines": ["persistence", "last_residual"],
            "candidate": "causal-prefix Gibbs mixture over finite velocity lags",
            "uncertainty": "diagonal residual covariance plus low-rank between-model spread",
            "velocity_lags": protocol["model"]["velocity_lags"],
        },
        "summary": aggregate(cases, protocol["analysis"]),
        "cases": cases,
        "failures": failures,
        "claim_authorized": False,
        "fresh_confirmation_authorized": False,
    }
    if result["selection"]["reserved_object_overlap"]:
        raise RuntimeError("reserved object reached the evaluated case roster")
    result["result_sha256"] = content_id(result)
    write_json(output / "result.json", result)
    save_csv(output / "case_metrics.csv", case_rows)
    save_csv(output / "step_metrics.csv", step_rows)
    (output / "report.md").write_text(report_markdown(result), encoding="utf-8")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).with_name("protocol.json"),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--profile", default="pilot")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = run(
        data_root=args.data_root,
        protocol_path=args.protocol,
        output_dir=args.output_dir,
        profile_name=args.profile,
        revision=os.environ.get("GITHUB_SHA"),
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
