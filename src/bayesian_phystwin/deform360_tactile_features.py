"""Reproducible causal tactile features for Deform360 belief guards."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .deform360_tactile_regret_guard import (
    align_baseline_subtracted_tactile,
    causal_tactile_regret_features,
)

TACTILE_SENSOR_NAMES = (
    "brics-odroid_tactilel_left",
    "brics-odroid_tactilel_right",
    "brics-odroid_tactiler_left",
    "brics-odroid_tactiler_right",
)
_FRAME_TOKEN = re.compile(r"^frame_(\d+)_\d+$")
_TRAILING_TIMESTAMP = re.compile(r"_(\d+)$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: str | Path) -> str:
    """Hash a file without loading it fully into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_artifact_sha256(payload: Mapping[str, Any]) -> str:
    """Hash canonical JSON while excluding its self-referential digest."""

    stripped = dict(payload)
    stripped.pop("artifact_sha256", None)
    encoded = json.dumps(stripped, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_frame_timestamps_us(path: str | Path) -> np.ndarray:
    """Read timestamps from Deform360 ``frame_<us>_<index>`` text rows."""

    rows = Path(path).read_text(encoding="utf-8").splitlines()
    timestamps = []
    for row in rows:
        fields = row.split()
        _require(bool(fields), f"empty timestamp row in {path}")
        match = _FRAME_TOKEN.fullmatch(fields[0])
        _require(match is not None, f"invalid timestamp token in {path}: {fields[0]}")
        timestamps.append(int(match.group(1)))
    values = np.asarray(timestamps, dtype=np.int64)
    _require(len(values) > 0, f"timestamp file is empty: {path}")
    _require(np.all(np.diff(values) >= 0), f"timestamps are unsorted: {path}")
    return values


def load_raw_tactile_frames(
    path: str | Path,
    *,
    frame_count: int,
    taxel_shape: tuple[int, int] = (16, 32),
) -> np.ndarray:
    """Load Deform360 raw float32 dumps or ordinary NumPy archives."""

    source = Path(path)
    with source.open("rb") as handle:
        is_numpy = handle.read(6) == b"\x93NUMPY"
    if is_numpy:
        frames = np.load(source, allow_pickle=False, mmap_mode="r")
    else:
        frames = np.memmap(source, dtype="<f4", mode="r")
    expected = int(frame_count) * int(np.prod(taxel_shape))
    _require(frames.size == expected, f"raw tactile shape changed: {source}")
    return np.asarray(frames).reshape((frame_count, *taxel_shape))


def _timestamp_suffix(path: Path) -> int:
    match = _TRAILING_TIMESTAMP.search(path.stem)
    _require(match is not None, f"timestamp suffix missing: {path}")
    return int(match.group(1))


def _select_raw_segment(
    sensor_root: Path,
    sensor_name: str,
    target_timestamps_us: np.ndarray,
) -> tuple[Path, Path, np.ndarray]:
    candidates = []
    for timestamp_path in sorted(sensor_root.glob(f"{sensor_name}_*.txt")):
        data_path = timestamp_path.with_suffix(".npy")
        if not data_path.is_file():
            continue
        timestamps = read_frame_timestamps_us(timestamp_path)
        contains = bool(
            timestamps[0] <= target_timestamps_us[0]
            and timestamps[-1] >= target_timestamps_us[-1]
        )
        midpoint_gap = abs(
            int(timestamps[len(timestamps) // 2])
            - int(target_timestamps_us[len(target_timestamps_us) // 2])
        )
        candidates.append((not contains, midpoint_gap, timestamp_path.name, data_path, timestamp_path, timestamps))
    _require(bool(candidates), f"no raw tactile segment found in {sensor_root}")
    selected = min(candidates, key=lambda row: row[:3])
    _require(not selected[0], f"no tactile segment covers the full causal window: {sensor_root}")
    return selected[3], selected[4], selected[5]


def _select_preceding_baseline(sensor_root: Path, data_path: Path) -> Path:
    segment_timestamp = _timestamp_suffix(data_path)
    candidates = [
        path
        for path in sensor_root.glob("median_*.npy")
        if _timestamp_suffix(path) <= segment_timestamp
    ]
    _require(bool(candidates), f"no preceding tactile baseline found in {sensor_root}")
    return max(candidates, key=_timestamp_suffix)


def _load_shared_target_timeline(episode_root: Path) -> tuple[np.ndarray, list[dict[str, str]]]:
    paths = sorted(episode_root.glob("brics-odroid-*_cam*/aligned_timestamps.txt"))
    _require(bool(paths), f"no aligned camera timeline found in {episode_root}")
    reference = read_frame_timestamps_us(paths[0])
    records = []
    for path in paths:
        timestamps = read_frame_timestamps_us(path)
        _require(
            np.array_equal(timestamps, reference),
            f"camera timelines disagree in {episode_root}",
        )
        records.append(
            {
                "path": path.relative_to(episode_root).as_posix(),
                "sha256": file_sha256(path),
            }
        )
    return reference, records


def extract_case_tactile_features(
    *,
    case_name: str,
    object_id: str,
    episode_index: int,
    episode_root: str | Path,
    raw_object_root: str | Path,
    update_frames: Sequence[int] = (19, 38, 57),
    initial_reference_frame_count: int = 6,
    history_frame_count: int = 3,
    available_frame_count: int | None = None,
) -> dict[str, Any]:
    """Extract one case with complete raw-input provenance."""

    episode = Path(episode_root).resolve()
    raw_object = Path(raw_object_root).resolve()
    full_target_timestamps, target_records = _load_shared_target_timeline(episode)
    causal_frame_count = (
        max(int(value) for value in update_frames) + 1
        if available_frame_count is None
        else int(available_frame_count)
    )
    _require(
        causal_frame_count >= max(int(value) for value in update_frames) + 1
        and causal_frame_count <= len(full_target_timestamps),
        "causal tactile availability does not cover the requested updates",
    )
    target_timestamps = full_target_timestamps[:causal_frame_count]
    responses = []
    source_records = []
    maximum_alignment_delta_us = 0
    for sensor_name in TACTILE_SENSOR_NAMES:
        sensor_root = raw_object / sensor_name
        _require(sensor_root.is_dir(), f"missing tactile sensor directory: {sensor_root}")
        data_path, timestamp_path, source_timestamps = _select_raw_segment(
            sensor_root,
            sensor_name,
            target_timestamps,
        )
        baseline_path = _select_preceding_baseline(sensor_root, data_path)
        baseline = np.load(baseline_path, allow_pickle=False)
        frames = load_raw_tactile_frames(
            data_path,
            frame_count=len(source_timestamps),
            taxel_shape=tuple(int(value) for value in baseline.shape),
        )
        causal_source_stop = min(
            int(np.searchsorted(source_timestamps, target_timestamps[-1])) + 1,
            len(source_timestamps),
        )
        aligned = align_baseline_subtracted_tactile(
            frames[:causal_source_stop],
            source_timestamps[:causal_source_stop],
            baseline,
            target_timestamps,
        )
        responses.append(aligned.response)
        maximum_alignment_delta_us = max(
            maximum_alignment_delta_us,
            int(np.max(np.abs(aligned.signed_delta_us))),
        )
        source_records.append(
            {
                "sensor": sensor_name,
                "data": data_path.name,
                "data_sha256": file_sha256(data_path),
                "timestamps": timestamp_path.name,
                "timestamps_sha256": file_sha256(timestamp_path),
                "baseline": baseline_path.name,
                "baseline_sha256": file_sha256(baseline_path),
                "source_frame_count": len(source_timestamps),
            }
        )
    _, diagnostics = causal_tactile_regret_features(
        np.stack(responses),
        update_frames=update_frames,
        initial_reference_frame_count=initial_reference_frame_count,
        history_frame_count=history_frame_count,
    )
    return {
        "case": str(case_name),
        "object": str(object_id),
        "episode_index": int(episode_index),
        "episode_frame_count": len(full_target_timestamps),
        "available_frame_count": causal_frame_count,
        "sensor_count": len(TACTILE_SENSOR_NAMES),
        "maximum_alignment_delta_us": maximum_alignment_delta_us,
        "target_timeline_records": target_records,
        "source_records": source_records,
        "updates": diagnostics,
    }


def build_tactile_feature_artifact(
    cases: Sequence[Mapping[str, Any]],
    *,
    window_root: str | Path,
    raw_root: str | Path,
    update_frames: Sequence[int] = (19, 38, 57),
    initial_reference_frame_count: int = 6,
    history_frame_count: int = 3,
    available_frame_count: int | None = None,
) -> dict[str, Any]:
    """Build a deterministic feature artifact from a target-free case manifest."""

    window = Path(window_root).resolve()
    raw = Path(raw_root).resolve()
    rows = []
    seen = set()
    for case in cases:
        case_name = str(case["case"])
        _require(case_name not in seen, f"case repeated in tactile manifest: {case_name}")
        seen.add(case_name)
        rows.append(
            extract_case_tactile_features(
                case_name=case_name,
                object_id=str(case["object"]),
                episode_index=int(case["episode_index"]),
                episode_root=window / str(case["episode_path"]),
                raw_object_root=raw / str(case["raw_object_path"]),
                update_frames=update_frames,
                initial_reference_frame_count=initial_reference_frame_count,
                history_frame_count=history_frame_count,
                available_frame_count=available_frame_count,
            )
        )
    payload: dict[str, Any] = {
        "artifact_kind": "Deform360CausalTactileFeatureAuditV2",
        "schema_version": 2,
        "information_boundary": {
            "opened_source_only": False,
            "target_outcomes_read": False,
            "held_v8_read": False,
            "future_tactile_used_for_update": False,
            "each_update_uses_tactile_at_or_before_update": True,
            "episode_wide_tactile_normalization_used": False,
        },
        "method": {
            "raw_baseline_subtracted_taxels": True,
            "invalid_columns": [-1],
            "update_frames": [int(value) for value in update_frames],
            "initial_reference_frame_count": int(initial_reference_frame_count),
            "history_frame_count": int(history_frame_count),
            "available_frame_count": (
                max(int(value) for value in update_frames) + 1
                if available_frame_count is None
                else int(available_frame_count)
            ),
        },
        "cases": sorted(rows, key=lambda row: str(row["case"])),
    }
    payload["artifact_sha256"] = canonical_artifact_sha256(payload)
    return payload


__all__ = [
    "TACTILE_SENSOR_NAMES",
    "build_tactile_feature_artifact",
    "canonical_artifact_sha256",
    "extract_case_tactile_features",
    "file_sha256",
    "load_raw_tactile_frames",
    "read_frame_timestamps_us",
]
