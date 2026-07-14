"""Causal tactile-baseline selection for the Deform360 replication."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np


_TIMESTAMP_PATTERN = re.compile(r"(?:^|_)(\d{10,})(?:\.[^.]+)?$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def timestamp_from_path(path: str | Path) -> int:
    """Return the terminal integer timestamp encoded in a Deform360 filename."""

    value = Path(path)
    match = _TIMESTAMP_PATTERN.search(value.name)
    _require(match is not None, f"filename has no terminal timestamp: {value.name}")
    return int(match.group(1))


def select_causal_tactile_baseline(
    recording_path: str | Path,
    baseline_paths: Sequence[str | Path],
) -> tuple[Path, dict[str, Any]]:
    """Select the newest baseline no later than the recording when possible."""

    recording = Path(recording_path).resolve()
    candidates = tuple(sorted(Path(path).resolve() for path in baseline_paths))
    _require(candidates, "no tactile baselines are available")
    recording_timestamp = timestamp_from_path(recording)
    timestamped = [(timestamp_from_path(path), path) for path in candidates]
    causal = [item for item in timestamped if item[0] <= recording_timestamp]
    if causal:
        selected_timestamp, selected = max(causal, key=lambda item: (item[0], item[1].name))
        rule = "latest-baseline-at-or-before-recording"
    else:
        selected_timestamp, selected = min(
            timestamped,
            key=lambda item: (
                abs(item[0] - recording_timestamp),
                item[0],
                item[1].name,
            ),
        )
        rule = "nearest-baseline-when-no-causal-baseline-exists"
    return selected, {
        "selection_rule": rule,
        "recording": recording.name,
        "recording_timestamp": recording_timestamp,
        "selected_baseline": selected.name,
        "selected_baseline_timestamp": selected_timestamp,
        "signed_baseline_age": recording_timestamp - selected_timestamp,
        "candidate_baselines": [
            {"name": path.name, "timestamp": timestamp}
            for timestamp, path in sorted(timestamped)
        ],
    }


def build_single_baseline_tactile_overlay(
    indexed_object_dir: str | Path,
    source_object_dir: str | Path,
    overlay_object_dir: str | Path,
    episode_index: int,
    *,
    sensors: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build an indexed symlink view with one causally selected baseline per sensor."""

    try:
        from deform360.layout import (
            list_tactile_names,
            recording_for_episode,
            tactile_recordings,
        )
    except ImportError as error:  # pragma: no cover - host integration
        raise RuntimeError("the pinned Deform360 runtime is required") from error

    indexed = Path(indexed_object_dir).resolve()
    source = Path(source_object_dir).resolve()
    output = Path(overlay_object_dir).resolve()
    _require(indexed.is_dir(), f"indexed tactile object is missing: {indexed}")
    _require(source.is_dir(), f"source tactile object is missing: {source}")
    available = list_tactile_names(indexed)
    selected_sensors = list(dict.fromkeys(sensors)) if sensors is not None else available
    _require(selected_sensors, "no tactile sensors were selected")
    _require(
        set(selected_sensors).issubset(available),
        "selected tactile sensor is absent from the indexed object",
    )
    records = []
    for sensor in selected_sensors:
        indexed_sensor = indexed / sensor
        source_sensor = source / sensor
        recordings = tactile_recordings(indexed_sensor, strict=True)
        _require(
            episode_index < len(recordings),
            f"episode {episode_index} is absent for tactile sensor {sensor}",
        )
        recording = recording_for_episode(recordings, episode_index)
        baseline, selection = select_causal_tactile_baseline(
            recording.data_path,
            sorted(source_sensor.glob("median_*.npy")),
        )
        output_sensor = output / sensor
        output_sensor.mkdir(parents=True, exist_ok=True)
        linked = []
        for path in sorted(indexed_sensor.iterdir()):
            if not path.is_file() or path.name.startswith("median_"):
                continue
            destination = output_sensor / path.name
            if destination.is_symlink():
                _require(
                    destination.resolve() == path.resolve(),
                    f"existing tactile link points elsewhere: {destination}",
                )
            else:
                _require(not destination.exists(), f"tactile overlay collision: {destination}")
                destination.symlink_to(path.resolve())
            linked.append(path.name)
        baseline_link = output_sensor / baseline.name
        if baseline_link.is_symlink():
            _require(
                baseline_link.resolve() == baseline,
                f"existing baseline link points elsewhere: {baseline_link}",
            )
        else:
            _require(not baseline_link.exists(), f"baseline overlay collision: {baseline_link}")
            baseline_link.symlink_to(baseline)
        records.append(
            {
                "sensor": sensor,
                "episode_index": episode_index,
                "recording_data": str(recording.data_path.resolve()),
                "recording_timestamps": str(recording.timestamp_path.resolve()),
                "recording_data_sha256": _sha256_file(recording.data_path),
                "recording_timestamps_sha256": _sha256_file(recording.timestamp_path),
                "selected_baseline": str(baseline),
                "selected_baseline_sha256": _sha256_file(baseline),
                "selection": selection,
                "indexed_link_count": len(linked),
            }
        )
    return {
        "artifact_kind": "Deform360ReplicationTactileBaselineOverlay",
        "episode_index": episode_index,
        "indexed_object_dir": str(indexed),
        "source_object_dir": str(source),
        "overlay_object_dir": str(output),
        "sensors": records,
        "information_boundary": {
            "selected_episode_recording_payload_hashed": True,
            "other_episode_payloads_read": False,
            "selection_uses_filename_timestamps_only": True,
        },
    }


def write_tactile_overlay_manifest(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def write_unavailable_tactile_stream(
    aligned_episode_dir: str | Path,
    sensor: str,
    *,
    episode_index: int,
    source_path: str | Path,
) -> Path:
    """Write an explicit zero evidence channel for a public zero-byte payload."""

    try:
        from deform360.tactile import TACTILE_SHAPE, load_episode_timeline
    except ImportError as error:  # pragma: no cover - host integration
        raise RuntimeError("the pinned Deform360 runtime is required") from error
    episode = Path(aligned_episode_dir).resolve()
    source = Path(source_path).resolve()
    _require(source.is_file(), f"unavailable tactile source is missing: {source}")
    _require(source.stat().st_size == 0, "only a zero-byte public payload may be unavailable")
    target_stream, timestamps = load_episode_timeline(episode)
    values = np.zeros((len(timestamps), *TACTILE_SHAPE), dtype=np.float32)
    output_dir = episode / sensor
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "synced_tactile.npy"
    with output.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    metadata = {
        "schema": "causal4d.deform360-unavailable-tactile/v1",
        "sensor": sensor,
        "episode_index": episode_index,
        "source_file": str(source),
        "source_sha256": _sha256_file(source),
        "source_size_bytes": 0,
        "target_stream": target_stream,
        "target_frame_count": len(timestamps),
        "output_sha256": _sha256_file(output),
        "semantics": (
            "All-zero unavailable-evidence channel; it must not be interpreted "
            "as a measured no-contact response."
        ),
    }
    (output_dir / "unavailable.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


__all__ = [
    "build_single_baseline_tactile_overlay",
    "select_causal_tactile_baseline",
    "timestamp_from_path",
    "write_tactile_overlay_manifest",
    "write_unavailable_tactile_stream",
]
