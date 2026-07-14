"""Pinned Deform360 ``001-rope`` inventory and information-boundary audit."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PINNED_DEFORM360_CODE_REPOSITORY = "https://github.com/lhy0807/deform360"
PINNED_DEFORM360_CODE_COMMIT = "0fe36f0b7a7a917ba62b5f8cee707299a9a4a317"
PINNED_DEFORM360_DATASET_REPOSITORY = "brownu/deform360"
PINNED_DEFORM360_DATASET_REVISION = "7fea8e20231a47641d1d2bc8791920ec4e62ec5e"
DEFORM360_OBJECT_ID = "001-rope"
DEFORM360_PROTOCOL_SCHEMA_VERSION = 1
DEFORM360_PREFLIGHT_SCHEMA_VERSION = 1
_CANONICAL_PROTOCOL_CONFIG_SHA256 = (
    "61f463b5f4b7cba3e103d83830167cb46cf28efd3be01f497bd323d7f203caa1"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Deform360ProtocolConfig:
    """Outcome-free choices fixed before processing or inspecting predictions."""

    protocol_id: str = "causal4d-deform360-001-rope-v1"
    code_repository: str = PINNED_DEFORM360_CODE_REPOSITORY
    code_commit: str = PINNED_DEFORM360_CODE_COMMIT
    dataset_repository: str = PINNED_DEFORM360_DATASET_REPOSITORY
    dataset_revision: str = PINNED_DEFORM360_DATASET_REVISION
    object_id: str = DEFORM360_OBJECT_ID
    expected_episode_count: int = 10
    expected_camera_count: int = 41
    minimum_calibrated_camera_count: int = 36
    expected_tactile_sensor_count: int = 4
    expected_non_audio_file_count: int = 908
    prefix_frame_count: int = 6
    prefix_trigger_method: str = "source-calibrated causal robot-opening trigger"
    prefix_trigger_confirmation_frames: int = 3
    prefix_trigger_aggregation: str = "all_target_grippers"
    minimum_tracking_camera_count: int = 3
    nominal_frame_interval_us: int = 33_333
    tactile_contact_threshold: float = 0.0
    split_seed: str = "causal4d-deform360-001-rope-v1"
    source_episode_ids: tuple[int, ...] = (0, 2, 3, 4, 5, 7, 8)
    calibration_episode_ids: tuple[int, ...] = (1, 9)
    target_episode_ids: tuple[int, ...] = (6,)

    def __post_init__(self) -> None:
        _require(
            self.code_repository == PINNED_DEFORM360_CODE_REPOSITORY,
            "unexpected code repository",
        )
        _require(
            self.code_commit == PINNED_DEFORM360_CODE_COMMIT, "unexpected code commit"
        )
        _require(
            self.dataset_repository == PINNED_DEFORM360_DATASET_REPOSITORY,
            "unexpected dataset repository",
        )
        _require(
            self.dataset_revision == PINNED_DEFORM360_DATASET_REVISION,
            "unexpected dataset revision",
        )
        _require(
            self.object_id == DEFORM360_OBJECT_ID, "this protocol is 001-rope only"
        )
        _require(
            self.expected_episode_count == 10, "001-rope must contain ten episodes"
        )
        _require(
            self.expected_camera_count >= self.minimum_calibrated_camera_count >= 1,
            "invalid camera counts",
        )
        _require(
            self.expected_tactile_sensor_count == 4, "unexpected tactile sensor count"
        )
        _require(
            self.prefix_frame_count >= 2, "prefix must contain at least two frames"
        )
        _require(
            self.prefix_trigger_method
            == "source-calibrated causal robot-opening trigger",
            "unexpected target-prefix trigger method",
        )
        _require(
            self.prefix_trigger_confirmation_frames >= 1,
            "prefix trigger must be confirmed for at least one frame",
        )
        _require(
            self.prefix_trigger_aggregation == "all_target_grippers",
            "unexpected target-prefix trigger aggregation",
        )
        _require(
            self.nominal_frame_interval_us > 0,
            "nominal frame interval must be positive",
        )
        _require(
            0.0 <= self.tactile_contact_threshold <= 1.0, "invalid tactile threshold"
        )
        groups = (
            self.source_episode_ids,
            self.calibration_episode_ids,
            self.target_episode_ids,
        )
        flattened = tuple(index for group in groups for index in group)
        _require(len(flattened) == len(set(flattened)), "episode splits overlap")
        _require(
            set(flattened) == set(range(self.expected_episode_count)),
            "episode splits must cover exactly 0..9",
        )
        expected = _metadata_ranked_split(self)
        actual = {
            "source": tuple(sorted(self.source_episode_ids)),
            "calibration": tuple(sorted(self.calibration_episode_ids)),
            "target": tuple(sorted(self.target_episode_ids)),
        }
        _require(
            actual == expected,
            "stored split differs from metadata-only SHA-256 ranking",
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Deform360ProtocolConfig:
        fields = cls.__dataclass_fields__
        unknown = set(value) - set(fields)
        _require(not unknown, f"unknown Deform360 config fields: {sorted(unknown)}")
        payload = dict(value)
        for key in (
            "source_episode_ids",
            "calibration_episode_ids",
            "target_episode_ids",
        ):
            if key in payload:
                payload[key] = tuple(int(item) for item in payload[key])
        return cls(**payload)


def _episode_id(index: int) -> str:
    return f"{DEFORM360_OBJECT_ID}/episode_{index:04d}"


def _split_rank(config: Deform360ProtocolConfig, index: int) -> str:
    return _sha256_bytes(f"{config.split_seed}:{_episode_id(index)}".encode("utf-8"))


def _metadata_ranked_split(
    config: Deform360ProtocolConfig,
) -> dict[str, tuple[int, ...]]:
    ranked = sorted(
        range(config.expected_episode_count),
        key=lambda index: _split_rank(config, index),
    )
    calibration_count = len(config.calibration_episode_ids)
    target_count = len(config.target_episode_ids)
    source_count = config.expected_episode_count - calibration_count - target_count
    return {
        "source": tuple(sorted(ranked[:source_count])),
        "calibration": tuple(
            sorted(ranked[source_count : source_count + calibration_count])
        ),
        "target": tuple(sorted(ranked[-target_count:])),
    }


def protocol_config_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    return _sha256_bytes(_canonical_bytes(canonical))


def load_deform360_protocol_config(path: str | Path) -> Deform360ProtocolConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(
        payload.get("schema_version") == DEFORM360_PROTOCOL_SCHEMA_VERSION,
        "unsupported Deform360 protocol schema",
    )
    _require(
        payload.get("config_sha256") == protocol_config_sha256(payload),
        "Deform360 protocol checksum mismatch",
    )
    if _CANONICAL_PROTOCOL_CONFIG_SHA256:
        _require(
            payload["config_sha256"] == _CANONICAL_PROTOCOL_CONFIG_SHA256,
            "Deform360 protocol differs from the canonical lock",
        )
    return Deform360ProtocolConfig.from_mapping(payload["config"])


def _stream_inventory(
    path: Path, data_suffix: str, *, exclude_prefix: str = ""
) -> dict[str, Any]:
    data_paths = sorted(
        item
        for item in path.glob(f"*{data_suffix}")
        if not exclude_prefix or not item.name.startswith(exclude_prefix)
    )
    timestamp_paths = sorted(path.glob("*.txt"))
    data_stems = {item.stem for item in data_paths}
    timestamp_stems = {item.stem for item in timestamp_paths}
    return {
        "stream": path.name,
        "recording_count": len(data_paths),
        "timestamp_count": len(timestamp_paths),
        "exact_stem_pairs": data_stems == timestamp_stems,
        "first_recording": data_paths[0].name if data_paths else None,
        "last_recording": data_paths[-1].name if data_paths else None,
    }


def _load_calibration_summary(object_dir: Path) -> dict[str, Any]:
    calibration_dir = object_dir / "calibration_refined"
    paths = {
        name: calibration_dir / f"{name}.npy"
        for name in ("intrinsics", "extrinsics", "dist")
    }
    if not all(path.is_file() for path in paths.values()):
        return {
            "complete": False,
            "camera_count": 0,
            "reason": "calibration files are missing",
        }
    dictionaries: dict[str, Mapping[str, Any]] = {}
    try:
        for name, path in paths.items():
            value = np.load(path, allow_pickle=True).item()
            _require(
                isinstance(value, Mapping), f"{path.name} is not a camera dictionary"
            )
            dictionaries[name] = value
    except (OSError, ValueError, AttributeError) as error:
        return {
            "complete": False,
            "camera_count": 0,
            "reason": f"calibration audit failed: {error}",
        }
    key_sets = [set(value) for value in dictionaries.values()]
    shared = set.intersection(*key_sets)
    shape_valid = all(
        np.asarray(dictionaries["intrinsics"][camera]).shape == (3, 3)
        and np.asarray(dictionaries["extrinsics"][camera]).shape == (4, 4)
        and np.asarray(dictionaries["dist"][camera]).shape == (4,)
        for camera in shared
    )
    return {
        "complete": bool(
            shared and all(keys == shared for keys in key_sets) and shape_valid
        ),
        "camera_count": len(shared),
        "camera_names": sorted(shared),
        "shape_valid": shape_valid,
        "trusted_pickle_boundary": "pinned official Deform360 dataset only",
        "files": {
            name: {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for name, path in paths.items()
        },
    }


def _raw_inventory(
    object_dir: Path, config: Deform360ProtocolConfig, hash_media: bool
) -> dict[str, Any]:
    files = sorted(path for path in object_dir.rglob("*") if path.is_file())
    records = []
    for path in files:
        record = {
            "path": path.relative_to(object_dir).as_posix(),
            "bytes": path.stat().st_size,
        }
        if hash_media:
            record["sha256"] = _sha256_file(path)
        records.append(record)
    cameras = sorted(
        path for path in object_dir.iterdir() if path.is_dir() and "_cam" in path.name
    )
    tactile = sorted(
        path
        for path in object_dir.iterdir()
        if path.is_dir() and "_tactile" in path.name
    )
    camera_streams = [_stream_inventory(path, ".mp4") for path in cameras]
    tactile_streams = [
        _stream_inventory(path, ".npy", exclude_prefix="median_") for path in tactile
    ]
    expected_episodes = config.expected_episode_count
    return {
        "file_count": len(files),
        "expected_non_audio_file_count": config.expected_non_audio_file_count,
        "file_count_matches": len(files) == config.expected_non_audio_file_count,
        "descriptor_sha256": _sha256_bytes(_canonical_bytes(records)),
        "all_file_contents_hashed": hash_media,
        "camera_count": len(cameras),
        "camera_streams": camera_streams,
        "camera_recordings_complete": bool(
            len(cameras) == config.expected_camera_count
            and all(
                stream["recording_count"] == expected_episodes
                and stream["timestamp_count"] == expected_episodes
                and stream["exact_stem_pairs"]
                for stream in camera_streams
            )
        ),
        "tactile_sensor_count": len(tactile),
        "tactile_streams": tactile_streams,
        "tactile_recordings_complete": bool(
            len(tactile) == config.expected_tactile_sensor_count
            and all(
                stream["recording_count"] == expected_episodes
                and stream["timestamp_count"] == expected_episodes
                and stream["exact_stem_pairs"]
                for stream in tactile_streams
            )
        ),
    }


def _episode_split(config: Deform360ProtocolConfig, index: int) -> str:
    if index in config.source_episode_ids:
        return "source"
    if index in config.calibration_episode_ids:
        return "calibration"
    return "target"


def _tactile_episode_summary(
    episode_dir: Path,
    sensor_names: Sequence[str],
    threshold: float,
    *,
    value_frame_start: int,
    value_frame_limit: int | None,
) -> dict[str, Any]:
    read_values = value_frame_limit != 0
    value_scope = (
        "sealed"
        if value_frame_limit == 0
        else "full"
        if value_frame_limit is None
        else "prefix"
    )
    _require(value_frame_start >= 0, "tactile value-frame start must be non-negative")
    sensors = []
    frame_counts = set()
    active_taxels_by_group: dict[str, np.ndarray] = {}
    for sensor in sensor_names:
        path = episode_dir / sensor / "synced_tactile.npy"
        if not path.is_file():
            sensors.append({"sensor": sensor, "available": False})
            continue
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        frame_counts.add(int(values.shape[0]) if values.ndim >= 1 else -1)
        summary: dict[str, Any] = {
            "sensor": sensor,
            "available": True,
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "sha256": _sha256_file(path),
            "values_read": read_values,
            "value_scope": value_scope,
        }
        alignment_path = episode_dir / sensor / "alignment.json"
        if alignment_path.is_file():
            try:
                alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
                if isinstance(alignment, Mapping):
                    summary["alignment_summary"] = alignment.get("summary")
                    summary["alignment_tolerance_us"] = alignment.get("tolerance_us")
            except (OSError, json.JSONDecodeError):
                summary["alignment_summary"] = None
        if read_values:
            stop = (
                int(values.shape[0])
                if value_frame_limit is None
                else value_frame_start + value_frame_limit
            )
            _require(
                stop <= int(values.shape[0]),
                f"requested tactile frame range [{value_frame_start}, {stop}) "
                f"exceeds {sensor} length {values.shape[0]}",
            )
            array = np.asarray(values[value_frame_start:stop])
            active_taxels = np.count_nonzero(
                array > threshold, axis=tuple(range(1, array.ndim))
            )
            active = active_taxels > 0
            active_indices = np.flatnonzero(active) + value_frame_start
            group = next(
                (
                    sensor[: -len(suffix)]
                    for suffix in ("_left", "_right")
                    if sensor.endswith(suffix)
                ),
                sensor,
            )
            active_taxels_by_group[group] = (
                active_taxels_by_group.get(group, np.zeros_like(active_taxels))
                + active_taxels
            )
            summary.update(
                {
                    "active_frame_count": int(np.count_nonzero(active)),
                    "value_frame_range": [value_frame_start, stop],
                    "contact_first_frame": int(active_indices[0])
                    if len(active_indices)
                    else None,
                    "contact_last_frame": int(active_indices[-1])
                    if len(active_indices)
                    else None,
                    "peak_response": float(np.max(array)) if array.size else None,
                    "peak_active_taxel_count": int(np.max(active_taxels))
                    if active_taxels.size
                    else 0,
                }
            )
        sensors.append(summary)
    groups = []
    episode_active = None
    for group, active_taxels in sorted(active_taxels_by_group.items()):
        active = active_taxels > 1
        active_indices = np.flatnonzero(active)
        episode_active = (
            active.copy() if episode_active is None else episode_active | active
        )
        groups.append(
            {
                "gripper_group": group,
                "active_frame_count": int(np.count_nonzero(active)),
                "contact_first_frame": int(active_indices[0])
                if len(active_indices)
                else None,
                "contact_last_frame": int(active_indices[-1])
                if len(active_indices)
                else None,
                "peak_active_taxel_count": int(np.max(active_taxels))
                if active_taxels.size
                else 0,
                "contact_rule": "sum of paired finger sensors > 1 active taxel",
            }
        )
    episode_active_indices = (
        np.flatnonzero(episode_active) + value_frame_start
        if episode_active is not None
        else np.asarray([], dtype=int)
    )
    return {
        "sensor_count": sum(sensor["available"] for sensor in sensors),
        "frame_count_consistent": len(frame_counts) <= 1,
        "values_read": read_values,
        "value_scope": value_scope,
        "value_frame_start": value_frame_start if read_values else None,
        "value_frame_limit": value_frame_limit if read_values else None,
        "sensors": sensors,
        "gripper_groups": groups,
        "episode_contact": {
            "active_frame_count": int(np.count_nonzero(episode_active))
            if episode_active is not None
            else 0,
            "contact_first_frame": int(episode_active_indices[0])
            if len(episode_active_indices)
            else None,
            "contact_last_frame": int(episode_active_indices[-1])
            if len(episode_active_indices)
            else None,
            "release_rule_matched": True,
        }
        if read_values
        else None,
    }


def _camera_alignment_quality(
    episode_dir: Path,
    camera_names: Sequence[str],
    nominal_frame_interval_us: int,
) -> dict[str, Any]:
    summaries = []
    for camera in camera_names:
        path = episode_dir / camera / "alignment.json"
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        summary = payload.get("summary") if isinstance(payload, Mapping) else None
        if not isinstance(summary, Mapping):
            continue
        count = int(summary.get("count") or 0)
        reused = int(summary.get("reused_source_frames") or 0)
        p95_delta = int(summary.get("p95_abs_delta_us") or 0)
        reused_fraction = reused / count if count else 1.0
        timing_reliability = float(
            np.exp(-0.5 * (p95_delta / nominal_frame_interval_us) ** 2)
        )
        summaries.append(
            {
                "camera": camera,
                "count": count,
                "within_tolerance_count": summary.get("within_tolerance_count"),
                "max_abs_delta_us": summary.get("max_abs_delta_us"),
                "p95_abs_delta_us": p95_delta,
                "reused_source_frames": reused,
                "reused_source_fraction": reused_fraction,
                "timing_reliability": timing_reliability,
                "synchronization_reliability": (1.0 - reused_fraction)
                * timing_reliability,
            }
        )
    complete = bool(summaries) and len(summaries) == len(camera_names)
    all_within_tolerance = complete and all(
        item["count"] == item["within_tolerance_count"] for item in summaries
    )
    reliability = [item["synchronization_reliability"] for item in summaries]
    return {
        "camera_summary_count": len(summaries),
        "all_within_tolerance": all_within_tolerance,
        "maximum_abs_delta_us": max(
            (int(item["max_abs_delta_us"]) for item in summaries), default=None
        ),
        "maximum_p95_abs_delta_us": max(
            (int(item["p95_abs_delta_us"]) for item in summaries), default=None
        ),
        "total_reused_source_frames": sum(
            int(item["reused_source_frames"] or 0) for item in summaries
        ),
        "minimum_synchronization_reliability": min(reliability, default=None),
        "median_synchronization_reliability": float(np.median(reliability))
        if reliability
        else None,
        "reliability_rule": (
            "(1 - reused_fraction) * exp(-0.5 * "
            "(p95_abs_delta_us / nominal_frame_interval_us)^2)"
        ),
        "nominal_frame_interval_us": nominal_frame_interval_us,
        "cameras": summaries,
    }


def _robot_summary(episode_dir: Path) -> dict[str, Any]:
    path = episode_dir / "robot" / "robot.npz"
    if not path.is_file():
        return {"available": False}
    try:
        with np.load(path, allow_pickle=False) as payload:
            keys = sorted(payload.files)
            actions = np.asarray(payload["actions"])
            transforms = np.asarray(payload["T_worlds"])
            openings = np.asarray(payload["openings"])
            finite = bool(
                np.all(np.isfinite(actions))
                and np.all(np.isfinite(transforms))
                and np.all(np.isfinite(openings))
            )
            batched = transforms[:, None] if transforms.ndim == 3 else transforms
            rotations = batched[..., :3, :3]
            translations = batched[..., :3, 3]
            translation_steps = np.linalg.norm(np.diff(translations, axis=0), axis=-1)
            relative_rotations = np.einsum(
                "tgij,tgkj->tgik", rotations[1:], rotations[:-1:]
            )
            rotation_cosines = np.clip(
                (np.trace(relative_rotations, axis1=-2, axis2=-1) - 1.0) / 2.0,
                -1.0,
                1.0,
            )
            rotation_steps_deg = np.degrees(np.arccos(rotation_cosines))
            orthonormal_error = np.max(
                np.abs(
                    np.einsum("...ji,...jk->...ik", rotations, rotations) - np.eye(3)
                )
            )
            determinant_error = np.max(np.abs(np.linalg.det(rotations) - 1.0))
            return {
                "available": True,
                "keys": keys,
                "actions_shape": list(actions.shape),
                "transforms_shape": list(transforms.shape),
                "openings_shape": list(openings.shape),
                "frame_count": int(actions.shape[0]),
                "finite": finite,
                "rigid_transform_valid": bool(
                    finite
                    and orthonormal_error <= 1e-5
                    and determinant_error <= 1e-5
                    and np.allclose(
                        batched[..., 3, :],
                        np.array([0.0, 0.0, 0.0, 1.0]),
                        rtol=0.0,
                        atol=1e-8,
                    )
                ),
                "maximum_rotation_orthonormal_error": float(orthonormal_error),
                "maximum_rotation_determinant_error": float(determinant_error),
                "translation_step_m": {
                    "median": float(np.median(translation_steps)),
                    "p95": float(np.quantile(translation_steps, 0.95)),
                    "maximum": float(np.max(translation_steps)),
                },
                "rotation_step_deg": {
                    "median": float(np.median(rotation_steps_deg)),
                    "p95": float(np.quantile(rotation_steps_deg, 0.95)),
                    "maximum": float(np.max(rotation_steps_deg)),
                },
                "opening_m": {
                    "minimum": float(np.min(openings)),
                    "maximum": float(np.max(openings)),
                    "range": float(np.max(openings) - np.min(openings)),
                },
                "sha256": _sha256_file(path),
                "signal_semantics": "measured/vision-recovered wrist trajectory, not commanded control",
            }
    except (OSError, KeyError, ValueError) as error:
        return {"available": False, "reason": f"robot audit failed: {error}"}


def _processed_episode_summary(
    processed_root: Path,
    index: int,
    split: str,
    sensor_names: Sequence[str],
    config: Deform360ProtocolConfig,
    *,
    unlock_target_prefix: bool,
    unlock_target_oracle: bool,
    target_prefix_start_frame: int | None,
) -> dict[str, Any]:
    episode_dir = processed_root / f"episode_{index:04d}"
    if not episode_dir.is_dir():
        return {
            "episode_id": _episode_id(index),
            "split": split,
            "available": False,
            "target_oracle_values_read": False,
        }
    alignment_path = episode_dir / "alignment.json"
    alignment: dict[str, Any] = {}
    if alignment_path.is_file():
        try:
            loaded = json.loads(alignment_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                alignment = loaded
        except (OSError, json.JSONDecodeError):
            alignment = {}
    aligned_cameras = sorted(
        path.name
        for path in episode_dir.iterdir()
        if path.is_dir()
        and (path / "undistorted.mp4").is_file()
        and (path / "aligned_timestamps.txt").is_file()
    )
    camera_alignment_quality = _camera_alignment_quality(
        episode_dir,
        aligned_cameras,
        config.nominal_frame_interval_us,
    )
    value_frame_limit = (
        None
        if split != "target" or unlock_target_oracle
        else config.prefix_frame_count
        if unlock_target_prefix
        else 0
    )
    value_frame_start = (
        int(target_prefix_start_frame)
        if split == "target" and unlock_target_prefix
        else 0
    )
    tactile = _tactile_episode_summary(
        episode_dir,
        sensor_names,
        config.tactile_contact_threshold,
        value_frame_start=value_frame_start,
        value_frame_limit=value_frame_limit,
    )
    robot = _robot_summary(episode_dir)
    tracking_cameras = sorted(
        camera
        for camera in aligned_cameras
        if (episode_dir / camera / "tracking" / "vel.h5").is_file()
        and (episode_dir / camera / "tracking" / "visibility.h5").is_file()
    )
    pcd_paths = sorted((episode_dir / "pcd_clean").glob("*.npz"))
    return {
        "episode_id": _episode_id(index),
        "split": split,
        "available": True,
        "alignment": {
            "manifest_available": alignment_path.is_file(),
            "anchor_camera": alignment.get("anchor_camera"),
            "frame_count": alignment.get("frame_count"),
            "timeline_sha256": alignment.get("timeline_sha256"),
            "aligned_camera_count": len(aligned_cameras),
            "quality": camera_alignment_quality,
        },
        "tactile": tactile,
        "robot": robot,
        "tracking": {
            "camera_count": len(tracking_cameras),
            "minimum_required": config.minimum_tracking_camera_count,
            "ready": len(tracking_cameras) >= config.minimum_tracking_camera_count,
        },
        "metric_geometry": {
            "pcd_frame_count": len(pcd_paths),
            "metadata_available": (
                episode_dir / "pcd_clean" / "pcd_clean.meta.json"
            ).is_file(),
            "ready": bool(pcd_paths),
            "material_identity_semantics": "frame-0 metric seed points advected through time",
        },
        "phystwin_adapter": {
            "final_data_available": (episode_dir / "final_data.pkl").is_file(),
            "start_point_cloud_available": (
                episode_dir / "start_obj_pcd.ply"
            ).is_file(),
        },
        "target_prefix_values_read": split == "target"
        and tactile["value_scope"] == "prefix",
        "target_oracle_values_read": split == "target"
        and tactile["value_scope"] == "full",
    }


def preflight_deform360_001_rope(
    raw_object_dir: str | Path,
    config: Deform360ProtocolConfig,
    *,
    processed_root: str | Path | None = None,
    hash_media: bool = False,
    unlock_target_prefix: bool = False,
    unlock_target_oracle: bool = False,
    target_prefix_start_frame: int | None = None,
) -> dict[str, Any]:
    """Audit the pinned public cohort without using target outcomes for selection."""

    object_dir = Path(raw_object_dir).resolve()
    _require(
        not (unlock_target_prefix and unlock_target_oracle),
        "target prefix and target oracle cannot be unlocked together",
    )
    if unlock_target_prefix:
        _require(
            target_prefix_start_frame is not None,
            "target-prefix unlock requires a visually selected start frame",
        )
        _require(
            target_prefix_start_frame >= 0,
            "target-prefix start frame must be non-negative",
        )
    else:
        _require(
            target_prefix_start_frame is None,
            "target-prefix start is only valid with target-prefix unlock",
        )
    _require(object_dir.is_dir(), "Deform360 raw object directory does not exist")
    _require(
        object_dir.name == config.object_id, "raw object directory must be 001-rope"
    )
    metadata_path = object_dir / "metadata.json"
    _require(metadata_path.is_file(), "metadata.json is missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _require(metadata.get("object") == config.object_id, "metadata object id mismatch")
    sequences = metadata.get("sequences")
    _require(isinstance(sequences, Mapping), "metadata sequences are missing")
    _require(
        set(sequences)
        == {str(index) for index in range(config.expected_episode_count)},
        "metadata must contain sequences 0..9",
    )

    inventory = _raw_inventory(object_dir, config, hash_media)
    calibration = _load_calibration_summary(object_dir)
    assignments = [
        {
            "episode_id": _episode_id(index),
            "episode_index": index,
            "action": str(sequences[str(index)]["action"]),
            "bimanual": sequences[str(index)].get("bimanual") == "yes",
            "nonprehensile": sequences[str(index)].get("nonprehensile") == "yes",
            "split": _episode_split(config, index),
            "metadata_rank_sha256": _split_rank(config, index),
        }
        for index in range(config.expected_episode_count)
    ]
    sensor_names = [stream["stream"] for stream in inventory["tactile_streams"]]
    processed = []
    if processed_root is not None:
        root = Path(processed_root).resolve()
        processed = [
            _processed_episode_summary(
                root,
                index,
                _episode_split(config, index),
                sensor_names,
                config,
                unlock_target_prefix=unlock_target_prefix,
                unlock_target_oracle=unlock_target_oracle,
                target_prefix_start_frame=target_prefix_start_frame,
            )
            for index in range(config.expected_episode_count)
        ]
    raw_ready = bool(
        inventory["file_count_matches"]
        and inventory["camera_recordings_complete"]
        and inventory["tactile_recordings_complete"]
        and calibration.get("complete")
        and calibration.get("camera_count", 0) >= config.minimum_calibrated_camera_count
    )
    result: dict[str, Any] = {
        "schema_version": DEFORM360_PREFLIGHT_SCHEMA_VERSION,
        "artifact_kind": "Deform360001RopePreflight",
        "protocol_id": config.protocol_id,
        "upstream": {
            "code_repository": config.code_repository,
            "code_commit": config.code_commit,
            "dataset_repository": config.dataset_repository,
            "dataset_revision": config.dataset_revision,
        },
        "information_boundary": {
            "split_frozen_before_prediction": True,
            "split_uses_metadata_only": True,
            "prediction_metrics_computed": False,
            "model_parameters_fitted": False,
            "target_tactile_values_read": any(
                episode.get("target_prefix_values_read", False)
                or episode.get("target_oracle_values_read", False)
                for episode in processed
            ),
            "target_prefix_values_read": any(
                episode.get("target_prefix_values_read", False) for episode in processed
            ),
            "target_oracle_values_read": any(
                episode.get("target_oracle_values_read", False) for episode in processed
            ),
            "target_prefix_unlock_requested": unlock_target_prefix,
            "target_oracle_unlock_requested": unlock_target_oracle,
            "target_prefix_selection": {
                "method": config.prefix_trigger_method,
                "confirmation_frames": config.prefix_trigger_confirmation_frames,
                "aggregation": config.prefix_trigger_aggregation,
                "start_frame": target_prefix_start_frame,
                "frame_count": config.prefix_frame_count,
                "target_tactile_used_to_select": False,
            },
        },
        "raw_inventory": inventory,
        "calibration": calibration,
        "metadata_sha256": _sha256_file(metadata_path),
        "split": {
            "method": "SHA-256 rank of metadata-only episode ids",
            "seed": config.split_seed,
            "assignments": assignments,
            "counts": {
                name: sum(item["split"] == name for item in assignments)
                for name in ("source", "calibration", "target")
            },
            "held_out_action": next(
                item["action"] for item in assignments if item["split"] == "target"
            ),
        },
        "processed_episodes": processed,
        "capability_gates": {
            "raw_cohort_complete": raw_ready,
            "synchronization_complete": bool(processed)
            and all(
                episode.get("alignment", {}).get("manifest_available")
                for episode in processed
            ),
            "tactile_preprocessing_complete": bool(processed)
            and all(
                episode.get("tactile", {}).get("sensor_count")
                == config.expected_tactile_sensor_count
                for episode in processed
            ),
            "robot_pose_complete": bool(processed)
            and all(episode.get("robot", {}).get("available") for episode in processed),
            "metric_geometry_complete": bool(processed)
            and all(
                episode.get("metric_geometry", {}).get("ready") for episode in processed
            ),
            "dense_tracks_complete": bool(processed)
            and all(episode.get("tracking", {}).get("ready") for episode in processed),
            "phystwin_adapter_complete": bool(processed)
            and all(
                episode.get("phystwin_adapter", {}).get("final_data_available")
                for episode in processed
            ),
            "command_vs_measured_inference_supported": False,
            "calibrated_force_inference_supported": False,
            "slip_ground_truth_supported": False,
        },
        "signal_contract": {
            "tactile": "unitless, per-episode peak-normalized normal response",
            "robot": "vision-recovered/measured wrist pose and opening",
            "not_available": [
                "commanded controller trajectory",
                "calibrated force or torque",
                "tangential tactile slip measurement",
            ],
        },
        "comparison_contract": {
            "visual_only": (
                "RGB/geometry and causally available robot pose/opening; no tactile values"
            ),
            "tactile_conditioned_z": (
                "source/calibration tactile and the visually triggered target tactile "
                "prefix only"
            ),
            "oracle_tactile": "full target tactile contact used only after predictions are sealed",
            "primary_target": "held-out interventional prediction for move both edges",
            "future_robot_trajectory": (
                "action-conditioning evidence for the benchmark, not a commanded-control "
                "measurement"
            ),
        },
        "preflight_passed": raw_ready,
    }
    result["result_sha256"] = preflight_result_sha256(result)
    return result


def preflight_result_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return _sha256_bytes(_canonical_bytes(canonical))


def validate_deform360_preflight(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        payload.get("schema_version") == DEFORM360_PREFLIGHT_SCHEMA_VERSION,
        "unsupported preflight schema",
    )
    _require(
        payload.get("artifact_kind") == "Deform360001RopePreflight",
        "unexpected artifact kind",
    )
    _require(
        payload.get("result_sha256") == preflight_result_sha256(payload),
        "preflight checksum mismatch",
    )
    boundary = payload["information_boundary"]
    _require(
        boundary["prediction_metrics_computed"] is False,
        "preflight computed prediction metrics",
    )
    _require(
        boundary["model_parameters_fitted"] is False,
        "preflight fitted model parameters",
    )
    _require(
        boundary["split_uses_metadata_only"] is True,
        "split crossed the information boundary",
    )
    return {
        "passed": True,
        "preflight_passed": bool(payload["preflight_passed"]),
        "result_sha256": payload["result_sha256"],
        "held_out_action": payload["split"]["held_out_action"],
    }


def write_deform360_preflight(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output
