"""Prediction-only prefix custody for the V12 causal-response experiment."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .observation_belief import array_sha256, file_sha256

CONTRACT = "deform360-causal-response-prefix-v12"
REPORT_FILENAME = "causal_response_prefix.json"
ARCHIVE_FILENAME = "causal_response_prefix.npz"


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _readonly(values: np.ndarray, *, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    array.setflags(write=False)
    return array


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(
        b"deform360-causal-response-prefix-v12\0"
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class CausalResponsePrefixConfig:
    """Registered dimensions and strict future-observation boundary."""

    prefix_frame_count: int = 58
    minimum_camera_count: int = 6

    def __post_init__(self) -> None:
        _require(self.prefix_frame_count >= 2, "prefix is too short")
        _require(
            self.minimum_camera_count >= 6,
            "two three-camera panels are required",
        )


@dataclass(frozen=True)
class CausalResponsePrefixInputs:
    """Depth, mask, tactile, and measured actuation through frame 57 only."""

    config: CausalResponsePrefixConfig
    camera_ids: tuple[str, ...]
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    depths_m: np.ndarray
    object_masks: np.ndarray
    tactile_contact_probability: np.ndarray
    measured_actuator_positions_m: np.ndarray

    def __post_init__(self) -> None:
        intrinsics = _readonly(self.intrinsics, dtype=np.float64)
        poses = _readonly(self.camera_to_world, dtype=np.float64)
        depths = _readonly(self.depths_m, dtype=np.float32)
        masks = _readonly(self.object_masks, dtype=bool)
        tactile = _readonly(
            self.tactile_contact_probability,
            dtype=np.float64,
        )
        actuator = _readonly(
            self.measured_actuator_positions_m,
            dtype=np.float64,
        )
        if actuator.ndim == 2:
            actuator = _readonly(actuator[:, None], dtype=np.float64)
        camera_count = len(self.camera_ids)
        frame_count = self.config.prefix_frame_count
        _require(
            camera_count >= self.config.minimum_camera_count
            and len(set(self.camera_ids)) == camera_count
            and all(str(camera).strip() for camera in self.camera_ids),
            "prefix camera identifiers are invalid",
        )
        _require(
            intrinsics.shape == (camera_count, 3, 3)
            and poses.shape == (camera_count, 4, 4)
            and depths.ndim == 4
            and depths.shape[:2] == (camera_count, frame_count)
            and masks.shape == depths.shape,
            "prefix camera arrays changed shape",
        )
        _require(
            tactile.shape == (frame_count,)
            and actuator.ndim == 3
            and actuator.shape[0] == frame_count
            and actuator.shape[2] == 3,
            "prefix causal-support arrays changed shape",
        )
        _require(
            np.all(np.isfinite(intrinsics))
            and np.all(np.isfinite(poses))
            and np.all(np.isfinite(depths))
            and np.all(depths >= 0.0)
            and np.all(np.isfinite(tactile))
            and np.all((tactile >= 0.0) & (tactile <= 1.0))
            and np.all(np.isfinite(actuator)),
            "prefix arrays contain invalid values",
        )
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "camera_to_world", poses)
        object.__setattr__(self, "depths_m", depths)
        object.__setattr__(self, "object_masks", masks)
        object.__setattr__(self, "tactile_contact_probability", tactile)
        object.__setattr__(
            self,
            "measured_actuator_positions_m",
            actuator,
        )

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "camera_ids": np.asarray(self.camera_ids, dtype=np.str_),
            "intrinsics": self.intrinsics,
            "camera_to_world": self.camera_to_world,
            "depths_m": self.depths_m,
            "object_masks": self.object_masks,
            "tactile_contact_probability": self.tactile_contact_probability,
            "measured_actuator_positions_m": (self.measured_actuator_positions_m),
        }

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_kind": "Deform360CausalResponsePrefixInputs",
            "contract": CONTRACT,
            "config": asdict(self.config),
            "camera_ids": list(self.camera_ids),
            "array_sha256": {
                name: array_sha256(values)
                for name, values in sorted(self.arrays().items())
            },
            "information_boundary": {
                "frame_range_half_open": [
                    0,
                    self.config.prefix_frame_count,
                ],
                "maximum_observation_frame": (self.config.prefix_frame_count - 1),
                "rgb_included": False,
                "future_depth_or_mask_included": False,
                "future_tactile_included": False,
                "future_actuator_measurement_included": False,
                "identity_or_metric_outcome_included": False,
                "held_v8_artifact_or_process_access": False,
            },
        }


def write_causal_response_prefix_artifacts(
    output_dir: str | Path,
    inputs: CausalResponsePrefixInputs,
    *,
    case_id: str,
    protocol_path: str | Path,
    source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Seal a strict prediction prefix before any outcome is authorized."""

    _require(bool(case_id.strip()), "case ID is empty")
    source_hashes = dict(sorted(source_sha256.items()))
    _require(
        source_hashes
        and all(
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            for digest in source_hashes.values()
        ),
        "prefix source digests are invalid",
    )
    output = Path(output_dir).resolve()
    _require(not output.exists(), "prefix output already exists")
    output.mkdir(parents=True)
    arrays = inputs.arrays()
    archive_path = output / ARCHIVE_FILENAME
    temporary = archive_path.with_name(archive_path.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        **{
            name: np.ascontiguousarray(np.asarray(values))
            for name, values in arrays.items()
        },
    )
    temporary.replace(archive_path)
    report: dict[str, Any] = {
        **inputs.descriptor(),
        "case": case_id,
        "protocol": {
            "path": str(Path(protocol_path)),
            "file_sha256": file_sha256(protocol_path),
        },
        "source_sha256": source_hashes,
        "archive": {
            "filename": ARCHIVE_FILENAME,
            "file_sha256": file_sha256(archive_path),
        },
    }
    report["result_sha256"] = _canonical_sha256(report)
    (output / REPORT_FILENAME).write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    validate_causal_response_prefix_artifacts(output)
    return report


def validate_causal_response_prefix_artifacts(
    output_dir: str | Path,
) -> tuple[dict[str, Any], CausalResponsePrefixInputs]:
    """Validate custody, archive hashes, and exact prefix cardinality."""

    output = Path(output_dir).resolve()
    report = json.loads((output / REPORT_FILENAME).read_text(encoding="utf-8"))
    _require(
        report.get("artifact_kind") == "Deform360CausalResponsePrefixInputs"
        and report.get("contract") == CONTRACT
        and report.get("result_sha256") == _canonical_sha256(report),
        "prefix report is invalid",
    )
    boundary = report["information_boundary"]
    _require(
        boundary.get("rgb_included") is False
        and boundary.get("future_depth_or_mask_included") is False
        and boundary.get("future_tactile_included") is False
        and boundary.get("future_actuator_measurement_included") is False
        and boundary.get("identity_or_metric_outcome_included") is False
        and boundary.get("held_v8_artifact_or_process_access") is False,
        "prefix crossed its information boundary",
    )
    archive_path = output / ARCHIVE_FILENAME
    _require(
        report["archive"]["file_sha256"] == file_sha256(archive_path),
        "prefix archive checksum changed",
    )
    with np.load(archive_path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    inputs = CausalResponsePrefixInputs(
        config=CausalResponsePrefixConfig(**report["config"]),
        camera_ids=tuple(map(str, arrays["camera_ids"])),
        intrinsics=arrays["intrinsics"],
        camera_to_world=arrays["camera_to_world"],
        depths_m=arrays["depths_m"],
        object_masks=arrays["object_masks"],
        tactile_contact_probability=arrays["tactile_contact_probability"],
        measured_actuator_positions_m=arrays["measured_actuator_positions_m"],
    )
    _require(
        inputs.descriptor()["array_sha256"] == report["array_sha256"],
        "prefix arrays differ from the report",
    )
    return report, inputs


__all__ = [
    "ARCHIVE_FILENAME",
    "CONTRACT",
    "CausalResponsePrefixConfig",
    "CausalResponsePrefixInputs",
    "REPORT_FILENAME",
    "validate_causal_response_prefix_artifacts",
    "write_causal_response_prefix_artifacts",
]
