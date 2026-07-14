"""Post-seal target-future rope geometry for Deform360 ``001-rope``."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .deform360 import Deform360ProtocolConfig
from .deform360_rope_evaluation import validate_held_out_rope_prediction_seal
from .deform360_rope_graph import RopeCenterlineConfig, extract_rope_centerline
from .deform360_rope_prefix import validate_target_prefix_rope_geometry
from .deform360_sam2_suffix import validate_sam2_suffix_mask_artifact
from .deform360_visual_hull import AdaptiveRopeHullConfig, adaptive_rope_visual_hull


DEFORM360_ROPE_FUTURE_GEOMETRY_SCHEMA_VERSION = 1


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    descriptor = _canonical_bytes(
        {"dtype": array.dtype.str, "shape": list(array.shape)}
    )
    return hashlib.sha256(
        descriptor + b"\0" + array.view(np.uint8).tobytes()
    ).hexdigest()


@dataclass(frozen=True)
class RopeFutureGeometryConfig:
    maximum_temporal_node_displacement_m: float = 0.05

    def __post_init__(self) -> None:
        _require(
            self.maximum_temporal_node_displacement_m > 0.0,
            "future temporal-displacement gate must be positive",
        )


def rope_future_geometry_artifact_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def build_target_future_rope_geometry(
    processed_root: str | Path,
    protocol: Deform360ProtocolConfig,
    held_out_prediction_seal: Mapping[str, Any],
    prefix_geometry: Mapping[str, Any],
    suffix_mask_audit: Mapping[str, Any],
    output_archive_path: str | Path,
    *,
    config: RopeFutureGeometryConfig = RopeFutureGeometryConfig(),
    centerline_config: RopeCenterlineConfig = RopeCenterlineConfig(),
    hull_config: AdaptiveRopeHullConfig = AdaptiveRopeHullConfig(),
) -> dict[str, Any]:
    """Reconstruct the untouched target future after prediction sealing."""

    validate_held_out_rope_prediction_seal(held_out_prediction_seal)
    validate_target_prefix_rope_geometry(prefix_geometry)
    validate_sam2_suffix_mask_artifact(suffix_mask_audit)
    _require(
        held_out_prediction_seal["protocol_id"]
        == prefix_geometry["protocol_id"]
        == suffix_mask_audit["protocol_id"]
        == protocol.protocol_id,
        "future geometry inputs belong to different protocols",
    )
    _require(
        suffix_mask_audit["held_out_prediction_seal_sha256"]
        == held_out_prediction_seal["result_sha256"],
        "suffix masks were opened under another prediction seal",
    )
    _require(
        held_out_prediction_seal["target_prefix_geometry_sha256"]
        == prefix_geometry["result_sha256"],
        "prediction seal used another target-prefix geometry",
    )
    target_index = protocol.target_episode_ids[0]
    episode_dir = Path(processed_root).resolve() / f"episode_{target_index:04d}"
    try:
        from deform360.processing.episode import load_episode_calibration
    except ImportError as error:  # pragma: no cover - pinned host integration
        raise RuntimeError(
            "the pinned Deform360 processing environment is required"
        ) from error
    intrinsics, extrinsics = load_episode_calibration(episode_dir)
    cameras = [row["camera"] for row in suffix_mask_audit["outputs"]]
    _require(len(cameras) >= 8, "future geometry has fewer than eight cameras")
    mask_start = int(suffix_mask_audit["frame_start"])
    mask_stop = int(suffix_mask_audit["frame_stop_exclusive"])
    future_start = int(held_out_prediction_seal["future_start_frame"])
    expected_frame_count = int(held_out_prediction_seal["prediction_shape"][0])
    _require(
        mask_stop - future_start == expected_frame_count,
        "future mask interval and sealed prediction length disagree",
    )
    masks = {}
    mask_inputs = []
    for output in suffix_mask_audit["outputs"]:
        path = Path(output["mask_path"])
        _require(_sha256_file(path) == output["mask_sha256"], "future mask changed")
        values = np.load(path, allow_pickle=False)
        _require(
            len(values) == mask_stop - mask_start,
            "future mask frame count mismatch",
        )
        masks[output["camera"]] = np.asarray(values, dtype=np.uint8)
        mask_inputs.append(
            {"camera": output["camera"], "mask_sha256": output["mask_sha256"]}
        )
    with np.load(prefix_geometry["archive"]["path"], allow_pickle=False) as stored:
        previous = np.asarray(stored["centerlines_m"][-1], dtype=np.float64)
    frame_indices = np.arange(future_start, mask_stop, dtype=np.int32)
    centerlines = []
    frame_diagnostics = []
    for frame_index in frame_indices:
        local_index = int(frame_index - mask_start)
        frame_masks = {camera: masks[camera][local_index] for camera in cameras}
        hull, hull_diagnostics = adaptive_rope_visual_hull(
            previous,
            frame_masks,
            {camera: intrinsics[camera] for camera in cameras},
            {camera: extrinsics[camera] for camera in cameras},
            config=hull_config,
        )
        current, centerline_diagnostics = extract_rope_centerline(
            hull,
            config=centerline_config,
            initial_centerline_m=previous,
            reference_centerline_m=previous,
        )
        centerlines.append(current)
        frame_diagnostics.append(
            {
                "frame_index": int(frame_index),
                "adaptive_hull": hull_diagnostics,
                "centerline": centerline_diagnostics,
            }
        )
        previous = current
    trajectories = np.asarray(centerlines, dtype=np.float64)
    _require(
        list(trajectories.shape)
        == [
            expected_frame_count,
            int(held_out_prediction_seal["prediction_shape"][1]),
            3,
        ],
        "future centerline shape differs from the prediction seal",
    )
    temporal = np.linalg.norm(np.diff(trajectories, axis=0), axis=2)
    maximum_temporal = float(np.max(temporal)) if temporal.size else 0.0
    quality = {
        "passed": maximum_temporal <= config.maximum_temporal_node_displacement_m,
        "maximum_temporal_node_displacement_m": {
            "value": maximum_temporal,
            "maximum": config.maximum_temporal_node_displacement_m,
        },
        "centerline_length_m": {
            "first": float(
                np.linalg.norm(np.diff(trajectories[0], axis=0), axis=1).sum()
            ),
            "last": float(
                np.linalg.norm(np.diff(trajectories[-1], axis=0), axis=1).sum()
            ),
        },
    }
    output = Path(output_archive_path).resolve()
    _require(output.suffix == ".npz", "future geometry archive must end in .npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, frame_indices=frame_indices, centerlines_m=trajectories)
    payload: dict[str, Any] = {
        "schema_version": DEFORM360_ROPE_FUTURE_GEOMETRY_SCHEMA_VERSION,
        "artifact_kind": "Deform360TargetFutureRopeGeometry",
        "protocol_id": protocol.protocol_id,
        "target_episode_id": held_out_prediction_seal["target_episode_id"],
        "held_out_prediction_seal_sha256": held_out_prediction_seal["result_sha256"],
        "prefix_geometry_result_sha256": prefix_geometry["result_sha256"],
        "suffix_mask_audit_result_sha256": suffix_mask_audit["result_sha256"],
        "parameters": {
            "future_geometry": asdict(config),
            "centerline": asdict(centerline_config),
            "adaptive_hull": asdict(hull_config),
        },
        "frame_indices": frame_indices.astype(int).tolist(),
        "mask_inputs": mask_inputs,
        "frame_diagnostics": frame_diagnostics,
        "quality": quality,
        "archive": {
            "path": str(output),
            "sha256": _sha256_file(output),
            "bytes": output.stat().st_size,
            "centerlines_sha256": _sha256_array(trajectories),
            "shape": list(trajectories.shape),
        },
        "information_boundary": {
            "deployable_predictions_previously_sealed": True,
            "target_future_geometry_read_for_evaluation": True,
            "target_future_geometry_used_for_fitting": False,
            "target_tactile_oracle_read": False,
        },
        "measurement_semantics": (
            "Ordered normalized-arc-length silhouette centerlines; Chamfer is the "
            "primary metric because nodes are pseudo-correspondences."
        ),
    }
    payload["result_sha256"] = rope_future_geometry_artifact_sha256(payload)
    return payload


def validate_target_future_rope_geometry(
    payload: Mapping[str, Any], *, verify_archive: bool = True
) -> dict[str, Any]:
    _require(
        payload.get("schema_version") == DEFORM360_ROPE_FUTURE_GEOMETRY_SCHEMA_VERSION,
        "unsupported target-future geometry schema",
    )
    _require(
        payload.get("artifact_kind") == "Deform360TargetFutureRopeGeometry",
        "unexpected target-future geometry artifact kind",
    )
    _require(
        payload.get("result_sha256") == rope_future_geometry_artifact_sha256(payload),
        "target-future geometry checksum mismatch",
    )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("deployable_predictions_previously_sealed") is True,
        "target-future geometry opened before prediction sealing",
    )
    _require(
        boundary.get("target_future_geometry_used_for_fitting") is False,
        "target-future geometry was used for fitting",
    )
    _require(
        payload.get("quality", {}).get("passed") is True, "future geometry failed QA"
    )
    if verify_archive:
        archive = Path(payload["archive"]["path"])
        _require(archive.is_file(), "target-future geometry archive is missing")
        _require(
            _sha256_file(archive) == payload["archive"]["sha256"],
            "target-future geometry archive checksum mismatch",
        )
        with np.load(archive, allow_pickle=False) as stored:
            _require(
                _sha256_array(stored["centerlines_m"])
                == payload["archive"]["centerlines_sha256"],
                "target-future centerline checksum mismatch",
            )
    return {
        "passed": True,
        "result_sha256": payload["result_sha256"],
        "frame_count": len(payload["frame_indices"]),
    }


def write_target_future_rope_geometry(
    path: str | Path, payload: Mapping[str, Any]
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


__all__ = [
    "DEFORM360_ROPE_FUTURE_GEOMETRY_SCHEMA_VERSION",
    "RopeFutureGeometryConfig",
    "build_target_future_rope_geometry",
    "rope_future_geometry_artifact_sha256",
    "validate_target_future_rope_geometry",
    "write_target_future_rope_geometry",
]
