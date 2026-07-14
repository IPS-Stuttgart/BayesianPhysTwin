"""Prefix-only SAM2 masks for the sealed Deform360 target episode."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .deform360 import (
    Deform360ProtocolConfig,
    validate_deform360_preflight,
)
from .deform360_contact import validate_contact_artifact
from .deform360_sam2 import (
    PINNED_SAM2_CHECKPOINT_SHA256,
    PINNED_SAM2_CHECKPOINT_URL,
    PINNED_SAM2_COMMIT,
    PINNED_SAM2_MODEL_CONFIG,
    PINNED_SAM2_REPOSITORY,
    RopeSam2VideoPredictor,
)
from .deform360_sam2_views import validate_sam2_view_audit


DEFORM360_SAM2_PREFIX_MASK_SCHEMA_VERSION = 2


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


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def target_prefix_bounds(
    config: Deform360ProtocolConfig,
    contact_prediction_seal: Mapping[str, Any],
) -> tuple[int, int]:
    """Return the target interval authorized by the immutable contact seal."""

    validate_contact_artifact(
        contact_prediction_seal,
        expected_kind="Deform360TargetContactPredictionSeal",
    )
    _require(
        contact_prediction_seal.get("protocol_id") == config.protocol_id,
        "contact prediction seal belongs to a different protocol",
    )
    target_index = config.target_episode_ids[0]
    _require(
        contact_prediction_seal.get("target_episode_id")
        == f"{config.object_id}/episode_{target_index:04d}",
        "contact prediction seal belongs to a different target episode",
    )
    boundary = contact_prediction_seal.get("information_boundary", {})
    _require(
        boundary.get("target_tactile_oracle_read") is False,
        "contact prediction seal has already opened the target oracle",
    )
    prefix = contact_prediction_seal.get("target_prefix", {})
    start = int(prefix["start_frame"])
    stop = int(prefix["stop_frame_exclusive"])
    frame_count = int(prefix["frame_count"])
    _require(start >= 0 and stop > start, "invalid sealed target-prefix interval")
    _require(
        stop - start == frame_count == config.prefix_frame_count,
        "sealed target-prefix length differs from the protocol",
    )
    return start, stop


def decode_video_frame_window(
    video_path: str | Path,
    start_frame: int,
    stop_frame_exclusive: int,
    *,
    capture_factory: Callable[[str], Any] | None = None,
) -> np.ndarray:
    """Decode exactly ``[start_frame, stop_frame_exclusive)`` as RGB frames."""

    _require(start_frame >= 0, "prefix start frame must be non-negative")
    _require(
        stop_frame_exclusive > start_frame,
        "prefix frame interval must be non-empty",
    )
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - GPU-host integration
        raise RuntimeError(
            "OpenCV is required for prefix-only video decoding"
        ) from error
    factory = capture_factory or cv2.VideoCapture
    capture = factory(str(Path(video_path)))
    try:
        is_opened = getattr(capture, "isOpened", None)
        _require(
            is_opened is None or bool(is_opened()),
            f"cannot open target-prefix video: {video_path}",
        )
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for frame_index in range(start_frame, stop_frame_exclusive):
            ok, bgr = capture.read()
            _require(ok, f"cannot read target-prefix frame {frame_index}: {video_path}")
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return np.stack(frames)


def select_source_locked_prefix_cameras(
    source_view_audit: Mapping[str, Any],
    preflight: Mapping[str, Any],
    *,
    minimum_synchronization_reliability: float,
    minimum_camera_count: int,
) -> dict[str, Any]:
    """Freeze target-prefix cameras from one source audit and source timing only."""

    validate_sam2_view_audit(source_view_audit)
    validate_deform360_preflight(preflight)
    _require(
        source_view_audit.get("protocol_id") == preflight.get("protocol_id"),
        "source view audit and preflight belong to different protocols",
    )
    _require(
        source_view_audit.get("episode_access", {}).get("split") == "source",
        "target-prefix camera policy requires a source view audit",
    )
    _require(
        0.0 <= minimum_synchronization_reliability <= 1.0,
        "invalid synchronization-reliability threshold",
    )
    _require(minimum_camera_count >= 2, "at least two prefix cameras are required")
    source_index = int(source_view_audit["episode_access"]["episode_index"])
    source_id = f"001-rope/episode_{source_index:04d}"
    episode = next(
        (
            item
            for item in preflight.get("processed_episodes", [])
            if item.get("episode_id") == source_id
        ),
        None,
    )
    _require(episode is not None, "source episode is absent from the preflight")
    quality = episode.get("alignment", {}).get("quality", {})
    reliability = {
        str(item["camera"]): float(item["synchronization_reliability"])
        for item in quality.get("cameras", [])
    }
    accepted_by_geometry = tuple(
        str(camera)
        for camera in source_view_audit["cross_view_consistency"]["accepted_cameras"]
    )
    missing = sorted(set(accepted_by_geometry) - set(reliability))
    _require(not missing, f"source timing reliability is missing for {missing}")
    selected = sorted(
        camera
        for camera in accepted_by_geometry
        if reliability[camera] >= minimum_synchronization_reliability
    )
    _require(
        len(selected) >= minimum_camera_count,
        "too few source-locked cameras pass the synchronization threshold",
    )
    rejected = sorted(set(accepted_by_geometry) - set(selected))
    return {
        "selection_scope": "source-only",
        "source_episode_id": source_id,
        "source_view_audit_result_sha256": source_view_audit["result_sha256"],
        "preflight_result_sha256": preflight["result_sha256"],
        "minimum_synchronization_reliability": minimum_synchronization_reliability,
        "selected_cameras": selected,
        "rejected_for_synchronization": [
            {"camera": camera, "reliability": reliability[camera]}
            for camera in rejected
        ],
        "selected_reliability": {camera: reliability[camera] for camera in selected},
    }


def _write_jpeg_frame_directory(frames_rgb: np.ndarray, output_dir: Path) -> None:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - GPU-host integration
        raise RuntimeError("OpenCV is required for SAM2 prefix frames") from error
    output_dir.mkdir(parents=True, exist_ok=False)
    for index, rgb in enumerate(frames_rgb):
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        ok = cv2.imwrite(
            str(output_dir / f"{index:06d}.jpg"),
            bgr,
            [cv2.IMWRITE_JPEG_QUALITY, 100],
        )
        _require(ok, f"cannot write temporary SAM2 prefix frame {index}")


def segment_target_prefix_camera(
    predictor: RopeSam2VideoPredictor,
    video_path: str | Path,
    output_mask_path: str | Path,
    *,
    start_frame: int,
    stop_frame_exclusive: int,
) -> dict[str, Any]:
    """Read, segment, and persist only the authorized target-prefix frames."""

    path = Path(video_path)
    frames = decode_video_frame_window(path, start_frame, stop_frame_exclusive)
    before = len(predictor.diagnostics)
    with tempfile.TemporaryDirectory(prefix="causal4d-sam2-prefix-") as temp:
        frame_dir = Path(temp) / "frames"
        _write_jpeg_frame_directory(frames, frame_dir)
        returned = list(predictor.segment(frame_dir, "striped rope"))
    _require(
        len(predictor.diagnostics) == before + 1,
        "SAM2 prefix predictor did not emit one camera diagnostic",
    )
    expected = list(range(len(frames)))
    _require(
        [index for index, _ in returned] == expected,
        "SAM2 prefix propagation did not return every authorized frame exactly once",
    )
    masks = np.stack([np.asarray(mask, dtype=np.uint8) for _, mask in returned])
    _require(masks.shape[:1] == frames.shape[:1], "SAM2 prefix mask count mismatch")
    output = Path(output_mask_path)
    _require(output.suffix == ".npy", "target-prefix mask output must end in .npy")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, masks, allow_pickle=False)
    diagnostic = dict(predictor.diagnostics[-1])
    diagnostic["camera"] = path.parent.name
    diagnostic["video"] = path.name
    return {
        "camera": path.parent.name,
        "source_video_path": str(path),
        "source_video_fully_hashed": False,
        "decoded_frame_indices": list(range(start_frame, stop_frame_exclusive)),
        "decoded_frame_sha256": [_sha256_array(frame) for frame in frames],
        "decoded_shape": list(frames.shape),
        "mask_path": str(output),
        "mask_sha256": _sha256_file(output),
        "mask_shape": list(masks.shape),
        "mask_dtype": str(masks.dtype),
        "diagnostic": diagnostic,
    }


def sam2_prefix_mask_artifact_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def build_sam2_prefix_mask_audit(
    *,
    config: Deform360ProtocolConfig,
    contact_prediction_seal: Mapping[str, Any],
    camera_policy: Mapping[str, Any],
    predictor: RopeSam2VideoPredictor,
    camera_outputs: Sequence[Mapping[str, Any]],
    camera_failures: Sequence[Mapping[str, Any]] = (),
    minimum_camera_count: int = 8,
) -> dict[str, Any]:
    start, stop = target_prefix_bounds(config, contact_prediction_seal)
    source_selected = list(camera_policy.get("selected_cameras", []))
    _require(source_selected, "target-prefix camera policy selected no cameras")
    outputs = [dict(item) for item in camera_outputs]
    failures = [dict(item) for item in camera_failures]
    successful = sorted(item["camera"] for item in outputs)
    failed = sorted(item["camera"] for item in failures)
    _require(
        not set(successful).intersection(failed),
        "a target-prefix camera is both successful and failed",
    )
    _require(
        sorted(successful + failed) == sorted(source_selected),
        "target-prefix attempts do not cover the source-locked camera policy",
    )
    _require(
        len(successful) >= minimum_camera_count,
        "too few target-prefix cameras produced SAM2 masks",
    )
    _require(
        all(
            item["decoded_frame_indices"] == list(range(start, stop))
            for item in outputs
        ),
        "a target-prefix output read frames outside the authorized interval",
    )
    _require(
        all(item["source_video_fully_hashed"] is False for item in outputs),
        "target source videos must not be fully hashed before prediction sealing",
    )
    effective_policy = dict(camera_policy)
    effective_policy["source_locked_selected_cameras"] = source_selected
    effective_policy["selected_cameras"] = successful
    effective_policy["target_prefix_segmentation_failures"] = failures
    effective_policy["minimum_successful_camera_count"] = minimum_camera_count
    payload: dict[str, Any] = {
        "schema_version": DEFORM360_SAM2_PREFIX_MASK_SCHEMA_VERSION,
        "artifact_kind": "Deform360RopeSam2PrefixMaskAudit",
        "protocol_id": config.protocol_id,
        "contact_prediction_seal_sha256": contact_prediction_seal["result_sha256"],
        "target_episode_id": contact_prediction_seal["target_episode_id"],
        "target_prefix": {
            "start_frame": start,
            "stop_frame_exclusive": stop,
            "frame_count": stop - start,
        },
        "camera_policy": effective_policy,
        "upstream": {
            "repository": PINNED_SAM2_REPOSITORY,
            "commit": PINNED_SAM2_COMMIT,
            "checkpoint_url": PINNED_SAM2_CHECKPOINT_URL,
            "checkpoint_sha256": PINNED_SAM2_CHECKPOINT_SHA256,
            "model_config": PINNED_SAM2_MODEL_CONFIG,
        },
        "parameters": asdict(predictor.config),
        "model_id": predictor.model_id,
        "information_boundary": {
            "target_visual_prefix_read": True,
            "target_future_visual_frames_read": False,
            "target_tactile_oracle_read": False,
            "target_prediction_metrics_computed": False,
            "full_target_video_hashes_computed": False,
            "camera_selection_used_target_prefix_measurement_availability": bool(
                failures
            ),
            "camera_selection_used_target_future_frames": False,
        },
        "outputs": outputs,
        "claim_boundary": (
            "Every source-locked camera was attempted on only the contact-sealed "
            "six-frame target RGB prefix. Deterministic prefix segmentation failures "
            "were recorded; target-future masks remain locked until predictions seal."
        ),
    }
    payload["result_sha256"] = sam2_prefix_mask_artifact_sha256(payload)
    return payload


def validate_sam2_prefix_mask_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        payload.get("schema_version") == DEFORM360_SAM2_PREFIX_MASK_SCHEMA_VERSION,
        "unsupported SAM2 prefix-mask artifact schema",
    )
    _require(
        payload.get("artifact_kind") == "Deform360RopeSam2PrefixMaskAudit",
        "unexpected SAM2 prefix-mask artifact kind",
    )
    _require(
        payload.get("result_sha256") == sam2_prefix_mask_artifact_sha256(payload),
        "SAM2 prefix-mask artifact checksum mismatch",
    )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("target_future_visual_frames_read") is False,
        "SAM2 prefix-mask artifact opened target-future visual frames",
    )
    _require(
        boundary.get("target_tactile_oracle_read") is False,
        "SAM2 prefix-mask artifact opened the target tactile oracle",
    )
    _require(
        _valid_sha256(payload.get("contact_prediction_seal_sha256")),
        "invalid contact prediction seal checksum",
    )
    return {
        "passed": True,
        "result_sha256": payload["result_sha256"],
        "camera_count": len(payload.get("outputs", [])),
    }


def write_sam2_prefix_mask_audit(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


__all__ = [
    "DEFORM360_SAM2_PREFIX_MASK_SCHEMA_VERSION",
    "build_sam2_prefix_mask_audit",
    "decode_video_frame_window",
    "sam2_prefix_mask_artifact_sha256",
    "segment_target_prefix_camera",
    "select_source_locked_prefix_cameras",
    "target_prefix_bounds",
    "validate_sam2_prefix_mask_artifact",
    "write_sam2_prefix_mask_audit",
]
