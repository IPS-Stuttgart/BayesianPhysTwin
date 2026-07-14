"""Post-seal SAM2 propagation over the Deform360 target suffix."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360 import Deform360ProtocolConfig
from .deform360_rope_evaluation import validate_held_out_rope_prediction_seal
from .deform360_sam2 import RopeSam2VideoPredictor
from .deform360_sam2_prefix import (
    _write_jpeg_frame_directory,
    decode_video_frame_window,
    validate_sam2_prefix_mask_artifact,
)


DEFORM360_SAM2_SUFFIX_MASK_SCHEMA_VERSION = 1


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


def sam2_suffix_mask_artifact_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def segment_target_suffix_camera(
    predictor: RopeSam2VideoPredictor,
    video_path: str | Path,
    prefix_output: Mapping[str, Any],
    output_mask_path: str | Path,
    *,
    start_frame: int,
    stop_frame_exclusive: int,
    prefix_frame_count: int,
) -> dict[str, Any]:
    """Propagate one sealed prefix mask through the now-authorized suffix."""

    path = Path(video_path)
    frames = decode_video_frame_window(path, start_frame, stop_frame_exclusive)
    _require(
        [_sha256_array(frame) for frame in frames[:prefix_frame_count]]
        == prefix_output["decoded_frame_sha256"],
        "suffix decode does not reproduce the sealed prefix frames",
    )
    prefix_mask_path = Path(prefix_output["mask_path"])
    _require(
        _sha256_file(prefix_mask_path) == prefix_output["mask_sha256"],
        "sealed prefix mask checksum mismatch",
    )
    prefix_masks = np.load(prefix_mask_path, allow_pickle=False)
    initial_mask = np.asarray(prefix_masks[0], dtype=bool)
    before = len(predictor.diagnostics)
    with tempfile.TemporaryDirectory(prefix="causal4d-sam2-suffix-") as temp:
        frame_dir = Path(temp) / "frames"
        _write_jpeg_frame_directory(frames, frame_dir)
        returned = list(
            predictor.segment_from_initial_mask(
                frame_dir,
                initial_mask,
                initialization={
                    "prefix_mask_sha256": prefix_output["mask_sha256"],
                    "absolute_frame_index": start_frame,
                },
            )
        )
    _require(
        len(predictor.diagnostics) == before + 1,
        "SAM2 suffix predictor did not emit one camera diagnostic",
    )
    expected = list(range(len(frames)))
    _require(
        [index for index, _ in returned] == expected,
        "SAM2 suffix propagation did not return every frame",
    )
    masks = np.stack([np.asarray(mask, dtype=np.uint8) for _, mask in returned])
    output = Path(output_mask_path).resolve()
    _require(output.suffix == ".npy", "suffix mask output must end in .npy")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, masks, allow_pickle=False)
    return {
        "camera": path.parent.name,
        "source_video_path": str(path),
        "source_video_fully_hashed": False,
        "frame_start": start_frame,
        "frame_stop_exclusive": stop_frame_exclusive,
        "frame_count": len(frames),
        "mask_path": str(output),
        "mask_sha256": _sha256_file(output),
        "mask_shape": list(masks.shape),
        "mask_dtype": str(masks.dtype),
        "diagnostic": dict(predictor.diagnostics[-1]),
    }


def build_sam2_suffix_mask_audit(
    *,
    protocol: Deform360ProtocolConfig,
    held_out_prediction_seal: Mapping[str, Any],
    prefix_mask_audit: Mapping[str, Any],
    predictor: RopeSam2VideoPredictor,
    camera_outputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    validate_held_out_rope_prediction_seal(held_out_prediction_seal)
    validate_sam2_prefix_mask_artifact(prefix_mask_audit)
    _require(
        held_out_prediction_seal["protocol_id"]
        == prefix_mask_audit["protocol_id"]
        == protocol.protocol_id,
        "suffix mask inputs belong to different protocols",
    )
    cameras = list(prefix_mask_audit["camera_policy"]["selected_cameras"])
    outputs = [dict(row) for row in camera_outputs]
    _require(
        sorted(row["camera"] for row in outputs) == sorted(cameras),
        "suffix outputs do not cover every prefix camera",
    )
    starts = {int(row["frame_start"]) for row in outputs}
    stops = {int(row["frame_stop_exclusive"]) for row in outputs}
    _require(len(starts) == len(stops) == 1, "suffix camera intervals disagree")
    start = starts.pop()
    stop = stops.pop()
    _require(
        start == int(prefix_mask_audit["target_prefix"]["start_frame"]),
        "suffix propagation does not begin at the sealed prefix",
    )
    _require(
        int(held_out_prediction_seal["future_start_frame"]) < stop,
        "suffix outputs contain no held-out future",
    )
    payload: dict[str, Any] = {
        "schema_version": DEFORM360_SAM2_SUFFIX_MASK_SCHEMA_VERSION,
        "artifact_kind": "Deform360TargetRopeSam2PostSealMaskAudit",
        "protocol_id": protocol.protocol_id,
        "target_episode_id": held_out_prediction_seal["target_episode_id"],
        "held_out_prediction_seal_sha256": held_out_prediction_seal["result_sha256"],
        "prefix_mask_audit_result_sha256": prefix_mask_audit["result_sha256"],
        "frame_start": start,
        "frame_stop_exclusive": stop,
        "future_start_frame": held_out_prediction_seal["future_start_frame"],
        "model_id": predictor.model_id,
        "outputs": outputs,
        "information_boundary": {
            "deployable_predictions_previously_sealed": True,
            "target_future_visual_frames_read": True,
            "target_future_visual_frames_used_for_fitting": False,
            "target_tactile_oracle_read": False,
        },
        "claim_boundary": (
            "Post-seal target annotation only. SAM2 object identity was initialized "
            "from each immutable frame-103 prefix mask."
        ),
    }
    payload["result_sha256"] = sam2_suffix_mask_artifact_sha256(payload)
    return payload


def validate_sam2_suffix_mask_artifact(
    payload: Mapping[str, Any], *, verify_outputs: bool = True
) -> dict[str, Any]:
    _require(
        payload.get("schema_version") == DEFORM360_SAM2_SUFFIX_MASK_SCHEMA_VERSION,
        "unsupported SAM2 suffix-mask schema",
    )
    _require(
        payload.get("artifact_kind") == "Deform360TargetRopeSam2PostSealMaskAudit",
        "unexpected SAM2 suffix-mask artifact kind",
    )
    _require(
        payload.get("result_sha256") == sam2_suffix_mask_artifact_sha256(payload),
        "SAM2 suffix-mask checksum mismatch",
    )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("deployable_predictions_previously_sealed") is True,
        "target suffix opened before prediction sealing",
    )
    _require(
        boundary.get("target_tactile_oracle_read") is False,
        "target suffix mask stage read the tactile oracle",
    )
    if verify_outputs:
        for output in payload.get("outputs", []):
            path = Path(output["mask_path"])
            _require(path.is_file(), "SAM2 suffix mask output is missing")
            _require(
                _sha256_file(path) == output["mask_sha256"],
                "SAM2 suffix mask checksum mismatch",
            )
    return {
        "passed": True,
        "result_sha256": payload["result_sha256"],
        "camera_count": len(payload.get("outputs", [])),
    }


def write_sam2_suffix_mask_audit(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


__all__ = [
    "DEFORM360_SAM2_SUFFIX_MASK_SCHEMA_VERSION",
    "build_sam2_suffix_mask_audit",
    "sam2_suffix_mask_artifact_sha256",
    "segment_target_suffix_camera",
    "validate_sam2_suffix_mask_artifact",
    "write_sam2_suffix_mask_audit",
]
