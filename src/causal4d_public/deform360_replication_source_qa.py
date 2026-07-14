"""Source-only multiview mask and camera gate for the locked replication."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360_object_sam2 import (
    DeformableObjectSam2MaskConfig,
    DeformableObjectSam2VideoPredictor,
)
from .deform360_replication import (
    CANONICAL_REPLICATION_CONFIG_SHA256,
    PINNED_DATASET_REVISION,
    validate_deform360_replication_protocol,
)
from .deform360_sam2_views import (
    CrossViewMaskReliabilityConfig,
    multiview_mask_consistency,
)


SOURCE_QA_POLICY_SCHEMA_VERSION = 1
SOURCE_QA_ARTIFACT_SCHEMA_VERSION = 1
SOURCE_QA_POLICY_ID = "deform360-replication-source-geometry-qa-v1"
CANONICAL_SOURCE_QA_POLICY_SHA256 = (
    "f5c3f4577bab648bf3da2537c28b7a77e87c055b32e1284c179a192e7569909b"
)


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


def source_qa_policy_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def validate_source_qa_policy(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        payload.get("schema_version") == SOURCE_QA_POLICY_SCHEMA_VERSION,
        "unsupported source-QA policy schema",
    )
    observed = source_qa_policy_sha256(payload)
    _require(payload.get("config_sha256") == observed, "source-QA checksum mismatch")
    _require(
        observed == CANONICAL_SOURCE_QA_POLICY_SHA256,
        "source-QA policy differs from the canonical lock",
    )
    config = payload["config"]
    _require(config["policy_id"] == SOURCE_QA_POLICY_ID, "source-QA id changed")
    _require(
        config["replication_config_sha256"]
        == CANONICAL_REPLICATION_CONFIG_SHA256,
        "source-QA replication lock changed",
    )
    _require(
        config["dataset_revision"] == PINNED_DATASET_REVISION,
        "source-QA dataset revision changed",
    )
    _require(
        config["minimum_cross_view_camera_count"]
        >= config["selected_camera_count"]
        >= 8,
        "source-QA camera gate is too small",
    )
    boundary = config["information_boundary"]
    _require(
        boundary == {
            "source_first_frames_only": True,
            "target_media_allowed": False,
            "target_outcomes_allowed": False,
            "selection_metrics_allowed": False,
        },
        "source-QA information boundary changed",
    )
    return {"passed": True, "policy_id": SOURCE_QA_POLICY_ID, "config_sha256": observed}


def load_source_qa_policy(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_source_qa_policy(payload)
    return payload


def select_diverse_cameras(
    accepted_cameras: Sequence[str],
    camera_to_world: Mapping[str, np.ndarray],
    *,
    reference_camera: str,
    selected_count: int,
) -> list[str]:
    """Keep the source anchor and greedily maximize camera-center coverage."""

    accepted = sorted(set(accepted_cameras))
    _require(reference_camera in accepted, "reference camera failed cross-view QA")
    _require(len(accepted) >= selected_count >= 1, "insufficient accepted cameras")
    positions = {
        camera: np.asarray(camera_to_world[camera], dtype=np.float64)[:3, 3]
        for camera in accepted
    }
    for camera, position in positions.items():
        _require(position.shape == (3,), f"invalid camera center for {camera}")
        _require(np.isfinite(position).all(), f"non-finite camera center for {camera}")
    selected = [reference_camera]
    while len(selected) < selected_count:
        candidates = [camera for camera in accepted if camera not in selected]
        next_camera = max(
            candidates,
            key=lambda camera: (
                min(
                    float(np.linalg.norm(positions[camera] - positions[chosen]))
                    for chosen in selected
                ),
                camera,
            ),
        )
        selected.append(next_camera)
    return selected


def source_qa_artifact_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def validate_source_qa_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        payload.get("schema_version") == SOURCE_QA_ARTIFACT_SCHEMA_VERSION,
        "unsupported source-QA artifact schema",
    )
    _require(
        payload.get("artifact_kind") == "Deform360ReplicationSourceGeometryQa",
        "unexpected source-QA artifact kind",
    )
    _require(
        payload.get("result_sha256") == source_qa_artifact_sha256(payload),
        "source-QA artifact checksum mismatch",
    )
    _require(payload.get("passed") is True, "source-QA artifact did not pass")
    boundary = payload.get("information_boundary", {})
    _require(boundary.get("target_media_read") is False, "source QA read target media")
    _require(
        boundary.get("target_metrics_computed") is False,
        "source QA computed target metrics",
    )
    return {
        "passed": True,
        "result_sha256": payload["result_sha256"],
        "object_count": len(payload.get("objects", [])),
    }


def _recording_path(camera_dir: Path, episode_index: int) -> tuple[Path, str]:
    try:
        from deform360.layout import camera_recordings, recording_for_episode
    except ImportError as error:  # pragma: no cover - GPU host integration
        raise RuntimeError("the pinned Deform360 runtime is required") from error
    recordings = camera_recordings(camera_dir)
    if len(recordings) == 1:
        return recordings[0].data_path, "single-source-recording-staging"
    return recording_for_episode(recordings, episode_index).data_path, "full-object-index"


def run_source_geometry_qa(
    raw_root: str | Path,
    replication_protocol: Mapping[str, Any],
    policy: Mapping[str, Any],
    predictor: DeformableObjectSam2VideoPredictor,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Select source-only camera subsets without reading target episodes."""

    validate_deform360_replication_protocol(replication_protocol)
    validate_source_qa_policy(policy)
    config = policy["config"]
    cohort = replication_protocol["config"]["cohort"]
    expected_objects = [record["object_id"] for record in cohort]
    _require(
        set(config["source_episode_by_object"]) == set(expected_objects),
        "source-QA object set differs from the replication cohort",
    )
    root = Path(raw_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        import cv2
        from deform360.calibration import load_calibration
        from deform360.undistort import build_undistort_maps, undistort_frame
    except ImportError as error:  # pragma: no cover - GPU host integration
        raise RuntimeError("the pinned Deform360 processing runtime is required") from error

    calibrations = {
        object_id: load_calibration(root / object_id) for object_id in expected_objects
    }
    candidate_cameras = sorted(
        set.intersection(*(set(value.cameras) for value in calibrations.values()))
    )
    _require(
        len(candidate_cameras) == config["expected_candidate_camera_count"],
        "common calibrated camera count changed",
    )
    reference_camera = config["reference_camera"]
    sam_config = DeformableObjectSam2MaskConfig(**config["sam2"])
    _require(asdict(predictor.config) == asdict(sam_config), "SAM2 parameters changed")
    reliability = CrossViewMaskReliabilityConfig(**config["cross_view"])
    object_records = []

    for object_id in expected_objects:
        source_episode = int(config["source_episode_by_object"][object_id])
        cohort_record = next(
            record for record in cohort if record["object_id"] == object_id
        )
        _require(
            source_episode in cohort_record["source_episode_ids"],
            f"{object_id} QA episode is not source-only",
        )
        object_dir = root / object_id
        calibration = calibrations[object_id]
        reference_video, reference_input_mode = _recording_path(
            object_dir / reference_camera, source_episode
        )
        raw_reference_mask, reference_diagnostic = predictor.select_initial_mask(
            reference_video
        )
        capture = cv2.VideoCapture(str(reference_video))
        ok, reference_raw = capture.read()
        capture.release()
        _require(ok, f"cannot decode source reference for {object_id}")
        height, width = reference_raw.shape[:2]
        reference_K, reference_map1, reference_map2 = build_undistort_maps(
            calibration.intrinsics[reference_camera],
            calibration.dist[reference_camera],
            (width, height),
        )
        reference_bgr = undistort_frame(
            reference_raw, reference_map1, reference_map2
        )
        reference_rgb = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2RGB)
        reference_mask = cv2.remap(
            raw_reference_mask.astype(np.uint8),
            reference_map1,
            reference_map2,
            cv2.INTER_NEAREST,
        ).astype(bool)
        masks = {reference_camera: reference_mask}
        intrinsics = {reference_camera: reference_K}
        view_records = []

        for camera in candidate_cameras:
            video, input_mode = _recording_path(object_dir / camera, source_episode)
            if camera == reference_camera:
                mask = reference_mask
                diagnostic = reference_diagnostic
                rectified = reference_bgr
                status = "reference"
                K_out = reference_K
            else:
                capture = cv2.VideoCapture(str(video))
                ok, raw_bgr = capture.read()
                capture.release()
                if not ok:
                    view_records.append(
                        {"camera": camera, "status": "decode-rejected"}
                    )
                    continue
                K_out, map1, map2 = build_undistort_maps(
                    calibration.intrinsics[camera],
                    calibration.dist[camera],
                    (raw_bgr.shape[1], raw_bgr.shape[0]),
                )
                rectified = undistort_frame(raw_bgr, map1, map2)
                frame_dir = output / "frames" / object_id / camera
                frame_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(frame_dir / "000000.jpg"), rectified)
                try:
                    mask, diagnostic = predictor.select_initial_mask_with_reference(
                        frame_dir,
                        reference_rgb,
                        reference_mask,
                        reference_camera=reference_camera,
                    )
                except ValueError as error:
                    view_records.append(
                        {
                            "camera": camera,
                            "status": "appearance-rejected",
                            "reason": str(error),
                            "input_mode": input_mode,
                            "video_sha256": _sha256_file(video),
                        }
                    )
                    continue
                status = "appearance-accepted"
                masks[camera] = mask
                intrinsics[camera] = K_out

            mask_path = output / "masks" / object_id / f"{camera}.png"
            overlay_path = output / "overlays" / object_id / f"{camera}.jpg"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
            overlay = rectified.copy()
            overlay[mask] = np.rint(
                0.45 * overlay[mask] + 0.55 * np.array([40, 220, 40])
            ).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
            cv2.imwrite(str(overlay_path), overlay)
            view_records.append(
                {
                    "camera": camera,
                    "status": status,
                    "input_mode": input_mode,
                    "video_sha256": _sha256_file(video),
                    "mask_sha256": _sha256_file(mask_path),
                    "overlay_sha256": _sha256_file(overlay_path),
                    "diagnostic": diagnostic,
                }
            )

        consistency = multiview_mask_consistency(
            masks,
            intrinsics,
            {camera: calibration.extrinsics[camera] for camera in masks},
            reliability,
        )
        accepted = consistency["accepted_cameras"]
        _require(
            len(accepted) >= config["minimum_cross_view_camera_count"],
            f"{object_id} has too few cross-view cameras",
        )
        selected = select_diverse_cameras(
            accepted,
            calibration.extrinsics,
            reference_camera=reference_camera,
            selected_count=config["selected_camera_count"],
        )
        object_records.append(
            {
                "object_id": object_id,
                "source_episode_index": source_episode,
                "reference_input_mode": reference_input_mode,
                "candidate_camera_count": len(candidate_cameras),
                "appearance_accepted_camera_count": len(masks),
                "cross_view_consistency": consistency,
                "selected_cameras": selected,
                "views": view_records,
            }
        )

    payload: dict[str, Any] = {
        "schema_version": SOURCE_QA_ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": "Deform360ReplicationSourceGeometryQa",
        "policy_id": SOURCE_QA_POLICY_ID,
        "policy_sha256": policy["config_sha256"],
        "replication_config_sha256": replication_protocol["config_sha256"],
        "candidate_cameras": candidate_cameras,
        "objects": object_records,
        "passed": True,
        "information_boundary": {
            "source_first_frames_only": True,
            "target_media_read": False,
            "target_metrics_computed": False,
        },
    }
    payload["result_sha256"] = source_qa_artifact_sha256(payload)
    validate_source_qa_artifact(payload)
    return payload


def write_source_qa_artifact(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


__all__ = [
    "CANONICAL_SOURCE_QA_POLICY_SHA256",
    "load_source_qa_policy",
    "run_source_geometry_qa",
    "select_diverse_cameras",
    "source_qa_artifact_sha256",
    "source_qa_policy_sha256",
    "validate_source_qa_artifact",
    "validate_source_qa_policy",
    "write_source_qa_artifact",
]
