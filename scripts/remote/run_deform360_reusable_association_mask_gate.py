#!/usr/bin/env python3
"""Run the frozen frame-zero mask gate on one Deform360 calibration episode."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from causal4d_public.deform360_dense_source import sha256_file
from causal4d_public.deform360_object_sam2 import (
    DeformableObjectSam2MaskConfig,
    DeformableObjectSam2VideoPredictor,
)
from causal4d_public.deform360_reusable_association import (
    load_reusable_association_config,
    validate_reusable_association_calibration_request,
)
from causal4d_public.deform360_sam2_views import (
    CrossViewMaskReliabilityConfig,
    JointMultiviewMaskSelectionConfig,
    select_joint_multiview_masks,
)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _first_frame_sha256(video_path: Path) -> str:
    capture = cv2.VideoCapture(str(video_path))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise ValueError(f"cannot read frame zero from {video_path}")
    array = np.ascontiguousarray(frame)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-qa-root", type=Path, required=True)
    parser.add_argument("--sam2-repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = (
        args.repo / "configs/causal4d_public/deform360_reusable_association_v2.json"
    )
    protocol = load_reusable_association_config(config_path)
    request = validate_reusable_association_calibration_request(
        protocol,
        object_id=args.object_id,
        episode_id=args.episode,
    )
    frozen = protocol["config"]

    source_policy_path = (
        args.repo / "configs/causal4d_public/deform360_replication_source_qa_v1.json"
    )
    source_policy = json.loads(source_policy_path.read_text(encoding="utf-8"))
    source_config = source_policy["config"]
    source_artifact_path = (
        args.repo
        / "milestones/deform360-replication-source-qa-v1/artifacts/source_geometry_qa.json"
    )
    source_artifact = json.loads(source_artifact_path.read_text(encoding="utf-8"))
    object_record = next(
        record
        for record in source_artifact["objects"]
        if record["object_id"] == args.object_id
    )
    cameras = tuple(object_record["selected_cameras"])
    reference_camera = source_config["reference_camera"]
    reference_episode = int(source_config["source_episode_by_object"][args.object_id])

    episode_root = args.data_root / args.object_id / f"episode_{args.episode:04d}"
    reference_root = (
        args.data_root / args.object_id / f"episode_{reference_episode:04d}"
    )
    alignment_path = episode_root / "alignment.json"
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    if not set(cameras).issubset(alignment["cameras"]):
        raise ValueError("frozen camera set is absent from calibration episode")

    intrinsics_path = episode_root / "undistorted_intrinsics.npy"
    extrinsics_path = episode_root / "extrinsics.npy"
    intrinsics_archive = np.load(intrinsics_path, allow_pickle=True).item()
    extrinsics_archive = np.load(extrinsics_path, allow_pickle=True).item()
    intrinsics = {camera: intrinsics_archive[camera] for camera in cameras}
    extrinsics = {camera: extrinsics_archive[camera] for camera in cameras}

    reference_image_path = reference_root / reference_camera / "undistorted_000000.png"
    reference_bgr = cv2.imread(str(reference_image_path), cv2.IMREAD_COLOR)
    if reference_bgr is None:
        raise ValueError("cannot read frozen source reference image")
    reference_rgb = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2RGB)
    reference_mask_path = (
        args.source_qa_root
        / "locked/masks"
        / args.object_id
        / f"{reference_camera}.png"
    )
    reference_mask_u8 = cv2.imread(str(reference_mask_path), cv2.IMREAD_GRAYSCALE)
    if reference_mask_u8 is None:
        raise ValueError("cannot read frozen source reference mask")
    reference_mask = reference_mask_u8 > 0

    predictor = DeformableObjectSam2VideoPredictor(
        args.sam2_repository,
        args.checkpoint,
        device=args.device,
        config=DeformableObjectSam2MaskConfig(**source_config["sam2"]),
    )
    candidates: dict[str, list[dict[str, Any]]] = {}
    candidate_diagnostics: list[dict[str, Any]] = []
    try:
        for camera in cameras:
            records, summary = predictor.initial_mask_candidates_with_reference(
                episode_root / camera / "undistorted.mp4",
                reference_rgb,
                reference_mask,
                reference_camera=reference_camera,
                maximum_candidates=frozen["mask_candidates"][
                    "maximum_candidates_per_camera"
                ],
                include_below_appearance_threshold=frozen["mask_candidates"][
                    "include_basic_candidates_below_appearance_threshold"
                ],
            )
            candidates[camera] = records
            candidate_diagnostics.append(
                {
                    **summary,
                    "camera": camera,
                    "candidates": [record["diagnostic"] for record in records],
                }
            )
    finally:
        predictor.close()

    selection = frozen["joint_selection"]
    selected_masks: dict[str, np.ndarray]
    try:
        selected_masks, joint = select_joint_multiview_masks(
            candidates,
            intrinsics,
            extrinsics,
            CrossViewMaskReliabilityConfig(
                **selection["full_resolution_cross_view_config"]
            ),
            JointMultiviewMaskSelectionConfig(
                maximum_candidates_per_camera=frozen["mask_candidates"][
                    "maximum_candidates_per_camera"
                ],
                voxel_resolution=selection["voxel_resolution"],
                coordinate_descent_passes=selection["coordinate_descent_passes"],
                appearance_weight=selection["appearance_weight"],
                projected_volume_penalty=selection["projected_volume_penalty"],
            ),
        )
        accepted = int(joint["cross_view_consistency"]["accepted_camera_count"])
    except ValueError as error:
        selected_masks = {}
        joint = {
            "passed": False,
            "failure_stage": "joint-calibrated-3d-selection",
            "reason": str(error),
        }
        accepted = 0
    required = int(
        frozen["calibration_gate"]["minimum_accepted_camera_count_per_episode"]
    )
    gates = {"accepted_camera_count": accepted >= required}
    passed = all(gates.values())

    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360ReusableAssociationCalibrationMaskGate",
        "protocol_id": frozen["protocol_id"],
        "config_sha256": request["config_sha256"],
        "object_id": args.object_id,
        "episode_id": args.episode,
        "split": "calibration",
        "allowed_frame_range": request["allowed_frame_range"],
        "frame_indices_read": [0],
        "reference_episode": reference_episode,
        "reference_camera": reference_camera,
        "selected_cameras": list(cameras),
        "candidate_diagnostics": candidate_diagnostics,
        "joint_selection": joint,
        "thresholds": {"minimum_accepted_camera_count": required},
        "gates": gates,
        "passed": passed,
        "input_sha256": {
            "protocol": sha256_file(config_path),
            "source_policy": sha256_file(source_policy_path),
            "source_geometry_qa": sha256_file(source_artifact_path),
            "alignment": sha256_file(alignment_path),
            "intrinsics": sha256_file(intrinsics_path),
            "extrinsics": sha256_file(extrinsics_path),
            "reference_image": sha256_file(reference_image_path),
            "reference_mask": sha256_file(reference_mask_path),
            "sam2_checkpoint": sha256_file(args.checkpoint),
            "calibration_frame_zero_rgb": {
                camera: _first_frame_sha256(episode_root / camera / "undistorted.mp4")
                for camera in cameras
            },
        },
        "information_boundary": {
            "source_reference_read": True,
            "calibration_first_frame_only": True,
            "calibration_future_geometry_read": False,
            "future_prediction_metrics_computed": False,
            "target_media_read": False,
        },
        "claim_boundary": (
            "prefix-only association calibration; no dynamics or target claim"
        ),
    }
    payload["result_sha256"] = _canonical_sha256(payload)

    args.output.mkdir(parents=True, exist_ok=False)
    result_path = args.output / "mask_gate.json"
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        args.output / "initial_masks.npz",
        **{
            camera: np.asarray(mask, dtype=np.uint8)
            for camera, mask in selected_masks.items()
        },
    )
    print(
        json.dumps(
            {
                "episode_id": args.episode,
                "accepted_camera_count": accepted,
                "passed": passed,
                "result_sha256": payload["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
