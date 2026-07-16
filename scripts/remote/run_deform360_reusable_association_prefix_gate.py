#!/usr/bin/env python3
"""Run the frozen six-frame Gaussian-identity gate on one calibration episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from plyfile import PlyData

from causal4d_public.deform360_dense_source import sha256_file
from causal4d_public.deform360_gaussian_identity import (
    GaussianIdentityConfig,
    match_gaussian_identities,
)
from causal4d_public.deform360_object_sam2 import (
    DeformableObjectSam2MaskConfig,
    DeformableObjectSam2VideoPredictor,
)
from causal4d_public.deform360_reusable_association import (
    load_reusable_association_config,
    load_reusable_association_source_evidence,
    validate_reusable_association_calibration_request,
)
from deform360.processing import reconstruct_stage


FRAME_COUNT = 6
STAGED_EPISODE = 0
MINIMUM_HULL_POINTS = 512
FIRST_FRAME_ITERATIONS = 500
WARM_START_ITERATIONS = 250


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_mask_gate(payload: dict[str, Any], request: dict[str, Any]) -> None:
    if payload.get("artifact_kind") != (
        "Deform360ReusableAssociationCalibrationMaskGate"
    ):
        raise ValueError("unexpected calibration mask artifact")
    claimed = payload.get("result_sha256")
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    if claimed != _canonical_sha256(canonical):
        raise ValueError("calibration mask artifact checksum mismatch")
    if payload.get("config_sha256") != request["config_sha256"]:
        raise ValueError("calibration mask artifact uses another protocol")
    if payload.get("object_id") != request["object_id"]:
        raise ValueError("calibration mask artifact uses another object")
    if int(payload.get("episode_id", -1)) != request["episode_id"]:
        raise ValueError("calibration mask artifact uses another episode")
    if payload.get("frame_indices_read") != [0] or payload.get("passed") is not True:
        raise ValueError("calibration mask gate did not pass at frame zero")
    boundary = payload.get("information_boundary", {})
    if boundary.get("calibration_future_geometry_read") is not False:
        raise ValueError("calibration mask gate crossed its information boundary")
    if boundary.get("target_media_read") is not False:
        raise ValueError("calibration mask gate read target media")


def _trim_video(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            f"select='between(n,0,{FRAME_COUNT - 1})',setpts=N/FRAME_RATE/TB",
            "-frames:v",
            str(FRAME_COUNT),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "12",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            str(destination),
        ],
        check=True,
    )


def _trim_timestamps(source: Path, destination: Path) -> None:
    lines = source.read_text(encoding="utf-8").splitlines()
    selected = lines[:FRAME_COUNT]
    if len(selected) != FRAME_COUNT:
        raise ValueError(
            f"requested {FRAME_COUNT} timestamps but found {len(selected)}"
        )
    destination.write_text("\n".join(selected) + "\n", encoding="utf-8")


def _subset_calibration(source: Path, destination: Path, cameras: list[str]) -> None:
    payload = np.load(source, allow_pickle=True).item()
    missing = sorted(set(cameras) - set(payload))
    if missing:
        raise ValueError(f"calibration {source.name} lacks cameras {missing}")
    np.save(destination, {camera: payload[camera] for camera in cameras})


def _write_masks(destination: Path, masks: list[np.ndarray]) -> None:
    values = np.asarray(masks, dtype=np.uint8)
    with h5py.File(destination, "w") as stream:
        stream.create_dataset(
            "data",
            data=values,
            dtype=np.uint8,
            compression="gzip",
            compression_opts=4,
        )


def _read_gaussian_positions(path: Path) -> np.ndarray:
    vertices = PlyData.read(str(path))["vertex"].data
    names = set(vertices.dtype.names or ())
    if not {"x", "y", "z"} <= names:
        raise ValueError("splat PLY lacks Gaussian centers")
    positions = np.column_stack((vertices["x"], vertices["y"], vertices["z"]))
    if not np.isfinite(positions).all():
        raise ValueError("splat PLY contains non-finite centers")
    return np.asarray(positions, dtype=np.float64)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mask-gate-dir", type=Path, required=True)
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
    gate = frozen["calibration_gate"]

    mask_gate_path = args.mask_gate_dir / "mask_gate.json"
    mask_archive_path = args.mask_gate_dir / "initial_masks.npz"
    mask_gate = json.loads(mask_gate_path.read_text(encoding="utf-8"))
    _validate_mask_gate(mask_gate, request)
    mask_summary = json.loads(
        (
            args.repo
            / "milestones/deform360-reusable-association-v2-calibration-mask/calibration_mask_summary.json"
        ).read_text(encoding="utf-8")
    )
    mask_summary_record = next(
        record
        for record in mask_summary["episodes"]
        if int(record["episode_id"]) == args.episode
    )
    if sha256_file(mask_gate_path) != mask_summary_record["result_file_sha256"]:
        raise ValueError("calibration mask gate file differs from sealed summary")
    if sha256_file(mask_archive_path) != mask_summary_record["mask_archive_sha256"]:
        raise ValueError("calibration mask archive differs from sealed summary")
    accepted_cameras = list(
        mask_gate["joint_selection"]["cross_view_consistency"]["accepted_cameras"]
    )
    if len(accepted_cameras) < int(gate["minimum_accepted_camera_count_per_episode"]):
        raise ValueError("sealed mask artifact has too few accepted cameras")
    with np.load(mask_archive_path, allow_pickle=False) as archive:
        initial_masks = {
            camera: np.asarray(archive[camera], dtype=bool)
            for camera in accepted_cameras
        }

    source_config_path = (
        args.repo / "configs/causal4d_public/deform360_replication_source_qa_v1.json"
    )
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))["config"]
    source_episode = args.data_root / args.object_id / f"episode_{args.episode:04d}"
    if args.output.exists():
        raise FileExistsError(f"prefix-gate output already exists: {args.output}")
    staged_episode = args.output / "staged" / f"episode_{STAGED_EPISODE:04d}"
    staged_episode.mkdir(parents=True)

    _subset_calibration(
        source_episode / "undistorted_intrinsics.npy",
        staged_episode / "undistorted_intrinsics.npy",
        accepted_cameras,
    )
    _subset_calibration(
        source_episode / "extrinsics.npy",
        staged_episode / "extrinsics.npy",
        accepted_cameras,
    )

    predictor = DeformableObjectSam2VideoPredictor(
        args.sam2_repository,
        args.checkpoint,
        device=args.device,
        config=DeformableObjectSam2MaskConfig(**source_config["sam2"]),
    )
    sam2_diagnostics: dict[str, Any] = {}
    try:
        for camera in accepted_cameras:
            camera_dir = staged_episode / camera
            camera_dir.mkdir()
            video_path = camera_dir / "undistorted.mp4"
            _trim_video(source_episode / camera / "undistorted.mp4", video_path)
            _trim_timestamps(
                source_episode / camera / "aligned_timestamps.txt",
                camera_dir / "aligned_timestamps.txt",
            )
            masks = list(
                predictor.segment_from_initial_mask(
                    video_path,
                    initial_masks[camera],
                    initialization={
                        "mask_gate_result_sha256": mask_gate["result_sha256"],
                        "raw_frame_index": 0,
                    },
                )
            )
            if [index for index, _ in masks] != list(range(FRAME_COUNT)):
                raise ValueError(f"SAM2 returned incomplete prefix for {camera}")
            _write_masks(
                camera_dir / "mask_refined.h5",
                [mask for _, mask in masks],
            )
            sam2_diagnostics[camera] = predictor.diagnostics[-1]
    finally:
        predictor.close()

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360ReusableAssociationCalibrationPrefix",
        "config_sha256": request["config_sha256"],
        "object_id": args.object_id,
        "episode_id": args.episode,
        "raw_frame_range": [0, FRAME_COUNT],
        "accepted_cameras": accepted_cameras,
        "mask_gate_result_sha256": mask_gate["result_sha256"],
        "mask_gate_file_sha256": sha256_file(mask_gate_path),
        "initial_masks_sha256": sha256_file(mask_archive_path),
        "sam2_checkpoint_sha256": sha256_file(args.checkpoint),
        "sam2_diagnostics": sam2_diagnostics,
        "information_boundary": {
            "calibration_prefix_only": True,
            "future_prediction_metrics_computed": False,
            "target_media_read": False,
        },
    }
    manifest["result_sha256"] = _canonical_sha256(manifest)
    manifest_path = staged_episode / "reusable_association_prefix.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    original_visual_hull = reconstruct_stage.visual_hull_points

    def strict_visual_hull_points(*call_args: Any, **call_kwargs: Any):
        call_kwargs["min_points"] = MINIMUM_HULL_POINTS
        return original_visual_hull(*call_args, **call_kwargs)

    reconstruct_stage.visual_hull_points = strict_visual_hull_points
    try:
        outputs = reconstruct_stage.process_reconstruction_episode(
            args.output / "staged",
            STAGED_EPISODE,
            cameras=accepted_cameras,
            first_frame_iterations=FIRST_FRAME_ITERATIONS,
            warm_start_iterations=WARM_START_ITERATIONS,
            voxel_resolution=frozen["joint_selection"][
                "full_resolution_cross_view_config"
            ]["voxel_resolution"],
            overwrite=True,
        )
    finally:
        reconstruct_stage.visual_hull_points = original_visual_hull

    identity_config = GaussianIdentityConfig(
        **{
            key: value
            for key, value in frozen["temporal_identity"].items()
            if key
            in {
                "maximum_distance_m",
                "stable_order_distance_m",
                "ambiguity_margin_m",
                "ambiguity_ratio",
                "maximum_neighbors",
            }
        }
    )
    positions = {
        frame: _read_gaussian_positions(path) for frame, path in sorted(outputs.items())
    }
    transitions: list[dict[str, Any]] = []
    for frame in range(FRAME_COUNT - 1):
        result = match_gaussian_identities(
            positions[frame], positions[frame + 1], identity_config
        )
        effective_fraction = float(
            result.diagnostics["effective_reliable_match_count"] / len(positions[frame])
        )
        transitions.append(
            {
                "from_frame": frame,
                "to_frame": frame + 1,
                **result.diagnostics,
                "effective_reliable_match_fraction": effective_fraction,
                "assignment_variance_m2": {
                    "median": float(np.median(result.assignment_variance_m2)),
                    "maximum": float(np.max(result.assignment_variance_m2)),
                },
            }
        )

    source_evidence = load_reusable_association_source_evidence(
        args.repo
        / "milestones/deform360-reusable-association-v2-source/artifacts/source_evidence.json"
    )
    source_maximum = int(
        source_evidence["temporal_identity_source_audit"]["maximum_source_splat_count"]
    )
    maximum_splat_count = max(len(value) for value in positions.values())
    thresholds = {
        "minimum_temporal_match_fraction_per_step": float(
            gate["minimum_temporal_match_fraction_per_step"]
        ),
        "minimum_effective_reliable_match_fraction_per_step": float(
            gate["minimum_effective_reliable_match_fraction_per_step"]
        ),
        "maximum_splat_count": int(
            np.floor(
                source_maximum
                * float(gate["maximum_splat_count_ratio_to_source_maximum"])
            )
        ),
    }
    gates = {
        "splat_cardinality": maximum_splat_count <= thresholds["maximum_splat_count"],
        "temporal_match_fraction": all(
            row["match_fraction"]
            >= thresholds["minimum_temporal_match_fraction_per_step"]
            for row in transitions
        ),
        "effective_reliable_match_fraction": all(
            row["effective_reliable_match_fraction"]
            >= thresholds["minimum_effective_reliable_match_fraction_per_step"]
            for row in transitions
        ),
    }
    passed = all(gates.values())
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360ReusableAssociationCalibrationPrefixGate",
        "protocol_id": frozen["protocol_id"],
        "config_sha256": request["config_sha256"],
        "object_id": args.object_id,
        "episode_id": args.episode,
        "raw_frame_range": [0, FRAME_COUNT],
        "accepted_cameras": accepted_cameras,
        "staging_manifest_sha256": sha256_file(manifest_path),
        "reconstruction": {
            "minimum_hull_points": MINIMUM_HULL_POINTS,
            "first_frame_iterations": FIRST_FRAME_ITERATIONS,
            "warm_start_iterations": WARM_START_ITERATIONS,
            "splat_counts": {
                str(frame): len(value) for frame, value in positions.items()
            },
            "splat_sha256": {
                str(frame): sha256_file(path) for frame, path in sorted(outputs.items())
            },
        },
        "transitions": transitions,
        "thresholds": thresholds,
        "gates": gates,
        "passed": passed,
        "information_boundary": {
            "calibration_prefix_frame_range": [0, FRAME_COUNT],
            "calibration_future_geometry_read": False,
            "future_prediction_metrics_computed": False,
            "target_media_read": False,
        },
        "claim_boundary": (
            "prefix-only material-identity calibration; no future dynamics claim"
        ),
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    result_path = args.output / "prefix_gate.json"
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "episode_id": args.episode,
                "splat_counts": payload["reconstruction"]["splat_counts"],
                "minimum_match_fraction": min(
                    row["match_fraction"] for row in transitions
                ),
                "minimum_effective_reliable_match_fraction": min(
                    row["effective_reliable_match_fraction"] for row in transitions
                ),
                "gates": gates,
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
