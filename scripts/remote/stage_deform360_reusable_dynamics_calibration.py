#!/usr/bin/env python3
"""Stage one frozen Deform360 dynamics calibration execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from causal4d_public.deform360_dense_source import sha256_file
from causal4d_public.deform360_object_sam2 import (
    DeformableObjectSam2MaskConfig,
    DeformableObjectSam2VideoPredictor,
)
from causal4d_public.deform360_reusable_dynamics import (
    load_reusable_dynamics_config,
    validate_reusable_dynamics_association_evidence,
    validate_reusable_dynamics_calibration_request,
    validate_reusable_dynamics_source_selection,
    validate_reusable_dynamics_source_trust_compatibility,
)
from deform360.robot import RobotState, load_robot_state, save_robot_state


STAGED_EPISODE = 0


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _trim_video(
    source: Path, destination: Path, *, start: int, count: int
) -> None:
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
            f"select='between(n,{start},{start + count - 1})',setpts=N/FRAME_RATE/TB",
            "-frames:v",
            str(count),
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


def _trim_timestamps(
    source: Path, destination: Path, *, start: int, count: int
) -> None:
    lines = source.read_text(encoding="utf-8").splitlines()
    selected = lines[start : start + count]
    if len(selected) != count:
        raise ValueError(f"requested {count} timestamps but found {len(selected)}")
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


def _trim_robot(
    source_episode: Path,
    output_episode: Path,
    *,
    start: int,
    count: int,
) -> Path:
    source = load_robot_state(source_episode / "robot" / "robot.npz")
    stop = start + count
    if stop > source.num_frames:
        raise ValueError("robot state is shorter than the requested frame slice")
    trimmed = RobotState(
        actions=source.actions[start:stop],
        T_worlds=source.T_worlds[start:stop],
        openings=source.openings[start:stop],
        bimanual=source.bimanual,
    )
    return save_robot_state(output_episode / "robot" / "robot.npz", trimmed)


def _trim_tactile_streams(
    source_episode: Path,
    output_episode: Path,
    *,
    start: int,
    count: int,
) -> dict[str, str]:
    stop = start + count
    outputs: dict[str, str] = {}
    for source_dir in sorted(source_episode.glob("*tactile*")):
        source = source_dir / "synced_tactile.npy"
        if not source.exists():
            continue
        values = np.load(source, allow_pickle=False)
        trimmed = values[start:stop]
        if len(trimmed) != count:
            raise ValueError(f"tactile stream {source_dir.name} is too short")
        output_dir = output_episode / source_dir.name
        output_dir.mkdir()
        destination = output_dir / "synced_tactile.npy"
        np.save(destination, trimmed)
        for filename in ("metadata.json", "alignment.json"):
            if (source_dir / filename).exists():
                shutil.copy2(source_dir / filename, output_dir / filename)
        outputs[source_dir.name] = sha256_file(destination)
    if not outputs:
        raise FileNotFoundError(f"no tactile streams found in {source_episode}")
    return outputs


def _validate_mask_gate(
    mask_gate_path: Path,
    mask_archive_path: Path,
    *,
    config_sha256: str,
    object_id: str,
    episode_id: int,
    raw_frame_index: int,
) -> tuple[dict[str, Any], list[str], dict[str, np.ndarray]]:
    gate = json.loads(mask_gate_path.read_text(encoding="utf-8"))
    if gate.get("artifact_kind") != "Deform360ReusableDynamicsInitialMaskGate":
        raise ValueError("unexpected dynamics mask artifact")
    if gate.get("config_sha256") != config_sha256:
        raise ValueError("dynamics mask artifact uses another protocol")
    if gate.get("object_id") != object_id or int(gate.get("episode_id", -1)) != episode_id:
        raise ValueError("dynamics mask artifact identifies another execution")
    if gate.get("raw_frame_indices_read") != [raw_frame_index]:
        raise ValueError("dynamics mask artifact read another initial frame")
    if gate.get("passed") is not True:
        raise ValueError("dynamics mask gate did not pass")
    canonical = dict(gate)
    result_sha256 = canonical.pop("result_sha256", None)
    if result_sha256 != _canonical_sha256(canonical):
        raise ValueError("dynamics mask artifact checksum mismatch")
    boundary = gate.get("information_boundary", {})
    if boundary.get("calibration_future_geometry_read") is not False:
        raise ValueError("dynamics mask artifact read calibration futures")
    if boundary.get("target_media_read") is not False:
        raise ValueError("dynamics mask artifact read target media")
    cameras = list(
        gate["joint_selection"]["cross_view_consistency"]["accepted_cameras"]
    )
    with np.load(mask_archive_path, allow_pickle=False) as archive:
        missing = sorted(set(cameras) - set(archive.files))
        if missing:
            raise ValueError(f"initial-mask archive lacks accepted cameras: {missing}")
        masks = {camera: np.asarray(archive[camera], dtype=bool) for camera in cameras}
    return gate, cameras, masks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mask-gate-root", type=Path, required=True)
    parser.add_argument("--sam2-repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    protocol_path = (
        args.repo / "configs/causal4d_public/deform360_reusable_dynamics_081_v1.json"
    )
    mask_summary_path = (
        args.repo
        / "milestones/deform360-reusable-association-v2-calibration-mask"
        / "calibration_mask_summary.json"
    )
    prefix_summary_path = (
        args.repo
        / "milestones/deform360-reusable-association-v2-calibration-prefix"
        / "calibration_prefix_summary.json"
    )
    milestone_root = args.repo / "milestones/deform360-reusable-dynamics-081-v1"
    source_selection_path = milestone_root / "artifacts/source_selection.json"
    source_trust_path = (
        milestone_root / "artifacts/source_trust_compatibility.json"
    )
    protocol = load_reusable_dynamics_config(protocol_path)
    evidence = validate_reusable_dynamics_association_evidence(
        protocol,
        mask_summary_path=mask_summary_path,
        prefix_summary_path=prefix_summary_path,
    )
    source_selection = json.loads(
        source_selection_path.read_text(encoding="utf-8")
    )
    source_selection_evidence = validate_reusable_dynamics_source_selection(
        source_selection, config=protocol
    )
    source_trust = json.loads(source_trust_path.read_text(encoding="utf-8"))
    source_trust_evidence = (
        validate_reusable_dynamics_source_trust_compatibility(
            source_trust,
            config=protocol,
            source_selection=source_selection,
        )
    )
    request = validate_reusable_dynamics_calibration_request(
        protocol,
        object_id=args.object_id,
        episode_id=args.episode,
        operation="staging",
    )
    start, stop = (int(value) for value in request["allowed_frame_range"])
    count = stop - start

    mask_gate_path = args.mask_gate_root / "mask_gate.json"
    mask_archive_path = args.mask_gate_root / "initial_masks.npz"
    mask_gate, cameras, initial_masks = _validate_mask_gate(
        mask_gate_path,
        mask_archive_path,
        config_sha256=request["config_sha256"],
        object_id=args.object_id,
        episode_id=args.episode,
        raw_frame_index=start,
    )

    source_config_path = (
        args.repo / "configs/causal4d_public/deform360_replication_source_qa_v1.json"
    )
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))[
        "config"
    ]
    source_episode = args.data_root / args.object_id / f"episode_{args.episode:04d}"
    if args.output.exists():
        raise FileExistsError(f"dynamics staging output already exists: {args.output}")
    staged_episode = args.output / "staged" / f"episode_{STAGED_EPISODE:04d}"
    staged_episode.mkdir(parents=True)

    intrinsics_path = staged_episode / "undistorted_intrinsics.npy"
    extrinsics_path = staged_episode / "extrinsics.npy"
    _subset_calibration(
        source_episode / "undistorted_intrinsics.npy", intrinsics_path, cameras
    )
    _subset_calibration(source_episode / "extrinsics.npy", extrinsics_path, cameras)
    robot_path = _trim_robot(
        source_episode, staged_episode, start=start, count=count
    )
    tactile_hashes = _trim_tactile_streams(
        source_episode, staged_episode, start=start, count=count
    )

    predictor = DeformableObjectSam2VideoPredictor(
        args.sam2_repository,
        args.checkpoint,
        device=args.device,
        config=DeformableObjectSam2MaskConfig(**source_config["sam2"]),
    )
    sam2_diagnostics: dict[str, Any] = {}
    output_hashes: dict[str, dict[str, str]] = {}
    try:
        for camera in cameras:
            source_camera = source_episode / camera
            output_camera = staged_episode / camera
            output_camera.mkdir()
            video_path = output_camera / "undistorted.mp4"
            timestamp_path = output_camera / "aligned_timestamps.txt"
            mask_path = output_camera / "mask_refined.h5"
            _trim_video(
                source_camera / "undistorted.mp4",
                video_path,
                start=start,
                count=count,
            )
            _trim_timestamps(
                source_camera / "aligned_timestamps.txt",
                timestamp_path,
                start=start,
                count=count,
            )
            metadata_path = source_camera / "metadata.json"
            if metadata_path.exists():
                shutil.copy2(metadata_path, output_camera / "metadata.json")
            masks = list(
                predictor.segment_from_initial_mask(
                    video_path,
                    initial_masks[camera],
                    initialization={
                        "mask_gate_result_sha256": mask_gate["result_sha256"],
                        "raw_frame_index": start,
                    },
                )
            )
            if [index for index, _ in masks] != list(range(count)):
                raise ValueError(f"SAM2 returned incomplete frames for {camera}")
            _write_masks(mask_path, [mask for _, mask in masks])
            sam2_diagnostics[camera] = predictor.diagnostics[-1]
            output_hashes[camera] = {
                "video": sha256_file(video_path),
                "timestamps": sha256_file(timestamp_path),
                "masks": sha256_file(mask_path),
            }
    finally:
        predictor.close()

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360ReusableDynamicsCalibrationStaging",
        "protocol_id": protocol["config"]["protocol_id"],
        "config_sha256": request["config_sha256"],
        "association_evidence": evidence,
        "source_selection_evidence": source_selection_evidence,
        "source_trust_evidence": source_trust_evidence,
        "object_id": args.object_id,
        "episode_id": args.episode,
        "staged_episode_id": STAGED_EPISODE,
        "raw_frame_range": [start, stop],
        "frame_count": count,
        "accepted_cameras": cameras,
        "mask_gate_result_sha256": mask_gate["result_sha256"],
        "input_sha256": {
            "protocol": sha256_file(protocol_path),
            "source_config": sha256_file(source_config_path),
            "source_selection": sha256_file(source_selection_path),
            "source_trust_compatibility": sha256_file(source_trust_path),
            "mask_gate": sha256_file(mask_gate_path),
            "initial_masks": sha256_file(mask_archive_path),
            "sam2_checkpoint": sha256_file(args.checkpoint),
        },
        "output_sha256": {
            "intrinsics": sha256_file(intrinsics_path),
            "extrinsics": sha256_file(extrinsics_path),
            "robot": sha256_file(robot_path),
            "tactile": tactile_hashes,
            "cameras": output_hashes,
        },
        "sam2_diagnostics": sam2_diagnostics,
        "information_boundary": {
            "calibration_raw_frame_range_read": [start, stop],
            "future_prediction_metrics_computed": False,
            "method_or_hyperparameter_changes_allowed": False,
            "target_media_read": False,
        },
        "claim_boundary": "staging only; no dynamics score or target claim",
    }
    manifest["result_sha256"] = _canonical_sha256(manifest)
    manifest_path = staged_episode / "reusable_dynamics_staging.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": True,
                "episode_id": args.episode,
                "episode_dir": str(staged_episode),
                "camera_count": len(cameras),
                "frame_count": count,
                "result_sha256": manifest["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
