#!/usr/bin/env python3
"""Inventory source-only Deform360 mask carriers without reading mask frames.

The audit lists the registered source episode, intersects aligned camera names
with the calibration dictionaries, and opens only HDF5 container/dataset
headers for ``mask_refined.h5`` and ``rendered_urdf.h5``. It never indexes a
mask dataset, decodes an RGB frame, opens a target, or writes into a dataset
tree. Its sole purpose is to decide whether the held-out-camera visual-hull
experiment has enough structurally compatible object-mask carriers to run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_camera_keys(path: Path) -> list[str]:
    """Load trusted first-party scalar NumPy dictionaries and return keys."""
    loaded = np.load(path, allow_pickle=True)
    try:
        value = loaded.item()
    finally:
        if hasattr(loaded, "close"):
            loaded.close()
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a camera dictionary in {path}")
    keys = []
    for key in value:
        if not isinstance(key, str) or not key:
            raise ValueError(f"invalid camera key in {path}")
        keys.append(key)
    return sorted(set(keys))


def _json_metadata(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"metadata at {path} is not a JSON object")
    return {
        "path": str(path),
        "file_size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
        "schema": value.get("schema"),
        "output_mask_sha256": value.get("outputs", {}).get("mask_sha256")
        if isinstance(value.get("outputs"), dict)
        else None,
        "frames_with_mask": value.get("outputs", {}).get("frames_with_mask")
        if isinstance(value.get("outputs"), dict)
        else None,
    }


def _carrier_header(path: Path, metadata_path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "header_valid": False,
        "dataset_payload_frames_opened": 0,
        "metadata": None,
    }
    if not path.is_file():
        return record
    record["file_size_bytes"] = int(path.stat().st_size)
    try:
        with h5py.File(path, "r") as handle:
            if "data" not in handle:
                raise KeyError("HDF5 container has no dataset named 'data'")
            data = handle["data"]
            shape = tuple(int(value) for value in data.shape)
            if data.ndim != 3:
                raise ValueError(f"dataset shape {shape} is not (T,H,W)")
            dtype = np.dtype(data.dtype)
            if dtype not in (np.dtype(np.bool_), np.dtype(np.uint8)):
                raise TypeError(f"dataset dtype {dtype} is not bool/uint8")
            record.update(
                {
                    "header_valid": True,
                    "shape": list(shape),
                    "frame_count": shape[0],
                    "height": shape[1],
                    "width": shape[2],
                    "dtype": str(dtype),
                    "chunks": list(data.chunks) if data.chunks is not None else None,
                    "compression": data.compression,
                    "compression_opts": data.compression_opts,
                    "shuffle": bool(data.shuffle),
                    "fletcher32": bool(data.fletcher32),
                }
            )
    except Exception as exc:  # preserve the structural failure in evidence
        record["error"] = f"{type(exc).__name__}: {exc}"
    try:
        record["metadata"] = _json_metadata(metadata_path)
    except Exception as exc:
        record["metadata_error"] = f"{type(exc).__name__}: {exc}"
    return record


def _shape_key(record: Mapping[str, Any]) -> str | None:
    shape = record.get("shape")
    dtype = record.get("dtype")
    if not record.get("header_valid") or not isinstance(shape, list):
        return None
    return f"{shape[0]}x{shape[1]}x{shape[2]}:{dtype}"


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-episode-root", required=True, type=Path)
    parser.add_argument("--source-object", required=True)
    parser.add_argument("--source-episode", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--minimum-object-mask-cameras", type=int, default=16)
    parser.add_argument("--minimum-heldout-cameras", type=int, default=6)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--workflow-run-id", required=True, type=int)
    parser.add_argument("--workflow-run-attempt", required=True, type=int)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.source_episode_root.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    observed = root.parent.name, int(root.name.split("_")[-1])
    expected = args.source_object, int(args.source_episode)
    if observed != expected:
        raise ValueError(f"source episode {observed} differs from {expected}")

    intrinsics_path = root / "undistorted_intrinsics.npy"
    extrinsics_path = root / "extrinsics.npy"
    intrinsic_cameras = _load_camera_keys(intrinsics_path)
    extrinsic_cameras = _load_camera_keys(extrinsics_path)
    calibrated_cameras = sorted(set(intrinsic_cameras) & set(extrinsic_cameras))
    aligned_camera_dirs = sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / "undistorted.mp4").is_file()
    )
    candidate_cameras = sorted(set(calibrated_cameras) & set(aligned_camera_dirs))

    camera_records = []
    object_shape_counts: Counter[str] = Counter()
    gripper_shape_counts: Counter[str] = Counter()
    for camera in candidate_cameras:
        camera_root = root / camera
        object_record = _carrier_header(
            camera_root / "mask_refined.h5",
            camera_root / "mask_refined.meta.json",
        )
        gripper_record = _carrier_header(
            camera_root / "rendered_urdf.h5",
            camera_root / "rendered_urdf.meta.json",
        )
        object_key = _shape_key(object_record)
        gripper_key = _shape_key(gripper_record)
        if object_key is not None:
            object_shape_counts[object_key] += 1
        if gripper_key is not None:
            gripper_shape_counts[gripper_key] += 1
        camera_records.append(
            {
                "camera": camera,
                "object_mask": object_record,
                "gripper_mask": gripper_record,
            }
        )

    object_present_count = sum(
        int(record["object_mask"]["exists"]) for record in camera_records
    )
    object_valid_count = sum(
        int(record["object_mask"]["header_valid"]) for record in camera_records
    )
    gripper_present_count = sum(
        int(record["gripper_mask"]["exists"]) for record in camera_records
    )
    gripper_valid_count = sum(
        int(record["gripper_mask"]["header_valid"]) for record in camera_records
    )
    dominant_object_shape = (
        object_shape_counts.most_common(1)[0] if object_shape_counts else (None, 0)
    )
    dominant_gripper_shape = (
        gripper_shape_counts.most_common(1)[0] if gripper_shape_counts else (None, 0)
    )
    minimum_object = int(args.minimum_object_mask_cameras)
    minimum_heldout = int(args.minimum_heldout_cameras)
    checks = {
        "minimum_aligned_calibrated_camera_count": len(candidate_cameras)
        >= minimum_object + minimum_heldout,
        "minimum_valid_object_mask_camera_count": object_valid_count
        >= minimum_object,
        "minimum_common_object_mask_shape_count": int(dominant_object_shape[1])
        >= minimum_object,
        "enough_remaining_cameras_for_heldout_split": len(candidate_cameras)
        - minimum_object
        >= minimum_heldout,
        "no_invalid_present_object_mask_headers": object_present_count
        == object_valid_count,
    }
    ready = all(checks.values())

    result: dict[str, Any] = {
        "schema": "bayesian-phystwin/deform360-source-mask-carrier-inventory-v1",
        "repository": args.repository,
        "revision": args.revision,
        "workflow_run_id": args.workflow_run_id,
        "workflow_run_attempt": args.workflow_run_attempt,
        "runner_name": os.environ.get("RUNNER_NAME"),
        "required_runner_label": "gpuserver4090",
        "source_object": args.source_object,
        "source_episode": args.source_episode,
        "source_episode_root": str(root),
        "calibration": {
            "intrinsics_path": str(intrinsics_path),
            "extrinsics_path": str(extrinsics_path),
            "intrinsics_file_sha256": _sha256(intrinsics_path),
            "extrinsics_file_sha256": _sha256(extrinsics_path),
            "intrinsic_camera_count": len(intrinsic_cameras),
            "extrinsic_camera_count": len(extrinsic_cameras),
            "calibrated_camera_count": len(calibrated_cameras),
        },
        "aligned_camera_count": len(aligned_camera_dirs),
        "aligned_calibrated_camera_count": len(candidate_cameras),
        "aligned_camera_names": aligned_camera_dirs,
        "aligned_calibrated_camera_names": candidate_cameras,
        "object_mask_inventory": {
            "present_count": object_present_count,
            "valid_header_count": object_valid_count,
            "shape_dtype_counts": _counter_dict(object_shape_counts),
            "dominant_shape_dtype": dominant_object_shape[0],
            "dominant_shape_dtype_count": int(dominant_object_shape[1]),
        },
        "gripper_mask_inventory": {
            "present_count": gripper_present_count,
            "valid_header_count": gripper_valid_count,
            "shape_dtype_counts": _counter_dict(gripper_shape_counts),
            "dominant_shape_dtype": dominant_gripper_shape[0],
            "dominant_shape_dtype_count": int(dominant_gripper_shape[1]),
        },
        "camera_records": camera_records,
        "readiness_checks": checks,
        "source_visual_hull_inputs_ready": ready,
        "decision": (
            "source-visual-hull-inputs-ready"
            if ready
            else "source-visual-hull-inputs-not-ready"
        ),
        "information_boundary": {
            "source_episode_directory_listed": True,
            "source_calibration_dictionaries_opened": True,
            "source_mask_container_headers_opened": True,
            "source_mask_dataset_payload_frames_opened": 0,
            "source_camera_pixels_opened": False,
            "persistent_dataset_write_performed": False,
            "target_directory_contents_listed": False,
            "target_numeric_payload_opened": False,
            "target_scoring_performed": False,
            "fresh_confirmation_authorized": False,
            "paper_claim_authorized": False,
        },
        "claim_boundary": (
            "Header-only source carrier inventory. It decides whether the "
            "registered source has enough structurally compatible object masks "
            "for held-out-camera visual-hull scoring. It does not establish mask "
            "content quality, geometry quality, transport, target benefit, "
            "generalization, safety, or a paper claim."
        ),
    }
    canonical = json.dumps(
        result, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    result["result_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    _write_json(output / "result.json", result)
    (output / "report.md").write_text(
        "# Deform360 source mask-carrier inventory v1\n\n"
        f"Decision: `{result['decision']}`\n\n"
        f"Aligned calibrated cameras: `{len(candidate_cameras)}`\n\n"
        f"Object carriers: `{object_valid_count}` valid / "
        f"`{object_present_count}` present\n\n"
        f"Dominant object shape/dtype: `{dominant_object_shape[0]}` on "
        f"`{dominant_object_shape[1]}` cameras\n\n"
        f"Gripper carriers: `{gripper_valid_count}` valid / "
        f"`{gripper_present_count}` present\n\n"
        "Only HDF5 headers were opened; no mask frame or RGB pixel was read.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
