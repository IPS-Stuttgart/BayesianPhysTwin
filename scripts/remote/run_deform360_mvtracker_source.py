#!/usr/bin/env python3
"""Run the sealed Deform360 MVTracker privileged-depth competence control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import types

import cv2
import h5py
import numpy as np

from bayesian_phystwin.deform360_mvtracker_source import (
    MVTRACKER_CHECKPOINT_SHA256,
    MVTRACKER_REVISION,
    MVTrackerSourceConfig,
    array_sha256,
    evaluate_prediction,
    file_sha256,
    seal_prediction,
    validate_source_contract,
    write_prediction_artifact,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--source-dir", type=Path, required=True)
    predict.add_argument("--output-dir", type=Path, required=True)
    predict.add_argument("--mvtracker-root", type=Path, required=True)
    predict.add_argument("--mvtracker-checkpoint", type=Path, required=True)
    predict.add_argument("--device", default="cuda:0")

    seal = subparsers.add_parser("seal")
    seal.add_argument("--prediction-dir", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--prediction-dir", type=Path, required=True)
    evaluate.add_argument("--source-case-dir", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _decode_rgb_prefix(path: Path, frame_count: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        for frame_index in range(frame_count):
            ok, bgr = capture.read()
            observed = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if not ok or observed != frame_index:
                raise ValueError(f"cannot decode exact RGB frame {frame_index}: {path}")
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return np.stack(frames)


def _read_depth_prefix(path: Path, frame_count: int, scale: float) -> np.ndarray:
    with h5py.File(path, "r") as stream:
        if "data" not in stream or stream["data"].ndim != 3:
            raise ValueError(f"invalid rendered-depth archive: {path}")
        if len(stream["data"]) < frame_count:
            raise ValueError(f"rendered-depth archive is too short: {path}")
        encoded = np.asarray(stream["data"][:frame_count])
    return encoded.astype(np.float32) * scale


def _load_prefix_inputs(
    source_dir: Path,
    frame_zero_m: np.ndarray,
    config: MVTrackerSourceConfig,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    episode = source_dir / "episode_0000"
    intrinsics_path = episode / "undistorted_intrinsics.npy"
    extrinsics_path = episode / "extrinsics.npy"
    intrinsics_dict = np.load(intrinsics_path, allow_pickle=True).item()
    extrinsics_dict = np.load(extrinsics_path, allow_pickle=True).item()
    rgbs: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    world_to_camera: list[np.ndarray] = []
    camera_inputs: dict[str, object] = {}
    for camera in config.selected_cameras:
        camera_dir = episode / camera
        rgb = _decode_rgb_prefix(
            camera_dir / "undistorted.mp4",
            config.prefix_frame_count,
        )
        depth = _read_depth_prefix(
            camera_dir / "rendered_depth.h5",
            config.prefix_frame_count,
            config.depth_scale_to_m,
        )
        if rgb.shape != (*depth.shape, 3):
            raise ValueError(f"RGB/depth shape differs for {camera}")
        k = np.asarray(intrinsics_dict[camera], dtype=np.float32)
        camera_to_world = np.asarray(extrinsics_dict[camera], dtype=np.float32)
        if k.shape != (3, 3) or camera_to_world.shape != (4, 4):
            raise ValueError(f"invalid camera calibration for {camera}")
        w2c = np.linalg.inv(camera_to_world).astype(np.float32)[:3]
        rgbs.append(np.moveaxis(rgb, -1, 1))
        depths.append(depth[:, None])
        intrinsics.append(
            np.repeat(k[None], config.prefix_frame_count, axis=0)
        )
        world_to_camera.append(
            np.repeat(w2c[None], config.prefix_frame_count, axis=0)
        )
        depth_meta = camera_dir / "rendered_depth.meta.json"
        camera_inputs[camera] = {
            "decoded_rgb_prefix_sha256": array_sha256(rgb),
            "rendered_depth_prefix_sha256": array_sha256(depth),
            "rendered_depth_metadata_sha256": file_sha256(depth_meta),
            "maximum_rgb_frame_read": config.update_frame,
            "rendered_depth_indices_read": list(range(config.prefix_frame_count)),
        }
    arrays = {
        "rgbs": np.stack(rgbs).astype(np.float32),
        "depths": np.stack(depths).astype(np.float32),
        "intrinsics": np.stack(intrinsics).astype(np.float32),
        "world_to_camera": np.stack(world_to_camera).astype(np.float32),
        "query_points": np.column_stack(
            (
                np.zeros(len(config.center_ids), dtype=np.float32),
                np.asarray(frame_zero_m, dtype=np.float32)[
                    np.asarray(config.center_ids, dtype=np.int64)
                ],
            )
        ),
    }
    provenance: dict[str, object] = {
        "source_prediction_seal_sha256": file_sha256(
            source_dir / "prediction_seal.json"
        ),
        "source_prediction_archive_sha256": file_sha256(
            source_dir / "prediction.npz"
        ),
        "open27_measurement_manifest_sha256": file_sha256(
            source_dir / "open27_measurement_manifest.json"
        ),
        "intrinsics_sha256": file_sha256(intrinsics_path),
        "extrinsics_sha256": file_sha256(extrinsics_path),
        "cameras": camera_inputs,
        "depth_source": (
            "released full-sequence splat rendered depth; privileged "
            "reconstruction control"
        ),
    }
    return arrays, provenance


def _run_prediction(args: argparse.Namespace) -> dict[str, object]:
    import torch

    source = args.source_dir.resolve()
    mvtracker_root = args.mvtracker_root.resolve()
    config = MVTrackerSourceConfig()
    _, physical, persistence, frame_zero = validate_source_contract(
        source,
        config=config,
    )
    revision = _git_revision(mvtracker_root)
    if revision != MVTRACKER_REVISION:
        raise ValueError("MVTracker repository revision changed")
    checkpoint = args.mvtracker_checkpoint.resolve()
    if file_sha256(checkpoint) != MVTRACKER_CHECKPOINT_SHA256:
        raise ValueError("MVTracker checkpoint checksum changed")
    arrays, input_provenance = _load_prefix_inputs(source, frame_zero, config)
    input_provenance["mvtracker_checkpoint_sha256"] = file_sha256(checkpoint)
    runner_path = Path(__file__).resolve()
    adapter_path = Path(validate_source_contract.__code__.co_filename).resolve()
    protocol_path = (
        runner_path.parents[2]
        / "configs"
        / "sota"
        / "deform360_mvtracker_privileged_depth_competence_v1.json"
    )
    input_provenance["bayesian_phystwin_implementation_sha256"] = {
        "runner": file_sha256(runner_path),
        "adapter": file_sha256(adapter_path),
        "protocol": file_sha256(protocol_path),
    }

    if str(mvtracker_root) not in sys.path:
        sys.path.insert(0, str(mvtracker_root))
    sys.modules.setdefault("rerun", types.ModuleType("rerun"))
    import hubconf
    from mvtracker.datasets.generic_scene_dataset import (
        compute_auto_scene_normalization,
    )
    from mvtracker.datasets.utils import transform_scene

    device = torch.device(args.device)
    torch.manual_seed(72)
    torch.cuda.manual_seed_all(72)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rgbs = torch.from_numpy(arrays["rgbs"])
    depths = torch.from_numpy(arrays["depths"])
    intrinsics = torch.from_numpy(arrays["intrinsics"])
    extrinsics = torch.from_numpy(arrays["world_to_camera"])
    query_points = torch.from_numpy(arrays["query_points"])
    depth_confidence = torch.zeros_like(depths)
    depth_confidence[depths > 0.0] = 1000.0
    scale, translation = compute_auto_scene_normalization(
        depths,
        depth_confidence,
        extrinsics,
        intrinsics,
        target_radius=config.normalization_target_camera_radius,
    )
    rotation = torch.eye(3, dtype=torch.float32)
    depths_normalized, extrinsics_normalized, queries_normalized, _, _ = (
        transform_scene(
            scale,
            rotation,
            translation,
            depths,
            extrinsics,
            query_points,
            None,
            None,
        )
    )
    torch.cuda.reset_peak_memory_stats(device)
    hubconf._WEIGHTS["mvtracker_main"] = str(checkpoint)
    model = hubconf.mvtracker(pretrained=True, device=str(device))
    start = time.perf_counter()
    with torch.no_grad(), torch.amp.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
    ):
        result = model(
            rgbs=rgbs[None].to(device) / 255.0,
            depths=depths_normalized[None].to(device),
            intrs=intrinsics[None].to(device),
            extrs=extrinsics_normalized[None].to(device),
            query_points_3d=queries_normalized[None].to(device),
        )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    normalized = result["traj_e"][0].detach().float().cpu()
    visibility = result["vis_e_as_prob"][0].detach().float().cpu().numpy()
    raw_world = (
        (normalized - translation[None, None]) / float(scale)
    ).numpy()
    runtime_provenance = {
        "device": str(device),
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_gib": (
            torch.cuda.max_memory_allocated(device) / (1024**3)
        ),
        "scene_normalization": {
            "scale": float(scale),
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
            "inputs": "frame-zero rendered depth and calibrated cameras only",
        },
        "checkpoint_sha256_verified": MVTRACKER_CHECKPOINT_SHA256,
        "pointops_fallback": True,
    }
    return write_prediction_artifact(
        args.output_dir,
        raw_tracker_m=raw_world,
        visibility_probability=visibility,
        physical_prior_m=physical,
        persistence_m=persistence,
        frame_zero_points_m=frame_zero,
        input_provenance=input_provenance,
        runtime_provenance=runtime_provenance,
        config=config,
    )


def main() -> int:
    args = _parse_args()
    if args.operation == "predict":
        result = _run_prediction(args)
    elif args.operation == "seal":
        result = seal_prediction(args.prediction_dir)
    else:
        result = evaluate_prediction(
            args.prediction_dir,
            args.source_case_dir,
            args.output,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
