#!/usr/bin/env python3
"""Run the frozen PhysTwin MVTracker prefix-only competence control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time
import types

import numpy as np

from bayesian_phystwin.phystwin_mvtracker_competence import (
    MVTRACKER_CHECKPOINT_SHA256,
    MVTRACKER_REVISION,
    PhysTwinMVTrackerCompetenceConfig,
    array_sha256,
    evaluate_competence,
    file_sha256,
    prepare_source_artifacts,
    seal_prediction,
    validate_query_input,
    write_prediction_artifact,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)

    prepare = subparsers.add_parser("prepare-source")
    prepare.add_argument("--manual-tracks", type=Path, required=True)
    prepare.add_argument("--split", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--raw-case-dir", type=Path, required=True)
    predict.add_argument("--query-input", type=Path, required=True)
    predict.add_argument("--query-sha256", required=True)
    predict.add_argument("--output-dir", type=Path, required=True)
    predict.add_argument("--mvtracker-root", type=Path, required=True)
    predict.add_argument("--mvtracker-checkpoint", type=Path, required=True)
    predict.add_argument("--protocol", type=Path, required=True)
    predict.add_argument("--device", default="cuda:0")

    seal = subparsers.add_parser("seal")
    seal.add_argument("--prediction-dir", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--prediction-dir", type=Path, required=True)
    evaluate.add_argument("--withheld-prefix", type=Path, required=True)
    evaluate.add_argument("--withheld-sha256", required=True)
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


def _load_rgb_frame(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - integration runtime
        raise RuntimeError("Pillow is required for PhysTwin RGB input") from error
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _load_prefix_inputs(
    raw_case_dir: Path,
    query_points_world_m: np.ndarray,
    config: PhysTwinMVTrackerCompetenceConfig,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    metadata_path = raw_case_dir / "metadata.json"
    calibration_path = raw_case_dir / "calibrate.pkl"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    intrinsics_all = np.asarray(metadata["intrinsics"], dtype=np.float32)
    with calibration_path.open("rb") as stream:
        camera_to_world_all = np.asarray(pickle.load(stream), dtype=np.float32)
    camera_count = len(config.selected_cameras)
    if intrinsics_all.ndim != 3 or intrinsics_all.shape[1:] != (3, 3):
        raise ValueError("PhysTwin intrinsics must have shape (C, 3, 3)")
    if (
        camera_to_world_all.ndim != 3
        or camera_to_world_all.shape[1:] != (4, 4)
    ):
        raise ValueError("PhysTwin calibration must have shape (C, 4, 4)")
    if max(config.selected_cameras) >= len(intrinsics_all):
        raise ValueError("selected camera exceeds PhysTwin calibration")

    rgbs: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    world_to_camera: list[np.ndarray] = []
    camera_inputs: dict[str, object] = {}
    for camera in config.selected_cameras:
        rgb_frames = []
        depth_frames = []
        frame_hashes = []
        for frame in range(
            config.source_frame_start,
            config.source_frame_end_exclusive,
        ):
            rgb_path = raw_case_dir / "color" / str(camera) / f"{frame}.png"
            depth_path = raw_case_dir / "depth" / str(camera) / f"{frame}.npy"
            rgb_frames.append(_load_rgb_frame(rgb_path))
            depth = np.asarray(np.load(depth_path), dtype=np.float32)
            depth_frames.append(depth * config.depth_scale_to_m)
            frame_hashes.append(
                {
                    "frame": frame,
                    "rgb_sha256": file_sha256(rgb_path),
                    "depth_sha256": file_sha256(depth_path),
                }
            )
        rgb = np.stack(rgb_frames)
        depth = np.stack(depth_frames)
        if rgb.shape[:3] != depth.shape or rgb.shape[3] != 3:
            raise ValueError(f"RGB/depth shape differs for camera {camera}")
        intrinsics_camera = intrinsics_all[camera]
        world_to_camera_camera = np.linalg.inv(
            camera_to_world_all[camera]
        ).astype(np.float32)[:3]
        rgbs.append(np.moveaxis(rgb, -1, 1))
        depths.append(depth[:, None])
        intrinsics.append(
            np.repeat(
                intrinsics_camera[None],
                config.prefix_frame_count,
                axis=0,
            )
        )
        world_to_camera.append(
            np.repeat(
                world_to_camera_camera[None],
                config.prefix_frame_count,
                axis=0,
            )
        )
        camera_inputs[str(camera)] = {
            "decoded_rgb_prefix_sha256": array_sha256(rgb),
            "sensor_depth_prefix_sha256": array_sha256(depth),
            "frame_file_hashes": frame_hashes,
        }
    arrays = {
        "rgbs": np.stack(rgbs).astype(np.float32),
        "depths": np.stack(depths).astype(np.float32),
        "intrinsics": np.stack(intrinsics).astype(np.float32),
        "world_to_camera": np.stack(world_to_camera).astype(np.float32),
        "query_points": np.column_stack(
            (
                np.zeros(len(query_points_world_m), dtype=np.float32),
                np.asarray(query_points_world_m, dtype=np.float32),
            )
        ),
    }
    provenance: dict[str, object] = {
        "raw_case_dir": str(raw_case_dir),
        "metadata_sha256": file_sha256(metadata_path),
        "calibration_sha256": file_sha256(calibration_path),
        "selected_camera_count": camera_count,
        "cameras": camera_inputs,
        "depth_source": "released causal RGB-D sensor depth",
    }
    return arrays, provenance


def _load_locked_protocol(
    protocol_path: Path,
    config: PhysTwinMVTrackerCompetenceConfig,
    query_sha256: str,
) -> dict[str, object]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "phystwin-mvtracker-prefix-competence-v1":
        raise ValueError("protocol ID differs from the implementation")
    if protocol.get("method_config") != json.loads(
        json.dumps(config.__dict__, allow_nan=False)
    ):
        raise ValueError("protocol method config differs from the implementation")
    if protocol.get("source_artifacts", {}).get("query_input_sha256") != query_sha256:
        raise ValueError("query hash differs from the locked protocol")
    if protocol.get("status") != "locked-before-mvtracker-prediction":
        raise ValueError("protocol is not prediction-locked")
    return protocol


def _run_prediction(args: argparse.Namespace) -> dict[str, object]:
    import torch

    config = PhysTwinMVTrackerCompetenceConfig()
    mvtracker_root = args.mvtracker_root.resolve()
    if _git_revision(mvtracker_root) != MVTRACKER_REVISION:
        raise ValueError("MVTracker repository revision changed")
    checkpoint = args.mvtracker_checkpoint.resolve()
    if file_sha256(checkpoint) != MVTRACKER_CHECKPOINT_SHA256:
        raise ValueError("MVTracker checkpoint checksum changed")
    query, identity_ids = validate_query_input(
        args.query_input,
        args.query_sha256,
        config=config,
    )
    protocol = _load_locked_protocol(
        args.protocol.resolve(),
        config,
        args.query_sha256,
    )
    arrays, input_provenance = _load_prefix_inputs(
        args.raw_case_dir.resolve(),
        query,
        config,
    )
    input_provenance.update(
        {
            "query_input_sha256": args.query_sha256,
            "protocol_sha256": file_sha256(args.protocol),
            "mvtracker_checkpoint_sha256": file_sha256(checkpoint),
        }
    )

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

    torch.cuda.reset_peak_memory_stats()
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
    runner_path = Path(__file__).resolve()
    adapter_path = Path(validate_query_input.__code__.co_filename).resolve()
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
            "inputs": (
                "released sensor depth and calibration from the frozen source "
                "prefix only"
            ),
        },
        "checkpoint_sha256_verified": MVTRACKER_CHECKPOINT_SHA256,
        "pointops_fallback": True,
        "locked_protocol_status": protocol["status"],
    }
    return write_prediction_artifact(
        args.output_dir,
        raw_tracker_m=raw_world,
        visibility_probability=visibility,
        query_points_world_m=query,
        identity_ids=identity_ids,
        input_provenance=input_provenance,
        runtime_provenance=runtime_provenance,
        implementation_sha256={
            "runner": file_sha256(runner_path),
            "adapter": file_sha256(adapter_path),
            "protocol": file_sha256(args.protocol),
        },
        config=config,
    )


def main() -> int:
    args = _parse_args()
    if args.operation == "prepare-source":
        result = prepare_source_artifacts(
            args.manual_tracks,
            args.split,
            args.output_dir,
        )
    elif args.operation == "predict":
        result = _run_prediction(args)
    elif args.operation == "seal":
        result = seal_prediction(args.prediction_dir)
    else:
        result = evaluate_competence(
            args.prediction_dir,
            args.withheld_prefix,
            args.withheld_sha256,
            args.output,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
