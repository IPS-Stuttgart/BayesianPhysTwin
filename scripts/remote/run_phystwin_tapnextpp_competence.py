#!/usr/bin/env python3
"""Run the frozen PhysTwin TAPNext++ prefix competence control."""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.phystwin_mvtracker_competence import array_sha256
from bayesian_phystwin.phystwin_tapnextpp_competence import (
    TAPNEXTPP_CHECKPOINT_SHA256,
    TAPNEXTPP_REVISION,
    PhysTwinTAPNextPPCompetenceConfig,
    evaluate_competence,
    file_sha256,
    prepare_source_artifacts,
    seal_prediction,
    validate_prediction_input,
    write_prediction_artifact,
)
from bayesian_phystwin.tapnextpp_multiview import (
    fuse_causal_multiview_tracks,
    project_world_point,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)

    prepare = subparsers.add_parser("prepare-source")
    prepare.add_argument("--manual-tracks", type=Path, required=True)
    prepare.add_argument("--split", type=Path, required=True)
    prepare.add_argument("--processed-masks", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--raw-case-dir", type=Path, required=True)
    predict.add_argument("--prediction-input", type=Path, required=True)
    predict.add_argument("--prediction-input-sha256", required=True)
    predict.add_argument("--output-dir", type=Path, required=True)
    predict.add_argument("--tapnet-root", type=Path, required=True)
    predict.add_argument("--tapnextpp-checkpoint", type=Path, required=True)
    predict.add_argument("--protocol", type=Path, required=True)
    predict.add_argument("--device", default="cuda:0")

    seal = subparsers.add_parser("seal")
    seal.add_argument("--prediction-dir", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--prediction-dir", type=Path, required=True)
    evaluate.add_argument("--withheld-prefix", type=Path, required=True)
    evaluate.add_argument("--withheld-sha256", required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--protocol", type=Path)
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
    object_masks: np.ndarray,
    config: PhysTwinTAPNextPPCompetenceConfig,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    metadata_path = raw_case_dir / "metadata.json"
    calibration_path = raw_case_dir / "calibrate.pkl"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    intrinsics_all = np.asarray(metadata["intrinsics"], dtype=np.float32)
    with calibration_path.open("rb") as stream:
        camera_to_world_all = np.asarray(
            pickle.load(stream),
            dtype=np.float32,
        )
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
    camera_inputs: dict[str, object] = {}
    for local_camera, camera in enumerate(config.selected_cameras):
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
        if object_masks[local_camera].shape != depth.shape:
            raise ValueError(f"mask/depth shape differs for camera {camera}")
        rgbs.append(rgb)
        depths.append(depth)
        camera_inputs[str(camera)] = {
            "decoded_rgb_prefix_sha256": array_sha256(rgb),
            "sensor_depth_prefix_sha256": array_sha256(depth),
            "object_mask_prefix_sha256": array_sha256(
                object_masks[local_camera]
            ),
            "frame_file_hashes": frame_hashes,
        }
    selected = np.asarray(config.selected_cameras, dtype=np.int64)
    arrays = {
        "rgbs": np.stack(rgbs),
        "depths_m": np.stack(depths).astype(np.float32),
        "intrinsics": intrinsics_all[selected].astype(np.float32),
        "camera_to_world": camera_to_world_all[selected].astype(np.float32),
    }
    provenance: dict[str, object] = {
        "raw_case_dir": str(raw_case_dir),
        "metadata_sha256": file_sha256(metadata_path),
        "calibration_sha256": file_sha256(calibration_path),
        "selected_camera_count": len(selected),
        "cameras": camera_inputs,
        "depth_source": "released causal RGB-D sensor depth",
        "mask_source": "prefix-only staged released object masks",
    }
    return arrays, provenance


def _load_locked_protocol(
    protocol_path: Path,
    config: PhysTwinTAPNextPPCompetenceConfig,
    prediction_input_sha256: str,
) -> dict[str, object]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != (
        "phystwin-tapnextpp-prefix-competence-v1"
    ):
        raise ValueError("protocol ID differs from the implementation")
    expected_config = json.loads(json.dumps(config.__dict__, allow_nan=False))
    if protocol.get("method_config") != expected_config:
        raise ValueError("protocol method config differs from the implementation")
    if (
        protocol.get("source_artifacts", {}).get(
            "prediction_input_sha256"
        )
        != prediction_input_sha256
    ):
        raise ValueError("prediction-input hash differs from the protocol")
    if protocol.get("status") != "locked-before-tapnextpp-prediction":
        raise ValueError("protocol is not prediction-locked")
    return protocol


def _config_from_protocol(protocol_path: Path) -> PhysTwinTAPNextPPCompetenceConfig:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    raw = dict(protocol.get("method_config", {}))
    if "selected_identity_ids" in raw:
        raw["selected_identity_ids"] = tuple(raw["selected_identity_ids"])
    if "selected_cameras" in raw:
        raw["selected_cameras"] = tuple(raw["selected_cameras"])
    return PhysTwinTAPNextPPCompetenceConfig(**raw)


def _grid_support_points(
    count: int,
    width: float,
    height: float,
) -> np.ndarray:
    if count <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    columns = max(1, round(float(np.sqrt(count * width / height))))
    rows = max(1, int(np.ceil(count / columns)))
    xs = (np.arange(columns) + 0.5) * (width / columns)
    ys = (np.arange(rows) + 0.5) * (height / rows)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack(
        [grid_x.ravel(), grid_y.ravel()],
        axis=-1,
    ).astype(np.float32)[:count]


def _local_support_points(
    query_xy: np.ndarray,
    count_per_query: int,
    radius_x: float,
    radius_y: float,
    width: int,
    height: int,
) -> np.ndarray:
    if count_per_query <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    output = []
    for query_x, query_y in np.asarray(query_xy, dtype=np.float32):
        local = _grid_support_points(
            count_per_query,
            2.0 * radius_x,
            2.0 * radius_y,
        )
        local -= np.asarray([radius_x, radius_y], dtype=np.float32)
        local += np.asarray([query_x, query_y], dtype=np.float32)
        local[:, 0] = np.clip(local[:, 0], 0, width - 1)
        local[:, 1] = np.clip(local[:, 1], 0, height - 1)
        output.append(local)
    return np.concatenate(output, axis=0).astype(np.float32)


def _project_queries(
    query_points_world_m: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
) -> np.ndarray:
    projected = []
    for intrinsic, pose in zip(
        intrinsics,
        camera_to_world,
        strict=True,
    ):
        projection = intrinsic @ np.linalg.inv(pose)[:3]
        camera_points = []
        for point in query_points_world_m:
            image_point, _ = project_world_point(point, projection)
            if not np.all(np.isfinite(image_point)):
                raise ValueError("query point is behind a selected camera")
            camera_points.append(image_point)
        projected.append(camera_points)
    return np.asarray(projected, dtype=np.float32)


def _track_frame_with_probability(
    model: Any,
    frame_bgr: np.ndarray,
    *,
    query_points_xy: np.ndarray | None,
    state: Any,
    tapnext_utils: Any,
) -> tuple[np.ndarray, np.ndarray, Any]:
    import torch

    if query_points_xy is None and state is None:
        raise ValueError("query points are required for a fresh tracker state")
    height, width = frame_bgr.shape[:2]
    frame_tensor = tapnext_utils.preprocess_frame(
        frame_bgr,
        model.device,
        model.input_resolution,
    )
    query_tensor = None
    if query_points_xy is not None:
        model_points = tapnext_utils.display_to_model(
            query_points_xy,
            height,
            width,
            model.MODEL_SIZE,
        )
        query_tensor = tapnext_utils.make_query_tensor(
            model_points,
            model.device,
        )
    context = (
        torch.amp.autocast("cuda", dtype=torch.float16)
        if model.device.type == "cuda"
        else torch.amp.autocast("cpu", enabled=False)
    )
    with torch.no_grad(), context:
        tracks, _, visibility_logits, new_state = model._model(
            video=frame_tensor,
            query_points=query_tensor,
            state=state,
        )
    tracks_xy = tracks[0, 0].detach().float().cpu().numpy()[:, ::-1].copy()
    positions_xy = tapnext_utils.model_to_display(
        tracks_xy,
        height,
        width,
        model.MODEL_SIZE,
    )
    probability = (
        torch.sigmoid(visibility_logits[0, 0, :, 0])
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    return positions_xy.astype(np.float32), probability.astype(np.float32), new_state


def _run_online_tracker(
    model: Any,
    rgbs: np.ndarray,
    projected_queries_xy: np.ndarray,
    config: PhysTwinTAPNextPPCompetenceConfig,
    tapnext_utils: Any,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    import torch

    camera_count, frame_count, height, width, _ = rgbs.shape
    point_count = projected_queries_xy.shape[1]
    tracks = np.zeros(
        (camera_count, frame_count, point_count, 2),
        dtype=np.float32,
    )
    visibility = np.zeros(
        (camera_count, frame_count, point_count),
        dtype=np.float32,
    )
    elapsed_seconds = []
    radius_x = config.support_radius_model_px * (
        width / config.input_resolution
    )
    radius_y = config.support_radius_model_px * (
        height / config.input_resolution
    )
    for camera in range(camera_count):
        supports = _local_support_points(
            projected_queries_xy[camera],
            config.support_points_per_query,
            radius_x,
            radius_y,
            width,
            height,
        )
        all_queries = np.concatenate(
            [projected_queries_xy[camera], supports],
            axis=0,
        )
        state = None
        start = time.perf_counter()
        for frame in range(frame_count):
            frame_bgr = rgbs[camera, frame, :, :, ::-1].copy()
            positions, probabilities, state = _track_frame_with_probability(
                model,
                frame_bgr,
                query_points_xy=all_queries if frame == 0 else None,
                state=state,
                tapnext_utils=tapnext_utils,
            )
            tracks[camera, frame] = positions[:point_count]
            visibility[camera, frame] = probabilities[:point_count]
        torch.cuda.synchronize(model.device)
        elapsed_seconds.append(time.perf_counter() - start)
        del state
        torch.cuda.empty_cache()
    return tracks, visibility, elapsed_seconds


def _run_prediction(args: argparse.Namespace) -> dict[str, object]:
    import torch

    config = _config_from_protocol(args.protocol.resolve())
    tapnet_root = args.tapnet_root.resolve()
    if _git_revision(tapnet_root) != TAPNEXTPP_REVISION:
        raise ValueError("TAPNet repository revision changed")
    checkpoint = args.tapnextpp_checkpoint.resolve()
    if file_sha256(checkpoint) != TAPNEXTPP_CHECKPOINT_SHA256:
        raise ValueError("TAPNext++ checkpoint checksum changed")
    query, identity_ids, object_masks = validate_prediction_input(
        args.prediction_input,
        args.prediction_input_sha256,
        config=config,
    )
    protocol = _load_locked_protocol(
        args.protocol.resolve(),
        config,
        args.prediction_input_sha256,
    )
    arrays, input_provenance = _load_prefix_inputs(
        args.raw_case_dir.resolve(),
        object_masks,
        config,
    )
    input_provenance.update(
        {
            "prediction_input_sha256": args.prediction_input_sha256,
            "protocol_sha256": file_sha256(args.protocol),
            "tapnextpp_checkpoint_sha256": file_sha256(checkpoint),
        }
    )
    if str(tapnet_root) not in sys.path:
        sys.path.insert(0, str(tapnet_root))
    from tapnet.tapnextpp.votsp2026 import utils as tapnext_utils
    from tapnet.tapnextpp.votsp2026.model import TAPNextPP

    device = torch.device(args.device)
    torch.manual_seed(72)
    torch.cuda.manual_seed_all(72)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    model = TAPNextPP.from_checkpoint(
        checkpoint,
        device=device,
        half_precision=False,
        compile_model=False,
        input_resolution=config.input_resolution,
    )
    projected_queries = _project_queries(
        query,
        arrays["intrinsics"],
        arrays["camera_to_world"],
    )
    tracks, camera_visibility, elapsed_seconds = _run_online_tracker(
        model,
        arrays["rgbs"],
        projected_queries,
        config,
        tapnext_utils,
    )
    fused = fuse_causal_multiview_tracks(
        tracks,
        camera_visibility,
        arrays["depths_m"],
        object_masks,
        arrays["intrinsics"],
        arrays["camera_to_world"],
        query,
        config=config.multiview_config,
    )
    runner_path = Path(__file__).resolve()
    adapter_path = Path(
        validate_prediction_input.__code__.co_filename
    ).resolve()
    multiview_path = Path(
        fuse_causal_multiview_tracks.__code__.co_filename
    ).resolve()
    runtime_provenance = {
        "device": str(device),
        "per_camera_elapsed_seconds": elapsed_seconds,
        "total_elapsed_seconds": float(np.sum(elapsed_seconds)),
        "peak_gpu_memory_gib": (
            torch.cuda.max_memory_allocated(device) / (1024**3)
        ),
        "input_resolution": config.input_resolution,
        "support_points_per_query": config.support_points_per_query,
        "support_radius_model_px": config.support_radius_model_px,
        "continuous_visibility_logits_retained": True,
        "checkpoint_sha256_verified": TAPNEXTPP_CHECKPOINT_SHA256,
        "locked_protocol_status": protocol["status"],
    }
    return write_prediction_artifact(
        args.output_dir,
        raw_tracker_m=fused["trajectory_world_m"],
        accepted_support=fused["accepted_support"],
        observation_reliability=fused["observation_reliability"],
        observation_covariance_m2=fused["observation_covariance_m2"],
        support_view_count=fused["support_view_count"],
        reprojection_rmse_px=fused["reprojection_rmse_px"],
        depth_residual_rmse_m=fused["depth_residual_rmse_m"],
        per_camera_tracks_xy=tracks,
        per_camera_visibility_probability=camera_visibility,
        query_points_world_m=query,
        identity_ids=identity_ids,
        input_provenance=input_provenance,
        runtime_provenance=runtime_provenance,
        implementation_sha256={
            "runner": file_sha256(runner_path),
            "adapter": file_sha256(adapter_path),
            "multiview": file_sha256(multiview_path),
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
            args.processed_masks,
            args.output_dir,
        )
    elif args.operation == "predict":
        result = _run_prediction(args)
    elif args.operation == "seal":
        result = seal_prediction(args.prediction_dir)
    else:
        config = (
            _config_from_protocol(args.protocol.resolve())
            if args.protocol is not None
            else PhysTwinTAPNextPPCompetenceConfig()
        )
        result = evaluate_competence(
            args.prediction_dir,
            args.withheld_prefix,
            args.withheld_sha256,
            args.output,
            config=config,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
