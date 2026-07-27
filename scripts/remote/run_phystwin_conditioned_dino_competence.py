#!/usr/bin/env python3
"""Run the frozen PhysTwin-conditioned DINO prefix competence control."""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from bayesian_phystwin.matphys_dino_features import project_world_points
from bayesian_phystwin.phystwin_conditioned_dino_competence import (
    PhysTwinConditionedDinoCompetenceConfig,
    evaluate_competence,
    prepare_source_artifacts,
    seal_prediction,
    validate_prediction_input,
    write_prediction_artifact,
)
from bayesian_phystwin.phystwin_conditioned_dino_correspondence import (
    ConditionedDinoConfig,
    MetricViewObservation,
    fuse_unknown_correlation,
    match_descriptor_near_prediction,
    refine_patch_correlation,
    unproject_rgbd_observation,
)
from bayesian_phystwin.phystwin_mvtracker_competence import (
    array_sha256,
    file_sha256,
)

DINO_MODEL_NAME = "dinov2_vits14_reg"
DINO_RESIZE_HEIGHT = 476
DINO_RESIZE_WIDTH = 840
MINIMUM_FUSED_RELIABILITY = 0.05
MASK_DISTANCE_FULL_SUPPORT_PX = 8.0
DEPTH_PATCH_RADIUS_PX = 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)

    prepare = subparsers.add_parser("prepare-source")
    prepare.add_argument("--manual-tracks", type=Path, required=True)
    prepare.add_argument("--split", type=Path, required=True)
    prepare.add_argument("--physical-trajectory", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--raw-case-dir", type=Path, required=True)
    predict.add_argument("--prediction-input", type=Path, required=True)
    predict.add_argument("--prediction-input-sha256", required=True)
    predict.add_argument("--output-dir", type=Path, required=True)
    predict.add_argument("--dino-root", type=Path, required=True)
    predict.add_argument("--dino-checkpoint", type=Path, required=True)
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


def _load_rgb(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - integration runtime
        raise RuntimeError("Pillow is required for PhysTwin RGB input") from error
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _load_mask(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - integration runtime
        raise RuntimeError("Pillow is required for PhysTwin masks") from error
    with Image.open(path) as image:
        mask = np.asarray(image)
    if mask.ndim == 3:
        mask = np.any(mask > 0, axis=2)
    else:
        mask = mask > 0
    return np.asarray(mask, dtype=bool)


def _metric_depth(path: Path) -> np.ndarray:
    depth = np.asarray(np.load(path))
    if np.issubdtype(depth.dtype, np.integer) or float(np.nanmax(depth)) > 50.0:
        return depth.astype(np.float32) / 1000.0
    return depth.astype(np.float32)


def _camera_data(case_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    with (case_dir / "calibrate.pkl").open("rb") as stream:
        camera_to_world = np.asarray(pickle.load(stream), dtype=np.float64)
    metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
    intrinsics = np.asarray(metadata["intrinsics"], dtype=np.float64)
    if camera_to_world.ndim != 3 or camera_to_world.shape[1:] != (4, 4):
        raise ValueError("camera calibration must have shape (C, 4, 4)")
    if intrinsics.shape != (len(camera_to_world), 3, 3):
        raise ValueError("camera intrinsics must have shape (C, 3, 3)")
    if np.max(np.abs(intrinsics[:, :2, :])) <= 2.0:
        widths_heights = metadata["WH"]
        if not isinstance(widths_heights[0], list):
            widths_heights = [widths_heights for _ in range(len(intrinsics))]
        for camera, (width, height) in enumerate(widths_heights):
            intrinsics[camera, 0, :] *= float(width)
            intrinsics[camera, 1, :] *= float(height)
    return camera_to_world, intrinsics


def _load_prefix(
    case_dir: Path,
    config: PhysTwinConditionedDinoCompetenceConfig,
) -> tuple[
    dict[int, list[np.ndarray]],
    dict[int, list[np.ndarray]],
    dict[int, list[np.ndarray]],
    np.ndarray,
    np.ndarray,
    dict[str, object],
]:
    camera_to_world, intrinsics = _camera_data(case_dir)
    rgbs: dict[int, list[np.ndarray]] = {}
    depths: dict[int, list[np.ndarray]] = {}
    masks: dict[int, list[np.ndarray]] = {}
    camera_records: dict[str, object] = {}
    for camera in config.selected_cameras:
        if camera >= len(camera_to_world):
            raise ValueError("selected camera exceeds released calibration")
        camera_rgbs: list[np.ndarray] = []
        camera_depths: list[np.ndarray] = []
        camera_masks: list[np.ndarray] = []
        frame_records: list[dict[str, object]] = []
        for frame in range(
            config.reference_frame,
            config.source_frame_end_exclusive,
        ):
            rgb_path = case_dir / "color" / str(camera) / f"{frame}.png"
            depth_path = case_dir / "depth" / str(camera) / f"{frame}.npy"
            mask_path = case_dir / "mask" / str(camera) / "0" / f"{frame}.png"
            rgb = _load_rgb(rgb_path)
            depth = _metric_depth(depth_path)
            mask = _load_mask(mask_path)
            if rgb.shape[:2] != depth.shape or depth.shape != mask.shape:
                raise ValueError("RGB, depth, and object mask shapes differ")
            camera_rgbs.append(rgb)
            camera_depths.append(depth)
            camera_masks.append(mask)
            frame_records.append(
                {
                    "frame": frame,
                    "rgb_sha256": file_sha256(rgb_path),
                    "depth_sha256": file_sha256(depth_path),
                    "object_mask_sha256": file_sha256(mask_path),
                }
            )
        rgbs[camera] = camera_rgbs
        depths[camera] = camera_depths
        masks[camera] = camera_masks
        camera_records[str(camera)] = {
            "decoded_rgb_prefix_sha256": array_sha256(np.stack(camera_rgbs)),
            "metric_depth_prefix_sha256": array_sha256(np.stack(camera_depths)),
            "object_mask_prefix_sha256": array_sha256(np.stack(camera_masks)),
            "frame_files": frame_records,
        }
    provenance: dict[str, object] = {
        "raw_case_dir": str(case_dir),
        "metadata_sha256": file_sha256(case_dir / "metadata.json"),
        "calibration_sha256": file_sha256(case_dir / "calibrate.pkl"),
        "cameras": camera_records,
        "depth_source": "released causal RGB-D sensor depth",
        "object_mask_source": "released causal object mask class 0",
    }
    return rgbs, depths, masks, camera_to_world, intrinsics, provenance


def _load_protocol(
    path: Path,
    *,
    config: PhysTwinConditionedDinoCompetenceConfig,
    correspondence_config: ConditionedDinoConfig,
    prediction_input_sha256: str,
) -> dict[str, object]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != (
        "phystwin-conditioned-dino-prefix-competence-v1"
    ):
        raise ValueError("protocol ID differs from the implementation")
    if protocol.get("status") != "locked-before-conditioned-dino-prediction":
        raise ValueError("protocol is not prediction-locked")
    serialized_config = json.loads(json.dumps(asdict(config), allow_nan=False))
    serialized_correspondence = json.loads(
        json.dumps(asdict(correspondence_config), allow_nan=False)
    )
    if protocol.get("method_config") != serialized_config:
        raise ValueError("protocol method config differs from the implementation")
    if protocol.get("correspondence_config") != serialized_correspondence:
        raise ValueError("protocol correspondence config differs")
    source = protocol.get("source_artifacts", {})
    if source.get("prediction_input_sha256") != prediction_input_sha256:
        raise ValueError("prediction input hash differs from the protocol")
    runtime = protocol.get("dino", {})
    if runtime.get("model_name") != DINO_MODEL_NAME:
        raise ValueError("DINO model name differs from the implementation")
    if runtime.get("resize_height") != DINO_RESIZE_HEIGHT:
        raise ValueError("DINO resize height differs from the implementation")
    if runtime.get("resize_width") != DINO_RESIZE_WIDTH:
        raise ValueError("DINO resize width differs from the implementation")
    return protocol


def _load_dino(
    repository: Path,
    checkpoint: Path,
    *,
    model_name: str,
    device: str,
):
    import torch

    model = torch.hub.load(
        str(repository),
        model_name,
        source="local",
        pretrained=False,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if not isinstance(state, dict):
        raise ValueError("DINO checkpoint does not contain a state dictionary")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            "DINO checkpoint differs from architecture: "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
    return model.eval().to(torch.device(device))


def _extract_feature_map(model, image: np.ndarray, *, device: str) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    tensor = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float()
    tensor = tensor.unsqueeze(0).to(device) / 255.0
    tensor = functional.interpolate(
        tensor,
        size=(DINO_RESIZE_HEIGHT, DINO_RESIZE_WIDTH),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    with (
        torch.no_grad(),
        torch.amp.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=str(device).startswith("cuda"),
        ),
    ):
        output = model.forward_features((tensor - mean) / std)
    tokens = output.get("x_norm_patchtokens")
    if tokens is None:
        raise RuntimeError("DINO model does not expose normalized patch tokens")
    rows = DINO_RESIZE_HEIGHT // 14
    columns = DINO_RESIZE_WIDTH // 14
    if tokens.shape[1] != rows * columns:
        raise RuntimeError("DINO patch-token shape differs from the frozen resize")
    return tokens[0].reshape(rows, columns, -1).detach().float().cpu().numpy()


def _sample_feature(feature_map: np.ndarray, uv_px: np.ndarray, shape) -> np.ndarray:
    rows, columns = feature_map.shape[:2]
    height, width = shape
    coordinate = np.asarray(
        [
            uv_px[0] * columns / width - 0.5,
            uv_px[1] * rows / height - 0.5,
        ],
        dtype=np.float64,
    )
    lower = np.floor(coordinate).astype(np.int64)
    fraction = coordinate - lower
    lower[0] = np.clip(lower[0], 0, columns - 1)
    lower[1] = np.clip(lower[1], 0, rows - 1)
    upper = np.minimum(lower + 1, [columns - 1, rows - 1])
    top = (1.0 - fraction[0]) * feature_map[lower[1], lower[0]] + fraction[
        0
    ] * feature_map[lower[1], upper[0]]
    bottom = (1.0 - fraction[0]) * feature_map[upper[1], lower[0]] + fraction[
        0
    ] * feature_map[upper[1], upper[0]]
    descriptor = (1.0 - fraction[1]) * top + fraction[1] * bottom
    norm = np.linalg.norm(descriptor)
    if norm <= 1e-12:
        raise ValueError("sampled DINO descriptor is zero")
    return descriptor / norm


def _feature_valid_mask(mask: np.ndarray, feature_shape) -> np.ndarray:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - integration runtime
        raise RuntimeError("OpenCV is required for mask resampling") from error
    rows, columns = feature_shape
    resized = cv2.resize(
        np.asarray(mask, dtype=np.uint8),
        (columns, rows),
        interpolation=cv2.INTER_AREA,
    )
    return resized >= 0.50


def _mask_distance(mask: np.ndarray) -> np.ndarray:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - integration runtime
        raise RuntimeError("OpenCV is required for mask distance") from error
    return cv2.distanceTransform(
        np.asarray(mask, dtype=np.uint8),
        cv2.DIST_L2,
        5,
    )


def _depth_at_match(
    depth: np.ndarray,
    mask: np.ndarray,
    uv_px: np.ndarray,
) -> tuple[float, float] | None:
    center = np.rint(np.asarray(uv_px, dtype=np.float64)).astype(np.int64)
    x, y = int(center[0]), int(center[1])
    radius = DEPTH_PATCH_RADIUS_PX
    x0, x1 = max(0, x - radius), min(depth.shape[1], x + radius + 1)
    y0, y1 = max(0, y - radius), min(depth.shape[0], y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    values = np.asarray(depth[y0:y1, x0:x1], dtype=np.float64)
    valid = (
        np.asarray(mask[y0:y1, x0:x1], dtype=bool)
        & np.isfinite(values)
        & (values > 0.0)
    )
    selected = values[valid]
    if len(selected) < 3:
        return None
    median = float(np.median(selected))
    robust_standard_deviation = 1.4826 * float(np.median(np.abs(selected - median)))
    return median, robust_standard_deviation


def _rejected_view(config: ConditionedDinoConfig) -> MetricViewObservation:
    scale = (
        config.maximum_cross_view_disagreement_m
        + config.shared_bias_standard_deviation_m
    )
    return MetricViewObservation(
        mean_world_m=np.zeros(3),
        covariance_world_m2=scale**2 * np.eye(3),
        prior_reliability=0.0,
        accepted=False,
    )


def _predict(args: argparse.Namespace) -> dict[str, object]:
    import torch

    config = PhysTwinConditionedDinoCompetenceConfig()
    correspondence_config = ConditionedDinoConfig()
    protocol = _load_protocol(
        args.protocol.resolve(),
        config=config,
        correspondence_config=correspondence_config,
        prediction_input_sha256=args.prediction_input_sha256,
    )
    dino_root = args.dino_root.resolve()
    checkpoint = args.dino_checkpoint.resolve()
    if _git_revision(dino_root) != protocol["dino"]["revision"]:
        raise ValueError("DINO repository revision changed")
    if file_sha256(checkpoint) != protocol["dino"]["checkpoint_sha256"]:
        raise ValueError("DINO checkpoint hash changed")
    query, physical, _, identity_ids = validate_prediction_input(
        args.prediction_input,
        args.prediction_input_sha256,
        config=config,
    )
    (
        rgbs,
        depths,
        masks,
        camera_to_world,
        intrinsics,
        input_provenance,
    ) = _load_prefix(args.raw_case_dir.resolve(), config)
    input_provenance.update(
        {
            "prediction_input_sha256": args.prediction_input_sha256,
            "protocol_sha256": file_sha256(args.protocol),
            "dino_checkpoint_sha256": file_sha256(checkpoint),
        }
    )

    torch.manual_seed(72)
    torch.cuda.manual_seed_all(72)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = _load_dino(
        dino_root,
        checkpoint,
        model_name=DINO_MODEL_NAME,
        device=args.device,
    )
    if str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    start = time.perf_counter()
    feature_maps: dict[tuple[int, int], np.ndarray] = {}
    for camera in config.selected_cameras:
        for local_frame, rgb in enumerate(rgbs[camera]):
            feature_maps[camera, local_frame] = _extract_feature_map(
                model,
                rgb,
                device=args.device,
            )

    shape = physical.shape[:2]
    observed = np.full((*shape, 3), np.nan, dtype=np.float32)
    covariance = np.full((*shape, 3, 3), np.nan, dtype=np.float32)
    reliability = np.zeros(shape, dtype=np.float32)
    accepted = np.zeros(shape, dtype=bool)
    accepted_view_count = np.zeros(shape, dtype=np.int16)
    observed[0] = query
    covariance[0] = correspondence_config.shared_bias_standard_deviation_m**2 * np.eye(
        3, dtype=np.float32
    )
    reliability[0] = 1.0
    accepted[0] = True
    accepted_view_count[0] = len(config.selected_cameras)
    reference_descriptors: dict[tuple[int, int], np.ndarray] = {}
    reference_uvs: dict[tuple[int, int], np.ndarray] = {}
    reference_valid: dict[tuple[int, int], bool] = {}
    diagnostics: list[dict[str, object]] = []

    for camera in config.selected_cameras:
        world_to_camera = np.linalg.inv(camera_to_world[camera])
        reference_uv, reference_depth = project_world_points(
            query,
            world_to_camera,
            intrinsics[camera],
        )
        image_shape = rgbs[camera][0].shape[:2]
        for identity_index in range(len(identity_ids)):
            uv = reference_uv[identity_index]
            pixel = np.rint(uv).astype(np.int64)
            in_image = (
                0 <= pixel[0] < image_shape[1]
                and 0 <= pixel[1] < image_shape[0]
                and reference_depth[identity_index] > 0.0
            )
            valid = bool(
                in_image
                and masks[camera][0][pixel[1], pixel[0]]
                and depths[camera][0][pixel[1], pixel[0]] > 0.0
            )
            reference_valid[camera, identity_index] = valid
            reference_uvs[camera, identity_index] = uv
            if valid:
                reference_descriptors[camera, identity_index] = _sample_feature(
                    feature_maps[camera, 0],
                    uv,
                    image_shape,
                )

    for local_frame in range(1, config.prefix_frame_count):
        for identity_index, identity_id in enumerate(identity_ids):
            views: list[MetricViewObservation] = []
            view_diagnostics: list[dict[str, object]] = []
            for camera in config.selected_cameras:
                if not reference_valid[camera, identity_index]:
                    views.append(_rejected_view(correspondence_config))
                    view_diagnostics.append(
                        {
                            "camera": camera,
                            "decision": "reference_identity_not_visible",
                        }
                    )
                    continue
                world_to_camera = np.linalg.inv(camera_to_world[camera])
                predicted_uv, predicted_depth = project_world_points(
                    physical[local_frame, identity_index][None],
                    world_to_camera,
                    intrinsics[camera],
                )
                image = rgbs[camera][local_frame]
                image_shape = image.shape[:2]
                predicted = predicted_uv[0]
                if (
                    predicted_depth[0] <= 0.0
                    or predicted[0] < 0.0
                    or predicted[0] >= image_shape[1]
                    or predicted[1] < 0.0
                    or predicted[1] >= image_shape[0]
                ):
                    views.append(_rejected_view(correspondence_config))
                    view_diagnostics.append(
                        {
                            "camera": camera,
                            "decision": "physical_search_center_outside_image",
                        }
                    )
                    continue
                descriptor = match_descriptor_near_prediction(
                    reference_descriptors[camera, identity_index],
                    feature_maps[camera, local_frame],
                    _feature_valid_mask(
                        masks[camera][local_frame],
                        feature_maps[camera, local_frame].shape[:2],
                    ),
                    predicted,
                    image_width=image_shape[1],
                    image_height=image_shape[0],
                    config=correspondence_config,
                )
                if not descriptor.accepted:
                    views.append(_rejected_view(correspondence_config))
                    view_diagnostics.append(
                        {
                            "camera": camera,
                            "decision": descriptor.decision,
                            "descriptor_similarity": descriptor.cosine_similarity,
                            "descriptor_entropy": descriptor.normalized_entropy,
                        }
                    )
                    continue
                patch = refine_patch_correlation(
                    rgbs[camera][0],
                    image,
                    reference_uvs[camera, identity_index],
                    descriptor.uv_px,
                    masks[camera][local_frame] & (depths[camera][local_frame] > 0.0),
                    config=correspondence_config,
                )
                depth_result = _depth_at_match(
                    depths[camera][local_frame],
                    masks[camera][local_frame],
                    patch.uv_px,
                )
                if not patch.accepted or depth_result is None:
                    views.append(_rejected_view(correspondence_config))
                    view_diagnostics.append(
                        {
                            "camera": camera,
                            "decision": (
                                patch.decision
                                if not patch.accepted
                                else "insufficient_metric_depth"
                            ),
                            "descriptor_similarity": descriptor.cosine_similarity,
                            "patch_correlation": patch.correlation,
                        }
                    )
                    continue
                depth_m, depth_spread_m = depth_result
                pixel = np.rint(patch.uv_px).astype(np.int64)
                distance = _mask_distance(masks[camera][local_frame])
                boundary_support = float(
                    np.clip(
                        distance[pixel[1], pixel[0]] / MASK_DISTANCE_FULL_SUPPORT_PX,
                        0.0,
                        1.0,
                    )
                )
                prior_reliability = float(
                    np.sqrt(descriptor.prior_reliability * patch.prior_reliability)
                    * boundary_support
                )
                pixel_covariance = descriptor.covariance_px2 + patch.covariance_px2
                metric = unproject_rgbd_observation(
                    patch.uv_px,
                    pixel_covariance,
                    depth_m,
                    intrinsics[camera],
                    camera_to_world[camera],
                    prior_reliability=prior_reliability,
                    depth_standard_deviation_m=max(
                        correspondence_config.depth_standard_deviation_m,
                        depth_spread_m,
                    ),
                    accepted=prior_reliability > 0.0,
                )
                views.append(metric)
                view_diagnostics.append(
                    {
                        "camera": camera,
                        "decision": "accepted",
                        "descriptor_similarity": descriptor.cosine_similarity,
                        "descriptor_entropy": descriptor.normalized_entropy,
                        "descriptor_probability": (descriptor.association_probability),
                        "patch_correlation": patch.correlation,
                        "patch_entropy": patch.normalized_entropy,
                        "patch_probability": patch.association_probability,
                        "boundary_support": boundary_support,
                        "prior_reliability": prior_reliability,
                        "depth_m": depth_m,
                        "depth_spread_m": depth_spread_m,
                        "matched_uv_px": patch.uv_px.tolist(),
                    }
                )
            fused = fuse_unknown_correlation(
                views,
                config=correspondence_config,
            )
            row_accepted = bool(
                fused.accepted and fused.prior_reliability >= MINIMUM_FUSED_RELIABILITY
            )
            if row_accepted:
                observed[local_frame, identity_index] = fused.mean_world_m
                covariance[local_frame, identity_index] = fused.covariance_world_m2
                reliability[local_frame, identity_index] = fused.prior_reliability
                accepted[local_frame, identity_index] = True
            accepted_view_count[local_frame, identity_index] = int(
                np.sum(fused.accepted_view_mask)
            )
            diagnostics.append(
                {
                    "frame": config.reference_frame + local_frame,
                    "identity_id": int(identity_id),
                    "accepted": row_accepted,
                    "decision": fused.decision,
                    "fused_prior_reliability": fused.prior_reliability,
                    "accepted_view_count": int(np.sum(fused.accepted_view_mask)),
                    "maximum_pair_disagreement_m": (fused.maximum_pair_disagreement_m),
                    "views": view_diagnostics,
                }
            )
    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(args.device))
    elapsed = time.perf_counter() - start
    runtime = {
        "device": args.device,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_gib": (
            torch.cuda.max_memory_allocated(torch.device(args.device)) / (1024**3)
            if str(args.device).startswith("cuda")
            else 0.0
        ),
        "dino_model_name": DINO_MODEL_NAME,
        "dino_revision": _git_revision(dino_root),
        "dino_checkpoint_sha256": file_sha256(checkpoint),
        "dino_resize": [DINO_RESIZE_HEIGHT, DINO_RESIZE_WIDTH],
        "minimum_fused_reliability": MINIMUM_FUSED_RELIABILITY,
        "mask_distance_full_support_px": MASK_DISTANCE_FULL_SUPPORT_PX,
        "depth_patch_radius_px": DEPTH_PATCH_RADIUS_PX,
        "per_identity_frame_diagnostics": diagnostics,
    }
    runner_path = Path(__file__).resolve()
    correspondence_path = Path(
        match_descriptor_near_prediction.__code__.co_filename
    ).resolve()
    competence_path = Path(validate_prediction_input.__code__.co_filename).resolve()
    return write_prediction_artifact(
        args.output_dir,
        observed_points_world_m=observed,
        observation_covariance_world_m2=covariance,
        prior_reliability=reliability,
        accepted=accepted,
        accepted_view_count=accepted_view_count,
        physical_points_world_m=physical,
        identity_ids=identity_ids,
        input_provenance=input_provenance,
        runtime_provenance=runtime,
        implementation_sha256={
            "runner": file_sha256(runner_path),
            "correspondence": file_sha256(correspondence_path),
            "competence": file_sha256(competence_path),
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
            args.physical_trajectory,
            args.output_dir,
        )
    elif args.operation == "predict":
        result = _predict(args)
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
