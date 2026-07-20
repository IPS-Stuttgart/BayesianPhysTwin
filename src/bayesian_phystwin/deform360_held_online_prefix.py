"""Outcome-blind causal RGB-prefix predictions for held Deform360 cases.

This stage starts only after a frame-zero physical prior has been sealed.  It
reads logical RGB prefixes ``[0,u]`` for ``u in (19, 38, 57)`` from the
action-aligned source interval recorded in the frame-zero manifest.  Masks,
depth, and calibration come exclusively from the extracted frame-zero NPZ;
the future-bearing processed HDF5 files are neither accepted nor opened.

The primary forecast is the method frozen by :mod:`deform360_held_protocol`:
current-observed-centre symmetric-Chamfer backbone selection followed by the
full-blend Euclidean RBF update.  Fewer than three observations produces exact
persistence.  No API in this module accepts a target, visibility, outcome, or
future-geometry input.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360_cpd_diagnostic import _symmetric_set_chamfer_m
from .deform360_frame_zero_assets import (
    FRAME_ZERO_CAMERA_SELECTION_POLICY_ID,
    FRAME_ZERO_CAMERA_SELECTION_RULE,
    frame_zero_view_diagnostics_sha256,
    validate_frame_zero_bundle_manifest as validate_frame_zero_asset_manifest,
)
from .deform360_held_protocol import (
    ONLINE_ARTIFACT_ROLES,
    PRIMARY_METHOD,
    PROTOCOL_ID,
    UPDATE_FRAMES,
    create_online_prediction_seal,
    held_artifact_sha256,
    validate_frame_zero_bundle_manifest,
    validate_physical_prior_seal,
    validate_prefix_stage_authorization,
)
from .deform360_held_physical_prior import (
    FRAME_ZERO_ARRAYS,
    validate_physical_prediction_manifest,
)
from .deform360_raw_camera_cycle_uncertainty import inflate_covariance_from_cycle
from .deform360_raw_camera_observation import (
    ALLTRACKER_CHECKPOINT_SHA256,
    ALLTRACKER_MOLMOMOTION_REVISION,
    ALLTRACKER_RUNTIME_SOURCE_SHA256,
    ALLTRACKER_SOURCE_TREE,
    AllTrackerPrefixRuntime,
    RawCameraObservationConfig,
    project_world_points,
    select_frame_zero_observation_plan,
    triangulate_observation_ransac,
)
from .deform360_raw_camera_uncertainty import (
    RawCameraUncertaintyConfig,
    jacobian_measurement_covariance,
    leave_one_camera_out_covariance,
)
from .phystwin_online_belief import (
    RecursiveRbfBeliefConfig,
    decode_recursive_rbf_belief,
    initialize_recursive_rbf_belief,
    update_recursive_rbf_belief,
)


SCHEMA_VERSION = 1
FRAME_COUNT = 76
MINIMUM_SELECTOR_SUPPORT = 3
HELD_OBSERVATION_CONFIG = RawCameraObservationConfig()
HELD_UNCERTAINTY_CONFIG = RawCameraUncertaintyConfig()
HELD_RBF_CONFIG = RecursiveRbfBeliefConfig(local_blend=1.0)
HELD_FRAME_ZERO_SAM2_PARAMETERS = {
    "points_per_side": 24,
    "points_per_batch": 64,
    "predicted_iou_threshold": 0.70,
    "stability_score_threshold": 0.80,
    "minimum_mask_region_area": 100,
    "minimum_area_fraction": 0.0005,
    "maximum_area_fraction": 0.65,
    "minimum_foreground_contrast": 0.06,
    "maximum_normalized_center_distance": 0.90,
    "border_side_penalty": 0.55,
    "minimum_reference_appearance_similarity": 0.35,
}
HELD_FRAME_ZERO_CONFIG = {
    "reference_camera": "brics-odroid-001_cam0",
    "minimum_camera_count": 8,
    "cube_half_extent_m": 0.5,
    "requested_voxel_size_m": 0.008,
    "maximum_grid_point_count": 1_500_000,
    "consensus_fraction_of_peak": 0.55,
    "minimum_consensus_votes": 8,
    "minimum_hull_point_count": 1_000,
    "minimum_largest_component_fraction": 0.50,
    "object_point_count": 10_000,
    "depth_splat_radius_pixels": 2,
    "minimum_median_depth_mask_coverage": 0.10,
    "minimum_median_hull_mask_containment": 0.60,
    "action_window_length_frames": 81,
    "prediction_frame_count": 76,
    "action_candidate_first_frame": 8,
    "action_candidate_stride_frames": 6,
    "rng_seed": 0,
    "sam2": HELD_FRAME_ZERO_SAM2_PARAMETERS,
}
HELD_FRAME_ZERO_SAM2 = {
    "repository": "https://github.com/facebookresearch/sam2",
    "commit": "2b90b9f5ceec907a1c18123530e92e794ad901a4",
    "checkpoint_sha256": (
        "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"
    ),
    "model_config": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "model_id": "bayesian_phystwin/frame-zero-sam2.1-small-automatic-v1@2b90b9f5ceec",
    "parameters": HELD_FRAME_ZERO_SAM2_PARAMETERS,
    "image_model_only": True,
    "video_propagator_constructed": False,
}
MEASUREMENT_ARCHIVE_FILENAME = "measurement.npz"
MEASUREMENT_MANIFEST_FILENAME = "measurement_manifest.json"
UNCERTAINTY_ARCHIVE_FILENAME = "measurement_uncertainty.npz"
UNCERTAINTY_MANIFEST_FILENAME = "measurement_uncertainty_manifest.json"
CYCLE_ARCHIVE_FILENAME = "measurement_cycle_uncertainty.npz"
CYCLE_MANIFEST_FILENAME = "measurement_cycle_uncertainty_manifest.json"
ONLINE_PREDICTION_ARCHIVE_FILENAME = "online_prediction.npz"
ONLINE_SEAL_FILENAME = "online_prediction_seal.json"

MEASUREMENT_KIND = "Deform360HeldCausalRawCameraMeasurement"
UNCERTAINTY_KIND = "Deform360HeldCausalRawCameraMeasurementUncertainty"
CYCLE_KIND = "Deform360HeldCausalRawCameraCycleUncertainty"

_EXPECTED_FRAME_ZERO_ARRAYS = FRAME_ZERO_ARRAYS
_EXPECTED_PHYSICAL_ARRAYS = frozenset(
    {
        "prediction_m",
        "persistence_m",
        "driven_readout_m",
        "zero_action_readout_m",
        "action_support",
        "frame_zero_points_m",
    }
)
_FORBIDDEN_SUFFIXES = frozenset({".h5", ".hdf5", ".hdf"})
_FORBIDDEN_NAMES = frozenset({"outcome.json", "target_data.pkl"})
_FORBIDDEN_TOKENS = (
    "mask_refined",
    "rendered_depth",
    "target_data",
    "outcome",
    "ground_truth",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    descriptor = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(
        descriptor + b"\0" + array.view(np.uint8).tobytes()
    ).hexdigest()


def _bound_file(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    _require(
        source.is_file() and not source.is_symlink(), f"missing regular file: {source}"
    )
    return {
        "path": str(source),
        "sha256": _sha256_file(source),
        "size_bytes": source.stat().st_size,
    }


def _write_manifest(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    payload["artifact_sha256"] = held_artifact_sha256(payload)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def reject_forbidden_prefix_input(path: str | Path, *, purpose: str) -> Path:
    """Reject any future-bearing processed or outcome path before opening it."""

    resolved = Path(path).resolve()
    lowered = resolved.as_posix().lower()
    _require(resolved.name.lower() not in _FORBIDDEN_NAMES, f"{purpose} is an outcome")
    _require(
        resolved.suffix.lower() not in _FORBIDDEN_SUFFIXES,
        f"{purpose} may not be an HDF5 container",
    )
    _require(
        not any(token in lowered for token in _FORBIDDEN_TOKENS),
        f"{purpose} appears future-derived",
    )
    return resolved


def _projection_matrix(
    intrinsics: np.ndarray, camera_to_world: np.ndarray
) -> np.ndarray:
    return (
        np.asarray(intrinsics, dtype=float)
        @ np.linalg.inv(np.asarray(camera_to_world, dtype=float))[:3]
    )


def _load_frame_zero_arrays(
    manifest_path: str | Path,
    lock_path: str | Path,
    *,
    case_name: str,
    role: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest = validate_frame_zero_bundle_manifest(
        manifest_path,
        lock_path,
        expected_case_name=case_name,
        expected_role=role,
    )
    validate_frame_zero_asset_manifest(manifest)
    _require(
        manifest.get("config") == HELD_FRAME_ZERO_CONFIG,
        "held frame-zero configuration changed",
    )
    sam2 = manifest.get("sam2")
    _require(
        isinstance(sam2, Mapping)
        and set(sam2)
        == set(HELD_FRAME_ZERO_SAM2) | {"view_diagnostics", "view_diagnostics_sha256"}
        and all(sam2.get(key) == value for key, value in HELD_FRAME_ZERO_SAM2.items())
        and isinstance(sam2.get("view_diagnostics"), list)
        and sam2.get("view_diagnostics_sha256")
        == frame_zero_view_diagnostics_sha256(sam2["view_diagnostics"]),
        "held frame-zero SAM2 configuration changed",
    )
    geometry_qa = manifest.get("geometry_qa")
    _require(
        isinstance(geometry_qa, Mapping)
        and geometry_qa.get("geometry_qa_passed") is True,
        "held frame-zero geometry did not pass QA",
    )
    bundle_record = manifest["bundle"]
    bundle_path = reject_forbidden_prefix_input(
        str(bundle_record["path"]), purpose="frame-zero bundle"
    )
    _require(
        _sha256_file(bundle_path) == bundle_record["sha256"],
        "frame-zero bundle changed",
    )
    manifest_arrays = manifest.get("arrays")
    _require(
        isinstance(manifest_arrays, Mapping)
        and set(manifest_arrays) == set(_EXPECTED_FRAME_ZERO_ARRAYS),
        "frame-zero manifest array set changed",
    )
    with np.load(bundle_path, allow_pickle=False) as stored:
        _require(
            set(stored.files) == set(_EXPECTED_FRAME_ZERO_ARRAYS),
            "frame-zero bundle array set changed",
        )
        for name in sorted(_EXPECTED_FRAME_ZERO_ARRAYS):
            value = np.asarray(stored[name])
            record = manifest_arrays[name]
            _require(
                isinstance(record, Mapping)
                and set(record) == {"shape", "dtype", "sha256"},
                f"invalid frame-zero array record: {name}",
            )
            _require(
                record["shape"] == list(value.shape)
                and record["dtype"] == value.dtype.str
                and record["sha256"] == _sha256_array(value),
                f"frame-zero array binding changed: {name}",
            )
        # Copy only the exact named contract after every stored array has been
        # checksum/shape/dtype verified against the sealed manifest.
        arrays = {
            name: np.asarray(stored[name]).copy()
            for name in sorted(_EXPECTED_FRAME_ZERO_ARRAYS)
        }
    _require(
        arrays["frame_indices"].dtype == np.dtype(np.int64)
        and np.array_equal(arrays["frame_indices"], np.asarray([0], dtype=np.int64)),
        "frame-zero bundle contains another logical frame",
    )
    cameras = np.asarray(arrays["camera_names"])
    camera_count = len(cameras)
    _require(
        cameras.ndim == 1
        and cameras.dtype.kind == "U"
        and camera_count >= HELD_OBSERVATION_CONFIG.selected_camera_count
        and all(str(value) for value in cameras.tolist())
        and len(set(map(str, cameras.tolist()))) == camera_count,
        "camera names are invalid",
    )
    camera_policy = manifest.get("camera_policy")
    _require(
        isinstance(camera_policy, Mapping)
        and camera_policy.get("policy_id") == FRAME_ZERO_CAMERA_SELECTION_POLICY_ID
        and camera_policy.get("rule") == FRAME_ZERO_CAMERA_SELECTION_RULE
        and camera_policy.get("reference_camera")
        == HELD_FRAME_ZERO_CONFIG["reference_camera"]
        and camera_policy.get("reference_camera")
        in [str(value) for value in cameras.tolist()]
        and camera_policy.get("selected_cameras")
        == [str(value) for value in cameras.tolist()]
        and camera_policy.get("selected_camera_count") == camera_count
        and camera_policy.get("minimum_selected_camera_count")
        == HELD_FRAME_ZERO_CONFIG["minimum_camera_count"],
        "frame-zero camera policy changed",
    )
    candidate_cameras = camera_policy["candidate_cameras"]
    abstained_cameras = camera_policy["abstained_cameras"]
    _require(
        camera_policy.get("candidate_camera_count") == len(candidate_cameras)
        and camera_policy.get("abstained_camera_count") == len(abstained_cameras)
        and len(sam2["view_diagnostics"]) == len(candidate_cameras)
        and [record.get("camera") for record in sam2["view_diagnostics"]]
        == candidate_cameras
        and [
            record.get("camera")
            for record in sam2["view_diagnostics"]
            if record.get("view_selected") is True
        ]
        == [str(value) for value in cameras.tolist()],
        "frame-zero SAM2 camera diagnostics changed",
    )
    _require(
        arrays["rgb_frame0"].dtype == np.dtype(np.uint8)
        and arrays["rgb_frame0"].ndim == 4
        and arrays["rgb_frame0"].shape[0] == camera_count
        and arrays["rgb_frame0"].shape[-1] == 3,
        "RGB frame-zero array changed",
    )
    _require(
        arrays["mask_frame0"].dtype == np.dtype(bool)
        and arrays["mask_frame0"].shape == arrays["rgb_frame0"].shape[:3],
        "mask/RGB shapes differ",
    )
    _require(
        arrays["depth_frame0_m"].dtype == np.dtype(np.float32)
        and arrays["depth_frame0_m"].shape == arrays["mask_frame0"].shape
        and arrays["depth_valid_frame0"].dtype == np.dtype(bool)
        and arrays["depth_valid_frame0"].shape == arrays["mask_frame0"].shape
        and np.all(np.isfinite(arrays["depth_frame0_m"])),
        "depth/mask shapes differ",
    )
    _require(
        arrays["intrinsics"].dtype == np.dtype(np.float64)
        and arrays["intrinsics"].shape == (camera_count, 3, 3)
        and arrays["camera_to_world"].dtype == np.dtype(np.float64)
        and arrays["camera_to_world"].shape == (camera_count, 4, 4)
        and arrays["projection_world_to_pixel"].dtype == np.dtype(np.float64)
        and arrays["projection_world_to_pixel"].shape == (camera_count, 3, 4)
        and np.all(np.isfinite(arrays["intrinsics"]))
        and np.all(np.isfinite(arrays["camera_to_world"]))
        and np.all(np.isfinite(arrays["projection_world_to_pixel"])),
        "calibration arrays changed",
    )
    points = arrays["object_points_world_m"]
    colors = arrays["object_colors_rgb"]
    color_support = arrays["object_color_support_count"]
    hull = arrays["visual_hull_points_world_m"]
    _require(
        points.dtype == np.dtype(np.float32)
        and points.ndim == 2
        and points.shape[1] == 3
        and len(points) >= HELD_OBSERVATION_CONFIG.center_count
        and np.all(np.isfinite(points))
        and colors.dtype == np.dtype(np.uint8)
        and colors.shape == points.shape
        and color_support.dtype == np.dtype(np.uint8)
        and color_support.shape == (len(points),)
        and np.all(color_support > 0)
        and hull.dtype == np.dtype(np.float32)
        and hull.ndim == 2
        and hull.shape[1] == 3
        and len(hull) > 0
        and np.all(np.isfinite(hull)),
        "frame-zero object arrays changed",
    )
    for index in range(camera_count):
        expected = _projection_matrix(
            arrays["intrinsics"][index], arrays["camera_to_world"][index]
        )
        _require(
            np.allclose(
                expected,
                arrays["projection_world_to_pixel"][index],
                rtol=0.0,
                atol=1e-9,
            ),
            "stored projection differs from immutable calibration",
        )
    return manifest, arrays


def _frame_zero_support_from_bundle(
    points_m: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    *,
    depth_tolerance_m: float,
) -> tuple[tuple[str, ...], np.ndarray, dict[str, np.ndarray]]:
    cameras = tuple(str(value) for value in arrays["camera_names"].tolist())
    points = np.asarray(points_m, dtype=float)
    support = np.zeros((len(points), len(cameras)), dtype=bool)
    projected: dict[str, np.ndarray] = {}
    for index, camera in enumerate(cameras):
        pixels, depth = project_world_points(
            points,
            arrays["intrinsics"][index],
            arrays["camera_to_world"][index],
        )
        projected[camera] = pixels
        mask = np.asarray(arrays["mask_frame0"][index], dtype=bool)
        depth_map = np.asarray(arrays["depth_frame0_m"][index], dtype=float)
        depth_valid = np.asarray(arrays["depth_valid_frame0"][index], dtype=bool)
        height, width = mask.shape
        rounded = np.zeros((len(points), 2), dtype=np.int64)
        finite_pixels = np.all(np.isfinite(pixels), axis=1)
        rounded[finite_pixels] = np.rint(pixels[finite_pixels]).astype(np.int64)
        inside = (
            (depth > 0.0)
            & finite_pixels
            & (rounded[:, 0] >= 0)
            & (rounded[:, 0] < width)
            & (rounded[:, 1] >= 0)
            & (rounded[:, 1] < height)
        )
        ids = np.flatnonzero(inside)
        if len(ids):
            rows = rounded[ids, 1]
            columns = rounded[ids, 0]
            support[ids, index] = (
                mask[rows, columns]
                & depth_valid[rows, columns]
                & (depth_map[rows, columns] > 0.0)
                & (np.abs(depth_map[rows, columns] - depth[ids]) <= depth_tolerance_m)
            )
    return cameras, support, projected


def _decode_action_aligned_prefix(
    video_path: Path,
    *,
    source_frame_start: int,
    logical_update_frame: int,
    expected_frame_zero: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode exactly raw ``[start,start+u]`` and call it logical ``[0,u]``."""

    path = reject_forbidden_prefix_input(video_path, purpose="aligned RGB video")
    _require(path.suffix.lower() in {".mp4", ".mov", ".mkv"}, "unsupported RGB video")
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - GPU integration
        raise RuntimeError("OpenCV is required for causal RGB prefixes") from error
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    frame_hashes: list[str] = []
    digest = hashlib.sha256()
    try:
        if source_frame_start:
            _require(
                bool(capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame_start)),
                "cannot seek to held action frame zero",
            )
        for logical_frame in range(logical_update_frame + 1):
            okay, bgr = capture.read()
            _require(
                bool(okay) and bgr is not None,
                f"cannot decode logical RGB frame {logical_frame}",
            )
            rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            frames.append(rgb)
            frame_hashes.append(_sha256_array(rgb))
            digest.update(str(rgb.dtype).encode("ascii"))
            digest.update(np.asarray(rgb.shape, dtype=np.int64).tobytes())
            digest.update(rgb.tobytes())
    finally:
        capture.release()
    _require(
        np.array_equal(frames[0], np.asarray(expected_frame_zero)),
        "decoded logical frame zero differs from the sealed bundle",
    )
    return np.stack(frames), {
        "logical_prefix_frame_range_half_open": [0, logical_update_frame + 1],
        "source_prefix_frame_range_half_open": [
            source_frame_start,
            source_frame_start + logical_update_frame + 1,
        ],
        "maximum_logical_rgb_frame_read": logical_update_frame,
        "maximum_source_rgb_frame_read": source_frame_start + logical_update_frame,
        "decoded_frame_count": logical_update_frame + 1,
        "decoded_rgb_prefix_sha256": digest.hexdigest(),
        "decoded_rgb_frame_sha256": frame_hashes,
        "logical_frame_zero_matches_bundle": True,
        "whole_video_hashed_or_read": False,
    }


def _infer_tracks_from_rgb(
    runtime: AllTrackerPrefixRuntime,
    rgb_frames: np.ndarray,
    query_pixels_xy: np.ndarray,
    *,
    logical_update_frame: int,
    reverse: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Use the pinned AllTracker network on an already bounded RGB prefix."""

    try:
        import cv2
    except ImportError as error:  # pragma: no cover - GPU integration
        raise RuntimeError("OpenCV is required for AllTracker resizing") from error
    queries = np.asarray(query_pixels_xy, dtype=float)
    _require(
        queries.ndim == 2 and queries.shape[1] == 2 and np.all(np.isfinite(queries)),
        "query pixels must have finite shape (N,2)",
    )
    rgb = np.asarray(rgb_frames)
    _require(
        rgb.ndim == 4 and len(rgb) == logical_update_frame + 1,
        "RGB prefix length changed",
    )
    if reverse:
        rgb = np.ascontiguousarray(rgb[::-1])
    original_height, original_width = rgb.shape[1:3]
    scale = min(
        1.0,
        runtime.config.alltracker_max_side / max(original_height, original_width),
    )
    height = max(8, int(original_height * scale) // 8 * 8)
    width = max(8, int(original_width * scale) // 8 * 8)
    if (height, width) != (original_height, original_width):
        rgb = np.stack(
            [
                cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                for frame in rgb
            ]
        )
    torch = runtime._torch
    video = (
        torch.from_numpy(np.ascontiguousarray(rgb))
        .permute(0, 3, 1, 2)[None]
        .float()
        .to(runtime._device)
    )
    started = time.perf_counter()
    with torch.no_grad():
        flows, confidence, _, _ = runtime._model(
            video,
            iters=runtime.config.alltracker_inference_iterations,
            sw=None,
            is_training=False,
        )
    if flows.ndim == 4:
        flows = flows[:, None]
        flows = torch.cat((torch.zeros_like(flows[:, :1]), flows), dim=1)
    if confidence.ndim == 4:
        confidence = confidence[:, None]
    if confidence.shape[1] == flows.shape[1] - 1:
        confidence = torch.cat((torch.ones_like(confidence[:, :1]), confidence), dim=1)
    _require(
        flows.shape[1] == logical_update_frame + 1, "AllTracker prefix length changed"
    )
    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, device=runtime._device),
        torch.arange(width, device=runtime._device),
        indexing="ij",
    )
    grid = torch.stack((x_grid, y_grid), dim=0)[None, None].float()
    trajectories = flows + grid
    x_query = np.clip(
        np.rint(queries[:, 0] * width / original_width).astype(np.int64),
        0,
        width - 1,
    )
    y_query = np.clip(
        np.rint(queries[:, 1] * height / original_height).astype(np.int64),
        0,
        height - 1,
    )
    endpoint = trajectories[0, logical_update_frame, :, y_query, x_query].T
    endpoint[:, 0] *= original_width / width
    endpoint[:, 1] *= original_height / height
    query_confidence = confidence[0, logical_update_frame, 0, y_query, x_query]
    visible = query_confidence > runtime.config.visibility_threshold
    tracks = endpoint.detach().cpu().numpy().astype(np.float32)
    mask = visible.detach().cpu().numpy().astype(bool)
    confidence_np = query_confidence.detach().cpu().numpy().astype(np.float32)
    elapsed = time.perf_counter() - started
    del video, flows, confidence, trajectories, endpoint, query_confidence
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (
        tracks,
        mask,
        {
            "direction": "reverse" if reverse else "forward",
            "model_frame_order": (
                [logical_update_frame, 0] if reverse else [0, logical_update_frame]
            ),
            "original_image_shape": [original_height, original_width],
            "inference_image_shape": [height, width],
            "query_count": len(queries),
            "visible_query_count": int(np.sum(mask)),
            "visibility_confidence_minimum": (
                None if not len(confidence_np) else float(np.min(confidence_np))
            ),
            "visibility_confidence_median": (
                None if not len(confidence_np) else float(np.median(confidence_np))
            ),
            "visibility_confidence_maximum": (
                None if not len(confidence_np) else float(np.max(confidence_np))
            ),
            "runtime_seconds": elapsed,
        },
    )


def predict_support_gated_selected_backbone_rbf(
    physical_prior_m: np.ndarray,
    persistence_m: np.ndarray,
    measurement_m: np.ndarray,
    measurement_validity: np.ndarray,
    *,
    center_ids: np.ndarray,
    rbf_config: RecursiveRbfBeliefConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Frozen target-free primary; array-equivalent to the open ungated arm."""

    prior_input = np.asarray(physical_prior_m)
    persistence_input = np.asarray(persistence_m)
    prior = np.asarray(prior_input, dtype=float)
    persistence = np.asarray(persistence_input, dtype=float)
    measurement = np.asarray(measurement_m, dtype=float)
    measurement_mask = np.asarray(measurement_validity, dtype=bool)
    centers = np.asarray(center_ids, dtype=np.int64)
    _require(prior.shape == persistence.shape, "physical and persistence shapes differ")
    _require(
        prior.ndim == 3 and prior.shape[0] == FRAME_COUNT and prior.shape[2] == 3,
        "physical prior must have shape (76,N,3)",
    )
    _require(
        measurement.shape == prior.shape and measurement_mask.shape == prior.shape[:2],
        "measurement arrays differ from physical prior",
    )
    _require(
        centers.ndim == 1
        and len(np.unique(centers)) == len(centers)
        and np.all((0 <= centers) & (centers < prior.shape[1])),
        "invalid center IDs",
    )
    config = rbf_config or HELD_RBF_CONFIG
    _require(config == HELD_RBF_CONFIG, "held RBF configuration changed")
    output_dtype = prior_input.dtype
    selected_raw = prior_input.copy()
    prediction = prior_input.copy()
    backbones = {"physical_prior": prior, "persistence": persistence}
    states = {
        name: initialize_recursive_rbf_belief(
            centers,
            trajectory[0, centers],
            trajectory[0],
            config=config,
        )
        for name, trajectory in backbones.items()
    }
    updates: list[dict[str, Any]] = []
    for update_index, update in enumerate(UPDATE_FRAMES):
        stop = (
            UPDATE_FRAMES[update_index + 1]
            if update_index + 1 < len(UPDATE_FRAMES)
            else len(prior)
        )
        available = (
            measurement_mask[update, centers]
            & np.all(np.isfinite(measurement[update, centers]), axis=1)
            & np.all(np.isfinite(prior[update, centers]), axis=1)
            & np.all(np.isfinite(persistence[update, centers]), axis=1)
        )
        available_ids = centers[available]
        observed = measurement[update, available_ids]
        sufficient = len(available_ids) >= MINIMUM_SELECTOR_SUPPORT
        if sufficient:
            chamfer = {
                name: _symmetric_set_chamfer_m(
                    trajectory[update, available_ids], observed
                )
                for name, trajectory in backbones.items()
            }
            selected_name = min(
                ("physical_prior", "persistence"),
                key=lambda name: (
                    chamfer[name],
                    0 if name == "physical_prior" else 1,
                ),
            )
        else:
            chamfer = (
                {
                    name: _symmetric_set_chamfer_m(
                        trajectory[update, available_ids], observed
                    )
                    for name, trajectory in backbones.items()
                }
                if len(available_ids)
                else {"physical_prior": None, "persistence": None}
            )
            selected_name = "persistence"
        selected = backbones[selected_name]
        selected_raw[update + 1 : stop] = selected[update + 1 : stop]
        prediction[update + 1 : stop] = selected[update + 1 : stop]
        applied = False
        if sufficient:
            for backbone_name, trajectory in backbones.items():
                residual = np.full((len(centers), 3), np.nan, dtype=float)
                residual[available] = observed - trajectory[update, available_ids]
                posterior, _ = update_recursive_rbf_belief(
                    states[backbone_name],
                    update,
                    trajectory[update, centers],
                    residual,
                    available,
                    config=config,
                )
                states[backbone_name] = posterior
            posterior = states[selected_name]
            for frame in range(update + 1, stop):
                decoded = decode_recursive_rbf_belief(
                    posterior,
                    selected[update],
                    forecast_frames=frame - update,
                    config=config,
                )
                prediction[frame] = (
                    selected[frame].astype(float) + decoded.mean_m
                ).astype(output_dtype, copy=False)
            applied = True
        if not applied:
            _require(
                np.array_equal(
                    prediction[update + 1 : stop], selected[update + 1 : stop]
                ),
                "insufficient-support fallback is not bit-exact",
            )
        updates.append(
            {
                "frame": int(update),
                "stop_frame_exclusive": int(stop),
                "available_center_count": int(len(available_ids)),
                "selector_support_sufficient": sufficient,
                "selected_backbone": selected_name,
                "selector_decision": (
                    "current_observed_center_symmetric_chamfer"
                    if sufficient
                    else "insufficient_support_persistence"
                ),
                "current_observation_chamfer_m": chamfer,
                "rbf_correction_applied": applied,
            }
        )
    return (
        prediction,
        selected_raw,
        {
            "primary_method": dict(PRIMARY_METHOD),
            "rbf_config": asdict(config),
            "updates": updates,
        },
    )


def _validate_physical_archive_arrays(
    arrays: Mapping[str, np.ndarray],
    bundle_frame_zero: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _require(
        set(arrays) == set(_EXPECTED_PHYSICAL_ARRAYS),
        "physical prediction archive array set changed",
    )
    prior = np.asarray(arrays["prediction_m"])
    persistence = np.asarray(arrays["persistence_m"])
    frame_zero = np.asarray(arrays["frame_zero_points_m"])
    driven = np.asarray(arrays["driven_readout_m"])
    zero_action = np.asarray(arrays["zero_action_readout_m"])
    support = np.asarray(arrays["action_support"])
    bundle = np.asarray(bundle_frame_zero)
    point_count = len(bundle)
    _require(
        frame_zero.dtype == bundle.dtype == np.dtype(np.float32)
        and frame_zero.shape == bundle.shape == (point_count, 3)
        and np.array_equal(frame_zero, bundle),
        "physical material identities differ from frame-zero bundle",
    )
    trajectory_shape = (FRAME_COUNT, point_count, 3)
    _require(
        all(
            value.dtype == np.dtype(np.float32)
            and value.shape == trajectory_shape
            and np.all(np.isfinite(value))
            for value in (prior, persistence, driven, zero_action)
        )
        and support.dtype == np.dtype(np.float32)
        and support.shape == (point_count,)
        and np.all(np.isfinite(support))
        and np.all((0.0 <= support) & (support <= 1.0)),
        "physical prediction arrays are invalid",
    )
    _require(
        np.array_equal(prior[0], frame_zero)
        and np.array_equal(persistence[0], frame_zero)
        and np.array_equal(
            persistence,
            np.repeat(frame_zero[None], FRAME_COUNT, axis=0),
        ),
        "physical frame zero or persistence contract changed",
    )
    return prior.copy(), persistence.copy(), frame_zero.copy()


def _pixel_sigma(median_reprojection_px: float, floor_px: float) -> float:
    return max(
        float(floor_px),
        float(median_reprojection_px) / np.sqrt(2.0 * np.log(2.0)),
    )


def _archive_record(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        **_bound_file(path),
        "array_sha256": {
            name: _sha256_array(np.asarray(value)) for name, value in arrays.items()
        },
    }


def logical_state_action_alignment(
    source_frame_start: int,
    logical_frame: int,
) -> dict[str, Any]:
    """Return the frozen state/action indices for one logical RGB frame.

    Selected robot rows are state-aligned controller poses.  Thus state ``u``
    is raw video/robot frame ``start+u`` and its incoming displacement uses
    selected action rows ``u-1 -> u`` (never ``u -> u+1``).
    """

    _require(source_frame_start >= 0, "negative source frame start")
    _require(0 <= logical_frame < FRAME_COUNT, "logical frame is out of range")
    return {
        "logical_state_frame": int(logical_frame),
        "source_state_frame": int(source_frame_start + logical_frame),
        "selected_action_state_index": int(logical_frame),
        "incoming_selected_action_transition": (
            None if logical_frame == 0 else [logical_frame - 1, logical_frame]
        ),
        "incoming_source_action_transition": (
            None
            if logical_frame == 0
            else [
                source_frame_start + logical_frame - 1,
                source_frame_start + logical_frame,
            ]
        ),
    }


def _validate_selected_action_bundle(
    action_alignment: Mapping[str, Any],
    *,
    source_frame_start: int,
) -> tuple[Path, dict[str, Any]]:
    record = action_alignment.get("selected_action_bundle")
    _require(isinstance(record, Mapping), "selected action bundle is missing")
    path = reject_forbidden_prefix_input(
        str(record.get("path")), purpose="selected action bundle"
    )
    _require(path.suffix.lower() == ".npz", "selected action bundle is not NPZ")
    observed = _bound_file(path)
    _require(
        all(
            observed[key] == record.get(key) for key in ("path", "sha256", "size_bytes")
        ),
        "selected action bundle binding changed",
    )
    expected_arrays = action_alignment.get("selected_action_arrays")
    _require(
        isinstance(expected_arrays, Mapping) and bool(expected_arrays),
        "selected action array bindings are missing",
    )
    with np.load(path, allow_pickle=False) as stored:
        _require(
            set(stored.files) == set(expected_arrays),
            "selected action array set changed",
        )
        for name in stored.files:
            value = np.asarray(stored[name])
            array_record = expected_arrays[name]
            _require(
                isinstance(array_record, Mapping)
                and array_record.get("shape") == list(value.shape)
                and array_record.get("dtype") == value.dtype.str
                and array_record.get("sha256") == _sha256_array(value),
                f"selected action array binding changed: {name}",
            )
        _require(
            "actions" in stored.files and "openings" in stored.files,
            "selected action fields are missing",
        )
        actions = np.asarray(stored["actions"])
        openings = np.asarray(stored["openings"])
        _require(
            actions.ndim in (3, 4)
            and len(actions) == FRAME_COUNT
            and actions.shape[-1] == 3
            and np.all(np.isfinite(actions)),
            "selected actions are invalid",
        )
        _require(
            openings.ndim in (1, 2)
            and len(openings) == FRAME_COUNT
            and np.all(np.isfinite(openings)),
            "selected openings are invalid",
        )
    source_count = action_alignment.get("source_robot_frame_count")
    _require(
        isinstance(source_count, int)
        and source_count >= source_frame_start + FRAME_COUNT,
        "source robot frame count is incompatible with the selected action",
    )
    _require(
        action_alignment.get("prediction_frame_count") == FRAME_COUNT,
        "selected action prediction frame count changed",
    )
    contract = {
        "dataset_layout": "deform360.processing/robot/v1 aligned raw index",
        "shared_state_index": (
            "raw frame k indexes the same aligned state in robot arrays and every "
            "camera undistorted.mp4"
        ),
        "selected_action_semantics": "state-aligned controller pose",
        "transition_semantics": (
            "logical state u uses raw state start+u; incoming displacement uses "
            "selected rows u-1 -> u"
        ),
        "source_robot_frame_count": source_count,
        "frame_zero": logical_state_action_alignment(source_frame_start, 0),
        "updates": [
            logical_state_action_alignment(source_frame_start, update)
            for update in UPDATE_FRAMES
        ],
    }
    return path, contract


def _validate_aligned_robot_video_metadata(
    frame_manifest: Mapping[str, Any],
    cameras: Sequence[str],
    *,
    source_frame_start: int,
    source_frame_stop: int,
) -> dict[str, Any]:
    record = frame_manifest.get("action_inputs", {}).get("robot_metadata")
    _require(isinstance(record, Mapping), "bound robot metadata is missing")
    path = reject_forbidden_prefix_input(
        str(record.get("path")), purpose="aligned robot/video metadata"
    )
    observed = _bound_file(path)
    _require(
        all(
            observed[key] == record.get(key) for key in ("path", "sha256", "size_bytes")
        ),
        "aligned robot/video metadata binding changed",
    )
    metadata = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(metadata, Mapping), "robot metadata is not a JSON object")
    _require(
        metadata.get("schema") == "deform360.processing/robot/v1",
        "robot/video alignment schema changed",
    )
    outputs = metadata.get("outputs")
    inputs = metadata.get("inputs")
    parameters = metadata.get("parameters")
    _require(
        isinstance(outputs, Mapping)
        and isinstance(inputs, Mapping)
        and isinstance(parameters, Mapping),
        "robot/video alignment metadata is incomplete",
    )
    frame_count = outputs.get("num_frames")
    _require(
        isinstance(frame_count, int)
        and 0 <= source_frame_start < source_frame_stop <= frame_count,
        "selected source window is outside aligned robot/video metadata",
    )
    camera_names = tuple(str(value) for value in cameras)
    declared_cameras = parameters.get("cameras")
    video_sha256 = inputs.get("video_sha256")
    _require(
        isinstance(declared_cameras, list)
        and tuple(str(value) for value in declared_cameras) == camera_names
        and isinstance(video_sha256, Mapping)
        and set(map(str, video_sha256)) == set(camera_names),
        "robot/video aligned camera set changed",
    )
    declared_hashes = {str(name): str(value) for name, value in video_sha256.items()}
    timestamp_sha256 = inputs.get("aligned_timestamps_sha256")
    _require(
        isinstance(timestamp_sha256, str)
        and len(timestamp_sha256) == 64
        and all(digit in "0123456789abcdef" for digit in timestamp_sha256)
        and all(
            isinstance(video_sha256[name], str)
            and len(video_sha256[name]) == 64
            and all(digit in "0123456789abcdef" for digit in video_sha256[name])
            for name in video_sha256
        )
        and len(declared_hashes) == len(camera_names),
        "robot/video declared checksum is invalid",
    )
    accesses = frame_manifest.get("camera_frame_zero_access")
    _require(
        isinstance(accesses, list) and len(accesses) == len(camera_names),
        "frame-zero camera access records changed",
    )
    access_by_camera = {
        str(value.get("camera")): value
        for value in accesses
        if isinstance(value, Mapping)
    }
    _require(
        set(access_by_camera) == set(camera_names)
        and all(
            access_by_camera[camera].get("source_aligned_frame_index")
            == source_frame_start
            for camera in camera_names
        ),
        "frame-zero camera source index differs from the action start",
    )
    return {
        "metadata": observed,
        "schema": metadata["schema"],
        "aligned_frame_count": frame_count,
        "aligned_timestamps_sha256": timestamp_sha256,
        "declared_video_sha256": declared_hashes,
        "declared_video_hashes_verified_by_whole_file_read": False,
        "camera_order": list(camera_names),
        "source_frame_range_half_open": [source_frame_start, source_frame_stop],
        "frame_zero_source_index_verified_for_every_camera": True,
    }


def _require_nested_prefix_hashes(
    previous: Sequence[str],
    current: Sequence[str],
    *,
    camera: str,
    update: int,
) -> tuple[str, ...]:
    current_tuple = tuple(str(value) for value in current)
    _require(
        len(current_tuple) == update + 1
        and all(
            len(value) == 64 and all(digit in "0123456789abcdef" for digit in value)
            for value in current_tuple
        ),
        f"invalid per-frame prefix hashes for {camera}",
    )
    _require(
        current_tuple[: len(previous)] == tuple(previous),
        f"nested causal prefixes differ for {camera}",
    )
    return current_tuple


def _canonical_json_utf8(value: Any) -> np.ndarray:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return np.frombuffer(encoded, dtype=np.uint8).copy()


def run_held_online_prefix_case(
    lock_path: str | Path,
    frame_zero_manifest_path: str | Path,
    physical_prior_seal_path: str | Path,
    prefix_authorization_path: str | Path,
    aligned_episode_dir: str | Path,
    output_dir: str | Path,
    runtime: AllTrackerPrefixRuntime,
    *,
    case_name: str,
    role: str = "calibration",
    observation_config: RawCameraObservationConfig | None = None,
    uncertainty_config: RawCameraUncertaintyConfig | None = None,
) -> dict[str, Any]:
    """Build all target-free held prefix artifacts and the online seal."""

    cfg = observation_config or runtime.config
    uncertainty_cfg = uncertainty_config or HELD_UNCERTAINTY_CONFIG
    uncertainty_cfg.validate()
    _require(cfg == HELD_OBSERVATION_CONFIG, "held observation configuration changed")
    _require(
        runtime.config == HELD_OBSERVATION_CONFIG,
        "AllTracker runtime configuration changed",
    )
    _require(
        uncertainty_cfg == HELD_UNCERTAINTY_CONFIG,
        "held uncertainty configuration changed",
    )
    _require(
        tuple(cfg.update_frames) == tuple(UPDATE_FRAMES), "held update frames changed"
    )
    _require(
        runtime.source_sha256 == ALLTRACKER_RUNTIME_SOURCE_SHA256,
        "tracker source changed",
    )
    _require(
        runtime.checkpoint_sha256 == ALLTRACKER_CHECKPOINT_SHA256,
        "tracker checkpoint changed",
    )
    authorization = validate_prefix_stage_authorization(
        prefix_authorization_path, lock_path
    )
    _require(
        authorization["case_name"] == case_name, "prefix authorization case changed"
    )
    _require(authorization["role"] == role, "prefix authorization role changed")
    physical = validate_physical_prior_seal(
        physical_prior_seal_path,
        lock_path,
        expected_case_name=case_name,
        expected_role=role,
    )
    _require(
        Path(authorization["physical_prior_seal"]["path"]).resolve()
        == Path(physical_prior_seal_path).resolve(),
        "prefix authorization binds another physical seal",
    )
    _require(
        Path(physical["frame_zero_manifest"]["path"]).resolve()
        == Path(frame_zero_manifest_path).resolve(),
        "physical seal binds another frame-zero manifest",
    )
    frame_manifest, frame_arrays = _load_frame_zero_arrays(
        frame_zero_manifest_path,
        lock_path,
        case_name=case_name,
        role=role,
    )
    episode = reject_forbidden_prefix_input(
        aligned_episode_dir, purpose="aligned RGB episode"
    )
    _require(episode.is_dir(), "aligned RGB episode is missing")
    _require(
        episode.name == f"episode_{int(authorization['episode_id']):04d}"
        and episode.parent.name == authorization["object_id"],
        "aligned RGB episode identity differs from authorization",
    )
    action_alignment = frame_manifest.get("action_alignment", {})
    raw_range = action_alignment.get("selected_raw_frame_range_half_open")
    prediction_range = action_alignment.get("prediction_raw_frame_range_half_open")
    _require(
        isinstance(raw_range, list)
        and len(raw_range) == 2
        and int(raw_range[1]) - int(raw_range[0]) == 81,
        "held action alignment is not 81 frames",
    )
    source_start = int(raw_range[0])
    _require(
        isinstance(prediction_range, list)
        and prediction_range == [source_start, source_start + FRAME_COUNT],
        "held prediction action range changed",
    )
    selected_action_path, state_action_contract = _validate_selected_action_bundle(
        action_alignment,
        source_frame_start=source_start,
    )

    prediction_archive = Path(
        physical["physical_artifacts"]["physical_prediction_archive"]["path"]
    )
    reject_forbidden_prefix_input(prediction_archive, purpose="physical prediction")
    prediction_manifest_path = Path(
        physical["physical_artifacts"]["physical_prediction_manifest"]["path"]
    )
    prediction_manifest = validate_physical_prediction_manifest(
        prediction_manifest_path,
        verify_archive=True,
    )
    physical_archive_record = physical["physical_artifacts"][
        "physical_prediction_archive"
    ]
    manifest_archive_record = prediction_manifest.get("physical_prediction_archive", {})
    _require(
        Path(str(manifest_archive_record.get("path"))).resolve()
        == prediction_archive.resolve()
        and all(
            manifest_archive_record.get(key) == physical_archive_record.get(key)
            for key in ("path", "sha256", "size_bytes")
        ),
        "physical manifest binds another prediction archive",
    )
    _require(
        prediction_manifest.get("case_name") == case_name
        and prediction_manifest.get("role") == role
        and prediction_manifest.get("held_lock_sha256") == _sha256_file(lock_path)
        and prediction_manifest.get("frame_zero_manifest_sha256")
        == _sha256_file(frame_zero_manifest_path)
        and prediction_manifest.get("frame_zero_manifest_artifact_sha256")
        == frame_manifest["artifact_sha256"],
        "physical prediction manifest identity changed",
    )
    expected_physical_hashes = manifest_archive_record.get("array_sha256")
    _require(
        isinstance(expected_physical_hashes, Mapping)
        and set(expected_physical_hashes) == set(_EXPECTED_PHYSICAL_ARRAYS),
        "physical prediction archive array contract changed",
    )
    with np.load(prediction_archive, allow_pickle=False) as stored:
        _require(
            set(stored.files) == set(_EXPECTED_PHYSICAL_ARRAYS),
            "physical prediction archive array set changed",
        )
        for name in stored.files:
            _require(
                _sha256_array(np.asarray(stored[name]))
                == expected_physical_hashes[name],
                f"physical prediction array changed: {name}",
            )
        physical_arrays = {
            name: np.asarray(stored[name]).copy()
            for name in sorted(_EXPECTED_PHYSICAL_ARRAYS)
        }
    bundle_frame_zero = np.asarray(frame_arrays["object_points_world_m"])
    prior, persistence, physical_frame_zero = _validate_physical_archive_arrays(
        physical_arrays,
        bundle_frame_zero,
    )
    _require(
        prediction_manifest.get("frozen_predictor", {}).get("point_count")
        == len(physical_frame_zero),
        "physical prediction point count changed",
    )
    cameras, support, projected = _frame_zero_support_from_bundle(
        physical_frame_zero,
        frame_arrays,
        depth_tolerance_m=cfg.frame_zero_depth_tolerance_m,
    )
    aligned_metadata = _validate_aligned_robot_video_metadata(
        frame_manifest,
        cameras,
        source_frame_start=source_start,
        source_frame_stop=int(raw_range[1]),
    )
    _require(
        aligned_metadata["aligned_frame_count"]
        == state_action_contract["source_robot_frame_count"],
        "selected action and aligned video frame counts differ",
    )
    intrinsic_by_camera = {
        camera: frame_arrays["intrinsics"][index]
        for index, camera in enumerate(cameras)
    }
    pose_by_camera = {
        camera: frame_arrays["camera_to_world"][index]
        for index, camera in enumerate(cameras)
    }
    rgb_zero_by_camera = {
        camera: frame_arrays["rgb_frame0"][index]
        for index, camera in enumerate(cameras)
    }
    plan = select_frame_zero_observation_plan(
        physical_frame_zero,
        cameras,
        support,
        projected,
        pose_by_camera,
        config=cfg,
    )
    centers = np.asarray(plan["center_ids"], dtype=np.int64)
    candidates = np.asarray(plan["candidate_ids"], dtype=np.int64)
    selected_cameras = tuple(plan["selected_cameras"])
    projection_matrices = {
        camera: _projection_matrix(intrinsic_by_camera[camera], pose_by_camera[camera])
        for camera in selected_cameras
    }
    camera_origins = {
        camera: np.asarray(pose_by_camera[camera], dtype=float)[:3, 3]
        for camera in selected_cameras
    }

    # Reserve the case output atomically before expensive GPU inference. A
    # failed attempt therefore leaves an obviously partial directory without
    # a seal, and no later invocation can silently mix or overwrite artifacts.
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)

    measurement = np.full(prior.shape, np.nan, dtype=np.float32)
    measurement_visibility = np.zeros(prior.shape[:2], dtype=bool)
    measurement_validity = np.zeros(prior.shape[:2], dtype=bool)
    measurement[0, candidates] = physical_frame_zero[candidates]
    measurement_visibility[0, candidates] = True
    measurement_validity[0, candidates] = True
    update_count = len(UPDATE_FRAMES)
    inlier_count = np.zeros((update_count, len(centers)), dtype=np.int16)
    reprojection_median = np.full(
        (update_count, len(centers)), np.nan, dtype=np.float32
    )
    ray_angle = np.full_like(reprojection_median, np.nan)
    tracker_view_count = np.zeros_like(inlier_count)

    covariance_shape = prior.shape[:2] + (3, 3)
    covariance = np.full(covariance_shape, np.nan, dtype=np.float32)
    covariance_valid = np.zeros(prior.shape[:2], dtype=bool)
    jacobian_covariance = np.full_like(covariance, np.nan)
    jackknife_covariance = np.full_like(covariance, np.nan)
    pixel_sigma = np.full(prior.shape[:2], np.nan, dtype=np.float32)
    principal_std = np.full(prior.shape[:2] + (3,), np.nan, dtype=np.float32)
    loo_count = np.zeros(prior.shape[:2], dtype=np.int16)

    cycle_covariance = np.full_like(covariance, np.nan)
    cycle_covariance_valid = np.zeros(prior.shape[:2], dtype=bool)
    cycle_error_median = np.full(prior.shape[:2], np.nan, dtype=np.float32)
    cycle_error_maximum = np.full(prior.shape[:2], np.nan, dtype=np.float32)
    cycle_view_count = np.zeros(prior.shape[:2], dtype=np.int16)
    cycle_sigma = np.full(prior.shape[:2], np.nan, dtype=np.float32)
    cycle_jacobian_scale = np.full(prior.shape[:2], np.nan, dtype=np.float32)

    measurement_updates: list[dict[str, Any]] = []
    uncertainty_updates: list[dict[str, Any]] = []
    cycle_updates: list[dict[str, Any]] = []
    selected_camera_inputs: dict[str, dict[str, Any]] = {
        camera: {
            "video_path": str((episode / camera / "undistorted.mp4").resolve()),
            "source_frame_start": source_start,
            "decoded_prefix_sha256_by_update": {},
            "decoded_frame_sha256_by_update": {},
            "nested_prefix_overlap_verified": False,
            "whole_video_hashed_or_read": False,
            "frame_zero_source": "sealed frame_zero_bundle.npz",
        }
        for camera in selected_cameras
    }
    prefix_frame_hashes: dict[str, tuple[str, ...]] = {
        camera: () for camera in selected_cameras
    }
    started = time.perf_counter()
    for update_index, update in enumerate(UPDATE_FRAMES):
        tracks_by_camera: dict[str, dict[int, np.ndarray]] = {}
        cycle_errors_by_camera: dict[str, dict[int, float]] = {}
        camera_records: list[dict[str, Any]] = []
        for camera in selected_cameras:
            query_ids = np.asarray(plan["query_ids"][camera], dtype=np.int64)
            query_pixels = np.asarray(plan["query_pixels"][camera], dtype=float)
            video_path = episode / camera / "undistorted.mp4"
            rgb_prefix, access = _decode_action_aligned_prefix(
                video_path,
                source_frame_start=source_start,
                logical_update_frame=update,
                expected_frame_zero=rgb_zero_by_camera[camera],
            )
            _require(
                access.get("logical_prefix_frame_range_half_open") == [0, update + 1]
                and access.get("source_prefix_frame_range_half_open")
                == [source_start, source_start + update + 1]
                and access.get("maximum_logical_rgb_frame_read") == update
                and access.get("maximum_source_rgb_frame_read") == source_start + update
                and access.get("logical_frame_zero_matches_bundle") is True
                and access.get("whole_video_hashed_or_read") is False,
                f"causal RGB access contract changed for {camera}",
            )
            prefix_frame_hashes[camera] = _require_nested_prefix_hashes(
                prefix_frame_hashes[camera],
                access.get("decoded_rgb_frame_sha256", ()),
                camera=camera,
                update=update,
            )
            endpoints, forward_visible, forward = _infer_tracks_from_rgb(
                runtime,
                rgb_prefix,
                query_pixels,
                logical_update_frame=update,
                reverse=False,
            )
            recovered, reverse_visible, reverse = _infer_tracks_from_rgb(
                runtime,
                rgb_prefix,
                endpoints,
                logical_update_frame=update,
                reverse=True,
            )
            tracks_by_camera[camera] = {
                int(point_id): endpoints[index]
                for index, point_id in enumerate(query_ids)
                if forward_visible[index]
            }
            cycle_valid = forward_visible & reverse_visible
            errors = np.linalg.norm(recovered - query_pixels, axis=1)
            cycle_errors_by_camera[camera] = {
                int(point_id): float(errors[index])
                for index, point_id in enumerate(query_ids)
                if cycle_valid[index]
            }
            selected_camera_inputs[camera]["decoded_prefix_sha256_by_update"][
                str(update)
            ] = access["decoded_rgb_prefix_sha256"]
            selected_camera_inputs[camera]["decoded_frame_sha256_by_update"][
                str(update)
            ] = list(prefix_frame_hashes[camera])
            camera_records.append(
                {
                    "camera": camera,
                    "query_ids": query_ids.tolist(),
                    "access": access,
                    "forward": forward,
                    "reverse": reverse,
                    "cycle_valid_count": int(np.sum(cycle_valid)),
                    "cycle_error_median_px": (
                        None
                        if not np.any(cycle_valid)
                        else float(np.median(errors[cycle_valid]))
                    ),
                }
            )

        for camera in selected_cameras:
            selected_camera_inputs[camera]["nested_prefix_overlap_verified"] = True

        measurement_centers: list[dict[str, Any]] = []
        uncertainty_centers: list[dict[str, Any]] = []
        cycle_centers: list[dict[str, Any]] = []
        for center_index, center_value in enumerate(centers):
            center = int(center_value)
            observations = {
                camera: tracks_by_camera[camera][center]
                for camera in selected_cameras
                if center in tracks_by_camera[camera]
            }
            tracker_view_count[update_index, center_index] = len(observations)
            point, diagnostic = triangulate_observation_ransac(
                observations,
                projection_matrices,
                camera_origins,
                physical_frame_zero[center],
                config=cfg,
            )
            diagnostic["center_id"] = center
            measurement_centers.append(diagnostic)
            uncertainty_record: dict[str, Any] = {
                "center_id": center,
                "source_measurement_accepted": point is not None,
                "covariance_valid": False,
                "decision": "source_measurement_rejected",
            }
            cycle_record: dict[str, Any] = dict(uncertainty_record)
            if point is None:
                uncertainty_centers.append(uncertainty_record)
                cycle_centers.append(cycle_record)
                continue
            measurement[update, center] = point
            measurement_visibility[update, center] = True
            measurement_validity[update, center] = True
            inlier_count[update_index, center_index] = int(
                diagnostic["inlier_view_count"]
            )
            reprojection_median[update_index, center_index] = float(
                diagnostic["median_reprojection_error_px"]
            )
            ray_angle[update_index, center_index] = float(
                diagnostic["maximum_ray_angle_degrees"]
            )
            inlier_cameras = tuple(str(value) for value in diagnostic["inlier_cameras"])
            inlier_observations = {
                camera: observations[camera] for camera in inlier_cameras
            }
            sigma = _pixel_sigma(
                float(diagnostic["median_reprojection_error_px"]),
                uncertainty_cfg.pixel_noise_floor_px,
            )
            geometric, geometric_diagnostic = jacobian_measurement_covariance(
                point,
                [projection_matrices[camera] for camera in sorted(inlier_cameras)],
                sigma,
                maximum_condition_number=(
                    uncertainty_cfg.maximum_information_condition_number
                ),
            )
            uncertainty_record["jacobian"] = geometric_diagnostic
            if geometric is None:
                uncertainty_record["decision"] = geometric_diagnostic["decision"]
                uncertainty_centers.append(uncertainty_record)
                cycle_centers.append(cycle_record)
                continue
            empirical, loo = leave_one_camera_out_covariance(
                inlier_observations, projection_matrices
            )
            combined = 0.5 * (geometric + empirical + (geometric + empirical).T)
            eigenvalues = np.linalg.eigvalsh(combined)
            if eigenvalues[0] <= 0.0 or not np.all(np.isfinite(eigenvalues)):
                uncertainty_record["decision"] = "combined_covariance_failure"
                uncertainty_centers.append(uncertainty_record)
                cycle_centers.append(cycle_record)
                continue
            covariance[update, center] = combined
            covariance_valid[update, center] = True
            jacobian_covariance[update, center] = geometric
            jackknife_covariance[update, center] = empirical
            pixel_sigma[update, center] = sigma
            principal_std[update, center] = np.sqrt(eigenvalues)
            loo_count[update, center] = len(loo)
            uncertainty_record.update(
                {
                    "covariance_valid": True,
                    "decision": "accepted",
                    "pixel_sigma": sigma,
                    "leave_one_out_sample_count": len(loo),
                    "principal_standard_deviation_m": np.sqrt(eigenvalues).tolist(),
                }
            )
            uncertainty_centers.append(uncertainty_record)

            cycle_errors = np.asarray(
                [
                    cycle_errors_by_camera[camera][center]
                    for camera in inlier_cameras
                    if center in cycle_errors_by_camera[camera]
                ],
                dtype=float,
            )
            cycle_view_count[update, center] = len(cycle_errors)
            cycle_record["source_inlier_view_count"] = len(inlier_cameras)
            cycle_record["cycle_valid_view_count"] = len(cycle_errors)
            if len(cycle_errors) < 2:
                cycle_record["decision"] = "insufficient_cycle_views"
                cycle_centers.append(cycle_record)
                continue
            try:
                inflated, cycle_diagnostic = inflate_covariance_from_cycle(
                    geometric,
                    empirical,
                    sigma,
                    cycle_errors,
                    pixel_noise_floor_px=uncertainty_cfg.pixel_noise_floor_px,
                )
            except ValueError as error:
                cycle_record["decision"] = "cycle_covariance_failure"
                cycle_record["error"] = str(error)
                cycle_centers.append(cycle_record)
                continue
            cycle_covariance[update, center] = inflated
            cycle_covariance_valid[update, center] = True
            cycle_error_median[update, center] = cycle_diagnostic[
                "cycle_error_median_px"
            ]
            cycle_error_maximum[update, center] = cycle_diagnostic[
                "cycle_error_maximum_px"
            ]
            cycle_sigma[update, center] = cycle_diagnostic["cycle_pixel_sigma"]
            cycle_jacobian_scale[update, center] = cycle_diagnostic[
                "jacobian_covariance_scale"
            ]
            cycle_record.update(
                {
                    "covariance_valid": True,
                    "decision": "accepted",
                    **cycle_diagnostic,
                }
            )
            cycle_centers.append(cycle_record)

        measurement_updates.append(
            {
                "logical_frame": int(update),
                "source_frame": source_start + int(update),
                "logical_prefix_frame_range_half_open": [0, int(update) + 1],
                "source_prefix_frame_range_half_open": [
                    source_start,
                    source_start + int(update) + 1,
                ],
                "tracker": camera_records,
                "centers": measurement_centers,
                "accepted_center_count": int(
                    np.sum(measurement_validity[update, centers])
                ),
            }
        )
        uncertainty_updates.append(
            {
                "logical_frame": int(update),
                "centers": uncertainty_centers,
                "valid_covariance_count": int(np.sum(covariance_valid[update])),
            }
        )
        cycle_updates.append(
            {
                "logical_frame": int(update),
                "centers": cycle_centers,
                "valid_covariance_count": int(np.sum(cycle_covariance_valid[update])),
            }
        )

    online_prediction, selected_raw, prediction_diagnostic = (
        predict_support_gated_selected_backbone_rbf(
            prior,
            persistence,
            measurement,
            measurement_validity,
            center_ids=centers,
            rbf_config=HELD_RBF_CONFIG,
        )
    )
    measurement_arrays = {
        "measurement_m": measurement,
        "measurement_visibility": measurement_visibility,
        "measurement_validity": measurement_validity,
        "candidate_ids": candidates,
        "center_ids": centers,
        "selected_cameras": np.asarray(selected_cameras),
        "update_frames": np.asarray(UPDATE_FRAMES, dtype=np.int64),
        "tracker_visible_view_count": tracker_view_count,
        "triangulation_inlier_view_count": inlier_count,
        "triangulation_median_reprojection_px": reprojection_median,
        "triangulation_maximum_ray_angle_degrees": ray_angle,
    }
    measurement_path = output / MEASUREMENT_ARCHIVE_FILENAME
    np.savez_compressed(measurement_path, **measurement_arrays)
    common_inputs = {
        "lock": _bound_file(lock_path),
        "frame_zero_manifest": _bound_file(frame_zero_manifest_path),
        "physical_prior_seal": _bound_file(physical_prior_seal_path),
        "prefix_authorization": _bound_file(prefix_authorization_path),
        "physical_prediction_archive": _bound_file(prediction_archive),
        "physical_prediction_manifest": _bound_file(prediction_manifest_path),
        "selected_action_bundle": _bound_file(selected_action_path),
        "robot_video_metadata": aligned_metadata["metadata"],
    }
    tracker_descriptor = {
        "name": "AllTracker",
        "molmomotion_revision": ALLTRACKER_MOLMOMOTION_REVISION,
        "source_tree": ALLTRACKER_SOURCE_TREE,
        "runtime_source_sha256": runtime.source_sha256,
        "checkpoint_sha256": runtime.checkpoint_sha256,
        "device": runtime.device_name,
        "query_routing": {
            "forward": (
                "chronological logical [0,u], with sealed frame-zero query pixels"
            ),
            "cycle_reverse": (
                "reversed logical [u,0], with forward endpoint query pixels"
            ),
            "legacy_open27_routing_reused": False,
            "reason": (
                "AllTracker queries belong to the first frame supplied to the model; "
                "the held path corrects the legacy open27 reverse-prefix routing"
            ),
            "equivalence_claim_scope": "target-free primary predictor arrays only",
        },
    }
    measurement_manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": MEASUREMENT_KIND,
        "protocol_id": PROTOCOL_ID,
        "case_name": case_name,
        "object_id": authorization["object_id"],
        "episode_id": authorization["episode_id"],
        "role": role,
        "config": asdict(cfg),
        "plan": {
            "candidate_ids": candidates.tolist(),
            "center_ids": centers.tolist(),
            "selected_cameras": list(selected_cameras),
            "selection_score": list(plan["selection_score"]),
            "selection_inputs": (
                "sealed frame-zero points/masks/depth/calibration from NPZ only"
            ),
        },
        "tracker": tracker_descriptor,
        "action_alignment": {
            "source_frame_start": source_start,
            "selected_raw_frame_range_half_open": raw_range,
            "prediction_raw_frame_range_half_open": prediction_range,
            "logical_update_frames": list(UPDATE_FRAMES),
            "maximum_logical_rgb_frame_read": UPDATE_FRAMES[-1],
            "maximum_source_rgb_frame_read": source_start + UPDATE_FRAMES[-1],
            "state_action_index_contract": state_action_contract,
            "aligned_robot_video_metadata": aligned_metadata,
        },
        "inputs": common_inputs,
        "selected_camera_inputs": selected_camera_inputs,
        "updates": measurement_updates,
        "output": _archive_record(measurement_path, measurement_arrays),
        "information_boundary": {
            "frame_zero_bundle_only_for_mask_depth_calibration": True,
            "processed_hdf5_read": False,
            "original_calibration_read_after_bundle": False,
            "logical_video_prefix_rule": "update u reads exactly logical frames [0,u]",
            "source_video_prefix_rule": "logical [0,u] maps to raw [start,start+u]",
            "nested_prefix_overlap_verified_per_camera": True,
            "whole_video_hashed_or_read": False,
            "maximum_logical_rgb_frame_read": UPDATE_FRAMES[-1],
            "future_tactile_read": False,
            "target_data_read": False,
            "target_visibility_read": False,
            "outcome_created": False,
            "outcome_read": False,
        },
    }
    _write_manifest(output / MEASUREMENT_MANIFEST_FILENAME, measurement_manifest)

    uncertainty_arrays = {
        "measurement_covariance_m2": covariance,
        "measurement_covariance_valid": covariance_valid,
        "jacobian_covariance_m2": jacobian_covariance,
        "jackknife_covariance_m2": jackknife_covariance,
        "pixel_sigma": pixel_sigma,
        "principal_standard_deviation_m": principal_std,
        "leave_one_out_sample_count": loo_count,
    }
    uncertainty_path = output / UNCERTAINTY_ARCHIVE_FILENAME
    np.savez_compressed(uncertainty_path, **uncertainty_arrays)
    uncertainty_manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": UNCERTAINTY_KIND,
        "protocol_id": PROTOCOL_ID,
        "case_name": case_name,
        "object_id": authorization["object_id"],
        "episode_id": authorization["episode_id"],
        "role": role,
        "config": asdict(uncertainty_cfg),
        "inputs": {
            **common_inputs,
            "measurement_archive": _bound_file(measurement_path),
            "measurement_manifest": _bound_file(output / MEASUREMENT_MANIFEST_FILENAME),
        },
        "updates": uncertainty_updates,
        "output": _archive_record(uncertainty_path, uncertainty_arrays),
        "information_boundary": {
            "causal_measurements_only": True,
            "processed_hdf5_read": False,
            "target_data_read": False,
            "target_visibility_read": False,
            "outcome_created": False,
            "outcome_read": False,
        },
    }
    _write_manifest(output / UNCERTAINTY_MANIFEST_FILENAME, uncertainty_manifest)

    cycle_arrays = {
        "measurement_covariance_m2": cycle_covariance,
        "measurement_covariance_valid": cycle_covariance_valid,
        "cycle_error_median_px": cycle_error_median,
        "cycle_error_maximum_px": cycle_error_maximum,
        "cycle_valid_view_count": cycle_view_count,
        "cycle_pixel_sigma": cycle_sigma,
        "jacobian_covariance_scale": cycle_jacobian_scale,
    }
    cycle_path = output / CYCLE_ARCHIVE_FILENAME
    np.savez_compressed(cycle_path, **cycle_arrays)
    cycle_manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": CYCLE_KIND,
        "protocol_id": PROTOCOL_ID,
        "case_name": case_name,
        "object_id": authorization["object_id"],
        "episode_id": authorization["episode_id"],
        "role": role,
        "inputs": {
            **common_inputs,
            "measurement_archive": _bound_file(measurement_path),
            "measurement_manifest": _bound_file(output / MEASUREMENT_MANIFEST_FILENAME),
            "measurement_uncertainty_archive": _bound_file(uncertainty_path),
            "measurement_uncertainty_manifest": _bound_file(
                output / UNCERTAINTY_MANIFEST_FILENAME
            ),
        },
        "updates": cycle_updates,
        "output": _archive_record(cycle_path, cycle_arrays),
        "information_boundary": {
            "forward_reverse_exact_causal_prefixes": True,
            "maximum_logical_rgb_frame_read": UPDATE_FRAMES[-1],
            "processed_hdf5_read": False,
            "target_data_read": False,
            "target_visibility_read": False,
            "outcome_created": False,
            "outcome_read": False,
        },
    }
    _write_manifest(output / CYCLE_MANIFEST_FILENAME, cycle_manifest)

    online_arrays = {
        "primary_prediction_m": online_prediction,
        "selected_raw_backbone_m": selected_raw,
        "frame_zero_points_m": physical_frame_zero,
        "physical_prior_m": prior.astype(online_prediction.dtype, copy=False),
        "persistence_m": persistence.astype(online_prediction.dtype, copy=False),
        "center_ids": centers,
        "update_frames": np.asarray(UPDATE_FRAMES, dtype=np.int64),
        "available_center_count_by_update": np.asarray(
            [
                record["available_center_count"]
                for record in prediction_diagnostic["updates"]
            ],
            dtype=np.int64,
        ),
        "selected_backbone_by_update": np.asarray(
            [record["selected_backbone"] for record in prediction_diagnostic["updates"]]
        ),
        "primary_method_id": np.asarray(PRIMARY_METHOD["method_id"]),
        "prediction_diagnostic_json_utf8": _canonical_json_utf8(prediction_diagnostic),
    }
    online_path = output / ONLINE_PREDICTION_ARCHIVE_FILENAME
    np.savez_compressed(online_path, **online_arrays)
    # Bind the target-free predictor diagnostics inside the NPZ as canonical JSON bytes.
    with np.load(online_path, allow_pickle=False) as stored:
        _require(
            set(stored.files) == set(online_arrays), "online archive array set changed"
        )
    online_artifacts = {
        "measurement_archive": measurement_path,
        "measurement_manifest": output / MEASUREMENT_MANIFEST_FILENAME,
        "measurement_uncertainty_archive": uncertainty_path,
        "measurement_uncertainty_manifest": output / UNCERTAINTY_MANIFEST_FILENAME,
        "cycle_uncertainty_archive": cycle_path,
        "cycle_uncertainty_manifest": output / CYCLE_MANIFEST_FILENAME,
        "online_prediction_archive": online_path,
    }
    _require(
        set(online_artifacts) == set(ONLINE_ARTIFACT_ROLES), "online role set changed"
    )
    online_seal = create_online_prediction_seal(
        output / ONLINE_SEAL_FILENAME,
        lock_path,
        prefix_authorization_path,
        online_artifacts,
    )
    return {
        "case_name": case_name,
        "role": role,
        "measurement_manifest": measurement_manifest,
        "uncertainty_manifest": uncertainty_manifest,
        "cycle_manifest": cycle_manifest,
        "prediction_diagnostic": prediction_diagnostic,
        "online_prediction_archive": _archive_record(online_path, online_arrays),
        "online_prediction_seal": online_seal,
        "runtime_seconds": time.perf_counter() - started,
    }


__all__ = [
    "CYCLE_ARCHIVE_FILENAME",
    "CYCLE_MANIFEST_FILENAME",
    "HELD_OBSERVATION_CONFIG",
    "HELD_RBF_CONFIG",
    "HELD_UNCERTAINTY_CONFIG",
    "MEASUREMENT_ARCHIVE_FILENAME",
    "MEASUREMENT_MANIFEST_FILENAME",
    "ONLINE_PREDICTION_ARCHIVE_FILENAME",
    "ONLINE_SEAL_FILENAME",
    "UNCERTAINTY_ARCHIVE_FILENAME",
    "UNCERTAINTY_MANIFEST_FILENAME",
    "predict_support_gated_selected_backbone_rbf",
    "reject_forbidden_prefix_input",
    "run_held_online_prefix_case",
]
