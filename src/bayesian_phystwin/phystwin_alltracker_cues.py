"""Causal AllTracker identity cues for the released PhysTwin videos.

This adapter reuses the checksum-locked AllTracker runtime from the Deform360
experiments but writes a distinct PhysTwin artifact.  Only RGB frames before
``train_end_frame`` are decoded; future cue rows are neutralized.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .deform360_raw_camera_observation import (
    ALLTRACKER_CHECKPOINT_SHA256,
    ALLTRACKER_MOLMOMOTION_REVISION,
    ALLTRACKER_RUNTIME_SOURCE_SHA256,
    AllTrackerPrefixRuntime,
    RawCameraObservationConfig,
)
from .mask_distance import interior_mask_distance
from .phystwin_cotracker3_cues import (
    _distribution,
    _initial_multiview_eligibility,
    _sha256,
    pack_multiview_triangulation,
    triangulate_multiview_tracks,
)
from .phystwin_raw_cues import (
    PhysTwinRawCueConfig,
    load_phystwin_raw_track_map,
)


@dataclass(frozen=True)
class PhysTwinAllTrackerCueConfig:
    """Fixed extraction settings for one leakage-free PhysTwin prefix."""

    train_end_frame: int
    max_side: int = 512
    inference_iterations: int = 4
    window_length: int = 16
    minimum_cycle_quality: float = 0.5
    visibility_threshold: float = 0.5
    initial_match_tolerance_m: float = 1e-6

    def __post_init__(self) -> None:
        if self.train_end_frame < 2:
            raise ValueError("train_end_frame must be at least two")
        if self.max_side < 64:
            raise ValueError("max_side is implausibly small")
        if self.inference_iterations < 1:
            raise ValueError("inference_iterations must be positive")
        if self.window_length < 2:
            raise ValueError("window_length must exceed one")
        if not 0.0 <= self.minimum_cycle_quality <= 1.0:
            raise ValueError("minimum_cycle_quality must lie in [0, 1]")
        if not 0.0 < self.visibility_threshold < 1.0:
            raise ValueError("visibility_threshold must lie in (0, 1)")
        if self.initial_match_tolerance_m <= 0.0:
            raise ValueError("initial_match_tolerance_m must be positive")


@dataclass(frozen=True)
class PhysTwinAllTrackerMultiviewCueConfig:
    """Settings for an opt-in redundant-view augmentation."""

    train_end_frame: int
    max_side: int = 512
    inference_iterations: int = 4
    window_length: int = 16
    minimum_cycle_quality: float = 0.5
    visibility_threshold: float = 0.5
    initial_match_tolerance_m: float = 1e-6
    minimum_multiview_quality: float = 0.5
    maximum_cycle_error_px: float = 5.0
    multiview_initial_depth_tolerance_m: float = 0.02

    def __post_init__(self) -> None:
        self.source_config()
        if not 0.0 <= self.minimum_multiview_quality <= 1.0:
            raise ValueError("minimum_multiview_quality must lie in [0, 1]")
        if self.maximum_cycle_error_px <= 0.0:
            raise ValueError("maximum_cycle_error_px must be positive")
        if self.multiview_initial_depth_tolerance_m <= 0.0:
            raise ValueError(
                "multiview_initial_depth_tolerance_m must be positive"
            )

    def source_config(self) -> PhysTwinAllTrackerCueConfig:
        """Return the exact config required of the source-only artifact."""

        return PhysTwinAllTrackerCueConfig(
            train_end_frame=self.train_end_frame,
            max_side=self.max_side,
            inference_iterations=self.inference_iterations,
            window_length=self.window_length,
            minimum_cycle_quality=self.minimum_cycle_quality,
            visibility_threshold=self.visibility_threshold,
            initial_match_tolerance_m=self.initial_match_tolerance_m,
        )


@dataclass(frozen=True)
class AllTrackerDensePrediction:
    """Frame-zero query trajectories and visibility confidence."""

    tracks_xy: np.ndarray
    quality_probability: np.ndarray

    def __post_init__(self) -> None:
        tracks = np.asarray(self.tracks_xy, dtype=np.float32)
        quality = np.asarray(self.quality_probability, dtype=np.float32)
        if tracks.ndim != 3 or tracks.shape[2] != 2:
            raise ValueError("tracks_xy must have shape (T, N, 2)")
        if quality.shape != tracks.shape[:2]:
            raise ValueError("quality_probability must have shape (T, N)")
        if np.any(~np.isfinite(tracks)) or np.any(~np.isfinite(quality)):
            raise ValueError("AllTracker predictions must be finite")
        if np.any((quality < 0.0) | (quality > 1.0)):
            raise ValueError("quality_probability must lie in [0, 1]")


class PhysTwinAllTrackerRunner:
    """Dense trajectory decoder around the frozen AllTracker runtime."""

    def __init__(
        self,
        source_root: str | Path,
        checkpoint: str | Path,
        *,
        device: str,
        config: PhysTwinAllTrackerCueConfig,
    ) -> None:
        runtime_config = RawCameraObservationConfig(
            alltracker_max_side=config.max_side,
            alltracker_inference_iterations=config.inference_iterations,
            alltracker_window_length=config.window_length,
            visibility_threshold=config.visibility_threshold,
            update_frames=(config.train_end_frame - 1,),
        )
        self.config = config
        self._runtime = AllTrackerPrefixRuntime(
            source_root,
            checkpoint,
            device=device,
            config=runtime_config,
        )

    @property
    def source_sha256(self) -> str:
        return self._runtime.source_sha256

    @property
    def checkpoint_sha256(self) -> str:
        return self._runtime.checkpoint_sha256

    def close(self) -> None:
        self._runtime.close()

    def track(
        self,
        video_rgb: np.ndarray,
        query_pixels_xy: np.ndarray,
    ) -> AllTrackerDensePrediction:
        """Track frame-zero pixels through one exact in-memory RGB prefix."""

        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - GPU integration only
            raise RuntimeError("OpenCV is required for AllTracker input") from exc
        rgb = np.asarray(video_rgb)
        queries = np.asarray(query_pixels_xy, dtype=float)
        if rgb.ndim != 4 or rgb.shape[3] != 3 or rgb.dtype != np.uint8:
            raise ValueError("video_rgb must contain uint8 shape (T, H, W, 3)")
        if len(rgb) != self.config.train_end_frame:
            raise ValueError("video length differs from train_end_frame")
        if (
            queries.ndim != 2
            or queries.shape[1] != 2
            or not np.all(np.isfinite(queries))
        ):
            raise ValueError("query_pixels_xy must have finite shape (N, 2)")

        original_height, original_width = rgb.shape[1:3]
        scale = min(
            1.0,
            self.config.max_side / max(original_height, original_width),
        )
        height = max(8, int(original_height * scale) // 8 * 8)
        width = max(8, int(original_width * scale) // 8 * 8)
        if (height, width) != (original_height, original_width):
            rgb = np.stack(
                [
                    cv2.resize(
                        frame,
                        (width, height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    for frame in rgb
                ]
            )

        torch = self._runtime._torch
        device = self._runtime._device
        video = (
            torch.from_numpy(np.ascontiguousarray(rgb))
            .permute(0, 3, 1, 2)[None]
            .float()
            .to(device)
        )
        with torch.no_grad():
            flows, confidence, _, _ = self._runtime._model(
                video,
                iters=self.config.inference_iterations,
                sw=None,
                is_training=False,
            )
        if flows.ndim == 4:
            flows = flows[:, None]
            flows = torch.cat((torch.zeros_like(flows[:, :1]), flows), dim=1)
        if confidence.ndim == 4:
            confidence = confidence[:, None]
        if confidence.shape[1] == flows.shape[1] - 1:
            confidence = torch.cat(
                (torch.ones_like(confidence[:, :1]), confidence),
                dim=1,
            )
        if flows.shape[1] != len(rgb) or confidence.shape[1] != len(rgb):
            raise ValueError("AllTracker output does not match the exact prefix")

        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
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
        sampled = trajectories[0, :, :, y_query, x_query].permute(0, 2, 1)
        sampled[:, :, 0] *= original_width / width
        sampled[:, :, 1] *= original_height / height
        sampled_confidence = confidence[0, :, 0, y_query, x_query]
        result = AllTrackerDensePrediction(
            tracks_xy=sampled.detach().cpu().numpy().astype(np.float32),
            quality_probability=np.clip(
                sampled_confidence.detach().cpu().numpy(),
                0.0,
                1.0,
            ).astype(np.float32),
        )
        del video, flows, confidence, trajectories, sampled, sampled_confidence
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result


def _load_video_prefix(
    raw_case_dir: Path,
    camera: int,
    end_frame: int,
) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("AllTracker extraction requires Pillow") from exc
    frames = []
    for frame in range(end_frame):
        path = raw_case_dir / "color" / str(camera) / f"{frame}.png"
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            frames.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
    return np.stack(frames)


def _pixels_inside_mask(tracks_xy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    tracks = np.asarray(tracks_xy, dtype=float)
    finite = np.all(np.isfinite(tracks), axis=1)
    pixels = np.rint(np.where(finite[:, None], tracks, 0.0)).astype(np.int64)
    height, width = mask.shape
    inside = (
        finite
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    selected = np.flatnonzero(inside)
    inside[selected] &= mask[pixels[selected, 1], pixels[selected, 0]]
    return inside


def build_phystwin_alltracker_cues(
    final_data_path: str | Path,
    raw_case_dir: str | Path,
    alltracker_source: str | Path,
    checkpoint_path: str | Path,
    output_npz_path: str | Path,
    *,
    config: PhysTwinAllTrackerCueConfig,
    device: str = "cuda",
) -> dict[str, Any]:
    """Build source-camera identity cues from an exact causal RGB prefix."""

    raw_path = Path(raw_case_dir)
    output = Path(output_npz_path)
    mapping = load_phystwin_raw_track_map(
        final_data_path,
        raw_path,
        config=PhysTwinRawCueConfig(
            initial_match_tolerance_m=config.initial_match_tolerance_m
        ),
    )
    frame_count, track_count = mapping.final_visible.shape
    if config.train_end_frame >= frame_count:
        raise ValueError("train_end_frame must leave at least one future frame")
    with (raw_path / "mask" / "processed_masks.pkl").open("rb") as handle:
        processed_masks = pickle.load(handle)

    tracks = np.full((frame_count, track_count, 2), np.nan, dtype=np.float32)
    quality = np.zeros((frame_count, track_count), dtype=np.float32)
    cycle_error = np.zeros((frame_count, track_count), dtype=np.float32)
    cycle_valid = np.zeros((frame_count, track_count), dtype=bool)
    boundary = np.zeros((frame_count, track_count), dtype=np.float32)
    cue_available = np.zeros((frame_count, track_count), dtype=bool)
    cue_available[: config.train_end_frame] = True
    camera_summaries: dict[str, Any] = {}

    runner = PhysTwinAllTrackerRunner(
        alltracker_source,
        checkpoint_path,
        device=device,
        config=config,
    )
    try:
        for camera in range(len(mapping.track_paths)):
            video = _load_video_prefix(
                raw_path,
                camera,
                config.train_end_frame,
            )
            archived = mapping.tracks_by_camera[camera]
            queries = archived[0, :, ::-1].astype(np.float32)
            forward = runner.track(video, queries)
            reverse = runner.track(
                np.ascontiguousarray(video[::-1]),
                forward.tracks_xy[-1],
            )
            reverse_tracks = reverse.tracks_xy[::-1]
            reverse_quality = reverse.quality_probability[::-1]
            selected = np.flatnonzero(mapping.source_camera == camera)
            raw_ids = mapping.source_track[selected]
            selected_tracks = forward.tracks_xy[:, raw_ids]
            selected_quality = forward.quality_probability[:, raw_ids]
            selected_cycle = np.linalg.norm(
                selected_tracks - reverse_tracks[:, raw_ids],
                axis=2,
            )
            selected_cycle_valid = (
                selected_quality >= config.minimum_cycle_quality
            ) & (reverse_quality[:, raw_ids] >= config.minimum_cycle_quality)
            tracks[: config.train_end_frame, selected] = selected_tracks
            quality[: config.train_end_frame, selected] = selected_quality
            cycle_error[: config.train_end_frame, selected] = selected_cycle
            cycle_valid[: config.train_end_frame, selected] = (
                selected_cycle_valid
            )
            for frame in range(config.train_end_frame):
                object_mask = np.asarray(
                    processed_masks[frame][camera]["object"],
                    dtype=bool,
                )
                distance = interior_mask_distance(object_mask) / max(
                    object_mask.shape
                )
                frame_tracks = selected_tracks[frame]
                pixels = np.rint(frame_tracks).astype(np.int64)
                inside = _pixels_inside_mask(frame_tracks, object_mask)
                ids = np.flatnonzero(inside)
                values = np.zeros(len(selected), dtype=np.float32)
                values[ids] = distance[pixels[ids, 1], pixels[ids, 0]]
                boundary[frame, selected] = values
            camera_summaries[str(camera)] = {
                "query_count": int(len(queries)),
                "selected_source_count": int(len(selected)),
                "quality": _distribution(selected_quality),
                "cycle_error_px": _distribution(
                    selected_cycle[selected_cycle_valid]
                ),
            }
    finally:
        runner.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        source_tracks_xy=tracks,
        source_quality_probability=quality,
        forward_backward_error_px=cycle_error,
        forward_backward_valid=cycle_valid,
        boundary_distance=boundary,
        cue_available=cue_available,
        source_camera=mapping.source_camera,
        source_track=mapping.source_track,
        initial_match_distance_m=mapping.initial_match_distance_m.astype(
            np.float32
        ),
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinAllTrackerCues",
        "config": asdict(config),
        "tracker": {
            "name": "AllTracker",
            "molmomotion_revision": ALLTRACKER_MOLMOMOTION_REVISION,
            "runtime_source_sha256": runner.source_sha256,
            "expected_runtime_source_sha256": (
                ALLTRACKER_RUNTIME_SOURCE_SHA256
            ),
            "checkpoint_sha256": runner.checkpoint_sha256,
            "expected_checkpoint_sha256": ALLTRACKER_CHECKPOINT_SHA256,
            "device": device,
        },
        "inputs": {
            "final_data": {
                "path": str(Path(final_data_path).resolve()),
                "sha256": _sha256(final_data_path),
            },
            "raw_case_dir": str(raw_path.resolve()),
        },
        "camera_summaries": camera_summaries,
        "output": {
            "path": str(output.resolve()),
            "sha256": _sha256(output),
        },
        "information_boundary": {
            "rgb_frame_range_half_open": [0, config.train_end_frame],
            "future_rgb_read": False,
            "future_cue_rows_neutralized": True,
            "future_outcome_read": False,
        },
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _load_validated_source_cues(
    cues_path: Path,
    *,
    config: PhysTwinAllTrackerMultiviewCueConfig,
    frame_count: int,
    track_count: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    summary_path = cues_path.with_suffix(".summary.json")
    if not cues_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            "AllTracker multiview augmentation requires source cues and summary"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("artifact_kind") != "PhysTwinAllTrackerCues":
        raise ValueError("base artifact is not PhysTwinAllTrackerCues")
    if summary.get("config") != asdict(config.source_config()):
        raise ValueError("base AllTracker config differs from the augmentation")
    if summary.get("output", {}).get("sha256") != _sha256(cues_path):
        raise ValueError("base AllTracker cue hash differs from its summary")
    tracker = summary.get("tracker", {})
    if tracker.get("runtime_source_sha256") != ALLTRACKER_RUNTIME_SOURCE_SHA256:
        raise ValueError("base AllTracker runtime hash differs from the lock")
    if tracker.get("checkpoint_sha256") != ALLTRACKER_CHECKPOINT_SHA256:
        raise ValueError("base AllTracker checkpoint hash differs from the lock")

    with np.load(cues_path) as archive:
        cues = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "source_tracks_xy",
        "source_quality_probability",
        "forward_backward_error_px",
        "forward_backward_valid",
        "boundary_distance",
        "cue_available",
        "source_camera",
        "source_track",
        "initial_match_distance_m",
    }
    missing = required.difference(cues)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"base AllTracker cues lack fields: {names}")
    if cues["source_tracks_xy"].shape != (frame_count, track_count, 2):
        raise ValueError("base source_tracks_xy shape differs from the case")
    for name in (
        "source_quality_probability",
        "forward_backward_error_px",
        "forward_backward_valid",
        "boundary_distance",
        "cue_available",
    ):
        if cues[name].shape != (frame_count, track_count):
            raise ValueError(f"base {name} shape differs from the case")
    future = slice(config.train_end_frame, None)
    if np.any(cues["cue_available"][future]):
        raise ValueError("base AllTracker cues expose future availability")
    if np.any(np.isfinite(cues["source_tracks_xy"][future])):
        raise ValueError("base AllTracker cues expose future tracks")
    if np.any(cues["source_quality_probability"][future] != 0.0):
        raise ValueError("base AllTracker cues expose future quality")
    return cues, summary


def build_phystwin_alltracker_multiview_cues(
    final_data_path: str | Path,
    raw_case_dir: str | Path,
    alltracker_source: str | Path,
    checkpoint_path: str | Path,
    base_cues_path: str | Path,
    output_npz_path: str | Path,
    *,
    config: PhysTwinAllTrackerMultiviewCueConfig,
    device: str = "cuda",
) -> dict[str, Any]:
    """Augment exact source-only cues with calibrated cross-view tracks."""

    raw_path = Path(raw_case_dir)
    output = Path(output_npz_path)
    base_path = Path(base_cues_path)
    if output.resolve() == base_path.resolve():
        raise ValueError("multiview output must not overwrite source-only cues")
    mapping = load_phystwin_raw_track_map(
        final_data_path,
        raw_path,
        config=PhysTwinRawCueConfig(
            initial_match_tolerance_m=config.initial_match_tolerance_m
        ),
    )
    frame_count, track_count = mapping.final_visible.shape
    if config.train_end_frame >= frame_count:
        raise ValueError("train_end_frame must leave at least one future frame")
    cues, base_summary = _load_validated_source_cues(
        base_path,
        config=config,
        frame_count=frame_count,
        track_count=track_count,
    )
    if not np.array_equal(cues["source_camera"], mapping.source_camera):
        raise ValueError("base source_camera differs from reconstructed mapping")
    if not np.array_equal(cues["source_track"], mapping.source_track):
        raise ValueError("base source_track differs from reconstructed mapping")
    reserved_prefix = "multiview_"
    existing_multiview = sorted(
        name for name in cues if name.startswith(reserved_prefix)
    )
    if existing_multiview:
        raise ValueError(
            "base source cues already contain multiview fields: "
            + ", ".join(existing_multiview)
        )

    metadata = json.loads(
        (raw_path / "metadata.json").read_text(encoding="utf-8")
    )
    intrinsics = np.asarray(metadata["intrinsics"], dtype=float)
    with (raw_path / "calibrate.pkl").open("rb") as handle:
        camera_to_world = np.asarray(pickle.load(handle), dtype=float)
    camera_count = len(mapping.track_paths)
    if intrinsics.shape != (camera_count, 3, 3):
        raise ValueError("metadata intrinsics do not match the raw cameras")
    if camera_to_world.shape != (camera_count, 4, 4):
        raise ValueError("calibrate.pkl does not match the raw cameras")
    with (raw_path / "mask" / "processed_masks.pkl").open("rb") as handle:
        processed_masks = pickle.load(handle)

    prefix_frames = config.train_end_frame
    tracks = np.full(
        (camera_count, prefix_frames, track_count, 2),
        np.nan,
        dtype=np.float32,
    )
    quality = np.zeros(
        (camera_count, prefix_frames, track_count),
        dtype=np.float32,
    )
    view_valid = np.zeros_like(quality, dtype=bool)
    cycle_error = np.full_like(quality, np.inf, dtype=np.float32)
    cycle_valid = np.zeros_like(quality, dtype=bool)
    initial_eligible = np.zeros((camera_count, track_count), dtype=bool)
    initial_surface_distance = np.full(
        (camera_count, track_count),
        np.inf,
        dtype=np.float32,
    )
    per_camera: dict[str, Any] = {}

    runner = PhysTwinAllTrackerRunner(
        alltracker_source,
        checkpoint_path,
        device=device,
        config=config.source_config(),
    )
    try:
        for camera in range(camera_count):
            video = _load_video_prefix(raw_path, camera, prefix_frames)
            projected, eligible, surface_distance = (
                _initial_multiview_eligibility(
                    mapping.source_world_points,
                    mapping.camera_points[camera],
                    np.asarray(
                        processed_masks[0][camera]["object"],
                        dtype=bool,
                    ),
                    intrinsics[camera],
                    camera_to_world[camera],
                    depth_tolerance_m=(
                        config.multiview_initial_depth_tolerance_m
                    ),
                )
            )
            initial_eligible[camera] = eligible
            initial_surface_distance[camera] = surface_distance.astype(
                np.float32
            )
            source_ids = np.flatnonzero(mapping.source_camera == camera)
            source_ids = source_ids[eligible[source_ids]]
            tracks[camera][:, source_ids] = cues["source_tracks_xy"][
                :prefix_frames, source_ids
            ]
            quality[camera][:, source_ids] = cues[
                "source_quality_probability"
            ][:prefix_frames, source_ids]
            cycle_error[camera][:, source_ids] = cues[
                "forward_backward_error_px"
            ][:prefix_frames, source_ids]
            cycle_valid[camera][:, source_ids] = cues[
                "forward_backward_valid"
            ][:prefix_frames, source_ids]

            cross_ids = np.flatnonzero(
                eligible & (mapping.source_camera != camera)
            )
            if len(cross_ids):
                forward = runner.track(
                    video,
                    projected[cross_ids].astype(np.float32),
                )
                reverse = runner.track(
                    np.ascontiguousarray(video[::-1]),
                    forward.tracks_xy[-1],
                )
                reverse_tracks = reverse.tracks_xy[::-1]
                reverse_quality = reverse.quality_probability[::-1]
                tracks[camera][:, cross_ids] = forward.tracks_xy
                quality[camera][:, cross_ids] = (
                    forward.quality_probability
                )
                cycle_error[camera][:, cross_ids] = np.linalg.norm(
                    forward.tracks_xy - reverse_tracks,
                    axis=2,
                )
                cycle_valid[camera][:, cross_ids] = (
                    forward.quality_probability
                    >= config.minimum_multiview_quality
                ) & (
                    reverse_quality >= config.minimum_multiview_quality
                )

            for frame in range(prefix_frames):
                object_mask = np.asarray(
                    processed_masks[frame][camera]["object"],
                    dtype=bool,
                )
                inside = _pixels_inside_mask(tracks[camera, frame], object_mask)
                view_valid[camera, frame] = (
                    eligible
                    & inside
                    & (
                        quality[camera, frame]
                        >= config.minimum_multiview_quality
                    )
                    & cycle_valid[camera, frame]
                    & (
                        cycle_error[camera, frame]
                        <= config.maximum_cycle_error_px
                    )
                )
            per_camera[str(camera)] = {
                "initial_eligible_count": int(np.sum(eligible)),
                "cross_view_query_count": int(len(cross_ids)),
                "valid_view_fraction": float(np.mean(view_valid[camera])),
                "cycle_error_px": _distribution(
                    cycle_error[camera][cycle_valid[camera]]
                ),
            }
    finally:
        runner.close()

    points, reprojection, support = triangulate_multiview_tracks(
        tracks,
        view_valid,
        np.where(view_valid, quality, 0.0),
        intrinsics,
        camera_to_world,
    )
    packed = pack_multiview_triangulation(
        points,
        reprojection,
        support,
        frame_count=frame_count,
    )
    full_reprojection = np.zeros(
        (frame_count, track_count),
        dtype=np.float32,
    )
    full_reprojection_valid = np.zeros(
        (frame_count, track_count),
        dtype=bool,
    )
    full_camera_count = np.zeros(
        (frame_count, track_count),
        dtype=np.int16,
    )
    triangulated = (
        np.all(np.isfinite(points), axis=2)
        & np.isfinite(reprojection)
        & (support >= 2)
    )
    full_reprojection[:prefix_frames][triangulated] = reprojection[
        triangulated
    ]
    full_reprojection_valid[:prefix_frames] = triangulated
    full_camera_count[:prefix_frames] = support
    augmented = {
        **cues,
        "multiview_reprojection_error_px": full_reprojection,
        "multiview_valid": full_reprojection_valid,
        "multiview_camera_count": full_camera_count,
        **packed,
        "multiview_initial_eligible": initial_eligible,
        "multiview_initial_surface_distance_m": initial_surface_distance,
        "multiview_tracks_xy_prefix": tracks,
        "multiview_quality_probability_prefix": quality,
        "multiview_view_valid_prefix": view_valid,
        "multiview_forward_backward_error_px_prefix": cycle_error,
        "multiview_forward_backward_valid_prefix": cycle_valid,
        "multiview_intrinsics": intrinsics.astype(np.float64),
        "multiview_camera_to_world": camera_to_world.astype(np.float64),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **augmented)

    source_arrays_preserved = all(
        np.array_equal(augmented[name], value, equal_nan=True)
        for name, value in cues.items()
    )
    selected = triangulated & mapping.final_visible[:prefix_frames]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinAllTrackerMultiviewCues",
        "config": asdict(config),
        "tracker": base_summary["tracker"],
        "inputs": {
            "final_data": {
                "path": str(Path(final_data_path).resolve()),
                "sha256": _sha256(final_data_path),
            },
            "raw_case_dir": str(raw_path.resolve()),
            "base_source_cues": {
                "path": str(base_path.resolve()),
                "sha256": _sha256(base_path),
                "summary_sha256": _sha256(
                    base_path.with_suffix(".summary.json")
                ),
            },
        },
        "camera_summaries": per_camera,
        "multiview": {
            "initial_eligible_fraction_by_camera": {
                str(camera): float(np.mean(initial_eligible[camera]))
                for camera in range(camera_count)
            },
            "triangulated_visible_fraction": float(np.mean(selected)),
            "three_view_visible_fraction": float(
                np.mean((support >= 3) & mapping.final_visible[:prefix_frames])
            ),
            "reprojection_error_px": _distribution(
                reprojection[selected]
            ),
            "camera_count": _distribution(support[selected]),
        },
        "compatibility": {
            "source_field_count": len(cues),
            "source_arrays_preserved_exactly": source_arrays_preserved,
        },
        "output": {
            "path": str(output.resolve()),
            "sha256": _sha256(output),
        },
        "information_boundary": {
            "rgb_frame_range_half_open": [0, prefix_frames],
            "future_rgb_read": False,
            "future_cue_rows_neutralized": True,
            "future_outcome_read": False,
        },
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
