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
from .phystwin_cotracker3_cues import _distribution, _sha256
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
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:
        raise RuntimeError("AllTracker cue extraction requires scipy") from exc

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
                distance = distance_transform_edt(object_mask) / max(
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
