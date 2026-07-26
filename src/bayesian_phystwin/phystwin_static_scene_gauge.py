"""PhysTwin adapter for target-free static-scene tracker-gauge artifacts."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .phystwin_cotracker3_cues import (
    CoTracker3OnlineRunner,
    _git_revision,
    _sha256,
)
from .static_scene_gauge import (
    STATIC_SCENE_GAUGE_SCHEMA_VERSION,
    StaticSceneGaugeConfig,
    apply_static_scene_gauge,
    estimate_static_scene_gauge,
    select_static_scene_queries,
)

PHYSTWIN_STATIC_SCENE_GAUGE_ARTIFACT_KIND = "PhysTwinStaticSceneGaugeV1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _prefix_tree_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


@dataclass(frozen=True)
class PhysTwinStaticSceneGaugeConfig:
    """Frozen causal extraction contract."""

    train_end_frame: int
    cotracker_iterations: int = 6
    cotracker_window_length: int = 16
    minimum_depth_mm: int = 1
    maximum_depth_mm: int = 10_000
    gauge: StaticSceneGaugeConfig = StaticSceneGaugeConfig()

    def __post_init__(self) -> None:
        _require(
            self.train_end_frame >= 2,
            "train_end_frame must be at least two",
        )
        _require(
            self.cotracker_iterations >= 1,
            "CoTracker iterations must be positive",
        )
        _require(
            self.cotracker_window_length >= 4
            and self.cotracker_window_length % 2 == 0,
            "CoTracker window length must be even and at least four",
        )
        _require(
            0 <= self.minimum_depth_mm < self.maximum_depth_mm,
            "depth range is invalid",
        )


def _load_video_prefix(
    raw_case_dir: Path,
    camera: int,
    end_frame: int,
) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - GPU integration only
        raise RuntimeError(
            "static-scene gauge extraction requires Pillow"
        ) from exc
    frames = []
    for frame in range(end_frame):
        path = raw_case_dir / "color" / str(camera) / f"{frame}.png"
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            frames.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
    return np.stack(frames)


def _camera_summary(
    estimate: Any,
    query_count: int,
    support_fraction: float,
) -> dict[str, Any]:
    return {
        "accepted": bool(estimate.accepted),
        "reason": str(estimate.reason),
        "static_query_count": int(query_count),
        "background_cluster_count": int(estimate.background_cluster_count),
        "requested_track_support_fraction": float(support_fraction),
        "cross_validation": {
            "count": int(estimate.cross_validation_count),
            "raw_error_px": estimate.cross_validation_raw_error_px,
            "corrected_error_px": (
                estimate.cross_validation_corrected_error_px
            ),
            "relative_gain": estimate.cross_validation_relative_gain,
        },
        "estimate_sha256": estimate.content_sha256,
    }


def build_phystwin_static_scene_gauge(
    cues_path: str | Path,
    raw_case_dir: str | Path,
    checkpoint_path: str | Path,
    cotracker_root: str | Path,
    output_npz_path: str | Path,
    *,
    config: PhysTwinStaticSceneGaugeConfig,
    device: str = "cuda",
) -> dict[str, Any]:
    """Build an opt-in nuisance artifact from only the allowed RGB-D prefix."""

    cues_file = Path(cues_path)
    raw_path = Path(raw_case_dir)
    output = Path(output_npz_path)
    required = {
        "source_tracks_xy",
        "source_camera",
        "multiview_tracks_xy_prefix",
    }
    with np.load(cues_file) as archive:
        missing = required.difference(archive.files)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"cue archive lacks required fields: {names}")
        source_tracks = np.asarray(archive["source_tracks_xy"])
        source_camera = np.asarray(archive["source_camera"], dtype=np.int64)
        multiview_tracks = np.asarray(
            archive["multiview_tracks_xy_prefix"]
        )
    end = config.train_end_frame
    _require(
        source_tracks.ndim == 3 and source_tracks.shape[2] == 2,
        "source tracks must have shape (T, N, 2)",
    )
    _require(
        multiview_tracks.ndim == 4 and multiview_tracks.shape[3] == 2,
        "multiview tracks must have shape (C, T, N, 2)",
    )
    _require(
        len(source_camera) == source_tracks.shape[1],
        "source camera inventory differs from source tracks",
    )
    _require(
        source_tracks.shape[0] >= end and multiview_tracks.shape[1] >= end,
        "cue archive is shorter than the configured prefix",
    )
    _require(
        source_tracks.shape[1] == multiview_tracks.shape[2],
        "source and multiview identity counts differ",
    )
    camera_count = multiview_tracks.shape[0]
    _require(
        np.all((source_camera >= 0) & (source_camera < camera_count)),
        "source camera index is invalid",
    )

    mask_path = raw_path / "mask" / "processed_masks.pkl"
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)
    with mask_path.open("rb") as handle:
        masks = pickle.load(handle)
    _require(len(masks) >= end, "processed masks are shorter than the prefix")

    source_correction = np.zeros(
        (end, source_tracks.shape[1], 2),
        dtype=np.float32,
    )
    source_variance = np.full(
        source_correction.shape[:2],
        config.gauge.minimum_variance_px2,
        dtype=np.float32,
    )
    source_supported = np.zeros(source_correction.shape[:2], dtype=bool)
    multiview_correction = np.zeros(
        (camera_count, end, multiview_tracks.shape[2], 2),
        dtype=np.float32,
    )
    multiview_variance = np.full(
        multiview_correction.shape[:3],
        config.gauge.minimum_variance_px2,
        dtype=np.float32,
    )
    multiview_supported = np.zeros(
        multiview_correction.shape[:3],
        dtype=bool,
    )
    camera_accepted = np.zeros(camera_count, dtype=bool)
    summaries: dict[str, Any] = {}
    rgb_paths: list[Path] = []
    depth_paths: list[Path] = []

    runner = CoTracker3OnlineRunner(
        checkpoint_path,
        cotracker_root=cotracker_root,
        device=device,
        window_length=config.cotracker_window_length,
        iterations=config.cotracker_iterations,
    )
    for camera in range(camera_count):
        dynamic = np.stack(
            [
                np.asarray(masks[frame][camera]["object"], dtype=bool)
                | np.asarray(
                    masks[frame][camera]["controller"],
                    dtype=bool,
                )
                for frame in range(end)
            ]
        )
        depth_path = raw_path / "depth" / str(camera) / "0.npy"
        if not depth_path.is_file():
            raise FileNotFoundError(depth_path)
        depth = np.asarray(np.load(depth_path))
        depth_valid = (
            (depth >= config.minimum_depth_mm)
            & (depth <= config.maximum_depth_mm)
        )
        camera_rgb_paths = [
            raw_path / "color" / str(camera) / f"{frame}.png"
            for frame in range(end)
        ]
        rgb_paths.extend(camera_rgb_paths)
        depth_paths.append(depth_path)
        try:
            queries = select_static_scene_queries(
                dynamic,
                depth_valid,
                config=config.gauge,
            )
        except ValueError as error:
            if str(error) != "static-scene query support is empty":
                raise
            rejected = {
                "accepted": False,
                "reason": "insufficient-static-scene-queries",
                "static_query_count": 0,
                "background_cluster_count": 0,
                "requested_track_support_fraction": 0.0,
                "cross_validation": {
                    "count": 0,
                    "raw_error_px": None,
                    "corrected_error_px": None,
                    "relative_gain": None,
                },
                "estimate_sha256": None,
            }
            summaries[str(camera)] = {
                "source": rejected,
                "multiview": rejected,
            }
            continue
        video = _load_video_prefix(raw_path, camera, end)
        prediction = runner.track(video, queries)
        background_quality = np.minimum(
            prediction.visibility_probability,
            prediction.confidence_probability,
        )

        selected = np.flatnonzero(source_camera == camera)
        source_estimate = estimate_static_scene_gauge(
            queries,
            prediction.tracks_xy,
            background_quality,
            source_tracks[0, selected],
            config=config.gauge,
        )
        source_correction[:, selected] = source_estimate.correction_px
        source_variance[:, selected] = source_estimate.variance_px2
        source_supported[:, selected] = source_estimate.supported

        multiview_estimate = estimate_static_scene_gauge(
            queries,
            prediction.tracks_xy,
            background_quality,
            multiview_tracks[camera, 0],
            config=config.gauge,
        )
        _require(
            source_estimate.accepted == multiview_estimate.accepted,
            "camera admission depends on requested object identities",
        )
        multiview_correction[camera] = multiview_estimate.correction_px
        multiview_variance[camera] = multiview_estimate.variance_px2
        multiview_supported[camera] = multiview_estimate.supported
        camera_accepted[camera] = source_estimate.accepted
        summaries[str(camera)] = {
            "source": _camera_summary(
                source_estimate,
                len(queries),
                (
                    float(np.mean(source_estimate.supported))
                    if source_estimate.supported.size
                    else 0.0
                ),
            ),
            "multiview": _camera_summary(
                multiview_estimate,
                len(queries),
                np.mean(multiview_estimate.supported),
            ),
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        source_track_correction_px=source_correction,
        source_track_variance_px2=source_variance,
        source_track_supported=source_supported,
        multiview_track_correction_px=multiview_correction,
        multiview_track_variance_px2=multiview_variance,
        multiview_track_supported=multiview_supported,
        camera_accepted=camera_accepted,
    )
    summary: dict[str, Any] = {
        "schema_version": STATIC_SCENE_GAUGE_SCHEMA_VERSION,
        "artifact_kind": PHYSTWIN_STATIC_SCENE_GAUGE_ARTIFACT_KIND,
        "config": asdict(config),
        "inputs": {
            "cues_path": str(cues_file.resolve()),
            "cues_sha256": _sha256(cues_file),
            "raw_case_dir": str(raw_path.resolve()),
            "processed_masks_sha256": _sha256(mask_path),
            "prefix_rgb_tree_sha256": _prefix_tree_sha256(rgb_paths),
            "frame_zero_depth_tree_sha256": _prefix_tree_sha256(depth_paths),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "cotracker_revision": _git_revision(cotracker_root),
        },
        "camera_count": camera_count,
        "camera_summaries": summaries,
        "output": {
            "path": str(output.resolve()),
            "sha256": _sha256(output),
        },
        "information_boundary": {
            "rgb_frame_range_half_open": [0, end],
            "future_rgb_read": False,
            "manual_identity_read": False,
            "physical_state_innovation_read": False,
            "reliability_source": (
                "spatially held-out static-scene tracker error"
            ),
            "rejected_camera_behavior": (
                "zero correction; downstream tracks remain byte-identical"
            ),
        },
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _load_bound_artifact(
    cues_path: str | Path,
    gauge_path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cues_file = Path(cues_path)
    gauge_file = Path(gauge_path)
    summary_path = gauge_file.with_suffix(".summary.json")
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _require(
        summary.get("artifact_kind")
        == PHYSTWIN_STATIC_SCENE_GAUGE_ARTIFACT_KIND,
        "static-scene gauge artifact kind changed",
    )
    _require(
        summary["inputs"]["cues_sha256"] == _sha256(cues_file),
        "static-scene gauge is bound to different source cues",
    )
    _require(
        summary["output"]["sha256"] == _sha256(gauge_file),
        "static-scene gauge output checksum changed",
    )
    with np.load(gauge_file) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    return arrays, summary


def load_static_scene_corrected_source_tracks(
    cues_path: str | Path,
    gauge_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load source-camera tracks and apply only an admitted gauge field."""

    with np.load(cues_path) as archive:
        tracks = np.asarray(archive["source_tracks_xy"])
    arrays, summary = _load_bound_artifact(cues_path, gauge_path)
    correction = arrays["source_track_correction_px"]
    supported = arrays["source_track_supported"].astype(bool)
    variance = arrays["source_track_variance_px2"].astype(np.float64)
    _require(
        correction.shape == tracks[: len(correction)].shape,
        "source correction shape differs from source tracks",
    )
    estimate_like = type(
        "_ArtifactEstimate",
        (),
        {
            "correction_px": correction,
            "supported": supported,
            "accepted": bool(np.any(arrays["camera_accepted"])),
        },
    )()
    prefix = apply_static_scene_gauge(
        tracks[: len(correction)],
        estimate_like,
    )
    corrected = tracks.copy()
    corrected[: len(prefix)] = prefix
    return corrected, variance, supported, summary


def load_static_scene_corrected_multiview_tracks(
    cues_path: str | Path,
    gauge_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load per-view tracks and apply admitted camera nuisance fields."""

    with np.load(cues_path) as archive:
        tracks = np.asarray(archive["multiview_tracks_xy_prefix"])
    arrays, summary = _load_bound_artifact(cues_path, gauge_path)
    correction = arrays["multiview_track_correction_px"]
    supported = arrays["multiview_track_supported"].astype(bool)
    variance = arrays["multiview_track_variance_px2"].astype(np.float64)
    _require(
        correction.shape == tracks[:, : correction.shape[1]].shape,
        "multiview correction shape differs from multiview tracks",
    )
    corrected = tracks.copy()
    camera_accepted = arrays["camera_accepted"].astype(bool)
    for camera in range(len(corrected)):
        if not camera_accepted[camera]:
            continue
        selected = supported[camera]
        corrected[camera, : correction.shape[1]][selected] -= correction[
            camera
        ][selected]
    return corrected, variance, supported, summary


__all__ = [
    "PHYSTWIN_STATIC_SCENE_GAUGE_ARTIFACT_KIND",
    "PhysTwinStaticSceneGaugeConfig",
    "build_phystwin_static_scene_gauge",
    "load_static_scene_corrected_multiview_tracks",
    "load_static_scene_corrected_source_tracks",
]
