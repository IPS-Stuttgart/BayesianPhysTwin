"""Source-only thin-rope Splatfacto reconstruction probe."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360 import Deform360ProtocolConfig
from .deform360_sam2_prefix import select_source_locked_prefix_cameras


DEFORM360_SPLAT_PROBE_SCHEMA_VERSION = 1


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    descriptor = _canonical_bytes(
        {"dtype": array.dtype.str, "shape": list(array.shape)}
    )
    return hashlib.sha256(
        descriptor + b"\0" + array.view(np.uint8).tobytes()
    ).hexdigest()


@dataclass(frozen=True)
class ThinRopeSplatProbeConfig:
    """Frozen development gate before expensive per-frame reconstruction."""

    source_episode_index: int = 0
    frame_index: int = 0
    minimum_synchronization_reliability: float = 0.85
    minimum_camera_count: int = 8
    cube_half_extent_m: float = 0.5
    voxel_resolution: int = 100
    minimum_hull_points: int = 256
    training_iterations: int = 1_000
    minimum_gaussian_count: int = 256
    minimum_principal_span_m: float = 0.15
    maximum_principal_span_m: float = 0.75
    maximum_minor_span_m: float = 0.15
    minimum_median_mask_containment: float = 0.75
    minimum_worst_view_mask_containment: float = 0.35

    def __post_init__(self) -> None:
        _require(
            self.source_episode_index >= 0, "source episode index must be nonnegative"
        )
        _require(self.frame_index >= 0, "probe frame index must be nonnegative")
        _require(
            0.0 <= self.minimum_synchronization_reliability <= 1.0,
            "invalid synchronization-reliability threshold",
        )
        _require(self.minimum_camera_count >= 2, "at least two cameras are required")
        _require(self.cube_half_extent_m > 0.0, "cube extent must be positive")
        _require(self.voxel_resolution >= 16, "voxel resolution is too small")
        _require(self.minimum_hull_points >= 16, "minimum hull size is too small")
        _require(self.training_iterations >= 1, "training iterations must be positive")
        _require(
            self.minimum_gaussian_count >= 1, "Gaussian count gate must be positive"
        )
        _require(
            0.0 < self.minimum_principal_span_m < self.maximum_principal_span_m,
            "invalid principal-span gate",
        )
        _require(self.maximum_minor_span_m > 0.0, "minor-span gate must be positive")
        _require(
            0.0 <= self.minimum_median_mask_containment <= 1.0,
            "invalid median containment gate",
        )
        _require(
            0.0 <= self.minimum_worst_view_mask_containment <= 1.0,
            "invalid worst-view containment gate",
        )


def _opacity_weights(opacity: np.ndarray | None, count: int) -> np.ndarray:
    if opacity is None:
        return np.ones(count, dtype=np.float64)
    values = np.asarray(opacity, dtype=np.float64).reshape(-1)
    _require(values.shape == (count,), "opacity count does not match Gaussian count")
    _require(np.all(np.isfinite(values)), "Gaussian opacity contains non-finite values")
    if np.any((values < 0.0) | (values > 1.0)):
        values = 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))
    return np.maximum(values, 1e-8)


def gaussian_splat_geometry_diagnostics(
    positions_world_m: np.ndarray,
    *,
    opacity: np.ndarray | None,
    masks_by_camera: Mapping[str, np.ndarray],
    intrinsics_by_camera: Mapping[str, np.ndarray],
    camera_to_world_by_camera: Mapping[str, np.ndarray],
    config: ThinRopeSplatProbeConfig,
) -> dict[str, Any]:
    """Audit robust shape and multiview mask containment of Gaussian centers."""

    positions = np.asarray(positions_world_m, dtype=np.float64)
    _require(
        positions.ndim == 2 and positions.shape[1] == 3 and len(positions) > 0,
        "Gaussian positions must have shape (N,3)",
    )
    _require(np.all(np.isfinite(positions)), "Gaussian positions are non-finite")
    cameras = tuple(sorted(masks_by_camera))
    _require(len(cameras) >= 2, "at least two masks are required for splat QA")
    _require(
        all(camera in intrinsics_by_camera for camera in cameras),
        "splat QA is missing camera intrinsics",
    )
    _require(
        all(camera in camera_to_world_by_camera for camera in cameras),
        "splat QA is missing camera extrinsics",
    )
    weights = _opacity_weights(opacity, len(positions))
    centered = positions - np.median(positions, axis=0)
    covariance = np.cov(centered.T, aweights=weights)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order]
    coordinates = centered @ axes
    quantiles = np.percentile(coordinates, [1.0, 50.0, 99.0], axis=0)
    spans = quantiles[2] - quantiles[0]

    per_camera = []
    for camera in cameras:
        mask = np.asarray(masks_by_camera[camera], dtype=bool)
        _require(mask.ndim == 2, f"mask for {camera} must be two-dimensional")
        height, width = mask.shape
        intrinsics = np.asarray(intrinsics_by_camera[camera], dtype=np.float64)
        camera_to_world = np.asarray(
            camera_to_world_by_camera[camera], dtype=np.float64
        )
        _require(intrinsics.shape == (3, 3), f"invalid intrinsics for {camera}")
        _require(camera_to_world.shape == (4, 4), f"invalid extrinsics for {camera}")
        world_to_camera = np.linalg.inv(camera_to_world)
        points_camera = positions @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        depth = points_camera[:, 2]
        in_front = depth > 1e-6
        safe_depth = np.where(in_front, depth, 1.0)
        u = points_camera[:, 0] / safe_depth * intrinsics[0, 0] + intrinsics[0, 2]
        v = points_camera[:, 1] / safe_depth * intrinsics[1, 1] + intrinsics[1, 2]
        visible = in_front & (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
        columns = np.clip(np.rint(u), 0, width - 1).astype(np.int64)
        rows = np.clip(np.rint(v), 0, height - 1).astype(np.int64)
        inside = visible & mask[rows, columns]
        visible_weight = float(np.sum(weights[visible]))
        containment = (
            float(np.sum(weights[inside]) / visible_weight)
            if visible_weight > 0.0
            else 0.0
        )
        per_camera.append(
            {
                "camera": camera,
                "visible_gaussian_count": int(np.count_nonzero(visible)),
                "inside_mask_gaussian_count": int(np.count_nonzero(inside)),
                "opacity_weighted_mask_containment": containment,
            }
        )
    containment = np.asarray(
        [item["opacity_weighted_mask_containment"] for item in per_camera],
        dtype=np.float64,
    )
    gates = {
        "gaussian_count": len(positions) >= config.minimum_gaussian_count,
        "principal_span": bool(
            config.minimum_principal_span_m
            <= spans[0]
            <= config.maximum_principal_span_m
        ),
        "minor_spans": bool(np.all(spans[1:] <= config.maximum_minor_span_m)),
        "median_mask_containment": bool(
            np.median(containment) >= config.minimum_median_mask_containment
        ),
        "worst_view_mask_containment": bool(
            np.min(containment) >= config.minimum_worst_view_mask_containment
        ),
    }
    return {
        "gaussian_count": len(positions),
        "opacity_weight_sum": float(np.sum(weights)),
        "pca_axis_world": axes.T.tolist(),
        "pca_q01_m": quantiles[0].tolist(),
        "pca_median_m": quantiles[1].tolist(),
        "pca_q99_m": quantiles[2].tolist(),
        "pca_q01_to_q99_span_m": spans.tolist(),
        "per_camera_projection": per_camera,
        "mask_containment": {
            "minimum": float(np.min(containment)),
            "median": float(np.median(containment)),
            "maximum": float(np.max(containment)),
        },
        "acceptance_gates": gates,
        "probe_passed": all(gates.values()),
    }


def _read_gaussian_ply(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    try:
        from plyfile import PlyData
    except ImportError as error:  # pragma: no cover - GPU-host integration
        raise RuntimeError("plyfile is required for the splat probe") from error
    vertices = PlyData.read(str(path))["vertex"].data
    names = set(vertices.dtype.names or ())
    _require({"x", "y", "z"} <= names, "splat PLY lacks Gaussian centers")
    positions = np.column_stack((vertices["x"], vertices["y"], vertices["z"]))
    opacity = np.asarray(vertices["opacity"]) if "opacity" in names else None
    return positions, opacity


def splat_probe_artifact_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def run_source_splat_probe(
    processed_root: str | Path,
    protocol: Deform360ProtocolConfig,
    source_view_audit: Mapping[str, Any],
    preflight: Mapping[str, Any],
    output_dir: str | Path,
    *,
    config: ThinRopeSplatProbeConfig | None = None,
    trainer: Any | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Fit one source-frame splat with a tight hull and emit a QA artifact."""

    cfg = config or ThinRopeSplatProbeConfig()
    _require(
        cfg.source_episode_index in protocol.source_episode_ids,
        "splat probe episode is not in the locked source split",
    )
    _require(
        source_view_audit.get("episode_access", {}).get("episode_index")
        == cfg.source_episode_index,
        "source view audit and splat probe episode differ",
    )
    policy = select_source_locked_prefix_cameras(
        source_view_audit,
        preflight,
        minimum_synchronization_reliability=(cfg.minimum_synchronization_reliability),
        minimum_camera_count=cfg.minimum_camera_count,
    )
    try:
        import cv2
        from deform360.annotations import H5Array
        from deform360.processing.episode import load_episode_calibration
        from deform360.processing.reconstruct_stage import (
            NerfstudioSplatTrainer,
            build_nerfstudio_dataset,
            visual_hull_points,
            write_seed_ply,
        )
    except ImportError as error:  # pragma: no cover - GPU-host integration
        raise RuntimeError(
            "the pinned Deform360 processing environment is required"
        ) from error

    root = Path(processed_root).resolve()
    episode_dir = root / f"episode_{cfg.source_episode_index:04d}"
    _require(episode_dir.is_dir(), f"source episode is missing: {episode_dir}")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    splat_path = output / f"splat_{cfg.frame_index}.ply"
    seed_path = output / f"visual_hull_seed_{cfg.frame_index}.ply"
    if splat_path.exists() and not overwrite:
        raise FileExistsError(f"splat probe output already exists: {splat_path}")
    cameras: Sequence[str] = policy["selected_cameras"]
    intrinsics, extrinsics = load_episode_calibration(episode_dir)
    masks: dict[str, np.ndarray] = {}
    images: dict[str, np.ndarray] = {}
    inputs = []
    for camera in cameras:
        camera_dir = episode_dir / camera
        mask_path = camera_dir / "mask_refined.h5"
        video_path = camera_dir / "undistorted.mp4"
        with H5Array(mask_path) as store:
            mask = np.asarray(store[cfg.frame_index], dtype=np.uint8)
        capture = cv2.VideoCapture(str(video_path))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, cfg.frame_index)
            ok, bgr = capture.read()
        finally:
            capture.release()
        _require(ok, f"cannot read source frame for {camera}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        _require(mask.shape == rgb.shape[:2], f"mask/video shape mismatch for {camera}")
        masks[camera] = mask
        images[camera] = rgb
        inputs.append(
            {
                "camera": camera,
                "mask_file_sha256": _sha256_file(mask_path),
                "selected_mask_sha256": _sha256_array(mask),
                "selected_rgb_sha256": _sha256_array(rgb),
                "full_video_hashed": False,
            }
        )

    points, colors = visual_hull_points(
        masks,
        images,
        {camera: intrinsics[camera] for camera in cameras},
        {camera: extrinsics[camera] for camera in cameras},
        cube_half_extent_m=cfg.cube_half_extent_m,
        voxel_resolution=cfg.voxel_resolution,
        min_points=cfg.minimum_hull_points,
    )
    write_seed_ply(seed_path, points, colors)
    if trainer is None:
        trainer = NerfstudioSplatTrainer()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(
        prefix=".causal4d-splat-probe-", dir=output
    ) as temp:
        dataset_dir = Path(temp) / "dataset"
        build_nerfstudio_dataset(
            episode_dir,
            cfg.frame_index,
            dataset_dir,
            cameras=cameras,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            seed_ply_path=seed_path,
        )
        if overwrite and splat_path.exists():
            splat_path.unlink()
        produced = Path(
            trainer.train(
                dataset_dir,
                output,
                splat_path.name,
                cfg.training_iterations,
            )
        )
    elapsed = time.monotonic() - started
    _require(
        produced == splat_path and splat_path.is_file(), "splat trainer output mismatch"
    )
    positions, opacity = _read_gaussian_ply(splat_path)
    diagnostics = gaussian_splat_geometry_diagnostics(
        positions,
        opacity=opacity,
        masks_by_camera=masks,
        intrinsics_by_camera={camera: intrinsics[camera] for camera in cameras},
        camera_to_world_by_camera={camera: extrinsics[camera] for camera in cameras},
        config=cfg,
    )
    payload: dict[str, Any] = {
        "schema_version": DEFORM360_SPLAT_PROBE_SCHEMA_VERSION,
        "artifact_kind": "Deform360ThinRopeSplatProbe",
        "protocol_id": protocol.protocol_id,
        "episode_index": cfg.source_episode_index,
        "split": "source",
        "frame_index": cfg.frame_index,
        "parameters": asdict(cfg),
        "camera_policy": policy,
        "inputs": inputs,
        "visual_hull_seed": {
            "path": str(seed_path),
            "sha256": _sha256_file(seed_path),
            "point_count": len(points),
            "q01_to_q99_span_m": (
                np.percentile(points, 99.0, axis=0) - np.percentile(points, 1.0, axis=0)
            ).tolist(),
        },
        "splat": {
            "path": str(splat_path),
            "sha256": _sha256_file(splat_path),
            "bytes": splat_path.stat().st_size,
            "training_elapsed_seconds": elapsed,
        },
        "diagnostics": diagnostics,
        "information_boundary": {
            "source_episode_only": True,
            "target_frames_read": False,
            "target_metrics_computed": False,
            "camera_selection_source_only": True,
        },
        "claim_boundary": (
            "Development QA for a public SAM2 thin-rope seed; this is not a held-out "
            "physics result or an exact reproduction of Deform360's SAM3 masks."
        ),
    }
    payload["result_sha256"] = splat_probe_artifact_sha256(payload)
    return payload


def validate_splat_probe_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        payload.get("schema_version") == DEFORM360_SPLAT_PROBE_SCHEMA_VERSION,
        "unsupported splat-probe artifact schema",
    )
    _require(
        payload.get("artifact_kind") == "Deform360ThinRopeSplatProbe",
        "unexpected splat-probe artifact kind",
    )
    _require(
        payload.get("result_sha256") == splat_probe_artifact_sha256(payload),
        "splat-probe artifact checksum mismatch",
    )
    _require(payload.get("split") == "source", "splat probe is not source-only")
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("target_frames_read") is False, "splat probe read target frames"
    )
    return {
        "passed": True,
        "probe_passed": bool(payload.get("diagnostics", {}).get("probe_passed")),
        "result_sha256": payload["result_sha256"],
    }


def write_splat_probe_artifact(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


__all__ = [
    "DEFORM360_SPLAT_PROBE_SCHEMA_VERSION",
    "ThinRopeSplatProbeConfig",
    "gaussian_splat_geometry_diagnostics",
    "run_source_splat_probe",
    "splat_probe_artifact_sha256",
    "validate_splat_probe_artifact",
    "write_splat_probe_artifact",
]
