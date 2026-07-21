#!/usr/bin/env python3
"""Apply the source-frozen depth-supported fallback to known failures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from bayesian_phystwin.deform360_frame_zero_depth_initializer import (
    DepthSupportedFrameZeroInitializerConfig,
    filter_surface_with_metric_depth,
    metric_depth_support_counts,
    select_depth_supported_frame_zero_point_cloud,
)
from bayesian_phystwin.deform360_frame_zero_initializer import (
    FrameZeroInitializerConfig,
    build_strict_multiview_surface,
    original_point_cloud_admissible,
)
from deform360.annotations import H5Array
from deform360.processing import pcd_stage
from deform360.processing.episode import load_episode_calibration


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


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    descriptor = _canonical_bytes(
        {"dtype": array.dtype.str, "shape": list(array.shape)}
    )
    return hashlib.sha256(
        descriptor + b"\0" + array.view(np.uint8).tobytes()
    ).hexdigest()


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_repository(repository: Path) -> str:
    revision = _git_revision(repository)
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _require(not status.strip(), "Bayesian-PhysTwin repository is not clean")
    return revision


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--staged-root", type=Path, required=True)
    parser.add_argument("--failure-root", type=Path, required=True)
    parser.add_argument("--deform360-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _load_sealed_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    observed = payload.get("result_sha256")
    expected = _canonical_sha256(
        {key: value for key, value in payload.items() if key != "result_sha256"}
    )
    _require(observed == expected, f"JSON checksum changed: {path}")
    return payload


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("schema_version") == 1, "config schema changed")
    config = payload.get("config")
    _require(isinstance(config, dict), "config is missing")
    expected = _canonical_sha256(config)
    _require(payload.get("config_sha256") == expected, "config checksum changed")
    return config, expected


def _load_initializer_config(
    repository: Path,
    config: dict[str, Any],
) -> DepthSupportedFrameZeroInitializerConfig:
    path = (
        repository
        / "configs/sota/deform360_frame_zero_depth_initializer_source_v2.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = config["source_candidate"]["source_config_sha256"]
    _require(payload.get("config_sha256") == expected, "source config changed")
    method = payload["config"]["method"]
    visual_hull = FrameZeroInitializerConfig(**method["visual_hull"])
    depth = method["depth_support"]
    return DepthSupportedFrameZeroInitializerConfig(
        visual_hull=visual_hull,
        depth_tolerance_m=float(depth["depth_tolerance_m"]),
        minimum_depth_camera_count=int(depth["minimum_depth_camera_count"]),
        minimum_depth_support_views=int(depth["minimum_depth_support_views"]),
        minimum_depth_supported_point_count=int(
            depth["minimum_depth_supported_point_count"]
        ),
    )


def _read_frame_zero_image(camera_dir: Path) -> np.ndarray:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - integration dependency
        raise RuntimeError("OpenCV is required for the post-open audit") from error
    capture = cv2.VideoCapture(str(camera_dir / "undistorted.mp4"))
    try:
        ok, image = capture.read()
    finally:
        capture.release()
    _require(ok and image is not None, f"cannot read frame zero from {camera_dir}")
    return image


def _load_frame_zero_views(
    episode_dir: Path,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    intrinsics, extrinsics = load_episode_calibration(episode_dir)
    masks: dict[str, np.ndarray] = {}
    images: dict[str, np.ndarray] = {}
    depths: dict[str, np.ndarray] = {}
    input_hashes: dict[str, Any] = {}
    for camera in sorted(intrinsics):
        camera_dir = episode_dir / camera
        mask_path = camera_dir / "mask_refined.h5"
        depth_path = camera_dir / "rendered_depth.h5"
        depth_meta_path = camera_dir / "rendered_depth.meta.json"
        if not mask_path.is_file():
            continue
        _require(depth_path.is_file(), f"rendered depth is missing for {camera}")
        _require(
            depth_meta_path.is_file(),
            f"rendered-depth metadata is missing for {camera}",
        )
        with H5Array(mask_path) as stored:
            mask = np.asarray(stored[0])
        with H5Array(depth_path) as stored:
            depth_mm = np.asarray(stored[0])
        while mask.ndim > 2 and mask.shape[0] == 1:
            mask = mask[0]
        while depth_mm.ndim > 2 and depth_mm.shape[0] == 1:
            depth_mm = depth_mm[0]
        if mask.ndim != 2 or not np.any(mask):
            continue
        image = _read_frame_zero_image(camera_dir)
        _require(mask.shape == image.shape[:2], f"mask/image mismatch for {camera}")
        _require(mask.shape == depth_mm.shape, f"mask/depth mismatch for {camera}")
        masks[camera] = mask.astype(bool)
        images[camera] = image
        depths[camera] = depth_mm.astype(np.float64) * 0.001
        input_hashes[camera] = {
            "mask_sha256": _file_sha256(mask_path),
            "frame_zero_image_sha256": _array_sha256(image),
            "rendered_depth_sha256": _file_sha256(depth_path),
            "rendered_depth_meta_sha256": _file_sha256(depth_meta_path),
        }
    return (
        masks,
        images,
        depths,
        {camera: intrinsics[camera] for camera in masks},
        {camera: extrinsics[camera] for camera in masks},
        input_hashes,
    )


def _evaluate_case(
    record: dict[str, Any],
    *,
    staged_root: Path,
    failure_root: Path,
    output_root: Path,
    initializer_config: DepthSupportedFrameZeroInitializerConfig,
) -> dict[str, Any]:
    staged = staged_root / record["case"]
    failure_path = failure_root / record["case"] / "quality_failure.json"
    failure = _load_sealed_json(failure_path)
    _require(
        failure.get("stage") == "frame-zero-reconstruction"
        and failure.get("case") == record["case"],
        "quality failure identity changed",
    )
    episode_dir = staged / "frame-zero" / "episode_0000"
    splat_path = episode_dir / "splatfacto" / "splat_0.ply"
    _require(splat_path.is_file(), "failed frame-zero splat is missing")
    original_points, original_colors = pcd_stage.seed_points_from_splat(
        splat_path,
        crop_half_extent_m=pcd_stage.CROP_HALF_EXTENT_M,
        seed_count=pcd_stage.SEED_POINT_COUNT,
        rng_seed=0,
    )
    _require(
        not original_point_cloud_admissible(
            original_points,
            minimum_point_count=(
                initializer_config.visual_hull.minimum_original_point_count
            ),
        ),
        "known quality failure now passes the original gate",
    )
    masks, images, depths, intrinsics, extrinsics, inputs = (
        _load_frame_zero_views(episode_dir)
    )
    visual_hull = build_strict_multiview_surface(
        masks,
        images,
        intrinsics,
        extrinsics,
        config=initializer_config.visual_hull,
    )
    fallback = filter_surface_with_metric_depth(
        visual_hull,
        depths,
        intrinsics,
        extrinsics,
        visual_hull_cameras=masks,
        config=initializer_config,
    )
    selected = select_depth_supported_frame_zero_point_cloud(
        original_points,
        original_colors,
        lambda: fallback,
        config=initializer_config,
    )
    support, support_diagnostics = metric_depth_support_counts(
        selected.points_m,
        depths,
        intrinsics,
        extrinsics,
        depth_tolerance_m=initializer_config.depth_tolerance_m,
    )
    passed = bool(
        selected.method == "depth-supported-strict-multiview-surface"
        and len(selected.points_m)
        >= initializer_config.minimum_depth_supported_point_count
        and np.all(np.isfinite(selected.points_m))
        and np.all(np.isfinite(selected.colors))
        and np.all(support >= initializer_config.minimum_depth_support_views)
        and support_diagnostics["informative_camera_count"]
        >= initializer_config.minimum_depth_camera_count
    )
    case_output = output_root / record["case"]
    case_output.mkdir(parents=True, exist_ok=False)
    archive = case_output / "frame_zero_depth_fallback.npz"
    np.savez_compressed(
        archive,
        points_m=selected.points_m,
        colors=selected.colors,
    )
    return {
        **record,
        "passed": passed,
        "known_failure_result_sha256": failure["result_sha256"],
        "camera_count": len(masks),
        "informative_depth_camera_count": support_diagnostics[
            "informative_camera_count"
        ],
        "original_filtered_point_count": len(original_points),
        "original_points_sha256": _array_sha256(original_points),
        "visual_hull_point_count": len(visual_hull.points_m),
        "fallback_point_count": len(selected.points_m),
        "fallback_points_sha256": _array_sha256(selected.points_m),
        "fallback_colors_sha256": _array_sha256(selected.colors),
        "fallback_archive_sha256": _file_sha256(archive),
        "minimum_depth_support_views": int(support.min()),
        "median_depth_support_views": float(np.median(support)),
        "maximum_depth_support_views": int(support.max()),
        "retained_fraction": float(len(selected.points_m) / len(visual_hull.points_m)),
        "frame_zero_inputs": inputs,
        "information_boundary": {
            "object_observation_frames_used": [0],
            "future_object_rgb_used": False,
            "future_dense_geometry_used": False,
            "future_tracks_used": False,
            "outcome_artifact_used": False,
            "target_metric_used": False,
        },
    }


def main() -> int:
    args = _parse_args()
    repository = Path(__file__).resolve().parents[2]
    code_revision = _require_clean_repository(repository)
    config, config_sha256 = _load_config(args.config.resolve())
    candidate = config["source_candidate"]
    module_path = (
        repository
        / "src/bayesian_phystwin/deform360_frame_zero_depth_initializer.py"
    )
    _require(
        _file_sha256(module_path) == candidate["initializer_module_sha256"],
        "source-frozen initializer changed",
    )
    source_result = _load_sealed_json(
        repository
        / "results/sota/deform360_frame_zero_depth_initializer_source_v2"
        / "source_audit.json"
    )
    _require(
        source_result.get("source_gate_passed") is True
        and source_result.get("result_sha256") == candidate["source_result_sha256"],
        "source candidate did not pass its gate",
    )
    deform360_repo = args.deform360_repo.resolve()
    _require(
        _git_revision(deform360_repo)
        == source_result["provenance"]["deform360_revision"],
        "Deform360 revision changed",
    )
    records = sorted(config["cases"], key=lambda row: row["case"])
    _require(
        len(records) == config["gate"]["required_case_count"],
        "post-open case count changed",
    )
    output_root = args.output_root.resolve()
    _require(not output_root.exists(), "post-open output root already exists")
    output_root.mkdir(parents=True)
    initializer_config = _load_initializer_config(repository, config)
    results = []
    for record in records:
        print(f"auditing {record['case']}", flush=True)
        try:
            result = _evaluate_case(
                record,
                staged_root=args.staged_root.resolve(),
                failure_root=args.failure_root.resolve(),
                output_root=output_root,
                initializer_config=initializer_config,
            )
        except Exception as error:  # noqa: BLE001 - account for every case
            result = {
                **record,
                "passed": False,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        results.append(result)
    recovery_count = sum(bool(row["passed"]) for row in results)
    gate_passed = recovery_count == config["gate"]["required_recovery_count"]
    output: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360FrameZeroDepthPostOpenAudit",
        "protocol_id": config["protocol_id"],
        "protocol_config_sha256": config_sha256,
        "postopen_gate_passed": gate_passed,
        "summary": {
            "case_count": len(results),
            "recovery_count": recovery_count,
            "failure_count": len(results) - recovery_count,
        },
        "cases": results,
        "provenance": {
            "bayesian_phystwin_revision": code_revision,
            "initializer_module_sha256": _file_sha256(module_path),
            "runner_sha256": _file_sha256(Path(__file__).resolve()),
            "source_result_sha256": source_result["result_sha256"],
            "deform360_revision": _git_revision(deform360_repo),
        },
        "information_boundary": dict(config["information_boundary"]),
        "claim_boundary": config["claim_boundary"],
    }
    output["result_sha256"] = _canonical_sha256(output)
    output_path = output_root / "postopen_audit.json"
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    print(f"postopen_gate_passed={gate_passed}")
    print(f"result_sha256={output['result_sha256']}")
    return 0 if gate_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
