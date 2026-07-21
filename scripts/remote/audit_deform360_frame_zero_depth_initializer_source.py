#!/usr/bin/env python3
"""Audit the frame-zero depth-supported fallback on the open source panel."""

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
    FrameZeroPointCloud,
    build_strict_multiview_surface,
    original_point_cloud_admissible,
)
from deform360.annotations import H5Array
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
    parser.add_argument("--deform360-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_config(path: Path) -> tuple[dict[str, Any], str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("schema_version") == 1, "config schema changed")
    protocol_id = str(payload.get("protocol_id"))
    _require(
        protocol_id == "deform360-frame-zero-depth-initializer-source-v2",
        "unexpected protocol id",
    )
    config = payload.get("config")
    _require(isinstance(config, dict), "config payload is missing")
    observed = payload.get("config_sha256")
    expected = _canonical_sha256(config)
    _require(observed == expected, "config checksum changed")
    return config, expected, protocol_id


def _case_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for stratum, objects in config["source_objects"].items():
        for object_id, episodes in objects.items():
            for episode_id in episodes:
                records.append(
                    {
                        "stratum": str(stratum),
                        "object_id": str(object_id),
                        "episode_id": int(episode_id),
                        "case": f"{object_id}-ep{int(episode_id):04d}",
                    }
                )
    return sorted(records, key=lambda row: row["case"])


def _read_frame_zero_image(camera_dir: Path) -> np.ndarray:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - integration dependency
        raise RuntimeError("OpenCV is required for source geometry audit") from error
    still = camera_dir / "undistorted_000000.png"
    if still.is_file():
        image = cv2.imread(str(still), cv2.IMREAD_COLOR)
        _require(image is not None, f"cannot read {still}")
        return image
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
    inputs: dict[str, Any] = {}
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
        inputs[camera] = {
            "mask_sha256": _file_sha256(mask_path),
            "frame_zero_image_sha256": _array_sha256(image),
            "rendered_depth_sha256": _file_sha256(depth_path),
            "rendered_depth_meta_sha256": _file_sha256(depth_meta_path),
        }
    selected_intrinsics = {camera: intrinsics[camera] for camera in masks}
    selected_extrinsics = {camera: extrinsics[camera] for camera in masks}
    return (
        masks,
        images,
        depths,
        selected_intrinsics,
        selected_extrinsics,
        inputs,
    )


def _load_original_cloud(staged_case: Path) -> tuple[np.ndarray, np.ndarray, Path]:
    path = staged_case / "frame_zero_points.npz"
    _require(path.is_file(), f"sealed frame-zero geometry is missing in {staged_case}")
    with np.load(path, allow_pickle=False) as stored:
        _require({"points_m", "colors"} <= set(stored.files), "geometry is incomplete")
        points = np.asarray(stored["points_m"])
        colors = np.asarray(stored["colors"])
    _require(
        points.ndim == 2
        and points.shape[1:] == (3,)
        and colors.shape == points.shape,
        "sealed frame-zero geometry is invalid",
    )
    return points, colors, path


def _symmetric_chamfer_m(first: np.ndarray, second: np.ndarray) -> float:
    try:
        from scipy.spatial import cKDTree
    except ImportError as error:  # pragma: no cover - integration dependency
        raise RuntimeError("SciPy is required for source geometry audit") from error
    first_to_second = cKDTree(second).query(first, k=1)[0]
    second_to_first = cKDTree(first).query(second, k=1)[0]
    return 0.5 * (float(np.mean(first_to_second)) + float(np.mean(second_to_first)))


def _initializer_config(
    config: dict[str, Any],
) -> DepthSupportedFrameZeroInitializerConfig:
    method = config["method"]
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


def _evaluate_case(
    record: dict[str, Any],
    *,
    staged_root: Path,
    initializer_config: DepthSupportedFrameZeroInitializerConfig,
) -> dict[str, Any]:
    staged_case = staged_root / record["case"]
    episode_dir = staged_case / "frame-zero" / "episode_0000"
    _require(staged_case.is_dir(), f"staged source case is missing: {record['case']}")
    _require(episode_dir.is_dir(), f"frame-zero episode is missing: {record['case']}")
    original_points, original_colors, original_path = _load_original_cloud(staged_case)
    original_admitted = original_point_cloud_admissible(
        original_points,
        minimum_point_count=(
            initializer_config.visual_hull.minimum_original_point_count
        ),
    )

    def forbidden_fallback() -> FrameZeroPointCloud:
        raise RuntimeError("admitted source path unexpectedly evaluated the fallback")

    exact_original_parity: bool | None = None
    if original_admitted:
        selected_original = select_depth_supported_frame_zero_point_cloud(
            original_points,
            original_colors,
            forbidden_fallback,
            config=initializer_config,
        )
        exact_original_parity = bool(
            selected_original.method == "original-splat"
            and selected_original.points_m.dtype == original_points.dtype
            and selected_original.colors.dtype == original_colors.dtype
            and selected_original.points_m.tobytes() == original_points.tobytes()
            and selected_original.colors.tobytes() == original_colors.tobytes()
        )

    masks, images, depths, intrinsics, extrinsics, view_inputs = (
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
    trigger_count = (
        len(original_points)
        if not original_admitted
        else initializer_config.visual_hull.minimum_original_point_count - 1
    )
    forced = select_depth_supported_frame_zero_point_cloud(
        original_points[:trigger_count],
        original_colors[:trigger_count],
        lambda: fallback,
        config=initializer_config,
    )
    support, support_diagnostics = metric_depth_support_counts(
        forced.points_m,
        depths,
        intrinsics,
        extrinsics,
        depth_tolerance_m=initializer_config.depth_tolerance_m,
    )
    visual_hull_cd = _symmetric_chamfer_m(visual_hull.points_m, original_points)
    depth_supported_cd = _symmetric_chamfer_m(forced.points_m, original_points)
    paired_improvement = visual_hull_cd - depth_supported_cd
    passed = bool(
        (not original_admitted or exact_original_parity)
        and forced.method == "depth-supported-strict-multiview-surface"
        and len(forced.points_m)
        >= initializer_config.minimum_depth_supported_point_count
        and np.all(np.isfinite(forced.points_m))
        and np.all(np.isfinite(forced.colors))
        and np.all(support >= initializer_config.minimum_depth_support_views)
        and support_diagnostics["informative_camera_count"]
        >= initializer_config.minimum_depth_camera_count
    )
    return {
        **record,
        "passed": passed,
        "camera_count": len(masks),
        "informative_depth_camera_count": support_diagnostics[
            "informative_camera_count"
        ],
        "original_point_count": len(original_points),
        "original_admitted": original_admitted,
        "admitted_original_exact_parity": exact_original_parity,
        "original_geometry": {
            "path": str(original_path),
            "sha256": _file_sha256(original_path),
            "points_sha256": _array_sha256(original_points),
            "frame_zero_only": True,
        },
        "forced_fallback": {
            "trigger_point_count": trigger_count,
            "trigger_is_natural_original_failure": not original_admitted,
            "method": forced.method,
            "visual_hull_point_count": len(visual_hull.points_m),
            "point_count": len(forced.points_m),
            "points_sha256": _array_sha256(forced.points_m),
            "minimum_support_views": int(support.min()),
            "median_support_views": float(np.median(support)),
            "maximum_support_views": int(support.max()),
            "retained_fraction": float(len(forced.points_m) / len(visual_hull.points_m)),
            "visual_hull_symmetric_chamfer_to_original_m": visual_hull_cd,
            "depth_supported_symmetric_chamfer_to_original_m": depth_supported_cd,
            "paired_symmetric_chamfer_improvement_m": paired_improvement,
            "depth_support": support_diagnostics,
        },
        "frame_zero_view_inputs": view_inputs,
    }


def _validate_predecessor(config: dict[str, Any], repository: Path) -> None:
    predecessor = config["predecessor"]
    source_config_path = (
        repository / "configs/sota/deform360_frame_zero_initializer_source_v1.json"
    )
    source_result_path = (
        repository
        / "results/sota/deform360_frame_zero_initializer_source_v1/source_audit.json"
    )
    source_module_path = (
        repository
        / "src/bayesian_phystwin/deform360_frame_zero_initializer.py"
    )
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
    _require(
        source_config["config_sha256"] == predecessor["source_config_sha256"],
        "predecessor source config changed",
    )
    _require(
        source_result["result_sha256"] == predecessor["source_result_sha256"],
        "predecessor source result changed",
    )
    _require(
        _file_sha256(source_module_path)
        == predecessor["source_initializer_module_sha256"],
        "predecessor initializer module changed",
    )


def main() -> int:
    args = _parse_args()
    repository = Path(__file__).resolve().parents[2]
    revision = _require_clean_repository(repository)
    config, config_sha256, protocol_id = _load_config(args.config)
    _validate_predecessor(config, repository)
    _require(
        not config["information_boundary"]["reserved_target_cases_read"],
        "reserved target cases must remain sealed",
    )
    initializer_config = _initializer_config(config)
    records = _case_records(config)
    cases = [
        _evaluate_case(
            record,
            staged_root=args.staged_root,
            initializer_config=initializer_config,
        )
        for record in records
    ]

    gates = config["source_gates"]
    improvements = np.asarray(
        [
            case["forced_fallback"]["paired_symmetric_chamfer_improvement_m"]
            for case in cases
        ],
        dtype=np.float64,
    )
    exact_parity_count = sum(
        case["admitted_original_exact_parity"] is True for case in cases
    )
    natural_failures = [case for case in cases if not case["original_admitted"]]
    natural_recovery_count = sum(case["passed"] for case in natural_failures)
    forced_success_count = sum(case["passed"] for case in cases)
    improved_count = int(np.count_nonzero(improvements > 0.0))
    maximum_regression = float(max(0.0, -float(np.min(improvements))))
    median_improvement = float(np.median(improvements))
    minimum_retained = min(case["forced_fallback"]["point_count"] for case in cases)
    source_gate_passed = bool(
        len(cases) >= gates["minimum_case_count"]
        and exact_parity_count
        >= gates["minimum_admitted_original_exact_parity_count"]
        and forced_success_count >= gates["minimum_forced_fallback_success_count"]
        and natural_recovery_count >= gates["minimum_natural_failure_recovery_count"]
        and minimum_retained >= gates["minimum_retained_point_count"]
        and improved_count >= gates["minimum_improved_symmetric_chamfer_case_count"]
        and median_improvement
        >= gates["minimum_median_paired_symmetric_chamfer_improvement_m"]
        and maximum_regression
        <= gates["maximum_allowed_symmetric_chamfer_regression_m"]
    )
    module_path = (
        repository
        / "src/bayesian_phystwin/deform360_frame_zero_depth_initializer.py"
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360FrameZeroDepthInitializerSourceAudit",
        "protocol_id": protocol_id,
        "protocol_config_sha256": config_sha256,
        "source_gate_passed": source_gate_passed,
        "summary": {
            "case_count": len(cases),
            "object_count": len({case["object_id"] for case in cases}),
            "strata": sorted({case["stratum"] for case in cases}),
            "admitted_original_case_count": sum(
                case["original_admitted"] for case in cases
            ),
            "exact_original_parity_count": exact_parity_count,
            "natural_original_failure_count": len(natural_failures),
            "natural_original_failure_recovery_count": natural_recovery_count,
            "forced_fallback_success_count": forced_success_count,
            "minimum_retained_point_count": minimum_retained,
            "improved_symmetric_chamfer_case_count": improved_count,
            "median_paired_symmetric_chamfer_improvement_m": median_improvement,
            "mean_paired_symmetric_chamfer_improvement_m": float(
                np.mean(improvements)
            ),
            "maximum_symmetric_chamfer_regression_m": maximum_regression,
        },
        "cases": cases,
        "provenance": {
            "bayesian_phystwin_revision": revision,
            "deform360_revision": _git_revision(args.deform360_repo),
            "config_path": str(args.config.resolve()),
            "config_file_sha256": _file_sha256(args.config),
            "initializer_module_sha256": _file_sha256(module_path),
            "runner_sha256": _file_sha256(Path(__file__).resolve()),
            "predecessor_source_result_sha256": config["predecessor"][
                "source_result_sha256"
            ],
            "predecessor_physical_result_sha256": config["predecessor"][
                "physical_result_sha256"
            ],
        },
        "information_boundary": config["information_boundary"],
        "claim_boundary": config["claim_boundary"],
    }
    result["result_sha256"] = _canonical_sha256(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "source_gate_passed": source_gate_passed,
                "summary": result["summary"],
                "result_sha256": result["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if source_gate_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
