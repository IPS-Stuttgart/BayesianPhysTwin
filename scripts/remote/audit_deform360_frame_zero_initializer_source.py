#!/usr/bin/env python3
"""Audit the robust frame-zero fallback on the opened Deform360 source panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from bayesian_phystwin.deform360_frame_zero_initializer import (
    FrameZeroInitializerConfig,
    FrameZeroPointCloud,
    build_strict_multiview_surface,
    multiview_mask_votes,
    original_point_cloud_admissible,
    select_frame_zero_point_cloud,
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


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("schema_version") == 1, "config schema changed")
    config = payload.get("config")
    _require(isinstance(config, dict), "config payload is missing")
    observed = payload.get("config_sha256")
    expected = _canonical_sha256(config)
    _require(observed == expected, "config checksum changed")
    return config, expected


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
    except ImportError as error:  # pragma: no cover - host integration
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
    dict[str, Any],
]:
    intrinsics, extrinsics = load_episode_calibration(episode_dir)
    masks: dict[str, np.ndarray] = {}
    images: dict[str, np.ndarray] = {}
    inputs: dict[str, Any] = {}
    for camera in sorted(intrinsics):
        camera_dir = episode_dir / camera
        mask_path = camera_dir / "mask_refined.h5"
        if not mask_path.is_file():
            continue
        with H5Array(mask_path) as stored:
            mask = np.asarray(stored[0])
        while mask.ndim > 2 and mask.shape[0] == 1:
            mask = mask[0]
        if mask.ndim != 2 or not np.any(mask):
            continue
        image = _read_frame_zero_image(camera_dir)
        _require(mask.shape == image.shape[:2], f"mask/image mismatch for {camera}")
        masks[camera] = mask.astype(bool)
        images[camera] = image
        inputs[camera] = {
            "mask_sha256": _file_sha256(mask_path),
            "frame_zero_image_sha256": _array_sha256(image),
        }
    selected_intrinsics = {camera: intrinsics[camera] for camera in masks}
    selected_extrinsics = {camera: extrinsics[camera] for camera in masks}
    return masks, images, selected_intrinsics, selected_extrinsics, inputs


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
    except ImportError as error:  # pragma: no cover - host integration
        raise RuntimeError("SciPy is required for source geometry audit") from error
    first_to_second = cKDTree(second).query(first, k=1)[0]
    second_to_first = cKDTree(first).query(second, k=1)[0]
    return 0.5 * (float(np.mean(first_to_second)) + float(np.mean(second_to_first)))


def _initializer_config(config: dict[str, Any]) -> FrameZeroInitializerConfig:
    method = config["method"]
    fields = FrameZeroInitializerConfig.__dataclass_fields__
    return FrameZeroInitializerConfig(
        **{name: method[name] for name in fields}
    )


def _evaluate_case(
    record: dict[str, Any],
    *,
    staged_root: Path,
    initializer_config: FrameZeroInitializerConfig,
    minimum_component_fraction: float,
) -> dict[str, Any]:
    staged_case = staged_root / record["case"]
    episode_dir = staged_case / "frame-zero" / "episode_0000"
    _require(staged_case.is_dir(), f"staged source case is missing: {record['case']}")
    _require(episode_dir.is_dir(), f"frame-zero episode is missing: {record['case']}")
    original_points, original_colors, original_path = _load_original_cloud(staged_case)
    original_admitted = original_point_cloud_admissible(
        original_points,
        minimum_point_count=initializer_config.minimum_original_point_count,
    )

    def forbidden_fallback() -> FrameZeroPointCloud:
        raise RuntimeError("admitted source path unexpectedly evaluated the fallback")

    exact_original_parity: bool | None = None
    if original_admitted:
        selected_original = select_frame_zero_point_cloud(
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
    masks, images, intrinsics, extrinsics, view_inputs = _load_frame_zero_views(
        episode_dir
    )
    fallback = build_strict_multiview_surface(
        masks,
        images,
        intrinsics,
        extrinsics,
        config=initializer_config,
    )
    trigger_count = (
        len(original_points)
        if not original_admitted
        else initializer_config.minimum_original_point_count - 1
    )
    forced = select_frame_zero_point_cloud(
        original_points[:trigger_count],
        original_colors[:trigger_count],
        lambda: fallback,
        config=initializer_config,
    )
    selected_attempt = fallback.diagnostics["attempts"][-1]
    required_votes = int(selected_attempt["required_vote_count"])
    votes, support = multiview_mask_votes(
        fallback.points_m,
        masks,
        intrinsics,
        extrinsics,
        mask_dilation_radius_pixels=int(
            fallback.diagnostics["selected_mask_dilation_radius_pixels"]
        ),
    )
    largest_fraction = float(
        selected_attempt["components"]["largest_component_fraction"]
    )
    strict_support = bool(np.all(votes >= required_votes))
    passed = bool(
        (not original_admitted or exact_original_parity)
        and forced.method == "strict-multiview-visual-hull-surface"
        and len(forced.points_m) >= initializer_config.minimum_fallback_point_count
        and np.all(np.isfinite(forced.points_m))
        and np.all(np.isfinite(forced.colors))
        and strict_support
        and largest_fraction >= minimum_component_fraction
    )
    return {
        **record,
        "passed": passed,
        "camera_count": len(masks),
        "original_point_count": len(original_points),
        "original_admitted": original_admitted,
        "original_geometry_sha256": _array_sha256(original_points),
        "original_geometry": {
            "path": str(original_path),
            "sha256": _file_sha256(original_path),
            "frame_zero_only": True,
        },
        "admitted_original_exact_parity": exact_original_parity,
        "forced_fallback": {
            "trigger_point_count": trigger_count,
            "trigger_is_natural_original_failure": not original_admitted,
            "method": forced.method,
            "point_count": len(forced.points_m),
            "points_sha256": _array_sha256(forced.points_m),
            "colors_sha256": _array_sha256(forced.colors),
            "finite": bool(
                np.all(np.isfinite(forced.points_m))
                and np.all(np.isfinite(forced.colors))
            ),
            "strict_support": strict_support,
            "minimum_observed_vote_count": int(votes.min()),
            "required_vote_count": required_votes,
            "largest_component_fraction": largest_fraction,
            "symmetric_chamfer_to_original_m": _symmetric_chamfer_m(
                forced.points_m,
                original_points,
            ),
            "diagnostics": fallback.diagnostics,
            "support_diagnostics": support,
        },
        "frame_zero_view_inputs": view_inputs,
        "information_boundary": {
            "object_observation_frames_used": [0],
            "future_source_geometry_used": False,
            "future_source_rgb_used": False,
            "prospective_calibration_cases_used": False,
            "prospective_target_cases_used": False,
            "target_metric_used": False,
        },
    }


def main() -> int:
    args = _parse_args()
    repository = Path(__file__).resolve().parents[2]
    code_revision = _require_clean_repository(repository)
    config, config_sha256 = _load_config(args.config.resolve())
    deform360_repo = args.deform360_repo.resolve()
    _require(
        _git_revision(deform360_repo) == config["dataset"]["deform360_code_revision"],
        "Deform360 code revision changed",
    )
    records = _case_records(config)
    gates = config["source_gates"]
    _require(len(records) == gates["required_case_count"], "source case count changed")
    initializer_config = _initializer_config(config)
    results = []
    for record in records:
        print(f"auditing {record['case']}", flush=True)
        try:
            result = _evaluate_case(
                record,
                staged_root=args.staged_root.resolve(),
                initializer_config=initializer_config,
                minimum_component_fraction=float(
                    gates["fallback_largest_component_fraction_at_least"]
                ),
            )
        except Exception as error:  # noqa: BLE001 - preserve every source failure
            result = {
                **record,
                "passed": False,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        results.append(result)
    completed = [row for row in results if "forced_fallback" in row]
    objects = sorted({row["object_id"] for row in results})
    strata = sorted({row["stratum"] for row in results})
    admitted_count = sum(bool(row.get("original_admitted")) for row in results)
    exact_count = sum(
        bool(row.get("admitted_original_exact_parity")) for row in results
    )
    natural_failure_count = sum(
        row.get("original_admitted") is False for row in results
    )
    natural_recovery_count = sum(
        row.get("original_admitted") is False
        and row.get("forced_fallback", {}).get("method")
        == "strict-multiview-visual-hull-surface"
        for row in results
    )
    fallback_count = len(completed)
    passing_count = sum(bool(row["passed"]) for row in results)
    largest_minimum = min(
        (
            row["forced_fallback"]["largest_component_fraction"]
            for row in completed
        ),
        default=0.0,
    )
    source_gate_passed = bool(
        len(results) == gates["required_case_count"]
        and len(objects) == gates["required_object_count"]
        and strata == sorted(gates["required_strata"])
        and exact_count == admitted_count
        and natural_recovery_count == natural_failure_count
        and fallback_count == len(results)
        and passing_count == len(results)
        and largest_minimum
        >= gates["fallback_largest_component_fraction_at_least"]
    )
    chamfers = [
        row["forced_fallback"]["symmetric_chamfer_to_original_m"]
        for row in completed
    ]
    output: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360FrameZeroInitializerSourceAudit",
        "protocol_id": config["protocol_id"],
        "protocol_config_sha256": config_sha256,
        "source_gate_passed": source_gate_passed,
        "summary": {
            "case_count": len(results),
            "object_count": len(objects),
            "strata": strata,
            "admitted_original_case_count": admitted_count,
            "exact_original_parity_count": exact_count,
            "natural_original_failure_count": natural_failure_count,
            "natural_original_failure_recovery_count": natural_recovery_count,
            "forced_fallback_build_count": fallback_count,
            "passing_case_count": passing_count,
            "minimum_largest_component_fraction": largest_minimum,
            "median_symmetric_chamfer_to_original_m": (
                float(np.median(chamfers)) if chamfers else None
            ),
            "maximum_symmetric_chamfer_to_original_m": (
                float(np.max(chamfers)) if chamfers else None
            ),
        },
        "cases": results,
        "provenance": {
            "config_path": str(args.config.resolve()),
            "config_file_sha256": _file_sha256(args.config.resolve()),
            "bayesian_phystwin_revision": code_revision,
            "initializer_module_sha256": _file_sha256(
                repository
                / "src"
                / "bayesian_phystwin"
                / "deform360_frame_zero_initializer.py"
            ),
            "deform360_revision": _git_revision(deform360_repo),
            "runner_sha256": _file_sha256(Path(__file__).resolve()),
        },
        "information_boundary": dict(config["information_boundary"]),
        "claim_boundary": config["claim_boundary"],
    }
    output["result_sha256"] = _canonical_sha256(output)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    print(f"source_gate_passed={source_gate_passed}")
    print(f"result_sha256={output['result_sha256']}")
    return 0 if source_gate_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
