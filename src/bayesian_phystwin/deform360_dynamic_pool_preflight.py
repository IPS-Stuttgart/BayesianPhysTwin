"""Outcome-blind frame-zero feasibility for the dynamic Open27 pool.

This module runs beside the processed source media before AllTracker. It reads
only sealed physical predictions, calibration, and HDF5 frame zero. The output
both closes the frozen arm when 64 multiview candidates are unavailable and
defines the minimal camera payload that may be staged for causal tracking.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .deform360_online_belief_evaluation import _resolve_prediction_archive, _sha256
from .deform360_raw_camera_observation import (
    RawCameraObservationConfig,
    _array_sha256,
    _canonical_sha256,
    _load_calibration,
    _read_h5_frame_zero,
    _validate_prediction_seal,
    expected_open_case_names,
    frame_zero_camera_support,
    select_frame_zero_observation_plan,
)

PROTOCOL_ID = "deform360-dynamic-pairwise-pool-preflight-v1"
PREFLIGHT_FILENAME = "pool_preflight.json"
FROZEN_CENTER_COUNT = 64
FROZEN_SELECTED_CAMERA_COUNT = 8
CAMERA_STAGING_FILENAMES = (
    "aligned_timestamps.txt",
    "mask_refined.h5",
    "metadata.json",
    "rendered_depth.h5",
    "rendered_depth.meta.json",
    "undistorted.mp4",
)


def frozen_preflight_config() -> RawCameraObservationConfig:
    """Return the exact frame-zero choices bound by the source protocol."""

    return RawCameraObservationConfig(
        center_count=FROZEN_CENTER_COUNT,
        selected_camera_count=FROZEN_SELECTED_CAMERA_COUNT,
    )


def _validate_frozen_config(config: RawCameraObservationConfig) -> None:
    if config.center_count != FROZEN_CENTER_COUNT:
        raise ValueError("dynamic-pool preflight requires exactly 64 centers")
    if config.selected_camera_count != FROZEN_SELECTED_CAMERA_COUNT:
        raise ValueError("dynamic-pool preflight requires exactly eight cameras")


def _frame_zero_camera_inputs(
    processed_episode_dir: Path,
    cameras: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    inputs: dict[str, dict[str, Any]] = {}
    for camera in cameras:
        camera_dir = processed_episode_dir / camera
        mask = _read_h5_frame_zero(camera_dir / "mask_refined.h5")
        depth = _read_h5_frame_zero(camera_dir / "rendered_depth.h5")
        inputs[camera] = {
            "frame_zero_mask_array_sha256": _array_sha256(mask),
            "frame_zero_depth_array_sha256": _array_sha256(depth),
            "hdf5_indices_read": [0],
            "future_hdf5_slice_read": False,
            "video_content_read": False,
        }
    return inputs


def _staging_paths(
    case: str,
    processed_episode_dir: Path,
    selected_cameras: tuple[str, ...],
) -> list[str]:
    paths = [
        f"{case}/episode_0000/extrinsics.npy",
        f"{case}/episode_0000/undistorted_intrinsics.npy",
    ]
    for camera in selected_cameras:
        for filename in CAMERA_STAGING_FILENAMES:
            path = processed_episode_dir / camera / filename
            if not path.is_file():
                raise FileNotFoundError(f"missing selected-camera input: {path}")
            paths.append(f"{case}/episode_0000/{camera}/{filename}")
    return sorted(paths)


def _stable_selection_score(values: tuple[int, int, int, float]) -> list[int | float]:
    """Remove backend-level roundoff from diagnostic-only ray angles."""

    return [
        round(float(value), 12)
        if isinstance(value, (float, np.floating))
        else int(value)
        for value in values
    ]


def preflight_dynamic_pool_case(
    panel_case_dir: str | Path,
    processed_episode_dir: str | Path,
    *,
    config: RawCameraObservationConfig | None = None,
) -> dict[str, Any]:
    """Evaluate one frame-zero contract without reading outcomes or RGB frames."""

    cfg = config or frozen_preflight_config()
    _validate_frozen_config(cfg)
    case_dir = Path(panel_case_dir).resolve()
    processed = Path(processed_episode_dir).resolve()
    if case_dir.name not in set(expected_open_case_names()):
        raise ValueError("case is outside the immutable Open27 source panel")

    seal_path = case_dir / "prediction_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    _validate_prediction_seal(seal)
    archive_path = _resolve_prediction_archive(case_dir, seal)
    with np.load(archive_path, allow_pickle=False) as stored:
        if "frame_zero_points_m" not in stored.files:
            raise ValueError("sealed prediction lacks frame-zero geometry")
        frame_zero = np.asarray(stored["frame_zero_points_m"], dtype=np.float64)
    if frame_zero.ndim != 2 or frame_zero.shape[1] != 3:
        raise ValueError("sealed frame-zero geometry must have shape (N, 3)")
    if len(frame_zero) < FROZEN_CENTER_COUNT:
        raise ValueError("physical state has fewer points than the frozen pool")

    intrinsics, extrinsics = _load_calibration(processed)
    cameras, support, projected = frame_zero_camera_support(
        frame_zero,
        processed,
        intrinsics,
        extrinsics,
        depth_tolerance_m=cfg.frame_zero_depth_tolerance_m,
    )
    plan = select_frame_zero_observation_plan(
        frame_zero,
        cameras,
        support,
        projected,
        extrinsics,
        config=cfg,
    )
    candidates = np.asarray(plan["candidate_ids"], dtype=np.int64)
    centers = np.asarray(plan["center_ids"], dtype=np.int64)
    selected_cameras = tuple(str(value) for value in plan["selected_cameras"])
    if len(centers) != FROZEN_CENTER_COUNT:
        raise AssertionError("frame-zero selector returned a nonfrozen pool size")

    return {
        "case": case_dir.name,
        "status": "passed",
        "object_id": str(seal["object_id"]),
        "episode_id": int(seal["episode_id"]),
        "episode_key": str(seal["episode_key"]),
        "physical_node_count": int(len(frame_zero)),
        "candidate_count": int(len(candidates)),
        "candidate_ids": candidates.tolist(),
        "center_ids": centers.tolist(),
        "selected_cameras": list(selected_cameras),
        "selection_score": _stable_selection_score(plan["selection_score"]),
        "staging_relative_paths": _staging_paths(
            case_dir.name,
            processed,
            selected_cameras,
        ),
        "inputs": {
            "prediction_seal_sha256": _sha256(seal_path),
            "prediction_archive_sha256": _sha256(archive_path),
            "intrinsics_sha256": _sha256(processed / "undistorted_intrinsics.npy"),
            "extrinsics_sha256": _sha256(processed / "extrinsics.npy"),
            "frame_zero_camera_inputs": _frame_zero_camera_inputs(
                processed,
                cameras,
            ),
        },
        "information_boundary": {
            "target_data_read": False,
            "outcome_manifest_read": False,
            "rgb_frame_read": False,
            "future_reconstruction_after_frame_zero_read": False,
            "hdf5_indices_read": [0],
        },
    }


def run_dynamic_pool_preflight(
    panel_root: str | Path,
    processed_root: str | Path,
    output_dir: str | Path,
    *,
    config: RawCameraObservationConfig | None = None,
) -> dict[str, Any]:
    """Run all 27 source cases and write one immutable feasibility artifact."""

    cfg = config or frozen_preflight_config()
    _validate_frozen_config(cfg)
    panel = Path(panel_root).resolve()
    processed = Path(processed_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)

    cases: list[dict[str, Any]] = []
    for case in expected_open_case_names():
        try:
            record = preflight_dynamic_pool_case(
                panel / case,
                processed / case / "episode_0000",
                config=cfg,
            )
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
            record = {
                "case": case,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        cases.append(record)

    passed = [record for record in cases if record["status"] == "passed"]
    staging_paths = sorted(
        {
            path
            for record in passed
            for path in record["staging_relative_paths"]
        }
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360DynamicPoolFrameZeroPreflight",
        "protocol_id": PROTOCOL_ID,
        "config": asdict(cfg),
        "expected_case_count": len(expected_open_case_names()),
        "passed_case_count": len(passed),
        "failed_case_count": len(cases) - len(passed),
        "preflight_gate_passed": len(passed) == len(cases),
        "cases": cases,
        "staging_relative_paths": staging_paths,
        "information_boundary": {
            "source_panel_only": True,
            "target_data_read": False,
            "outcome_manifest_read": False,
            "rgb_frame_read": False,
            "future_reconstruction_after_frame_zero_read": False,
            "hdf5_indices_read": [0],
        },
        "claim_boundary": (
            "target-free operational feasibility for the already-open Open27 "
            "source panel; no tracker competence or outcome claim"
        ),
    }
    result["result_sha256"] = _canonical_sha256(result)
    (output / PREFLIGHT_FILENAME).write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


__all__ = [
    "CAMERA_STAGING_FILENAMES",
    "FROZEN_CENTER_COUNT",
    "FROZEN_SELECTED_CAMERA_COUNT",
    "PREFLIGHT_FILENAME",
    "PROTOCOL_ID",
    "frozen_preflight_config",
    "preflight_dynamic_pool_case",
    "run_dynamic_pool_preflight",
]
