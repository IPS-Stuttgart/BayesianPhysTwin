#!/usr/bin/env python3
"""Score a sealed render-to-real AllTracker source competence prediction.

The prediction report and archive are fully validated before this process
opens the manual future identities or the CoTracker3 comparator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.rendered_alltracker_competence import (
    covariance_diagnostics,
    evaluate_competence_gates,
    shared_support_metrics,
    trajectory_metrics,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _validate_prediction(
    prediction_dir: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, str]]:
    report_path = prediction_dir / "prediction_report.json"
    prediction_path = prediction_dir / "prediction.npz"
    seal_path = prediction_dir / "PREDICTION_SEAL"
    for path in (report_path, prediction_path, seal_path):
        _require(path.is_file(), f"missing sealed prediction artifact {path}")
    report_sha = _sha256(report_path)
    _require(
        seal_path.read_text(encoding="ascii").strip() == report_sha,
        "prediction seal does not bind the report",
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(
        report.get("artifact_kind")
        == "PhysTwinRenderedAllTrackerPrefixPrediction",
        "prediction artifact kind changed",
    )
    _require(
        report.get("status") == "prediction_complete_unscored",
        "prediction was not sealed in the unscored state",
    )
    _require(
        report.get("prediction_sha256") == _sha256(prediction_path),
        "prediction archive differs from the sealed report",
    )
    for field in (
        "later_manual_identity_trajectory_read",
        "future_frame_after_120_read",
        "cotracker_comparator_read",
        "held_v8_read",
    ):
        _require(report.get(field) is False, f"prediction boundary failed: {field}")

    with np.load(prediction_path) as archive:
        prediction = {
            name: np.asarray(archive[name])
            for name in archive.files
        }
    required = {
        "frames",
        "query_point",
        "node_indices",
        "anchored_points_world_m",
        "covariance_m2",
        "valid",
    }
    missing = required.difference(prediction)
    _require(not missing, f"prediction archive lacks {sorted(missing)}")
    _require(
        report.get("prediction_points_sha256")
        == _array_sha256(prediction["anchored_points_world_m"]),
        "sealed point prediction changed",
    )
    _require(
        report.get("prediction_valid_sha256")
        == _array_sha256(prediction["valid"]),
        "sealed prediction support changed",
    )
    return report, prediction, {
        "prediction_report_sha256": report_sha,
        "prediction_archive_sha256": _sha256(prediction_path),
        "prediction_seal_sha256": _sha256(seal_path),
    }


def _load_cotracker_comparator(
    cues_path: Path,
    *,
    frames: np.ndarray,
    node_indices: np.ndarray,
    query_positions_m: np.ndarray,
    minimum_view_quality: float,
    maximum_reprojection_error_px: float,
    maximum_cycle_error_px: float,
    minimum_camera_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    required = {
        "multiview_points_world_m",
        "multiview_point_valid",
        "multiview_camera_count",
        "multiview_reprojection_error_px",
        "multiview_quality_probability_prefix",
        "forward_backward_valid",
        "forward_backward_error_px",
    }
    with np.load(cues_path) as archive:
        missing = required.difference(archive.files)
        _require(not missing, f"CoTracker3 archive lacks {sorted(missing)}")
        points = np.asarray(archive["multiview_points_world_m"], dtype=np.float64)
        point_valid = np.asarray(archive["multiview_point_valid"], dtype=bool)
        camera_count = np.asarray(
            archive["multiview_camera_count"],
            dtype=np.int64,
        )
        reprojection = np.asarray(
            archive["multiview_reprojection_error_px"],
            dtype=np.float64,
        )
        quality = np.asarray(
            archive["multiview_quality_probability_prefix"],
            dtype=np.float64,
        )
        cycle_valid = np.asarray(
            archive["forward_backward_valid"],
            dtype=bool,
        )
        cycle_error = np.asarray(
            archive["forward_backward_error_px"],
            dtype=np.float64,
        )

    _require(
        points.ndim == 3 and points.shape[2] == 3,
        "CoTracker3 world points must have shape (frame, track, 3)",
    )
    track_shape = points.shape[:2]
    for name, values in {
        "point validity": point_valid,
        "camera count": camera_count,
        "reprojection": reprojection,
        "cycle validity": cycle_valid,
        "cycle error": cycle_error,
    }.items():
        _require(values.shape == track_shape, f"CoTracker3 {name} shape changed")
    _require(
        quality.ndim == 3
        and quality.shape[1:] == track_shape,
        "CoTracker3 quality must have shape (camera, frame, track)",
    )
    _require(int(np.max(frames)) < points.shape[0], "CoTracker3 prefix is too short")
    _require(
        int(np.max(node_indices)) < points.shape[1],
        "physical node index exceeds the CoTracker3 identity inventory",
    )

    selected_points = points[np.ix_(frames, node_indices)]
    selected_point_valid = point_valid[np.ix_(frames, node_indices)]
    selected_camera_count = camera_count[np.ix_(frames, node_indices)]
    selected_reprojection = reprojection[np.ix_(frames, node_indices)]
    selected_cycle_valid = cycle_valid[np.ix_(frames, node_indices)]
    selected_cycle_error = cycle_error[np.ix_(frames, node_indices)]
    selected_quality = quality[:, frames][:, :, node_indices]

    initial_points = points[0, node_indices]
    initial_quality = quality[:, 0, node_indices]
    initial_valid = (
        point_valid[0, node_indices]
        & np.all(np.isfinite(initial_points), axis=1)
        & (camera_count[0, node_indices] >= minimum_camera_count)
        & np.isfinite(reprojection[0, node_indices])
        & (
            reprojection[0, node_indices]
            <= maximum_reprojection_error_px
        )
        & cycle_valid[0, node_indices]
        & np.isfinite(cycle_error[0, node_indices])
        & (cycle_error[0, node_indices] <= maximum_cycle_error_px)
        & np.all(
            np.isfinite(initial_quality)
            & (initial_quality >= minimum_view_quality),
            axis=0,
        )
    )
    valid = (
        initial_valid[None]
        & selected_point_valid
        & np.all(np.isfinite(selected_points), axis=2)
        & (selected_camera_count >= minimum_camera_count)
        & np.isfinite(selected_reprojection)
        & (selected_reprojection <= maximum_reprojection_error_px)
        & selected_cycle_valid
        & np.isfinite(selected_cycle_error)
        & (selected_cycle_error <= maximum_cycle_error_px)
        & np.all(
            np.isfinite(selected_quality)
            & (selected_quality >= minimum_view_quality),
            axis=0,
        )
    )
    anchored = (
        query_positions_m[None]
        + selected_points
        - initial_points[None]
    )
    anchored[~valid] = np.nan
    return anchored, valid, {
        "supported_count": int(np.sum(valid)),
        "initial_supported_identity_count": int(np.sum(initial_valid)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--manual-track", type=Path, required=True)
    parser.add_argument("--cotracker-cues", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-manual-track-sha256", required=True)
    parser.add_argument("--expected-cotracker-sha256", required=True)
    parser.add_argument("--minimum-support-fraction", type=float, default=0.5)
    parser.add_argument("--maximum-position-rmse-m", type=float, default=0.005)
    parser.add_argument("--maximum-final-frame-rmse-m", type=float, default=0.008)
    parser.add_argument(
        "--minimum-physical-improvement-fraction",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--minimum-cotracker-improvement-fraction",
        type=float,
        default=0.20,
    )
    parser.add_argument("--minimum-view-quality", type=float, default=0.5)
    parser.add_argument("--maximum-reprojection-error-px", type=float, default=3.0)
    parser.add_argument("--maximum-cycle-error-px", type=float, default=5.0)
    parser.add_argument("--minimum-camera-count", type=int, default=3)
    args = parser.parse_args()

    _require(not args.output.exists(), "score output already exists")
    _require(args.trajectory.is_file(), "physical trajectory is missing")
    report, prediction, prediction_hashes = _validate_prediction(
        args.prediction_dir
    )

    # The outcome boundary opens only after the sealed prediction has passed
    # every provenance check above.
    for path in (args.manual_track, args.cotracker_cues):
        _require(path.is_file(), f"missing scoring input {path}")
    manual_sha = _sha256(args.manual_track)
    cotracker_sha = _sha256(args.cotracker_cues)
    _require(
        manual_sha == args.expected_manual_track_sha256,
        "manual trajectory differs from the frozen target",
    )
    _require(
        cotracker_sha == args.expected_cotracker_sha256,
        "CoTracker3 archive differs from the frozen comparator",
    )

    frames = np.asarray(prediction["frames"], dtype=np.int64)
    query_point = np.asarray(prediction["query_point"], dtype=np.float64)
    query_positions = query_point[:, 1:]
    node_indices = np.asarray(prediction["node_indices"], dtype=np.int64)
    candidate = np.asarray(
        prediction["anchored_points_world_m"],
        dtype=np.float64,
    )
    candidate_valid = np.asarray(prediction["valid"], dtype=bool)
    covariance = np.asarray(prediction["covariance_m2"], dtype=np.float64)
    _require(
        np.array_equal(frames, np.asarray(report["frames"], dtype=np.int64)),
        "prediction frame inventory differs from the report",
    )

    with args.manual_track.open("rb") as handle:
        manual = np.asarray(pickle.load(handle), dtype=np.float64)
    _require(
        manual.ndim == 3
        and manual.shape[2] == 3
        and int(np.max(frames)) < manual.shape[0],
        "manual trajectory shape changed",
    )
    _require(
        manual.shape[1] == len(query_positions),
        "manual identity inventory differs from the oracle query inventory",
    )
    _require(
        np.allclose(
            manual[0],
            query_positions,
            atol=1e-6,
            rtol=0.0,
            equal_nan=False,
        ),
        "frame-zero oracle query order differs from the manual identities",
    )
    target = manual[frames]

    with args.trajectory.open("rb") as handle:
        trajectory = np.asarray(pickle.load(handle), dtype=np.float64)
    _require(
        trajectory.ndim == 3
        and trajectory.shape[2] == 3
        and int(np.max(frames)) < trajectory.shape[0]
        and int(np.max(node_indices)) < trajectory.shape[1],
        "physical trajectory shape changed",
    )
    physical = (
        query_positions[None]
        + trajectory[frames][:, node_indices]
        - trajectory[0, node_indices][None]
    )
    physical_valid = np.all(np.isfinite(physical), axis=2)

    cotracker, cotracker_valid, cotracker_summary = (
        _load_cotracker_comparator(
            args.cotracker_cues,
            frames=frames,
            node_indices=node_indices,
            query_positions_m=query_positions,
            minimum_view_quality=args.minimum_view_quality,
            maximum_reprojection_error_px=args.maximum_reprojection_error_px,
            maximum_cycle_error_px=args.maximum_cycle_error_px,
            minimum_camera_count=args.minimum_camera_count,
        )
    )

    candidate_metrics = trajectory_metrics(candidate, candidate_valid, target)
    final_frame_metrics = trajectory_metrics(
        candidate[-1:],
        candidate_valid[-1:],
        target[-1:],
    )
    physical_metrics = trajectory_metrics(physical, physical_valid, target)
    cotracker_metrics = trajectory_metrics(
        cotracker,
        cotracker_valid,
        target,
    )
    physical_shared = shared_support_metrics(
        candidate,
        candidate_valid,
        physical,
        physical_valid,
        target,
    )
    cotracker_shared = shared_support_metrics(
        candidate,
        candidate_valid,
        cotracker,
        cotracker_valid,
        target,
    )
    covariance_report = covariance_diagnostics(
        candidate,
        covariance,
        candidate_valid,
        target,
    )
    gates = evaluate_competence_gates(
        candidate_metrics,
        final_frame_metrics,
        physical_shared,
        cotracker_shared,
        minimum_support_fraction=args.minimum_support_fraction,
        maximum_position_rmse_m=args.maximum_position_rmse_m,
        maximum_final_frame_rmse_m=args.maximum_final_frame_rmse_m,
        minimum_physical_improvement_fraction=(
            args.minimum_physical_improvement_fraction
        ),
        minimum_cotracker_improvement_fraction=(
            args.minimum_cotracker_improvement_fraction
        ),
    )
    output = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinRenderedAllTrackerPrefixCompetenceResult",
        "case": "single_lift_cloth",
        "status": (
            "source_competence_gate_passed"
            if gates["competence_gate_passed"]
            else "source_competence_gate_failed"
        ),
        "frames": frames.tolist(),
        "candidate": candidate_metrics,
        "candidate_final_frame": final_frame_metrics,
        "physical_comparator": physical_metrics,
        "cotracker_comparator": cotracker_metrics,
        "candidate_vs_physical_shared_support": physical_shared,
        "candidate_vs_cotracker_shared_support": cotracker_shared,
        "cotracker_strict_support": cotracker_summary,
        "candidate_covariance_conditional_diagnostic": covariance_report,
        "gates": gates,
        "thresholds": {
            "minimum_support_fraction": args.minimum_support_fraction,
            "maximum_position_rmse_m": args.maximum_position_rmse_m,
            "maximum_final_frame_rmse_m": args.maximum_final_frame_rmse_m,
            "minimum_physical_improvement_fraction": (
                args.minimum_physical_improvement_fraction
            ),
            "minimum_cotracker_improvement_fraction": (
                args.minimum_cotracker_improvement_fraction
            ),
            "minimum_view_quality": args.minimum_view_quality,
            "maximum_reprojection_error_px": (
                args.maximum_reprojection_error_px
            ),
            "maximum_cycle_error_px": args.maximum_cycle_error_px,
            "minimum_camera_count": args.minimum_camera_count,
        },
        "inputs_sha256": {
            **prediction_hashes,
            "trajectory": _sha256(args.trajectory),
            "manual_track": manual_sha,
            "cotracker_cues": cotracker_sha,
        },
        "information_boundary": {
            "prediction_validated_before_outcome_open": True,
            "latest_rgb_frame": int(np.max(frames)),
            "future_rgb_after_prefix_read": False,
            "manual_identity_trajectory_opened_by_scorer": True,
            "cotracker_comparator_opened_by_scorer": True,
            "held_v8_read": False,
        },
        "claim_boundary": (
            "Opened-source, nine-identity frame-zero association-oracle "
            "competence control. It is neither deployable nor a SOTA result."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
