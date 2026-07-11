"""Training-only camera selection for MotionCrafter graph association."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _optional_mean(values: list[float | None]) -> float | None:
    finite = np.asarray([value for value in values if value is not None], dtype=float)
    finite = finite[np.isfinite(finite)]
    return None if len(finite) == 0 else float(np.mean(finite))


def _future_correlation(summary: dict[str, Any]) -> float | None:
    automatic = np.asarray(
        summary["released_dense_track_error"]["by_sampled_frame_m"], dtype=float
    )
    manual = np.asarray(
        summary["manual_identity_audit"].get(
            "error_by_sampled_frame_m", np.full(len(automatic), np.nan)
        ),
        dtype=float,
    )
    frame_indices = np.asarray(summary["frame_indices"], dtype=int)
    usable = (
        (frame_indices >= int(summary["train_end_frame"]))
        & np.isfinite(automatic)
        & np.isfinite(manual)
    )
    if np.sum(usable) < 3:
        return None
    automatic_selected = automatic[usable]
    manual_selected = manual[usable]
    if np.std(automatic_selected) == 0.0 or np.std(manual_selected) == 0.0:
        return None
    return float(np.corrcoef(automatic_selected, manual_selected)[0, 1])


def _summarize_candidate(path: str | Path) -> dict[str, Any]:
    summary_path = Path(path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("alignment", {}).get("view_count", 1)) != 1:
        raise ValueError("camera selection requires one-view association summaries")
    frame_indices = np.asarray(summary["frame_indices"], dtype=int)
    training = np.flatnonzero(frame_indices < int(summary["train_end_frame"]))
    if len(training) == 0:
        raise ValueError("association summary has no training frames")
    coverage = np.asarray(
        summary["graph"]["valid_vertex_fraction_by_sampled_frame"], dtype=float
    )
    training_error = float(
        summary["released_dense_track_error"]["training_mean_m"]
    )
    endpoint_coverage = float(coverage[training[-1]])
    initial_error = float(summary["graph"]["association_initial_error_m"]["mean"])
    training_motion_error = float(
        summary["graph"]["training_motion_error_m"]["mean"]
    )
    future_coverage = coverage[frame_indices >= int(summary["train_end_frame"])]
    manual = summary.get("manual_identity_audit", {})
    automatic_by_frame = np.asarray(
        summary["released_dense_track_error"]["by_sampled_frame_m"], dtype=float
    )
    manual_by_frame = np.asarray(
        manual.get(
            "error_by_sampled_frame_m", np.full(len(frame_indices), np.nan)
        ),
        dtype=float,
    )
    future_mask = frame_indices >= int(summary["train_end_frame"])
    future_frame_audit = [
        {
            "frame_index": int(frame),
            "dense_error_m": (
                None if not np.isfinite(dense_error) else float(dense_error)
            ),
            "graph_coverage": float(frame_coverage),
            "audit_only_manual_error_m": (
                None if not np.isfinite(manual_error) else float(manual_error)
            ),
        }
        for frame, dense_error, frame_coverage, manual_error in zip(
            frame_indices[future_mask],
            automatic_by_frame[future_mask],
            coverage[future_mask],
            manual_by_frame[future_mask],
            strict=True,
        )
    ]
    return {
        "case": str(summary["case"]),
        "camera_index": int(summary["config"]["camera_index"]),
        "selection_score_m": training_error / max(endpoint_coverage, 1e-6),
        "perception_only_sensitivity_score_m": (
            initial_error + training_motion_error
        )
        / max(endpoint_coverage, 1e-6),
        "training_dense_error_m": training_error,
        "association_initial_error_m": initial_error,
        "training_motion_error_m": training_motion_error,
        "training_endpoint_graph_coverage": endpoint_coverage,
        "future_dense_error_m": summary["released_dense_track_error"][
            "future_mean_m"
        ],
        "future_mean_graph_coverage": (
            None if len(future_coverage) == 0 else float(np.mean(future_coverage))
        ),
        "audit_only_manual_training_error_m": manual.get("training_mean_m"),
        "audit_only_manual_future_error_m": manual.get("future_mean_m"),
        "audit_only_future_frame_correlation": _future_correlation(summary),
        "audit_only_future_frames": future_frame_audit,
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": _sha256(summary_path),
    }


def select_motioncrafter_views(
    summary_paths: list[str | Path],
) -> dict[str, Any]:
    """Select one view per case without reading future or manual metrics."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in summary_paths:
        candidate = _summarize_candidate(path)
        grouped[candidate["case"]].append(candidate)
    if not grouped:
        raise ValueError("at least one association summary is required")

    cases: list[dict[str, Any]] = []
    selected_candidates: list[dict[str, Any]] = []
    fixed_candidates: list[dict[str, Any]] = []
    for case, candidates in sorted(grouped.items()):
        camera_indices = [candidate["camera_index"] for candidate in candidates]
        if len(camera_indices) != len(set(camera_indices)):
            raise ValueError(f"duplicate camera summary for {case}")
        ordered = sorted(
            candidates,
            key=lambda candidate: (
                candidate["selection_score_m"],
                candidate["camera_index"],
            ),
        )
        selected = ordered[0]
        perception_only_selected = min(
            candidates,
            key=lambda candidate: (
                candidate["perception_only_sensitivity_score_m"],
                candidate["camera_index"],
            ),
        )
        fixed = next(
            (candidate for candidate in candidates if candidate["camera_index"] == 0),
            None,
        )
        selected_candidates.append(selected)
        if fixed is not None:
            fixed_candidates.append(fixed)
        cases.append(
            {
                "case": case,
                "selected_camera_index": selected["camera_index"],
                "perception_only_sensitivity_camera_index": (
                    perception_only_selected["camera_index"]
                ),
                "fixed_camera_zero_available": fixed is not None,
                "candidates": sorted(
                    candidates, key=lambda candidate: candidate["camera_index"]
                ),
            }
        )

    def aggregate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "case_count": len(candidates),
            "future_dense_error_m": _optional_mean(
                [candidate["future_dense_error_m"] for candidate in candidates]
            ),
            "future_mean_graph_coverage": _optional_mean(
                [
                    candidate["future_mean_graph_coverage"]
                    for candidate in candidates
                ]
            ),
            "audit_only_manual_future_error_m": _optional_mean(
                [
                    candidate["audit_only_manual_future_error_m"]
                    for candidate in candidates
                ]
            ),
            "audit_only_future_frame_correlation": _optional_mean(
                [
                    candidate["audit_only_future_frame_correlation"]
                    for candidate in candidates
                ]
            ),
        }

    return {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_contract": {
            "score": "training dense error / training-end graph coverage",
            "allowed_inputs": "released training simulation, automatic PhysTwin tracks, masks, depth, and MotionCrafter outputs",
            "forbidden_inputs": "future errors, future coverage, and sparse manual tracks",
            "manual_metrics": "reported only after the camera map is frozen",
            "perception_only_sensitivity": "(frame-zero association error + training motion disagreement) / training-end graph coverage",
        },
        "cases": cases,
        "selected_aggregate": aggregate(selected_candidates),
        "fixed_camera_zero_aggregate": aggregate(fixed_candidates),
    }
