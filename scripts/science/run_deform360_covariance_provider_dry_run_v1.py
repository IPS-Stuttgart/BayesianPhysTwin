#!/usr/bin/env python3
"""Run the source-only synthetic gate for the Deform360 covariance provider."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.deform360_covariance_provider_v1 import (
    Deform360ObservationSplitV1,
    build_deform360_covariance_only_forecast_v1,
    estimate_deform360_causal_residual_history_v1,
    plan_deform360_camera_partition_v1,
)
from bayesian_phystwin.deform360_joint_sparse_materializer_v5 import (
    Deform360JointSparseVisualWindowRowsV5,
)

DRY_RUN_SCHEMA = "bayesian-phystwin/deform360-covariance-provider-dry-run-v1"


def _canonical_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _window(
    camera_id: str,
    frame_indices: np.ndarray,
    identity_indices: np.ndarray,
    point_world_m: np.ndarray,
) -> Deform360JointSparseVisualWindowRowsV5:
    count = len(frame_indices)
    return Deform360JointSparseVisualWindowRowsV5(
        camera_id=camera_id,
        window_id=f"synthetic-{camera_id}",
        frame_indices=frame_indices,
        pixel_yx=np.column_stack((identity_indices, identity_indices)),
        point_world_m=point_world_m,
        point_covariance_m2=np.broadcast_to(np.eye(3) * 1e-6, (count, 3, 3)),
        source_confidence=np.ones(count),
        mask_distance_pixels=np.ones(count),
        overlap_disagreement_m=np.zeros(count),
        contributor_count=np.ones(count, dtype=np.int64),
        source_artifact_ids={"synthetic/provider-input": "1" * 64},
    )


def run_source_only_dry_run() -> dict[str, Any]:
    """Exercise history, support, covariance, split, and fallback contracts."""

    nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    physical = np.broadcast_to(nodes[None], (4, 4, 3)).copy()
    camera_a_frames = np.asarray([0, 0, 1, 1, 2, 3], dtype=np.int64)
    camera_a_identities = np.asarray([0, 1, 0, 1, 0, 0], dtype=np.int64)
    camera_b_frames = np.asarray([0, 0, 1, 1, 2, 3], dtype=np.int64)
    camera_b_identities = np.asarray([2, 3, 2, 3, 2, 2], dtype=np.int64)
    camera_a_points = np.asarray(
        [
            physical[frame, identity] + [0.001 * (frame + 1), 0.0, 0.0]
            for frame, identity in zip(
                camera_a_frames,
                camera_a_identities,
                strict=True,
            )
        ]
    )
    camera_b_points = np.asarray(
        [
            physical[frame, identity] + [0.001 * (frame + 1), 0.0, 0.0]
            for frame, identity in zip(
                camera_b_frames,
                camera_b_identities,
                strict=True,
            )
        ]
    )
    provider_cameras, scoring_cameras = plan_deform360_camera_partition_v1(
        camera_ids=("camera-a", "camera-b", "camera-c", "camera-d"),
        object_session_hash="3" * 64,
    )
    history = estimate_deform360_causal_residual_history_v1(
        visual_windows=(
            _window(
                provider_cameras[0],
                camera_a_frames,
                camera_a_identities,
                camera_a_points,
            ),
            _window(
                provider_cameras[1],
                camera_b_frames,
                camera_b_identities,
                camera_b_points,
            ),
        ),
        physical_prediction_world_m=physical,
        frame_indices=np.arange(4),
        material_identity_ids=("node-0", "node-1", "node-2", "node-3"),
        source_artifact_ids={"synthetic/physical-prefix": "2" * 64},
        association_candidate_count=1,
    )
    split = Deform360ObservationSplitV1(
        provider_camera_ids=provider_cameras,
        scoring_camera_ids=scoring_cameras,
        provider_reconstruction_artifact_id="synthetic-provider-reconstruction",
        scoring_reconstruction_artifact_id="synthetic-scoring-reconstruction",
    )
    mean = np.arange(48, dtype=np.float32).reshape(4, 4, 3) / 1000.0
    fallback_covariance = np.broadcast_to(
        np.eye(3) * 0.02**2,
        (4, 4, 3, 3),
    ).copy()
    candidate = build_deform360_covariance_only_forecast_v1(
        reference_mean_world_m=mean,
        fallback_covariance_world_m2=fallback_covariance,
        future_frame_indices=np.arange(4, 8),
        horizon_labels=("early", "middle", "middle", "late"),
        history=history,
    )
    unsupported = replace(
        history,
        residual_world_m=np.zeros_like(history.residual_world_m),
        valid_mask=np.zeros_like(history.valid_mask),
    )
    fallback = build_deform360_covariance_only_forecast_v1(
        reference_mean_world_m=mean,
        fallback_covariance_world_m2=fallback_covariance,
        future_frame_indices=np.arange(4, 8),
        horizon_labels=("early", "middle", "middle", "late"),
        history=unsupported,
    )
    payload: dict[str, Any] = {
        "schema": DRY_RUN_SCHEMA,
        "schema_version": 1,
        "source_only": True,
        "synthetic_inputs_only": True,
        "target_roster_read": False,
        "target_payload_read": False,
        "target_outcome_read": False,
        "history": {
            "artifact_id": history.artifact_id,
            "shape": list(history.residual_world_m.shape),
            "valid_shape": list(history.valid_mask.shape),
            "observed_update_count": np.sum(history.valid_mask, axis=0).tolist(),
            "missing_entries_are_zero": bool(
                np.all(history.residual_world_m[~history.valid_mask] == 0.0)
            ),
            "coordinate_frame": history.coordinate_frame,
            "position_units": history.position_units,
            "covariance_units": history.covariance_units,
        },
        "observation_split": {
            "artifact_id": split.artifact_id,
            "provider_camera_ids": list(split.provider_camera_ids),
            "scoring_camera_ids": list(split.scoring_camera_ids),
            "camera_sets_disjoint": bool(
                set(split.provider_camera_ids).isdisjoint(split.scoring_camera_ids)
            ),
            "reconstruction_artifacts_distinct": True,
        },
        "admitted_candidate": {
            "artifact_id": candidate.artifact_id,
            "case_donor_admitted": candidate.case_donor_admitted,
            "empirical_identity_count": int(np.sum(candidate.empirical_donor_mask)),
            "prior_only_identity_count": int(np.sum(candidate.prior_only_mask)),
            "mean_byte_identical": candidate.mean_world_m.tobytes() == mean.tobytes(),
            "minimum_covariance_eigenvalue_m2": float(
                np.min(np.linalg.eigvalsh(candidate.covariance_world_m2))
            ),
        },
        "failed_support_fallback": {
            "artifact_id": fallback.artifact_id,
            "case_donor_admitted": fallback.case_donor_admitted,
            "fallback_reason": fallback.fallback_reason,
            "mean_byte_identical": fallback.mean_world_m.tobytes() == mean.tobytes(),
            "covariance_byte_identical": (
                fallback.covariance_world_m2.tobytes()
                == fallback_covariance.tobytes()
            ),
            "empirical_identity_count": int(np.sum(fallback.empirical_donor_mask)),
        },
        "gate_passed": bool(
            candidate.case_donor_admitted
            and candidate.mean_world_m.tobytes() == mean.tobytes()
            and np.min(np.linalg.eigvalsh(candidate.covariance_world_m2)) >= 0.0
            and not fallback.case_donor_admitted
            and fallback.mean_world_m.tobytes() == mean.tobytes()
            and fallback.covariance_world_m2.tobytes()
            == fallback_covariance.tobytes()
            and set(split.provider_camera_ids).isdisjoint(split.scoring_camera_ids)
        ),
        "claim_boundary": (
            "Implementation dry run only; no target, official benchmark, calibration, "
            "point-accuracy, or state-of-the-art claim."
        ),
    }
    payload["dry_run_sha256"] = _canonical_sha256(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_source_only_dry_run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"dry_run_sha256": result["dry_run_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
