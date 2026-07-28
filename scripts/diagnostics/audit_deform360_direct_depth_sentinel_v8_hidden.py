#!/usr/bin/env python3
"""Score the sealed V8 source prediction on disjoint hidden identities."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.deform360_dynamic_tapnextpp_assimilation import (
    CANDIDATE_ARM,
    PERSISTENCE_ARM,
    PHYSICAL_ARM,
    SELECTED_BACKBONE_ARM,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_evaluation import (
    _score_hidden_frames,
)
from bayesian_phystwin.deform360_sentinel_query_schedule import (
    DIRECT_DEPTH_PROTOCOL_ID,
)
from bayesian_phystwin.observation_belief import file_sha256

FROZEN_PROVIDER_REVISION = "e776afc74804909dd2c7126e1e0069337f4b7d2c"
FUTURE_FRAMES = tuple(range(58, 76))
LATE_FUTURE_FRAMES = tuple(range(70, 76))
ARMS = (
    PHYSICAL_ARM,
    PERSISTENCE_ARM,
    SELECTED_BACKBONE_ARM,
    CANDIDATE_ARM,
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _canonical_sha256(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {
            name: np.ascontiguousarray(np.asarray(stored[name]))
            for name in stored.files
        }


def _load_target(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    _require(isinstance(payload, dict), "target payload is invalid")
    target = np.asarray(payload["object_points"], dtype=np.float64)
    visibility = np.asarray(payload["object_visibilities"], dtype=bool)
    validity = np.asarray(payload["object_motions_valid"], dtype=bool)
    _require(
        target.ndim == 3
        and target.shape[0] == 76
        and target.shape[2] == 3
        and visibility.shape == validity.shape == target.shape[:2],
        "target arrays changed shape",
    )
    return target, visibility, validity


def _score(
    trajectory: np.ndarray,
    target: np.ndarray,
    visibility: np.ndarray,
    validity: np.ndarray,
    hidden: np.ndarray,
) -> dict[str, float]:
    future = _score_hidden_frames(
        trajectory,
        target,
        visibility,
        validity,
        hidden,
        FUTURE_FRAMES,
    )
    late = _score_hidden_frames(
        trajectory,
        target,
        visibility,
        validity,
        hidden,
        LATE_FUTURE_FRAMES,
    )
    return {
        "future_hidden_identity_rmse_m": future["identity_rmse_m"],
        "future_hidden_symmetric_chamfer_m": (future["symmetric_chamfer_m"]),
        "late_future_hidden_identity_rmse_m": late["identity_rmse_m"],
        "future_hidden_target_to_prediction_chamfer_m": (
            future["target_to_prediction_chamfer_m"]
        ),
    }


def _relative_change(candidate: float, reference: float) -> float:
    _require(
        np.isfinite(candidate) and np.isfinite(reference) and reference > 0.0,
        "relative-change inputs are invalid",
    )
    return candidate / reference - 1.0


def main() -> int:
    args = _parse_args()
    run = args.run_dir.resolve()
    report_path = run / "source_development_report.json"
    schedule_path = run / "query_schedule.json"
    assimilation_path = run / "assimilation_arrays.npz"
    for path in (report_path, schedule_path, assimilation_path):
        _require(path.is_file(), f"V8 artifact is missing: {path.name}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(
        report.get("artifact_kind") == "Deform360DirectDepthSentinelV8SourceDevelopment"
        and report.get("protocol_id") == DIRECT_DEPTH_PROTOCOL_ID
        and report.get("status") == "post_open_source_development_not_confirmation"
        and report.get("repository_revision") == FROZEN_PROVIDER_REVISION
        and report.get("result_sha256") == _canonical_sha256(report),
        "V8 source report is invalid",
    )
    output_hashes = report.get("outputs", {})
    _require(
        output_hashes.get("assimilation_arrays_file_sha256")
        == file_sha256(assimilation_path)
        and output_hashes.get("query_schedule_file_sha256")
        == file_sha256(schedule_path),
        "V8 prediction artifact hash changed",
    )
    boundary = report.get("information_boundary", {})
    _require(
        boundary.get("future_identity_read") is False
        and boundary.get("future_object_observation_read") is False
        and boundary.get("target_metric_read") is False
        and boundary.get("v1_sealed_target_cohort_read") is False
        and boundary.get("held_v8_artifact_read") is False,
        "V8 prediction crossed its information boundary",
    )
    support = report.get("support", {})
    _require(
        support.get("birth_and_update_supported_count") == 12
        and support.get("active_endpoint_supported_count") == 9
        and support.get("sentinel_endpoint_supported_count") == 3
        and report.get("sentinel_debias", {}).get("applied") is True,
        "V8 provider did not pass its pre-outcome gate",
    )
    updates = report.get("assimilation_report", {}).get("updates", [])
    _require(
        isinstance(updates, list) and len(updates) == 3,
        "V8 assimilation update record changed",
    )
    final_update = updates[-1]
    _require(
        final_update.get("pairwise_gate", {}).get("accepted") is True
        and final_update.get("pairwise_gate", {}).get("inlier_count") == 9,
        "V8 active pairwise gate did not pass",
    )

    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    query_ids = np.asarray(schedule["entity_ids"], dtype=np.int64)
    _require(
        schedule.get("protocol_id") == DIRECT_DEPTH_PROTOCOL_ID
        and schedule.get("schedule_sha256") == report["inputs_sha256"]["query_schedule"]
        and query_ids.shape == (12,)
        and len(np.unique(query_ids)) == 12,
        "V8 query schedule is invalid",
    )
    assimilation = _load_npz(assimilation_path)
    _require(
        set(ARMS).issubset(assimilation)
        and all(
            assimilation[name].shape == assimilation[PHYSICAL_ARM].shape
            for name in ARMS
        ),
        "V8 assimilation arrays changed",
    )
    _require(
        not np.array_equal(
            assimilation[CANDIDATE_ARM][58:],
            assimilation[SELECTED_BACKBONE_ARM][58:],
        ),
        "V8 candidate is exact fallback",
    )

    target_path = args.target.resolve()
    target, visibility, validity = _load_target(target_path)
    _require(
        assimilation[PHYSICAL_ARM].shape == target.shape
        and np.all((query_ids >= 0) & (query_ids < target.shape[1])),
        "V8 prediction and target identities differ",
    )
    hidden_ids = np.setdiff1d(
        np.arange(target.shape[1], dtype=np.int64),
        query_ids,
        assume_unique=True,
    )
    scores = {
        name: _score(
            assimilation[name],
            target,
            visibility,
            validity,
            hidden_ids,
        )
        for name in ARMS
    }
    candidate = scores[CANDIDATE_ARM]
    comparison: dict[str, Any] = {}
    for reference_name in (
        PERSISTENCE_ARM,
        SELECTED_BACKBONE_ARM,
    ):
        reference = scores[reference_name]
        identity_change = _relative_change(
            candidate["future_hidden_identity_rmse_m"],
            reference["future_hidden_identity_rmse_m"],
        )
        chamfer_change = _relative_change(
            candidate["future_hidden_symmetric_chamfer_m"],
            reference["future_hidden_symmetric_chamfer_m"],
        )
        comparison[reference_name] = {
            "relative_identity_rmse_change": identity_change,
            "relative_symmetric_chamfer_change": chamfer_change,
            "joint_improvement": (identity_change < 0.0 and chamfer_change < 0.0),
        }
    passed = all(row["joint_improvement"] for row in comparison.values())

    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360DirectDepthSentinelV8HiddenSourceAudit",
        "status": "post_open_source_diagnostic_not_confirmation",
        "protocol_id": DIRECT_DEPTH_PROTOCOL_ID,
        "source_case_hash": report["case_hash"],
        "frozen_provider_revision": FROZEN_PROVIDER_REVISION,
        "source_report_file_sha256": file_sha256(report_path),
        "query_schedule_file_sha256": file_sha256(schedule_path),
        "assimilation_file_sha256": file_sha256(assimilation_path),
        "target_file_sha256": file_sha256(target_path),
        "query_identity_count": len(query_ids),
        "hidden_identity_count": len(hidden_ids),
        "scored_future_range_half_open": [58, 76],
        "late_future_range_half_open": [70, 76],
        "scores": scores,
        "comparison": comparison,
        "advancement_gate_passed": passed,
        "decision": (
            "advance-to-immutable-opened-source-transfer"
            if passed
            else "close-direct-depth-sentinel-v8"
        ),
        "information_boundary": {
            "query_identities_excluded_from_all_scores": True,
            "already_open_source_target_read": True,
            "fresh_target_read": False,
            "v1_sealed_target_cohort_read": False,
            "held_v8_artifact_read": False,
        },
        "claim_boundary": (
            "One already-open source smoke case; not transfer, confirmation, "
            "calibration, or state-of-the-art evidence."
        ),
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    output = args.output.resolve()
    _require(not output.exists(), "V8 hidden-audit output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
