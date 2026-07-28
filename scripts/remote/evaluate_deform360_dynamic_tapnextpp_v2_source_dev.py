#!/usr/bin/env python3
"""Score complete V2 development outputs on already-open source futures."""

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
    SELECTED_BACKBONE_ARM,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_evaluation import (
    score_assimilation_trajectory,
    score_provider_case_arrays,
)
from bayesian_phystwin.observation_belief import file_sha256
from bayesian_phystwin.tapnextpp_dynamic_multiview import (
    DynamicMultiviewConfig,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
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
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True)
    return parser.parse_args()


def _load_target(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    _require(isinstance(payload, dict), "source target payload is invalid")
    target = np.asarray(payload["object_points"], dtype=np.float64)
    visibility = np.asarray(payload["object_visibilities"], dtype=bool)
    validity = np.asarray(payload["object_motions_valid"], dtype=bool)
    _require(
        target.ndim == 3
        and target.shape[0] == 76
        and target.shape[2] == 3
        and visibility.shape == validity.shape == target.shape[:2],
        "source target arrays changed shape",
    )
    return target, visibility, validity


def _score_case(run_root: Path, processed_root: Path, case: str) -> dict[str, Any]:
    run = run_root / case
    report_path = run / "source_development_report.json"
    provider_path = run / "provider_arrays.npz"
    assimilation_path = run / "assimilation_arrays.npz"
    for path in (report_path, provider_path, assimilation_path):
        _require(path.is_file(), f"complete development output is missing: {case}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(
        report.get("status") == "post_open_source_development_not_confirmation"
        and report.get("case") == case
        and report.get("result_sha256") == _canonical_sha256(report),
        f"development report is invalid: {case}",
    )
    _require(
        file_sha256(provider_path)
        == report["outputs"]["provider_arrays_file_sha256"]
        and file_sha256(assimilation_path)
        == report["outputs"]["assimilation_arrays_file_sha256"],
        f"development arrays changed: {case}",
    )
    with np.load(provider_path, allow_pickle=False) as stored:
        provider = {name: np.asarray(stored[name]).copy() for name in stored.files}
    with np.load(assimilation_path, allow_pickle=False) as stored:
        assimilation = {
            name: np.asarray(stored[name]).copy() for name in stored.files
        }
    object_id, episode_token = case.rsplit("-ep", maxsplit=1)
    _require(
        len(episode_token) == 4 and episode_token.isdigit(),
        f"development case identity is invalid: {case}",
    )
    target_path = (
        processed_root
        / object_id
        / f"episode_{episode_token}"
        / "final_data.pkl"
    )
    target, visibility, validity = _load_target(target_path)
    provider_score = score_provider_case_arrays(
        trajectory_world_m=provider["trajectory_world_m"],
        accepted_support=provider["accepted_support"],
        local_covariance_m2=provider["local_covariance_m2"],
        shared_bias_standard_deviation_m=(
            DynamicMultiviewConfig().shared_bias_standard_deviation_m
        ),
        target_m=target,
        target_visibility=visibility,
        target_validity=validity,
        entity_ids=provider["entity_ids"],
        birth_frames=provider["birth_frames"],
        update_frames=provider["update_frames"],
        expected_query_count=None,
    )
    measured = np.unique(provider["entity_ids"].astype(np.int64))
    hidden = np.setdiff1d(
        np.arange(target.shape[1], dtype=np.int64),
        measured,
        assume_unique=True,
    )
    _require(len(hidden) > 0, f"development case has no hidden identities: {case}")
    baseline = score_assimilation_trajectory(
        assimilation[SELECTED_BACKBONE_ARM],
        target,
        visibility,
        validity,
        hidden,
    )
    candidate = score_assimilation_trajectory(
        assimilation[CANDIDATE_ARM],
        target,
        visibility,
        validity,
        hidden,
    )
    return {
        "case": case,
        "case_hash": report["case_hash"],
        "development_report_file_sha256": file_sha256(report_path),
        "provider": provider_score,
        "assimilation": {
            "measurement_identity_count": len(measured),
            "hidden_identity_count": len(hidden),
            "selected_backbone": baseline,
            "candidate": candidate,
            "candidate_minus_selected_identity_rmse_m": (
                candidate["hidden_identity_rmse_m"]
                - baseline["hidden_identity_rmse_m"]
            ),
            "candidate_minus_selected_chamfer_m": (
                candidate["hidden_symmetric_chamfer_m"]
                - baseline["hidden_symmetric_chamfer_m"]
            ),
            "maximum_absolute_prediction_difference_m": float(
                np.max(
                    np.abs(
                        assimilation[CANDIDATE_ARM]
                        - assimilation[SELECTED_BACKBONE_ARM]
                    )
                )
            ),
        },
    }


def main() -> int:
    args = _parse_args()
    cases = tuple(map(str, args.case))
    _require(len(cases) == len(set(cases)), "development cases are repeated")
    rows = [
        _score_case(
            args.run_root.resolve(),
            args.processed_root.resolve(),
            case,
        )
        for case in cases
    ]
    scored_provider = [
        row["provider"]
        for row in rows
        if row["provider"]["provider_rmse_m"] is not None
    ]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360DynamicTAPNextPPV2SourceDevelopmentResult",
        "protocol_id": (
            "deform360-dynamic-tapnextpp-provider-v2-source-development"
        ),
        "status": "post_open_two_case_source_development_not_confirmation",
        "counts": {
            "complete_development_cases": len(rows),
            "provider_scored_cases": len(scored_provider),
        },
        "cases": rows,
        "aggregate": {
            "mean_provider_supported_fraction": float(
                np.mean([row["provider"]["supported_fraction"] for row in rows])
            ),
            "mean_provider_rmse_m": (
                None
                if not scored_provider
                else float(
                    np.mean(
                        [row["provider_rmse_m"] for row in scored_provider]
                    )
                )
            ),
            "mean_persistence_rmse_m": (
                None
                if not scored_provider
                else float(
                    np.mean(
                        [row["persistence_rmse_m"] for row in scored_provider]
                    )
                )
            ),
            "provider_joint_win_count": sum(
                bool(row["provider"]["provider_wins"]) for row in rows
            ),
            "assimilation_nonzero_case_count": sum(
                row["assimilation"]["maximum_absolute_prediction_difference_m"]
                > 0.0
                for row in rows
            ),
        },
        "information_boundary": {
            "all_two_development_outputs_complete_before_future_open": True,
            "already_open_source_futures_read": True,
            "v1_sealed_target_cohort_read": False,
            "fresh_target_read": False,
            "held_v8_artifact_read": False,
        },
        "claim_boundary": (
            "Mechanics and competence development on two already-open source "
            "objects only; not a transfer, confirmation, or SOTA result."
        ),
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
