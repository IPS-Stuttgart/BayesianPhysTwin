#!/usr/bin/env python3
"""Apply unchanged source v4 to the opened fresh-pairwise cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
from typing import Any, Mapping

import numpy as np

from bayesian_phystwin.deform360_fresh_bias_guard_diagnostic import (
    apply_frozen_fresh_bias_guard_arrays,
)
from bayesian_phystwin.deform360_fresh_pairwise_outcome import (
    _cluster_bootstrap,
    _load_prediction_case,
)
from bayesian_phystwin.deform360_fresh_pairwise_protocol import (
    canonical_sha256,
    file_sha256,
    load_bound_cohort,
    load_fresh_pairwise_protocol,
    load_json,
    write_json,
)
from bayesian_phystwin.deform360_online_belief_evaluation import (
    score_deform360_hidden_trajectory,
)
from bayesian_phystwin.deform360_raw_camera_observation import (
    MANIFEST_FILENAME,
    MEASUREMENT_FILENAME,
)


SOURCE_LOCK_SHA256 = (
    "5f5672d35aa41e276f1dd5ace54b6694b0139ff2a562e3c3a24558fa555c9dd6"
)
METRICS = (
    "post_update_hidden_identity_rmse_m",
    "post_update_hidden_symmetric_chamfer_m",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cohort-lock", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--outcome-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _score(
    trajectory: np.ndarray,
    target: np.ndarray,
    visibility: np.ndarray,
    validity: np.ndarray,
    centers: np.ndarray,
    frames: tuple[int, ...],
) -> dict[str, object]:
    return score_deform360_hidden_trajectory(
        trajectory,
        target,
        visibility,
        validity,
        center_ids=centers,
        scored_frames=frames,
    )


def _metric_values(score: Mapping[str, Any]) -> dict[str, float]:
    return {metric: float(score[metric]) for metric in METRICS}


def main() -> int:
    args = _parse_args()
    repo = args.repo.resolve()
    source_lock_path = args.source_lock.resolve()
    _require(
        file_sha256(source_lock_path) == SOURCE_LOCK_SHA256,
        "source-v4 lock changed",
    )
    source_lock = load_json(source_lock_path)
    protocol = load_fresh_pairwise_protocol(
        args.protocol,
        repository_root=repo,
    )
    cohort = load_bound_cohort(args.cohort_lock, protocol)
    outcome_summary_path = args.outcome_root.resolve() / "summary.json"
    outcome_summary = load_json(outcome_summary_path)
    _require(
        outcome_summary.get("result_sha256")
        == canonical_sha256(outcome_summary, digest_key="result_sha256")
        and outcome_summary.get("case_count") == 12,
        "fresh outcome summary changed",
    )
    scored_frames = tuple(outcome_summary["scoring"]["scored_frames"])
    cases: list[dict[str, Any]] = []
    for case_record in cohort["cases"]:
        case = str(case_record["case"])
        prediction_dir = args.prediction_root.resolve() / case
        seal, prediction_report, arrays = _load_prediction_case(
            prediction_dir,
            protocol_config_sha256=protocol["config_file_sha256"],
            cohort_lock_sha256=cohort["cohort_lock_sha256"],
            expected_case=case,
        )
        measurement_dir = args.measurement_root.resolve() / case
        measurement_manifest_path = measurement_dir / MANIFEST_FILENAME
        measurement_manifest = load_json(measurement_manifest_path)
        _require(
            file_sha256(measurement_manifest_path)
            == seal["inputs"]["measurement_manifest"]["file_sha256"]
            and measurement_manifest.get("result_sha256")
            == canonical_sha256(
                measurement_manifest, digest_key="result_sha256"
            ),
            f"measurement manifest changed: {case}",
        )
        measurement_path = measurement_dir / MEASUREMENT_FILENAME
        _require(
            file_sha256(measurement_path)
            == measurement_manifest["output"]["measurement_archive_sha256"],
            f"measurement archive changed: {case}",
        )
        with np.load(measurement_path, allow_pickle=False) as stored:
            measurement = {
                name: np.asarray(stored[name]).copy() for name in stored.files
            }
        selected_baseline = np.asarray(arrays["selected_raw_backbone_m"])
        physical = np.asarray(arrays["physical_prior_m"])
        persistence = np.asarray(arrays["persistence_m"])
        centers = np.asarray(arrays["center_ids"], dtype=np.int64)
        _require(
            np.array_equal(centers, measurement["center_ids"]),
            f"measurement identities changed: {case}",
        )
        target_free_report, guarded = apply_frozen_fresh_bias_guard_arrays(
            selected_baseline,
            physical,
            persistence,
            measurement["measurement_m"],
            measurement["measurement_visibility"],
            measurement["measurement_validity"],
            center_ids=centers,
            update_frames=measurement["update_frames"],
            selected_camera_count=len(measurement["selected_cameras"]),
            triangulation_inlier_view_count=measurement[
                "triangulation_inlier_view_count"
            ],
            triangulation_median_reprojection_px=measurement[
                "triangulation_median_reprojection_px"
            ],
            source_lock=source_lock,
        )

        opened_report_path = args.outcome_root.resolve() / f"{case}.json"
        opened_report = load_json(opened_report_path)
        _require(
            opened_report.get("result_sha256")
            == canonical_sha256(opened_report, digest_key="result_sha256"),
            f"opened report changed: {case}",
        )
        target_path = (
            args.processed_root.resolve()
            / str(case_record["object_id"])
            / f"episode_{int(case_record['episode_id']):04d}"
            / "final_data.pkl"
        )
        _require(
            file_sha256(target_path)
            == opened_report["inputs"]["future_payload"]["file_sha256"],
            f"opened target changed: {case}",
        )
        with target_path.open("rb") as handle:
            target_payload = pickle.load(handle)
        target = np.asarray(target_payload["object_points"])
        visibility = np.asarray(
            target_payload["object_visibilities"], dtype=bool
        )
        validity = np.asarray(
            target_payload["object_motions_valid"], dtype=bool
        )
        baseline_score = _score(
            selected_baseline,
            target,
            visibility,
            validity,
            centers,
            scored_frames,
        )
        guarded_score = _score(
            guarded,
            target,
            visibility,
            validity,
            centers,
            scored_frames,
        )
        for metric in METRICS:
            _require(
                baseline_score[metric]
                == opened_report["scores"][
                    "selected_raw_backbone_persistence_insufficient_default"
                ][metric],
                f"baseline score changed: {case} {metric}",
            )
        interval_outcomes = []
        for decision in target_free_report["decisions"]:
            frames = tuple(
                range(
                    int(decision["frame"]) + 1,
                    int(decision["interval_end_exclusive"]),
                )
            )
            baseline_interval = _score(
                selected_baseline,
                target,
                visibility,
                validity,
                centers,
                frames,
            )
            guarded_interval = _score(
                guarded,
                target,
                visibility,
                validity,
                centers,
                frames,
            )
            regrets = {
                metric: float(
                    guarded_interval[metric] - baseline_interval[metric]
                )
                for metric in METRICS
            }
            interval_outcomes.append(
                {
                    "frame": int(decision["frame"]),
                    "candidate_accepted": bool(
                        decision["candidate_accepted"]
                    ),
                    "regret_m": regrets,
                    "harmful_on_any_primary_metric": bool(
                        any(value > 0.0 for value in regrets.values())
                    ),
                }
            )
        cases.append(
            {
                "case": case,
                "object_id": str(case_record["object_id"]),
                "category": str(case_record["category"]),
                "target_free_guard": target_free_report,
                "scores": {
                    "selected_raw_baseline": _metric_values(baseline_score),
                    "guarded_source_v4": _metric_values(guarded_score),
                },
                "interval_outcomes": interval_outcomes,
                "input_sha256": {
                    "prediction_seal": file_sha256(
                        prediction_dir / "belief_prediction_seal.json"
                    ),
                    "prediction_report": file_sha256(
                        prediction_dir / "belief_prediction_report.json"
                    ),
                    "prediction_archive": file_sha256(
                        prediction_dir / "belief_prediction.npz"
                    ),
                    "measurement_manifest": file_sha256(
                        measurement_manifest_path
                    ),
                    "measurement_archive": file_sha256(measurement_path),
                    "opened_report": file_sha256(opened_report_path),
                    "opened_target": file_sha256(target_path),
                },
            }
        )
    aggregate = {
        arm: {
            metric: float(
                np.mean([case["scores"][arm][metric] for case in cases])
            )
            for metric in METRICS
        }
        for arm in ("selected_raw_baseline", "guarded_source_v4")
    }
    groups = {case["case"]: case["object_id"] for case in cases}
    comparison = {}
    for metric in METRICS:
        differences = {
            case["case"]: float(
                case["scores"]["guarded_source_v4"][metric]
                - case["scores"]["selected_raw_baseline"][metric]
            )
            for case in cases
        }
        result = _cluster_bootstrap(differences, groups)
        result["relative_change"] = (
            aggregate["guarded_source_v4"][metric]
            / aggregate["selected_raw_baseline"][metric]
            - 1.0
        )
        result["wins"] = sum(value < 0.0 for value in differences.values())
        result["ties"] = sum(value == 0.0 for value in differences.values())
        result["regressions"] = sum(
            value > 0.0 for value in differences.values()
        )
        comparison[metric] = result
    accepted_intervals = [
        interval
        for case in cases
        for interval in case["interval_outcomes"]
        if interval["candidate_accepted"]
    ]
    payload = {
        "schema_version": 1,
        "artifact_kind": "Deform360FreshBiasGuardPostOpenResult",
        "source_v4_lock_sha256": file_sha256(source_lock_path),
        "fresh_outcome_summary_sha256": file_sha256(outcome_summary_path),
        "case_count": len(cases),
        "object_count": len({case["object_id"] for case in cases}),
        "candidate_available_interval_count": sum(
            case["target_free_guard"]["candidate_available_count"]
            for case in cases
        ),
        "accepted_interval_count": len(accepted_intervals),
        "exact_fallback_interval_count": sum(
            case["target_free_guard"]["exact_fallback_interval_count"]
            for case in cases
        ),
        "harmful_accepted_interval_count": sum(
            interval["harmful_on_any_primary_metric"]
            for interval in accepted_intervals
        ),
        "aggregate": aggregate,
        "guarded_vs_selected_raw": comparison,
        "cases": cases,
        "information_boundary": {
            "source_v4_lock_changed": False,
            "candidate_built_without_target_argument": True,
            "opened_outcomes_used_to_change_guard": False,
            "cohort_already_exhausted_before_diagnostic": True,
        },
        "claim_boundary": (
            "post-open stress test of unchanged source v4 on an exhausted "
            "fresh cohort; not prospective confirmation, selector tuning, "
            "calibration, or SOTA evidence"
        ),
    }
    payload["result_sha256"] = canonical_sha256(
        payload, digest_key="result_sha256"
    )
    write_json(args.output, payload)
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "case_count",
                    "candidate_available_interval_count",
                    "accepted_interval_count",
                    "exact_fallback_interval_count",
                    "harmful_accepted_interval_count",
                    "aggregate",
                    "guarded_vs_selected_raw",
                    "result_sha256",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
