#!/usr/bin/env python3
"""Run the opened-case disjoint-camera guard mechanism diagnostic.

Prediction and evaluation are separate subcommands.  ``supplements`` and
``predict`` accept no target path.  Only ``evaluate`` joins the already-open
source/calibration outcomes after every guarded prediction is checksummed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from bayesian_phystwin.deform360_bias_aware_belief_development import (
    _load_source_target_pickle,
)
from bayesian_phystwin.deform360_bias_aware_prospective_artifacts import (
    canonical_sha256,
    file_sha256,
)
from bayesian_phystwin.deform360_crossview_guard_artifact import (
    build_crossview_guard_prediction,
    load_crossview_guard_prediction,
)
from bayesian_phystwin.deform360_crossview_2d_guard_artifact import (
    ARCHIVE_FILENAME as DIRECT_ARCHIVE_FILENAME,
    REPORT_FILENAME as DIRECT_REPORT_FILENAME,
    build_direct_crossview_guard_prediction,
    load_direct_crossview_guard_prediction,
)
from bayesian_phystwin.deform360_crossview_observation import (
    build_crossview_track_supplement,
    load_crossview_track_supplement,
    load_source_raw_camera_config,
)
from bayesian_phystwin.deform360_online_belief_evaluation import (
    score_deform360_hidden_trajectory,
)
from bayesian_phystwin.deform360_raw_camera_observation import (
    AllTrackerPrefixRuntime,
)


SOURCE_CASES = (
    "002-rope-silk-ep0006",
    "083-blanket-cloth-ep0002",
    "083-blanket-cloth-ep0008",
    "085-scarf-cloth-ep0004",
    "085-scarf-cloth-ep0008",
    "170-spider-ep0000",
    "170-spider-ep0001",
)
CALIBRATION_V1_CASES = (
    "076-rubber-bands-ep0000",
    "163-bear-ep0001",
    "175-plastic-bag-cloth-ep0003",
)
CALIBRATION_V2_CASES = ("078-fishing-line-ep0004",)
SCORED_FRAMES = tuple([*range(20, 38), *range(39, 57), *range(58, 76)])
METRICS = (
    "post_update_hidden_identity_rmse_m",
    "post_update_hidden_symmetric_chamfer_m",
)


@dataclass(frozen=True)
class CaseRecord:
    case: str
    cohort: str
    measurement_dir: Path
    baseline_archive: Path
    baseline_key: str
    target_path: Path

    @property
    def object_id(self) -> str:
        return self.case.rsplit("-ep", 1)[0]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/mnt/corsair/florianpfaff/"
            "deform360-crossview-guard-v1-postopen"
        ),
    )
    parser.add_argument(
        "--source-measurement-root",
        type=Path,
        default=Path(
            "/mnt/corsair/florianpfaff/bpt-online-belief-v1/runs/"
            "deform360-raw-camera-alltracker-v1-measurements"
        ),
    )
    parser.add_argument(
        "--source-baseline-root",
        type=Path,
        default=Path(
            "/mnt/corsair/florianpfaff/"
            "bpt-bias-aware-open27-v4-locked-development"
        ),
    )
    parser.add_argument(
        "--source-panel-root",
        type=Path,
        default=Path(
            "/mnt/corsair/florianpfaff/deform360-dense-reusable-panel-v1/"
            "independent-source-v1"
        ),
    )
    parser.add_argument(
        "--prospective-v1-root",
        type=Path,
        default=Path(
            "/mnt/corsair/florianpfaff/"
            "deform360-bias-aware-prospective-v1"
        ),
    )
    parser.add_argument(
        "--prospective-v2-root",
        type=Path,
        default=Path(
            "/mnt/corsair/florianpfaff/"
            "deform360-bias-aware-prospective-v2"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    supplement = subparsers.add_parser("supplements")
    supplement.add_argument("--alltracker-source", type=Path, required=True)
    supplement.add_argument("--checkpoint", type=Path, required=True)
    supplement.add_argument("--device", required=True)
    supplement.add_argument("--shard-index", type=int, default=0)
    supplement.add_argument("--shard-count", type=int, default=1)
    predict = subparsers.add_parser("predict")
    predict.add_argument("--shard-index", type=int, default=0)
    predict.add_argument("--shard-count", type=int, default=1)
    predict_direct = subparsers.add_parser("predict-direct-2d")
    predict_direct.add_argument("--shard-index", type=int, default=0)
    predict_direct.add_argument("--shard-count", type=int, default=1)
    subparsers.add_parser("evaluate")
    subparsers.add_parser("evaluate-direct-2d")
    return parser


def _records(args: argparse.Namespace) -> tuple[CaseRecord, ...]:
    records: list[CaseRecord] = []
    for case in SOURCE_CASES:
        records.append(
            CaseRecord(
                case=case,
                cohort="opened_source",
                measurement_dir=args.source_measurement_root / case,
                baseline_archive=args.source_baseline_root / f"{case}.npz",
                baseline_key="selected_raw_baseline",
                target_path=args.source_panel_root / case / "target_data.pkl",
            )
        )
    for case in CALIBRATION_V1_CASES:
        records.append(
            CaseRecord(
                case=case,
                cohort="opened_calibration",
                measurement_dir=(
                    args.prospective_v1_root / "measurements" / case
                ),
                baseline_archive=(
                    args.prospective_v1_root
                    / "predictions"
                    / case
                    / "bias_aware_prediction.npz"
                ),
                baseline_key="selected_raw_backbone",
                target_path=(
                    args.prospective_v2_root
                    / "authorized-outcomes"
                    / case
                    / "target_trajectory.npz"
                ),
            )
        )
    for case in CALIBRATION_V2_CASES:
        records.append(
            CaseRecord(
                case=case,
                cohort="opened_calibration",
                measurement_dir=(
                    args.prospective_v2_root / "measurements" / case
                ),
                baseline_archive=(
                    args.prospective_v2_root
                    / "predictions-fresh"
                    / case
                    / "bias_aware_prediction.npz"
                ),
                baseline_key="selected_raw_backbone",
                target_path=(
                    args.prospective_v2_root
                    / "authorized-outcomes"
                    / case
                    / "target_trajectory.npz"
                ),
            )
        )
    return tuple(records)


def _selected_records(
    records: tuple[CaseRecord, ...], shard_index: int, shard_count: int
) -> tuple[CaseRecord, ...]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard")
    return tuple(
        record
        for index, record in enumerate(records)
        if index % shard_count == shard_index
    )


def _build_supplements(args: argparse.Namespace) -> dict[str, Any]:
    selected = _selected_records(
        _records(args), args.shard_index, args.shard_count
    )
    completed: list[dict[str, Any]] = []
    for record in selected:
        output = args.output_root / "supplements" / record.case
        if output.exists():
            manifest, _ = load_crossview_track_supplement(output)
            completed.append(
                {
                    "case": record.case,
                    "status": "preexisting-validated",
                    "result_sha256": manifest["result_sha256"],
                }
            )
            continue
        config = load_source_raw_camera_config(record.measurement_dir)
        runtime = AllTrackerPrefixRuntime(
            args.alltracker_source,
            args.checkpoint,
            device=args.device,
            config=config,
        )
        try:
            manifest = build_crossview_track_supplement(
                record.measurement_dir,
                output,
                runtime,
            )
        finally:
            runtime.close()
        completed.append(
            {
                "case": record.case,
                "status": "built",
                "result_sha256": manifest["result_sha256"],
            }
        )
    return {
        "command": "supplements",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "cases": completed,
    }


def _build_predictions(args: argparse.Namespace) -> dict[str, Any]:
    selected = _selected_records(
        _records(args), args.shard_index, args.shard_count
    )
    completed: list[dict[str, Any]] = []
    for record in selected:
        output = args.output_root / "predictions" / record.case
        if output.exists():
            report, _ = load_crossview_guard_prediction(output)
            status = "preexisting-validated"
        else:
            report = build_crossview_guard_prediction(
                record.measurement_dir,
                args.output_root / "supplements" / record.case,
                record.baseline_archive,
                record.baseline_key,
                output,
            )
            status = "built"
        completed.append(
            {
                "case": record.case,
                "status": status,
                "result_sha256": report["result_sha256"],
                "accepted_update_count": report["output"][
                    "accepted_update_count"
                ],
            }
        )
    return {
        "command": "predict",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "cases": completed,
    }


def _build_direct_predictions(args: argparse.Namespace) -> dict[str, Any]:
    selected = _selected_records(
        _records(args), args.shard_index, args.shard_count
    )
    completed: list[dict[str, Any]] = []
    for record in selected:
        output = args.output_root / "predictions-direct-2d" / record.case
        if output.exists():
            report, _ = load_direct_crossview_guard_prediction(output)
            status = "preexisting-validated"
        else:
            report = build_direct_crossview_guard_prediction(
                record.measurement_dir,
                args.output_root / "supplements" / record.case,
                record.baseline_archive,
                record.baseline_key,
                output,
            )
            status = "built"
        completed.append(
            {
                "case": record.case,
                "status": status,
                "result_sha256": report["result_sha256"],
                "accepted_update_count": report["output"][
                    "accepted_update_count"
                ],
            }
        )
    return {
        "command": "predict-direct-2d",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "cases": completed,
    }


def _target_arrays(record: CaseRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if record.cohort == "opened_source":
        target = _load_source_target_pickle(record.target_path)
        return (
            np.asarray(target["object_points"]),
            np.asarray(target["object_visibilities"], dtype=bool),
            np.asarray(target["object_motions_valid"], dtype=bool),
        )
    with np.load(record.target_path, allow_pickle=False) as stored:
        return (
            np.asarray(stored["target_m"]),
            np.asarray(stored["target_visibility"], dtype=bool),
            np.asarray(stored["target_validity"], dtype=bool),
        )


def _score(
    trajectory: np.ndarray,
    target: np.ndarray,
    visibility: np.ndarray,
    validity: np.ndarray,
    centers: np.ndarray,
    frames: tuple[int, ...],
) -> dict[str, Any]:
    return score_deform360_hidden_trajectory(
        trajectory,
        target,
        visibility,
        validity,
        center_ids=centers,
        scored_frames=frames,
    )


def _evaluate_case(
    record: CaseRecord, output_root: Path, *, direct_2d: bool
) -> dict[str, Any]:
    prediction_subdir = "predictions-direct-2d" if direct_2d else "predictions"
    prediction_dir = output_root / prediction_subdir / record.case
    if direct_2d:
        report, arrays = load_direct_crossview_guard_prediction(prediction_dir)
        guarded_key = "direct_2d_crossview_guarded_m"
        report_filename = DIRECT_REPORT_FILENAME
        archive_filename = DIRECT_ARCHIVE_FILENAME
    else:
        report, arrays = load_crossview_guard_prediction(prediction_dir)
        guarded_key = "crossview_guarded_m"
        report_filename = "crossview_guarded_prediction.json"
        archive_filename = "crossview_guarded_prediction.npz"
    target, visibility, validity = _target_arrays(record)
    baseline = arrays["baseline_m"]
    guarded = arrays[guarded_key]
    centers = np.asarray(arrays["center_ids"], dtype=np.int64)
    if target.shape != baseline.shape:
        raise ValueError(f"target shape changed: {record.case}")
    scores = {
        "baseline": _score(
            baseline, target, visibility, validity, centers, SCORED_FRAMES
        ),
        "crossview_guarded": _score(
            guarded, target, visibility, validity, centers, SCORED_FRAMES
        ),
    }
    differences = {
        metric: float(
            scores["crossview_guarded"][metric] - scores["baseline"][metric]
        )
        for metric in METRICS
    }
    intervals: list[dict[str, Any]] = []
    for update in report["method"]["updates"]:
        frame = int(update["frame"])
        stop = int(update["interval_end_exclusive"])
        frames = tuple(range(frame + 1, stop))
        interval_scores = {
            "baseline": _score(
                baseline, target, visibility, validity, centers, frames
            ),
            "crossview_guarded": _score(
                guarded, target, visibility, validity, centers, frames
            ),
        }
        regret = {
            metric: float(
                interval_scores["crossview_guarded"][metric]
                - interval_scores["baseline"][metric]
            )
            for metric in METRICS
        }
        intervals.append(
            {
                "frame": frame,
                "accepted": bool(update["accepted"]),
                "exact_fallback": bool(
                    np.array_equal(
                        guarded[frame + 1 : stop], baseline[frame + 1 : stop]
                    )
                ),
                "scores": interval_scores,
                "regret_m": regret,
                "worst_primary_regret_m": float(max(regret.values())),
            }
        )
    payload: dict[str, Any] = {
        "artifact_kind": (
            "Deform360Direct2DCrossViewGuardPostOpenEvaluation"
            if direct_2d
            else "Deform360CrossViewGuardPostOpenEvaluation"
        ),
        "schema_version": 1,
        "case": record.case,
        "object_id": record.object_id,
        "cohort": record.cohort,
        "prediction_result_sha256": report["result_sha256"],
        "scores": scores,
        "difference_m": differences,
        "intervals": intervals,
        "inputs_sha256": {
            "prediction_report": file_sha256(
                prediction_dir / report_filename
            ),
            "prediction_archive": file_sha256(
                prediction_dir / archive_filename
            ),
            "opened_target": file_sha256(record.target_path),
        },
        "information_boundary": {
            "prediction_preexisted_before_outcome_join": True,
            "opened_outcome_used_to_change_prediction": False,
        },
        "claim_boundary": "post-open mechanism development only",
    }
    payload["result_sha256"] = canonical_sha256(
        payload, digest_key="result_sha256"
    )
    evaluation_subdir = "evaluations-direct-2d" if direct_2d else "evaluations"
    evaluation_path = output_root / evaluation_subdir / f"{record.case}.json"
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def _object_balanced(
    cases: list[Mapping[str, Any]], value_key: str
) -> dict[str, float]:
    by_object: dict[str, list[Mapping[str, Any]]] = {}
    for case in cases:
        by_object.setdefault(str(case["object_id"]), []).append(case)
    return {
        metric: float(
            np.mean(
                [
                    np.mean([member[value_key][metric] for member in members])
                    for members in by_object.values()
                ]
            )
        )
        for metric in METRICS
    }


def _aggregate_cohort(cases: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [
        interval
        for case in cases
        for interval in case["intervals"]
        if interval["accepted"]
    ]
    accepted_objects = {
        case["object_id"]
        for case in cases
        if any(interval["accepted"] for interval in case["intervals"])
    }
    differences = _object_balanced(cases, "difference_m")
    return {
        "case_count": len(cases),
        "object_count": len({case["object_id"] for case in cases}),
        "accepted_update_count": len(accepted),
        "accepted_object_count": len(accepted_objects),
        "harmful_accepted_interval_count": int(
            sum(interval["worst_primary_regret_m"] > 0.0 for interval in accepted)
        ),
        "object_balanced_difference_m": differences,
        "object_balanced_baseline_m": _object_balanced(
            [
                {
                    **case,
                    "baseline_metric": {
                        metric: case["scores"]["baseline"][metric]
                        for metric in METRICS
                    },
                }
                for case in cases
            ],
            "baseline_metric",
        ),
        "object_balanced_guarded_m": _object_balanced(
            [
                {
                    **case,
                    "guarded_metric": {
                        metric: case["scores"]["crossview_guarded"][metric]
                        for metric in METRICS
                    },
                }
                for case in cases
            ],
            "guarded_metric",
        ),
    }


def _evaluate(args: argparse.Namespace, *, direct_2d: bool = False) -> dict[str, Any]:
    cases = [
        _evaluate_case(record, args.output_root, direct_2d=direct_2d)
        for record in _records(args)
    ]
    source = [case for case in cases if case["cohort"] == "opened_source"]
    calibration = [
        case for case in cases if case["cohort"] == "opened_calibration"
    ]
    source_summary = _aggregate_cohort(source)
    calibration_summary = _aggregate_cohort(calibration)
    gates = {
        "source_updates_span_at_least_two_objects": bool(
            source_summary["accepted_object_count"] >= 2
        ),
        "source_object_balanced_identity_improves": bool(
            source_summary["object_balanced_difference_m"][METRICS[0]] < 0.0
        ),
        "source_object_balanced_chamfer_improves": bool(
            source_summary["object_balanced_difference_m"][METRICS[1]] < 0.0
        ),
        "zero_harmful_calibration_acceptances": bool(
            calibration_summary["harmful_accepted_interval_count"] == 0
        ),
    }
    payload: dict[str, Any] = {
        "artifact_kind": (
            "Deform360Direct2DCrossViewGuardPostOpenResult"
            if direct_2d
            else "Deform360CrossViewGuardPostOpenResult"
        ),
        "schema_version": 1,
        "protocol_id": (
            "deform360-direct-2d-crossview-guard-v1-postopen-development"
            if direct_2d
            else "deform360-disjoint-crossview-guard-v1-postopen-development"
        ),
        "source": source_summary,
        "calibration": calibration_summary,
        "gates": gates,
        "fresh_preregistered_evaluation_justified": bool(all(gates.values())),
        "cases": cases,
        "information_boundary": {
            "all_predictions_hashed_before_outcome_join": True,
            "sealed_prospective_target_accessed": False,
            "opened_source_and_calibration_outcomes_only": True,
        },
        "claim_boundary": (
            "Post-open mechanism evidence on exhausted source/calibration "
            "cases. Passing can justify a fresh preregistration but cannot be "
            "reported as prospective confirmation."
        ),
    }
    payload["result_sha256"] = canonical_sha256(
        payload, digest_key="result_sha256"
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_name = "summary-direct-2d.json" if direct_2d else "summary.json"
    (args.output_root / summary_name).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    args = _parser().parse_args()
    if args.command == "supplements":
        result = _build_supplements(args)
    elif args.command == "predict":
        result = _build_predictions(args)
    elif args.command == "predict-direct-2d":
        result = _build_direct_predictions(args)
    elif args.command == "evaluate-direct-2d":
        result = _evaluate(args, direct_2d=True)
    else:
        result = _evaluate(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
