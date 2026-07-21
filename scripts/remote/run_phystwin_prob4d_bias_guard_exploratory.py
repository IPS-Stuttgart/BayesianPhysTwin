#!/usr/bin/env python3
"""Stage, predict, and evaluate the exploratory Prob4D bias-aware guard."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
from typing import Any, Mapping

import numpy as np

import bayesian_phystwin.phystwin_prob4d_action_guard as action_guard_module
from bayesian_phystwin.phystwin_comparison import (
    phystwin_physical_object_cluster,
)
from bayesian_phystwin.phystwin_official_evaluation import (
    evaluate_official_phystwin_interval,
)
from bayesian_phystwin.phystwin_prob4d_bias_guard import (
    Prob4DBiasGuardConfig,
    build_guarded_prob4d_prefix_candidate,
)


PREFIX_FILENAME = "prob4d_prefix.npz"
PREFIX_MANIFEST_FILENAME = "prefix_manifest.json"
CANDIDATE_FILENAME = "candidate.pkl"
GUARDED_FILENAME = "guarded.pkl"
PREDICTION_REPORT_FILENAME = "prediction_report.json"
PREDICTION_COHORT_SEAL_FILENAME = "prediction_cohort_seal.json"
METRICS = ("chamfer_distance_m", "track_error_m")
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20260721


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage = subparsers.add_parser("stage")
    stage.add_argument("--protocol", type=Path, required=True)
    stage.add_argument("--source-summary", type=Path, required=True)
    stage.add_argument("--output-root", type=Path, required=True)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--protocol", type=Path, required=True)
    predict.add_argument("--source-lock", type=Path, required=True)
    predict.add_argument("--prefix-root", type=Path, required=True)
    predict.add_argument("--selected-baseline-root", type=Path, required=True)
    predict.add_argument("--output-root", type=Path, required=True)
    predict.add_argument(
        "--candidate-family",
        choices=("static", "action_conditioned"),
        default="static",
    )

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--protocol", type=Path, required=True)
    evaluate.add_argument("--source-summary", type=Path, required=True)
    evaluate.add_argument("--prediction-root", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("result_sha256", None)
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path, *, verify: bool = False) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if verify and payload.get("result_sha256") != _canonical_sha256(payload):
        raise ValueError(f"canonical JSON checksum changed: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    payload["result_sha256"] = _canonical_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def _load_pickle(path: Path) -> Any:
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except ModuleNotFoundError as error:
        if error.name != "numpy._core.numeric":
            raise
        import numpy.core as numpy_core
        import numpy.core.numeric as numpy_core_numeric

        sys.modules.setdefault("numpy._core", numpy_core)
        sys.modules.setdefault("numpy._core.numeric", numpy_core_numeric)
        with path.open("rb") as handle:
            return pickle.load(handle)


def _write_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _protocol_cases(protocol: Mapping[str, Any]) -> tuple[str, ...]:
    cases = tuple(str(value) for value in protocol["cohort"]["cases"])
    if len(cases) != len(set(cases)) or not cases:
        raise ValueError("protocol cases must be unique and nonempty")
    return cases


def _source_rows(summary: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = {str(row["case"]): row for row in summary["inputs"]}
    if len(rows) != len(summary["inputs"]):
        raise ValueError("source summary contains duplicate cases")
    return rows


def _require_hash(path: Path, expected: str, role: str) -> None:
    if _sha256(path) != expected:
        raise ValueError(f"{role} checksum changed: {path}")


def _stage_case(
    case: str,
    row: Mapping[str, Any],
    *,
    protocol_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    assimilation_path = Path(str(row["C_assimilation_npz"]))
    assimilation_summary_path = Path(str(row["C_summary"]))
    _require_hash(
        assimilation_path,
        str(row["C_assimilation_npz_sha256"]),
        "arm-C assimilation",
    )
    _require_hash(
        assimilation_summary_path,
        str(row["C_summary_sha256"]),
        "arm-C summary",
    )
    assimilation_summary = _load_json(assimilation_summary_path)
    inputs = assimilation_summary["inputs"]
    physical_path = Path(str(inputs["baseline"]["path"]))
    final_data_path = Path(str(inputs["final_data"]["path"]))
    split_path = Path(str(inputs["split"]["path"]))
    for role, descriptor, path in (
        ("physical baseline", inputs["baseline"], physical_path),
        ("final data", inputs["final_data"], final_data_path),
        ("split", inputs["split"], split_path),
    ):
        _require_hash(path, str(descriptor["sha256"]), role)

    split = _load_json(split_path)
    train_end = int(split["train"][1])
    test_end = int(split["test"][1])
    if int(assimilation_summary["train_end_frame"]) != train_end:
        raise ValueError(f"arm-C and released train boundaries differ: {case}")
    physical = np.asarray(_load_pickle(physical_path), dtype=np.float32)
    final_data = _load_pickle(final_data_path)
    object_points = np.asarray(final_data["object_points"], dtype=np.float32)
    visibility = np.asarray(final_data["object_visibilities"], dtype=bool)
    motion_validity = np.asarray(final_data["object_motions_valid"], dtype=bool)
    num_surface_points = object_points.shape[1] + len(
        np.asarray(final_data["surface_points"])
    )
    with np.load(assimilation_path, allow_pickle=False) as stored:
        frame_indices = np.asarray(stored["frame_indices"], dtype=np.int64)
        if not np.array_equal(frame_indices, np.arange(len(frame_indices))):
            raise ValueError(f"arm-C frame map is not identity: {case}")
        required = {
            "position_flow_positions",
            "position_flow_valid",
            "position_flow_prior_reliability",
            "position_flow_observation_covariance_m2",
        }
        if not required.issubset(stored.files):
            raise ValueError(f"arm-C archive lacks bias-aware inputs: {case}")
        archive = {name: np.asarray(stored[name]) for name in required}
    if min(len(physical), len(object_points), len(frame_indices)) < test_end:
        raise ValueError(f"source arrays do not cover released split: {case}")
    observed_vertex_count = archive["position_flow_positions"].shape[1]
    physical_vertex_count = physical.shape[1]
    if not 1 <= observed_vertex_count <= physical_vertex_count:
        raise ValueError(f"arm-C vertex count exceeds physical state: {case}")
    if archive["position_flow_valid"].shape[1] != observed_vertex_count:
        raise ValueError(f"arm-C validity vertex count changed: {case}")
    padded_positions = np.full(
        (len(frame_indices), physical_vertex_count, 3), np.nan, dtype=np.float32
    )
    padded_validity = np.zeros(
        (len(frame_indices), physical_vertex_count), dtype=bool
    )
    padded_reliability = np.full(
        (len(frame_indices), physical_vertex_count), np.nan, dtype=np.float32
    )
    padded_covariance = np.full(
        (len(frame_indices), physical_vertex_count, 3, 3),
        np.nan,
        dtype=np.float32,
    )
    padded_positions[:, :observed_vertex_count] = archive[
        "position_flow_positions"
    ]
    padded_validity[:, :observed_vertex_count] = archive["position_flow_valid"]
    padded_reliability[:, :observed_vertex_count] = archive[
        "position_flow_prior_reliability"
    ]
    padded_covariance[:, :observed_vertex_count] = archive[
        "position_flow_observation_covariance_m2"
    ]

    case_dir = output_root / case
    prefix_path = case_dir / PREFIX_FILENAME
    case_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prefix_path,
        physical_prefix_m=physical[:train_end],
        prob4d_prefix_positions_m=padded_positions[:train_end],
        prob4d_prefix_validity=padded_validity[:train_end],
        prob4d_prefix_prior_reliability=padded_reliability[:train_end],
        prob4d_prefix_observation_covariance_m2=padded_covariance[:train_end],
        prefix_object_points_m=object_points[:train_end],
        prefix_object_visibility=visibility[:train_end],
        prefix_object_motion_validity=motion_validity[:train_end],
        train_end_exclusive=np.asarray(train_end, dtype=np.int64),
        test_end_exclusive=np.asarray(test_end, dtype=np.int64),
        num_surface_points=np.asarray(num_surface_points, dtype=np.int64),
    )
    manifest = {
        "artifact_kind": "PhysTwinProb4DPrefixArtifact",
        "schema_version": 1,
        "case": case,
        "train_end_exclusive": train_end,
        "test_end_exclusive": test_end,
        "num_surface_points": num_surface_points,
        "physical_vertex_count": physical_vertex_count,
        "prob4d_observed_vertex_count": observed_vertex_count,
        "padded_unobserved_interior_vertex_count": (
            physical_vertex_count - observed_vertex_count
        ),
        "inputs_sha256": {
            "protocol": _sha256(protocol_path),
            "arm_c_assimilation": _sha256(assimilation_path),
            "arm_c_summary": _sha256(assimilation_summary_path),
            "physical_baseline": _sha256(physical_path),
            "final_data": _sha256(final_data_path),
            "split": _sha256(split_path),
        },
        "output": {
            "prefix_archive": str(prefix_path),
            "prefix_archive_sha256": _sha256(prefix_path),
        },
        "information_boundary": {
            "source_files_contain_future": True,
            "emitted_prob4d_frames": [0, train_end - 1],
            "emitted_object_observation_frames": [0, train_end - 1],
            "emitted_future_prob4d": False,
            "emitted_future_object_observation": False,
            "manual_tracks_read": False,
            "role": (
                "trusted prefix materializer; prediction command accepts only "
                "this truncated artifact"
            ),
        },
    }
    return _write_json(case_dir / PREFIX_MANIFEST_FILENAME, manifest)


def stage(args: argparse.Namespace) -> None:
    protocol = _load_json(args.protocol)
    summary = _load_json(args.source_summary)
    if _sha256(args.source_summary) != protocol["cohort"][
        "source_summary_sha256"
    ]:
        raise ValueError("source summary differs from protocol lock")
    rows = _source_rows(summary)
    cases = _protocol_cases(protocol)
    if set(cases) != set(rows):
        raise ValueError("protocol and source-summary cohorts differ")
    manifests = [
        _stage_case(
            case,
            rows[case],
            protocol_path=args.protocol,
            output_root=args.output_root,
        )
        for case in cases
    ]
    cohort = {
        "artifact_kind": "PhysTwinProb4DPrefixCohort",
        "schema_version": 1,
        "case_count": len(cases),
        "cases": [
            {
                "case": case,
                "prefix_manifest_result_sha256": manifest["result_sha256"],
            }
            for case, manifest in zip(cases, manifests, strict=True)
        ],
        "information_boundary": {
            "future_prob4d_emitted": False,
            "future_object_observation_emitted": False,
            "manual_tracks_read": False,
        },
    }
    cohort = _write_json(args.output_root / "prefix_cohort.json", cohort)
    print(json.dumps(cohort, indent=2, sort_keys=True))


def _predict_case(
    case: str,
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    source_lock: Mapping[str, Any],
    source_lock_path: Path,
    prefix_root: Path,
    selected_baseline_root: Path,
    output_root: Path,
    candidate_family: str,
) -> dict[str, Any]:
    prefix_dir = prefix_root / case
    prefix_manifest_path = prefix_dir / PREFIX_MANIFEST_FILENAME
    prefix_manifest = _load_json(prefix_manifest_path, verify=True)
    if prefix_manifest["inputs_sha256"]["protocol"] != _sha256(protocol_path):
        raise ValueError(f"prefix artifact used a different protocol: {case}")
    prefix_path = prefix_dir / PREFIX_FILENAME
    _require_hash(
        prefix_path,
        str(prefix_manifest["output"]["prefix_archive_sha256"]),
        "prefix artifact",
    )
    selected_path = (
        selected_baseline_root / case / "validation_selected" / "trajectory.pkl"
    )
    selected = np.asarray(_load_pickle(selected_path), dtype=np.float32)
    with np.load(prefix_path, allow_pickle=False) as stored:
        prefix = {name: np.asarray(stored[name]) for name in stored.files}
    method = protocol["method"]
    config = Prob4DBiasGuardConfig(
        fit_fraction=float(method["fit_fraction"]),
        minimum_validation_frame_count=int(
            method["minimum_validation_frame_count"]
        ),
        minimum_balanced_validation_improvement_fraction=float(
            method["minimum_balanced_validation_improvement_fraction"]
        ),
    )
    candidate_arguments = (
        selected,
        prefix["physical_prefix_m"],
        prefix["prob4d_prefix_positions_m"],
        prefix["prob4d_prefix_validity"],
        prefix["prob4d_prefix_prior_reliability"],
        prefix["prob4d_prefix_observation_covariance_m2"],
        prefix["prefix_object_points_m"],
        prefix["prefix_object_visibility"],
        prefix["prefix_object_motion_validity"],
    )
    candidate_keywords = {
        "num_surface_points": int(prefix["num_surface_points"]),
        "source_lock": source_lock,
    }
    if candidate_family == "static":
        report, candidate, guarded = build_guarded_prob4d_prefix_candidate(
            *candidate_arguments,
            **candidate_keywords,
            config=config,
        )
    else:
        action_config = action_guard_module.Prob4DActionGuardConfig(
            static_guard=config
        )
        report, candidate, guarded = (
            action_guard_module.build_guarded_action_conditioned_prob4d_candidate(
                *candidate_arguments,
                **candidate_keywords,
                config=action_config,
            )
        )
    case_dir = output_root / case
    candidate_path = case_dir / CANDIDATE_FILENAME
    guarded_path = case_dir / GUARDED_FILENAME
    _write_pickle(candidate_path, candidate)
    _write_pickle(guarded_path, guarded)
    payload = {
        "artifact_kind": (
            "PhysTwinProb4DBiasAwarePrediction"
            if candidate_family == "static"
            else "PhysTwinProb4DActionConditionedPrediction"
        ),
        "schema_version": 1,
        "case": case,
        "target_free_prediction": report,
        "inputs": {
            "protocol": {
                "path": str(protocol_path),
                "sha256": _sha256(protocol_path),
            },
            "source_lock": {
                "path": str(source_lock_path),
                "sha256": _sha256(source_lock_path),
            },
            "prefix_manifest": {
                "path": str(prefix_manifest_path),
                "sha256": _sha256(prefix_manifest_path),
                "result_sha256": prefix_manifest["result_sha256"],
            },
            "prefix_archive": {
                "path": str(prefix_path),
                "sha256": _sha256(prefix_path),
            },
            "selected_baseline": {
                "path": str(selected_path),
                "sha256": _sha256(selected_path),
            },
        },
        "outputs": {
            "candidate": {
                "path": str(candidate_path),
                "sha256": _sha256(candidate_path),
            },
            "guarded": {
                "path": str(guarded_path),
                "sha256": _sha256(guarded_path),
            },
        },
        "information_boundary": {
            "future_prob4d_read": False,
            "future_object_observation_read": False,
            "future_manual_tracks_read": False,
            "target_metric_read": False,
        },
    }
    if candidate_family == "action_conditioned":
        implementation_path = Path(action_guard_module.__file__)
        payload["candidate_family"] = candidate_family
        payload["action_conditioned_implementation"] = {
            "path": str(implementation_path),
            "sha256": _sha256(implementation_path),
        }
    return _write_json(case_dir / PREDICTION_REPORT_FILENAME, payload)


def predict(args: argparse.Namespace) -> None:
    protocol = _load_json(args.protocol)
    source_lock = _load_json(args.source_lock)
    if _sha256(args.source_lock) != protocol["method"]["source_v4_lock_sha256"]:
        raise ValueError("source-v4 lock differs from protocol")
    implementation_path = Path(protocol["method"]["implementation"])
    if _sha256(implementation_path) != protocol["method"][
        "implementation_sha256"
    ]:
        raise ValueError("bias-aware implementation differs from protocol")
    cases = _protocol_cases(protocol)
    reports = [
        _predict_case(
            case,
            protocol=protocol,
            protocol_path=args.protocol,
            source_lock=source_lock,
            source_lock_path=args.source_lock,
            prefix_root=args.prefix_root,
            selected_baseline_root=args.selected_baseline_root,
            output_root=args.output_root,
            candidate_family=args.candidate_family,
        )
        for case in cases
    ]
    seal = {
        "artifact_kind": "PhysTwinProb4DBiasAwarePredictionCohortSeal",
        "schema_version": 1,
        "case_count": len(cases),
        "cases": [
            {
                "case": case,
                "prediction_result_sha256": report["result_sha256"],
                "candidate_accepted": report["target_free_prediction"][
                    "candidate_accepted"
                ],
            }
            for case, report in zip(cases, reports, strict=True)
        ],
        "information_boundary": {
            "all_predictions_completed_before_future_scoring": True,
            "future_prob4d_read": False,
            "future_object_observation_read": False,
            "future_manual_tracks_read": False,
        },
    }
    if args.candidate_family == "action_conditioned":
        seal["candidate_family"] = args.candidate_family
    seal = _write_json(
        args.output_root / PREDICTION_COHORT_SEAL_FILENAME, seal
    )
    print(json.dumps(seal, indent=2, sort_keys=True))


def _score_trajectory(
    trajectory: np.ndarray,
    final_data: Mapping[str, Any],
    manual_tracks: np.ndarray,
    *,
    start_frame: int,
    end_frame: int,
) -> dict[str, float]:
    object_points = np.asarray(final_data["object_points"])
    num_surface_points = object_points.shape[1] + len(
        np.asarray(final_data["surface_points"])
    )
    result = evaluate_official_phystwin_interval(
        trajectory,
        object_points,
        np.asarray(final_data["object_visibilities"], dtype=bool),
        manual_tracks,
        num_surface_points=num_surface_points,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    return {metric: float(result[metric]) for metric in METRICS}


def _evaluate_case(
    case: str,
    row: Mapping[str, Any],
    *,
    prediction_root: Path,
) -> dict[str, Any]:
    case_dir = prediction_root / case
    report_path = case_dir / PREDICTION_REPORT_FILENAME
    report = _load_json(report_path, verify=True)
    for role in ("candidate", "guarded"):
        descriptor = report["outputs"][role]
        _require_hash(Path(descriptor["path"]), descriptor["sha256"], role)
    selected_descriptor = report["inputs"]["selected_baseline"]
    selected_path = Path(selected_descriptor["path"])
    _require_hash(
        selected_path, selected_descriptor["sha256"], "selected baseline"
    )
    source_summary_path = Path(str(row["C_summary"]))
    _require_hash(
        source_summary_path, str(row["C_summary_sha256"]), "arm-C summary"
    )
    source_summary = _load_json(source_summary_path)
    final_descriptor = source_summary["inputs"]["final_data"]
    manual_descriptor = source_summary["inputs"]["manual_tracks"]
    split_descriptor = source_summary["inputs"]["split"]
    final_path = Path(str(final_descriptor["path"]))
    manual_path = Path(str(manual_descriptor["path"]))
    split_path = Path(str(split_descriptor["path"]))
    _require_hash(final_path, final_descriptor["sha256"], "final data")
    _require_hash(manual_path, manual_descriptor["sha256"], "manual tracks")
    _require_hash(split_path, split_descriptor["sha256"], "split")
    final_data = _load_pickle(final_path)
    manual_tracks = np.asarray(_load_pickle(manual_path), dtype=np.float64)
    split = _load_json(split_path)
    train_end = int(split["train"][1])
    test_end = int(split["test"][1])
    late_start = train_end + (2 * (test_end - train_end)) // 3
    trajectories = {
        "selected_baseline": np.asarray(_load_pickle(selected_path)),
        "raw_candidate": np.asarray(
            _load_pickle(Path(report["outputs"]["candidate"]["path"]))
        ),
        "guarded_candidate": np.asarray(
            _load_pickle(Path(report["outputs"]["guarded"]["path"]))
        ),
    }
    scores = {
        arm: {
            "future": _score_trajectory(
                trajectory,
                final_data,
                manual_tracks,
                start_frame=train_end,
                end_frame=test_end,
            ),
            "late": _score_trajectory(
                trajectory,
                final_data,
                manual_tracks,
                start_frame=late_start,
                end_frame=test_end,
            ),
        }
        for arm, trajectory in trajectories.items()
    }
    accepted = bool(report["target_free_prediction"]["candidate_accepted"])
    exact_fallback = accepted or np.array_equal(
        trajectories["guarded_candidate"], trajectories["selected_baseline"]
    )
    if not exact_fallback:
        raise AssertionError(f"rejected prediction changed baseline: {case}")
    future_difference = {
        metric: scores["guarded_candidate"]["future"][metric]
        - scores["selected_baseline"]["future"][metric]
        for metric in METRICS
    }
    return {
        "case": case,
        "physical_object_cluster": phystwin_physical_object_cluster(case),
        "candidate_accepted": accepted,
        "bit_exact_fallback": exact_fallback,
        "scores": scores,
        "guarded_minus_selected_future_m": future_difference,
        "accepted_harmful": bool(
            accepted and any(value > 0.0 for value in future_difference.values())
        ),
        "prediction_result_sha256": report["result_sha256"],
        "opened_outcome_sha256": {
            "final_data": _sha256(final_path),
            "manual_tracks": _sha256(manual_path),
        },
    }


def _case_balanced_scores(
    cases: list[dict[str, Any]], arm: str, interval: str
) -> dict[str, float]:
    return {
        metric: float(
            np.mean([case["scores"][arm][interval][metric] for case in cases])
        )
        for metric in METRICS
    }


def _cluster_bootstrap(
    cases: list[dict[str, Any]], metric: str
) -> dict[str, float]:
    by_cluster: dict[str, list[float]] = {}
    for case in cases:
        by_cluster.setdefault(case["physical_object_cluster"], []).append(
            float(case["guarded_minus_selected_future_m"][metric])
        )
    cluster_means = np.asarray(
        [np.mean(values) for values in by_cluster.values()], dtype=np.float64
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.mean(
        rng.choice(
            cluster_means,
            size=(BOOTSTRAP_DRAWS, len(cluster_means)),
            replace=True,
        ),
        axis=1,
    )
    return {
        "cluster_count": len(cluster_means),
        "mean_difference_m": float(np.mean(cluster_means)),
        "lower_95_m": float(np.quantile(draws, 0.025)),
        "upper_95_m": float(np.quantile(draws, 0.975)),
    }


def evaluate(args: argparse.Namespace) -> None:
    protocol = _load_json(args.protocol)
    source_summary = _load_json(args.source_summary)
    rows = _source_rows(source_summary)
    cases_expected = _protocol_cases(protocol)
    seal_path = args.prediction_root / PREDICTION_COHORT_SEAL_FILENAME
    seal = _load_json(seal_path, verify=True)
    candidate_family = str(seal.get("candidate_family", "static"))
    if candidate_family not in {"static", "action_conditioned"}:
        raise ValueError("prediction seal candidate family is unsupported")
    if tuple(row["case"] for row in seal["cases"]) != cases_expected:
        raise ValueError("prediction seal case order differs from protocol")
    cases = [
        _evaluate_case(case, rows[case], prediction_root=args.prediction_root)
        for case in cases_expected
    ]
    scores = {
        arm: {
            interval: _case_balanced_scores(cases, arm, interval)
            for interval in ("future", "late")
        }
        for arm in ("selected_baseline", "raw_candidate", "guarded_candidate")
    }
    intervals = {metric: _cluster_bootstrap(cases, metric) for metric in METRICS}
    accepted_count = sum(case["candidate_accepted"] for case in cases)
    harmful_count = sum(case["accepted_harmful"] for case in cases)
    rejected_exact = all(
        case["candidate_accepted"] or case["bit_exact_fallback"] for case in cases
    )
    two_metric_win_or_tie = sum(
        all(value <= 0.0 for value in case["guarded_minus_selected_future_m"].values())
        for case in cases
    )
    gates = {
        "both_case_balanced_future_means_improve": all(
            scores["guarded_candidate"]["future"][metric]
            < scores["selected_baseline"]["future"][metric]
            for metric in METRICS
        ),
        "both_cluster_upper_95_bounds_below_zero": all(
            intervals[metric]["upper_95_m"] < 0.0 for metric in METRICS
        ),
        "two_metric_win_or_tie_count": two_metric_win_or_tie,
        "two_metric_win_or_tie_gate": two_metric_win_or_tie
        >= int(
            protocol["transfer_gate_for_fresh_evaluation"][
                "two_metric_win_or_tie_count_at_least"
            ]
        ),
        "accepted_harmful_case_count": harmful_count,
        "accepted_harmful_gate": harmful_count
        == int(
            protocol["transfer_gate_for_fresh_evaluation"][
                "accepted_harmful_case_count"
            ]
        ),
        "all_rejections_bit_exact": rejected_exact,
        "accepted_case_count": accepted_count,
        "minimum_accepted_case_gate": accepted_count
        >= int(
            protocol["transfer_gate_for_fresh_evaluation"][
                "minimum_accepted_case_count"
            ]
        ),
    }
    gate_names = (
        "both_case_balanced_future_means_improve",
        "both_cluster_upper_95_bounds_below_zero",
        "two_metric_win_or_tie_gate",
        "accepted_harmful_gate",
        "all_rejections_bit_exact",
        "minimum_accepted_case_gate",
    )
    gates["all_pass"] = all(bool(gates[name]) for name in gate_names)
    payload = {
        "artifact_kind": (
            "PhysTwinProb4DBiasAwareExploratoryResult"
            if candidate_family == "static"
            else "PhysTwinProb4DActionConditionedExploratoryResult"
        ),
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "case_count": len(cases),
        "physical_object_cluster_count": len(
            {case["physical_object_cluster"] for case in cases}
        ),
        "scores": scores,
        "guarded_minus_selected_cluster_bootstrap": intervals,
        "gates": gates,
        "decision": (
            "freeze-independent-prospective-evaluation"
            if gates["all_pass"]
            else "reject-prob4d-bias-guard-transfer-family"
        ),
        "cases": cases,
        "inputs_sha256": {
            "protocol": _sha256(args.protocol),
            "source_summary": _sha256(args.source_summary),
            "prediction_cohort_seal": _sha256(seal_path),
        },
        "information_boundary": {
            "prediction_cohort_sealed_before_this_command": True,
            "future_prob4d_used": False,
            "future_object_observation_used_for_prediction_or_selection": False,
            "future_manual_tracks_used_for_prediction_or_selection": False,
            "future_outcomes_used_only_here": True,
        },
        "claim_boundary": (
            "post-open exploratory PhysTwin-19 method development; passing "
            "would justify but not replace an independent evaluation"
        ),
    }
    if candidate_family == "action_conditioned":
        payload["candidate_family"] = candidate_family
    payload = _write_json(args.output, payload)
    print(
        json.dumps(
            {
                "scores": payload["scores"],
                "cluster_bootstrap": payload[
                    "guarded_minus_selected_cluster_bootstrap"
                ],
                "gates": payload["gates"],
                "decision": payload["decision"],
                "result_sha256": payload["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    args = _parse_args()
    if args.command == "stage":
        stage(args)
    elif args.command == "predict":
        predict(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
