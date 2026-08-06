#!/usr/bin/env python3
"""Open staged source outcomes after TAPNext++ assimilation predictions seal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.phystwin_official_evaluation import _nearest_distances
from bayesian_phystwin.phystwin_tapnextpp_competence import (
    canonical_sha256,
    file_sha256,
)

PREDICTION_FILENAME = "tapnextpp_sparse_assimilation_prediction.npz"
REPORT_FILENAME = "tapnextpp_sparse_assimilation_prediction_report.json"
SEAL_FILENAME = "tapnextpp_sparse_assimilation_prediction_seal.json"
CASE_RESULT_FILENAME = "tapnextpp_sparse_assimilation_case_result.json"
SUMMARY_FILENAME = "tapnextpp_sparse_assimilation_source_summary.json"
CHI_SQUARE_3D_90 = 6.251388631170325


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON artifact is not an object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _case_record(manifest: dict[str, Any], case_name: str) -> dict[str, Any]:
    records = [
        record
        for record in manifest["case_records"]
        if record.get("case") == case_name
    ]
    _require(len(records) == 1, f"source manifest does not bind {case_name} once")
    return records[0]


def _safe_mean(values: np.ndarray) -> float | None:
    finite = np.isfinite(values)
    return float(np.mean(values[finite])) if np.any(finite) else None


def _horizon_slices(frame_count: int) -> dict[str, np.ndarray]:
    chunks = np.array_split(np.arange(frame_count, dtype=np.int64), 3)
    return dict(zip(("early", "middle", "late"), chunks, strict=True))


def _evaluate_arm(
    trajectory: np.ndarray,
    variance_m2: np.ndarray,
    frame_zero: np.ndarray,
    observed_points: np.ndarray,
    visible: np.ndarray,
    manual_frame_zero: np.ndarray,
    manual_future: np.ndarray,
    provider_ids: np.ndarray,
    *,
    num_surface_points: int,
) -> dict[str, Any]:
    frame_count = len(trajectory)
    _require(
        trajectory.ndim == 3
        and trajectory.shape[0] == frame_count
        and trajectory.shape[2] == 3,
        "future trajectory shape changed",
    )
    _require(variance_m2.shape == trajectory.shape[:2], "variance shape changed")
    _require(observed_points.shape[0] == frame_count, "outcome frame count changed")
    _require(manual_future.shape[0] == frame_count, "manual frame count changed")
    finite_initial = np.all(np.isfinite(manual_frame_zero), axis=1)
    initial_ids = np.flatnonzero(finite_initial)
    _, mapped_nodes = _nearest_distances(
        frame_zero,
        manual_frame_zero[finite_initial],
        p=2,
    )
    observed_identity = np.isin(initial_ids, provider_ids)
    hidden_identity = ~observed_identity
    chamfer = np.empty(frame_count, dtype=np.float64)
    track = {
        "all": np.full(frame_count, np.nan, dtype=np.float64),
        "observed": np.full(frame_count, np.nan, dtype=np.float64),
        "hidden": np.full(frame_count, np.nan, dtype=np.float64),
    }
    nees = {name: np.full(frame_count, np.nan) for name in track}
    coverage = {name: np.full(frame_count, np.nan) for name in track}
    groups = {
        "all": np.ones(len(initial_ids), dtype=bool),
        "observed": observed_identity,
        "hidden": hidden_identity,
    }
    for frame in range(frame_count):
        current = observed_points[frame, visible[frame]]
        distance, _ = _nearest_distances(
            trajectory[frame, :num_surface_points],
            current,
            p=1,
        )
        chamfer[frame] = float(np.mean(distance))
        target = manual_future[frame, initial_ids]
        finite_target = np.all(np.isfinite(target), axis=1)
        prediction = trajectory[frame, mapped_nodes]
        residual = prediction - target
        squared = np.sum(np.square(residual), axis=1)
        predictive_variance = np.maximum(
            variance_m2[frame, mapped_nodes],
            1e-12,
        )
        for name, group in groups.items():
            selected = finite_target & group
            if not np.any(selected):
                continue
            radial = np.sqrt(squared[selected])
            track[name][frame] = float(np.mean(radial))
            row_nees = squared[selected] / predictive_variance[selected]
            nees[name][frame] = float(np.mean(row_nees))
            coverage[name][frame] = float(
                np.mean(row_nees <= CHI_SQUARE_3D_90)
            )
    horizons: dict[str, Any] = {}
    for name, indices in _horizon_slices(frame_count).items():
        horizons[name] = {
            "chamfer_distance_m": float(np.mean(chamfer[indices])),
            "track_error_m": _safe_mean(track["all"][indices]),
            "observed_track_error_m": _safe_mean(track["observed"][indices]),
            "hidden_track_error_m": _safe_mean(track["hidden"][indices]),
        }
    return {
        "chamfer_distance_m": float(np.mean(chamfer)),
        "track_error_m": _safe_mean(track["all"]),
        "observed_track_error_m": _safe_mean(track["observed"]),
        "hidden_track_error_m": _safe_mean(track["hidden"]),
        "late_track_error_m": horizons["late"]["track_error_m"],
        "late_hidden_track_error_m": horizons["late"]["hidden_track_error_m"],
        "all_track_nees": _safe_mean(nees["all"]),
        "all_track_coverage_90": _safe_mean(coverage["all"]),
        "observed_track_coverage_90": _safe_mean(coverage["observed"]),
        "hidden_track_coverage_90": _safe_mean(coverage["hidden"]),
        "hidden_future_frame_support": float(np.mean(np.isfinite(track["hidden"]))),
        "horizons": horizons,
        "by_frame": {
            "chamfer_distance_m": chamfer.tolist(),
            "track_error_m": track["all"].tolist(),
            "observed_track_error_m": track["observed"].tolist(),
            "hidden_track_error_m": track["hidden"].tolist(),
            "all_track_nees": nees["all"].tolist(),
            "all_track_coverage_90": coverage["all"].tolist(),
        },
        "track_identity_counts": {
            "all": int(len(initial_ids)),
            "observed": int(np.sum(observed_identity)),
            "hidden": int(np.sum(hidden_identity)),
        },
        "official_track_node_indices": mapped_nodes.tolist(),
    }


def evaluate_case(
    protocol_path: str | Path,
    source_manifest_path: str | Path,
    case_name: str,
    prediction_dir: str | Path,
    withheld_outcome_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Open one staged source future only after validating its prediction seal."""

    protocol_file = Path(protocol_path).resolve()
    manifest_file = Path(source_manifest_path).resolve()
    prediction = Path(prediction_dir).resolve()
    withheld = Path(withheld_outcome_path).resolve()
    output = Path(output_path).resolve()
    _require(not output.exists(), "case result already exists")
    protocol = _load_json(protocol_file)
    manifest = _load_json(manifest_file)
    _require(
        manifest.get("result_sha256") == canonical_sha256(manifest),
        "source manifest hash changed",
    )
    record = _case_record(manifest, case_name)
    _require(
        file_sha256(withheld) == record["withheld_outcome"]["sha256"],
        "withheld source outcome changed after staging",
    )
    archive_path = prediction / PREDICTION_FILENAME
    report_path = prediction / REPORT_FILENAME
    seal_path = prediction / SEAL_FILENAME
    for path in (archive_path, report_path, seal_path):
        _require(path.is_file(), f"prediction artifact is missing: {path}")
    seal = _load_json(seal_path)
    _require(
        seal.get("result_sha256") == canonical_sha256(seal),
        "prediction seal hash changed",
    )
    _require(
        seal.get("prediction_archive_sha256") == file_sha256(archive_path)
        and seal.get("prediction_report_sha256") == file_sha256(report_path),
        "prediction changed after sealing",
    )
    report = _load_json(report_path)
    _require(report.get("case") == case_name, "prediction case changed")
    _require(
        report.get("result_sha256") == canonical_sha256(report),
        "prediction report hash changed",
    )
    _require(
        report["information_boundary"]["future_real_outcome_read"] is False,
        "prediction report crossed the future-outcome boundary",
    )
    with np.load(archive_path, allow_pickle=False) as stored:
        frame_zero = np.asarray(stored["physical_frame_zero_m"], np.float64)
        trajectories = {
            "physical": np.asarray(stored["physical_future_m"], np.float64),
            "dense_persistence": np.asarray(
                stored["dense_persistence_future_m"],
                np.float64,
            ),
            "tapnext_direct": np.asarray(
                stored["tapnext_direct_future_m"],
                np.float64,
            ),
            "tapnext_graph": np.asarray(
                stored["tapnext_graph_future_m"],
                np.float64,
            ),
        }
        variances = {
            "physical": np.asarray(stored["physical_variance_m2"], np.float64),
            "dense_persistence": np.asarray(
                stored["dense_persistence_variance_m2"],
                np.float64,
            ),
            "tapnext_direct": np.asarray(
                stored["tapnext_direct_variance_m2"],
                np.float64,
            ),
            "tapnext_graph": np.asarray(
                stored["tapnext_graph_variance_m2"],
                np.float64,
            ),
        }
        provider_ids = np.asarray(stored["provider_identity_ids"], np.int64)
        num_surface_points = int(stored["num_surface_points"])
        train_end = int(stored["train_end_frame_exclusive"])
        future_end = int(stored["future_end_frame_exclusive"])
    with np.load(withheld, allow_pickle=False) as stored:
        observed_points = np.asarray(stored["future_object_points_m"], np.float64)
        visible = np.asarray(stored["future_object_visibilities"], bool)
        manual_frame_zero = np.asarray(
            stored["manual_track_frame_zero_m"],
            np.float64,
        )
        manual_future = np.asarray(stored["future_manual_tracks_m"], np.float64)
        withheld_ids = np.asarray(stored["provider_identity_ids"], np.int64)
        withheld_train_end = int(stored["train_end_frame_exclusive"])
        withheld_future_end = int(stored["future_end_frame_exclusive"])
    _require(np.array_equal(provider_ids, withheld_ids), "provider identities changed")
    _require(
        (train_end, future_end) == (withheld_train_end, withheld_future_end),
        "future interval changed",
    )
    arms = {
        name: _evaluate_arm(
            trajectory,
            variances[name],
            frame_zero,
            observed_points,
            visible,
            manual_frame_zero,
            manual_future,
            provider_ids,
            num_surface_points=num_surface_points,
        )
        for name, trajectory in trajectories.items()
    }
    dense = arms["dense_persistence"]
    graph = arms["tapnext_graph"]
    direct = arms["tapnext_direct"]

    def gain(candidate: dict[str, Any], metric: str) -> float | None:
        baseline_value = dense[metric]
        candidate_value = candidate[metric]
        if baseline_value is None or candidate_value is None or baseline_value <= 0.0:
            return None
        return 1.0 - float(candidate_value) / float(baseline_value)

    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinTAPNextPPSparseAssimilationCaseResult",
        "protocol_id": protocol["protocol_id"],
        "case": case_name,
        "provider_gate_passed": bool(
            report["sparse_update"]["provider_gate_passed"]
        ),
        "sparse_update_accepted": bool(report["sparse_update"]["accepted"]),
        "exact_dense_fallback": bool(
            report["sparse_update"]["exact_dense_fallback"]
        ),
        "arms": arms,
        "incremental_gain_over_dense_persistence": {
            "tapnext_direct": {
                metric: gain(direct, metric)
                for metric in (
                    "chamfer_distance_m",
                    "track_error_m",
                    "observed_track_error_m",
                    "hidden_track_error_m",
                    "late_track_error_m",
                )
            },
            "tapnext_graph": {
                metric: gain(graph, metric)
                for metric in (
                    "chamfer_distance_m",
                    "track_error_m",
                    "observed_track_error_m",
                    "hidden_track_error_m",
                    "late_track_error_m",
                )
            },
        },
        "primary_joint_nonregression": bool(
            graph["chamfer_distance_m"] <= dense["chamfer_distance_m"]
            and graph["track_error_m"] <= dense["track_error_m"]
        ),
        "inputs": {
            "protocol_sha256": file_sha256(protocol_file),
            "source_manifest_sha256": file_sha256(manifest_file),
            "prediction_archive_sha256": file_sha256(archive_path),
            "prediction_report_sha256": file_sha256(report_path),
            "prediction_seal_sha256": file_sha256(seal_path),
            "withheld_outcome_sha256": file_sha256(withheld),
        },
        "information_boundary": {
            "prediction_sealed_before_future_open": True,
            "source_future_opened_for_evaluation": True,
            "cohort_previously_opened_exploratory": True,
            "held_v8_accessed": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    result["result_sha256"] = canonical_sha256(result)
    _write_json(output, result)
    return result


def aggregate_results(
    protocol_path: str | Path,
    source_manifest_path: str | Path,
    result_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Apply the frozen source transfer and calibration gates."""

    protocol_file = Path(protocol_path).resolve()
    manifest_file = Path(source_manifest_path).resolve()
    root = Path(result_root).resolve()
    output = Path(output_path).resolve()
    _require(not output.exists(), "aggregate result already exists")
    protocol = _load_json(protocol_file)
    manifest = _load_json(manifest_file)
    results = []
    dispositions = []
    for case_name in protocol["fixed_source_cases"]:
        path = root / case_name / CASE_RESULT_FILENAME
        _require(path.is_file(), f"case result is missing: {case_name}")
        result = _load_json(path)
        _require(
            result.get("result_sha256") == canonical_sha256(result),
            f"case result hash changed: {case_name}",
        )
        results.append(result)
        dispositions.append(
            {
                "case": case_name,
                "result_sha256": file_sha256(path),
                "provider_gate_passed": result["provider_gate_passed"],
                "sparse_update_accepted": result["sparse_update_accepted"],
                "primary_joint_nonregression": result[
                    "primary_joint_nonregression"
                ],
            }
        )

    metric_names = (
        "chamfer_distance_m",
        "track_error_m",
        "observed_track_error_m",
        "hidden_track_error_m",
        "late_track_error_m",
        "late_hidden_track_error_m",
        "all_track_nees",
        "all_track_coverage_90",
        "hidden_future_frame_support",
    )
    arm_names = ("physical", "dense_persistence", "tapnext_direct", "tapnext_graph")
    aggregate: dict[str, Any] = {}
    for arm in arm_names:
        aggregate[arm] = {}
        for metric in metric_names:
            values = [result["arms"][arm][metric] for result in results]
            finite = [float(value) for value in values if value is not None]
            aggregate[arm][metric] = float(np.mean(finite)) if finite else None

    dense = aggregate["dense_persistence"]
    graph = aggregate["tapnext_graph"]

    def relative_gain(metric: str) -> float | None:
        if dense[metric] is None or graph[metric] is None or dense[metric] <= 0.0:
            return None
        return 1.0 - graph[metric] / dense[metric]

    gains = {
        metric: relative_gain(metric)
        for metric in (
            "chamfer_distance_m",
            "track_error_m",
            "observed_track_error_m",
            "hidden_track_error_m",
            "late_track_error_m",
        )
    }
    gate_config = protocol["advancement_gates"]
    joint_wins = sum(result["primary_joint_nonregression"] for result in results)
    failed_provider_results = [
        result for result in results if not result["provider_gate_passed"]
    ]
    exact_fallback = all(
        result["exact_dense_fallback"] for result in failed_provider_results
    )
    graph_coverage = graph["all_track_coverage_90"]
    dense_coverage = dense["all_track_coverage_90"]
    gates = {
        "chamfer_gain": gains["chamfer_distance_m"] is not None
        and gains["chamfer_distance_m"]
        >= float(gate_config["minimum_chamfer_gain"]),
        "all_track_gain": gains["track_error_m"] is not None
        and gains["track_error_m"]
        >= float(gate_config["minimum_all_track_gain"]),
        "observed_track_gain": gains["observed_track_error_m"] is not None
        and gains["observed_track_error_m"]
        >= float(gate_config["minimum_observed_track_gain"]),
        "hidden_track_nonregression": gains["hidden_track_error_m"] is not None
        and gains["hidden_track_error_m"]
        >= -float(gate_config["maximum_hidden_track_regression"]),
        "joint_case_nonregression": joint_wins
        >= int(gate_config["minimum_joint_case_nonregression_count"]),
        "failed_provider_exact_fallback": exact_fallback,
        "hidden_future_support": graph["hidden_future_frame_support"]
        >= float(gate_config["minimum_hidden_future_frame_support"]),
        "coverage_floor": graph_coverage is not None
        and graph_coverage >= float(gate_config["minimum_coverage_90"]),
        "coverage_not_farther_from_nominal": graph_coverage is not None
        and dense_coverage is not None
        and abs(graph_coverage - 0.9) <= abs(dense_coverage - 0.9),
    }
    passed = all(gates.values())
    summary: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinTAPNextPPSparseAssimilationSourceSummary",
        "protocol_id": protocol["protocol_id"],
        "case_count": len(results),
        "arms": aggregate,
        "tapnext_graph_incremental_gain_over_dense_persistence": gains,
        "joint_case_nonregression_count": joint_wins,
        "gates": gates,
        "gate_passed": passed,
        "decision": (
            "authorize-independent-preregistered-evaluation"
            if passed
            else "stop-before-independent-evaluation"
        ),
        "case_dispositions": dispositions,
        "inputs": {
            "protocol_sha256": file_sha256(protocol_file),
            "source_manifest_sha256": file_sha256(manifest_file),
            "source_manifest_result_sha256": manifest["result_sha256"],
        },
        "information_boundary": {
            "cohort": "eight fixed previously opened source cases",
            "future_metrics_opened_after_per_case_prediction_seals": True,
            "independent_target_opened": False,
            "held_v8_accessed": False,
            "failed_cases_replaced": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    summary["result_sha256"] = canonical_sha256(summary)
    _write_json(output, summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    case = subparsers.add_parser("case")
    case.add_argument("--protocol", type=Path, required=True)
    case.add_argument("--source-manifest", type=Path, required=True)
    case.add_argument("--case", required=True)
    case.add_argument("--prediction-dir", type=Path, required=True)
    case.add_argument("--withheld-outcome", type=Path, required=True)
    case.add_argument("--output", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--protocol", type=Path, required=True)
    aggregate.add_argument("--source-manifest", type=Path, required=True)
    aggregate.add_argument("--result-root", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "case":
        result = evaluate_case(
            args.protocol,
            args.source_manifest,
            args.case,
            args.prediction_dir,
            args.withheld_outcome,
            args.output,
        )
    else:
        result = aggregate_results(
            args.protocol,
            args.source_manifest,
            args.result_root,
            args.output,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

