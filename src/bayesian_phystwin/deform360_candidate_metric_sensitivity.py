"""Source-only Deform360 metric-convention sensitivity evaluation.

This evaluator holds the already-open hidden-identity population fixed while
varying point-distance and aggregation conventions. It is deliberately
separate from official Deform360 parity: no result from this module may be
labelled an official benchmark score.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360_official_parity import (
    PUBLISHED_3D_REFERENCE_SCORES,
    aggregate_metric_sensitivity,
    candidate_chamfer_metrics,
    candidate_track_metrics,
)
from .deform360_online_belief_evaluation import _sha256
from .deform360_raw_camera_observation import _load_open_case_for_evaluation


SCHEMA_VERSION = 1
ARTIFACT_KIND = "Deform360OpenSourceCandidateMetricSensitivity"
PROTOCOL_ID = "deform360-open27-candidate-metric-sensitivity-v1-development"
SOURCE_PROTOCOL_ID = "deform360-open27-raw-alltracker-pairwise-gate-v1-development"
CLAIM_LABEL = "candidate_convention_sensitivity_only"

PHYSICAL_ARM = "physical_prior"
PERSISTENCE_ARM = "persistence"
PAIRWISE_ARM = "raw_selected_backbone_full_blend_rbf_pairwise_clique"
SUPPORT_ARM = "raw_selected_backbone_full_blend_rbf_support_gated"
DEFAULT_METHODS = (PHYSICAL_ARM, PERSISTENCE_ARM, PAIRWISE_ARM, SUPPORT_ARM)
COMPARATORS = (PHYSICAL_ARM, PERSISTENCE_ARM)

METRIC_UNITS = {
    "track_coordinate_mse_m2": "m^2",
    "track_coordinate_rmse_m": "m",
    "track_mean_point_euclidean_m": "m",
    "track_point_rmse_m": "m",
    "chamfer_pred_to_target_mean_euclidean_m": "m",
    "chamfer_target_to_pred_mean_euclidean_m": "m",
    "chamfer_symmetric_mean_euclidean_m": "m",
    "chamfer_pred_to_target_mean_squared_m2": "m^2",
    "chamfer_target_to_pred_mean_squared_m2": "m^2",
    "chamfer_symmetric_mean_squared_m2": "m^2",
}
HEADLINE_METRICS = (
    "track_mean_point_euclidean_m",
    "chamfer_pred_to_target_mean_euclidean_m",
    "chamfer_symmetric_mean_euclidean_m",
)
AGGREGATIONS = (
    "frame_pooled_mean",
    "episode_balanced_mean",
    "object_balanced_mean",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evaluate_candidate_metric_arrays(
    predictions_m: Mapping[str, np.ndarray],
    target_m: np.ndarray,
    visibility: np.ndarray,
    validity: np.ndarray,
    *,
    center_ids: np.ndarray,
    scored_frames: Sequence[int],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Evaluate explicit metric variants on permanently hidden identities."""

    target = np.asarray(target_m, dtype=float)
    visible = np.asarray(visibility, dtype=bool)
    valid = np.asarray(validity, dtype=bool)
    centers = np.asarray(center_ids, dtype=np.int64)
    frames = tuple(int(frame) for frame in scored_frames)
    _require(
        target.ndim == 3 and target.shape[2] == 3,
        "target must have shape (T,N,3)",
    )
    _require(visible.shape == target.shape[:2], "visibility shape differs")
    _require(valid.shape == target.shape[:2], "validity shape differs")
    _require(bool(frames), "scored frames are empty")
    _require(
        min(frames) >= 0 and max(frames) < len(target),
        "scored frame exceeds target",
    )
    _require(
        centers.ndim == 1 and len(centers) == len(np.unique(centers)),
        "center IDs must be a unique vector",
    )
    _require(
        np.all((centers >= 0) & (centers < target.shape[1])),
        "center ID exceeds target",
    )
    _require(bool(predictions_m), "predictions are empty")

    hidden = np.ones(target.shape[1], dtype=bool)
    hidden[centers] = False
    results: dict[str, dict[str, list[float]]] = {
        str(method): {metric: [] for metric in METRIC_UNITS} for method in predictions_m
    }
    support_counts: list[int] = []
    for frame in frames:
        target_finite = np.all(np.isfinite(target[frame]), axis=1)
        base_mask = hidden & visible[frame] & valid[frame] & target_finite
        support_counts.append(int(np.sum(base_mask)))
        _require(support_counts[-1] > 0, f"frame {frame} has no hidden support")
        for method, trajectory_m in predictions_m.items():
            trajectory = np.asarray(trajectory_m, dtype=float)
            _require(
                trajectory.shape == target.shape,
                f"{method} trajectory shape differs from target",
            )
            mask = base_mask & np.all(np.isfinite(trajectory[frame]), axis=1)
            _require(np.any(mask), f"{method} frame {frame} has no finite support")
            prediction = trajectory[frame, mask]
            truth = target[frame, mask]
            track = candidate_track_metrics(prediction, truth)
            chamfer = candidate_chamfer_metrics(prediction, truth)
            values = {
                "track_coordinate_mse_m2": track["coordinate_mse_m2"],
                "track_coordinate_rmse_m": track["coordinate_rmse_m"],
                "track_mean_point_euclidean_m": track["mean_point_euclidean_m"],
                "track_point_rmse_m": track["point_rmse_m"],
                "chamfer_pred_to_target_mean_euclidean_m": chamfer[
                    "pred_to_target_mean_euclidean_m"
                ],
                "chamfer_target_to_pred_mean_euclidean_m": chamfer[
                    "target_to_pred_mean_euclidean_m"
                ],
                "chamfer_symmetric_mean_euclidean_m": chamfer[
                    "symmetric_mean_euclidean_m"
                ],
                "chamfer_pred_to_target_mean_squared_m2": chamfer[
                    "pred_to_target_mean_squared_m2"
                ],
                "chamfer_target_to_pred_mean_squared_m2": chamfer[
                    "target_to_pred_mean_squared_m2"
                ],
                "chamfer_symmetric_mean_squared_m2": chamfer[
                    "symmetric_mean_squared_m2"
                ],
            }
            for metric, value in values.items():
                results[str(method)][metric].append(float(value))

    arrays = {
        method: {
            metric: np.asarray(values, dtype=float)
            for metric, values in method_metrics.items()
        }
        for method, method_metrics in results.items()
    }
    support = {
        "permanently_excluded_center_count": int(len(centers)),
        "scored_frame_count": int(len(frames)),
        "hidden_support_per_frame": {
            "minimum": int(np.min(support_counts)),
            "mean": float(np.mean(support_counts)),
            "maximum": int(np.max(support_counts)),
        },
    }
    return arrays, support


def _object_cluster_interval(
    differences: Mapping[str, float],
    case_to_object: Mapping[str, str],
    *,
    draws: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    object_ids = tuple(sorted(set(case_to_object.values())))
    _require(len(object_ids) >= 2, "object bootstrap requires multiple objects")
    object_differences = {
        object_id: float(
            np.mean(
                [
                    differences[case]
                    for case, assigned in case_to_object.items()
                    if assigned == object_id
                ]
            )
        )
        for object_id in object_ids
    }
    values = np.asarray(list(object_differences.values()), dtype=float)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(
        axis=1
    )
    return {
        "episode_mean_difference": float(np.mean(list(differences.values()))),
        "object_balanced_mean_difference": float(np.mean(values)),
        "object_cluster_lower_95": float(np.quantile(sampled, 0.025)),
        "object_cluster_upper_95": float(np.quantile(sampled, 0.975)),
        "object_cluster_probability_improved": float(np.mean(sampled < 0.0)),
        "object_differences": object_differences,
    }


def summarize_candidate_metric_sensitivity(
    case_metrics: Mapping[
        str, Mapping[str, Mapping[str, Sequence[float] | np.ndarray]]
    ],
    case_to_object: Mapping[str, str],
    *,
    primary_method: str = PAIRWISE_ARM,
    comparators: Sequence[str] = COMPARATORS,
) -> dict[str, Any]:
    """Aggregate candidate conventions and test metric-robust improvement."""

    _require(primary_method in case_metrics, "primary method is missing")
    _require(set(comparators).issubset(case_metrics), "comparator is missing")
    methods = tuple(case_metrics)
    expected_cases = tuple(sorted(case_to_object))
    expected_case_set = set(expected_cases)
    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    case_means: dict[str, dict[str, dict[str, float]]] = {}
    object_means: dict[str, dict[str, dict[str, float]]] = {}
    for method in methods:
        _require(
            set(case_metrics[method]) == set(METRIC_UNITS),
            f"{method} metric set changed",
        )
        aggregate[method] = {}
        case_means[method] = {}
        object_means[method] = {}
        for metric, values_by_case in case_metrics[method].items():
            _require(
                set(values_by_case) == expected_case_set,
                f"{method} {metric} case set changed",
            )
            aggregate[method][metric] = aggregate_metric_sensitivity(
                values_by_case, case_to_object
            )
            case_means[method][metric] = {
                case: float(np.mean(np.asarray(values, dtype=float)))
                for case, values in values_by_case.items()
            }
            object_means[method][metric] = {
                object_id: float(
                    np.mean(
                        [
                            case_means[method][metric][case]
                            for case, assigned in case_to_object.items()
                            if assigned == object_id
                        ]
                    )
                )
                for object_id in sorted(set(case_to_object.values()))
            }

    comparisons: dict[str, Any] = {}
    for comparator in comparators:
        comparisons[comparator] = {}
        for metric, unit in METRIC_UNITS.items():
            differences = {
                case: (
                    case_means[primary_method][metric][case]
                    - case_means[comparator][metric][case]
                )
                for case in expected_cases
            }
            interval = _object_cluster_interval(differences, case_to_object)
            relative = {}
            for aggregation in AGGREGATIONS:
                candidate_value = aggregate[primary_method][metric][aggregation]
                comparator_value = aggregate[comparator][metric][aggregation]
                relative[aggregation] = (
                    None
                    if comparator_value == 0.0
                    else candidate_value / comparator_value - 1.0
                )
            interval.update(
                {
                    "unit": unit,
                    "relative_change": relative,
                    "episode_wins": int(
                        np.sum(np.asarray(list(differences.values())) < 0.0)
                    ),
                    "episode_count": len(differences),
                    "object_wins": int(
                        np.sum(
                            np.asarray(
                                [
                                    object_means[primary_method][metric][object_id]
                                    - object_means[comparator][metric][object_id]
                                    for object_id in sorted(
                                        set(case_to_object.values())
                                    )
                                ]
                            )
                            < 0.0
                        )
                    ),
                    "object_count": len(set(case_to_object.values())),
                }
            )
            comparisons[comparator][metric] = interval

    checks = []
    for comparator in comparators:
        for metric in HEADLINE_METRICS:
            for aggregation in AGGREGATIONS:
                change = comparisons[comparator][metric]["relative_change"][aggregation]
                checks.append(
                    {
                        "comparator": comparator,
                        "metric": metric,
                        "aggregation": aggregation,
                        "relative_change": change,
                        "passed": change is not None and change < 0.0,
                    }
                )
    return {
        "aggregate": aggregate,
        "case_means": case_means,
        "object_means": object_means,
        "comparisons": comparisons,
        "metric_robustness_gate": {
            "rule": (
                "The frozen pairwise arm must improve over physical prior and "
                "persistence for all three explicit metre-valued headline metrics "
                "under frame-, episode-, and object-balanced aggregation."
            ),
            "passed": all(bool(check["passed"]) for check in checks),
            "checks": checks,
        },
    }


def evaluate_open_source_candidate_metric_sensitivity(
    prediction_root: str | Path,
    source_panel_root: str | Path,
) -> dict[str, Any]:
    """Evaluate the fixed open-27 result without claiming official parity."""

    prediction_dir = Path(prediction_root).resolve()
    panel_dir = Path(source_panel_root).resolve()
    summary_path = prediction_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _require(
        summary.get("protocol_id") == SOURCE_PROTOCOL_ID,
        "source prediction protocol changed",
    )
    _require(
        "not held-target or official benchmark evidence"
        in str(summary.get("claim_boundary")),
        "source prediction claim boundary changed",
    )
    artifacts = summary.get("artifacts")
    _require(isinstance(artifacts, list) and len(artifacts) == 27, "open panel changed")

    case_metrics: dict[str, dict[str, dict[str, np.ndarray]]] = {
        method: {metric: {} for metric in METRIC_UNITS} for method in DEFAULT_METHODS
    }
    case_to_object: dict[str, str] = {}
    support_by_case: dict[str, Any] = {}
    input_artifacts: list[dict[str, Any]] = []
    for artifact in artifacts:
        case = str(artifact["case"])
        report_path = prediction_dir / f"{case}.json"
        archive_path = prediction_dir / f"{case}.npz"
        _require(
            _sha256(report_path) == artifact["report_sha256"],
            f"{case} report checksum changed",
        )
        _require(
            _sha256(archive_path) == artifact["archive_sha256"],
            f"{case} archive checksum changed",
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        _require(report.get("case") == case, f"{case} report identity changed")
        seal, prior, persistence, target, visibility, validity = (
            _load_open_case_for_evaluation(panel_dir / case)
        )
        with np.load(archive_path, allow_pickle=False) as stored:
            _require(
                set(DEFAULT_METHODS).issubset(stored.files),
                f"{case} candidate arm is missing",
            )
            predictions = {
                method: np.asarray(stored[method]).copy() for method in DEFAULT_METHODS
            }
        _require(
            np.array_equal(predictions[PHYSICAL_ARM], prior),
            f"{case} physical prior lineage changed",
        )
        _require(
            np.array_equal(predictions[PERSISTENCE_ARM], persistence),
            f"{case} persistence lineage changed",
        )
        evaluated, support = evaluate_candidate_metric_arrays(
            predictions,
            target,
            visibility,
            validity,
            center_ids=np.asarray(report["center_ids"], dtype=np.int64),
            scored_frames=report["scored_frames"],
        )
        for method, metrics in evaluated.items():
            for metric, values in metrics.items():
                case_metrics[method][metric][case] = values
        object_id = str(report["object_id"])
        _require(object_id == str(seal["object_id"]), f"{case} object changed")
        case_to_object[case] = object_id
        support_by_case[case] = support
        target_path = panel_dir / case / "target_data.pkl"
        input_artifacts.append(
            {
                "case": case,
                "prediction_report_sha256": artifact["report_sha256"],
                "prediction_archive_sha256": artifact["archive_sha256"],
                "source_target_sha256": _sha256(target_path),
            }
        )

    summary_metrics = summarize_candidate_metric_sensitivity(
        case_metrics, case_to_object
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "claim_label": CLAIM_LABEL,
        "official_parity_ready": False,
        "case_count": len(case_to_object),
        "physical_object_count": len(set(case_to_object.values())),
        "primary_method": PAIRWISE_ARM,
        "diagnostic_control_method": SUPPORT_ARM,
        "comparators": list(COMPARATORS),
        "metric_units": METRIC_UNITS,
        "population_contract": {
            "identities": (
                "permanently hidden material identities; all assimilation centres "
                "are excluded from every metric"
            ),
            "per_frame_mask": (
                "source visibility and validity, finite target, and finite prediction"
            ),
            "future_frames": (
                "the source report's already-open post-update scoring frames"
            ),
            "purpose": (
                "hold the evaluated population fixed while varying only distance "
                "and aggregation conventions"
            ),
        },
        "source_protocol": {
            "protocol_id": SOURCE_PROTOCOL_ID,
            "summary_sha256": _sha256(summary_path),
            "claim_boundary": summary["claim_boundary"],
        },
        "input_artifacts": input_artifacts,
        "support_by_case": support_by_case,
        **summary_metrics,
        "published_reference_context": {
            "scores": PUBLISHED_3D_REFERENCE_SCORES,
            "comparison_allowed": False,
            "reason": (
                "The official Deform360 evaluator, exact split, frame policy, "
                "coordinate convention, units, and aggregation contract remain "
                "unresolved. Published values are context, not denominators."
            ),
        },
        "decision": {
            "larger_fresh_evaluation_justified": bool(
                summary_metrics["metric_robustness_gate"]["passed"]
            ),
            "next_gate": (
                "Obtain an authoritative evaluator contract, then preregister a "
                "fresh-object evaluation of the unchanged pairwise-gated method."
            ),
            "method_change_authorized": False,
        },
        "claim_boundary": (
            "Already-open five-object/27-episode source-development evidence only. "
            "This audit tests metric-convention robustness; it is neither held "
            "confirmation nor official Deform360 parity and cannot support a SOTA "
            "claim."
        ),
    }
    result["result_sha256"] = _canonical_sha256(result)
    return result


def write_candidate_metric_sensitivity(
    payload: Mapping[str, Any], output_path: str | Path
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "ARTIFACT_KIND",
    "CLAIM_LABEL",
    "PAIRWISE_ARM",
    "PROTOCOL_ID",
    "evaluate_candidate_metric_arrays",
    "evaluate_open_source_candidate_metric_sensitivity",
    "summarize_candidate_metric_sensitivity",
    "write_candidate_metric_sensitivity",
]
