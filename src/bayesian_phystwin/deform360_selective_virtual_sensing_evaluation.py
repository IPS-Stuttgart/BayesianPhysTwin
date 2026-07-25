"""Frozen scoring and object-level inference for selective virtual sensing."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360_online_belief_evaluation import (
    _sha256,
    score_deform360_hidden_trajectory,
)
from .deform360_raw_camera_observation import MANIFEST_FILENAME, MEASUREMENT_FILENAME
from .deform360_raw_pairwise_correspondence_diagnostic import (
    CPD_ARM,
    PERSISTENCE_CLIQUE_RBF_ARM,
    UNGATED_RBF_ARM,
)
from .deform360_selective_virtual_sensing_artifacts import (
    VIRTUAL_SENSING_ARCHIVE_FILENAME,
    VIRTUAL_SENSING_REPORT_FILENAME,
    authorize_selective_target_case,
    validate_selective_prediction_cohort_seal,
)
from .deform360_selective_virtual_sensing_protocol import (
    EXPECTED_STRATA,
    PROTOCOL_ID,
    load_selective_virtual_sensing_protocol,
)


TARGET_ARCHIVE_FILENAME = "target_trajectory.npz"
OUTCOME_MANIFEST_FILENAME = "authorized_outcome_manifest.json"
CASE_EVALUATION_FILENAME = "evaluation.json"
PRIMARY_METRICS = (
    "post_update_hidden_identity_rmse_m",
    "post_update_hidden_symmetric_chamfer_m",
)
SCORED_FRAMES = tuple(
    [*range(20, 38), *range(39, 57), *range(58, 76)]
)
ARM_TO_ARCHIVE_KEY = {
    PERSISTENCE_CLIQUE_RBF_ARM: "prediction_m",
    "persistence": "persistence_m",
    UNGATED_RBF_ARM: "ungated_rbf_m",
    CPD_ARM: "independent_cpd_m",
}
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("result_sha256", None)
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def score_selective_virtual_sensing_arrays(
    trajectories_m: Mapping[str, np.ndarray],
    target_m: np.ndarray,
    target_visibility: np.ndarray,
    target_validity: np.ndarray,
    *,
    center_ids: np.ndarray,
) -> dict[str, dict[str, object]]:
    """Score every sealed arm on permanently hidden post-update identities."""

    _require(
        tuple(trajectories_m) == tuple(ARM_TO_ARCHIVE_KEY),
        "trajectory arm order or inventory changed",
    )
    centers = np.asarray(center_ids, dtype=np.int64)
    _require(
        centers.shape == (16,) and len(np.unique(centers)) == 16,
        "scoring center inventory changed",
    )
    target = np.asarray(target_m)
    _require(
        target.ndim == 3 and target.shape[0] == 76 and target.shape[2] == 3,
        "target trajectory must have shape (76,N,3)",
    )
    return {
        arm: score_deform360_hidden_trajectory(
            trajectory,
            target,
            target_visibility,
            target_validity,
            center_ids=centers,
            scored_frames=SCORED_FRAMES,
        )
        for arm, trajectory in trajectories_m.items()
    }


def _validate_outcome_manifest(
    outcome: Mapping[str, Any],
    *,
    protocol_config_sha256: str,
    record: Mapping[str, Any],
    cohort_result_sha256: str,
    prediction_result_sha256: str,
    outcome_dir: Path,
) -> Path:
    _require(
        outcome.get("artifact_kind") == "Deform360SelectiveAuthorizedOutcome"
        and outcome.get("protocol_id") == PROTOCOL_ID
        and outcome.get("protocol_config_sha256") == protocol_config_sha256
        and outcome.get("result_sha256") == _canonical_sha256(outcome),
        "authorized outcome manifest changed",
    )
    _require(
        all(outcome.get(key) == value for key, value in record.items()),
        "authorized outcome case identity changed",
    )
    _require(
        outcome.get("authorization", {}).get(
            "prediction_cohort_result_sha256"
        )
        == cohort_result_sha256
        and outcome.get("authorization", {}).get("prediction_result_sha256")
        == prediction_result_sha256,
        "authorized outcome belongs to another prediction seal",
    )
    boundary = outcome.get("information_boundary", {})
    _require(
        boundary.get(
            "eligible_prediction_cohort_verified_before_target_construction"
        )
        is True
        and boundary.get("future_tactile_read") is False
        and boundary.get("prediction_metric_computed") is False,
        "outcome construction crossed its metric boundary",
    )
    archive = outcome_dir / TARGET_ARCHIVE_FILENAME
    _require(
        archive.is_file()
        and outcome.get("output", {}).get("target_archive_sha256")
        == _sha256(archive),
        "authorized target archive changed",
    )
    return archive


def _measurement_target_audit(
    measurement_dir: Path,
    prediction_report: Mapping[str, Any],
    target: np.ndarray,
    center_ids: np.ndarray,
) -> dict[str, object]:
    manifest_path = measurement_dir / MANIFEST_FILENAME
    archive_path = measurement_dir / MEASUREMENT_FILENAME
    expected = prediction_report["inputs_sha256"]
    _require(
        _sha256(manifest_path) == expected["measurement_manifest"]
        and _sha256(archive_path) == expected["measurement_archive"],
        "measurement audit input differs from the sealed prediction",
    )
    with np.load(archive_path, allow_pickle=False) as stored:
        measurement = np.asarray(stored["measurement_m"])
        visibility = np.asarray(stored["measurement_visibility"], dtype=bool)
        validity = np.asarray(stored["measurement_validity"], dtype=bool)
    errors = []
    per_update = []
    for update in (19, 38, 57):
        supported = (
            visibility[update, center_ids]
            & validity[update, center_ids]
            & np.all(np.isfinite(measurement[update, center_ids]), axis=1)
        )
        values = np.linalg.norm(
            measurement[update, center_ids[supported]]
            - target[update, center_ids[supported]],
            axis=1,
        )
        errors.extend(values.tolist())
        per_update.append(
            {
                "frame": update,
                "count": int(len(values)),
                "mean_error_m": None if not len(values) else float(np.mean(values)),
            }
        )
    values = np.asarray(errors, dtype=float)
    return {
        "count": int(len(values)),
        "mean_error_m": None if not len(values) else float(np.mean(values)),
        "median_error_m": None if not len(values) else float(np.median(values)),
        "p90_error_m": (
            None if not len(values) else float(np.quantile(values, 0.90))
        ),
        "maximum_error_m": None if not len(values) else float(np.max(values)),
        "per_update": per_update,
        "role": "observed-center audit only; never scored as a full-field arm",
    }


def evaluate_selective_virtual_sensing_case(
    protocol_path: str | Path,
    cohort_seal: Mapping[str, Any],
    prediction_root: str | Path,
    failure_root: str | Path,
    measurement_root: str | Path,
    outcome_root: str | Path,
    *,
    object_id: str,
    episode_id: int,
) -> dict[str, Any]:
    """Score one authorized case after the complete prediction cohort was sealed."""

    protocol = load_selective_virtual_sensing_protocol(protocol_path)
    record, prediction_seal = authorize_selective_target_case(
        cohort_seal,
        protocol_path=protocol_path,
        prediction_root=prediction_root,
        failure_root=failure_root,
        object_id=object_id,
        episode_id=episode_id,
    )
    case_name = str(record["case"])
    prediction_dir = Path(prediction_root).resolve() / case_name
    prediction_archive = prediction_dir / VIRTUAL_SENSING_ARCHIVE_FILENAME
    prediction_report_path = prediction_dir / VIRTUAL_SENSING_REPORT_FILENAME
    prediction_report = json.loads(
        prediction_report_path.read_text(encoding="utf-8")
    )
    outcome_dir = Path(outcome_root).resolve() / case_name
    outcome_path = outcome_dir / OUTCOME_MANIFEST_FILENAME
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    target_archive = _validate_outcome_manifest(
        outcome,
        protocol_config_sha256=str(protocol["config_sha256"]),
        record=record,
        cohort_result_sha256=str(cohort_seal["result_sha256"]),
        prediction_result_sha256=str(prediction_seal["result_sha256"]),
        outcome_dir=outcome_dir,
    )
    with np.load(prediction_archive, allow_pickle=False) as stored:
        trajectories = {
            arm: np.asarray(stored[key]).copy()
            for arm, key in ARM_TO_ARCHIVE_KEY.items()
        }
        center_ids = np.asarray(stored["center_ids"], dtype=np.int64)
        selected_cameras = np.asarray(stored["selected_cameras"]).astype(str)
    with np.load(target_archive, allow_pickle=False) as stored:
        target = np.asarray(stored["target_m"]).copy()
        visibility = np.asarray(stored["target_visibility"], dtype=bool)
        validity = np.asarray(stored["target_validity"], dtype=bool)
    _require(
        np.array_equal(trajectories["persistence"][0], target[0]),
        "evaluation frame-zero material identities differ",
    )
    _require(
        selected_cameras.shape == (8,), "evaluation camera panel changed"
    )
    scores = score_selective_virtual_sensing_arrays(
        trajectories,
        target,
        visibility,
        validity,
        center_ids=center_ids,
    )
    measurement_dir = Path(measurement_root).resolve() / case_name
    audit = _measurement_target_audit(
        measurement_dir, prediction_report, target, center_ids
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360SelectiveVirtualSensingEvaluation",
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": protocol["config_sha256"],
        **record,
        "scored_frames": list(SCORED_FRAMES),
        "center_ids": center_ids.tolist(),
        "hidden_material_identity_count": int(target.shape[1] - len(center_ids)),
        "selected_cameras": selected_cameras.tolist(),
        "scores": scores,
        "raw_measurement_target_open_audit": audit,
        "inputs_sha256": {
            "prediction_archive": _sha256(prediction_archive),
            "prediction_report": _sha256(prediction_report_path),
            "outcome_manifest": _sha256(outcome_path),
            "target_archive": _sha256(target_archive),
            "measurement_manifest": _sha256(measurement_dir / MANIFEST_FILENAME),
            "measurement_archive": _sha256(measurement_dir / MEASUREMENT_FILENAME),
        },
        "authorization": {
            "prediction_cohort_result_sha256": cohort_seal["result_sha256"],
            "prediction_result_sha256": prediction_seal["result_sha256"],
            "outcome_result_sha256": outcome["result_sha256"],
        },
        "claim_boundary": (
            "prospective public-data confirmation against persistence; not "
            "official Deform360 Table-4 parity or a direct state-of-the-art claim"
        ),
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    return payload


def _exact_one_sided_sign_paired_p(differences: np.ndarray) -> dict[str, object]:
    values = np.asarray(differences, dtype=float)
    _require(values.ndim == 1 and len(values) > 0, "sign test needs object differences")
    non_ties = values[values != 0.0]
    wins = int(np.sum(non_ties < 0.0))
    count = int(len(non_ties))
    p_value = (
        1.0
        if count == 0
        else sum(math.comb(count, index) for index in range(wins, count + 1))
        / (2**count)
    )
    return {
        "object_count_excluding_exact_ties": count,
        "improved_object_count": wins,
        "exact_tie_count": int(len(values) - count),
        "one_sided_exact_p": float(p_value),
    }


def _object_cluster_bootstrap(differences: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(differences, dtype=float)
    _require(values.ndim == 1 and len(values) > 1, "bootstrap needs multiple objects")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    samples = np.mean(values[indices], axis=1)
    return {
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "lower_95_difference_m": float(np.quantile(samples, 0.025)),
        "upper_95_difference_m": float(np.quantile(samples, 0.975)),
        "probability_improved": float(np.mean(samples < 0.0)),
    }


def aggregate_selective_case_reports(
    reports: Sequence[Mapping[str, Any]],
    *,
    protocol_path: str | Path,
) -> dict[str, Any]:
    """Apply every preregistered object-level success gate to case reports."""

    protocol = load_selective_virtual_sensing_protocol(protocol_path)
    _require(reports, "no successful prospective evaluation exists")
    by_object: dict[str, list[Mapping[str, Any]]] = {}
    object_stratum: dict[str, str] = {}
    seen_cases: set[str] = set()
    for report in reports:
        _require(
            report.get("artifact_kind")
            == "Deform360SelectiveVirtualSensingEvaluation"
            and report.get("protocol_id") == PROTOCOL_ID,
            "unsupported case evaluation",
        )
        case_name = str(report["case"])
        _require(case_name not in seen_cases, "case evaluation repeated")
        seen_cases.add(case_name)
        object_id = str(report["object_id"])
        stratum = str(report["stratum"])
        by_object.setdefault(object_id, []).append(report)
        _require(
            object_id not in object_stratum or object_stratum[object_id] == stratum,
            "object appears in multiple strata",
        )
        object_stratum[object_id] = stratum

    cohort = protocol["config"]["cohort"]
    per_stratum = {
        stratum: sum(value == stratum for value in object_stratum.values())
        for stratum in EXPECTED_STRATA
    }
    _require(
        len(by_object) >= int(cohort["minimum_evaluable_objects"])
        and all(
            count >= int(cohort["minimum_evaluable_objects_per_stratum"])
            for count in per_stratum.values()
        ),
        "case reports do not meet the locked evaluability threshold",
    )

    arms = tuple(ARM_TO_ARCHIVE_KEY)
    object_rows = []
    for object_id, object_reports in by_object.items():
        means = {
            arm: {
                metric: float(
                    np.mean(
                        [report["scores"][arm][metric] for report in object_reports]
                    )
                )
                for metric in PRIMARY_METRICS
            }
            for arm in arms
        }
        object_rows.append(
            {
                "object_id": object_id,
                "stratum": object_stratum[object_id],
                "episode_count": len(object_reports),
                "episodes": [int(report["episode_id"]) for report in object_reports],
                "scores": means,
            }
        )
    object_rows.sort(key=lambda row: (EXPECTED_STRATA.index(row["stratum"]), row["object_id"]))
    aggregate_scores = {
        arm: {
            metric: float(np.mean([row["scores"][arm][metric] for row in object_rows]))
            for metric in PRIMARY_METRICS
        }
        for arm in arms
    }

    evaluation_config = protocol["config"]["evaluation"]
    comparisons: dict[str, Any] = {}
    metric_passes = []
    for metric in PRIMARY_METRICS:
        primary = np.asarray(
            [row["scores"][PERSISTENCE_CLIQUE_RBF_ARM][metric] for row in object_rows]
        )
        baseline = np.asarray(
            [row["scores"]["persistence"][metric] for row in object_rows]
        )
        _require(
            np.all(baseline > 0.0),
            "persistence metric must be positive for relative success gates",
        )
        differences = primary - baseline
        relative_by_object = differences / baseline
        relative_change = (
            aggregate_scores[PERSISTENCE_CLIQUE_RBF_ARM][metric]
            / aggregate_scores["persistence"][metric]
            - 1.0
        )
        bootstrap = _object_cluster_bootstrap(differences)
        sign_test = _exact_one_sided_sign_paired_p(differences)
        stratum_differences = {
            stratum: float(
                np.mean(
                    [
                        differences[index]
                        for index, row in enumerate(object_rows)
                        if row["stratum"] == stratum
                    ]
                )
            )
            for stratum in EXPECTED_STRATA
        }
        gates = {
            "minimum_ten_percent_relative_improvement": bool(
                relative_change
                <= -float(
                    evaluation_config[
                        "minimum_relative_improvement_each_primary_metric"
                    ]
                )
            ),
            "object_cluster_upper_95_below_zero": bool(
                bootstrap["upper_95_difference_m"]
                < float(
                    evaluation_config[
                        "object_cluster_upper_95_difference_m_must_be_below"
                    ]
                )
            ),
            "one_sided_exact_sign_p_at_most_0_05": bool(
                sign_test["one_sided_exact_p"]
                <= float(evaluation_config["one_sided_exact_sign_p_must_be_at_most"])
            ),
            "no_stratum_mean_regression": bool(
                all(value <= 0.0 for value in stratum_differences.values())
            ),
            "no_object_regression_over_ten_percent": bool(
                np.max(relative_by_object)
                <= float(
                    evaluation_config[
                        "maximum_object_regression_fraction_each_primary_metric"
                    ]
                )
            ),
        }
        passed = all(gates.values())
        metric_passes.append(passed)
        comparisons[metric] = {
            "primary_object_balanced_mean_m": aggregate_scores[
                PERSISTENCE_CLIQUE_RBF_ARM
            ][metric],
            "persistence_object_balanced_mean_m": aggregate_scores["persistence"][
                metric
            ],
            "relative_change": float(relative_change),
            "object_mean_difference_m": differences.tolist(),
            "object_relative_change": relative_by_object.tolist(),
            "maximum_object_relative_regression": float(
                np.max(relative_by_object)
            ),
            "stratum_mean_difference_m": stratum_differences,
            "object_cluster_bootstrap": bootstrap,
            "exact_sign_test": sign_test,
            "gates": gates,
            "passed": passed,
        }

    threshold_passed = all(metric_passes)
    secondary_comparisons = {}
    for comparator in (UNGATED_RBF_ARM, CPD_ARM):
        secondary_comparisons[f"{PERSISTENCE_CLIQUE_RBF_ARM}_vs_{comparator}"] = {
            metric: {
                "primary_object_balanced_mean_m": aggregate_scores[
                    PERSISTENCE_CLIQUE_RBF_ARM
                ][metric],
                "comparator_object_balanced_mean_m": aggregate_scores[comparator][
                    metric
                ],
                "relative_change": float(
                    aggregate_scores[PERSISTENCE_CLIQUE_RBF_ARM][metric]
                    / aggregate_scores[comparator][metric]
                    - 1.0
                )
                if aggregate_scores[comparator][metric] > 0.0
                else None,
                "improved_object_count": int(
                    sum(
                        row["scores"][PERSISTENCE_CLIQUE_RBF_ARM][metric]
                        < row["scores"][comparator][metric]
                        for row in object_rows
                    )
                ),
            }
            for metric in PRIMARY_METRICS
        }
    return {
        "object_count": len(object_rows),
        "episode_count": len(reports),
        "object_count_by_stratum": per_stratum,
        "object_results": object_rows,
        "object_balanced_scores": aggregate_scores,
        "primary_vs_persistence": comparisons,
        "secondary_comparisons": secondary_comparisons,
        "unavailable_secondary_comparators": {
            "selected_raw_physical_backbone": (
                "not available in the persistence-only prospective primary arm"
            )
        },
        "all_success_gates_conjunctive": True,
        "paper_threshold_passed": threshold_passed,
        "permitted_claim": (
            "The frozen sparse-camera virtual sensor prospectively improves "
            "hidden future full-field state over persistence across new "
            "deformable objects."
            if threshold_passed
            else (
                "No general prospective improvement claim; report the mixed or "
                "negative result without selecting a replacement method."
            )
        ),
    }


def aggregate_selective_virtual_sensing_evaluations(
    protocol_path: str | Path,
    cohort_seal: Mapping[str, Any],
    evaluation_root: str | Path,
    prediction_root: str | Path,
    failure_root: str | Path,
) -> dict[str, Any]:
    """Load every authorized case report and emit the final prospective result."""

    protocol = load_selective_virtual_sensing_protocol(protocol_path)
    validate_selective_prediction_cohort_seal(
        cohort_seal,
        protocol_path=protocol_path,
        require_eligible=True,
        prediction_root=prediction_root,
        failure_root=failure_root,
    )
    root = Path(evaluation_root).resolve()
    reports = []
    failed_cases = []
    for row in cohort_seal["cases"]:
        if row["status"] == "quality-failure":
            failure_path = (
                Path(failure_root).resolve()
                / str(row["case"])
                / "quality_failure.json"
            )
            failure = json.loads(failure_path.read_text(encoding="utf-8"))
            failed_cases.append(
                {
                    **dict(row),
                    "error_type": failure["error_type"],
                    "error_message": failure["error_message"],
                    "disposition": failure["disposition"],
                }
            )
            continue
        path = root / str(row["case"]) / CASE_EVALUATION_FILENAME
        report = json.loads(path.read_text(encoding="utf-8"))
        _require(
            report.get("result_sha256") == _canonical_sha256(report),
            f"case evaluation checksum changed: {row['case']}",
        )
        _require(
            all(report.get(key) == value for key, value in row.items() if key in report),
            f"case evaluation identity changed: {row['case']}",
        )
        _require(
            report.get("authorization", {}).get(
                "prediction_cohort_result_sha256"
            )
            == cohort_seal["result_sha256"],
            f"case evaluation used another cohort seal: {row['case']}",
        )
        _require(
            report.get("authorization", {}).get("prediction_result_sha256")
            == row["artifact_result_sha256"],
            f"case evaluation used another prediction: {row['case']}",
        )
        reports.append(report)
    aggregate = aggregate_selective_case_reports(
        reports, protocol_path=protocol_path
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360SelectiveVirtualSensingProspectiveResult",
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": protocol["config_sha256"],
        "prediction_cohort_result_sha256": cohort_seal["result_sha256"],
        "quality_failure_count": len(failed_cases),
        "quality_failures": failed_cases,
        **aggregate,
        "claim_boundary": (
            "prospective public-data confirmation against persistence; not "
            "official Deform360 Table-4 parity or a direct state-of-the-art claim"
        ),
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    return payload


__all__ = [
    "ARM_TO_ARCHIVE_KEY",
    "CASE_EVALUATION_FILENAME",
    "OUTCOME_MANIFEST_FILENAME",
    "PRIMARY_METRICS",
    "SCORED_FRAMES",
    "TARGET_ARCHIVE_FILENAME",
    "aggregate_selective_case_reports",
    "aggregate_selective_virtual_sensing_evaluations",
    "evaluate_selective_virtual_sensing_case",
    "score_selective_virtual_sensing_arrays",
]
