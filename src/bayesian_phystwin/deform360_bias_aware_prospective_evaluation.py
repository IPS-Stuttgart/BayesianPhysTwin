"""Outcome-gated evaluation for the fresh Deform360 bias-aware protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bias_aware_belief import fit_source_group_regret_bound
from .deform360_bias_aware_prospective_artifacts import (
    PREDICTION_ARCHIVE_FILENAME,
    PREDICTION_REPORT_FILENAME,
    PREDICTION_SEAL_FILENAME,
    QUALITY_FAILURE_FILENAME,
    authorize_prospective_outcome_case,
    canonical_sha256,
    file_sha256,
    prospective_case_record,
    prospective_case_records,
    validate_prospective_prediction_cohort_seal,
)
from .deform360_bias_aware_prospective_protocol import (
    EXPECTED_STRATA,
    EXPECTED_UPDATE_FRAMES,
    PROTOCOL_ID,
    SOURCE_LOCK_GROUP_COUNT,
    SOURCE_LOCK_SHA256,
    SOURCE_MINIMUM_IMPROVEMENT_M,
    load_bias_aware_prospective_protocol,
)
from .deform360_online_belief_evaluation import (
    score_deform360_hidden_trajectory,
)


AUTHORIZED_FUTURE_MANIFEST_FILENAME = "authorized_future_manifest.json"
TARGET_ARCHIVE_FILENAME = "target_trajectory.npz"
OUTCOME_MANIFEST_FILENAME = "authorized_outcome_manifest.json"
OUTCOME_FAILURE_FILENAME = "outcome_failure.json"
CASE_EVALUATION_FILENAME = "evaluation.json"
CALIBRATION_GATE_FILENAME = "calibration_gate.json"
TARGET_RESULT_FILENAME = "target_result.json"

AUTHORIZED_FUTURE_ARTIFACT_KIND = "Deform360BiasAwareProspectiveAuthorizedFuture"
AUTHORIZED_OUTCOME_ARTIFACT_KIND = "Deform360BiasAwareProspectiveAuthorizedOutcome"
OUTCOME_FAILURE_ARTIFACT_KIND = "Deform360BiasAwareProspectiveOutcomeFailure"
CASE_EVALUATION_ARTIFACT_KIND = "Deform360BiasAwareProspectiveEvaluation"
CALIBRATION_GATE_ARTIFACT_KIND = "Deform360BiasAwareProspectiveCalibrationGate"
TARGET_RESULT_ARTIFACT_KIND = "Deform360BiasAwareProspectiveTargetResult"

PRIMARY_METRICS = (
    "post_update_hidden_identity_rmse_m",
    "post_update_hidden_symmetric_chamfer_m",
)
SCORED_FRAMES = tuple([*range(20, 38), *range(39, 57), *range(58, 76)])
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 0
FROZEN_SOURCE_GROUP_WORST_REGRET_M = {
    "002-rope-silk": -0.001998798742991026,
    "083-blanket-cloth": -1.887323425379516e-05,
    "085-scarf-cloth": -8.871136285708119e-06,
    "170-spider": -1.527995763579877e-05,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _object_cluster_bootstrap(differences: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(differences, dtype=np.float64)
    _require(values.ndim == 1 and len(values) > 1, "bootstrap needs objects")
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


def score_bias_aware_prospective_arrays(
    prediction_m: np.ndarray,
    baseline_m: np.ndarray,
    target_m: np.ndarray,
    target_visibility: np.ndarray,
    target_validity: np.ndarray,
    *,
    center_ids: np.ndarray,
    update_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score sealed candidate and baseline, including each guarded interval."""

    prediction = np.asarray(prediction_m)
    baseline = np.asarray(baseline_m)
    target = np.asarray(target_m)
    visibility = np.asarray(target_visibility, dtype=bool)
    validity = np.asarray(target_validity, dtype=bool)
    centers = np.asarray(center_ids, dtype=np.int64)
    _require(
        prediction.shape == baseline.shape == target.shape
        and target.ndim == 3
        and target.shape[0] == 76
        and target.shape[2] == 3,
        "prediction, baseline, and target shapes changed",
    )
    _require(
        visibility.shape == validity.shape == target.shape[:2],
        "target support shape changed",
    )
    _require(
        centers.shape == (16,) and len(np.unique(centers)) == 16,
        "center inventory changed",
    )
    _require(
        np.array_equal(prediction[0], baseline[0])
        and np.array_equal(baseline[0], target[0]),
        "frame-zero material identity changed",
    )
    _require(len(update_records) == 3, "update inventory changed")
    expected_stops = (38, 57, 76)
    intervals: list[dict[str, Any]] = []
    for record, update, stop in zip(
        update_records, EXPECTED_UPDATE_FRAMES, expected_stops, strict=True
    ):
        _require(
            int(record.get("frame", -1)) == update
            and int(record.get("interval_end_exclusive", -1)) == stop
            and isinstance(record.get("candidate_available"), bool),
            "candidate interval contract changed",
        )
        frames = tuple(range(update + 1, stop))
        baseline_score = score_deform360_hidden_trajectory(
            baseline,
            target,
            visibility,
            validity,
            center_ids=centers,
            scored_frames=frames,
        )
        candidate_score = score_deform360_hidden_trajectory(
            prediction,
            target,
            visibility,
            validity,
            center_ids=centers,
            scored_frames=frames,
        )
        regrets = {
            metric: float(candidate_score[metric] - baseline_score[metric])
            for metric in PRIMARY_METRICS
        }
        fallback_exact = bool(
            np.array_equal(prediction[update + 1 : stop], baseline[update + 1 : stop])
        )
        if not bool(record["candidate_available"]):
            _require(fallback_exact, "rejected update is not exact fallback")
        intervals.append(
            {
                "frame": update,
                "interval_end_exclusive": stop,
                "candidate_available": bool(record["candidate_available"]),
                "exact_baseline_fallback": fallback_exact,
                "scores": {
                    "prediction": candidate_score,
                    "selected_raw_baseline": baseline_score,
                },
                "regret_m": regrets,
                "worst_primary_regret_m": float(max(regrets.values())),
            }
        )
    scores = {
        "prediction": score_deform360_hidden_trajectory(
            prediction,
            target,
            visibility,
            validity,
            center_ids=centers,
            scored_frames=SCORED_FRAMES,
        ),
        "selected_raw_baseline": score_deform360_hidden_trajectory(
            baseline,
            target,
            visibility,
            validity,
            center_ids=centers,
            scored_frames=SCORED_FRAMES,
        ),
    }
    return {
        "scores": scores,
        "intervals": intervals,
        "candidate_update_count": int(
            sum(row["candidate_available"] for row in intervals)
        ),
        "all_rejections_bit_exact_fallback": bool(
            all(
                row["candidate_available"] or row["exact_baseline_fallback"]
                for row in intervals
            )
        ),
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
        outcome.get("artifact_kind") == AUTHORIZED_OUTCOME_ARTIFACT_KIND
        and outcome.get("protocol_id") == PROTOCOL_ID
        and outcome.get("protocol_config_sha256") == protocol_config_sha256
        and outcome.get("result_sha256")
        == canonical_sha256(outcome, digest_key="result_sha256"),
        "authorized outcome manifest changed",
    )
    _require(
        all(outcome.get(key) == value for key, value in record.items()),
        "authorized outcome case identity changed",
    )
    authorization = outcome.get("authorization", {})
    _require(
        authorization.get("prediction_cohort_result_sha256") == cohort_result_sha256
        and authorization.get("prediction_result_sha256") == prediction_result_sha256,
        "authorized outcome belongs to another prediction",
    )
    boundary = outcome.get("information_boundary", {})
    _require(
        boundary.get("prediction_cohort_verified_before_target_construction") is True
        and boundary.get("future_tactile_read") is False
        and boundary.get("prediction_metric_computed") is False,
        "outcome construction crossed its metric boundary",
    )
    archive = outcome_dir / TARGET_ARCHIVE_FILENAME
    _require(
        archive.is_file()
        and outcome.get("output", {}).get("target_archive_sha256")
        == file_sha256(archive),
        "authorized target archive changed",
    )
    return archive


def evaluate_bias_aware_prospective_case(
    protocol_path: str | Path,
    cohort_seal: Mapping[str, Any],
    prediction_root: str | Path,
    outcome_root: str | Path,
    *,
    role: str,
    object_id: str,
    episode_id: int,
    calibration_gate_result_sha256: str | None = None,
) -> dict[str, Any]:
    """Score one case only after its full role cohort was prediction-sealed."""

    protocol = load_bias_aware_prospective_protocol(protocol_path)
    record, prediction_seal = authorize_prospective_outcome_case(
        cohort_seal,
        protocol_path=protocol_path,
        role=role,
        artifact_root=prediction_root,
        object_id=object_id,
        episode_id=episode_id,
    )
    case_name = str(record["case"])
    prediction_dir = Path(prediction_root).resolve() / case_name
    prediction_path = prediction_dir / PREDICTION_ARCHIVE_FILENAME
    report_path = prediction_dir / PREDICTION_REPORT_FILENAME
    prediction_report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(
        file_sha256(prediction_path)
        == prediction_seal["prediction_archive"]["file_sha256"]
        and file_sha256(report_path)
        == prediction_seal["prediction_report"]["file_sha256"],
        "sealed prediction inputs changed",
    )
    outcome_dir = Path(outcome_root).resolve() / case_name
    outcome_path = outcome_dir / OUTCOME_MANIFEST_FILENAME
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    if role == "target":
        _require(
            calibration_gate_result_sha256 is not None
            and outcome.get("authorization", {}).get("calibration_gate_result_sha256")
            == calibration_gate_result_sha256,
            "target outcome lacks the passed calibration gate",
        )
    else:
        _require(
            calibration_gate_result_sha256 is None
            and outcome.get("authorization", {}).get("calibration_gate_result_sha256")
            is None,
            "calibration outcome consumed a target gate",
        )
    target_path = _validate_outcome_manifest(
        outcome,
        protocol_config_sha256=str(protocol["config_sha256"]),
        record=record,
        cohort_result_sha256=str(cohort_seal["result_sha256"]),
        prediction_result_sha256=str(prediction_seal["result_sha256"]),
        outcome_dir=outcome_dir,
    )
    with np.load(prediction_path, allow_pickle=False) as stored:
        prediction = np.asarray(stored["prediction_m"]).copy()
        baseline = np.asarray(stored["selected_raw_backbone"]).copy()
        centers = np.asarray(stored["center_ids"], dtype=np.int64)
    with np.load(target_path, allow_pickle=False) as stored:
        target = np.asarray(stored["target_m"]).copy()
        visibility = np.asarray(stored["target_visibility"], dtype=bool)
        validity = np.asarray(stored["target_validity"], dtype=bool)
    scored = score_bias_aware_prospective_arrays(
        prediction,
        baseline,
        target,
        visibility,
        validity,
        center_ids=centers,
        update_records=prediction_report["method"]["bias_aware_candidate"]["updates"],
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": CASE_EVALUATION_ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": protocol["config_sha256"],
        **record,
        "scored_frames": list(SCORED_FRAMES),
        "center_ids": centers.tolist(),
        **scored,
        "input_sha256": {
            "prediction_seal": file_sha256(prediction_dir / PREDICTION_SEAL_FILENAME),
            "prediction_archive": file_sha256(prediction_path),
            "prediction_report": file_sha256(report_path),
            "outcome_manifest": file_sha256(outcome_path),
            "target_archive": file_sha256(target_path),
        },
        "authorization": {
            "prediction_cohort_result_sha256": cohort_seal["result_sha256"],
            "prediction_result_sha256": prediction_seal["result_sha256"],
            "outcome_result_sha256": outcome["result_sha256"],
            "calibration_gate_result_sha256": calibration_gate_result_sha256,
        },
        "claim_boundary": (
            "prospective comparison with the exact selected raw/physical "
            "backbone; not official Deform360 Table-4 parity"
        ),
    }
    payload["result_sha256"] = canonical_sha256(payload, digest_key="result_sha256")
    return payload


def _validate_case_report(
    report: Mapping[str, Any],
    *,
    protocol_config_sha256: str,
    role: str,
) -> None:
    _require(
        report.get("artifact_kind") == CASE_EVALUATION_ARTIFACT_KIND
        and report.get("protocol_id") == PROTOCOL_ID
        and report.get("protocol_config_sha256") == protocol_config_sha256
        and report.get("role") == role
        and report.get("result_sha256")
        == canonical_sha256(report, digest_key="result_sha256"),
        "case evaluation changed",
    )


def _object_rows(reports: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_object: dict[str, list[Mapping[str, Any]]] = {}
    strata: dict[str, str] = {}
    for report in reports:
        object_id = str(report["object_id"])
        stratum = str(report["stratum"])
        by_object.setdefault(object_id, []).append(report)
        _require(
            object_id not in strata or strata[object_id] == stratum,
            "object appears in multiple strata",
        )
        strata[object_id] = stratum
    rows: list[dict[str, Any]] = []
    for object_id, members in by_object.items():
        scores = {
            arm: {
                metric: float(
                    np.mean([member["scores"][arm][metric] for member in members])
                )
                for metric in PRIMARY_METRICS
            }
            for arm in ("prediction", "selected_raw_baseline")
        }
        eligible_intervals = [
            interval
            for member in members
            for interval in member["intervals"]
            if interval["candidate_available"]
        ]
        rows.append(
            {
                "object_id": object_id,
                "stratum": strata[object_id],
                "episode_count": len(members),
                "episodes": sorted(int(member["episode_id"]) for member in members),
                "cases": sorted(str(member["case"]) for member in members),
                "evaluation_result_sha256": {
                    str(member["case"]): str(member["result_sha256"])
                    for member in members
                },
                "scores": scores,
                "regret_m": {
                    metric: float(
                        scores["prediction"][metric]
                        - scores["selected_raw_baseline"][metric]
                    )
                    for metric in PRIMARY_METRICS
                },
                "eligible_update_count": len(eligible_intervals),
                "worst_eligible_interval_regret_m": (
                    None
                    if not eligible_intervals
                    else float(
                        max(
                            interval["worst_primary_regret_m"]
                            for interval in eligible_intervals
                        )
                    )
                ),
                "all_rejections_bit_exact_fallback": bool(
                    all(
                        member["all_rejections_bit_exact_fallback"]
                        for member in members
                    )
                ),
            }
        )
    rows.sort(key=lambda row: (EXPECTED_STRATA.index(row["stratum"]), row["object_id"]))
    return rows


def fit_bias_aware_calibration_gate(
    reports: Sequence[Mapping[str, Any]],
    *,
    protocol_path: str | Path,
    source_lock: Mapping[str, Any],
    calibration_cohort_result_sha256: str,
    quality_failures: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Apply the locked calibration arithmetic without changing the method."""

    protocol = load_bias_aware_prospective_protocol(protocol_path)
    for report in reports:
        _validate_case_report(
            report,
            protocol_config_sha256=str(protocol["config_sha256"]),
            role="calibration",
        )
        _require(
            report.get("authorization", {}).get("prediction_cohort_result_sha256")
            == calibration_cohort_result_sha256,
            "calibration report used another cohort seal",
        )
    expected = prospective_case_records(protocol_path, role="calibration")
    seen = {str(report["case"]) for report in reports}
    failure_cases = {str(row["case"]) for row in quality_failures}
    _require(
        len(seen) == len(reports)
        and len(failure_cases) == len(quality_failures)
        and not seen & failure_cases
        and seen | failure_cases == {str(row["case"]) for row in expected},
        "calibration dispositions are incomplete or duplicated",
    )
    _require(
        source_lock.get("source_group_count") == SOURCE_LOCK_GROUP_COUNT
        and source_lock.get("candidate_certified") is True
        and source_lock.get("fresh_accuracy_evaluation_allowed") is True,
        "source lock is incompatible",
    )
    source_scores = source_lock.get("source_group_worst_regret_m")
    _require(
        isinstance(source_scores, Mapping)
        and {str(key): float(value) for key, value in source_scores.items()}
        == FROZEN_SOURCE_GROUP_WORST_REGRET_M,
        "source group scores changed",
    )
    rows = _object_rows(reports)
    eligible = [row for row in rows if row["eligible_update_count"] > 0]
    combined_scores = {
        **{str(key): float(value) for key, value in source_scores.items()},
        **{
            str(row["object_id"]): float(row["worst_eligible_interval_regret_m"])
            for row in eligible
        },
    }
    _require(
        len(combined_scores) == len(source_scores) + len(eligible), "group overlap"
    )
    bound = fit_source_group_regret_bound(
        np.asarray(list(combined_scores.values()), dtype=np.float64),
        list(combined_scores),
        nominal_coverage=0.90,
        within_group_coverage=1.0,
        minimum_improvement_m=SOURCE_MINIMUM_IMPROVEMENT_M,
    )
    gate = protocol["config"]["calibration_gate"]
    stratum_counts = {
        stratum: sum(row["stratum"] == stratum for row in rows)
        for stratum in EXPECTED_STRATA
    }
    mean_regret = {
        metric: float(np.mean([row["regret_m"][metric] for row in rows]))
        for metric in PRIMARY_METRICS
    }
    harmful = [
        row["object_id"]
        for row in eligible
        if any(row["regret_m"][metric] > 0.0 for metric in PRIMARY_METRICS)
    ]
    gates = {
        "minimum_evaluable_objects": len(rows)
        >= int(gate["minimum_evaluable_objects"]),
        "minimum_evaluable_objects_per_stratum": all(
            count >= int(gate["minimum_evaluable_objects_per_stratum"])
            for count in stratum_counts.values()
        ),
        "minimum_new_eligible_object_groups": len(eligible)
        >= int(gate["minimum_new_eligible_object_groups"]),
        "minimum_combined_eligible_object_groups": len(combined_scores)
        >= int(gate["minimum_combined_eligible_object_groups"]),
        "required_finite_sample_coverage": bound.finite_sample_coverage
        >= float(gate["required_finite_sample_coverage"]),
        "required_upper_regret": bound.upper_regret_m
        < float(gate["required_upper_regret_m"]),
        "co_primary_object_balanced_mean_regret_negative": all(
            value < 0.0 for value in mean_regret.values()
        ),
        "accepted_harmful_object_count": len(harmful)
        <= int(gate["accepted_harmful_object_count_allowed"]),
        "every_rejection_bit_exact_fallback": all(
            row["all_rejections_bit_exact_fallback"] for row in rows
        ),
        "no_replacement": all(
            row.get("replacement_allowed") is False for row in quality_failures
        ),
    }
    passed = all(gates.values())
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": CALIBRATION_GATE_ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": protocol["config_sha256"],
        "calibration_prediction_cohort_result_sha256": (
            calibration_cohort_result_sha256
        ),
        "source_lock_sha256": SOURCE_LOCK_SHA256,
        "source_group_count": len(source_scores),
        "new_eligible_object_group_count": len(eligible),
        "combined_eligible_object_group_count": len(combined_scores),
        "combined_group_worst_regret_m": combined_scores,
        "finite_sample_rank": bound.finite_sample_rank,
        "finite_sample_coverage": bound.finite_sample_coverage,
        "upper_regret_m": bound.upper_regret_m,
        "minimum_improvement_m": bound.minimum_improvement_m,
        "evaluable_object_count": len(rows),
        "evaluable_object_count_by_stratum": stratum_counts,
        "quality_failure_count": len(quality_failures),
        "quality_failures": [dict(row) for row in quality_failures],
        "calibration_evaluation_result_sha256": {
            str(report["case"]): str(report["result_sha256"]) for report in reports
        },
        "object_balanced_regret_m": mean_regret,
        "accepted_harmful_objects": harmful,
        "object_results": rows,
        "gates": gates,
        "calibration_gate_passed": passed,
        "target_access_authorized": passed,
        "failed_gate_action": (
            None
            if passed
            else "publish calibration failure and keep every target future sealed"
        ),
        "information_boundary": {
            "method_family_changed": False,
            "candidate_threshold_changed": False,
            "observation_model_changed": False,
            "calibration_futures_opened_after_prediction_cohort_seal": True,
            "target_object_media_read": False,
            "target_future_read": False,
        },
        "claim_boundary": (
            "calibration gate for a fresh accuracy and non-regression test; not "
            "official Deform360 Table-4 parity or universal safety"
        ),
    }
    payload["result_sha256"] = canonical_sha256(payload, digest_key="result_sha256")
    return payload


def validate_bias_aware_calibration_gate(
    gate: Mapping[str, Any],
    *,
    protocol_path: str | Path,
    require_passed: bool,
) -> None:
    """Validate the one artifact that may authorize target-object access."""

    protocol = load_bias_aware_prospective_protocol(protocol_path)
    _require(
        gate.get("artifact_kind") == CALIBRATION_GATE_ARTIFACT_KIND
        and gate.get("protocol_id") == PROTOCOL_ID
        and gate.get("protocol_config_sha256") == protocol["config_sha256"]
        and gate.get("source_lock_sha256") == SOURCE_LOCK_SHA256
        and gate.get("result_sha256")
        == canonical_sha256(gate, digest_key="result_sha256"),
        "calibration gate changed",
    )
    config = protocol["config"]["calibration_gate"]
    scores = gate.get("combined_group_worst_regret_m")
    rows = gate.get("object_results")
    failures = gate.get("quality_failures")
    _require(
        isinstance(scores, Mapping)
        and isinstance(rows, Sequence)
        and isinstance(failures, Sequence)
        and len(scores) >= SOURCE_LOCK_GROUP_COUNT
        and len(rows) > 0,
        "calibration gate evidence is incomplete",
    )
    group_values = np.asarray(list(scores.values()), dtype=np.float64)
    _require(
        len(scores) == int(gate.get("combined_eligible_object_group_count", -1))
        and int(gate.get("source_group_count", -1)) == SOURCE_LOCK_GROUP_COUNT
        and int(gate.get("new_eligible_object_group_count", -1))
        == len(scores) - SOURCE_LOCK_GROUP_COUNT
        and np.all(np.isfinite(group_values)),
        "calibration group inventory changed",
    )
    _require(
        all(
            scores.get(group) == value
            for group, value in FROZEN_SOURCE_GROUP_WORST_REGRET_M.items()
        ),
        "frozen source group scores changed",
    )
    rank = min(len(group_values), int(np.ceil((len(group_values) + 1) * 0.90)))
    upper = float(np.partition(group_values, rank - 1)[rank - 1])
    _require(
        int(gate.get("finite_sample_rank", -1)) == rank
        and float(gate.get("finite_sample_coverage", -1.0))
        == rank / (len(group_values) + 1)
        and float(gate.get("upper_regret_m", np.inf)) == upper
        and float(gate.get("minimum_improvement_m", -1.0))
        == SOURCE_MINIMUM_IMPROVEMENT_M,
        "calibration finite-sample arithmetic changed",
    )
    stratum_counts = {
        stratum: sum(row.get("stratum") == stratum for row in rows)
        for stratum in EXPECTED_STRATA
    }
    mean_regret = {
        metric: float(np.mean([row["regret_m"][metric] for row in rows]))
        for metric in PRIMARY_METRICS
    }
    eligible = [row for row in rows if int(row["eligible_update_count"]) > 0]
    expected_score_keys = set(FROZEN_SOURCE_GROUP_WORST_REGRET_M) | {
        str(row["object_id"]) for row in eligible
    }
    _require(
        set(scores) == expected_score_keys
        and int(gate.get("new_eligible_object_group_count", -1)) == len(eligible)
        and all(
            scores[str(row["object_id"])]
            == row["worst_eligible_interval_regret_m"]
            for row in eligible
        ),
        "fresh calibration group scores changed",
    )
    harmful = [
        row["object_id"]
        for row in eligible
        if any(row["regret_m"][metric] > 0.0 for metric in PRIMARY_METRICS)
    ]
    expected_gates = {
        "minimum_evaluable_objects": len(rows)
        >= int(config["minimum_evaluable_objects"]),
        "minimum_evaluable_objects_per_stratum": all(
            count >= int(config["minimum_evaluable_objects_per_stratum"])
            for count in stratum_counts.values()
        ),
        "minimum_new_eligible_object_groups": len(eligible)
        >= int(config["minimum_new_eligible_object_groups"]),
        "minimum_combined_eligible_object_groups": len(scores)
        >= int(config["minimum_combined_eligible_object_groups"]),
        "required_finite_sample_coverage": rank / (len(scores) + 1)
        >= float(config["required_finite_sample_coverage"]),
        "required_upper_regret": upper < float(config["required_upper_regret_m"]),
        "co_primary_object_balanced_mean_regret_negative": all(
            value < 0.0 for value in mean_regret.values()
        ),
        "accepted_harmful_object_count": len(harmful)
        <= int(config["accepted_harmful_object_count_allowed"]),
        "every_rejection_bit_exact_fallback": all(
            row["all_rejections_bit_exact_fallback"] for row in rows
        ),
        "no_replacement": all(
            row.get("replacement_allowed") is False for row in failures
        ),
    }
    _require(
        gate.get("gates") == expected_gates
        and gate.get("evaluable_object_count") == len(rows)
        and gate.get("evaluable_object_count_by_stratum") == stratum_counts
        and gate.get("quality_failure_count") == len(failures)
        and gate.get("object_balanced_regret_m") == mean_regret
        and gate.get("accepted_harmful_objects") == harmful,
        "calibration gate summaries changed",
    )
    _require(
        gate.get("calibration_evaluation_result_sha256")
        == {
            case: digest
            for row in rows
            for case, digest in row["evaluation_result_sha256"].items()
        }
        and isinstance(
            gate.get("calibration_prediction_cohort_result_sha256"), str
        )
        and bool(gate["calibration_prediction_cohort_result_sha256"]),
        "calibration evaluation provenance changed",
    )
    passed = all(expected_gates.values())
    _require(
        gate.get("calibration_gate_passed") is passed
        and gate.get("target_access_authorized") is passed
        and gate.get("information_boundary")
        == {
            "method_family_changed": False,
            "candidate_threshold_changed": False,
            "observation_model_changed": False,
            "calibration_futures_opened_after_prediction_cohort_seal": True,
            "target_object_media_read": False,
            "target_future_read": False,
        },
        "target authorization disagrees with calibration gate",
    )
    if require_passed:
        _require(passed, "calibration gate forbids target access")


def aggregate_bias_aware_target_result(
    reports: Sequence[Mapping[str, Any]],
    *,
    protocol_path: str | Path,
    target_cohort_result_sha256: str,
    calibration_gate_result_sha256: str,
    quality_failures: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Apply the frozen object-clustered target success gates."""

    protocol = load_bias_aware_prospective_protocol(protocol_path)
    for report in reports:
        _validate_case_report(
            report,
            protocol_config_sha256=str(protocol["config_sha256"]),
            role="target",
        )
        _require(
            report.get("authorization", {}).get("prediction_cohort_result_sha256")
            == target_cohort_result_sha256
            and report.get("authorization", {}).get("calibration_gate_result_sha256")
            == calibration_gate_result_sha256,
            "target report used another authorization",
        )
    expected = prospective_case_records(protocol_path, role="target")
    seen = {str(report["case"]) for report in reports}
    failure_cases = {str(row["case"]) for row in quality_failures}
    _require(
        len(seen) == len(reports)
        and len(failure_cases) == len(quality_failures)
        and not seen & failure_cases
        and seen | failure_cases == {str(row["case"]) for row in expected},
        "target dispositions are incomplete or duplicated",
    )
    rows = _object_rows(reports)
    config = protocol["config"]["target_evaluation"]
    stratum_counts = {
        stratum: sum(row["stratum"] == stratum for row in rows)
        for stratum in EXPECTED_STRATA
    }
    comparisons: dict[str, Any] = {}
    metric_passes = []
    for metric in PRIMARY_METRICS:
        differences = np.asarray(
            [row["regret_m"][metric] for row in rows], dtype=np.float64
        )
        bootstrap = _object_cluster_bootstrap(differences)
        strata = {
            stratum: float(
                np.mean(
                    [
                        row["regret_m"][metric]
                        for row in rows
                        if row["stratum"] == stratum
                    ]
                )
            )
            for stratum in EXPECTED_STRATA
            if any(row["stratum"] == stratum for row in rows)
        }
        gates = {
            "object_balanced_mean_difference_negative": float(np.mean(differences))
            < 0.0,
            "object_cluster_upper_95_bound_negative": bootstrap["upper_95_difference_m"]
            < 0.0,
            "no_stratum_mean_regression": len(strata) == len(EXPECTED_STRATA)
            and all(value <= 0.0 for value in strata.values()),
        }
        metric_passes.append(all(gates.values()))
        comparisons[metric] = {
            "object_balanced_mean_difference_m": float(np.mean(differences)),
            "object_differences_m": differences.tolist(),
            "stratum_mean_difference_m": strata,
            "object_cluster_bootstrap": bootstrap,
            "gates": gates,
        }
    accepted = [row for row in rows if row["eligible_update_count"] > 0]
    harmful = [
        row["object_id"]
        for row in accepted
        if any(row["regret_m"][metric] > 0.0 for metric in PRIMARY_METRICS)
    ]
    harmful_rate = 0.0 if not accepted else len(harmful) / len(accepted)
    global_gates = {
        "minimum_evaluable_objects": len(rows)
        >= int(config["minimum_evaluable_objects"]),
        "minimum_evaluable_objects_per_stratum": all(
            count >= int(config["minimum_evaluable_objects_per_stratum"])
            for count in stratum_counts.values()
        ),
        "both_primary_metric_gates": all(metric_passes),
        "accepted_harmful_object_rate": harmful_rate
        <= float(config["success_gates"]["accepted_harmful_object_rate_at_most"]),
        "every_rejection_bit_exact_fallback": all(
            row["all_rejections_bit_exact_fallback"] for row in rows
        ),
        "all_quality_failures_reported": len(seen) + len(failure_cases)
        == len(expected),
        "no_replacement": all(
            row.get("replacement_allowed") is False for row in quality_failures
        ),
    }
    passed = all(global_gates.values())
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": TARGET_RESULT_ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": protocol["config_sha256"],
        "calibration_gate_result_sha256": calibration_gate_result_sha256,
        "target_prediction_cohort_result_sha256": target_cohort_result_sha256,
        "object_count": len(rows),
        "episode_count": len(reports),
        "object_count_by_stratum": stratum_counts,
        "quality_failure_count": len(quality_failures),
        "quality_failures": [dict(row) for row in quality_failures],
        "target_evaluation_result_sha256": {
            str(report["case"]): str(report["result_sha256"]) for report in reports
        },
        "object_results": rows,
        "primary_comparisons": comparisons,
        "accepted_object_count": len(accepted),
        "accepted_harmful_objects": harmful,
        "accepted_harmful_object_rate": harmful_rate,
        "gates": global_gates,
        "paper_threshold_passed": passed,
        "permitted_claim": (
            protocol["config"]["target_evaluation"]["permitted_positive_claim"]
            if passed
            else (
                "No general fresh-object improvement claim; report the frozen "
                "negative or mixed result without replacement tuning."
            )
        ),
        "claim_boundary": (
            "fresh public-data test against the exact selected raw/physical "
            "backbone; not official Deform360 Table-4 parity"
        ),
    }
    payload["result_sha256"] = canonical_sha256(payload, digest_key="result_sha256")
    return payload


def collect_prospective_case_evaluations(
    protocol_path: str | Path,
    cohort_seal: Mapping[str, Any],
    artifact_root: str | Path,
    evaluation_root: str | Path,
    outcome_failure_root: str | Path,
    *,
    role: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load exactly one post-seal disposition for every locked role case."""

    protocol = load_bias_aware_prospective_protocol(protocol_path)
    validate_prospective_prediction_cohort_seal(
        cohort_seal,
        protocol_path=protocol_path,
        role=role,
        artifact_root=artifact_root,
    )
    evaluations = Path(evaluation_root).resolve()
    outcome_failures = Path(outcome_failure_root).resolve()
    reports: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    cohort_by_case = {str(row["case"]): row for row in cohort_seal["cases"]}
    for record in prospective_case_records(protocol_path, role=role):
        case = str(record["case"])
        cohort_row = cohort_by_case[case]
        if cohort_row["disposition"] == "quality_failure":
            source = Path(artifact_root).resolve() / case / QUALITY_FAILURE_FILENAME
            failure = json.loads(source.read_text(encoding="utf-8"))
            failures.append(
                {
                    **record,
                    "stage": failure["stage"],
                    "error_type": failure["error_type"],
                    "error_message": failure["error_message"],
                    "replacement_allowed": False,
                    "pre_outcome": True,
                    "artifact_sha256": file_sha256(source),
                }
            )
            continue
        evaluation_path = evaluations / case / CASE_EVALUATION_FILENAME
        failure_path = outcome_failures / case / OUTCOME_FAILURE_FILENAME
        _require(
            evaluation_path.is_file() != failure_path.is_file(),
            f"case needs exactly one post-seal disposition: {case}",
        )
        if evaluation_path.is_file():
            report = json.loads(evaluation_path.read_text(encoding="utf-8"))
            _validate_case_report(
                report,
                protocol_config_sha256=str(protocol["config_sha256"]),
                role=role,
            )
            _require(
                all(report.get(key) == value for key, value in record.items()),
                f"evaluation identity changed: {case}",
            )
            reports.append(report)
        else:
            failure = json.loads(failure_path.read_text(encoding="utf-8"))
            _require(
                failure.get("artifact_kind") == OUTCOME_FAILURE_ARTIFACT_KIND
                and failure.get("result_sha256")
                == canonical_sha256(failure, digest_key="result_sha256")
                and all(failure.get(key) == value for key, value in record.items()),
                f"outcome failure changed: {case}",
            )
            _require(
                failure.get("replacement_allowed") is False
                and failure.get("authorization", {}).get(
                    "prediction_cohort_result_sha256"
                )
                == cohort_seal["result_sha256"]
                and failure.get("authorization", {}).get("prediction_result_sha256")
                == cohort_row["artifact_result_sha256"]
                and failure.get("information_boundary")
                == {
                    "future_read_after_prediction_cohort_seal": True,
                    "failure_retained_without_replacement": True,
                },
                f"outcome-failure authorization changed: {case}",
            )
            failures.append(failure)
    return reports, failures


def record_prospective_outcome_failure(
    protocol_path: str | Path,
    output_dir: str | Path,
    *,
    object_id: str,
    episode_id: int,
    stage: str,
    error_type: str,
    error_message: str,
    prediction_cohort_result_sha256: str,
    prediction_result_sha256: str,
    evidence_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Record a post-authorization failure without replacing the case."""

    protocol = load_bias_aware_prospective_protocol(protocol_path)
    record = prospective_case_record(
        protocol_path, object_id=object_id, episode_id=episode_id
    )
    _require(
        stage in {"authorized-future", "authorized-outcome", "evaluation"},
        "invalid post-outcome failure stage",
    )
    evidence: dict[str, str] = {}
    for name, value in sorted((evidence_paths or {}).items()):
        path = Path(value).resolve()
        _require(path.is_file(), f"outcome failure evidence is missing: {name}")
        evidence[name] = file_sha256(path)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": OUTCOME_FAILURE_ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": protocol["config_sha256"],
        **record,
        "stage": stage,
        "error_type": error_type,
        "error_message": error_message,
        "replacement_allowed": False,
        "evidence_sha256": evidence,
        "authorization": {
            "prediction_cohort_result_sha256": prediction_cohort_result_sha256,
            "prediction_result_sha256": prediction_result_sha256,
        },
        "information_boundary": {
            "future_read_after_prediction_cohort_seal": True,
            "failure_retained_without_replacement": True,
        },
    }
    payload["result_sha256"] = canonical_sha256(payload, digest_key="result_sha256")
    _write_json(output / OUTCOME_FAILURE_FILENAME, payload)
    return payload


__all__ = [
    "AUTHORIZED_FUTURE_ARTIFACT_KIND",
    "AUTHORIZED_FUTURE_MANIFEST_FILENAME",
    "AUTHORIZED_OUTCOME_ARTIFACT_KIND",
    "CALIBRATION_GATE_ARTIFACT_KIND",
    "CALIBRATION_GATE_FILENAME",
    "CASE_EVALUATION_ARTIFACT_KIND",
    "CASE_EVALUATION_FILENAME",
    "OUTCOME_FAILURE_ARTIFACT_KIND",
    "OUTCOME_FAILURE_FILENAME",
    "OUTCOME_MANIFEST_FILENAME",
    "PRIMARY_METRICS",
    "SCORED_FRAMES",
    "TARGET_ARCHIVE_FILENAME",
    "TARGET_RESULT_ARTIFACT_KIND",
    "TARGET_RESULT_FILENAME",
    "aggregate_bias_aware_target_result",
    "collect_prospective_case_evaluations",
    "evaluate_bias_aware_prospective_case",
    "fit_bias_aware_calibration_gate",
    "record_prospective_outcome_failure",
    "score_bias_aware_prospective_arrays",
    "validate_bias_aware_calibration_gate",
]
