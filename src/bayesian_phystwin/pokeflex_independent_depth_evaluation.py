"""Causal evaluation for PokeFlex independent-depth arm selection."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import json

import numpy as np
from scipy.stats import pearsonr, spearmanr


BASELINE_ARM = "released_checkpoint"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_float(value: object, name: str) -> float:
    result = float(value)
    _require(np.isfinite(result), f"{name} must be finite")
    return result


def _correlation(
    left: Sequence[float],
    right: Sequence[float],
    *,
    kind: str,
) -> float | None:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if len(first) < 2 or np.ptp(first) <= 1e-12 or np.ptp(second) <= 1e-12:
        return None
    value = spearmanr(first, second).statistic if kind == "spearman" else pearsonr(
        first, second
    ).statistic
    return float(value)


def _take_identity(take_id: str) -> tuple[str, str]:
    object_name, separator, take_number = take_id.rpartition("_T")
    _require(
        bool(separator) and bool(object_name) and take_number.isdigit(),
        f"invalid PokeFlex take id: {take_id}",
    )
    return object_name, f"T{take_number}"


def _validate_artifact(
    payload: Mapping[str, Any],
) -> tuple[str, list[Mapping[str, Any]], Mapping[str, Any]]:
    _require(
        payload.get("artifact_kind")
        == "PokeFlexCheckpointBayesianRegistrationDevelopmentSmoke",
        "unexpected PokeFlex source artifact kind",
    )
    _require(payload.get("future_observation_used") is False, "future input was used")
    anchor = payload.get("independent_depth_anchor")
    _require(isinstance(anchor, Mapping), "independent-depth evidence is missing")
    _require(anchor.get("future_observation_used") is False, "future anchor was used")
    take = payload.get("take")
    _require(isinstance(take, Mapping), "take metadata is missing")
    take_id = str(take.get("id", ""))
    _take_identity(take_id)
    targets = payload.get("targets")
    _require(isinstance(targets, list) and targets, "target records are missing")
    frames = [int(record["target_frame"]) for record in targets]
    _require(frames == sorted(set(frames)), "target frames must be unique and sorted")
    return take_id, targets, anchor


def _candidate_regrets(
    target: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Any]]:
    value = target.get("independent_anchor_regret", {})
    _require(isinstance(value, Mapping), "anchor-regret record has invalid type")
    return value


def _hidden_regret(target: Mapping[str, Any], candidate: str) -> float:
    baseline = _finite_float(
        target["released_checkpoint_CD_UL1_mm"], "released checkpoint error"
    )
    _require(candidate in target, f"candidate outcome is missing: {candidate}")
    return _finite_float(target[candidate], "candidate error") - baseline


def evaluate_independent_depth_take(
    payload: Mapping[str, Any],
    *,
    minimum_anchor_improvement_mm: float = 0.0,
    maximum_calibration_median_residual_mm: float = 10.0,
) -> dict[str, Any]:
    """Evaluate anchor competence and a strictly causal one-frame-lag selector."""

    _require(
        minimum_anchor_improvement_mm >= 0.0,
        "anchor improvement margin must be non-negative",
    )
    _require(
        maximum_calibration_median_residual_mm > 0.0,
        "calibration residual limit must be positive",
    )
    take_id, targets, anchor_metadata = _validate_artifact(payload)
    object_name, take_number = _take_identity(take_id)
    by_frame = {int(record["target_frame"]): record for record in targets}
    calibration_median = np.asarray(
        anchor_metadata.get("median_residual_mm", ()), dtype=np.float64
    )
    _require(
        calibration_median.ndim == 1
        and len(calibration_median) >= 1
        and np.all(np.isfinite(calibration_median)),
        "per-sensor calibration residuals are missing",
    )
    eligible_sensors = np.flatnonzero(
        calibration_median <= maximum_calibration_median_residual_mm
    )

    def qualified_regret(record: Mapping[str, Any]) -> float:
        per_sensor = np.asarray(record.get("per_sensor_mm", ()), dtype=np.float64)
        _require(
            per_sensor.shape == calibration_median.shape
            and np.all(np.isfinite(per_sensor)),
            "per-sensor anchor regret inventory changed",
        )
        _require(len(eligible_sensors) >= 1, "no eligible independent-depth sensor")
        return float(np.max(per_sensor[eligible_sensors]))

    predicted_regrets: list[float] = []
    hidden_regrets: list[float] = []
    accepted_hidden_regrets: list[float] = []
    competence_rows: list[dict[str, Any]] = []
    for evidence_target in targets:
        evidence_frame = int(evidence_target["target_frame"])
        if len(eligible_sensors) == 0:
            continue
        for candidate, record in sorted(_candidate_regrets(evidence_target).items()):
            _require(isinstance(record, Mapping), "candidate regret record is invalid")
            evaluated_frame = int(record["evaluated_prediction_frame"])
            _require(
                evaluated_frame == evidence_frame - 1,
                "independent anchor is not aligned to the preceding prediction",
            )
            _require(
                evaluated_frame in by_frame,
                "anchor evaluates a prediction outside the scored target set",
            )
            predicted = qualified_regret(record)
            hidden = _hidden_regret(by_frame[evaluated_frame], candidate)
            accepted = predicted < -minimum_anchor_improvement_mm
            if accepted:
                accepted_hidden_regrets.append(hidden)
            predicted_regrets.append(predicted)
            hidden_regrets.append(hidden)
            competence_rows.append(
                {
                    "evidence_frame": evidence_frame,
                    "evaluated_prediction_frame": evaluated_frame,
                    "candidate": candidate,
                    "predicted_regret_mm": predicted,
                    "hidden_regret_mm": hidden,
                    "accepted": accepted,
                    "false_safe": bool(accepted and hidden > 0.0),
                }
            )

    predicted_array = np.asarray(predicted_regrets, dtype=np.float64)
    hidden_array = np.asarray(hidden_regrets, dtype=np.float64)
    accepted_count = int(sum(row["accepted"] for row in competence_rows))
    false_safe_count = int(sum(row["false_safe"] for row in competence_rows))
    sign_agreement = (
        float(np.mean(np.signbit(predicted_array) == np.signbit(hidden_array)))
        if competence_rows
        else None
    )
    competence = {
        "candidate_frame_pair_count": len(competence_rows),
        "spearman": _correlation(predicted_regrets, hidden_regrets, kind="spearman"),
        "pearson": _correlation(predicted_regrets, hidden_regrets, kind="pearson"),
        "sign_agreement": sign_agreement,
        "acceptance_rate": (
            accepted_count / len(competence_rows) if competence_rows else 0.0
        ),
        "false_safe_count": false_safe_count,
        "false_safe_rate_among_accepted": (
            false_safe_count / accepted_count if accepted_count else 0.0
        ),
        "mean_hidden_regret_accepted_mm": (
            float(np.mean(accepted_hidden_regrets))
            if accepted_hidden_regrets
            else None
        ),
        "rows": competence_rows,
    }

    selection_rows: list[dict[str, Any]] = []
    for target in targets:
        target_frame = int(target["target_frame"])
        baseline = _finite_float(
            target["released_checkpoint_CD_UL1_mm"], "released checkpoint error"
        )
        evidence = _candidate_regrets(target)
        selected_arm = BASELINE_ARM
        predicted_regret = None
        reason = "no-preceding-independent-anchor"
        if evidence and len(eligible_sensors) >= 1:
            selected_arm, selected_record = min(
                evidence.items(),
                key=lambda item: (
                    qualified_regret(item[1]),
                    item[0],
                ),
            )
            predicted_regret = qualified_regret(selected_record)
            if predicted_regret >= -minimum_anchor_improvement_mm:
                selected_arm = BASELINE_ARM
                reason = "independent-anchor-fallback"
            else:
                reason = "independent-anchor-selected"
        selected_error = (
            baseline
            if selected_arm == BASELINE_ARM
            else _finite_float(target[selected_arm], "selected candidate error")
        )
        selection_rows.append(
            {
                "target_frame": target_frame,
                "selected_arm": selected_arm,
                "reason": reason,
                "predicted_regret_mm": predicted_regret,
                "baseline_CD_UL1_mm": baseline,
                "selected_CD_UL1_mm": selected_error,
                "hidden_difference_mm": selected_error - baseline,
            }
        )

    baseline_values = np.asarray(
        [row["baseline_CD_UL1_mm"] for row in selection_rows], dtype=np.float64
    )
    selected_values = np.asarray(
        [row["selected_CD_UL1_mm"] for row in selection_rows], dtype=np.float64
    )
    differences = selected_values - baseline_values
    nonzero = np.abs(differences) > 1e-12
    baseline_mean = float(np.mean(baseline_values))
    selected_mean = float(np.mean(selected_values))
    selector = {
        "target_frame_count": len(selection_rows),
        "baseline_mean_CD_UL1_mm": baseline_mean,
        "selected_mean_CD_UL1_mm": selected_mean,
        "relative_improvement": (baseline_mean - selected_mean) / baseline_mean,
        "mean_difference_mm": float(np.mean(differences)),
        "wins": int(np.sum(differences < -1e-12)),
        "losses": int(np.sum(differences > 1e-12)),
        "fallback_ties": int(np.sum(~nonzero)),
        "nonbaseline_selection_count": int(
            sum(row["selected_arm"] != BASELINE_ARM for row in selection_rows)
        ),
        "rows": selection_rows,
    }
    return {
        "take_id": take_id,
        "object": object_name,
        "take": take_number,
        "minimum_anchor_improvement_mm": minimum_anchor_improvement_mm,
        "sensor_quality": {
            "maximum_calibration_median_residual_mm": (
                maximum_calibration_median_residual_mm
            ),
            "calibration_median_residual_mm": calibration_median.tolist(),
            "eligible_sensor_indices": eligible_sensors.tolist(),
            "eligible_sensor_count": len(eligible_sensors),
            "policy": "exclude failed sensor; exact fallback if none remain",
        },
        "competence": competence,
        "selector": selector,
    }


def evaluate_independent_depth_artifacts(
    payloads: Sequence[Mapping[str, Any]],
    *,
    minimum_anchor_improvement_mm: float = 0.0,
    maximum_calibration_median_residual_mm: float = 10.0,
) -> dict[str, Any]:
    """Aggregate causal selector results with equal object weight."""

    _require(bool(payloads), "at least one source artifact is required")
    takes = [
        evaluate_independent_depth_take(
            payload,
            minimum_anchor_improvement_mm=minimum_anchor_improvement_mm,
            maximum_calibration_median_residual_mm=(
                maximum_calibration_median_residual_mm
            ),
        )
        for payload in payloads
    ]
    take_ids = [result["take_id"] for result in takes]
    _require(len(set(take_ids)) == len(take_ids), "duplicate take artifact")

    object_takes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in takes:
        object_takes[result["object"]].append(result)
    objects = []
    for object_name, results in sorted(object_takes.items()):
        baseline = float(
            np.mean(
                [result["selector"]["baseline_mean_CD_UL1_mm"] for result in results]
            )
        )
        selected = float(
            np.mean(
                [result["selector"]["selected_mean_CD_UL1_mm"] for result in results]
            )
        )
        objects.append(
            {
                "object": object_name,
                "take_count": len(results),
                "baseline_mean_CD_UL1_mm": baseline,
                "selected_mean_CD_UL1_mm": selected,
                "difference_mm": selected - baseline,
                "relative_improvement": (baseline - selected) / baseline,
            }
        )

    predicted = []
    hidden = []
    accepted_count = 0
    false_safe_count = 0
    for result in takes:
        rows = result["competence"]["rows"]
        predicted.extend(row["predicted_regret_mm"] for row in rows)
        hidden.extend(row["hidden_regret_mm"] for row in rows)
        accepted_count += sum(row["accepted"] for row in rows)
        false_safe_count += sum(row["false_safe"] for row in rows)
    baseline_object_mean = float(
        np.mean([result["baseline_mean_CD_UL1_mm"] for result in objects])
    )
    selected_object_mean = float(
        np.mean([result["selected_mean_CD_UL1_mm"] for result in objects])
    )
    competence_sign_agreement = (
        float(
            np.mean(
                np.signbit(np.asarray(predicted))
                == np.signbit(np.asarray(hidden))
            )
        )
        if predicted
        else None
    )
    return {
        "schema_version": 1,
        "artifact_kind": "PokeFlexIndependentDepthSourceEvaluation",
        "method": (
            "strictly causal one-frame-lag selection using minimum worst-sensor "
            "baseline-relative RealSense regret with exact checkpoint fallback"
        ),
        "minimum_anchor_improvement_mm": minimum_anchor_improvement_mm,
        "maximum_calibration_median_residual_mm": (
            maximum_calibration_median_residual_mm
        ),
        "take_count": len(takes),
        "object_count": len(objects),
        "competence": {
            "candidate_frame_pair_count": len(predicted),
            "spearman": _correlation(predicted, hidden, kind="spearman"),
            "pearson": _correlation(predicted, hidden, kind="pearson"),
            "sign_agreement": competence_sign_agreement,
            "acceptance_rate": accepted_count / len(predicted) if predicted else 0.0,
            "false_safe_count": false_safe_count,
            "false_safe_rate_among_accepted": (
                false_safe_count / accepted_count if accepted_count else 0.0
            ),
        },
        "object_balanced_selector": {
            "baseline_mean_CD_UL1_mm": baseline_object_mean,
            "selected_mean_CD_UL1_mm": selected_object_mean,
            "relative_improvement": (
                baseline_object_mean - selected_object_mean
            )
            / baseline_object_mean,
            "object_wins": sum(result["difference_mm"] < -1e-12 for result in objects),
            "object_losses": sum(result["difference_mm"] > 1e-12 for result in objects),
            "object_ties": sum(abs(result["difference_mm"]) <= 1e-12 for result in objects),
        },
        "objects": objects,
        "takes": takes,
    }


def evaluate_locked_independent_depth_source_validation(
    payloads: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the exact v2 source panel and its preregistered gates."""

    _require(
        protocol.get("artifact_kind")
        == "PokeFlexIndependentDepthSourceValidationProtocol",
        "source-validation protocol is required",
    )
    boundary = protocol["evidence_boundary"]
    method = protocol["method_lock"]
    gates = protocol["source_validation"]
    expected = {
        f"{object_name}_{take}"
        for object_name in boundary["development_objects"]
        for take in boundary["source_validation_takes"]
    }
    observed = {str(payload.get("take", {}).get("id", "")) for payload in payloads}
    _require(observed == expected, "source-validation take inventory changed")
    protocol_sha256 = str(protocol["protocol_sha256"])
    for payload in payloads:
        anchor = payload.get("independent_depth_anchor", {})
        _require(
            anchor.get("protocol_sha256") == protocol_sha256,
            "source artifact protocol checksum changed",
        )
        _require(
            float(anchor.get("maximum_template_distance_m", -1.0))
            == float(method["static_template_support_radius_mm"]) / 1000.0,
            "source artifact template support radius changed",
        )

    result = evaluate_independent_depth_artifacts(
        payloads,
        minimum_anchor_improvement_mm=float(
            method["minimum_anchor_improvement_mm"]
        ),
        maximum_calibration_median_residual_mm=float(
            method["maximum_calibration_median_residual_mm"]
        ),
    )
    competence = result["competence"]
    selector = result["object_balanced_selector"]
    maximum_regression = max(
        max(0.0, -float(row["relative_improvement"]))
        for row in result["objects"]
    )
    checks = {
        "regret_sign_agreement": bool(
            competence["sign_agreement"] is not None
            and competence["sign_agreement"]
            >= float(gates["minimum_regret_sign_agreement"])
        ),
        "false_safe_rate": bool(
            competence["false_safe_rate_among_accepted"]
            <= float(gates["maximum_false_safe_rate"])
        ),
        "regret_spearman": bool(
            competence["spearman"] is not None
            and competence["spearman"]
            >= float(gates["minimum_regret_spearman"])
        ),
        "object_balanced_improvement": bool(
            selector["relative_improvement"]
            >= float(
                gates["minimum_object_balanced_CD_UL1_relative_improvement"]
            )
        ),
        "object_wins": bool(
            selector["object_wins"] >= int(gates["minimum_object_wins"])
        ),
        "maximum_object_regression": bool(
            maximum_regression
            <= float(gates["maximum_per_object_relative_regression"])
        ),
    }
    result["artifact_kind"] = "PokeFlexIndependentDepthSourceValidationResult"
    result["protocol_sha256"] = protocol_sha256
    result["registered_gate"] = {
        "checks": checks,
        "maximum_per_object_relative_regression": maximum_regression,
        "all_passed": all(checks.values()),
        "T2_access_permitted": bool(
            gates["all_required_before_T2_access"] and all(checks.values())
        ),
    }
    return result


def load_and_evaluate_independent_depth_artifacts(
    paths: Sequence[str | Path],
    *,
    minimum_anchor_improvement_mm: float = 0.0,
    maximum_calibration_median_residual_mm: float = 10.0,
) -> dict[str, Any]:
    """Load JSON smoke artifacts and evaluate them causally."""

    sources = [Path(path).resolve() for path in paths]
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in sources]
    result = evaluate_independent_depth_artifacts(
        payloads,
        minimum_anchor_improvement_mm=minimum_anchor_improvement_mm,
        maximum_calibration_median_residual_mm=(
            maximum_calibration_median_residual_mm
        ),
    )
    result["sources"] = [str(path) for path in sources]
    return result
