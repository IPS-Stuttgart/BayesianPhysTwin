"""Evaluation for same-time PokeFlex independent-depth state updates."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr


BASELINE_ARM = "released_checkpoint"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _correlation(left: Sequence[float], right: Sequence[float], kind: str) -> float | None:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if len(first) < 2 or np.ptp(first) <= 1e-12 or np.ptp(second) <= 1e-12:
        return None
    statistic = (
        spearmanr(first, second).statistic
        if kind == "spearman"
        else pearsonr(first, second).statistic
    )
    return float(statistic)


def _take_identity(take_id: str) -> tuple[str, str]:
    object_name, separator, take_number = take_id.rpartition("_T")
    _require(
        bool(separator) and bool(object_name) and take_number.isdigit(),
        f"invalid PokeFlex take id: {take_id}",
    )
    return object_name, f"T{take_number}"


def evaluate_current_state_take(
    payload: Mapping[str, Any],
    *,
    maximum_calibration_median_residual_mm: float = 10.0,
    minimum_anchor_improvement_mm: float = 0.0,
) -> dict[str, Any]:
    """Score current state candidates using only same-time causal evidence."""

    _require(
        payload.get("artifact_kind")
        == "PokeFlexIndependentDepthCurrentStateDiagnostic",
        "unexpected current-state diagnostic kind",
    )
    _require(payload.get("future_observation_used") is False, "future input was used")
    _require(
        payload.get("target_mesh_used_by_diagnostic_runner") is False,
        "diagnostic runner reopened target meshes",
    )
    take_id = str(payload.get("take", {}).get("id", ""))
    object_name, take_number = _take_identity(take_id)
    targets = payload.get("targets")
    _require(isinstance(targets, list) and targets, "diagnostic targets are missing")
    calibration = np.asarray(
        payload.get("independent_depth_anchor", {}).get("median_residual_mm", ()),
        dtype=np.float64,
    )
    _require(
        calibration.ndim == 1
        and len(calibration) >= 1
        and np.all(np.isfinite(calibration)),
        "calibration residual inventory changed",
    )
    eligible = np.flatnonzero(calibration <= maximum_calibration_median_residual_mm)

    predicted: list[float] = []
    hidden: list[float] = []
    false_safe = 0
    accepted = 0
    selector_rows = []
    competence_rows = []
    for target in targets:
        source_frame = int(target["source_frame"])
        target_frame = int(target["target_frame"])
        _require(source_frame == target_frame - 1, "diagnostic timing changed")
        baseline = float(target["released_checkpoint_CD_UL1_mm"])
        evidence = target.get("current_state_anchor_regret", {})
        _require(isinstance(evidence, Mapping), "current-state evidence is invalid")
        scored = []
        if len(eligible) >= 1:
            for candidate, record in sorted(evidence.items()):
                _require(
                    int(record["evidence_frame"]) == source_frame,
                    "D405 evidence is not same-time with the source state",
                )
                per_sensor = np.asarray(record["per_sensor_mm"], dtype=np.float64)
                _require(
                    per_sensor.shape == calibration.shape,
                    "per-sensor regret inventory changed",
                )
                estimate = float(np.max(per_sensor[eligible]))
                outcome = float(target[candidate]) - baseline
                is_accepted = estimate < -minimum_anchor_improvement_mm
                accepted += is_accepted
                false_safe += is_accepted and outcome > 0.0
                predicted.append(estimate)
                hidden.append(outcome)
                scored.append((estimate, candidate))
                competence_rows.append(
                    {
                        "source_frame": source_frame,
                        "target_frame": target_frame,
                        "candidate": candidate,
                        "predicted_regret_mm": estimate,
                        "hidden_target_regret_mm": outcome,
                        "accepted": is_accepted,
                        "false_safe": bool(is_accepted and outcome > 0.0),
                    }
                )
        selected_arm = BASELINE_ARM
        selected_error = baseline
        predicted_regret = None
        if scored:
            predicted_regret, candidate = min(scored)
            if predicted_regret < -minimum_anchor_improvement_mm:
                selected_arm = candidate
                selected_error = float(target[candidate])
        selector_rows.append(
            {
                "source_frame": source_frame,
                "target_frame": target_frame,
                "selected_arm": selected_arm,
                "predicted_regret_mm": predicted_regret,
                "baseline_CD_UL1_mm": baseline,
                "selected_CD_UL1_mm": selected_error,
                "hidden_difference_mm": selected_error - baseline,
            }
        )

    baseline_values = np.asarray(
        [row["baseline_CD_UL1_mm"] for row in selector_rows], dtype=np.float64
    )
    selected_values = np.asarray(
        [row["selected_CD_UL1_mm"] for row in selector_rows], dtype=np.float64
    )
    differences = selected_values - baseline_values
    baseline_mean = float(np.mean(baseline_values))
    selected_mean = float(np.mean(selected_values))
    return {
        "take_id": take_id,
        "object": object_name,
        "take": take_number,
        "sensor_quality": {
            "calibration_median_residual_mm": calibration.tolist(),
            "eligible_sensor_indices": eligible.tolist(),
        },
        "competence": {
            "candidate_frame_pair_count": len(predicted),
            "spearman": _correlation(predicted, hidden, "spearman"),
            "pearson": _correlation(predicted, hidden, "pearson"),
            "sign_agreement": (
                float(
                    np.mean(
                        np.signbit(np.asarray(predicted))
                        == np.signbit(np.asarray(hidden))
                    )
                )
                if predicted
                else None
            ),
            "acceptance_rate": accepted / len(predicted) if predicted else 0.0,
            "false_safe_rate_among_accepted": (
                false_safe / accepted if accepted else 0.0
            ),
            "rows": competence_rows,
        },
        "selector": {
            "target_frame_count": len(selector_rows),
            "baseline_mean_CD_UL1_mm": baseline_mean,
            "selected_mean_CD_UL1_mm": selected_mean,
            "relative_improvement": (baseline_mean - selected_mean) / baseline_mean,
            "wins": int(np.sum(differences < -1e-12)),
            "losses": int(np.sum(differences > 1e-12)),
            "fallback_ties": int(np.sum(np.abs(differences) <= 1e-12)),
            "rows": selector_rows,
        },
    }


def evaluate_current_state_artifacts(
    payloads: Sequence[Mapping[str, Any]],
    *,
    maximum_calibration_median_residual_mm: float = 10.0,
    minimum_anchor_improvement_mm: float = 0.0,
) -> dict[str, Any]:
    """Aggregate same-time diagnostics with equal object and take weight."""

    _require(bool(payloads), "at least one current-state artifact is required")
    takes = [
        evaluate_current_state_take(
            payload,
            maximum_calibration_median_residual_mm=(
                maximum_calibration_median_residual_mm
            ),
            minimum_anchor_improvement_mm=minimum_anchor_improvement_mm,
        )
        for payload in payloads
    ]
    _require(
        len({result["take_id"] for result in takes}) == len(takes),
        "duplicate current-state take",
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in takes:
        grouped[result["object"]].append(result)
    objects = []
    for object_name, results in sorted(grouped.items()):
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
                "relative_improvement": (baseline - selected) / baseline,
            }
        )
    baseline = float(np.mean([row["baseline_mean_CD_UL1_mm"] for row in objects]))
    selected = float(np.mean([row["selected_mean_CD_UL1_mm"] for row in objects]))
    predicted = [
        row["predicted_regret_mm"]
        for take in takes
        for row in take["competence"]["rows"]
    ]
    hidden = [
        row["hidden_target_regret_mm"]
        for take in takes
        for row in take["competence"]["rows"]
    ]
    accepted_rows = [
        row
        for take in takes
        for row in take["competence"]["rows"]
        if row["accepted"]
    ]
    return {
        "schema_version": 1,
        "artifact_kind": "PokeFlexIndependentDepthCurrentStateEvaluation",
        "claim_status": "post-open source-only mechanism diagnostic",
        "take_count": len(takes),
        "object_count": len(objects),
        "competence": {
            "candidate_frame_pair_count": len(predicted),
            "spearman": _correlation(predicted, hidden, "spearman"),
            "pearson": _correlation(predicted, hidden, "pearson"),
            "sign_agreement": float(
                np.mean(
                    np.signbit(np.asarray(predicted))
                    == np.signbit(np.asarray(hidden))
                )
            ),
            "acceptance_rate": len(accepted_rows) / len(predicted),
            "false_safe_rate_among_accepted": (
                sum(row["false_safe"] for row in accepted_rows) / len(accepted_rows)
                if accepted_rows
                else 0.0
            ),
        },
        "object_balanced_selector": {
            "baseline_mean_CD_UL1_mm": baseline,
            "selected_mean_CD_UL1_mm": selected,
            "relative_improvement": (baseline - selected) / baseline,
            "object_wins": sum(row["relative_improvement"] > 1e-12 for row in objects),
            "object_losses": sum(row["relative_improvement"] < -1e-12 for row in objects),
            "object_ties": sum(abs(row["relative_improvement"]) <= 1e-12 for row in objects),
        },
        "objects": objects,
        "takes": takes,
    }
