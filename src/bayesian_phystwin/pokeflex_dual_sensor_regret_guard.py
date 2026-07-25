"""Target-free delayed consensus for PokeFlex candidate selection."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


_CANDIDATE_PATTERN = re.compile(
    r"^checkpoint_action_local_state_relative_"
    r"(?:0\.4|0\.55|0\.7)_residual_scale_"
    r"(?:0\.125|0\.25|0\.5|1)$"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _take_identity(take_id: str) -> tuple[str, str]:
    object_name, separator, take_number = take_id.rpartition("_T")
    _require(
        bool(separator) and bool(object_name) and take_number.isdigit(),
        f"invalid PokeFlex take id: {take_id}",
    )
    return object_name, f"T{take_number}"


def _correlated_family_upper(
    evidence: Mapping[str, Any],
    *,
    values_key: str,
    recorded_upper_key: str,
) -> float:
    values = np.asarray(evidence.get(values_key), dtype=np.float64)
    _require(
        values.ndim == 1
        and len(values) >= 1
        and np.all(np.isfinite(values)),
        "sensor-family regret is invalid",
    )
    upper = float(np.max(values))
    recorded = float(evidence.get(recorded_upper_key))
    _require(np.isfinite(recorded), "recorded sensor-family upper is invalid")
    _require(
        np.isclose(recorded, upper, rtol=0.0, atol=1e-12),
        "sensor-family upper does not match its correlated observations",
    )
    return upper


def evaluate_pokeflex_dual_sensor_consensus(
    payloads: Sequence[Mapping[str, Any]],
    *,
    expected_take_ids: Sequence[str] | None = None,
    minimum_improvement_mm: float = 0.0,
) -> dict[str, Any]:
    """Select arms using only delayed Kinect and independent D405 evidence.

    Both evidence families score the candidate generated for frame ``f-1``
    before an arm with the same frozen radius and scale is selected for frame
    ``f``. Unknown within-family correlation is handled by the maximum regret;
    agreement across the two sensor families is also required by a maximum.
    """

    _require(bool(payloads), "at least one candidate artifact is required")
    _require(minimum_improvement_mm >= 0.0, "improvement margin is negative")
    decisions: list[dict[str, Any]] = []
    seen_takes: set[str] = set()
    for payload in payloads:
        _require(
            payload.get("artifact_kind")
            == "PokeFlexCheckpointBayesianRegistrationDevelopmentSmoke",
            "unexpected candidate artifact kind",
        )
        _require(payload.get("future_observation_used") is False, "future input used")
        _require(
            payload.get("online_observation_regret_recorded") is True,
            "delayed Kinect evidence is missing",
        )
        take_id = str(payload.get("take", {}).get("id", ""))
        object_name, take_number = _take_identity(take_id)
        _require(take_id not in seen_takes, f"duplicate take: {take_id}")
        seen_takes.add(take_id)
        targets = payload.get("targets")
        _require(isinstance(targets, list) and targets, "candidate targets missing")
        for target in targets:
            target_frame = int(target["target_frame"])
            baseline = float(target["released_checkpoint_CD_UL1_mm"])
            _require(np.isfinite(baseline) and baseline > 0.0, "baseline invalid")
            d405_bank = target.get("independent_anchor_regret")
            kinect_bank = target.get("online_observation_regret")
            _require(isinstance(d405_bank, Mapping), "D405 evidence bank invalid")
            _require(isinstance(kinect_bank, Mapping), "Kinect evidence bank invalid")
            candidates: list[tuple[float, str, float, float]] = []
            for name in sorted(set(d405_bank) & set(kinect_bank)):
                if _CANDIDATE_PATTERN.match(str(name)) is None:
                    continue
                d405 = d405_bank[name]
                kinect = kinect_bank[name]
                _require(isinstance(d405, Mapping), "D405 candidate evidence invalid")
                _require(
                    isinstance(kinect, Mapping), "Kinect candidate evidence invalid"
                )
                d405_upper = _correlated_family_upper(
                    d405,
                    values_key="per_sensor_mm",
                    recorded_upper_key="covariance_intersection_upper_mm",
                )
                kinect_upper = _correlated_family_upper(
                    kinect,
                    values_key="per_view_mm",
                    recorded_upper_key="covariance_intersection_upper_mm",
                )
                evaluated_frame = int(d405["evaluated_prediction_frame"])
                _require(
                    evaluated_frame == target_frame - 1,
                    "D405 evidence is not a one-step delayed score",
                )
                candidates.append(
                    (max(d405_upper, kinect_upper), str(name), d405_upper, kinect_upper)
                )
            selected_name = "released_checkpoint"
            selected_error = baseline
            consensus_upper = None
            d405_upper = None
            kinect_upper = None
            if candidates:
                consensus_upper, proposed, d405_upper, kinect_upper = min(candidates)
                if consensus_upper < -minimum_improvement_mm:
                    outcome = float(target[proposed])
                    _require(np.isfinite(outcome), "candidate outcome is invalid")
                    selected_name = proposed
                    selected_error = outcome
            decisions.append(
                {
                    "frame_id": f"{take_id}:f{target_frame:05d}",
                    "take_id": take_id,
                    "object": object_name,
                    "take": take_number,
                    "target_frame": target_frame,
                    "baseline_error_mm": baseline,
                    "selected_error_mm": selected_error,
                    "hidden_regret_mm": selected_error - baseline,
                    "selected_arm": selected_name,
                    "accepted": selected_name != "released_checkpoint",
                    "consensus_upper_regret_mm": consensus_upper,
                    "d405_upper_regret_mm": d405_upper,
                    "delayed_kinect_upper_regret_mm": kinect_upper,
                }
            )

    observed_takes = sorted(seen_takes)
    if expected_take_ids is not None:
        _require(
            observed_takes == sorted(map(str, expected_take_ids)),
            "candidate take inventory changed",
        )
    take_rows = []
    for take_id in observed_takes:
        current = [value for value in decisions if value["take_id"] == take_id]
        baseline = float(np.mean([value["baseline_error_mm"] for value in current]))
        selected = float(np.mean([value["selected_error_mm"] for value in current]))
        take_rows.append(
            {
                "take_id": take_id,
                "object": current[0]["object"],
                "target_frame_count": len(current),
                "baseline_mean_CD_UL1_mm": baseline,
                "selected_mean_CD_UL1_mm": selected,
                "relative_improvement": (baseline - selected) / baseline,
            }
        )
    object_rows = []
    for object_name in sorted({str(value["object"]) for value in take_rows}):
        current = [value for value in take_rows if value["object"] == object_name]
        baseline = float(np.mean([value["baseline_mean_CD_UL1_mm"] for value in current]))
        selected = float(np.mean([value["selected_mean_CD_UL1_mm"] for value in current]))
        object_rows.append(
            {
                "object": object_name,
                "take_count": len(current),
                "baseline_mean_CD_UL1_mm": baseline,
                "selected_mean_CD_UL1_mm": selected,
                "relative_improvement": (baseline - selected) / baseline,
            }
        )
    baseline_mean = float(
        np.mean([value["baseline_mean_CD_UL1_mm"] for value in object_rows])
    )
    selected_mean = float(
        np.mean([value["selected_mean_CD_UL1_mm"] for value in object_rows])
    )
    accepted = [value for value in decisions if value["accepted"]]
    return {
        "schema_version": 1,
        "artifact_kind": "PokeFlexDualSensorConsensusDevelopmentEvaluation",
        "claim_status": "post-open source/calibration method development",
        "selection_rule": (
            "choose the arm minimizing max(D405-family regret, delayed-Kinect-family "
            "regret); accept only when that upper regret is below the fixed margin"
        ),
        "correlation_treatment": (
            "maximum within D405 cameras, maximum within Kinect cameras, and maximum "
            "between sensor families"
        ),
        "minimum_improvement_mm": minimum_improvement_mm,
        "aggregation": "equal frames within take, equal takes within object, equal objects",
        "take_ids": observed_takes,
        "object_count": len(object_rows),
        "take_count": len(take_rows),
        "baseline_object_mean_CD_UL1_mm": baseline_mean,
        "selected_object_mean_CD_UL1_mm": selected_mean,
        "object_balanced_relative_improvement": (
            baseline_mean - selected_mean
        ) / baseline_mean,
        "object_wins": sum(value["relative_improvement"] > 1e-12 for value in object_rows),
        "object_losses": sum(value["relative_improvement"] < -1e-12 for value in object_rows),
        "accepted_frame_count": len(accepted),
        "accepted_frame_wins": sum(
            value["hidden_regret_mm"] < -1e-12 for value in accepted
        ),
        "accepted_frame_losses": sum(
            value["hidden_regret_mm"] > 1e-12 for value in accepted
        ),
        "exact_fallback_frame_count": len(decisions) - len(accepted),
        "objects": object_rows,
        "takes": take_rows,
        "decisions": decisions,
    }
