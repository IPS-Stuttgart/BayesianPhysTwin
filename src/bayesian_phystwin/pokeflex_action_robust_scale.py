"""Repeated-action robust shrinkage scales for PokeFlex belief updates."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ACTION_ROBUST_SCALE_KIND = "PokeFlexActionRobustScaleCalibration"
ACTION_ROBUST_SCALE_ID = "pokeflex-action-robust-scale-v3"
ACTION_ROBUST_SCALE_SHA256 = (
    "78d3c74e4246ec6b69cbcfe113ed04324bf1a9f49d543194df8a7a87d7f09157"
)
ACTION_ROBUST_SCALE_FILE_SHA256 = (
    "96fe0046d15dfdd150b3f2f695b678a5b2b8a6acd790978624b120f6fa6408b0"
)
FIRST_SOURCE_AUDIT_KIND = "PokeFlexFresh12PostopenScaleHeadroomAudit"
FIRST_SOURCE_AUDIT_SHA256 = (
    "78179996296b5ed47692e3ee716308c4525deeb71ce2881442331b5643b4bf94"
)
FIRST_SOURCE_AUDIT_FILE_SHA256 = (
    "114fbf7c3437311f625c9c022742a2c707d6db1e61ca1548b0d8f2500f83d494"
)
SECOND_SOURCE_AUDIT_KIND = (
    "PokeFlexInstanceFresh12PostopenGlobalScaleHeadroomCompactAudit"
)
SECOND_SOURCE_AUDIT_SHA256 = (
    "08bc71efec3c8c99bb469efada2a82048b978a0f5fdde3a149c47b60ac395587"
)
SECOND_SOURCE_AUDIT_FILE_SHA256 = (
    "2583791fa6e64d9d7f9f1493c1fc7dec5c9cf7a2d96bfe27ef0010a561d516d5"
)
BASE_EFFECTIVE_SCALE = 0.125
GLOBAL_MULTIPLIER = 1.0
CANDIDATE_MULTIPLIERS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object_name(take_id: str) -> str:
    object_name, separator, take_number = str(take_id).rpartition("_T")
    _require(bool(separator) and take_number.isdigit(), "invalid source take id")
    return object_name


def calibration_sha256(payload: Mapping[str, Any]) -> str:
    """Return the canonical digest of an action-robust scale calibration."""

    canonical = dict(payload)
    canonical.pop("calibration_sha256", None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _scale_scores(row: Mapping[str, Any]) -> dict[float, float]:
    raw = row.get("mean_CD_UL1_mm_by_multiplier")
    _require(isinstance(raw, Mapping), "source scale scores are missing")
    scores = {value: float(raw[str(value)]) for value in CANDIDATE_MULTIPLIERS}
    _require(
        all(np.isfinite(score) and score > 0.0 for score in scores.values()),
        "source scale score is invalid",
    )
    return scores


def scale_relative_improvement(row: Mapping[str, Any], multiplier: float) -> float:
    """Return improvement relative to the validated global scale."""

    scores = _scale_scores(row)
    selected = float(multiplier)
    _require(selected in scores, "scale multiplier is outside the frozen bank")
    reference = scores[GLOBAL_MULTIPLIER]
    return float((reference - scores[selected]) / reference)


def select_action_robust_multiplier(
    source_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select the scale with the best lower envelope over two interactions."""

    rows = tuple(source_rows)
    _require(len(rows) == 2, "exactly two source interactions are required")
    take_ids = tuple(str(row.get("take_id", "")) for row in rows)
    object_names = tuple(_object_name(take_id) for take_id in take_ids)
    _require(len(set(take_ids)) == 2, "source interactions repeat")
    _require(len(set(object_names)) == 1, "source interactions change object")
    support = tuple(int(row.get("supported_frame_count", -1)) for row in rows)
    _require(all(value >= 0 for value in support), "source support count is invalid")

    if any(value == 0 for value in support):
        selected = GLOBAL_MULTIPLIER
        reason = "insufficient-repeated-action-support"
    else:
        candidates = []
        for multiplier in CANDIDATE_MULTIPLIERS:
            gains = tuple(
                scale_relative_improvement(row, multiplier) for row in rows
            )
            candidates.append(
                (
                    min(gains),
                    float(np.mean(gains)),
                    -abs(math.log(multiplier / GLOBAL_MULTIPLIER)),
                    -multiplier,
                    multiplier,
                    gains,
                )
            )
        selected_record = max(candidates)
        selected = float(selected_record[4])
        reason = (
            "global-lower-envelope-fallback"
            if selected == GLOBAL_MULTIPLIER
            else "repeated-action-maximin"
        )

    gains = tuple(scale_relative_improvement(row, selected) for row in rows)
    _require(min(gains) >= -1e-12, "selected scale regresses on a source action")
    best_by_action = tuple(
        min(
            CANDIDATE_MULTIPLIERS,
            key=lambda multiplier: (
                _scale_scores(row)[multiplier],
                abs(math.log(multiplier / GLOBAL_MULTIPLIER)),
                multiplier,
            ),
        )
        for row in rows
    )
    return {
        "object_name": object_names[0],
        "source_take_ids": list(take_ids),
        "source_supported_frame_counts": list(support),
        "source_best_multipliers": list(best_by_action),
        "multiplier": selected,
        "effective_scale": BASE_EFFECTIVE_SCALE * selected,
        "source_relative_improvements": list(gains),
        "minimum_source_relative_improvement": float(min(gains)),
        "mean_source_relative_improvement": float(np.mean(gains)),
        "selection_reason": reason,
    }


def _validate_source_audit(
    payload: Mapping[str, Any],
    *,
    expected_kind: str,
    expected_sha256: str,
) -> list[Mapping[str, Any]]:
    _require(payload.get("schema_version") == 1, "source audit schema changed")
    _require(payload.get("artifact_kind") == expected_kind, "source audit kind changed")
    _require(payload.get("audit_sha256") == expected_sha256, "source audit changed")
    _require(
        "not prospective evidence" in str(payload.get("status", "")),
        "source audit boundary changed",
    )
    rows = payload.get("takes")
    _require(isinstance(rows, list) and len(rows) == 12, "source cohort changed")
    _require(
        len({_object_name(str(row.get("take_id", ""))) for row in rows}) == 12,
        "source object inventory changed",
    )
    return rows


def action_robust_control_summary() -> dict[str, Any]:
    """Exercise the production selector on positive and conflict controls."""

    positive_detections = 0
    placebo_deviations = 0
    for index in range(12):
        base = {str(value): 10.3 for value in CANDIDATE_MULTIPLIERS}
        positive_rows = []
        placebo_rows = []
        for action in range(2):
            positive = dict(base)
            positive["1.0"] = 10.0
            positive["2.0"] = 9.8 + 0.02 * action
            positive_rows.append(
                {
                    "take_id": f"Positive{index}_T{action + 1}",
                    "supported_frame_count": 20,
                    "mean_CD_UL1_mm_by_multiplier": positive,
                }
            )
            placebo = dict(base)
            placebo["1.0"] = 10.0
            placebo["0.5"] = 9.8 if action == 0 else 10.2
            placebo["4.0"] = 10.2 if action == 0 else 9.8
            placebo_rows.append(
                {
                    "take_id": f"Placebo{index}_T{action + 1}",
                    "supported_frame_count": 20,
                    "mean_CD_UL1_mm_by_multiplier": placebo,
                }
            )
        positive_detections += int(
            select_action_robust_multiplier(positive_rows)["multiplier"] == 2.0
        )
        placebo_deviations += int(
            select_action_robust_multiplier(placebo_rows)["multiplier"]
            != GLOBAL_MULTIPLIER
        )
    return {
        "positive_control_count": 12,
        "positive_detection_count": positive_detections,
        "placebo_control_count": 12,
        "placebo_deviation_count": placebo_deviations,
        "passed": positive_detections == 12 and placebo_deviations == 0,
    }


def build_action_robust_scale_calibration(
    first_source_audit: Mapping[str, Any],
    second_source_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the repeated-action maximin scale artifact."""

    first = _validate_source_audit(
        first_source_audit,
        expected_kind=FIRST_SOURCE_AUDIT_KIND,
        expected_sha256=FIRST_SOURCE_AUDIT_SHA256,
    )
    second = _validate_source_audit(
        second_source_audit,
        expected_kind=SECOND_SOURCE_AUDIT_KIND,
        expected_sha256=SECOND_SOURCE_AUDIT_SHA256,
    )
    first_by_object = {_object_name(str(row["take_id"])): row for row in first}
    second_by_object = {_object_name(str(row["take_id"])): row for row in second}
    _require(
        first_by_object.keys() == second_by_object.keys(),
        "source action object inventories differ",
    )

    objects = {
        object_name: select_action_robust_multiplier(
            (first_by_object[object_name], second_by_object[object_name])
        )
        for object_name in sorted(first_by_object)
    }
    source_action_gains = [
        gain
        for row in objects.values()
        for gain in row["source_relative_improvements"]
    ]
    controls = action_robust_control_summary()
    source_gate = {
        "source_object_count": len(objects),
        "source_action_count": len(source_action_gains),
        "adjusted_object_count": sum(
            row["multiplier"] != GLOBAL_MULTIPLIER for row in objects.values()
        ),
        "source_action_regression_count": sum(
            gain < -1e-12 for gain in source_action_gains
        ),
        "minimum_source_action_relative_improvement": float(
            min(source_action_gains)
        ),
        "mean_source_action_relative_improvement": float(
            np.mean(source_action_gains)
        ),
        "controls_passed": controls["passed"],
    }
    source_gate["passed"] = bool(
        source_gate["adjusted_object_count"] >= 8
        and source_gate["source_action_regression_count"] == 0
        and source_gate["mean_source_action_relative_improvement"] > 0.0
        and controls["passed"]
    )
    _require(source_gate["passed"], "action-robust source gate failed")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": ACTION_ROBUST_SCALE_KIND,
        "calibration_id": ACTION_ROBUST_SCALE_ID,
        "first_source_audit_sha256": FIRST_SOURCE_AUDIT_SHA256,
        "first_source_audit_file_sha256": FIRST_SOURCE_AUDIT_FILE_SHA256,
        "second_source_audit_sha256": SECOND_SOURCE_AUDIT_SHA256,
        "second_source_audit_file_sha256": SECOND_SOURCE_AUDIT_FILE_SHA256,
        "source_cohort_role": "two opened development interactions per object",
        "base_effective_scale": BASE_EFFECTIVE_SCALE,
        "global_multiplier": GLOBAL_MULTIPLIER,
        "candidate_multipliers": list(CANDIDATE_MULTIPLIERS),
        "selection_rule": (
            "per physical object, choose the multiplier maximizing the minimum "
            "relative improvement over the validated global scale across two "
            "source interactions; mean gain, prior distance to one, and smaller "
            "magnitude break exact ties"
        ),
        "objects": objects,
        "source_gate": source_gate,
        "synthetic_controls": controls,
        "future_take_outcomes_opened": False,
        "claim_boundary": (
            "Both source interactions are opened development data. The maximin "
            "mapping has no prospective status until it is frozen and evaluated "
            "on another interaction."
        ),
    }
    payload["calibration_sha256"] = calibration_sha256(payload)
    return payload


def validate_action_robust_scale_calibration(
    payload: Mapping[str, Any],
    *,
    bind_registered_digest: bool = True,
) -> dict[str, Any]:
    """Validate a complete repeated-action scale calibration."""

    _require(payload.get("schema_version") == 1, "calibration schema changed")
    _require(
        payload.get("artifact_kind") == ACTION_ROBUST_SCALE_KIND,
        "calibration kind changed",
    )
    _require(
        payload.get("calibration_id") == ACTION_ROBUST_SCALE_ID,
        "calibration id changed",
    )
    _require(
        payload.get("calibration_sha256") == calibration_sha256(payload),
        "calibration checksum mismatch",
    )
    if bind_registered_digest:
        _require(bool(ACTION_ROBUST_SCALE_SHA256), "registered calibration is unset")
        _require(
            payload.get("calibration_sha256") == ACTION_ROBUST_SCALE_SHA256,
            "registered calibration changed",
        )
    _require(
        payload.get("first_source_audit_sha256") == FIRST_SOURCE_AUDIT_SHA256,
        "first source audit changed",
    )
    _require(
        payload.get("first_source_audit_file_sha256")
        == FIRST_SOURCE_AUDIT_FILE_SHA256,
        "first source audit bytes changed",
    )
    _require(
        payload.get("second_source_audit_sha256") == SECOND_SOURCE_AUDIT_SHA256,
        "second source audit changed",
    )
    _require(
        payload.get("second_source_audit_file_sha256")
        == SECOND_SOURCE_AUDIT_FILE_SHA256,
        "second source audit bytes changed",
    )
    _require(
        float(payload.get("base_effective_scale", -1.0)) == BASE_EFFECTIVE_SCALE,
        "base effective scale changed",
    )
    _require(
        float(payload.get("global_multiplier", -1.0)) == GLOBAL_MULTIPLIER,
        "global multiplier changed",
    )
    _require(
        tuple(float(value) for value in payload.get("candidate_multipliers", ()))
        == CANDIDATE_MULTIPLIERS,
        "candidate multiplier bank changed",
    )
    _require(
        payload.get("future_take_outcomes_opened") is False,
        "calibration followed future target access",
    )
    source_gate = payload.get("source_gate")
    _require(
        isinstance(source_gate, Mapping) and source_gate.get("passed") is True,
        "source gate failed",
    )
    controls = payload.get("synthetic_controls")
    _require(
        isinstance(controls, Mapping) and controls.get("passed") is True,
        "synthetic controls failed",
    )
    objects = payload.get("objects")
    _require(isinstance(objects, Mapping) and len(objects) == 12, "object map changed")
    multipliers: dict[str, float] = {}
    for object_name, raw in objects.items():
        _require(isinstance(raw, Mapping), "object calibration row is invalid")
        row = dict(raw)
        _require(row.get("object_name") == object_name, "object identity changed")
        take_ids = row.get("source_take_ids")
        _require(
            isinstance(take_ids, list)
            and len(take_ids) == 2
            and all(_object_name(value) == object_name for value in take_ids),
            "object source interactions changed",
        )
        multiplier = float(row.get("multiplier", -1.0))
        _require(multiplier in CANDIDATE_MULTIPLIERS, "object multiplier changed")
        _require(
            float(row.get("effective_scale", -1.0))
            == BASE_EFFECTIVE_SCALE * multiplier,
            "object effective scale changed",
        )
        _require(
            float(row.get("minimum_source_relative_improvement", -1.0)) >= -1e-12,
            "object scale regresses on a source interaction",
        )
        multipliers[str(object_name)] = multiplier
    return {
        "passed": True,
        "calibration_sha256": payload["calibration_sha256"],
        "multipliers": multipliers,
    }


def load_action_robust_scale_calibration(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_action_robust_scale_calibration(payload)
    return payload


__all__ = [
    "ACTION_ROBUST_SCALE_FILE_SHA256",
    "ACTION_ROBUST_SCALE_ID",
    "ACTION_ROBUST_SCALE_KIND",
    "ACTION_ROBUST_SCALE_SHA256",
    "BASE_EFFECTIVE_SCALE",
    "CANDIDATE_MULTIPLIERS",
    "GLOBAL_MULTIPLIER",
    "action_robust_control_summary",
    "build_action_robust_scale_calibration",
    "calibration_sha256",
    "load_action_robust_scale_calibration",
    "scale_relative_improvement",
    "select_action_robust_multiplier",
    "validate_action_robust_scale_calibration",
]
