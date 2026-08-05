"""Source-calibrated shrinkage magnitudes for repeated PokeFlex interactions."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

INSTANCE_SCALE_CALIBRATION_KIND = "PokeFlexInstanceScaleCalibration"
INSTANCE_SCALE_CALIBRATION_ID = "pokeflex-instance-scale-calibration-v2"
INSTANCE_SCALE_CALIBRATION_SHA256 = (
    "74c2f5fe6b57215fdebedd18cc31cb1b4bca010aac905b1c91f185fb34b10390"
)
INSTANCE_SCALE_CALIBRATION_FILE_SHA256 = (
    "bfde9f3572b694d4dffe008b889d45dccea888162886a307fd3b96cfd6b475f3"
)
SOURCE_SCALE_AUDIT_SHA256 = (
    "78179996296b5ed47692e3ee716308c4525deeb71ce2881442331b5643b4bf94"
)
SOURCE_SCALE_AUDIT_FILE_SHA256 = (
    "114fbf7c3437311f625c9c022742a2c707d6db1e61ca1548b0d8f2500f83d494"
)
BASE_EFFECTIVE_SCALE = 0.125
DEFAULT_MULTIPLIER = 1.0
CANDIDATE_MULTIPLIERS = (0.5, 1.0, 1.5, 2.0)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object_name(take_id: str) -> str:
    object_name, separator, take_number = str(take_id).rpartition("_T")
    _require(bool(separator) and take_number.isdigit(), "invalid source take id")
    return object_name


def calibration_sha256(payload: Mapping[str, Any]) -> str:
    """Return the canonical digest of an instance-scale calibration."""

    canonical = dict(payload)
    canonical.pop("calibration_sha256", None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def select_source_multiplier(
    source_row: Mapping[str, Any],
    *,
    candidate_multipliers: Sequence[float] = CANDIDATE_MULTIPLIERS,
) -> tuple[float, str]:
    """Select one bounded scale using only a completed source interaction."""

    bank = tuple(float(value) for value in candidate_multipliers)
    _require(bank == CANDIDATE_MULTIPLIERS, "instance scale bank changed")
    _require(
        int(source_row.get("supported_frame_count", -1)) >= 0,
        "source support count is invalid",
    )
    if int(source_row["supported_frame_count"]) == 0:
        return DEFAULT_MULTIPLIER, "no-supported-source-update"

    means = source_row.get("mean_CD_UL1_mm_by_multiplier")
    _require(isinstance(means, Mapping), "source scale scores are missing")
    values = {multiplier: float(means[str(multiplier)]) for multiplier in bank}
    _require(
        all(np.isfinite(value) and value > 0.0 for value in values.values()),
        "source scale score is invalid",
    )
    # The log-distance tie break is a discrete prior centered on the globally
    # validated multiplier. It matters only when source scores are exactly tied.
    selected = min(
        bank,
        key=lambda multiplier: (
            values[multiplier],
            abs(math.log(multiplier / DEFAULT_MULTIPLIER)),
            multiplier,
        ),
    )
    return selected, "source-map"


def build_instance_scale_calibration(
    source_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a typed empirical-Bayes calibration from the opened source cohort."""

    _require(
        source_audit.get("artifact_kind")
        == "PokeFlexFresh12PostopenScaleHeadroomAudit",
        "source scale artifact changed",
    )
    _require(
        source_audit.get("audit_sha256") == SOURCE_SCALE_AUDIT_SHA256,
        "source scale audit digest changed",
    )
    _require(
        source_audit.get("status") == "post-open diagnostic; not prospective evidence",
        "source audit boundary changed",
    )
    rows = source_audit.get("takes")
    _require(isinstance(rows, list) and len(rows) == 12, "source cohort changed")

    objects: dict[str, dict[str, Any]] = {}
    for raw in rows:
        _require(isinstance(raw, Mapping), "source take row is invalid")
        row = dict(raw)
        take_id = str(row.get("take_id", ""))
        object_name = _object_name(take_id)
        _require(object_name not in objects, "source object repeats")
        multiplier, reason = select_source_multiplier(row)
        means = row["mean_CD_UL1_mm_by_multiplier"]
        global_score = float(means[str(DEFAULT_MULTIPLIER)])
        selected_score = float(means[str(multiplier)])
        objects[object_name] = {
            "source_take_id": take_id,
            "source_scored_frame_count": int(row["scored_frame_count"]),
            "source_supported_frame_count": int(row["supported_frame_count"]),
            "multiplier": multiplier,
            "effective_scale": BASE_EFFECTIVE_SCALE * multiplier,
            "selection_reason": reason,
            "source_global_scale_CD_UL1_mm": global_score,
            "source_selected_scale_CD_UL1_mm": selected_score,
            "source_selected_minus_global_CD_UL1_mm": selected_score - global_score,
        }

    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": INSTANCE_SCALE_CALIBRATION_KIND,
        "calibration_id": INSTANCE_SCALE_CALIBRATION_ID,
        "source_scale_audit_sha256": SOURCE_SCALE_AUDIT_SHA256,
        "source_scale_audit_file_sha256": SOURCE_SCALE_AUDIT_FILE_SHA256,
        "source_cohort_role": "opened development interactions only",
        "base_effective_scale": BASE_EFFECTIVE_SCALE,
        "default_multiplier": DEFAULT_MULTIPLIER,
        "candidate_multipliers": list(CANDIDATE_MULTIPLIERS),
        "selection_rule": (
            "per physical object, minimize source-action CD_UL1 over the frozen "
            "capped multiplier bank; exact ties prefer log-distance to multiplier "
            "one; no supported source update uses multiplier one"
        ),
        "objects": {name: objects[name] for name in sorted(objects)},
        "future_take_outcomes_opened": False,
        "claim_boundary": (
            "This artifact calibrates one scalar from an opened interaction of "
            "each physical object. It is method-development evidence only until "
            "the frozen mapping transfers to untouched interactions."
        ),
    }
    payload["calibration_sha256"] = calibration_sha256(payload)
    validate_instance_scale_calibration(payload)
    return payload


def validate_instance_scale_calibration(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the complete source calibration and return its scale mapping."""

    _require(payload.get("schema_version") == 1, "calibration schema changed")
    _require(
        payload.get("artifact_kind") == INSTANCE_SCALE_CALIBRATION_KIND,
        "calibration kind changed",
    )
    _require(
        payload.get("calibration_id") == INSTANCE_SCALE_CALIBRATION_ID,
        "calibration id changed",
    )
    _require(
        payload.get("calibration_sha256") == calibration_sha256(payload),
        "calibration checksum mismatch",
    )
    _require(
        payload.get("source_scale_audit_sha256") == SOURCE_SCALE_AUDIT_SHA256,
        "calibration source changed",
    )
    _require(
        payload.get("source_scale_audit_file_sha256") == SOURCE_SCALE_AUDIT_FILE_SHA256,
        "calibration source bytes changed",
    )
    _require(
        float(payload.get("base_effective_scale", -1.0)) == BASE_EFFECTIVE_SCALE,
        "base effective scale changed",
    )
    _require(
        float(payload.get("default_multiplier", -1.0)) == DEFAULT_MULTIPLIER,
        "default multiplier changed",
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
    objects = payload.get("objects")
    _require(isinstance(objects, Mapping) and len(objects) == 12, "object map changed")
    scales: dict[str, float] = {}
    for object_name, raw in objects.items():
        _require(isinstance(raw, Mapping), "object calibration row is invalid")
        row = dict(raw)
        _require(
            _object_name(str(row.get("source_take_id", ""))) == object_name,
            "source object identity changed",
        )
        multiplier = float(row.get("multiplier", -1.0))
        _require(multiplier in CANDIDATE_MULTIPLIERS, "object multiplier changed")
        _require(
            float(row.get("effective_scale", -1.0))
            == BASE_EFFECTIVE_SCALE * multiplier,
            "object effective scale changed",
        )
        support = int(row.get("source_supported_frame_count", -1))
        _require(support >= 0, "object support count changed")
        if support == 0:
            _require(
                multiplier == DEFAULT_MULTIPLIER, "unsupported source changed scale"
            )
            _require(
                row.get("selection_reason") == "no-supported-source-update",
                "unsupported source reason changed",
            )
        else:
            _require(
                row.get("selection_reason") == "source-map", "source reason changed"
            )
            _require(
                float(row.get("source_selected_minus_global_CD_UL1_mm", 1.0)) <= 0.0,
                "selected source scale is worse than the global scale",
            )
        scales[str(object_name)] = multiplier
    return {
        "passed": True,
        "calibration_sha256": payload["calibration_sha256"],
        "multipliers": scales,
    }


def load_instance_scale_calibration(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_instance_scale_calibration(payload)
    return payload


__all__ = [
    "BASE_EFFECTIVE_SCALE",
    "CANDIDATE_MULTIPLIERS",
    "DEFAULT_MULTIPLIER",
    "INSTANCE_SCALE_CALIBRATION_ID",
    "INSTANCE_SCALE_CALIBRATION_FILE_SHA256",
    "INSTANCE_SCALE_CALIBRATION_KIND",
    "INSTANCE_SCALE_CALIBRATION_SHA256",
    "build_instance_scale_calibration",
    "calibration_sha256",
    "load_instance_scale_calibration",
    "select_source_multiplier",
    "validate_instance_scale_calibration",
]
