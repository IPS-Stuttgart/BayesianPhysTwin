"""Outcome-blind support accounting for the prospective Deform360 v2 study."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import deform360_bias_aware_prospective_artifacts as artifacts
from .deform360_bias_aware_prospective_protocol import EXPECTED_STRATA
from .deform360_bias_aware_prospective_v2_protocol import (
    EXPECTED_FRESH_CALIBRATION,
    PROTOCOL_ID,
    load_bias_aware_prospective_v2_protocol,
)
from .deform360_bias_aware_prospective_v2_runtime import (
    activate_v2_prediction_runtime,
    prospective_v2_case_records,
)


COHORT_ARTIFACT_KIND = "Deform360BiasAwareProspectiveV2CalibrationCohortSeal"
SUPPORT_ARTIFACT_KIND = "Deform360BiasAwareProspectiveV2CalibrationSupportGate"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"JSON object expected: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _validate_physical_disposition(
    case: Mapping[str, Any],
    *,
    protocol_path: Path,
    backbone_dir: Path,
) -> tuple[str, bool, dict[str, str]]:
    seal_path = backbone_dir / artifacts.BACKBONE_SEAL_FILENAME
    seal = _load_json(seal_path)
    artifacts.validate_prospective_backbone_seal(
        seal,
        protocol_path=protocol_path,
        case_dir=backbone_dir,
    )
    _require(
        all(seal.get(key) == value for key, value in case.items()),
        f"physical case identity changed: {case['case']}",
    )
    manifest_path = backbone_dir / artifacts.PHYSICAL_MANIFEST_FILENAME
    manifest = _load_json(manifest_path)
    _require(
        manifest.get("artifact_kind")
        == "Deform360BiasAwareProspectivePhysicalPrediction"
        and manifest.get("protocol_id") == seal["protocol_id"]
        and manifest.get("protocol_config_sha256") == seal["protocol_config_sha256"]
        and manifest.get("result_sha256")
        == artifacts.canonical_sha256(manifest, digest_key="result_sha256"),
        f"physical manifest is incompatible: {case['case']}",
    )
    _require(
        all(manifest.get(key) == value for key, value in case.items()),
        f"physical manifest identity changed: {case['case']}",
    )
    mode = str(manifest.get("physical_mode", ""))
    _require(
        mode in {"warp_twin", "persistence_fallback"},
        f"unknown physical disposition: {case['case']}",
    )
    admitted = mode == "warp_twin"
    _require(
        manifest.get("physical_admitted") is admitted,
        f"physical admission flag changed: {case['case']}",
    )
    return (
        mode,
        admitted,
        {
            "backbone_seal_file_sha256": artifacts.file_sha256(seal_path),
            "backbone_seal_result_sha256": str(seal["result_sha256"]),
            "physical_manifest_file_sha256": artifacts.file_sha256(manifest_path),
            "physical_manifest_result_sha256": str(manifest["result_sha256"]),
        },
    )


def _prediction_row(
    case: Mapping[str, Any],
    *,
    origin: str,
    protocol_path: Path,
    prediction_dir: Path,
    backbone_dir: Path,
) -> dict[str, Any]:
    prediction_path = prediction_dir / artifacts.PREDICTION_SEAL_FILENAME
    prediction = _load_json(prediction_path)
    artifacts.validate_prospective_prediction_seal(
        prediction,
        protocol_path=protocol_path,
        prediction_dir=prediction_dir,
    )
    _require(
        all(prediction.get(key) == value for key, value in case.items()),
        f"prediction identity changed: {case['case']}",
    )
    mode, admitted, physical = _validate_physical_disposition(
        case,
        protocol_path=protocol_path,
        backbone_dir=backbone_dir,
    )
    return {
        **dict(case),
        "origin": origin,
        "disposition": "prediction",
        "physical_mode": mode,
        "automatic_twin": admitted,
        "eligible_for_accuracy_and_calibration": admitted,
        "prediction_seal_file_sha256": artifacts.file_sha256(prediction_path),
        "prediction_seal_result_sha256": str(prediction["result_sha256"]),
        **physical,
    }


def _quality_failure_row(
    case: Mapping[str, Any],
    *,
    origin: str,
    prediction_dir: Path,
    protocol_id: str,
    protocol_config_sha256: str,
) -> dict[str, Any]:
    path = prediction_dir / artifacts.QUALITY_FAILURE_FILENAME
    failure = _load_json(path)
    _require(
        failure.get("artifact_kind") == artifacts.QUALITY_FAILURE_ARTIFACT_KIND
        and failure.get("protocol_id") == protocol_id
        and failure.get("protocol_config_sha256") == protocol_config_sha256
        and failure.get("result_sha256")
        == artifacts.canonical_sha256(failure, digest_key="result_sha256"),
        f"quality failure is incompatible: {case['case']}",
    )
    _require(
        all(failure.get(key) == value for key, value in case.items())
        and failure.get("replacement_allowed") is False
        and failure.get("information_boundary")
        == {
            "target_data_read": False,
            "outcome_manifest_read": False,
            "failure_recorded_before_future_open": True,
        },
        f"quality failure crossed its boundary: {case['case']}",
    )
    return {
        **dict(case),
        "origin": origin,
        "disposition": "quality_failure",
        "failure_stage": str(failure["stage"]),
        "failure_type": str(failure["error_type"]),
        "physical_mode": None,
        "automatic_twin": False,
        "eligible_for_accuracy_and_calibration": False,
        "quality_failure_file_sha256": artifacts.file_sha256(path),
        "quality_failure_result_sha256": str(failure["result_sha256"]),
    }


def _case_disposition(
    case: Mapping[str, Any],
    *,
    origin: str,
    protocol_path: Path,
    prediction_root: Path,
    backbone_root: Path,
    protocol_id: str,
    protocol_config_sha256: str,
) -> dict[str, Any]:
    prediction_dir = prediction_root / str(case["case"])
    prediction = prediction_dir / artifacts.PREDICTION_SEAL_FILENAME
    failure = prediction_dir / artifacts.QUALITY_FAILURE_FILENAME
    _require(
        prediction.is_file() != failure.is_file(),
        f"case needs exactly one sealed disposition: {case['case']}",
    )
    if prediction.is_file():
        return _prediction_row(
            case,
            origin=origin,
            protocol_path=protocol_path,
            prediction_dir=prediction_dir,
            backbone_dir=backbone_root / str(case["case"]),
        )
    return _quality_failure_row(
        case,
        origin=origin,
        prediction_dir=prediction_dir,
        protocol_id=protocol_id,
        protocol_config_sha256=protocol_config_sha256,
    )


def build_v2_calibration_cohort_seal(
    protocol_path: str | Path,
    *,
    base_protocol_path: str | Path,
    base_prediction_root: str | Path,
    base_backbone_root: str | Path,
    base_cohort_seal_path: str | Path,
    base_support_rejection_path: str | Path,
    fresh_prediction_root: str | Path,
    fresh_backbone_root: str | Path,
    execution_lock_record: Mapping[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Seal inherited and fresh dispositions before any v2 future is opened."""

    protocol_file = Path(protocol_path).resolve()
    protocol = load_bias_aware_prospective_v2_protocol(protocol_file)
    config = protocol["config"]
    base_config = config["base_protocol"]
    base_protocol_file = Path(base_protocol_path).resolve()
    _require(
        artifacts.file_sha256(base_protocol_file) == base_config["file_sha256"],
        "base protocol file changed",
    )
    base_cohort_file = Path(base_cohort_seal_path).resolve()
    base_cohort = _load_json(base_cohort_file)
    _require(
        artifacts.file_sha256(base_cohort_file)
        == base_config["calibration_prediction_cohort_file_sha256"]
        and base_cohort.get("result_sha256")
        == base_config["calibration_prediction_cohort_result_sha256"],
        "base calibration cohort changed",
    )
    base_predictions = Path(base_prediction_root).resolve()
    artifacts.validate_prospective_prediction_cohort_seal(
        base_cohort,
        protocol_path=base_protocol_file,
        role="calibration",
        artifact_root=base_predictions,
    )
    base_rejection_file = Path(base_support_rejection_path).resolve()
    base_rejection = _load_json(base_rejection_file)
    _require(
        artifacts.file_sha256(base_rejection_file)
        == base_config["support_rejection_file_sha256"]
        and base_rejection.get("result_sha256")
        == base_config["support_rejection_result_sha256"],
        "base support rejection changed",
    )
    artifacts.validate_prospective_calibration_support_rejection(
        base_rejection,
        protocol_path=base_protocol_file,
        cohort_seal=base_cohort,
        artifact_root=base_predictions,
    )

    base_rows = {str(row["case"]): row for row in base_cohort["cases"]}
    fresh_cases = {
        f"{object_id}-ep{episode_ids[0]:04d}"
        for object_id, episode_ids in EXPECTED_FRESH_CALIBRATION.items()
    }
    expected = prospective_v2_case_records(protocol_file, role="calibration")
    _require(
        set(base_rows) | fresh_cases == {str(row["case"]) for row in expected},
        "v2 calibration composition changed",
    )
    fresh_predictions = Path(fresh_prediction_root).resolve()
    observed_fresh = {
        path.name for path in fresh_predictions.iterdir() if path.is_dir()
    }
    _require(observed_fresh == fresh_cases, "fresh prediction inventory changed")

    rows: list[dict[str, Any]] = []
    base_backbones = Path(base_backbone_root).resolve()
    fresh_backbones = Path(fresh_backbone_root).resolve()
    for case in expected:
        case_name = str(case["case"])
        if case_name in fresh_cases:
            with activate_v2_prediction_runtime():
                row = _case_disposition(
                    case,
                    origin="fresh_v2",
                    protocol_path=protocol_file,
                    prediction_root=fresh_predictions,
                    backbone_root=fresh_backbones,
                    protocol_id=PROTOCOL_ID,
                    protocol_config_sha256=str(protocol["config_sha256"]),
                )
        else:
            inherited = base_rows[case_name]
            _require(
                all(inherited.get(key) == value for key, value in case.items()),
                f"inherited case identity changed: {case_name}",
            )
            row = _case_disposition(
                case,
                origin="inherited_v1",
                protocol_path=base_protocol_file,
                prediction_root=base_predictions,
                backbone_root=base_backbones,
                protocol_id=str(base_config["protocol_id"]),
                protocol_config_sha256=str(base_config["config_sha256"]),
            )
            observed_result = row.get(
                "prediction_seal_result_sha256",
                row.get("quality_failure_result_sha256"),
            )
            _require(
                inherited["disposition"] == row["disposition"]
                and inherited["artifact_result_sha256"] == observed_result,
                f"inherited disposition changed: {case_name}",
            )
        rows.append(row)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": COHORT_ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": protocol["config_sha256"],
        "role": "calibration",
        "expected_case_count": len(expected),
        "prediction_count": sum(row["disposition"] == "prediction" for row in rows),
        "quality_failure_count": sum(
            row["disposition"] == "quality_failure" for row in rows
        ),
        "automatic_twin_count": sum(row["automatic_twin"] for row in rows),
        "replacement_count": 0,
        "cases": rows,
        "base_protocol": {
            "file_sha256": artifacts.file_sha256(base_protocol_file),
            "calibration_cohort_file_sha256": artifacts.file_sha256(base_cohort_file),
            "calibration_cohort_result_sha256": base_cohort["result_sha256"],
            "support_rejection_file_sha256": artifacts.file_sha256(base_rejection_file),
            "support_rejection_result_sha256": base_rejection["result_sha256"],
        },
        "execution_lock": dict(execution_lock_record),
        "complete": True,
        "information_boundary": {
            "base_dispositions_inherited_without_relabeling": True,
            "fresh_predictions_or_failures_sealed_before_future_open": True,
            "calibration_future_read": False,
            "calibration_outcome_read": False,
            "target_media_read": False,
            "target_future_read": False,
            "replacement_allowed": False,
        },
    }
    payload["result_sha256"] = artifacts.canonical_sha256(
        payload, digest_key="result_sha256"
    )
    destination = Path(output_path).resolve()
    _require(not destination.exists(), "v2 calibration cohort is already sealed")
    validate_v2_calibration_cohort_seal(payload, protocol_path=protocol_file)
    _write_json(destination, payload)
    return payload


def validate_v2_calibration_cohort_seal(
    seal: Mapping[str, Any], *, protocol_path: str | Path
) -> None:
    """Validate the self-contained v2 calibration disposition inventory."""

    protocol = load_bias_aware_prospective_v2_protocol(protocol_path)
    expected = prospective_v2_case_records(protocol_path, role="calibration")
    fresh_cases = {
        f"{object_id}-ep{episode_ids[0]:04d}"
        for object_id, episode_ids in EXPECTED_FRESH_CALIBRATION.items()
    }
    rows = seal.get("cases")
    _require(
        seal.get("artifact_kind") == COHORT_ARTIFACT_KIND
        and seal.get("protocol_id") == PROTOCOL_ID
        and seal.get("protocol_config_sha256") == protocol["config_sha256"]
        and seal.get("role") == "calibration"
        and seal.get("result_sha256")
        == artifacts.canonical_sha256(seal, digest_key="result_sha256"),
        "v2 calibration cohort contract changed",
    )
    _require(
        isinstance(rows, Sequence)
        and len(rows) == len(expected)
        and seal.get("expected_case_count") == len(expected)
        and seal.get("complete") is True
        and seal.get("replacement_count") == 0,
        "v2 calibration cohort is incomplete",
    )
    for row, case in zip(rows, expected, strict=True):
        expected_origin = "fresh_v2" if case["case"] in fresh_cases else "inherited_v1"
        disposition = row.get("disposition") if isinstance(row, Mapping) else None
        physical_mode = row.get("physical_mode") if isinstance(row, Mapping) else None
        _require(
            isinstance(row, Mapping)
            and all(row.get(key) == value for key, value in case.items())
            and row.get("origin") == expected_origin
            and disposition in {"prediction", "quality_failure"}
            and (
                (
                    disposition == "prediction"
                    and physical_mode
                    in {
                        "warp_twin",
                        "persistence_fallback",
                    }
                )
                or (disposition == "quality_failure" and physical_mode is None)
            )
            and row.get("automatic_twin") is (physical_mode == "warp_twin")
            and row.get("eligible_for_accuracy_and_calibration")
            is row.get("automatic_twin"),
            f"v2 calibration row changed: {case['case']}",
        )
    _require(
        seal.get("prediction_count")
        == sum(row["disposition"] == "prediction" for row in rows)
        and seal.get("quality_failure_count")
        == sum(row["disposition"] == "quality_failure" for row in rows)
        and seal.get("automatic_twin_count")
        == sum(row["automatic_twin"] for row in rows),
        "v2 calibration cohort counts changed",
    )
    _require(
        seal.get("information_boundary")
        == {
            "base_dispositions_inherited_without_relabeling": True,
            "fresh_predictions_or_failures_sealed_before_future_open": True,
            "calibration_future_read": False,
            "calibration_outcome_read": False,
            "target_media_read": False,
            "target_future_read": False,
            "replacement_allowed": False,
        },
        "v2 calibration cohort crossed its boundary",
    )


def calibration_support_summary(
    rows: Sequence[Mapping[str, Any]], gate: Mapping[str, Any], *, source_groups: int
) -> dict[str, Any]:
    """Compute the locked support arithmetic without reading outcomes."""

    automatic = [row for row in rows if row.get("automatic_twin") is True]
    automatic_objects = {str(row["object_id"]) for row in automatic}
    by_stratum = {
        stratum: len(
            {str(row["object_id"]) for row in automatic if row["stratum"] == stratum}
        )
        for stratum in EXPECTED_STRATA
    }
    fresh_filament = len(
        {
            str(row["object_id"])
            for row in automatic
            if row["origin"] == "fresh_v2" and row["stratum"] == "filament"
        }
    )
    new_groups = len(automatic_objects)
    combined_groups = int(source_groups) + new_groups
    rank = min(combined_groups, int(np.ceil((combined_groups + 1) * 0.90)))
    coverage = rank / (combined_groups + 1)
    checks = {
        "minimum_automatic_twin_objects": new_groups
        >= int(gate["minimum_automatic_twin_objects"]),
        "minimum_automatic_twin_objects_per_stratum": all(
            value >= int(gate["minimum_automatic_twin_objects_per_stratum"])
            for value in by_stratum.values()
        ),
        "fresh_filament_automatic_twins_required": fresh_filament
        >= int(gate["fresh_filament_automatic_twins_required"]),
        "minimum_new_eligible_object_groups": new_groups
        >= int(gate["minimum_new_eligible_object_groups"]),
        "minimum_combined_eligible_object_groups": combined_groups
        >= int(gate["minimum_combined_eligible_object_groups"]),
        "required_finite_sample_coverage": coverage
        >= float(gate["required_finite_sample_coverage"]),
    }
    return {
        "automatic_twin_object_count": new_groups,
        "automatic_twin_object_count_by_stratum": by_stratum,
        "fresh_filament_automatic_twin_count": fresh_filament,
        "new_eligible_object_group_count": new_groups,
        "source_group_count": int(source_groups),
        "combined_eligible_object_group_count": combined_groups,
        "finite_sample_rank": rank,
        "finite_sample_coverage": coverage,
        "support_gates": checks,
        "failed_support_gates": sorted(
            name for name, passed in checks.items() if not passed
        ),
        "support_passed": all(checks.values()),
    }


def build_v2_calibration_support_gate(
    protocol_path: str | Path,
    *,
    cohort_seal_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Evaluate only target-free support and authorize no target access."""

    protocol = load_bias_aware_prospective_v2_protocol(protocol_path)
    cohort_path = Path(cohort_seal_path).resolve()
    cohort = _load_json(cohort_path)
    validate_v2_calibration_cohort_seal(cohort, protocol_path=protocol_path)
    summary = calibration_support_summary(
        cohort["cases"],
        protocol["config"]["calibration_support_gate"],
        source_groups=int(protocol["config"]["method"]["source_group_count"]),
    )
    passed = bool(summary["support_passed"])
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": SUPPORT_ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": protocol["config_sha256"],
        "decision_stage": "pre-outcome-support",
        "calibration_cohort_file_sha256": artifacts.file_sha256(cohort_path),
        "calibration_cohort_result_sha256": cohort["result_sha256"],
        **summary,
        "calibration_future_access_authorized": passed,
        "target_access_authorized": False,
        "next_action": (
            "open calibration futures under a separate locked evaluator"
            if passed
            else protocol["config"]["calibration_support_gate"]["failed_gate_action"]
        ),
        "information_boundary": {
            "method_family_changed": False,
            "candidate_threshold_changed": False,
            "observation_model_changed": False,
            "calibration_future_read": False,
            "calibration_outcome_read": False,
            "target_media_read": False,
            "target_future_read": False,
            "support_decision_uses_target_free_artifacts_only": True,
        },
        "claim_boundary": (
            "target-free calibration-support decision only; no accuracy, "
            "calibration, or state-of-the-art claim"
        ),
    }
    payload["result_sha256"] = artifacts.canonical_sha256(
        payload, digest_key="result_sha256"
    )
    destination = Path(output_path).resolve()
    _require(not destination.exists(), "v2 calibration support is already decided")
    validate_v2_calibration_support_gate(
        payload,
        protocol_path=protocol_path,
        cohort_seal_path=cohort_path,
    )
    _write_json(destination, payload)
    return payload


def validate_v2_calibration_support_gate(
    support: Mapping[str, Any],
    *,
    protocol_path: str | Path,
    cohort_seal_path: str | Path,
) -> None:
    """Recompute the support-only decision from the sealed cohort."""

    protocol = load_bias_aware_prospective_v2_protocol(protocol_path)
    cohort_path = Path(cohort_seal_path).resolve()
    cohort = _load_json(cohort_path)
    validate_v2_calibration_cohort_seal(cohort, protocol_path=protocol_path)
    summary = calibration_support_summary(
        cohort["cases"],
        protocol["config"]["calibration_support_gate"],
        source_groups=int(protocol["config"]["method"]["source_group_count"]),
    )
    _require(
        support.get("artifact_kind") == SUPPORT_ARTIFACT_KIND
        and support.get("protocol_id") == PROTOCOL_ID
        and support.get("protocol_config_sha256") == protocol["config_sha256"]
        and support.get("decision_stage") == "pre-outcome-support"
        and support.get("result_sha256")
        == artifacts.canonical_sha256(support, digest_key="result_sha256")
        and support.get("calibration_cohort_file_sha256")
        == artifacts.file_sha256(cohort_path)
        and support.get("calibration_cohort_result_sha256") == cohort["result_sha256"],
        "v2 support artifact contract changed",
    )
    _require(
        all(support.get(key) == value for key, value in summary.items()),
        "v2 support arithmetic changed",
    )
    passed = bool(summary["support_passed"])
    _require(
        support.get("calibration_future_access_authorized") is passed
        and support.get("target_access_authorized") is False
        and support.get("information_boundary")
        == {
            "method_family_changed": False,
            "candidate_threshold_changed": False,
            "observation_model_changed": False,
            "calibration_future_read": False,
            "calibration_outcome_read": False,
            "target_media_read": False,
            "target_future_read": False,
            "support_decision_uses_target_free_artifacts_only": True,
        },
        "v2 support authorization changed",
    )


__all__ = [
    "COHORT_ARTIFACT_KIND",
    "SUPPORT_ARTIFACT_KIND",
    "build_v2_calibration_cohort_seal",
    "build_v2_calibration_support_gate",
    "calibration_support_summary",
    "validate_v2_calibration_cohort_seal",
    "validate_v2_calibration_support_gate",
]
