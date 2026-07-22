"""Calibration-only outcome authorization and scoring for prospective v2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

import numpy as np

from . import deform360_bias_aware_prospective_artifacts as artifacts
from . import deform360_bias_aware_prospective_evaluation as evaluation
from .deform360_bias_aware_prospective_evaluation import (
    FROZEN_SOURCE_GROUP_WORST_REGRET_M,
)
from .bias_aware_belief import fit_source_group_regret_bound
from .deform360_bias_aware_prospective_protocol import (
    PROTOCOL_ID as V1_PROTOCOL_ID,
    SOURCE_LOCK_GROUP_COUNT,
    SOURCE_LOCK_SHA256,
    SOURCE_MINIMUM_IMPROVEMENT_M,
)
from .deform360_bias_aware_prospective_v2_protocol import (
    PROTOCOL_ID,
    load_bias_aware_prospective_v2_protocol,
)
from .deform360_bias_aware_prospective_v2_runtime import (
    activate_v2_prediction_runtime,
    load_bias_aware_prospective_v2_execution_protocol,
    prospective_v2_case_record,
)
from .deform360_bias_aware_prospective_v2_support import (
    validate_v2_calibration_cohort_seal,
    validate_v2_calibration_support_gate,
)


AUTHORIZATION_ARTIFACT_KIND = (
    "Deform360BiasAwareProspectiveV2CalibrationOutcomeAuthorization"
)
CALIBRATION_GATE_ARTIFACT_KIND = (
    "Deform360BiasAwareProspectiveV2CalibrationAccuracyGate"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"JSON object expected: {path}")
    return payload


def validate_v2_calibration_access(
    protocol_path: str | Path,
    *,
    cohort_seal_path: str | Path,
    support_gate_path: str | Path,
    object_id: str,
    episode_id: int,
    expected_origin: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Authorize one automatic-twin calibration case, never a target case."""

    cohort_path = Path(cohort_seal_path).resolve()
    support_path = Path(support_gate_path).resolve()
    cohort = _load_json(cohort_path)
    support = _load_json(support_path)
    validate_v2_calibration_cohort_seal(cohort, protocol_path=protocol_path)
    validate_v2_calibration_support_gate(
        support,
        protocol_path=protocol_path,
        cohort_seal_path=cohort_path,
    )
    _require(
        support.get("support_passed") is True
        and support.get("calibration_future_access_authorized") is True
        and support.get("target_access_authorized") is False,
        "v2 support gate forbids calibration outcome access",
    )
    record = prospective_v2_case_record(
        protocol_path,
        object_id=object_id,
        episode_id=episode_id,
    )
    _require(record["role"] == "calibration", "target case is not authorized")
    matches = [row for row in cohort["cases"] if row["case"] == record["case"]]
    _require(len(matches) == 1, "calibration case is missing from cohort seal")
    disposition = dict(matches[0])
    _require(
        disposition.get("disposition") == "prediction"
        and disposition.get("automatic_twin") is True
        and disposition.get("eligible_for_accuracy_and_calibration") is True,
        "calibration case has no eligible automatic twin",
    )
    if expected_origin is not None:
        _require(
            disposition.get("origin") == expected_origin,
            "calibration origin changed",
        )
    return record, disposition, support


def fresh_v2_prediction_authorizer(
    *,
    protocol_path: str | Path,
    cohort_seal_path: str | Path,
    support_gate_path: str | Path,
    prediction_root: str | Path,
):
    """Build the closure expected by the frozen future/outcome stages."""

    expected_protocol = Path(protocol_path).resolve()
    expected_root = Path(prediction_root).resolve()
    expected_cohort = _load_json(cohort_seal_path)

    def authorize(
        cohort_seal: Mapping[str, Any],
        *,
        protocol_path: str | Path,
        role: str,
        artifact_root: str | Path,
        object_id: str,
        episode_id: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        _require(role == "calibration", "fresh adapter cannot authorize targets")
        _require(
            Path(protocol_path).resolve() == expected_protocol
            and Path(artifact_root).resolve() == expected_root
            and cohort_seal.get("result_sha256")
            == expected_cohort.get("result_sha256"),
            "fresh adapter inputs changed",
        )
        record, disposition, _ = validate_v2_calibration_access(
            expected_protocol,
            cohort_seal_path=cohort_seal_path,
            support_gate_path=support_gate_path,
            object_id=object_id,
            episode_id=episode_id,
            expected_origin="fresh_v2",
        )
        prediction_dir = expected_root / str(record["case"])
        prediction_path = prediction_dir / artifacts.PREDICTION_SEAL_FILENAME
        prediction = _load_json(prediction_path)
        artifacts.validate_prospective_prediction_seal(
            prediction,
            protocol_path=expected_protocol,
            prediction_dir=prediction_dir,
        )
        _require(
            prediction["result_sha256"] == disposition["prediction_seal_result_sha256"],
            "fresh prediction differs from v2 cohort seal",
        )
        return record, prediction

    return authorize


def patch_fresh_v2_calibration_stage(
    module: ModuleType,
    *,
    protocol_path: str | Path,
    cohort_seal_path: str | Path,
    support_gate_path: str | Path,
    prediction_root: str | Path,
) -> None:
    """Patch aliases imported by one frozen calibration-only stage."""

    module.PROTOCOL_ID = PROTOCOL_ID
    module.load_bias_aware_prospective_protocol = (
        load_bias_aware_prospective_v2_execution_protocol
    )
    module.authorize_prospective_outcome_case = fresh_v2_prediction_authorizer(
        protocol_path=protocol_path,
        cohort_seal_path=cohort_seal_path,
        support_gate_path=support_gate_path,
        prediction_root=prediction_root,
    )


@contextmanager
def activate_fresh_v2_evaluation_runtime(
    *,
    protocol_path: str | Path,
    cohort_seal_path: str | Path,
    support_gate_path: str | Path,
    prediction_root: str | Path,
) -> Iterator[None]:
    """Bind the frozen scorer to fresh v2 identities for one process."""

    changes = {
        "PROTOCOL_ID": evaluation.PROTOCOL_ID,
        "load_bias_aware_prospective_protocol": (
            evaluation.load_bias_aware_prospective_protocol
        ),
        "prospective_case_record": evaluation.prospective_case_record,
        "prospective_case_records": evaluation.prospective_case_records,
        "authorize_prospective_outcome_case": (
            evaluation.authorize_prospective_outcome_case
        ),
    }
    evaluation.PROTOCOL_ID = PROTOCOL_ID
    evaluation.load_bias_aware_prospective_protocol = (
        load_bias_aware_prospective_v2_execution_protocol
    )
    evaluation.prospective_case_record = prospective_v2_case_record
    evaluation.authorize_prospective_outcome_case = fresh_v2_prediction_authorizer(
        protocol_path=protocol_path,
        cohort_seal_path=cohort_seal_path,
        support_gate_path=support_gate_path,
        prediction_root=prediction_root,
    )
    try:
        with activate_v2_prediction_runtime():
            yield
    finally:
        for name, value in changes.items():
            setattr(evaluation, name, value)


def build_v2_calibration_authorization_sidecar(
    protocol_path: str | Path,
    *,
    cohort_seal_path: str | Path,
    support_gate_path: str | Path,
    object_id: str,
    episode_id: int,
    origin: str,
    stage: str,
    stage_artifact_path: str | Path,
) -> dict[str, Any]:
    """Bind a v1 or v2 stage artifact to the passed v2 support decision."""

    record, disposition, support = validate_v2_calibration_access(
        protocol_path,
        cohort_seal_path=cohort_seal_path,
        support_gate_path=support_gate_path,
        object_id=object_id,
        episode_id=episode_id,
        expected_origin=origin,
    )
    _require(
        stage in {"authorized-future", "authorized-outcome", "evaluation"},
        "unknown calibration stage",
    )
    artifact_path = Path(stage_artifact_path).resolve()
    artifact = _load_json(artifact_path)
    expected_protocol = V1_PROTOCOL_ID if origin == "inherited_v1" else PROTOCOL_ID
    _require(
        artifact.get("protocol_id") == expected_protocol
        and artifact.get("result_sha256")
        == artifacts.canonical_sha256(artifact, digest_key="result_sha256")
        and all(artifact.get(key) == value for key, value in record.items()),
        "calibration stage artifact changed",
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": AUTHORIZATION_ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": load_bias_aware_prospective_v2_protocol(
            protocol_path
        )["config_sha256"],
        **record,
        "origin": origin,
        "stage": stage,
        "source_artifact_protocol_id": expected_protocol,
        "source_artifact_file_sha256": artifacts.file_sha256(artifact_path),
        "source_artifact_result_sha256": artifact["result_sha256"],
        "prediction_disposition_result_sha256": disposition[
            "prediction_seal_result_sha256"
        ],
        "calibration_cohort_result_sha256": _load_json(cohort_seal_path)[
            "result_sha256"
        ],
        "calibration_support_gate_result_sha256": support["result_sha256"],
        "target_access_authorized": False,
        "information_boundary": {
            "calibration_support_verified_before_stage": True,
            "target_media_read": False,
            "target_future_read": False,
            "method_or_gate_changed": False,
        },
    }
    payload["result_sha256"] = artifacts.canonical_sha256(
        payload, digest_key="result_sha256"
    )
    return payload


def _validate_report(
    report: Mapping[str, Any],
    *,
    expected_protocol_id: str,
    expected_config_sha256: str,
) -> None:
    _require(
        report.get("artifact_kind") == evaluation.CASE_EVALUATION_ARTIFACT_KIND
        and report.get("protocol_id") == expected_protocol_id
        and report.get("protocol_config_sha256") == expected_config_sha256
        and report.get("role") == "calibration"
        and report.get("result_sha256")
        == artifacts.canonical_sha256(report, digest_key="result_sha256"),
        "calibration report changed",
    )


def fit_v2_calibration_accuracy_gate(
    reports: Sequence[Mapping[str, Any]],
    *,
    protocol_path: str | Path,
    base_protocol_config_sha256: str,
    cohort_seal_path: str | Path,
    support_gate_path: str | Path,
    source_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the unchanged accuracy gates to all eight automatic twins."""

    protocol = load_bias_aware_prospective_v2_protocol(protocol_path)
    cohort = _load_json(cohort_seal_path)
    support = _load_json(support_gate_path)
    validate_v2_calibration_cohort_seal(cohort, protocol_path=protocol_path)
    validate_v2_calibration_support_gate(
        support,
        protocol_path=protocol_path,
        cohort_seal_path=cohort_seal_path,
    )
    _require(support["support_passed"] is True, "support gate did not pass")
    expected_rows = [row for row in cohort["cases"] if row["automatic_twin"]]
    expected_by_case = {str(row["case"]): row for row in expected_rows}
    _require(
        len(reports) == len(expected_by_case)
        and {str(report["case"]) for report in reports} == set(expected_by_case),
        "calibration evaluations are incomplete",
    )
    for report in reports:
        disposition = expected_by_case[str(report["case"])]
        _validate_report(
            report,
            expected_protocol_id=(
                V1_PROTOCOL_ID
                if disposition["origin"] == "inherited_v1"
                else PROTOCOL_ID
            ),
            expected_config_sha256=(
                base_protocol_config_sha256
                if disposition["origin"] == "inherited_v1"
                else str(protocol["config_sha256"])
            ),
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
        "source scores changed",
    )
    rows = evaluation._object_rows(reports)
    eligible = [row for row in rows if row["eligible_update_count"] > 0]
    combined_scores = {
        **{str(key): float(value) for key, value in source_scores.items()},
        **{
            str(row["object_id"]): float(row["worst_eligible_interval_regret_m"])
            for row in eligible
        },
    }
    _require(
        len(combined_scores) == len(source_scores) + len(eligible),
        "calibration and source groups overlap",
    )
    bound = fit_source_group_regret_bound(
        np.asarray(list(combined_scores.values()), dtype=np.float64),
        list(combined_scores),
        nominal_coverage=0.90,
        within_group_coverage=1.0,
        minimum_improvement_m=SOURCE_MINIMUM_IMPROVEMENT_M,
    )
    gate = protocol["config"]["calibration_accuracy_gate"]
    mean_regret = {
        metric: float(np.mean([row["regret_m"][metric] for row in rows]))
        for metric in evaluation.PRIMARY_METRICS
    }
    harmful = [
        row["object_id"]
        for row in eligible
        if any(row["regret_m"][metric] > 0.0 for metric in evaluation.PRIMARY_METRICS)
    ]
    gates = {
        "support_gate_passed": support["support_passed"] is True,
        "minimum_new_eligible_object_groups": len(eligible)
        >= int(
            protocol["config"]["calibration_support_gate"][
                "minimum_new_eligible_object_groups"
            ]
        ),
        "minimum_combined_eligible_object_groups": len(combined_scores)
        >= int(
            protocol["config"]["calibration_support_gate"][
                "minimum_combined_eligible_object_groups"
            ]
        ),
        "required_finite_sample_coverage": bound.finite_sample_coverage
        >= float(
            protocol["config"]["calibration_support_gate"][
                "required_finite_sample_coverage"
            ]
        ),
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
    }
    passed = all(gates.values())
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": CALIBRATION_GATE_ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": protocol["config_sha256"],
        "calibration_prediction_cohort_result_sha256": cohort["result_sha256"],
        "calibration_support_gate_result_sha256": support["result_sha256"],
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
        "object_balanced_regret_m": mean_regret,
        "accepted_harmful_objects": harmful,
        "object_results": rows,
        "calibration_evaluation_result_sha256": {
            str(report["case"]): str(report["result_sha256"]) for report in reports
        },
        "gates": gates,
        "calibration_gate_passed": passed,
        "target_access_authorized": passed,
        "failed_gate_action": (
            None
            if passed
            else "publish calibration failure and keep every target sealed"
        ),
        "information_boundary": {
            "method_family_changed": False,
            "candidate_threshold_changed": False,
            "observation_model_changed": False,
            "calibration_futures_opened_after_support_gate": True,
            "target_object_media_read": False,
            "target_future_read": False,
        },
        "claim_boundary": (
            "fresh calibration accuracy and non-regression gate only; not an "
            "official Deform360 or state-of-the-art result"
        ),
    }
    payload["result_sha256"] = artifacts.canonical_sha256(
        payload, digest_key="result_sha256"
    )
    return payload


__all__ = [
    "AUTHORIZATION_ARTIFACT_KIND",
    "CALIBRATION_GATE_ARTIFACT_KIND",
    "activate_fresh_v2_evaluation_runtime",
    "build_v2_calibration_authorization_sidecar",
    "fit_v2_calibration_accuracy_gate",
    "fresh_v2_prediction_authorizer",
    "patch_fresh_v2_calibration_stage",
    "validate_v2_calibration_access",
]
