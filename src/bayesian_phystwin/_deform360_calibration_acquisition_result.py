# ruff: noqa: F403, F405
"""Internal implementation slice for Deform360 calibration acquisition."""

from __future__ import annotations

from ._deform360_calibration_acquisition_common import *

def _validated_cases(
    plan: Deform360CalibrationAcquisitionPlanV1,
    cases: Sequence[Deform360CalibrationAcquisitionCaseV1],
) -> tuple[Deform360CalibrationAcquisitionCaseV1, ...]:
    if isinstance(cases, (str, bytes)):
        raise ValueError("cases must be a sequence")
    values = tuple(cases)
    if any(
        not isinstance(case, Deform360CalibrationAcquisitionCaseV1)
        for case in values
    ):
        raise ValueError("cases must contain acquisition case records")
    values = tuple(sorted(values, key=lambda case: (case.stratum, case.object_id)))
    expected = {
        (unit.object_id, unit.episode_id, unit.stratum)
        for unit in plan.calibration_units
    }
    observed = {(case.object_id, case.episode_id, case.stratum) for case in values}
    _require(len(observed) == len(values), "acquisition cases repeat a unit")
    _require(observed == expected, "acquisition cases do not cover the locked cohort")
    if any(case.plan_id != plan.plan_id for case in values):
        raise ValueError("acquisition case names a different plan")
    forbidden = set(plan.forbidden_confirmation_object_ids)
    if any(case.object_id in forbidden for case in values):
        raise ValueError("acquisition cases contain confirmation-object evidence")
    return values


def build_calibration_evidence_ledger(
    plan: Deform360CalibrationAcquisitionPlanV1,
    cases: Sequence[Deform360CalibrationAcquisitionCaseV1],
) -> EvidenceUseLedgerV1:
    """Build one conservative calibration-only raw-factor entry per object."""

    values = _validated_cases(plan, cases)
    entries: list[EvidenceUseV1] = []
    for case in values:
        raw_factor_sha256 = content_id(
            {
                "raw_factor_artifacts": case.raw_factor_artifacts,
            }
        )
        raw_factor_id = content_id(
            {
                "plan_id": plan.plan_id,
                "object_id": case.object_id,
                "episode_id": case.episode_id,
                "raw_factor_sha256": raw_factor_sha256,
            }
        )
        frame_stop = case.aligned_frame_count or 1
        sensor_family = (
            "multimodal-calibration-source"
            if case.status == "prepared"
            else "technical-failure"
        )
        entries.append(
            EvidenceUseV1(
                evidence_artifact_id=case.case_id,
                raw_factor_id=raw_factor_id,
                raw_factor_sha256=raw_factor_sha256,
                source_repository=plan.dataset_repository,
                source_revision=plan.dataset_revision,
                source_artifacts=case.raw_factor_artifacts,
                sensor_family=sensor_family,
                stream_id=(
                    f"{case.object_id}/episode_{case.episode_id:04d}/"
                    "calibration-source-panel"
                ),
                clock_id=cast(
                    str,
                    case.metadata.get("timeline_sha256", raw_factor_sha256),
                ),
                causal_frame_start=0,
                causal_frame_stop=frame_stop,
                correlation_group_ids=(
                    f"deform360:calibration:{case.object_id}:"
                    f"episode:{case.episode_id}",
                ),
                inference_role="calibration_only",
                metadata={
                    "object_id": case.object_id,
                    "episode_id": case.episode_id,
                    "stratum": case.stratum,
                    "status": case.status,
                    "case_id": case.case_id,
                    "technical_failure_retained": (
                        case.status == "technical_failure"
                    ),
                },
            )
        )
    return EvidenceUseLedgerV1(
        protocol_id=plan.protocol_id,
        case_id=DEFORM360_CALIBRATION_LEDGER_CASE_ID,
        causal_frame_stop=max(entry.causal_frame_stop for entry in entries),
        entries=entries,
        metadata={
            "plan_id": plan.plan_id,
            "calibration_object_ids": sorted(case.object_id for case in values),
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
            "technical_failures_retained_without_replacement": True,
        },
    )


def build_calibration_acquisition_result(
    plan: Deform360CalibrationAcquisitionPlanV1,
    cases: Sequence[Deform360CalibrationAcquisitionCaseV1],
    ledger: EvidenceUseLedgerV1,
    *,
    source_artifacts: Mapping[str, str],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Build the portable acquisition result after all ten units have a disposition."""

    values = _validated_cases(plan, cases)
    if not isinstance(ledger, EvidenceUseLedgerV1):
        raise TypeError("ledger must be an EvidenceUseLedgerV1")
    expected_ledger = build_calibration_evidence_ledger(plan, values)
    if ledger.ledger_id != expected_ledger.ledger_id:
        raise ValueError("evidence ledger differs from acquisition cases")
    prepared = sum(case.status == "prepared" for case in values)
    failed = len(values) - prepared
    status = "complete" if failed == 0 else "complete_with_technical_failures"
    descriptor: dict[str, object] = {
        "schema": DEFORM360_CALIBRATION_ACQUISITION_RESULT_SCHEMA,
        "schema_version": DEFORM360_CALIBRATION_ACQUISITION_VERSION,
        "semantics": DEFORM360_CALIBRATION_ACQUISITION_SEMANTICS,
        "protocol_id": plan.protocol_id,
        "plan_id": plan.plan_id,
        "implementation_revision": plan.implementation_revision,
        "case_ids": [case.case_id for case in values],
        "evidence_use_ledger_id": ledger.ledger_id,
        "prepared_object_count": prepared,
        "technical_failure_count": failed,
        "status": status,
        "source_artifacts": source_artifact_mapping(
            source_artifacts,
            name="acquisition result source_artifacts",
        ),
        "calibration_payloads_opened": True,
        "confirmation_payloads_opened": False,
        "target_outcomes_used": False,
        "replacement_allowed": False,
        "metadata": plain_json(
            frozen_finite_json_mapping(
                metadata or {},
                name="acquisition result metadata",
            )
        ),
        "claim_boundary": DEFORM360_CALIBRATION_ACQUISITION_CLAIM_BOUNDARY,
    }
    return {**descriptor, "result_id": content_id(descriptor)}


def validate_calibration_acquisition_result(value: object) -> dict[str, object]:
    """Strictly validate a portable acquisition-result record."""

    if not isinstance(value, Mapping):
        raise ValueError("acquisition result must be a JSON object")
    require_exact_fields(value, expected=_RESULT_FIELDS, name="acquisition result")
    if value["schema"] != DEFORM360_CALIBRATION_ACQUISITION_RESULT_SCHEMA:
        raise ValueError("acquisition result schema changed")
    if value["semantics"] != DEFORM360_CALIBRATION_ACQUISITION_SEMANTICS:
        raise ValueError("acquisition result semantics changed")
    if value["claim_boundary"] != DEFORM360_CALIBRATION_ACQUISITION_CLAIM_BOUNDARY:
        raise ValueError("acquisition result claim boundary changed")
    version = genuine_integer(
        value["schema_version"],
        name="acquisition result schema_version",
        minimum=1,
    )
    if version != DEFORM360_CALIBRATION_ACQUISITION_VERSION:
        raise ValueError("acquisition result schema_version changed")
    if value["protocol_id"] != DEFORM360_CALIBRATION_PROTOCOL_ID:
        raise ValueError("acquisition result protocol changed")
    exact_revision(value["implementation_revision"], name="implementation_revision")
    sha256_digest(value["plan_id"], name="plan_id")
    sha256_digest(value["evidence_use_ledger_id"], name="evidence_use_ledger_id")
    case_ids = value["case_ids"]
    if not isinstance(case_ids, list) or len(case_ids) != 10:
        raise ValueError("acquisition result must contain exactly ten case IDs")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("acquisition result case IDs must be unique")
    for case_id in case_ids:
        sha256_digest(case_id, name="case_id")
    prepared = genuine_integer(
        value["prepared_object_count"],
        name="prepared_object_count",
        minimum=0,
    )
    failed = genuine_integer(
        value["technical_failure_count"],
        name="technical_failure_count",
        minimum=0,
    )
    if prepared + failed != 10:
        raise ValueError("acquisition result object accounting changed")
    expected_status = "complete" if failed == 0 else "complete_with_technical_failures"
    if value["status"] != expected_status:
        raise ValueError("acquisition result status disagrees with failures")
    source_artifact_mapping(value["source_artifacts"], name="source_artifacts")
    metadata = value["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("acquisition result metadata must be a mapping")
    frozen_finite_json_mapping(metadata, name="acquisition result metadata")
    for key, expected in (
        ("calibration_payloads_opened", True),
        ("confirmation_payloads_opened", False),
        ("target_outcomes_used", False),
        ("replacement_allowed", False),
    ):
        if genuine_boolean(value[key], name=key) is not expected:
            raise ValueError(f"acquisition result information boundary changed: {key}")
    descriptor = dict(value)
    declared = sha256_digest(descriptor.pop("result_id"), name="result_id")
    if content_id(descriptor) != declared:
        raise ValueError("acquisition result_id does not match content")
    return dict(value)


def save_calibration_acquisition_plan(
    path: str | Path,
    plan: Deform360CalibrationAcquisitionPlanV1,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(plan, Deform360CalibrationAcquisitionPlanV1):
        raise TypeError("plan must be a Deform360CalibrationAcquisitionPlanV1")
    write_atomic_json(plan.to_record(), path, overwrite=overwrite)


def save_calibration_acquisition_case(
    path: str | Path,
    case: Deform360CalibrationAcquisitionCaseV1,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(case, Deform360CalibrationAcquisitionCaseV1):
        raise TypeError("case must be a Deform360CalibrationAcquisitionCaseV1")
    write_atomic_json(case.to_record(), path, overwrite=overwrite)


def save_calibration_evidence_ledger(
    path: str | Path,
    ledger: EvidenceUseLedgerV1,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(ledger, EvidenceUseLedgerV1):
        raise TypeError("ledger must be an EvidenceUseLedgerV1")
    write_atomic_json(ledger.to_record(), path, overwrite=overwrite)


def save_calibration_acquisition_result(
    path: str | Path,
    result: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    validated = validate_calibration_acquisition_result(result)
    write_atomic_json(validated, path, overwrite=overwrite)
