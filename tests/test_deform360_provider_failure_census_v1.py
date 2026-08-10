from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

import pytest

from bayesian_phystwin.deform360_provider_failure_census_v1 import (
    ALLOWED_DEFORM360_CENSUS_STATISTICAL_UNITS,
    validate_deform360_provider_failure_census_payload,
)
from bayesian_phystwin.provider_failure_decomposition import (
    PROVIDER_FAILURE_EVIDENCE_SCHEMA,
    PROVIDER_FAILURE_EVIDENCE_VERSION,
)
from bayesian_phystwin.provider_failure_evidence_adapters import (
    CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_CLAIM_BOUNDARY,
    CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_SCHEMA,
    CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_VERSION,
)

PROVIDER_ID = "c" * 64
UPDATE_ID = "a" * 64
ADMISSION_ID = "b" * 64
INFERENCE_ID = "d" * 64
OBSERVATION_ID = "e" * 64
LINEARIZATION_ID = "f" * 64
CALIBRATION_ID = "1" * 64


def _source_metadata(
    statistical_unit: str = "physical-object",
) -> dict[str, object]:
    return {
        "split": "source-only",
        "statistical_unit": statistical_unit,
        "confirmation_payloads_opened": False,
        "adaptive_confirmation_payloads_opened": False,
        "target_outcomes_used": False,
        "future_frames_used": False,
        "replacement_allowed": False,
    }


def _generic_payload(
    statistical_unit: str = "physical-object",
) -> dict[str, object]:
    return {
        "schema": PROVIDER_FAILURE_EVIDENCE_SCHEMA,
        "schema_version": PROVIDER_FAILURE_EVIDENCE_VERSION,
        "provider_id": "generic-source-provider-v1",
        "records": [
            {
                "case_id": "object-01",
                "accepted": False,
                "result_reason": "rejected",
                "signals": {"technical_valid": True},
                "metrics": {"source_gate_id": "source-gate-v1"},
            }
        ],
        "metadata": _source_metadata(statistical_unit),
    }


def _adapter_payload() -> dict[str, object]:
    metadata = {
        **_source_metadata(),
        "adapter_schema": CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_SCHEMA,
        "adapter_schema_version": (CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_VERSION),
        "adapter_claim_boundary": (
            CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_CLAIM_BOUNDARY
        ),
        "source_contract": "ClaimBearingProb4DUpdateV1",
        "strict_result_contract": "PriorAwareGaugeBeliefResultV2",
        "record_update_ids": [{"case_id": "object-01", "update_id": UPDATE_ID}],
    }
    metrics = {
        "adapter_schema": CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_SCHEMA,
        "adapter_schema_version": (CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_VERSION),
        "adapter_claim_boundary": (
            CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_CLAIM_BOUNDARY
        ),
        "claim_bearing_update_id": UPDATE_ID,
        "claim_bearing_admission_id": ADMISSION_ID,
        "claim_bearing_inference_result_id": INFERENCE_ID,
        "observation_artifact_id": OBSERVATION_ID,
        "linearization_artifact_id": LINEARIZATION_ID,
        "provider_manifest_id": PROVIDER_ID,
        "calibration_artifact_ids": {"gauge": CALIBRATION_ID},
        "runtime_revision_source": "independent-vcs-check",
        "runtime_revision_independently_verified": True,
        "strict_result_implementation_id": "prior-aware-gauge-belief-v2",
        "strict_admission_certificate": {
            "passed": False,
            "underlying_inference_admissible": True,
            "reason": "strict-v2-fixed-point-not-converged",
        },
    }
    return {
        "schema": PROVIDER_FAILURE_EVIDENCE_SCHEMA,
        "schema_version": PROVIDER_FAILURE_EVIDENCE_VERSION,
        "provider_id": PROVIDER_ID,
        "records": [
            {
                "case_id": "object-01",
                "accepted": False,
                "result_reason": "strict-v2-fixed-point-not-converged",
                "signals": {
                    "technical_valid": True,
                    "provider_support_complete": True,
                    "numerically_converged": False,
                    "query_identifiable": True,
                    "physical_guard_passed": True,
                },
                "metrics": metrics,
            }
        ],
        "metadata": metadata,
    }


def _metadata(payload: dict[str, object]) -> dict[str, object]:
    return cast(dict[str, object], payload["metadata"])


def _record(payload: dict[str, object]) -> dict[str, object]:
    records = cast(list[dict[str, object]], payload["records"])
    return records[0]


def _metrics(payload: dict[str, object]) -> dict[str, object]:
    return cast(dict[str, object], _record(payload)["metrics"])


@pytest.mark.parametrize(
    "unit",
    sorted(ALLOWED_DEFORM360_CENSUS_STATISTICAL_UNITS),
)
def test_generic_source_payload_accepts_only_registered_equal_case_units(
    unit: str,
) -> None:
    report = validate_deform360_provider_failure_census_payload(_generic_payload(unit))

    assert report["record_count"] == 1
    assert report["accepted_count"] == 0
    assert report["unresolved_rejection_count"] == 1


def test_validator_requires_a_mapping() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        validate_deform360_provider_failure_census_payload(cast(Any, []))


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("split", "target"),
        ("confirmation_payloads_opened", True),
        ("adaptive_confirmation_payloads_opened", True),
        ("target_outcomes_used", True),
        ("future_frames_used", True),
        ("replacement_allowed", True),
    ],
)
def test_validator_requires_the_closed_source_metadata(
    field: str,
    invalid: object,
) -> None:
    payload = _generic_payload()
    _metadata(payload)[field] = invalid

    with pytest.raises(ValueError, match=field):
        validate_deform360_provider_failure_census_payload(payload)


@pytest.mark.parametrize(
    "unit",
    ["frame", "frames", "view", "camera-view", "physical-object ", "", None],
)
def test_validator_rejects_unregistered_statistical_units(unit: object) -> None:
    payload = _generic_payload()
    _metadata(payload)["statistical_unit"] = unit

    with pytest.raises(ValueError, match="statistical_unit must be one of"):
        validate_deform360_provider_failure_census_payload(payload)


def test_claim_bearing_adapter_payload_binds_every_record() -> None:
    payload = _adapter_payload()

    report = validate_deform360_provider_failure_census_payload(payload)

    assert report["record_count"] == 1
    assert report["classified_rejection_count"] == 1
    assert report["primary_category_counts"]["numerical-non-convergence"] == 1


def test_claim_bearing_adapter_accepts_a_certificate_bound_acceptance() -> None:
    payload = _adapter_payload()
    record = _record(payload)
    record["accepted"] = True
    record["result_reason"] = "strict-admission-passed"
    signals = cast(dict[str, object], record["signals"])
    signals["numerically_converged"] = True
    certificate = cast(
        dict[str, object],
        _metrics(payload)["strict_admission_certificate"],
    )
    certificate["passed"] = True
    certificate["reason"] = "strict-admission-passed"

    report = validate_deform360_provider_failure_census_payload(payload)

    assert report["accepted_count"] == 1


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("adapter_schema", "wrong", "unsupported.*schema"),
        ("adapter_schema_version", 2, "unsupported.*version"),
        ("adapter_claim_boundary", "wrong", "boundary text"),
        ("source_contract", "wrong", "source_contract"),
        ("strict_result_contract", "wrong", "strict_result_contract"),
        ("record_update_ids", {}, "JSON array"),
    ],
)
def test_adapter_metadata_is_exact(
    field: str,
    invalid: object,
    message: str,
) -> None:
    payload = _adapter_payload()
    _metadata(payload)[field] = invalid

    with pytest.raises(ValueError, match=message):
        validate_deform360_provider_failure_census_payload(payload)


def test_adapter_binding_count_must_match_record_count() -> None:
    payload = _adapter_payload()
    _metadata(payload)["record_update_ids"] = []

    with pytest.raises(ValueError, match="one binding per record"):
        validate_deform360_provider_failure_census_payload(payload)


@pytest.mark.parametrize(
    ("binding", "message"),
    [
        ([], "must be a mapping"),
        ({1: UPDATE_ID}, "literal string keys"),
        ({"case_id": "other", "update_id": UPDATE_ID}, "case_id differs"),
        ({"case_id": "object-01", "update_id": "bad"}, "lowercase SHA-256"),
    ],
)
def test_adapter_record_binding_is_literal_ordered_and_content_addressed(
    binding: object,
    message: str,
) -> None:
    payload = _adapter_payload()
    _metadata(payload)["record_update_ids"] = [binding]

    with pytest.raises(ValueError, match=message):
        validate_deform360_provider_failure_census_payload(payload)


def test_adapter_update_id_must_match_record_metrics() -> None:
    payload = _adapter_payload()
    _metrics(payload)["claim_bearing_update_id"] = "2" * 64

    with pytest.raises(ValueError, match="update ID differs"):
        validate_deform360_provider_failure_census_payload(payload)


def test_adapter_requires_every_owned_metric() -> None:
    payload = _adapter_payload()
    del _metrics(payload)["claim_bearing_admission_id"]

    with pytest.raises(ValueError, match="lacks adapter-owned metrics"):
        validate_deform360_provider_failure_census_payload(payload)


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("adapter_schema", "wrong", "adapter schema differs"),
        ("adapter_schema_version", 2, "adapter version differs"),
        ("adapter_claim_boundary", "wrong", "adapter boundary"),
    ],
)
def test_adapter_record_contract_matches_metadata(
    field: str,
    invalid: object,
    message: str,
) -> None:
    payload = _adapter_payload()
    _metrics(payload)[field] = invalid

    with pytest.raises(ValueError, match=message):
        validate_deform360_provider_failure_census_payload(payload)


@pytest.mark.parametrize(
    "field",
    [
        "claim_bearing_admission_id",
        "claim_bearing_inference_result_id",
        "observation_artifact_id",
        "linearization_artifact_id",
    ],
)
def test_adapter_record_identity_fields_are_lowercase_digests(field: str) -> None:
    payload = _adapter_payload()
    _metrics(payload)[field] = "BAD"

    with pytest.raises(ValueError, match=field):
        validate_deform360_provider_failure_census_payload(payload)


def test_adapter_payload_provider_identity_is_content_addressed() -> None:
    payload = _adapter_payload()
    payload["provider_id"] = "not-a-digest"

    with pytest.raises(ValueError, match="payload provider_id"):
        validate_deform360_provider_failure_census_payload(payload)


def test_adapter_record_provider_identity_matches_payload() -> None:
    payload = _adapter_payload()
    _metrics(payload)["provider_manifest_id"] = "2" * 64

    with pytest.raises(ValueError, match="provider identity differs"):
        validate_deform360_provider_failure_census_payload(payload)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([], "must be a mapping"),
        ({1: CALIBRATION_ID}, "literal string keys"),
        ({}, "must not be empty"),
        ({"": CALIBRATION_ID}, "names must be nonempty"),
        ({"gauge": "bad"}, "lowercase SHA-256"),
    ],
)
def test_adapter_calibration_identity_is_nonempty_and_content_addressed(
    value: object,
    message: str,
) -> None:
    payload = _adapter_payload()
    _metrics(payload)["calibration_artifact_ids"] = value

    with pytest.raises(ValueError, match=message):
        validate_deform360_provider_failure_census_payload(payload)


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("runtime_revision_source", "", "must be nonempty text"),
        (
            "runtime_revision_independently_verified",
            False,
            "must be independently verified",
        ),
        ("strict_result_implementation_id", "", "must be nonempty text"),
    ],
)
def test_adapter_runtime_and_implementation_are_explicit(
    field: str,
    invalid: object,
    message: str,
) -> None:
    payload = _adapter_payload()
    _metrics(payload)[field] = invalid

    with pytest.raises(ValueError, match=message):
        validate_deform360_provider_failure_census_payload(payload)


@pytest.mark.parametrize(
    ("certificate", "message"),
    [
        ([], "must be a mapping"),
        ({1: False}, "literal string keys"),
        (
            {
                "passed": 0,
                "underlying_inference_admissible": True,
                "reason": "rejected",
            },
            "field 'passed' must be a bool",
        ),
        (
            {
                "passed": False,
                "underlying_inference_admissible": 0,
                "reason": "rejected",
            },
            "underlying_inference_admissible.*bool",
        ),
        (
            {
                "passed": False,
                "underlying_inference_admissible": True,
                "reason": "",
            },
            "reason must be nonempty text",
        ),
        (
            {
                "passed": True,
                "underlying_inference_admissible": True,
                "reason": "strict-admission-passed",
            },
            "decision differs from accepted",
        ),
    ],
)
def test_adapter_certificate_is_literal_and_decision_bound(
    certificate: object,
    message: str,
) -> None:
    payload = _adapter_payload()
    _metrics(payload)["strict_admission_certificate"] = certificate

    with pytest.raises(ValueError, match=message):
        validate_deform360_provider_failure_census_payload(payload)


def test_adapter_evidence_must_establish_technical_validity() -> None:
    payload = _adapter_payload()
    signals = cast(dict[str, object], _record(payload)["signals"])
    signals["technical_valid"] = False

    with pytest.raises(ValueError, match="technical_valid=true"):
        validate_deform360_provider_failure_census_payload(payload)


@pytest.mark.parametrize(
    "fragment_location",
    ["metadata", "metrics"],
)
def test_partial_adapter_fields_are_rejected(fragment_location: str) -> None:
    payload = _generic_payload()
    if fragment_location == "metadata":
        _metadata(payload)["source_contract"] = "ClaimBearingProb4DUpdateV1"
    else:
        _metrics(payload)["adapter_schema"] = (
            CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_SCHEMA
        )

    with pytest.raises(ValueError, match="complete canonical adapter metadata"):
        validate_deform360_provider_failure_census_payload(payload)


def test_adapter_schema_without_owned_record_metrics_fails_closed() -> None:
    payload = _adapter_payload()
    _record(payload)["metrics"] = {"source_gate_id": "source-gate-v1"}

    with pytest.raises(ValueError, match="lacks adapter-owned metrics"):
        validate_deform360_provider_failure_census_payload(payload)


def test_generic_metrics_do_not_trigger_adapter_validation() -> None:
    payload = deepcopy(_generic_payload())
    _metrics(payload)["claim_bearing_comment"] = "not an adapter-owned field"

    report = validate_deform360_provider_failure_census_payload(payload)

    assert report["record_count"] == 1
