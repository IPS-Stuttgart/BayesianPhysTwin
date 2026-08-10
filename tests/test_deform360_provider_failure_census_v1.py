from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, cast

import pytest

from bayesian_phystwin.deform360_provider_failure_census_v1 import (
    ALLOWED_DEFORM360_CENSUS_STATISTICAL_UNITS,
    validate_deform360_provider_failure_census_payload,
)
from bayesian_phystwin.prior_aware_gauge_belief_v2 import (
    PRIOR_AWARE_GAUGE_BELIEF_V2_IMPLEMENTATION,
)
from bayesian_phystwin.prospective_prob4d_update import (
    CLAIM_BEARING_PROB4D_UPDATE_IDENTITY_VERSION,
    CLAIM_BEARING_PROB4D_UPDATE_VERSION,
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
INFERENCE_ID = "d" * 64
OBSERVATION_ID = "e" * 64
LINEARIZATION_ID = "f" * 64
CALIBRATION_ID = "1" * 64
VALID_DIGEST = "a" * 64


def _canonical_id(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_metadata(unit: str = "physical-object") -> dict[str, object]:
    return {
        "split": "source-only",
        "statistical_unit": unit,
        "confirmation_payloads_opened": False,
        "adaptive_confirmation_payloads_opened": False,
        "target_outcomes_used": False,
        "future_frames_used": False,
        "replacement_allowed": False,
    }


def _generic_payload(unit: str = "physical-object") -> dict[str, object]:
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
        "metadata": _source_metadata(unit),
    }


def _certificate(accepted: bool) -> dict[str, object]:
    return {
        "schema": "bayesian_phystwin.prior_aware_gauge_admission_certificate",
        "schema_version": 1,
        "underlying_inference_admissible": True,
        "underlying_inference_reason": "accepted",
        "exact_mixture_objective": True,
        "fixed_point_converged": accepted,
        "diagnostics_valid": True,
        "positive_exact_mixture_curvature": True,
        "condition_number_within_limit": True,
        "mixture_solution_delta": 0.0,
        "mixture_stationarity_norm": 0.0,
        "exact_hessian_minimum_eigenvalue": 1.0,
        "exact_hessian_maximum_eigenvalue": 2.0,
        "exact_hessian_condition_number": 2.0,
        "maximum_exact_hessian_condition_number": 1.0e14,
        "passed": accepted,
        "reason": (
            "strict-admission-passed"
            if accepted
            else "strict-v2-fixed-point-not-converged"
        ),
    }


def _adapter_payload(accepted: bool = False) -> dict[str, object]:
    certificate = _certificate(accepted)
    reason = "accepted" if accepted else cast(str, certificate["reason"])
    calibration_ids = {"gauge": CALIBRATION_ID}
    admission: dict[str, object] = {
        "schema": "bayesian_phystwin.claim_bearing_prob4d_update",
        "schema_version": CLAIM_BEARING_PROB4D_UPDATE_VERSION,
        "observation_artifact_id": OBSERVATION_ID,
        "linearization_artifact_id": LINEARIZATION_ID,
        "provider_manifest_id": PROVIDER_ID,
        "calibration_artifact_ids": calibration_ids,
        "runtime_revision_source": "independent-vcs-check",
        "runtime_revision_independently_verified": True,
        "inference_admissible": accepted,
        "reason": reason,
    }
    admission_id = _canonical_id(admission)
    update_id = _canonical_id(
        {
            **admission,
            "identity_version": CLAIM_BEARING_PROB4D_UPDATE_IDENTITY_VERSION,
            "admission_id": admission_id,
            "inference_result_id": INFERENCE_ID,
        }
    )
    return {
        "schema": PROVIDER_FAILURE_EVIDENCE_SCHEMA,
        "schema_version": PROVIDER_FAILURE_EVIDENCE_VERSION,
        "provider_id": PROVIDER_ID,
        "records": [
            {
                "case_id": "object-01",
                "accepted": accepted,
                "result_reason": reason,
                "signals": {
                    "technical_valid": True,
                    "provider_support_complete": True,
                    "numerically_converged": accepted,
                    "query_identifiable": True,
                    "physical_guard_passed": True,
                },
                "metrics": {
                    "adapter_schema": (CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_SCHEMA),
                    "adapter_schema_version": (
                        CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_VERSION
                    ),
                    "adapter_claim_boundary": (
                        CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_CLAIM_BOUNDARY
                    ),
                    "claim_bearing_update_id": update_id,
                    "claim_bearing_admission_id": admission_id,
                    "claim_bearing_inference_result_id": INFERENCE_ID,
                    "observation_artifact_id": OBSERVATION_ID,
                    "linearization_artifact_id": LINEARIZATION_ID,
                    "provider_manifest_id": PROVIDER_ID,
                    "calibration_artifact_ids": calibration_ids,
                    "runtime_revision_source": "independent-vcs-check",
                    "runtime_revision_independently_verified": True,
                    "strict_result_implementation_id": (
                        PRIOR_AWARE_GAUGE_BELIEF_V2_IMPLEMENTATION
                    ),
                    "strict_admission_certificate": certificate,
                },
            }
        ],
        "metadata": {
            **_source_metadata(),
            "adapter_schema": CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_SCHEMA,
            "adapter_schema_version": (CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_VERSION),
            "adapter_claim_boundary": (
                CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_CLAIM_BOUNDARY
            ),
            "source_contract": "ClaimBearingProb4DUpdateV1",
            "strict_result_contract": "PriorAwareGaugeBeliefResultV2",
            "record_update_ids": [{"case_id": "object-01", "update_id": update_id}],
        },
    }


def _metadata(payload: dict[str, object]) -> dict[str, object]:
    return cast(dict[str, object], payload["metadata"])


def _record(payload: dict[str, object]) -> dict[str, object]:
    return cast(list[dict[str, object]], payload["records"])[0]


def _metrics(payload: dict[str, object]) -> dict[str, object]:
    return cast(dict[str, object], _record(payload)["metrics"])


def _certificate_from(payload: dict[str, object]) -> dict[str, object]:
    return cast(dict[str, object], _metrics(payload)["strict_admission_certificate"])


@pytest.mark.parametrize("unit", sorted(ALLOWED_DEFORM360_CENSUS_STATISTICAL_UNITS))
def test_generic_units(unit: str) -> None:
    report = validate_deform360_provider_failure_census_payload(_generic_payload(unit))
    assert report["record_count"] == 1
    assert report["unresolved_rejection_count"] == 1


def test_generic_optional_metrics() -> None:
    payload = _generic_payload()
    del _record(payload)["metrics"]
    assert (
        validate_deform360_provider_failure_census_payload(payload)["record_count"] == 1
    )


def test_top_level_requires_mapping() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        validate_deform360_provider_failure_census_payload(cast(Any, []))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("split", "target"),
        ("confirmation_payloads_opened", True),
        ("adaptive_confirmation_payloads_opened", True),
        ("target_outcomes_used", True),
        ("future_frames_used", True),
        ("replacement_allowed", True),
    ],
)
def test_source_metadata_is_closed(field: str, value: object) -> None:
    payload = _generic_payload()
    _metadata(payload)[field] = value
    with pytest.raises(ValueError, match=field):
        validate_deform360_provider_failure_census_payload(payload)


@pytest.mark.parametrize(
    "unit",
    ["frame", "frames", "view", "camera-view", "physical-object ", "", None],
)
def test_unregistered_units_fail(unit: object) -> None:
    payload = _generic_payload()
    _metadata(payload)["statistical_unit"] = unit
    with pytest.raises(ValueError, match="statistical_unit must be one of"):
        validate_deform360_provider_failure_census_payload(payload)


@pytest.mark.parametrize("accepted", [False, True])
def test_adapter_valid_paths(accepted: bool) -> None:
    report = validate_deform360_provider_failure_census_payload(
        _adapter_payload(accepted)
    )
    assert report["accepted_count"] == int(accepted)
    assert report["record_count"] == 1


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("adapter_schema", "wrong", "unsupported.*schema"),
        ("adapter_schema_version", 2, "unsupported.*version"),
        ("adapter_claim_boundary", "wrong", "boundary text"),
        ("source_contract", "wrong", "source_contract"),
        ("strict_result_contract", "wrong", "strict_result_contract"),
        ("record_update_ids", {}, "JSON array"),
    ],
)
def test_adapter_metadata_exact(field: str, value: object, match: str) -> None:
    payload = _adapter_payload()
    _metadata(payload)[field] = value
    with pytest.raises(ValueError, match=match):
        validate_deform360_provider_failure_census_payload(payload)


def test_binding_count_exact() -> None:
    payload = _adapter_payload()
    _metadata(payload)["record_update_ids"] = []
    with pytest.raises(ValueError, match="one binding per record"):
        validate_deform360_provider_failure_census_payload(payload)


@pytest.mark.parametrize(
    ("binding", "match"),
    [
        ([], "must be a mapping"),
        (cast(Any, {1: VALID_DIGEST}), "literal string.*keys"),
        ({"case_id": "other", "update_id": VALID_DIGEST}, "case_id differs"),
        ({"case_id": "object-01", "update_id": "bad"}, "lowercase SHA-256"),
    ],
)
def test_binding_contract(binding: object, match: str) -> None:
    payload = _adapter_payload()
    _metadata(payload)["record_update_ids"] = [binding]
    with pytest.raises(ValueError, match=match):
        validate_deform360_provider_failure_census_payload(payload)


def test_binding_matches_record_update() -> None:
    payload = _adapter_payload()
    _metrics(payload)["claim_bearing_update_id"] = "2" * 64
    with pytest.raises(ValueError, match="update ID differs"):
        validate_deform360_provider_failure_census_payload(payload)


def test_all_adapter_metrics_required() -> None:
    payload = _adapter_payload()
    del _metrics(payload)["claim_bearing_admission_id"]
    with pytest.raises(ValueError, match="lacks adapter-owned metrics"):
        validate_deform360_provider_failure_census_payload(payload)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("adapter_schema", "wrong", "adapter schema differs"),
        ("adapter_schema_version", 2, "adapter version differs"),
        ("adapter_claim_boundary", "wrong", "adapter boundary"),
    ],
)
def test_record_adapter_contract(field: str, value: object, match: str) -> None:
    payload = _adapter_payload()
    _metrics(payload)[field] = value
    with pytest.raises(ValueError, match=match):
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
def test_record_ids_are_digests(field: str) -> None:
    payload = _adapter_payload()
    _metrics(payload)[field] = "BAD"
    with pytest.raises(ValueError, match=field):
        validate_deform360_provider_failure_census_payload(payload)


def test_payload_provider_is_digest() -> None:
    payload = _adapter_payload()
    payload["provider_id"] = "bad"
    with pytest.raises(ValueError, match="payload provider_id"):
        validate_deform360_provider_failure_census_payload(payload)


def test_record_provider_matches_payload() -> None:
    payload = _adapter_payload()
    _metrics(payload)["provider_manifest_id"] = "2" * 64
    with pytest.raises(ValueError, match="provider identity differs"):
        validate_deform360_provider_failure_census_payload(payload)


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ([], "must be a mapping"),
        (cast(Any, {1: CALIBRATION_ID}), "literal string.*keys"),
        ({}, "must not be empty"),
        ({"": CALIBRATION_ID}, "names must be nonempty"),
        ({"gauge": "bad"}, "lowercase SHA-256"),
    ],
)
def test_calibration_ids(value: object, match: str) -> None:
    payload = _adapter_payload()
    _metrics(payload)["calibration_artifact_ids"] = value
    with pytest.raises(ValueError, match=match):
        validate_deform360_provider_failure_census_payload(payload)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("runtime_revision_source", "", "must be nonempty text"),
        (
            "runtime_revision_independently_verified",
            False,
            "must be independently verified",
        ),
        (
            "strict_result_implementation_id",
            "wrong",
            "strict result implementation is unsupported",
        ),
    ],
)
def test_runtime_and_implementation(field: str, value: object, match: str) -> None:
    payload = _adapter_payload()
    _metrics(payload)[field] = value
    with pytest.raises(ValueError, match=match):
        validate_deform360_provider_failure_census_payload(payload)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("schema", "unsupported schema"),
        ("version", "unsupported schema_version"),
        ("underlying-reason", "underlying reason must be nonempty text"),
        ("boolean", "exact_mixture_objective.*bool"),
        ("real", "mixture_solution_delta.*finite real or null"),
        ("curvature", "curvature invariant is inconsistent"),
        ("condition", "condition number is inconsistent"),
        ("passed", "pass invariant is inconsistent"),
        ("reason", "reason invariant is inconsistent"),
        ("fields", "certificate fields changed"),
    ],
)
def test_complete_certificate_contract(mutation: str, match: str) -> None:
    payload = _adapter_payload()
    certificate = _certificate_from(payload)
    if mutation == "schema":
        certificate["schema"] = "wrong"
    elif mutation == "version":
        certificate["schema_version"] = 2
    elif mutation == "underlying-reason":
        certificate["underlying_inference_reason"] = ""
    elif mutation == "boolean":
        certificate["exact_mixture_objective"] = 1
    elif mutation == "real":
        certificate["mixture_solution_delta"] = "zero"
    elif mutation == "curvature":
        certificate["positive_exact_mixture_curvature"] = False
    elif mutation == "condition":
        certificate["exact_hessian_condition_number"] = 3.0
    elif mutation == "passed":
        certificate["passed"] = True
    elif mutation == "reason":
        certificate["reason"] = "wrong"
    else:
        del certificate["schema"]
    with pytest.raises(ValueError, match=match):
        validate_deform360_provider_failure_census_payload(payload)


@pytest.mark.parametrize(
    ("certificate", "match"),
    [
        ([], "must be a mapping"),
        (cast(Any, {1: False}), "literal string.*keys"),
    ],
)
def test_certificate_container(certificate: object, match: str) -> None:
    payload = _adapter_payload()
    _metrics(payload)["strict_admission_certificate"] = certificate
    with pytest.raises(ValueError, match=match):
        validate_deform360_provider_failure_census_payload(payload)


def test_certificate_decision_matches_record() -> None:
    payload = _adapter_payload()
    _record(payload)["accepted"] = True
    _record(payload)["result_reason"] = "accepted"
    cast(dict[str, object], _record(payload)["signals"])["numerically_converged"] = True
    with pytest.raises(ValueError, match="decision differs from accepted"):
        validate_deform360_provider_failure_census_payload(payload)


def test_result_reason_matches_certificate_path() -> None:
    payload = _adapter_payload()
    _record(payload)["result_reason"] = "rejected"
    with pytest.raises(ValueError, match="result reason differs"):
        validate_deform360_provider_failure_census_payload(payload)


@pytest.mark.parametrize(
    ("field", "bind_update", "match"),
    [
        ("claim_bearing_admission_id", False, "admission ID does not match"),
        ("claim_bearing_update_id", True, "update ID does not bind"),
        ("claim_bearing_inference_result_id", False, "update ID does not bind"),
        ("observation_artifact_id", False, "admission ID does not match"),
    ],
)
def test_content_ids(field: str, bind_update: bool, match: str) -> None:
    payload = _adapter_payload()
    replacement = "2" * 64
    _metrics(payload)[field] = replacement
    if bind_update:
        bindings = cast(
            list[dict[str, object]],
            _metadata(payload)["record_update_ids"],
        )
        bindings[0]["update_id"] = replacement
    with pytest.raises(ValueError, match=match):
        validate_deform360_provider_failure_census_payload(payload)


def test_adapter_requires_technical_validity() -> None:
    payload = _adapter_payload()
    cast(dict[str, object], _record(payload)["signals"])["technical_valid"] = False
    with pytest.raises(ValueError, match="technical_valid=true"):
        validate_deform360_provider_failure_census_payload(payload)


@pytest.mark.parametrize("location", ["metadata", "metrics"])
def test_partial_adapter_fields(location: str) -> None:
    payload = _generic_payload()
    if location == "metadata":
        _metadata(payload)["source_contract"] = "ClaimBearingProb4DUpdateV1"
    else:
        _metrics(payload)["adapter_schema"] = (
            CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_SCHEMA
        )
    with pytest.raises(ValueError, match="complete canonical adapter metadata"):
        validate_deform360_provider_failure_census_payload(payload)


def test_adapter_metrics_cannot_be_partial() -> None:
    payload = _adapter_payload()
    _record(payload)["metrics"] = {"source_gate_id": "source-gate-v1"}
    with pytest.raises(ValueError, match="lacks adapter-owned metrics"):
        validate_deform360_provider_failure_census_payload(payload)


def test_unrelated_metrics_do_not_enable_adapter_validation() -> None:
    payload = deepcopy(_generic_payload())
    _metrics(payload)["claim_bearing_comment"] = "not adapter-owned"
    assert (
        validate_deform360_provider_failure_census_payload(payload)["record_count"] == 1
    )
