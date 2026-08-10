"""Strict source-only input validation for the Deform360 failure census."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Final, cast

from ._canonical_contracts import plain_json
from .prior_aware_gauge_belief_v2 import (
    PRIOR_AWARE_GAUGE_BELIEF_V2_IMPLEMENTATION,
    _validate_certificate_mapping,
)
from .prospective_prob4d_update import (
    CLAIM_BEARING_PROB4D_UPDATE_IDENTITY_VERSION,
    CLAIM_BEARING_PROB4D_UPDATE_VERSION,
)
from .provider_failure_decomposition import analyze_provider_failure_evidence
from .provider_failure_evidence_adapters import (
    CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_CLAIM_BOUNDARY,
    CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_SCHEMA,
    CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_VERSION,
)

DEFORM360_PROVIDER_FAILURE_CENSUS_INPUT_SCHEMA: Final = (
    "bayesian_phystwin.deform360_provider_failure_census_input"
)
DEFORM360_PROVIDER_FAILURE_CENSUS_INPUT_VERSION: Final = 1
DEFORM360_PROVIDER_FAILURE_CENSUS_CLAIM_BOUNDARY: Final = (
    "Source-only equal-case failure census. Validation does not authorize raw-tree "
    "traversal, confirmation or adaptive-confirmation access, target outcomes, "
    "future frames, replacement cases, provider promotion, or physical-benefit claims."
)

ALLOWED_DEFORM360_CENSUS_STATISTICAL_UNITS: Final[frozenset[str]] = frozenset(
    {"physical-object", "acquisition-session", "physical-object-session"}
)

_REQUIRED_SOURCE_METADATA: Final[dict[str, object]] = {
    "split": "source-only",
    "confirmation_payloads_opened": False,
    "adaptive_confirmation_payloads_opened": False,
    "target_outcomes_used": False,
    "future_frames_used": False,
    "replacement_allowed": False,
}

_ADAPTER_METADATA_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "adapter_schema",
        "adapter_schema_version",
        "adapter_claim_boundary",
        "source_contract",
        "strict_result_contract",
        "record_update_ids",
    }
)
_ADAPTER_METRIC_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "adapter_schema",
        "adapter_schema_version",
        "adapter_claim_boundary",
        "claim_bearing_update_id",
        "claim_bearing_admission_id",
        "claim_bearing_inference_result_id",
        "observation_artifact_id",
        "linearization_artifact_id",
        "provider_manifest_id",
        "calibration_artifact_ids",
        "runtime_revision_source",
        "runtime_revision_independently_verified",
        "strict_result_implementation_id",
        "strict_admission_certificate",
    }
)
_DIGEST_PATTERN: Final = re.compile(r"[0-9a-f]{64}")


def _canonical_id(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        plain_json(payload),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _literal_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} must use literal string keys")
    return cast(Mapping[str, object], value)


def _lowercase_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _DIGEST_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return cast(str, value)


def _nonempty_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be nonempty text")
    return cast(str, value)


def _record_metrics(record: Mapping[str, object]) -> Mapping[str, object]:
    value = record.get("metrics")
    if value is None:
        return {}
    return cast(Mapping[str, object], value)


def _contains_adapter_fields(records: Sequence[Mapping[str, object]]) -> bool:
    return any(
        bool(set(_record_metrics(record)).intersection(_ADAPTER_METRIC_FIELDS))
        for record in records
    )


def _validate_calibration_ids(
    value: object,
    *,
    index: int,
) -> dict[str, str]:
    calibration_ids = _literal_mapping(
        value,
        name=f"record {index} calibration_artifact_ids",
    )
    if not calibration_ids:
        raise ValueError(f"record {index} calibration_artifact_ids must not be empty")
    result: dict[str, str] = {}
    for name, digest in calibration_ids.items():
        if not name:
            raise ValueError(
                f"record {index} calibration artifact names must be nonempty"
            )
        result[name] = _lowercase_sha256(
            digest,
            name=f"record {index} calibration artifact {name!r}",
        )
    return result


def _validate_strict_certificate(
    value: object,
    *,
    accepted: bool,
    result_reason: str,
    index: int,
) -> Mapping[str, object]:
    certificate = _literal_mapping(
        value,
        name=f"record {index} strict_admission_certificate",
    )
    certificate_passed = _validate_certificate_mapping(certificate)
    if certificate_passed is not accepted:
        raise ValueError(
            f"record {index} certificate passed decision differs from accepted"
        )
    underlying_admissible = (
        certificate["underlying_inference_admissible"] is True
    )
    expected_result_reason = (
        certificate["underlying_inference_reason"]
        if certificate_passed or not underlying_admissible
        else certificate["reason"]
    )
    if result_reason != expected_result_reason:
        raise ValueError(
            f"record {index} result reason differs from the certificate path"
        )
    return certificate


def _validate_adapter_metadata(metadata: Mapping[str, object]) -> list[object]:
    if metadata.get("adapter_schema") != CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_SCHEMA:
        raise ValueError("unsupported claim-bearing provider-failure adapter schema")
    if (
        metadata.get("adapter_schema_version")
        != CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_VERSION
    ):
        raise ValueError("unsupported claim-bearing provider-failure adapter version")
    if (
        metadata.get("adapter_claim_boundary")
        != CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_CLAIM_BOUNDARY
    ):
        raise ValueError("claim-bearing adapter boundary text is not canonical")
    if metadata.get("source_contract") != "ClaimBearingProb4DUpdateV1":
        raise ValueError("claim-bearing adapter source_contract is invalid")
    if metadata.get("strict_result_contract") != "PriorAwareGaugeBeliefResultV2":
        raise ValueError("claim-bearing adapter strict_result_contract is invalid")
    bindings = metadata.get("record_update_ids")
    if not isinstance(bindings, list):
        raise ValueError("claim-bearing adapter record_update_ids must be a JSON array")
    return bindings


def _validate_adapter_record(
    *,
    record: Mapping[str, object],
    binding: object,
    provider_id: str,
    index: int,
) -> None:
    case_id = cast(str, record["case_id"])
    accepted = cast(bool, record["accepted"])
    result_reason = cast(str, record["result_reason"])
    binding_map = _literal_mapping(binding, name=f"record binding {index}")
    if binding_map.get("case_id") != case_id:
        raise ValueError(f"record binding {index} case_id differs from record order")
    bound_update_id = _lowercase_sha256(
        binding_map.get("update_id"),
        name=f"record binding {index} update_id",
    )

    metrics = _record_metrics(record)
    missing = _ADAPTER_METRIC_FIELDS.difference(metrics)
    if missing:
        raise ValueError(
            f"record {index} lacks adapter-owned metrics: {sorted(missing)}"
        )
    if metrics.get("adapter_schema") != CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_SCHEMA:
        raise ValueError(f"record {index} adapter schema differs from metadata")
    if (
        metrics.get("adapter_schema_version")
        != CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_VERSION
    ):
        raise ValueError(f"record {index} adapter version differs from metadata")
    if (
        metrics.get("adapter_claim_boundary")
        != CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_CLAIM_BOUNDARY
    ):
        raise ValueError(f"record {index} adapter boundary is not canonical")

    update_id = _lowercase_sha256(
        metrics.get("claim_bearing_update_id"),
        name=f"record {index} claim_bearing_update_id",
    )
    if update_id != bound_update_id:
        raise ValueError(f"record {index} update ID differs from metadata binding")
    admission_id = _lowercase_sha256(
        metrics.get("claim_bearing_admission_id"),
        name=f"record {index} claim_bearing_admission_id",
    )
    inference_result_id = _lowercase_sha256(
        metrics.get("claim_bearing_inference_result_id"),
        name=f"record {index} claim_bearing_inference_result_id",
    )
    observation_artifact_id = _lowercase_sha256(
        metrics.get("observation_artifact_id"),
        name=f"record {index} observation_artifact_id",
    )
    linearization_artifact_id = _lowercase_sha256(
        metrics.get("linearization_artifact_id"),
        name=f"record {index} linearization_artifact_id",
    )
    bound_provider_id = _lowercase_sha256(
        metrics.get("provider_manifest_id"),
        name=f"record {index} provider_manifest_id",
    )
    if bound_provider_id != provider_id:
        raise ValueError(f"record {index} provider identity differs from payload")
    calibration_ids = _validate_calibration_ids(
        metrics.get("calibration_artifact_ids"),
        index=index,
    )
    runtime_revision_source = _nonempty_text(
        metrics.get("runtime_revision_source"),
        name=f"record {index} runtime_revision_source",
    )
    if metrics.get("runtime_revision_independently_verified") is not True:
        raise ValueError(
            f"record {index} runtime revision must be independently verified"
        )
    if (
        metrics.get("strict_result_implementation_id")
        != PRIOR_AWARE_GAUGE_BELIEF_V2_IMPLEMENTATION
    ):
        raise ValueError(
            f"record {index} strict result implementation is unsupported"
        )
    _validate_strict_certificate(
        metrics.get("strict_admission_certificate"),
        accepted=accepted,
        result_reason=result_reason,
        index=index,
    )

    signals = cast(Mapping[str, object], record["signals"])
    if signals.get("technical_valid") is not True:
        raise ValueError(
            f"record {index} adapter evidence must establish technical_valid=true"
        )

    admission_payload: dict[str, object] = {
        "schema": "bayesian_phystwin.claim_bearing_prob4d_update",
        "schema_version": CLAIM_BEARING_PROB4D_UPDATE_VERSION,
        "observation_artifact_id": observation_artifact_id,
        "linearization_artifact_id": linearization_artifact_id,
        "provider_manifest_id": bound_provider_id,
        "calibration_artifact_ids": calibration_ids,
        "runtime_revision_source": runtime_revision_source,
        "runtime_revision_independently_verified": True,
        "inference_admissible": accepted,
        "reason": result_reason,
    }
    expected_admission_id = _canonical_id(admission_payload)
    if admission_id != expected_admission_id:
        raise ValueError(
            f"record {index} admission ID does not match its bound decision"
        )
    expected_update_id = _canonical_id(
        {
            **admission_payload,
            "identity_version": CLAIM_BEARING_PROB4D_UPDATE_IDENTITY_VERSION,
            "admission_id": expected_admission_id,
            "inference_result_id": inference_result_id,
        }
    )
    if update_id != expected_update_id:
        raise ValueError(
            f"record {index} update ID does not bind admission and inference result"
        )


def _validate_adapter_binding(
    payload: Mapping[str, object],
    metadata: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
) -> None:
    bindings = _validate_adapter_metadata(metadata)
    if len(bindings) != len(records):
        raise ValueError("record_update_ids must contain one binding per record")
    provider_id = _lowercase_sha256(
        payload.get("provider_id"),
        name="adapter payload provider_id",
    )
    for index, (record, binding) in enumerate(zip(records, bindings, strict=True)):
        _validate_adapter_record(
            record=record,
            binding=binding,
            provider_id=provider_id,
            index=index,
        )


def validate_deform360_provider_failure_census_payload(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Validate one immutable, source-only, equal-case Deform360 census input."""

    if not isinstance(payload, Mapping):
        raise ValueError("Deform360 provider-failure census input must be a mapping")
    report = analyze_provider_failure_evidence(payload)
    metadata = cast(Mapping[str, object], payload["metadata"])
    for key, expected in _REQUIRED_SOURCE_METADATA.items():
        if metadata.get(key) != expected:
            raise ValueError(f"metadata field {key!r} must equal {expected!r}")

    unit = metadata.get("statistical_unit")
    if unit not in ALLOWED_DEFORM360_CENSUS_STATISTICAL_UNITS:
        raise ValueError(
            "metadata.statistical_unit must be one of "
            f"{sorted(ALLOWED_DEFORM360_CENSUS_STATISTICAL_UNITS)}"
        )

    records = cast(Sequence[Mapping[str, object]], payload["records"])
    adapter_schema = metadata.get("adapter_schema")
    adapter_metadata_present = bool(
        set(metadata).intersection(_ADAPTER_METADATA_FIELDS)
    )
    adapter_metrics_present = _contains_adapter_fields(records)
    if adapter_schema is None:
        if adapter_metadata_present or adapter_metrics_present:
            raise ValueError(
                "claim-bearing adapter fields require complete canonical adapter metadata"
            )
    else:
        _validate_adapter_binding(payload, metadata, records)
    return report


__all__ = [
    "ALLOWED_DEFORM360_CENSUS_STATISTICAL_UNITS",
    "DEFORM360_PROVIDER_FAILURE_CENSUS_CLAIM_BOUNDARY",
    "DEFORM360_PROVIDER_FAILURE_CENSUS_INPUT_SCHEMA",
    "DEFORM360_PROVIDER_FAILURE_CENSUS_INPUT_VERSION",
    "validate_deform360_provider_failure_census_payload",
]
