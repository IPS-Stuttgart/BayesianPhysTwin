"""Conservative adapters for source-only provider-failure evidence.

The adapter consumes an already-validated claim-bearing Prob4D update and its
strict-v2 admission certificate. It never reruns inference, changes a decision,
or infers externally owned calibration, identity, or robustness gates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final, TypeAlias, cast

from ._canonical_contracts import plain_json
from .prior_aware_gauge_belief_v2 import PriorAwareGaugeBeliefResultV2
from .prospective_prob4d_update import ClaimBearingProb4DUpdateV1
from .provider_failure_decomposition import (
    PROVIDER_FAILURE_EVIDENCE_SCHEMA,
    PROVIDER_FAILURE_EVIDENCE_VERSION,
    ProviderFailureEvidenceV1,
    ProviderFailureSignalsV1,
    analyze_provider_failure_evidence,
    decompose_provider_failure,
)

CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_SCHEMA: Final = (
    "bayesian_phystwin.claim_bearing_provider_failure_evidence_adapter"
)
CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_VERSION: Final = 1
CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_CLAIM_BOUNDARY: Final = (
    "Source-only evidence adaptation from an immutable claim-bearing Prob4D "
    "update and strict-v2 admission certificate. The adapter does not establish "
    "provider covariance calibration, material identity, robust support, physical "
    "benefit, deployment safety, target access, Causal4D benefit, or state of the art."
)

_STRICT_CERTIFICATE_KEY: Final = "strict_admission_certificate"
_ADAPTER_METADATA_KEYS: Final = frozenset(
    {
        "adapter_schema",
        "adapter_schema_version",
        "adapter_claim_boundary",
        "source_contract",
        "strict_result_contract",
        "record_update_ids",
    }
)

SourceSignalsInput: TypeAlias = (
    ProviderFailureSignalsV1 | Mapping[str, object] | None
)
ClaimBearingFailureCase: TypeAlias = tuple[str, ClaimBearingProb4DUpdateV1]


def _literal_mapping(
    value: Mapping[str, Any] | None,
    *,
    name: str,
) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping or null")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} must use literal string keys")
    return dict(value)


def _strict_result_and_certificate(
    update: ClaimBearingProb4DUpdateV1,
) -> tuple[PriorAwareGaugeBeliefResultV2, Mapping[str, object]]:
    if not isinstance(update, ClaimBearingProb4DUpdateV1):
        raise TypeError("update must be a ClaimBearingProb4DUpdateV1")
    result = update.result
    if not isinstance(result, PriorAwareGaugeBeliefResultV2):
        raise TypeError(
            "update.result must be a PriorAwareGaugeBeliefResultV2 with strict "
            "admission evidence"
        )
    certificate = result.diagnostics.get(_STRICT_CERTIFICATE_KEY)
    if not isinstance(certificate, Mapping):
        raise ValueError("strict-v2 result lacks an admission certificate")
    for name in ("passed", "underlying_inference_admissible"):
        if type(certificate.get(name)) is not bool:
            raise ValueError(f"strict-v2 certificate field {name!r} must be a bool")
    if type(certificate.get("reason")) is not str or not certificate.get("reason"):
        raise ValueError("strict-v2 certificate reason must be nonempty text")
    return result, cast(Mapping[str, object], certificate)


def _reason_derived_signal(reason: str) -> str | None:
    probe = ProviderFailureEvidenceV1(
        case_id="claim-bearing-adapter-reason-probe",
        accepted=False,
        result_reason=reason,
        signals=ProviderFailureSignalsV1(),
        metrics={},
    )
    return decompose_provider_failure(probe).reason_derived_signal


def _derived_signals(
    result: PriorAwareGaugeBeliefResultV2,
    certificate: Mapping[str, object],
) -> ProviderFailureSignalsV1:
    passed = certificate["passed"] is True
    underlying_admissible = certificate["underlying_inference_admissible"] is True
    result_signal = _reason_derived_signal(result.reason)
    strict_signal = _reason_derived_signal(cast(str, certificate["reason"]))

    return ProviderFailureSignalsV1(
        technical_valid=True,
        provider_support_complete=(
            False
            if result_signal == "provider_support_complete"
            else True if underlying_admissible else None
        ),
        numerically_converged=(
            True
            if passed
            else False
            if result_signal == "numerically_converged"
            or strict_signal == "numerically_converged"
            else None
        ),
        query_identifiable=(
            False
            if result_signal == "query_identifiable"
            else True if underlying_admissible else None
        ),
        physical_guard_passed=(
            False
            if result_signal == "physical_guard_passed"
            else True if underlying_admissible else None
        ),
    )


def _supplied_signals(value: SourceSignalsInput) -> ProviderFailureSignalsV1:
    if value is None:
        return ProviderFailureSignalsV1()
    if isinstance(value, ProviderFailureSignalsV1):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(
            "source_signals must be ProviderFailureSignalsV1, a mapping, or null"
        )
    return ProviderFailureSignalsV1.from_mapping(value)


def _merge_signals(
    derived: ProviderFailureSignalsV1,
    supplied: ProviderFailureSignalsV1,
) -> ProviderFailureSignalsV1:
    derived_values = derived.to_dict()
    supplied_values = supplied.to_dict()
    merged: dict[str, object] = {}
    for name, derived_value in derived_values.items():
        supplied_value = supplied_values[name]
        if (
            derived_value is not None
            and supplied_value is not None
            and derived_value is not supplied_value
        ):
            raise ValueError(
                f"source signal {name!r} contradicts immutable claim-bearing evidence"
            )
        merged[name] = supplied_value if supplied_value is not None else derived_value
    return ProviderFailureSignalsV1.from_mapping(merged)


def _adapter_metrics(
    update: ClaimBearingProb4DUpdateV1,
    result: PriorAwareGaugeBeliefResultV2,
    certificate: Mapping[str, object],
) -> dict[str, object]:
    return {
        "adapter_schema": CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_SCHEMA,
        "adapter_schema_version": CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_VERSION,
        "adapter_claim_boundary": (
            CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_CLAIM_BOUNDARY
        ),
        "claim_bearing_update_id": update.update_id,
        "claim_bearing_admission_id": update.admission_id,
        "claim_bearing_inference_result_id": update.inference_result_id,
        "observation_artifact_id": update.observation_artifact_id,
        "linearization_artifact_id": update.linearization_artifact_id,
        "provider_manifest_id": update.provider_manifest_id,
        "calibration_artifact_ids": dict(update.calibration_artifact_ids),
        "runtime_revision_source": update.runtime_revision_source,
        "runtime_revision_independently_verified": (
            update.runtime_revision_independently_verified
        ),
        "strict_result_implementation_id": result.implementation_id,
        "strict_admission_certificate": plain_json(certificate),
    }


def provider_failure_evidence_from_claim_bearing_update(
    case_id: str,
    update: ClaimBearingProb4DUpdateV1,
    *,
    source_signals: SourceSignalsInput = None,
    metrics: Mapping[str, Any] | None = None,
) -> ProviderFailureEvidenceV1:
    """Adapt one immutable strict claim-bearing update into diagnostic evidence.

    Independently owned source/calibration signals may fill otherwise unknown
    fields. They cannot contradict facts derived from the update or certificate.
    """

    result, certificate = _strict_result_and_certificate(update)
    supplied_metrics = _literal_mapping(metrics, name="metrics")
    bound_metrics = _adapter_metrics(update, result, certificate)
    overlap = set(supplied_metrics).intersection(bound_metrics)
    if overlap:
        raise ValueError(
            "metrics cannot replace adapter-owned fields: " f"{sorted(overlap)}"
        )
    evidence = ProviderFailureEvidenceV1(
        case_id=case_id,
        accepted=update.inference_admissible,
        result_reason=result.reason,
        signals=_merge_signals(
            _derived_signals(result, certificate),
            _supplied_signals(source_signals),
        ),
        metrics={**bound_metrics, **supplied_metrics},
    )
    decompose_provider_failure(evidence)
    return evidence


def _case_mapping(
    value: Mapping[str, Any] | None,
    *,
    name: str,
    case_ids: set[str],
) -> dict[str, Any]:
    result = _literal_mapping(value, name=name)
    unknown = set(result).difference(case_ids)
    if unknown:
        raise ValueError(f"{name} contains unknown case IDs: {sorted(unknown)}")
    return result


def _evidence_record(evidence: ProviderFailureEvidenceV1) -> dict[str, object]:
    return {
        "case_id": evidence.case_id,
        "accepted": evidence.accepted,
        "result_reason": evidence.result_reason,
        "signals": evidence.signals.to_dict(),
        "metrics": plain_json(evidence.metrics),
    }


def build_provider_failure_payload_from_claim_bearing_updates(
    cases: Sequence[ClaimBearingFailureCase],
    *,
    source_signals_by_case: Mapping[str, SourceSignalsInput] | None = None,
    metrics_by_case: Mapping[str, Mapping[str, Any]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Build a directly analyzable payload from ordered claim-bearing updates."""

    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)):
        raise TypeError("cases must be a sequence of (case_id, update) pairs")
    if not cases:
        raise ValueError("cases must not be empty")

    parsed: list[ClaimBearingFailureCase] = []
    for index, item in enumerate(cases):
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)):
            raise TypeError(f"case entry {index} must be a two-item sequence")
        if len(item) != 2:
            raise ValueError(f"case entry {index} must contain case_id and update")
        case_id, update = item
        if type(case_id) is not str or not case_id:
            raise ValueError(f"case entry {index} has an invalid case_id")
        if not isinstance(update, ClaimBearingProb4DUpdateV1):
            raise TypeError(
                f"case entry {index} update must be ClaimBearingProb4DUpdateV1"
            )
        parsed.append((case_id, update))

    ordered_ids = [case_id for case_id, _ in parsed]
    case_ids = set(ordered_ids)
    if len(case_ids) != len(ordered_ids):
        raise ValueError("case_id values must be unique")

    signals = _case_mapping(
        source_signals_by_case,
        name="source_signals_by_case",
        case_ids=case_ids,
    )
    case_metrics = _case_mapping(
        metrics_by_case,
        name="metrics_by_case",
        case_ids=case_ids,
    )
    supplied_metadata = _literal_mapping(metadata, name="metadata")
    metadata_overlap = set(supplied_metadata).intersection(_ADAPTER_METADATA_KEYS)
    if metadata_overlap:
        raise ValueError(
            "metadata cannot replace adapter-owned fields: "
            f"{sorted(metadata_overlap)}"
        )

    provider_ids = {update.provider_manifest_id for _, update in parsed}
    if len(provider_ids) != 1:
        raise ValueError("all cases must bind the same provider_manifest_id")
    provider_id = next(iter(provider_ids))

    evidence = [
        provider_failure_evidence_from_claim_bearing_update(
            case_id,
            update,
            source_signals=signals.get(case_id),
            metrics=case_metrics.get(case_id),
        )
        for case_id, update in parsed
    ]
    payload: dict[str, object] = {
        "schema": PROVIDER_FAILURE_EVIDENCE_SCHEMA,
        "schema_version": PROVIDER_FAILURE_EVIDENCE_VERSION,
        "provider_id": provider_id,
        "records": [_evidence_record(record) for record in evidence],
        "metadata": {
            **supplied_metadata,
            "adapter_schema": CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_SCHEMA,
            "adapter_schema_version": (
                CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_VERSION
            ),
            "adapter_claim_boundary": (
                CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_CLAIM_BOUNDARY
            ),
            "source_contract": "ClaimBearingProb4DUpdateV1",
            "strict_result_contract": "PriorAwareGaugeBeliefResultV2",
            "record_update_ids": [
                {"case_id": case_id, "update_id": update.update_id}
                for case_id, update in parsed
            ],
        },
    }
    analyze_provider_failure_evidence(payload)
    return payload


__all__ = [
    "CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_CLAIM_BOUNDARY",
    "CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_SCHEMA",
    "CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_VERSION",
    "ClaimBearingFailureCase",
    "SourceSignalsInput",
    "build_provider_failure_payload_from_claim_bearing_updates",
    "provider_failure_evidence_from_claim_bearing_update",
]
