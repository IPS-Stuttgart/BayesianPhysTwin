"""Source-only attribution of rejected observation-to-belief updates.

The diagnostic deliberately classifies already-produced gate evidence. It does
not choose thresholds, inspect target outcomes, or turn a failed provider into an
accepted update. Unknown evidence remains explicit rather than being treated as
passing evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, Final, cast

from ._canonical_contracts import frozen_finite_json_mapping, plain_json

PROVIDER_FAILURE_EVIDENCE_SCHEMA: Final = (
    "bayesian_phystwin.provider_failure_evidence"
)
PROVIDER_FAILURE_EVIDENCE_VERSION: Final = 1
PROVIDER_FAILURE_REPORT_SCHEMA: Final = (
    "bayesian_phystwin.provider_failure_decomposition"
)
PROVIDER_FAILURE_REPORT_VERSION: Final = 1
PROVIDER_FAILURE_INFORMATION_BOUNDARY: Final = (
    "Source-only diagnostic attribution. The report consumes frozen gate outcomes "
    "and result reasons, never authorizes target access, never changes an acceptance "
    "decision, and is not observation-competence or physical-benefit evidence."
)

ACCEPTED_CATEGORY: Final = "accepted"
UNRESOLVED_REJECTION_CATEGORY: Final = "unresolved-rejection"

_SIGNAL_TO_CATEGORY: Final[dict[str, str]] = {
    "technical_valid": "technical-failure",
    "provider_support_complete": "unsupported-provider-geometry",
    "numerically_converged": "numerical-non-convergence",
    "query_identifiable": "unidentifiable-physical-query",
    "gauge_or_common_mode_consistent": "coherent-gauge-or-common-mode-bias",
    "covariance_calibrated": "provider-covariance-miscalibration",
    "material_identity_reliable": "association-or-material-identity-failure",
    "robust_support_sufficient": "outlier-dominated-evidence",
    "physical_guard_passed": "physical-model-or-readout-mismatch",
}

CLASSIFICATION_PRECEDENCE: Final[tuple[str, ...]] = tuple(
    _SIGNAL_TO_CATEGORY.values()
)

_REASON_TO_SIGNAL: Final[dict[str, str]] = {
    "no-observation-support": "provider_support_complete",
    "released-robot-geometry-outside-fixed-camera-prefix": (
        "provider_support_complete"
    ),
    "no-identifiable-query-state": "query_identifiable",
    "mixture-fixed-point-not-converged": "numerically_converged",
    "strict-v2-fixed-point-not-converged": "numerically_converged",
    "strict-v2-non-exact-mixture-objective": "numerically_converged",
    "strict-v2-invalid-admission-diagnostics": "numerically_converged",
    "strict-v2-non-positive-exact-mixture-curvature": "numerically_converged",
    "strict-v2-ill-conditioned-exact-mixture-curvature": "numerically_converged",
    "singular-posterior": "numerically_converged",
    "ill-conditioned-posterior": "numerically_converged",
    "implausible-state-update": "physical_guard_passed",
}


def _optional_boolean(value: object, *, name: str) -> bool | None:
    if value is None:
        return None
    if type(value) is not bool:
        raise ValueError(f"signal {name!r} must be a bool or null")
    return cast(bool, value)


def _require_literal_keys(values: Mapping[object, object], *, name: str) -> None:
    if any(type(key) is not str for key in values):
        raise ValueError(f"{name} must use literal string keys")


@dataclass(frozen=True, slots=True)
class ProviderFailureSignalsV1:
    """Tri-state outcomes from independently owned provider and consumer gates."""

    technical_valid: bool | None = None
    provider_support_complete: bool | None = None
    numerically_converged: bool | None = None
    query_identifiable: bool | None = None
    gauge_or_common_mode_consistent: bool | None = None
    covariance_calibrated: bool | None = None
    material_identity_reliable: bool | None = None
    robust_support_sufficient: bool | None = None
    physical_guard_passed: bool | None = None

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None and type(value) is not bool:
                raise ValueError(f"signal {field.name!r} must be a bool or null")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, object] | None
    ) -> ProviderFailureSignalsV1:
        if values is None:
            return cls()
        if not isinstance(values, Mapping):
            raise ValueError("signals must be a mapping")
        _require_literal_keys(values, name="signals")
        expected = {field.name for field in fields(cls)}
        extra = set(values) - expected
        if extra:
            raise ValueError(f"signals contain unknown fields: {sorted(extra)}")
        return cls(
            technical_valid=_optional_boolean(
                values.get("technical_valid"), name="technical_valid"
            ),
            provider_support_complete=_optional_boolean(
                values.get("provider_support_complete"),
                name="provider_support_complete",
            ),
            numerically_converged=_optional_boolean(
                values.get("numerically_converged"),
                name="numerically_converged",
            ),
            query_identifiable=_optional_boolean(
                values.get("query_identifiable"), name="query_identifiable"
            ),
            gauge_or_common_mode_consistent=_optional_boolean(
                values.get("gauge_or_common_mode_consistent"),
                name="gauge_or_common_mode_consistent",
            ),
            covariance_calibrated=_optional_boolean(
                values.get("covariance_calibrated"),
                name="covariance_calibrated",
            ),
            material_identity_reliable=_optional_boolean(
                values.get("material_identity_reliable"),
                name="material_identity_reliable",
            ),
            robust_support_sufficient=_optional_boolean(
                values.get("robust_support_sufficient"),
                name="robust_support_sufficient",
            ),
            physical_guard_passed=_optional_boolean(
                values.get("physical_guard_passed"),
                name="physical_guard_passed",
            ),
        )

    def to_dict(self) -> dict[str, bool | None]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True, slots=True)
class ProviderFailureEvidenceV1:
    """One frozen update decision and the source-only evidence available for it."""

    case_id: str
    accepted: bool
    result_reason: str
    signals: ProviderFailureSignalsV1
    metrics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if type(self.case_id) is not str or not self.case_id:
            raise ValueError("case_id must be a nonempty literal string")
        if type(self.accepted) is not bool:
            raise ValueError("accepted must be a bool")
        if type(self.result_reason) is not str or not self.result_reason:
            raise ValueError("result_reason must be a nonempty literal string")
        if not isinstance(self.signals, ProviderFailureSignalsV1):
            raise ValueError("signals must be ProviderFailureSignalsV1")
        object.__setattr__(
            self,
            "metrics",
            frozen_finite_json_mapping(self.metrics, name="metrics"),
        )

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> ProviderFailureEvidenceV1:
        if not isinstance(values, Mapping):
            raise ValueError("each provider-failure record must be a mapping")
        _require_literal_keys(values, name="provider-failure record")
        expected = {"case_id", "accepted", "result_reason", "signals", "metrics"}
        extra = set(values) - expected
        if extra:
            raise ValueError(
                f"provider-failure record contains unknown fields: {sorted(extra)}"
            )
        missing = {"case_id", "accepted", "result_reason"} - set(values)
        if missing:
            raise ValueError(
                f"provider-failure record is missing fields: {sorted(missing)}"
            )
        case_id = values["case_id"]
        accepted = values["accepted"]
        result_reason = values["result_reason"]
        if type(case_id) is not str or not case_id:
            raise ValueError("case_id must be a nonempty literal string")
        if type(accepted) is not bool:
            raise ValueError("accepted must be a bool")
        if type(result_reason) is not str or not result_reason:
            raise ValueError("result_reason must be a nonempty literal string")
        raw_signals = values.get("signals")
        if raw_signals is not None and not isinstance(raw_signals, Mapping):
            raise ValueError("signals must be a mapping or null")
        signals = ProviderFailureSignalsV1.from_mapping(raw_signals)
        metrics = values.get("metrics", {})
        if not isinstance(metrics, Mapping):
            raise ValueError("metrics must be a mapping")
        return cls(
            case_id=cast(str, case_id),
            accepted=cast(bool, accepted),
            result_reason=cast(str, result_reason),
            signals=signals,
            metrics=metrics,
        )


@dataclass(frozen=True, slots=True)
class ProviderFailureAttributionV1:
    """Deterministic primary and multi-cause attribution for one update."""

    case_id: str
    accepted: bool
    result_reason: str
    primary_category: str
    failed_categories: tuple[str, ...]
    explicit_failed_signals: tuple[str, ...]
    reason_derived_signal: str | None
    unresolved_signals: tuple[str, ...]
    classification_complete: bool
    metrics: Mapping[str, Any]

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "accepted": self.accepted,
            "result_reason": self.result_reason,
            "primary_category": self.primary_category,
            "failed_categories": list(self.failed_categories),
            "explicit_failed_signals": list(self.explicit_failed_signals),
            "reason_derived_signal": self.reason_derived_signal,
            "unresolved_signals": list(self.unresolved_signals),
            "classification_complete": self.classification_complete,
            "metrics": plain_json(self.metrics),
        }


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        plain_json(payload),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def decompose_provider_failure(
    evidence: ProviderFailureEvidenceV1,
) -> ProviderFailureAttributionV1:
    """Classify a frozen update without changing its acceptance decision."""

    signals = evidence.signals.to_dict()
    explicit_failed = tuple(
        signal for signal in _SIGNAL_TO_CATEGORY if signals[signal] is False
    )
    reason_derived = _REASON_TO_SIGNAL.get(evidence.result_reason)
    if reason_derived is not None and signals[reason_derived] is True:
        raise ValueError(
            f"case {evidence.case_id!r} declares {reason_derived}=true but "
            f"result_reason={evidence.result_reason!r} implies failure"
        )

    failed_signal_set = set(explicit_failed)
    if reason_derived is not None:
        failed_signal_set.add(reason_derived)
    failed_categories = tuple(
        category
        for signal, category in _SIGNAL_TO_CATEGORY.items()
        if signal in failed_signal_set
    )
    unresolved = tuple(signal for signal, value in signals.items() if value is None)

    if evidence.accepted:
        if failed_categories:
            raise ValueError(
                f"accepted case {evidence.case_id!r} contains failed gate evidence"
            )
        primary = ACCEPTED_CATEGORY
        complete = True
    elif failed_categories:
        primary = failed_categories[0]
        complete = True
    else:
        primary = UNRESOLVED_REJECTION_CATEGORY
        complete = False

    return ProviderFailureAttributionV1(
        case_id=evidence.case_id,
        accepted=evidence.accepted,
        result_reason=evidence.result_reason,
        primary_category=primary,
        failed_categories=failed_categories,
        explicit_failed_signals=explicit_failed,
        reason_derived_signal=reason_derived,
        unresolved_signals=unresolved,
        classification_complete=complete,
        metrics=evidence.metrics,
    )


def _parse_payload(
    payload: Mapping[str, object],
) -> tuple[str, tuple[ProviderFailureEvidenceV1, ...], Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        raise ValueError("provider-failure input must be a mapping")
    _require_literal_keys(payload, name="provider-failure input")
    expected = {"schema", "schema_version", "provider_id", "records", "metadata"}
    extra = set(payload) - expected
    if extra:
        raise ValueError(
            f"provider-failure input contains unknown fields: {sorted(extra)}"
        )
    if payload.get("schema") != PROVIDER_FAILURE_EVIDENCE_SCHEMA:
        raise ValueError("provider-failure input has an unsupported schema")
    if payload.get("schema_version") != PROVIDER_FAILURE_EVIDENCE_VERSION:
        raise ValueError("provider-failure input has an unsupported schema version")
    raw_provider_id = payload.get("provider_id")
    if type(raw_provider_id) is not str or not raw_provider_id:
        raise ValueError("provider_id must be a nonempty literal string")
    provider_id = cast(str, raw_provider_id)
    raw_records = payload.get("records")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        raise ValueError("records must be a sequence")
    records_list: list[ProviderFailureEvidenceV1] = []
    for record in raw_records:
        if not isinstance(record, Mapping):
            raise ValueError("each provider-failure record must be a mapping")
        records_list.append(ProviderFailureEvidenceV1.from_mapping(record))
    records = tuple(records_list)
    if not records:
        raise ValueError("records must not be empty")
    case_ids = [record.case_id for record in records]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case_id values must be unique")
    raw_metadata = payload.get("metadata")
    if raw_metadata is not None and not isinstance(raw_metadata, Mapping):
        raise ValueError("metadata must be a mapping or null")
    metadata = frozen_finite_json_mapping(raw_metadata, name="metadata")
    return provider_id, records, metadata


def analyze_provider_failure_evidence(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Build a content-addressed, equal-case failure decomposition report."""

    provider_id, records, metadata = _parse_payload(payload)
    attributions = tuple(decompose_provider_failure(record) for record in records)
    primary_categories = (*CLASSIFICATION_PRECEDENCE, UNRESOLVED_REJECTION_CATEGORY)
    primary_counts = {
        ACCEPTED_CATEGORY: sum(item.accepted for item in attributions),
        **{
            category: sum(item.primary_category == category for item in attributions)
            for category in primary_categories
        },
    }
    any_category_counts = {
        category: sum(category in item.failed_categories for item in attributions)
        for category in CLASSIFICATION_PRECEDENCE
    }
    accepted_count = primary_counts[ACCEPTED_CATEGORY]
    unresolved_count = primary_counts[UNRESOLVED_REJECTION_CATEGORY]
    report: dict[str, object] = {
        "schema": PROVIDER_FAILURE_REPORT_SCHEMA,
        "schema_version": PROVIDER_FAILURE_REPORT_VERSION,
        "provider_id": provider_id,
        "record_count": len(attributions),
        "accepted_count": accepted_count,
        "rejected_count": len(attributions) - accepted_count,
        "classified_rejection_count": (
            len(attributions) - accepted_count - unresolved_count
        ),
        "unresolved_rejection_count": unresolved_count,
        "primary_category_counts": primary_counts,
        "any_category_counts": any_category_counts,
        "classification_precedence": list(CLASSIFICATION_PRECEDENCE),
        "records": [item.to_dict() for item in attributions],
        "metadata": plain_json(metadata),
        "equal_case_weighting": True,
        "information_boundary": PROVIDER_FAILURE_INFORMATION_BOUNDARY,
        "input_content_sha256": _canonical_sha256(payload),
    }
    report["report_id"] = _canonical_sha256(report)
    return report


__all__ = [
    "ACCEPTED_CATEGORY",
    "CLASSIFICATION_PRECEDENCE",
    "PROVIDER_FAILURE_EVIDENCE_SCHEMA",
    "PROVIDER_FAILURE_EVIDENCE_VERSION",
    "PROVIDER_FAILURE_INFORMATION_BOUNDARY",
    "PROVIDER_FAILURE_REPORT_SCHEMA",
    "PROVIDER_FAILURE_REPORT_VERSION",
    "UNRESOLVED_REJECTION_CATEGORY",
    "ProviderFailureAttributionV1",
    "ProviderFailureEvidenceV1",
    "ProviderFailureSignalsV1",
    "analyze_provider_failure_evidence",
    "decompose_provider_failure",
]
