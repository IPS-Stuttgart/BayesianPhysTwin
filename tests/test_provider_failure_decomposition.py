from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from bayesian_phystwin.cli.provider_failure_decomposition import main
from bayesian_phystwin.provider_failure_decomposition import (
    CLASSIFICATION_PRECEDENCE,
    PROVIDER_FAILURE_EVIDENCE_SCHEMA,
    PROVIDER_FAILURE_EVIDENCE_VERSION,
    ProviderFailureEvidenceV1,
    ProviderFailureSignalsV1,
    analyze_provider_failure_evidence,
    decompose_provider_failure,
)

_SIGNAL_NAMES = (
    "technical_valid",
    "provider_support_complete",
    "numerically_converged",
    "query_identifiable",
    "gauge_or_common_mode_consistent",
    "covariance_calibrated",
    "material_identity_reliable",
    "robust_support_sufficient",
    "physical_guard_passed",
)


def _signals(**overrides: bool | None) -> dict[str, bool | None]:
    result: dict[str, bool | None] = {name: True for name in _SIGNAL_NAMES}
    result.update(overrides)
    return result


def _record(
    case_id: str,
    *,
    accepted: bool = False,
    result_reason: str = "rejected",
    signals: dict[str, bool | None] | None = None,
    metrics: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "accepted": accepted,
        "result_reason": result_reason,
        "signals": _signals() if signals is None else signals,
        "metrics": {} if metrics is None else metrics,
    }


def _payload(records: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema": PROVIDER_FAILURE_EVIDENCE_SCHEMA,
        "schema_version": PROVIDER_FAILURE_EVIDENCE_VERSION,
        "provider_id": "prob4d-provider-v2-source-lock",
        "records": records,
        "metadata": {"split": "source-only"},
    }


def test_report_decomposes_primary_and_multi_cause_failures() -> None:
    payload = _payload(
        [
            _record(
                "accepted",
                accepted=True,
                result_reason="strict-admission-passed",
                metrics={"guard_margin": 0.2},
            ),
            _record(
                "support",
                result_reason="no-observation-support",
                signals=_signals(provider_support_complete=False),
            ),
            _record(
                "identifiability",
                result_reason="no-identifiable-query-state",
                signals=_signals(query_identifiable=None),
            ),
            _record(
                "gauge-and-covariance",
                signals=_signals(
                    gauge_or_common_mode_consistent=False,
                    covariance_calibrated=False,
                ),
            ),
            _record(
                "fixed-point",
                result_reason="strict-v2-fixed-point-not-converged",
                signals=_signals(numerically_converged=None),
            ),
            _record("unresolved", signals=_signals()),
            _record(
                "physical-guard",
                signals=_signals(physical_guard_passed=False),
            ),
            _record("technical", signals=_signals(technical_valid=False)),
            _record(
                "identity",
                signals=_signals(material_identity_reliable=False),
            ),
            _record(
                "robust-support",
                signals=_signals(robust_support_sufficient=False),
            ),
        ]
    )

    report = analyze_provider_failure_evidence(payload)

    assert report["record_count"] == 10
    assert report["accepted_count"] == 1
    assert report["rejected_count"] == 9
    assert report["classified_rejection_count"] == 8
    assert report["unresolved_rejection_count"] == 1
    assert report["classification_precedence"] == list(CLASSIFICATION_PRECEDENCE)
    assert report["equal_case_weighting"] is True
    assert len(report["report_id"]) == 64
    records = {record["case_id"]: record for record in report["records"]}
    assert records["accepted"]["primary_category"] == "accepted"
    assert records["support"]["primary_category"] == (
        "unsupported-provider-geometry"
    )
    assert records["identifiability"]["primary_category"] == (
        "unidentifiable-physical-query"
    )
    assert records["identifiability"]["reason_derived_signal"] == (
        "query_identifiable"
    )
    assert records["gauge-and-covariance"]["failed_categories"] == [
        "coherent-gauge-or-common-mode-bias",
        "provider-covariance-miscalibration",
    ]
    assert records["fixed-point"]["primary_category"] == (
        "numerical-non-convergence"
    )
    assert records["unresolved"]["classification_complete"] is False
    assert report["any_category_counts"][
        "provider-covariance-miscalibration"
    ] == 1


def test_reason_derived_failure_rejects_explicit_pass_contradiction() -> None:
    evidence = ProviderFailureEvidenceV1(
        case_id="contradiction",
        accepted=False,
        result_reason="no-identifiable-query-state",
        signals=ProviderFailureSignalsV1(query_identifiable=True),
        metrics={},
    )
    with pytest.raises(ValueError, match="implies failure"):
        decompose_provider_failure(evidence)


def test_accepted_record_rejects_failed_gate_evidence() -> None:
    evidence = ProviderFailureEvidenceV1(
        case_id="accepted-contradiction",
        accepted=True,
        result_reason="accepted",
        signals=ProviderFailureSignalsV1(covariance_calibrated=False),
        metrics={},
    )
    with pytest.raises(ValueError, match="accepted case"):
        decompose_provider_failure(evidence)


@pytest.mark.parametrize(
    "payload, message",
    [
        ({}, "unsupported schema"),
        (
            _payload([_record("duplicate"), _record("duplicate")]),
            "case_id values must be unique",
        ),
        (
            _payload([_record("invalid", accepted=1)]),
            "accepted must be a bool",
        ),
        (
            _payload(
                [
                    _record(
                        "invalid-signal",
                        signals={**_signals(), "technical_valid": 1},
                    )
                ]
            ),
            "must be a bool or null",
        ),
        (
            _payload([_record("invalid-metric", metrics={"score": float("nan")})]),
            "finite JSON values",
        ),
    ],
)
def test_invalid_evidence_fails_closed(
    payload: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        analyze_provider_failure_evidence(payload)


def test_report_identity_is_canonical_and_input_sensitive() -> None:
    first = _payload([_record("case", metrics={"b": 2, "a": 1})])
    second = deepcopy(first)
    second["metadata"] = {"split": "source-only"}
    second_records = second["records"]
    assert isinstance(second_records, list)
    second_records[0]["metrics"] = {"a": 1, "b": 2}

    first_report = analyze_provider_failure_evidence(first)
    second_report = analyze_provider_failure_evidence(second)
    assert first_report["report_id"] == second_report["report_id"]
    assert first_report["input_content_sha256"] == second_report[
        "input_content_sha256"
    ]

    second_records[0]["metrics"] = {"a": 1, "b": 3}
    changed = analyze_provider_failure_evidence(second)
    assert changed["report_id"] != first_report["report_id"]


def test_cli_writes_atomically_and_requires_explicit_overwrite(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "report.json"
    input_path.write_text(
        json.dumps(_payload([_record("case")]), sort_keys=True),
        encoding="utf-8",
    )

    assert main([str(input_path), str(output_path)]) == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["input_artifact"]["sha256"]
    assert report["unresolved_rejection_count"] == 1

    with pytest.raises(FileExistsError):
        main([str(input_path), str(output_path)])
    assert main([str(input_path), str(output_path), "--overwrite"]) == 0
