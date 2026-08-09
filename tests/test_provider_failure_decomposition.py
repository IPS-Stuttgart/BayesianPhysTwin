from __future__ import annotations

import json
import runpy
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

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
from bayesian_phystwin.provider_failure_report_io import (
    canonical_json_sha256,
    load_provider_failure_input,
    publish_provider_failure_report,
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

    report = cast(dict[str, Any], analyze_provider_failure_evidence(payload))

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
    assert records["support"]["primary_category"] == ("unsupported-provider-geometry")
    assert records["identifiability"]["primary_category"] == (
        "unidentifiable-physical-query"
    )
    assert records["identifiability"]["reason_derived_signal"] == ("query_identifiable")
    assert records["gauge-and-covariance"]["failed_categories"] == [
        "coherent-gauge-or-common-mode-bias",
        "provider-covariance-miscalibration",
    ]
    assert records["fixed-point"]["primary_category"] == ("numerical-non-convergence")
    assert records["unresolved"]["classification_complete"] is False
    assert report["any_category_counts"]["provider-covariance-miscalibration"] == 1


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
            _payload([_record("invalid", accepted=cast(Any, 1))]),
            "accepted must be a bool",
        ),
        (
            _payload(
                [
                    _record(
                        "invalid-signal",
                        signals={**_signals(), "technical_valid": cast(Any, 1)},
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

    first_report = cast(dict[str, Any], analyze_provider_failure_evidence(first))
    second_report = cast(dict[str, Any], analyze_provider_failure_evidence(second))
    assert first_report["report_id"] == second_report["report_id"]
    assert first_report["input_content_sha256"] == second_report["input_content_sha256"]

    second_records[0]["metrics"] = {"a": 1, "b": 3}
    changed = cast(dict[str, Any], analyze_provider_failure_evidence(second))
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
    report = cast(
        dict[str, Any],
        json.loads(output_path.read_text(encoding="utf-8")),
    )
    assert report["input_artifact"]["sha256"]
    assert report["unresolved_rejection_count"] == 1
    status_sha256 = report.pop("status_sha256")
    assert status_sha256 == canonical_json_sha256(report)

    with pytest.raises(FileExistsError):
        main([str(input_path), str(output_path)])
    assert main([str(input_path), str(output_path), "--overwrite"]) == 0


def test_cli_rejects_duplicate_keys_and_nonfinite_constants(tmp_path: Path) -> None:
    output_path = tmp_path / "report.json"
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(
        '{"schema":"first","schema":"second"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate key"):
        main([str(duplicate_path), str(output_path)])

    nonfinite_path = tmp_path / "nonfinite.json"
    nonfinite_path.write_text(
        '{"schema":NaN}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-finite constant"):
        main([str(nonfinite_path), str(output_path)])


def test_cli_rejects_non_object_and_invalid_json(tmp_path: Path) -> None:
    output_path = tmp_path / "report.json"
    array_path = tmp_path / "array.json"
    array_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        main([str(array_path), str(output_path)])

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        main([str(invalid_path), str(output_path)])


def test_strict_input_rejects_duplicate_keys_and_nonfinite_constants(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        load_provider_failure_input(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite constant"):
        load_provider_failure_input(nonfinite)


def test_strict_input_binds_raw_bytes_and_enforces_budget(tmp_path: Path) -> None:
    path = tmp_path / "input.json"
    path.write_text('{"a":1}', encoding="utf-8")
    payload, artifact = load_provider_failure_input(path)
    assert payload == {"a": 1}
    assert artifact["path"] == str(path.resolve())
    assert artifact["bytes"] == path.stat().st_size
    assert len(cast(str, artifact["sha256"])) == 64

    with pytest.raises(ValueError, match="byte budget"):
        load_provider_failure_input(path, maximum_input_bytes=1)
    with pytest.raises(TypeError, match="genuine integer"):
        load_provider_failure_input(path, maximum_input_bytes=True)
    with pytest.raises(ValueError, match="positive"):
        load_provider_failure_input(path, maximum_input_bytes=0)


def test_atomic_publication_adds_host_local_status_and_no_clobber(
    tmp_path: Path,
) -> None:
    output = tmp_path / "report.json"
    report = {"schema": "example", "report_id": "0" * 64, "value": 3}
    artifact = {"path": "/source/input.json", "sha256": "1" * 64, "bytes": 7}

    emitted = publish_provider_failure_report(
        output,
        report,
        input_artifact=artifact,
    )
    loaded = cast(dict[str, Any], json.loads(output.read_text(encoding="utf-8")))
    assert loaded == emitted
    status = loaded.pop("status_sha256")
    assert status == canonical_json_sha256(loaded)

    with pytest.raises(FileExistsError):
        publish_provider_failure_report(
            output,
            report,
            input_artifact=artifact,
        )
    replaced = publish_provider_failure_report(
        output,
        {**report, "value": 4},
        input_artifact=artifact,
        overwrite=True,
    )
    assert replaced["value"] == 4


def test_publication_rejects_owned_fields_and_invalid_overwrite(
    tmp_path: Path,
) -> None:
    output = tmp_path / "report.json"
    with pytest.raises(ValueError, match="publication-owned fields"):
        publish_provider_failure_report(
            output,
            {"input_artifact": {}},
            input_artifact={},
        )
    with pytest.raises(TypeError, match="overwrite must be a bool"):
        publish_provider_failure_report(
            output,
            {},
            input_artifact={},
            overwrite=cast(Any, 1),
        )


def test_signal_contract_boundary_paths() -> None:
    assert ProviderFailureSignalsV1.from_mapping(None) == ProviderFailureSignalsV1()
    with pytest.raises(ValueError, match="signals must be a mapping"):
        ProviderFailureSignalsV1.from_mapping(cast(Any, []))
    with pytest.raises(ValueError, match="literal string keys"):
        ProviderFailureSignalsV1.from_mapping(cast(Any, {1: True}))
    with pytest.raises(ValueError, match="unknown fields"):
        ProviderFailureSignalsV1.from_mapping({"unknown": True})
    with pytest.raises(ValueError, match="must be a bool or null"):
        ProviderFailureSignalsV1(technical_valid=cast(Any, 1))


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"case_id": cast(Any, 1)}, "case_id"),
        ({"accepted": cast(Any, 1)}, "accepted"),
        ({"result_reason": cast(Any, 1)}, "result_reason"),
        ({"signals": cast(Any, {})}, "signals must be ProviderFailureSignalsV1"),
    ],
)
def test_direct_evidence_contract_boundary_paths(
    updates: dict[str, Any], message: str
) -> None:
    arguments: dict[str, Any] = {
        "case_id": "case",
        "accepted": False,
        "result_reason": "rejected",
        "signals": ProviderFailureSignalsV1(),
        "metrics": {},
    }
    arguments.update(updates)
    with pytest.raises(ValueError, match=message):
        ProviderFailureEvidenceV1(**arguments)


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (cast(Any, []), "must be a mapping"),
        ({**_record("case"), "extra": True}, "unknown fields"),
        ({"case_id": "case"}, "missing fields"),
        ({**_record("case"), "case_id": 1}, "case_id"),
        ({**_record("case"), "accepted": 1}, "accepted"),
        ({**_record("case"), "result_reason": 1}, "result_reason"),
        ({**_record("case"), "signals": []}, "signals must be a mapping"),
        ({**_record("case"), "metrics": []}, "metrics must be a mapping"),
    ],
)
def test_record_mapping_boundary_paths(record: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ProviderFailureEvidenceV1.from_mapping(record)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (cast(Any, []), "must be a mapping"),
        (cast(Any, {1: True}), "literal string keys"),
        ({**_payload([_record("case")]), "extra": True}, "unknown fields"),
        (
            {**_payload([_record("case")]), "schema_version": 99},
            "unsupported schema version",
        ),
        ({**_payload([_record("case")]), "provider_id": 1}, "provider_id"),
        ({**_payload([_record("case")]), "records": "case"}, "records must"),
        ({**_payload([_record("case")]), "records": [1]}, "record must be a mapping"),
        ({**_payload([_record("case")]), "records": []}, "must not be empty"),
        ({**_payload([_record("case")]), "metadata": []}, "metadata must"),
    ],
)
def test_payload_boundary_paths(payload: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        analyze_provider_failure_evidence(payload)


def test_strict_input_rejects_nonordinary_and_non_utf8_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ordinary file"):
        load_provider_failure_input(tmp_path)

    invalid_utf8 = tmp_path / "invalid-utf8.json"
    invalid_utf8.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="UTF-8 JSON"):
        load_provider_failure_input(invalid_utf8)


def test_cli_module_entrypoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "report.json"
    input_path.write_text(
        json.dumps(_payload([_record("case")]), sort_keys=True),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "provider_failure_decomposition",
            str(input_path),
            str(output_path),
        ],
    )
    with pytest.raises(SystemExit) as error:
        runpy.run_module(
            "bayesian_phystwin.cli.provider_failure_decomposition",
            run_name="__main__",
        )
    assert error.value.code == 0
    assert output_path.is_file()
