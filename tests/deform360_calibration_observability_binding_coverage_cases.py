"""Adversarial coverage cases imported by the stable-core Deform360 suite."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import bayesian_phystwin.deform360_calibration_observability_binding as binding
from bayesian_phystwin.deform360_calibration_observability_binding import (
    DEFORM360_CALIBRATION_OBSERVABILITY_SOURCE_KEY,
    DEFORM360_CALIBRATION_SOURCE_RUN_RECORD_SOURCE_KEY,
    Deform360ConfirmationOpeningAuthorizationV1,
    save_deform360_confirmation_opening_authorization,
    validate_deform360_calibration_observability_binding,
)
from test_deform360_calibration_observability_binding import (
    REPORT_FILE_SHA256,
    RUN_FILE_SHA256,
    _artifacts,
    _bound_products,
    _ledger,
    _provider,
    _report,
    _run_record,
    _sources,
    _stage0,
)


def _stub_run_record_validator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        binding,
        "validate_deform360_calibration_source_run_record",
        lambda value: dict(value),
    )


def _context(monkeypatch: pytest.MonkeyPatch):
    _stub_run_record_validator(monkeypatch)
    stage0 = _stage0()
    provider = _provider()
    report = _report()
    report_id = report.report_id
    assert report_id is not None
    return (
        stage0,
        provider,
        report,
        _artifacts(stage0, report_id),
        _run_record(stage0, provider),
        _sources(stage0),
    )


def _validate(
    report: object,
    run_record: object,
    *,
    stage0: object,
    provider: object,
    artifacts: object,
    sources: object,
) -> None:
    validate_deform360_calibration_observability_binding(
        report,  # type: ignore[arg-type]
        run_record,  # type: ignore[arg-type]
        stage0_selection=stage0,  # type: ignore[arg-type]
        visual_provider_lock=provider,  # type: ignore[arg-type]
        calibration_artifacts=artifacts,  # type: ignore[arg-type]
        source_artifacts=sources,  # type: ignore[arg-type]
        calibration_source_run_record_file_sha256=RUN_FILE_SHA256,
        calibration_observability_report_file_sha256=REPORT_FILE_SHA256,
    )


def _authorization(monkeypatch: pytest.MonkeyPatch):
    _stub_run_record_validator(monkeypatch)
    stage0, provider, ledger, report, _artifacts_value, products = _bound_products()
    authorization = binding.build_deform360_confirmation_opening_authorization(
        products,
        calibration_source_run_record=_run_record(stage0, provider),
        calibration_observability_report=report,
        calibration_source_run_record_file_sha256=RUN_FILE_SHA256,
        calibration_observability_report_file_sha256=REPORT_FILE_SHA256,
        stage0_selection=stage0,
        visual_provider_lock=provider,
        evidence_use_ledger=ledger,
    )
    return stage0, provider, ledger, report, products, authorization


def test_binding_rejects_missing_report_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage0, provider, report, artifacts, run_record, sources = _context(monkeypatch)
    object.__setattr__(report, "report_id", None)

    with pytest.raises(ValueError, match="lacks a report_id"):
        _validate(
            report,
            run_record,
            stage0=stage0,
            provider=provider,
            artifacts=artifacts,
            sources=sources,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "confirmation_boundary_verified",
            False,
            "confirmation boundary is unverified",
        ),
        (
            "confirmation_payloads_opened",
            True,
            "confirmation payload access",
        ),
        ("support_gate", None, "support gate did not pass"),
    ],
)
def test_binding_rejects_invalid_run_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    stage0, provider, report, artifacts, run_record, sources = _context(monkeypatch)
    run_record[field] = value

    with pytest.raises(ValueError, match=message):
        _validate(
            report,
            run_record,
            stage0=stage0,
            provider=provider,
            artifacts=artifacts,
            sources=sources,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("report", "calibration_observability_report"),
        ("stage0", "stage0_selection"),
        ("provider", "visual_provider_lock"),
        ("run", "calibration_source_run_record"),
    ],
)
def test_binding_rejects_wrong_input_types(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    stage0, provider, report, artifacts, run_record, sources = _context(monkeypatch)
    values: dict[str, object] = {
        "report": report,
        "stage0": stage0,
        "provider": provider,
        "run": run_record,
    }
    values[case] = object()

    with pytest.raises(TypeError, match=message):
        _validate(
            values["report"],
            values["run"],
            stage0=values["stage0"],
            provider=values["provider"],
            artifacts=artifacts,
            sources=sources,
        )


def test_binding_rejects_missing_retained_source_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage0, provider, report, artifacts, run_record, sources = _context(monkeypatch)
    sources.pop(DEFORM360_CALIBRATION_SOURCE_RUN_RECORD_SOURCE_KEY)

    with pytest.raises(ValueError, match="run record bytes are not retained"):
        _validate(
            report,
            run_record,
            stage0=stage0,
            provider=provider,
            artifacts=artifacts,
            sources=sources,
        )

    report = replace(
        _report(),
        source_artifacts={"sources/other.json": "a" * 64},
        report_id=None,
    )
    report_id = report.report_id
    assert report_id is not None
    with pytest.raises(ValueError, match="calibration source run record bytes"):
        _validate(
            report,
            run_record,
            stage0=stage0,
            provider=provider,
            artifacts=_artifacts(stage0, report_id),
            sources=_sources(stage0),
        )


def test_binding_rejects_invalid_artifact_collections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage0, provider, report, artifacts, run_record, sources = _context(monkeypatch)

    with pytest.raises(ValueError, match="must be a sequence"):
        _validate(
            report,
            run_record,
            stage0=stage0,
            provider=provider,
            artifacts="not-artifacts",
            sources=sources,
        )

    with pytest.raises(ValueError, match="unsupported value"):
        _validate(
            report,
            run_record,
            stage0=stage0,
            provider=provider,
            artifacts=(*artifacts[:-1], object()),
            sources=sources,
        )

    with pytest.raises(ValueError, match="repeat a role"):
        _validate(
            report,
            run_record,
            stage0=stage0,
            provider=provider,
            artifacts=(*artifacts, artifacts[0]),
            sources=sources,
        )

    missing = tuple(
        artifact
        for artifact in artifacts
        if artifact.role != binding.DEFORM360_OBSERVABILITY_BOUND_ROLES[0]
    )
    with pytest.raises(ValueError, match="roles are missing"):
        _validate(
            report,
            run_record,
            stage0=stage0,
            provider=provider,
            artifacts=missing,
            sources=sources,
        )


def test_builder_rejects_reserved_or_nonmapping_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage0, provider, report, artifacts, run_record, sources = _context(monkeypatch)
    ledger = _ledger(stage0)
    common: dict[str, Any] = {
        "stage0_selection": stage0,
        "visual_provider_lock": provider,
        "evidence_use_ledger": ledger,
        "calibration_artifacts": artifacts,
        "calibration_source_run_record": run_record,
        "calibration_observability_report": report,
        "calibration_source_run_record_file_sha256": RUN_FILE_SHA256,
        "calibration_observability_report_file_sha256": REPORT_FILE_SHA256,
        "implementation_revision": report.implementation_revision,
        "source_artifacts": sources,
    }

    with pytest.raises(ValueError, match="reserves observability fields"):
        binding.build_deform360_calibration_execution_seal_with_observability(
            **common,
            metadata={"calibration_observability_report_id": "reserved"},
        )

    monkeypatch.setattr(binding, "plain_json", lambda value: [])
    with pytest.raises(ValueError, match="metadata must be a mapping"):
        binding.build_deform360_calibration_execution_seal_with_observability(
            **common,
            metadata={},
        )


def test_verifier_rejects_partial_or_changed_binding_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage0, provider, ledger, report, _artifacts_value, products = _bound_products()
    run_record = _run_record(stage0, provider)
    seal = products.execution_seal
    partial = replace(
        seal,
        metadata={"calibration_observability_report_id": report.report_id},
    )
    partial_products = products._replace(execution_seal=partial)

    with pytest.raises(ValueError, match="incomplete observability metadata"):
        binding.verify_deform360_calibration_execution_observability_binding(
            partial_products,
            calibration_source_run_record=run_record,
            calibration_observability_report=report,
            calibration_source_run_record_file_sha256=RUN_FILE_SHA256,
            calibration_observability_report_file_sha256=REPORT_FILE_SHA256,
            stage0_selection=stage0,
            visual_provider_lock=provider,
            evidence_use_ledger=ledger,
        )

    changed_metadata = dict(seal.metadata)
    changed_metadata["calibration_observability_report_id"] = "f" * 64
    changed = replace(seal, metadata=changed_metadata)
    changed_products = products._replace(execution_seal=changed)
    with pytest.raises(ValueError, match="metadata changed"):
        binding.verify_deform360_calibration_execution_observability_binding(
            changed_products,
            calibration_source_run_record=run_record,
            calibration_observability_report=report,
            calibration_source_run_record_file_sha256=RUN_FILE_SHA256,
            calibration_observability_report_file_sha256=REPORT_FILE_SHA256,
            stage0_selection=stage0,
            visual_provider_lock=provider,
            evidence_use_ledger=ledger,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("root", "must be a JSON object"),
        ("schema", "schema changed"),
        ("version", "schema_version changed"),
        ("semantics", "semantics changed"),
        ("claim", "claim boundary changed"),
        ("sources", "source_artifacts must be a JSON object"),
        ("metadata", "metadata must be a JSON object"),
    ],
)
def test_authorization_loader_rejects_malformed_contracts(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    *_context_values, authorization = _authorization(monkeypatch)
    record: object = authorization.to_record()
    if case == "root":
        record = []
    else:
        assert isinstance(record, dict)
        if case == "schema":
            record["schema"] = "changed"
        elif case == "version":
            record["schema_version"] = 2
        elif case == "semantics":
            record["semantics"] = "changed"
        elif case == "claim":
            record["claim_boundary"] = "changed"
        elif case == "sources":
            record["source_artifacts"] = []
        elif case == "metadata":
            record["metadata"] = []

    with pytest.raises(ValueError, match=message):
        Deform360ConfirmationOpeningAuthorizationV1.from_mapping(record)


def test_authorization_constructor_rejects_invalid_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    *_context_values, authorization = _authorization(monkeypatch)

    with pytest.raises(ValueError, match="status changed"):
        replace(authorization, status="changed", authorization_id=None)

    sources = dict(authorization.source_artifacts)
    sources[DEFORM360_CALIBRATION_OBSERVABILITY_SOURCE_KEY] = "f" * 64
    with pytest.raises(ValueError, match="source bytes changed"):
        replace(authorization, source_artifacts=sources, authorization_id=None)

    with pytest.raises(ValueError, match="opened before authorization"):
        replace(
            authorization,
            confirmation_payloads_opened=True,
            authorization_id=None,
        )

    with pytest.raises(ValueError, match="used before authorization"):
        replace(authorization, target_outcomes_used=True, authorization_id=None)

    with pytest.raises(ValueError, match="does not match content"):
        replace(authorization, authorization_id="f" * 64)


def test_authorization_writer_rejects_wrong_type(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="Deform360ConfirmationOpeningAuthorizationV1"):
        save_deform360_confirmation_opening_authorization(
            object(),  # type: ignore[arg-type]
            tmp_path / "authorization.json",
        )
