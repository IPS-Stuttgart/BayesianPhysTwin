from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import bayesian_phystwin.deform360_calibration_observability_binding as binding
from bayesian_phystwin.deform360_calibration_bundle import (
    DEFORM360_CALIBRATION_ROLES,
    Deform360CalibrationArtifactRefV1,
)
from bayesian_phystwin.deform360_calibration_execution import (
    DEFORM360_CALIBRATION_LEDGER_CASE_ID,
    build_deform360_calibration_execution_seal,
    load_deform360_stage0_selection,
)
from bayesian_phystwin.deform360_calibration_observability_binding import (
    DEFORM360_CALIBRATION_OBSERVABILITY_SOURCE_KEY,
    DEFORM360_CALIBRATION_SOURCE_RUN_RECORD_SOURCE_KEY,
    build_deform360_calibration_execution_seal_with_observability,
    build_deform360_confirmation_opening_authorization,
    load_deform360_confirmation_opening_authorization,
    save_deform360_confirmation_opening_authorization,
    validate_deform360_calibration_observability_binding,
)
from bayesian_phystwin.deform360_calibration_observability_report import (
    Deform360CalibrationObservabilityCaseV1,
    Deform360CalibrationObservabilityReportV1,
)
from bayesian_phystwin.deform360_visual_provider_lock import (
    Deform360VisualProviderLockV1,
)
from bayesian_phystwin.evidence_use_ledger import (
    EvidenceUseLedgerV1,
    EvidenceUseV1,
)

RUN_RECORD_ID = "3" * 64
RUN_FILE_SHA256 = "4" * 64
REPORT_FILE_SHA256 = "5" * 64
SOURCE_REVISION = "6" * 40
IMPLEMENTATION_REVISION = "7" * 40
QUERY_ID = hashlib.sha256(b"deform360-physical-query-v1").hexdigest()


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _stage0():
    return load_deform360_stage0_selection(
        _repository_root() / "protocols/locks/"
        "deform360_official_hub_visuotactile_v1_selection.json"
    )


def _provider() -> Deform360VisualProviderLockV1:
    return Deform360VisualProviderLockV1(
        provider_revision="1" * 40,
        provider_manifest_id="2" * 64,
        provider_attestation_sha256="3" * 64,
        motioncrafter_revision="4" * 40,
        model_set_id="5" * 64,
        root_seed=20260805,
        seed_policy="per-object-derived-seed-v1",
        window_size=25,
        overlap=8,
        height=320,
        width=640,
        storage_dtype="float32",
        initial_metric_frame_prior_id="6" * 64,
        additional_metric_anchor_policy="none",
        max_gauge_rank=64,
        minimum_retained_gauge_trace=0.999,
    )


def _run_record(stage0, provider) -> dict[str, object]:
    return {
        "record_sha256": RUN_RECORD_ID,
        "source_revision": SOURCE_REVISION,
        "status": "succeeded",
        "exit_code": 0,
        "confirmation_boundary_verified": True,
        "confirmation_payloads_opened": False,
        "selection_artifact_sha256": stage0.selection_artifact_sha256,
        "visual_provider_lock_id": provider.artifact_id,
        "support_gate": {"support_passed": True},
    }


def _case(unit, *, evaluated: bool = True):
    common: dict[str, Any] = {
        "selection_artifact_sha256": _stage0().selection_artifact_sha256,
        "visual_provider_lock_id": _provider().artifact_id,
        "calibration_source_run_record_sha256": RUN_RECORD_ID,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "object_id": unit.object_id,
        "episode_id": unit.episode_id,
        "stratum": unit.stratum,
        "physical_query_id": QUERY_ID,
        "status": (
            "evaluated" if evaluated else "technical_failure_without_replacement"
        ),
        "source_artifacts": {
            f"sources/cases/{unit.object_id}.json": hashlib.sha256(
                unit.object_id.encode("utf-8")
            ).hexdigest()
        },
        "information_boundary": {
            "calibration_payloads_opened": evaluated,
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
            "replacement_allowed": False,
        },
    }
    if not evaluated:
        return Deform360CalibrationObservabilityCaseV1(
            **common,
            failure_reason="registered technical failure",
        )
    reference = np.diag([1.0, 2.0, 3.0])
    candidate = reference + np.diag([0.5, 0.25, 0.75])
    return Deform360CalibrationObservabilityCaseV1(
        **common,
        reference_state_artifact_id=hashlib.sha256(
            f"reference-{unit.object_id}".encode()
        ).hexdigest(),
        candidate_state_artifact_id=hashlib.sha256(
            f"candidate-{unit.object_id}".encode()
        ).hexdigest(),
        contact_anchor_artifact_id=hashlib.sha256(
            f"anchor-{unit.object_id}".encode()
        ).hexdigest(),
        reference_marginal_precision=reference,
        candidate_marginal_precision=candidate,
        query_jacobian=np.eye(3),
    )


def _report(*, insufficient: bool = False):
    stage0 = _stage0()
    cases = []
    for index, unit in enumerate(stage0.calibration_units):
        evaluated = not (insufficient and unit.stratum == "sheet" and index < 2)
        cases.append(_case(unit, evaluated=evaluated))
    return Deform360CalibrationObservabilityReportV1(
        selection_artifact_sha256=stage0.selection_artifact_sha256,
        visual_provider_lock_id=_provider().artifact_id,
        calibration_source_run_record_sha256=RUN_RECORD_ID,
        calibration_source_revision=SOURCE_REVISION,
        implementation_revision=IMPLEMENTATION_REVISION,
        physical_query_id=QUERY_ID,
        cases=tuple(cases),
        source_artifacts={
            binding.DEFORM360_OBSERVABILITY_REPORT_RUN_RECORD_SOURCE_KEY: (
                RUN_FILE_SHA256
            )
        },
        metadata={"stage": "calibration-only"},
    )


def _artifacts(stage0, report_id: str):
    groups = tuple(unit.object_id for unit in stage0.calibration_units)
    return tuple(
        Deform360CalibrationArtifactRefV1(
            role=role,
            artifact_id=f"{index + 1:064x}",
            implementation_revision=IMPLEMENTATION_REVISION,
            selection_evidence_id=(
                report_id
                if role in binding.DEFORM360_OBSERVABILITY_BOUND_ROLES
                else f"{index + 101:064x}"
            ),
            selected_candidate_id=f"candidate-{index}",
            candidate_count=index + 2,
            calibration_group_ids=groups,
            source_artifacts={f"calibration/{role}.json": f"{index + 201:064x}"},
        )
        for index, role in enumerate(DEFORM360_CALIBRATION_ROLES)
    )


def _ledger(stage0) -> EvidenceUseLedgerV1:
    entries = tuple(
        EvidenceUseV1(
            evidence_artifact_id=f"{index + 301:064x}",
            raw_factor_id=f"{index + 401:064x}",
            raw_factor_sha256=f"{index + 501:064x}",
            source_repository="brownu/deform360",
            source_revision="b" * 40,
            source_artifacts={unit.metadata_path: unit.metadata_sha256},
            sensor_family="deform360-calibration-source",
            stream_id=f"{unit.object_id}:episode-{unit.episode_id}",
            clock_id="deform360-frame-clock",
            causal_frame_start=0,
            causal_frame_stop=10,
            correlation_group_ids=(f"group-{index}",),
            inference_role="calibration_only",
            metadata={"object_id": unit.object_id},
        )
        for index, unit in enumerate(stage0.calibration_units)
    )
    return EvidenceUseLedgerV1(
        protocol_id=stage0.protocol_id,
        case_id=DEFORM360_CALIBRATION_LEDGER_CASE_ID,
        causal_frame_stop=10,
        entries=entries,
    )


def _sources(stage0) -> dict[str, str]:
    sources = {
        "sources/stage0/selection.json": stage0.source_sha256,
        "sources/locks/visual-provider-lock.json": "8" * 64,
        "sources/calibration/evidence-use-ledger.json": "9" * 64,
        DEFORM360_CALIBRATION_SOURCE_RUN_RECORD_SOURCE_KEY: RUN_FILE_SHA256,
        DEFORM360_CALIBRATION_OBSERVABILITY_SOURCE_KEY: REPORT_FILE_SHA256,
    }
    for index, role in enumerate(DEFORM360_CALIBRATION_ROLES):
        sources[f"sources/calibration/artifacts/{role}.json"] = f"{index + 10:064x}"
    return sources


@pytest.fixture(autouse=True)
def _strict_run_record_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        binding,
        "validate_deform360_calibration_source_run_record",
        lambda value: dict(value),
    )


def _bound_products(*, low_level: bool = False):
    stage0 = _stage0()
    provider = _provider()
    report = _report()
    report_id = report.report_id
    assert report_id is not None
    artifacts = _artifacts(stage0, report_id)
    ledger = _ledger(stage0)
    values = {
        "stage0_selection": stage0,
        "visual_provider_lock": provider,
        "evidence_use_ledger": ledger,
        "calibration_artifacts": artifacts,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "source_artifacts": _sources(stage0),
    }
    if low_level:
        products = build_deform360_calibration_execution_seal(**values)
    else:
        products = build_deform360_calibration_execution_seal_with_observability(
            **values,
            calibration_source_run_record=_run_record(stage0, provider),
            calibration_observability_report=report,
            calibration_source_run_record_file_sha256=RUN_FILE_SHA256,
            calibration_observability_report_file_sha256=REPORT_FILE_SHA256,
        )
    return stage0, provider, ledger, report, artifacts, products


def test_bound_builder_and_authorization_round_trip(tmp_path: Path) -> None:
    stage0, provider, ledger, report, _artifacts_value, products = _bound_products()
    authorization = build_deform360_confirmation_opening_authorization(
        products,
        calibration_source_run_record=_run_record(stage0, provider),
        calibration_observability_report=report,
        calibration_source_run_record_file_sha256=RUN_FILE_SHA256,
        calibration_observability_report_file_sha256=REPORT_FILE_SHA256,
        stage0_selection=stage0,
        visual_provider_lock=provider,
        evidence_use_ledger=ledger,
    )

    assert authorization.confirmation_opening_token == (
        products.calibration_bundle.confirmation_opening_token
    )
    assert authorization.confirmation_payloads_opened is False
    assert authorization.target_outcomes_used is False
    path = tmp_path / "authorization.json"
    save_deform360_confirmation_opening_authorization(authorization, path)
    loaded = load_deform360_confirmation_opening_authorization(path)
    assert loaded == authorization
    with pytest.raises(FileExistsError):
        save_deform360_confirmation_opening_authorization(authorization, path)

    record = json.loads(path.read_text(encoding="utf-8"))
    record["calibration_observability_report_id"] = "f" * 64
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="authorization_id"):
        load_deform360_confirmation_opening_authorization(path)


def test_low_level_seal_can_be_authorized_only_after_strict_binding() -> None:
    stage0, provider, ledger, report, _artifacts_value, products = _bound_products(
        low_level=True
    )
    authorization = build_deform360_confirmation_opening_authorization(
        products,
        calibration_source_run_record=_run_record(stage0, provider),
        calibration_observability_report=report,
        calibration_source_run_record_file_sha256=RUN_FILE_SHA256,
        calibration_observability_report_file_sha256=REPORT_FILE_SHA256,
        stage0_selection=stage0,
        visual_provider_lock=provider,
        evidence_use_ledger=ledger,
    )
    assert authorization.authorization_id is not None


def test_binding_rejects_wrong_bytes_and_unbound_role() -> None:
    stage0, provider, _ledger_value, report, artifacts, _products = _bound_products()
    run_record = _run_record(stage0, provider)
    sources = _sources(stage0)
    sources[DEFORM360_CALIBRATION_OBSERVABILITY_SOURCE_KEY] = "f" * 64
    with pytest.raises(ValueError, match="observability-report bytes changed"):
        validate_deform360_calibration_observability_binding(
            report,
            run_record,
            stage0_selection=stage0,
            visual_provider_lock=provider,
            calibration_artifacts=artifacts,
            source_artifacts=sources,
            calibration_source_run_record_file_sha256=RUN_FILE_SHA256,
            calibration_observability_report_file_sha256=REPORT_FILE_SHA256,
        )

    changed = list(artifacts)
    index = next(
        index
        for index, artifact in enumerate(changed)
        if artifact.role == "physical_response_and_closure"
    )
    changed[index] = replace(changed[index], selection_evidence_id="e" * 64)
    with pytest.raises(ValueError, match="does not bind"):
        validate_deform360_calibration_observability_binding(
            report,
            run_record,
            stage0_selection=stage0,
            visual_provider_lock=provider,
            calibration_artifacts=tuple(changed),
            source_artifacts=_sources(stage0),
            calibration_source_run_record_file_sha256=RUN_FILE_SHA256,
            calibration_observability_report_file_sha256=REPORT_FILE_SHA256,
        )


def test_binding_rejects_failed_or_different_source_run() -> None:
    stage0, provider, _ledger_value, report, artifacts, _products = _bound_products()
    failed = _run_record(stage0, provider)
    failed["status"] = "failed"
    failed["exit_code"] = 1
    with pytest.raises(ValueError, match="did not succeed"):
        validate_deform360_calibration_observability_binding(
            report,
            failed,
            stage0_selection=stage0,
            visual_provider_lock=provider,
            calibration_artifacts=artifacts,
            source_artifacts=_sources(stage0),
            calibration_source_run_record_file_sha256=RUN_FILE_SHA256,
            calibration_observability_report_file_sha256=REPORT_FILE_SHA256,
        )

    different = _run_record(stage0, provider)
    different["record_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="another source run"):
        validate_deform360_calibration_observability_binding(
            report,
            different,
            stage0_selection=stage0,
            visual_provider_lock=provider,
            calibration_artifacts=artifacts,
            source_artifacts=_sources(stage0),
            calibration_source_run_record_file_sha256=RUN_FILE_SHA256,
            calibration_observability_report_file_sha256=REPORT_FILE_SHA256,
        )


def test_binding_rejects_insufficient_object_support() -> None:
    stage0 = _stage0()
    provider = _provider()
    report = _report(insufficient=True)
    report_id = report.report_id
    assert report_id is not None
    with pytest.raises(ValueError, match="did not complete with support"):
        validate_deform360_calibration_observability_binding(
            report,
            _run_record(stage0, provider),
            stage0_selection=stage0,
            visual_provider_lock=provider,
            calibration_artifacts=_artifacts(stage0, report_id),
            source_artifacts=_sources(stage0),
            calibration_source_run_record_file_sha256=RUN_FILE_SHA256,
            calibration_observability_report_file_sha256=REPORT_FILE_SHA256,
        )
