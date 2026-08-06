from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import bayesian_phystwin.deform360_calibration_observability_report as module
from bayesian_phystwin.deform360_calibration_bundle import (
    Deform360CohortUnitV1,
)
from bayesian_phystwin.deform360_calibration_execution import (
    Deform360Stage0SelectionV1,
)
from bayesian_phystwin.deform360_calibration_observability_report import (
    Deform360CalibrationObservabilityCaseV1,
    Deform360CalibrationObservabilityReportV1,
    build_deform360_calibration_observability_report,
    build_report_from_paths,
    load_deform360_calibration_observability_case,
    load_deform360_calibration_observability_report,
    save_deform360_calibration_observability_case,
    save_deform360_calibration_observability_report,
)

SELECTION_ID = "1" * 64
VISUAL_LOCK_ID = "2" * 64
RUN_RECORD_ID = "3" * 64
IMPLEMENTATION_REVISION = "4" * 40
SOURCE_REVISION = "5" * 40
PROCESSING_REVISION = "6" * 40
QUERY_ID = hashlib.sha256(b"physical-query-v1").hexdigest()


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _boundary(*, calibration_opened: bool = True) -> dict[str, bool]:
    return {
        "calibration_payloads_opened": calibration_opened,
        "confirmation_payloads_opened": False,
        "target_outcomes_used": False,
        "replacement_allowed": False,
    }


def _case(
    index: int,
    *,
    status: str = "evaluated",
    query: np.ndarray | None = None,
) -> Deform360CalibrationObservabilityCaseV1:
    stratum = "sheet" if index < 5 else "volumetric"
    object_id = f"calibration-{index:02d}"
    common: dict[str, Any] = {
        "selection_artifact_sha256": SELECTION_ID,
        "visual_provider_lock_id": VISUAL_LOCK_ID,
        "calibration_source_run_record_sha256": RUN_RECORD_ID,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "object_id": object_id,
        "episode_id": index,
        "stratum": stratum,
        "physical_query_id": QUERY_ID,
        "status": status,
        "source_artifacts": {
            f"sources/{object_id}/case-input.json": _digest(object_id)
        },
        "information_boundary": _boundary(
            calibration_opened=status == "evaluated"
        ),
    }
    if status == "evaluated":
        reference = np.diag([1.0, 2.0, 3.0])
        candidate = reference + np.diag([0.5, 0.25, 0.75])
        return Deform360CalibrationObservabilityCaseV1(
            **common,
            reference_state_artifact_id=_digest(f"reference-{index}"),
            candidate_state_artifact_id=_digest(f"candidate-{index}"),
            contact_anchor_artifact_id=_digest(f"anchor-{index}"),
            reference_marginal_precision=reference,
            candidate_marginal_precision=candidate,
            query_jacobian=np.eye(3) if query is None else query,
        )
    return Deform360CalibrationObservabilityCaseV1(
        **common,
        failure_reason="registered technical failure",
    )


def _copy_case(
    case: Deform360CalibrationObservabilityCaseV1,
    **changes: object,
) -> Deform360CalibrationObservabilityCaseV1:
    values: dict[str, object] = {
        "selection_artifact_sha256": case.selection_artifact_sha256,
        "visual_provider_lock_id": case.visual_provider_lock_id,
        "calibration_source_run_record_sha256": (
            case.calibration_source_run_record_sha256
        ),
        "implementation_revision": case.implementation_revision,
        "object_id": case.object_id,
        "episode_id": case.episode_id,
        "stratum": case.stratum,
        "physical_query_id": case.physical_query_id,
        "status": case.status,
        "source_artifacts": case.source_artifacts,
        "information_boundary": case.information_boundary,
        "reference_state_artifact_id": case.reference_state_artifact_id,
        "candidate_state_artifact_id": case.candidate_state_artifact_id,
        "contact_anchor_artifact_id": case.contact_anchor_artifact_id,
        "reference_marginal_precision": case.reference_marginal_precision,
        "candidate_marginal_precision": case.candidate_marginal_precision,
        "query_jacobian": case.query_jacobian,
        "failure_reason": case.failure_reason,
        "protocol_id": case.protocol_id,
    }
    values.update(changes)
    return Deform360CalibrationObservabilityCaseV1(**values)


def _cases(*, evaluated_sheet: int = 5, evaluated_volumetric: int = 5):
    result = []
    for index in range(10):
        within = index if index < 5 else index - 5
        limit = evaluated_sheet if index < 5 else evaluated_volumetric
        status = (
            "evaluated"
            if within < limit
            else "technical_failure_without_replacement"
        )
        result.append(_case(index, status=status))
    return tuple(result)


def _report(
    *,
    evaluated_sheet: int = 5,
    evaluated_volumetric: int = 5,
) -> Deform360CalibrationObservabilityReportV1:
    return Deform360CalibrationObservabilityReportV1(
        selection_artifact_sha256=SELECTION_ID,
        visual_provider_lock_id=VISUAL_LOCK_ID,
        calibration_source_run_record_sha256=RUN_RECORD_ID,
        calibration_source_revision=SOURCE_REVISION,
        implementation_revision=IMPLEMENTATION_REVISION,
        physical_query_id=QUERY_ID,
        cases=_cases(
            evaluated_sheet=evaluated_sheet,
            evaluated_volumetric=evaluated_volumetric,
        ),
        source_artifacts={"sources/report-inputs.json": "7" * 64},
        metadata={"protocol_stage": "calibration-only"},
    )


def _unit(index: int, *, confirmation: bool) -> Deform360CohortUnitV1:
    offset = index + (100 if confirmation else 0)
    object_id = f"{'confirmation' if confirmation else 'calibration'}-{index:02d}"
    return Deform360CohortUnitV1(
        object_id=object_id,
        episode_id=offset,
        stratum="sheet" if index < (6 if confirmation else 5) else "volumetric",
        metadata_path=f"raw/{object_id}/metadata.json",
        metadata_sha256=_digest(f"metadata-{object_id}"),
    )


def _selection() -> Deform360Stage0SelectionV1:
    calibration = tuple(_unit(index, confirmation=False) for index in range(10))
    confirmation = tuple(_unit(index, confirmation=True) for index in range(12))
    return Deform360Stage0SelectionV1(
        source_sha256="8" * 64,
        selection_artifact_sha256=SELECTION_ID,
        selection_sha256="9" * 64,
        content_selection_sha256="a" * 64,
        protocol_sha256="b" * 64,
        dataset_revision="c" * 40,
        processing_revision=PROCESSING_REVISION,
        implementation_revision="d" * 40,
        calibration_units=calibration,
        confirmation_units=confirmation,
    )


@dataclass(frozen=True)
class _VisualLock:
    artifact_id: str = VISUAL_LOCK_ID


def _source_run() -> dict[str, object]:
    return {
        "record_sha256": RUN_RECORD_ID,
        "source_revision": SOURCE_REVISION,
        "status": "succeeded",
        "exit_code": 0,
        "confirmation_boundary_verified": True,
        "confirmation_payloads_opened": False,
        "selection_artifact_sha256": SELECTION_ID,
        "visual_provider_lock_id": VISUAL_LOCK_ID,
        "support_gate": {"support_passed": True},
    }


def test_evaluated_case_recomputes_comparison_and_freezes_arrays() -> None:
    case = _case(0)

    assert case.comparison is not None
    assert case.comparison.mutual_information_gain_nats > 0.0
    assert case.to_record()["case_id"] == case.case_id
    with pytest.raises(ValueError):
        cast(np.ndarray, case.reference_marginal_precision)[0, 0] = 99.0


def test_case_round_trip_rejects_changed_derived_comparison(tmp_path: Path) -> None:
    case = _case(0)
    path = tmp_path / "case.json"
    save_deform360_calibration_observability_case(case, path)
    loaded = load_deform360_calibration_observability_case(path)
    assert loaded.case_id == case.case_id

    record = case.to_record()
    comparison = dict(cast(dict[str, object], record["comparison"]))
    comparison["mutual_information_gain_nats"] = 100.0
    record["comparison"] = comparison
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="comparison changed"):
        load_deform360_calibration_observability_case(path)


def test_technical_failure_is_retained_without_numerical_result() -> None:
    case = _case(2, status="technical_failure_without_replacement")

    assert case.comparison is None
    assert case.failure_reason == "registered technical failure"
    assert case.information_boundary["calibration_payloads_opened"] is False

    with pytest.raises(ValueError, match="must not carry numerical results"):
        Deform360CalibrationObservabilityCaseV1(
            selection_artifact_sha256=SELECTION_ID,
            visual_provider_lock_id=VISUAL_LOCK_ID,
            calibration_source_run_record_sha256=RUN_RECORD_ID,
            implementation_revision=IMPLEMENTATION_REVISION,
            object_id="broken",
            episode_id=0,
            stratum="sheet",
            physical_query_id=QUERY_ID,
            status="technical_failure_without_replacement",
            source_artifacts={"failure.json": "e" * 64},
            information_boundary=_boundary(calibration_opened=False),
            failure_reason="failure",
            query_jacobian=np.eye(3),
        )


def test_case_rejects_candidate_information_loss_and_forbidden_access() -> None:
    with pytest.raises(ValueError, match="candidate"):
        Deform360CalibrationObservabilityCaseV1(
            selection_artifact_sha256=SELECTION_ID,
            visual_provider_lock_id=VISUAL_LOCK_ID,
            calibration_source_run_record_sha256=RUN_RECORD_ID,
            implementation_revision=IMPLEMENTATION_REVISION,
            object_id="lossy",
            episode_id=0,
            stratum="sheet",
            physical_query_id=QUERY_ID,
            status="evaluated",
            source_artifacts={"case.json": "f" * 64},
            information_boundary=_boundary(),
            reference_state_artifact_id="0" * 64,
            candidate_state_artifact_id="1" * 64,
            contact_anchor_artifact_id="2" * 64,
            reference_marginal_precision=np.eye(2) * 2.0,
            candidate_marginal_precision=np.eye(2),
            query_jacobian=np.eye(2),
        )

    with pytest.raises(ValueError, match="confirmation payload"):
        _copy_case(
            _case(0),
            information_boundary={
                **_boundary(),
                "confirmation_payloads_opened": True,
            },
        )


def test_report_is_object_balanced_and_order_independent() -> None:
    report = _report()
    reverse = Deform360CalibrationObservabilityReportV1(
        selection_artifact_sha256=SELECTION_ID,
        visual_provider_lock_id=VISUAL_LOCK_ID,
        calibration_source_run_record_sha256=RUN_RECORD_ID,
        calibration_source_revision=SOURCE_REVISION,
        implementation_revision=IMPLEMENTATION_REVISION,
        physical_query_id=QUERY_ID,
        cases=tuple(reversed(_cases())),
        source_artifacts={"sources/report-inputs.json": "7" * 64},
        metadata={"protocol_stage": "calibration-only"},
    )

    assert report.report_id == reverse.report_id
    assert report.support_gate["support_passed"] is True
    assert report.overall["object_count"] == 10
    assert report.overall["evaluated_object_count"] == 10
    assert report.by_stratum["sheet"]["object_count"] == 5
    assert report.overall["mean_mutual_information_gain_nats"] > 0.0


def test_support_gate_retains_failures_without_replacement() -> None:
    supported = _report(evaluated_sheet=4, evaluated_volumetric=4)
    insufficient = _report(evaluated_sheet=3, evaluated_volumetric=5)

    assert supported.support_gate["evaluated_object_count"] == 8
    assert supported.support_gate["support_passed"] is True
    assert supported.status == "completed-supported-calibration-observability"
    assert insufficient.support_gate["evaluated_object_count"] == 8
    assert insufficient.support_gate["support_passed"] is False
    assert insufficient.status == "completed-insufficient-calibration-support"


def test_report_rejects_duplicates_query_drift_and_revision_drift() -> None:
    cases = list(_cases())
    cases[-1] = cases[0]
    with pytest.raises(ValueError, match="repeats a calibration unit"):
        Deform360CalibrationObservabilityReportV1(
            selection_artifact_sha256=SELECTION_ID,
            visual_provider_lock_id=VISUAL_LOCK_ID,
            calibration_source_run_record_sha256=RUN_RECORD_ID,
            calibration_source_revision=SOURCE_REVISION,
            implementation_revision=IMPLEMENTATION_REVISION,
            physical_query_id=QUERY_ID,
            cases=cases,
            source_artifacts={"source.json": "3" * 64},
        )

    cases = list(_cases())
    cases[4] = _case(4, query=np.asarray([[1.0, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="different query Jacobians"):
        Deform360CalibrationObservabilityReportV1(
            selection_artifact_sha256=SELECTION_ID,
            visual_provider_lock_id=VISUAL_LOCK_ID,
            calibration_source_run_record_sha256=RUN_RECORD_ID,
            calibration_source_revision=SOURCE_REVISION,
            implementation_revision=IMPLEMENTATION_REVISION,
            physical_query_id=QUERY_ID,
            cases=cases,
            source_artifacts={"source.json": "4" * 64},
        )

    cases = list(_cases())
    cases[0] = _copy_case(cases[0], implementation_revision="e" * 40)
    with pytest.raises(ValueError, match="implementation revision"):
        Deform360CalibrationObservabilityReportV1(
            selection_artifact_sha256=SELECTION_ID,
            visual_provider_lock_id=VISUAL_LOCK_ID,
            calibration_source_run_record_sha256=RUN_RECORD_ID,
            calibration_source_revision=SOURCE_REVISION,
            implementation_revision=IMPLEMENTATION_REVISION,
            physical_query_id=QUERY_ID,
            cases=cases,
            source_artifacts={"source.json": "5" * 64},
        )


def test_report_round_trip_and_tamper_rejection(tmp_path: Path) -> None:
    report = _report(evaluated_sheet=4, evaluated_volumetric=4)
    path = tmp_path / "report.json"
    save_deform360_calibration_observability_report(report, path)
    loaded = load_deform360_calibration_observability_report(path)
    assert loaded.report_id == report.report_id
    assert loaded.status == report.status

    record = report.to_record()
    overall = dict(cast(dict[str, object], record["overall"]))
    overall["mean_mutual_information_gain_nats"] = 999.0
    record["overall"] = overall
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="overall summary changed"):
        load_deform360_calibration_observability_report(path)


def test_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":"first","schema":"second"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_deform360_calibration_observability_report(path)


def test_builder_binds_source_and_analysis_revisions_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _selection()
    cases = []
    for index, unit in enumerate(selection.calibration_units):
        case = _case(index)
        cases.append(
            _copy_case(
                case,
                object_id=unit.object_id,
                episode_id=unit.episode_id,
                stratum=unit.stratum,
            )
        )
    source_run = _source_run()
    monkeypatch.setattr(
        module,
        "validate_deform360_calibration_source_run_record",
        lambda value: dict(value),
    )

    report = build_deform360_calibration_observability_report(
        selection,
        cast(Any, _VisualLock()),
        source_run,
        cases,
        implementation_revision=IMPLEMENTATION_REVISION,
        physical_query_id=QUERY_ID,
        source_artifacts={"source.json": "6" * 64},
    )

    assert report.calibration_source_revision == SOURCE_REVISION
    assert report.implementation_revision == IMPLEMENTATION_REVISION
    assert report.calibration_source_revision != report.implementation_revision


def test_builder_rejects_failed_source_run(monkeypatch: pytest.MonkeyPatch) -> None:
    source_run = _source_run()
    source_run["status"] = "failed"
    source_run["exit_code"] = 3
    monkeypatch.setattr(
        module,
        "validate_deform360_calibration_source_run_record",
        lambda value: dict(value),
    )

    with pytest.raises(ValueError, match="did not succeed"):
        build_deform360_calibration_observability_report(
            _selection(),
            cast(Any, _VisualLock()),
            source_run,
            _cases(),
            implementation_revision=IMPLEMENTATION_REVISION,
            physical_query_id=QUERY_ID,
            source_artifacts={"source.json": "7" * 64},
        )


def test_path_builder_hashes_exact_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _selection()
    case_paths = []
    for index, unit in enumerate(selection.calibration_units):
        original = _case(index)
        case = _copy_case(
            original,
            object_id=unit.object_id,
            episode_id=unit.episode_id,
            stratum=unit.stratum,
        )
        path = tmp_path / f"case-{index}.json"
        save_deform360_calibration_observability_case(case, path)
        case_paths.append(path)

    selection_path = tmp_path / "selection.json"
    protocol_path = tmp_path / "protocol.json"
    visual_path = tmp_path / "visual.json"
    run_path = tmp_path / "run.json"
    for path, payload in (
        (selection_path, {"kind": "selection"}),
        (protocol_path, {"kind": "protocol"}),
        (visual_path, {"kind": "visual"}),
        (run_path, {"kind": "run"}),
    ):
        path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        module,
        "load_deform360_stage0_selection",
        lambda path, protocol_path: selection,
    )
    monkeypatch.setattr(
        module,
        "load_deform360_visual_provider_lock",
        lambda path: _VisualLock(),
    )
    monkeypatch.setattr(
        module,
        "load_deform360_calibration_source_run_record",
        lambda path: _source_run(),
    )
    monkeypatch.setattr(
        module,
        "validate_deform360_calibration_source_run_record",
        lambda value: dict(value),
    )

    report = build_report_from_paths(
        selection_lock_path=selection_path,
        stage0_protocol_path=protocol_path,
        visual_provider_lock_path=visual_path,
        calibration_source_run_record_path=run_path,
        case_paths=case_paths,
        implementation_revision=IMPLEMENTATION_REVISION,
        physical_query_id=QUERY_ID,
    )

    assert report.report_id is not None
    assert len(report.source_artifacts) == 14
    assert report.support_gate["support_passed"] is True


def test_workflow_is_hosted_read_only_and_payload_free() -> None:
    path = Path(
        ".github/workflows/deform360-calibration-observability-report.yml"
    )
    text = path.read_text(encoding="utf-8")

    assert "contents: read" in text
    assert "contents: write" not in text
    assert "runs-on: ubuntu-latest" in text
    assert "self-hosted" not in text
    assert "git push" not in text
    assert "open_calibration_payloads" not in text
    assert "confirmation" not in text.lower()
