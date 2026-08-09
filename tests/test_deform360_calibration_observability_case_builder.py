"""Contracts for claim-bearing Deform360 observability case production."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import test_deform360_calibration_source_run_record as source_run_cases

from bayesian_phystwin.deform360_calibration_observability_case_builder import (
    build_evaluated_case_from_paths,
    build_technical_failure_case_from_paths,
    physical_query_id_from_path,
)
from bayesian_phystwin.deform360_calibration_observability_report import (
    load_deform360_calibration_observability_case,
    save_deform360_calibration_observability_case,
)
from bayesian_phystwin.deform360_calibration_source_run_record import (
    save_deform360_calibration_source_run_record,
)

IMPLEMENTATION_REVISION = "7" * 40
ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = ROOT / "scripts/science/build_deform360_calibration_observability_case.py"
SPEC = importlib.util.spec_from_file_location("deform360_case_cli", CLI_PATH)
assert SPEC is not None and SPEC.loader is not None
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


@dataclass
class Inputs:
    chain: source_run_cases.Chain
    run_record_path: Path
    reference_path: Path
    candidate_path: Path
    query_path: Path
    anchor_path: Path
    failure_evidence_path: Path


def _inputs(
    tmp_path: Path,
    *,
    planned_sheet: int = 5,
    planned_volumetric: int = 5,
    prepared_sheet: int = 5,
    prepared_volumetric: int = 5,
) -> Inputs:
    chain = source_run_cases._build_chain(
        tmp_path / "chain",
        planned_sheet=planned_sheet,
        planned_volumetric=planned_volumetric,
        prepared_sheet=prepared_sheet,
        prepared_volumetric=prepared_volumetric,
    )
    run_record_path = tmp_path / "execution-manifest.json"
    save_deform360_calibration_source_run_record(
        source_run_cases._record(chain),
        run_record_path,
    )
    reference_path = tmp_path / "reference.npy"
    candidate_path = tmp_path / "candidate.npy"
    query_path = tmp_path / "query.npy"
    np.save(reference_path, np.diag([1.0, 2.0, 3.0]), allow_pickle=False)
    np.save(candidate_path, np.diag([1.5, 2.25, 3.75]), allow_pickle=False)
    np.save(query_path, np.eye(3), allow_pickle=False)
    anchor_path = tmp_path / "contact-anchor.json"
    anchor_path.write_text(
        '{"anchor":"calibration-only"}\n',
        encoding="utf-8",
    )
    failure_evidence_path = tmp_path / "failure.txt"
    failure_evidence_path.write_text(
        "retained technical failure\n",
        encoding="utf-8",
    )
    return Inputs(
        chain=chain,
        run_record_path=run_record_path,
        reference_path=reference_path,
        candidate_path=candidate_path,
        query_path=query_path,
        anchor_path=anchor_path,
        failure_evidence_path=failure_evidence_path,
    )


def _common(inputs: Inputs, *, object_id: str) -> dict[str, Any]:
    return {
        "source_protocol_path": inputs.chain.source_protocol_path,
        "stage0_protocol_path": inputs.chain.stage0_protocol_path,
        "selection_lock_path": inputs.chain.selection_path,
        "visual_provider_lock_path": inputs.chain.provider_path,
        "calibration_source_plan_path": inputs.chain.plan_path,
        "calibration_source_download_path": inputs.chain.download_path,
        "calibration_source_run_record_path": inputs.run_record_path,
        "calibration_source_result_path": inputs.chain.result_path,
        "object_id": object_id,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "query_jacobian_path": inputs.query_path,
    }


def _evaluated(inputs: Inputs, *, object_id: str = "cal-sheet-0"):
    return build_evaluated_case_from_paths(
        **_common(inputs, object_id=object_id),
        reference_marginal_precision_path=inputs.reference_path,
        candidate_marginal_precision_path=inputs.candidate_path,
        contact_anchor_artifact_path=inputs.anchor_path,
    )


def test_evaluated_builder_binds_exact_lineage_and_round_trips(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)

    case = _evaluated(inputs)

    assert case.status == "evaluated"
    assert case.object_id == "cal-sheet-0"
    assert case.physical_query_id == physical_query_id_from_path(inputs.query_path)
    assert case.comparison is not None
    assert case.comparison.mutual_information_gain_nats > 0.0
    assert case.reference_state_artifact_id != case.candidate_state_artifact_id
    assert case.contact_anchor_artifact_id not in {
        case.reference_state_artifact_id,
        case.candidate_state_artifact_id,
    }
    assert "sources/calibration-source/download.json" in case.source_artifacts
    serialized = json.dumps(case.to_record(), sort_keys=True)
    assert str(tmp_path) not in serialized
    assert "confirm-sheet" not in serialized

    output = tmp_path / "case.json"
    save_deform360_calibration_observability_case(case, output)
    assert load_deform360_calibration_observability_case(output).case_id == (
        case.case_id
    )


def test_builder_rejects_confirmation_and_unknown_objects(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)

    with pytest.raises(ValueError, match="confirmation objects are forbidden"):
        _evaluated(inputs, object_id="confirm-sheet-0")
    with pytest.raises(ValueError, match="not in the frozen calibration cohort"):
        _evaluated(inputs, object_id="not-selected")


def test_evaluated_builder_requires_source_prepared_object(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, prepared_sheet=4)

    with pytest.raises(ValueError, match="requires a source_prepared object"):
        _evaluated(inputs, object_id="cal-sheet-4")


@pytest.mark.parametrize(
    ("path_name", "message"),
    (
        ("source_protocol_path", "source-lock summary differs"),
        ("plan_path", "plan summary differs"),
        ("download_path", "download summary differs"),
        ("result_path", "result summary differs"),
    ),
)
def test_terminal_record_prevents_artifact_file_substitution(
    tmp_path: Path,
    path_name: str,
    message: str,
) -> None:
    inputs = _inputs(tmp_path)
    path = getattr(inputs.chain, path_name)
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n")

    with pytest.raises(ValueError, match=message):
        _evaluated(inputs)


def test_candidate_information_loss_is_rejected(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    np.save(inputs.candidate_path, np.eye(3) * 0.5, allow_pickle=False)

    with pytest.raises(ValueError, match="candidate"):
        _evaluated(inputs)


def test_symlinked_matrix_is_rejected(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    symlink = tmp_path / "reference-link.npy"
    symlink.symlink_to(inputs.reference_path)

    with pytest.raises(ValueError, match="must not contain symlinks"):
        build_evaluated_case_from_paths(
            **_common(inputs, object_id="cal-sheet-0"),
            reference_marginal_precision_path=symlink,
            candidate_marginal_precision_path=inputs.candidate_path,
            contact_anchor_artifact_path=inputs.anchor_path,
        )


def test_pickled_query_payload_is_rejected(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    np.save(
        inputs.query_path,
        np.asarray([{"not": "numeric"}], dtype=object),
        allow_pickle=True,
    )

    with pytest.raises(ValueError, match="cannot load physical query Jacobian"):
        _evaluated(inputs)


def test_empty_contact_anchor_is_rejected(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    inputs.anchor_path.write_bytes(b"")

    with pytest.raises(ValueError, match="must not be empty"):
        _evaluated(inputs)


def test_failure_builder_retains_source_and_analysis_failures(tmp_path: Path) -> None:
    source_failure = _inputs(tmp_path / "source", prepared_sheet=4)
    source_case = build_technical_failure_case_from_paths(
        **_common(source_failure, object_id="cal-sheet-4"),
        failure_evidence_path=source_failure.failure_evidence_path,
        failure_reason="source preparation failed",
    )
    assert source_case.status == "technical_failure_without_replacement"
    assert source_case.information_boundary["calibration_payloads_opened"] is True
    assert source_case.physical_query_id == physical_query_id_from_path(
        source_failure.query_path
    )

    analysis_failure = _inputs(tmp_path / "analysis")
    analysis_case = build_technical_failure_case_from_paths(
        **_common(analysis_failure, object_id="cal-sheet-0"),
        failure_evidence_path=analysis_failure.failure_evidence_path,
        failure_reason="observability factorization failed",
    )
    assert analysis_case.information_boundary["calibration_payloads_opened"] is True
    assert analysis_case.comparison is None


def test_unsupported_object_failure_records_no_payload_access(tmp_path: Path) -> None:
    inputs = _inputs(
        tmp_path,
        planned_sheet=4,
        prepared_sheet=4,
    )

    case = build_technical_failure_case_from_paths(
        **_common(inputs, object_id="cal-sheet-4"),
        failure_evidence_path=inputs.failure_evidence_path,
        failure_reason="names-only support unavailable",
    )

    assert case.information_boundary["calibration_payloads_opened"] is False
    assert case.status == "technical_failure_without_replacement"


def test_cli_publishes_evaluated_case_without_overwrite(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    output = tmp_path / "case.json"
    arguments = [
        "evaluated",
        "--source-protocol",
        str(inputs.chain.source_protocol_path),
        "--stage0-protocol",
        str(inputs.chain.stage0_protocol_path),
        "--selection-lock",
        str(inputs.chain.selection_path),
        "--visual-provider-lock",
        str(inputs.chain.provider_path),
        "--calibration-source-plan",
        str(inputs.chain.plan_path),
        "--calibration-source-download",
        str(inputs.chain.download_path),
        "--calibration-source-run-record",
        str(inputs.run_record_path),
        "--calibration-source-result",
        str(inputs.chain.result_path),
        "--object-id",
        "cal-sheet-0",
        "--implementation-revision",
        IMPLEMENTATION_REVISION,
        "--query-jacobian",
        str(inputs.query_path),
        "--reference-marginal-precision",
        str(inputs.reference_path),
        "--candidate-marginal-precision",
        str(inputs.candidate_path),
        "--contact-anchor-artifact",
        str(inputs.anchor_path),
        "--output",
        str(output),
    ]

    assert CLI.main(arguments) == 0
    assert load_deform360_calibration_observability_case(output).status == ("evaluated")
    assert CLI.main(arguments) == 2
