from __future__ import annotations

import json
from pathlib import Path

from bayesian_phystwin.phystwin_tapnextpp_competence import (
    canonical_sha256,
    file_sha256,
)
from bayesian_phystwin.tapnextpp_transfer_evaluation import (
    CASE_RESULT_FILENAME,
    evaluate_transfer_panel,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _protocol_path() -> Path:
    return (
        Path(__file__).parents[1]
        / "configs"
        / "sota"
        / "phystwin_tapnextpp_depth_completion_transfer_v1.json"
    )


def _source_manifest(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    protocol_path = _protocol_path()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    source = {
        "artifact_kind": "PhysTwinTAPNextPPDepthCompletionTransferSourceManifest",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": file_sha256(protocol_path),
        "case_records": [
            {"case": case, "status": "prediction-ready"}
            for case in protocol["fixed_source_cases"]
        ],
    }
    source["result_sha256"] = canonical_sha256(source)
    path = tmp_path / "source.json"
    _write_json(path, source)
    return path, protocol


def _write_result(root: Path, case: str, *, passed: bool = True) -> None:
    result = {
        "artifact_kind": "PhysTwinTAPNextPPDepthCompletionTransferCaseResult",
        "case": case,
        "provider_gate_passed": passed,
        "metrics": {
            "eligible_rows": 100,
            "completed_supported_rows": 90,
            "candidate_identity_rmse_m": 0.005,
            "relative_gain_over_persistence": 0.5,
        },
    }
    result["result_sha256"] = canonical_sha256(result)
    _write_json(root / case / CASE_RESULT_FILENAME, result)


def test_aggregate_gate_retains_missing_cases_but_allows_six_of_eight(
    tmp_path: Path,
) -> None:
    source_path, protocol = _source_manifest(tmp_path)
    result_root = tmp_path / "results"
    for case in protocol["fixed_source_cases"][:6]:
        _write_result(result_root, case)

    summary = evaluate_transfer_panel(
        _protocol_path(),
        source_path,
        result_root,
        tmp_path / "summary.json",
    )
    assert summary["case_count"] == 8
    assert summary["evaluated_case_count"] == 6
    assert summary["technical_failure_count"] == 2
    assert summary["passing_case_count"] == 6
    assert summary["transfer_gate_passed"] is True


def test_aggregate_gate_stops_when_only_five_cases_pass(tmp_path: Path) -> None:
    source_path, protocol = _source_manifest(tmp_path)
    result_root = tmp_path / "results"
    for case in protocol["fixed_source_cases"][:5]:
        _write_result(result_root, case)

    summary = evaluate_transfer_panel(
        _protocol_path(),
        source_path,
        result_root,
        tmp_path / "summary.json",
    )
    assert summary["passing_case_count"] == 5
    assert summary["gates"]["passing_case_count"] is False
    assert summary["transfer_gate_passed"] is False
