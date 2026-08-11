from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path("scripts/science/analyze_full22_uncertainty_value_v1.py")
PROTOCOL = Path("protocols/full22_discrepancy_candidate_tournament_v1.json")


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "full22_uncertainty_value_analysis",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


analysis = _load_module()


def _protocol() -> dict[str, object]:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def _payload(
    *,
    structured_effect: float = -0.5,
    graph_effect: float = 0.2,
    reject_structured: bool = False,
) -> dict[str, object]:
    protocol = _protocol()
    protocol_id = analysis._canonical_sha256(protocol)
    rows: list[dict[str, object]] = []
    effects = {
        "independent_endpoint_v1": 0.02,
        "dynamic_endpoint_v2": 0.0,
        "structured_kernel_rank4_v1": structured_effect,
        "graph_dynamic_kernel_rank4_v1": graph_effect,
    }
    for case_index in range(analysis.EXPECTED_CASE_COUNT):
        case_id = f"object-{case_index:02d}"
        case_shift = 0.01 * (case_index - 10)
        for horizon_index, horizon in enumerate(analysis.HORIZONS):
            fallback_point = {
                "chamfer_distance_m": 0.14 + 0.01 * horizon_index,
                "track_error_m": 0.16 + 0.01 * horizon_index,
            }
            fallback_nll = 10.0 + horizon_index + case_shift
            reference_point = {
                "chamfer_distance_m": 0.10 + 0.01 * horizon_index,
                "track_error_m": 0.12 + 0.01 * horizon_index,
            }
            reference_nll = 8.0 + horizon_index + case_shift
            for candidate_id in analysis.EXPECTED_CANDIDATES:
                if candidate_id == analysis.PHYSICAL_FALLBACK:
                    point = fallback_point
                    proper_score = fallback_nll
                    accepted = False
                else:
                    point = dict(reference_point)
                    proper_score = reference_nll + effects.get(candidate_id, 0.0)
                    accepted = not (
                        reject_structured
                        and candidate_id == "structured_kernel_rank4_v1"
                    )
                rows.append(
                    {
                        "case_id": case_id,
                        "candidate_id": candidate_id,
                        "horizon": horizon,
                        "accepted": accepted,
                        "point": point,
                        "fallback_point": fallback_point,
                        "proper_score": proper_score,
                        "fallback_proper_score": fallback_nll,
                    }
                )
    return {
        "protocol_id": protocol_id,
        "claim_authorized": False,
        "rows": rows,
    }


def _analyze(payload: dict[str, object]) -> dict[str, object]:
    return analysis.analyze_uncertainty_value(
        payload,
        _protocol(),
        bootstrap_replicates=1000,
        bootstrap_seed=17,
        bootstrap_confidence=0.95,
        source_metadata={"unit_test": True},
    )


def _comparison(
    report: dict[str, object],
    *,
    candidate_id: str,
    stream: str,
    endpoint: str,
    aggregation: str = "overall",
) -> dict[str, object]:
    return next(
        row
        for row in report["comparisons"]
        if row["candidate_id"] == candidate_id
        and row["stream"] == stream
        and row["endpoint"] == endpoint
        and row["aggregation"] == aggregation
    )


def test_localizes_uncertainty_gain_without_point_gain() -> None:
    report = _analyze(_payload())
    summary = report["summary"]
    assert summary["primary_conclusion"] == (
        "retrospective-uncertainty-score-signal"
    )
    assert summary["familywise_supported_raw_nll_candidates"] == [
        "structured_kernel_rank4_v1"
    ]
    structured = _comparison(
        report,
        candidate_id="structured_kernel_rank4_v1",
        stream="raw",
        endpoint="gaussian_nll",
    )
    assert structured["mean_difference"] == pytest.approx(-0.5)
    assert structured["familywise_decision"] == "candidate_better"
    assert structured["leave_one_case_out_sign_stable"] is True
    track = _comparison(
        report,
        candidate_id="structured_kernel_rank4_v1",
        stream="raw",
        endpoint="track_error_m",
    )
    assert track["mean_difference"] == pytest.approx(0.0)
    assert track["familywise_decision"] == "inconclusive"
    assert report["claim_authorized"] is False
    assert report["promotion_authorized"] is False


def test_raw_and_deployed_scientific_questions_remain_separate() -> None:
    report = _analyze(_payload(reject_structured=True))
    raw = _comparison(
        report,
        candidate_id="structured_kernel_rank4_v1",
        stream="raw",
        endpoint="gaussian_nll",
    )
    deployed = _comparison(
        report,
        candidate_id="structured_kernel_rank4_v1",
        stream="deployed",
        endpoint="gaussian_nll",
    )
    assert raw["familywise_decision"] == "candidate_better"
    assert deployed["familywise_decision"] == "candidate_worse"


def test_mixed_object_effects_do_not_create_supported_gain() -> None:
    payload = _payload(structured_effect=0.0, graph_effect=0.0)
    for row in payload["rows"]:
        if row["candidate_id"] == "structured_kernel_rank4_v1":
            case_index = int(str(row["case_id"]).split("-")[-1])
            row["proper_score"] += -0.5 if case_index % 2 == 0 else 0.5
    report = _analyze(payload)
    structured = _comparison(
        report,
        candidate_id="structured_kernel_rank4_v1",
        stream="raw",
        endpoint="gaussian_nll",
    )
    assert structured["familywise_decision"] == "inconclusive"
    assert report["summary"]["primary_conclusion"] == (
        "no-familywise-supported-uncertainty-score-gain"
    )


def test_analysis_is_deterministic_and_row_order_invariant() -> None:
    payload = _payload()
    forward = _analyze(payload)
    reverse_payload = copy.deepcopy(payload)
    reverse_payload["rows"].reverse()
    reverse = _analyze(reverse_payload)
    assert reverse == forward
    assert len(forward["report_id"]) == 64


def test_source_contracts_fail_closed() -> None:
    payload = _payload()
    payload["rows"].pop()
    with pytest.raises(ValueError, match="sealed rows"):
        _analyze(payload)

    payload = _payload()
    payload["rows"][1]["fallback_proper_score"] += 1.0
    with pytest.raises(ValueError, match="fallback proper"):
        _analyze(payload)

    payload = _payload()
    payload["claim_authorized"] = True
    with pytest.raises(ValueError, match="authorize a claim"):
        _analyze(payload)

    payload = _payload()
    payload["protocol_id"] = "0" * 64
    with pytest.raises(ValueError, match="frozen protocol"):
        _analyze(payload)


def test_cli_publishes_a_compact_reproduction_capsule(tmp_path: Path) -> None:
    scored_path = tmp_path / "raw_scored_rows.json"
    protocol_path = tmp_path / "protocol.json"
    output_dir = tmp_path / "result"
    scored_path.write_text(json.dumps(_payload()), encoding="utf-8")
    protocol_path.write_text(json.dumps(_protocol()), encoding="utf-8")
    arguments = [
        str(scored_path),
        str(protocol_path),
        str(output_dir),
        "--bootstrap-replicates",
        "1000",
        "--source-run-id",
        "31410594302",
        "--source-run-attempt",
        "1",
        "--source-head-sha",
        "c5d22a369f87a7ce2e10fcad532bcc5cc5207ff9",
        "--source-artifact-id",
        "9074451004",
        "--source-artifact-name",
        "bpt-full22-discrepancy-31410594302-1",
        "--source-artifact-digest",
        "22984bd34992ef7693c7577045c7496f8de2990641c3d2592ce230b9fbc97220",
        "--analyzer-revision",
        "a" * 40,
    ]
    assert analysis.main(arguments) == 0
    report = json.loads(
        (output_dir / "full22_uncertainty_value_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["source"]["raw_scored_rows_bytes"] > 0
    assert report["source"]["artifact_id"] == "9074451004"
    assert (output_dir / "full22_uncertainty_value_table.csv").is_file()
    assert (output_dir / "full22_uncertainty_value_summary.md").is_file()
    assert (output_dir / "raw_scored_rows.json").is_file()
    with pytest.raises(FileExistsError):
        analysis.main(arguments)
    assert analysis.main([*arguments, "--force"]) == 0
