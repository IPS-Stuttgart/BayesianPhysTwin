from __future__ import annotations

import json
from pathlib import Path

import pytest

from bayesian_phystwin.cli.decisive_evidence import main as evidence_main
from bayesian_phystwin.cli.main import main as grouped_main
from bayesian_phystwin.decisive_evidence import analyze_decisive_evidence


def _payload() -> dict[str, object]:
    units = [
        {
            "unit_id": "u1",
            "group_id": "object-a",
            "horizon": "early",
            "fallback_loss": 10.0,
            "bayesian": (8.0, 0.10, True, 0.90, 2, True, 2.0),
            "last_residual": (9.0, 0.15, True, 0.70, 1, True, 3.0),
        },
        {
            "unit_id": "u2",
            "group_id": "object-a",
            "horizon": "early",
            "fallback_loss": 10.0,
            "bayesian": (12.0, 0.40, False, 0.40, 1, False, 4.0),
            "last_residual": (11.0, 0.25, True, 0.60, 1, False, 5.0),
        },
        {
            "unit_id": "u3",
            "group_id": "object-b",
            "horizon": "late",
            "fallback_loss": 10.0,
            "bayesian": (7.0, 0.20, True, 0.80, 2, True, 2.5),
            "last_residual": (8.0, 0.35, False, 0.30, 0, True, 4.0),
        },
        {
            "unit_id": "u4",
            "group_id": "object-b",
            "horizon": "late",
            "fallback_loss": 10.0,
            "bayesian": (11.0, 0.30, False, 0.20, 0, True, 3.0),
            "last_residual": (13.0, 0.45, False, 0.10, 0, False, 6.0),
        },
    ]
    records: list[dict[str, object]] = []
    for unit in units:
        for method in ("bayesian", "last_residual"):
            loss, risk, accepted, reliability, rank, covered, width = unit[method]
            fallback = float(unit["fallback_loss"])
            records.append(
                {
                    "unit_id": unit["unit_id"],
                    "group_id": unit["group_id"],
                    "metric": "track_error_m",
                    "method": method,
                    "loss": loss,
                    "fallback_loss": fallback,
                    "risk_score": risk,
                    "accepted": accepted,
                    "deployed_loss": loss if accepted else fallback,
                    "horizon": unit["horizon"],
                    "reliability": reliability,
                    "identifiable_rank": rank,
                    "intervals": [
                        {
                            "nominal_coverage": 0.9,
                            "covered": covered,
                            "width": width,
                        }
                    ],
                }
            )
    return {
        "schema_version": 1,
        "contract": "bayesian-phystwin-decisive-evidence-v1",
        "protocol_id": "prospective-prob4d-bpt-test-v1",
        "statistical_unit": "case-horizon",
        "claim_boundary": "synthetic unit test only",
        "reference_method": "last_residual",
        "records": records,
    }


def test_summary_reports_matched_selective_and_operational_risk() -> None:
    summary = analyze_decisive_evidence(
        _payload(),
        target_coverages=(0.0, 0.5, 1.0),
        regression_quantiles=(0.5, 0.95),
        reliability_edges=(0.0, 0.5, 1.0),
    )
    metric = summary["metrics"]["track_error_m"]
    bayesian = metric["methods"]["bayesian"]
    operational = bayesian["operational_policy"]

    assert operational["accepted_count"] == 2
    assert operational["fallback_frequency"] == 0.5
    assert operational["harmful_accepted_count"] == 0
    assert operational["deployed"]["mean_loss"] == 8.75
    assert operational["exact_fallback_verified"] is True

    curve = metric["matched_risk_coverage"]["methods"]["bayesian"]
    assert curve[0]["deployed"]["mean_loss"] == 10.0
    assert curve[1]["accepted_count"] == 2
    assert curve[1]["deployed"]["mean_loss"] == 8.75
    assert curve[1]["vs_reference_method"]["relative_change_of_means"] == -0.125
    assert curve[2]["deployed"]["mean_loss"] == 9.5

    comparator = metric["methods"]["last_residual"]["operational_policy"]
    assert comparator["harmful_accepted_count"] == 1
    assert comparator["harmful_update_frequency_accepted"] == 0.5


def test_summary_reports_horizon_calibration_reliability_and_rank() -> None:
    summary = analyze_decisive_evidence(
        _payload(),
        target_coverages=(0.0, 1.0),
        reliability_edges=(0.0, 0.5, 1.0),
    )
    bayesian = summary["metrics"]["track_error_m"]["methods"]["bayesian"]
    horizons = {
        entry["horizon"]: entry for entry in bayesian["performance_by_horizon"]
    }
    early = horizons["early"]["interval_calibration"]["by_nominal_coverage"][0]
    assert early["nominal_coverage"] == 0.9
    assert early["empirical_coverage"] == 0.5
    assert early["mean_width"] == 3.0

    reliability = bayesian["performance_by_reliability"]
    assert reliability["available_count"] == 4
    assert reliability["bins"][0]["summary"]["unit_count"] == 2
    assert reliability["bins"][1]["summary"]["unit_count"] == 2

    ranks = {
        entry["rank"]: entry["summary"]
        for entry in bayesian["performance_by_identifiable_rank"]["by_rank"]
    }
    assert ranks[0]["unit_count"] == 1
    assert ranks[1]["unit_count"] == 1
    assert ranks[2]["unit_count"] == 2


def test_exact_fallback_and_matched_method_contracts_fail_closed() -> None:
    payload = _payload()
    records = payload["records"]
    records[2]["deployed_loss"] = 9.5
    with pytest.raises(ValueError, match="exact fallback loss"):
        analyze_decisive_evidence(payload)

    payload = _payload()
    payload["records"] = payload["records"][:-1]
    with pytest.raises(ValueError, match="matched comparisons require every method"):
        analyze_decisive_evidence(payload)

    payload = _payload()
    payload["records"][1]["fallback_loss"] = 9.0
    payload["records"][1]["deployed_loss"] = payload["records"][1]["loss"]
    with pytest.raises(ValueError, match="method-dependent fallback losses"):
        analyze_decisive_evidence(payload)


def test_cli_writes_checksummed_summary_and_grouped_route(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "evidence.json"
    output_path = tmp_path / "summary.json"
    input_path.write_text(json.dumps(_payload()), encoding="utf-8")

    assert grouped_main(["evidence"]) == 0
    assert "summarize" in capsys.readouterr().out
    assert (
        grouped_main(
            [
                "evidence",
                "summarize",
                str(input_path),
                str(output_path),
                "--coverage",
                "0.0",
                "--coverage",
                "0.5",
                "--coverage",
                "1.0",
                "--reference-method",
                "last_residual",
            ]
        )
        == 0
    )
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["input_artifact"]["sha256"]
    assert written["analysis_configuration"]["matched_fallback"] is True
    assert written["reference_method"] == "last_residual"

    with pytest.raises(FileExistsError):
        evidence_main([str(input_path), str(output_path)])
