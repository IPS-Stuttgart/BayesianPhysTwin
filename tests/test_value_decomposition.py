from __future__ import annotations

import json
from pathlib import Path

import pytest

from bayesian_phystwin.cli.value_decomposition import main as decomposition_main
from bayesian_phystwin.value_decomposition import (
    analyze_bayesian_value_decomposition,
)


ARMS = ("last_residual", "last_residual_guarded", "bpt_mean", "bpt_full")


def _payload() -> dict[str, object]:
    units = [
        ("u1", "object-a", 8.0, True, 7.0, 6.0),
        ("u2", "object-a", 14.0, False, 12.0, 9.0),
        ("u3", "object-b", 7.0, True, 6.0, 5.0),
        ("u4", "object-b", 13.0, False, 11.0, 8.0),
    ]
    records: list[dict[str, object]] = []
    risks = {"u1": 0.1, "u2": 0.4, "u3": 0.2, "u4": 0.3}
    for unit_id, group_id, deterministic_loss, accepted, mean_loss, full_loss in units:
        method_values = {
            "last_residual": (deterministic_loss, True, 0.0, None, None, 2.0),
            "last_residual_guarded": (
                deterministic_loss,
                accepted,
                risks[unit_id],
                0.8,
                2,
                3.0,
            ),
            "bpt_mean": (
                mean_loss,
                accepted,
                risks[unit_id],
                0.8,
                2,
                3.0,
            ),
            "bpt_full": (full_loss, True, risks[unit_id] / 2.0, 0.9, 2, 2.5),
        }
        for method, (
            loss,
            method_accepted,
            risk,
            reliability,
            rank,
            width,
        ) in method_values.items():
            records.append(
                {
                    "unit_id": unit_id,
                    "group_id": group_id,
                    "metric": "endpoint/energy_score",
                    "method": method,
                    "loss": loss,
                    "fallback_loss": 10.0,
                    "risk_score": risk,
                    "accepted": method_accepted,
                    "deployed_loss": loss if method_accepted else 10.0,
                    "horizon": "late",
                    "reliability": reliability,
                    "identifiable_rank": rank,
                    "intervals": [
                        {
                            "nominal_coverage": 0.9,
                            "covered": loss <= 10.0,
                            "width": width,
                        }
                    ],
                }
            )
    return {
        "schema_version": 1,
        "contract": "bayesian-phystwin-decisive-evidence-v1",
        "protocol_id": "bayesian-value-unit-test-v1",
        "statistical_unit": "object-horizon",
        "claim_boundary": "unit-test evidence only",
        "records": records,
    }


def _analyze(payload: dict[str, object]) -> dict[str, object]:
    return analyze_bayesian_value_decomposition(
        payload,
        deterministic_reference=ARMS[0],
        guarded_reference=ARMS[1],
        bayesian_mean=ARMS[2],
        full_belief=ARMS[3],
        bootstrap_replicates=200,
        bootstrap_seed=7,
    )


def test_four_arm_decomposition_is_matched_and_telescoping() -> None:
    report = _analyze(_payload())
    metric = report["metrics"]["endpoint/energy_score"]
    assert metric["invariants"] == {
        "deterministic_reference_always_deployed": True,
        "guarded_reference_raw_mean_unchanged": True,
        "guarded_and_bayesian_mean_share_guard": True,
        "common_exact_fallback_validated": True,
    }
    raw = metric["unit_weighted"]["raw"]
    deployed = metric["unit_weighted"]["deployed"]
    assert raw["steps"][0]["mean_improvement"] == pytest.approx(0.0)
    assert deployed["steps"][0]["mean_improvement"] == pytest.approx(1.75)
    assert deployed["steps"][1]["mean_improvement"] == pytest.approx(0.5)
    assert deployed["steps"][2]["mean_improvement"] == pytest.approx(1.25)
    assert deployed["total"]["mean_improvement"] == pytest.approx(3.5)
    assert deployed["telescoping_mean_loss_difference_residual"] == pytest.approx(
        0.0
    )
    assert deployed["steps"][0][
        "fraction_of_total_mean_improvement"
    ] == pytest.approx(0.5)
    assert metric["group_clustered_bootstrap"]["deployed"]["total"]
    assert len(str(report["source_evidence_id"])) == 64
    assert len(str(report["report_id"])) == 64


def test_decomposition_fails_when_an_arm_does_not_isolate_its_role() -> None:
    payload = _payload()
    records = payload["records"]
    assert isinstance(records, list)
    deterministic = next(
        item
        for item in records
        if item["method"] == "last_residual" and item["unit_id"] == "u1"
    )
    deterministic["accepted"] = False
    deterministic["deployed_loss"] = 10.0
    with pytest.raises(ValueError, match="must always deploy"):
        _analyze(payload)

    payload = _payload()
    records = payload["records"]
    assert isinstance(records, list)
    guarded = next(
        item
        for item in records
        if item["method"] == "last_residual_guarded" and item["unit_id"] == "u1"
    )
    guarded["loss"] = 8.1
    guarded["deployed_loss"] = 8.1
    with pytest.raises(ValueError, match="changed the deterministic mean"):
        _analyze(payload)

    payload = _payload()
    records = payload["records"]
    assert isinstance(records, list)
    mean = next(
        item
        for item in records
        if item["method"] == "bpt_mean" and item["unit_id"] == "u2"
    )
    mean["accepted"] = True
    mean["deployed_loss"] = mean["loss"]
    with pytest.raises(ValueError, match="acceptance differ"):
        _analyze(payload)


def test_cli_publishes_decomposition_with_input_identity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "evidence.json"
    output_path = tmp_path / "decomposition.json"
    input_path.write_text(json.dumps(_payload()), encoding="utf-8")
    arguments = [
        str(input_path),
        str(output_path),
        "--deterministic-reference",
        ARMS[0],
        "--guarded-reference",
        ARMS[1],
        "--bayesian-mean",
        ARMS[2],
        "--full-belief",
        ARMS[3],
        "--bootstrap-replicates",
        "100",
    ]
    assert decomposition_main(arguments) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["metric_count"] == 1
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(written["report_id"]) == 64
    assert len(written["input_artifact"]["sha256"]) == 64
    assert len(written["status_sha256"]) == 64
