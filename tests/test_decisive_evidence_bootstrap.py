from __future__ import annotations

import json
from pathlib import Path

import pytest

from bayesian_phystwin.cli.decisive_evidence import main as evidence_main
from bayesian_phystwin.decisive_evidence_bootstrap import (
    GROUP_CLUSTERED_BOOTSTRAP_CONTRACT,
    group_clustered_paired_bootstrap,
)


def _record(
    *,
    unit_id: str,
    group_id: str,
    method: str,
    loss: float,
    fallback_loss: float = 10.0,
    accepted: bool = True,
) -> dict[str, object]:
    return {
        "unit_id": unit_id,
        "group_id": group_id,
        "metric": "track_error_m",
        "method": method,
        "loss": loss,
        "fallback_loss": fallback_loss,
        "risk_score": 0.1,
        "accepted": accepted,
        "deployed_loss": loss if accepted else fallback_loss,
        "horizon": "late",
        "reliability": 0.8,
        "identifiable_rank": 2,
        "intervals": [],
    }


def _payload() -> dict[str, object]:
    records = [
        _record(
            unit_id="a-1",
            group_id="object-a",
            method="bayesian",
            loss=8.0,
        ),
        _record(
            unit_id="a-1",
            group_id="object-a",
            method="last_residual",
            loss=9.0,
        ),
        _record(
            unit_id="a-2",
            group_id="object-a",
            method="bayesian",
            loss=12.0,
            accepted=False,
        ),
        _record(
            unit_id="a-2",
            group_id="object-a",
            method="last_residual",
            loss=11.0,
        ),
        _record(
            unit_id="b-1",
            group_id="object-b",
            method="bayesian",
            loss=7.0,
        ),
        _record(
            unit_id="b-1",
            group_id="object-b",
            method="last_residual",
            loss=8.0,
            accepted=False,
        ),
        _record(
            unit_id="b-2",
            group_id="object-b",
            method="bayesian",
            loss=11.0,
            accepted=False,
        ),
        _record(
            unit_id="b-2",
            group_id="object-b",
            method="last_residual",
            loss=13.0,
            accepted=False,
        ),
    ]
    return {
        "schema_version": 1,
        "contract": "bayesian-phystwin-decisive-evidence-v1",
        "protocol_id": "paired-bootstrap-test-v1",
        "statistical_unit": "object-horizon",
        "claim_boundary": "unit test",
        "reference_method": "last_residual",
        "records": records,
    }


def test_bootstrap_is_paired_equal_group_and_deterministic() -> None:
    result = group_clustered_paired_bootstrap(
        _payload(), replicates=1000, seed=17, confidence=0.9
    )
    assert result["contract"] == GROUP_CLUSTERED_BOOTSTRAP_CONTRACT
    assert result["resampling_unit"] == "group_id"
    assert result["group_weighting"] == "equal"

    metric = result["metrics"]["track_error_m"]
    assert metric["group_count"] == 2
    assert metric["minimum_independent_group_requirement_met"] is True
    bayesian = metric["methods"]["bayesian"]
    deployed = bayesian["deployed_vs_fallback"]
    assert deployed["status"] == "complete"
    assert deployed["observed"]["mean_loss_difference"] == pytest.approx(-1.25)
    assert deployed["bootstrap_probability_candidate_better"] == 1.0
    assert deployed["mean_loss_difference_interval"]["lower"] == pytest.approx(-1.5)
    assert deployed["mean_loss_difference_interval"]["upper"] == pytest.approx(-1.0)

    paired = bayesian["deployed_vs_reference_method"]
    assert paired["reference_method"] == "last_residual"
    assert paired["observed"]["mean_loss_difference"] == pytest.approx(-1.25)
    assert paired["bootstrap_probability_candidate_better"] == 1.0

    reversed_payload = json.loads(json.dumps(_payload()))
    reversed_payload["records"].reverse()
    assert group_clustered_paired_bootstrap(
        reversed_payload, replicates=1000, seed=17, confidence=0.9
    ) == result


def test_bootstrap_averages_within_group_before_equal_weighting() -> None:
    records: list[dict[str, object]] = []
    for index in range(10):
        records.extend(
            [
                _record(
                    unit_id=f"large-{index}",
                    group_id="large-group",
                    method="candidate",
                    loss=0.0,
                ),
                _record(
                    unit_id=f"large-{index}",
                    group_id="large-group",
                    method="reference",
                    loss=10.0,
                ),
            ]
        )
    records.extend(
        [
            _record(
                unit_id="small-0",
                group_id="small-group",
                method="candidate",
                loss=20.0,
            ),
            _record(
                unit_id="small-0",
                group_id="small-group",
                method="reference",
                loss=10.0,
            ),
        ]
    )
    payload = {
        "schema_version": 1,
        "contract": "bayesian-phystwin-decisive-evidence-v1",
        "protocol_id": "equal-group-test-v1",
        "statistical_unit": "object",
        "claim_boundary": "unit test",
        "reference_method": "reference",
        "records": records,
    }
    result = group_clustered_paired_bootstrap(payload, replicates=100, seed=3)
    comparison = result["metrics"]["track_error_m"]["methods"]["candidate"]
    assert comparison["raw_vs_fallback"]["observed"][
        "candidate_mean_loss"
    ] == pytest.approx(10.0)
    assert comparison["raw_vs_reference_method"]["observed"][
        "mean_loss_difference"
    ] == pytest.approx(0.0)
    assert comparison["registered_unit_counts_by_group"] == {
        "large-group": 10,
        "small-group": 1,
    }


def test_single_group_is_reported_as_insufficient() -> None:
    payload = _payload()
    payload["records"] = [
        record for record in payload["records"] if record["group_id"] == "object-a"
    ]
    result = group_clustered_paired_bootstrap(payload, replicates=10)
    comparison = result["metrics"]["track_error_m"]["methods"]["bayesian"][
        "deployed_vs_fallback"
    ]
    assert comparison["status"] == "insufficient_independent_groups"
    assert comparison["mean_loss_difference_interval"] is None
    assert comparison["bootstrap_probability_candidate_better"] is None


def test_cli_embeds_bootstrap_configuration(tmp_path: Path) -> None:
    input_path = tmp_path / "evidence.json"
    output_path = tmp_path / "summary.json"
    input_path.write_text(json.dumps(_payload()), encoding="utf-8")
    assert (
        evidence_main(
            [
                str(input_path),
                str(output_path),
                "--bootstrap-replicates",
                "250",
                "--bootstrap-seed",
                "23",
                "--bootstrap-confidence",
                "0.9",
            ]
        )
        == 0
    )
    summary = json.loads(output_path.read_text(encoding="utf-8"))
    bootstrap = summary["group_clustered_bootstrap"]
    assert bootstrap["replicates"] == 250
    assert bootstrap["seed"] == 23
    assert bootstrap["confidence"] == 0.9
    assert summary["analysis_configuration"][
        "group_clustered_bootstrap_contract"
    ] == GROUP_CLUSTERED_BOOTSTRAP_CONTRACT
