from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.cli.proper_scoring import main as scoring_main
from bayesian_phystwin.decisive_evidence import parse_decisive_evidence
from bayesian_phystwin.proper_scoring import (
    build_proper_score_evidence,
    empirical_energy_score,
    gaussian_log_score,
)


def _vector_prediction(mean: list[float], samples: list[list[float]]) -> dict:
    return {
        "samples": samples,
        "gaussian": {
            "mean": mean,
            "covariance": [[1.0, 0.0], [0.0, 1.0]],
        },
    }


def _scalar_prediction(
    median: float,
    intervals: list[tuple[float, float, float]],
) -> dict:
    return {
        "scalar_intervals": {
            "median": median,
            "central_intervals": [
                {
                    "nominal_coverage": nominal,
                    "lower": lower,
                    "upper": upper,
                }
                for nominal, lower, upper in intervals
            ],
        }
    }


def _payload() -> dict[str, object]:
    common_vector = {
        "unit_id": "vector-u1",
        "group_id": "object-a",
        "query_id": "endpoint_xy",
        "horizon": "late",
        "observation": [0.0, 0.0],
        "fallback_prediction": _vector_prediction(
            [1.0, 0.0], [[1.0, 0.0], [1.0, 0.0]]
        ),
        "variogram_pairs": [{"left": 0, "right": 1, "weight": 1.0}],
    }
    common_scalar = {
        "unit_id": "scalar-u1",
        "group_id": "object-a",
        "query_id": "scalar_endpoint",
        "horizon": 2,
        "observation": [0.0],
        "fallback_prediction": _scalar_prediction(
            1.0,
            [(0.5, 0.0, 2.0), (0.9, -1.0, 3.0)],
        ),
    }
    records = [
        {
            **common_vector,
            "method": "bayesian",
            "risk_score": 0.1,
            "accepted": True,
            "reliability": 0.9,
            "identifiable_rank": 2,
            "prediction": _vector_prediction(
                [0.0, 0.0], [[0.0, 0.0], [2.0, 0.0]]
            ),
        },
        {
            **common_vector,
            "method": "last_residual",
            "risk_score": 0.3,
            "accepted": False,
            "reliability": 0.7,
            "identifiable_rank": 1,
            "prediction": _vector_prediction(
                [0.5, 0.0], [[0.5, 0.0], [0.5, 0.0]]
            ),
        },
        {
            **common_scalar,
            "method": "bayesian",
            "risk_score": 0.1,
            "accepted": True,
            "prediction": _scalar_prediction(
                0.0,
                [(0.5, -1.0, 1.0), (0.9, -2.0, 2.0)],
            ),
        },
        {
            **common_scalar,
            "method": "last_residual",
            "risk_score": 0.3,
            "accepted": False,
            "prediction": _scalar_prediction(
                0.5,
                [(0.5, -0.5, 1.5), (0.9, -1.5, 2.5)],
            ),
        },
    ]
    return {
        "schema_version": 1,
        "contract": "bayesian-phystwin-proper-scoring-input-v1",
        "protocol_id": "proper-score-unit-test-v1",
        "statistical_unit": "object-query-horizon",
        "claim_boundary": "unit-test evidence only",
        "reference_method": "last_residual",
        "score_configuration": {
            "variogram_power": 1.0,
            "gaussian_log_score_offset": 0.0,
        },
        "records": records,
    }


def _record(
    evidence: dict[str, object],
    *,
    metric_suffix: str,
    method: str,
) -> dict[str, object]:
    records = evidence["records"]
    assert isinstance(records, list)
    return next(
        item
        for item in records
        if isinstance(item, dict)
        and str(item["metric"]).endswith(metric_suffix)
        and item["method"] == method
    )


def test_formulae_and_exact_fallback_conversion() -> None:
    evidence = build_proper_score_evidence(_payload())
    parse_decisive_evidence(evidence)

    energy = _record(
        evidence,
        metric_suffix="/energy_score",
        method="bayesian",
    )
    assert energy["loss"] == pytest.approx(0.5)
    assert energy["fallback_loss"] == pytest.approx(1.0)

    variogram = _record(
        evidence,
        metric_suffix="variogram_score_power_1",
        method="bayesian",
    )
    assert variogram["loss"] == pytest.approx(1.0)
    assert variogram["fallback_loss"] == pytest.approx(1.0)

    log_score = _record(
        evidence,
        metric_suffix="/gaussian_log_score_shifted",
        method="bayesian",
    )
    assert log_score["loss"] == pytest.approx(np.log(2.0 * np.pi))
    assert log_score["fallback_loss"] == pytest.approx(
        np.log(2.0 * np.pi) + 0.5
    )

    rejected = _record(
        evidence,
        metric_suffix="/energy_score",
        method="last_residual",
    )
    assert rejected["accepted"] is False
    assert rejected["deployed_loss"] == rejected["fallback_loss"]

    wis = _record(
        evidence,
        metric_suffix="/weighted_interval_score",
        method="bayesian",
    )
    assert wis["loss"] == pytest.approx(0.28)
    assert wis["fallback_loss"] == pytest.approx(0.48)
    assert [interval["width"] for interval in wis["intervals"]] == [2.0, 4.0]

    rejected_wis = _record(
        evidence,
        metric_suffix="/weighted_interval_score",
        method="last_residual",
    )
    assert rejected_wis["deployed_loss"] == rejected_wis["fallback_loss"]
    assert [interval["width"] for interval in rejected_wis["intervals"]] == [
        2.0,
        4.0,
    ]
    metadata = evidence["proper_scoring"]
    assert isinstance(metadata, dict)
    assert len(str(metadata["evidence_id"])) == 64


def test_public_numeric_helpers_match_closed_form() -> None:
    score = empirical_energy_score(
        np.asarray([0.0, 0.0]),
        np.asarray([[0.0, 0.0], [2.0, 0.0]]),
        maximum_pair_evaluations=4,
    )
    assert score == pytest.approx(0.5)
    nll = gaussian_log_score(
        np.asarray([0.0, 0.0]),
        np.asarray([0.0, 0.0]),
        np.eye(2),
    )
    assert nll == pytest.approx(np.log(2.0 * np.pi))


def test_converter_fails_closed_on_contract_and_numerical_drift() -> None:
    payload = _payload()
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        build_proper_score_evidence(payload)

    payload = _payload()
    records = payload["records"]
    assert isinstance(records, list)
    records[1]["fallback_prediction"] = _vector_prediction(
        [2.0, 0.0], [[2.0, 0.0], [2.0, 0.0]]
    )
    with pytest.raises(ValueError, match="changed truth, fallback"):
        build_proper_score_evidence(payload)

    payload = _payload()
    records = payload["records"]
    assert isinstance(records, list)
    records[0]["prediction"]["gaussian"]["covariance"] = [
        [1.0, 2.0],
        [2.0, 1.0],
    ]
    with pytest.raises(ValueError, match="positive definite"):
        build_proper_score_evidence(payload)

    payload = _payload()
    records = payload["records"]
    assert isinstance(records, list)
    records[2]["prediction"]["scalar_intervals"]["central_intervals"] = [
        {"nominal_coverage": 0.5, "lower": -2.0, "upper": 2.0},
        {"nominal_coverage": 0.9, "lower": -1.0, "upper": 1.0},
    ]
    with pytest.raises(ValueError, match="nested"):
        build_proper_score_evidence(payload)


def test_converter_enforces_score_resource_and_offset_boundaries() -> None:
    with pytest.raises(ValueError, match="pair-evaluation budget"):
        build_proper_score_evidence(
            _payload(), maximum_energy_pair_evaluations=3
        )
    with pytest.raises(ValueError, match="variogram-score evaluation budget"):
        build_proper_score_evidence(
            _payload(), maximum_variogram_evaluations=1
        )

    payload = _payload()
    records = payload["records"]
    assert isinstance(records, list)
    records[0]["prediction"]["gaussian"]["covariance"] = [
        [1e-6, 0.0],
        [0.0, 1e-6],
    ]
    with pytest.raises(ValueError, match="larger common"):
        build_proper_score_evidence(payload)


def test_cli_publishes_bounded_content_addressed_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "evidence.json"
    input_path.write_text(json.dumps(_payload()), encoding="utf-8")
    assert scoring_main([str(input_path), str(output_path)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["record_count"] == 8
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(written["status_sha256"]) == 64
    assert len(written["input_artifact"]["sha256"]) == 64
    with pytest.raises(FileExistsError):
        scoring_main([str(input_path), str(output_path)])
