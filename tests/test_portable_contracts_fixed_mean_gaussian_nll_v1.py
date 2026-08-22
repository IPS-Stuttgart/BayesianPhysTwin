from __future__ import annotations

import copy
import importlib.util
import json
import math
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCIENCE = ROOT / "scripts/science"
MODULE_PATH = SCIENCE / "fixed_mean_gaussian_nll_v1.py"
SCRIPT = SCIENCE / "analyze_fixed_mean_gaussian_nll_v1.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fixed_mean_nll", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load fixed-mean NLL analyzer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def _record(
    group: str,
    horizon: str,
    *,
    suffix: str = "0",
    error: float = 2.0,
    candidate_variance: float = 4.0,
) -> dict[str, object]:
    return {
        "candidate_covariance": [[candidate_variance, 0.0], [0.0, 1.0]],
        "group_id": group,
        "horizon": horizon,
        "mean": [0.0, 0.0],
        "observation": [error, 0.0],
        "reference_covariance": [[1.0, 0.0], [0.0, 1.0]],
        "unit_id": f"{group}/{horizon}/{suffix}",
    }


def _payload() -> dict[str, object]:
    horizons = ["early", "middle", "late"]
    return {
        "analysis_id": "synthetic-fixed-mean-source-diagnostic-v1",
        "analysis_status": MODULE.ANALYSIS_STATUS,
        "candidate_arm_id": "same_mean_with_candidate_covariance",
        "claim_authorized": False,
        "contract": MODULE.INPUT_CONTRACT,
        "horizon_order": horizons,
        "maximum_condition_number": 1e8,
        "nominal_coverage": 0.9,
        "observation_model_id": "gaussian-observation-5mm-v1",
        "protocol_id": "synthetic-fixed-mean-protocol-v1",
        "query_id": "synthetic-two-component-query-v1",
        "records": [
            _record(group, horizon, error=2.0 + 0.1 * group_index)
            for group_index, group in enumerate(("group-a", "group-b"))
            for horizon in horizons
        ],
        "reference_arm_id": "same_mean_with_reference_covariance",
        "schema_version": 1,
        "scientific_boundary": "synthetic source-only diagnostic; no claim",
        "source_artifact_id": "synthetic-sealed-source-table-v1",
        "statistical_unit": "physical-object-session",
    }


def test_terms_match_registered_gaussian_score() -> None:
    terms = MODULE.gaussian_nll_terms(
        [0.0, 0.0],
        [[4.0, 0.0], [0.0, 1.0]],
        [2.0, 0.0],
    )
    expected = 0.5 * math.log(2.0 * math.pi) + 0.25 * math.log(4.0) + 0.25
    assert terms["total_per_dimension"] == pytest.approx(expected)
    assert terms["sharpness_per_dimension"] == pytest.approx(0.25 * math.log(4.0))
    assert terms["standardized_error_per_dimension"] == pytest.approx(0.25)

    from bayesian_phystwin.probabilistic_scoring import (
        gaussian_nll_per_dimension,
    )

    assert terms["total_per_dimension"] == pytest.approx(
        gaussian_nll_per_dimension(
            [0.0, 0.0],
            [[4.0, 0.0], [0.0, 1.0]],
            [2.0, 0.0],
        )
    )


def test_report_exposes_fit_gain_and_width_cost() -> None:
    report = MODULE.analyze_fixed_mean_gaussian_nll(_payload())
    overall = report["summary"]["overall"]
    assert overall["nll_difference_per_dimension"] < 0.0
    assert overall["sharpness_difference_per_dimension"] > 0.0
    assert overall["standardized_error_difference_per_dimension"] < 0.0
    assert overall["decomposition_residual"] == pytest.approx(0.0, abs=1e-14)
    assert overall["candidate_to_reference_width_ratio"] == pytest.approx(1.5)
    assert report["summary"]["better_worse_tie_groups"] == [2, 0, 0]
    assert report["fixed_mean_by_construction"] is True
    assert report["claim_authorized"] is False
    assert len(report["input_id"]) == 64
    assert len(report["report_id"]) == 64
    assert report == MODULE.analyze_fixed_mean_gaussian_nll(_payload())


def test_weighting_does_not_pool_extra_records() -> None:
    payload = _payload()
    records = payload["records"]
    assert isinstance(records, list)
    records.insert(1, _record("group-a", "early", suffix="1", error=0.0))
    report = MODULE.analyze_fixed_mean_gaussian_nll(payload)
    groups = {row["group_id"]: row for row in report["group_analyses"]}
    first, second = report["record_analyses"][:2]
    expected_cell = 0.5 * (
        first["metrics"]["nll_difference_per_dimension"]
        + second["metrics"]["nll_difference_per_dimension"]
    )
    assert groups["group-a"]["by_horizon"]["early"][
        "nll_difference_per_dimension"
    ] == pytest.approx(expected_cell)
    expected_overall = np.mean(
        [
            groups["group-a"]["overall"]["nll_difference_per_dimension"],
            groups["group-b"]["overall"]["nll_difference_per_dimension"],
        ]
    )
    assert report["summary"]["overall"][
        "nll_difference_per_dimension"
    ] == pytest.approx(expected_overall)


def test_terms_are_orthogonally_invariant() -> None:
    angle = 0.37
    rotation = np.asarray(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ]
    )
    mean = np.asarray([0.2, -0.1])
    observation = np.asarray([1.3, 0.4])
    covariance = np.asarray([[2.0, 0.4], [0.4, 0.7]])
    original = MODULE.gaussian_nll_terms(mean, covariance, observation)
    transformed = MODULE.gaussian_nll_terms(
        rotation @ mean,
        rotation @ covariance @ rotation.T,
        rotation @ observation,
    )
    assert transformed["log_determinant"] == pytest.approx(original["log_determinant"])
    assert transformed["mahalanobis_squared"] == pytest.approx(
        original["mahalanobis_squared"]
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(claim_authorized=True), "claim_authorized"),
        (
            lambda value: value.update(analysis_status="claim-bearing"),
            "retrospective",
        ),
        (
            lambda value: value.update(reference_arm_id=value["candidate_arm_id"]),
            "identities must differ",
        ),
        (lambda value: value.update(nominal_coverage=0.5), "minimum"),
        (lambda value: value["records"].reverse(), "records must be sorted"),
        (lambda value: value["records"].pop(), "every group-by-horizon"),
        (
            lambda value: value["records"][0].update(
                candidate_covariance=[[1.0, 1.0], [1.0, 1.0]]
            ),
            "positive definite",
        ),
        (
            lambda value: value["records"][0].update(
                candidate_covariance=[[1.0, 0.0], [0.0, 1e-12]]
            ),
            "condition number",
        ),
        (
            lambda value: value["records"][0].update(observation=[0.0]),
            "shape differs",
        ),
    ],
)
def test_contract_fails_closed(
    mutation: Callable[[dict[str, object]], object],
    message: str,
) -> None:
    payload = copy.deepcopy(_payload())
    mutation(payload)
    with pytest.raises(ValueError, match=message):
        MODULE.analyze_fixed_mean_gaussian_nll(payload)


def test_contract_rejects_duplicate_units_and_noncanonical_shapes() -> None:
    payload = _payload()
    records = payload["records"]
    assert isinstance(records, list)
    records[1]["unit_id"] = records[0]["unit_id"]
    with pytest.raises(ValueError, match="repeat a unit_id"):
        MODULE.analyze_fixed_mean_gaussian_nll(payload)

    with pytest.raises(ValueError, match="symmetric"):
        MODULE.gaussian_nll_terms(
            [0.0, 0.0],
            [[1.0, 0.1], [0.0, 1.0]],
            [0.0, 0.0],
        )
    with pytest.raises(ValueError, match="real numeric"):
        MODULE.gaussian_nll_terms([0.0], [["x"]], [0.0])


def test_cli_is_atomic_and_refuses_unrequested_overwrite(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "nested/report.json"
    input_path.write_text(json.dumps(_payload()), encoding="utf-8")
    command = [
        sys.executable,
        str(SCRIPT),
        str(input_path),
        "--output",
        str(output_path),
    ]
    first = subprocess.run(command, check=False, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["contract"] == MODULE.REPORT_CONTRACT

    second = subprocess.run(command, check=False, capture_output=True, text=True)
    assert second.returncode == 2
    assert "output already exists" in second.stderr

    forced = subprocess.run(
        [*command, "--force"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert forced.returncode == 0, forced.stderr
