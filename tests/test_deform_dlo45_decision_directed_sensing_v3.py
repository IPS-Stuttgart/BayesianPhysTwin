from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.deform_dlo45_decision_directed_sensing_v3 import (
    evaluate as module,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "experiments"
    / "deform_dlo45_decision_directed_sensing_v3"
    / "protocol.json"
)
SCRIPT = (
    ROOT
    / "experiments"
    / "deform_dlo45_decision_directed_sensing_v3"
    / "evaluate.py"
)


def protocol() -> dict[str, object]:
    return module.load_protocol(PROTOCOL)


def test_protocol_fixes_predecessor_and_keeps_official_eval_closed() -> None:
    value = protocol()
    predecessor = value["predecessor"]
    transport = value["transport_calibration"]
    evaluation = value["evaluation"]
    assert predecessor["fixed_likelihood_scale"] == 2.0
    assert predecessor["fixed_action_prototype_scale"] == 1.0
    assert predecessor["fixed_support_regret_tolerance"] == 0.05
    assert predecessor["fixed_measurement_budget"] == 4
    assert transport["miscoverage_level"] == 0.1
    assert evaluation["official_evaluation_split_opened"] is False
    assert evaluation["new_data_collection"] is False
    assert evaluation["target_tuning"] is False


def test_conformal_quantile_uses_finite_sample_rank() -> None:
    scores = [float(value) for value in range(18)]
    result = module.conformal_quantile(scores, 0.1)
    assert result["calibration_count"] == 18
    assert result["finite_sample_rank"] == 18
    assert result["additive_slack"] == 17.0


def test_conformal_quantile_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        module.conformal_quantile([], 0.1)
    with pytest.raises(ValueError, match="strictly"):
        module.conformal_quantile([1.0], 0.0)


def test_trajectory_score_is_mean_positive_certificate_excess() -> None:
    rows = [
        {
            "policy": "decision_regret",
            "budget": 4,
            "certified": True,
            "dlo": "DLO4",
            "trajectory": "a.pkl",
            "normalized_realized_regret": 0.2,
            "certificate_worst_case_regret": 0.1,
        },
        {
            "policy": "decision_regret",
            "budget": 4,
            "certified": True,
            "dlo": "DLO4",
            "trajectory": "a.pkl",
            "normalized_realized_regret": 0.05,
            "certificate_worst_case_regret": 0.1,
        },
    ]
    result = module.trajectory_certificate_scores(
        rows,
        policy="decision_regret",
        budget=4,
        minimum_certified=1,
    )
    assert len(result) == 1
    assert result[0]["mean_excess"] == pytest.approx(0.025)
    assert result[0]["mean_positive_excess"] == pytest.approx(0.05)


def test_exact_sign_test_and_holm_adjustment() -> None:
    assert module.one_sided_sign_test(16, 0) == pytest.approx(2.0**-16)
    assert module.one_sided_sign_test(0, 0) == 1.0
    adjusted = module.holm_adjust({"a": 0.001, "b": 0.02, "c": 0.2})
    assert adjusted["a"] == pytest.approx(0.003)
    assert adjusted["b"] == pytest.approx(0.04)
    assert adjusted["c"] == pytest.approx(0.2)


def test_transport_calibration_operates_on_complete_trajectory_units() -> None:
    value = protocol()
    calibration = []
    source_test = []
    for dlo in module.DLOS:
        for index in range(9):
            calibration.append(
                {
                    "policy": "decision_regret",
                    "budget": 4,
                    "certified": True,
                    "dlo": dlo,
                    "trajectory": f"cal-{dlo}-{index}.pkl",
                    "normalized_realized_regret": 0.01 * index,
                    "certificate_worst_case_regret": 0.0,
                }
            )
        for index in range(8):
            source_test.append(
                {
                    "policy": "decision_regret",
                    "budget": 4,
                    "certified": True,
                    "dlo": dlo,
                    "trajectory": f"test-{dlo}-{index}.pkl",
                    "normalized_realized_regret": 0.005 * index,
                    "certificate_worst_case_regret": 0.0,
                }
            )
    result = module.transport_calibration(calibration, source_test, value)
    assert result["quantile"]["calibration_count"] == 18
    assert result["source_test_trajectory_count"] == 16
    assert result["source_test_coverage_fraction"] == 1.0


def test_overlap_audit_detects_predecessor_reuse() -> None:
    value = protocol()
    excluded = value["predecessor"]["excluded_source_test_roster"]
    core_result = {
        "source_split": {
            dlo: {
                "source_test": [excluded[dlo][0]]
                + [f"fresh-{dlo}-{index}.pkl" for index in range(7)]
            }
            for dlo in module.DLOS
        }
    }
    result = module.overlap_audit(core_result, value)
    assert result["total_overlap_count"] == 2
    for dlo in module.DLOS:
        assert result["by_dlo"][dlo]["overlap_count"] == 1


def test_wrapper_runs_core_before_opening_replication_outputs() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    section = text[text.index("def run(args: argparse.Namespace)") :]
    execute = section.index("core_status = core.run")
    read_result = section.index("core_result = read_json")
    assert execute < read_result
    assert "source_test_cases.jsonl" in section[read_result:]


def test_outer_protocol_is_valid_json() -> None:
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert value["contract"] == module.CONTRACT
