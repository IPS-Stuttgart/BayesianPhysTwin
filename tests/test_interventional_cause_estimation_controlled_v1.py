from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_runner():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/science/run_interventional_cause_estimation_design_v1.py"
    )
    spec = importlib.util.spec_from_file_location("cause_estimation_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registered_controlled_design_passes() -> None:
    runner = _load_runner()
    result = runner.run_study(
        trial_count=2_000,
        seed=20_260_902,
        noise_standard_deviation=0.05,
    )

    assert result["passed"]
    assert result["decision"] == (
        "minimum-cost-interventions-recover-attribution-with-calibrated-uncertainty"
    )
    assert result["plan"]["selected_intervention_ids"] == [
        "action-0-source",
        "action-1-view-change",
        "action-3-control-change",
    ]
    planned = result["methods"]["minimum_cost_planned_portfolio"]
    random = result["methods"]["random_equal_count_portfolio"]
    wrong = result["methods"]["wrong_action_relation"]
    assert planned["resolved_coverage"] == 1.0
    assert planned["accuracy_among_resolved"] >= 0.99
    assert 0.93 <= planned["nominal_95_coverage"] <= 0.97
    assert random["resolved_coverage"] < planned["resolved_coverage"]
    assert wrong["accuracy_among_resolved"] < planned["accuracy_among_resolved"]
