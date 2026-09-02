from __future__ import annotations

import importlib.util
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "deform_dlo45_conformal_active_sensing_v1"
    / "analyze.py"
)
SPEC = importlib.util.spec_from_file_location(
    "deform_dlo45_conformal_active_sensing_v1_analysis",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def row(
    *,
    dlo: str,
    trajectory: str,
    nonfallback: bool,
    realized: float,
    bound: float,
    physical_mse: float = 1.0,
    fallback_mse: float = 4.0,
) -> dict[str, object]:
    return {
        "policy": "decision_regret",
        "budget": 4,
        "dlo": dlo,
        "trajectory": trajectory,
        "nonfallback": nonfallback,
        "normalized_realized_regret": realized,
        "certificate_worst_case_regret": bound,
        "physical_task_mse": physical_mse,
        "fallback_task_mse": fallback_mse,
        "sensor_count": 1,
        "effective_hypothesis_count": 2.0,
    }


def test_conformal_quantile_uses_finite_sample_ceiling() -> None:
    values = [float(index) for index in range(1, 10)]
    at_eighty = MODULE.conformal_quantile(values, 0.2)
    at_ninety = MODULE.conformal_quantile(values, 0.1)
    at_ninety_five = MODULE.conformal_quantile(values, 0.05)
    assert at_eighty["finite_sample_rank"] == 8
    assert at_eighty["radius"] == 8.0
    assert at_ninety["finite_sample_rank"] == 9
    assert at_ninety["radius"] == 9.0
    assert at_ninety_five["finite_sample_rank"] == 10
    assert at_ninety_five["radius"] == "infinite"


def test_trajectory_score_is_maximum_positive_excess() -> None:
    rows = []
    for decision in range(19):
        rows.append(
            row(
                dlo="DLO4",
                trajectory="one.pkl",
                nonfallback=decision in {2, 7, 11},
                realized={2: 0.10, 7: 0.01, 11: 0.40}.get(decision, 0.0),
                bound={2: 0.04, 7: 0.03, 11: 0.05}.get(decision, 0.0),
            )
        )
    score = MODULE.grouped_trajectory_scores(rows)
    assert len(score) == 1
    assert score[0]["parent_nonfallback_count"] == 3
    assert math.isclose(float(score[0]["score"]), 0.35)


def test_no_nonfallback_action_has_zero_vacuous_score() -> None:
    rows = [
        row(
            dlo="DLO5",
            trajectory="fallback.pkl",
            nonfallback=False,
            realized=0.9,
            bound=0.0,
        )
        for _ in range(19)
    ]
    score = MODULE.grouped_trajectory_scores(rows)
    assert score[0]["score"] == 0.0


def test_primary_budget_is_chosen_without_target_rows() -> None:
    calibration = [
        row(
            dlo="DLO4",
            trajectory="a.pkl",
            nonfallback=True,
            realized=0.0,
            bound=0.04,
        ),
        row(
            dlo="DLO5",
            trajectory="b.pkl",
            nonfallback=True,
            realized=0.0,
            bound=0.05,
        ),
    ]
    envelope = {
        "by_dlo": {
            "DLO4": {"radius": 0.20},
            "DLO5": {"radius": 0.18},
        }
    }
    selected = MODULE.choose_primary_budget(
        calibration,
        envelope,
        [0.20, 0.23, 0.25, 0.30],
    )
    assert selected == 0.25


def test_operational_wrapper_restores_fallback_when_bound_is_too_large() -> None:
    rows = []
    for dlo, trajectory in (("DLO4", "a.pkl"), ("DLO5", "b.pkl")):
        for _ in range(19):
            rows.append(
                row(
                    dlo=dlo,
                    trajectory=trajectory,
                    nonfallback=True,
                    realized=0.2,
                    bound=0.05,
                    physical_mse=1.0,
                    fallback_mse=4.0,
                )
            )
    rows = [
        {**item, "trajectory": f"{copy}-{item['trajectory']}"}
        for copy in range(8)
        for item in rows
    ]
    envelope = {
        "by_dlo": {
            "DLO4": {"radius": 0.30},
            "DLO5": {"radius": 0.30},
        }
    }
    result = MODULE.evaluate_frontier_point(
        rows,
        envelope,
        0.30,
        bootstrap_replicates=100,
        bootstrap_seed=1,
    )
    assert result["nonfallback_count"] == 0
    assert result["pooled_task_rmse_mm"] == 2000.0
    assert result["harmful_nonfallback_count"] == 0


def test_registered_artifact_reproduces_primary_numbers(tmp_path: Path) -> None:
    supplied = os.environ.get("DEFORM_DLO45_V3_PARENT_DIR")
    if not supplied:
        return
    parent = Path(supplied)
    protocol = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "deform_dlo45_conformal_active_sensing_v1"
        / "protocol.json"
    )
    output = tmp_path / "result"
    result = MODULE.run(parent, protocol, output)
    primary = result["primary_result"]
    assert result["primary_selection"]["selected_regret_budget"] == 0.3
    assert primary["nonfallback_count"] == 251
    assert primary["harmful_nonfallback_count"] == 0
    assert primary["envelope_exceed_trajectory_count"] == 3
    assert primary["regret_budget_exceed_trajectory_count"] == 2
    np.testing.assert_allclose(primary["pooled_task_rmse_mm"], 87.21625971864873)
    np.testing.assert_allclose(
        primary["mean_trajectory_rmse_reduction"],
        0.4810555327821773,
    )
    committed = protocol.with_name("compact_result.json")
    generated = output / "compact_result.json"
    assert generated.read_bytes() == committed.read_bytes()
    shutil.rmtree(output)
