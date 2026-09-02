from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.deform_dlo45_decision_identifiability_v1.support_envelope import (
    WindowMeasurement,
    _selected_action_tensors,
    _summarize_policy,
    _trajectory_tensors,
    load_envelope_protocol,
    validate_request,
)


def _record(
    *,
    dlo: str,
    trajectory: str,
    current: int,
    base_action: int,
    realized_regret: tuple[float, float, float] = (0.4, 0.03, 0.1),
    physical_mse: tuple[float, float, float] = (4.0, 1.0, 2.0),
) -> WindowMeasurement:
    return WindowMeasurement(
        stable_id=f"{dlo}/{trajectory}/{current}",
        dlo=dlo,
        trajectory=trajectory,
        current_frame=current,
        registered_regret=np.asarray([0.5, 0.02, 0.2]),
        realized_regret=np.asarray(realized_regret),
        physical_mse=np.asarray(physical_mse),
        fallback_mse=float(physical_mse[0]),
        base_certificate_action=base_action,
    )


def _protocol_value() -> dict[str, object]:
    return {
        "contract": "deform-dlo45-conformal-regret-envelope-v1",
        "schema_version": 1,
        "parent_contract": "deform-dlo45-decision-identifiability-v1",
        "parent_workflow_run_id": 33473378340,
        "parent_source_result_sha256": "a" * 64,
        "dataset_repository": "roahmlab/DEFORM",
        "dataset_commit": "b" * 40,
        "calibration": {
            "partition": "source_test",
            "unit": "complete_trajectory",
            "candidate_action_mask": [False, True, True],
            "miscoverage_levels": [0.1, 0.2, 0.3],
            "primary_miscoverage": 0.2,
        },
        "decision": {
            "regret_budget_grid": [0.05, 0.3, 1.0],
            "primary_regret_budget": 0.3,
        },
        "bootstrap": {"replicates": 100, "seed": 1},
        "claim_boundary": "bounded test protocol",
    }


def test_protocol_and_request_are_content_bound(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(_protocol_value()), encoding="utf-8")
    protocol = load_envelope_protocol(protocol_path)
    request_path = tmp_path / "request.json"
    request = {
        "contract": "deform-dlo45-conformal-regret-envelope-request-v1",
        "schema_version": 1,
        "status": "authorized",
        "run_key": "unit-test",
        "parent_workflow_run_id": 33473378340,
        "protocol_sha256": "expected",
        "dlos": ["DLO4", "DLO5"],
        "source_only_calibration": True,
        "target_tuning": False,
        "target_retries": False,
        "report_complete_frontier": True,
    }
    request_path.write_text(json.dumps(request), encoding="utf-8")
    assert validate_request(request_path, protocol, "expected")["run_key"] == (
        "unit-test"
    )
    with pytest.raises(ValueError, match="invalid conformal-regret-envelope request"):
        validate_request(request_path, protocol, "different")


def test_trajectory_tensors_order_frames_and_zero_fallback_selections() -> None:
    records = [
        _record(dlo="DLO4", trajectory="b.pkl", current=29, base_action=1),
        _record(dlo="DLO4", trajectory="a.pkl", current=29, base_action=0),
        _record(dlo="DLO4", trajectory="a.pkl", current=4, base_action=2),
        _record(dlo="DLO4", trajectory="b.pkl", current=4, base_action=1),
    ]
    names, realized, registered, actions = _trajectory_tensors(records, 3)
    assert names == ("a.pkl", "b.pkl")
    assert realized.shape == (2, 2, 3)
    np.testing.assert_array_equal(actions, [[2, 0], [1, 1]])
    selected_realized, selected_registered = _selected_action_tensors(
        realized, registered, actions, 0
    )
    assert selected_realized.shape == (2, 2, 1)
    assert selected_realized[0, 1, 0] == 0.0
    assert selected_registered[0, 1, 0] == 0.0


def test_policy_summary_keeps_equal_filenames_in_distinct_dlo_units() -> None:
    records = [
        _record(
            dlo="DLO4",
            trajectory="1.pkl",
            current=4,
            base_action=1,
            realized_regret=(0.4, 0.4, 0.1),
        ),
        _record(
            dlo="DLO5",
            trajectory="1.pkl",
            current=4,
            base_action=1,
            realized_regret=(0.4, 0.1, 0.2),
        ),
    ]
    summary = _summarize_policy(
        records,
        [1, 1],
        regret_budget=0.3,
        bootstrap_replicates=100,
        bootstrap_seed=5,
    )
    assert summary.nonfallback_count == 2
    assert summary.trajectory_budget_violation_count == 1
    assert summary.trajectory_budget_violation_fraction == pytest.approx(0.5)


def test_policy_summary_reports_exact_fallback_without_division_failure() -> None:
    records = [
        _record(dlo="DLO4", trajectory="a.pkl", current=4, base_action=0),
        _record(dlo="DLO4", trajectory="a.pkl", current=29, base_action=0),
    ]
    summary = _summarize_policy(
        records,
        [0, 0],
        regret_budget=0.05,
        bootstrap_replicates=100,
        bootstrap_seed=9,
    )
    assert summary.nonfallback_count == 0
    assert summary.harmful_nonfallback_count == 0
    assert summary.budget_violation_count_nonfallback == 0
    assert summary.rmse_ratio_to_fallback == pytest.approx(1.0)
    assert summary.rmse_reduction == pytest.approx(0.0)
