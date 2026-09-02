"""Focused contracts for the DEFORM support-adequacy prototype."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.deform_dlo45_decision_identifiability_v1.support_adequacy_audit import (
    FEATURE_NAMES,
    RidgeModel,
    _threshold_candidates,
    fit_ridge,
    load_support_protocol,
    predict_ridge,
    validate_request,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "experiments"
    / "deform_dlo45_decision_identifiability_v1"
    / "support_adequacy_protocol.json"
)
REQUEST = ROOT / ".github" / "requests" / "deform-dlo45-support-adequacy-v1.json"


def test_protocol_and_request_keep_target_selection_closed() -> None:
    protocol = load_support_protocol(PROTOCOL)
    request = validate_request(REQUEST, protocol)

    assert protocol.parent_workflow_run_id == 33473378340
    assert protocol.tolerance == 0.05
    assert request["target_tuning"] is False
    assert request["target_retries"] is False
    assert request["paper_claim_authorized"] is False


def test_registered_features_exclude_future_residual_distance() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    assert protocol["future_residual_distance_used_for_policy"] is False
    assert "nearest_selected_residual_rmse" not in FEATURE_NAMES
    assert "nearest_global_residual_rmse" not in FEATURE_NAMES
    assert "selected_feature_distance_median" in FEATURE_NAMES


def test_group_weighted_ridge_serializes_exactly() -> None:
    x = np.arange(6 * len(FEATURE_NAMES), dtype=np.float64).reshape(
        6,
        len(FEATURE_NAMES),
    )
    y = np.asarray([0.0, 0.2, 0.4, 1.0, 1.2, 1.4])
    groups = ["a", "a", "a", "b", "b", "b"]
    model = fit_ridge(x, y, groups, 1.0)
    restored = RidgeModel.from_record(model.record())

    assert np.allclose(
        predict_ridge(model, x),
        predict_ridge(restored, x),
    )
    assert np.all(np.isfinite(predict_ridge(restored, x)))


def test_threshold_candidates_include_a_select_none_boundary() -> None:
    candidates = _threshold_candidates([0.2, 0.2, 0.4])

    assert candidates[0] < 0.2
    assert candidates[1:] == [0.2, 0.4]
