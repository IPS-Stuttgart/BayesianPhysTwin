from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from experiments.deform_dlo45_support_adequacy_v1.run import (
    FEATURE_NAMES,
    FORBIDDEN_FEATURE_TOKENS,
    LinearRiskModel,
    _finite_group_quantile,
    fit_logistic,
    fit_ridge,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT / "experiments" / "deform_dlo45_support_adequacy_v1" / "protocol.json"
)


def test_feature_roster_is_explicitly_outcome_free() -> None:
    assert len(FEATURE_NAMES) == 15
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)
    for name in FEATURE_NAMES:
        assert not any(token in name for token in FORBIDDEN_FEATURE_TOKENS)


def test_protocol_forbids_outcome_and_suffix_gate_inputs() -> None:
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert tuple(value["outcome_free_features"]) == FEATURE_NAMES
    assert value["target_outcomes_used_for_model_or_threshold_selection"] is False
    assert value["target_retries_authorized"] is False
    forbidden = set(value["forbidden_gate_inputs"])
    assert "nearest_selected_residual_rmse" in forbidden
    assert "certificate_realized_regret" in forbidden
    assert "oracle_action" in forbidden
    assert "future_internal_node_coordinates" in forbidden


def test_linear_models_round_trip_and_rank_expected_risk() -> None:
    values = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [2.0, 3.0],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([False, False, False, True, True, True])
    model = fit_logistic(values, labels, 1.0)
    probabilities = model.predict_bad_probability(values)
    assert np.all(np.isfinite(probabilities))
    assert np.mean(probabilities[3:]) > np.mean(probabilities[:3])

    record = model.to_record()
    record["feature_names"] = list(FEATURE_NAMES)
    record["feature_mean"] = [0.0] * len(FEATURE_NAMES)
    record["feature_scale"] = [1.0] * len(FEATURE_NAMES)
    record["coefficients"] = [0.0] * len(FEATURE_NAMES)
    restored = LinearRiskModel.from_record(record)
    assert restored.feature_names == FEATURE_NAMES

    targets = np.asarray([0.0, 0.5, 0.5, 1.0, 2.0, 2.5])
    ridge = fit_ridge(values, targets, 1.0)
    assert np.all(np.isfinite(ridge.predict_excess(values)))


def test_group_conformal_quantile_uses_finite_sample_rank() -> None:
    assert math.isclose(
        _finite_group_quantile([0.0, 0.1, 0.2, 0.3], 0.25),
        0.3,
        rel_tol=0.0,
        abs_tol=0.0,
    )
    assert math.isinf(_finite_group_quantile([0.0, 0.1, 0.2], 0.01))
