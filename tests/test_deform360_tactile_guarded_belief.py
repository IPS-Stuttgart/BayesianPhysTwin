from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bayesian_phystwin.deform360_pairwise_regret_guard import (
    DUAL_BACKBONE_ARM,
    SELECTED_BACKBONE_ARM,
)
from bayesian_phystwin.deform360_tactile_guarded_belief import (
    TACTILE_GUARDED_ARM,
    predict_tactile_guarded_belief_arrays,
    tactile_guard_model_from_source_result,
)
from bayesian_phystwin.deform360_tactile_regret_guard import (
    TACTILE_REGRET_FEATURE_NAMES,
    TactileRegretGuardModel,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_RESULT = (
    ROOT
    / "results/sota/diagnostics/deform360_tactile_regret_guard_source_v1/result.json"
)


def _inputs() -> tuple[np.ndarray, ...]:
    frame_count = 76
    point_count = 16
    frame_zero = np.zeros((point_count, 3), dtype=np.float64)
    frame_zero[:, 0] = np.linspace(0.0, 0.15, point_count)
    physical = np.repeat(frame_zero[None], frame_count, axis=0)
    persistence = physical.copy()
    measurement = physical.copy()
    for update in (19, 38, 57):
        measurement[update, :, 1] += 0.01
    visible = np.ones((frame_count, point_count), dtype=bool)
    valid = visible.copy()
    features = np.zeros((3, len(TACTILE_REGRET_FEATURE_NAMES)))
    return physical, persistence, measurement, visible, valid, features


def _model(intercept: float) -> TactileRegretGuardModel:
    feature_count = len(TACTILE_REGRET_FEATURE_NAMES)
    return TactileRegretGuardModel(
        feature_center=(0.0,) * feature_count,
        feature_scale=(1.0,) * feature_count,
        coefficients=(intercept,) + (0.0,) * feature_count,
        ridge_penalty=10.0,
        admission_threshold=0.7,
        source_object_count=2,
        source_row_count=6,
    )


def test_rejection_is_bit_exact_selected_backbone() -> None:
    report, arrays = predict_tactile_guarded_belief_arrays(
        *_inputs(),
        _model(0.0),
        center_ids=np.arange(16),
    )
    assert np.array_equal(arrays[TACTILE_GUARDED_ARM], arrays[SELECTED_BACKBONE_ARM])
    assert all(
        row["bit_exact_baseline_fallback"]
        for row in report["tactile_guard"]["updates"]
    )


def test_acceptance_selects_the_frozen_camera_candidate() -> None:
    report, arrays = predict_tactile_guarded_belief_arrays(
        *_inputs(),
        _model(1.0),
        center_ids=np.arange(16),
    )
    assert np.array_equal(arrays[TACTILE_GUARDED_ARM], arrays[DUAL_BACKBONE_ARM])
    assert all(row["candidate_accepted"] for row in report["tactile_guard"]["updates"])
    assert report["information_boundary"]["tactile_features_use_camera_state_residual"] is False


def test_source_model_loads_without_refitting() -> None:
    source = json.loads(SOURCE_RESULT.read_text(encoding="utf-8"))
    model = tactile_guard_model_from_source_result(source)
    expected = source["full_source_model_for_future_lock"]
    assert model.admission_threshold == 0.7
    assert model.coefficients == tuple(expected["coefficients"])
    assert model.source_object_count == 17
    assert model.source_row_count == 117
