from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.deform360_causal_support_guard import (
    CausalSupportGuardModel,
    apply_causal_support_guard,
    causal_support_decisions,
    causal_support_feature_vector,
    fit_causal_support_guard,
)


def _source_rows() -> tuple[np.ndarray, np.ndarray, list[str]]:
    features = np.asarray(
        [
            [0.5, 0.10, 0.10],
            [2.0, 0.90, 0.10],
            [2.0, 0.10, 0.90],
            [1.0, 0.20, 0.20],
            [2.5, 0.50, 0.30],
            [3.0, 0.30, 0.50],
            [0.4, 0.95, 0.95],
        ],
        dtype=np.float64,
    )
    regret = np.asarray((-0.1, -0.2, -0.3, 0.1, 0.2, 0.3, 0.0))
    objects = ["a", "b", "c", "a", "b", "c", "d"]
    return features, regret, objects


def _model() -> CausalSupportGuardModel:
    features, regret, objects = _source_rows()
    return fit_causal_support_guard(
        features,
        regret,
        objects,
        candidate_nontrivial=[True, True, True, True, True, True, False],
    )


def test_fit_selects_safe_monotone_routes() -> None:
    model = _model()
    assert [route.threshold for route in model.routes] == [0.5, 0.9, 0.9]
    assert all(route.enabled for route in model.routes)
    assert all(route.source_regressive_admission_count == 0 for route in model.routes)
    assert model.source_object_count == 4
    assert model.source_row_count == 7
    assert model.source_informative_row_count == 6


def test_fit_is_invariant_to_source_row_order() -> None:
    features, regret, objects = _source_rows()
    order = np.asarray((4, 1, 6, 3, 0, 5, 2))
    first = fit_causal_support_guard(
        features,
        regret,
        objects,
        candidate_nontrivial=regret != 0.0,
    )
    second = fit_causal_support_guard(
        features[order],
        regret[order],
        [objects[index] for index in order],
        candidate_nontrivial=(regret != 0.0)[order],
    )
    assert second == first


def test_decision_union_rejects_unsupported_camera_only_row() -> None:
    decisions = causal_support_decisions(
        np.asarray(
            [
                [0.4, 0.2, 0.2],
                [2.0, 0.95, 0.2],
                [2.0, 0.2, 0.95],
                [2.0, 0.2, 0.2],
            ]
        ),
        _model(),
    )
    assert [row["support_available"] for row in decisions] == [
        True,
        True,
        True,
        False,
    ]
    assert decisions[0]["admitting_routes"] == ["low_tactile_loading"]
    assert decisions[-1]["admitting_routes"] == []


def test_apply_preserves_exact_fallback_and_does_not_admit_noop() -> None:
    baseline = np.arange(12, dtype=np.float64)[:, None]
    candidate = baseline.copy()
    candidate[2:4] += 10.0
    candidate[8:] += 30.0
    report, guarded = apply_causal_support_guard(
        baseline,
        candidate,
        np.asarray(
            [
                [0.4, 0.2, 0.2],
                [2.0, 0.2, 0.2],
                [0.4, 0.2, 0.2],
            ]
        ),
        _model(),
        update_frames=(1, 4, 7),
    )
    np.testing.assert_array_equal(guarded[2:4], candidate[2:4])
    np.testing.assert_array_equal(guarded[5:7], baseline[5:7])
    np.testing.assert_array_equal(guarded[8:], candidate[8:])
    updates = report["updates"]
    assert updates[0]["candidate_accepted"] is True
    assert updates[1]["reason"] == "exact-candidate-noop"
    assert updates[1]["candidate_accepted"] is False
    assert updates[1]["bit_exact_baseline_fallback"] is True
    assert updates[2]["candidate_accepted"] is True


def test_rejected_nontrivial_candidate_is_bit_exact_baseline() -> None:
    baseline = np.zeros((8, 3), dtype=np.float32)
    candidate = np.ones_like(baseline)
    report, guarded = apply_causal_support_guard(
        baseline,
        candidate,
        np.asarray([[2.0, 0.2, 0.2]]),
        _model(),
        update_frames=(2,),
    )
    assert np.array_equal(guarded, baseline)
    assert guarded.dtype == baseline.dtype
    assert report["updates"][0]["candidate_accepted"] is False
    assert report["updates"][0]["bit_exact_baseline_fallback"] is True


def test_feature_vector_separates_tactile_and_pairwise_inputs() -> None:
    vector = causal_support_feature_vector(
        {"sensor_ratio_max": 0.7},
        {
            "correction_change_rms_over_object_scale": 0.08,
            "prior_consistency_gain": 0.75,
        },
    )
    np.testing.assert_allclose(vector, (0.7, 0.08, 0.75))
    with pytest.raises(KeyError):
        causal_support_feature_vector({}, {})


def test_fit_rejects_bad_source_contracts() -> None:
    features, regret, objects = _source_rows()
    with pytest.raises(ValueError, match="multiple objects"):
        fit_causal_support_guard(features, regret, ["same"] * len(objects))
    with pytest.raises(ValueError, match="no candidate updates"):
        fit_causal_support_guard(
            features,
            regret,
            objects,
            candidate_nontrivial=[False] * len(objects),
        )
