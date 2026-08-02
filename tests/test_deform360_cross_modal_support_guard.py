from __future__ import annotations

import numpy as np

from bayesian_phystwin.deform360_cross_modal_support_guard import (
    apply_cross_modal_support_guard,
    cross_modal_support_decisions,
    cross_modal_support_feature_vector,
    fit_cross_modal_support_guard,
)


def _source_rows() -> tuple[np.ndarray, np.ndarray, list[str]]:
    features = np.asarray(
        [
            [0.5, 0.10, 0.10, 0.20, 0.20],
            [2.0, 0.90, 0.10, 0.20, 0.20],
            [2.0, 0.10, 0.90, 0.20, 0.20],
            [2.0, 0.10, 0.10, -0.20, 0.90],
            [1.0, 0.20, 0.20, 0.00, 0.50],
            [2.5, 0.50, 0.30, 0.10, 0.40],
            [3.0, 0.30, 0.50, -0.30, 0.40],
            [0.4, 0.95, 0.95, -0.50, 0.99],
        ],
        dtype=np.float64,
    )
    regret = np.asarray((-0.1, -0.2, -0.3, -0.4, 0.1, 0.2, 0.3, 0.0))
    objects = ["a", "b", "c", "d", "a", "b", "c", "e"]
    return features, regret, objects


def _model():
    features, regret, objects = _source_rows()
    return fit_cross_modal_support_guard(
        features,
        regret,
        objects,
        candidate_nontrivial=regret != 0.0,
    )


def test_fit_adds_safe_conjunctive_route() -> None:
    model = _model()
    route = model.stable_tactile_coherent_correction
    assert route.enabled is True
    assert route.maximum_cumulative_energy_change == -0.2
    assert route.minimum_correction_coherence == 0.9
    assert route.source_beneficial_admission_count == 1
    assert route.source_regressive_admission_count == 0


def test_conjunctive_route_requires_both_modalities() -> None:
    decisions = cross_modal_support_decisions(
        np.asarray(
            [
                [2.0, 0.2, 0.2, -0.3, 0.95],
                [2.0, 0.2, 0.2, 0.1, 0.95],
                [2.0, 0.2, 0.2, -0.3, 0.5],
            ]
        ),
        _model(),
    )
    assert [row["support_available"] for row in decisions] == [True, False, False]
    assert decisions[0]["admitting_routes"] == [
        "stable_tactile_coherent_correction"
    ]
    assert decisions[1]["cross_modal_route"]["passed"] is False
    assert decisions[2]["cross_modal_route"]["passed"] is False


def test_v1_route_and_cross_modal_route_form_union() -> None:
    decisions = cross_modal_support_decisions(
        np.asarray(
            [
                [0.4, 0.2, 0.2, 0.1, 0.2],
                [2.0, 0.2, 0.2, -0.3, 0.95],
            ]
        ),
        _model(),
    )
    assert decisions[0]["admitting_routes"] == ["low_tactile_loading"]
    assert decisions[1]["admitting_routes"] == [
        "stable_tactile_coherent_correction"
    ]


def test_apply_preserves_exact_fallback_and_noop_semantics() -> None:
    baseline = np.arange(12, dtype=np.float64)[:, None]
    candidate = baseline.copy()
    candidate[2:4] += 10.0
    candidate[8:] += 30.0
    report, guarded = apply_cross_modal_support_guard(
        baseline,
        candidate,
        np.asarray(
            [
                [2.0, 0.2, 0.2, -0.3, 0.95],
                [2.0, 0.2, 0.2, -0.3, 0.95],
                [2.0, 0.2, 0.2, 0.1, 0.5],
            ]
        ),
        _model(),
        update_frames=(1, 4, 7),
    )
    np.testing.assert_array_equal(guarded[2:4], candidate[2:4])
    np.testing.assert_array_equal(guarded[5:7], baseline[5:7])
    np.testing.assert_array_equal(guarded[8:], baseline[8:])
    assert report["updates"][0]["candidate_accepted"] is True
    assert report["updates"][1]["reason"] == "exact-candidate-noop"
    assert report["updates"][2]["bit_exact_baseline_fallback"] is True


def test_feature_vector_keeps_cross_modal_inputs_separate() -> None:
    vector = cross_modal_support_feature_vector(
        {
            "sensor_ratio_max": 0.7,
            "cumulative_energy_change_from_frame0_fraction": -0.1,
        },
        {
            "correction_change_rms_over_object_scale": 0.08,
            "prior_consistency_gain": 0.75,
            "correction_coherence": 0.9,
        },
    )
    np.testing.assert_allclose(vector, (0.7, 0.08, 0.75, -0.1, 0.9))


def test_fit_is_row_order_invariant() -> None:
    features, regret, objects = _source_rows()
    order = np.asarray((4, 1, 7, 3, 0, 6, 5, 2))
    first = fit_cross_modal_support_guard(
        features,
        regret,
        objects,
        candidate_nontrivial=regret != 0.0,
    )
    second = fit_cross_modal_support_guard(
        features[order],
        regret[order],
        [objects[index] for index in order],
        candidate_nontrivial=(regret != 0.0)[order],
    )
    assert second == first
