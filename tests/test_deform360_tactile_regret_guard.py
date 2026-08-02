from __future__ import annotations

import numpy as np

from bayesian_phystwin.deform360_tactile_regret_guard import (
    TACTILE_REGRET_FEATURE_NAMES,
    TactileRegretGuardModel,
    align_baseline_subtracted_tactile,
    apply_tactile_regret_guard,
    causal_tactile_regret_features,
    fit_object_balanced_tactile_regret_guard,
    tactile_benefit_scores,
)


def _tactile_response() -> np.ndarray:
    response = np.zeros((4, 76, 3, 4), dtype=np.float64)
    for sensor in range(4):
        response[sensor, :, sensor % 3, sensor % 4] = np.linspace(
            1.0 + sensor,
            3.0 + sensor,
            76,
        )
    response[:, 35:42, 1, 1] += 2.0
    return response


def _constant_model(intercept: float) -> TactileRegretGuardModel:
    count = len(TACTILE_REGRET_FEATURE_NAMES)
    return TactileRegretGuardModel(
        feature_center=(0.0,) * count,
        feature_scale=(1.0,) * count,
        coefficients=(intercept,) + (0.0,) * count,
        ridge_penalty=10.0,
        admission_threshold=0.7,
        source_object_count=2,
        source_row_count=6,
    )


def test_raw_tactile_alignment_uses_earlier_tie_without_future_scale() -> None:
    frames = np.zeros((3, 2, 3), dtype=np.float32)
    frames[:, 0, 0] = (1.0, 3.0, 7.0)
    frames[:, 0, 2] = 100.0
    result = align_baseline_subtracted_tactile(
        frames,
        np.asarray((100, 200, 300)),
        np.ones((2, 3)),
        np.asarray((150, 290)),
        invalid_columns=(-1,),
    )

    np.testing.assert_array_equal(result.source_indices, (0, 2))
    np.testing.assert_array_equal(result.signed_delta_us, (-50, 10))
    np.testing.assert_array_equal(result.response[:, 0, 0], (0.0, 6.0))
    np.testing.assert_array_equal(result.response[:, :, -1], 0.0)


def test_tactile_features_do_not_read_after_the_update() -> None:
    response = _tactile_response()
    original, diagnostics = causal_tactile_regret_features(response)
    mutated = response.copy()
    mutated[:, 20:] += 1_000_000.0
    changed, _ = causal_tactile_regret_features(mutated)

    np.testing.assert_array_equal(changed[0], original[0])
    assert not np.array_equal(changed[1], original[1])
    assert original.shape == (3, len(TACTILE_REGRET_FEATURE_NAMES))
    assert diagnostics[0]["update_frame"] == 19


def test_object_balanced_source_fit_returns_finite_scores() -> None:
    features = np.vstack(
        (
            np.zeros((2, len(TACTILE_REGRET_FEATURE_NAMES))),
            np.ones((2, len(TACTILE_REGRET_FEATURE_NAMES))),
            np.full((2, len(TACTILE_REGRET_FEATURE_NAMES)), 2.0),
        )
    )
    model = fit_object_balanced_tactile_regret_guard(
        features,
        np.asarray((-0.1, -0.1, 0.1, 0.1, -0.2, -0.2)),
        ("a", "a", "b", "b", "c", "c"),
    )
    scores = tactile_benefit_scores(features, model)

    assert model.source_object_count == 3
    assert model.source_row_count == 6
    assert scores.shape == (6,)
    assert np.all(np.isfinite(scores))


def test_rejected_tactile_updates_preserve_baseline_bit_exactly() -> None:
    baseline = np.arange(76 * 2 * 3, dtype=np.float32).reshape(76, 2, 3)
    candidate = baseline + np.float32(1.0)
    features = np.zeros((3, len(TACTILE_REGRET_FEATURE_NAMES)))
    report, guarded = apply_tactile_regret_guard(
        baseline,
        candidate,
        features,
        _constant_model(0.0),
    )

    np.testing.assert_array_equal(guarded, baseline)
    assert all(row["bit_exact_baseline_fallback"] for row in report["updates"])
    assert report["information_boundary"]["future_tactile_read"] is False


def test_accepted_tactile_updates_select_candidate_intervals() -> None:
    baseline = np.zeros((76, 2, 3), dtype=np.float32)
    candidate = np.ones_like(baseline)
    features = np.zeros((3, len(TACTILE_REGRET_FEATURE_NAMES)))
    report, guarded = apply_tactile_regret_guard(
        baseline,
        candidate,
        features,
        _constant_model(0.8),
    )

    baseline_frames = np.asarray((19, 38, 57))
    np.testing.assert_array_equal(guarded[:20], baseline[:20])
    np.testing.assert_array_equal(guarded[baseline_frames], baseline[baseline_frames])
    continuation = np.ones(len(guarded), dtype=bool)
    continuation[:20] = False
    continuation[baseline_frames] = False
    np.testing.assert_array_equal(guarded[continuation], candidate[continuation])
    assert all(row["candidate_accepted"] for row in report["updates"])
