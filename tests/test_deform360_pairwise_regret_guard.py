from __future__ import annotations

import numpy as np

from bayesian_phystwin.bias_aware_belief import fit_source_regret_certificate
from bayesian_phystwin.deform360_pairwise_regret_guard import (
    DUAL_BACKBONE_ARM,
    PRIOR_CORRECTION_AT_UPDATE,
    PRIOR_VARIANCE_AT_UPDATE,
    SELECTED_BACKBONE_ARM,
    SELECTED_BACKBONE_AT_UPDATE,
    apply_pairwise_regret_certificate,
    pairwise_regret_features,
    predict_dual_backbone_pairwise_rbf_arrays,
)
from bayesian_phystwin.deform360_raw_pairwise_correspondence_diagnostic import (
    CLIQUE_RBF_ARM,
    SELECTED_RAW_ARM,
    evaluate_raw_pairwise_correspondence_arrays,
)


def _case() -> tuple[np.ndarray, ...]:
    frames = 76
    x = np.linspace(0.0, 0.11, 12)
    frame_zero = np.column_stack((x, np.zeros_like(x), np.zeros_like(x)))
    physical = np.repeat(frame_zero[None], frames, axis=0)
    persistence = physical.copy()
    physical[:, :, 1] += np.linspace(0.0, 0.006, frames)[:, None]
    measurement = physical.copy()
    measurement[19:, :, 0] += 0.004
    visible = np.ones((frames, len(x)), dtype=bool)
    valid = visible.copy()
    centers = np.arange(9, dtype=np.int64)
    return physical, persistence, measurement, visible, valid, centers


def test_target_free_dual_backbone_matches_diagnostic_candidate() -> None:
    physical, persistence, measurement, visible, valid, centers = _case()
    report, arrays = predict_dual_backbone_pairwise_rbf_arrays(
        physical,
        persistence,
        measurement,
        visible,
        valid,
        center_ids=centers,
    )
    target = physical.copy()
    diagnostic, diagnostic_arrays = evaluate_raw_pairwise_correspondence_arrays(
        physical,
        persistence,
        target,
        visible,
        valid,
        measurement,
        visible,
        valid,
        center_ids=centers,
        scored_frames=tuple(range(20, 76)),
    )

    np.testing.assert_array_equal(
        arrays[SELECTED_BACKBONE_ARM],
        diagnostic_arrays[SELECTED_RAW_ARM],
    )
    np.testing.assert_array_equal(
        arrays[DUAL_BACKBONE_ARM],
        diagnostic_arrays[CLIQUE_RBF_ARM],
    )
    assert [row["selected_backbone"] for row in report["updates"]] == [
        row["selected_backbone"] for row in diagnostic["updates"]
    ]


def test_pairwise_regret_features_are_target_free_and_finite() -> None:
    physical, persistence, measurement, visible, valid, centers = _case()
    _, arrays = predict_dual_backbone_pairwise_rbf_arrays(
        physical,
        persistence,
        measurement,
        visible,
        valid,
        center_ids=centers,
    )
    features, diagnostics = pairwise_regret_features(
        physical,
        arrays[SELECTED_BACKBONE_ARM],
        arrays[DUAL_BACKBONE_ARM],
        measurement,
        visible,
        valid,
        center_ids=centers,
        update_frame=19,
        previous_update_frame=0,
        interval_end_exclusive=38,
        inlier_view_count=np.full(len(centers), 2),
        prior_correction_at_update_m=arrays[PRIOR_CORRECTION_AT_UPDATE][19],
        prior_variance_at_update_m2=arrays[PRIOR_VARIANCE_AT_UPDATE][19],
        selected_backbone_at_update_m=arrays[SELECTED_BACKBONE_AT_UPDATE][19],
    )

    assert features.shape == (15,)
    assert np.all(np.isfinite(features))
    assert diagnostics["available_center_count"] == len(centers)
    assert diagnostics["prior_predicted_residual_rms_m"] == 0.0


def test_rejected_regret_interval_is_bit_exact_baseline() -> None:
    physical, persistence, measurement, visible, valid, centers = _case()
    _, arrays = predict_dual_backbone_pairwise_rbf_arrays(
        physical,
        persistence,
        measurement,
        visible,
        valid,
        center_ids=centers,
    )
    feature_rows = []
    for previous, update, stop in ((0, 19, 38), (19, 38, 57), (38, 57, 76)):
        features, _ = pairwise_regret_features(
            physical,
            arrays[SELECTED_BACKBONE_ARM],
            arrays[DUAL_BACKBONE_ARM],
            measurement,
            visible,
            valid,
            center_ids=centers,
            update_frame=update,
            previous_update_frame=previous,
            interval_end_exclusive=stop,
            prior_correction_at_update_m=arrays[
                PRIOR_CORRECTION_AT_UPDATE
            ][update],
            prior_variance_at_update_m2=arrays[
                PRIOR_VARIANCE_AT_UPDATE
            ][update],
            selected_backbone_at_update_m=arrays[
                SELECTED_BACKBONE_AT_UPDATE
            ][update],
        )
        feature_rows.append(features)
    features = np.asarray(feature_rows)
    training = np.repeat(features, 3, axis=0)
    certificate = fit_source_regret_certificate(
        training,
        np.full(len(training), 0.01),
        ("a",) * 3 + ("b",) * 3 + ("c",) * 3,
        nominal_coverage=0.5,
    )
    report, guarded = apply_pairwise_regret_certificate(
        arrays[SELECTED_BACKBONE_ARM],
        arrays[DUAL_BACKBONE_ARM],
        features,
        certificate,
    )

    np.testing.assert_array_equal(guarded, arrays[SELECTED_BACKBONE_ARM])
    assert all(row["bit_exact_baseline_fallback"] for row in report["updates"])
