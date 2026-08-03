from __future__ import annotations

import inspect

import numpy as np
import pytest

from bayesian_phystwin.bias_aware_belief import SourceRegretCertificate
from bayesian_phystwin.deform360_pairwise_regret_guard import (
    FEATURE_NAMES,
    PairwiseRegretGuardConfig,
    apply_pairwise_regret_guard,
    build_pairwise_regret_candidate_arrays,
)
from bayesian_phystwin.deform360_raw_pairwise_correspondence_diagnostic import (
    CLIQUE_RBF_ARM,
    evaluate_raw_pairwise_correspondence_arrays,
)


def _inputs(*, physical_motion: bool = True) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(73)
    frame_count = 76
    point_count = 24
    frame_zero = rng.uniform(-0.2, 0.2, size=(point_count, 3)).astype(np.float32)
    persistence = np.repeat(frame_zero[None], frame_count, axis=0)
    physical = persistence.copy()
    if physical_motion:
        physical[:, :, 0] += np.arange(frame_count)[:, None] * 0.001
    measurement = np.full_like(physical, np.nan)
    visible = np.zeros((frame_count, point_count), dtype=bool)
    valid = visible.copy()
    centers = np.arange(16, dtype=np.int64)
    measurement[0, centers] = physical[0, centers]
    visible[0, centers] = True
    valid[0, centers] = True
    for index, frame in enumerate((19, 38, 57), start=1):
        measurement[frame, centers] = physical[frame, centers]
        measurement[frame, centers, 1] += 0.002 * index
        visible[frame, centers] = True
        valid[frame, centers] = True
    cameras = np.asarray(["cam0", "cam1", "cam2", "cam3"])
    view_count = np.full((3, len(centers)), 4, dtype=np.int16)
    reprojection = np.full((3, len(centers)), 0.5, dtype=np.float32)
    return (
        physical,
        persistence,
        measurement,
        visible,
        valid,
        centers,
        cameras,
        view_count,
        reprojection,
    )


def _build(values: tuple[np.ndarray, ...]):
    (
        physical,
        persistence,
        measurement,
        visible,
        valid,
        centers,
        cameras,
        view_count,
        reprojection,
    ) = values
    return build_pairwise_regret_candidate_arrays(
        physical,
        persistence,
        measurement,
        visible,
        valid,
        center_ids=centers,
        selected_camera_ids=cameras,
        triangulation_inlier_view_count=view_count,
        triangulation_median_reprojection_px=reprojection,
    )


def _certificate(features: np.ndarray, *, upper_regret: float):
    zeros = np.zeros(len(features), dtype=np.float64)
    return SourceRegretCertificate(
        feature_center=features,
        feature_scale=np.ones(len(features)),
        standardized_feature_lower=zeros,
        standardized_feature_upper=zeros,
        coefficients=np.r_[upper_regret, zeros],
        upper_residual_quantile=0.0,
        nominal_coverage=0.80,
        minimum_improvement=0.0,
        ridge_penalty=1.0,
        support_margin_std=0.0,
        source_group_count=4,
        finite_sample_rank=4,
        finite_sample_coverage=0.80,
    )


def test_candidate_builder_cannot_accept_target_or_outcome() -> None:
    parameters = inspect.signature(build_pairwise_regret_candidate_arrays).parameters

    assert "target" not in parameters
    assert "outcome" not in parameters


def test_aligned_redundant_physical_response_produces_candidate() -> None:
    report, baseline, candidate = _build(_inputs())

    assert report["candidate_available_count"] == 3
    assert all(update["candidate_available"] for update in report["updates"])
    assert not np.array_equal(candidate, baseline)
    assert all(len(update["features"]) == len(FEATURE_NAMES) for update in report["updates"])


def test_permissive_builder_preserves_exact_source_positive_candidate() -> None:
    values = _inputs()
    (
        physical,
        persistence,
        measurement,
        visible,
        valid,
        centers,
        cameras,
        view_count,
        reprojection,
    ) = values
    _, development = evaluate_raw_pairwise_correspondence_arrays(
        physical,
        persistence,
        physical,
        np.ones_like(visible),
        np.ones_like(valid),
        measurement,
        visible,
        valid,
        center_ids=centers,
        scored_frames=tuple(range(20, 76)),
    )
    config = PairwiseRegretGuardConfig(
        minimum_motion_center_count=1,
        minimum_physical_support_m=1e-12,
        minimum_observed_motion_m=1e-12,
        minimum_physical_agreement_gain=0.0,
        maximum_correction_to_physical_response=1e9,
    )

    _, _, candidate = build_pairwise_regret_candidate_arrays(
        physical,
        persistence,
        measurement,
        visible,
        valid,
        center_ids=centers,
        selected_camera_ids=cameras,
        triangulation_inlier_view_count=view_count,
        triangulation_median_reprojection_px=reprojection,
        config=config,
    )

    np.testing.assert_array_equal(candidate, development[CLIQUE_RBF_ARM])


def test_two_view_rows_do_not_claim_three_view_redundancy() -> None:
    values = list(_inputs())
    values[-2][:] = 2

    report, baseline, candidate = _build(tuple(values))

    assert report["candidate_available_count"] == 3
    assert not np.array_equal(candidate, baseline)
    assert all(
        update["redundant_center_count"] == 0
        and update["features"][FEATURE_NAMES.index("redundant_center_fraction")]
        == 0.0
        for update in report["updates"]
    )


def test_coherent_camera_shift_without_physical_response_falls_back() -> None:
    report, baseline, candidate = _build(_inputs(physical_motion=False))

    np.testing.assert_array_equal(candidate, baseline)
    assert all(
        "physical-support-too-small" in update["rejection_reasons"]
        for update in report["updates"]
    )


def test_correction_is_shrunk_without_changing_its_direction() -> None:
    values = list(_inputs())
    values[2][19, values[5], 1] += 0.08

    report, baseline, candidate = _build(tuple(values))

    first = report["updates"][0]
    assert first["candidate_available"]
    assert 0.0 < first["applied_correction_scale"] < 1.0
    delta = candidate[20] - baseline[20]
    assert np.any(delta)


def test_duplicate_camera_identity_cannot_inflate_redundancy() -> None:
    values = list(_inputs())
    values[-3] = np.asarray(["cam0", "cam1", "cam2", "cam2"])

    with pytest.raises(ValueError, match="not independent"):
        _build(tuple(values))


def test_missing_regret_certificate_is_exact_fallback() -> None:
    candidate_report, baseline, candidate = _build(_inputs())

    guard_report, selected = apply_pairwise_regret_guard(
        baseline, candidate, candidate_report, None
    )

    np.testing.assert_array_equal(selected, baseline)
    assert guard_report["accepted_count"] == 0
    assert guard_report["exact_fallback_count"] == 3


def test_negative_regret_bound_admits_only_bound_candidate() -> None:
    candidate_report, baseline, candidate = _build(_inputs())
    first_features = np.asarray(candidate_report["updates"][0]["features"])
    certificate = _certificate(first_features, upper_regret=-0.001)

    guard_report, selected = apply_pairwise_regret_guard(
        baseline, candidate, candidate_report, certificate
    )

    assert guard_report["accepted_count"] >= 1
    assert not np.array_equal(selected, baseline)


def test_nonnegative_regret_bound_is_exact_fallback() -> None:
    candidate_report, baseline, candidate = _build(_inputs())
    first_features = np.asarray(candidate_report["updates"][0]["features"])
    certificate = _certificate(first_features, upper_regret=0.001)

    guard_report, selected = apply_pairwise_regret_guard(
        baseline, candidate, candidate_report, certificate
    )

    np.testing.assert_array_equal(selected, baseline)
    assert guard_report["accepted_count"] == 0
    assert guard_report["exact_fallback_count"] == 3
