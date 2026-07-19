import numpy as np
import pytest

import bayesian_phystwin.deform360_recursive_cpd_diagnostic as diagnostic
from bayesian_phystwin.cpd_registration import NonrigidCpdConfig, fit_nonrigid_cpd
from bayesian_phystwin.deform360_recursive_cpd_diagnostic import (
    ARMS,
    CpdObservationStress,
    OBSERVATION_STRESSES,
    _evaluate_stress,
    corrupt_current_unordered_set,
    decode_tempered_cpd_field,
    effective_support_tempering_gain,
    update_tempered_cpd_field,
)


def _points() -> np.ndarray:
    rng = np.random.default_rng(11)
    return rng.normal(size=(12, 3)) * np.array([0.05, 0.03, 0.02])


def test_tempered_cpd_field_is_recursive_posterior_mean() -> None:
    points = _points()
    first_shift = np.array([0.010, 0.0, 0.0])
    second_shift = np.array([0.030, 0.0, 0.0])
    first = fit_nonrigid_cpd(points, points + first_shift)
    second = fit_nonrigid_cpd(points, points + second_shift)

    state = update_tempered_cpd_field(None, first, gain=0.75, update_index=0)
    state = update_tempered_cpd_field(state, second, gain=0.75, update_index=1)
    decoded = decode_tempered_cpd_field(state, points)

    np.testing.assert_allclose(
        decoded,
        points + 0.25 * first_shift + 0.75 * second_shift,
        atol=1e-6,
    )
    np.testing.assert_allclose(state.weights, [0.25, 0.75])


def test_unit_gain_exactly_matches_latest_cpd_field() -> None:
    points = _points()
    first = fit_nonrigid_cpd(points, points + np.array([0.02, 0.0, 0.0]))
    second = fit_nonrigid_cpd(points, points + np.array([0.0, 0.01, 0.0]))

    state = update_tempered_cpd_field(None, first, gain=1.0, update_index=0)
    state = update_tempered_cpd_field(state, second, gain=1.0, update_index=1)

    assert len(state.transforms) == 1
    np.testing.assert_allclose(
        decode_tempered_cpd_field(state, points),
        second.transform(points),
        atol=1e-12,
    )


def test_corrupt_current_set_is_deterministic_and_unordered() -> None:
    ids = np.arange(16, dtype=np.int64)
    points = _points()
    points = np.vstack((points, np.ones((4, 3))))
    stress = next(
        value
        for value in OBSERVATION_STRESSES
        if value.name == "combined_2mm_25pct_30mm_50pct"
    )

    first = corrupt_current_unordered_set(
        ids,
        points,
        case_name="002-rope-silk-ep0002",
        frame=19,
        stress=stress,
    )
    second = corrupt_current_unordered_set(
        ids,
        points,
        case_name="002-rope-silk-ep0002",
        frame=19,
        stress=stress,
    )

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2] == second[2]
    assert first[2]["retained_count"] == 8
    assert first[2]["outlier_count"] == 2
    assert first[2]["target_order_permuted"] is True


def test_stress_config_rejects_partial_outlier_specification() -> None:
    with pytest.raises(ValueError, match="both be zero or positive"):
        CpdObservationStress(name="bad", outlier_fraction=0.25)
    with pytest.raises(ValueError, match="retained_fraction"):
        CpdObservationStress(name="bad", retained_fraction=0.0)


def test_tempered_update_rejects_noncausal_indices() -> None:
    points = _points()
    transform = fit_nonrigid_cpd(points, points + 0.01)
    state = update_tempered_cpd_field(None, transform, gain=0.5, update_index=1)
    with pytest.raises(ValueError, match="strictly increasing"):
        update_tempered_cpd_field(state, transform, gain=0.5, update_index=1)


def test_effective_support_gain_uses_frozen_clipped_ratio() -> None:
    assert effective_support_tempering_gain(16.0) == 1.0
    assert effective_support_tempering_gain(14.0) == pytest.approx(0.875)
    assert effective_support_tempering_gain(8.0) == 0.75
    assert effective_support_tempering_gain(3.0) == 0.75
    with pytest.raises(ValueError, match="must be positive"):
        effective_support_tempering_gain(0.0)


def test_insufficient_support_falls_back_to_selected_raw_backbone_exactly() -> None:
    rng = np.random.default_rng(29)
    base = rng.normal(size=(20, 3)) * np.array([0.05, 0.03, 0.02])
    persistence = np.repeat(base[None], 76, axis=0)
    prior = persistence.copy()
    prior[:, :, 0] += np.linspace(0.0, 0.03, len(prior))[:, None]
    target = persistence.copy()
    supported = np.ones(target.shape[:2], dtype=bool)
    centers = np.arange(16, dtype=np.int64)

    result, trajectories = _evaluate_stress(
        case_name="synthetic-open-unit-case",
        prior=prior,
        persistence=persistence,
        target=target,
        visibility=supported,
        validity=supported,
        centers=centers,
        scored_frames=(20, 39, 58),
        stress=CpdObservationStress(name="two_points", retained_fraction=0.10),
        config=NonrigidCpdConfig(),
    )

    for update in result["updates"]:
        assert update["fit_performed"] is False
        assert update["fallback"] == {
            "applied": True,
            "reason": "insufficient_support",
            "selected_raw_backbone": "persistence",
            "bit_exact_for_all_arms": True,
        }
        start = update["frame"] + 1
        stop = update["interval_end_exclusive"]
        for arm in ARMS:
            np.testing.assert_array_equal(
                trajectories[arm][start:stop], persistence[start:stop]
            )


def test_gain_one_copies_independent_cpd_output_bit_exactly(monkeypatch) -> None:
    rng = np.random.default_rng(31)
    base = rng.normal(size=(20, 3)) * np.array([0.05, 0.03, 0.02])
    persistence = np.repeat(base[None], 76, axis=0)
    prior = persistence.copy()
    prior[:, :, 0] += np.linspace(0.0, 0.01, len(prior))[:, None]
    target = persistence.copy()
    target[:, :, 1] += np.linspace(0.0, 0.02, len(target))[:, None]
    supported = np.ones(target.shape[:2], dtype=bool)
    centers = np.arange(16, dtype=np.int64)
    monkeypatch.setattr(
        diagnostic,
        "effective_support_tempering_gain",
        lambda _effective_count: 1.0,
    )

    result, trajectories = _evaluate_stress(
        case_name="synthetic-gain-one-unit-case",
        prior=prior,
        persistence=persistence,
        target=target,
        visibility=supported,
        validity=supported,
        centers=centers,
        scored_frames=(20, 39, 58),
        stress=CpdObservationStress(name="clean_gain_one"),
        config=NonrigidCpdConfig(),
    )

    reasons = [
        update["adaptive_effective_support_gain"]["exact_independent_reason"]
        for update in result["updates"]
    ]
    assert reasons == ["first_successful_update", "gain_one", "gain_one"]
    np.testing.assert_array_equal(
        trajectories[diagnostic.ADAPTIVE_ARM],
        trajectories[diagnostic.INDEPENDENT_ARM],
    )
