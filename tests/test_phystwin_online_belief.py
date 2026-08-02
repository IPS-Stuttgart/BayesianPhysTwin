import numpy as np
import pytest

from bayesian_phystwin.phystwin_online_belief import (
    RecursiveRbfBeliefConfig,
    decode_recursive_rbf_belief,
    deterministic_farthest_point_ids,
    finite_sample_absolute_residual_quantile_m,
    initialize_recursive_rbf_belief,
    robust_huber_continuation_gain,
    update_recursive_rbf_belief,
)


def _line_problem() -> tuple[np.ndarray, np.ndarray, RecursiveRbfBeliefConfig]:
    points = np.stack((np.linspace(0.0, 1.0, 9), np.zeros(9), np.zeros(9)), axis=1)
    centre_ids = np.asarray([0, 4, 8])
    config = RecursiveRbfBeliefConfig(
        length_scale_fraction=0.25,
        local_blend=0.25,
    )
    return points, centre_ids, config


def test_farthest_point_selection_is_order_independent_and_tie_stable() -> None:
    points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.1, 0.0]]
    )
    first = deterministic_farthest_point_ids(points, np.asarray([3, 2, 1, 0]), 3)
    second = deterministic_farthest_point_ids(points, np.asarray([0, 1, 2, 3]), 3)
    np.testing.assert_array_equal(first, np.asarray([0, 1, 2]))
    np.testing.assert_array_equal(second, first)


def test_farthest_point_selection_keeps_ids_unique_for_coincident_points() -> None:
    points = np.zeros((5, 3), dtype=float)

    selected = deterministic_farthest_point_ids(
        points,
        np.asarray([4, 2, 0, 3, 1]),
        5,
    )

    np.testing.assert_array_equal(selected, np.arange(5))


def test_finite_sample_absolute_residual_quantile_uses_corrected_rank() -> None:
    residual_m = np.asarray(
        [[0.001, -0.002, 0.003], [0.004, -0.005, 0.006], [np.nan] * 3]
    )
    available = np.asarray([True, True, True])

    assert (
        finite_sample_absolute_residual_quantile_m(residual_m, available, 0.50) == 0.004
    )
    assert (
        finite_sample_absolute_residual_quantile_m(residual_m, available, 0.90) == 0.006
    )
    with pytest.raises(ValueError, match="no finite available"):
        finite_sample_absolute_residual_quantile_m(
            residual_m,
            np.zeros(3, dtype=bool),
            0.90,
        )


def test_robust_huber_continuation_gain_recovers_causal_motion_scale() -> None:
    physical = np.asarray(
        [
            [0.01, 0.00, 0.00],
            [0.00, 0.02, 0.00],
            [0.00, 0.00, 0.03],
            [0.02, 0.01, 0.00],
            [-0.01, 0.02, 0.01],
        ]
    )
    observed = 0.4 * physical
    observed[-1] += 0.5

    gain = robust_huber_continuation_gain(physical, observed)

    assert 0.35 < gain < 0.45
    assert robust_huber_continuation_gain(physical, physical) == pytest.approx(1.0)
    assert robust_huber_continuation_gain(physical, np.zeros_like(physical)) == 0.0


def test_continuation_gain_freezes_when_overlap_is_insufficient() -> None:
    physical = np.asarray([[0.01, 0.0, 0.0], [0.02, 0.0, 0.0]])

    assert robust_huber_continuation_gain(physical, physical) == 0.0


def test_sparse_translation_updates_arbitrary_queries() -> None:
    points, centre_ids, config = _line_problem()
    belief = initialize_recursive_rbf_belief(
        centre_ids,
        points[centre_ids],
        points,
        config=config,
    )
    translation = np.asarray([0.01, -0.02, 0.03])
    residual = np.repeat(translation[None], len(centre_ids), axis=0)
    posterior, reliability = update_recursive_rbf_belief(
        belief,
        5,
        points[centre_ids],
        residual,
        np.ones(len(centre_ids), dtype=bool),
        config=config,
    )
    prediction = decode_recursive_rbf_belief(
        posterior,
        points,
        forecast_frames=2,
        config=config,
    )
    assert np.all(reliability > 0.9)
    np.testing.assert_allclose(
        prediction.mean_m,
        np.repeat(translation[None], len(points), axis=0),
        atol=1e-3,
    )
    assert np.all(prediction.variance_m2 > 0.0)


def test_missing_centres_survive_occlusion_and_variance_grows() -> None:
    points, centre_ids, config = _line_problem()
    prior = initialize_recursive_rbf_belief(
        centre_ids,
        points[centre_ids],
        points,
        config=config,
    )
    residual = np.asarray([[0.01, 0.0, 0.0], [0.02, 0.0, 0.0], [0.03, 0.0, 0.0]])
    first, _ = update_recursive_rbf_belief(
        prior,
        2,
        points[centre_ids],
        residual,
        np.ones(3, dtype=bool),
        config=config,
    )
    second, reliability = update_recursive_rbf_belief(
        first,
        5,
        points[centre_ids],
        np.full((3, 3), np.nan),
        np.zeros(3, dtype=bool),
        config=config,
    )
    np.testing.assert_array_equal(second.update_count, first.update_count)
    np.testing.assert_array_equal(second.local_mean_m, first.local_mean_m)
    assert np.all(second.local_variance_m2 > first.local_variance_m2)
    np.testing.assert_array_equal(reliability, np.zeros(3))


def test_isolated_large_residual_is_downweighted() -> None:
    points, centre_ids, config = _line_problem()
    prior = initialize_recursive_rbf_belief(
        centre_ids,
        points[centre_ids],
        points,
        config=config,
    )
    residual = np.asarray([[0.01, 0.0, 0.0], [0.01, 0.0, 0.0], [0.20, 0.0, 0.0]])
    posterior, reliability = update_recursive_rbf_belief(
        prior,
        1,
        points[centre_ids],
        residual,
        np.ones(3, dtype=bool),
        config=config,
    )
    assert reliability[2] < reliability[0]
    prediction = decode_recursive_rbf_belief(
        posterior,
        points,
        forecast_frames=0,
        config=config,
    )
    assert np.max(prediction.mean_m[:, 0]) < 0.04


def test_explicit_default_observation_model_is_behavior_compatible() -> None:
    points, centre_ids, config = _line_problem()
    prior = initialize_recursive_rbf_belief(
        centre_ids,
        points[centre_ids],
        points,
        config=config,
    )
    residual = np.asarray([[0.01, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0]])
    available = np.ones(3, dtype=bool)

    legacy, legacy_reliability = update_recursive_rbf_belief(
        prior,
        1,
        points[centre_ids],
        residual,
        available,
        config=config,
    )
    explicit, explicit_reliability = update_recursive_rbf_belief(
        prior,
        1,
        points[centre_ids],
        residual,
        available,
        config=config,
        prior_reliability=np.ones(3),
        observation_variance_m2=np.full(3, config.observation_std_m**2),
    )

    for name in (
        "center_positions_m",
        "global_mean_m",
        "global_variance_m2",
        "local_mean_m",
        "local_variance_m2",
        "update_count",
    ):
        np.testing.assert_array_equal(getattr(legacy, name), getattr(explicit, name))
    np.testing.assert_array_equal(legacy_reliability, explicit_reliability)


def test_metric_variance_and_prior_reliability_conservatively_scale_update() -> None:
    points, centre_ids, config = _line_problem()
    prior = initialize_recursive_rbf_belief(
        centre_ids,
        points[centre_ids],
        points,
        config=config,
    )
    residual = np.repeat(np.asarray([[0.01, 0.0, 0.0]]), 3, axis=0)
    available = np.ones(3, dtype=bool)
    confident, _ = update_recursive_rbf_belief(
        prior,
        1,
        points[centre_ids],
        residual,
        available,
        config=config,
        prior_reliability=np.ones(3),
        observation_variance_m2=np.full(3, 0.002**2),
    )
    conservative, reliability = update_recursive_rbf_belief(
        prior,
        1,
        points[centre_ids],
        residual,
        available,
        config=config,
        prior_reliability=np.full(3, 0.25),
        observation_variance_m2=np.full(3, 0.010**2),
    )

    assert conservative.global_mean_m[0] < confident.global_mean_m[0]
    assert np.all(conservative.local_variance_m2 > confident.local_variance_m2)
    assert np.all(reliability <= 0.25)


def test_update_rejects_noncausal_frame_order() -> None:
    points, centre_ids, config = _line_problem()
    prior = initialize_recursive_rbf_belief(
        centre_ids,
        points[centre_ids],
        points,
        config=config,
    )
    posterior, _ = update_recursive_rbf_belief(
        prior,
        3,
        points[centre_ids],
        np.zeros((3, 3)),
        np.ones(3, dtype=bool),
        config=config,
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        update_recursive_rbf_belief(
            posterior,
            3,
            points[centre_ids],
            np.zeros((3, 3)),
            np.ones(3, dtype=bool),
            config=config,
        )
