import numpy as np

from bayesian_phystwin.phystwin_residual_shrinkage import (
    HierarchicalResidualShrinkageProtocol,
    ScaleLikelihoodStatistics,
    frame_balanced_scale_statistics,
    normalize_residual_shape,
    scale_posterior,
    select_shared_hyperparameters,
)


def test_normalize_residual_shape_sets_rms_without_pointwise_clipping() -> None:
    values = np.array([[[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]])

    shape, raw_rms = normalize_residual_shape(values)

    assert raw_rms == np.sqrt(12.5)
    np.testing.assert_allclose(
        np.sqrt(np.mean(np.sum(np.square(shape), axis=2))), 1.0
    )
    assert np.max(np.linalg.norm(shape, axis=2)) > 1.0


def test_frame_balanced_statistics_match_direct_squared_error() -> None:
    target = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.01, 0.0, 0.0], [0.02, 0.0, 0.0]],
            [[0.02, 0.0, 0.0], [0.04, 0.0, 0.0]],
        ]
    )
    shape = np.array(
        [
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ]
    )
    valid = np.ones((3, 2), dtype=bool)
    scale = 0.012

    stats = frame_balanced_scale_statistics(
        target,
        valid,
        shape,
        start_frame=1,
        end_frame=3,
    )
    quadratic_error = (
        stats.quadratic * scale**2
        - 2.0 * stats.linear * scale
        + stats.constant
    )
    direct_error = sum(
        np.mean(np.sum(np.square(target[frame] - scale * shape[frame - 1]), axis=1))
        for frame in range(1, 3)
    )

    np.testing.assert_allclose(quadratic_error, direct_error)


def test_scale_posterior_shrinks_likelihood_toward_population() -> None:
    scales = np.linspace(0.0, 0.03, 61)
    statistics = ScaleLikelihoodStatistics(
        quadratic=10.0,
        linear=0.2,
        constant=0.01,
        frame_count=10,
        raw_rms_m=1.0,
    )

    weights, _, _ = scale_posterior(
        statistics,
        scales,
        observation_std_m=0.005,
        population_mean_m=0.005,
        population_std_m=0.001,
    )

    posterior_mean = float(np.sum(scales * weights))
    assert 0.005 < posterior_mean < 0.02


def test_shared_selection_does_not_use_held_out_statistics() -> None:
    protocol = HierarchicalResidualShrinkageProtocol(
        rank_candidates=(1,),
        persistence_candidates=(0.0,),
        ridge_candidates=(1.0,),
        observation_std_candidates_m=(0.005,),
        population_mean_candidates_m=(0.005, 0.02),
        population_std_candidates_m=(0.0025,),
        scale_grid_maximum_m=0.03,
        scale_grid_step_m=0.001,
    )
    key = (1, 0.0, 1.0)

    def stats(optimum: float) -> ScaleLikelihoodStatistics:
        return ScaleLikelihoodStatistics(
            quadratic=20.0,
            linear=20.0 * optimum,
            constant=20.0 * optimum**2 + 1e-4,
            frame_count=10,
            raw_rms_m=1.0,
        )

    common = {"train_a": {key: stats(0.005)}, "train_b": {key: stats(0.006)}}
    first, _ = select_shared_hyperparameters(
        {**common, "held_out": {key: stats(0.025)}},
        "held_out",
        protocol,
    )
    second, _ = select_shared_hyperparameters(
        {**common, "held_out": {key: stats(0.001)}},
        "held_out",
        protocol,
    )

    assert first == second
    assert first.population_mean_m == 0.005
