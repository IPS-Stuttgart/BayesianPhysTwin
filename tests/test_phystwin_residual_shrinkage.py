import numpy as np

from bayesian_phystwin.phystwin_residual_shrinkage import (
    HierarchicalResidualShrinkageProtocol,
    ScaleLikelihoodStatistics,
    frame_balanced_scale_statistics,
    scale_posterior,
    select_shared_hyperparameters,
    smooth_radial_shrinkage,
)


def test_smooth_radial_shrinkage_is_continuous_and_below_scale() -> None:
    values = np.array([[[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]])

    shrunk = smooth_radial_shrinkage(values, 2.0)

    norms = np.linalg.norm(shrunk, axis=2)
    assert np.all(norms < 2.0)
    assert norms[0, 1] > norms[0, 0]
    np.testing.assert_array_equal(smooth_radial_shrinkage(values, 0.0), 0.0)


def test_frame_balanced_statistics_match_direct_squared_error() -> None:
    target = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.01, 0.0, 0.0], [0.02, 0.0, 0.0]],
            [[0.02, 0.0, 0.0], [0.04, 0.0, 0.0]],
        ]
    )
    raw = np.array(
        [
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ]
    )
    valid = np.ones((3, 2), dtype=bool)
    scales = np.array([0.005, 0.012, 0.02])

    stats = frame_balanced_scale_statistics(
        target,
        valid,
        raw,
        scales,
        start_frame=1,
        end_frame=3,
    )
    for index, scale in enumerate(scales):
        corrected = smooth_radial_shrinkage(raw, float(scale))
        direct_error = sum(
            np.mean(
                np.sum(
                    np.square(target[frame] - corrected[frame - 1]),
                    axis=1,
                )
            )
            for frame in range(1, 3)
        )
        np.testing.assert_allclose(stats.squared_error_by_scale[index], direct_error)


def test_scale_posterior_shrinks_likelihood_toward_population() -> None:
    scales = np.linspace(0.0, 0.03, 61)
    statistics = ScaleLikelihoodStatistics(
        squared_error_by_scale=np.square(scales - 0.02),
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
        scales = np.linspace(0.0, 0.03, 31)
        return ScaleLikelihoodStatistics(
            squared_error_by_scale=20.0 * np.square(scales - optimum) + 1e-4,
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
