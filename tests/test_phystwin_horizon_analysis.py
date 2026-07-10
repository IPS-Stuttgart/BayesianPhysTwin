import numpy as np

from bayesian_phystwin.phystwin_horizon_analysis import (
    bootstrap_case_frame_mean,
    centered_spatial_correlation,
    split_future_horizon,
)


def test_split_future_horizon_is_contiguous_and_balanced() -> None:
    result = split_future_horizon(8)

    assert tuple(result) == ("early", "middle", "late")
    assert [len(values) for values in result.values()] == [3, 3, 2]
    np.testing.assert_array_equal(
        np.concatenate(tuple(result.values())),
        np.arange(8),
    )


def test_centered_spatial_correlation_removes_translation() -> None:
    endpoint = np.array(
        [
            [-1.0, 0.0, 0.5],
            [0.0, 1.0, -0.5],
            [1.0, -1.0, 0.0],
        ]
    )
    future = 2.0 * endpoint + np.array([4.0, -3.0, 2.0])

    correlation = centered_spatial_correlation(
        endpoint,
        future,
        np.ones(3, dtype=bool),
    )

    np.testing.assert_allclose(correlation, 1.0, atol=1e-12)


def test_bootstrap_case_frame_mean_weights_cases_and_clusters() -> None:
    result = bootstrap_case_frame_mean(
        {
            "a_first": np.full(5, 0.8),
            "a_second": np.full(5, 0.8),
            "b": np.full(5, 0.2),
        },
        samples=20,
        block_length=2,
        seed=3,
        clusters={"a_first": "a", "a_second": "a", "b": "b"},
    )

    np.testing.assert_allclose(
        result["macro"]["observed_equal_case_mean"],
        0.6,
    )
    np.testing.assert_allclose(
        result["cluster_macro"]["observed_equal_cluster_mean"],
        0.5,
    )
