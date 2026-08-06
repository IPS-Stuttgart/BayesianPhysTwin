import numpy as np

from bayesian_phystwin.phystwin_graph_discrepancy import (
    normalized_spring_laplacian,
)
from bayesian_phystwin.tapnextpp_sparse_assimilation import (
    MetricEndpointPosterior,
    SparseAssimilationConfig,
    associate_fixed_material_displacements,
    associate_sparse_observations,
    build_material_transport_graph_update,
    build_sparse_graph_update,
    robust_metric_random_walk_endpoint,
)


def _association_inputs():
    baseline = np.zeros((4, 5, 3), dtype=float)
    baseline[:, :, 0] = np.arange(5)[None] * 0.01
    observations = np.zeros((4, 2, 3), dtype=float)
    observations[:, 0, 0] = np.array([0.001, 0.002, 0.003, 0.004])
    observations[:, 1, 0] = np.array([0.019, 0.018, 0.017, 0.016])
    support = np.ones((4, 2), dtype=bool)
    reliability = np.full((4, 2), 0.8)
    covariance = np.broadcast_to(
        np.eye(3) * 4e-6,
        (4, 2, 3, 3),
    ).copy()
    return observations, support, reliability, covariance, baseline


def test_state_residual_does_not_change_association_or_prior_reliability() -> None:
    observations, support, reliability, covariance, baseline = _association_inputs()
    first = associate_sparse_observations(
        observations,
        support,
        reliability,
        covariance,
        baseline,
    )
    shifted = observations.copy()
    shifted[1:] += np.array([0.1, -0.1, 0.05])
    second = associate_sparse_observations(
        shifted,
        support,
        reliability,
        covariance,
        baseline,
    )

    np.testing.assert_array_equal(
        first.candidate_indices,
        second.candidate_indices,
    )
    np.testing.assert_allclose(
        first.candidate_probabilities,
        second.candidate_probabilities,
    )
    np.testing.assert_array_equal(first.prior_reliability, reliability)
    np.testing.assert_array_equal(second.prior_reliability, reliability)
    assert not np.array_equal(first.innovation_m, second.innovation_m)


def test_ambiguous_assignment_adds_metric_covariance() -> None:
    observations, support, reliability, covariance, baseline = _association_inputs()
    observations[:, 0, 0] = 0.005
    association = associate_sparse_observations(
        observations,
        support,
        reliability,
        covariance,
        baseline,
    )

    assert association.entropy[0] > 0.0
    assert np.max(
        association.covariance_m2[:, 0] - covariance[:, 0]
    ) > 0.0


def test_temporal_effective_sample_cap_bounds_duplicate_confidence() -> None:
    config = SparseAssimilationConfig(
        process_std_m=1e-9,
        maximum_effective_rows_per_identity=4.0,
    )
    covariance = np.eye(3) * 4e-6

    def posterior(repeats: int):
        return robust_metric_random_walk_endpoint(
            np.zeros((repeats, 1, 3), dtype=float),
            np.ones((repeats, 1), dtype=bool),
            np.ones((repeats, 1), dtype=float),
            np.broadcast_to(covariance, (repeats, 1, 3, 3)),
            config=config,
        )

    four = posterior(4)
    forty = posterior(40)
    assert np.all(
        np.diag(forty.covariance_m2[0])
        >= 0.95 * np.diag(four.covariance_m2[0])
    )
    assert four.effective_row_count[0] == 4.0
    assert forty.effective_row_count[0] == 4.0
    assert forty.temporal_covariance_inflation[0] == 10.0


def test_robust_endpoint_rejects_gross_innovation_once() -> None:
    innovation = np.zeros((5, 1, 3), dtype=float)
    innovation[:4, 0, 0] = 0.002
    innovation[4, 0, 0] = 0.100
    posterior = robust_metric_random_walk_endpoint(
        innovation,
        np.ones((5, 1), dtype=bool),
        np.ones((5, 1), dtype=float),
        np.broadcast_to(np.eye(3) * 4e-6, (5, 1, 3, 3)),
        config=SparseAssimilationConfig(process_std_m=1e-6),
    )

    assert posterior.final_inlier_probability[0] < 0.1
    assert abs(posterior.mean_m[0, 0]) < 0.02
    assert posterior.update_count[0] == 5


def test_unknown_correlation_does_not_accumulate_duplicate_node_precision() -> None:
    observations, support, reliability, covariance, baseline = _association_inputs()
    observations[:, 1] = observations[:, 0]
    association = associate_sparse_observations(
        observations,
        support,
        reliability,
        covariance,
        baseline,
    )
    endpoint = MetricEndpointPosterior(
        mean_m=np.array([[0.010, 0.0, 0.0], [0.020, 0.0, 0.0]]),
        covariance_m2=np.array([np.eye(3) * 1e-4, np.eye(3) * 4e-4]),
        final_inlier_probability=np.ones(2),
        update_count=np.ones(2, dtype=np.int64),
        effective_row_count=np.ones(2),
        temporal_covariance_inflation=np.ones(2),
    )
    laplacian = normalized_spring_laplacian(
        5,
        np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64),
    )
    result = build_sparse_graph_update(
        endpoint,
        association,
        np.zeros((5, 3), dtype=float),
        laplacian,
        config=SparseAssimilationConfig(graph_covariance_probes=0),
    )

    assert result.accepted
    assert len(result.observed_nodes) == 1
    assert result.observed_variance_m2[0] == 1e-4


def test_rejected_endpoint_returns_exact_zero_fallback() -> None:
    observations, support, reliability, covariance, baseline = _association_inputs()
    association = associate_sparse_observations(
        observations,
        support,
        reliability,
        covariance,
        baseline,
    )
    endpoint = MetricEndpointPosterior(
        mean_m=np.ones((2, 3)),
        covariance_m2=np.broadcast_to(np.eye(3), (2, 3, 3)),
        final_inlier_probability=np.zeros(2),
        update_count=np.ones(2, dtype=np.int64),
        effective_row_count=np.ones(2),
        temporal_covariance_inflation=np.ones(2),
    )
    laplacian = normalized_spring_laplacian(
        5,
        np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64),
    )
    result = build_sparse_graph_update(
        endpoint,
        association,
        np.zeros((5, 3), dtype=float),
        laplacian,
    )

    assert not result.accepted
    np.testing.assert_array_equal(result.direct_delta_m, np.zeros((5, 3)))
    np.testing.assert_array_equal(result.graph_delta_m, np.zeros((5, 3)))


def test_fixed_material_transport_cancels_static_attachment_offset() -> None:
    baseline = np.zeros((4, 5, 3), dtype=float)
    baseline[:, :, 0] = np.arange(5)[None] * 0.01
    baseline[:, 2, 0] += np.arange(4) * 0.001
    observations = baseline[:, [2]].copy()
    observations[:, 0, 0] += 0.004
    observations[:, 0, 1] += np.arange(4) * 0.002
    support = np.ones((4, 1), dtype=bool)
    reliability = np.full((4, 1), 0.8)
    covariance = np.broadcast_to(np.eye(3) * 4e-6, (4, 1, 3, 3)).copy()

    association = associate_fixed_material_displacements(
        observations,
        support,
        reliability,
        covariance,
        baseline,
        np.array([2]),
    )

    np.testing.assert_array_equal(association.map_indices, np.array([2]))
    np.testing.assert_allclose(association.innovation_m[:, 0, 0], 0.0)
    np.testing.assert_allclose(
        association.innovation_m[:, 0, 1],
        np.array([0.0, 0.002, 0.004, 0.006]),
    )
    assert not association.support[0, 0]
    np.testing.assert_array_equal(association.prior_reliability[0], 0.0)
    np.testing.assert_allclose(association.prior_reliability[1:], 0.8)
    assert np.min(np.linalg.eigvalsh(association.covariance_m2[1, 0])) >= 24e-6


def test_material_transport_reliability_is_residual_independent() -> None:
    observations, support, reliability, covariance, baseline = _association_inputs()
    first = associate_fixed_material_displacements(
        observations,
        support,
        reliability,
        covariance,
        baseline,
        np.array([0, 2]),
    )
    shifted = observations.copy()
    shifted[1:] += np.array([0.02, -0.01, 0.03])
    second = associate_fixed_material_displacements(
        shifted,
        support,
        reliability,
        covariance,
        baseline,
        np.array([0, 2]),
    )

    np.testing.assert_array_equal(first.map_indices, second.map_indices)
    np.testing.assert_array_equal(
        first.prior_reliability,
        second.prior_reliability,
    )
    assert not np.array_equal(first.innovation_m, second.innovation_m)


def test_material_transport_graph_update_does_not_subtract_dense_field() -> None:
    observations, support, reliability, covariance, baseline = _association_inputs()
    association = associate_fixed_material_displacements(
        observations,
        support,
        reliability,
        covariance,
        baseline,
        np.array([0, 2]),
    )
    endpoint = MetricEndpointPosterior(
        mean_m=np.array([[0.003, 0.0, 0.0], [-0.002, 0.0, 0.0]]),
        covariance_m2=np.broadcast_to(np.eye(3) * 1e-4, (2, 3, 3)),
        final_inlier_probability=np.ones(2),
        update_count=np.ones(2, dtype=np.int64),
        effective_row_count=np.ones(2),
        temporal_covariance_inflation=np.ones(2),
    )
    laplacian = normalized_spring_laplacian(
        5,
        np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64),
    )

    result = build_material_transport_graph_update(
        endpoint,
        association,
        laplacian,
        config=SparseAssimilationConfig(graph_covariance_probes=0),
    )

    np.testing.assert_allclose(result.direct_delta_m[0], endpoint.mean_m[0])
    np.testing.assert_allclose(result.direct_delta_m[2], endpoint.mean_m[1])
