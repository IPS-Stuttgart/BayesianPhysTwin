import numpy as np
import pytest

from bayesian_phystwin.contracts.fixed_anchor import FixedBayesianAnchorConfigV1
from bayesian_phystwin.endpoint_model_average import (
    ModelAveragedEndpointConfigV1,
    infer_model_averaged_endpoint,
)
from bayesian_phystwin.structured_discrepancy import (
    STRUCTURED_DISCREPANCY_CLAIM_BOUNDARY,
    StructuredDiscrepancyConfigV1,
    infer_structured_discrepancy,
    predict_structured_discrepancy,
    structured_discrepancy_query_moments,
)


def _single_component(
    *,
    process_std_m: float = 0.001,
    observation_std_m: float = 0.0025,
    initial_std_m: float = 0.01,
) -> ModelAveragedEndpointConfigV1:
    return ModelAveragedEndpointConfigV1(
        components=(
            FixedBayesianAnchorConfigV1(
                process_std_m=process_std_m,
                observation_std_m=observation_std_m,
                initial_std_m=initial_std_m,
            ),
        )
    )


def test_identity_basis_matches_single_component_endpoint() -> None:
    residual = np.array(
        [
            [[0.010, 0.000, 0.000], [0.000, 0.010, 0.000]],
            [[0.012, 0.000, 0.000], [0.000, 0.008, 0.000]],
            [[0.011, 0.000, 0.000], [0.000, 0.009, 0.000]],
        ]
    )
    valid = np.array([[True, True], [True, False], [True, True]])
    endpoint_config = _single_component()
    structured = infer_structured_discrepancy(
        residual,
        valid,
        np.eye(2),
        end_frame=3,
        config=StructuredDiscrepancyConfigV1(
            endpoint_config=endpoint_config,
        ),
    )
    reference = infer_model_averaged_endpoint(
        residual,
        valid,
        end_frame=3,
        config=endpoint_config,
    )

    assert np.array_equal(structured.update_count, reference.update_count)
    assert np.allclose(structured.mean_m, reference.mean_m)
    assert np.allclose(
        structured.marginal_covariance_m2,
        reference.covariance_m2,
    )
    assert np.allclose(
        structured.final_nominal_probability,
        reference.final_nominal_probability,
    )
    assert np.allclose(structured.component_weights, 1.0)


def test_complete_rotated_basis_matches_independent_endpoint() -> None:
    rng = np.random.default_rng(41)
    basis, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    residual = rng.normal(scale=0.004, size=(5, 4, 3))
    valid = rng.random((5, 4)) > 0.2
    endpoint_config = _single_component()
    structured = infer_structured_discrepancy(
        residual,
        valid,
        basis,
        end_frame=5,
        config=StructuredDiscrepancyConfigV1(
            endpoint_config=endpoint_config,
        ),
    )
    reference = infer_model_averaged_endpoint(
        residual,
        valid,
        end_frame=5,
        config=endpoint_config,
    )
    identity_query = np.eye(12)
    moments = structured_discrepancy_query_moments(
        structured,
        identity_query,
    )

    assert np.allclose(structured.mean_m, reference.mean_m)
    assert np.allclose(
        structured.marginal_covariance_m2,
        reference.covariance_m2,
    )
    assert np.allclose(
        moments.covariance,
        np.diag(np.repeat(reference.component_variance_m2[0], 3)),
    )


def test_shared_translation_basis_produces_cross_track_covariance() -> None:
    track_count = 4
    basis = np.ones((track_count, 1)) / np.sqrt(track_count)
    residual = np.zeros((5, track_count, 3))
    residual[..., 0] = 0.01
    posterior = infer_structured_discrepancy(
        residual,
        np.ones((5, track_count), dtype=bool),
        basis,
        end_frame=5,
        config=StructuredDiscrepancyConfigV1(
            endpoint_config=_single_component(
                process_std_m=0.0,
                observation_std_m=0.001,
            )
        ),
    )

    query = np.zeros((2, track_count, 3))
    query[0, 0, 0] = 1.0
    query[1, 1, 0] = 1.0
    moments = structured_discrepancy_query_moments(posterior, query)

    assert np.allclose(posterior.mean_m[:, 0], posterior.mean_m[0, 0])
    assert posterior.mean_m[0, 0] > 0.0
    assert moments.covariance[0, 1] > 0.0
    assert np.allclose(moments.mean, posterior.mean_m[:2, 0])


def test_structured_filter_is_vertex_permutation_equivariant() -> None:
    rng = np.random.default_rng(17)
    track_count = 7
    rank = 3
    basis, _ = np.linalg.qr(rng.normal(size=(track_count, rank)))
    residual = rng.normal(scale=0.004, size=(6, track_count, 3))
    valid = rng.random((6, track_count)) > 0.15
    reliability = rng.uniform(0.2, 1.0, size=(6, track_count))
    config = StructuredDiscrepancyConfigV1(
        endpoint_config=_single_component(),
    )
    reference = infer_structured_discrepancy(
        residual,
        valid,
        basis,
        prior_reliability=reliability,
        end_frame=6,
        config=config,
    )
    permutation = rng.permutation(track_count)
    permuted = infer_structured_discrepancy(
        residual[:, permutation],
        valid[:, permutation],
        basis[permutation],
        prior_reliability=reliability[:, permutation],
        end_frame=6,
        config=config,
    )

    assert np.allclose(
        permuted.component_coefficient_mean_m,
        reference.component_coefficient_mean_m,
    )
    assert np.allclose(
        permuted.component_coefficient_covariance_m2,
        reference.component_coefficient_covariance_m2,
    )
    assert np.allclose(permuted.mean_m, reference.mean_m[permutation])
    assert np.allclose(
        permuted.marginal_covariance_m2,
        reference.marginal_covariance_m2[permutation],
    )
    assert np.array_equal(permuted.update_count, reference.update_count[permutation])


def test_query_operator_matches_marginal_blocks_and_dense_psd() -> None:
    rng = np.random.default_rng(9)
    track_count = 5
    rank = 2
    basis, _ = np.linalg.qr(rng.normal(size=(track_count, rank)))
    endpoint_config = ModelAveragedEndpointConfigV1(
        components=(
            FixedBayesianAnchorConfigV1(
                process_std_m=0.0,
                observation_std_m=0.002,
            ),
            FixedBayesianAnchorConfigV1(
                process_std_m=0.003,
                observation_std_m=0.004,
            ),
        ),
        component_prior_probability=(0.4, 0.6),
    )
    posterior = infer_structured_discrepancy(
        rng.normal(scale=0.006, size=(4, track_count, 3)),
        np.ones((4, track_count), dtype=bool),
        basis,
        end_frame=4,
        config=StructuredDiscrepancyConfigV1(
            endpoint_config=endpoint_config,
        ),
    )
    identity_query = np.eye(3 * track_count)
    moments = structured_discrepancy_query_moments(
        posterior,
        identity_query,
    )

    assert moments.covariance.shape == (3 * track_count, 3 * track_count)
    assert np.min(np.linalg.eigvalsh(moments.covariance)) >= -1e-12
    expected_dense = np.zeros_like(moments.covariance)
    mean_flat = posterior.mean_m.reshape(-1)
    for component_index, weight in enumerate(posterior.component_weights):
        space_covariance = (
            basis
            @ posterior.component_coefficient_covariance_m2[component_index]
            @ basis.T
            + np.diag(
                posterior.component_local_variance_m2[component_index]
            )
        )
        component_flat = posterior.component_mean_m[component_index].reshape(-1)
        centered = component_flat - mean_flat
        expected_dense += weight * (
            np.kron(space_covariance, np.eye(3))
            + np.outer(centered, centered)
        )
    assert np.allclose(moments.covariance, expected_dense)
    for index in range(track_count):
        start = 3 * index
        assert np.allclose(
            moments.covariance[start : start + 3, start : start + 3],
            posterior.marginal_covariance_m2[index],
        )


def test_horizon_prediction_increases_unresolved_and_shared_uncertainty() -> None:
    basis = np.ones((3, 1)) / np.sqrt(3.0)
    posterior = infer_structured_discrepancy(
        np.zeros((2, 3, 3)),
        np.ones((2, 3), dtype=bool),
        basis,
        end_frame=2,
        config=StructuredDiscrepancyConfigV1(
            endpoint_config=_single_component(process_std_m=0.002),
        ),
    )
    now = predict_structured_discrepancy(posterior, horizon_steps=0)
    future = predict_structured_discrepancy(posterior, horizon_steps=12)
    query = np.eye(9)
    now_moments = structured_discrepancy_query_moments(now, query)
    future_moments = structured_discrepancy_query_moments(future, query)

    assert np.allclose(now.mean_m, future.mean_m)
    assert np.trace(future_moments.covariance) > np.trace(now_moments.covariance)
    assert np.all(
        future.component_local_variance_m2
        >= now.component_local_variance_m2
    )


def test_large_outlier_receives_low_nominal_probability() -> None:
    track_count = 6
    basis = np.ones((track_count, 1)) / np.sqrt(track_count)
    residual = np.full((4, track_count, 3), 0.0)
    residual[..., 0] = 0.01
    residual[-1, -1, 0] = 0.5
    posterior = infer_structured_discrepancy(
        residual,
        np.ones((4, track_count), dtype=bool),
        basis,
        end_frame=4,
        config=StructuredDiscrepancyConfigV1(
            endpoint_config=_single_component(
                process_std_m=0.0,
                observation_std_m=0.002,
            )
        ),
    )

    assert posterior.final_nominal_probability[-1] < 0.1
    assert np.median(posterior.mean_m[:, 0]) < 0.05
    assert np.median(posterior.mean_m[:, 0]) > 0.0


def test_no_observations_preserve_global_component_priors() -> None:
    endpoint_config = ModelAveragedEndpointConfigV1(
        components=(
            FixedBayesianAnchorConfigV1(process_std_m=0.0),
            FixedBayesianAnchorConfigV1(process_std_m=0.002),
        ),
        component_prior_probability=(0.25, 0.75),
    )
    basis = np.ones((4, 1)) / 2.0
    posterior = infer_structured_discrepancy(
        np.zeros((3, 4, 3)),
        np.zeros((3, 4), dtype=bool),
        basis,
        end_frame=3,
        config=StructuredDiscrepancyConfigV1(
            endpoint_config=endpoint_config,
        ),
    )

    assert np.allclose(posterior.component_weights, [0.25, 0.75])
    assert np.allclose(posterior.mean_m, 0.0)
    assert not np.any(posterior.updated_mask)
    assert np.all(np.diagonal(posterior.marginal_covariance_m2, axis1=1, axis2=2) > 0)


def test_contracts_are_read_only_and_claim_boundary_is_explicit() -> None:
    posterior = infer_structured_discrepancy(
        np.zeros((1, 2, 3)),
        np.ones((1, 2), dtype=bool),
        np.eye(2),
        end_frame=1,
        config=StructuredDiscrepancyConfigV1(
            endpoint_config=_single_component(),
        ),
    )
    assert "not an identified simulator-state" in STRUCTURED_DISCREPANCY_CLAIM_BOUNDARY
    assert not posterior.mean_m.flags.writeable
    assert not posterior.spatial_basis.flags.writeable
    assert not posterior.component_weights.flags.writeable
    with pytest.raises(ValueError):
        posterior.mean_m[0, 0] = 1.0


@pytest.mark.parametrize(
    ("basis", "message"),
    [
        (np.ones((3, 0)), "rank"),
        (np.ones((2, 1)), "track_count"),
        (np.ones((3, 2)), "orthonormal"),
        (np.full((3, 1), np.nan), "finite"),
    ],
)
def test_invalid_basis_fails_closed(basis: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        infer_structured_discrepancy(
            np.zeros((2, 3, 3)),
            np.ones((2, 3), dtype=bool),
            basis,
            end_frame=2,
            config=StructuredDiscrepancyConfigV1(
                endpoint_config=_single_component(),
            ),
        )


def test_invalid_reliability_and_query_fail_closed() -> None:
    basis = np.ones((3, 1)) / np.sqrt(3.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        infer_structured_discrepancy(
            np.zeros((2, 3, 3)),
            np.ones((2, 3), dtype=bool),
            basis,
            prior_reliability=np.full((2, 3), 1.1),
            end_frame=2,
            config=StructuredDiscrepancyConfigV1(
                endpoint_config=_single_component(),
            ),
        )
    posterior = infer_structured_discrepancy(
        np.zeros((2, 3, 3)),
        np.ones((2, 3), dtype=bool),
        basis,
        end_frame=2,
        config=StructuredDiscrepancyConfigV1(
            endpoint_config=_single_component(),
        ),
    )
    with pytest.raises(ValueError, match="query_jacobian"):
        structured_discrepancy_query_moments(posterior, np.zeros((2, 2)))
