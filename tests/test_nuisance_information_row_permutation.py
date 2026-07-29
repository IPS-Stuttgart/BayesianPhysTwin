from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.nuisance_aware_information import (
    NuisanceAwareInformationState,
    greedy_nuisance_aware_selection,
)


def _permuted_block(
    state_jacobian: np.ndarray,
    nuisance_jacobian: np.ndarray,
    covariance: np.ndarray,
    reliability: np.ndarray,
    permutation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        state_jacobian[permutation],
        nuisance_jacobian[permutation],
        covariance[np.ix_(permutation, permutation)],
        reliability[permutation],
    )


def test_correlated_row_reliability_is_permutation_invariant() -> None:
    prior = NuisanceAwareInformationState.from_independent_priors(
        np.asarray([[2.0, 0.2], [0.2, 1.5]]),
        np.asarray([[1.2]]),
    )
    state_jacobian = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, -0.5]],
    )
    nuisance_jacobian = np.asarray([[1.0], [0.2], [-0.4]])
    covariance = np.asarray(
        [[1.0, 0.8, 0.2], [0.8, 1.5, 0.5], [0.2, 0.5, 0.9]],
    )
    reliability = np.asarray([0.15, 0.65, 0.95])
    permutation = np.asarray([2, 0, 1])

    original = prior.observation_information_gain(
        state_jacobian,
        nuisance_jacobian,
        covariance,
        reliability=reliability,
    )
    permuted_block = _permuted_block(
        state_jacobian,
        nuisance_jacobian,
        covariance,
        reliability,
        permutation,
    )
    permuted = prior.observation_information_gain(
        permuted_block[0],
        permuted_block[1],
        permuted_block[2],
        reliability=permuted_block[3],
    )

    assert original.mutual_information_nats == pytest.approx(
        permuted.mutual_information_nats,
        rel=1e-12,
        abs=1e-12,
    )
    np.testing.assert_allclose(
        original.updated_state.state_precision,
        permuted.updated_state.state_precision,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        original.updated_state.nuisance_precision,
        permuted.updated_state.nuisance_precision,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        original.updated_state.state_nuisance_precision,
        permuted.updated_state.state_nuisance_precision,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("seed", range(16))
def test_random_correlated_information_blocks_are_row_permutation_invariant(
    seed: int,
) -> None:
    rng = np.random.default_rng(seed + 8_000)
    state_dimension = int(rng.integers(1, 5))
    nuisance_dimension = int(rng.integers(1, 4))
    row_count = int(rng.integers(2, 7))

    state_prior_root = rng.normal(size=(state_dimension, state_dimension))
    nuisance_prior_root = rng.normal(size=(nuisance_dimension, nuisance_dimension))
    prior = NuisanceAwareInformationState.from_independent_priors(
        state_prior_root.T @ state_prior_root + np.eye(state_dimension),
        nuisance_prior_root.T @ nuisance_prior_root + np.eye(nuisance_dimension),
    )
    state_jacobian = rng.normal(size=(row_count, state_dimension))
    nuisance_jacobian = rng.normal(size=(row_count, nuisance_dimension))
    covariance_root = rng.normal(size=(row_count, row_count))
    covariance = covariance_root @ covariance_root.T + 0.25 * np.eye(row_count)
    reliability = rng.uniform(0.05, 1.0, size=row_count)
    permutation = rng.permutation(row_count)
    permuted = _permuted_block(
        state_jacobian,
        nuisance_jacobian,
        covariance,
        reliability,
        permutation,
    )

    original_update = prior.observation_information_gain(
        state_jacobian,
        nuisance_jacobian,
        covariance,
        reliability=reliability,
    )
    permuted_update = prior.observation_information_gain(
        permuted[0],
        permuted[1],
        permuted[2],
        reliability=permuted[3],
    )

    assert original_update.mutual_information_nats == pytest.approx(
        permuted_update.mutual_information_nats,
        rel=1e-10,
        abs=1e-12,
    )
    np.testing.assert_allclose(
        original_update.updated_state.state_precision,
        permuted_update.updated_state.state_precision,
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        original_update.updated_state.nuisance_precision,
        permuted_update.updated_state.nuisance_precision,
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        original_update.updated_state.state_nuisance_precision,
        permuted_update.updated_state.state_nuisance_precision,
        rtol=1e-10,
        atol=1e-12,
    )


def test_greedy_selection_is_invariant_to_rows_within_candidate_blocks() -> None:
    prior = NuisanceAwareInformationState.from_independent_priors(
        np.eye(2),
        np.asarray([[0.5]]),
    )
    state_jacobians = (
        np.asarray([[1.0, 0.0], [0.5, 1.0], [0.2, -0.3]]),
        np.asarray([[0.0, 1.0], [1.0, 0.5], [-0.4, 0.2]]),
        np.asarray([[0.6, 0.2], [-0.1, 0.9], [0.7, -0.5]]),
    )
    nuisance_jacobians = (
        np.asarray([[1.0], [0.0], [0.4]]),
        np.asarray([[0.2], [1.0], [-0.3]]),
        np.asarray([[0.8], [-0.2], [0.1]]),
    )
    covariances = (
        np.asarray([[1.2, 0.5, 0.1], [0.5, 1.0, 0.4], [0.1, 0.4, 0.8]]),
        np.asarray([[0.9, 0.3, 0.2], [0.3, 1.4, 0.6], [0.2, 0.6, 1.1]]),
        np.asarray([[1.1, 0.2, 0.5], [0.2, 0.7, 0.1], [0.5, 0.1, 1.3]]),
    )
    reliabilities = (
        np.asarray([0.2, 0.8, 0.6]),
        np.asarray([0.9, 0.3, 0.7]),
        np.asarray([0.4, 0.95, 0.5]),
    )
    permutations = (
        np.asarray([2, 0, 1]),
        np.asarray([1, 2, 0]),
        np.asarray([2, 1, 0]),
    )
    permuted_blocks = tuple(
        _permuted_block(
            state_jacobians[index],
            nuisance_jacobians[index],
            covariances[index],
            reliabilities[index],
            permutations[index],
        )
        for index in range(3)
    )

    original = greedy_nuisance_aware_selection(
        prior,
        state_jacobians,
        nuisance_jacobians,
        covariances,
        reliabilities=reliabilities,
        count=2,
    )
    permuted = greedy_nuisance_aware_selection(
        prior,
        tuple(block[0] for block in permuted_blocks),
        tuple(block[1] for block in permuted_blocks),
        tuple(block[2] for block in permuted_blocks),
        reliabilities=tuple(block[3] for block in permuted_blocks),
        count=2,
    )

    np.testing.assert_array_equal(original.selected_indices, permuted.selected_indices)
    np.testing.assert_allclose(
        original.mutual_information_nats,
        permuted.mutual_information_nats,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        original.final_state.marginal_state_precision(),
        permuted.final_state.marginal_state_precision(),
        rtol=1e-12,
        atol=1e-12,
    )
