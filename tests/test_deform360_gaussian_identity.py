from __future__ import annotations

import numpy as np

from causal4d_public.deform360_gaussian_identity import (
    GaussianIdentityConfig,
    match_gaussian_identities,
)


def test_identity_path_preserves_stable_export_order() -> None:
    previous = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])

    result = match_gaussian_identities(previous, previous.copy())

    np.testing.assert_array_equal(result.current_index_by_previous, [0, 1])
    np.testing.assert_allclose(result.distance_m, 0.0)
    np.testing.assert_allclose(result.reliability, 1.0)
    assert result.diagnostics["stable_order_match_count"] == 2


def test_dense_neighbors_do_not_make_stable_export_order_ambiguous() -> None:
    previous = np.column_stack((np.arange(8) * 0.0002, np.zeros((8, 2))))

    result = match_gaussian_identities(previous, previous.copy())

    np.testing.assert_allclose(result.reliability, 1.0)
    assert result.diagnostics["ambiguous_match_count"] == 0


def test_equal_cardinality_permutation_is_spatially_rematched() -> None:
    previous = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    current = previous[[2, 0, 1]]

    result = match_gaussian_identities(previous, current)

    np.testing.assert_array_equal(result.current_index_by_previous, [1, 2, 0])
    np.testing.assert_allclose(result.distance_m, 0.0)
    assert result.diagnostics["spatial_rematch_count"] == 3


def test_one_identity_outlier_is_quarantined_without_episode_failure() -> None:
    previous = np.column_stack((np.arange(5) * 0.1, np.zeros((5, 2))))
    current = previous.copy()
    current[-1, 0] = 2.0

    result = match_gaussian_identities(previous, current)

    np.testing.assert_array_equal(result.current_index_by_previous[:4], [0, 1, 2, 3])
    assert result.current_index_by_previous[4] == -1
    assert result.diagnostics["match_fraction"] == 0.8
    assert result.assignment_variance_m2[4] > 0.0


def test_ambiguous_duplicate_identity_keeps_assignment_uncertain() -> None:
    previous = np.array([[0.0, 0.0, 0.0]])
    current = np.array([[-0.001, 0.0, 0.0], [0.001, 0.0, 0.0]])
    config = GaussianIdentityConfig(
        maximum_distance_m=0.03,
        ambiguity_margin_m=0.002,
    )

    result = match_gaussian_identities(previous, current, config)

    assert result.current_index_by_previous[0] in (0, 1)
    assert result.reliability[0] == 0.0
    assert result.assignment_variance_m2[0] >= config.maximum_distance_m**2
    assert result.diagnostics["ambiguous_match_count"] == 1
