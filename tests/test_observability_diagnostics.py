"""Tests for nuisance-marginalized observability diagnostics."""

from __future__ import annotations

import json

import numpy as np
import pytest

from bayesian_phystwin.nuisance_aware_information import (
    NuisanceAwareInformationState,
)
from bayesian_phystwin.observability_diagnostics import (
    compare_marginal_observability,
    summarize_marginal_observability,
)


def test_contact_anchor_resolves_visual_nuisance_in_query_direction() -> None:
    prior = NuisanceAwareInformationState.from_independent_priors(
        np.eye(2),
        np.eye(1),
    )
    visual = prior.add_observation(
        np.array([[1.0, 0.0]]),
        np.array([[1.0]]),
        np.array([[0.1]]),
    )
    visuotactile = visual.add_observation(
        np.array([[1.0, 0.0]]),
        np.array([[0.0]]),
        np.array([[0.1]]),
    )

    query = np.array([[1.0, 0.0]])
    comparison = compare_marginal_observability(
        visual,
        visuotactile,
        query_jacobian=query,
    )

    assert comparison.log_determinant_gain > 0.0
    assert comparison.mutual_information_gain_nats > 0.0
    assert comparison.weakest_direction_precision_ratio > 1.0
    assert comparison.mean_variance_reduction_fraction > 0.0
    assert comparison.information_increment_eigenvalues[0] > 0.0


def test_full_state_report_retains_unchanged_directions() -> None:
    reference = NuisanceAwareInformationState.from_independent_priors(
        np.diag([2.0, 3.0]),
    )
    candidate = reference.add_observation(
        np.array([[1.0, 0.0]]),
        None,
        np.array([[0.5]]),
    )

    comparison = compare_marginal_observability(reference, candidate)

    assert comparison.weakest_direction_precision_ratio == pytest.approx(1.5)
    assert comparison.maximum_variance_reduction_fraction == pytest.approx(0.5)
    assert comparison.mean_variance_reduction_fraction == pytest.approx(0.25)
    assert comparison.information_increment_eigenvalues.tolist() == pytest.approx(
        [0.0, 2.0]
    )


def test_near_unobservable_direction_is_reported_by_numerical_rank() -> None:
    state = NuisanceAwareInformationState.from_independent_priors(
        np.diag([1.0, 1e-14]),
    )

    summary = summarize_marginal_observability(state)

    assert summary.numerical_rank == 1
    assert 1.0 <= summary.effective_rank < 1.01
    assert summary.condition_number == pytest.approx(1e14)
    assert summary.weakest_direction_variance == pytest.approx(1e14)


def test_query_basis_rotation_preserves_spectrum() -> None:
    state = NuisanceAwareInformationState.from_independent_priors(
        np.diag([2.0, 5.0]),
    )
    angle = 0.37
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )

    identity = summarize_marginal_observability(state)
    rotated = summarize_marginal_observability(
        state,
        query_jacobian=rotation,
    )

    assert rotated.precision_eigenvalues == pytest.approx(
        identity.precision_eigenvalues
    )
    assert rotated.log_determinant_precision == pytest.approx(
        identity.log_determinant_precision
    )


def test_comparison_fails_closed_on_information_loss() -> None:
    reference = NuisanceAwareInformationState.from_independent_priors(np.eye(2))
    candidate = NuisanceAwareInformationState.from_independent_priors(0.5 * np.eye(2))

    with pytest.raises(ValueError, match="information is lower"):
        compare_marginal_observability(reference, candidate)


def test_query_rows_must_be_independent() -> None:
    state = NuisanceAwareInformationState.from_independent_priors(np.eye(2))

    with pytest.raises(ValueError, match="numerically independent"):
        summarize_marginal_observability(
            state,
            query_jacobian=np.array([[1.0, 0.0], [2.0, 0.0]]),
        )


def test_records_are_json_compatible_and_arrays_are_immutable() -> None:
    state = NuisanceAwareInformationState.from_independent_priors(np.eye(2))
    candidate = state.add_observation(
        np.array([[1.0, 0.0]]),
        None,
        np.array([[1.0]]),
    )
    comparison = compare_marginal_observability(state, candidate)

    json.dumps(comparison.to_record(), sort_keys=True)
    assert not comparison.reference.precision_eigenvalues.flags.writeable
    assert not comparison.reference.marginal_variances.flags.writeable
    assert not comparison.reference.query_jacobian.flags.writeable
    assert not comparison.information_increment_eigenvalues.flags.writeable
