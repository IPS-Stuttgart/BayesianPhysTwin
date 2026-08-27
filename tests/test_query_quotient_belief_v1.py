from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.query_quotient_belief_v1 import (
    QUERY_QUOTIENT_BELIEF_CLAIM_BOUNDARY,
    aggregate_to_query_quotient,
    minimum_information_query_lift,
    query_ambiguity_envelope,
    query_quotient_information_decomposition,
)


def test_minimum_information_lift_matches_quotient_and_preserves_conditionals() -> None:
    prior = np.array([0.1, 0.2, 0.3, 0.4])
    classes = np.array([0, 0, 1, 1])
    quotient = np.array([0.75, 0.25])

    result = minimum_information_query_lift(prior, classes, quotient)

    np.testing.assert_allclose(
        result.lifted_weights,
        np.array([0.25, 0.5, 3.0 / 28.0, 1.0 / 7.0]),
    )
    np.testing.assert_allclose(
        aggregate_to_query_quotient(result.lifted_weights, classes),
        quotient,
    )
    np.testing.assert_allclose(
        result.lifted_weights[:2] / np.sum(result.lifted_weights[:2]),
        prior[:2] / np.sum(prior[:2]),
    )
    np.testing.assert_allclose(
        result.lifted_weights[2:] / np.sum(result.lifted_weights[2:]),
        prior[2:] / np.sum(prior[2:]),
    )
    assert result.information.unsupported_specificity_nats == pytest.approx(
        0.0, abs=1e-12
    )
    assert result.information.total_information_nats == pytest.approx(
        result.information.quotient_information_nats,
        abs=1e-12,
    )
    assert "does not establish" in QUERY_QUOTIENT_BELIEF_CLAIM_BOUNDARY


def test_kl_chain_rule_exposes_extra_within_class_specificity() -> None:
    prior = np.array([0.1, 0.2, 0.3, 0.4])
    classes = np.array([0, 0, 1, 1])
    quotient = np.array([0.75, 0.25])
    canonical = minimum_information_query_lift(prior, classes, quotient)
    alternative = np.array([0.6, 0.15, 0.05, 0.2])

    information = query_quotient_information_decomposition(
        prior,
        alternative,
        classes,
    )

    np.testing.assert_allclose(
        information.posterior_quotient_weights,
        quotient,
    )
    assert information.unsupported_specificity_nats > 0.0
    assert information.total_information_nats == pytest.approx(
        information.quotient_information_nats
        + information.unsupported_specificity_nats,
        abs=1e-12,
    )
    assert information.total_information_nats > (
        canonical.information.total_information_nats
    )
    assert information.supported_information_fraction < 1.0


def test_canonical_lift_minimizes_information_over_feasible_alternatives() -> None:
    prior = np.array([0.12, 0.18, 0.28, 0.42])
    classes = np.array([0, 0, 1, 1])
    quotient = np.array([0.4, 0.6])
    canonical = minimum_information_query_lift(prior, classes, quotient)
    alternatives = (
        np.array([0.05, 0.35, 0.1, 0.5]),
        np.array([0.3, 0.1, 0.5, 0.1]),
        np.array([0.2, 0.2, 0.3, 0.3]),
    )

    for alternative in alternatives:
        information = query_quotient_information_decomposition(
            prior,
            alternative,
            classes,
        )
        assert information.total_information_nats >= (
            canonical.information.total_information_nats - 1e-12
        )


def test_ambiguity_envelope_distinguishes_query_from_physical_specificity() -> None:
    classes = np.array([0, 0, 1, 1])
    quotient = np.array([0.25, 0.75])
    query_values = np.array([10.0, 10.0, 20.0, 20.0])
    physical_values = np.array([0.0, 2.0, -1.0, 3.0])

    query_envelope = query_ambiguity_envelope(
        quotient,
        classes,
        query_values,
    )
    physical_envelope = query_ambiguity_envelope(
        quotient,
        classes,
        physical_values,
    )

    np.testing.assert_allclose(query_envelope.lower, np.array([17.5]))
    np.testing.assert_allclose(query_envelope.upper, np.array([17.5]))
    assert query_envelope.all_identified
    np.testing.assert_allclose(physical_envelope.lower, np.array([-0.75]))
    np.testing.assert_allclose(physical_envelope.upper, np.array([2.75]))
    np.testing.assert_allclose(physical_envelope.width, np.array([3.5]))
    assert not physical_envelope.all_identified


def test_ambiguity_envelope_supports_multiple_registered_endpoints() -> None:
    classes = np.array([0, 0, 1])
    quotient = np.array([0.4, 0.6])
    values = np.array([[1.0, -2.0], [1.0, 2.0], [3.0, 4.0]])

    envelope = query_ambiguity_envelope(
        quotient,
        classes,
        values,
        identifiability_tolerance=1e-14,
    )

    np.testing.assert_allclose(envelope.lower, np.array([2.2, 1.6]))
    np.testing.assert_allclose(envelope.upper, np.array([2.2, 3.2]))
    np.testing.assert_array_equal(envelope.identified_mask, np.array([True, False]))
    assert envelope.maximum_width == pytest.approx(1.6)


def test_outputs_are_immutable() -> None:
    result = minimum_information_query_lift(
        [0.2, 0.3, 0.5],
        [0, 0, 1],
        [0.6, 0.4],
    )
    envelope = query_ambiguity_envelope(
        [0.6, 0.4],
        [0, 0, 1],
        [1.0, 2.0, 3.0],
    )

    with pytest.raises(ValueError, match="read-only"):
        result.lifted_weights[0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        result.class_index[0] = 1
    with pytest.raises(ValueError, match="read-only"):
        envelope.width[0] = 0.0


@pytest.mark.parametrize(
    ("prior", "classes", "quotient", "match"),
    [
        ([0.5, 0.5], [0, 2], [0.5, 0.0, 0.5], "contiguous"),
        ([0.5, 0.5], [0.0, 1.0], [0.5, 0.5], "integer"),
        ([0.5, 0.5], [-1, 0], [0.5], "nonnegative"),
        ([0.5, 0.4], [0, 1], [0.5, 0.5], "sum to one"),
        ([0.5, 0.5], [0, 1], [0.6, 0.3], "sum to one"),
        ([0.0, 1.0], [0, 1], [0.5, 0.5], "zero prior support"),
    ],
)
def test_minimum_information_lift_rejects_invalid_contracts(
    prior: object,
    classes: object,
    quotient: object,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        minimum_information_query_lift(prior, classes, quotient)


def test_decomposition_rejects_posterior_outside_prior_support() -> None:
    with pytest.raises(ValueError, match="absolutely continuous"):
        query_quotient_information_decomposition(
            [0.0, 0.5, 0.5],
            [0.1, 0.4, 0.5],
            [0, 0, 1],
        )


def test_ambiguity_envelope_rejects_misaligned_or_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="one row"):
        query_ambiguity_envelope([0.5, 0.5], [0, 1], [1.0])
    with pytest.raises(ValueError, match="finite"):
        query_ambiguity_envelope([0.5, 0.5], [0, 1], [1.0, np.nan])
    with pytest.raises(ValueError, match="nonnegative"):
        query_ambiguity_envelope(
            [0.5, 0.5],
            [0, 1],
            [1.0, 2.0],
            identifiability_tolerance=-1.0,
        )
