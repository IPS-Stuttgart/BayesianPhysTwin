import numpy as np
import pytest

from bayesian_phystwin.observation_timing_nuisance import (
    ObservationTimingPrior,
    append_timing_nuisance,
    assess_timing_identifiability,
    build_timing_jacobian,
    condition_timing_prior,
    timing_prior_precision,
)


def test_source_only_calibration_recovers_known_offset() -> None:
    derivative = np.asarray([1.0, -2.0, 0.5, 3.0])
    true_offset_s = 0.012
    timing = build_timing_jacobian(derivative)
    posterior = condition_timing_prior(
        derivative * true_offset_s,
        timing,
        np.eye(len(derivative)) * 1e-6,
        np.asarray([0.0]),
        np.asarray([[0.05**2]]),
    )

    assert posterior.mean_offset_s[0] == pytest.approx(true_offset_s, abs=2e-6)
    assert posterior.covariance_s2[0, 0] < 0.05**2
    assert posterior.information_gain_nats > 0.0


def test_spatial_bias_and_timing_can_be_exactly_confounded() -> None:
    timing = build_timing_jacobian(np.ones(4))
    result = assess_timing_identifiability(
        timing,
        np.ones((4, 1)),
        maximum_subspace_cosine=0.99,
    )

    assert not result.identifiable
    assert result.subspace_cosines[0] == pytest.approx(1.0)
    assert result.residual_fractions[0] == pytest.approx(0.0, abs=1e-12)


def test_independent_sync_anchor_breaks_spatial_bias_confounding() -> None:
    timing = build_timing_jacobian(np.ones(4))
    result = assess_timing_identifiability(
        timing,
        np.ones((4, 1)),
        independent_timing_jacobian=np.asarray([[3.0], [-2.0]]),
        maximum_subspace_cosine=0.99,
    )

    assert result.identifiable
    assert result.residual_fractions[0] > 0.0
    assert result.subspace_cosines[0] < 0.99


def test_multiple_clock_domains_preserve_row_mapping() -> None:
    timing = build_timing_jacobian(
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        stream_indices=np.asarray([0, 1, 0, 1]),
        stream_count=2,
    )

    np.testing.assert_array_equal(
        timing,
        np.asarray(
            [
                [1.0, 0.0],
                [0.0, 2.0],
                [3.0, 0.0],
                [0.0, 4.0],
            ]
        ),
    )
    combined = append_timing_nuisance(np.ones((4, 1)), timing)
    assert combined.shape == (4, 3)


def test_timing_prior_precision_uses_declared_order() -> None:
    priors = [
        ObservationTimingPrior("camera", 0.0, 0.01, "sha256:camera"),
        ObservationTimingPrior("actuator", 0.002, 0.02, "sha256:actuator"),
    ]
    np.testing.assert_allclose(
        timing_prior_precision(priors),
        np.diag([10_000.0, 2_500.0]),
    )


def test_empty_timing_stream_is_rejected() -> None:
    with pytest.raises(ValueError, match="nonzero derivative"):
        build_timing_jacobian(
            np.asarray([1.0, 2.0]),
            stream_indices=np.asarray([0, 0]),
            stream_count=2,
        )
