import numpy as np
import pytest

from bayesian_phystwin import (
    PseudoMeasurementBatch,
    ReliabilityConfig,
    reliability_weighted_loss,
    score_reliability,
)


def test_low_confidence_occluded_outlier_is_downweighted() -> None:
    batch = PseudoMeasurementBatch(
        observed=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        predicted=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        variance=1.0,
        confidence=[1.0, 0.2],
        occluded=[False, True],
        flow_inconsistency=[0.0, 1.0],
    )

    result = score_reliability(batch, ReliabilityConfig(residual_scale=1.0))

    assert result.weights[0] > 0.99
    assert result.weights[1] < 0.01
    assert result.inflated_variance[1, 0] > result.inflated_variance[0, 0]


def test_weighted_loss_is_finite() -> None:
    batch = PseudoMeasurementBatch(
        observed=np.ones((4, 3)),
        predicted=np.zeros((4, 3)),
        variance=np.full((4, 3), 0.5),
    )

    loss = reliability_weighted_loss(batch)

    assert np.isfinite(loss)
    assert loss >= 0.0


def test_weighted_loss_applies_reliability_once() -> None:
    batch = PseudoMeasurementBatch(
        observed=[[2.0]],
        predicted=[[0.0]],
        variance=1.0,
        confidence=[0.5],
    )

    result = score_reliability(batch)
    loss = reliability_weighted_loss(batch)

    assert result.weights[0] == pytest.approx(0.5)
    assert result.inflated_variance[0, 0] == pytest.approx(2.0)
    assert loss == pytest.approx(2.0)


def test_shape_mismatch_raises() -> None:
    batch = PseudoMeasurementBatch(
        observed=np.ones((4, 3)),
        predicted=np.ones((5, 3)),
    )

    with pytest.raises(ValueError):
        score_reliability(batch)


def test_residual_gating_is_opt_in() -> None:
    batch = PseudoMeasurementBatch(
        observed=[[10.0]],
        predicted=[[0.0]],
        confidence=[1.0],
    )

    cue_only = score_reliability(batch)
    residual_gated = score_reliability(batch, ReliabilityConfig(residual_scale=1.0))

    assert cue_only.weights[0] > 0.99
    assert residual_gated.weights[0] == pytest.approx(1e-3)
