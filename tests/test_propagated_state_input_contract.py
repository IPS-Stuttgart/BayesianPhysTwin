from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.propagated_state_belief import infer_propagated_state_belief
from bayesian_phystwin.propagated_state_correction import select_propagated_state_update


_NON_BOOLEAN_AVAILABILITY = (
    np.ones((5, 2), dtype=np.int64),
    np.full((5, 2), 0.5, dtype=np.float64),
    np.full((5, 2), -1.0, dtype=np.float64),
    np.full((5, 2), np.inf, dtype=np.float64),
    # NaN is truthy under NumPy's bool coercion and must never mean "available".
    np.full((5, 2), np.nan, dtype=np.float64),
)


def _problem() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    frame_count = 5
    point_count = 2
    rank = 1
    innovation = np.zeros((frame_count, point_count, 3), dtype=np.float64)
    available = np.ones((frame_count, point_count), dtype=np.bool_)
    response = np.zeros((frame_count, point_count, 3, 6 * rank), dtype=np.float64)
    basis = np.array([[1.0], [-1.0]], dtype=np.float64) / np.sqrt(2.0)
    position_steps = np.ones(rank, dtype=np.float64)
    velocity_steps = np.ones(rank, dtype=np.float64)
    return (
        innovation,
        available,
        response,
        basis,
        basis,
        position_steps,
        velocity_steps,
    )


def _call(
    *,
    available: np.ndarray | None = None,
    prior_reliability: np.ndarray | None = None,
    belief_config: object | None = None,
    selection_config: object | None = None,
) -> object:
    problem = list(_problem())
    if available is not None:
        problem[1] = available
    return select_propagated_state_update(
        *problem,
        prior_reliability=prior_reliability,
        belief_config=belief_config,  # type: ignore[arg-type]
        selection_config=selection_config,  # type: ignore[arg-type]
    )


def _call_belief(
    *,
    available: np.ndarray | None = None,
    config: object | None = None,
) -> object:
    innovation, default_available, response, _, _, _, _ = _problem()
    selected_available = default_available if available is None else available
    return infer_propagated_state_belief(
        innovation,
        selected_available,
        response,
        np.zeros((response.shape[1], 0), dtype=np.float64),
        config=config,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("available", _NON_BOOLEAN_AVAILABILITY)
def test_selection_rejects_non_boolean_availability(available: np.ndarray) -> None:
    with pytest.raises(ValueError, match="availability must contain only booleans"):
        _call(available=available)


@pytest.mark.parametrize("available", _NON_BOOLEAN_AVAILABILITY)
def test_direct_belief_rejects_non_boolean_availability(
    available: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="availability must contain only booleans"):
        _call_belief(available=available)


@pytest.mark.parametrize(
    "prior_reliability",
    [
        np.full((5, 2), np.nan, dtype=np.float64),
        np.full((5, 2), -0.1, dtype=np.float64),
        np.full((5, 2), 1.1, dtype=np.float64),
    ],
)
def test_selection_rejects_invalid_prior_reliability(
    prior_reliability: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="prior reliability must lie in \[0, 1\]"):
        _call(prior_reliability=prior_reliability)


def test_selection_rejects_falsey_non_config() -> None:
    with pytest.raises(
        TypeError,
        match="selection_config must be a PropagatedStateSelectionConfig",
    ):
        _call(selection_config=0)


def test_selection_rejects_invalid_belief_config_before_inference() -> None:
    with pytest.raises(
        TypeError,
        match="belief_config must be a PropagatedStateBeliefConfig",
    ):
        _call(belief_config=0)


def test_direct_belief_rejects_falsey_non_config() -> None:
    with pytest.raises(
        TypeError,
        match="config must be a PropagatedStateBeliefConfig",
    ):
        _call_belief(config=0)
