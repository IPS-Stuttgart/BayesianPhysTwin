from collections.abc import Callable

import numpy as np
import pytest

import bayesian_phystwin
from bayesian_phystwin.bias_aware_belief import (
    BiasAwareStateUpdateResult,
)
from bayesian_phystwin.bias_aware_belief import (
    update_bias_aware_state as update_bias_aware_state_frozen_v1,
)
from bayesian_phystwin.bias_aware_belief_v2 import update_bias_aware_state_v2

UpdateFunction = Callable[..., BiasAwareStateUpdateResult]
_UPDATE_FUNCTIONS: tuple[UpdateFunction, ...] = (
    bayesian_phystwin.update_bias_aware_state,
    update_bias_aware_state_v2,
)


def _minimal_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((1, 2, 3), dtype=np.float64),
        np.asarray([[1.0], [-1.0]], dtype=np.float64),
        np.zeros((2, 0), dtype=np.float64),
    )


@pytest.mark.parametrize("update", _UPDATE_FUNCTIONS)
@pytest.mark.parametrize(
    "camera_available",
    [
        np.asarray([[np.nan, 0.0]]),
        np.asarray([[2, 0]]),
        np.asarray([["true", "false"]], dtype=object),
    ],
)
def test_bias_aware_public_updates_reject_truthiness_coerced_camera_masks(
    update: UpdateFunction,
    camera_available: np.ndarray,
) -> None:
    innovation, state_basis, shared_bias_basis = _minimal_problem()

    with pytest.raises(
        ValueError,
        match=r"camera_available.*booleans or exact 0/1",
    ):
        update(
            innovation,
            camera_available,
            state_basis,
            shared_bias_basis,
        )


@pytest.mark.parametrize("update", _UPDATE_FUNCTIONS)
def test_bias_aware_public_updates_accept_exact_binary_numeric_camera_masks(
    update: UpdateFunction,
) -> None:
    innovation, state_basis, shared_bias_basis = _minimal_problem()

    result = update(
        innovation,
        np.asarray([[1.0, 0.0]]),
        state_basis,
        shared_bias_basis,
    )

    np.testing.assert_array_equal(result.prior_reliability, [[1.0, 0.0]])


@pytest.mark.parametrize("invalid_config", [False, 0])
def test_public_v1_adapter_rejects_falsey_invalid_config(
    invalid_config: object,
) -> None:
    innovation, state_basis, shared_bias_basis = _minimal_problem()

    with pytest.raises(TypeError, match="BiasAwareStateUpdateConfig"):
        bayesian_phystwin.update_bias_aware_state(
            innovation,
            np.asarray([[True, False]]),
            state_basis,
            shared_bias_basis,
            config=invalid_config,  # type: ignore[arg-type]
        )


def test_public_v1_adapter_preserves_frozen_v1_results_for_valid_inputs() -> None:
    innovation, state_basis, shared_bias_basis = _minimal_problem()
    available = np.asarray([[True, False]])

    checked = bayesian_phystwin.update_bias_aware_state(
        innovation,
        available,
        state_basis,
        shared_bias_basis,
    )
    frozen = update_bias_aware_state_frozen_v1(
        innovation,
        available,
        state_basis,
        shared_bias_basis,
    )

    assert checked.accepted == frozen.accepted
    assert checked.reason == frozen.reason
    np.testing.assert_array_equal(
        checked.state_coefficients_m,
        frozen.state_coefficients_m,
    )
    np.testing.assert_array_equal(
        checked.posterior_covariance_m2,
        frozen.posterior_covariance_m2,
    )
    np.testing.assert_array_equal(
        checked.prior_reliability,
        frozen.prior_reliability,
    )
