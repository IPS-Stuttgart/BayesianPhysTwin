from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin_experiments.dlolab_slingshot_policy_certificate_v2 import (
    combined_competence_features,
    posterior_diagnostic_features,
)


def _inference() -> dict[str, np.ndarray]:
    weights = np.arange(1, 28, dtype=np.float64)
    weights /= weights.sum()
    return {
        "weights": weights,
        "iid_weights": weights[::-1],
        "expected_losses": np.linspace(-7.0, -6.4, 7),
        "iid_expected_losses": np.linspace(-6.9, -6.3, 7),
        "map_losses": np.linspace(-6.8, -6.2, 7),
        "nominal_losses": np.linspace(-6.7, -6.1, 7),
        "prior_losses": np.linspace(-6.6, -6.0, 7),
        "raw_upper": np.arange(21, dtype=np.float64).reshape(3, 7) / 100.0,
    }


def test_posterior_diagnostics_are_offset_invariant_and_fixed_width() -> None:
    first = _inference()
    second = {name: value.copy() for name, value in first.items()}
    for name in (
        "expected_losses",
        "iid_expected_losses",
        "map_losses",
        "nominal_losses",
        "prior_losses",
    ):
        second[name] += 10.0

    np.testing.assert_array_equal(
        posterior_diagnostic_features(first), posterior_diagnostic_features(second)
    )
    assert posterior_diagnostic_features(first).shape == (110,)


def test_combined_features_cancel_shared_observation_bias() -> None:
    rng = np.random.default_rng(262001)
    observation = rng.normal(size=(3, 4, 3))
    shifted = observation + np.asarray([0.2, -0.1, 0.3])

    np.testing.assert_allclose(
        combined_competence_features(observation, _inference()),
        combined_competence_features(shifted, _inference()),
        rtol=0.0,
        atol=1e-15,
    )
    assert combined_competence_features(observation, _inference()).shape == (161,)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("weights", np.ones(27), "normalized"),
        ("expected_losses", np.zeros(6), "shape"),
        ("raw_upper", np.full((3, 7), np.nan), "finite"),
    ],
)
def test_posterior_diagnostics_reject_invalid_inputs(
    field: str, value: np.ndarray, message: str
) -> None:
    inference = _inference()
    inference[field] = value

    with pytest.raises(ValueError, match=message):
        posterior_diagnostic_features(inference)
