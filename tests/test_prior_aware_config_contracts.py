from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from bayesian_phystwin.prior_aware_gauge_belief import PriorAwareGaugeConfigV1


@pytest.mark.parametrize(
    "value",
    (
        True,
        False,
        1.0,
        np.float64(2.0),
        "2",
    ),
)
def test_maximum_iterations_requires_a_genuine_integer(value: object) -> None:
    with pytest.raises(ValueError, match="maximum_iterations must be an integer"):
        PriorAwareGaugeConfigV1(maximum_iterations=cast(int, value))


@pytest.mark.parametrize("value", (0, -1, np.int64(0)))
def test_maximum_iterations_must_be_positive(value: object) -> None:
    with pytest.raises(ValueError, match="maximum_iterations must be positive"):
        PriorAwareGaugeConfigV1(maximum_iterations=cast(int, value))


def test_numpy_integer_iteration_count_is_normalized() -> None:
    config = PriorAwareGaugeConfigV1(maximum_iterations=cast(int, np.int64(3)))

    assert config.maximum_iterations == 3
    assert type(config.maximum_iterations) is int
