from __future__ import annotations

import pytest

from bayesian_phystwin.guard_harm_risk import (
    minimum_zero_harm_groups_for_certificate,
    one_sided_binomial_upper_bound,
)


def test_documented_zero_harm_support_table() -> None:
    assert minimum_zero_harm_groups_for_certificate(0.30, 0.95) == 9
    assert minimum_zero_harm_groups_for_certificate(0.25, 0.95) == 11
    assert minimum_zero_harm_groups_for_certificate(0.20, 0.95) == 14
    assert minimum_zero_harm_groups_for_certificate(0.10, 0.95) == 29
    assert minimum_zero_harm_groups_for_certificate(0.05, 0.95) == 59


def test_ten_zero_harm_objects_have_only_a_loose_upper_bound() -> None:
    upper = one_sided_binomial_upper_bound(0, 10, 0.95)

    assert upper == pytest.approx(0.2588655508930523)
    assert upper > 0.25
