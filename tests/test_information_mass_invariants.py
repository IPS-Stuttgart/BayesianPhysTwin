from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin._gauge_aware_contracts import (
    COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,
)
from bayesian_phystwin._gauge_aware_solver import _correlation_group_weights
from bayesian_phystwin._prior_aware_gauge_math import _group_layout


@pytest.mark.parametrize("seed", range(16))
def test_information_mass_multiplies_prior_terms_before_group_capping(
    seed: int,
) -> None:
    rng = np.random.default_rng(seed + 4_000)
    row_count = int(rng.integers(4, 25))
    group_ids = tuple(
        f"group-{group_id}"
        for group_id in rng.integers(0, 5, size=row_count)
    )
    reliability = rng.uniform(0.0, 1.0, size=row_count)
    nominal_probability = rng.uniform(0.0, 1.0, size=row_count)
    composite_weight = rng.uniform(0.01, 1.0, size=row_count)
    effective_samples = float(rng.uniform(0.25, row_count + 1.0))

    actual, counts = _correlation_group_weights(
        group_ids,
        reliability,
        nominal_probability,
        composite_weight,
        effective_samples,
    )

    expected = reliability * nominal_probability * composite_weight
    expected_counts: dict[str, int] = {}
    for group_id in group_ids:
        expected_counts[group_id] = expected_counts.get(group_id, 0) + 1
    for index, group_id in enumerate(group_ids):
        count = expected_counts[group_id]
        expected[index] *= min(effective_samples, float(count)) / count

    assert counts == expected_counts
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0.0)


@pytest.mark.parametrize("zero_term", ["reliability", "nominal", "composite"])
def test_zero_prior_term_contributes_zero_information_mass(zero_term: str) -> None:
    reliability = np.asarray([0.8, 0.7, 0.6], dtype=np.float64)
    nominal_probability = np.asarray([0.9, 0.8, 0.7], dtype=np.float64)
    composite_weight = np.asarray([0.5, 0.4, 0.3], dtype=np.float64)
    arrays = {
        "reliability": reliability,
        "nominal": nominal_probability,
        "composite": composite_weight,
    }
    arrays[zero_term][1] = 0.0

    actual, counts = _correlation_group_weights(
        ("shared", "shared", "shared"),
        reliability,
        nominal_probability,
        composite_weight,
        effective_samples_per_group=2.0,
    )

    assert counts == {"shared": 3}
    assert actual[1] == 0.0



def test_provider_final_per_row_power_is_not_capped_twice() -> None:
    effective = 64.0

    def standard_mass(row_count: int) -> float:
        weights, _ = _correlation_group_weights(
            tuple("provider-group" for _ in range(row_count)),
            np.ones(row_count),
            np.ones(row_count),
            np.full(row_count, effective / row_count),
            effective,
            composite_weight_mode=COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,
        )
        return float(np.sum(weights))

    assert standard_mass(128) == pytest.approx(effective)
    assert standard_mass(1_024) == pytest.approx(effective)


def test_prior_aware_provider_power_is_duplication_invariant() -> None:
    effective = 32.0

    def layout(row_count: int) -> tuple[np.ndarray, np.ndarray]:
        _, _, base, _, group_power = _group_layout(
            tuple("provider-group" for _ in range(row_count)),
            np.ones(row_count),
            np.ones(row_count),
            np.full(row_count, effective / row_count),
            cap=effective,
            composite_weight_mode=COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,
        )
        return base, group_power

    base_small, power_small = layout(64)
    base_large, power_large = layout(640)
    assert float(np.sum(base_small)) == pytest.approx(effective)
    assert float(np.sum(base_large)) == pytest.approx(effective)
    assert power_small[0] == pytest.approx(effective / 64)
    assert power_large[0] == pytest.approx(effective / 640)
