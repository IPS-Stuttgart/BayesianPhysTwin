"""Unit tests for the Deform360 dependence robustness replay."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

RUNNER = (
    Path(__file__).resolve().parents[1]
    / "scripts/science/run_deform360_dependence_robustness_v1.py"
)
SPEC = importlib.util.spec_from_file_location("deform360_robustness_v1", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def covariance_matrix(model: object) -> np.ndarray:
    diagonal = np.asarray(model.diagonal, dtype=np.float64)
    factor = np.asarray(model.factor, dtype=np.float64)
    return float(model.multiplier) * (np.diag(diagonal) + factor @ factor.T)


def example_covariance() -> object:
    return MODULE._DummyCovariance(
        mean_error=np.zeros(5),
        diagonal=np.asarray([0.1, 0.2, 0.3, 0.4, 0.5]),
        factor=np.asarray(
            [
                [1.0, 0.1, 0.0],
                [0.8, -0.2, 0.3],
                [0.0, 1.1, 0.2],
                [-0.4, 0.2, 0.9],
                [0.3, 0.5, -0.6],
            ]
        ),
        multiplier=1.7,
    )


def test_dependence_continuum_preserves_marginals_and_endpoints() -> None:
    covariance = example_covariance()
    reference = MODULE.marginal_variance(covariance)
    zero = MODULE.dependence_strength_model(MODULE._DummyBase, covariance, 0.0)
    half = MODULE.dependence_strength_model(MODULE._DummyBase, covariance, 0.5)
    full = MODULE.dependence_strength_model(MODULE._DummyBase, covariance, 1.0)

    for model in (zero, half, full):
        np.testing.assert_allclose(MODULE.marginal_variance(model), reference)
    np.testing.assert_allclose(
        covariance_matrix(zero),
        np.diag(reference),
    )
    np.testing.assert_allclose(covariance_matrix(full), covariance_matrix(covariance))
    expected_half = 0.5 * covariance_matrix(covariance) + 0.5 * np.diag(reference)
    np.testing.assert_allclose(covariance_matrix(half), expected_half)


def test_rank_energy_path_preserves_marginals_and_nondecreasing_rank() -> None:
    covariance = example_covariance()
    reference = MODULE.marginal_variance(covariance)
    ranks: list[int] = []
    for fraction in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0):
        model, metadata = MODULE.rank_energy_model(
            MODULE._DummyBase,
            covariance,
            fraction,
        )
        np.testing.assert_allclose(MODULE.marginal_variance(model), reference)
        ranks.append(int(metadata["retained_rank"]))
    assert ranks == sorted(ranks)
    assert ranks[0] == 0
    assert ranks[-1] == np.linalg.matrix_rank(covariance.factor)


def test_lowest_risk_selection_has_exact_stable_coverage() -> None:
    predicted = np.asarray([0.3, 0.1, 0.1, 0.8, 0.2])
    mask = MODULE.lowest_risk_mask(predicted, 3)
    assert mask.tolist() == [False, True, True, False, True]
    assert MODULE.coverage_count(30, 0.1) == 3
    assert MODULE.coverage_count(30, 0.25) == 8


def test_decision_loss_matches_registered_execute_or_fallback_rule() -> None:
    predicted = np.asarray([0.01, 0.2, 0.03, 0.8])
    labels = np.asarray([False, True, True, False])
    result = MODULE.decision_metrics(
        predicted,
        labels,
        fallback_cost=0.1,
    )
    assert result["acceptance_fraction"] == 0.5
    assert result["harmful_accept_fraction_all"] == 0.25
    assert np.isclose(result["decision_loss"], 0.3)
