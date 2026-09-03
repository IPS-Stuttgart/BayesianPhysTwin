from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/science/run_deform360_bootstrap_comparator_v1.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "deform360_bootstrap_comparator_v1_test_module",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_self_test() -> None:
    load_module().self_test()


def test_episode_balancing_assigns_equal_mass() -> None:
    module = load_module()
    weights = module.episode_balanced_weights([2, 4])
    assert np.isclose(weights[:2].sum(), 0.5)
    assert np.isclose(weights[2:].sum(), 0.5)
    assert np.isclose(weights.sum(), 1.0)


def test_moment_matching_is_exact() -> None:
    module = load_module()
    source = np.asarray([-3.0, -1.0, 2.0, 5.0])
    weights = module.episode_balanced_weights([1, 3])
    distribution, parity = module.moment_matched_distribution(
        source,
        weights,
        2.5,
    )
    mean = np.sum(distribution.weights * distribution.values)
    variance = np.sum(
        distribution.weights * distribution.values * distribution.values
    )
    assert abs(float(mean)) < 1e-12
    assert abs(float(variance) - 2.5) < 1e-12
    assert parity["mean_parity_abs"] < 1e-12
    assert parity["variance_parity_abs"] < 1e-12


def test_empirical_probabilities_are_monotone_in_mean() -> None:
    module = load_module()
    distribution = module.weighted_empirical(
        np.asarray([-1.0, 0.0, 1.0]),
        np.asarray([1.0, 1.0, 1.0]),
    )
    probabilities = module.empirical_event_probability(
        distribution,
        np.asarray([-1.0, 0.0, 1.0]),
        0.5,
        "upper",
    )
    assert np.all(np.diff(probabilities) >= 0.0)


def test_crps_is_nonnegative() -> None:
    module = load_module()
    distribution = module.weighted_empirical(
        np.asarray([-1.0, 0.0, 1.0]),
        np.asarray([1.0, 1.0, 1.0]),
    )
    errors = np.asarray([-0.5, 0.0, 0.5])
    assert module.empirical_standardized_crps(distribution, errors, 1.0) >= 0.0
    assert module.gaussian_standardized_crps(errors) >= 0.0
