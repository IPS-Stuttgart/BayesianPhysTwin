from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

RUNNER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "science"
    / "run_deform360_query_sufficient_dependence_v1.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "deform360_query_sufficient_dependence_v1_for_tests",
        RUNNER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_exact_projection_preserves_complete_query_covariance() -> None:
    module = load_runner()
    rng = np.random.default_rng(11)
    factor = rng.normal(size=(30, 8))
    query = rng.normal(size=(5, 30))
    result = module.exact_query_projection(
        factor,
        query,
        relative_rank_tolerance=1e-13,
        absolute_rank_tolerance=1e-15,
    )
    assert result["retained_rank"] <= 5
    full = (query @ factor) @ (query @ factor).T
    reduced_factor = result["compressed_factor"]
    reduced = (query @ reduced_factor) @ (query @ reduced_factor).T
    np.testing.assert_allclose(reduced, full, rtol=1e-11, atol=1e-11)


def test_scalar_query_requires_at_most_one_latent_direction() -> None:
    module = load_runner()
    rng = np.random.default_rng(12)
    factor = rng.normal(size=(24, 8))
    query = rng.normal(size=(1, 24))
    result = module.exact_query_projection(
        factor,
        query,
        relative_rank_tolerance=1e-13,
        absolute_rank_tolerance=1e-15,
    )
    assert result["retained_rank"] <= 1
    assert result["query_covariance_relative_frobenius_error"] < 1e-11


def test_zero_query_has_zero_minimum_rank() -> None:
    module = load_runner()
    rng = np.random.default_rng(13)
    factor = rng.normal(size=(21, 6))
    result = module.exact_query_projection(
        factor,
        np.zeros((3, 21)),
        relative_rank_tolerance=1e-13,
        absolute_rank_tolerance=1e-15,
    )
    assert result["retained_rank"] == 0
    assert result["compressed_factor"].shape == (21, 0)


def test_leading_energy_is_not_query_sufficient_in_general() -> None:
    module = load_runner()
    factor = np.diag([100.0, 80.0, 60.0, 0.25])
    query = np.asarray([[0.0, 0.0, 0.0, 1.0]])
    exact = module.exact_query_projection(
        factor,
        query,
        relative_rank_tolerance=1e-13,
        absolute_rank_tolerance=1e-15,
    )
    energy = module.leading_energy_projection(factor, exact["retained_rank"])
    assert exact["retained_rank"] == 1
    exact_variance = float(np.square(query @ exact["compressed_factor"]).sum())
    energy_variance = float(np.square(query @ energy["compressed_factor"]).sum())
    full_variance = float(np.square(query @ factor).sum())
    assert exact_variance == full_variance
    assert energy_variance == 0.0


def test_canonical_basis_preserves_projector() -> None:
    module = load_runner()
    rng = np.random.default_rng(14)
    vectors, _ = np.linalg.qr(rng.normal(size=(9, 4)))
    canonical = module.canonical_projector_basis(vectors)
    np.testing.assert_allclose(
        canonical @ canonical.T,
        vectors @ vectors.T,
        rtol=1e-11,
        atol=1e-11,
    )
