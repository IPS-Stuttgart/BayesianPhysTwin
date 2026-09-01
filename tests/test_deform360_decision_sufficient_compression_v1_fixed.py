"""Corrected contracts for decision-sufficient covariance compression."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODULE_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "science"
    / "run_deform360_decision_sufficient_compression_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "deform360_decision_sufficient_compression_v1_fixed",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@dataclass
class Covariance:
    mean_error: np.ndarray
    diagonal: np.ndarray
    factor: np.ndarray
    multiplier: float = 1.0
    marginal_z: float = 1.0
    source_marginal_coverage: float = 0.9
    source_joint_nanees: float = 1.0


class Base:
    CovarianceModel = Covariance


def test_portfolio_projection_preserves_complete_query_covariance() -> None:
    rng = np.random.default_rng(12)
    dimension = 31
    latent_rank = 9
    query_count = 5
    factor = rng.normal(size=(dimension, latent_rank))
    query = rng.normal(size=(query_count, dimension))
    covariance = Covariance(
        mean_error=np.zeros(dimension),
        diagonal=np.linspace(0.1, 0.5, dimension),
        factor=factor,
        multiplier=1.7,
    )
    basis, _ = MODULE.orthonormal_range(factor.T @ query.T, 1e-12)
    compressed = MODULE.projected_model(Base, covariance, basis)
    np.testing.assert_allclose(
        MODULE.query_covariance(compressed, query),
        MODULE.query_covariance(covariance, query),
        rtol=1e-11,
        atol=1e-11,
    )
    assert basis.shape[1] <= query_count


def test_scalar_query_needs_at_most_one_shared_factor() -> None:
    rng = np.random.default_rng(23)
    factor = rng.normal(size=(17, 7))
    weight = rng.normal(size=17)
    basis, _ = MODULE.orthonormal_range(factor.T @ weight[:, None], 1e-12)
    assert basis.shape[1] <= 1
    full = weight @ factor
    compressed = weight @ (factor @ basis)
    np.testing.assert_allclose(
        np.sum(compressed**2),
        np.sum(full**2),
        rtol=1e-11,
        atol=1e-11,
    )


def test_zero_query_visible_factor_has_rank_zero() -> None:
    factor = np.eye(4)
    weight = np.zeros(4)
    basis, singular = MODULE.orthonormal_range(
        factor.T @ weight[:, None],
        1e-12,
    )
    assert basis.shape == (4, 0)
    assert singular.shape == (1,)


def test_spectral_control_has_same_rank_but_need_not_preserve_query() -> None:
    factor = np.asarray(
        [
            [10.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    query = np.asarray([[0.0, 0.0, 1.0]])
    sufficient, _ = MODULE.orthonormal_range(factor.T @ query.T, 1e-12)
    spectral = MODULE.spectral_basis(factor, sufficient.shape[1])
    sufficient_variance = np.sum((query @ factor @ sufficient) ** 2)
    spectral_variance = np.sum((query @ factor @ spectral) ** 2)
    full_variance = np.sum((query @ factor) ** 2)
    np.testing.assert_allclose(sufficient_variance, full_variance)
    assert spectral_variance != full_variance
