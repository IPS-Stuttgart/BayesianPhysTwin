from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "science"
    / "run_deform360_source_response_competence_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_deform360_source_response_competence_v1", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_rotation_log_recovers_z_rotation() -> None:
    angle = 0.2
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(
        MODULE._rotation_log(rotation), [0.0, 0.0, angle], atol=1e-10
    )


def test_block_permutation_preserves_rows_within_each_split() -> None:
    values = np.arange(80, dtype=np.float64).reshape(40, 2)
    splits = [slice(0, 20), slice(20, 40)]
    permuted = MODULE._block_permute(
        values, splits, block_size=4, seed=20260902
    )
    assert not np.array_equal(permuted, values)
    for split in splits:
        expected = sorted(map(tuple, values[split].tolist()))
        observed = sorted(map(tuple, permuted[split].tolist()))
        assert observed == expected


def test_multivariate_belief_is_positive_definite_and_finite() -> None:
    rng = np.random.default_rng(11)
    features = rng.normal(size=(180, 5))
    design = MODULE._with_intercept(features)
    coefficient = rng.normal(size=(design.shape[1], 3))
    noise_covariance = np.array(
        [[0.25, 0.12, -0.03], [0.12, 0.30, 0.04], [-0.03, 0.04, 0.20]]
    )
    targets = design @ coefficient + rng.multivariate_normal(
        np.zeros(3), noise_covariance, size=len(design)
    )
    model = MODULE._fit(
        design[:120], targets[:120], ridge=0.01, eigenvalue_floor=1e-8
    )
    means, covariances = MODULE._predict(
        model, design[120:], covariance_scale=1.0
    )
    assert means.shape == (60, 3)
    assert covariances.shape == (60, 3, 3)
    assert np.all(np.linalg.eigvalsh(covariances) > 0.0)
    for diagonal in (False, True):
        metrics = MODULE._probabilistic_metrics(
            targets[120:], means, covariances, diagonal=diagonal
        )
        assert all(np.isfinite(value) for value in metrics.values())
        assert metrics["rmse"] >= 0.0
        assert metrics["normalized_joint_nees"] >= 0.0
        assert 0.0 <= metrics["marginal_90_coverage"] <= 1.0


def test_tuning_selects_frozen_grid_values() -> None:
    rng = np.random.default_rng(17)
    features = rng.normal(size=(130, 4))
    design = MODULE._with_intercept(features)
    coefficient = rng.normal(size=(design.shape[1], 2))
    targets = design @ coefficient + rng.normal(scale=0.3, size=(130, 2))
    ridge_grid = [1e-4, 0.01, 1.0]
    scale_grid = [0.5, 1.0, 2.0]
    model, scale, metrics = MODULE._tune(
        design[:80],
        targets[:80],
        design[80:105],
        targets[80:105],
        ridge_grid=ridge_grid,
        covariance_scale_grid=scale_grid,
        eigenvalue_floor=1e-8,
        diagonal=False,
    )
    assert model["ridge"] in ridge_grid
    assert scale in scale_grid
    assert np.isfinite(metrics["nll_per_dimension"])
