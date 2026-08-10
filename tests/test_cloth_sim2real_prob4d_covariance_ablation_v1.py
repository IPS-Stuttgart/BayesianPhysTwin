from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/science/run_cloth_sim2real_prob4d_covariance_ablation_v1.py"


def _module():
    spec = importlib.util.spec_from_file_location("cloth_covariance_runner", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _statistics(module):
    residual = np.asarray(
        [
            [[1.0, 0.2, -0.1], [2.0, 0.1, 0.3], [3.0, -0.2, 0.4]],
            [[1.5, 0.3, -0.2], [1.0, 0.0, 0.2], [4.0, -0.1, 0.5]],
        ]
    )
    precision = np.asarray([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
    return module.FitStatistics(
        residual_mean_rows_m=residual,
        row_precision_per_m2=precision,
        vertex_mean_m=np.zeros((3, 3)),
        local_mean_variance_m2=np.ones(3),
        robust_probability=np.ones((2, 3)),
        assignment_entropy=np.zeros((2, 3)),
    )


def _explicit_normal_equations(residual, precision, covariance):
    frame_count, node_count = precision.shape
    design = np.zeros((frame_count * node_count, node_count))
    for frame in range(frame_count):
        for node in range(node_count):
            design[frame * node_count + node, node] = 1.0
    inverse = np.linalg.inv(covariance)
    values = residual.reshape(frame_count * node_count, 3)
    return design.T @ inverse @ design, design.T @ inverse @ values


def test_covariance_policy_matches_public_five_way_contract() -> None:
    module = _module()
    policies = {
        name: module.CovariancePolicy.from_treatment(name) for name in module.TREATMENTS
    }
    assert policies["full_joint"].shared_uncertainty_scale == 1.0
    assert policies["full_joint"].gauge_factors_enabled is True
    assert policies["full_joint"].construction == "persistent"
    assert policies["block_diagonal"].shared_uncertainty_scale == 1.0
    assert policies["block_diagonal"].gauge_factors_enabled is False
    assert policies["block_diagonal"].construction == "frame_blocks"
    assert (
        policies["full_joint"].construction != policies["block_diagonal"].construction
    )
    assert policies["independent_rows"].shared_uncertainty_scale == 0.0
    assert policies["independent_rows"].construction == "independent_marginals"
    assert policies["shared_uncertainty_removed"].shared_uncertainty_scale == 0.0
    assert policies["shared_uncertainty_removed"].construction == "removed"
    assert policies["shared_uncertainty_underreported"].shared_uncertainty_scale == 0.5
    assert policies["shared_uncertainty_underreported"].gauge_factors_enabled
    assert (
        policies["shared_uncertainty_underreported"].construction == "persistent_half"
    )


def test_full_joint_matches_explicit_persistent_shared_covariance() -> None:
    module = _module()
    statistics = _statistics(module)
    shared_variance = 0.25
    information, rhs, _ = module.covariance_normal_equations(
        statistics,
        module.CovariancePolicy.from_treatment("full_joint"),
        shared_bias_variance_m2=shared_variance,
    )
    row_variance = 1.0 / statistics.row_precision_per_m2.reshape(-1)
    covariance = np.diag(row_variance) + shared_variance * np.ones(
        (len(row_variance), len(row_variance))
    )
    expected_information, expected_rhs = _explicit_normal_equations(
        statistics.residual_mean_rows_m,
        statistics.row_precision_per_m2,
        covariance,
    )
    assert np.allclose(information, expected_information)
    assert np.allclose(rhs, expected_rhs)


def test_block_diagonal_matches_explicit_frame_local_shared_covariance() -> None:
    module = _module()
    statistics = _statistics(module)
    shared_variance = 0.25
    information, rhs, _ = module.covariance_normal_equations(
        statistics,
        module.CovariancePolicy.from_treatment("block_diagonal"),
        shared_bias_variance_m2=shared_variance,
    )
    precision = statistics.row_precision_per_m2
    frame_count, node_count = precision.shape
    covariance = np.zeros((frame_count * node_count, frame_count * node_count))
    for frame in range(frame_count):
        start = frame * node_count
        stop = start + node_count
        covariance[start:stop, start:stop] = np.diag(
            1.0 / precision[frame]
        ) + shared_variance * np.ones((node_count, node_count))
    expected_information, expected_rhs = _explicit_normal_equations(
        statistics.residual_mean_rows_m,
        precision,
        covariance,
    )
    assert np.allclose(information, expected_information)
    assert np.allclose(rhs, expected_rhs)


def test_ablation_changes_only_dependence_and_has_expected_information_order() -> None:
    module = _module()
    statistics = _statistics(module)
    matrices = {}
    diagnostics = {}
    for treatment in module.TREATMENTS:
        matrices[treatment], _, diagnostics[treatment] = (
            module.covariance_normal_equations(
                statistics,
                module.CovariancePolicy.from_treatment(treatment),
                shared_bias_variance_m2=0.25,
            )
        )
    assert np.allclose(
        matrices["independent_rows"],
        np.diag(np.diag(matrices["independent_rows"])),
    )
    assert np.allclose(
        matrices["shared_uncertainty_removed"],
        np.diag(np.diag(matrices["shared_uncertainty_removed"])),
    )
    assert np.any(np.abs(np.triu(matrices["full_joint"], 1)) > 0.0)
    assert np.any(np.abs(np.triu(matrices["block_diagonal"], 1)) > 0.0)
    assert not np.allclose(matrices["full_joint"], matrices["block_diagonal"])
    assert (
        diagnostics["full_joint"]["constant_mode_information"]
        < diagnostics["shared_uncertainty_underreported"]["constant_mode_information"]
        < diagnostics["shared_uncertainty_removed"]["constant_mode_information"]
    )


def test_graph_posterior_returns_finite_mean_and_covariance() -> None:
    module = _module()
    statistics = _statistics(module)
    information, rhs, _ = module.covariance_normal_equations(
        statistics,
        module.CovariancePolicy.from_treatment("full_joint"),
        shared_bias_variance_m2=0.25,
    )
    laplacian = np.asarray([[1.0, -1.0, 0.0], [-0.5, 1.0, -0.5], [0.0, -1.0, 1.0]])
    mean, variance, diagnostics = module._posterior_solution(
        information,
        rhs,
        laplacian.T @ laplacian,
        reference_variance_m2=0.5,
        prior_strength=1.0,
        covariance=True,
    )
    assert mean.shape == (3, 3)
    assert variance is not None and variance.shape == (3,)
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(variance)) and np.all(variance >= 0.0)
    assert diagnostics["posterior_precision_condition_estimate"] >= 1.0
