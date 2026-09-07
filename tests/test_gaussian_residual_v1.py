"""Numerical and information-boundary tests, not physical evidence."""

import numpy as np
import pytest

from bayesian_phystwin_experiments.gaussian_residual_v1 import (
    GaussianResidual,
    apply_canonical_correction,
    grouped_support,
    matern32,
    require_disjoint_groups,
)


def fixture():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(24, 3))
    y = np.column_stack((np.sin(x[:, 0]), x[:, 1] ** 2))
    groups = np.repeat(np.arange(4), 6)
    return x, y, groups


def test_gp_mean_covariance_match_independent_dense_conditioning():
    x, y, groups = fixture()
    model = GaussianResidual.fit(x, y, groups, length_scale=0.8, noise_variance=0.03)
    query = x[:5] + 0.2
    z = (query - model.location) / model.scale
    kernel = matern32(model.support, model.support, 0.8) + 0.03 * np.eye(24)
    cross = matern32(z, model.support, 0.8)
    expected_mean = (
        cross @ np.linalg.solve(kernel, y / model.output_scale) * model.output_scale
    )
    expected_cov = matern32(z, z, 0.8) - cross @ np.linalg.solve(kernel, cross.T)
    mean, covariance = model.predict(query, full_covariance=True)
    np.testing.assert_allclose(mean, expected_mean, rtol=1e-11, atol=1e-12)
    for axis in range(2):
        np.testing.assert_allclose(
            covariance[axis], expected_cov * model.output_scale[axis] ** 2
        )
        assert np.linalg.eigvalsh(covariance[axis]).min() > 0
    _, marginal = model.predict(query)
    np.testing.assert_allclose(marginal.T, np.diagonal(covariance, axis1=1, axis2=2))
    np.testing.assert_allclose(model.predict_mean(query), mean)


def test_matches_sklearn_gp_and_kernel_ridge():
    sklearn_gp = pytest.importorskip("sklearn.gaussian_process")
    kernels = pytest.importorskip("sklearn.gaussian_process.kernels")
    krr = pytest.importorskip("sklearn.kernel_ridge")
    x, y, groups = fixture()
    model = GaussianResidual.fit(x, y, groups, length_scale=0.7, noise_variance=0.05)
    z = (x[:4] + 0.1 - model.location) / model.scale
    kernel = kernels.Matern(length_scale=0.7 * np.sqrt(3), nu=1.5)
    reference = sklearn_gp.GaussianProcessRegressor(
        kernel=kernel, alpha=0.05, optimizer=None
    )
    reference.fit(model.support, y / model.output_scale)
    mean, _ = model.predict(x[:4] + 0.1)
    np.testing.assert_allclose(
        mean, reference.predict(z) * model.output_scale, atol=1e-10
    )
    ridge = krr.KernelRidge(alpha=0.05, kernel="precomputed")
    ridge.fit(matern32(model.support, model.support, 0.7), y / model.output_scale)
    np.testing.assert_allclose(
        mean,
        ridge.predict(matern32(z, model.support, 0.7)) * model.output_scale,
        atol=1e-10,
    )


def test_group_balanced_support_is_outcome_blind():
    groups = np.array(["a"] * 100 + ["b"] * 7)
    indices = grouped_support(groups, 4)
    assert sum(groups[indices] == "a") == sum(groups[indices] == "b") == 4
    assert list(indices[:4]) == [0, 33, 66, 99]


def test_fit_only_normalization_and_duplicate_features():
    x, y, groups = fixture()
    x[:, 2] = 4
    x[1] = x[0]
    model = GaussianResidual.fit(x, y, groups, length_scale=1, noise_variance=0.1)
    before = model.location.copy()
    model.predict_mean(x + 1000)
    np.testing.assert_array_equal(model.location, before)
    assert model.scale[2] == 1


def test_permutation_equivariance_when_all_rows_retained():
    x, y, groups = fixture()
    order = np.random.default_rng(11).permutation(len(x))
    a = GaussianResidual.fit(x, y, groups, length_scale=1, noise_variance=0.1)
    b = GaussianResidual.fit(
        x[order], y[order], groups[order], length_scale=1, noise_variance=0.1
    )
    np.testing.assert_allclose(a.predict_mean(x), b.predict_mean(x), atol=1e-11)


def test_observation_noise_is_explicit():
    x, y, groups = fixture()
    model = GaussianResidual.fit(x, y, groups, length_scale=1, noise_variance=0.1)
    _, latent = model.predict(x[:3])
    _, observation = model.predict(x[:3], include_noise=True)
    np.testing.assert_allclose(
        observation - latent,
        np.broadcast_to(0.1 * model.output_scale**2, latent.shape),
    )


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_invalid_noise_and_length_fail(value):
    x, y, groups = fixture()
    with pytest.raises(ValueError):
        GaussianResidual.fit(x, y, groups, length_scale=1, noise_variance=value)
    with pytest.raises(ValueError):
        matern32(x, x, value)


@pytest.mark.parametrize(
    "partitions", [(["a"], ["a"]), (["a", "a"], ["b"]), ([], ["b"])]
)
def test_group_leakage_rejected(partitions):
    with pytest.raises(ValueError):
        require_disjoint_groups(*partitions)


def test_clamped_nodes_and_exact_zero_correction():
    baseline = np.zeros((2, 5, 8, 3), dtype=np.float64)
    correction = np.ones((2, 5, 4, 3))
    frames = np.broadcast_to(np.eye(3), (2, 3, 3))
    assert apply_canonical_correction(baseline, correction, frames, 0) is baseline
    result = apply_canonical_correction(baseline, correction, frames, 0.25)
    np.testing.assert_array_equal(
        result[:, :, [0, 1, -2, -1]], baseline[:, :, [0, 1, -2, -1]]
    )
    np.testing.assert_array_equal(result[:, :, 2:-2], 0.25)
    assert not baseline.any()


def test_query_invalid_inputs_fail():
    x, y, groups = fixture()
    model = GaussianResidual.fit(x, y, groups, length_scale=1, noise_variance=0.1)
    with pytest.raises(ValueError):
        model.predict(np.ones((3, 4)))
    with pytest.raises(ValueError):
        model.predict(np.full((3, 3), np.nan))


def test_nested_splits_keep_recordings_and_holdout_separate():
    from scripts.remote.run_gp_residual_development_v1 import build_splits

    names = [f"{index}.pkl" for index in range(48)]
    for seed in (7, 19, 41):
        for budget in (8, 16, 32):
            train, inner, outer = build_splits(names, seed, budget)
            assert len(train) == budget and len(inner) == len(outer) == 8
            assert not set(train) & set(inner)
            assert not (set(train) | set(inner)) & set(outer)
            np.testing.assert_array_equal(outer, np.arange(40, 48))


def test_source_manifest_rejects_eval_and_overlap():
    from scripts.remote.run_gp_residual_development_v1 import validate_source_manifest

    manifest = {
        "contract": "deform-dlo-source-reproduction-v1",
        "dlo_type": "DLO2",
        "partition": "train",
        "official_eval_read": False,
        "split": {
            "fit": [f"{i}.pkl" for i in range(40)],
            "validation": [f"{i}.pkl" for i in range(40, 48)],
            "source_test": [f"{i}.pkl" for i in range(48, 56)],
        },
    }
    assert len(validate_source_manifest(manifest)[0]) == 40
    manifest["partition"] = "eval"
    with pytest.raises(ValueError):
        validate_source_manifest(manifest)
    manifest["partition"] = "train"
    manifest["split"]["validation"][0] = "0.pkl"
    with pytest.raises(ValueError):
        validate_source_manifest(manifest)


def test_payload_guard_blocks_source_test_and_official_eval(tmp_path):
    import os
    import subprocess
    import sys

    allowed = tmp_path / "fit.pkl"
    forbidden = tmp_path / "source_test.pkl"
    official = tmp_path / "eval" / "result.txt"
    official.parent.mkdir()
    for path in (allowed, forbidden, official):
        path.write_text("sentinel")
    code = f"""
from pathlib import Path
from scripts.remote.run_gp_residual_development_v1 import install_payload_guard
install_payload_guard({{Path({str(allowed)!r})}})
assert Path({str(allowed)!r}).read_text() == 'sentinel'
for name in ({str(forbidden)!r}, {str(official)!r}):
    try:
        Path(name).read_text()
    except PermissionError:
        pass
    else:
        raise AssertionError('forbidden payload opened')
"""
    subprocess.run([sys.executable, "-c", code], env=dict(os.environ), check=True)


def test_selection_deterministic_and_invalid_predictions_fail():
    from scripts.remote.run_gp_residual_development_v1 import select_strength

    truth = np.ones((2, 4, 6, 3))
    assert select_strength([truth * 0, truth, truth], truth) == 1
    with pytest.raises(ValueError):
        select_strength([np.full_like(truth, np.nan)], truth)
