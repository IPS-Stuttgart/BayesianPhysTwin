from __future__ import annotations

import numpy as np

from bayesian_phystwin.photometric_graph_update import (
    PhotometricGraphConfig,
    select_photometric_graph_update,
)


def _synthetic(
    *,
    camera_count: int = 2,
    frame_count: int = 6,
    size: int = 16,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    baseline = rng.uniform(0.15, 0.85, size=(frame_count, camera_count, size, size, 3))
    jacobian = rng.normal(
        scale=0.035,
        size=(frame_count, camera_count, size, size, 3, 2),
    )
    true_weights = np.array([0.55, -0.35])
    observed = baseline + np.einsum("fchwrp,p->fchwr", jacobian, true_weights)
    for frame in range(frame_count):
        for camera in range(camera_count):
            gain = 0.88 + 0.03 * frame + 0.02 * camera
            offset = np.array([0.025, -0.01, 0.015])
            observed[frame, camera] = gain * observed[frame, camera] + offset
    observed += rng.normal(scale=2e-4, size=observed.shape)
    mask = np.ones(observed.shape[:-1], dtype=bool)
    return observed, baseline, jacobian, mask, true_weights


def _config(**changes: object) -> PhotometricGraphConfig:
    values = {
        "fit_frame_count": 4,
        "correlation_block_size": 8,
        "state_ridge": 1e-4,
        "minimum_fit_groups": 8,
        "minimum_validation_groups": 4,
        "minimum_validation_improvement_fraction": 0.01,
        "minimum_validation_improvement_absolute": 1e-5,
    }
    values.update(changes)
    return PhotometricGraphConfig(**values)


def test_recovers_state_despite_camera_exposure_nuisance() -> None:
    observed, baseline, jacobian, mask, truth = _synthetic()
    result = select_photometric_graph_update(
        observed,
        baseline,
        jacobian,
        mask,
        config=_config(),
    )
    assert result.accepted
    assert np.allclose(result.state_weights, truth, atol=0.04)
    assert result.diagnostics["validation_improvement_fraction"] > 0.5


def test_zero_innovation_returns_exact_fallback() -> None:
    _, baseline, jacobian, mask, _ = _synthetic()
    result = select_photometric_graph_update(
        baseline,
        baseline,
        jacobian,
        mask,
        config=_config(),
    )
    assert not result.accepted
    assert np.array_equal(result.state_weights, np.zeros(2))


def test_duplicate_camera_does_not_create_extra_information() -> None:
    observed, baseline, jacobian, mask, _ = _synthetic(camera_count=1)
    one = select_photometric_graph_update(
        observed,
        baseline,
        jacobian,
        mask,
        config=_config(),
    )
    duplicated = select_photometric_graph_update(
        np.repeat(observed, 2, axis=1),
        np.repeat(baseline, 2, axis=1),
        np.repeat(jacobian, 2, axis=1),
        np.repeat(mask, 2, axis=1),
        config=_config(),
    )
    assert one.accepted and duplicated.accepted
    assert np.allclose(one.state_weights, duplicated.state_weights, atol=1e-5)
    assert np.trace(duplicated.posterior_covariance) >= (
        0.9999 * np.trace(one.posterior_covariance)
    )


def test_dense_pixel_duplication_does_not_increase_confidence() -> None:
    observed, baseline, jacobian, mask, _ = _synthetic(size=8)
    base = select_photometric_graph_update(
        observed,
        baseline,
        jacobian,
        mask,
        config=_config(correlation_block_size=8),
    )
    repeated = select_photometric_graph_update(
        np.repeat(np.repeat(observed, 2, axis=2), 2, axis=3),
        np.repeat(np.repeat(baseline, 2, axis=2), 2, axis=3),
        np.repeat(np.repeat(jacobian, 2, axis=2), 2, axis=3),
        np.repeat(np.repeat(mask, 2, axis=2), 2, axis=3),
        config=_config(correlation_block_size=16),
    )
    assert base.accepted and repeated.accepted
    assert np.allclose(base.state_weights, repeated.state_weights, atol=5e-4)
    assert np.allclose(
        base.posterior_covariance,
        repeated.posterior_covariance,
        atol=5e-4,
    )


def test_gross_outlier_block_is_downweighted() -> None:
    observed, baseline, jacobian, mask, truth = _synthetic(size=24)
    observed[:, :, :8, :8, :] += 2.0
    result = select_photometric_graph_update(
        observed,
        baseline,
        jacobian,
        mask,
        config=_config(),
    )
    assert result.accepted
    assert np.linalg.norm(result.state_weights - truth) < 0.15
    assert result.diagnostics["full_prefix_refit"]["minimum_group_weight"] < 0.5


def test_validity_support_is_residual_independent() -> None:
    observed, baseline, jacobian, mask, _ = _synthetic()
    ordinary = select_photometric_graph_update(
        observed,
        baseline,
        jacobian,
        mask,
        config=_config(),
    )
    shifted = select_photometric_graph_update(
        observed + 10.0,
        baseline,
        jacobian,
        mask,
        config=_config(),
    )
    assert ordinary.diagnostics["fit_group_count"] == shifted.diagnostics[
        "fit_group_count"
    ]
    assert ordinary.diagnostics["validation_group_count"] == shifted.diagnostics[
        "validation_group_count"
    ]
