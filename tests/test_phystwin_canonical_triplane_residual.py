from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from bayesian_phystwin.phystwin_canonical_triplane_residual import (
    CANONICAL_TRIPLANE_RESIDUAL_CONTRACT,
    CanonicalTriplaneResidualConfig,
    _build_model,
    _canonical_frame,
    _load_protocol,
    _point_feature_dimension,
    _prepare_episode,
    _rotate_exogenous,
    _triplane_stencil,
)


def test_canonical_frame_is_right_handed_and_rigid_invariant() -> None:
    points = np.array(
        [
            [-3.0, -0.2, 0.1],
            [-1.0, 0.4, -0.3],
            [0.5, -0.5, 0.2],
            [2.0, 0.1, 0.6],
            [3.5, 0.3, -0.4],
        ]
    )
    angle = 0.63
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    center, basis, scale = _canonical_frame(points)
    transformed = points @ rotation.T + np.array([7.0, -2.0, 4.0])
    other_center, other_basis, other_scale = _canonical_frame(transformed)
    canonical = ((points - center) @ basis) / scale
    other = ((transformed - other_center) @ other_basis) / other_scale
    np.testing.assert_allclose(canonical, other, atol=1.0e-12)
    assert np.linalg.det(basis) == pytest.approx(1.0)
    assert np.linalg.det(other_basis) == pytest.approx(1.0)


def test_triplane_stencil_is_bounded_and_partitions_unity() -> None:
    coordinates = np.array(
        [[-1.0, -1.0, -1.0], [0.25, -0.5, 0.75], [1.0, 1.0, 1.0]]
    )
    indices, weights = _triplane_stencil(coordinates, 8)
    assert indices.shape == (3, 3, 4)
    assert weights.shape == (3, 3, 4)
    assert np.min(indices) >= 0
    assert np.max(indices) < 64
    np.testing.assert_allclose(np.sum(weights, axis=2), 1.0, atol=1.0e-7)


def test_exogenous_rotation_leaves_scalar_proximity_unchanged() -> None:
    features = np.arange(50, dtype=float).reshape(2, 25)
    basis = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated = _rotate_exogenous(features, basis)
    np.testing.assert_array_equal(rotated[:, 18], features[:, 18])
    for start in (0, 3, 6, 9, 12, 15, 19, 22):
        np.testing.assert_allclose(
            rotated[:, start : start + 3],
            features[:, start : start + 3] @ basis,
        )


def test_zero_initialized_triplane_model_predicts_zero_velocity() -> None:
    torch = pytest.importorskip("torch")
    config = CanonicalTriplaneResidualConfig(
        grid_resolution=8,
        plane_channels=4,
        point_hidden_dim=8,
        decoder_hidden_dim=12,
        latent_dim=3,
    )
    model = _build_model(torch, config)
    count = 7
    coordinates = np.linspace(-1.0, 1.0, count * 3).reshape(count, 3)
    indices, weights = _triplane_stencil(coordinates, config.grid_resolution)
    output = model(
        torch.randn(count, 3),
        torch.randn(count, 3),
        torch.randn(count, 25),
        torch.as_tensor(indices),
        torch.as_tensor(weights),
        torch.zeros(config.latent_dim),
    )
    assert _point_feature_dimension() == 31
    torch.testing.assert_close(output, torch.zeros_like(output))


def test_prefix_preparation_is_invariant_to_future_observation_mutation() -> None:
    frame_count = 8
    point_count = 5
    reference = np.array(
        [
            [-2.0, 0.0, 0.0],
            [-1.0, 0.2, 0.1],
            [0.0, -0.3, 0.2],
            [1.0, 0.1, -0.2],
            [2.5, 0.0, 0.3],
        ]
    )
    baseline = np.repeat(reference[None], frame_count, axis=0)
    baseline += np.arange(frame_count)[:, None, None] * np.array([0.01, 0.0, 0.0])
    residual = np.zeros_like(baseline)
    residual[:4, :4, 1] = np.arange(4)[:, None] * 0.002
    residual[4:, :, :] = 0.4
    valid = np.zeros((frame_count, point_count), dtype=bool)
    valid[:4, :4] = True
    valid[4:] = True
    common = {
        "spec": SimpleNamespace(case="synthetic"),
        "observed": baseline + residual,
        "baseline": baseline,
        "controllers": np.zeros((frame_count, 1, 3), dtype=float),
        "object_scale": 1.0,
        "controller_kernel_fraction": 0.25,
    }
    first_loaded = SimpleNamespace(**common, residual=residual, valid=valid)
    mutated_residual = residual.copy()
    mutated_residual[4:] = -7.0
    mutated_valid = valid.copy()
    mutated_valid[4:, 4] = False
    second_loaded = SimpleNamespace(
        **common, residual=mutated_residual, valid=mutated_valid
    )
    config = CanonicalTriplaneResidualConfig(
        grid_resolution=8, maximum_training_points=5
    )
    first = _prepare_episode(
        first_loaded, config, maximum_points=None, evidence_end=4
    )
    second = _prepare_episode(
        second_loaded, config, maximum_points=None, evidence_end=4
    )
    np.testing.assert_array_equal(first.state, second.state)
    np.testing.assert_array_equal(first.velocity, second.velocity)
    np.testing.assert_array_equal(first.exogenous, second.exogenous)
    assert first.velocity_cap == second.velocity_cap


def test_protocol_requires_disjoint_complete_fold_coverage(tmp_path) -> None:
    payload = {
        "contract": CANONICAL_TRIPLANE_RESIDUAL_CONTRACT,
        "source_cases": ["a", "b", "c"],
        "target_cases": ["target"],
        "source_folds": [
            {"held_out_cases": ["a", "b"]},
            {"held_out_cases": ["c"]},
        ],
        "model": {},
    }
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded, config = _load_protocol(path)
    assert loaded["source_cases"] == ["a", "b", "c"]
    assert config == CanonicalTriplaneResidualConfig()

    payload["target_cases"] = ["a"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="disjoint"):
        _load_protocol(path)
