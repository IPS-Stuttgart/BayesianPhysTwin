import numpy as np
import pytest

from bayesian_phystwin.phystwin_spring_field import (
    build_canonical_spring_basis,
    build_canonical_triplane_spring_basis,
)


def _graph():
    vertices = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.07, 0.01, 0.00],
            [0.01, 0.09, 0.02],
            [0.02, 0.03, 0.11],
            [0.10, 0.08, 0.06],
        ],
        dtype=np.float32,
    )
    springs = np.array(
        [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [4, 1]],
        dtype=np.int32,
    )
    return vertices, springs


def test_canonical_basis_is_smooth_separated_and_identity_centered():
    vertices, springs = _graph()

    basis = build_canonical_spring_basis(
        vertices,
        springs,
        num_object_springs=6,
        rank=3,
    )

    assert basis.weights.shape == (7, 4)
    assert basis.object_rank == 3
    assert basis.controller_parameter_index == 3
    np.testing.assert_allclose(np.sum(basis.weights[:6, :3], axis=1), 1.0)
    np.testing.assert_array_equal(basis.weights[:6, 3], 0.0)
    np.testing.assert_array_equal(basis.weights[6], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_array_equal(basis.weights @ np.zeros(4), np.zeros(7))

    coefficients = np.array([0.2, 0.2, 0.2, -0.4])
    np.testing.assert_allclose((basis.weights @ coefficients)[:6], 0.2)
    assert (basis.weights @ coefficients)[6] == pytest.approx(-0.4)


def test_canonical_basis_is_rigid_transform_invariant():
    vertices, springs = _graph()
    rotation = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    transformed = vertices @ rotation.T + np.array([0.4, -0.2, 0.8])

    reference = build_canonical_spring_basis(
        vertices,
        springs,
        num_object_springs=6,
        rank=4,
    )
    candidate = build_canonical_spring_basis(
        transformed,
        springs,
        num_object_springs=6,
        rank=4,
    )

    np.testing.assert_array_equal(
        candidate.center_spring_indices,
        reference.center_spring_indices,
    )
    np.testing.assert_allclose(candidate.weights, reference.weights, atol=1e-6)
    assert candidate.length_scale_m == pytest.approx(
        reference.length_scale_m,
        abs=1e-7,
    )


def test_canonical_basis_caps_rank_at_object_spring_count():
    vertices, springs = _graph()

    basis = build_canonical_spring_basis(
        vertices,
        springs,
        num_object_springs=6,
        rank=20,
        length_scale_multiplier=1.5,
    )

    assert basis.object_rank == 6
    assert basis.weights.shape == (7, 7)
    assert basis.length_scale_m > 0.0


@pytest.mark.parametrize(
    ("rank", "multiplier"),
    [(0, 1.0), (2, 0.0), (2, np.inf)],
)
def test_canonical_basis_rejects_invalid_hyperparameters(rank, multiplier):
    vertices, springs = _graph()

    with pytest.raises(ValueError):
        build_canonical_spring_basis(
            vertices,
            springs,
            num_object_springs=6,
            rank=rank,
            length_scale_multiplier=multiplier,
        )


def test_canonical_triplane_is_sparse_separated_and_identity_centered():
    vertices, springs = _graph()

    basis = build_canonical_triplane_spring_basis(
        vertices,
        springs,
        num_object_springs=6,
        resolution=4,
    )

    assert basis.parameter_indices.shape == (7, 12)
    assert basis.interpolation_weights.shape == (7, 12)
    assert basis.parameter_count == 49
    assert basis.controller_parameter_index == 48
    assert np.all(basis.parameter_indices[:6] < 48)
    np.testing.assert_allclose(
        np.sum(basis.interpolation_weights[:6], axis=1),
        1.0,
        atol=1e-7,
    )
    np.testing.assert_array_equal(basis.parameter_indices[6], np.full(12, 48))
    np.testing.assert_array_equal(
        basis.interpolation_weights[6],
        np.array([1.0] + [0.0] * 11, dtype=np.float32),
    )

    coefficients = np.zeros(basis.parameter_count, dtype=np.float32)
    np.testing.assert_array_equal(
        np.sum(
            coefficients[basis.parameter_indices]
            * basis.interpolation_weights,
            axis=1,
        ),
        np.zeros(7, dtype=np.float32),
    )
    coefficients[:48] = 0.2
    coefficients[48] = -0.4
    field = np.sum(
        coefficients[basis.parameter_indices] * basis.interpolation_weights,
        axis=1,
    )
    np.testing.assert_allclose(field[:6], 0.2, atol=1e-7)
    assert field[6] == pytest.approx(-0.4)


def test_canonical_triplane_is_rigid_transform_invariant():
    vertices, springs = _graph()
    rotation = np.array(
        [
            [0.36, -0.80, 0.48],
            [0.80, 0.52, 0.30],
            [-0.48, 0.30, 0.82],
        ],
        dtype=np.float64,
    )
    u, _, vh = np.linalg.svd(rotation)
    rotation = u @ vh
    transformed = vertices @ rotation.T + np.array([0.4, -0.2, 0.8])

    reference = build_canonical_triplane_spring_basis(
        vertices,
        springs,
        num_object_springs=6,
        resolution=5,
    )
    candidate = build_canonical_triplane_spring_basis(
        transformed,
        springs,
        num_object_springs=6,
        resolution=5,
    )

    np.testing.assert_array_equal(
        candidate.parameter_indices,
        reference.parameter_indices,
    )
    np.testing.assert_allclose(
        candidate.interpolation_weights,
        reference.interpolation_weights,
        atol=1e-5,
    )


def test_canonical_triplane_uses_paper_resolution_rule_by_default():
    vertices, springs = _graph()

    basis = build_canonical_triplane_spring_basis(
        vertices,
        springs,
        num_object_springs=6,
    )

    assert basis.resolution == max(2, round(0.85 * np.sqrt(6)))


@pytest.mark.parametrize("resolution", [0, 1])
def test_canonical_triplane_rejects_invalid_resolution(resolution):
    vertices, springs = _graph()

    with pytest.raises(ValueError):
        build_canonical_triplane_spring_basis(
            vertices,
            springs,
            num_object_springs=6,
            resolution=resolution,
        )
