import numpy as np
import pytest

from bayesian_phystwin.phystwin_spring_field import (
    build_canonical_spring_basis,
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
