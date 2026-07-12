from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.structural_artifact import (
    build_rigid_free_graph_basis,
    corrected_rest_geometry,
    identity_structural_twin_correction,
    load_structural_twin_correction,
    write_structural_twin_correction,
)
from bayesian_phystwin.structural_warp import (
    assert_zero_configuration_parity,
    prepare_structural_warp_configuration,
    write_structural_warp_configuration,
)


def _grid():
    x, y = np.meshgrid(np.linspace(-0.1, 0.1, 4), np.linspace(0.0, -0.15, 4))
    positions = np.column_stack((x.reshape(-1), y.reshape(-1), 0.01 * x.reshape(-1)))

    def node(row, column):
        return 4 * row + column

    edges = []
    triangles = []
    for row in range(4):
        for column in range(4):
            if column < 3:
                edges.append((node(row, column), node(row, column + 1)))
            if row < 3:
                edges.append((node(row, column), node(row + 1, column)))
            if row < 3 and column < 3:
                edges.append((node(row, column), node(row + 1, column + 1)))
                triangles.extend(
                    (
                        (node(row, column), node(row + 1, column), node(row + 1, column + 1)),
                        (node(row, column), node(row + 1, column + 1), node(row, column + 1)),
                    )
                )
    springs = np.asarray(edges, dtype=np.int64)
    lengths = np.linalg.norm(
        positions[springs[:, 0]] - positions[springs[:, 1]], axis=1
    )
    return positions, springs, lengths, np.asarray(triangles), np.asarray((0, 3))


def _identity():
    positions, springs, lengths, triangles, support = _grid()
    basis, frequencies, diagnostics = build_rigid_free_graph_basis(
        positions, springs, rank=4, support_node_indices=support
    )
    correction = identity_structural_twin_correction(
        positions,
        springs,
        lengths,
        num_object_springs=len(springs),
        graph_basis=basis,
        graph_frequencies=frequencies,
        session_ids=("session_a",),
        support_node_indices=support,
        surface_triangles=triangles,
        source_checksums={"source": "a" * 64},
    )
    return positions, springs, lengths, support, basis, correction, diagnostics


def test_rigid_free_basis_is_orthonormal_and_anchored():
    positions, _, _, support, basis, _, diagnostics = _identity()
    flat = basis.reshape(-1, basis.shape[2])
    np.testing.assert_allclose(flat.T @ flat, np.eye(4), atol=1e-10)
    np.testing.assert_array_equal(basis[support], 0.0)
    centered = positions - np.mean(positions[np.setdiff1d(np.arange(len(positions)), support)], axis=0)
    rigid = []
    free = np.ones(len(positions), dtype=bool)
    free[support] = False
    for axis in np.eye(3):
        field = np.zeros_like(positions)
        field[free] = axis
        rigid.append(field.reshape(-1))
    for axis in np.eye(3):
        field = np.zeros_like(positions)
        field[free] = np.cross(axis, centered[free])
        rigid.append(field.reshape(-1))
    assert np.max(np.abs(np.asarray(rigid) @ flat)) < 1e-10
    assert diagnostics["maximum_rigid_mode_overlap"] < 1e-10


def test_structural_artifact_roundtrip_and_recomputed_lengths(tmp_path):
    positions, springs, lengths, _, _, identity, _ = _identity()
    correction = replace(
        identity,
        persistent_rest_coefficients=np.asarray((0.001, -0.0005, 0.0004, 0.0)),
        metadata={"identity_artifact": False, "posterior_stage": "deferred"},
    )
    geometry = corrected_rest_geometry(
        correction,
        positions,
        springs,
        lengths,
        num_object_springs=len(springs),
    )
    expected = np.linalg.norm(
        geometry.rest_positions[springs[:, 0]]
        - geometry.rest_positions[springs[:, 1]],
        axis=1,
    )
    np.testing.assert_allclose(geometry.rest_lengths, expected)
    written = write_structural_twin_correction(tmp_path, correction)
    loaded = load_structural_twin_correction(written["manifest_path"])
    assert loaded.artifact_id == correction.artifact_id
    np.testing.assert_array_equal(
        loaded.persistent_rest_coefficients,
        correction.persistent_rest_coefficients,
    )


def test_structural_artifact_rejects_excessive_strain():
    positions, springs, lengths, _, _, identity, _ = _identity()
    correction = replace(
        identity,
        persistent_rest_coefficients=np.asarray((0.2, 0.0, 0.0, 0.0)),
        allowed_edge_strain=0.01,
    )
    with pytest.raises(ValueError, match="allowed edge strain"):
        corrected_rest_geometry(
            correction,
            positions,
            springs,
            lengths,
            num_object_springs=len(springs),
        )


def test_zero_structural_configuration_is_byte_identical(tmp_path):
    positions, springs, lengths, support, _, identity, _ = _identity()
    initial = positions.copy()
    velocity = np.zeros_like(initial)
    controls = np.repeat(initial[support][None], 6, axis=0)
    configuration = prepare_structural_warp_configuration(
        identity,
        positions,
        springs,
        lengths,
        num_object_springs=len(springs),
        session_id="session_a",
        nominal_initial_position_m=initial,
        nominal_initial_velocity_mps=velocity,
        controller_points_m=controls,
    )
    parity = assert_zero_configuration_parity(
        configuration,
        nominal_rest_positions_m=positions,
        nominal_rest_lengths_m=lengths,
        nominal_initial_position_m=initial,
        nominal_initial_velocity_mps=velocity,
        controller_points_m=controls,
    )
    assert parity["passed"] is True
    written = write_structural_warp_configuration(tmp_path, configuration)
    assert written["configuration_id"] == configuration.configuration_id


def test_nonzero_rest_geometry_requires_simulated_equilibrium():
    positions, springs, lengths, support, _, identity, _ = _identity()
    correction = replace(
        identity,
        persistent_rest_coefficients=np.asarray((0.001, 0.0, 0.0, 0.0)),
    )
    with pytest.raises(ValueError, match="simulated equilibrium"):
        prepare_structural_warp_configuration(
            correction,
            positions,
            springs,
            lengths,
            num_object_springs=len(springs),
            session_id="session_a",
            nominal_initial_position_m=positions,
            nominal_initial_velocity_mps=np.zeros_like(positions),
            controller_points_m=np.repeat(positions[support][None], 6, axis=0),
        )
