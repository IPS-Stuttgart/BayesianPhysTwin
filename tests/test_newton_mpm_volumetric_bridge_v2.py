from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.newton_mpm_volumetric_bridge_v2 import (
    MaterialQueryMapV2,
    build_material_query_map,
    compliant_contact_projection,
    read_material_displacements,
    regular_convex_hull_particles,
    transfer_query_contacts_to_material,
)


def _cube() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )


def test_regular_particleization_is_deterministic_and_translation_equivariant() -> None:
    source = _cube()
    first = regular_convex_hull_particles(source, spacing_m=0.25)
    second = regular_convex_hull_particles(source, spacing_m=0.25)
    translated = regular_convex_hull_particles(source + 2.5, spacing_m=0.25)

    assert first.shape == (125, 3)
    assert np.array_equal(first, second)
    assert np.array_equal(translated, first + 2.5)
    assert np.all((first >= 0.0) & (first <= 1.0))


def test_regular_particleization_rejects_flat_cloud_and_particle_explosion() -> None:
    flat = _cube().copy()
    flat[:, 2] = 0.0
    with pytest.raises(ValueError, match="non-degenerate 3D hull"):
        regular_convex_hull_particles(flat, spacing_m=0.25)
    with pytest.raises(ValueError, match="candidate grid is too large"):
        regular_convex_hull_particles(
            _cube(), spacing_m=0.01, maximum_particle_count=10
        )


@pytest.mark.parametrize(
    ("source", "spacing", "maximum", "message"),
    [
        (np.zeros((3, 3)), 0.25, 100, r"N>=4"),
        (np.full((4, 3), "bad"), 0.25, 100, "numeric"),
        (np.full((4, 3), np.nan), 0.25, 100, "finite"),
        (_cube(), 0.0, 100, "spacing_m"),
        (_cube(), True, 100, "spacing_m"),
        (_cube(), 0.25, 0, "maximum_particle_count"),
        (_cube(), 0.25, True, "maximum_particle_count"),
        (_cube(), 2.0, 100, "too large"),
        (_cube(), 0.5, 4, "exceeds maximum_particle_count"),
    ],
)
def test_regular_particleization_validates_inputs(
    source: object,
    spacing: object,
    maximum: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        regular_convex_hull_particles(
            source,
            spacing_m=spacing,  # type: ignore[arg-type]
            maximum_particle_count=maximum,  # type: ignore[arg-type]
        )


def test_query_readout_is_frame_zero_exact_and_translation_equivariant() -> None:
    material = regular_convex_hull_particles(_cube(), spacing_m=0.5)
    queries = np.asarray([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]], dtype=np.float64)
    query_map = build_material_query_map(queries, material, neighbour_count=8)
    translation = np.asarray([0.03, -0.04, 0.05])
    trajectory = np.stack([material, material + translation])

    readout = read_material_displacements(trajectory, material, queries, query_map)

    assert np.array_equal(readout[0], queries)
    assert np.allclose(readout[1], queries + translation, rtol=0.0, atol=1.0e-15)
    assert np.allclose(query_map.weights.sum(axis=1), 1.0)


def test_query_map_uses_one_exact_material_identity_without_duplicate_evidence() -> (
    None
):
    material = regular_convex_hull_particles(_cube(), spacing_m=0.5)
    query_map = build_material_query_map(material[[4]], material, neighbour_count=8)

    assert np.count_nonzero(query_map.weights[0]) == 1
    selected = query_map.indices[0, np.argmax(query_map.weights[0])]
    assert np.array_equal(material[selected], material[4])


@pytest.mark.parametrize(
    ("queries", "material", "neighbours", "power", "minimum", "message"),
    [
        (np.empty((0, 3)), _cube(), 1, 2.0, 1.0e-9, r"N>=1"),
        (np.full((1, 3), "bad"), _cube(), 1, 2.0, 1.0e-9, "numeric"),
        (np.full((1, 3), np.nan), _cube(), 1, 2.0, 1.0e-9, "finite"),
        (_cube()[:1], np.empty((0, 3)), 1, 2.0, 1.0e-9, r"N>=1"),
        (_cube()[:1], _cube(), 0, 2.0, 1.0e-9, "neighbour_count"),
        (_cube()[:1], _cube(), 1, 0.0, 1.0e-9, "inverse_distance_power"),
        (_cube()[:1], _cube(), 1, 2.0, 0.0, "minimum_distance_m"),
    ],
)
def test_query_map_validates_inputs(
    queries: object,
    material: object,
    neighbours: object,
    power: object,
    minimum: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_material_query_map(
            queries,
            material,
            neighbour_count=neighbours,  # type: ignore[arg-type]
            inverse_distance_power=power,  # type: ignore[arg-type]
            minimum_distance_m=minimum,  # type: ignore[arg-type]
        )


def test_readout_rejects_unnormalized_or_out_of_range_map() -> None:
    material = _cube()
    queries = material[:2]
    trajectory = np.stack([material, material])
    bad_weights = MaterialQueryMapV2(
        indices=np.asarray([[0, 1], [1, 2]], dtype=np.int64),
        weights=np.full((2, 2), 0.2),
        maximum_distance_m=1.0,
    )
    with pytest.raises(ValueError, match="sum to one"):
        read_material_displacements(trajectory, material, queries, bad_weights)

    bad_indices = MaterialQueryMapV2(
        indices=np.asarray([[0, 99], [1, 2]], dtype=np.int64),
        weights=np.full((2, 2), 0.5),
        maximum_distance_m=1.0,
    )
    with pytest.raises(ValueError, match="outside the material"):
        read_material_displacements(trajectory, material, queries, bad_indices)


@pytest.mark.parametrize(
    ("trajectory", "material", "queries", "query_map", "message"),
    [
        (np.zeros((2, 3)), _cube(), _cube()[:2], None, r"shape \(T, P, 3\)"),
        (
            np.full((2, 8, 3), "bad"),
            _cube(),
            _cube()[:2],
            None,
            "numeric",
        ),
        (
            np.zeros((2, 7, 3)),
            _cube(),
            _cube()[:2],
            None,
            "counts differ",
        ),
        (
            np.full((2, 8, 3), np.nan),
            _cube(),
            _cube()[:2],
            None,
            "finite",
        ),
    ],
)
def test_readout_validates_trajectory_before_map(
    trajectory: object,
    material: object,
    queries: object,
    query_map: object,
    message: str,
) -> None:
    if query_map is None:
        query_map = MaterialQueryMapV2(
            indices=np.asarray([[0], [1]], dtype=np.int64),
            weights=np.ones((2, 1)),
            maximum_distance_m=0.0,
        )
    with pytest.raises(ValueError, match=message):
        read_material_displacements(
            trajectory,
            material,
            queries,
            query_map,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("indices", "weights", "message"),
    [
        (np.asarray([[0]], dtype=np.int64), np.ones((1, 1)), "indices"),
        (np.asarray([[0], [1]], dtype=np.int64), np.ones((2, 2)), "weights"),
        (np.asarray([[0.0], [1.0]]), np.ones((2, 1)), "outside"),
        (np.asarray([[0], [1]], dtype=np.int64), np.full((2, 1), np.nan), "finite"),
        (np.asarray([[0], [1]], dtype=np.int64), -np.ones((2, 1)), "nonnegative"),
    ],
)
def test_readout_validates_query_map_branches(
    indices: np.ndarray,
    weights: np.ndarray,
    message: str,
) -> None:
    material = _cube()
    trajectory = np.stack([material, material])
    query_map = MaterialQueryMapV2(
        indices=indices,
        weights=weights,
        maximum_distance_m=0.0,
    )
    with pytest.raises(ValueError, match=message):
        read_material_displacements(
            trajectory,
            material,
            material[:2],
            query_map,
        )


def test_contact_transfer_preserves_controller_mixture_and_sparse_support() -> None:
    query_map = MaterialQueryMapV2(
        indices=np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int64),
        weights=np.asarray([[0.75, 0.25], [0.5, 0.5], [0.25, 0.75]]),
        maximum_distance_m=0.01,
    )
    contacts = transfer_query_contacts_to_material(
        np.asarray([0, 2], dtype=np.int64),
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        query_map,
        material_particle_count=5,
    )

    assert np.array_equal(contacts.material_indices, np.asarray([0, 1, 2, 3]))
    assert np.allclose(contacts.controller_weights.sum(axis=1), 1.0)
    assert np.array_equal(contacts.controller_weights[0], np.asarray([1.0, 0.0]))
    assert np.array_equal(contacts.controller_weights[-1], np.asarray([0.0, 1.0]))


@pytest.mark.parametrize(
    ("attached", "controller", "query_map", "material_count", "message"),
    [
        (np.asarray([], dtype=np.int64), np.empty((0, 1)), None, 4, "nonempty"),
        (np.asarray([0]), np.ones((2, 1)), None, 4, "match attached"),
        (np.asarray([0]), np.full((1, 1), "bad"), None, 4, "numeric"),
        (np.asarray([0]), np.full((1, 1), np.nan), None, 4, "finite"),
        (np.asarray([0]), -np.ones((1, 1)), None, 4, "nonnegative"),
        (np.asarray([0]), np.full((1, 1), 0.5), None, 4, "sum to one"),
        (np.asarray([3]), np.ones((1, 1)), None, 4, "outside the query map"),
        (
            np.asarray([0]),
            np.ones((1, 1)),
            MaterialQueryMapV2(
                indices=np.asarray([[9]], dtype=np.int64),
                weights=np.ones((1, 1)),
                maximum_distance_m=0.0,
            ),
            4,
            "unavailable material",
        ),
        (
            np.asarray([0]),
            np.ones((1, 1)),
            MaterialQueryMapV2(
                indices=np.asarray([[0]], dtype=np.int64),
                weights=np.zeros((1, 1)),
                maximum_distance_m=0.0,
            ),
            4,
            "produced no material",
        ),
    ],
)
def test_contact_transfer_validates_inputs(
    attached: object,
    controller: object,
    query_map: MaterialQueryMapV2 | None,
    material_count: object,
    message: str,
) -> None:
    if query_map is None:
        query_map = MaterialQueryMapV2(
            indices=np.asarray([[0], [1]], dtype=np.int64),
            weights=np.ones((2, 1)),
            maximum_distance_m=0.0,
        )
    with pytest.raises(ValueError, match=message):
        transfer_query_contacts_to_material(
            attached,
            controller,
            query_map,
            material_particle_count=material_count,  # type: ignore[arg-type]
        )


def test_compliant_projection_has_exact_endpoints_and_finite_slip() -> None:
    current_position = np.zeros((2, 3))
    current_velocity = np.zeros((2, 3))
    target_position = np.full((2, 3), 2.0)
    target_velocity = np.full((2, 3), 4.0)

    unchanged = compliant_contact_projection(
        current_position,
        current_velocity,
        target_position,
        target_velocity,
        coupling=0.0,
    )
    partial = compliant_contact_projection(
        current_position,
        current_velocity,
        target_position,
        target_velocity,
        coupling=0.25,
    )
    exact = compliant_contact_projection(
        current_position,
        current_velocity,
        target_position,
        target_velocity,
        coupling=1.0,
    )

    assert np.array_equal(unchanged[0], current_position)
    assert np.array_equal(unchanged[1], current_velocity)
    assert np.array_equal(partial[0], np.full((2, 3), 0.5))
    assert np.array_equal(partial[1], np.full((2, 3), 1.0))
    assert np.array_equal(exact[0], target_position)
    assert np.array_equal(exact[1], target_velocity)


def test_compliant_projection_rejects_shape_mismatch() -> None:
    one = np.zeros((1, 3))
    two = np.zeros((2, 3))
    with pytest.raises(ValueError, match="identical shapes"):
        compliant_contact_projection(one, two, one, one, coupling=0.5)


@pytest.mark.parametrize("coupling", [-0.1, 1.1, np.nan, True])
def test_compliant_projection_rejects_invalid_coupling(coupling: object) -> None:
    points = np.zeros((1, 3))
    with pytest.raises(ValueError, match="coupling"):
        compliant_contact_projection(points, points, points, points, coupling=coupling)  # type: ignore[arg-type]
