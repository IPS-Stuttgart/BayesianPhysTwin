import inspect

import numpy as np
import pytest

import bayesian_phystwin.deform360_frozen_query_field as query_field
from bayesian_phystwin.deform360_frozen_query_field import (
    FrameZeroQuerySet,
    FrozenFieldConfig,
    FrozenFieldGeometry,
    build_frozen_nodal_field,
    map_assimilation_centers_to_queries,
    query_frozen_nodal_field,
)


def _points() -> np.ndarray:
    return np.asarray(
        [
            [0.00, 0.00, 0.00],
            [0.10, 0.00, 0.00],
            [0.00, 0.10, 0.00],
            [0.10, 0.10, 0.00],
        ],
        dtype=np.float32,
    )


def _trajectories(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    primary = np.repeat(points[None], 4, axis=0)
    comparator = primary.copy()
    anchor_scale = np.arange(1, len(points) + 1, dtype=np.float32)
    for frame in range(1, len(primary)):
        primary[frame, :, 0] += np.float32(0.01 * frame) * anchor_scale
        primary[frame, :, 1] -= np.float32(0.002 * frame) * anchor_scale
        comparator[frame, :, 1] += np.float32(0.005 * frame) * anchor_scale
    return primary, comparator


def _nearest_config(*, support: float = 0.08) -> FrozenFieldConfig:
    return FrozenFieldConfig(
        operator_id="nearest-v1",
        maximum_support_distance_m=support,
        unsupported_query_policy="emit-prediction-and-mask-v1",
    )


def _gaussian_config(
    *, support: float = 0.08, neighbors: int = 3, length_scale: float = 0.05
) -> FrozenFieldConfig:
    return FrozenFieldConfig(
        operator_id="gaussian-knn-normalized-v1",
        maximum_support_distance_m=support,
        unsupported_query_policy="emit-prediction-and-mask-v1",
        gaussian_neighbor_count=neighbors,
        gaussian_length_scale_m=length_scale,
    )


def _field(config: FrozenFieldConfig):
    points = _points()
    primary, comparator = _trajectories(points)
    return build_frozen_nodal_field(
        points,
        primary,
        comparator,
        np.asarray([0, 2], dtype=np.int64),
        config=config,
    )


def test_configuration_and_builder_have_no_implicit_operator_default() -> None:
    signature = inspect.signature(build_frozen_nodal_field)
    assert signature.parameters["config"].default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        FrozenFieldConfig()  # type: ignore[call-arg]

    points = _points()
    primary, comparator = _trajectories(points)
    with pytest.raises(TypeError):
        build_frozen_nodal_field(
            points,
            primary,
            comparator,
            np.asarray([0], dtype=np.int64),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "operator_id": "other",
                "maximum_support_distance_m": 0.1,
                "unsupported_query_policy": "emit-prediction-and-mask-v1",
            },
            "unsupported",
        ),
        (
            {
                "operator_id": "nearest-v1",
                "maximum_support_distance_m": 0.0,
                "unsupported_query_policy": "emit-prediction-and-mask-v1",
            },
            "numerical minimum",
        ),
        (
            {
                "operator_id": "nearest-v1",
                "maximum_support_distance_m": 0.1,
                "unsupported_query_policy": "emit-prediction-and-mask-v1",
                "gaussian_neighbor_count": 2,
            },
            "does not accept",
        ),
        (
            {
                "operator_id": "gaussian-knn-normalized-v1",
                "maximum_support_distance_m": 0.1,
                "unsupported_query_policy": "emit-prediction-and-mask-v1",
                "gaussian_length_scale_m": 0.1,
            },
            "neighbor count",
        ),
        (
            {
                "operator_id": "gaussian-knn-normalized-v1",
                "maximum_support_distance_m": 0.1,
                "unsupported_query_policy": "emit-prediction-and-mask-v1",
                "gaussian_neighbor_count": 2,
                "gaussian_length_scale_m": -0.1,
            },
            "numerical minimum",
        ),
        (
            {
                "operator_id": "nearest-v1",
                "maximum_support_distance_m": 0.1,
                "unsupported_query_policy": "score-unsupported",
            },
            "unsupported-query policy",
        ),
    ],
)
def test_configuration_rejects_unfrozen_or_invalid_choices(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        FrozenFieldConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("config", [_nearest_config(), _gaussian_config()])
def test_exact_anchor_queries_return_both_nodal_trajectories_bit_exactly(
    config: FrozenFieldConfig,
) -> None:
    field = _field(config)
    source_indices = np.asarray([2, 0, 3], dtype=np.int64)
    queries = FrameZeroQuerySet(
        identity_ids=np.asarray([90, 12, 47], dtype=np.int64),
        positions_m=field.geometry.anchor_positions_m[source_indices],
    )

    result = query_frozen_nodal_field(field, queries)

    np.testing.assert_array_equal(
        result.primary_prediction_m,
        field.primary_nodal_trajectory_m[:, source_indices],
    )
    np.testing.assert_array_equal(
        result.comparator_prediction_m,
        field.comparator_nodal_trajectory_m[:, source_indices],
    )
    np.testing.assert_array_equal(result.exact_anchor_mask, True)
    np.testing.assert_array_equal(
        result.nearest_anchor_ids, field.geometry.anchor_ids[source_indices]
    )
    assert not result.primary_prediction_m.flags.writeable
    assert not result.neighbor_weights.flags.writeable


def test_distance_then_anchor_id_breaks_exact_nearest_neighbor_ties() -> None:
    points = np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    primary, comparator = _trajectories(points)
    query = FrameZeroQuerySet(
        identity_ids=np.asarray([7], dtype=np.int64),
        positions_m=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
    )
    nearest = build_frozen_nodal_field(
        points,
        primary,
        comparator,
        np.asarray([0], dtype=np.int64),
        config=_nearest_config(support=2.0),
    )
    gaussian = build_frozen_nodal_field(
        points,
        primary,
        comparator,
        np.asarray([0], dtype=np.int64),
        config=_gaussian_config(support=2.0, neighbors=2, length_scale=1.0),
    )

    nearest_result = query_frozen_nodal_field(nearest, query)
    gaussian_result = query_frozen_nodal_field(gaussian, query)

    assert nearest_result.nearest_anchor_ids.tolist() == [0]
    assert gaussian_result.neighbor_anchor_ids.tolist() == [[0, 1]]
    np.testing.assert_array_equal(gaussian_result.neighbor_weights, [[0.5, 0.5]])


def test_gaussian_knn_uses_fixed_shared_weights_for_both_arms() -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    primary = np.repeat(points[None], 2, axis=0)
    comparator = primary.copy()
    primary[1, :, 1] += np.asarray([1.0, 3.0], dtype=np.float32)
    comparator[1, :, 2] += np.asarray([4.0, 8.0], dtype=np.float32)
    field = build_frozen_nodal_field(
        points,
        primary,
        comparator,
        np.asarray([0], dtype=np.int64),
        config=_gaussian_config(support=3.0, neighbors=2, length_scale=1.0),
    )
    query = FrameZeroQuerySet(
        identity_ids=np.asarray([100], dtype=np.int64),
        positions_m=np.asarray([[0.5, 0.0, 0.0]], dtype=np.float32),
    )

    result = query_frozen_nodal_field(field, query)

    raw = np.exp(-0.5 * np.square(np.asarray([0.5, 1.5])))
    expected_weights = raw / np.sum(raw)
    np.testing.assert_allclose(result.neighbor_weights[0], expected_weights, atol=1e-15)
    assert result.primary_prediction_m[1, 0, 1] == pytest.approx(
        float(expected_weights @ np.asarray([1.0, 3.0])), abs=1e-7
    )
    assert result.comparator_prediction_m[1, 0, 2] == pytest.approx(
        float(expected_weights @ np.asarray([4.0, 8.0])), abs=1e-7
    )


def test_gaussian_weights_remain_finite_at_the_minimum_locked_scale() -> None:
    points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    primary, comparator = _trajectories(points)
    field = build_frozen_nodal_field(
        points,
        primary,
        comparator,
        np.asarray([0], dtype=np.int64),
        config=_gaussian_config(
            support=3.0,
            neighbors=3,
            length_scale=1e-12,
        ),
    )
    query = FrameZeroQuerySet(
        identity_ids=np.asarray([1], dtype=np.int64),
        positions_m=np.asarray([[0.25, 0.0, 0.0]], dtype=np.float32),
    )

    result = query_frozen_nodal_field(field, query)

    np.testing.assert_array_equal(result.neighbor_weights, [[1.0, 0.0, 0.0]])
    assert np.all(np.isfinite(result.primary_prediction_m))


@pytest.mark.parametrize("config", [_nearest_config(), _gaussian_config()])
def test_queries_are_permutation_batch_and_cardinality_invariant(
    config: FrozenFieldConfig,
) -> None:
    field = _field(config)
    positions = np.asarray(
        [
            [0.02, 0.01, 0.00],
            [0.08, 0.03, 0.00],
            [0.03, 0.09, 0.00],
            [0.02, 0.01, 0.00],
        ],
        dtype=np.float32,
    )
    identities = np.asarray([41, 5, 83, 99], dtype=np.int64)
    full = query_frozen_nodal_field(
        field, FrameZeroQuerySet(identity_ids=identities, positions_m=positions)
    )
    permutation = np.asarray([2, 0, 3, 1], dtype=np.int64)
    permuted = query_frozen_nodal_field(
        field,
        FrameZeroQuerySet(
            identity_ids=identities[permutation],
            positions_m=positions[permutation],
        ),
    )
    inverse = np.argsort(permutation)

    np.testing.assert_array_equal(
        permuted.primary_prediction_m[:, inverse], full.primary_prediction_m
    )
    np.testing.assert_array_equal(
        permuted.comparator_prediction_m[:, inverse], full.comparator_prediction_m
    )
    np.testing.assert_array_equal(
        permuted.neighbor_anchor_ids[inverse], full.neighbor_anchor_ids
    )
    np.testing.assert_array_equal(
        permuted.neighbor_weights[inverse], full.neighbor_weights
    )
    np.testing.assert_array_equal(
        full.primary_prediction_m[:, 0], full.primary_prediction_m[:, 3]
    )
    for index in range(len(positions)):
        alone = query_frozen_nodal_field(
            field,
            FrameZeroQuerySet(
                identity_ids=identities[index : index + 1],
                positions_m=positions[index : index + 1],
            ),
        )
        np.testing.assert_array_equal(
            alone.primary_prediction_m[:, 0], full.primary_prediction_m[:, index]
        )
        np.testing.assert_array_equal(
            alone.neighbor_weights[0], full.neighbor_weights[index]
        )


def test_shared_operator_is_arm_symmetric() -> None:
    base = _field(_gaussian_config())
    equal = build_frozen_nodal_field(
        base.geometry.anchor_positions_m,
        base.primary_nodal_trajectory_m,
        base.primary_nodal_trajectory_m,
        base.geometry.assimilation_anchor_ids,
        config=base.config,
    )
    swapped = build_frozen_nodal_field(
        base.geometry.anchor_positions_m,
        base.comparator_nodal_trajectory_m,
        base.primary_nodal_trajectory_m,
        base.geometry.assimilation_anchor_ids,
        config=base.config,
    )
    queries = FrameZeroQuerySet(
        identity_ids=np.asarray([1, 2], dtype=np.int64),
        positions_m=np.asarray(
            [[0.025, 0.035, 0.0], [0.075, 0.065, 0.0]], dtype=np.float32
        ),
    )

    base_result = query_frozen_nodal_field(base, queries)
    equal_result = query_frozen_nodal_field(equal, queries)
    swapped_result = query_frozen_nodal_field(swapped, queries)

    np.testing.assert_array_equal(
        equal_result.primary_prediction_m, equal_result.comparator_prediction_m
    )
    np.testing.assert_array_equal(
        swapped_result.primary_prediction_m, base_result.comparator_prediction_m
    )
    np.testing.assert_array_equal(
        swapped_result.comparator_prediction_m, base_result.primary_prediction_m
    )
    np.testing.assert_array_equal(
        swapped_result.neighbor_weights, base_result.neighbor_weights
    )
    np.testing.assert_array_equal(
        swapped_result.supported_identity_mask, base_result.supported_identity_mask
    )


def test_support_distances_are_explicit_and_boundary_is_inclusive() -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    primary, comparator = _trajectories(points)
    field = build_frozen_nodal_field(
        points,
        primary,
        comparator,
        np.asarray([0], dtype=np.int64),
        config=_gaussian_config(support=0.5, neighbors=2, length_scale=1.0),
    )
    queries = FrameZeroQuerySet(
        identity_ids=np.asarray([1, 2, 3], dtype=np.int64),
        positions_m=np.asarray(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5001, 0.0, 0.0]],
            dtype=np.float32,
        ),
    )

    result = query_frozen_nodal_field(field, queries)

    np.testing.assert_array_equal(result.supported_identity_mask, [True, True, False])
    np.testing.assert_allclose(
        result.nearest_anchor_distance_m, [0.0, 0.5, 0.5001000165939331]
    )
    np.testing.assert_allclose(
        result.kth_anchor_distance_m, [2.0, 1.5, 1.499899983406067]
    )
    assert np.all(np.isfinite(result.primary_prediction_m[:, 2]))


def test_field_and_query_validation_reject_ambiguous_or_future_shaped_inputs() -> None:
    points = _points()
    primary, comparator = _trajectories(points)
    duplicate = points.copy()
    duplicate[1] = duplicate[0]
    duplicate_primary, duplicate_comparator = _trajectories(duplicate)
    with pytest.raises(ValueError, match="positions must be unique"):
        build_frozen_nodal_field(
            duplicate,
            duplicate_primary,
            duplicate_comparator,
            np.asarray([0], dtype=np.int64),
            config=_nearest_config(),
        )

    changed = primary.copy()
    changed[0, 0, 0] += np.float32(0.01)
    with pytest.raises(ValueError, match="frame zero"):
        build_frozen_nodal_field(
            points,
            changed,
            comparator,
            np.asarray([0], dtype=np.int64),
            config=_nearest_config(),
        )
    with pytest.raises(ValueError, match="exceeds the anchor count"):
        build_frozen_nodal_field(
            points,
            primary,
            comparator,
            np.asarray([0], dtype=np.int64),
            config=_gaussian_config(neighbors=5),
        )
    with pytest.raises(ValueError, match="dtype float32"):
        FrameZeroQuerySet(
            identity_ids=np.asarray([0], dtype=np.int64),
            positions_m=np.zeros((1, 3), dtype=np.float64),
        )
    with pytest.raises(ValueError, match="rank 2"):
        FrameZeroQuerySet(
            identity_ids=np.asarray([0, 1], dtype=np.int64),
            positions_m=np.zeros((2, 4, 3), dtype=np.float32),
        )


def _center_geometry() -> FrozenFieldGeometry:
    return FrozenFieldGeometry(
        anchor_ids=np.asarray([0, 1, 2], dtype=np.int64),
        anchor_positions_m=np.asarray(
            [[0.012, 0.0, 0.0], [0.000, 0.0, 0.0], [0.100, 0.0, 0.0]],
            dtype=np.float32,
        ),
        assimilation_anchor_ids=np.asarray([0, 1], dtype=np.int64),
    )


def test_center_exclusion_solves_greedy_counterexample_using_only_frame_zero() -> None:
    geometry = _center_geometry()
    queries = FrameZeroQuerySet(
        identity_ids=np.asarray([200, 100], dtype=np.int64),
        positions_m=np.asarray(
            [[0.000, 0.0, 0.0], [0.025, 0.0, 0.0]], dtype=np.float32
        ),
    )

    exclusion = map_assimilation_centers_to_queries(
        geometry, queries, maximum_distance_m=0.015
    )

    np.testing.assert_array_equal(exclusion.assimilation_anchor_ids, [0, 1])
    np.testing.assert_array_equal(exclusion.mapped_query_identity_ids, [100, 200])
    np.testing.assert_array_equal(exclusion.mapped_query_indices, [1, 0])
    np.testing.assert_allclose(exclusion.assignment_distance_m, [0.013, 0.0])
    np.testing.assert_array_equal(exclusion.excluded_query_mask, [True, True])
    assert set(inspect.signature(map_assimilation_centers_to_queries).parameters) == {
        "geometry",
        "queries",
        "maximum_distance_m",
    }


def test_center_exclusion_is_query_order_invariant_and_uses_identity_ties() -> None:
    geometry = FrozenFieldGeometry(
        anchor_ids=np.asarray([0, 1], dtype=np.int64),
        anchor_positions_m=np.asarray(
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float32
        ),
        assimilation_anchor_ids=np.asarray([0], dtype=np.int64),
    )
    queries = FrameZeroQuerySet(
        identity_ids=np.asarray([9, 3], dtype=np.int64),
        positions_m=np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
    )
    reversed_queries = FrameZeroQuerySet(
        identity_ids=queries.identity_ids[::-1],
        positions_m=queries.positions_m[::-1],
    )

    first = map_assimilation_centers_to_queries(
        geometry, queries, maximum_distance_m=2.0
    )
    second = map_assimilation_centers_to_queries(
        geometry, reversed_queries, maximum_distance_m=2.0
    )

    assert first.mapped_query_identity_ids.tolist() == [3]
    np.testing.assert_array_equal(
        first.mapped_query_identity_ids, second.mapped_query_identity_ids
    )


def test_center_exclusion_fails_closed_without_a_full_radius_assignment() -> None:
    geometry = FrozenFieldGeometry(
        anchor_ids=np.asarray([0, 1], dtype=np.int64),
        anchor_positions_m=np.asarray(
            [[0.000, 0.0, 0.0], [0.001, 0.0, 0.0]], dtype=np.float32
        ),
        assimilation_anchor_ids=np.asarray([0, 1], dtype=np.int64),
    )
    queries = FrameZeroQuerySet(
        identity_ids=np.asarray([10, 11], dtype=np.int64),
        positions_m=np.asarray(
            [[0.000, 0.0, 0.0], [1.000, 0.0, 0.0]], dtype=np.float32
        ),
    )

    with pytest.raises(ValueError, match="no collision-free"):
        map_assimilation_centers_to_queries(geometry, queries, maximum_distance_m=0.015)
    with pytest.raises(ValueError, match="too few query identities"):
        map_assimilation_centers_to_queries(
            geometry,
            FrameZeroQuerySet(
                identity_ids=np.asarray([10], dtype=np.int64),
                positions_m=queries.positions_m[:1],
            ),
            maximum_distance_m=0.015,
        )


def test_empty_assimilation_centers_support_offline_fields_but_not_mapping() -> None:
    points = _points()
    primary, comparator = _trajectories(points)
    field = build_frozen_nodal_field(
        points,
        primary,
        comparator,
        np.empty(0, dtype=np.int64),
        config=_nearest_config(),
    )
    queries = FrameZeroQuerySet(
        identity_ids=np.asarray([10], dtype=np.int64),
        positions_m=np.asarray([[0.02, 0.01, 0.0]], dtype=np.float32),
    )

    result = query_frozen_nodal_field(field, queries)

    assert len(result.identity_ids) == 1
    assert field.geometry.assimilation_anchor_ids.tolist() == []
    with pytest.raises(ValueError, match="at least one assimilation center"):
        map_assimilation_centers_to_queries(
            field.geometry,
            queries,
            maximum_distance_m=0.1,
        )


def test_core_module_has_no_scipy_or_target_scoring_dependency() -> None:
    source = inspect.getsource(query_field)
    assert "scipy" not in source.lower()
    assert "object_visibilities" not in source
    assert "object_motions_valid" not in source
    assert "score_" not in source
