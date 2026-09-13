from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.decision_capability_atlas_v1 import (
    DECISION_CAPABILITY_ATLAS_CLAIM_BOUNDARY,
    affine_capability_halfspaces,
    affine_decision_capability_atlas,
    capability_polygon_2d,
    polygon_area_2d,
)
from bayesian_phystwin.query_decision_certificate_v1 import (
    query_decision_certificate,
)


def _controlled_family() -> tuple[np.ndarray, ...]:
    displacement = np.array(
        [
            [-1.1, -0.1, 0.7],
            [-0.7, 0.1, 1.1],
            [-1.0, 0.0, 0.6],
            [-0.6, 0.0, 1.0],
        ]
    )
    physical_risk = np.array(
        [
            [0.4, 0.05, 0.8],
            [0.8, 0.05, 0.4],
            [0.5, 0.02, 0.9],
            [0.9, 0.02, 0.5],
        ]
    )
    intercept = np.square(displacement)
    coefficient = np.stack((-2.0 * displacement, physical_risk), axis=2)
    return (
        np.full(4, 0.25),
        np.array([0.5, 0.5]),
        np.array([0, 0, 1, 1]),
        intercept,
        coefficient,
    )


def test_atlas_matches_independent_pointwise_certificates() -> None:
    rng = np.random.default_rng(7)
    hypotheses = 7
    actions = 4
    dimension = 3
    tasks = rng.normal(size=(19, dimension))
    intercept = rng.normal(size=(hypotheses, actions))
    coefficient = rng.normal(size=(hypotheses, actions, dimension))
    prior = np.array([0.1, 0.0, 0.15, 0.25, 0.2, 0.1, 0.2])
    classes = np.array([0, 0, 1, 1, 2, 2, 2])
    quotient = np.array([0.25, 0.35, 0.4])

    atlas = affine_decision_capability_atlas(
        prior,
        quotient,
        classes,
        intercept,
        coefficient,
        tasks,
        regret_tolerance=0.3,
        task_batch_size=5,
    )

    for task_index, task in enumerate(tasks):
        losses = intercept + np.tensordot(coefficient, task, axes=(2, 0))
        certificate = query_decision_certificate(
            prior,
            quotient,
            classes,
            losses,
            regret_tolerance=0.3,
        )
        np.testing.assert_allclose(
            atlas.pairwise_worst_case_loss_gap[task_index],
            certificate.pairwise_worst_case_loss_gap,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            atlas.worst_case_regret[task_index],
            certificate.worst_case_regret,
            atol=1e-12,
        )
        assert atlas.minimax_action_index[task_index] == (
            certificate.minimax_action_index
        )
        np.testing.assert_array_equal(
            atlas.tolerance_admissible_action_mask[task_index],
            certificate.tolerance_admissible_action_mask,
        )
        np.testing.assert_array_equal(
            atlas.robustly_optimal_action_mask[task_index],
            certificate.robustly_optimal_action_mask,
        )

    assert atlas.summary()["task_count"] == len(tasks)
    assert atlas.summary()["claim_boundary"] == DECISION_CAPABILITY_ATLAS_CLAIM_BOUNDARY


def test_exact_halfspaces_match_atlas_membership() -> None:
    prior, quotient, classes, intercept, coefficient = _controlled_family()
    target = np.linspace(-1.5, 1.5, 31)
    risk = np.linspace(0.0, 4.0, 29)
    tasks = np.asarray([(x, y) for x in target for y in risk])
    atlas = affine_decision_capability_atlas(
        prior,
        quotient,
        classes,
        intercept,
        coefficient,
        tasks,
    )

    for action in range(3):
        halfspaces = affine_capability_halfspaces(
            prior,
            quotient,
            classes,
            intercept,
            coefficient,
            action_index=action,
        )
        np.testing.assert_array_equal(
            halfspaces.contains(tasks),
            atlas.robustly_optimal_action_mask[:, action],
        )
        assert halfspaces.halfspace_count == 8


def test_controlled_atlas_has_three_action_regions_and_fallback_gaps() -> None:
    prior, quotient, classes, intercept, coefficient = _controlled_family()
    bounds = np.array([[-1.5, 1.5], [0.0, 4.0]])
    polygons = []
    for action in range(3):
        halfspaces = affine_capability_halfspaces(
            prior,
            quotient,
            classes,
            intercept,
            coefficient,
            action_index=action,
        )
        polygons.append(capability_polygon_2d(halfspaces, bounds))

    areas = np.asarray([polygon_area_2d(polygon) for polygon in polygons])
    np.testing.assert_allclose(areas[0], areas[2], atol=1e-12)
    assert np.all(areas > 0.5)
    assert np.sum(areas) < 12.0
    assert 0.15 < 1.0 - float(np.sum(areas)) / 12.0 < 0.22

    probe_tasks = np.array(
        [
            [-1.2, 0.2],
            [0.0, 2.0],
            [1.2, 0.2],
            [-0.45, 0.0],
        ]
    )
    atlas = affine_decision_capability_atlas(
        prior,
        quotient,
        classes,
        intercept,
        coefficient,
        probe_tasks,
    )
    np.testing.assert_array_equal(
        atlas.robustly_optimal_action_mask,
        np.array(
            [
                [True, False, False],
                [False, True, False],
                [False, False, True],
                [False, False, False],
            ]
        ),
    )


def test_capability_expands_monotonically_with_tolerance() -> None:
    prior, quotient, classes, intercept, coefficient = _controlled_family()
    tasks = np.asarray(
        [(x, y) for x in np.linspace(-1.5, 1.5, 41) for y in np.linspace(0, 4, 31)]
    )
    exact = affine_decision_capability_atlas(
        prior,
        quotient,
        classes,
        intercept,
        coefficient,
        tasks,
        regret_tolerance=0.0,
    )
    relaxed = affine_decision_capability_atlas(
        prior,
        quotient,
        classes,
        intercept,
        coefficient,
        tasks,
        regret_tolerance=0.1,
    )
    assert np.all(~exact.capability_mask | relaxed.capability_mask)
    assert np.count_nonzero(relaxed.capability_mask) > np.count_nonzero(
        exact.capability_mask
    )


def test_support_magnitudes_do_not_change_atlas() -> None:
    _, quotient, classes, intercept, coefficient = _controlled_family()
    tasks = np.array([[-1.0, 0.5], [0.0, 1.0], [1.0, 0.5]])
    first = affine_decision_capability_atlas(
        [0.1, 0.2, 0.3, 0.4],
        quotient,
        classes,
        intercept,
        coefficient,
        tasks,
    )
    second = affine_decision_capability_atlas(
        [0.4, 0.3, 0.2, 0.1],
        quotient,
        classes,
        intercept,
        coefficient,
        tasks,
    )
    np.testing.assert_allclose(first.worst_case_regret, second.worst_case_regret)


def test_outputs_are_immutable_and_policy_falls_back() -> None:
    prior, quotient, classes, intercept, coefficient = _controlled_family()
    tasks = np.array([[-1.2, 0.2], [-0.45, 0.0]])
    atlas = affine_decision_capability_atlas(
        prior,
        quotient,
        classes,
        intercept,
        coefficient,
        tasks,
    )
    np.testing.assert_array_equal(atlas.policy_action_index(1), np.array([0, 1]))
    with pytest.raises(ValueError, match="read-only"):
        atlas.worst_case_regret[0, 0] = 0.0
    with pytest.raises(ValueError, match="outside"):
        atlas.policy_action_index(3)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"task_parameters": [[0.0, 1.0, 2.0]]}, "wrong task dimension"),
        ({"loss_coefficients": np.zeros((4, 3, 0))}, "nonempty"),
        ({"task_batch_size": 0}, "positive integer"),
    ],
)
def test_invalid_contracts_fail_closed(kwargs: dict[str, object], match: str) -> None:
    prior, quotient, classes, intercept, coefficient = _controlled_family()
    parameters: dict[str, object] = {
        "prior_weights": prior,
        "quotient_weights": quotient,
        "class_index": classes,
        "loss_intercepts": intercept,
        "loss_coefficients": coefficient,
        "task_parameters": [[0.0, 1.0]],
    }
    parameters.update(kwargs)
    with pytest.raises(ValueError, match=match):
        affine_decision_capability_atlas(**parameters)


def test_halfspace_enumeration_cap_fails_closed() -> None:
    prior, quotient, classes, intercept, coefficient = _controlled_family()
    with pytest.raises(ValueError, match="exceeding"):
        affine_capability_halfspaces(
            prior,
            quotient,
            classes,
            intercept,
            coefficient,
            action_index=0,
            maximum_halfspaces=7,
        )


def test_numeric_and_finiteness_boundaries_fail_closed() -> None:
    prior, quotient, classes, intercept, coefficient = _controlled_family()
    with pytest.raises(ValueError, match="real numeric"):
        affine_decision_capability_atlas(
            prior,
            quotient,
            classes,
            [["bad"] * 3] * 4,
            coefficient,
            [[0.0, 1.0]],
        )
    bad_tasks = np.array([[0.0, np.inf]])
    with pytest.raises(ValueError, match="finite"):
        affine_decision_capability_atlas(
            prior,
            quotient,
            classes,
            intercept,
            coefficient,
            bad_tasks,
        )
    with pytest.raises(ValueError, match="positive integer"):
        affine_decision_capability_atlas(
            prior,
            quotient,
            classes,
            intercept,
            coefficient,
            [[0.0, 1.0]],
            task_batch_size=True,
        )


def test_policy_index_and_halfspace_contains_validate_inputs() -> None:
    prior, quotient, classes, intercept, coefficient = _controlled_family()
    atlas = affine_decision_capability_atlas(
        prior,
        quotient,
        classes,
        intercept,
        coefficient,
        [[0.0, 1.0]],
    )
    with pytest.raises(ValueError, match="integer"):
        atlas.policy_action_index(True)

    halfspaces = affine_capability_halfspaces(
        prior,
        quotient,
        classes,
        intercept,
        coefficient,
        action_index=1,
    )
    with pytest.raises(ValueError, match="wrong task dimension"):
        halfspaces.contains([[0.0]])

    empty_halfspaces = type(halfspaces)(
        action_index=1,
        regret_tolerance=0.0,
        active_class_index=np.array([0], dtype=np.int64),
        normal=np.empty((0, 2), dtype=np.float64),
        offset=np.empty(0, dtype=np.float64),
        benchmark_action_index=np.empty(0, dtype=np.int64),
        witness_hypothesis_index=np.empty((0, 1), dtype=np.int64),
    )
    np.testing.assert_array_equal(
        empty_halfspaces.contains([[0.0, 0.0], [1.0, 2.0]]),
        np.array([True, True]),
    )


def test_zero_posterior_class_is_ignored() -> None:
    displacement = np.array([[-1.0, 0.0], [-0.5, 0.5], [0.0, 1.0]])
    intercept = np.square(displacement)
    coefficient = np.stack((-2.0 * displacement, np.zeros_like(displacement)), axis=2)
    atlas = affine_decision_capability_atlas(
        [0.3, 0.3, 0.4],
        [1.0, 0.0],
        [0, 0, 1],
        intercept,
        coefficient,
        [[-0.75, 0.0]],
    )
    direct = query_decision_certificate(
        [0.3, 0.3, 0.4],
        [1.0, 0.0],
        [0, 0, 1],
        intercept + np.tensordot(coefficient, [-0.75, 0.0], axes=(2, 0)),
    )
    np.testing.assert_allclose(atlas.worst_case_regret[0], direct.worst_case_regret)


def test_halfspace_action_and_shape_boundaries_fail_closed() -> None:
    prior, quotient, classes, intercept, coefficient = _controlled_family()
    with pytest.raises(ValueError, match="shape"):
        affine_capability_halfspaces(
            prior,
            quotient,
            classes,
            intercept,
            np.zeros((4, 2, 2)),
            action_index=0,
        )
    with pytest.raises(ValueError, match="integer"):
        affine_capability_halfspaces(
            prior,
            quotient,
            classes,
            intercept,
            coefficient,
            action_index=True,
        )
    with pytest.raises(ValueError, match="outside"):
        affine_capability_halfspaces(
            prior,
            quotient,
            classes,
            intercept,
            coefficient,
            action_index=3,
        )


def test_polygon_and_area_input_boundaries() -> None:
    prior, quotient, classes, intercept, coefficient = _controlled_family()
    halfspaces = affine_capability_halfspaces(
        prior,
        quotient,
        classes,
        intercept,
        coefficient,
        action_index=1,
    )
    one_dimensional = type(halfspaces)(
        action_index=1,
        regret_tolerance=0.0,
        active_class_index=np.array([0], dtype=np.int64),
        normal=np.array([[1.0]], dtype=np.float64),
        offset=np.array([0.0], dtype=np.float64),
        benchmark_action_index=np.array([0], dtype=np.int64),
        witness_hypothesis_index=np.array([[0]], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="two-dimensional"):
        capability_polygon_2d(one_dimensional, [[-1.0, 1.0]])
    with pytest.raises(ValueError, match="shape"):
        capability_polygon_2d(halfspaces, [[-1.0, 1.0]])
    with pytest.raises(ValueError, match="lower < upper"):
        capability_polygon_2d(halfspaces, [[1.0, -1.0], [0.0, 1.0]])

    impossible = type(halfspaces)(
        action_index=1,
        regret_tolerance=0.0,
        active_class_index=np.array([0], dtype=np.int64),
        normal=np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float64),
        offset=np.array([-2.0, -2.0], dtype=np.float64),
        benchmark_action_index=np.array([0, 2], dtype=np.int64),
        witness_hypothesis_index=np.array([[0], [0]], dtype=np.int64),
    )
    empty = capability_polygon_2d(impossible, [[-1.0, 1.0], [-1.0, 1.0]])
    assert empty.shape == (0, 2)
    assert polygon_area_2d(empty) == 0.0
    assert polygon_area_2d([[0.0, 0.0], [1.0, 0.0]]) == 0.0
    with pytest.raises(ValueError, match="shape"):
        polygon_area_2d([0.0, 1.0])
