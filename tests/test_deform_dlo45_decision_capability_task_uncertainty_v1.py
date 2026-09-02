from __future__ import annotations

import itertools

import numpy as np
import pytest

from bayesian_phystwin.decision_capability_atlas_v1 import AffineCapabilityHalfspacesV1
from bayesian_phystwin.decision_capability_task_uncertainty_v1 import (
    TASK_UNCERTAINTY_CERTIFICATE_CLAIM_BOUNDARY,
    box_robust_center_halfspaces,
    box_task_set_capability,
    ellipsoid_robust_center_halfspaces,
    ellipsoid_task_set_capability,
    norm_ball_capability_margin,
    task_uncertainty_action_mask,
)


def _region(action: int) -> AffineCapabilityHalfspacesV1:
    normals = (
        np.array(
            [
                [2.0, 0.415],
                [1.6, 0.615],
                [1.8, 0.615],
                [1.4, 0.815],
                [3.4, -0.4],
                [3.4, 0.0],
                [3.4, 0.0],
                [3.4, 0.4],
            ]
        ),
        np.array(
            [
                [-2.0, -0.415],
                [-1.6, -0.615],
                [-1.8, -0.615],
                [-1.4, -0.815],
                [1.4, -0.815],
                [1.8, -0.615],
                [1.6, -0.615],
                [2.0, -0.415],
            ]
        ),
        np.array(
            [
                [-3.4, 0.4],
                [-3.4, 0.0],
                [-3.4, 0.0],
                [-3.4, -0.4],
                [-1.4, 0.815],
                [-1.8, 0.615],
                [-1.6, 0.615],
                [-2.0, 0.415],
            ]
        ),
    )[action]
    offsets = (
        np.array([-1.1, -0.78, -0.74, -0.42, -0.68, -0.04, 0.04, 0.68]),
        np.array([1.1, 0.78, 0.74, 0.42, 0.42, 0.74, 0.78, 1.1]),
        np.array([0.68, 0.04, -0.04, -0.68, -0.42, -0.74, -0.78, -1.1]),
    )[action]
    return AffineCapabilityHalfspacesV1(
        action_index=action,
        regret_tolerance=0.0,
        active_class_index=np.array([0, 1], dtype=np.int64),
        normal=normals,
        offset=offsets,
        benchmark_action_index=np.zeros(8, dtype=np.int64),
        witness_hypothesis_index=np.zeros((8, 2), dtype=np.int64),
    )


def test_box_certificate_matches_all_vertices() -> None:
    region = _region(0)
    centers = np.array([[-1.2, 0.2], [-0.6, 0.0], [-0.45, 0.0]])
    widths = np.array([[0.2, 0.2], [0.03, 0.005], [0.03, 0.01]])
    result = box_task_set_capability(region, centers, widths)
    expected = []
    for center, width in zip(centers, widths, strict=True):
        vertices = np.asarray(
            [
                center + width * np.asarray(signs)
                for signs in itertools.product((-1, 1), repeat=2)
            ]
        )
        expected.append(bool(np.all(region.contains(vertices))))
    np.testing.assert_array_equal(result.capable_mask, expected)
    np.testing.assert_allclose(result.minimum_slack, -result.worst_excess)
    assert (
        result.summary()["claim_boundary"]
        == TASK_UNCERTAINTY_CERTIFICATE_CLAIM_BOUNDARY
    )


def test_nominal_capability_can_fail_under_task_uncertainty() -> None:
    region = _region(0)
    center = np.array([[-0.6, 0.0]])
    assert bool(region.contains(center)[0])
    small = box_task_set_capability(region, center, [0.03, 0.005])
    large = box_task_set_capability(region, center, [0.06, 0.01])
    assert bool(small.capable_mask[0])
    assert not bool(large.capable_mask[0])


def test_box_eroded_halfspaces_match_box_certificate() -> None:
    region = _region(1)
    centers = np.array([[0.0, 2.0], [0.0, 0.1], [0.4, 1.0]])
    widths = np.array([0.2, 0.4])
    direct = box_task_set_capability(region, centers, widths)
    eroded = box_robust_center_halfspaces(region, widths)
    np.testing.assert_array_equal(eroded.contains(centers), direct.capable_mask)


def test_ellipsoid_certificate_and_shifted_halfspaces_agree() -> None:
    region = _region(0)
    centers = np.array([[-1.2, 0.2], [-0.6, 0.0], [-0.45, 0.0]])
    generator = np.diag([0.05, 0.1])
    direct = ellipsoid_task_set_capability(region, centers, generator)
    shifted = ellipsoid_robust_center_halfspaces(region, generator)
    np.testing.assert_array_equal(shifted.contains(centers), direct.capable_mask)
    expected_support = np.linalg.norm(region.normal @ generator, axis=1)
    np.testing.assert_allclose(direct.support_value[0], expected_support)


def test_batched_ellipsoids_accept_different_latent_dimensions() -> None:
    region = _region(1)
    centers = np.array([[0.0, 2.0], [0.0, 0.2]])
    generators = np.array(
        [
            [[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]],
            [[0.2, 0.0, 0.0], [0.0, 0.1, 0.1]],
        ]
    )
    result = ellipsoid_task_set_capability(region, centers, generators)
    assert result.support_value.shape == (2, region.halfspace_count)


def test_norm_ball_radius_hits_the_critical_boundary() -> None:
    region = _region(0)
    center = np.array([[-0.6, 0.0]])
    result = norm_ball_capability_margin(region, center, task_norm="l2")
    radius = float(result.normalized_constraint_margin[0])
    critical = int(result.critical_halfspace_index[0])
    assert radius == pytest.approx(0.0489571513187252)
    normal = region.normal[critical]
    boundary = center[0] + radius * normal / np.linalg.norm(normal)
    assert float(normal @ boundary) == pytest.approx(float(region.offset[critical]))
    assert bool(result.center_capable_mask[0])


def test_norm_ball_dual_norms_and_outside_sign() -> None:
    region = _region(0)
    centers = np.array([[-1.2, 0.2], [-0.45, 0.0]])
    l1 = norm_ball_capability_margin(region, centers, task_norm="l1")
    l2 = norm_ball_capability_margin(region, centers, task_norm="l2")
    linf = norm_ball_capability_margin(region, centers, task_norm="linf")
    assert (
        l1.normalized_constraint_margin[0]
        >= l2.normalized_constraint_margin[0]
        >= linf.normalized_constraint_margin[0]
    )
    assert l2.normalized_constraint_margin[1] < 0.0
    assert l2.guaranteed_radius[1] == 0.0
    with pytest.raises(ValueError, match="task_norm"):
        norm_ball_capability_margin(
            region, centers, task_norm="bad"  # type: ignore[arg-type]
        )


def test_uncertainty_atlas_preserves_unique_and_fallback_sets() -> None:
    centers = np.array(
        [[-1.2, 0.2], [0.0, 2.0], [1.2, 0.2], [-0.6, 0.0]]
    )
    widths = np.array(
        [[0.2, 0.2], [0.2, 0.4], [0.2, 0.2], [0.06, 0.01]]
    )
    reports = [box_task_set_capability(_region(a), centers, widths) for a in range(3)]
    mask = task_uncertainty_action_mask(reports)
    np.testing.assert_array_equal(
        mask,
        np.array(
            [
                [True, False, False],
                [False, True, False],
                [False, False, True],
                [False, False, False],
            ]
        ),
    )
    np.testing.assert_array_equal(
        ~np.any(mask, axis=1), np.array([False, False, False, True])
    )


def test_empty_halfspaces_certify_every_task_with_infinite_radius() -> None:
    empty = AffineCapabilityHalfspacesV1(
        action_index=0,
        regret_tolerance=0.0,
        active_class_index=np.array([], dtype=np.int64),
        normal=np.empty((0, 2)),
        offset=np.empty(0),
        benchmark_action_index=np.empty(0, dtype=np.int64),
        witness_hypothesis_index=np.empty((0, 0), dtype=np.int64),
    )
    box = box_task_set_capability(empty, [[0.0, 0.0], [1.0, 2.0]], [0.1, 0.2])
    radius = norm_ball_capability_margin(empty, [[0.0, 0.0]], task_norm="l2")
    assert box.all_capable
    assert np.isinf(radius.normalized_constraint_margin[0])
    assert radius.critical_halfspace_index[0] == -1


def test_zero_normal_infeasible_constraint_is_detected() -> None:
    region = AffineCapabilityHalfspacesV1(
        action_index=0,
        regret_tolerance=0.0,
        active_class_index=np.array([0]),
        normal=np.array([[0.0, 0.0]]),
        offset=np.array([-1.0]),
        benchmark_action_index=np.array([1]),
        witness_hypothesis_index=np.array([[0]]),
    )
    radius = norm_ball_capability_margin(region, [[0.0, 0.0]])
    assert radius.normalized_constraint_margin[0] == -np.inf
    assert not bool(radius.center_capable_mask[0])


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: box_task_set_capability(_region(0), [[0.0]], [0.1, 0.2]), "shape"),
        (
            lambda: box_task_set_capability(
                _region(0), [[0.0, 0.0]], [-0.1, 0.2]
            ),
            "nonnegative",
        ),
        (
            lambda: box_task_set_capability(
                _region(0), [[0.0, np.nan]], [0.1, 0.2]
            ),
            "finite",
        ),
        (
            lambda: ellipsoid_task_set_capability(
                _region(0), [[0.0, 0.0]], np.eye(3)
            ),
            "shape",
        ),
        (
            lambda: ellipsoid_robust_center_halfspaces(
                _region(0), [[1.0], [np.inf]]
            ),
            "finite",
        ),
        (lambda: box_robust_center_halfspaces(_region(0), [0.1]), "one value"),
    ],
)
def test_invalid_task_uncertainty_contracts_fail_closed(call, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()


def test_outputs_are_immutable_and_report_roster_validation_is_strict() -> None:
    first = box_task_set_capability(_region(0), [[-1.2, 0.2]], [0.1, 0.1])
    with pytest.raises(ValueError, match="read-only"):
        first.capable_mask[0] = False
    with pytest.raises(TypeError, match="sequence"):
        task_uncertainty_action_mask("bad")
    with pytest.raises(ValueError, match="nonempty"):
        task_uncertainty_action_mask([])
    with pytest.raises(ValueError, match="contiguous"):
        task_uncertainty_action_mask(
            [
                box_task_set_capability(
                    _region(1), [[0.0, 2.0]], [0.1, 0.1]
                )
            ]
        )


def test_helper_input_boundaries_cover_malformed_arrays() -> None:
    import bayesian_phystwin.decision_capability_task_uncertainty_v1 as module

    with pytest.raises(ValueError, match="real numeric"):
        box_task_set_capability(_region(0), [["bad", "task"]], [0.1, 0.2])
    with pytest.raises(ValueError, match="real numeric"):
        box_task_set_capability(_region(0), [[0.0, 0.0]], ["bad", "width"])
    with pytest.raises(ValueError, match="shape"):
        box_task_set_capability(
            _region(0), [[0.0, 0.0], [1.0, 1.0]], [[0.1, 0.2]]
        )
    with pytest.raises(ValueError, match="one- or two-dimensional"):
        box_task_set_capability(_region(0), [[0.0, 0.0]], [[[0.1, 0.2]]])
    with pytest.raises(ValueError, match="finite"):
        box_task_set_capability(_region(0), [[0.0, 0.0]], [np.inf, 0.2])
    with pytest.raises(TypeError, match="AffineCapabilityHalfspacesV1"):
        box_task_set_capability(  # type: ignore[arg-type]
            "bad", [[0.0, 0.0]], [0.1, 0.2]
        )

    malformed = AffineCapabilityHalfspacesV1(
        action_index=0,
        regret_tolerance=0.0,
        active_class_index=np.array([0]),
        normal=np.zeros((2, 2)),
        offset=np.zeros(1),
        benchmark_action_index=np.zeros(2, dtype=np.int64),
        witness_hypothesis_index=np.zeros((2, 1), dtype=np.int64),
    )
    with pytest.raises(ValueError, match="inconsistent"):
        box_task_set_capability(malformed, [[0.0, 0.0]], [0.1, 0.2])

    zero_dimension = malformed._replace(
        normal=np.empty((0, 0)),
        offset=np.empty(0),
        benchmark_action_index=np.empty(0, dtype=np.int64),
        witness_hypothesis_index=np.empty((0, 1), dtype=np.int64),
    )
    with pytest.raises(ValueError, match="positive task dimension"):
        box_task_set_capability(zero_dimension, [[]], [])

    nonfinite = malformed._replace(
        normal=np.array([[np.nan, 0.0]]),
        offset=np.array([0.0]),
        benchmark_action_index=np.array([1]),
        witness_hypothesis_index=np.array([[0]]),
    )
    with pytest.raises(ValueError, match="finite"):
        box_task_set_capability(nonfinite, [[0.0, 0.0]], [0.1, 0.2])

    centers = module._centers([[0.0, 0.0]], dimension=2)
    with pytest.raises(ValueError, match="wrong task-set"):
        module._task_set_result(
            _region(0),
            centers,
            np.zeros((1, 7)),
            uncertainty_kind="test",
        )
    with pytest.raises(ValueError, match="nonnegative"):
        module._task_set_result(
            _region(0),
            centers,
            -np.ones((1, 8)),
            uncertainty_kind="test",
        )


def test_ellipsoid_malformed_contracts_cover_batched_branches() -> None:
    region = _region(0)
    center = [[0.0, 0.0]]
    with pytest.raises(ValueError, match="real numeric"):
        ellipsoid_task_set_capability(region, center, [["bad"], ["bad"]])
    with pytest.raises(ValueError, match="batched generators"):
        ellipsoid_task_set_capability(region, center, np.zeros((2, 2, 1)))
    with pytest.raises(ValueError, match="two- or three-dimensional"):
        ellipsoid_task_set_capability(region, center, [0.1, 0.2])
    with pytest.raises(ValueError, match="finite"):
        ellipsoid_task_set_capability(
            region, center, np.array([[[np.inf], [0.0]]])
        )
    with pytest.raises(ValueError, match="real numeric"):
        ellipsoid_robust_center_halfspaces(region, [["bad"], ["bad"]])
    with pytest.raises(ValueError, match="shape"):
        ellipsoid_robust_center_halfspaces(region, np.zeros((2, 0)))


def test_radius_and_atlas_summaries_and_no_unique_branch() -> None:
    radius = norm_ball_capability_margin(_region(1), [[0.0, 2.0]])
    summary = radius.summary()
    assert summary["task_count"] == radius.task_count == 1
    assert summary["task_dimension"] == radius.task_dimension == 2
    assert summary["capable_center_count"] == 1

    centers = [[-0.45, 0.0]]
    reports = [
        box_task_set_capability(_region(action), centers, [0.01, 0.01])
        for action in range(3)
    ]
    mask = task_uncertainty_action_mask(reports)
    assert mask.shape == (1, 3)
    assert not np.any(mask)


def test_report_aggregation_rejects_mismatched_payloads() -> None:
    first = box_task_set_capability(_region(0), [[-1.2, 0.2]], [0.1, 0.1])
    second = box_task_set_capability(_region(1), [[-1.2, 0.2]], [0.1, 0.1])
    third = box_task_set_capability(_region(2), [[-1.2, 0.2]], [0.1, 0.1])
    with pytest.raises(TypeError, match="contain"):
        task_uncertainty_action_mask([first, "bad", third])  # type: ignore[list-item]
    mismatched_kind = second._replace(uncertainty_kind="centered-ellipsoid")
    with pytest.raises(ValueError, match="same uncertainty kind"):
        task_uncertainty_action_mask([first, mismatched_kind, third])
    mismatched_center = second._replace(task_centers=np.array([[0.0, 2.0]]))
    with pytest.raises(ValueError, match="identical task centers"):
        task_uncertainty_action_mask([first, mismatched_center, third])
