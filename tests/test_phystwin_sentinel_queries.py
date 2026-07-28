import numpy as np
import pytest

from bayesian_phystwin.phystwin_active_queries import PhysicsGuidedQueryConfig
from bayesian_phystwin.phystwin_sentinel_queries import (
    ACTIVE_QUERY_ROLE,
    SENTINEL_QUERY_ROLE,
    MotionStratifiedQueryConfig,
    SentinelBiasConfig,
    debias_active_displacements,
    estimate_sentinel_common_bias,
    plan_motion_stratified_queries,
)


def _query_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rollout = np.zeros((4, 10, 3), dtype=np.float64)
    rollout[:, :, 0] = np.arange(10, dtype=np.float64)[None]
    for node_id in range(6):
        rollout[:, node_id, 1] = np.linspace(
            0.0,
            0.004 + 0.001 * node_id,
            len(rollout),
        )
    for node_id in range(6, 10):
        rollout[:, node_id, 2] = np.linspace(0.0, 0.0001, len(rollout))
    pixels = np.empty((3, 4, 10, 2), dtype=np.float64)
    for camera in range(3):
        pixels[camera, ..., 0] = 20.0 + rollout[..., 0] + camera
        pixels[camera, ..., 1] = 30.0 + rollout[..., 1]
    support = np.ones((3, 4, 10), dtype=np.float64)
    return rollout, pixels, support


def test_motion_stratified_plan_reserves_fixed_sentinel_budget() -> None:
    rollout, pixels, support = _query_inputs()
    plan = plan_motion_stratified_queries(
        rollout,
        pixels,
        support,
        mode_basis=np.eye(10, 3),
        active_config=PhysicsGuidedQueryConfig(
            maximum_reseeds=0,
            minimum_motion_m=0.002,
            contact_exclusion_fraction=0.0,
        ),
        config=MotionStratifiedQueryConfig(
            total_query_count=8,
            sentinel_query_count=2,
            sentinel_maximum_motion_m=0.0005,
            sentinel_maximum_reseeds=0,
        ),
    )

    assert plan.initial_budget_met
    assert plan.initial_query_count == 8
    assert len(plan.active_node_ids) == 6
    assert len(plan.sentinel_node_ids) == 2
    assert set(plan.active_node_ids).isdisjoint(plan.sentinel_node_ids)
    active_motion = np.max(
        np.linalg.norm(
            rollout[:, plan.active_node_ids] - rollout[0, plan.active_node_ids],
            axis=2,
        ),
        axis=0,
    )
    sentinel_motion = np.max(
        np.linalg.norm(
            rollout[:, plan.sentinel_node_ids]
            - rollout[0, plan.sentinel_node_ids],
            axis=2,
        ),
        axis=0,
    )
    assert np.all(active_motion >= 0.002)
    assert np.all(sentinel_motion <= 0.0005)

    node_ids, queries, replacements, roles = plan.camera_queries_txy(0)
    assert len(node_ids) == len(queries) == len(replacements) == len(roles) == 8
    assert np.sum(roles == ACTIVE_QUERY_ROLE) == 6
    assert np.sum(roles == SENTINEL_QUERY_ROLE) == 2
    assert not roles.flags.writeable


def test_missing_sentinels_do_not_expand_the_active_budget() -> None:
    rollout, pixels, support = _query_inputs()
    rollout[:, 6:, 2] = np.linspace(0.0, 0.004, len(rollout))[:, None]
    plan = plan_motion_stratified_queries(
        rollout,
        pixels,
        support,
        active_config=PhysicsGuidedQueryConfig(
            maximum_reseeds=0,
            minimum_motion_m=0.002,
            contact_exclusion_fraction=0.0,
        ),
        config=MotionStratifiedQueryConfig(
            total_query_count=8,
            sentinel_query_count=2,
            sentinel_maximum_motion_m=0.0005,
            sentinel_maximum_reseeds=0,
        ),
    )

    assert len(plan.active_node_ids) == 6
    assert len(plan.sentinel_node_ids) == 0
    assert plan.initial_query_count == 6
    assert not plan.initial_budget_met


def test_motion_stratified_selection_is_candidate_order_invariant() -> None:
    rollout, pixels, support = _query_inputs()
    kwargs = {
        "active_config": PhysicsGuidedQueryConfig(
            maximum_reseeds=0,
            minimum_motion_m=0.002,
            contact_exclusion_fraction=0.0,
        ),
        "config": MotionStratifiedQueryConfig(
            total_query_count=6,
            sentinel_query_count=2,
            sentinel_maximum_motion_m=0.0005,
            sentinel_maximum_reseeds=0,
        ),
    }
    first = plan_motion_stratified_queries(
        rollout,
        pixels,
        support,
        candidate_ids=np.asarray([9, 3, 5, 0, 7, 1, 8, 2, 6, 4]),
        **kwargs,
    )
    second = plan_motion_stratified_queries(
        rollout,
        pixels,
        support,
        candidate_ids=np.arange(10),
        **kwargs,
    )

    np.testing.assert_array_equal(first.active_node_ids, second.active_node_ids)
    np.testing.assert_array_equal(first.sentinel_node_ids, second.sentinel_node_ids)


def test_sentinel_bias_recovers_common_mode_without_duplicate_confidence() -> None:
    bias = np.asarray([0.01, -0.005, 0.002])
    covariance = np.repeat((1e-6 * np.eye(3))[None], 4, axis=0)
    estimate = estimate_sentinel_common_bias(
        np.repeat(bias[None], 4, axis=0),
        np.zeros((4, 3)),
        covariance,
        np.ones(4),
        np.asarray([0, 0, 1, 1]),
    )

    assert estimate.usable
    assert estimate.decision == "sentinel-common-mode-estimated"
    np.testing.assert_allclose(estimate.bias_m, bias, atol=1e-12)

    single = estimate_sentinel_common_bias(
        bias[None],
        np.zeros((1, 3)),
        covariance[:1],
        np.ones(1),
        np.asarray([0]),
    )
    duplicate = estimate_sentinel_common_bias(
        np.repeat(bias[None], 8, axis=0),
        np.zeros((8, 3)),
        np.repeat(covariance[:1], 8, axis=0),
        np.ones(8),
        np.zeros(8, dtype=np.int64),
    )
    np.testing.assert_allclose(
        duplicate.covariance_m2,
        single.covariance_m2,
        rtol=0.0,
        atol=1e-15,
    )


def test_unknown_correlation_is_not_as_confident_as_independence() -> None:
    covariance = np.repeat((2e-6 * np.eye(3))[None], 2, axis=0)
    estimate = estimate_sentinel_common_bias(
        np.zeros((2, 3)),
        np.zeros((2, 3)),
        covariance,
        np.ones(2),
        np.asarray([0, 1]),
    )
    independent_covariance = np.linalg.inv(
        np.linalg.inv(covariance[0]) + np.linalg.inv(covariance[1])
    )

    assert estimate.usable
    assert np.all(
        np.linalg.eigvalsh(estimate.covariance_m2)
        >= np.linalg.eigvalsh(independent_covariance)
    )


def test_inconsistent_sentinels_force_abstention() -> None:
    covariance = np.repeat((1e-6 * np.eye(3))[None], 2, axis=0)
    estimate = estimate_sentinel_common_bias(
        np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        np.zeros((2, 3)),
        covariance,
        np.ones(2),
        np.asarray([0, 0]),
        config=SentinelBiasConfig(maximum_inconsistency_sigma=4.0),
    )

    assert not estimate.usable
    assert estimate.decision == "sentinel-common-mode-inconsistent"
    with pytest.raises(ValueError, match="exact baseline"):
        debias_active_displacements(
            np.zeros((1, 3)),
            covariance[:1],
            estimate,
        )


def test_debiasing_propagates_sentinel_covariance() -> None:
    bias = np.asarray([0.01, 0.0, 0.0])
    sentinel_covariance = np.repeat((1e-6 * np.eye(3))[None], 2, axis=0)
    estimate = estimate_sentinel_common_bias(
        np.repeat(bias[None], 2, axis=0),
        np.zeros((2, 3)),
        sentinel_covariance,
        np.ones(2),
        np.asarray([0, 1]),
    )
    active_covariance = np.repeat((4e-6 * np.eye(3))[None], 3, axis=0)
    corrected, corrected_covariance = debias_active_displacements(
        np.repeat((bias + np.asarray([0.02, 0.0, 0.0]))[None], 3, axis=0),
        active_covariance,
        estimate,
    )

    np.testing.assert_allclose(
        corrected,
        np.repeat(np.asarray([[0.02, 0.0, 0.0]]), 3, axis=0),
    )
    np.testing.assert_allclose(
        corrected_covariance,
        active_covariance + estimate.covariance_m2[None],
    )
    assert not corrected.flags.writeable
    assert not corrected_covariance.flags.writeable
