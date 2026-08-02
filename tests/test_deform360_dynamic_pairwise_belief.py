from __future__ import annotations

import copy

import numpy as np

from bayesian_phystwin.deform360_dynamic_pairwise_belief import (
    DYNAMIC_PAIRWISE_ARM,
    SELECTED_BACKBONE_ARM,
    DynamicPairwiseBeliefConfig,
    predict_dynamic_pairwise_belief_arrays,
    select_dynamic_centers,
    update_metric_recursive_rbf_belief,
)
from bayesian_phystwin.phystwin_online_belief import (
    RecursiveRbfBeliefConfig,
    initialize_recursive_rbf_belief,
)


def _synthetic_inputs() -> dict[str, np.ndarray]:
    frame_count = 7
    point_count = 80
    angle = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    frame_zero = np.column_stack(
        (0.10 * np.cos(angle), 0.10 * np.sin(angle), np.zeros(point_count))
    )
    local_mode = 0.5 + 0.5 * np.sin(angle)
    response = np.zeros((frame_count, point_count, 3), dtype=np.float64)
    response[:, :, 2] = (
        np.linspace(0.0, 0.006, frame_count)[:, None] * local_mode
    )
    physical = frame_zero[None] + response
    persistence = np.repeat(frame_zero[None], frame_count, axis=0)
    measurement = np.full_like(physical, np.nan)
    visibility = np.zeros((frame_count, point_count), dtype=bool)
    validity = np.zeros_like(visibility)
    measurement[0] = frame_zero
    visibility[0] = True
    validity[0] = True
    update = 3
    measurement[update] = physical[update]
    measurement[update, :, 2] += 0.0015 * local_mode
    visibility[update] = True
    validity[update] = True
    return {
        "physical": physical.astype(np.float32),
        "persistence": persistence.astype(np.float32),
        "response": response,
        "frame_zero": frame_zero,
        "action_support": np.ones(point_count),
        "measurement": measurement,
        "visibility": visibility,
        "validity": validity,
        "pool_ids": np.arange(64, dtype=np.int64),
        "reliability": np.ones((1, 64), dtype=np.float64),
        "variance": np.full((1, 64), 0.002**2, dtype=np.float64),
        "view_count": np.full((1, 64), 3, dtype=np.int64),
    }


def _predict(
    inputs: dict[str, np.ndarray],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    return predict_dynamic_pairwise_belief_arrays(
        inputs["physical"],
        inputs["persistence"],
        inputs["response"],
        inputs["frame_zero"],
        inputs["action_support"],
        inputs["measurement"],
        inputs["visibility"],
        inputs["validity"],
        pool_ids=inputs["pool_ids"],
        prior_reliability=inputs["reliability"],
        observation_variance_m2=inputs["variance"],
        inlier_view_count=inputs["view_count"],
        config=DynamicPairwiseBeliefConfig(update_frames=(3,)),
    )


def test_aligned_causal_update_passes_physical_guards() -> None:
    inputs = _synthetic_inputs()

    report, arrays = _predict(inputs)

    update = report["updates"][0]
    assert report["accepted_update_count"] == 1
    assert update["accepted"] is True
    assert update["selected_backbone"] == "physical_prior"
    assert len(update["preassociation_pool_ids"]) == 24
    assert len(update["active_center_ids"]) == 16
    assert update["effective_reliability_sum"] <= 8.0 + 1e-12
    assert update["correction_physical_cosine"] > 0.9
    assert not np.array_equal(
        arrays[DYNAMIC_PAIRWISE_ARM][4:],
        arrays[SELECTED_BACKBONE_ARM][4:],
    )


def test_pool_archive_order_does_not_change_prediction() -> None:
    inputs = _synthetic_inputs()
    first_report, first_arrays = _predict(inputs)
    permutation = np.random.default_rng(4).permutation(64)
    permuted = copy.deepcopy(inputs)
    permuted["pool_ids"] = permuted["pool_ids"][permutation]
    for name in ("reliability", "variance", "view_count"):
        permuted[name] = permuted[name][:, permutation]

    second_report, second_arrays = _predict(permuted)

    assert first_report["observation_pool_ids"] == second_report["observation_pool_ids"]
    assert first_report["updates"] == second_report["updates"]
    for arm in (SELECTED_BACKBONE_ARM, DYNAMIC_PAIRWISE_ARM):
        np.testing.assert_array_equal(first_arrays[arm], second_arrays[arm])


def test_prior_selection_reliability_does_not_reuse_state_innovation() -> None:
    inputs = _synthetic_inputs()
    first_report, _ = _predict(inputs)
    shifted = copy.deepcopy(inputs)
    shifted["measurement"][3, :, 0] += 0.020

    second_report, _ = _predict(shifted)

    first = first_report["updates"][0]
    second = second_report["updates"][0]
    assert first["preassociation_pool_ids"] == second["preassociation_pool_ids"]
    assert first["active_center_ids"] == second["active_center_ids"]
    assert first["effective_reliability_sum"] == second["effective_reliability_sum"]
    assert report_boundary(first_report, "prior_reliability_uses_state_innovation") is False


def report_boundary(report: dict[str, object], name: str) -> object:
    return report["information_boundary"][name]


def test_correlated_duplicate_pool_cannot_increase_information() -> None:
    row_count = 64
    x = np.linspace(-1.0, 1.0, row_count)
    state = np.column_stack((np.ones(row_count), x))
    nuisance = np.column_stack((np.ones(row_count), x**2))
    variance = np.full(row_count, 0.002**2)
    reliability = np.ones(row_count)
    available = np.ones(row_count, dtype=bool)
    config = DynamicPairwiseBeliefConfig()
    original = select_dynamic_centers(
        state,
        nuisance,
        variance,
        reliability,
        available,
        count=16,
        config=config,
    )
    duplicated = select_dynamic_centers(
        np.repeat(state, 2, axis=0),
        np.repeat(nuisance, 2, axis=0),
        np.repeat(variance, 2),
        np.repeat(reliability, 2),
        np.repeat(available, 2),
        count=16,
        config=config,
    )

    assert np.sum(original.effective_reliability) <= 8.0 + 1e-12
    assert np.sum(duplicated.effective_reliability) <= 8.0 + 1e-12
    assert np.sum(duplicated.mutual_information_nats) <= np.sum(
        original.mutual_information_nats
    ) + 1e-12


def test_metric_variance_and_prior_reliability_scale_recursive_update() -> None:
    points = np.column_stack((np.linspace(0.0, 1.0, 5), np.zeros((5, 2))))
    center_ids = np.asarray([0, 2, 4])
    config = RecursiveRbfBeliefConfig()
    prior = initialize_recursive_rbf_belief(
        center_ids,
        points[center_ids],
        points,
        config=config,
    )
    residual = np.repeat(np.asarray([[0.01, 0.0, 0.0]]), 3, axis=0)
    available = np.ones(3, dtype=bool)
    confident, _ = update_metric_recursive_rbf_belief(
        prior,
        1,
        points[center_ids],
        residual,
        available,
        prior_reliability=np.ones(3),
        observation_variance_m2=np.full(3, 0.002**2),
        config=config,
    )
    conservative, reliability = update_metric_recursive_rbf_belief(
        prior,
        1,
        points[center_ids],
        residual,
        available,
        prior_reliability=np.full(3, 0.25),
        observation_variance_m2=np.full(3, 0.010**2),
        config=config,
    )

    assert conservative.global_mean_m[0] < confident.global_mean_m[0]
    assert np.all(conservative.local_variance_m2 > confident.local_variance_m2)
    assert np.all(reliability <= 0.25)


def test_insufficient_independent_views_is_exact_selected_fallback() -> None:
    inputs = _synthetic_inputs()
    inputs["view_count"][:] = 2

    report, arrays = _predict(inputs)

    assert report["accepted_update_count"] == 0
    assert report["updates"][0]["bit_exact_selected_backbone_fallback"] is True
    np.testing.assert_array_equal(
        arrays[DYNAMIC_PAIRWISE_ARM],
        arrays[SELECTED_BACKBONE_ARM],
    )


def test_future_measurement_mutation_cannot_change_prediction() -> None:
    inputs = _synthetic_inputs()
    first_report, first_arrays = _predict(inputs)
    mutated = copy.deepcopy(inputs)
    mutated["measurement"][4:] = np.random.default_rng(8).normal(
        size=mutated["measurement"][4:].shape
    )
    mutated["visibility"][4:] = True
    mutated["validity"][4:] = True

    second_report, second_arrays = _predict(mutated)

    assert first_report == second_report
    for arm in (SELECTED_BACKBONE_ARM, DYNAMIC_PAIRWISE_ARM):
        np.testing.assert_array_equal(first_arrays[arm], second_arrays[arm])
