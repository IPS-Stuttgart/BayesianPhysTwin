from __future__ import annotations

import numpy as np

from bayesian_phystwin.deform360_raw_camera_gated_evaluation import (
    CHI2_DF3_95,
    LEGACY_SELECTED_BACKBONE_ARM,
    SELECTED_BACKBONE_ARM,
    covariance_innovation_gate,
    evaluate_covariance_gated_arrays,
)


def test_covariance_gate_uses_nearest_backbone_not_array_identity() -> None:
    backbone = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    observation = backbone[[2, 0, 1]] + np.array([0.01, 0.0, 0.0])
    covariance = np.repeat((np.eye(3) * 0.001**2)[None], 3, axis=0)

    result = covariance_innovation_gate(
        observation,
        backbone,
        covariance,
        np.ones(3, dtype=bool),
    )

    assert result["accepted"] is True
    assert np.isclose(result["median_squared_mahalanobis_innovation"], 100.0)
    assert result["threshold"] == CHI2_DF3_95


def test_covariance_gate_abstains_for_uncertainty_scale_innovation() -> None:
    backbone = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    observation = backbone + np.array([0.001, 0.0, 0.0])
    covariance = np.repeat((np.eye(3) * 0.001**2)[None], 3, axis=0)

    result = covariance_innovation_gate(
        observation,
        backbone,
        covariance,
        np.ones(3, dtype=bool),
    )

    assert result["accepted"] is False
    assert result["decision"] == "covariance_gate_rejected"
    assert np.isclose(result["median_squared_mahalanobis_innovation"], 1.0)


def test_covariance_gate_requires_three_valid_covariances() -> None:
    backbone = np.zeros((3, 3))
    observation = np.ones((3, 3))
    covariance = np.repeat(np.eye(3)[None], 3, axis=0)

    result = covariance_innovation_gate(
        observation,
        backbone,
        covariance,
        np.array([True, True, False]),
    )

    assert result["accepted"] is False
    assert result["decision"] == "insufficient_valid_covariance"
    assert result["valid_count"] == 2


def test_rbf_selector_switch_keeps_one_state_per_backbone() -> None:
    frame_count = 76
    point_count = 5
    centers = np.array([0, 1, 2])
    frame_zero = np.stack(
        (np.arange(point_count), np.zeros(point_count), np.ones(point_count)), axis=1
    ).astype(np.float32)
    prior = np.repeat(frame_zero[None], frame_count, axis=0)
    persistence = prior.copy()
    prior[38:, :, 1] = 0.1
    persistence[19:38, :, 1] = 0.1
    target = prior.copy()
    measurement = np.full_like(prior, np.nan)
    measurement_validity = np.zeros((frame_count, point_count), dtype=bool)
    measurement[19, centers] = prior[19, centers] + np.array([0.01, 0.0, 0.0])
    measurement[38, centers] = persistence[38, centers] + np.array([0.01, 0.0, 0.0])
    measurement[57, centers] = persistence[57, centers] + np.array([0.01, 0.0, 0.0])
    measurement_validity[np.ix_([19, 38, 57], centers)] = True
    covariance = np.full((frame_count, point_count, 3, 3), np.nan)
    covariance_validity = measurement_validity.copy()
    covariance[measurement_validity] = np.eye(3) * 1.0e-6

    report, _ = evaluate_covariance_gated_arrays(
        prior,
        persistence,
        target,
        np.ones((frame_count, point_count), dtype=bool),
        np.ones((frame_count, point_count), dtype=bool),
        measurement,
        measurement_validity,
        covariance,
        covariance_validity,
        center_ids=centers,
        scored_frames=(20, 39, 58),
        gate_thresholds={"ungated": -np.inf},
    )

    assert [update["selected_backbone"] for update in report["updates"]] == [
        "physical_prior",
        "persistence",
        "persistence",
    ]
    for update_index, update in enumerate(report["updates"], start=1):
        assert update["gates"]["ungated"]["rbf_state_update_count_by_backbone"] == {
            "physical_prior": update_index,
            "persistence": update_index,
        }


def test_insufficient_selector_support_defaults_to_persistence_with_legacy_ablation() -> (
    None
):
    frame_count = 76
    point_count = 5
    centers = np.array([0, 1, 2])
    frame_zero = np.stack(
        (np.arange(point_count), np.zeros(point_count), np.ones(point_count)), axis=1
    ).astype(np.float32)
    prior = np.repeat(frame_zero[None], frame_count, axis=0)
    persistence = prior.copy()
    prior[:, :, 1] = 0.1
    prior[0] = frame_zero
    measurement = np.full_like(prior, np.nan)
    measurement_validity = np.zeros((frame_count, point_count), dtype=bool)
    covariance = np.full((frame_count, point_count, 3, 3), np.nan)
    covariance_validity = np.zeros((frame_count, point_count), dtype=bool)

    report, trajectories = evaluate_covariance_gated_arrays(
        prior,
        persistence,
        prior.copy(),
        np.ones((frame_count, point_count), dtype=bool),
        np.ones((frame_count, point_count), dtype=bool),
        measurement,
        measurement_validity,
        covariance,
        covariance_validity,
        center_ids=centers,
        scored_frames=(20, 39, 58),
        gate_thresholds={"ungated": -np.inf},
    )

    assert report["observed_backbone_selector"]["insufficient_support_count"] == 3
    assert report["observed_backbone_selector"]["persistence_count"] == 3
    scored = np.asarray(
        [*range(20, 38), *range(39, 57), *range(58, frame_count)], dtype=np.int64
    )
    np.testing.assert_array_equal(
        trajectories[SELECTED_BACKBONE_ARM][scored], persistence[scored]
    )
    np.testing.assert_array_equal(
        trajectories[LEGACY_SELECTED_BACKBONE_ARM][scored], prior[scored]
    )
