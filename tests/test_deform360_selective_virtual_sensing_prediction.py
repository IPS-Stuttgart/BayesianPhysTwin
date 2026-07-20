from __future__ import annotations

import inspect

import numpy as np

from bayesian_phystwin.cpd_registration import NonrigidCpdConfig
from bayesian_phystwin.deform360_raw_pairwise_correspondence_diagnostic import (
    CPD_ARM,
    PERSISTENCE_CLIQUE_RBF_ARM,
    UNGATED_RBF_ARM,
    evaluate_raw_pairwise_correspondence_arrays,
)
from bayesian_phystwin.deform360_selective_virtual_sensing_prediction import (
    predict_persistence_control_arrays,
    predict_persistence_pairwise_rbf_arrays,
)


def _inputs(seed: int, *, corrupt: bool) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    point_count = 24
    frame_count = 76
    frame_zero = rng.uniform(-0.25, 0.25, size=(point_count, 3)).astype(np.float32)
    persistence = np.repeat(frame_zero[None], frame_count, axis=0)
    physical = persistence.copy()
    physical[:, :, 0] += np.arange(frame_count, dtype=np.float32)[:, None] * 0.002
    target = physical.copy()
    visible = np.ones((frame_count, point_count), dtype=bool)
    valid = visible.copy()
    measurement = np.full_like(persistence, np.nan)
    measurement_visible = np.zeros((frame_count, point_count), dtype=bool)
    measurement_valid = measurement_visible.copy()
    centers = np.arange(16, dtype=np.int64)
    for frame in (19, 38, 57):
        measurement[frame, centers] = target[frame, centers]
        if corrupt:
            bad = centers[:8]
            measurement[frame, bad] = target[frame, np.roll(bad, 1)]
        measurement_visible[frame, centers] = True
        measurement_valid[frame, centers] = True
    return (
        physical,
        persistence,
        target,
        visible,
        valid,
        measurement,
        measurement_visible,
        measurement_valid,
        centers,
    )


def test_primary_predictor_signature_cannot_accept_outcomes() -> None:
    parameters = inspect.signature(
        predict_persistence_pairwise_rbf_arrays
    ).parameters

    assert "target" not in parameters
    assert "outcome" not in parameters
    assert "physical_prior" not in parameters


def test_target_free_predictor_matches_development_primary_arm_bit_exactly() -> None:
    for corrupt in (False, True):
        values = _inputs(31, corrupt=corrupt)
        (
            physical,
            persistence,
            target,
            visible,
            valid,
            measurement,
            measurement_visible,
            measurement_valid,
            centers,
        ) = values
        _, development = evaluate_raw_pairwise_correspondence_arrays(
            physical,
            persistence,
            target,
            visible,
            valid,
            measurement,
            measurement_visible,
            measurement_valid,
            center_ids=centers,
            scored_frames=tuple(range(20, 76)),
            cpd_config=NonrigidCpdConfig(maximum_iterations=3),
        )
        report, prospective = predict_persistence_pairwise_rbf_arrays(
            persistence,
            measurement,
            measurement_visible,
            measurement_valid,
            center_ids=centers,
        )

        np.testing.assert_array_equal(
            prospective, development[PERSISTENCE_CLIQUE_RBF_ARM]
        )
        assert report["information_boundary"]["target_argument_accepted"] is False


def test_insufficient_support_is_bit_exact_persistence() -> None:
    values = list(_inputs(33, corrupt=False))
    persistence = values[1]
    measurement = values[5]
    measurement_visible = values[6]
    measurement_valid = values[7]
    centers = values[8]
    measurement_visible[:] = False
    measurement_valid[:] = False

    report, prediction = predict_persistence_pairwise_rbf_arrays(
        persistence,
        measurement,
        measurement_visible,
        measurement_valid,
        center_ids=centers,
    )

    np.testing.assert_array_equal(prediction, persistence)
    assert all(
        update["bit_exact_persistence_fallback"] for update in report["updates"]
    )


def test_target_free_controls_match_development_controls_bit_exactly() -> None:
    values = _inputs(41, corrupt=True)
    (
        _,
        persistence,
        target,
        visible,
        valid,
        measurement,
        measurement_visible,
        measurement_valid,
        centers,
    ) = values
    cpd_config = NonrigidCpdConfig(maximum_iterations=3)
    _, development = evaluate_raw_pairwise_correspondence_arrays(
        persistence,
        persistence,
        target,
        visible,
        valid,
        measurement,
        measurement_visible,
        measurement_valid,
        center_ids=centers,
        scored_frames=tuple(range(20, 76)),
        cpd_config=cpd_config,
    )

    report, prospective = predict_persistence_control_arrays(
        persistence,
        measurement,
        measurement_visible,
        measurement_valid,
        center_ids=centers,
        cpd_config=cpd_config,
    )

    np.testing.assert_array_equal(
        prospective[UNGATED_RBF_ARM], development[UNGATED_RBF_ARM]
    )
    np.testing.assert_array_equal(prospective[CPD_ARM], development[CPD_ARM])
    assert report["information_boundary"]["target_argument_accepted"] is False
