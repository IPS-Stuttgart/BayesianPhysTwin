from __future__ import annotations

import inspect

import numpy as np

from bayesian_phystwin.cpd_registration import NonrigidCpdConfig
from bayesian_phystwin.deform360_fresh_pairwise_prediction import (
    CANDIDATE_ARM,
    SELECTED_RAW_ARM,
    predict_fresh_pairwise_arrays,
)
from bayesian_phystwin.deform360_raw_pairwise_correspondence_diagnostic import (
    CLIQUE_RBF_ARM,
    SELECTED_RAW_ARM as DEVELOPMENT_SELECTED_RAW_ARM,
    evaluate_raw_pairwise_correspondence_arrays,
)


def _arrays(seed: int, *, contaminated: bool = False) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    frame_count = 76
    point_count = 32
    frame_zero = rng.normal(0.0, 0.08, size=(point_count, 3)).astype(np.float32)
    physical = np.repeat(frame_zero[None], frame_count, axis=0)
    for frame in range(frame_count):
        physical[frame, :, 0] += np.float32(0.0015 * frame)
        physical[frame, :, 1] += (
            np.sin(np.arange(point_count) * 0.3) * 0.0002 * frame
        ).astype(np.float32)
    persistence = np.repeat(frame_zero[None], frame_count, axis=0)
    centers = np.arange(16, dtype=np.int64)
    measurement = np.full_like(physical, np.nan)
    visible = np.zeros((frame_count, point_count), dtype=bool)
    valid = visible.copy()
    for frame in (19, 38, 57):
        measurement[frame, centers] = physical[frame, centers]
        measurement[frame, centers, 2] += np.float32(0.004)
        if contaminated:
            measurement[frame, :8] = physical[frame, np.roll(np.arange(8), 1)]
        visible[frame, centers] = True
        valid[frame, centers] = True
    return physical, persistence, measurement, visible, valid, centers


def test_target_free_signature_has_no_target_or_outcome() -> None:
    parameters = inspect.signature(predict_fresh_pairwise_arrays).parameters

    assert "target" not in parameters
    assert "outcome" not in parameters


def test_target_free_predictor_is_exactly_equivalent_to_frozen_development_arm() -> (
    None
):
    for seed, contaminated in ((2, False), (9, True)):
        physical, persistence, measurement, visible, valid, centers = _arrays(
            seed, contaminated=contaminated
        )
        report, fresh = predict_fresh_pairwise_arrays(
            physical,
            persistence,
            measurement,
            visible,
            valid,
            center_ids=centers,
        )
        target = physical.copy()
        target_visibility = np.ones(target.shape[:2], dtype=bool)
        development_report, development = (
            evaluate_raw_pairwise_correspondence_arrays(
                physical,
                persistence,
                target,
                target_visibility,
                target_visibility,
                measurement,
                visible,
                valid,
                center_ids=centers,
                scored_frames=tuple(range(20, 76)),
                cpd_config=NonrigidCpdConfig(maximum_iterations=3),
            )
        )

        np.testing.assert_array_equal(
            fresh[CANDIDATE_ARM], development[CLIQUE_RBF_ARM]
        )
        np.testing.assert_array_equal(
            fresh[SELECTED_RAW_ARM],
            development[DEVELOPMENT_SELECTED_RAW_ARM],
        )
        assert [
            item["selected_backbone"] for item in report["updates"]
        ] == [
            item["selected_backbone"] for item in development_report["updates"]
        ]
        assert [
            item["selected_pairwise_gate"] for item in report["updates"]
        ] == [
            item["selected_pairwise_gate"]
            for item in development_report["updates"]
        ]


def test_rejected_update_preserves_selected_backbone_bit_exactly() -> None:
    physical, persistence, measurement, visible, valid, centers = _arrays(
        15, contaminated=True
    )

    report, arrays = predict_fresh_pairwise_arrays(
        physical,
        persistence,
        measurement,
        visible,
        valid,
        center_ids=centers,
    )

    for update in report["updates"]:
        if update["selected_pairwise_gate"]["accepted"]:
            continue
        start = int(update["frame"]) + 1
        stop = int(update["interval_end_exclusive"])
        np.testing.assert_array_equal(
            arrays[CANDIDATE_ARM][start:stop],
            arrays[SELECTED_RAW_ARM][start:stop],
        )
        assert update["selected_pairwise_gate"]["bit_exact_raw_fallback"] is True
