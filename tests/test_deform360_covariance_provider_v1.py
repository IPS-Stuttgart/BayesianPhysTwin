from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.deform360_covariance_provider_v1 import (
    Deform360CausalResidualHistoryV1,
    Deform360CovarianceDonorSupportConfigV1,
    Deform360ObservationSplitV1,
    build_deform360_covariance_only_forecast_v1,
    estimate_deform360_causal_residual_history_v1,
    plan_deform360_camera_partition_v1,
)
from bayesian_phystwin.deform360_joint_sparse_materializer_v5 import (
    Deform360JointSparseVisualWindowRowsV5,
)
from scripts.science.run_deform360_covariance_provider_dry_run_v1 import (
    run_source_only_dry_run,
)


def _visual_window(
    *,
    camera_id: str,
    frame_points: tuple[tuple[int, int, np.ndarray], ...],
) -> Deform360JointSparseVisualWindowRowsV5:
    frames = np.asarray([row[0] for row in frame_points], dtype=np.int64)
    identity_indices = np.asarray([row[1] for row in frame_points], dtype=np.int64)
    points = np.asarray([row[2] for row in frame_points], dtype=np.float64)
    count = len(frames)
    return Deform360JointSparseVisualWindowRowsV5(
        camera_id=camera_id,
        window_id=f"window-{camera_id}",
        frame_indices=frames,
        pixel_yx=np.column_stack((identity_indices, identity_indices)),
        point_world_m=points,
        point_covariance_m2=np.broadcast_to(
            np.eye(3) * 1e-6,
            (count, 3, 3),
        ),
        source_confidence=np.ones(count),
        mask_distance_pixels=np.ones(count),
        overlap_disagreement_m=np.zeros(count),
        contributor_count=np.ones(count, dtype=np.int64),
        source_artifact_ids={"fixture/input": "1" * 64},
    )


def _history() -> Deform360CausalResidualHistoryV1:
    physical = np.broadcast_to(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.3, 0.0, 0.0],
            ]
        )[None],
        (3, 4, 3),
    ).copy()
    rows = (
        (10, 0, physical[0, 0] + [0.001, 0.0, 0.0]),
        (10, 1, physical[0, 1] + [0.002, 0.0, 0.0]),
        (11, 0, physical[1, 0] + [0.002, 0.0, 0.0]),
        (11, 1, physical[1, 1] + [0.003, 0.0, 0.0]),
        (12, 0, physical[2, 0] + [0.003, 0.0, 0.0]),
    )
    return estimate_deform360_causal_residual_history_v1(
        visual_windows=(_visual_window(camera_id="provider-a", frame_points=rows),),
        physical_prediction_world_m=physical,
        frame_indices=np.arange(10, 13),
        material_identity_ids=("node-0", "node-1", "node-2", "node-3"),
        source_artifact_ids={"physical/prefix": "2" * 64},
        association_candidate_count=1,
    )


def _forecast_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    mean = np.arange(36, dtype=np.float32).reshape(3, 4, 3) / 1000.0
    fallback = np.broadcast_to(
        np.eye(3, dtype=np.float64) * 0.02**2,
        (3, 4, 3, 3),
    ).copy()
    return mean, fallback, np.arange(13, 16), ("early", "middle", "late")


def test_causal_history_preserves_identity_order_and_missingness() -> None:
    history = _history()

    assert history.material_identity_ids == (
        "node-0",
        "node-1",
        "node-2",
        "node-3",
    )
    np.testing.assert_array_equal(
        history.valid_mask,
        [
            [True, True, False, False],
            [True, True, False, False],
            [True, False, False, False],
        ],
    )
    assert np.all(history.residual_world_m[~history.valid_mask] == 0.0)
    assert history.coordinate_frame == "deform360_world"
    assert history.position_units == "m"
    assert history.covariance_units == "m^2"


def test_duplicate_correlated_window_does_not_change_residual_history() -> None:
    original = _history()
    physical = np.broadcast_to(
        np.asarray(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]
        )[None],
        (3, 4, 3),
    ).copy()
    rows = tuple(
        (int(frame), int(identity), physical[frame - 10, identity] + [0.001, 0, 0])
        for frame, identity in ((10, 0), (10, 1), (11, 0), (11, 1), (12, 0))
    )
    window = _visual_window(camera_id="provider-a", frame_points=rows)
    duplicate = replace(window, window_id="duplicate")
    repeated = estimate_deform360_causal_residual_history_v1(
        visual_windows=(window, duplicate),
        physical_prediction_world_m=physical,
        frame_indices=np.arange(10, 13),
        material_identity_ids=("node-0", "node-1", "node-2", "node-3"),
        source_artifact_ids={"physical/prefix": "2" * 64},
        association_candidate_count=1,
    )

    np.testing.assert_allclose(
        repeated.residual_world_m,
        np.asarray(
            [
                [[0.001, 0, 0], [0.001, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0.001, 0, 0], [0.001, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0.001, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ]
        ),
        atol=1e-14,
        rtol=0.0,
    )
    np.testing.assert_array_equal(repeated.valid_mask, original.valid_mask)


def test_covariance_donor_keeps_mean_bytes_and_falls_back_per_identity() -> None:
    history = _history()
    mean, fallback, future, labels = _forecast_inputs()
    result = build_deform360_covariance_only_forecast_v1(
        reference_mean_world_m=mean,
        fallback_covariance_world_m2=fallback,
        future_frame_indices=future,
        horizon_labels=labels,
        history=history,
    )

    assert result.case_donor_admitted is True
    np.testing.assert_array_equal(result.empirical_donor_mask, [True, True, False, False])
    np.testing.assert_array_equal(result.prior_only_mask, [False, False, True, True])
    assert result.mean_world_m.dtype == mean.dtype
    assert result.mean_world_m.tobytes() == mean.tobytes()
    assert np.array_equal(result.covariance_world_m2[:, 2:], fallback[:, 2:])
    assert np.min(np.linalg.eigvalsh(result.covariance_world_m2)) >= 0.0


def test_weak_case_and_all_zero_updates_are_exact_fallback() -> None:
    history = _history()
    valid = np.zeros_like(history.valid_mask)
    residual = np.zeros_like(history.residual_world_m)
    unsupported = replace(history, valid_mask=valid, residual_world_m=residual)
    mean, fallback, future, labels = _forecast_inputs()
    result = build_deform360_covariance_only_forecast_v1(
        reference_mean_world_m=mean,
        fallback_covariance_world_m2=fallback,
        future_frame_indices=future,
        horizon_labels=labels,
        history=unsupported,
    )

    assert result.case_donor_admitted is False
    assert result.fallback_reason == (
        "insufficient-observed-frames+insufficient-empirical-identities"
    )
    assert not np.any(result.empirical_donor_mask)
    assert np.array_equal(result.covariance_world_m2, fallback)
    assert result.covariance_world_m2.tobytes() == fallback.tobytes()


def test_support_gate_is_registered_and_not_inferred_from_prior_covariance() -> None:
    history = _history()
    mean, fallback, future, labels = _forecast_inputs()
    strict = Deform360CovarianceDonorSupportConfigV1(
        minimum_observed_frame_count=3,
        minimum_updates_per_identity=3,
        minimum_empirical_identity_fraction=0.5,
    )
    result = build_deform360_covariance_only_forecast_v1(
        reference_mean_world_m=mean,
        fallback_covariance_world_m2=fallback,
        future_frame_indices=future,
        horizon_labels=labels,
        history=history,
        support_config=strict,
    )

    assert result.update_count.tolist() == [3, 2, 0, 0]
    assert result.case_donor_admitted is False
    assert not np.any(result.empirical_donor_mask)


def test_history_rejects_nonzero_values_marked_missing() -> None:
    history = _history()
    residual = np.array(history.residual_world_m, copy=True)
    residual[0, 3, 0] = 1.0

    with pytest.raises(ValueError, match="never filled"):
        replace(history, residual_world_m=residual)


def test_observation_split_rejects_shared_views_or_reconstruction() -> None:
    split = Deform360ObservationSplitV1(
        provider_camera_ids=("camera-a", "camera-b"),
        scoring_camera_ids=("camera-c", "camera-d"),
        provider_reconstruction_artifact_id="provider-reconstruction",
        scoring_reconstruction_artifact_id="scoring-reconstruction",
    )
    assert set(split.provider_camera_ids).isdisjoint(split.scoring_camera_ids)

    with pytest.raises(ValueError, match="cameras must be disjoint"):
        replace(split, scoring_camera_ids=("camera-b", "camera-c"))
    with pytest.raises(ValueError, match="distinct artifacts"):
        replace(
            split,
            scoring_reconstruction_artifact_id="provider-reconstruction",
        )


def test_camera_partition_is_names_only_deterministic_and_disjoint() -> None:
    cameras = ("camera-d", "camera-a", "camera-c", "camera-b", "camera-e")
    first = plan_deform360_camera_partition_v1(
        camera_ids=cameras,
        object_session_hash="3" * 64,
    )
    second = plan_deform360_camera_partition_v1(
        camera_ids=tuple(reversed(cameras)),
        object_session_hash="3" * 64,
    )

    assert first == second
    assert set(first[0]).isdisjoint(first[1])
    assert set(first[0]) | set(first[1]) == set(cameras)
    with pytest.raises(ValueError, match="at least four"):
        plan_deform360_camera_partition_v1(
            camera_ids=("camera-a", "camera-b", "camera-c"),
            object_session_hash="3" * 64,
        )


def test_future_horizon_must_start_after_prefix() -> None:
    history = _history()
    mean, fallback, _, labels = _forecast_inputs()

    with pytest.raises(ValueError, match="after the causal prefix"):
        build_deform360_covariance_only_forecast_v1(
            reference_mean_world_m=mean,
            fallback_covariance_world_m2=fallback,
            future_frame_indices=np.arange(12, 15),
            horizon_labels=labels,
            history=history,
        )


def test_source_only_end_to_end_dry_run_passes_without_target_access() -> None:
    result = run_source_only_dry_run()

    assert result["gate_passed"] is True
    assert result["source_only"] is True
    assert result["target_roster_read"] is False
    assert result["target_payload_read"] is False
    assert result["target_outcome_read"] is False
    assert result["history"]["missing_entries_are_zero"] is True
    assert result["observation_split"]["camera_sets_disjoint"] is True
    assert result["admitted_candidate"]["mean_byte_identical"] is True
    assert result["failed_support_fallback"]["covariance_byte_identical"] is True
