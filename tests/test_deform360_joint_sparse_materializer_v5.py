from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.deform360_joint_sparse_materializer_v5 import (
    Deform360JointSparseContactRowsV5,
    Deform360JointSparseExtractionConfigV5,
    Deform360JointSparsePrefixFitV5,
    Deform360JointSparseVisualWindowRowsV5,
    extract_deform360_joint_sparse_visual_rows_v5,
    materialize_deform360_joint_sparse_prediction_v5,
)
from bayesian_phystwin.deform360_joint_sparse_prediction_v5 import (
    VT2_VISUOTACTILE_UNGUARDED,
    run_deform360_joint_sparse_prediction_v5,
)


def _digest(character: str) -> str:
    return character * 64


def _fit() -> Deform360JointSparsePrefixFitV5:
    return Deform360JointSparsePrefixFitV5(
        fit_object_ids=tuple(f"source-{index}" for index in range(9)),
        source_artifact_ids={"fit/prefix.json": _digest("a")},
        maximum_association_distance_m=0.080,
    )


def _physical() -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(42)
    initial = generator.normal(size=(16, 3)) * np.asarray([0.055, 0.045, 0.035])
    initial += np.asarray([0.02, -0.01, 0.80])
    physical = np.stack(
        [
            initial,
            initial + np.asarray([0.001, 0.000, 0.000]),
            initial + np.asarray([0.002, 0.001, 0.000]),
            initial + np.asarray([0.004, 0.002, 0.001]),
        ]
    ).astype(np.float64)
    persistence = np.broadcast_to(initial, physical.shape).copy()
    return physical, persistence


def _visual_windows(
    physical: np.ndarray,
    *,
    cameras: tuple[str, ...] = ("camera-0", "camera-1"),
    point_shift_m: np.ndarray | None = None,
) -> tuple[Deform360JointSparseVisualWindowRowsV5, ...]:
    shift = np.zeros(3) if point_shift_m is None else np.asarray(point_shift_m)
    covariance = np.broadcast_to(2.5e-5 * np.eye(3), (16, 3, 3)).copy()
    digest_characters = ("b", "c", "d", "e", "f", "0")
    windows: list[Deform360JointSparseVisualWindowRowsV5] = []
    for camera_index, camera in enumerate(cameras):
        for frame in (0, 1):
            window_id = f"{camera}-window-{frame}"
            windows.append(
                Deform360JointSparseVisualWindowRowsV5(
                    camera_id=camera,
                    window_id=window_id,
                    frame_indices=np.full(16, frame, dtype=np.int64),
                    pixel_yx=np.column_stack(
                        (np.arange(16), np.arange(16) + frame)
                    ),
                    point_world_m=physical[frame] + shift,
                    point_covariance_m2=covariance,
                    source_confidence=np.full(16, 0.9),
                    mask_distance_pixels=np.full(16, 8.0),
                    overlap_disagreement_m=np.full(16, 0.002),
                    contributor_count=np.full(16, 3, dtype=np.int64),
                    source_artifact_ids={
                        f"visual/{camera_index}/{frame}.npz": _digest(
                            digest_characters[camera_index * 2 + frame]
                        )
                    },
                )
            )
    return tuple(windows)


def _contact(physical: np.ndarray) -> Deform360JointSparseContactRowsV5:
    return Deform360JointSparseContactRowsV5(
        frame_indices=np.asarray([0, 1], dtype=np.int64),
        observed_point_world_m=np.stack((physical[0, 0], physical[1, 1])),
        graph_node_indices=np.asarray([[0, 2], [1, 3]], dtype=np.int64),
        graph_node_weights=np.asarray([[0.8, 0.2], [0.7, 0.3]]),
        covariance_m2=np.broadcast_to(1e-5 * np.eye(3), (2, 3, 3)).copy(),
        prior_reliability=np.asarray([0.95, 0.90]),
        correlation_group_ids=("contact-0", "contact-1"),
        source_artifact_ids={"contact/anchors.npz": _digest("f")},
    )


def _materialize(
    physical: np.ndarray,
    persistence: np.ndarray,
    *,
    windows: tuple[Deform360JointSparseVisualWindowRowsV5, ...] | None = None,
    contact: Deform360JointSparseContactRowsV5 | None = None,
    association_candidate_count: int = 4,
):
    return materialize_deform360_joint_sparse_prediction_v5(
        object_id="development-object",
        episode_id=1,
        stratum="sheet",
        physical_prediction_m=physical,
        persistence_m=persistence,
        last_causal_residual_m=np.zeros_like(physical[0]),
        physical_mode="warp_twin",
        causal_frame_stop=2,
        evaluation_frame_range_half_open=(2, 4),
        visual_windows=_visual_windows(physical) if windows is None else windows,
        contact_rows=contact,
        fit=_fit(),
        implementation_revision="1" * 40,
        source_artifact_ids={"physical/b0.npz": _digest("9")},
        association_candidate_count=association_candidate_count,
        spatial_cluster_size_m=0.005,
    )


def test_dense_extraction_is_prefix_only_and_preserves_metric_covariance() -> None:
    fit = _fit()
    points = np.zeros((2, 8, 8, 3), dtype=np.float64)
    points[..., 2] = 0.8
    points[0, 0, 0] = np.nan
    valid = np.ones((2, 8, 8), dtype=np.bool_)
    valid[0, 0, 0] = False
    mask = np.ones_like(valid)
    covariance = np.broadcast_to(7e-6 * np.eye(3), (2, 8, 8, 3, 3)).copy()
    extracted = extract_deform360_joint_sparse_visual_rows_v5(
        camera_id="camera-0",
        window_id="window-0",
        frame_indices=np.asarray([0, 1]),
        point_map_world_m=points,
        valid_mask=valid,
        object_mask=mask,
        causal_frame_stop=2,
        fit=fit,
        source_artifact_ids={"visual/window.npz": _digest("b")},
        point_covariance_m2=covariance,
        source_confidence=np.full(valid.shape, 0.75),
        overlap_disagreement_m=np.full(valid.shape, 0.003),
        contributor_count=np.full(valid.shape, 2, dtype=np.uint16),
        config=Deform360JointSparseExtractionConfigV5(
            measurement_stride_pixels=2,
            maximum_rows_per_window=12,
        ),
    )
    assert len(extracted.frame_indices) == 12
    assert np.all(extracted.frame_indices < 2)
    np.testing.assert_allclose(
        extracted.point_covariance_m2,
        np.broadcast_to(7e-6 * np.eye(3), (12, 3, 3)),
    )
    np.testing.assert_array_equal(extracted.contributor_count, 2)
    with pytest.raises(ValueError, match="causal prefix"):
        extract_deform360_joint_sparse_visual_rows_v5(
            camera_id="camera-0",
            window_id="window-0",
            frame_indices=np.asarray([0, 2]),
            point_map_world_m=points,
            valid_mask=valid,
            object_mask=mask,
            causal_frame_stop=2,
            fit=fit,
            source_artifact_ids={"visual/window.npz": _digest("b")},
        )


def test_prior_perception_reliability_is_independent_of_physical_residual() -> None:
    physical, persistence = _physical()
    windows = _visual_windows(physical)
    nominal = _materialize(physical, persistence, windows=windows)
    shifted_physical = physical + np.asarray([0.025, 0.0, 0.0])
    shifted = _materialize(shifted_physical, persistence, windows=windows)
    np.testing.assert_array_equal(
        nominal.problem.observation_batch.prior_reliability,
        shifted.problem.observation_batch.prior_reliability,
    )
    assert (
        nominal.problem.observation_batch.metadata[
            "prior_reliability_uses_physical_innovation"
        ]
        is False
    )


def test_duplicate_correlated_camera_does_not_increase_effective_mass() -> None:
    physical, persistence = _physical()
    two = _materialize(
        physical,
        persistence,
        windows=_visual_windows(physical, cameras=("camera-0", "camera-1")),
    )
    three = _materialize(
        physical,
        persistence,
        windows=_visual_windows(
            physical,
            cameras=("camera-0", "camera-1", "camera-copy"),
        ),
    )
    assert three.visual_row_count > two.visual_row_count
    assert three.admission.effective_row_weight_sum == pytest.approx(
        two.admission.effective_row_weight_sum,
        rel=1e-12,
        abs=1e-12,
    )
    assert (
        three.problem.observation_batch.metadata[
            "unknown_cross_view_correlation_treatment"
        ]
        == "camera-independent-physical-frame-voxel-group-power-cap-v1"
    )


def test_assignment_mixture_spread_increases_observation_covariance() -> None:
    physical, persistence = _physical()
    physical = physical.copy()
    physical[:, 1] = physical[:, 0] + np.asarray([0.010, 0.0, 0.0])
    windows = list(_visual_windows(physical))
    midpoint = 0.5 * (physical[0, 0] + physical[0, 1])
    first = windows[0]
    points = first.point_world_m.copy()
    points[0] = midpoint
    windows[0] = replace(first, point_world_m=points)
    single = _materialize(
        physical,
        persistence,
        windows=tuple(windows),
        association_candidate_count=1,
    )
    mixture = _materialize(
        physical,
        persistence,
        windows=tuple(windows),
        association_candidate_count=2,
    )
    single_trace = float(
        np.trace(single.problem.observation_batch.observation_covariance_m2[0])
    )
    mixture_trace = float(
        np.trace(mixture.problem.observation_batch.observation_covariance_m2[0])
    )
    assert mixture_trace > single_trace
    assert (
        mixture.problem.observation_batch.metadata[
            "assignment_mixture_spread_in_covariance"
        ]
        is True
    )


def test_contact_rows_are_metric_anchors_and_do_not_change_visual_prior() -> None:
    physical, persistence = _physical()
    visual_only = _materialize(physical, persistence)
    with_contact = _materialize(
        physical,
        persistence,
        contact=_contact(physical),
    )
    np.testing.assert_array_equal(
        visual_only.problem.observation_batch.prior_reliability,
        with_contact.problem.observation_batch.prior_reliability,
    )
    assert with_contact.contact_row_count == 2
    assert with_contact.problem.observation_batch.anchor_innovation_m is not None
    assert with_contact.problem.observation_batch.anchor_covariance_m2 is not None


def test_gross_innovation_reaches_the_final_robust_likelihood_once() -> None:
    physical, persistence = _physical()
    clean_windows = _visual_windows(physical)
    outlier_windows = list(clean_windows)
    first = outlier_windows[0]
    points = first.point_world_m.copy()
    points[0] += np.asarray([0.060, 0.0, 0.0])
    outlier_windows[0] = replace(first, point_world_m=points)
    clean = _materialize(
        physical,
        persistence,
        windows=clean_windows,
        contact=_contact(physical),
    )
    outlier = _materialize(
        physical,
        persistence,
        windows=tuple(outlier_windows),
        contact=_contact(physical),
    )
    np.testing.assert_array_equal(
        clean.problem.observation_batch.prior_reliability,
        outlier.problem.observation_batch.prior_reliability,
    )
    clean_result = run_deform360_joint_sparse_prediction_v5(clean.problem)
    outlier_result = run_deform360_joint_sparse_prediction_v5(outlier.problem)
    clean_weights = clean_result.inference_results[
        VT2_VISUOTACTILE_UNGUARDED
    ].robust_weights
    outlier_weights = outlier_result.inference_results[
        VT2_VISUOTACTILE_UNGUARDED
    ].robust_weights
    assert np.min(outlier_weights) < np.min(clean_weights)
    assert (
        outlier.problem.observation_batch.metadata[
            "state_innovation_robust_processing_count"
        ]
        == 1
    )


def test_materialization_declares_public_data_without_human_approval() -> None:
    physical, persistence = _physical()
    result = _materialize(physical, persistence, contact=_contact(physical))
    metadata = result.problem.observation_batch.metadata
    assert metadata["released_real_world_measurements_used"] is True
    assert metadata["human_approval_required"] is False
    assert metadata["new_measurements_required"] is False
    assert metadata["future_object_observations_used"] is False
    assert metadata["confirmation_payloads_opened"] is False
