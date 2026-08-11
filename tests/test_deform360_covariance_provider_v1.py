from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.contracts.fixed_anchor import FixedBayesianAnchorConfigV1
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
from bayesian_phystwin.endpoint_model_average import ModelAveragedEndpointConfigV1
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


def _split(
    *,
    provider_camera_ids: tuple[str, ...] = ("provider-a",),
) -> Deform360ObservationSplitV1:
    return Deform360ObservationSplitV1(
        provider_camera_ids=provider_camera_ids,
        scoring_camera_ids=("scoring-a", "scoring-b"),
        provider_reconstruction_artifact_id="a" * 64,
        scoring_reconstruction_artifact_id="b" * 64,
    )


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


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
        observation_split=_split(),
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


def _forecast(
    *,
    history: Deform360CausalResidualHistoryV1,
    support_config: Deform360CovarianceDonorSupportConfigV1 | None = None,
    endpoint_config: ModelAveragedEndpointConfigV1 | None = None,
):
    mean, fallback, future, labels = _forecast_inputs()
    return build_deform360_covariance_only_forecast_v1(
        reference_mean_world_m=mean,
        fallback_covariance_world_m2=fallback,
        future_frame_indices=future,
        horizon_labels=labels,
        history=history,
        observation_split=_split(),
        registered_reference_mean_sha256=_array_sha256(mean),
        support_config=support_config,
        endpoint_config=endpoint_config,
    )


def _single_component_config() -> ModelAveragedEndpointConfigV1:
    return ModelAveragedEndpointConfigV1(
        components=(
            FixedBayesianAnchorConfigV1(
                process_std_m=0.001,
                observation_std_m=0.001,
                initial_std_m=0.010,
                inlier_prior=0.95,
                outlier_variance_multiplier=100.0,
            ),
        )
    )


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
    assert np.all(
        history.observation_covariance_world_m2[~history.valid_mask] == 0.0
    )
    assert np.all(history.prior_reliability[~history.valid_mask] == 0.0)
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
        observation_split=_split(),
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
    np.testing.assert_allclose(
        repeated.observation_covariance_world_m2,
        original.observation_covariance_world_m2,
        atol=1e-14,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        repeated.prior_reliability,
        original.prior_reliability,
        atol=1e-14,
        rtol=0.0,
    )


def test_far_points_cannot_manufacture_support_and_force_exact_fallback() -> None:
    physical = np.broadcast_to(
        np.asarray(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]
        )[None],
        (2, 4, 3),
    ).copy()
    rows = (
        (0, 0, np.asarray([17.0, 0.0, 0.0])),
        (1, 0, np.asarray([17.0, 0.0, 0.0])),
    )
    history = estimate_deform360_causal_residual_history_v1(
        visual_windows=(_visual_window(camera_id="provider-a", frame_points=rows),),
        physical_prediction_world_m=physical,
        frame_indices=np.arange(2),
        material_identity_ids=("node-0", "node-1", "node-2", "node-3"),
        observation_split=_split(),
        source_artifact_ids={"physical/prefix": "2" * 64},
    )
    result = _forecast(history=history)

    assert not np.any(history.valid_mask)
    assert not np.any(result.update_count)
    assert result.case_donor_admitted is False
    assert result.covariance_world_m2.tobytes() == (
        result.fallback_covariance_world_m2.tobytes()
    )


def test_ambiguous_midpoint_uses_candidate_specific_residual_and_spread() -> None:
    physical = np.broadcast_to(
        np.asarray(
            [[0.0, 0.0, 0.0], [0.010, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]
        )[None],
        (2, 4, 3),
    ).copy()
    midpoint = np.asarray([0.005, 0.0, 0.0])
    rows = ((0, 0, midpoint), (1, 0, midpoint))
    history = estimate_deform360_causal_residual_history_v1(
        visual_windows=(_visual_window(camera_id="provider-a", frame_points=rows),),
        physical_prediction_world_m=physical,
        frame_indices=np.arange(2),
        material_identity_ids=("node-0", "node-1", "node-2", "node-3"),
        observation_split=_split(),
        source_artifact_ids={"physical/prefix": "2" * 64},
        association_candidate_count=2,
    )

    np.testing.assert_array_equal(
        history.valid_mask,
        [[True, True, False, False], [True, True, False, False]],
    )
    np.testing.assert_allclose(history.residual_world_m[:, 0, 0], 0.005)
    np.testing.assert_allclose(history.residual_world_m[:, 1, 0], -0.005)
    assert np.all(history.observation_covariance_world_m2[:, :2, 0, 0] > 25e-6)
    without_assignment_spread = np.zeros_like(
        history.observation_covariance_world_m2
    )
    without_assignment_spread[history.valid_mask] = np.eye(3) * 5e-6
    narrow_history = replace(
        history,
        observation_covariance_world_m2=without_assignment_spread,
    )
    config = _single_component_config()
    narrow = _forecast(history=narrow_history, endpoint_config=config)
    ambiguous = _forecast(history=history, endpoint_config=config)

    assert np.all(
        np.trace(ambiguous.covariance_world_m2[:, :2], axis1=-2, axis2=-1)
        >= np.trace(narrow.covariance_world_m2[:, :2], axis1=-2, axis2=-1)
        - 1e-12
    )
    assert (
        ambiguous.covariance_world_m2.tobytes()
        != narrow.covariance_world_m2.tobytes()
    )


def test_admitted_innovation_is_not_clipped_before_robust_endpoint() -> None:
    physical = np.asarray(
        [[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]]
    )
    rows = ((0, 0, np.asarray([0.039, 0.0, 0.0])),)
    history = estimate_deform360_causal_residual_history_v1(
        visual_windows=(_visual_window(camera_id="provider-a", frame_points=rows),),
        physical_prediction_world_m=physical,
        frame_indices=np.arange(1),
        material_identity_ids=("node-0", "node-1", "node-2", "node-3"),
        observation_split=_split(),
        source_artifact_ids={"physical/prefix": "2" * 64},
        association_candidate_count=1,
    )

    assert history.valid_mask[0, 0]
    assert history.residual_world_m[0, 0, 0] == pytest.approx(0.039)


def test_zero_cue_reliability_cannot_create_an_endpoint_update() -> None:
    physical = np.asarray(
        [[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]]
    )
    window = _visual_window(
        camera_id="provider-a",
        frame_points=((0, 0, np.asarray([0.001, 0.0, 0.0])),),
    )
    window = replace(window, source_confidence=np.zeros(1))
    history = estimate_deform360_causal_residual_history_v1(
        visual_windows=(window,),
        physical_prediction_world_m=physical,
        frame_indices=np.arange(1),
        material_identity_ids=("node-0", "node-1", "node-2", "node-3"),
        observation_split=_split(),
        source_artifact_ids={"physical/prefix": "2" * 64},
        association_candidate_count=1,
    )

    assert not np.any(history.valid_mask)


def test_state_residual_does_not_change_admitted_prior_reliability() -> None:
    physical = np.asarray(
        [[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]]
    )

    def build(offset_m: float) -> Deform360CausalResidualHistoryV1:
        return estimate_deform360_causal_residual_history_v1(
            visual_windows=(
                _visual_window(
                    camera_id="provider-a",
                    frame_points=(
                        (0, 0, np.asarray([offset_m, 0.0, 0.0])),
                    ),
                ),
            ),
            physical_prediction_world_m=physical,
            frame_indices=np.arange(1),
            material_identity_ids=("node-0", "node-1", "node-2", "node-3"),
            observation_split=_split(),
            source_artifact_ids={"physical/prefix": "2" * 64},
            association_candidate_count=1,
        )

    near = build(0.001)
    farther = build(0.030)

    assert near.valid_mask[0, 0] and farther.valid_mask[0, 0]
    assert farther.residual_world_m[0, 0, 0] > near.residual_world_m[0, 0, 0]
    assert farther.prior_reliability[0, 0] == pytest.approx(
        near.prior_reliability[0, 0]
    )


def test_metric_point_covariance_is_retained_without_changing_reliability() -> None:
    physical = np.asarray(
        [[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]]
    )
    base = _visual_window(
        camera_id="provider-a",
        frame_points=((0, 0, np.asarray([0.001, 0.0, 0.0])),),
    )

    def build(variance_m2: float) -> Deform360CausalResidualHistoryV1:
        window = replace(
            base,
            point_covariance_m2=np.eye(3)[None] * variance_m2,
        )
        return estimate_deform360_causal_residual_history_v1(
            visual_windows=(window,),
            physical_prediction_world_m=physical,
            frame_indices=np.arange(1),
            material_identity_ids=("node-0", "node-1", "node-2", "node-3"),
            observation_split=_split(),
            source_artifact_ids={"physical/prefix": "2" * 64},
            association_candidate_count=1,
        )

    low = build(1e-6)
    high = build(9e-6)

    assert high.observation_covariance_world_m2[0, 0, 0, 0] > (
        low.observation_covariance_world_m2[0, 0, 0, 0]
    )
    assert high.prior_reliability[0, 0] == pytest.approx(
        low.prior_reliability[0, 0]
    )


def test_larger_row_covariance_survives_into_forecast_covariance() -> None:
    history = _history()
    inflated = replace(
        history,
        observation_covariance_world_m2=(
            history.observation_covariance_world_m2 * 1000.0
        ),
    )
    config = _single_component_config()

    baseline = _forecast(history=history, endpoint_config=config)
    wider = _forecast(history=inflated, endpoint_config=config)
    empirical = baseline.empirical_donor_mask
    difference = (
        wider.covariance_world_m2[:, empirical]
        - baseline.covariance_world_m2[:, empirical]
    )

    assert baseline.update_count.tolist() == wider.update_count.tolist()
    assert baseline.case_donor_admitted and wider.case_donor_admitted
    assert np.min(np.linalg.eigvalsh(difference)) >= -1e-12
    assert np.any(np.linalg.eigvalsh(difference) > 1e-12)


def test_lower_cue_reliability_cannot_make_forecast_more_confident() -> None:
    history = _history()
    lower_reliability = replace(
        history,
        prior_reliability=history.prior_reliability * 0.1,
    )
    config = _single_component_config()

    baseline = _forecast(history=history, endpoint_config=config)
    wider = _forecast(history=lower_reliability, endpoint_config=config)
    empirical = baseline.empirical_donor_mask
    difference = (
        wider.covariance_world_m2[:, empirical]
        - baseline.covariance_world_m2[:, empirical]
    )

    assert baseline.update_count.tolist() == wider.update_count.tolist()
    assert baseline.case_donor_admitted and wider.case_donor_admitted
    assert np.min(np.linalg.eigvalsh(difference)) >= -1e-12
    assert np.any(np.linalg.eigvalsh(difference) > 1e-12)


def test_unclipped_gross_innovation_is_robustified_once() -> None:
    history = _history()
    residual_30 = np.array(history.residual_world_m, copy=True)
    residual_39 = np.array(history.residual_world_m, copy=True)
    residual_30[history.valid_mask, 0] = 0.030
    residual_39[history.valid_mask, 0] = 0.039
    thirty = replace(history, residual_world_m=residual_30)
    thirty_nine = replace(history, residual_world_m=residual_39)
    config = _single_component_config()

    forecast_30 = _forecast(history=thirty, endpoint_config=config)
    forecast_39 = _forecast(history=thirty_nine, endpoint_config=config)

    assert np.all(thirty_nine.residual_world_m[history.valid_mask, 0] == 0.039)
    assert forecast_30.update_count.tolist() == forecast_39.update_count.tolist()
    assert forecast_30.case_donor_admitted and forecast_39.case_donor_admitted
    assert (
        forecast_30.covariance_world_m2.tobytes()
        != forecast_39.covariance_world_m2.tobytes()
    )
    assert np.min(np.linalg.eigvalsh(forecast_39.covariance_world_m2)) >= 0.0


def test_covariance_donor_keeps_mean_bytes_and_falls_back_per_identity() -> None:
    history = _history()
    mean, fallback, future, labels = _forecast_inputs()
    result = build_deform360_covariance_only_forecast_v1(
        reference_mean_world_m=mean,
        fallback_covariance_world_m2=fallback,
        future_frame_indices=future,
        horizon_labels=labels,
        history=history,
        observation_split=_split(),
        registered_reference_mean_sha256=_array_sha256(mean),
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
    unsupported = replace(
        history,
        valid_mask=valid,
        residual_world_m=residual,
        observation_covariance_world_m2=np.zeros_like(
            history.observation_covariance_world_m2
        ),
        prior_reliability=np.zeros_like(history.prior_reliability),
    )
    mean, fallback, future, labels = _forecast_inputs()
    result = build_deform360_covariance_only_forecast_v1(
        reference_mean_world_m=mean,
        fallback_covariance_world_m2=fallback,
        future_frame_indices=future,
        horizon_labels=labels,
        history=unsupported,
        observation_split=_split(),
        registered_reference_mean_sha256=_array_sha256(mean),
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
        observation_split=_split(),
        registered_reference_mean_sha256=_array_sha256(mean),
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
        provider_reconstruction_artifact_id="a" * 64,
        scoring_reconstruction_artifact_id="b" * 64,
    )
    assert set(split.provider_camera_ids).isdisjoint(split.scoring_camera_ids)

    with pytest.raises(ValueError, match="cameras must be disjoint"):
        replace(split, scoring_camera_ids=("camera-b", "camera-c"))
    with pytest.raises(ValueError, match="distinct artifacts"):
        replace(
            split,
            scoring_reconstruction_artifact_id="a" * 64,
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        replace(split, provider_reconstruction_artifact_id="not-a-digest")


def test_history_and_forecast_enforce_registered_observation_split() -> None:
    history = _history()
    mean, fallback, future, labels = _forecast_inputs()
    changed_split = replace(
        _split(),
        scoring_reconstruction_artifact_id="c" * 64,
    )

    with pytest.raises(ValueError, match="registered observation split"):
        build_deform360_covariance_only_forecast_v1(
            reference_mean_world_m=mean,
            fallback_covariance_world_m2=fallback,
            future_frame_indices=future,
            horizon_labels=labels,
            history=history,
            observation_split=changed_split,
            registered_reference_mean_sha256=_array_sha256(mean),
        )
    with pytest.raises(ValueError, match="registered baseline digest"):
        build_deform360_covariance_only_forecast_v1(
            reference_mean_world_m=mean,
            fallback_covariance_world_m2=fallback,
            future_frame_indices=future,
            horizon_labels=labels,
            history=history,
            observation_split=_split(),
            registered_reference_mean_sha256="f" * 64,
        )


def test_history_is_window_order_invariant_and_rejects_duplicate_keys() -> None:
    physical = np.broadcast_to(
        np.asarray(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]
        )[None],
        (2, 4, 3),
    ).copy()
    first_window = _visual_window(
        camera_id="provider-a",
        frame_points=((0, 0, physical[0, 0] + [0.001, 0, 0]),),
    )
    second_window = _visual_window(
        camera_id="provider-b",
        frame_points=((1, 1, physical[1, 1] + [0.001, 0, 0]),),
    )
    split = _split(provider_camera_ids=("provider-a", "provider-b"))

    def build(windows):
        return estimate_deform360_causal_residual_history_v1(
            visual_windows=windows,
            physical_prediction_world_m=physical,
            frame_indices=np.arange(2),
            material_identity_ids=("node-0", "node-1", "node-2", "node-3"),
            observation_split=split,
            source_artifact_ids={"physical/prefix": "2" * 64},
            association_candidate_count=1,
        )

    assert build((first_window, second_window)).artifact_id == build(
        (second_window, first_window)
    ).artifact_id
    with pytest.raises(ValueError, match="camera/window repeats"):
        build((first_window, first_window))


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
            observation_split=_split(),
            registered_reference_mean_sha256=_array_sha256(mean),
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
    assert result["heteroscedastic_endpoint"] == {
        "cue_reliability_consumed": True,
        "inflated_row_covariance_trace_not_lower": True,
        "innovation_clipping_before_mixture": False,
        "lower_reliability_trace_not_lower": True,
        "metric_row_covariance_consumed": True,
        "robust_mixture_application_count": 1,
    }
    assert result["failed_support_fallback"]["covariance_byte_identical"] is True
    assert result["far_point_rejection"] == {
        "case_donor_admitted": False,
        "covariance_byte_identical": True,
        "update_count": [0, 0, 0, 0],
        "valid_entry_count": 0,
    }
    assert result["candidate_specific_midpoint"]["opposite_signed"] is True
    assert result["candidate_specific_midpoint"][
        "assignment_spread_retained"
    ] is True
