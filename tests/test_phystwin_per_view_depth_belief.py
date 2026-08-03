from __future__ import annotations

from pathlib import Path

import numpy as np

from bayesian_phystwin.bias_aware_belief import BiasAwareStateUpdateConfig
from bayesian_phystwin.phystwin_per_view_depth_belief import (
    PerViewDepthLoaderConfig,
    PerViewDepthObservations,
    PerViewDepthStateConfig,
    infer_per_view_depth_state_correction,
    load_cotracker3_per_view_depth_observations,
)


def _write_cues(path: Path, tracks: np.ndarray) -> None:
    view_count, frame_count, identity_count, _ = tracks.shape
    intrinsics = np.repeat(np.eye(3)[None], view_count, axis=0)
    intrinsics[:, 0, 0] = 100.0
    intrinsics[:, 1, 1] = 100.0
    camera_to_world = np.repeat(np.eye(4)[None], view_count, axis=0)
    np.savez_compressed(
        path,
        multiview_tracks_xy_prefix=tracks.astype(np.float32),
        multiview_quality_probability_prefix=np.full(
            (view_count, frame_count, identity_count), 0.8, dtype=np.float32
        ),
        multiview_view_valid_prefix=np.ones(
            (view_count, frame_count, identity_count), dtype=bool
        ),
        multiview_intrinsics=intrinsics,
        multiview_camera_to_world=camera_to_world,
        forward_backward_error_px=np.full(
            (frame_count, identity_count), 1.0, dtype=np.float32
        ),
        forward_backward_valid=np.ones((frame_count, identity_count), dtype=bool),
        boundary_distance=np.ones((frame_count, identity_count), dtype=np.float32),
        cue_available=np.ones((frame_count, identity_count), dtype=bool),
    )


def test_per_view_depth_loader_preserves_cameras_and_metric_variance(
    tmp_path: Path,
) -> None:
    view_count, frame_count, identity_count = 2, 3, 4
    tracks = np.zeros((view_count, frame_count, identity_count, 2), dtype=float)
    tracks[..., 0] = np.arange(identity_count)[None, None] + 2.0
    tracks[..., 1] = 3.0
    tracks[:, 1:, :, 0] += np.arange(1, frame_count)[None, :, None]
    cues = tmp_path / "cues.npz"
    _write_cues(cues, tracks)
    raw = tmp_path / "raw"
    for view in range(view_count):
        (raw / "depth" / str(view)).mkdir(parents=True)
        for frame in range(frame_count):
            np.save(raw / "depth" / str(view) / f"{frame}.npy", np.full((8, 8), 1000))
    initial = np.column_stack(
        ((np.arange(identity_count) + 2.0) / 100.0, np.full(identity_count, 0.03), np.ones(identity_count))
    )

    observations = load_cotracker3_per_view_depth_observations(
        cues,
        raw,
        initial,
        train_end_frame=frame_count,
        config=PerViewDepthLoaderConfig(depth_patch_radius_px=0),
    )

    assert observations.points_world_m.shape == (2, 3, 4, 3)
    assert np.all(observations.valid)
    assert np.allclose(observations.points_world_m[:, 0], initial)
    assert np.allclose(
        observations.points_world_m[:, 2, :, 0] - initial[None, :, 0],
        0.02,
    )
    assert np.all(observations.variance_m2 > 0.0)
    assert np.all(observations.prior_reliability > 0.0)
    assert not observations.points_world_m.flags.writeable
    assert not observations.prior_reliability.flags.writeable


def _synthetic_problem(
    *,
    view_count: int = 2,
    outlier: bool = False,
) -> tuple[PerViewDepthObservations, np.ndarray, np.ndarray, np.ndarray]:
    frame_count, node_count = 8, 24
    coordinate = np.linspace(-1.0, 1.0, node_count)
    frame_zero = np.column_stack(
        (0.10 * coordinate, np.zeros(node_count), np.zeros(node_count))
    )
    mode = np.sin(3.0 * np.pi * coordinate)
    baseline = np.repeat(frame_zero[None], frame_count, axis=0)
    ramp = np.linspace(0.0, 1.0, frame_count)
    baseline[:, :, 1] += 0.010 * ramp[:, None] * mode[None]
    correction = np.column_stack(
        (np.zeros(node_count), np.zeros(node_count), 0.003 * mode)
    )
    points = np.repeat(baseline[None], view_count, axis=0)
    camera_bias = np.linspace(-0.004, 0.005, view_count)
    points[:, 4:] += correction[None, None]
    points[:, 4:, :, 0] += camera_bias[:, None, None]
    if outlier:
        points[0, -1, 0] += np.asarray([0.5, -0.4, 0.3])
    valid = np.ones((view_count, frame_count, node_count), dtype=bool)
    reliability = np.ones_like(valid, dtype=float)
    variance = np.full_like(reliability, 0.001**2)
    observations = PerViewDepthObservations(
        points_world_m=points,
        valid=valid,
        prior_reliability=reliability,
        variance_m2=variance,
        local_depth_mad_m=np.zeros_like(reliability),
    )
    return observations, baseline, frame_zero, correction


def _state_config() -> PerViewDepthStateConfig:
    return PerViewDepthStateConfig(
        window_frames=4,
        minimum_unique_identities=8,
        maximum_correction_m=0.02,
        maximum_correction_to_response_ratio=2.0,
        update=BiasAwareStateUpdateConfig(
            observation_std_m=0.001,
            state_prior_std_m=0.02,
            shared_bias_prior_std_m=0.02,
            camera_bias_prior_std_m=0.02,
            effective_samples_per_view=16.0,
            maximum_state_update_m=0.02,
        ),
    )


def test_per_view_depth_state_recovers_nonrigid_mode_with_camera_bias() -> None:
    observations, baseline, frame_zero, expected = _synthetic_problem()

    result = infer_per_view_depth_state_correction(
        observations,
        baseline,
        frame_zero,
        end_frame=len(baseline),
        config=_state_config(),
    )

    assert result.accepted, result.reason
    assert np.sqrt(np.mean(np.square(result.correction_m - expected))) < 0.001
    assert result.diagnostics["prior_reliability_uses_innovation"] is False
    assert result.diagnostics["state_update"]["active_view_count"] == 2


def test_per_view_depth_state_rejects_globally_confounded_response() -> None:
    observations, baseline, frame_zero, _ = _synthetic_problem()
    baseline = baseline.copy()
    baseline[:] = frame_zero
    baseline[:, :, 0] += np.linspace(0.0, 0.01, len(baseline))[:, None]

    result = infer_per_view_depth_state_correction(
        observations,
        baseline,
        frame_zero,
        end_frame=len(baseline),
        config=_state_config(),
    )

    assert not result.accepted
    assert result.reason == "unidentifiable-physical-response"
    assert np.array_equal(result.correction_m, np.zeros_like(result.correction_m))


def test_duplicate_correlated_views_do_not_increase_information() -> None:
    observations, baseline, frame_zero, _ = _synthetic_problem(view_count=2)
    duplicated = PerViewDepthObservations(
        points_world_m=np.repeat(observations.points_world_m, 2, axis=0),
        valid=np.repeat(observations.valid, 2, axis=0),
        prior_reliability=np.repeat(observations.prior_reliability, 2, axis=0),
        variance_m2=np.repeat(observations.variance_m2, 2, axis=0),
        local_depth_mad_m=np.repeat(observations.local_depth_mad_m, 2, axis=0),
    )

    original = infer_per_view_depth_state_correction(
        observations,
        baseline,
        frame_zero,
        end_frame=len(baseline),
        config=_state_config(),
    )
    repeated = infer_per_view_depth_state_correction(
        duplicated,
        baseline,
        frame_zero,
        end_frame=len(baseline),
        config=_state_config(),
    )

    assert original.accepted and repeated.accepted
    assert np.allclose(original.correction_m, repeated.correction_m, atol=1e-10)
    assert np.allclose(
        original.coefficient_covariance_m2,
        repeated.coefficient_covariance_m2,
        atol=1e-12,
    )


def test_robust_update_downweights_gross_outlier_once() -> None:
    observations, baseline, frame_zero, expected = _synthetic_problem(outlier=True)

    result = infer_per_view_depth_state_correction(
        observations,
        baseline,
        frame_zero,
        end_frame=len(baseline),
        config=_state_config(),
    )

    assert result.accepted, result.reason
    assert np.sqrt(np.mean(np.square(result.correction_m - expected))) < 0.0015
    assert result.diagnostics["state_update"]["minimum_camera_robust_weight"] < 0.1


def test_large_inferred_update_is_shrunk_to_physical_cap() -> None:
    observations, baseline, frame_zero, _ = _synthetic_problem()
    config = PerViewDepthStateConfig(
        window_frames=4,
        minimum_unique_identities=8,
        maximum_correction_m=0.001,
        maximum_correction_to_response_ratio=2.0,
        update=BiasAwareStateUpdateConfig(
            observation_std_m=0.001,
            state_prior_std_m=0.02,
            shared_bias_prior_std_m=0.02,
            camera_bias_prior_std_m=0.02,
            effective_samples_per_view=16.0,
            maximum_state_update_m=0.10,
        ),
    )

    result = infer_per_view_depth_state_correction(
        observations,
        baseline,
        frame_zero,
        end_frame=len(baseline),
        config=config,
    )

    assert result.accepted, result.reason
    assert result.diagnostics["correction_cap_applied"] is True
    assert result.diagnostics["raw_maximum_correction_m"] > 0.001
    assert result.diagnostics["maximum_correction_m"] <= 0.001 + 1e-12
    assert np.max(np.linalg.norm(result.correction_m, axis=1)) <= 0.001 + 1e-12


def test_state_residual_cannot_change_prior_perception_reliability() -> None:
    observations, baseline, frame_zero, _ = _synthetic_problem()
    reliability_before = observations.prior_reliability.copy()
    shifted_baseline = baseline + np.asarray([0.03, -0.02, 0.01])

    infer_per_view_depth_state_correction(
        observations,
        shifted_baseline,
        frame_zero,
        end_frame=len(baseline),
        config=_state_config(),
    )

    assert np.array_equal(observations.prior_reliability, reliability_before)
