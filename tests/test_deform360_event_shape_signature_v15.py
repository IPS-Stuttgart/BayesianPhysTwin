from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.deform360_event_conditioned_window_v15 import (
    EventConditionedWindowConfig,
    select_event_conditioned_window,
)
from bayesian_phystwin.deform360_event_shape_signature_v15 import (
    EventShapeSignatureConfig,
    build_event_panel_shape_signature,
    pairwise_shape_signature,
)


def _rgbd_scene(
    *,
    camera_count: int = 2,
    frame_count: int = 4,
    event_frame: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height = width = 64
    depth = np.ones(
        (camera_count, frame_count, height, width),
        dtype=np.float64,
    )
    object_mask = np.zeros_like(depth, dtype=bool)
    object_mask[:, :, 8:56, 8:56] = True
    gripper_mask = np.zeros_like(depth, dtype=bool)
    if event_frame is not None:
        depth[:, event_frame:, 8:56, 32:56] += 0.08
    intrinsics = np.repeat(np.eye(3)[None], camera_count, axis=0)
    intrinsics[:, 0, 0] = 120.0
    intrinsics[:, 1, 1] = 120.0
    intrinsics[:, 0, 2] = 32.0
    intrinsics[:, 1, 2] = 32.0
    return depth, object_mask, gripper_mask, intrinsics


def _build(
    scene: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    config: EventShapeSignatureConfig | None = None,
):
    depth, object_mask, gripper_mask, intrinsics = scene
    return build_event_panel_shape_signature(
        depth,
        object_mask,
        gripper_mask,
        intrinsics,
        camera_ids=tuple(
            f"camera-{index}" for index in range(depth.shape[0])
        ),
        config=config,
    )


def test_pairwise_shape_signature_is_rigid_transform_invariant() -> None:
    rng = np.random.default_rng(4)
    points = rng.normal(size=(64, 3))
    angle = 0.7
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transformed = points @ rotation.T + np.asarray([0.4, -0.2, 1.7])
    quantiles = (0.1, 0.3, 0.5, 0.7, 0.9)

    first = pairwise_shape_signature(
        points,
        quantiles=quantiles,
        maximum_points=64,
    )
    second = pairwise_shape_signature(
        transformed,
        quantiles=quantiles,
        maximum_points=64,
    )

    assert np.allclose(first, second, rtol=0.0, atol=1e-12)


def test_panel_builder_produces_metric_shape_components() -> None:
    result = _build(_rgbd_scene())

    assert result.evidence.component_signature_m.shape == (4, 5)
    assert np.all(result.evidence.available)
    assert np.all(result.evidence.camera_support == 2)
    assert np.all(result.evidence.gripper_clear)
    assert np.all(np.isfinite(result.evidence.component_signature_m))
    assert np.all(result.evidence.variance_m2 >= 2.0 * 0.005**2)
    assert result.descriptor()["information_boundary"][
        "tracker_or_material_identity_used"
    ] is False


def test_duplicating_identical_cameras_does_not_reduce_variance() -> None:
    ordinary = _build(_rgbd_scene(camera_count=2))
    duplicated = _build(_rgbd_scene(camera_count=4))

    assert np.array_equal(
        ordinary.evidence.component_signature_m,
        duplicated.evidence.component_signature_m,
    )
    assert np.array_equal(
        ordinary.evidence.variance_m2,
        duplicated.evidence.variance_m2,
    )
    assert np.all(ordinary.evidence.camera_support == 2)
    assert np.all(duplicated.evidence.camera_support == 4)


def test_gripper_overlap_is_excluded_and_reported() -> None:
    depth, object_mask, gripper_mask, intrinsics = _rgbd_scene()
    gripper_mask[:, :, 20:44, 20:44] = True
    result = _build((depth, object_mask, gripper_mask, intrinsics))

    assert np.all(
        result.per_camera_gripper_overlap_fraction
        > result.config.maximum_gripper_overlap_fraction
    )
    assert not np.any(result.per_camera_gripper_clear)
    assert not np.any(result.evidence.gripper_clear)


def test_each_frame_signature_is_independent_of_later_frames() -> None:
    original_scene = _rgbd_scene()
    changed_scene = tuple(value.copy() for value in original_scene)
    changed_scene[0][:, 2:] *= 1.7
    changed_scene[1][:, 2:, :8, :8] = True
    changed_scene[2][:, 2:, 12:52, 12:52] = True

    first = _build(original_scene)
    second = _build(changed_scene)

    assert np.array_equal(
        first.evidence.component_signature_m[:2],
        second.evidence.component_signature_m[:2],
    )
    assert np.array_equal(
        first.evidence.variance_m2[:2],
        second.evidence.variance_m2[:2],
    )


def test_shape_change_is_visible_without_a_tracker_or_physical_model() -> None:
    result = _build(_rgbd_scene(event_frame=2))

    change = (
        result.evidence.component_signature_m[2]
        - result.evidence.component_signature_m[1]
    )

    assert np.sqrt(np.mean(np.square(change))) > 0.001


def test_shape_provider_drives_the_causal_event_selector() -> None:
    proposal_scene = _rgbd_scene(
        camera_count=2,
        frame_count=32,
        event_frame=10,
    )
    validation_scene = tuple(value.copy() for value in proposal_scene)
    provider_config = EventShapeSignatureConfig(
        depth_standard_deviation_m=0.0005,
    )
    proposal = _build(proposal_scene, config=provider_config)
    validation = _build(validation_scene, config=provider_config)
    tactile = np.zeros(32, dtype=np.float64)
    tactile[6:] = 0.9
    actuator = np.zeros((32, 1, 3), dtype=np.float64)
    actuator[:, 0, 0] = 0.00025 * np.arange(32)

    event = select_event_conditioned_window(
        "synthetic-provider-control",
        proposal.evidence,
        validation.evidence,
        tactile,
        actuator,
        config=EventConditionedWindowConfig(
            lag_frames=6,
            first_candidate_frame=8,
            forecast_horizon_frames=10,
            shared_bias_variance_m2=1e-8,
        ),
    )

    assert event.admitted
    assert event.selected_attempt is not None
    assert event.selected_attempt.branch_frame == 10


def test_builder_rejects_a_single_camera_panel() -> None:
    with pytest.raises(ValueError, match="camera panel"):
        _build(_rgbd_scene(camera_count=1))
