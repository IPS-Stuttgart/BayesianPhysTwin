from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.projected_observability_planner import (
    ProjectedObservabilityConfig,
    plan_projected_observability,
)


def _fixture() -> dict[str, object]:
    camera_names = tuple(f"camera-{index}" for index in range(6))
    groups = ("a", "a", "b", "b", "c", "c")
    frame_count = 4
    point_count = 18
    positions = np.column_stack(
        (
            np.linspace(0.0, 0.17, point_count),
            np.zeros(point_count),
            np.zeros(point_count),
        )
    )
    pixels = np.zeros(
        (len(camera_names), frame_count, point_count, 2),
        dtype=np.float64,
    )
    pixels[..., 0] = np.linspace(100.0, 300.0, point_count)
    pixels[..., 1] = 200.0
    support = np.ones((point_count, len(camera_names)), dtype=bool)
    blocks = (
        np.arange(0, 6),
        np.arange(0, 6),
        np.arange(6, 12),
        np.arange(6, 12),
        np.arange(12, 18),
        np.arange(12, 18),
    )
    shape = np.asarray([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
    for camera, block in enumerate(blocks):
        for frame, progress in enumerate(np.linspace(0.0, 1.0, frame_count)):
            pixels[camera, frame, block, 1] += 4.0 * progress * shape
    # This camera has only coherent translation, which is a bias nuisance.
    pixels[5] = pixels[5, :1]
    for frame, progress in enumerate(np.linspace(0.0, 1.0, frame_count)):
        pixels[5, frame, :, 0] += 8.0 * progress
    return {
        "positions_m": positions,
        "camera_names": camera_names,
        "spatial_group_ids": groups,
        "physical_pixels_px": pixels,
        "initial_depth_m": np.ones((len(camera_names), point_count)),
        "focal_lengths_px": np.full((len(camera_names), 2), 500.0),
        "frame_zero_support": support,
        "config": ProjectedObservabilityConfig(
            center_count=12,
            minimum_camera_count=3,
            maximum_camera_count=3,
            minimum_points_per_camera=4,
            minimum_projected_response_rms_m=0.0005,
        ),
    }


def test_plan_spans_groups_and_selects_only_observable_queries() -> None:
    plan = plan_projected_observability(**_fixture())

    assert len(plan.selected_camera_names) == 3
    assert len(
        {
            plan.spatial_group_ids[index]
            for index in plan.selected_camera_indices
        }
    ) == 3
    assert "camera-5" not in plan.selected_camera_names
    for camera in plan.selected_camera_names:
        query_ids = plan.query_ids(camera)
        index = plan.camera_names.index(camera)
        assert len(query_ids) >= 4
        assert np.all(plan.eligible_camera_point[index, query_ids])
        assert np.all(
            plan.projected_response_rms_m[index, query_ids] >= 0.0005
        )
    assert not plan.center_ids.flags.writeable
    assert plan.artifact_id.startswith("sha256:")


def test_shared_camera_translation_is_not_observable_shape_response() -> None:
    inputs = _fixture()
    pixels = np.asarray(inputs["physical_pixels_px"]).copy()
    pixels[:] = pixels[:, :1]
    for frame, progress in enumerate(np.linspace(0.0, 1.0, 4)):
        pixels[:, frame, :, 0] += 10.0 * progress
        pixels[:, frame, :, 1] -= 4.0 * progress
    inputs["physical_pixels_px"] = pixels

    with pytest.raises(ValueError, match="too few cameras"):
        plan_projected_observability(**inputs)


def test_camera_input_order_does_not_change_plan() -> None:
    inputs = _fixture()
    first = plan_projected_observability(**inputs)
    order = np.asarray([5, 2, 0, 4, 1, 3])
    names = tuple(inputs["camera_names"])
    groups = tuple(inputs["spatial_group_ids"])
    inputs["camera_names"] = tuple(names[index] for index in order)
    inputs["spatial_group_ids"] = tuple(groups[index] for index in order)
    inputs["physical_pixels_px"] = np.asarray(
        inputs["physical_pixels_px"]
    )[order]
    inputs["initial_depth_m"] = np.asarray(inputs["initial_depth_m"])[order]
    inputs["focal_lengths_px"] = np.asarray(
        inputs["focal_lengths_px"]
    )[order]
    inputs["frame_zero_support"] = np.asarray(
        inputs["frame_zero_support"]
    )[:, order]

    second = plan_projected_observability(**inputs)

    assert second.artifact_id == first.artifact_id
    assert second.selected_camera_names == first.selected_camera_names
    np.testing.assert_array_equal(second.center_ids, first.center_ids)


def test_geometry_changes_content_address_even_when_selection_is_same() -> None:
    inputs = _fixture()
    first = plan_projected_observability(**inputs)
    positions = np.asarray(inputs["positions_m"]).copy()
    positions *= 2.0
    inputs["positions_m"] = positions

    second = plan_projected_observability(**inputs)

    np.testing.assert_array_equal(second.center_ids, first.center_ids)
    assert second.artifact_id != first.artifact_id


def test_duplicate_camera_name_is_rejected() -> None:
    inputs = _fixture()
    names = list(inputs["camera_names"])
    names[-1] = names[0]
    inputs["camera_names"] = tuple(names)

    with pytest.raises(ValueError, match="unique"):
        plan_projected_observability(**inputs)
