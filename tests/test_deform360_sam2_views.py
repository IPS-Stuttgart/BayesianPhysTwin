from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from causal4d_public.deform360_sam2_views import (
    CrossViewMaskReliabilityConfig,
    JointMultiviewMaskSelectionConfig,
    build_sam2_view_audit,
    camera_reliability_from_multiview_consistency,
    select_joint_mask_candidate_hits,
    select_joint_multiview_masks,
    summarize_multiview_mask_hits,
    validate_sam2_view_audit,
    write_sam2_view_audit,
)


def _consistency_fixture() -> tuple[dict[str, object], CrossViewMaskReliabilityConfig]:
    cameras = tuple(f"camera_{index}" for index in range(5))
    hits = np.zeros((5, 20), dtype=bool)
    hits[:4, :10] = True
    hits[4, 10:] = True
    grid = np.column_stack(
        (
            np.linspace(0.0, 1.0, 20),
            np.zeros(20),
            np.zeros(20),
        )
    )
    config = CrossViewMaskReliabilityConfig(
        voxel_resolution=16,
        consensus_fraction_of_peak=0.75,
        minimum_consensus_votes=3,
        minimum_leave_one_out_recall=0.5,
    )
    return summarize_multiview_mask_hits(hits, cameras, grid, config), config


def test_cross_view_consistency_rejects_disjoint_placebo_mask() -> None:
    consistency, _ = _consistency_fixture()

    assert consistency["peak_vote_count"] == 4
    assert consistency["consensus_vote_count"] == 3
    assert consistency["accepted_cameras"] == [
        "camera_0",
        "camera_1",
        "camera_2",
        "camera_3",
    ]
    assert consistency["rejected_cameras"] == ["camera_4"]
    bad = next(
        item for item in consistency["per_camera"] if item["camera"] == "camera_4"
    )
    assert bad["leave_one_out_core_recall"] == 0.0


def test_camera_reliability_is_soft_and_residual_independent() -> None:
    consistency, _ = _consistency_fixture()

    reliability = camera_reliability_from_multiview_consistency(consistency)

    assert reliability["camera_0"] == 1.0
    assert reliability["camera_4"] == 0.05
    assert set(reliability) == {item["camera"] for item in consistency["per_camera"]}


def test_view_audit_is_checksummed_and_round_trips(tmp_path: Path) -> None:
    consistency, config = _consistency_fixture()
    access = {
        "episode_index": 0,
        "split": "source",
        "target_future_annotation_unlocked": False,
        "held_out_prediction_seal_sha256": None,
    }
    artifact = build_sam2_view_audit(
        protocol_id="fixture",
        episode_access=access,
        automatic_view_diagnostics=[],
        consistency=consistency,
        reliability_config=config,
    )

    assert validate_sam2_view_audit(artifact)["accepted_camera_count"] == 4
    output = write_sam2_view_audit(tmp_path / "view-audit.json", artifact)
    assert output.is_file()
    artifact["parameters"]["minimum_leave_one_out_recall"] = 0.1
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_sam2_view_audit(artifact)


def test_consensus_requires_three_views() -> None:
    config = CrossViewMaskReliabilityConfig(voxel_resolution=16)
    with pytest.raises(ValueError, match="at least three"):
        summarize_multiview_mask_hits(
            np.ones((2, 4), dtype=bool),
            ("left", "right"),
            np.zeros((4, 3)),
            config,
        )


def test_joint_selection_prefers_calibrated_consensus_over_top_appearance() -> None:
    candidates = {}
    for camera_index in range(4):
        wrong = np.zeros(50, dtype=bool)
        wrong[10 + 10 * camera_index : 20 + 10 * camera_index] = True
        correct = np.zeros(50, dtype=bool)
        correct[:8] = True
        candidates[f"camera_{camera_index}"] = [
            {"hits": wrong, "prior_score": 1.0, "candidate_index": 10},
            {"hits": correct, "prior_score": 0.8, "candidate_index": 11},
        ]

    result = select_joint_mask_candidate_hits(
        candidates,
        minimum_consensus_votes=4,
        config=JointMultiviewMaskSelectionConfig(
            voxel_resolution=16,
            appearance_weight=0.05,
        ),
    )

    assert result["selected_candidate_by_camera"] == {
        f"camera_{index}": 1 for index in range(4)
    }
    assert result["objective_components"]["peak_vote_fraction"] == 1.0


def test_joint_selection_preserves_single_candidate_behavior() -> None:
    shared = np.array([True, True, False, False])
    candidates = {
        camera: [{"hits": shared, "prior_score": 0.4, "candidate_index": 2}]
        for camera in ("a", "b", "c")
    }

    result = select_joint_mask_candidate_hits(
        candidates,
        minimum_consensus_votes=3,
        config=JointMultiviewMaskSelectionConfig(voxel_resolution=16),
    )

    assert result["selected_candidate_by_camera"] == {"a": 0, "b": 0, "c": 0}


def test_joint_selection_does_not_replace_geometry_valid_top_appearance() -> None:
    full = np.ones((4, 4), dtype=bool)
    small = np.zeros((4, 4), dtype=bool)
    small[1:3, 1:3] = True
    cameras = ("a", "b", "c")
    candidates = {
        camera: [
            {"mask": full, "prior_score": 1.0, "candidate_index": 0},
            {"mask": small, "prior_score": 0.5, "candidate_index": 1},
        ]
        for camera in cameras
    }
    intrinsics = {
        camera: np.array([[2.0, 0.0, 1.5], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]])
        for camera in cameras
    }
    extrinsics = {camera: np.eye(4) for camera in cameras}

    _, result = select_joint_multiview_masks(
        candidates,
        intrinsics,
        extrinsics,
        CrossViewMaskReliabilityConfig(
            cube_half_extent_m=0.5,
            voxel_resolution=16,
            minimum_consensus_votes=3,
        ),
        JointMultiviewMaskSelectionConfig(voxel_resolution=16),
    )

    assert result["selection"]["search_required"] is False
    assert all(
        record["candidate_rank"] == 0 for record in result["selected_candidates"]
    )
