from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from causal4d_public.deform360_sam2_views import (
    CrossViewMaskReliabilityConfig,
    build_sam2_view_audit,
    camera_reliability_from_multiview_consistency,
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
