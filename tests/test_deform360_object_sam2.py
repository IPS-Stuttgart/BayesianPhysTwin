from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from causal4d_public.deform360_object_sam2 import (
    DeformableObjectSam2MaskConfig,
    DeformableObjectSam2VideoPredictor,
    deformable_object_mask_candidate_diagnostics,
    mask_appearance_descriptor,
    mask_appearance_similarity,
)


def test_central_salient_object_beats_small_border_marker() -> None:
    config = DeformableObjectSam2MaskConfig()
    rgb = np.full((120, 180, 3), 245, dtype=np.uint8)
    rgb[42:82, 48:136] = (30, 80, 180)
    rgb[0:14, 0:14] = 0
    object_mask = np.zeros(rgb.shape[:2], dtype=bool)
    object_mask[40:84, 45:139] = True
    marker_mask = np.zeros_like(object_mask)
    marker_mask[0:14, 0:14] = True

    object_result = deformable_object_mask_candidate_diagnostics(
        rgb, object_mask, config
    )
    marker_result = deformable_object_mask_candidate_diagnostics(
        rgb, marker_mask, config
    )

    assert object_result["eligible"] is True
    assert object_result["score"] > marker_result["score"]


def test_bright_background_region_is_ineligible() -> None:
    config = DeformableObjectSam2MaskConfig()
    rgb = np.full((100, 140, 3), 248, dtype=np.uint8)
    background = np.zeros(rgb.shape[:2], dtype=bool)
    background[20:80, 20:120] = True

    result = deformable_object_mask_candidate_diagnostics(rgb, background, config)

    assert result["eligible"] is False
    assert result["foreground_contrast"] < config.minimum_foreground_contrast


def test_large_dark_sheet_remains_eligible() -> None:
    config = DeformableObjectSam2MaskConfig()
    rgb = np.full((120, 180, 3), 250, dtype=np.uint8)
    rgb[18:108, 24:160] = (38, 42, 47)
    sheet = np.zeros(rgb.shape[:2], dtype=bool)
    sheet[18:108, 24:160] = True

    result = deformable_object_mask_candidate_diagnostics(rgb, sheet, config)

    assert result["eligible"] is True
    assert result["area_fraction"] > 0.5


def test_reference_appearance_prefers_same_object_color() -> None:
    pytest.importorskip("cv2")
    rgb = np.full((100, 160, 3), 245, dtype=np.uint8)
    rgb[30:70, 20:75] = (180, 105, 65)
    rgb[25:75, 95:145] = (20, 190, 45)
    reference_mask = np.zeros(rgb.shape[:2], dtype=bool)
    reference_mask[30:70, 20:75] = True
    same_color = np.zeros_like(reference_mask)
    same_color[32:68, 22:73] = True
    distractor = np.zeros_like(reference_mask)
    distractor[25:75, 95:145] = True

    reference = mask_appearance_descriptor(rgb, reference_mask)
    same = mask_appearance_similarity(
        reference, mask_appearance_descriptor(rgb, same_color)
    )
    other = mask_appearance_similarity(
        reference, mask_appearance_descriptor(rgb, distractor)
    )

    assert same["combined"] > 0.95
    assert same["combined"] > other["combined"]


def test_joint_opt_in_retains_basic_candidate_below_appearance_gate() -> None:
    pytest.importorskip("cv2")
    config = DeformableObjectSam2MaskConfig(
        minimum_mask_region_area=20,
        minimum_reference_appearance_similarity=0.99,
    )
    reference_rgb = np.full((60, 80, 3), 245, dtype=np.uint8)
    reference_rgb[20:40, 20:50] = (200, 30, 30)
    reference_mask = np.zeros(reference_rgb.shape[:2], dtype=bool)
    reference_mask[20:40, 20:50] = True
    target_rgb = np.full_like(reference_rgb, 245)
    target_rgb[20:40, 20:50] = (20, 30, 210)
    target_mask = reference_mask.copy()
    annotation = {
        "segmentation": target_mask,
        "predicted_iou": 0.95,
        "stability_score": 0.95,
    }
    predictor = object.__new__(DeformableObjectSam2VideoPredictor)
    predictor.config = config
    predictor._first_frame_rgb = lambda _: target_rgb
    predictor._automatic_annotations = lambda _: [annotation]

    with pytest.raises(ValueError, match="no reference-consistent mask"):
        predictor.initial_mask_candidates_with_reference(
            Path("camera/video.mp4"),
            reference_rgb,
            reference_mask,
            reference_camera="reference",
        )
    candidates, summary = predictor.initial_mask_candidates_with_reference(
        Path("camera/video.mp4"),
        reference_rgb,
        reference_mask,
        reference_camera="reference",
        include_below_appearance_threshold=True,
    )

    assert len(candidates) == 1
    assert candidates[0]["diagnostic"]["eligible"] is False
    assert candidates[0]["diagnostic"]["joint_selection_eligible"] is True
    assert summary["appearance_eligible_candidate_count"] == 0
