from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from bayesian_phystwin import deform360_frame_zero_assets as frame_zero_assets
from bayesian_phystwin import deform360_frame_zero_semantic_gate as semantic_gate
from bayesian_phystwin.deform360_frame_zero_assets import FrameZeroAssetConfig
from bayesian_phystwin.deform360_frame_zero_semantic_gate import (
    FRAME_ZERO_REFERENCE_OPTIONAL_GEOMETRY_STRATEGY,
    FRAME_ZERO_SEMANTIC_GATE_CONTRACT,
    background_control_crop,
    evaluate_rank_gate,
    filter_robot_dominated_proposals,
    masked_square_crop,
    semantic_label_for_object_id,
    subtract_robot_from_selected_masks,
)


def test_contract_has_exact_fourth_position_and_excludes_failed_gates() -> None:
    assert FRAME_ZERO_SEMANTIC_GATE_CONTRACT["application_order"][-1] == (
        FRAME_ZERO_REFERENCE_OPTIONAL_GEOMETRY_STRATEGY
    )
    excluded = " ".join(FRAME_ZERO_SEMANTIC_GATE_CONTRACT["explicitly_excluded"])
    assert "GroundingDINO" in excluded
    assert "taxel" in excluded
    assert FRAME_ZERO_SEMANTIC_GATE_CONTRACT["siglip2"]["labels_in_order"] == [
        "rope",
        "blanket",
        "scarf",
        "squirrel toy",
        "spider toy",
    ]


def test_deterministic_siglip_state_is_narrow_and_restored_after_error() -> None:
    class FakeTorch:
        def __init__(self) -> None:
            self.deterministic = False
            self.warn_only = True
            self.precision = "medium"
            self.backends = SimpleNamespace(
                cuda=SimpleNamespace(
                    matmul=SimpleNamespace(allow_tf32=True),
                ),
                cudnn=SimpleNamespace(allow_tf32=True),
            )

        def are_deterministic_algorithms_enabled(self) -> bool:
            return self.deterministic

        def is_deterministic_algorithms_warn_only_enabled(self) -> bool:
            return self.warn_only

        def get_float32_matmul_precision(self) -> str:
            return self.precision

        def use_deterministic_algorithms(
            self, enabled: bool, *, warn_only: bool = False
        ) -> None:
            self.deterministic = enabled
            self.warn_only = warn_only

        def set_float32_matmul_precision(self, value: str) -> None:
            self.precision = value

    torch = FakeTorch()
    with pytest.raises(RuntimeError, match="synthetic inference"):
        with semantic_gate._deterministic_siglip_torch_state(torch):
            assert torch.deterministic is True
            assert torch.warn_only is False
            assert torch.precision == "highest"
            assert torch.backends.cuda.matmul.allow_tf32 is False
            assert torch.backends.cudnn.allow_tf32 is False
            raise RuntimeError("synthetic inference")
    assert torch.deterministic is False
    assert torch.warn_only is True
    assert torch.precision == "medium"
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True


@pytest.mark.parametrize(
    ("object_id", "expected"),
    [
        ("002-rope-silk", "rope"),
        ("081-stripe-rope", "rope"),
        ("083-blanket-cloth", "blanket"),
        ("085-scarf-cloth", "scarf"),
        ("092-squirrel", "squirrel toy"),
        ("170-spider", "spider toy"),
    ],
)
def test_true_label_comes_only_from_frozen_object_prefix(
    object_id: str, expected: str
) -> None:
    assert semantic_label_for_object_id(object_id) == expected


def test_unknown_true_label_fails_closed() -> None:
    with pytest.raises(ValueError, match="outside the frozen"):
        semantic_label_for_object_id("999-unregistered")


def test_masked_crop_uses_inclusive_bbox_edge_center_and_gray_padding() -> None:
    rgb = np.arange(5 * 6 * 3, dtype=np.uint8).reshape(5, 6, 3)
    mask = np.zeros((5, 6), dtype=bool)
    mask[0:2, 0:2] = True

    crop, audit = masked_square_crop(rgb, mask)

    assert crop.shape == (224, 224, 3)
    assert audit["mask_bbox_xyxy_inclusive"] == [0, 0, 1, 1]
    assert audit["mask_bbox_width_pixels"] == 2
    assert audit["mask_bbox_height_pixels"] == 2
    assert audit["square_side_pixels"] == 3
    assert audit["square_xyxy_exclusive_unclipped"] == [-1, -1, 2, 2]
    assert audit["square_source_xyxy_exclusive_clipped"] == [0, 0, 2, 2]
    assert audit["neutral_background_rgb"] == [128, 128, 128]


def test_background_control_uses_zero_overlap_farthest_corner_tie_break() -> None:
    rgb = np.zeros((12, 12, 3), dtype=np.uint8)
    mask = np.zeros((12, 12), dtype=bool)
    mask[5:8, 5:8] = True

    crop, audit = background_control_crop(rgb, mask, side=3)

    assert crop is not None
    assert audit["available"] is True
    # All four corners have zero overlap and equal centroid distance, so the
    # fixed top-left order is the final tie-break.
    assert audit["chosen"]["name"] == "top-left"
    assert audit["chosen"]["selected_mask_overlap_pixels"] == 0


def test_urdf_majority_boundary_preserves_indices_and_subtraction_audit() -> None:
    masks = []
    for start in (0, 10):
        mask = np.zeros((4, 10), dtype=bool)
        mask.flat[start : start + 10] = True
        masks.append(mask)
    robot = np.zeros((4, 10), dtype=bool)
    robot.flat[:5] = True
    robot.flat[10:14] = True
    proposals = {
        "camera": [
            {
                "segmentation": masks[0],
                "predicted_iou": 0.95,
                "stability_score": 0.95,
            },
            {
                "segmentation": masks[1],
                "predicted_iou": 0.96,
                "stability_score": 0.96,
            },
        ]
    }

    filtered, audit = filter_robot_dominated_proposals(
        proposals, {"camera": robot}, {"camera": robot}
    )

    records = audit["per_camera"][0]["candidates"]
    assert [record["candidate_index"] for record in records] == [0, 1]
    assert records[0]["dilated_robot_overlap_fraction"] == 0.5
    assert records[0]["rejected_as_robot_dominated"] is True
    assert records[1]["dilated_robot_overlap_fraction"] == 0.4
    assert records[1]["rejected_as_robot_dominated"] is False
    assert filtered["camera"][0]["predicted_iou"] == -1.0
    assert filtered["camera"][1]["predicted_iou"] == 0.96

    selected = {"camera": masks[1]}
    final, subtraction = subtract_robot_from_selected_masks(
        selected, {"camera": robot}, {"camera": robot}
    )
    assert int(final["camera"].sum()) == 6
    record = subtraction["per_camera"][0]
    assert record["selected_pixel_count_before_subtraction"] == 10
    assert record["removed_pixel_count"] == 4
    assert record["selected_pixel_count_after_subtraction"] == 6


def _passing_logits(background_count: int = 4) -> np.ndarray:
    object_logits = np.zeros((8, 5), dtype=np.float64)
    object_logits[:5, 4] = 10.0
    object_logits[5:, 0] = 1.0
    background_logits = np.zeros((background_count, 5), dtype=np.float64)
    if background_count:
        background_logits[:, 0] = 2.0
    return np.concatenate([object_logits, background_logits], axis=0)


def test_rank_gate_passes_with_five_of_eight_and_four_inapplicable_controls() -> None:
    cameras = [f"camera-{index}" for index in range(8)]
    result = evaluate_rank_gate(
        _passing_logits(),
        object_cameras=cameras,
        background_cameras=cameras[:4],
        true_label="spider toy",
    )

    assert result["decision"]["status"] == "pass"
    assert result["decision"]["true_label_object_top1_vote_count"] == 5
    assert result["background_control_rank"]["gate_applicable"] is False


def test_rank_gate_fails_exact_tie_and_applicable_background_bias() -> None:
    cameras = [f"camera-{index}" for index in range(8)]
    tied = _passing_logits()
    tied[0, 0] = tied[0, 4]
    tie_result = evaluate_rank_gate(
        tied,
        object_cameras=cameras,
        background_cameras=cameras[:4],
        true_label="spider toy",
    )
    assert tie_result["decision"]["status"] == "fail"
    assert tie_result["decision"]["all_applicable_ranks_have_unique_top1"] is False

    biased = _passing_logits(background_count=5)
    biased[8:, :] = 0.0
    biased[8:, 4] = 3.0
    bias_result = evaluate_rank_gate(
        biased,
        object_cameras=cameras,
        background_cameras=cameras[:5],
        true_label="spider toy",
    )
    assert bias_result["background_control_rank"]["gate_applicable"] is True
    assert bias_result["decision"]["background_vote_gate_passed"] is False
    assert bias_result["decision"]["status"] == "fail"


def test_rank_gate_audit_is_sensitive_to_logit_tampering() -> None:
    cameras = [f"camera-{index}" for index in range(8)]
    logits = _passing_logits()
    original = evaluate_rank_gate(
        logits,
        object_cameras=cameras,
        background_cameras=cameras[:4],
        true_label="spider toy",
    )
    changed = deepcopy(logits)
    changed[:5, 4] = 0.0
    changed[:5, 0] = 20.0
    tampered = evaluate_rank_gate(
        changed,
        object_cameras=cameras,
        background_cameras=cameras[:4],
        true_label="spider toy",
    )
    assert original["decision"]["status"] == "pass"
    assert tampered["decision"]["status"] == "fail"


def test_reference_optional_selector_abstains_reference_only_in_fourth_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = FrameZeroAssetConfig().reference_camera
    nonreference = [f"camera-{index:02d}" for index in range(8)]
    cameras = sorted([reference, *nonreference])
    rgb = {camera: np.zeros((64, 64, 3), dtype=np.uint8) for camera in cameras}
    reference_mask = np.zeros((64, 64), dtype=bool)
    reference_mask[28:37, 9:14] = True
    common_mask = np.zeros((64, 64), dtype=bool)
    common_mask[28:37, 30:35] = True
    proposals = {
        camera: [
            {
                "segmentation": (
                    reference_mask.copy() if camera == reference else common_mask.copy()
                ),
                "predicted_iou": 0.95,
                "stability_score": 0.95,
            }
        ]
        for camera in cameras
    }
    intrinsic = np.asarray(
        [[40.0, 0.0, 32.0], [0.0, 40.0, 32.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    intrinsics = {camera: intrinsic for camera in cameras}
    extrinsics = {camera: np.eye(4, dtype=np.float64) for camera in cameras}

    def diagnostic(
        _rgb: np.ndarray, mask: np.ndarray, _config: object
    ) -> dict[str, object]:
        rows, columns = np.nonzero(mask)
        return {
            "eligible": True,
            "score": 1.0,
            "area_pixels": int(mask.sum()),
            "area_fraction": float(mask.mean()),
            "centroid_xy": [float(columns.mean()), float(rows.mean())],
            "normalized_center_distance": 0.1,
            "bounding_box_wh": [5, 9],
            "bounding_box_fill_fraction": 1.0,
            "border_side_count": 0,
            "foreground_contrast": 0.5,
            "border_background_rgb": [0.0, 0.0, 0.0],
        }

    monkeypatch.setattr(
        frame_zero_assets,
        "deformable_object_mask_candidate_diagnostics",
        diagnostic,
    )
    monkeypatch.setattr(
        frame_zero_assets,
        "mask_appearance_descriptor",
        lambda _rgb, mask: {"centroid": float(np.nonzero(mask)[1].mean())},
    )
    monkeypatch.setattr(
        frame_zero_assets,
        "mask_appearance_similarity",
        lambda _reference, _candidate: {
            "combined": 1.0,
            "hs_histogram_intersection": 1.0,
            "lab_similarity": 1.0,
            "shape_similarity": 1.0,
        },
    )

    with pytest.raises(frame_zero_assets.FrameZeroGeometryQAError):
        frame_zero_assets._common_voxel_mask_assignment(
            rgb,
            proposals,
            intrinsics,
            extrinsics,
            reference_camera=reference,
            config=FrameZeroAssetConfig(),
        )

    monkeypatch.setattr(
        frame_zero_assets,
        "HELD_PROTOCOL_ID",
        "deform360-held-online-belief-v8",
    )
    masks, diagnostics, audit = frame_zero_assets._common_voxel_mask_assignment(
        rgb,
        proposals,
        intrinsics,
        extrinsics,
        reference_camera=reference,
        config=FrameZeroAssetConfig(),
        reference_optional=True,
    )

    assert tuple(sorted(masks)) == tuple(nonreference)
    assert reference not in masks
    assert audit["strategy"] == (
        "reference-conditioned-reference-optional-exhaustive-exact-eight-assignment"
    )
    assert audit["evaluated_exact_eight_subset_count"] == 9
    assert audit["selected_exact_eight_cameras"] == nonreference
    bounded = audit["exact_eight_subset_evaluations"]
    assert bounded["schema_id"] == (
        frame_zero_assets.EXACT_EIGHT_SUBSET_BOUNDED_AUDIT_SCHEMA_ID
    )
    assert bounded["record_count"] == 9
    assert bounded["selected_record"]["cameras"] == nonreference
    reference_diagnostic = next(
        record for record in diagnostics if record["camera"] == reference
    )
    assert reference_diagnostic["geometry_inlier_selection"]["retained"] is False
