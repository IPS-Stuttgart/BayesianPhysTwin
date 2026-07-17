import json
from pathlib import Path

import numpy as np
import pytest

from causal4d_public.deform360_dense_source import unpack_sampled_mask
from causal4d_public.deform360_grounded_sam2 import (
    GroundedSam2MaskConfig,
    rank_grounded_sam2_candidates,
)
from causal4d_public.deform360_object_sam2 import (
    DeformableObjectSam2MaskConfig,
    DeformableObjectSam2VideoPredictor,
)
from causal4d_public.deform360_reusable_trust_masks import (
    authorize_reusable_trust_mask_episode,
    load_reusable_trust_mask_addendum,
    write_sampled_mask_archive,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/causal4d_public"
PARENT = CONFIG / "deform360_reusable_trust_fresh_v1.json"
PHYSICS = CONFIG / "deform360_reusable_trust_physics_addendum_v1.json"
EXECUTION = CONFIG / "deform360_reusable_trust_execution_v1.json"
MASKS = CONFIG / "deform360_reusable_trust_mask_addendum_v2.json"
GROUNDED_MASKS = CONFIG / "deform360_reusable_trust_mask_addendum_v3.json"
GEOMETRY_CONTACT_MASKS = CONFIG / "deform360_reusable_trust_mask_addendum_v4.json"
SOURCE_TRAINED_CAMERA_MASKS = CONFIG / "deform360_reusable_trust_mask_addendum_v5.json"


def test_mask_addendum_loads_and_preserves_held_boundary() -> None:
    protocol = load_reusable_trust_mask_addendum(PARENT, PHYSICS, EXECUTION, MASKS)
    authorization = authorize_reusable_trust_mask_episode(
        protocol,
        object_id="171-penguin",
        episode_id=0,
        operation="held-prediction",
    )

    assert len(protocol["mask_addendum"]["objects"]["171-penguin"]["cameras"]) == 12
    assert not authorization["held_outcome_allowed"]
    assert (
        authorization["mask_addendum_file_sha256"]
        == protocol["mask_addendum_file_sha256"]
    )


def test_mask_addendum_rejects_prior_held_media_access(tmp_path: Path) -> None:
    payload = json.loads(MASKS.read_text(encoding="utf-8"))
    payload["lock_timing"]["held_out_media_inspected"] = True
    changed = tmp_path / "mask.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="held access"):
        load_reusable_trust_mask_addendum(PARENT, PHYSICS, EXECUTION, changed)


def test_grounded_mask_addendum_locks_prompt_and_held_boundary() -> None:
    protocol = load_reusable_trust_mask_addendum(
        PARENT, PHYSICS, EXECUTION, GROUNDED_MASKS
    )
    authorization = authorize_reusable_trust_mask_episode(
        protocol,
        object_id="003-cable",
        episode_id=8,
        operation="held-prediction",
    )

    assert protocol["mask_addendum"]["objects"]["003-cable"]["text_prompt"] == (
        "long black cable."
    )
    assert authorization["mask_addendum_id"].endswith("v3")
    assert not authorization["held_outcome_allowed"]


def test_grounded_mask_addendum_rejects_unpinned_model_revision(
    tmp_path: Path,
) -> None:
    payload = json.loads(GROUNDED_MASKS.read_text(encoding="utf-8"))
    payload["observation_initializer"]["grounding_dino"]["model_revision"] = "main"
    changed = tmp_path / "mask.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="full Git SHA"):
        load_reusable_trust_mask_addendum(PARENT, PHYSICS, EXECUTION, changed)


def test_geometry_contact_mask_addendum_locks_all_views_and_delayed_contact() -> None:
    protocol = load_reusable_trust_mask_addendum(
        PARENT, PHYSICS, EXECUTION, GEOMETRY_CONTACT_MASKS
    )
    policy = protocol["mask_addendum"]

    assert len(policy["objects"]["171-penguin"]["cameras"]) == 32
    assert policy["geometry_contact_policy"]["confirmation_frames"] == 2
    assert policy["geometry_contact_policy"]["target_tactile_used"] is False
    assert (
        policy["geometry_contact_policy"]["release_inferred_from_initial_geometry"]
        is False
    )


def test_geometry_contact_mask_addendum_rejects_mask_selected_cameras(
    tmp_path: Path,
) -> None:
    payload = json.loads(GEOMETRY_CONTACT_MASKS.read_text(encoding="utf-8"))
    payload["camera_policy"]["camera_set_selected_from_mask_quality"] = True
    changed = tmp_path / "mask.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="all-camera"):
        load_reusable_trust_mask_addendum(PARENT, PHYSICS, EXECUTION, changed)


def test_source_trained_camera_addendum_freezes_fit_only_transfer_panel() -> None:
    protocol = load_reusable_trust_mask_addendum(
        PARENT, PHYSICS, EXECUTION, SOURCE_TRAINED_CAMERA_MASKS
    )
    policy = protocol["mask_addendum"]

    assert len(policy["objects"]["003-cable"]["cameras"]) == 6
    assert policy["camera_policy"]["validation_episode_ids"] == [3, 4, 6, 7, 9]
    assert policy["camera_policy"]["selection_uses_object_outcomes"] is False
    assert policy["source_qa_gates"]["minimum_accepted_camera_count"] == 4


def test_sampled_mask_archive_is_staging_compatible(tmp_path: Path) -> None:
    masks = {
        "a": np.array([[True, False, True], [False, True, False]]),
        "b": np.array([[False, True, False], [True, False, True]]),
    }
    path = write_sampled_mask_archive(
        tmp_path / "masks.npz",
        cameras=["a", "b"],
        frame_index=17,
        masks=masks,
    )

    with np.load(path, allow_pickle=False) as archive:
        assert np.array_equal(unpack_sampled_mask(archive, "a", 17), masks["a"])
        assert np.array_equal(unpack_sampled_mask(archive, "b", 17), masks["b"])


def test_exact_rgb_generic_selection_keeps_old_candidate_scoring() -> None:
    predictor = object.__new__(DeformableObjectSam2VideoPredictor)
    predictor.config = DeformableObjectSam2MaskConfig()
    rgb = np.full((32, 32, 3), 240, dtype=np.uint8)
    rgb[8:24, 8:24] = 40
    small = np.zeros((32, 32), dtype=bool)
    small[12:20, 12:20] = True
    large = np.zeros((32, 32), dtype=bool)
    large[8:24, 8:24] = True
    predictor._automatic_annotations = lambda _rgb: [
        {"segmentation": small, "predicted_iou": 0.8, "stability_score": 0.9},
        {"segmentation": large, "predicted_iou": 0.9, "stability_score": 0.9},
    ]

    selected, diagnostics = predictor.select_initial_mask_from_rgb(
        rgb, camera="camera", video_name="frame-000017"
    )

    assert np.array_equal(selected, large)
    assert diagnostics["selected"]["candidate_index"] == 1


def test_grounded_candidates_rank_and_deduplicate_without_state_residual() -> None:
    rgb = np.full((32, 32, 3), 240, dtype=np.uint8)
    rgb[8:24, 8:24] = 30
    large = np.zeros((32, 32), dtype=bool)
    large[8:24, 8:24] = True
    small = np.zeros((32, 32), dtype=bool)
    small[10:20, 10:20] = True
    config = GroundedSam2MaskConfig(
        model_id="model",
        model_revision="1" * 40,
        transformers_version="version",
        maximum_candidates_per_camera=2,
    )

    candidates = rank_grounded_sam2_candidates(
        rgb,
        boxes_xyxy=np.array([[7, 7, 25, 25], [9, 9, 21, 21]], dtype=float),
        box_scores=np.array([0.9, 0.5]),
        masks_by_box=[np.stack([large, large]), np.stack([small])],
        mask_scores_by_box=[np.array([0.8, 0.7]), np.array([0.9])],
        object_config=DeformableObjectSam2MaskConfig(),
        grounded_config=config,
    )

    assert len(candidates) == 2
    assert np.array_equal(candidates[0]["mask"], large)
    assert np.array_equal(candidates[1]["mask"], small)
    assert all("state" not in record["diagnostic"] for record in candidates)
