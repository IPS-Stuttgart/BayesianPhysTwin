"""Locked source-reference masks for the fresh reusable-twin panel."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .deform360_object_sam2 import DeformableObjectSam2MaskConfig
from .deform360_grounded_sam2 import GroundedSam2MaskConfig
from .deform360_reusable_trust_protocol import (
    EXPECTED_SPLITS,
    authorize_reusable_trust_episode,
    load_reusable_trust_protocol,
)
from .deform360_sam2_views import (
    CrossViewMaskReliabilityConfig,
    JointMultiviewMaskSelectionConfig,
)


MASK_ADDENDUM_ID = "deform360-reusable-trust-mask-addendum-v2"
GROUNDED_MASK_ADDENDUM_ID = "deform360-reusable-trust-mask-addendum-v3"
GEOMETRY_CONTACT_MASK_ADDENDUM_ID = "deform360-reusable-trust-mask-addendum-v4"
SOURCE_TRAINED_CAMERA_MASK_ADDENDUM_ID = "deform360-reusable-trust-mask-addendum-v5"
MASK_ADDENDUM_IDS = {
    MASK_ADDENDUM_ID,
    GROUNDED_MASK_ADDENDUM_ID,
    GEOMETRY_CONTACT_MASK_ADDENDUM_ID,
    SOURCE_TRAINED_CAMERA_MASK_ADDENDUM_ID,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reusable_trust_mask_addendum(
    parent_path: str | Path,
    physics_path: str | Path,
    execution_path: str | Path,
    mask_path: str | Path,
) -> dict[str, Any]:
    """Validate the source-only mask repair against all preceding locks."""

    protocol = load_reusable_trust_protocol(parent_path, physics_path, execution_path)
    mask_file = Path(mask_path).resolve()
    payload = json.loads(mask_file.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "mask addendum must contain an object")
    _require(payload.get("schema_version") == 1, "mask addendum schema changed")
    protocol_id = payload.get("protocol_id")
    _require(protocol_id in MASK_ADDENDUM_IDS, "mask addendum identity changed")
    parents = payload.get("parent_locks", {})
    _require(
        parents.get("fresh_protocol_file_sha256") == protocol["parent_file_sha256"]
        and parents.get("physics_addendum_file_sha256")
        == protocol["addendum_file_sha256"]
        and parents.get("execution_lock_file_sha256")
        == protocol["execution_file_sha256"],
        "mask addendum uses another parent lock",
    )
    timing = payload.get("lock_timing", {})
    _require(
        timing.get("held_out_media_inspected") is False
        and timing.get("held_out_outcomes_inspected") is False,
        "mask repair was not locked before held access",
    )
    if protocol_id == MASK_ADDENDUM_ID:
        reference = payload.get("source_reference", {})
        _require(
            int(reference.get("episode_id", -1)) == 1
            and reference.get("camera") == "brics-odroid-001_cam0"
            and reference.get("bootstrap")
            == "generic_sam2_top_candidate_on_exact_action_window_start"
            and reference.get("reused_across_fit_and_held_episodes") is True,
            "source-reference mask policy changed",
        )
    else:
        initializer = payload.get("observation_initializer", {})
        _require(
            initializer.get("mode") == "grounding_dino_box_prompted_sam2"
            and initializer.get("candidate_prior_uses_simulator_residual") is False
            and initializer.get("candidate_prior_uses_future_object_observation")
            is False
            and initializer.get("candidate_prior_uses_tactile") is False,
            "grounded observation boundary changed",
        )
        GroundedSam2MaskConfig(**initializer["grounding_dino"])
    objects = payload.get("objects", {})
    _require(set(objects) == set(EXPECTED_SPLITS), "mask object set changed")
    for object_id, expected in EXPECTED_SPLITS.items():
        record = objects[object_id]
        cameras = tuple(str(value) for value in record.get("cameras", ()))
        _require(
            record.get("topology") == expected["topology"], "mask topology changed"
        )
        if protocol_id == GEOMETRY_CONTACT_MASK_ADDENDUM_ID:
            expected_camera_count = 32
        elif protocol_id == SOURCE_TRAINED_CAMERA_MASK_ADDENDUM_ID:
            expected_camera_count = {
                "003-cable": 6,
                "086-cotton-scarf-cloth": 9,
                "171-penguin": 9,
            }[object_id]
        else:
            expected_camera_count = 12
        _require(
            len(cameras) == expected_camera_count,
            "each object must use the frozen camera count",
        )
        _require(len(set(cameras)) == len(cameras), "mask cameras are duplicated")
        if protocol_id == MASK_ADDENDUM_ID:
            _require(reference["camera"] in cameras, "reference camera is missing")
        else:
            _require(bool(record.get("text_prompt")), "object text prompt is empty")
        if protocol_id == GEOMETRY_CONTACT_MASK_ADDENDUM_ID:
            _require(
                cameras == tuple(sorted(cameras)), "all-camera policy is not sorted"
            )
        if protocol_id == SOURCE_TRAINED_CAMERA_MASK_ADDENDUM_ID:
            _require(
                cameras == tuple(sorted(cameras)),
                "source-trained camera panel is not sorted",
            )

    sam2 = payload.get("sam2", {})
    DeformableObjectSam2MaskConfig(**sam2["candidate_config"])
    selection = payload.get("joint_multiview_selection", {})
    CrossViewMaskReliabilityConfig(**selection["cross_view_config"])
    JointMultiviewMaskSelectionConfig(
        maximum_candidates_per_camera=int(selection["maximum_candidates_per_camera"]),
        voxel_resolution=int(selection["voxel_resolution"]),
        coordinate_descent_passes=int(selection["coordinate_descent_passes"]),
        appearance_weight=float(selection["appearance_weight"]),
        projected_volume_penalty=float(selection["projected_volume_penalty"]),
    )
    _require(
        selection.get("simulator_residual_used") is False
        and selection.get("post_initial_object_observation_used") is False,
        "mask selection crossed the observation boundary",
    )
    gates = payload.get("source_qa_gates", {})
    minimum_camera_gate = (
        4 if protocol_id == SOURCE_TRAINED_CAMERA_MASK_ADDENDUM_ID else 8
    )
    _require(
        int(gates.get("minimum_accepted_camera_count", 0)) >= minimum_camera_gate
        and int(gates.get("minimum_visual_hull_point_count", 0)) >= 512
        and 0.0
        < float(gates.get("maximum_hull_to_controller_distance_m", 0.0))
        <= 0.03,
        "mask source-QA gates changed",
    )
    if protocol_id in {
        GEOMETRY_CONTACT_MASK_ADDENDUM_ID,
        SOURCE_TRAINED_CAMERA_MASK_ADDENDUM_ID,
    }:
        camera_policy = payload.get("camera_policy", {})
        contact = payload.get("geometry_contact_policy", {})
        if protocol_id == GEOMETRY_CONTACT_MASK_ADDENDUM_ID:
            _require(
                camera_policy.get("rule")
                == "all_common_calibrated_cameras_sorted_lexicographically"
                and int(camera_policy.get("frozen_camera_count", 0)) == 32
                and camera_policy.get("camera_set_selected_from_mask_quality") is False,
                "all-camera observation policy changed",
            )
        else:
            _require(
                camera_policy.get("rule")
                == "fit_episode_one_leave_one_view_reliability_then_frozen_transfer"
                and int(camera_policy.get("selection_episode_id", -1)) == 1
                and camera_policy.get("selection_uses_object_outcomes") is False
                and camera_policy.get("validation_episode_ids") == [3, 4, 6, 7, 9]
                and camera_policy.get(
                    "held_camera_set_is_identical_to_source_selected_set"
                )
                is True,
                "source-trained camera policy changed",
            )
        _require(
            contact.get("rule")
            == "first_consecutive_entry_into_initial_geometry_envelope_then_latched"
            and int(contact.get("controller_group_size", 0)) == 768
            and int(contact.get("confirmation_frames", 0)) >= 1
            and float(contact.get("maximum_contact_distance_m", 0.0))
            == float(gates["maximum_hull_to_controller_distance_m"])
            and contact.get("release_inferred_from_initial_geometry") is False
            and contact.get("future_object_observation_used") is False
            and contact.get("target_tactile_used") is False,
            "geometry contact policy changed",
        )
    return {
        **protocol,
        "mask_addendum": payload,
        "mask_addendum_path": str(mask_file),
        "mask_addendum_file_sha256": sha256_file(mask_file),
    }


def authorize_reusable_trust_mask_episode(
    protocol: Mapping[str, Any],
    *,
    object_id: str,
    episode_id: int,
    operation: str,
) -> dict[str, Any]:
    """Authorize one exact-frame mask without granting outcome access."""

    authorization = authorize_reusable_trust_episode(
        protocol,
        object_id=object_id,
        episode_id=episode_id,
        operation=operation,
    )
    return {
        **authorization,
        "mask_addendum_id": protocol["mask_addendum"]["protocol_id"],
        "mask_addendum_file_sha256": protocol["mask_addendum_file_sha256"],
    }


def write_sampled_mask_archive(
    destination: str | Path,
    *,
    cameras: list[str],
    frame_index: int,
    masks: Mapping[str, np.ndarray],
) -> Path:
    """Write the packed exact-frame archive accepted by the staging runner."""

    values = np.stack([np.asarray(masks[camera], dtype=bool) for camera in cameras])
    _require(values.ndim == 3, "sampled masks must have shape camera x height x width")
    packed = np.packbits(values[:, None], axis=-1)
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        frame_indices=np.asarray([frame_index], dtype=np.int64),
        cameras=np.asarray(cameras),
        packed_masks=packed,
        image_shape=np.asarray(values.shape[1:], dtype=np.int64),
    )
    return output


__all__ = [
    "GEOMETRY_CONTACT_MASK_ADDENDUM_ID",
    "GROUNDED_MASK_ADDENDUM_ID",
    "MASK_ADDENDUM_ID",
    "SOURCE_TRAINED_CAMERA_MASK_ADDENDUM_ID",
    "authorize_reusable_trust_mask_episode",
    "load_reusable_trust_mask_addendum",
    "sha256_file",
    "write_sampled_mask_archive",
]
