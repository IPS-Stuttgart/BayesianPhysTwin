"""Lock and validate the prospective Deform360 virtual-sensing protocol."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .deform360_raw_pairwise_correspondence_diagnostic import (
    PERSISTENCE_CLIQUE_RBF_ARM,
)


SCHEMA_VERSION = 1
PROTOCOL_ID = "deform360-selective-virtual-sensing-v1"
DATASET_REPOSITORY = "brownu/deform360"
DATASET_REVISION = "7fea8e20231a47641d1d2bc8791920ec4e62ec5e"
SELECTION_SEED = PROTOCOL_ID
EXPECTED_UPDATE_FRAMES = (19, 38, 57)
EXPECTED_FRAME_COUNT = 76
EXPECTED_CENTER_COUNT = 16
EXPECTED_CAMERA_COUNT = 8
EXPECTED_STRATA = ("filament", "sheet", "volumetric")
EXPECTED_COHORT = {
    "filament": {
        "005-thread": (5, 2),
        "069-jump-rope": (1, 4),
        "071-climbing-rope": (0, 7),
        "077-hemp-rope": (8, 0),
    },
    "sheet": {
        "009-yellow-cloth": (9, 3),
        "022-handkerchief": (0, 4),
        "035-wipe-cloth": (8, 5),
        "041-wrap-paper-cloth": (2, 8),
    },
    "volumetric": {
        "044-doll": (0, 1),
        "049-ball": (5, 2),
        "125-rabbit": (8, 9),
        "146-frog": (5, 2),
    },
}
INELIGIBLE_OBJECTS = frozenset(
    {
        "001-rope",
        "002-rope-silk",
        "003-cable",
        "004-rubber-band",
        "008-pink-cloth",
        "016-shirt-cloth",
        "021-bag-cloth",
        "033-mask-cloth",
        "040-paper-cloth",
        "043-dog",
        "046-sponge",
        "052-rubber-duck",
        "067-paracord",
        "068-nylon-rope",
        "073-shoelace",
        "074-string",
        "079-chain-metal",
        "081-stripe-rope",
        "083-blanket-cloth",
        "085-scarf-cloth",
        "086-cotton-scarf-cloth",
        "090-sloth",
        "092-squirrel",
        "096-octopus",
        "117-bubble-wrap-cloth",
        "145-rubber-toy",
        "170-spider",
        "171-penguin",
    }
)
_CANONICAL_CONFIG_SHA256 = (
    "d231b0eb06e724ec131c569e88ca482bca44b3340f5e13d11f973feab0cc53dd"
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def protocol_config_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a protocol payload while excluding its declared digest."""

    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def metadata_ranked_episode_ids(object_id: str, count: int = 2) -> tuple[int, ...]:
    """Select episodes using object names and IDs only."""

    if not 1 <= count <= 10:
        raise ValueError("episode selection count must be in [1, 10]")
    ranked = sorted(
        range(10),
        key=lambda episode: hashlib.sha256(
            (f"{SELECTION_SEED}:{object_id}/episode_{episode:04d}").encode("utf-8")
        ).hexdigest(),
    )
    return tuple(ranked[:count])


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_cohort(raw: object) -> dict[str, dict[str, tuple[int, ...]]]:
    _require(isinstance(raw, Mapping), "cohort strata must be an object")
    _require(tuple(raw) == EXPECTED_STRATA, "cohort strata or order changed")
    normalized: dict[str, dict[str, tuple[int, ...]]] = {}
    all_objects: list[str] = []
    for stratum in EXPECTED_STRATA:
        records = raw[stratum]
        _require(isinstance(records, Sequence), f"{stratum} cohort must be a list")
        object_map: dict[str, tuple[int, ...]] = {}
        for record in records:
            _require(isinstance(record, Mapping), "cohort record must be an object")
            object_id = str(record.get("object_id", ""))
            episodes = tuple(int(value) for value in record.get("episode_ids", ()))
            _require(object_id, "cohort object ID is empty")
            _require(
                object_id not in INELIGIBLE_OBJECTS,
                f"cohort object was previously accessed or reserved: {object_id}",
            )
            _require(
                object_id not in object_map, f"duplicate cohort object: {object_id}"
            )
            _require(
                episodes == metadata_ranked_episode_ids(object_id),
                f"episode selection is not metadata-ranked: {object_id}",
            )
            object_map[object_id] = episodes
            all_objects.append(object_id)
        normalized[stratum] = object_map
    _require(len(all_objects) == len(set(all_objects)), "object appears in two strata")
    _require(normalized == EXPECTED_COHORT, "prospective cohort changed")
    return normalized


def _validate_method(config: Mapping[str, Any]) -> None:
    observation = config.get("observation")
    _require(isinstance(observation, Mapping), "observation contract is missing")
    _require(
        int(observation.get("frame_count", 0)) == EXPECTED_FRAME_COUNT,
        "frame count changed",
    )
    _require(
        tuple(observation.get("update_frames", ())) == EXPECTED_UPDATE_FRAMES,
        "update frames changed",
    )
    _require(
        int(observation.get("center_count", 0)) == EXPECTED_CENTER_COUNT,
        "center count changed",
    )
    _require(
        int(observation.get("camera_count", 0)) == EXPECTED_CAMERA_COUNT,
        "camera count changed",
    )
    _require(
        observation.get("initial_object_masks")
        == {
            "selector": (
                "generic automatic SAM2 object selector frozen before "
                "prospective access"
            ),
            "selector_source_sha256": (
                "79b161fa66489f75b5b078c7ae409387feed74c51a38b86e89800d0aa578b1df"
            ),
            "sam2_repository_revision": (
                "2b90b9f5ceec907a1c18123530e92e794ad901a4"
            ),
            "sam2_checkpoint_sha256": (
                "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"
            ),
            "minimum_accepted_cameras": 8,
            "future_propagation_allowed_only_after_prediction_cohort_seal": True,
        },
        "initial object-mask policy changed",
    )

    method = config.get("method")
    _require(isinstance(method, Mapping), "method contract is missing")
    _require(
        method.get("primary_arm") == PERSISTENCE_CLIQUE_RBF_ARM,
        "primary arm changed",
    )
    _require(method.get("base_backbone") == "persistence", "base backbone changed")
    _require(
        method.get("insufficient_support_fallback") == "bit-exact persistence",
        "fallback changed",
    )
    gate = method.get("pairwise_gate")
    _require(isinstance(gate, Mapping), "pairwise gate is missing")
    _require(
        dict(gate)
        == {
            "absolute_pair_strain_m": 0.03,
            "maximum_exact_center_count": 24,
            "minimum_inlier_count": 9,
            "minimum_inlier_fraction": 0.7,
            "relative_pair_strain": 0.1,
        },
        "pairwise gate changed",
    )
    belief = method.get("belief")
    _require(isinstance(belief, Mapping), "belief configuration is missing")
    _require(
        dict(belief)
        == {
            "degrees_of_freedom": 4.0,
            "global_prior_std_m": 0.1,
            "length_scale_fraction": 0.1,
            "local_blend": 1.0,
            "local_prior_std_m": 0.02,
            "maximum_correction_m": 0.1,
            "minimum_length_scale_m": 0.0001,
            "minimum_reliability": 0.02,
            "observation_std_m": 0.005,
            "process_std_m_per_sqrt_frame": 0.003,
        },
        "belief configuration changed",
    )
    _require(
        method.get("ungated_control")
        == (
            "same recursive RBF belief with every finite supported centre and "
            "no pairwise gate"
        ),
        "ungated control changed",
    )
    _require(
        method.get("independent_cpd_control")
        == {
            "beta": 2.0,
            "regularization": 3.0,
            "outlier_weight": 0.1,
            "maximum_iterations": 100,
            "relative_tolerance": 0.00001,
            "minimum_variance": 0.00000001,
            "minimum_scale_m": 0.000001,
            "refit_independently_at_each_update": True,
            "failure_fallback": "bit-exact persistence",
        },
        "independent CPD control changed",
    )


def _validate_temporal_staging(config: Mapping[str, Any]) -> None:
    staging = config.get("temporal_staging")
    _require(isinstance(staging, Mapping), "temporal staging contract is missing")
    _require(
        dict(staging)
        == {
            "rule": (
                "maximize mean gripper-centre path weighted at each step by the "
                "minimum endpoint closure confidence"
            ),
            "raw_window_length_frames": 81,
            "processed_frame_count": 76,
            "tracking_tail_frames_skipped": 5,
            "controller_centre": (
                "arithmetic mean of the five released action points per gripper"
            ),
            "closure_confidence": (
                "per gripper, clip((episode aperture p90 - aperture)/(episode "
                "aperture p90 - episode aperture p10), 0, 1); smaller released "
                "aperture means more closed"
            ),
            "static_aperture_fallback": (
                "confidence one, reducing exactly to unweighted path when "
                "aperture cannot identify closure"
            ),
            "step_weight": (
                "minimum closure confidence at the two endpoints of each "
                "gripper-centre path step"
            ),
            "gripper_aggregation": (
                "mean of per-gripper closure-weighted path lengths; gripper "
                "vectors are never averaged before taking their norms"
            ),
            "candidate_starts": {
                "first": 8,
                "stride": 6,
                "stop": "last start whose complete 81-frame window exists",
            },
            "tie_break": "earliest start",
            "input_fields": [
                "robot/robot.npz:actions",
                "robot/robot.npz:openings",
                "episode frame count",
            ],
            "known_future_action_is_conditioning_input": True,
            "object_geometry_or_tactile_used_for_selection": False,
            "inheritance": (
                "unchanged deform360-reusable-sota-window-v1 action-only rule "
                "locked on 2026-07-17"
            ),
        },
        "temporal staging rule changed",
    )


def _validate_information_boundary(config: Mapping[str, Any]) -> None:
    boundary = config.get("information_boundary")
    _require(isinstance(boundary, Mapping), "information boundary is missing")
    required_true = (
        "measurement_artifacts_hashed_before_target_open",
        "predictions_hashed_before_target_open",
        "observed_centers_permanently_excluded_from_scoring",
        "existing_frame_zero_confirmation_remains_sealed",
        "no_failed_object_replacement",
        "window_selection_may_read_full_robot_action",
    )
    for key in required_true:
        _require(boundary.get(key) is True, f"information boundary changed: {key}")
    required_false = (
        "future_dense_reconstruction_allowed_before_prediction_seal",
        "future_particle_tracks_allowed_before_prediction_seal",
        "target_metrics_allowed_before_prediction_seal",
        "window_selection_may_read_object_geometry",
        "window_selection_may_read_object_tracks",
        "window_selection_may_read_tactile",
    )
    for key in required_false:
        _require(boundary.get(key) is False, f"information boundary changed: {key}")


def _validate_target_construction(config: Mapping[str, Any]) -> None:
    target = config.get("target_construction")
    _require(isinstance(target, Mapping), "target construction contract is missing")
    _require(
        target
        == {
            "role": (
                "independent Deform360 outcome construction after an eligible "
                "all-case prediction cohort seal"
            ),
            "deform360_processing_revision": (
                "0fe36f0b7a7a917ba62b5f8cee707299a9a4a317"
            ),
            "source_sha256": {
                "reconstruct_stage": (
                    "53a1e8b73e56a1c68a0c4344b279c2817ed4b3ed93e8f5ea792def26d5099c7c"
                ),
                "urdf_render": (
                    "c4d6a10e980ed4952f974d2e8a991c6fb819a3e6fdc6c121d3ce6925c94c2467"
                ),
                "depth_stage": (
                    "34befb732107b805f1e1924699f1e26fc2ca5d3041561b920d8c23d8e85feef0"
                ),
                "tracking_stage": (
                    "04533cd9cd900ae2f5bd139568ed1a2442661f14ceda009dd7bb85e4fbd83ec2"
                ),
                "pcd_stage": (
                    "87553e1ea3dac5a90e46114c76aaf65901b43a064025626ae6871523065c864d"
                ),
            },
            "camera_policy": (
                "exact eight cameras sealed by the target-free raw-camera "
                "measurement"
            ),
            "future_mask_policy": (
                "propagate each sealed frame-zero mask with the pinned SAM2 "
                "model only after cohort authorization"
            ),
            "reconstruction": {
                "minimum_visual_hull_points": 512,
                "voxel_resolution": 120,
                "cube_half_extent_m": 0.5,
                "first_frame_iterations": 500,
                "warm_start_iterations": 250,
                "warm_start_from_previous_frame": True,
                "sealed_frame_zero_splat_reused_bit_exactly": True,
            },
            "depth": {
                "expected_depth": True,
                "object_mask_applied": True,
                "gripper_urdf_mask_applied": True,
                "preview_video": False,
            },
            "tracking": {
                "model": "facebook/cotracker3-scaled-offline",
                "repository_revision": (
                    "82e02e8029753ad4ef13cf06be7f4fc5facdda4d"
                ),
                "repository_tree": (
                    "f0296ad047b50c1530063b67e575908257478cab"
                ),
                "predictor_source_sha256": (
                    "783536d6c77790fa6c8e005f2df1d6bd4f0d8955c2c67d464bfb4e64d366375f"
                ),
                "checkpoint_sha256": (
                    "2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834"
                ),
                "pivot_skip": 5,
                "sequence_length": 15,
                "gap": 5,
                "grid_size": 40,
                "resize_factor": 4,
            },
            "point_cloud": {
                "seed_point_count": 10000,
                "crop_half_extent_m": 0.5,
                "radius_neighbors": 30,
                "radius_m": 0.02,
                "statistical_neighbors": 30,
                "statistical_std_ratio": 3.5,
                "fusion_ransac_threshold_m_per_s": 0.01,
                "fusion_minimum_inliers": 4,
                "rng_seed": 0,
                "raw_frame_count": 81,
                "output_frame_count": 76,
                "tracking_tail_frames_skipped": 5,
                "frame_rate_hz": 30.0,
            },
            "material_identity_contract": (
                "the full outcome frame-zero points must be bit-exact to the "
                "sealed prediction persistence frame zero"
            ),
            "future_tactile_read": False,
        },
        "target construction changed",
    )


def load_selective_virtual_sensing_protocol(
    path: str | Path,
) -> dict[str, Any]:
    """Load the canonical prospective lock and reject any silent amendment."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    _require(payload.get("schema_version") == SCHEMA_VERSION, "unsupported schema")
    declared_hash = payload.get("config_sha256")
    _require(
        declared_hash == protocol_config_sha256(payload),
        "selective virtual-sensing protocol checksum mismatch",
    )
    if _CANONICAL_CONFIG_SHA256:
        _require(
            declared_hash == _CANONICAL_CONFIG_SHA256,
            "selective virtual-sensing protocol differs from canonical lock",
        )
    config = payload.get("config")
    _require(isinstance(config, Mapping), "protocol config is missing")
    _require(config.get("protocol_id") == PROTOCOL_ID, "protocol ID changed")
    _require(
        config.get("status")
        == (
            "locked-before-selected-object-download-or-media-access-amended-"
            "window-and-target-rules"
        ),
        "protocol was not locked before media access",
    )
    dataset = config.get("dataset")
    _require(isinstance(dataset, Mapping), "dataset contract is missing")
    _require(dataset.get("repository") == DATASET_REPOSITORY, "dataset changed")
    _require(dataset.get("revision") == DATASET_REVISION, "dataset revision changed")
    cohort = config.get("cohort")
    _require(isinstance(cohort, Mapping), "cohort contract is missing")
    _require(cohort.get("selection_seed") == SELECTION_SEED, "selection seed changed")
    normalized = _validate_cohort(cohort.get("strata"))
    _validate_temporal_staging(config)
    _validate_method(config)
    _validate_information_boundary(config)
    _validate_target_construction(config)
    result = dict(payload)
    result["config"] = dict(config)
    result["normalized_cohort"] = normalized
    result["protocol_path"] = str(source)
    return result


__all__ = [
    "DATASET_REPOSITORY",
    "DATASET_REVISION",
    "EXPECTED_COHORT",
    "INELIGIBLE_OBJECTS",
    "PROTOCOL_ID",
    "load_selective_virtual_sensing_protocol",
    "metadata_ranked_episode_ids",
    "protocol_config_sha256",
]
