"""Pinned semantic and robot-exclusion gate for the fourth frame-zero fallback.

The module is deliberately light at import time.  OpenGL, Torch, Transformers,
and the 1.5 GB model snapshot are imported or opened only if all three earlier
frame-zero strategies have failed and the fourth strategy is actually tried.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

FRAME_ZERO_REFERENCE_OPTIONAL_FALLBACK_POLICY_ID = (
    "deform360-frame-zero-reference-conditioned-reference-optional-exact-eight-v1"
)
FRAME_ZERO_REFERENCE_OPTIONAL_ASSIGNMENT_STRATEGY = (
    "reference-conditioned-reference-optional-exhaustive-exact-eight-assignment"
)
FRAME_ZERO_REFERENCE_OPTIONAL_GEOMETRY_STRATEGY = (
    "reference-optional-semantic-urdf-projected-footprint"
)

_OFFICIAL_DEFORM360_COMMIT = "0fe36f0b7a7a917ba62b5f8cee707299a9a4a317"
_OFFICIAL_DEFORM360_BINDINGS = {
    "deform360/processing/urdf_render.py": (
        "c4d6a10e980ed4952f974d2e8a991c6fb819a3e6fdc6c121d3ce6925c94c2467"
    ),
    "deform360/processing/control_points_stage.py": (
        "9ff82c86c22e38c56dd2ce5d872850afb6ffeb502da7338baf0b55108afb7373"
    ),
    "deform360/robot.py": (
        "376e4dec6f2340a3ee03af1a3bd5462e06e3284cc82f312872a7bedbe863825f"
    ),
    "deform360/processing/assets/umi/umi.urdf": (
        "77d7fcc5d4e33aa6c0038a7998628a0b41cd4b629d1e4ce6357a37a98f0d08a0"
    ),
    "deform360/processing/assets/umi/meshes/finger.stl": (
        "93b610cf4eba79542e16c3186347deaf7f388d0ad35bde0ae85d21bacde40e76"
    ),
    "deform360/processing/assets/umi/meshes/holder_left.stl": (
        "5048f0a3fd320c039e938cae0b0e6ee2dd248ad55880c57fc2c0a2b70edf2df7"
    ),
    "deform360/processing/assets/umi/meshes/holder_right.stl": (
        "e783a0a39fcd8501e5917cb867f7cf0c6e90dcded63b983cb47549f744bb6aae"
    ),
    "deform360/processing/assets/umi/meshes/umi_gripper_base.stl": (
        "defb654cc54c81d338ff5397af1fc781e70e94f205c37adb8dcf35f05f019831"
    ),
}
_URDF_POLICY_ID = "official-umi-urdf-majority-overlap-rejection-v1"
_URDF_DILATION_RADIUS_PIXELS = 5
_URDF_REJECTION_FRACTION = 0.5

_SIGLIP_POLICY_ID = "siglip2-masked-crop-exclusive-five-label-exact8-v1"
_SIGLIP_REPO_ID = "google/siglip2-base-patch16-224"
_SIGLIP_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
_SIGLIP_MODEL_LOCK_SHA256 = (
    "e5696dc4650194fe2d773a7c5a197862e9d87dda6d7ee5cc45401d5b71f55239"
)
_SIGLIP_MODEL_TREE_SHA256 = (
    "b293f9bb1cd86272626d211d7c297c99d3aa1adfc4a3072b83e69f1fa70773ad"
)
_SIGLIP_WEIGHTS_SHA256 = (
    "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b"
)
_SIGLIP_LABELS = ("rope", "blanket", "scarf", "squirrel toy", "spider toy")
_SIGLIP_PROMPTS = tuple(f"a photo of a {label}" for label in _SIGLIP_LABELS)
_SIGLIP_BACKGROUND_RGB = (128, 128, 128)
_SIGLIP_CROP_MARGIN_SCALE = 1.10
_SIGLIP_OUTPUT_SIZE = 224
_SIGLIP_MINIMUM_TRUE_TOP1_VIEWS = 5
_TRANSFORMERS_SOURCE_BINDINGS = {
    "transformers.models.auto.modeling_auto": (
        "6e4fa67c88e02a8b84d46d7b1719e760f197073b7b233bcf30eeb596f5a5f07a"
    ),
    "transformers.models.siglip.modeling_siglip": (
        "60874564b9fd1fbd78d9aa3748b3adb506cf4b570f1ebd51dfcde32e0ba51d0d"
    ),
    "transformers.models.siglip.processing_siglip": (
        "3a2b7c83ac25331042a7d7e1eaabd2b8d64241c43164196ca16e9aefb1e3cc2e"
    ),
    "transformers.models.siglip.image_processing_siglip": (
        "f686c098214b194a88fb970e665fd86cdc9e322d34d6b0356acc36e6a1f6b13e"
    ),
    "transformers.models.gemma.tokenization_gemma_fast": (
        "884266d1b7a34986f50e69976dbe94b9165a7865f448b1b5b82c79bfadb98532"
    ),
}
_RUNTIME_VERSION_BINDINGS = {
    "PIL": "11.3.0",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "torch": "2.4.0+cu121",
    "transformers": "4.57.6",
}
_OBJECT_PREFIX_TO_LABEL = {
    "002": "rope",
    "081": "rope",
    "083": "blanket",
    "085": "scarf",
    "092": "squirrel toy",
    "170": "spider toy",
}

FRAME_ZERO_OFFICIAL_DEFORM360_COMMIT = _OFFICIAL_DEFORM360_COMMIT
FRAME_ZERO_OFFICIAL_DEFORM360_BINDINGS = dict(_OFFICIAL_DEFORM360_BINDINGS)
FRAME_ZERO_SIGLIP2_MODEL_REVISION = _SIGLIP_REVISION
FRAME_ZERO_SIGLIP2_MODEL_LOCK_SHA256 = _SIGLIP_MODEL_LOCK_SHA256
FRAME_ZERO_SIGLIP2_MODEL_TREE_SHA256 = _SIGLIP_MODEL_TREE_SHA256
FRAME_ZERO_SIGLIP2_WEIGHTS_SHA256 = _SIGLIP_WEIGHTS_SHA256
FRAME_ZERO_SIGLIP2_TRANSFORMERS_SOURCE_BINDINGS = tuple(
    _TRANSFORMERS_SOURCE_BINDINGS.items()
)
FRAME_ZERO_SIGLIP2_TRANSFORMERS_SOURCE_AGGREGATE_SHA256 = hashlib.sha256(
    json.dumps(
        list(FRAME_ZERO_SIGLIP2_TRANSFORMERS_SOURCE_BINDINGS),
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


FRAME_ZERO_SEMANTIC_GATE_CONTRACT: dict[str, Any] = {
    "contract_id": "deform360-frame-zero-fourth-fallback-semantic-gate-v1",
    "application_order": [
        "legacy",
        "same-masks-projected-footprint",
        "common-voxel-assignment-projected-footprint",
        FRAME_ZERO_REFERENCE_OPTIONAL_GEOMETRY_STRATEGY,
    ],
    "fourth_fallback": {
        "policy_id": FRAME_ZERO_REFERENCE_OPTIONAL_FALLBACK_POLICY_ID,
        "assignment_strategy": FRAME_ZERO_REFERENCE_OPTIONAL_ASSIGNMENT_STRATEGY,
        "geometry_strategy": FRAME_ZERO_REFERENCE_OPTIONAL_GEOMETRY_STRATEGY,
        "semantic_conditioning": (
            "fixed reference and frozen top-eight local-score seeds"
        ),
        "geometry_support": (
            "strict at-least-eight per-camera union support without a "
            "reference-hit intersection"
        ),
        "camera_selection": (
            "exhaustive lexicographic evaluation of every eligible exact-eight "
            "camera subset; the fixed reference may abstain"
        ),
    },
    "official_urdf": {
        "policy_id": _URDF_POLICY_ID,
        "repository_commit": _OFFICIAL_DEFORM360_COMMIT,
        "source_and_asset_sha256": dict(_OFFICIAL_DEFORM360_BINDINGS),
        "render": (
            "official PyrenderGripperRenderer; camera pose equals "
            "invert_transform(T_worlds[0,g]) @ camera_to_world"
        ),
        "dilation": "square/Chebyshev radius 5 pixels",
        "proposal_overlap_fraction": (
            "count(proposal AND dilated_URDF) / count(proposal)"
        ),
        "reject_if_fraction_greater_than_or_equal": _URDF_REJECTION_FRACTION,
        "application": (
            "reject every robot-dominated proposal before basic eligibility, "
            "preserving proposal indices; subtract the same dilated silhouette "
            "from selected masks before geometry"
        ),
    },
    "siglip2": {
        "policy_id": _SIGLIP_POLICY_ID,
        "repo_id": _SIGLIP_REPO_ID,
        "revision": _SIGLIP_REVISION,
        "model_lock_sha256": _SIGLIP_MODEL_LOCK_SHA256,
        "model_tree_sha256": _SIGLIP_MODEL_TREE_SHA256,
        "model_safetensors_sha256": _SIGLIP_WEIGHTS_SHA256,
        "labels_in_order": list(_SIGLIP_LABELS),
        "prompts_in_order": list(_SIGLIP_PROMPTS),
        "crop": {
            "outside_mask_and_padding_rgb": list(_SIGLIP_BACKGROUND_RGB),
            "bbox": "tight inclusive nonzero-mask pixel bounding box",
            "square_side": "ceil(1.10 * max(inclusive width, inclusive height))",
            "center": "tight-bbox center in pixel-edge coordinates",
            "origin": "floor(center - side/2) independently per axis",
            "resize": "PIL bicubic to exactly 224x224",
        },
        "rank": {
            "exact_top_ties_fail": True,
            "minimum_true_label_object_top1_views": 5,
            "true_label_unique_top1_by_mean_raw_logit": True,
            "background_gate_minimum_available_controls": 5,
            "background_true_label_top1_votes_must_be_less_than": 5,
            "background_mean_must_not_have_true_label_unique_top1": True,
        },
        "inference": {
            "dtype": "float32",
            "autocast": False,
            "tf32": False,
            "deterministic_algorithms": True,
            "seed": 0,
            "offline_local_files_only": True,
            "score": "raw logits_per_image",
        },
        "runtime_versions": dict(_RUNTIME_VERSION_BINDINGS),
        "transformers_source_sha256": dict(_TRANSFORMERS_SOURCE_BINDINGS),
        "transformers_source_ordered_aggregate_sha256": (
            FRAME_ZERO_SIGLIP2_TRANSFORMERS_SOURCE_AGGREGATE_SHA256
        ),
    },
    "explicitly_excluded": [
        "GroundingDINO/GroundedSAM prompt segmentation",
        "taxel or interaction proximity gate",
        "target, outcome, tactile, future RGB, or confirmation data",
    ],
}
FRAME_ZERO_SEMANTIC_GATE_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        FRAME_ZERO_SEMANTIC_GATE_CONTRACT,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def artifact_sha256(value: Mapping[str, Any]) -> str:
    canonical = dict(value)
    canonical.pop("artifact_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    descriptor = _canonical_bytes(
        {"dtype": array.dtype.str, "shape": list(array.shape)}
    )
    return hashlib.sha256(
        descriptor + b"\0" + array.view(np.uint8).tobytes()
    ).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _pil_image_module() -> Any:
    try:
        return importlib.import_module("PIL.Image")
    except ImportError as error:  # pragma: no cover - optional fourth path
        raise RuntimeError(
            "Pillow is required by the fourth frame-zero fallback"
        ) from error


@contextmanager
def _deterministic_siglip_torch_state(torch: Any) -> Any:
    """Temporarily enforce the frozen SigLIP math without leaking globals."""

    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_matmul_precision = torch.get_float32_matmul_precision()
    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.use_deterministic_algorithms(
            previous_deterministic, warn_only=previous_warn_only
        )
        torch.set_float32_matmul_precision(previous_matmul_precision)
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32


def semantic_label_for_object_id(object_id: str) -> str:
    """Map the public object name to the preregistered exclusive label."""

    _require(isinstance(object_id, str) and object_id, "object id is missing")
    prefix = object_id.split("-", 1)[0]
    _require(
        prefix in _OBJECT_PREFIX_TO_LABEL,
        "object id is outside the frozen SigLIP2 true-label map",
    )
    return _OBJECT_PREFIX_TO_LABEL[prefix]


def masked_square_crop(
    rgb: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the exact preregistered neutral-background object crop."""

    image = np.asarray(rgb, dtype=np.uint8)
    selected = np.asarray(mask, dtype=bool)
    _require(
        image.ndim == 3 and image.shape[2] == 3 and selected.shape == image.shape[:2],
        "SigLIP2 RGB/mask shape mismatch",
    )
    ys, xs = np.nonzero(selected)
    _require(bool(len(xs)), "cannot crop an empty selected mask")
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    box_width = x_max - x_min + 1
    box_height = y_max - y_min + 1
    side = max(
        1,
        int(math.ceil(_SIGLIP_CROP_MARGIN_SCALE * max(box_width, box_height))),
    )
    center_x_edge = 0.5 * (x_min + x_max + 1)
    center_y_edge = 0.5 * (y_min + y_max + 1)
    x0 = int(math.floor(center_x_edge - side / 2.0))
    y0 = int(math.floor(center_y_edge - side / 2.0))
    x1, y1 = x0 + side, y0 + side
    masked = np.full_like(image, _SIGLIP_BACKGROUND_RGB, dtype=np.uint8)
    masked[selected] = image[selected]
    square = np.full((side, side, 3), _SIGLIP_BACKGROUND_RGB, dtype=np.uint8)
    image_height, image_width = image.shape[:2]
    source_x0, source_y0 = max(0, x0), max(0, y0)
    source_x1, source_y1 = min(image_width, x1), min(image_height, y1)
    if source_x1 > source_x0 and source_y1 > source_y0:
        square[
            source_y0 - y0 : source_y1 - y0,
            source_x0 - x0 : source_x1 - x0,
        ] = masked[source_y0:source_y1, source_x0:source_x1]
    image_module = _pil_image_module()
    resized = np.asarray(
        image_module.fromarray(square).resize(
            (_SIGLIP_OUTPUT_SIZE, _SIGLIP_OUTPUT_SIZE),
            image_module.Resampling.BICUBIC,
        ),
        dtype=np.uint8,
    )
    return resized, {
        "mask_bbox_xyxy_inclusive": [x_min, y_min, x_max, y_max],
        "mask_bbox_width_pixels": box_width,
        "mask_bbox_height_pixels": box_height,
        "crop_margin_scale": _SIGLIP_CROP_MARGIN_SCALE,
        "square_side_pixels": side,
        "square_xyxy_exclusive_unclipped": [x0, y0, x1, y1],
        "square_source_xyxy_exclusive_clipped": [
            source_x0,
            source_y0,
            source_x1,
            source_y1,
        ],
        "neutral_background_rgb": list(_SIGLIP_BACKGROUND_RGB),
        "resize": "PIL bicubic to 224x224",
        "crop_rgb_sha256": sha256_array(resized),
    }


def background_control_crop(
    rgb: np.ndarray, mask: np.ndarray, *, side: int
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Select the exact preregistered zero-overlap corner control."""

    image = np.asarray(rgb, dtype=np.uint8)
    selected = np.asarray(mask, dtype=bool)
    _require(
        image.ndim == 3 and image.shape[2] == 3 and selected.shape == image.shape[:2],
        "background-control RGB/mask shape mismatch",
    )
    height, width = selected.shape
    if side > min(height, width):
        return None, {
            "available": False,
            "reason": "object square side exceeds at least one image dimension",
            "requested_side_pixels": side,
        }
    ys, xs = np.nonzero(selected)
    _require(bool(len(xs)), "cannot build a control for an empty mask")
    centroid_x, centroid_y = float(np.mean(xs)), float(np.mean(ys))
    candidates = (
        (0, 0, "top-left"),
        (width - side, 0, "top-right"),
        (0, height - side, "bottom-left"),
        (width - side, height - side, "bottom-right"),
    )
    records = []
    for order, (x0, y0, name) in enumerate(candidates):
        x1, y1 = x0 + side, y0 + side
        overlap = int(selected[y0:y1, x0:x1].sum())
        distance_squared = (x0 + side / 2.0 - centroid_x) ** 2 + (
            y0 + side / 2.0 - centroid_y
        ) ** 2
        records.append(
            {
                "order": order,
                "name": name,
                "xyxy_exclusive": [x0, y0, x1, y1],
                "selected_mask_overlap_pixels": overlap,
                "squared_distance_to_selected_mask_centroid_pixels2": (
                    distance_squared
                ),
            }
        )
    chosen = min(
        records,
        key=lambda item: (
            item["selected_mask_overlap_pixels"],
            -item["squared_distance_to_selected_mask_centroid_pixels2"],
            item["order"],
        ),
    )
    if chosen["selected_mask_overlap_pixels"] != 0:
        return None, {
            "available": False,
            "reason": "no corner crop has zero selected-mask overlap",
            "requested_side_pixels": side,
            "candidates": records,
            "chosen_if_overlap_were_allowed": chosen,
        }
    x0, y0, x1, y1 = chosen["xyxy_exclusive"]
    image_module = _pil_image_module()
    crop = np.asarray(
        image_module.fromarray(image[y0:y1, x0:x1]).resize(
            (_SIGLIP_OUTPUT_SIZE, _SIGLIP_OUTPUT_SIZE),
            image_module.Resampling.BICUBIC,
        ),
        dtype=np.uint8,
    )
    return crop, {
        "available": True,
        "selection_key": (
            "minimum selected-mask overlap, then farthest squared centroid "
            "distance, then fixed corner order"
        ),
        "requested_side_pixels": side,
        "candidates": records,
        "chosen": chosen,
        "resize": "PIL bicubic to 224x224",
        "crop_rgb_sha256": sha256_array(crop),
    }


def unique_rank_record(logits: np.ndarray) -> dict[str, Any]:
    values = np.asarray(logits, dtype=np.float64)
    _require(values.shape == (len(_SIGLIP_LABELS),), "invalid SigLIP2 logit vector")
    _require(np.all(np.isfinite(values)), "SigLIP2 logits are non-finite")
    maximum = float(np.max(values))
    winners = np.flatnonzero(values == maximum).astype(int).tolist()
    ranking = sorted(
        range(len(_SIGLIP_LABELS)), key=lambda index: (-values[index], index)
    )
    return {
        "logits_by_label": {
            label: float(values[index]) for index, label in enumerate(_SIGLIP_LABELS)
        },
        "ranking": [_SIGLIP_LABELS[index] for index in ranking],
        "unique_top1": len(winners) == 1,
        "top1_label": (_SIGLIP_LABELS[winners[0]] if len(winners) == 1 else None),
        "exact_top_tie_labels": [_SIGLIP_LABELS[index] for index in winners],
    }


def evaluate_rank_gate(
    logits: np.ndarray,
    *,
    object_cameras: Sequence[str],
    background_cameras: Sequence[str],
    true_label: str,
) -> dict[str, Any]:
    """Recompute the entire exclusive-rank admission decision from logits."""

    object_order = list(object_cameras)
    background_order = list(background_cameras)
    _require(
        object_order == sorted(set(object_order))
        and len(object_order) == 8
        and background_order == sorted(set(background_order))
        and set(background_order) <= set(object_order),
        "invalid SigLIP2 image order",
    )
    _require(true_label in _SIGLIP_LABELS, "invalid true SigLIP2 label")
    values = np.asarray(logits, dtype=np.float64)
    _require(
        values.shape
        == (len(object_order) + len(background_order), len(_SIGLIP_LABELS)),
        "unexpected SigLIP2 logits shape",
    )
    object_ranks = {
        camera: unique_rank_record(values[index])
        for index, camera in enumerate(object_order)
    }
    offset = len(object_order)
    background_ranks = {
        camera: unique_rank_record(values[offset + index])
        for index, camera in enumerate(background_order)
    }
    object_votes = {
        label: sum(record["top1_label"] == label for record in object_ranks.values())
        for label in _SIGLIP_LABELS
    }
    background_votes = {
        label: sum(
            record["top1_label"] == label for record in background_ranks.values()
        )
        for label in _SIGLIP_LABELS
    }
    object_mean = unique_rank_record(values[:offset].mean(axis=0, dtype=np.float64))
    background_mean = (
        unique_rank_record(values[offset:].mean(axis=0, dtype=np.float64))
        if background_order
        else None
    )
    background_gate_applicable = (
        len(background_order) >= _SIGLIP_MINIMUM_TRUE_TOP1_VIEWS
    )
    no_ties = all(record["unique_top1"] for record in object_ranks.values()) and (
        all(record["unique_top1"] for record in background_ranks.values())
        if background_gate_applicable
        else True
    )
    object_vote_gate = object_votes[true_label] >= _SIGLIP_MINIMUM_TRUE_TOP1_VIEWS
    object_mean_gate = bool(
        object_mean["unique_top1"] and object_mean["top1_label"] == true_label
    )
    background_vote_gate = (
        background_votes[true_label] < _SIGLIP_MINIMUM_TRUE_TOP1_VIEWS
        if background_gate_applicable
        else True
    )
    background_mean_gate = (
        bool(
            background_mean is not None
            and background_mean["unique_top1"]
            and background_mean["top1_label"] != true_label
        )
        if background_gate_applicable
        else True
    )
    passed = bool(
        no_ties
        and object_vote_gate
        and object_mean_gate
        and background_vote_gate
        and background_mean_gate
    )
    return {
        "object_rank": {
            "per_camera": object_ranks,
            "exclusive_top1_votes": object_votes,
            "mean_logit_rank": object_mean,
        },
        "background_control_rank": {
            "available_camera_count": len(background_order),
            "per_camera": background_ranks,
            "exclusive_top1_votes": background_votes,
            "mean_logit_rank": background_mean,
            "gate_applicable": background_gate_applicable,
        },
        "decision": {
            "status": "pass" if passed else "fail",
            "all_applicable_ranks_have_unique_top1": no_ties,
            "true_label": true_label,
            "minimum_true_top1_views": _SIGLIP_MINIMUM_TRUE_TOP1_VIEWS,
            "true_label_object_top1_vote_count": object_votes[true_label],
            "object_vote_gate_passed": object_vote_gate,
            "object_mean_logit_top1_gate_passed": object_mean_gate,
            "background_control_gate_applicable": background_gate_applicable,
            "background_true_label_top1_vote_count": background_votes[true_label],
            "background_vote_gate_passed": background_vote_gate,
            "background_mean_logit_not_true_label_gate_passed": (background_mean_gate),
            "prospective_optional_fallback_admission": passed,
            "no_prompt_crop_model_or_rank_tuning_performed": True,
        },
    }


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_semantic_gate_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute all rank decisions and require every immutable binding."""

    top_keys = {
        "policy_id",
        "contract_sha256",
        "bindings",
        "selected_exact8",
        "model_input",
        "object_rank",
        "background_control_rank",
        "decision",
        "artifact_sha256",
    }
    _require(
        isinstance(audit, Mapping)
        and set(audit) == top_keys
        and audit.get("artifact_sha256") == artifact_sha256(audit)
        and audit.get("policy_id") == _SIGLIP_POLICY_ID
        and audit.get("contract_sha256") == FRAME_ZERO_SEMANTIC_GATE_CONTRACT_SHA256,
        "invalid SigLIP2 semantic-gate audit",
    )
    bindings = audit["bindings"]
    binding_keys = {
        "model_root",
        "model_lock",
        "repo_id",
        "revision",
        "tree_artifact_sha256",
        "model_safetensors_sha256",
        "regular_file_count",
        "runtime_versions",
        "transformers_source_files",
        "transformers_source_ordered_aggregate_sha256",
    }
    _require(
        isinstance(bindings, Mapping)
        and set(bindings) == binding_keys
        and Path(str(bindings.get("model_root"))).is_absolute()
        and isinstance(bindings.get("model_lock"), Mapping)
        and set(bindings["model_lock"]) == {"path", "sha256"}
        and Path(str(bindings["model_lock"].get("path"))).is_absolute()
        and bindings["model_lock"].get("sha256") == _SIGLIP_MODEL_LOCK_SHA256
        and bindings.get("repo_id") == _SIGLIP_REPO_ID
        and bindings.get("revision") == _SIGLIP_REVISION
        and bindings.get("tree_artifact_sha256") == _SIGLIP_MODEL_TREE_SHA256
        and bindings.get("model_safetensors_sha256") == _SIGLIP_WEIGHTS_SHA256
        and bindings.get("regular_file_count") == 19
        and bindings.get("runtime_versions") == _RUNTIME_VERSION_BINDINGS
        and bindings.get("transformers_source_ordered_aggregate_sha256")
        == FRAME_ZERO_SIGLIP2_TRANSFORMERS_SOURCE_AGGREGATE_SHA256,
        "SigLIP2 immutable binding audit changed",
    )
    source_records = bindings["transformers_source_files"]
    _require(
        isinstance(source_records, list)
        and len(source_records) == len(_TRANSFORMERS_SOURCE_BINDINGS)
        and all(
            isinstance(record, Mapping)
            and set(record) == {"module", "path", "sha256"}
            and record.get("module") == module_name
            and Path(str(record.get("path"))).is_absolute()
            and record.get("sha256") == expected_sha256
            for record, (module_name, expected_sha256) in zip(
                source_records,
                _TRANSFORMERS_SOURCE_BINDINGS.items(),
                strict=True,
            )
        ),
        "ordered Transformers source binding audit changed",
    )
    selected = audit["selected_exact8"]
    selected_keys = {
        "camera",
        "candidate_index",
        "rgb_sha256",
        "selected_mask_sha256",
        "object",
        "background_control",
    }
    object_crop_keys = {
        "mask_bbox_xyxy_inclusive",
        "mask_bbox_width_pixels",
        "mask_bbox_height_pixels",
        "crop_margin_scale",
        "square_side_pixels",
        "square_xyxy_exclusive_unclipped",
        "square_source_xyxy_exclusive_clipped",
        "neutral_background_rgb",
        "resize",
        "crop_rgb_sha256",
    }
    _require(
        isinstance(selected, list)
        and len(selected) == 8
        and all(
            isinstance(record, Mapping) and set(record) == selected_keys
            for record in selected
        ),
        "SigLIP2 selected exact-eight audit changed",
    )
    cameras = [str(record["camera"]) for record in selected]
    _require(cameras == sorted(set(cameras)), "SigLIP2 camera order changed")
    background_cameras = []
    for record in selected:
        object_crop = record["object"]
        background = record["background_control"]
        _require(
            isinstance(record.get("candidate_index"), int)
            and not isinstance(record.get("candidate_index"), bool)
            and int(record["candidate_index"]) >= 0
            and _valid_sha256(record.get("rgb_sha256"))
            and _valid_sha256(record.get("selected_mask_sha256"))
            and isinstance(object_crop, Mapping)
            and set(object_crop) == object_crop_keys
            and object_crop.get("crop_margin_scale") == _SIGLIP_CROP_MARGIN_SCALE
            and object_crop.get("neutral_background_rgb")
            == list(_SIGLIP_BACKGROUND_RGB)
            and object_crop.get("resize") == "PIL bicubic to 224x224"
            and _valid_sha256(object_crop.get("crop_rgb_sha256"))
            and isinstance(object_crop.get("mask_bbox_xyxy_inclusive"), list)
            and len(object_crop["mask_bbox_xyxy_inclusive"]) == 4
            and isinstance(object_crop.get("mask_bbox_width_pixels"), int)
            and object_crop["mask_bbox_width_pixels"] >= 1
            and isinstance(object_crop.get("mask_bbox_height_pixels"), int)
            and object_crop["mask_bbox_height_pixels"] >= 1
            and object_crop.get("square_side_pixels")
            == math.ceil(
                _SIGLIP_CROP_MARGIN_SCALE
                * max(
                    object_crop["mask_bbox_width_pixels"],
                    object_crop["mask_bbox_height_pixels"],
                )
            )
            and isinstance(background, Mapping)
            and isinstance(background.get("available"), bool),
            "invalid SigLIP2 crop audit",
        )
        x_min, y_min, x_max, y_max = object_crop["mask_bbox_xyxy_inclusive"]
        _require(
            object_crop["mask_bbox_width_pixels"] == x_max - x_min + 1
            and object_crop["mask_bbox_height_pixels"] == y_max - y_min + 1,
            "SigLIP2 inclusive crop bounds changed",
        )
        if background["available"]:
            _require(
                background.get("selection_key")
                == (
                    "minimum selected-mask overlap, then farthest squared centroid "
                    "distance, then fixed corner order"
                )
                and background.get("requested_side_pixels")
                == object_crop["square_side_pixels"]
                and background.get("resize") == "PIL bicubic to 224x224"
                and _valid_sha256(background.get("crop_rgb_sha256"))
                and isinstance(background.get("chosen"), Mapping)
                and background["chosen"].get("selected_mask_overlap_pixels") == 0,
                "available SigLIP2 background crop changed",
            )
            background_cameras.append(str(record["camera"]))
    model_input = audit["model_input"]
    input_keys = {
        "labels_in_exclusive_order",
        "text_prompts_in_order",
        "image_order",
        "processor_tensor_sha256",
        "logits_shape",
        "logits_float64",
        "logits_float64_sha256",
    }
    expected_image_order = [
        {"kind": "object", "camera": camera} for camera in cameras
    ] + [
        {"kind": "background_control", "camera": camera}
        for camera in background_cameras
    ]
    _require(
        isinstance(model_input, Mapping)
        and set(model_input) == input_keys
        and model_input.get("labels_in_exclusive_order") == list(_SIGLIP_LABELS)
        and model_input.get("text_prompts_in_order") == list(_SIGLIP_PROMPTS)
        and model_input.get("image_order") == expected_image_order
        and isinstance(model_input.get("processor_tensor_sha256"), Mapping)
        and bool(model_input["processor_tensor_sha256"])
        and all(
            isinstance(key, str) and _valid_sha256(value)
            for key, value in model_input["processor_tensor_sha256"].items()
        ),
        "SigLIP2 model-input audit changed",
    )
    logits = np.asarray(model_input.get("logits_float64"), dtype=np.float64)
    _require(
        model_input.get("logits_shape") == list(logits.shape)
        and logits.shape == (len(expected_image_order), len(_SIGLIP_LABELS))
        and np.all(np.isfinite(logits))
        and model_input.get("logits_float64_sha256") == sha256_array(logits),
        "SigLIP2 logits binding changed",
    )
    true_label = audit["decision"].get("true_label")
    _require(true_label in _SIGLIP_LABELS, "SigLIP2 true label changed")
    expected_rank = evaluate_rank_gate(
        logits,
        object_cameras=cameras,
        background_cameras=background_cameras,
        true_label=true_label,
    )
    _require(
        audit.get("object_rank") == expected_rank["object_rank"]
        and audit.get("background_control_rank")
        == expected_rank["background_control_rank"]
        and audit.get("decision") == expected_rank["decision"]
        and expected_rank["decision"]["status"] == "pass",
        "SigLIP2 rank decision is not recomputable",
    )
    return {
        "passed": True,
        "artifact_sha256": audit["artifact_sha256"],
        "true_label": true_label,
        "selected_cameras": cameras,
    }


def verify_model_tree(
    model_root: str | Path, model_lock_path: str | Path
) -> dict[str, Any]:
    """Verify every snapshot byte, its immutable lock, modes, and revision."""

    root_input = Path(model_root)
    lock_input = Path(model_lock_path)
    _require(
        root_input.is_absolute()
        and root_input.exists()
        and not root_input.is_symlink()
        and root_input == root_input.resolve(strict=True),
        "SigLIP2 snapshot path is not absolute canonical non-symlink material",
    )
    _require(
        lock_input.is_absolute()
        and lock_input.exists()
        and not lock_input.is_symlink()
        and lock_input == lock_input.resolve(strict=True),
        "SigLIP2 model-lock path is not absolute canonical non-symlink material",
    )
    root = root_input
    lock_path = lock_input
    _require(root.is_dir(), "SigLIP2 snapshot is missing")
    _require(
        lock_path.is_file()
        and not lock_path.is_symlink()
        and stat.S_IMODE(lock_path.stat().st_mode) == 0o400
        and sha256_file(lock_path) == _SIGLIP_MODEL_LOCK_SHA256,
        "SigLIP2 model lock differs from the frozen lock",
    )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    _require(
        lock.get("upstream", {}).get("repo_id") == _SIGLIP_REPO_ID
        and lock.get("upstream", {}).get("resolved_revision") == _SIGLIP_REVISION,
        "SigLIP2 upstream binding changed",
    )
    expected = [
        {key: item[key] for key in ("path", "bytes", "sha256")}
        for item in lock["local_snapshot"]["files"]
    ]
    all_entries = list(root.rglob("*"))
    _require(
        all(not path.is_symlink() for path in all_entries),
        "SigLIP2 snapshot contains a symlink",
    )
    actual = []
    for path in sorted(
        (item for item in all_entries if item.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        actual.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    _require(actual == expected, "SigLIP2 snapshot inventory/hash drift")
    directories = ["."] + [
        path.relative_to(root).as_posix()
        for path in sorted(
            (item for item in all_entries if item.is_dir()),
            key=lambda item: (len(item.parts), item.as_posix()),
        )
    ]
    tree_artifact = hashlib.sha256(
        json.dumps(
            {"directories": directories, "files": actual},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    _require(
        tree_artifact
        == lock["local_snapshot"]["tree_artifact_sha256"]
        == _SIGLIP_MODEL_TREE_SHA256,
        "SigLIP2 snapshot tree artifact drift",
    )
    _require(
        stat.S_IMODE(root.stat().st_mode) == 0o500
        and all(
            stat.S_IMODE(path.stat().st_mode) == 0o500
            for path in all_entries
            if path.is_dir()
        )
        and all(
            stat.S_IMODE(path.stat().st_mode) == 0o400
            for path in all_entries
            if path.is_file()
        ),
        "SigLIP2 snapshot is not sealed read-only",
    )
    _require(
        sha256_file(root / "model.safetensors") == _SIGLIP_WEIGHTS_SHA256,
        "SigLIP2 model weights changed",
    )
    return {
        "model_root": str(root),
        "model_lock": {
            "path": str(lock_path),
            "sha256": _SIGLIP_MODEL_LOCK_SHA256,
        },
        "repo_id": _SIGLIP_REPO_ID,
        "revision": _SIGLIP_REVISION,
        "tree_artifact_sha256": tree_artifact,
        "model_safetensors_sha256": _SIGLIP_WEIGHTS_SHA256,
        "regular_file_count": len(actual),
    }


def robot_proposal_overlap_record(
    mask: np.ndarray,
    exact_robot_mask: np.ndarray,
    dilated_robot_mask: np.ndarray,
) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    exact_robot = np.asarray(exact_robot_mask, dtype=bool)
    dilated_robot = np.asarray(dilated_robot_mask, dtype=bool)
    _require(
        selected.shape == exact_robot.shape == dilated_robot.shape,
        "proposal/URDF shape mismatch",
    )
    count = int(np.count_nonzero(selected))
    _require(count > 0, "proposal mask is empty")
    exact = int(np.count_nonzero(selected & exact_robot))
    broad = int(np.count_nonzero(selected & dilated_robot))
    fraction = broad / count
    return {
        "mask_pixel_count": count,
        "exact_robot_intersection_pixel_count": exact,
        "exact_robot_overlap_fraction": exact / count,
        "dilated_robot_intersection_pixel_count": broad,
        "dilated_robot_overlap_fraction": fraction,
        "rejected_as_robot_dominated": fraction >= _URDF_REJECTION_FRACTION,
    }


def filter_robot_dominated_proposals(
    proposals_by_camera: Mapping[str, Sequence[Mapping[str, Any]]],
    exact_robot_masks: Mapping[str, np.ndarray],
    dilated_robot_masks: Mapping[str, np.ndarray],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Apply majority exclusion without renumbering automatic proposals."""

    cameras = tuple(sorted(proposals_by_camera))
    _require(
        cameras
        == tuple(sorted(exact_robot_masks))
        == tuple(sorted(dilated_robot_masks)),
        "proposal/URDF camera sets differ",
    )
    filtered: dict[str, list[dict[str, Any]]] = {}
    per_camera = []
    for camera in cameras:
        records = []
        copied_records = []
        for candidate_index, proposal in enumerate(proposals_by_camera[camera]):
            mask = np.asarray(proposal["segmentation"], dtype=bool)
            overlap = robot_proposal_overlap_record(
                mask,
                exact_robot_masks[camera],
                dilated_robot_masks[camera],
            )
            copied = dict(proposal)
            if overlap["rejected_as_robot_dominated"]:
                copied["predicted_iou"] = -1.0
            copied_records.append(copied)
            records.append(
                {
                    "candidate_index": candidate_index,
                    "mask_sha256": sha256_array(mask),
                    **overlap,
                }
            )
        filtered[camera] = copied_records
        per_camera.append(
            {
                "camera": camera,
                "automatic_candidate_count": len(copied_records),
                "rejected_candidate_count": sum(
                    record["rejected_as_robot_dominated"] for record in records
                ),
                "exact_robot_mask_sha256": sha256_array(exact_robot_masks[camera]),
                "dilated_robot_mask_sha256": sha256_array(dilated_robot_masks[camera]),
                "candidates": records,
            }
        )
    audit = {
        "policy_id": _URDF_POLICY_ID,
        "dilation_shape": "square/Chebyshev",
        "dilation_radius_pixels": _URDF_DILATION_RADIUS_PIXELS,
        "kernel_shape_pixels": [11, 11],
        "overlap_fraction": ("count(proposal AND dilated_URDF) / count(proposal)"),
        "reject_if_overlap_fraction_greater_than_or_equal": (_URDF_REJECTION_FRACTION),
        "candidate_indices_preserved": True,
        "per_camera": per_camera,
    }
    audit["artifact_sha256"] = artifact_sha256(audit)
    return filtered, audit


def subtract_robot_from_selected_masks(
    selected_masks: Mapping[str, np.ndarray],
    exact_robot_masks: Mapping[str, np.ndarray],
    dilated_robot_masks: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cameras = tuple(sorted(selected_masks))
    _require(
        set(cameras) <= set(exact_robot_masks)
        and set(cameras) <= set(dilated_robot_masks),
        "selected-mask URDF cameras differ",
    )
    subtracted = {}
    records = []
    for camera in cameras:
        original = np.asarray(selected_masks[camera], dtype=bool)
        exact = np.asarray(exact_robot_masks[camera], dtype=bool)
        dilated = np.asarray(dilated_robot_masks[camera], dtype=bool)
        _require(original.shape == exact.shape == dilated.shape, "URDF shape drift")
        final = original & ~dilated
        _require(np.any(final), "URDF subtraction emptied a selected mask")
        subtracted[camera] = final
        records.append(
            {
                "camera": camera,
                "selected_mask_sha256_before_subtraction": sha256_array(original),
                "exact_robot_mask_sha256": sha256_array(exact),
                "dilated_robot_mask_sha256": sha256_array(dilated),
                "selected_pixel_count_before_subtraction": int(original.sum()),
                "removed_pixel_count": int(np.count_nonzero(original & dilated)),
                "selected_pixel_count_after_subtraction": int(final.sum()),
                "selected_mask_sha256_after_subtraction": sha256_array(final),
            }
        )
    audit = {
        "policy_id": _URDF_POLICY_ID,
        "operation": "selected_mask AND NOT dilated_official_URDF",
        "per_camera": records,
    }
    audit["artifact_sha256"] = artifact_sha256(audit)
    return subtracted, audit


class PinnedFrameZeroSemanticGateRuntime:
    """Lazy official-URDF renderer and pinned offline SigLIP2 ranker."""

    def __init__(
        self,
        model_root: str | Path,
        model_lock_path: str | Path,
        deform360_code: str | Path,
        *,
        device: str = "cuda",
    ) -> None:
        # Keep the supplied paths un-resolved until the lazy verifier can
        # detect final-component and ancestor symlinks rather than erasing
        # that evidence with ``resolve()`` in the constructor.
        self.model_root = Path(model_root)
        self.model_lock_path = Path(model_lock_path)
        self.deform360_code = Path(deform360_code)
        self.device = device
        self.model_id = f"{_SIGLIP_REPO_ID}@{_SIGLIP_REVISION}"
        self._bindings: dict[str, Any] | None = None

    def _verify_official_runtime(self) -> dict[str, Any]:
        root = self.deform360_code
        _require(
            root.is_absolute()
            and root.exists()
            and root.is_dir()
            and not root.is_symlink()
            and root == root.resolve(strict=True),
            "Deform360 code path is not absolute canonical non-symlink material",
        )
        _require(
            all(not path.is_symlink() for path in root.rglob("*")),
            "official Deform360 tree contains a symlink",
        )
        try:
            revision = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip()
            dirty = subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain=v1"], text=True
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError("official Deform360 repository is unavailable") from error
        _require(
            revision == _OFFICIAL_DEFORM360_COMMIT and not dirty,
            "official Deform360 source binding drift",
        )
        observed = {
            relative: sha256_file(root / relative)
            for relative in _OFFICIAL_DEFORM360_BINDINGS
        }
        _require(
            observed == _OFFICIAL_DEFORM360_BINDINGS,
            "official URDF source/asset hash drift",
        )
        return {
            "repository": str(root),
            "commit": revision,
            "source_and_asset_sha256": observed,
        }

    def _verify_bindings(self) -> dict[str, Any]:
        if self._bindings is None:
            self._bindings = {
                "official_deform360": self._verify_official_runtime(),
                "siglip2": verify_model_tree(self.model_root, self.model_lock_path),
            }
        return self._bindings

    def prepare_proposals(
        self,
        proposals_by_camera: Mapping[str, Sequence[Mapping[str, Any]]],
        rgb_by_camera: Mapping[str, np.ndarray],
        intrinsics_by_camera: Mapping[str, np.ndarray],
        camera_to_world_by_camera: Mapping[str, np.ndarray],
        selected_action_arrays: Mapping[str, np.ndarray],
    ) -> tuple[
        dict[str, list[dict[str, Any]]],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, Any],
    ]:
        """Render frame-zero URDF masks and exclude robot-majority proposals."""

        bindings = self._verify_bindings()
        cameras = tuple(sorted(proposals_by_camera))
        _require(
            cameras == tuple(sorted(rgb_by_camera))
            and set(cameras) <= set(intrinsics_by_camera)
            and set(cameras) <= set(camera_to_world_by_camera),
            "URDF input camera sets differ",
        )
        shapes = {np.asarray(rgb_by_camera[camera]).shape for camera in cameras}
        _require(len(shapes) == 1, "URDF RGB shapes differ")
        height, width, channels = next(iter(shapes))
        _require(channels == 3, "URDF RGB is not three-channel")
        transforms_all = np.asarray(selected_action_arrays["T_worlds"])
        openings_all = np.asarray(selected_action_arrays["openings"])
        bimanual = bool(np.asarray(selected_action_arrays["bimanual"]).item())
        _require(
            len(transforms_all) >= 1 and len(openings_all) >= 1, "robot slice empty"
        )
        transforms = transforms_all[0]
        openings = openings_all[0]
        if bimanual:
            _require(
                transforms.shape == (2, 4, 4) and openings.shape == (2,),
                "invalid bimanual frame-zero state",
            )
        else:
            _require(
                transforms.shape == (4, 4) and np.asarray(openings).shape == (),
                "invalid monomanual frame-zero state",
            )
            transforms = transforms[None]
            openings = np.asarray([openings])
        _require(
            np.all(np.isfinite(transforms)) and np.all(np.isfinite(openings)),
            "non-finite frame-zero robot state",
        )
        if str(self.deform360_code) not in sys.path:
            sys.path.insert(0, str(self.deform360_code))
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        urdf_module = importlib.import_module("deform360.processing.urdf_render")
        robot_module = importlib.import_module("deform360.robot")
        for module in (urdf_module, robot_module):
            _require(
                Path(str(module.__file__))
                .resolve()
                .is_relative_to(self.deform360_code),
                "official Deform360 import escaped its pinned tree",
            )
        renderer = urdf_module.PyrenderGripperRenderer(
            width, height, urdf_path=urdf_module.default_urdf_path()
        )
        exact_masks: dict[str, np.ndarray] = {}
        try:
            for camera in cameras:
                combined = np.zeros((height, width), dtype=bool)
                for transform, opening in zip(transforms, openings, strict=True):
                    camera_pose = (
                        robot_module.invert_transform(transform)
                        @ camera_to_world_by_camera[camera]
                    )
                    combined |= np.asarray(
                        renderer.render(
                            urdf_module.opening_to_umi_joints(float(opening)),
                            camera_pose,
                            intrinsics_by_camera[camera],
                        ),
                        dtype=bool,
                    )
                exact_masks[camera] = combined
        finally:
            renderer.close()
        kernel = np.ones((11, 11), dtype=np.uint8)
        try:
            cv2 = importlib.import_module("cv2")
        except ImportError as error:  # pragma: no cover - integration dependency
            raise RuntimeError("OpenCV is required for URDF dilation") from error
        dilated_masks = {
            camera: cv2.dilate(exact_masks[camera].astype(np.uint8), kernel).astype(
                bool
            )
            for camera in cameras
        }
        filtered, overlap_audit = filter_robot_dominated_proposals(
            proposals_by_camera, exact_masks, dilated_masks
        )
        audit = {
            "bindings": bindings["official_deform360"],
            "selected_action_frame_index": 0,
            "selected_action": {
                "bimanual": bimanual,
                "gripper_count": len(transforms),
                "T_worlds_sha256": sha256_array(np.asarray(transforms)),
                "openings_sha256": sha256_array(np.asarray(openings)),
            },
            "render": {
                "implementation": (
                    "official deform360.processing.urdf_render."
                    "PyrenderGripperRenderer"
                ),
                "camera_pose": ("invert_transform(T_worlds[0,g]) @ camera_to_world"),
                "multi_gripper_union": "boolean union per camera",
                "image_shape": [height, width],
            },
            "proposal_exclusion": overlap_audit,
        }
        audit["artifact_sha256"] = artifact_sha256(audit)
        return filtered, exact_masks, dilated_masks, audit

    def subtract_robot(
        self,
        selected_masks: Mapping[str, np.ndarray],
        exact_robot_masks: Mapping[str, np.ndarray],
        dilated_robot_masks: Mapping[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        return subtract_robot_from_selected_masks(
            selected_masks, exact_robot_masks, dilated_robot_masks
        )

    def evaluate(
        self,
        rgb_by_camera: Mapping[str, np.ndarray],
        selected_masks: Mapping[str, np.ndarray],
        *,
        object_id: str,
        selected_proposals: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Run the exact frozen exclusive-rank semantic admission gate."""

        bindings = self._verify_bindings()
        cameras = tuple(sorted(selected_masks))
        _require(
            len(cameras) == 8
            and set(cameras) <= set(rgb_by_camera)
            and [record.get("camera") for record in selected_proposals]
            == list(cameras),
            "semantic exact-eight assignment binding changed",
        )
        true_label = semantic_label_for_object_id(object_id)
        object_crops: dict[str, np.ndarray] = {}
        background_crops: dict[str, np.ndarray] = {}
        crop_records = []
        proposal_by_camera = {
            str(record["camera"]): record for record in selected_proposals
        }
        for camera in cameras:
            mask = np.asarray(selected_masks[camera], dtype=bool)
            _require(
                proposal_by_camera[camera].get("mask_sha256") == sha256_array(mask),
                "semantic selected mask binding changed",
            )
            object_crop, object_record = masked_square_crop(rgb_by_camera[camera], mask)
            background_crop, background_record = background_control_crop(
                rgb_by_camera[camera],
                mask,
                side=int(object_record["square_side_pixels"]),
            )
            object_crops[camera] = object_crop
            if background_crop is not None:
                background_crops[camera] = background_crop
            crop_records.append(
                {
                    "camera": camera,
                    "candidate_index": int(
                        proposal_by_camera[camera]["candidate_index"]
                    ),
                    "rgb_sha256": sha256_array(rgb_by_camera[camera]),
                    "selected_mask_sha256": sha256_array(mask),
                    "object": object_record,
                    "background_control": background_record,
                }
            )
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        _require(
            os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
            "CUBLAS_WORKSPACE_CONFIG must be :4096:8 before Torch/CUDA startup",
        )
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            safetensors = importlib.import_module("safetensors")
            tokenizers = importlib.import_module("tokenizers")
            pil = importlib.import_module("PIL")
            AutoModel = transformers.AutoModel
            GemmaTokenizerFast = transformers.GemmaTokenizerFast
            SiglipImageProcessor = transformers.SiglipImageProcessor
            SiglipProcessor = transformers.SiglipProcessor
        except ImportError as error:  # pragma: no cover - integration dependency
            raise RuntimeError("pinned SigLIP2 dependencies are unavailable") from error
        runtime_versions = {
            "PIL": pil.__version__,
            "safetensors": safetensors.__version__,
            "tokenizers": tokenizers.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }
        _require(
            runtime_versions == _RUNTIME_VERSION_BINDINGS,
            "SigLIP2 runtime versions changed",
        )
        transformer_sources = []
        for module_name, expected_sha256 in _TRANSFORMERS_SOURCE_BINDINGS.items():
            module = importlib.import_module(module_name)
            path = Path(str(module.__file__)).resolve()
            observed_sha256 = sha256_file(path)
            _require(
                observed_sha256 == expected_sha256,
                f"Transformers source binding drift: {module_name}",
            )
            transformer_sources.append(
                {
                    "module": module_name,
                    "path": str(path),
                    "sha256": observed_sha256,
                }
            )
        _require(
            hashlib.sha256(
                json.dumps(
                    [
                        [record["module"], record["sha256"]]
                        for record in transformer_sources
                    ],
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            == FRAME_ZERO_SIGLIP2_TRANSFORMERS_SOURCE_AGGREGATE_SHA256,
            "ordered Transformers source aggregate drift",
        )
        image_processor = SiglipImageProcessor.from_pretrained(
            self.model_root, local_files_only=True
        )
        tokenizer = GemmaTokenizerFast.from_pretrained(
            self.model_root, local_files_only=True
        )
        processor = SiglipProcessor(
            image_processor=image_processor, tokenizer=tokenizer
        )
        model = AutoModel.from_pretrained(
            self.model_root, local_files_only=True, dtype=torch.float32
        ).to(self.device)
        model.eval()
        torch.manual_seed(0)
        if self.device.startswith("cuda"):
            torch.cuda.manual_seed_all(0)
        object_cameras = sorted(object_crops)
        background_cameras = sorted(background_crops)
        image_order = [
            ("object", camera, object_crops[camera]) for camera in object_cameras
        ] + [
            ("background_control", camera, background_crops[camera])
            for camera in background_cameras
        ]
        image_module = _pil_image_module()
        inputs = processor(
            text=list(_SIGLIP_PROMPTS),
            images=[image_module.fromarray(item[2]) for item in image_order],
            padding="max_length",
            return_tensors="pt",
        )
        input_hashes = {
            key: sha256_array(value.detach().cpu().numpy())
            for key, value in inputs.items()
        }
        device_inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with _deterministic_siglip_torch_state(torch):
            with torch.inference_mode():
                outputs = model(**device_inputs)
            logits = outputs.logits_per_image.detach().cpu().to(torch.float64).numpy()
        rank = evaluate_rank_gate(
            logits,
            object_cameras=object_cameras,
            background_cameras=background_cameras,
            true_label=true_label,
        )
        audit = {
            "policy_id": _SIGLIP_POLICY_ID,
            "contract_sha256": hashlib.sha256(
                _canonical_bytes(FRAME_ZERO_SEMANTIC_GATE_CONTRACT)
            ).hexdigest(),
            "bindings": {
                **bindings["siglip2"],
                "runtime_versions": runtime_versions,
                "transformers_source_files": transformer_sources,
                "transformers_source_ordered_aggregate_sha256": (
                    FRAME_ZERO_SIGLIP2_TRANSFORMERS_SOURCE_AGGREGATE_SHA256
                ),
            },
            "selected_exact8": crop_records,
            "model_input": {
                "labels_in_exclusive_order": list(_SIGLIP_LABELS),
                "text_prompts_in_order": list(_SIGLIP_PROMPTS),
                "image_order": [
                    {"kind": kind, "camera": camera}
                    for kind, camera, _image in image_order
                ],
                "processor_tensor_sha256": input_hashes,
                "logits_shape": list(logits.shape),
                "logits_float64": logits.tolist(),
                "logits_float64_sha256": sha256_array(logits),
            },
            **rank,
        }
        audit["artifact_sha256"] = artifact_sha256(audit)
        _require(
            audit["decision"]["status"] == "pass",
            "reference-optional SigLIP2 semantic gate failed",
        )
        return audit

    def close(self) -> None:
        """Compatibility hook; the lazy runtime retains no model or renderer."""

        self._bindings = None


__all__ = [
    "FRAME_ZERO_REFERENCE_OPTIONAL_ASSIGNMENT_STRATEGY",
    "FRAME_ZERO_REFERENCE_OPTIONAL_FALLBACK_POLICY_ID",
    "FRAME_ZERO_REFERENCE_OPTIONAL_GEOMETRY_STRATEGY",
    "FRAME_ZERO_SEMANTIC_GATE_CONTRACT",
    "FRAME_ZERO_SEMANTIC_GATE_CONTRACT_SHA256",
    "FRAME_ZERO_OFFICIAL_DEFORM360_BINDINGS",
    "FRAME_ZERO_OFFICIAL_DEFORM360_COMMIT",
    "FRAME_ZERO_SIGLIP2_MODEL_LOCK_SHA256",
    "FRAME_ZERO_SIGLIP2_MODEL_REVISION",
    "FRAME_ZERO_SIGLIP2_MODEL_TREE_SHA256",
    "FRAME_ZERO_SIGLIP2_TRANSFORMERS_SOURCE_AGGREGATE_SHA256",
    "FRAME_ZERO_SIGLIP2_TRANSFORMERS_SOURCE_BINDINGS",
    "FRAME_ZERO_SIGLIP2_WEIGHTS_SHA256",
    "PinnedFrameZeroSemanticGateRuntime",
    "background_control_crop",
    "evaluate_rank_gate",
    "filter_robot_dominated_proposals",
    "masked_square_crop",
    "robot_proposal_overlap_record",
    "semantic_label_for_object_id",
    "subtract_robot_from_selected_masks",
    "unique_rank_record",
    "validate_semantic_gate_audit",
    "verify_model_tree",
]
