"""Permit-gated official Deform360 reconstruction for the held protocol.

No path supplied by the operator is resolved, opened, enumerated, copied, or
hashed when a callback is constructed.  The callback first crosses
``run_outcome_operation`` with the complete cohort capability.  Only inside
that callback may the exact realized-robot-kinematics-selected 81-frame
tracking window be read.
The official Deform360 point-cloud stage consumes the final five frames only as
fixed tracking context and emits exactly 76 target frames.

The production backend wraps the pinned released stages directly.  It does not
call the older independent-source scripts because those scripts validate an
incompatible seal type.  Numerical settings are unchanged: sealed-mask SAM2
propagation, strict-hull reconstruction, depth, CoTracker3, and ``pcd_clean``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Literal, Mapping, Protocol, Sequence

import numpy as np

from . import deform360_held_outcome_scoring as outcome_scoring
from .deform360_frame_zero_assets import (
    validate_frame_zero_bundle_manifest as validate_asset_manifest,
)
from .deform360_held_protocol import (
    DATASET_REVISION,
    FRAME_COUNT,
    METRIC_LOCK,
    PROTOCOL_ID,
    OutcomePhasePermit,
    held_artifact_sha256,
    held_contract_sha256,
    load_held_protocol_lock,
    run_outcome_operation,
    validate_frame_zero_bundle_manifest,
    validate_online_prediction_seal,
    validate_physical_prior_seal,
    validate_prefix_stage_authorization,
)
from .deform360_held_outcome_scoring import (
    OUTCOME_ARTIFACT_KIND,
    TARGET_ARTIFACT_KIND,
    OfficialTarget,
    TargetOperation,
    official_target_array_sha256,
)
from .deform360_robot_kinematics import (
    ROBOT_KINEMATICS_WINDOW_CONTRACT,
    ROBOT_KINEMATICS_WINDOW_CONTRACT_SHA256,
    ROBOT_KINEMATICS_WINDOW_POLICY_ID,
    load_robot_kinematics_archive,
    robot_kinematics_array_records,
    sha256_array as robot_array_sha256,
    slice_robot_kinematics,
    validate_robot_kinematics_selection_audit,
    validate_selected_robot_kinematics_bundle,
)


RECONSTRUCTION_ADAPTER_KIND = "Deform360HeldOfficialReconstructionAdapter"
RECONSTRUCTION_STAGE_KIND = "Deform360HeldOfficialReconstructionStage"
TRACKING_CONTEXT_FRAME_COUNT = 81
TRACKING_TAIL_FRAME_COUNT = 5
STAGED_EPISODE_ID = 0

STAGE_IDS = (
    "held-action-window-staging-v1",
    "sealed-frame-zero-sam2-propagation-v1",
    "official-strict-hull-reconstruct-v1",
    "official-depth-v1",
    "official-cotracker3-v1",
    "official-pcd-clean-v1",
    "held-official-target-v1",
)

SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SAM2_CHECKPOINT_SHA256 = (
    "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"
)
# Hydra resolves this package-relative identifier after ``sam2`` is imported.
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
# The same configuration's location relative to the checkout root is distinct.
SAM2_MODEL_CONFIG_REPOSITORY_PATH = "sam2/configs/sam2.1/sam2.1_hiera_s.yaml"
COTRACKER_CHECKPOINT_SHA256 = (
    "2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834"
)
COTRACKER_COMMIT = "82e02e8029753ad4ef13cf06be7f4fc5facdda4d"
DEFORM360_PROCESSING_REVISION = "0fe36f0b7a7a917ba62b5f8cee707299a9a4a317"

STRICT_HULL_PARAMETERS = {
    "minimum_visual_hull_points": 512,
    "voxel_resolution": 120,
    "cube_half_extent_m": 0.5,
    "first_frame_iterations": 500,
    "warm_start_iterations": 250,
    "warm_start_from_previous_frame": True,
}
DEPTH_PARAMETERS = {
    "expected_depth": True,
    "object_mask_applied": True,
    "gripper_mask_applied_when_available": True,
    "preview_video": False,
}
TRACKING_PARAMETERS = {
    "model_id": "facebook/cotracker3-scaled-offline",
    "pivot_skip": 5,
    "sequence_length": 15,
    "gap": 5,
    "grid_size": 40,
    "resize_factor": 4,
}
PCD_PARAMETERS = {
    "seed_point_count": 10_000,
    "radius_neighbours": 30,
    "radius_m": 0.02,
    "statistical_neighbours": 30,
    "statistical_std_ratio": 3.5,
    "crop_half_extent_m": 0.5,
    "tail_frames_skipped": 5,
    "frame_rate_hz": 30.0,
    "fusion_maximum_speed_m_per_s": 0.05,
    "fusion_minimum_camera_inlier_count": 2,
    "rng_seed": 0,
    "expected_output_frame_count": FRAME_COUNT,
}
VIDEO_STAGING_PARAMETERS = {
    "selector": "between(n,start,start+80)",
    "timestamp_reset": "N/FRAME_RATE/TB",
    "codec": "libx264rgb",
    "crf": 0,
    "preset": "medium",
    "pixel_format": "rgb24",
    "audio": False,
    "decoded_frame_zero_must_equal_sealed_rgb": True,
}
CLAIM_LIMITATION = (
    "official released reconstruction proxy built after complete prediction "
    "sealing; later one-to-one frame-zero transport is not native material "
    "identity or Deform360 Table-4 parity"
)

DEFORM360_SOURCE_SHA256 = {
    "deform360/processing/reconstruct_stage.py": (
        "53a1e8b73e56a1c68a0c4344b279c2817ed4b3ed93e8f5ea792def26d5099c7c"
    ),
    "deform360/processing/depth_stage.py": (
        "34befb732107b805f1e1924699f1e26fc2ca5d3041561b920d8c23d8e85feef0"
    ),
    "deform360/processing/tracking_stage.py": (
        "04533cd9cd900ae2f5bd139568ed1a2442661f14ceda009dd7bb85e4fbd83ec2"
    ),
    "deform360/processing/pcd_stage.py": (
        "87553e1ea3dac5a90e46114c76aaf65901b43a064025626ae6871523065c864d"
    ),
    "deform360/processing/episode.py": (
        "7bd865a461788f7bae1992fd7e21577045ccffd54b58abd18781df4584e13db9"
    ),
    "deform360/robot.py": (
        "376e4dec6f2340a3ee03af1a3bd5462e06e3284cc82f312872a7bedbe863825f"
    ),
}

LOCAL_PROPAGATION_SOURCE_SHA256 = {
    "causal4d_public/deform360_object_sam2.py": (
        "c10391578c73dde47fbce160312559a7e638007e9053ec89373fe575cc64d7e5"
    ),
    "causal4d_public/deform360_sam2.py": (
        "419be2e98ab2b01627ea188c8658b43b39d8b3d4e34e8b33559f32ccdcd04184"
    ),
}

_FRAME_ZERO_ARRAYS = frozenset(
    {
        "frame_indices",
        "camera_names",
        "rgb_frame0",
        "mask_frame0",
        "depth_frame0_m",
        "depth_valid_frame0",
        "intrinsics",
        "camera_to_world",
        "projection_world_to_pixel",
        "object_points_world_m",
        "object_colors_rgb",
        "object_color_support_count",
        "visual_hull_points_world_m",
    }
)
_TARGET_ARCHIVE_ARRAYS = frozenset(
    {
        "object_points",
        "object_colors",
        "object_visibilities",
        "object_motions_valid",
    }
)
_OUTCOME_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "protocol_id",
        "dataset_revision",
        "case_name",
        "object_id",
        "episode_id",
        "role",
        "cohort_barrier_sha256",
        "outcome_reconstruction_contract",
        "outcome_reconstruction_contract_sha256",
        "inputs",
        "target_file",
        "target_array_sha256",
        "backend_audit",
        "information_boundary",
        "claim_limitation",
        "artifact_sha256",
    }
)
_BACKEND_AUDIT_FIELDS = frozenset(
    {
        "stage_ids",
        "contract_sha256",
        "tracking_context_raw_frame_range_half_open",
        "tracking_context_frame_count",
        "prediction_output_frame_range_half_open",
        "tracking_tail_frame_range_half_open",
        "frame_zero_anchor",
        "staging",
        "mask_propagation",
        "official_stages",
        "ffmpeg_runtime",
        "tactile_read",
        "target_dependent_parameter_selection_or_tuning",
        "runtime_seconds",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    descriptor = _canonical_bytes(
        {"dtype": array.dtype.str, "shape": list(array.shape)}
    )
    return hashlib.sha256(
        descriptor + b"\0" + array.view(np.uint8).tobytes()
    ).hexdigest()


def _bound_file(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    _require(
        source.is_file() and not source.is_symlink(),
        f"missing regular non-symlink file: {source}",
    )
    resolved = source.resolve()
    return {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _validate_bound_file(record: Mapping[str, Any], *, label: str) -> Path:
    _require(
        isinstance(record, Mapping) and set(record) == {"path", "sha256", "size_bytes"},
        f"{label} binding fields changed",
    )
    observed = _bound_file(str(record.get("path", "")))
    _require(observed == dict(record), f"{label} binding changed")
    return Path(observed["path"])


def _literal_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _git_stdout(repository: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
    ).stdout


def _git_runtime_binding(repository: str | Path) -> dict[str, str]:
    """Reproduce the prospective lock's exact Git digest semantics."""

    root = Path(repository).resolve()
    _require(root.is_dir() and not root.is_symlink(), "Git runtime root is invalid")
    _require(
        _git_stdout(root, "status", "--porcelain=v1", "--untracked-files=no") == b"",
        "pinned Git runtime has modified tracked files",
    )
    revision = _git_stdout(root, "rev-parse", "HEAD").decode("ascii").strip()
    commit_object = _git_stdout(root, "cat-file", "commit", "HEAD")
    tree_lines = _git_stdout(root, "ls-tree", "-r", "--full-tree", "HEAD").splitlines()
    tree_manifest = b"".join(line + b"\n" for line in sorted(tree_lines))
    return {
        "revision": revision,
        "revision_literal_sha256": _literal_sha256(revision),
        "commit_object_sha256": hashlib.sha256(commit_object).hexdigest(),
        "git_tree_manifest_sha256": hashlib.sha256(tree_manifest).hexdigest(),
    }


def _validate_git_runtime_binding(
    repository: str | Path,
    immutable_bindings: Mapping[str, str],
    *,
    prefix: str,
    expected_revision: str,
) -> dict[str, str]:
    observed = _git_runtime_binding(repository)
    expected = {
        "revision_literal_sha256": immutable_bindings[f"{prefix}_revision_literal"],
        "commit_object_sha256": immutable_bindings[f"{prefix}_commit_object"],
        "git_tree_manifest_sha256": immutable_bindings[f"{prefix}_git_tree_manifest"],
    }
    _require(observed["revision"] == expected_revision, f"{prefix} revision changed")
    _require(
        {key: observed[key] for key in expected} == expected,
        f"{prefix} Git binding differs from the immutable lock",
    )
    return observed


def _tree_binding(root: str | Path) -> dict[str, Any]:
    directory = Path(root).resolve()
    _require(directory.is_dir() and not directory.is_symlink(), "tree root is invalid")
    entries = sorted(directory.rglob("*"))
    _require(
        all(not path.is_symlink() for path in entries),
        f"bound output tree contains a symlink: {directory}",
    )
    _require(
        all(path.is_dir() or path.is_file() for path in entries),
        f"bound output tree contains a non-regular entry: {directory}",
    )
    records = {
        path.relative_to(directory).as_posix(): _bound_file(path)
        for path in entries
        if path.is_file()
    }
    _require(bool(records), f"bound output tree is empty: {directory}")
    return {
        "root": str(directory),
        "file_count": len(records),
        "files": records,
        "tree_sha256": _canonical_sha256(
            {
                name: {
                    "sha256": record["sha256"],
                    "size_bytes": record["size_bytes"],
                }
                for name, record in records.items()
            }
        ),
    }


def _validate_tree_binding(record: Mapping[str, Any], *, label: str) -> Path:
    _require(
        isinstance(record, Mapping)
        and set(record) == {"root", "file_count", "files", "tree_sha256"},
        f"{label} tree binding fields changed",
    )
    observed = _tree_binding(str(record.get("root", "")))
    _require(observed == dict(record), f"{label} tree binding changed")
    return Path(observed["root"])


def _validate_nested_audit_bindings(
    value: Any, *, label: str = "backend audit"
) -> None:
    """Revalidate every immutable file/tree binding nested in an audit."""

    if isinstance(value, Mapping):
        fields = set(value)
        if fields == {"path", "sha256", "size_bytes"}:
            _validate_bound_file(value, label=label)
            return
        if fields == {"root", "file_count", "files", "tree_sha256"}:
            _validate_tree_binding(value, label=label)
            return
        for name, child in value.items():
            _validate_nested_audit_bindings(child, label=f"{label}.{name}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_nested_audit_bindings(child, label=f"{label}[{index}]")


def _write_new_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination.resolve()


def _write_new_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination.resolve()


def _copy_contract() -> dict[str, Any]:
    contract = getattr(outcome_scoring, "OUTCOME_RECONSTRUCTION_CONTRACT", None)
    _require(
        isinstance(contract, Mapping),
        "OUTCOME_RECONSTRUCTION_CONTRACT is not exported",
    )
    return json.loads(json.dumps(contract, sort_keys=True, allow_nan=False))


def _contract_sha256(contract: Mapping[str, Any]) -> str:
    return held_contract_sha256(contract)


def _validate_contract_semantics(contract: Mapping[str, Any]) -> None:
    """Require the exported contract to describe this exact backend."""

    expected = {
        "contract_id": "deform360-held-official-reconstruction-v1",
        "dataset_revision": DATASET_REVISION,
        "ordered_stages": list(STAGE_IDS),
        "temporal_contract": {
            "staged_raw_interval": "[selected_start, selected_start + 81)",
            "staged_rgb_and_mask_context_frame_count": 81,
            "logical_target_interval": "[0, 76)",
            "logical_target_frame_count": 76,
            "prediction_raw_interval": "[selected_start, selected_start + 76)",
            "final_context_only_frame_count": 5,
            "final_context_frames_scored": False,
            "reason": "official pcd_stage TAIL_FRAMES_SKIPPED=5",
        },
        "sealed_frame_zero_anchor": {
            "camera_order": "exact frame_zero_bundle camera_names order",
            "arrays": [
                "intrinsics",
                "camera_to_world",
                "rgb_frame0",
                "mask_frame0",
            ],
            "automatic_initial_mask_selection": False,
            "decoded_staged_rgb_frame0_bit_exact": True,
            "propagated_mask_frame0_bit_exact": True,
            "mask_seed_source": "sealed mask_frame0 only",
        },
        "video_staging": {
            "tool": "ffmpeg",
            "selection": "exact selected raw 81-frame interval",
            "video_codec": "libx264rgb",
            "crf": 0,
            "pixel_format": "rgb24",
            "audio": False,
        },
        "sam2_video_propagation": {
            "commit": SAM2_COMMIT,
            "checkpoint_sha256": SAM2_CHECKPOINT_SHA256,
            "model_config": SAM2_MODEL_CONFIG,
            "sealed_frame_zero_seed_only": True,
        },
        "strict_visual_hull": {
            "minimum_visual_hull_points": 512,
            "voxel_resolution": 120,
            "cube_half_extent_m": 0.5,
            "iteration_limit": 500,
            "warm_iteration_count": 250,
            "warm_start_previous_frame": True,
        },
        "depth": {
            "mode": "expected_depth",
            "object_mask": True,
            "gripper_mask_when_available": True,
            "preview": False,
        },
        "cotracker": {
            "revision": COTRACKER_COMMIT,
            "model_id": "facebook/cotracker3-scaled-offline",
            "checkpoint_sha256": COTRACKER_CHECKPOINT_SHA256,
            "pivot_skip": 5,
            "sequence_length": 15,
            "gap": 5,
            "grid_size": 40,
            "resize_factor": 4,
        },
        "point_cloud": {
            "seed_point_count": 10_000,
            "radius_outlier_neighbour_count": 30,
            "radius_outlier_radius_m": 0.02,
            "statistical_outlier_neighbour_count": 30,
            "statistical_outlier_std_ratio": 3.5,
            "crop_half_extent_m": 0.5,
            "tail_frames_skipped": 5,
            "frame_rate_hz": 30,
            "fusion_max_speed_m_per_s": 0.05,
            "minimum_camera_inliers": 2,
            "rng_seed": 0,
            "exact_output_file_count": 76,
        },
        "tactile": {"copied": False, "read": False},
        "official_processing": {
            "revision": DEFORM360_PROCESSING_REVISION,
            "pipeline_config_file_sha256": (
                "8692dc89651a91dcb1732a7b7185983ffd0aa2312aeb2bd202bafaf85309d7e8"
            ),
            "pipeline_config_semantic_sha256": (
                "e32c20e98442e7112a79c1d54de3f58a4608d9c382739f05a10085df53d42039"
            ),
            "stage_runner_sha256": (
                "2f379581786b6b6072eaf88cf4430b514b1bf47b4ab3b8dacd46201edc6a7739"
            ),
            "strict_reconstruction_sha256": (
                "14ab64761037074314383d67af5ce56d744ae0fbc8c64cccddce3ea7a57fd450"
            ),
            "official_outcome_builder_sha256": (
                "3e188e4b8d507543a4472c62453ce2c94fa1696370fc4ac1de31bc196f1da827"
            ),
            "object_sam2_source_sha256": (
                "c10391578c73dde47fbce160312559a7e638007e9053ec89373fe575cc64d7e5"
            ),
            "sam2_source_sha256": (
                "419be2e98ab2b01627ea188c8658b43b39d8b3d4e34e8b33559f32ccdcd04184"
            ),
            "stage_source_sha256": {
                (
                    "processing/robot.py"
                    if key == "deform360/robot.py"
                    else key.removeprefix("deform360/")
                ): value
                for key, value in DEFORM360_SOURCE_SHA256.items()
            },
        },
        "target_arrays": [
            "object_points",
            "object_visibilities",
            "object_motions_valid",
        ],
        "identity_transport": {
            "method": "sparse minimum-cost one-to-one frame-zero assignment",
            "maximum_assignment_distance_m": 0.015,
            "required_coverage_fraction": 1.0,
            "required_collision_count": 0,
            "assimilation_centres_excluded_from_scores": True,
        },
        "metric_lock": METRIC_LOCK,
    }
    _require(dict(contract) == expected, "exported reconstruction contract changed")


def _validate_locked_adapter_contract(
    lock: Mapping[str, Any], contract: Mapping[str, Any]
) -> Mapping[str, str]:
    """Bind both the reconstruction semantics and this adapter to the lock."""

    bindings = lock.get("immutable_bindings", {})
    _require(isinstance(bindings, Mapping), "held lock has no immutable bindings")
    _require(
        bindings.get("outcome_reconstruction_contract") == _contract_sha256(contract),
        "held lock binds another outcome reconstruction contract",
    )
    _require(
        bindings.get("held_outcome_reconstruction_adapter_source")
        == _sha256_file(Path(__file__)),
        "held lock binds another outcome reconstruction adapter",
    )
    return bindings


@dataclass(frozen=True)
class ReconstructionRequest:
    """Materialized only inside a successfully revalidated outcome callback."""

    case_name: str
    object_id: str
    episode_id: int
    role: str
    cohort_barrier_sha256: str
    aligned_episode_dir: Path | None
    output_dir: Path
    source_frame_start: int
    source_frame_stop: int
    camera_names: tuple[str, ...]
    frame_zero_arrays: Mapping[str, np.ndarray]
    frame_zero_manifest: Mapping[str, Any]
    frame_zero_manifest_path: Path
    online_seal_path: Path
    contract: Mapping[str, Any]
    immutable_bindings: Mapping[str, str]


@dataclass(frozen=True)
class ReconstructionBackendResult:
    """Raw official target material and a complete backend audit."""

    object_points: np.ndarray
    object_colors: np.ndarray
    object_visibilities: np.ndarray
    object_motions_valid: np.ndarray
    audit: Mapping[str, Any]


class ReconstructionBackend(Protocol):
    def build(self, request: ReconstructionRequest) -> ReconstructionBackendResult:
        """Read the permitted future and return the official reconstruction."""


@dataclass(frozen=True)
class PinnedOfficialPipelineBackend:
    """Path-only configuration; construction performs no filesystem access."""

    deform360_repo: str
    sam2_repository: str
    sam2_checkpoint: str
    cotracker_repo: str
    cotracker_checkpoint: str
    device: str = "cuda:0"
    ffmpeg: str = "ffmpeg"

    def build(self, request: ReconstructionRequest) -> ReconstructionBackendResult:
        return _run_pinned_official_pipeline(request, self)


def _load_json(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def _load_sealed_request(
    permit: OutcomePhasePermit,
    case_name: str,
    aligned_episode_dir: str | None,
    output_dir: str,
    *,
    operation: Literal["create", "read"],
) -> ReconstructionRequest:
    contract = _copy_contract()
    _validate_contract_semantics(contract)
    lock = load_held_protocol_lock(permit.lock_path)
    bindings = _validate_locked_adapter_contract(lock, contract)
    seal_paths = dict(permit.seal_paths)
    _require(case_name in seal_paths, "reconstruction case is outside the permit")
    seal_path = Path(seal_paths[case_name]).resolve()
    seal = validate_online_prediction_seal(
        seal_path,
        permit.lock_path,
        expected_case_name=case_name,
        expected_role=permit.role,
    )
    authorization_path = Path(seal["prefix_authorization"]["path"])
    authorization = validate_prefix_stage_authorization(
        authorization_path, permit.lock_path
    )
    physical_path = Path(authorization["physical_prior_seal"]["path"])
    physical = validate_physical_prior_seal(
        physical_path,
        permit.lock_path,
        expected_case_name=case_name,
        expected_role=permit.role,
    )
    manifest_path = Path(physical["frame_zero_manifest"]["path"])
    manifest = validate_frame_zero_bundle_manifest(
        manifest_path,
        permit.lock_path,
        expected_case_name=case_name,
        expected_role=permit.role,
    )
    asset_manifest = _load_json(manifest_path)
    validate_asset_manifest(asset_manifest)
    _require(
        asset_manifest.get("artifact_sha256") == manifest.get("artifact_sha256"),
        "frame-zero validators disagree on the sealed manifest",
    )
    bundle_path = _validate_bound_file(
        asset_manifest["bundle"], label="frame-zero bundle"
    )
    with np.load(bundle_path, allow_pickle=False) as stored:
        _require(
            set(stored.files) == _FRAME_ZERO_ARRAYS,
            "frame-zero bundle array set changed",
        )
        arrays = {name: np.asarray(stored[name]).copy() for name in stored.files}
    expected_arrays = asset_manifest.get("arrays", {})
    _require(
        isinstance(expected_arrays, Mapping)
        and set(expected_arrays) == _FRAME_ZERO_ARRAYS,
        "frame-zero manifest array bindings changed",
    )
    for name, value in arrays.items():
        record = expected_arrays[name]
        _require(
            record
            == {
                "shape": list(value.shape),
                "dtype": value.dtype.str,
                "sha256": _sha256_array(value),
            },
            f"sealed frame-zero array changed: {name}",
        )
    cameras = tuple(str(value) for value in arrays["camera_names"].tolist())
    _require(len(cameras) >= 2 and len(set(cameras)) == len(cameras), "invalid cameras")
    _require(
        arrays["rgb_frame0"].shape[0] == len(cameras)
        and arrays["mask_frame0"].shape == arrays["rgb_frame0"].shape[:3],
        "frame-zero RGB/mask camera axes changed",
    )
    policy = asset_manifest.get("camera_policy", {})
    _require(
        policy.get("selected_cameras") == list(cameras),
        "frame-zero camera order changed",
    )
    alignment = asset_manifest.get("action_alignment", {})
    selected_range = alignment.get("selected_raw_frame_range_half_open")
    prediction_range = alignment.get("prediction_raw_frame_range_half_open")
    _require(
        isinstance(selected_range, list)
        and len(selected_range) == 2
        and int(selected_range[1]) - int(selected_range[0])
        == TRACKING_CONTEXT_FRAME_COUNT,
        "sealed tracking context is not 81 frames",
    )
    start = int(selected_range[0])
    stop = int(selected_range[1])
    _require(
        prediction_range == [start, start + FRAME_COUNT]
        and stop == start + FRAME_COUNT + TRACKING_TAIL_FRAME_COUNT,
        "sealed prediction/output window changed",
    )
    # Operator-supplied paths are intentionally touched only now, after the
    # enclosing run_outcome_operation has revalidated the complete cohort.
    output = Path(output_dir).resolve()
    aligned: Path | None = None
    if operation == "create":
        _require(
            aligned_episode_dir is not None,
            "aligned outcome episode is required for creation",
        )
        aligned = Path(aligned_episode_dir).resolve()
        _require(
            aligned.is_dir() and not aligned.is_symlink(),
            "aligned outcome episode is missing or a symlink",
        )
        _require(
            aligned.name == f"episode_{int(seal['episode_id']):04d}"
            and aligned.parent.name == seal["object_id"],
            "aligned outcome episode identity changed",
        )
        _require(
            not os.path.lexists(output),
            "outcome reconstruction output already exists",
        )
    else:
        _require(operation == "read", "unsupported reconstruction operation")
        _require(
            output.is_dir() and not output.is_symlink(),
            "outcome reconstruction output is absent or not a regular directory",
        )
    return ReconstructionRequest(
        case_name=case_name,
        object_id=str(seal["object_id"]),
        episode_id=int(seal["episode_id"]),
        role=str(seal["role"]),
        cohort_barrier_sha256=permit.cohort_barrier_sha256,
        aligned_episode_dir=aligned,
        output_dir=output,
        source_frame_start=start,
        source_frame_stop=stop,
        camera_names=cameras,
        frame_zero_arrays=arrays,
        frame_zero_manifest=asset_manifest,
        frame_zero_manifest_path=manifest_path.resolve(),
        online_seal_path=seal_path,
        contract=contract,
        immutable_bindings=dict(bindings),
    )


def _decode_video_frame(path: Path, frame_index: int) -> np.ndarray:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - integration runtime
        raise RuntimeError("OpenCV is required for held outcome staging") from error
    capture = cv2.VideoCapture(str(path))
    try:
        if frame_index:
            _require(
                bool(capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)),
                "cannot seek outcome video",
            )
        ok, bgr = capture.read()
    finally:
        capture.release()
    _require(bool(ok) and bgr is not None, f"cannot decode video frame: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _video_frame_count(path: Path) -> int:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - integration runtime
        raise RuntimeError("OpenCV is required for held outcome staging") from error
    capture = cv2.VideoCapture(str(path))
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    return count


def _trim_lossless_video(
    executable: str,
    source: Path,
    destination: Path,
    *,
    start: int,
    frame_count: int,
) -> None:
    stop = start + frame_count - 1
    subprocess.run(
        [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            f"select=between(n\\,{start}\\,{stop}),setpts=N/FRAME_RATE/TB",
            "-frames:v",
            str(frame_count),
            "-an",
            "-c:v",
            "libx264rgb",
            "-crf",
            "0",
            "-preset",
            "medium",
            "-pix_fmt",
            "rgb24",
            str(destination),
        ],
        check=True,
    )


def _write_mask_h5(path: Path, masks: Sequence[np.ndarray]) -> None:
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - integration runtime
        raise RuntimeError("h5py is required for official mask staging") from error
    values = np.asarray(masks, dtype=np.uint8)
    _require(
        values.ndim == 3 and len(values) == TRACKING_CONTEXT_FRAME_COUNT,
        "propagated mask stack changed",
    )
    with h5py.File(path, "w") as stream:
        stream.create_dataset(
            "data", data=values, dtype=np.uint8, compression="gzip", compression_opts=4
        )


def _verify_ffmpeg_runtime(
    request: ReconstructionRequest, executable: str
) -> tuple[Path, dict[str, Any]]:
    located = shutil.which(executable)
    _require(located is not None, "pinned ffmpeg executable is unavailable")
    path = Path(located).resolve()
    _require(
        _sha256_file(path) == request.immutable_bindings["ffmpeg_executable"],
        "ffmpeg executable differs from the immutable lock",
    )
    version = subprocess.run([str(path), "-version"], check=True, capture_output=True)
    _require(
        hashlib.sha256(version.stdout).hexdigest()
        == request.immutable_bindings["ffmpeg_version_literal"],
        "ffmpeg version differs from the immutable lock",
    )
    return path, {
        "executable": _bound_file(path),
        "version_stdout_sha256": hashlib.sha256(version.stdout).hexdigest(),
        "version_stderr_sha256": hashlib.sha256(version.stderr).hexdigest(),
    }


def _validate_sam2_model_config(
    repository: str | Path, immutable_bindings: Mapping[str, str]
) -> Path:
    """Resolve Hydra's model ID to its distinct checkout-relative file."""

    model_config = Path(repository).resolve() / SAM2_MODEL_CONFIG_REPOSITORY_PATH
    _require(
        model_config.is_file()
        and not model_config.is_symlink()
        and _sha256_file(model_config) == immutable_bindings["sam2_model_config"],
        "SAM2 model configuration differs from the immutable lock",
    )
    return model_config


def _stage_action_window(
    request: ReconstructionRequest,
    ffmpeg_executable: Path,
) -> tuple[Path, dict[str, Any]]:
    _require(
        request.aligned_episode_dir is not None,
        "creation request has no aligned outcome episode",
    )
    staged_root = request.output_dir / "staged-aligned"
    episode = staged_root / f"episode_{STAGED_EPISODE_ID:04d}"
    episode.mkdir(parents=True)
    arrays = request.frame_zero_arrays
    intrinsics = {
        camera: np.asarray(arrays["intrinsics"])[index]
        for index, camera in enumerate(request.camera_names)
    }
    extrinsics = {
        camera: np.asarray(arrays["camera_to_world"])[index]
        for index, camera in enumerate(request.camera_names)
    }
    np.save(episode / "undistorted_intrinsics.npy", intrinsics)
    np.save(episode / "extrinsics.npy", extrinsics)

    action_inputs = request.frame_zero_manifest["action_inputs"]
    raw_robot_path = _validate_bound_file(
        action_inputs["robot_trajectory"], label="bound raw robot action"
    )
    robot_meta_path = _validate_bound_file(
        action_inputs["robot_metadata"], label="bound robot metadata"
    )
    selected_action_path = _validate_bound_file(
        request.frame_zero_manifest["action_alignment"]["selected_action_bundle"],
        label="selected 76-frame realized robot kinematics",
    )
    source_state = load_robot_kinematics_archive(raw_robot_path)
    alignment = request.frame_zero_manifest["action_alignment"]
    config = request.frame_zero_manifest.get("config")
    _require(isinstance(config, Mapping), "frame-zero configuration is missing")
    candidate_first = config.get("action_candidate_first_frame")
    candidate_stride = config.get("action_candidate_stride_frames")
    _require(
        type(candidate_first) is int
        and candidate_first >= 0
        and type(candidate_stride) is int
        and candidate_stride >= 1
        and config.get("action_window_length_frames")
        == TRACKING_CONTEXT_FRAME_COUNT
        and config.get("prediction_frame_count") == FRAME_COUNT,
        "frame-zero robot-window configuration changed",
    )
    selection = alignment.get("selection_audit")
    _require(isinstance(selection, Mapping), "robot selection audit is missing")
    selection_audit = validate_robot_kinematics_selection_audit(
        selection,
        source_state,
        window_length_frames=TRACKING_CONTEXT_FRAME_COUNT,
        prediction_frame_count=FRAME_COUNT,
        candidate_first_frame=candidate_first,
        candidate_stride_frames=candidate_stride,
    )
    _require(
        alignment.get("policy_id") == ROBOT_KINEMATICS_WINDOW_POLICY_ID
        and alignment.get("trajectory_semantics")
        == ROBOT_KINEMATICS_WINDOW_CONTRACT["trajectory_semantics"]
        and alignment.get("selected_raw_frame_range_half_open")
        == selection_audit["selected_raw_frame_range_half_open"]
        == [request.source_frame_start, request.source_frame_stop]
        and alignment.get("prediction_raw_frame_range_half_open")
        == selection_audit["prediction_raw_frame_range_half_open"]
        == [request.source_frame_start, request.source_frame_start + FRAME_COUNT]
        and alignment.get("source_robot_frame_count") == source_state.frame_count
        and alignment.get("prediction_frame_count") == FRAME_COUNT
        and alignment.get("tracking_tail_frame_count") == TRACKING_TAIL_FRAME_COUNT,
        "outcome robot selection alignment changed",
    )
    _require(
        alignment.get("selected_action_bundle")
        == alignment.get("selected_robot_kinematics_bundle")
        and alignment.get("selected_action_bundle_is_compatibility_alias") is True,
        "outcome selected-action compatibility alias changed",
    )
    selected_state = load_robot_kinematics_archive(
        selected_action_path, expected_frame_count=FRAME_COUNT
    )
    _require(
        alignment.get("selected_action_arrays")
        == robot_kinematics_array_records(selected_state),
        "sealed selected robot kinematics array bindings changed",
    )
    exact_slice_audit = validate_selected_robot_kinematics_bundle(
        selected_state,
        source_state=source_state,
        prediction_start_frame=request.source_frame_start,
        prediction_frame_count=FRAME_COUNT,
    )
    _require(
        alignment.get("selected_bundle_exact_slice_audit") == exact_slice_audit,
        "sealed selected robot kinematics exact-slice proof changed",
    )

    staged_state = slice_robot_kinematics(
        source_state,
        start_frame=request.source_frame_start,
        frame_count=TRACKING_CONTEXT_FRAME_COUNT,
    )
    staged_robot = staged_state.archive_arrays()
    source_arrays = source_state.archive_arrays()
    selected_arrays = selected_state.archive_arrays()
    temporal_fields = ("actions", "T_worlds", "openings")
    scalar_fields = ("format_version", "bimanual")
    for name in temporal_fields:
        _require(
            np.array_equal(
                staged_robot[name],
                source_arrays[name][
                    request.source_frame_start : request.source_frame_stop
                ],
            ),
            f"staged robot temporal slice changed: {name}",
        )
        _require(
            np.array_equal(staged_robot[name][:FRAME_COUNT], selected_arrays[name]),
            f"staged robot prediction prefix differs from sealed bundle: {name}",
        )
    for name in scalar_fields:
        _require(
            np.array_equal(staged_robot[name], source_arrays[name])
            and np.array_equal(staged_robot[name], selected_arrays[name]),
            f"staged robot scalar field changed: {name}",
        )
    robot_dir = episode / "robot"
    robot_dir.mkdir()
    np.savez_compressed(robot_dir / "robot.npz", **staged_robot)
    shutil.copy2(robot_meta_path, robot_dir / "robot.meta.json")

    source_inputs: dict[str, Any] = {
        "robot_kinematics": _bound_file(raw_robot_path),
        "robot_trajectory": _bound_file(raw_robot_path),
        "robot_metadata": _bound_file(robot_meta_path),
        "selected_prediction_robot_kinematics": _bound_file(selected_action_path),
        "selected_prediction_action": _bound_file(selected_action_path),
        "cameras": {},
    }
    staged_outputs: dict[str, Any] = {
        "intrinsics": _bound_file(episode / "undistorted_intrinsics.npy"),
        "extrinsics": _bound_file(episode / "extrinsics.npy"),
        "robot": _bound_file(robot_dir / "robot.npz"),
        "robot_metadata": _bound_file(robot_dir / "robot.meta.json"),
        "cameras": {},
    }
    robot_kinematics_staging = {
        "policy_id": ROBOT_KINEMATICS_WINDOW_POLICY_ID,
        "contract_sha256": ROBOT_KINEMATICS_WINDOW_CONTRACT_SHA256,
        "trajectory_semantics": ROBOT_KINEMATICS_WINDOW_CONTRACT[
            "trajectory_semantics"
        ],
        "selection_audit": selection_audit,
        "selected_bundle_exact_slice_audit": exact_slice_audit,
        "temporal_fields_sliced_exactly_81": list(temporal_fields),
        "scalar_fields_copied_unchanged": list(scalar_fields),
        "all_five_fields_first_76_equal_selected_bundle": True,
        "source_array_sha256": {
            name: robot_array_sha256(value)
            for name, value in sorted(source_arrays.items())
        },
        "staged_array_sha256": {
            name: robot_array_sha256(value)
            for name, value in sorted(staged_robot.items())
        },
        "selected_array_sha256": {
            name: robot_array_sha256(value)
            for name, value in sorted(selected_arrays.items())
        },
        "commanded_control_or_delta_action_used": False,
    }
    rgb_zero = np.asarray(arrays["rgb_frame0"])
    for index, camera in enumerate(request.camera_names):
        source_camera = request.aligned_episode_dir / camera
        source_video = source_camera / "undistorted.mp4"
        source_timestamps = source_camera / "aligned_timestamps.txt"
        _require(
            source_video.is_file()
            and not source_video.is_symlink()
            and source_timestamps.is_file()
            and not source_timestamps.is_symlink(),
            f"outcome camera inputs are missing: {camera}",
        )
        _require(
            np.array_equal(
                _decode_video_frame(source_video, request.source_frame_start),
                rgb_zero[index],
            ),
            f"raw outcome frame zero differs from the sealed RGB: {camera}",
        )
        output_camera = episode / camera
        output_camera.mkdir()
        output_video = output_camera / "undistorted.mp4"
        _trim_lossless_video(
            str(ffmpeg_executable),
            source_video,
            output_video,
            start=request.source_frame_start,
            frame_count=TRACKING_CONTEXT_FRAME_COUNT,
        )
        _require(
            _video_frame_count(output_video) == TRACKING_CONTEXT_FRAME_COUNT,
            f"staged video frame count changed: {camera}",
        )
        _require(
            np.array_equal(_decode_video_frame(output_video, 0), rgb_zero[index]),
            f"lossless staged frame zero differs from sealed RGB: {camera}",
        )
        timestamps = source_timestamps.read_text(encoding="utf-8").splitlines()
        selected_timestamps = timestamps[
            request.source_frame_start : request.source_frame_stop
        ]
        _require(
            len(selected_timestamps) == TRACKING_CONTEXT_FRAME_COUNT,
            f"timestamp window is incomplete: {camera}",
        )
        output_timestamps = output_camera / "aligned_timestamps.txt"
        output_timestamps.write_text(
            "\n".join(selected_timestamps) + "\n", encoding="utf-8"
        )
        metadata = source_camera / "metadata.json"
        metadata_record = None
        output_metadata_record = None
        if metadata.is_file() and not metadata.is_symlink():
            shutil.copy2(metadata, output_camera / "metadata.json")
            metadata_record = _bound_file(metadata)
            output_metadata_record = _bound_file(output_camera / "metadata.json")
        source_inputs["cameras"][camera] = {
            "video": _bound_file(source_video),
            "timestamps": _bound_file(source_timestamps),
            "metadata": metadata_record,
        }
        staged_outputs["cameras"][camera] = {
            "video": _bound_file(output_video),
            "timestamps": _bound_file(output_timestamps),
            "metadata": output_metadata_record,
            "frame_zero_rgb_sha256": _sha256_array(rgb_zero[index]),
        }
    return staged_root, {
        "source_inputs": source_inputs,
        "staged_outputs": staged_outputs,
        "robot_kinematics": robot_kinematics_staging,
        "source_frame_range_half_open": [
            request.source_frame_start,
            request.source_frame_stop,
        ],
        "tracking_context_frame_count": TRACKING_CONTEXT_FRAME_COUNT,
        "prediction_frame_range_half_open": [0, FRAME_COUNT],
        "tracking_tail_frame_range_half_open": [
            FRAME_COUNT,
            TRACKING_CONTEXT_FRAME_COUNT,
        ],
        "video_staging": dict(VIDEO_STAGING_PARAMETERS),
        "tactile_paths_read_or_copied": [],
    }


def _propagate_sealed_masks(
    request: ReconstructionRequest,
    backend: PinnedOfficialPipelineBackend,
    staged_root: Path,
) -> dict[str, Any]:
    from causal4d_public.deform360_object_sam2 import (
        DeformableObjectSam2VideoPredictor,
    )

    source_root = Path(__file__).resolve().parents[1]
    for relative, expected in LOCAL_PROPAGATION_SOURCE_SHA256.items():
        path = source_root / relative
        _require(
            _sha256_file(path) == expected, f"propagation source changed: {relative}"
        )
    sam2_repo = Path(backend.sam2_repository).resolve()
    checkpoint = Path(backend.sam2_checkpoint).resolve()
    _require(
        _sha256_file(checkpoint)
        == SAM2_CHECKPOINT_SHA256
        == request.immutable_bindings["sam2_checkpoint"],
        "SAM2 checkpoint changed",
    )
    git_binding = _validate_git_runtime_binding(
        sam2_repo,
        request.immutable_bindings,
        prefix="sam2",
        expected_revision=SAM2_COMMIT,
    )
    model_config = _validate_sam2_model_config(sam2_repo, request.immutable_bindings)
    episode = staged_root / f"episode_{STAGED_EPISODE_ID:04d}"
    sealed_masks = np.asarray(request.frame_zero_arrays["mask_frame0"], dtype=bool)
    records: dict[str, Any] = {}
    predictor = DeformableObjectSam2VideoPredictor(
        sam2_repo, checkpoint, device=backend.device
    )
    try:
        for index, camera in enumerate(request.camera_names):
            video = episode / camera / "undistorted.mp4"
            initialization = {
                "case_name": request.case_name,
                "cohort_barrier_sha256": request.cohort_barrier_sha256,
                "frame_zero_manifest_sha256": _sha256_file(
                    request.frame_zero_manifest_path
                ),
                "sealed_mask_sha256": _sha256_array(sealed_masks[index]),
                "automatic_initial_mask_selection": False,
            }
            propagated = list(
                predictor.segment_from_initial_mask(
                    video, sealed_masks[index], initialization=initialization
                )
            )
            _require(
                [frame for frame, _ in propagated]
                == list(range(TRACKING_CONTEXT_FRAME_COUNT)),
                f"SAM2 propagation is incomplete: {camera}",
            )
            masks = [np.asarray(mask, dtype=bool) for _, mask in propagated]
            _require(
                np.array_equal(masks[0], sealed_masks[index]),
                f"propagated frame-zero mask differs from seal: {camera}",
            )
            destination = episode / camera / "mask_refined.h5"
            _write_mask_h5(destination, masks)
            records[camera] = {
                "mask_archive": _bound_file(destination),
                "frame_count": len(masks),
                "sealed_frame_zero_mask_sha256": _sha256_array(sealed_masks[index]),
                "propagated_frame_zero_mask_sha256": _sha256_array(masks[0]),
                "initialization": initialization,
            }
    finally:
        predictor.close()
    return {
        "stage_id": STAGE_IDS[1],
        "sam2_git": git_binding,
        "sam2_checkpoint": _bound_file(checkpoint),
        "sam2_model_config": _bound_file(model_config),
        "camera_masks": records,
        "sealed_mask_is_only_initialization": True,
        "target_dependent_mask_selection_or_tuning": False,
    }


def _verify_deform360_runtime(
    repository: Path, immutable_bindings: Mapping[str, str]
) -> dict[str, Any]:
    git_binding = _validate_git_runtime_binding(
        repository,
        immutable_bindings,
        prefix="deform360_code",
        expected_revision=DEFORM360_PROCESSING_REVISION,
    )
    records: dict[str, Any] = {}
    for relative, expected in DEFORM360_SOURCE_SHA256.items():
        path = repository / relative
        _require(
            _sha256_file(path) == expected, f"Deform360 source changed: {relative}"
        )
        records[relative] = _bound_file(path)
    return {"git": git_binding, "source_files": records}


def _verify_cotracker_runtime(
    request: ReconstructionRequest,
    backend: PinnedOfficialPipelineBackend,
) -> tuple[Path, Any, dict[str, Any]]:
    """Verify the exact CoTracker checkout, checkpoint, and imported module."""

    repository = Path(backend.cotracker_repo).resolve()
    checkpoint = Path(backend.cotracker_checkpoint).resolve()
    git_binding = _validate_git_runtime_binding(
        repository,
        request.immutable_bindings,
        prefix="cotracker",
        expected_revision=COTRACKER_COMMIT,
    )
    _require(
        checkpoint.is_file() and not checkpoint.is_symlink(),
        "CoTracker checkpoint is missing or a symlink",
    )
    _require(
        _sha256_file(checkpoint)
        == COTRACKER_CHECKPOINT_SHA256
        == request.immutable_bindings["cotracker_checkpoint"],
        "CoTracker checkpoint changed",
    )
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    importlib.invalidate_caches()
    predictor_module = importlib.import_module("cotracker.predictor")
    expected_module = (repository / "cotracker" / "predictor.py").resolve()
    imported_file = Path(predictor_module.__file__).resolve()
    _require(
        expected_module.is_file()
        and not expected_module.is_symlink()
        and imported_file == expected_module,
        "imported CoTracker runtime comes from another repository",
    )
    return (
        checkpoint,
        predictor_module,
        {
            "git": git_binding,
            "checkpoint": _bound_file(checkpoint),
            "predictor_module": _bound_file(expected_module),
        },
    )


def _run_deform360_stages(
    request: ReconstructionRequest,
    backend: PinnedOfficialPipelineBackend,
    staged_root: Path,
) -> tuple[ReconstructionBackendResult, dict[str, Any]]:
    repository = Path(backend.deform360_repo).resolve()
    runtime = _verify_deform360_runtime(repository, request.immutable_bindings)
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    from deform360.processing import (
        depth_stage,
        pcd_stage,
        reconstruct_stage,
        tracking_stage,
    )

    for module, relative in (
        (reconstruct_stage, "deform360/processing/reconstruct_stage.py"),
        (depth_stage, "deform360/processing/depth_stage.py"),
        (tracking_stage, "deform360/processing/tracking_stage.py"),
        (pcd_stage, "deform360/processing/pcd_stage.py"),
    ):
        _require(
            Path(module.__file__).resolve() == (repository / relative).resolve(),
            "imported Deform360 runtime comes from another repository",
        )
    episode = staged_root / f"episode_{STAGED_EPISODE_ID:04d}"
    original_hull = reconstruct_stage.visual_hull_points

    def strict_hull(*args: Any, **kwargs: Any) -> Any:
        kwargs["min_points"] = STRICT_HULL_PARAMETERS["minimum_visual_hull_points"]
        return original_hull(*args, **kwargs)

    reconstruct_stage.visual_hull_points = strict_hull
    try:
        reconstruction = reconstruct_stage.process_reconstruction_episode(
            staged_root,
            STAGED_EPISODE_ID,
            cameras=list(request.camera_names),
            first_frame_iterations=STRICT_HULL_PARAMETERS["first_frame_iterations"],
            warm_start_iterations=STRICT_HULL_PARAMETERS["warm_start_iterations"],
            cube_half_extent_m=STRICT_HULL_PARAMETERS["cube_half_extent_m"],
            voxel_resolution=STRICT_HULL_PARAMETERS["voxel_resolution"],
            overwrite=True,
        )
    finally:
        reconstruct_stage.visual_hull_points = original_hull
    _require(
        sorted(reconstruction) == list(range(TRACKING_CONTEXT_FRAME_COUNT)),
        "strict-hull reconstruction did not produce 81 tracking frames",
    )
    cameras = list(request.camera_names)
    depth = depth_stage.process_depth_episode(
        staged_root,
        STAGED_EPISODE_ID,
        cameras=cameras,
        overwrite=True,
        preview=False,
    )
    cotracker, _, cotracker_runtime = _verify_cotracker_runtime(request, backend)
    tracking = tracking_stage.process_tracking_episode(
        staged_root,
        STAGED_EPISODE_ID,
        cameras=cameras,
        checkpoint=cotracker,
        overwrite=True,
    )
    original_threshold = pcd_stage.FUSE_RANSAC_THRESHOLD
    original_inliers = pcd_stage.FUSE_RANSAC_MIN_INLIERS
    pcd_stage.FUSE_RANSAC_THRESHOLD = PCD_PARAMETERS["fusion_maximum_speed_m_per_s"]
    pcd_stage.FUSE_RANSAC_MIN_INLIERS = PCD_PARAMETERS[
        "fusion_minimum_camera_inlier_count"
    ]
    try:
        pcd_directory = pcd_stage.process_pcd_episode(
            staged_root,
            STAGED_EPISODE_ID,
            cameras=cameras,
            overwrite=True,
            rng_seed=PCD_PARAMETERS["rng_seed"],
        )
    finally:
        pcd_stage.FUSE_RANSAC_THRESHOLD = original_threshold
        pcd_stage.FUSE_RANSAC_MIN_INLIERS = original_inliers
    pcd_files = sorted(pcd_directory.glob("*.npz"))
    _require(len(pcd_files) == FRAME_COUNT, "official PCD output is not 76 frames")
    points: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    for path in pcd_files:
        with np.load(path, allow_pickle=False) as stored:
            point = np.asarray(stored["pts"], dtype=np.float32)
            color = np.asarray(stored["colors"], dtype=np.float32)
        _require(
            point.ndim == 2
            and point.shape[1] == 3
            and color.shape == point.shape
            and np.all(np.isfinite(point))
            and np.all(np.isfinite(color)),
            f"invalid official PCD frame: {path.name}",
        )
        points.append(point)
        colors.append(color)
    _require(
        len({value.shape for value in points}) == 1,
        "official PCD material identity count changes across frames",
    )
    point_array = np.stack(points).astype(np.float32, copy=False)
    color_array = np.stack(colors).astype(np.float32, copy=False)
    valid = np.ones(point_array.shape[:2], dtype=bool)
    result = ReconstructionBackendResult(
        object_points=point_array,
        object_colors=color_array,
        object_visibilities=valid,
        object_motions_valid=valid.copy(),
        audit={},
    )
    output_bindings = {
        "reconstruction": {
            str(frame): _bound_file(path)
            for frame, path in sorted(reconstruction.items())
        },
        "depth": {camera: _bound_file(path) for camera, path in sorted(depth.items())},
        "tracking": {
            camera: _tree_binding(path) for camera, path in sorted(tracking.items())
        },
        "pcd": {path.name: _bound_file(path) for path in pcd_files},
    }
    stage_audit = {
        "runtime": runtime,
        "cotracker_runtime": cotracker_runtime,
        "strict_hull_parameters": dict(STRICT_HULL_PARAMETERS),
        "depth_parameters": dict(DEPTH_PARAMETERS),
        "tracking_parameters": dict(TRACKING_PARAMETERS),
        "pcd_parameters": dict(PCD_PARAMETERS),
        "output_bindings": output_bindings,
        "staged_episode_tree": _tree_binding(episode),
    }
    return result, stage_audit


def _run_pinned_official_pipeline(
    request: ReconstructionRequest,
    backend: PinnedOfficialPipelineBackend,
) -> ReconstructionBackendResult:
    started = time.perf_counter()
    ffmpeg_executable, ffmpeg_runtime = _verify_ffmpeg_runtime(request, backend.ffmpeg)
    staged_root, staging = _stage_action_window(request, ffmpeg_executable)
    propagation = _propagate_sealed_masks(request, backend, staged_root)
    result, stages = _run_deform360_stages(request, backend, staged_root)
    audit = {
        "stage_ids": list(STAGE_IDS),
        "contract_sha256": _contract_sha256(request.contract),
        "tracking_context_raw_frame_range_half_open": [
            request.source_frame_start,
            request.source_frame_stop,
        ],
        "tracking_context_frame_count": TRACKING_CONTEXT_FRAME_COUNT,
        "prediction_output_frame_range_half_open": [0, FRAME_COUNT],
        "tracking_tail_frame_range_half_open": [
            FRAME_COUNT,
            TRACKING_CONTEXT_FRAME_COUNT,
        ],
        "frame_zero_anchor": {
            "bundle": dict(request.frame_zero_manifest["bundle"]),
            "intrinsics_sha256": _sha256_array(request.frame_zero_arrays["intrinsics"]),
            "camera_to_world_sha256": _sha256_array(
                request.frame_zero_arrays["camera_to_world"]
            ),
            "rgb_frame0_sha256": _sha256_array(request.frame_zero_arrays["rgb_frame0"]),
            "mask_frame0_sha256": _sha256_array(
                request.frame_zero_arrays["mask_frame0"]
            ),
        },
        "staging": staging,
        "mask_propagation": propagation,
        "official_stages": stages,
        "ffmpeg_runtime": ffmpeg_runtime,
        "tactile_read": False,
        "target_dependent_parameter_selection_or_tuning": False,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    return ReconstructionBackendResult(
        object_points=result.object_points,
        object_colors=result.object_colors,
        object_visibilities=result.object_visibilities,
        object_motions_valid=result.object_motions_valid,
        audit=audit,
    )


_ROBOT_KINEMATICS_STAGING_FIELDS = frozenset(
    {
        "policy_id",
        "contract_sha256",
        "trajectory_semantics",
        "selection_audit",
        "selected_bundle_exact_slice_audit",
        "temporal_fields_sliced_exactly_81",
        "scalar_fields_copied_unchanged",
        "all_five_fields_first_76_equal_selected_bundle",
        "source_array_sha256",
        "staged_array_sha256",
        "selected_array_sha256",
        "commanded_control_or_delta_action_used",
    }
)


def _validate_staged_robot_kinematics(
    request: ReconstructionRequest,
    staging: Mapping[str, Any],
) -> None:
    """Replay the raw-to-81-to-76 robot staging proof after the barrier."""

    source_inputs = staging.get("source_inputs")
    staged_outputs = staging.get("staged_outputs")
    evidence = staging.get("robot_kinematics")
    _require(
        isinstance(source_inputs, Mapping)
        and isinstance(staged_outputs, Mapping)
        and isinstance(evidence, Mapping)
        and set(evidence) == set(_ROBOT_KINEMATICS_STAGING_FIELDS),
        "backend robot kinematics staging evidence changed",
    )
    _require(
        source_inputs.get("robot_kinematics")
        == source_inputs.get("robot_trajectory")
        == request.frame_zero_manifest.get("action_inputs", {}).get(
            "robot_trajectory"
        )
        and source_inputs.get("selected_prediction_robot_kinematics")
        == source_inputs.get("selected_prediction_action")
        == request.frame_zero_manifest.get("action_alignment", {}).get(
            "selected_robot_kinematics_bundle"
        ),
        "backend robot compatibility aliases changed",
    )
    raw_path = _validate_bound_file(
        source_inputs["robot_kinematics"], label="staging source robot kinematics"
    )
    selected_path = _validate_bound_file(
        source_inputs["selected_prediction_robot_kinematics"],
        label="staging selected robot kinematics",
    )
    staged_path = _validate_bound_file(
        staged_outputs["robot"], label="staged 81-frame robot kinematics"
    )
    source_state = load_robot_kinematics_archive(raw_path)
    selected_state = load_robot_kinematics_archive(
        selected_path, expected_frame_count=FRAME_COUNT
    )
    staged_state = load_robot_kinematics_archive(
        staged_path, expected_frame_count=TRACKING_CONTEXT_FRAME_COUNT
    )
    alignment = request.frame_zero_manifest.get("action_alignment")
    config = request.frame_zero_manifest.get("config")
    _require(
        isinstance(alignment, Mapping) and isinstance(config, Mapping),
        "sealed robot alignment or configuration is missing",
    )
    selection = validate_robot_kinematics_selection_audit(
        alignment.get("selection_audit", {}),
        source_state,
        window_length_frames=TRACKING_CONTEXT_FRAME_COUNT,
        prediction_frame_count=FRAME_COUNT,
        candidate_first_frame=int(config["action_candidate_first_frame"]),
        candidate_stride_frames=int(config["action_candidate_stride_frames"]),
    )
    _require(
        selection.get("selected_raw_frame_range_half_open")
        == alignment.get("selected_raw_frame_range_half_open")
        == [request.source_frame_start, request.source_frame_stop]
        and selection.get("prediction_raw_frame_range_half_open")
        == alignment.get("prediction_raw_frame_range_half_open")
        == [request.source_frame_start, request.source_frame_start + FRAME_COUNT],
        "backend robot kinematics range changed",
    )
    exact_slice = validate_selected_robot_kinematics_bundle(
        selected_state,
        source_state=source_state,
        prediction_start_frame=request.source_frame_start,
        prediction_frame_count=FRAME_COUNT,
    )
    expected_staged = slice_robot_kinematics(
        source_state,
        start_frame=request.source_frame_start,
        frame_count=TRACKING_CONTEXT_FRAME_COUNT,
    )
    staged_arrays = staged_state.archive_arrays()
    expected_staged_arrays = expected_staged.archive_arrays()
    selected_arrays = selected_state.archive_arrays()
    for name in ("actions", "T_worlds", "openings"):
        _require(
            np.array_equal(staged_arrays[name], expected_staged_arrays[name])
            and np.array_equal(staged_arrays[name][:FRAME_COUNT], selected_arrays[name]),
            f"backend staged robot temporal field changed: {name}",
        )
    for name in ("format_version", "bimanual"):
        _require(
            np.array_equal(staged_arrays[name], expected_staged_arrays[name])
            and np.array_equal(staged_arrays[name], selected_arrays[name]),
            f"backend staged robot scalar field changed: {name}",
        )
    source_arrays = source_state.archive_arrays()
    expected_source_sha = {
        name: robot_array_sha256(value)
        for name, value in sorted(source_arrays.items())
    }
    expected_staged_sha = {
        name: robot_array_sha256(value)
        for name, value in sorted(staged_arrays.items())
    }
    expected_selected_sha = {
        name: robot_array_sha256(value)
        for name, value in sorted(selected_arrays.items())
    }
    _require(
        evidence.get("policy_id") == ROBOT_KINEMATICS_WINDOW_POLICY_ID
        and evidence.get("contract_sha256")
        == ROBOT_KINEMATICS_WINDOW_CONTRACT_SHA256
        and evidence.get("trajectory_semantics")
        == ROBOT_KINEMATICS_WINDOW_CONTRACT["trajectory_semantics"]
        and evidence.get("selection_audit") == selection
        and evidence.get("selected_bundle_exact_slice_audit") == exact_slice
        and evidence.get("temporal_fields_sliced_exactly_81")
        == ["actions", "T_worlds", "openings"]
        and evidence.get("scalar_fields_copied_unchanged")
        == ["format_version", "bimanual"]
        and evidence.get("all_five_fields_first_76_equal_selected_bundle") is True
        and evidence.get("source_array_sha256") == expected_source_sha
        and evidence.get("staged_array_sha256") == expected_staged_sha
        and evidence.get("selected_array_sha256") == expected_selected_sha
        and evidence.get("commanded_control_or_delta_action_used") is False,
        "backend robot kinematics staging proof changed",
    )


def _validate_backend_result(
    request: ReconstructionRequest,
    result: ReconstructionBackendResult,
) -> ReconstructionBackendResult:
    points = np.asarray(result.object_points)
    colors = np.asarray(result.object_colors)
    visible = np.asarray(result.object_visibilities)
    valid = np.asarray(result.object_motions_valid)
    _require(
        points.dtype == np.dtype(np.float32)
        and points.ndim == 3
        and points.shape[0] == FRAME_COUNT
        and points.shape[2] == 3
        and points.shape[1] > 0
        and np.all(np.isfinite(points)),
        "official target points must be finite float32 (76, M, 3)",
    )
    _require(
        colors.dtype == np.dtype(np.float32)
        and colors.shape == points.shape
        and np.all(np.isfinite(colors)),
        "official target colors changed",
    )
    _require(
        visible.dtype == np.dtype(bool)
        and valid.dtype == np.dtype(bool)
        and visible.shape == points.shape[:2]
        and valid.shape == points.shape[:2],
        "official target masks must be bool (76, M)",
    )
    audit = result.audit
    _require(
        isinstance(audit, Mapping) and set(audit) == _BACKEND_AUDIT_FIELDS,
        "reconstruction backend audit fields changed",
    )
    _require(audit.get("stage_ids") == list(STAGE_IDS), "backend stage order changed")
    _require(
        audit.get("contract_sha256") == _contract_sha256(request.contract),
        "backend used another reconstruction contract",
    )
    _require(
        audit.get("tracking_context_raw_frame_range_half_open")
        == [request.source_frame_start, request.source_frame_stop]
        and audit.get("tracking_context_frame_count") == TRACKING_CONTEXT_FRAME_COUNT
        and audit.get("prediction_output_frame_range_half_open") == [0, FRAME_COUNT]
        and audit.get("tracking_tail_frame_range_half_open")
        == [FRAME_COUNT, TRACKING_CONTEXT_FRAME_COUNT],
        "backend temporal window changed",
    )
    anchor = audit.get("frame_zero_anchor", {})
    _require(
        isinstance(anchor, Mapping)
        and set(anchor)
        == {
            "bundle",
            "intrinsics_sha256",
            "camera_to_world_sha256",
            "rgb_frame0_sha256",
            "mask_frame0_sha256",
        }
        and anchor.get("bundle") == request.frame_zero_manifest["bundle"]
        and anchor.get("intrinsics_sha256")
        == _sha256_array(request.frame_zero_arrays["intrinsics"])
        and anchor.get("camera_to_world_sha256")
        == _sha256_array(request.frame_zero_arrays["camera_to_world"])
        and anchor.get("rgb_frame0_sha256")
        == _sha256_array(request.frame_zero_arrays["rgb_frame0"])
        and anchor.get("mask_frame0_sha256")
        == _sha256_array(request.frame_zero_arrays["mask_frame0"]),
        "backend used another frame-zero anchor",
    )
    propagation = audit.get("mask_propagation", {})
    camera_masks = propagation.get("camera_masks", {})
    _require(
        isinstance(propagation, Mapping)
        and set(propagation)
        == {
            "stage_id",
            "sam2_git",
            "sam2_checkpoint",
            "sam2_model_config",
            "camera_masks",
            "sealed_mask_is_only_initialization",
            "target_dependent_mask_selection_or_tuning",
        }
        and propagation.get("stage_id") == STAGE_IDS[1]
        and propagation.get("sealed_mask_is_only_initialization") is True
        and propagation.get("target_dependent_mask_selection_or_tuning") is False
        and isinstance(camera_masks, Mapping)
        and set(camera_masks) == set(request.camera_names),
        "backend mask camera set changed",
    )
    sealed_masks = np.asarray(request.frame_zero_arrays["mask_frame0"])
    for index, camera in enumerate(request.camera_names):
        record = camera_masks[camera]
        expected = _sha256_array(sealed_masks[index])
        initialization = record.get("initialization", {})
        _require(
            isinstance(record, Mapping)
            and set(record)
            == {
                "mask_archive",
                "frame_count",
                "sealed_frame_zero_mask_sha256",
                "propagated_frame_zero_mask_sha256",
                "initialization",
            }
            and record.get("frame_count") == TRACKING_CONTEXT_FRAME_COUNT
            and record.get("sealed_frame_zero_mask_sha256") == expected
            and record.get("propagated_frame_zero_mask_sha256") == expected
            and initialization
            == {
                "case_name": request.case_name,
                "cohort_barrier_sha256": request.cohort_barrier_sha256,
                "frame_zero_manifest_sha256": _sha256_file(
                    request.frame_zero_manifest_path
                ),
                "sealed_mask_sha256": expected,
                "automatic_initial_mask_selection": False,
            },
            f"backend changed the sealed mask anchor: {camera}",
        )
        _validate_bound_file(record["mask_archive"], label=f"{camera} mask archive")
    staging = audit.get("staging", {})
    _require(
        isinstance(staging, Mapping)
        and set(staging)
        == {
            "source_inputs",
            "staged_outputs",
            "robot_kinematics",
            "source_frame_range_half_open",
            "tracking_context_frame_count",
            "prediction_frame_range_half_open",
            "tracking_tail_frame_range_half_open",
            "video_staging",
            "tactile_paths_read_or_copied",
        }
        and staging.get("source_frame_range_half_open")
        == [request.source_frame_start, request.source_frame_stop]
        and staging.get("tracking_context_frame_count") == TRACKING_CONTEXT_FRAME_COUNT
        and staging.get("prediction_frame_range_half_open") == [0, FRAME_COUNT]
        and staging.get("tracking_tail_frame_range_half_open")
        == [FRAME_COUNT, TRACKING_CONTEXT_FRAME_COUNT]
        and staging.get("video_staging") == VIDEO_STAGING_PARAMETERS
        and staging.get("tactile_paths_read_or_copied") == [],
        "backend staging contract changed",
    )
    _validate_staged_robot_kinematics(request, staging)
    official = audit.get("official_stages", {})
    _require(
        isinstance(official, Mapping)
        and set(official)
        == {
            "runtime",
            "cotracker_runtime",
            "strict_hull_parameters",
            "depth_parameters",
            "tracking_parameters",
            "pcd_parameters",
            "output_bindings",
            "staged_episode_tree",
        }
        and official.get("strict_hull_parameters") == STRICT_HULL_PARAMETERS
        and official.get("depth_parameters") == DEPTH_PARAMETERS
        and official.get("tracking_parameters") == TRACKING_PARAMETERS
        and official.get("pcd_parameters") == PCD_PARAMETERS,
        "official backend parameters changed",
    )
    output_bindings = official.get("output_bindings", {})
    _require(
        isinstance(output_bindings, Mapping)
        and set(output_bindings) == {"reconstruction", "depth", "tracking", "pcd"}
        and set(output_bindings["reconstruction"])
        == {str(frame) for frame in range(TRACKING_CONTEXT_FRAME_COUNT)}
        and set(output_bindings["depth"]) == set(request.camera_names)
        and set(output_bindings["tracking"]) == set(request.camera_names)
        and set(output_bindings["pcd"])
        == {f"{frame:06d}.npz" for frame in range(FRAME_COUNT)},
        "official backend output bindings changed",
    )
    ffmpeg = audit.get("ffmpeg_runtime", {})
    _require(
        isinstance(ffmpeg, Mapping)
        and set(ffmpeg)
        == {
            "executable",
            "version_stdout_sha256",
            "version_stderr_sha256",
        },
        "ffmpeg runtime audit changed",
    )
    _require(
        audit.get("tactile_read") is False
        and audit.get("target_dependent_parameter_selection_or_tuning") is False,
        "backend used forbidden evidence or tuning",
    )
    runtime_seconds = audit.get("runtime_seconds")
    _require(
        isinstance(runtime_seconds, (int, float))
        and np.isfinite(runtime_seconds)
        and runtime_seconds >= 0.0,
        "backend runtime audit is invalid",
    )
    _validate_nested_audit_bindings(audit)
    return ReconstructionBackendResult(
        object_points=points.copy(),
        object_colors=colors.copy(),
        object_visibilities=visible.copy(),
        object_motions_valid=valid.copy(),
        audit=dict(audit),
    )


def _write_target_and_outcome(
    request: ReconstructionRequest,
    result: ReconstructionBackendResult,
) -> OfficialTarget:
    target_path = request.output_dir / "official_target.npz"
    outcome_path = request.output_dir / "held_outcome.json"
    target_arrays = {
        "object_points": result.object_points,
        "object_colors": result.object_colors,
        "object_visibilities": result.object_visibilities,
        "object_motions_valid": result.object_motions_valid,
    }
    _write_new_npz(target_path, target_arrays)
    target_record = _bound_file(target_path)
    target = OfficialTarget(
        object_points=result.object_points,
        object_visibilities=result.object_visibilities,
        object_motions_valid=result.object_motions_valid,
        provenance={},
    )
    array_sha256 = official_target_array_sha256(target)
    outcome: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": OUTCOME_ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "dataset_revision": DATASET_REVISION,
        "case_name": request.case_name,
        "object_id": request.object_id,
        "episode_id": request.episode_id,
        "role": request.role,
        "cohort_barrier_sha256": request.cohort_barrier_sha256,
        "outcome_reconstruction_contract": dict(request.contract),
        "outcome_reconstruction_contract_sha256": _contract_sha256(request.contract),
        "inputs": {
            "online_prediction_seal": _bound_file(request.online_seal_path),
            "frame_zero_manifest": _bound_file(request.frame_zero_manifest_path),
            "frame_zero_bundle": dict(request.frame_zero_manifest["bundle"]),
        },
        "target_file": target_record,
        "target_array_sha256": array_sha256,
        "backend_audit": dict(result.audit),
        "information_boundary": {
            "complete_cohort_barrier_validated_before_future_open": True,
            "official_target_constructed_or_read_after_barrier": True,
            "prediction_metric_computed_during_target_construction": False,
            "tracking_tail_used_only_as_pinned_context": True,
            "tactile_read": False,
            "target_dependent_parameter_selection_or_tuning": False,
        },
        "claim_limitation": CLAIM_LIMITATION,
    }
    outcome["artifact_sha256"] = held_artifact_sha256(outcome)
    _write_new_json(outcome_path, outcome)
    outcome_record = _bound_file(outcome_path)
    provenance = {
        "target_artifact_kind": TARGET_ARTIFACT_KIND,
        "outcome_artifact_kind": OUTCOME_ARTIFACT_KIND,
        "case_name": request.case_name,
        "object_id": request.object_id,
        "episode_id": request.episode_id,
        "dataset_revision": DATASET_REVISION,
        "cohort_barrier_sha256": request.cohort_barrier_sha256,
        "target_file": target_record,
        "outcome_file": outcome_record,
        "array_sha256": array_sha256,
        "information_boundary": {
            "complete_cohort_barrier_validated_before_future_open": True,
            "official_target_constructed_or_read_after_barrier": True,
            "prediction_metric_computed_during_target_construction": False,
        },
    }
    return OfficialTarget(
        object_points=result.object_points,
        object_visibilities=result.object_visibilities,
        object_motions_valid=result.object_motions_valid,
        provenance=provenance,
    )


def _load_completed_target(request: ReconstructionRequest) -> OfficialTarget:
    """Validate and return one complete write-once reconstruction output."""

    output = request.output_dir
    target_path = output / "official_target.npz"
    outcome_path = output / "held_outcome.json"
    _require(
        target_path.is_file()
        and not target_path.is_symlink()
        and outcome_path.is_file()
        and not outcome_path.is_symlink(),
        "outcome reconstruction directory is partial",
    )
    allowed_children = {target_path.name, outcome_path.name}
    staged = output / "staged-aligned"
    if staged.exists():
        _require(staged.is_dir() and not staged.is_symlink(), "staged output is unsafe")
        allowed_children.add(staged.name)
    _require(
        {path.name for path in output.iterdir()} == allowed_children,
        "outcome reconstruction directory contains partial or unexpected entries",
    )
    outcome = _load_json(outcome_path)
    _require(set(outcome) == _OUTCOME_FIELDS, "held outcome fields changed")
    _require(
        outcome.get("schema_version") == 1
        and outcome.get("artifact_kind") == OUTCOME_ARTIFACT_KIND
        and outcome.get("protocol_id") == PROTOCOL_ID
        and outcome.get("dataset_revision") == DATASET_REVISION,
        "held outcome schema changed",
    )
    _require(
        outcome.get("case_name") == request.case_name
        and outcome.get("object_id") == request.object_id
        and outcome.get("episode_id") == request.episode_id
        and outcome.get("role") == request.role
        and outcome.get("cohort_barrier_sha256") == request.cohort_barrier_sha256,
        "held outcome identity or cohort changed",
    )
    _require(
        outcome.get("outcome_reconstruction_contract") == request.contract
        and outcome.get("outcome_reconstruction_contract_sha256")
        == _contract_sha256(request.contract),
        "held outcome reconstruction contract changed",
    )
    inputs = outcome.get("inputs", {})
    _require(
        isinstance(inputs, Mapping)
        and set(inputs)
        == {
            "online_prediction_seal",
            "frame_zero_manifest",
            "frame_zero_bundle",
        }
        and inputs.get("online_prediction_seal")
        == _bound_file(request.online_seal_path)
        and inputs.get("frame_zero_manifest")
        == _bound_file(request.frame_zero_manifest_path)
        and inputs.get("frame_zero_bundle") == request.frame_zero_manifest["bundle"],
        "held outcome sealed inputs changed",
    )
    _require(
        outcome.get("target_file") == _bound_file(target_path),
        "held target file binding changed",
    )
    _require(
        outcome.get("information_boundary")
        == {
            "complete_cohort_barrier_validated_before_future_open": True,
            "official_target_constructed_or_read_after_barrier": True,
            "prediction_metric_computed_during_target_construction": False,
            "tracking_tail_used_only_as_pinned_context": True,
            "tactile_read": False,
            "target_dependent_parameter_selection_or_tuning": False,
        }
        and outcome.get("claim_limitation") == CLAIM_LIMITATION,
        "held outcome information boundary changed",
    )
    _require(
        outcome.get("artifact_sha256") == held_artifact_sha256(outcome),
        "held outcome content checksum changed",
    )
    with np.load(target_path, allow_pickle=False) as stored:
        _require(
            set(stored.files) == _TARGET_ARCHIVE_ARRAYS,
            "official target archive array set changed",
        )
        result = ReconstructionBackendResult(
            object_points=np.asarray(stored["object_points"]).copy(),
            object_colors=np.asarray(stored["object_colors"]).copy(),
            object_visibilities=np.asarray(stored["object_visibilities"]).copy(),
            object_motions_valid=np.asarray(stored["object_motions_valid"]).copy(),
            audit=dict(outcome["backend_audit"]),
        )
    validated = _validate_backend_result(request, result)
    target = OfficialTarget(
        object_points=validated.object_points,
        object_visibilities=validated.object_visibilities,
        object_motions_valid=validated.object_motions_valid,
        provenance={},
    )
    array_sha256 = official_target_array_sha256(target)
    _require(
        outcome.get("target_array_sha256") == array_sha256,
        "official target arrays changed",
    )
    provenance = {
        "target_artifact_kind": TARGET_ARTIFACT_KIND,
        "outcome_artifact_kind": OUTCOME_ARTIFACT_KIND,
        "case_name": request.case_name,
        "object_id": request.object_id,
        "episode_id": request.episode_id,
        "dataset_revision": DATASET_REVISION,
        "cohort_barrier_sha256": request.cohort_barrier_sha256,
        "target_file": _bound_file(target_path),
        "outcome_file": _bound_file(outcome_path),
        "array_sha256": array_sha256,
        "information_boundary": {
            "complete_cohort_barrier_validated_before_future_open": True,
            "official_target_constructed_or_read_after_barrier": True,
            "prediction_metric_computed_during_target_construction": False,
        },
    }
    return OfficialTarget(
        object_points=validated.object_points,
        object_visibilities=validated.object_visibilities,
        object_motions_valid=validated.object_motions_valid,
        provenance=provenance,
    )


@dataclass(frozen=True)
class PermitGatedOfficialReconstructionCallback:
    """Callable accepted by ``TargetOperation``; all paths remain unopened."""

    permit: OutcomePhasePermit
    case_name: str
    aligned_episode_dir: str
    output_dir: str
    backend: ReconstructionBackend

    def __call__(self) -> OfficialTarget:
        return run_outcome_operation(
            self.permit,
            case_name=self.case_name,
            operation="create",
            callback=self._after_permit,
        )

    def _after_permit(self) -> OfficialTarget:
        request = _load_sealed_request(
            self.permit,
            self.case_name,
            self.aligned_episode_dir,
            self.output_dir,
            operation="create",
        )
        request.output_dir.mkdir(parents=True, exist_ok=False)
        result = _validate_backend_result(request, self.backend.build(request))
        _write_target_and_outcome(request, result)
        return _load_completed_target(request)


@dataclass(frozen=True)
class PermitGatedOfficialReconstructionReadCallback:
    """Read one complete immutable output only after cohort revalidation."""

    permit: OutcomePhasePermit
    case_name: str
    output_dir: str

    def __call__(self) -> OfficialTarget:
        return run_outcome_operation(
            self.permit,
            case_name=self.case_name,
            operation="read",
            callback=self._after_permit,
        )

    def _after_permit(self) -> OfficialTarget:
        request = _load_sealed_request(
            self.permit,
            self.case_name,
            None,
            self.output_dir,
            operation="read",
        )
        return _load_completed_target(request)


def make_official_reconstruction_target_operation(
    permit: OutcomePhasePermit,
    *,
    case_name: str,
    aligned_episode_dir: str | Path,
    output_dir: str | Path,
    backend: ReconstructionBackend,
) -> TargetOperation:
    """Return a scorer-compatible operation without touching future evidence."""

    callback = PermitGatedOfficialReconstructionCallback(
        permit=permit,
        case_name=str(case_name),
        aligned_episode_dir=os.fspath(aligned_episode_dir),
        output_dir=os.fspath(output_dir),
        backend=backend,
    )
    return TargetOperation(operation="create", callback=callback)


def make_official_reconstruction_read_target_operation(
    permit: OutcomePhasePermit,
    *,
    case_name: str,
    output_dir: str | Path,
) -> TargetOperation:
    """Return a scorer-compatible, permit-gated immutable resume read."""

    callback = PermitGatedOfficialReconstructionReadCallback(
        permit=permit,
        case_name=str(case_name),
        output_dir=os.fspath(output_dir),
    )
    return TargetOperation(operation="read", callback=callback)


def plan_official_reconstruction_target_operation(
    permit: OutcomePhasePermit,
    *,
    case_name: str,
    aligned_episode_dir: str | Path,
    output_dir: str | Path,
    backend: ReconstructionBackend,
) -> TargetOperation:
    """Classify an absent/complete output behind the live cohort barrier.

    An absent path produces a CREATE operation.  A complete output is fully
    revalidated before a READ operation is returned.  Every other existing
    path is treated as a partial/invalid write and fails closed.
    """

    case = str(case_name)
    aligned = os.fspath(aligned_episode_dir)
    output = os.fspath(output_dir)

    def after_permit() -> TargetOperation:
        if not os.path.lexists(output):
            return make_official_reconstruction_target_operation(
                permit,
                case_name=case,
                aligned_episode_dir=aligned,
                output_dir=output,
                backend=backend,
            )
        operation = make_official_reconstruction_read_target_operation(
            permit,
            case_name=case,
            output_dir=output,
        )
        try:
            operation.callback()
        except Exception as error:
            raise ValueError(
                "existing outcome reconstruction is partial or invalid"
            ) from error
        return operation

    return run_outcome_operation(
        permit,
        case_name=case,
        operation="read",
        callback=after_permit,
    )


__all__ = [
    "DEFORM360_PROCESSING_REVISION",
    "DEFORM360_SOURCE_SHA256",
    "DEPTH_PARAMETERS",
    "LOCAL_PROPAGATION_SOURCE_SHA256",
    "PCD_PARAMETERS",
    "PermitGatedOfficialReconstructionCallback",
    "PermitGatedOfficialReconstructionReadCallback",
    "PinnedOfficialPipelineBackend",
    "RECONSTRUCTION_ADAPTER_KIND",
    "RECONSTRUCTION_STAGE_KIND",
    "ReconstructionBackend",
    "ReconstructionBackendResult",
    "ReconstructionRequest",
    "SAM2_CHECKPOINT_SHA256",
    "SAM2_COMMIT",
    "SAM2_MODEL_CONFIG",
    "SAM2_MODEL_CONFIG_REPOSITORY_PATH",
    "STAGE_IDS",
    "STRICT_HULL_PARAMETERS",
    "TRACKING_CONTEXT_FRAME_COUNT",
    "TRACKING_PARAMETERS",
    "TRACKING_TAIL_FRAME_COUNT",
    "VIDEO_STAGING_PARAMETERS",
    "make_official_reconstruction_target_operation",
    "make_official_reconstruction_read_target_operation",
    "plan_official_reconstruction_target_operation",
]
