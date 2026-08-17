"""Outcome-blind MotionCrafter schedule for the public Deform360 v5 source panel.

The archived official-Hub provider products were generated for an earlier
contact-centered prefix.  They are useful runtime provenance, but their frame
ranges are not interchangeable with the action-only prefixes frozen by the v5
source execution lock.  This module binds the same released videos and frozen
Prob4D/MotionCrafter stack to a deterministic, source-specific causal range.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

from ._portable_contracts import (
    canonical_json_bytes,
    content_id,
    exact_revision,
    load_strict_json_object,
    nonempty_string,
    require_exact_fields,
    sha256_digest,
    write_atomic_json,
)
from .deform360_calibration_visual_execution_admission import (
    validate_deform360_prepared_source_inventory,
)
from .deform360_joint_sparse_endpoint_v5 import select_reserved_endpoint_views_v5
from .deform360_joint_sparse_source_gate_v5 import (
    load_deform360_joint_sparse_source_execution_lock_v5,
)

SOURCE_PROVIDER_SCHEMA: Final = (
    "bayesian-phystwin.deform360-joint-sparse-motioncrafter-source-plan"
)
SOURCE_PROVIDER_VERSION: Final = 1
SOURCE_PROVIDER_SEMANTICS: Final = (
    "latest-42-of-locked-58-frame-public-prefix-prob4d-v1"
)
SOURCE_PROVIDER_STATUS: Final = "locked-before-source-provider-inference"

V5_EXECUTION_LOCK_ID: Final = (
    "76b74483790ace51d642889be2e3dbb22149e30f7919b5855a18066434e25189"
)
V5_PREPARED_INVENTORY_ID: Final = (
    "6994aa621b38dc8fb21cd38e43363bde3ea12dd644532addeecfc07a30f84e7b"
)
V5_PREPARED_INVENTORY_FILE_SHA256: Final = (
    "4da96c4f636d195f7aea5d971fbd83bd3b0f35b1c66a77af68007bbd08a69007"
)
LEGACY_CAMERA_ROSTER_MANIFEST_ID: Final = (
    "9726e7ae12d442956ff81376fe52cdc2f8360fdcd3e5cccbc12543ca584b30f9"
)
LEGACY_CAMERA_ROSTER_FILE_SHA256: Final = (
    "b9302a27d779a6de619baffc04e624eee629a226a140b90278fa9dd06b213fe2"
)
PROB4D_REVISION: Final = "25d90ef7f78ba4307f4555cb636d666004e1bf66"
MOTIONCRAFTER_REVISION: Final = "1d6a8947ec6ebabbcf4fc1e0f6d06828fcf6f257"
MOTIONCRAFTER_MODEL_SET_ID: Final = (
    "b072956636612ca1a31d1edb83bd7d1bd27b8962cb617c6e615b9b310a16de6e"
)
MOTIONCRAFTER_SEED_SCHEDULE_SCHEMA: Final = "prob4d.motioncrafter-seed-schedule.v1"

PREFIX_FRAME_COUNT: Final = 58
PROVIDER_FRAME_COUNT: Final = 42
WINDOW_SIZE: Final = 25
WINDOW_OVERLAP: Final = 8
WINDOW_STEP: Final = WINDOW_SIZE - WINDOW_OVERLAP
OBJECT_COUNT: Final = 10
CAMERAS_PER_OBJECT: Final = 3

RUN_CONFIGURATION: Final = {
    "model_type": "determ",
    "height": 320,
    "width": 640,
    "window_size": WINDOW_SIZE,
    "overlap": WINDOW_OVERLAP,
    "num_inference_steps": 5,
    "guidance_scale": 1.0,
    "decode_chunk_size": 25,
    "seed": 20260805,
    "seed_policy": "derived-per-call",
    "low_memory_usage": True,
    "frame_stride": 1,
    "model_source_set_sha256": MOTIONCRAFTER_MODEL_SET_ID,
    "products": [
        "disjoint_baseline",
        "latent_linear_baseline",
        "independently_decoded_overlap_windows",
    ],
    "provider_consumed_product": "independently_decoded_overlap_windows",
}

TEMPORAL_POLICY: Final = {
    "locked_prefix_frame_count": PREFIX_FRAME_COUNT,
    "provider_frame_count": PROVIDER_FRAME_COUNT,
    "selection": "latest-42-frames-within-each-locked-58-frame-prefix",
    "window_size": WINDOW_SIZE,
    "window_overlap": WINDOW_OVERLAP,
    "window_step": WINDOW_STEP,
    "window_count": 2,
    "future_frames_permitted": False,
    "legacy_provider_frame_ranges_reused": False,
    "legacy_provider_outputs_reused": False,
}

INFORMATION_BOUNDARY: Final = {
    "public_source_prefix_payloads_authorized": True,
    "provider_outputs_opened": False,
    "development_suffix_opened": False,
    "future_object_observations_used": False,
    "confirmation_payloads_opened": False,
    "target_outcomes_used": False,
    "human_approval_required": False,
    "new_measurements_required": False,
}

CLAIM_BOUNDARY: Final = (
    "Source-only provider scheduling and input custody repair. The plan uses "
    "released real-world Deform360 videos inside the already locked causal "
    "prefix and establishes no prediction benefit, calibration, confirmation, "
    "Causal4D benefit, safety, or state-of-the-art claim."
)

_PLAN_FIELDS = frozenset(
    {
        "manifest_sha256",
        "schema",
        "schema_version",
        "semantics",
        "status",
        "role",
        "implementation",
        "source_execution_lock",
        "prepared_source_inventory",
        "camera_roster_source",
        "provider_lock",
        "motioncrafter",
        "run_configuration",
        "temporal_policy",
        "objects",
        "object_count",
        "jobs",
        "job_count",
        "smoke_job_id",
        "information_boundary",
        "claim_boundary",
    }
)
_OBJECT_FIELDS = frozenset(
    {
        "object_id",
        "episode_id",
        "stratum",
        "raw_prefix_range_half_open",
        "provider_range_half_open",
        "all_camera_ids",
        "reserved_endpoint_camera_ids",
        "provider_camera_ids",
        "likelihood_camera_ids",
    }
)
_JOB_FIELDS = frozenset(
    {
        "job_id",
        "object_id",
        "episode_id",
        "source_episode",
        "stratum",
        "camera",
        "likelihood_eligible",
        "source_video",
        "source_frame_start",
        "source_frame_stop_exclusive",
        "source_frame_count",
        "windows",
        "seed_schedule",
        "output_relative_path",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return cast(Sequence[Any], value)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative_path(value: object, *, name: str) -> str:
    result = nonempty_string(value, name=name)
    path = PurePosixPath(result)
    _require(
        result == path.as_posix()
        and not path.is_absolute()
        and all(part not in {"", ".", ".."} for part in path.parts),
        f"{name} must be a safe POSIX relative path",
    )
    return result


def _nonnegative_integer(value: object, *, name: str) -> int:
    _require(type(value) is int, f"{name} must be an integer")
    result = cast(int, value)
    _require(result >= 0, f"{name} must be non-negative")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    _require(type(value) is int, f"{name} must be an integer")
    result = cast(int, value)
    _require(result > 0, f"{name} must be positive")
    return result


def motioncrafter_effective_seed(root_seed: int, *, call_id: str) -> int:
    """Mirror the frozen Prob4D derived-per-call seed schedule."""

    _require(
        type(root_seed) is int and 0 <= root_seed < 2**32,
        "root_seed must lie in [0, 2**32)",
    )
    call = nonempty_string(call_id, name="call_id")
    descriptor = {
        "schema": MOTIONCRAFTER_SEED_SCHEDULE_SCHEMA,
        "root_seed": root_seed,
        "call_id": call,
    }
    return int.from_bytes(
        hashlib.sha256(canonical_json_bytes(descriptor)).digest()[:4], "big"
    )


def _windows(source_start: int, source_stop: int) -> list[dict[str, object]]:
    _require(
        source_stop - source_start == PROVIDER_FRAME_COUNT,
        "provider range must contain exactly 42 frames",
    )
    windows = [
        {
            "window_id": "window_0000",
            "source_frame_start": source_start,
            "source_frame_stop_exclusive": source_start + WINDOW_SIZE,
        },
        {
            "window_id": "window_0001",
            "source_frame_start": source_start + WINDOW_STEP,
            "source_frame_stop_exclusive": source_stop,
        },
    ]
    _require(
        all(
            cast(int, item["source_frame_stop_exclusive"])
            - cast(int, item["source_frame_start"])
            == WINDOW_SIZE
            for item in windows
        ),
        "provider windows changed",
    )
    return windows


def _seed_schedule(
    root_seed: int, windows: Sequence[Mapping[str, Any]]
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = [
        {"call_id": "baseline-disjoint", "product": "disjoint_baseline"},
        {
            "call_id": "baseline-latent-linear",
            "product": "latent_linear_baseline",
        },
    ]
    for window in windows:
        window_id = str(window["window_id"])
        start = int(window["source_frame_start"])
        stop = int(window["source_frame_stop_exclusive"])
        calls.append(
            {
                "call_id": f"overlap-window:{window_id}:{start}:{stop}",
                "product": "independently_decoded_overlap_window",
                "window_id": window_id,
                "source_frame_start": start,
                "source_frame_stop_exclusive": stop,
            }
        )
    for call in calls:
        call["effective_seed"] = motioncrafter_effective_seed(
            root_seed, call_id=str(call["call_id"])
        )
    return calls


def _content_addressed(value: Mapping[str, Any], *, identity: str) -> None:
    declared = sha256_digest(value.get(identity), name=identity)
    descriptor = dict(value)
    descriptor.pop(identity)
    _require(content_id(descriptor) == declared, f"{identity} changed")


def _legacy_camera_rosters(value: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    _content_addressed(value, identity="manifest_sha256")
    _require(
        value.get("manifest_sha256") == LEGACY_CAMERA_ROSTER_MANIFEST_ID,
        "unexpected legacy camera-roster manifest",
    )
    _require(
        value.get("schema")
        == "bayesian-phystwin.deform360-official-hub-motioncrafter-jobs"
        and value.get("schema_version") == 1
        and value.get("status") == "locked-pre-provider-inference",
        "legacy camera-roster manifest contract changed",
    )
    provider = _mapping(value.get("provider_lock"), name="legacy provider lock")
    motion = _mapping(value.get("motioncrafter"), name="legacy MotionCrafter")
    _require(
        provider.get("provider_revision") == PROB4D_REVISION,
        "legacy Prob4D revision changed",
    )
    _require(
        motion.get("revision") == MOTIONCRAFTER_REVISION
        and motion.get("model_set_id") == MOTIONCRAFTER_MODEL_SET_ID,
        "legacy MotionCrafter binding changed",
    )
    model_set = _mapping(
        motion.get("model_set_manifest"), name="MotionCrafter model-set manifest"
    )
    _require(content_id(model_set) == MOTIONCRAFTER_MODEL_SET_ID, "model set changed")
    _require(
        value.get("run_configuration") == RUN_CONFIGURATION,
        "legacy run configuration changed",
    )
    boundary = _mapping(value.get("information_boundary"), name="legacy boundary")
    _require(
        boundary.get("confirmation_payloads_opened") is False
        and boundary.get("target_outcomes_used") is False
        and boundary.get("future_frames_used_for_prediction") is False,
        "legacy manifest crossed its information boundary",
    )
    jobs = _sequence(value.get("jobs"), name="legacy jobs")
    _require(len(jobs) == OBJECT_COUNT * CAMERAS_PER_OBJECT, "legacy job count changed")
    rosters: dict[str, list[str]] = {}
    for raw_job in jobs:
        job = _mapping(raw_job, name="legacy job")
        _content_addressed(job, identity="job_id")
        object_id = nonempty_string(job.get("object_id"), name="legacy object_id")
        camera = nonempty_string(job.get("camera"), name="legacy camera")
        _require(job.get("episode") == "episode_0000", "legacy episode changed")
        rosters.setdefault(object_id, []).append(camera)
    _require(len(rosters) == OBJECT_COUNT, "legacy object count changed")
    result: dict[str, tuple[str, ...]] = {}
    for object_id, cameras in rosters.items():
        _require(
            len(cameras) == CAMERAS_PER_OBJECT and len(set(cameras)) == len(cameras),
            f"legacy camera roster changed: {object_id}",
        )
        result[object_id] = tuple(sorted(cameras))
    return result


def _cohort(lock: Mapping[str, Any]) -> dict[str, tuple[int, str]]:
    _require(lock.get("execution_lock_id") == V5_EXECUTION_LOCK_ID, "lock changed")
    cohort = _mapping(lock.get("cohort"), name="cohort")
    rows = _sequence(cohort.get("development_objects"), name="development objects")
    result: dict[str, tuple[int, str]] = {}
    for raw in rows:
        row = _mapping(raw, name="development object")
        object_id = nonempty_string(row.get("object_id"), name="object_id")
        episode_id = _nonnegative_integer(row.get("episode_id"), name="episode_id")
        stratum = row.get("stratum")
        _require(stratum in {"sheet", "volumetric"}, "stratum changed")
        _require(object_id not in result, "development object repeats")
        result[object_id] = (episode_id, cast(str, stratum))
    _require(len(result) == OBJECT_COUNT, "source cohort must contain ten objects")
    return result


def _inventory_rows(inventory: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    _require(
        inventory.get("inventory_id") == V5_PREPARED_INVENTORY_ID,
        "prepared inventory identity changed",
    )
    boundary = _mapping(
        inventory.get("information_boundary"), name="inventory boundary"
    )
    _require(
        boundary.get("calibration_target_metrics_computed") is False
        and boundary.get("confirmation_payloads_opened") is False
        and boundary.get("target_outcomes_used") is False,
        "prepared inventory crossed its information boundary",
    )
    rows = _sequence(inventory.get("objects"), name="inventory objects")
    result: dict[str, Mapping[str, Any]] = {}
    for raw in rows:
        row = _mapping(raw, name="inventory object")
        object_id = nonempty_string(row.get("object_id"), name="inventory object_id")
        _require(object_id not in result, "inventory object repeats")
        result[object_id] = row
    _require(len(result) == OBJECT_COUNT, "inventory object count changed")
    return result


def _camera_video(row: Mapping[str, Any], camera: str) -> dict[str, object]:
    cameras = _sequence(row.get("cameras"), name="inventory cameras")
    matches = [
        _mapping(item, name="inventory camera")
        for item in cameras
        if isinstance(item, Mapping) and item.get("camera") == camera
    ]
    _require(len(matches) == 1, f"inventory camera is missing or repeated: {camera}")
    video = _mapping(matches[0].get("video"), name=f"{camera} video")
    path = _safe_relative_path(video.get("path"), name=f"{camera} video path")
    expected_suffix = f"/{camera}/undistorted.mp4"
    _require(path.endswith(expected_suffix), f"{camera} video path changed")
    return {
        "path": path,
        "sha256": sha256_digest(video.get("sha256"), name=f"{camera} video sha256"),
        "bytes": _positive_integer(video.get("byte_count"), name=f"{camera} bytes"),
    }


def build_deform360_joint_sparse_motioncrafter_source_plan_v5(
    *,
    lock: Mapping[str, Any],
    execution_lock_file_sha256: str,
    inventory: Mapping[str, Any],
    inventory_file_sha256: str,
    legacy_job_manifest: Mapping[str, Any],
    legacy_job_manifest_file_sha256: str,
    implementation_revision: str,
    runner_source_sha256: str,
) -> dict[str, Any]:
    """Build the source-only provider plan before any provider output is opened."""

    cohort = _cohort(lock)
    inventory_rows = _inventory_rows(inventory)
    rosters = _legacy_camera_rosters(legacy_job_manifest)
    _require(set(cohort) == set(inventory_rows) == set(rosters), "cohort changed")
    _require(
        sha256_digest(inventory_file_sha256, name="inventory file sha256")
        == V5_PREPARED_INVENTORY_FILE_SHA256,
        "prepared inventory file changed",
    )
    _require(
        sha256_digest(
            legacy_job_manifest_file_sha256,
            name="legacy job-manifest file sha256",
        )
        == LEGACY_CAMERA_ROSTER_FILE_SHA256,
        "legacy camera-roster file changed",
    )
    revision = exact_revision(implementation_revision, name="implementation revision")
    runner_sha = sha256_digest(runner_source_sha256, name="runner source sha256")
    root_seed = cast(int, RUN_CONFIGURATION["seed"])

    objects: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    for object_id in sorted(cohort):
        episode_id, stratum = cohort[object_id]
        row = inventory_rows[object_id]
        _require(row.get("episode_id") == episode_id, f"episode changed: {object_id}")
        _require(row.get("stratum") == stratum, f"stratum changed: {object_id}")
        action = _mapping(row.get("action_window"), name=f"{object_id} action window")
        prefix = list(
            _sequence(
                action.get("prefix_raw_frame_range_half_open"),
                name=f"{object_id} prefix range",
            )
        )
        _require(
            len(prefix) == 2
            and all(type(item) is int for item in prefix)
            and prefix[1] - prefix[0] == PREFIX_FRAME_COUNT,
            f"locked prefix changed: {object_id}",
        )
        source_stop = cast(int, prefix[1])
        source_start = source_stop - PROVIDER_FRAME_COUNT
        windows = _windows(source_start, source_stop)
        all_cameras = tuple(
            sorted(
                nonempty_string(
                    _mapping(item, name="inventory camera").get("camera"),
                    name="camera",
                )
                for item in _sequence(row.get("cameras"), name="inventory cameras")
            )
        )
        _require(len(all_cameras) >= 4, f"camera roster too small: {object_id}")
        reserved = select_reserved_endpoint_views_v5(object_id, all_cameras, count=2)
        provider_cameras = rosters[object_id]
        _require(
            set(provider_cameras).issubset(all_cameras),
            f"provider camera is outside inventory: {object_id}",
        )
        likelihood_cameras = tuple(
            camera for camera in provider_cameras if camera not in reserved
        )
        _require(
            len(likelihood_cameras) >= 2,
            f"fewer than two non-reserved provider cameras: {object_id}",
        )
        objects.append(
            {
                "object_id": object_id,
                "episode_id": episode_id,
                "stratum": stratum,
                "raw_prefix_range_half_open": prefix,
                "provider_range_half_open": [source_start, source_stop],
                "all_camera_ids": list(all_cameras),
                "reserved_endpoint_camera_ids": list(reserved),
                "provider_camera_ids": list(provider_cameras),
                "likelihood_camera_ids": list(likelihood_cameras),
            }
        )
        for camera in provider_cameras:
            descriptor: dict[str, Any] = {
                "object_id": object_id,
                "episode_id": episode_id,
                "source_episode": "episode_0000",
                "stratum": stratum,
                "camera": camera,
                "likelihood_eligible": camera in likelihood_cameras,
                "source_video": _camera_video(row, camera),
                "source_frame_start": source_start,
                "source_frame_stop_exclusive": source_stop,
                "source_frame_count": PROVIDER_FRAME_COUNT,
                "windows": windows,
                "seed_schedule": _seed_schedule(root_seed, windows),
                "output_relative_path": (
                    f"objects/{object_id}/episode_{episode_id:04d}/views/{camera}"
                ),
            }
            jobs.append({"job_id": content_id(descriptor), **descriptor})

    motion = _mapping(
        legacy_job_manifest.get("motioncrafter"), name="legacy MotionCrafter"
    )
    provider = _mapping(
        legacy_job_manifest.get("provider_lock"), name="legacy provider lock"
    )
    descriptor = {
        "schema": SOURCE_PROVIDER_SCHEMA,
        "schema_version": SOURCE_PROVIDER_VERSION,
        "semantics": SOURCE_PROVIDER_SEMANTICS,
        "status": SOURCE_PROVIDER_STATUS,
        "role": "development-source",
        "implementation": {
            "revision": revision,
            "runner_source_sha256": runner_sha,
        },
        "source_execution_lock": {
            "execution_lock_id": V5_EXECUTION_LOCK_ID,
            "file_sha256": sha256_digest(
                execution_lock_file_sha256, name="execution lock file sha256"
            ),
        },
        "prepared_source_inventory": {
            "inventory_id": V5_PREPARED_INVENTORY_ID,
            "file_sha256": V5_PREPARED_INVENTORY_FILE_SHA256,
        },
        "camera_roster_source": {
            "manifest_sha256": LEGACY_CAMERA_ROSTER_MANIFEST_ID,
            "file_sha256": LEGACY_CAMERA_ROSTER_FILE_SHA256,
            "use": "camera-identities-and-frozen-runtime-bindings-only",
            "legacy_frame_ranges_rejected": True,
            "legacy_provider_outputs_rejected": True,
        },
        "provider_lock": dict(provider),
        "motioncrafter": dict(motion),
        "run_configuration": dict(RUN_CONFIGURATION),
        "temporal_policy": dict(TEMPORAL_POLICY),
        "objects": objects,
        "object_count": len(objects),
        "jobs": jobs,
        "job_count": len(jobs),
        "smoke_job_id": jobs[0]["job_id"],
        "information_boundary": dict(INFORMATION_BOUNDARY),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    plan = {"manifest_sha256": content_id(descriptor), **descriptor}
    validate_deform360_joint_sparse_motioncrafter_source_plan_v5(plan)
    return plan


def validate_deform360_joint_sparse_motioncrafter_source_plan_v5(
    value: Mapping[str, Any],
) -> str:
    """Strictly validate one v5 source-provider plan and return its identity."""

    require_exact_fields(value, expected=_PLAN_FIELDS, name="source-provider plan")
    _content_addressed(value, identity="manifest_sha256")
    declared = sha256_digest(value.get("manifest_sha256"), name="manifest_sha256")
    _require(
        value.get("schema") == SOURCE_PROVIDER_SCHEMA
        and value.get("schema_version") == SOURCE_PROVIDER_VERSION
        and value.get("semantics") == SOURCE_PROVIDER_SEMANTICS
        and value.get("status") == SOURCE_PROVIDER_STATUS
        and value.get("role") == "development-source",
        "source-provider contract changed",
    )
    implementation = _mapping(value.get("implementation"), name="implementation")
    exact_revision(implementation.get("revision"), name="implementation revision")
    sha256_digest(
        implementation.get("runner_source_sha256"), name="runner source sha256"
    )
    lock = _mapping(value.get("source_execution_lock"), name="source lock binding")
    _require(lock.get("execution_lock_id") == V5_EXECUTION_LOCK_ID, "lock changed")
    sha256_digest(lock.get("file_sha256"), name="source lock file sha256")
    inventory = _mapping(
        value.get("prepared_source_inventory"), name="inventory binding"
    )
    _require(
        inventory
        == {
            "inventory_id": V5_PREPARED_INVENTORY_ID,
            "file_sha256": V5_PREPARED_INVENTORY_FILE_SHA256,
        },
        "inventory binding changed",
    )
    roster = _mapping(value.get("camera_roster_source"), name="camera roster source")
    _require(
        roster
        == {
            "manifest_sha256": LEGACY_CAMERA_ROSTER_MANIFEST_ID,
            "file_sha256": LEGACY_CAMERA_ROSTER_FILE_SHA256,
            "use": "camera-identities-and-frozen-runtime-bindings-only",
            "legacy_frame_ranges_rejected": True,
            "legacy_provider_outputs_rejected": True,
        },
        "camera-roster provenance changed",
    )
    provider = _mapping(value.get("provider_lock"), name="provider lock")
    motion = _mapping(value.get("motioncrafter"), name="MotionCrafter")
    _require(provider.get("provider_revision") == PROB4D_REVISION, "Prob4D changed")
    _require(
        motion.get("revision") == MOTIONCRAFTER_REVISION
        and motion.get("model_set_id") == MOTIONCRAFTER_MODEL_SET_ID,
        "MotionCrafter changed",
    )
    model_set = _mapping(motion.get("model_set_manifest"), name="model set")
    _require(content_id(model_set) == MOTIONCRAFTER_MODEL_SET_ID, "model set changed")
    _require(value.get("run_configuration") == RUN_CONFIGURATION, "config changed")
    _require(value.get("temporal_policy") == TEMPORAL_POLICY, "time policy changed")
    _require(
        value.get("information_boundary") == INFORMATION_BOUNDARY,
        "information boundary changed",
    )
    _require(value.get("claim_boundary") == CLAIM_BOUNDARY, "claim boundary changed")

    objects = _sequence(value.get("objects"), name="objects")
    jobs = _sequence(value.get("jobs"), name="jobs")
    _require(
        len(objects) == OBJECT_COUNT and value.get("object_count") == OBJECT_COUNT,
        "object count changed",
    )
    _require(
        len(jobs) == OBJECT_COUNT * CAMERAS_PER_OBJECT
        and value.get("job_count") == len(jobs),
        "job count changed",
    )
    object_map: dict[str, Mapping[str, Any]] = {}
    for raw_object in objects:
        item = _mapping(raw_object, name="object plan")
        require_exact_fields(item, expected=_OBJECT_FIELDS, name="object plan")
        object_id = nonempty_string(item.get("object_id"), name="object_id")
        _require(object_id not in object_map, "object plan repeats")
        prefix = list(_sequence(item.get("raw_prefix_range_half_open"), name="prefix"))
        provider_range = list(
            _sequence(item.get("provider_range_half_open"), name="provider range")
        )
        _require(
            len(prefix) == len(provider_range) == 2
            and all(type(index) is int for index in prefix + provider_range)
            and prefix[1] - prefix[0] == PREFIX_FRAME_COUNT
            and provider_range == [prefix[1] - PROVIDER_FRAME_COUNT, prefix[1]],
            f"provider range is not the latest causal prefix: {object_id}",
        )
        all_cameras = tuple(_sequence(item.get("all_camera_ids"), name="all cameras"))
        reserved = select_reserved_endpoint_views_v5(object_id, all_cameras, count=2)
        provider_cameras = tuple(
            _sequence(item.get("provider_camera_ids"), name="provider cameras")
        )
        likelihood = tuple(
            _sequence(item.get("likelihood_camera_ids"), name="likelihood cameras")
        )
        _require(
            tuple(item.get("reserved_endpoint_camera_ids", ())) == reserved
            and len(provider_cameras) == CAMERAS_PER_OBJECT
            and provider_cameras == tuple(sorted(set(provider_cameras)))
            and set(provider_cameras).issubset(all_cameras)
            and likelihood
            == tuple(camera for camera in provider_cameras if camera not in reserved)
            and len(likelihood) >= 2,
            f"camera policy changed: {object_id}",
        )
        object_map[object_id] = item
    _require(list(object_map) == sorted(object_map), "objects are not sorted")

    job_ids: list[str] = []
    seen_pairs: set[tuple[str, str]] = set()
    for raw_job in jobs:
        job = _mapping(raw_job, name="job")
        require_exact_fields(job, expected=_JOB_FIELDS, name="job")
        _content_addressed(job, identity="job_id")
        job_id = sha256_digest(job.get("job_id"), name="job_id")
        object_id = nonempty_string(job.get("object_id"), name="object_id")
        camera = nonempty_string(job.get("camera"), name="camera")
        _require(object_id in object_map, "job object is outside cohort")
        object_plan = object_map[object_id]
        provider_range = cast(list[int], object_plan["provider_range_half_open"])
        _require(
            job.get("episode_id") == object_plan.get("episode_id")
            and job.get("stratum") == object_plan.get("stratum")
            and job.get("source_episode") == "episode_0000"
            and camera in object_plan["provider_camera_ids"],
            "job identity changed",
        )
        _require(
            job.get("likelihood_eligible")
            is (camera in object_plan["likelihood_camera_ids"]),
            "likelihood eligibility changed",
        )
        start = _nonnegative_integer(job.get("source_frame_start"), name="source start")
        stop = _positive_integer(
            job.get("source_frame_stop_exclusive"), name="source stop"
        )
        _require(
            [start, stop] == provider_range
            and job.get("source_frame_count") == PROVIDER_FRAME_COUNT,
            "job source range changed",
        )
        windows = list(_sequence(job.get("windows"), name="windows"))
        _require(windows == _windows(start, stop), "job windows changed")
        _require(
            job.get("seed_schedule")
            == _seed_schedule(cast(int, RUN_CONFIGURATION["seed"]), windows),
            "job seed schedule changed",
        )
        source = _mapping(job.get("source_video"), name="source video")
        path = _safe_relative_path(source.get("path"), name="source video path")
        _require(
            path.endswith(f"/{camera}/undistorted.mp4"),
            "source video camera changed",
        )
        sha256_digest(source.get("sha256"), name="source video sha256")
        _positive_integer(source.get("bytes"), name="source video bytes")
        expected_output = (
            f"objects/{object_id}/episode_{int(job['episode_id']):04d}/views/{camera}"
        )
        _require(
            _safe_relative_path(job.get("output_relative_path"), name="output path")
            == expected_output,
            "job output path changed",
        )
        pair = (object_id, camera)
        _require(pair not in seen_pairs, "object/camera job repeats")
        seen_pairs.add(pair)
        job_ids.append(job_id)
    _require(len(set(job_ids)) == len(job_ids), "job IDs repeat")
    _require(value.get("smoke_job_id") == job_ids[0], "smoke job changed")
    return declared


def build_deform360_joint_sparse_motioncrafter_source_plan_from_paths_v5(
    *,
    execution_lock_path: str | Path,
    prepared_source_inventory_path: str | Path,
    legacy_job_manifest_path: str | Path,
    implementation_revision: str,
    runner_source_path: str | Path,
) -> dict[str, Any]:
    """Load exact public inputs and build one immutable source-provider plan."""

    lock_path = Path(execution_lock_path).resolve(strict=True)
    inventory_path = Path(prepared_source_inventory_path).resolve(strict=True)
    legacy_path = Path(legacy_job_manifest_path).resolve(strict=True)
    runner_path = Path(runner_source_path).resolve(strict=True)
    lock = load_deform360_joint_sparse_source_execution_lock_v5(lock_path)
    inventory = validate_deform360_prepared_source_inventory(
        load_strict_json_object(inventory_path, label="prepared source inventory")
    )
    legacy = load_strict_json_object(legacy_path, label="legacy camera-roster manifest")
    return build_deform360_joint_sparse_motioncrafter_source_plan_v5(
        lock=lock,
        execution_lock_file_sha256=_file_sha256(lock_path),
        inventory=inventory,
        inventory_file_sha256=_file_sha256(inventory_path),
        legacy_job_manifest=legacy,
        legacy_job_manifest_file_sha256=_file_sha256(legacy_path),
        implementation_revision=implementation_revision,
        runner_source_sha256=_file_sha256(runner_path),
    )


def load_deform360_joint_sparse_motioncrafter_source_plan_v5(
    path: str | Path,
) -> Mapping[str, Any]:
    """Load and validate one source-provider plan."""

    value = load_strict_json_object(path, label="source-provider plan")
    validate_deform360_joint_sparse_motioncrafter_source_plan_v5(value)
    return value


def save_deform360_joint_sparse_motioncrafter_source_plan_v5(
    path: str | Path,
    value: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    """Validate and atomically save one source-provider plan."""

    validate_deform360_joint_sparse_motioncrafter_source_plan_v5(value)
    write_atomic_json(value, path, overwrite=overwrite)


__all__ = [
    "LEGACY_CAMERA_ROSTER_FILE_SHA256",
    "LEGACY_CAMERA_ROSTER_MANIFEST_ID",
    "MOTIONCRAFTER_MODEL_SET_ID",
    "MOTIONCRAFTER_REVISION",
    "PROB4D_REVISION",
    "SOURCE_PROVIDER_SCHEMA",
    "SOURCE_PROVIDER_SEMANTICS",
    "SOURCE_PROVIDER_VERSION",
    "build_deform360_joint_sparse_motioncrafter_source_plan_from_paths_v5",
    "build_deform360_joint_sparse_motioncrafter_source_plan_v5",
    "load_deform360_joint_sparse_motioncrafter_source_plan_v5",
    "motioncrafter_effective_seed",
    "save_deform360_joint_sparse_motioncrafter_source_plan_v5",
    "validate_deform360_joint_sparse_motioncrafter_source_plan_v5",
]
