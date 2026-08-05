"""Frozen MotionCrafter jobs for the official-Hub calibration cohort.

The job manifest is created before provider inference. It binds every camera
video, causal frame interval, complete Prob4D product set, and deterministic
seed schedule without reading prediction values or calibration outcomes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from ._portable_contracts import (
    canonical_json_bytes,
    content_id,
    exact_revision,
    load_strict_json_object,
    sha256_digest,
    write_atomic_json,
)
from .deform360_official_hub_causal_windows import (
    DEFORM360_VISUAL_PROVIDER_RECOVERY_LOCK_ID,
    validate_deform360_official_hub_causal_window_manifest_v2,
)
from .deform360_visual_provider_recovery_lock import (
    Deform360VisualProviderRecoveryLockV1,
)

DEFORM360_MOTIONCRAFTER_JOB_MANIFEST_SCHEMA = (
    "bayesian-phystwin.deform360-official-hub-motioncrafter-jobs"
)
DEFORM360_MOTIONCRAFTER_JOB_MANIFEST_VERSION = 1
DEFORM360_V2_CAUSAL_WINDOW_MANIFEST_ID = (
    "9fe5fdf4ae6449182d2e5064ad99417b4252dd04b76831df722b44614c2351dd"
)
DEFORM360_MOTIONCRAFTER_MODEL_SET_ID = (
    "2e5cf9bbf1fa0a61b985ed440a437ba8ea736ae643964d8449429e75a836de02"
)
DEFORM360_V2_CAUSAL_WINDOW_MANIFEST_FILE_SHA256 = (
    "7398575f32ea8868f241da9356264d5b44b815f41425f6f259afcbbd10f336de"
)
DEFORM360_PROVIDER_LOCK_FILE_SHA256 = (
    "ca60602a799f42d151f58de80d71bda2966f7e01eeedbeca911fb82860aa2656"
)
DEFORM360_MODEL_SET_MANIFEST_FILE_SHA256 = (
    "5968a8664ab1dd936a1609a5581d6bde7d40b16d929b7f1bbef7bc042b6c52f5"
)
DEFORM360_PROB4D_REVISION = "364f216c14f7770c1b360bb1b836b11ecf0c18b8"
DEFORM360_MOTIONCRAFTER_REVISION = "1d6a8947ec6ebabbcf4fc1e0f6d06828fcf6f257"
MOTIONCRAFTER_SEED_SCHEDULE_SCHEMA = "prob4d.motioncrafter-seed-schedule.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _safe_relative_path(value: object, *, name: str) -> str:
    _require(type(value) is str and bool(value), f"{name} must be a string")
    result = str(value)
    pure = PurePosixPath(result)
    _require(
        result == pure.as_posix()
        and not pure.is_absolute()
        and all(part not in {"", ".", ".."} for part in pure.parts),
        f"{name} must be a safe POSIX relative path",
    )
    return result


def _positive_integer(value: object, *, name: str) -> int:
    _require(
        type(value) is int and int(value) > 0,
        f"{name} must be a positive integer",
    )
    return int(value)


def _nonnegative_integer(value: object, *, name: str) -> int:
    _require(
        type(value) is int and int(value) >= 0,
        f"{name} must be a non-negative integer",
    )
    return int(value)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def motioncrafter_effective_seed(root_seed: int, *, call_id: str) -> int:
    """Mirror Prob4D's frozen ``derived-per-call`` seed derivation."""

    _require(
        type(root_seed) is int and 0 <= root_seed < 2**32,
        "root_seed must lie in [0, 2**32)",
    )
    _require(type(call_id) is str and bool(call_id), "call_id must be non-empty")
    descriptor = {
        "schema": MOTIONCRAFTER_SEED_SCHEDULE_SCHEMA,
        "root_seed": root_seed,
        "call_id": call_id,
    }
    return int.from_bytes(hashlib.sha256(canonical_json_bytes(descriptor)).digest()[:4], "big")


def _expected_seed_schedule(
    *,
    root_seed: int,
    windows: list[Mapping[str, Any]],
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
            root_seed,
            call_id=str(call["call_id"]),
        )
    return calls


def _video_descriptor(
    case: Mapping[str, Any],
    *,
    camera: str,
) -> dict[str, object]:
    suffix = f"/{camera}/undistorted.mp4"
    files = case.get("bound_input_files")
    _require(isinstance(files, list), "case lacks bound input files")
    matches = [
        item
        for item in files
        if isinstance(item, Mapping)
        and type(item.get("path")) is str
        and str(item["path"]).endswith(suffix)
    ]
    _require(len(matches) == 1, f"camera {camera!r} has no unique bound video")
    source = matches[0]
    return {
        "path": _safe_relative_path(source.get("path"), name="source video path"),
        "sha256": sha256_digest(source.get("sha256"), name="source video sha256"),
        "bytes": _positive_integer(source.get("size"), name="source video bytes"),
    }


def build_deform360_motioncrafter_job_manifest(
    *,
    causal_window_manifest: Mapping[str, Any],
    causal_window_manifest_file_sha256: str,
    provider_lock: Deform360VisualProviderRecoveryLockV1,
    provider_lock_file_sha256: str,
    model_set_manifest: Mapping[str, Any],
    model_set_manifest_file_sha256: str,
    implementation_revision: str,
    runner_source_sha256: str,
) -> dict[str, Any]:
    """Build the frozen 30-camera inference plan without opening image values."""

    parent_id = validate_deform360_official_hub_causal_window_manifest_v2(
        causal_window_manifest
    )
    _require(
        parent_id == DEFORM360_V2_CAUSAL_WINDOW_MANIFEST_ID,
        "unexpected v2 causal-window manifest",
    )
    _require(
        provider_lock.artifact_id
        == str(causal_window_manifest["visual_provider_recovery_lock_id"]),
        "provider lock differs from causal-window manifest",
    )
    _require(
        provider_lock.seed_policy == "derived-per-call",
        "job manifest requires derived-per-call seeds",
    )
    model_set_id = content_id(model_set_manifest)
    _require(
        model_set_id == DEFORM360_MOTIONCRAFTER_MODEL_SET_ID,
        "MotionCrafter model-set identity changed",
    )
    _require(
        model_set_id == provider_lock.motioncrafter_model_set_id,
        "provider lock and model-set identity differ",
    )
    implementation = exact_revision(
        implementation_revision,
        name="implementation_revision",
    )

    jobs: list[dict[str, Any]] = []
    cases = causal_window_manifest.get("cases")
    _require(isinstance(cases, list), "causal-window manifest lacks cases")
    for case in cases:
        _require(isinstance(case, Mapping), "causal-window case must be an object")
        _require(case.get("status") == "success", "all v2 cases must be successful")
        object_id = _safe_relative_path(case.get("object_id"), name="object_id")
        _require(len(PurePosixPath(object_id).parts) == 1, "object_id must be one segment")
        panel = case.get("camera_panel")
        _require(
            isinstance(panel, list) and len(panel) == 3,
            "camera panel must contain three cameras",
        )
        causal = case.get("causal_window")
        provider_windows = case.get("provider_windows")
        _require(isinstance(causal, Mapping), "case lacks causal window")
        _require(
            isinstance(provider_windows, list) and len(provider_windows) == 2,
            "case must contain two provider windows",
        )
        source_start = _nonnegative_integer(
            causal.get("source_start_frame"),
            name="source_start_frame",
        )
        cutoff = _positive_integer(
            causal.get("causal_cutoff_frame"),
            name="causal_cutoff_frame",
        )
        _require(cutoff - source_start == 42, "provider source must contain 42 frames")
        windows: list[dict[str, object]] = []
        for index, source in enumerate(provider_windows):
            _require(isinstance(source, Mapping), "provider window must be an object")
            start = _nonnegative_integer(
                source.get("frame_start"),
                name="provider window start",
            )
            stop = _positive_integer(
                source.get("frame_stop_exclusive"),
                name="provider window stop",
            )
            _require(
                source.get("window_index") == index
                and stop - start == provider_lock.descriptor()["window_size"]
                and source_start <= start < stop <= cutoff,
                "provider window changed",
            )
            windows.append(
                {
                    "window_id": f"window_{index:04d}",
                    "source_frame_start": start,
                    "source_frame_stop_exclusive": stop,
                }
            )
        for camera_value in panel:
            camera = _safe_relative_path(camera_value, name="camera name")
            _require(len(PurePosixPath(camera).parts) == 1, "camera must be one segment")
            descriptor: dict[str, Any] = {
                "object_id": object_id,
                "episode": "episode_0000",
                "camera": camera,
                "source_video": _video_descriptor(case, camera=camera),
                "source_frame_start": source_start,
                "source_frame_stop_exclusive": cutoff,
                "source_frame_count": cutoff - source_start,
                "windows": windows,
                "seed_schedule": _expected_seed_schedule(
                    root_seed=provider_lock.root_seed,
                    windows=windows,
                ),
                "output_relative_path": (
                    f"{object_id}/episode_0000/{camera}"
                ),
            }
            jobs.append({"job_id": content_id(descriptor), **descriptor})
    _require(len(jobs) == 30, "expected exactly 30 camera jobs")
    _require(
        len({str(job["job_id"]) for job in jobs}) == len(jobs),
        "job identities are not unique",
    )

    descriptor = {
        "schema": DEFORM360_MOTIONCRAFTER_JOB_MANIFEST_SCHEMA,
        "schema_version": DEFORM360_MOTIONCRAFTER_JOB_MANIFEST_VERSION,
        "protocol_id": provider_lock.protocol_id,
        "role": "calibration",
        "status": "locked-pre-provider-inference",
        "implementation": {
            "revision": implementation,
            "runner_source_sha256": sha256_digest(
                runner_source_sha256,
                name="runner source sha256",
            ),
        },
        "causal_window_manifest": {
            "manifest_sha256": parent_id,
            "file_sha256": sha256_digest(
                causal_window_manifest_file_sha256,
                name="causal-window manifest file sha256",
            ),
        },
        "provider_lock": {
            "artifact_id": provider_lock.artifact_id,
            "file_sha256": sha256_digest(
                provider_lock_file_sha256,
                name="provider lock file sha256",
            ),
            "provider_revision": provider_lock.provider_revision,
        },
        "motioncrafter": {
            "repository": provider_lock.motioncrafter_repository,
            "revision": provider_lock.motioncrafter_revision,
            "model_set_id": model_set_id,
            "model_set_manifest_file_sha256": sha256_digest(
                model_set_manifest_file_sha256,
                name="model-set manifest file sha256",
            ),
            "model_set_manifest": dict(model_set_manifest),
        },
        "run_configuration": {
            "model_type": provider_lock.motioncrafter_model_type,
            "height": provider_lock.height,
            "width": provider_lock.width,
            "window_size": 25,
            "overlap": 8,
            "num_inference_steps": 5,
            "guidance_scale": 1.0,
            "decode_chunk_size": 25,
            "seed": provider_lock.root_seed,
            "seed_policy": provider_lock.seed_policy,
            "low_memory_usage": True,
            "frame_stride": 1,
            "model_source_set_sha256": model_set_id,
            "products": [
                "disjoint_baseline",
                "latent_linear_baseline",
                "independently_decoded_overlap_windows",
            ],
            "provider_consumed_product": "independently_decoded_overlap_windows",
        },
        "object_count": 10,
        "job_count": len(jobs),
        "smoke_job_id": jobs[0]["job_id"],
        "jobs": jobs,
        "information_boundary": {
            "calibration_camera_payload_authorized_for_provider_inference": True,
            "calibration_provider_outputs_opened": False,
            "calibration_scores_opened": False,
            "calibration_policy_fit": False,
            "future_frames_used_for_prediction": False,
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
        },
        "claim_boundary": (
            "Pre-inference calibration job schedule and input custody only. The "
            "complete Prob4D product set is retained for strict calibration "
            "compatibility, while only independently decoded overlap windows may "
            "enter the causal provider. This artifact establishes no provider "
            "competence, calibration, physical improvement, confirmation, or SOTA claim."
        ),
    }
    return {"manifest_sha256": content_id(descriptor), **descriptor}


def validate_deform360_motioncrafter_job_manifest(
    value: Mapping[str, Any],
) -> str:
    """Validate one frozen calibration job manifest and return its identity."""

    declared = sha256_digest(value.get("manifest_sha256"), name="manifest_sha256")
    descriptor = dict(value)
    descriptor.pop("manifest_sha256")
    _require(content_id(descriptor) == declared, "job-manifest identity changed")
    _require(
        value.get("schema") == DEFORM360_MOTIONCRAFTER_JOB_MANIFEST_SCHEMA
        and value.get("schema_version")
        == DEFORM360_MOTIONCRAFTER_JOB_MANIFEST_VERSION,
        "unsupported job manifest",
    )
    _require(value.get("role") == "calibration", "job-manifest role changed")
    _require(
        value.get("status") == "locked-pre-provider-inference",
        "job-manifest status changed",
    )
    _require(value.get("object_count") == 10, "object count changed")
    implementation = value.get("implementation")
    _require(isinstance(implementation, Mapping), "implementation binding is missing")
    exact_revision(implementation.get("revision"), name="implementation revision")
    sha256_digest(
        implementation.get("runner_source_sha256"),
        name="runner source sha256",
    )
    causal = value.get("causal_window_manifest")
    provider = value.get("provider_lock")
    motioncrafter = value.get("motioncrafter")
    _require(isinstance(causal, Mapping), "causal-window binding is missing")
    _require(isinstance(provider, Mapping), "provider binding is missing")
    _require(isinstance(motioncrafter, Mapping), "MotionCrafter binding is missing")
    _require(
        causal.get("manifest_sha256") == DEFORM360_V2_CAUSAL_WINDOW_MANIFEST_ID
        and causal.get("file_sha256")
        == DEFORM360_V2_CAUSAL_WINDOW_MANIFEST_FILE_SHA256,
        "v2 causal-window binding changed",
    )
    _require(
        provider.get("artifact_id") == DEFORM360_VISUAL_PROVIDER_RECOVERY_LOCK_ID
        and provider.get("file_sha256") == DEFORM360_PROVIDER_LOCK_FILE_SHA256
        and provider.get("provider_revision") == DEFORM360_PROB4D_REVISION,
        "provider binding changed",
    )
    _require(
        motioncrafter.get("repository") == "TencentARC/MotionCrafter"
        and motioncrafter.get("revision") == DEFORM360_MOTIONCRAFTER_REVISION
        and motioncrafter.get("model_set_id") == DEFORM360_MOTIONCRAFTER_MODEL_SET_ID
        and motioncrafter.get("model_set_manifest_file_sha256")
        == DEFORM360_MODEL_SET_MANIFEST_FILE_SHA256,
        "MotionCrafter binding changed",
    )
    model_set = motioncrafter.get("model_set_manifest")
    _require(
        isinstance(model_set, Mapping)
        and content_id(model_set) == DEFORM360_MOTIONCRAFTER_MODEL_SET_ID,
        "embedded MotionCrafter model set changed",
    )
    jobs = value.get("jobs")
    _require(isinstance(jobs, list) and len(jobs) == 30, "job count changed")
    _require(value.get("job_count") == len(jobs), "declared job count changed")
    configuration = value.get("run_configuration")
    _require(isinstance(configuration, Mapping), "run configuration is missing")
    root_seed = _nonnegative_integer(configuration.get("seed"), name="root seed")
    expected_configuration = {
        "model_type": "determ",
        "height": 320,
        "width": 640,
        "window_size": 25,
        "overlap": 8,
        "num_inference_steps": 5,
        "guidance_scale": 1.0,
        "decode_chunk_size": 25,
        "seed": 20260805,
        "seed_policy": "derived-per-call",
        "low_memory_usage": True,
        "frame_stride": 1,
        "model_source_set_sha256": DEFORM360_MOTIONCRAFTER_MODEL_SET_ID,
        "products": [
            "disjoint_baseline",
            "latent_linear_baseline",
            "independently_decoded_overlap_windows",
        ],
        "provider_consumed_product": "independently_decoded_overlap_windows",
    }
    _require(
        dict(configuration) == expected_configuration,
        "run configuration changed",
    )
    job_ids: list[str] = []
    for job in jobs:
        _require(isinstance(job, Mapping), "job must be an object")
        job_id = sha256_digest(job.get("job_id"), name="job_id")
        job_descriptor = dict(job)
        job_descriptor.pop("job_id")
        _require(content_id(job_descriptor) == job_id, "job identity changed")
        job_ids.append(job_id)
        source_start = _nonnegative_integer(
            job.get("source_frame_start"),
            name="source frame start",
        )
        source_stop = _positive_integer(
            job.get("source_frame_stop_exclusive"),
            name="source frame stop",
        )
        _require(source_stop - source_start == 42, "job source extent changed")
        _safe_relative_path(job.get("output_relative_path"), name="job output path")
        source = job.get("source_video")
        windows = job.get("windows")
        schedule = job.get("seed_schedule")
        _require(isinstance(source, Mapping), "job source video is missing")
        _safe_relative_path(source.get("path"), name="source video path")
        sha256_digest(source.get("sha256"), name="source video sha256")
        _positive_integer(source.get("bytes"), name="source video bytes")
        _require(
            isinstance(windows, list) and len(windows) == 2,
            "job windows changed",
        )
        expected_schedule = _expected_seed_schedule(
            root_seed=root_seed,
            windows=windows,
        )
        _require(schedule == expected_schedule, "job seed schedule changed")
    _require(len(set(job_ids)) == len(job_ids), "job IDs are not unique")
    _require(value.get("smoke_job_id") == job_ids[0], "smoke job changed")
    boundary = value.get("information_boundary")
    _require(isinstance(boundary, Mapping), "information boundary is missing")
    _require(
        boundary.get("calibration_camera_payload_authorized_for_provider_inference")
        is True
        and boundary.get("calibration_provider_outputs_opened") is False
        and boundary.get("calibration_scores_opened") is False
        and boundary.get("calibration_policy_fit") is False
        and boundary.get("future_frames_used_for_prediction") is False
        and boundary.get("confirmation_payloads_opened") is False
        and boundary.get("target_outcomes_used") is False,
        "job manifest crossed its information boundary",
    )
    return declared


def load_deform360_motioncrafter_job_manifest(
    path: str | Path,
) -> Mapping[str, Any]:
    """Load and strictly validate one frozen job manifest."""

    value = load_strict_json_object(path, label="MotionCrafter job manifest")
    validate_deform360_motioncrafter_job_manifest(value)
    return value


def save_deform360_motioncrafter_job_manifest(
    path: str | Path,
    value: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    """Validate and atomically save one job manifest."""

    validate_deform360_motioncrafter_job_manifest(value)
    write_atomic_json(value, path, overwrite=overwrite)


__all__ = [
    "DEFORM360_MOTIONCRAFTER_JOB_MANIFEST_SCHEMA",
    "DEFORM360_MOTIONCRAFTER_JOB_MANIFEST_VERSION",
    "build_deform360_motioncrafter_job_manifest",
    "load_deform360_motioncrafter_job_manifest",
    "motioncrafter_effective_seed",
    "save_deform360_motioncrafter_job_manifest",
    "validate_deform360_motioncrafter_job_manifest",
]
