"""Portable contracts for admitted Deform360 calibration visual production.

The metadata-only execution admission fixes each camera job and its exact
retained source bytes.  This module defines the command, prediction seal,
technical-failure, and complete-accounting contracts used by the protected
self-hosted producer.  It contains no model loading and opens no dataset files.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

from ._canonical_contracts import canonical_relative_posix_path, plain_json
from ._portable_contracts import (
    content_id,
    exact_revision,
    require_exact_fields,
    sha256_digest,
)
from .deform360_visual_provider_lock import Deform360VisualProviderLockV1

DEFORM360_CALIBRATION_VISUAL_PREDICTION_SEAL_SCHEMA: Final = (
    "bayesian-phystwin.deform360-calibration-visual-prediction-seal"
)
DEFORM360_CALIBRATION_VISUAL_TECHNICAL_FAILURE_SCHEMA: Final = (
    "bayesian-phystwin.deform360-calibration-visual-technical-failure"
)
DEFORM360_CALIBRATION_VISUAL_PRODUCTION_RESULT_SCHEMA: Final = (
    "bayesian-phystwin.deform360-calibration-visual-production-result"
)

_MODEL_BINDING_SCHEMA: Final = (
    "bayesian-phystwin/deform360-motioncrafter-model-set-binding-v1"
)
_MODEL_SET_SCHEMA: Final = "prob4d.motioncrafter-model-set.v2"
_MODEL_SOURCE_SCHEMA: Final = "prob4d.motioncrafter-model-source.v1"
_MOTIONCRAFTER_INTEGRITY_SCHEMA: Final = (
    "prob4d.motioncrafter-artifact-integrity.v1"
)
_MOTIONCRAFTER_RUN_SPEC_SCHEMA: Final = "prob4d.motioncrafter-run-spec.v1"

PRODUCTION_INFORMATION_BOUNDARY: Final = {
    "retained_calibration_camera_payloads_opened": True,
    "motioncrafter_prediction_payloads_opened": True,
    "reserved_evaluation_frames_opened": False,
    "calibration_tactile_payloads_opened": False,
    "calibration_robot_state_opened": False,
    "confirmation_payloads_opened": False,
    "target_outcomes_used": False,
    "replacement_allowed": False,
}

_MODEL_BINDING_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "model_set_id",
        "model_set_manifest",
        "cached_revisions",
        "cache_path_recorded",
        "selected_raw_payloads_opened",
        "target_outcomes_used",
    }
)
_MODEL_MANIFEST_FIELDS = frozenset(
    {"schema", "model_type", "sources", "loader_module"}
)
_MODEL_SOURCE_FIELDS = frozenset(
    {"schema", "role", "kind", "repository", "revision"}
)
_FILE_FIELDS = frozenset({"path", "sha256", "byte_count"})
_SEAL_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "seal_id",
        "implementation_revision",
        "admission_id",
        "job_id",
        "object_id",
        "episode_id",
        "stratum",
        "camera_id",
        "provider_revision",
        "motioncrafter_revision",
        "visual_provider_lock_id",
        "model_set_id",
        "command_id",
        "source_video",
        "source_timestamps",
        "causal_prefix_frame_range_half_open",
        "reserved_evaluation_frame_range_half_open",
        "prediction_manifest",
        "run_spec_sha256",
        "verified_member_count",
        "output_relative_directory",
        "information_boundary",
    }
)
_FAILURE_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "failure_id",
        "implementation_revision",
        "admission_id",
        "job_id",
        "object_id",
        "episode_id",
        "stratum",
        "camera_id",
        "provider_revision",
        "motioncrafter_revision",
        "visual_provider_lock_id",
        "model_set_id",
        "command_id",
        "stage",
        "return_code",
        "detail_sha256",
        "stdout",
        "stderr",
        "output_relative_directory",
        "information_boundary",
    }
)
_RESULT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "result_id",
        "implementation_revision",
        "admission_id",
        "visual_provider_lock_id",
        "provider_revision",
        "motioncrafter_revision",
        "model_set_id",
        "object_count",
        "camera_view_count",
        "succeeded_job_count",
        "technical_failure_job_count",
        "completely_succeeded_object_count",
        "status",
        "jobs",
        "information_boundary",
    }
)
_RESULT_JOB_FIELDS = frozenset(
    {"job_id", "object_id", "camera_id", "status", "receipt"}
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty literal string")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _file_record(
    value: object,
    *,
    name: str,
    minimum_bytes: int = 1,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    require_exact_fields(value, expected=_FILE_FIELDS, name=name)
    return {
        "path": canonical_relative_posix_path(value["path"], name=f"{name}.path"),
        "sha256": sha256_digest(value["sha256"], name=f"{name}.sha256"),
        "byte_count": _integer(
            value["byte_count"],
            name=f"{name}.byte_count",
            minimum=minimum_bytes,
        ),
    }


def validate_deform360_motioncrafter_model_set_binding(
    value: object,
    *,
    expected_model_set_id: str,
) -> dict[str, Any]:
    """Validate the portable model-set binding committed by the provider freeze."""

    if not isinstance(value, Mapping):
        raise ValueError("model-set binding must be a JSON object")
    binding = cast(dict[str, Any], plain_json(value))
    require_exact_fields(
        binding,
        expected=_MODEL_BINDING_FIELDS,
        name="model-set binding",
    )
    _require(
        binding["schema"] == _MODEL_BINDING_SCHEMA
        and binding["schema_version"] == 1,
        "unsupported model-set binding",
    )
    for field in (
        "cache_path_recorded",
        "selected_raw_payloads_opened",
        "target_outcomes_used",
    ):
        _require(type(binding[field]) is bool and not binding[field], f"{field} changed")
    model_set_id = sha256_digest(binding["model_set_id"], name="model_set_id")
    _require(model_set_id == expected_model_set_id, "model-set identity changed")
    manifest = binding["model_set_manifest"]
    if not isinstance(manifest, Mapping):
        raise ValueError("model_set_manifest must be a JSON object")
    require_exact_fields(
        manifest,
        expected=_MODEL_MANIFEST_FIELDS,
        name="model-set manifest",
    )
    _require(
        manifest["schema"] == _MODEL_SET_SCHEMA and manifest["model_type"] == "determ",
        "model-set manifest changed",
    )
    _require(
        hashlib.sha256(_canonical_json(manifest)).hexdigest() == model_set_id,
        "model-set digest mismatch",
    )
    sources = manifest["sources"]
    roles = ("unet", "vae", "image_vae", "base_pipeline")
    _require(isinstance(sources, Mapping) and set(sources) == set(roles), "model roles changed")
    cached = binding["cached_revisions"]
    _require(isinstance(cached, Mapping) and set(cached) == set(roles), "cache roles changed")
    normalized_sources: dict[str, dict[str, str]] = {}
    for role in roles:
        source = cast(Mapping[str, Any], sources[role])
        require_exact_fields(source, expected=_MODEL_SOURCE_FIELDS, name=f"source {role}")
        _require(
            source["schema"] == _MODEL_SOURCE_SCHEMA
            and source["kind"] == "huggingface_revision",
            f"model source {role} changed kind",
        )
        revision = exact_revision(source["revision"], name=f"{role} revision")
        _require(cached[role] == revision, f"cached {role} revision changed")
        normalized_sources[role] = {
            "repository": _string(source["repository"], name=f"{role} repository"),
            "revision": revision,
        }
    loader = manifest["loader_module"]
    _require(isinstance(loader, Mapping), "model loader descriptor is missing")
    require_exact_fields(
        cast(Mapping[str, Any], loader),
        expected=frozenset({"module", "sha256", "bytes"}),
        name="model loader descriptor",
    )
    _require(loader["module"] == "prob4d.motioncrafter_models", "model loader changed")
    return {
        "model_set_id": model_set_id,
        "manifest": plain_json(manifest),
        "sources": normalized_sources,
        "loader_sha256": sha256_digest(loader["sha256"], name="loader sha256"),
        "loader_bytes": _integer(loader["bytes"], name="loader bytes", minimum=1),
    }


def deform360_calibration_visual_command_descriptor(
    *,
    admission: Mapping[str, Any],
    job: Mapping[str, Any],
    provider_lock: Deform360VisualProviderLockV1,
    model_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the path-free command identity for one admitted camera job."""

    prefix = cast(Sequence[int], job["prefix_source_frame_range_half_open"])
    return {
        "schema": "bayesian-phystwin.deform360-calibration-visual-command-v1",
        "admission_id": admission["admission_id"],
        "job_id": job["job_id"],
        "provider_revision": provider_lock.provider_revision,
        "motioncrafter_revision": provider_lock.motioncrafter_revision,
        "model_set_id": model_binding["model_set_id"],
        "model_type": "determ",
        "height": provider_lock.height,
        "width": provider_lock.width,
        "window_size": provider_lock.window_size,
        "overlap": provider_lock.overlap,
        "num_inference_steps": 5,
        "guidance_scale": 1.0,
        "decode_chunk_size": 25,
        "seed": job["view_root_seed"],
        "seed_policy": "derived-per-call",
        "frame_start": prefix[0],
        "frame_stop": prefix[1],
        "frame_stride": 1,
        "source_video": job["source_video"],
        "output_relative_directory": job["output_relative_directory"],
        "model_sources": model_binding["sources"],
    }


def build_deform360_calibration_visual_command(
    *,
    executable: str | Path,
    source_video_path: str | Path,
    output_directory: str | Path,
    motioncrafter_root: str | Path,
    cache_directory: str | Path,
    job: Mapping[str, Any],
    provider_lock: Deform360VisualProviderLockV1,
    model_binding: Mapping[str, Any],
    resume: bool,
) -> tuple[str, ...]:
    """Build the exact pinned Prob4D command for one admitted causal prefix."""

    sources = cast(Mapping[str, Mapping[str, str]], model_binding["sources"])
    prefix = cast(Sequence[int], job["prefix_source_frame_range_half_open"])
    command = [
        str(executable),
        str(source_video_path),
        "--upstream-root",
        str(motioncrafter_root),
        "--output-dir",
        str(output_directory),
        "--model-type",
        "determ",
    ]
    for role, option in (
        ("unet", "--unet"),
        ("vae", "--vae"),
        ("image_vae", "--image-vae"),
        ("base_pipeline", "--base-pipeline"),
    ):
        command.extend(
            (
                f"{option}-path",
                sources[role]["repository"],
                f"{option}-revision",
                sources[role]["revision"],
            )
        )
    command.extend(
        (
            "--cache-dir",
            str(cache_directory),
            "--height",
            str(provider_lock.height),
            "--width",
            str(provider_lock.width),
            "--window-size",
            str(provider_lock.window_size),
            "--overlap",
            str(provider_lock.overlap),
            "--num-inference-steps",
            "5",
            "--guidance-scale",
            "1.0",
            "--decode-chunk-size",
            "25",
            "--seed",
            str(job["view_root_seed"]),
            "--seed-policy",
            "derived-per-call",
            "--frame-start",
            str(prefix[0]),
            "--frame-stop",
            str(prefix[1]),
            "--frame-stride",
            "1",
        )
    )
    if resume:
        command.append("--resume")
    return tuple(command)


def validate_deform360_motioncrafter_prediction_manifest(
    manifest: object,
    *,
    verification: Mapping[str, Any],
    job: Mapping[str, Any],
    provider_lock: Deform360VisualProviderLockV1,
    model_binding: Mapping[str, Any],
) -> dict[str, object]:
    """Validate the run identity and causal boundary after Prob4D hash verification."""

    if not isinstance(manifest, Mapping):
        raise ValueError("prediction manifest must be a JSON object")
    _require(manifest.get("format_version") == 1, "prediction format changed")
    integrity = manifest.get("artifact_integrity")
    _require(isinstance(integrity, Mapping), "prediction is not integrity bound")
    integrity = cast(Mapping[str, Any], integrity)
    _require(integrity.get("schema") == _MOTIONCRAFTER_INTEGRITY_SCHEMA, "integrity schema changed")
    run_spec = integrity.get("run_spec")
    _require(isinstance(run_spec, Mapping), "prediction run spec is missing")
    run_spec = cast(Mapping[str, Any], run_spec)
    _require(run_spec.get("schema") == _MOTIONCRAFTER_RUN_SPEC_SCHEMA, "run-spec schema changed")
    run_spec_sha = sha256_digest(
        integrity.get("run_spec_sha256"),
        name="run_spec_sha256",
    )
    _require(
        hashlib.sha256(_canonical_json(run_spec)).hexdigest() == run_spec_sha,
        "run-spec digest mismatch",
    )
    _require(
        verification.get("integrity_bound") is True
        and verification.get("hashes_verified") is True
        and verification.get("run_spec_sha256") == run_spec_sha,
        "Prob4D did not verify every prediction member",
    )
    member_count = _integer(
        verification.get("member_count"),
        name="member_count",
        minimum=1,
    )
    admitted_video = _file_record(job["source_video"], name="source video")
    video = run_spec.get("input_video")
    _require(
        isinstance(video, Mapping)
        and video.get("sha256") == admitted_video["sha256"]
        and video.get("bytes") == admitted_video["byte_count"],
        "prediction input video differs from the admission",
    )
    upstream = run_spec.get("motioncrafter_upstream")
    _require(
        isinstance(upstream, Mapping)
        and upstream.get("commit") == provider_lock.motioncrafter_revision
        and upstream.get("clean") is True
        and upstream.get("status_entry_count") == 0,
        "MotionCrafter checkout differs from the provider lock",
    )
    config = run_spec.get("inference_config")
    _require(isinstance(config, Mapping), "prediction inference config is missing")
    config = cast(Mapping[str, Any], config)
    model_set_id = cast(str, model_binding["model_set_id"])
    identity = f"{_MODEL_SET_SCHEMA}:{model_set_id}"
    prefix = cast(Sequence[int], job["prefix_source_frame_range_half_open"])
    expected = {
        "model_type": "determ",
        "unet_path": f"{identity}#unet",
        "vae_path": f"{identity}#geometry-motion-vae",
        "base_pipeline_path": f"{identity}#base-video-pipeline",
        "height": provider_lock.height,
        "width": provider_lock.width,
        "window_size": provider_lock.window_size,
        "overlap": provider_lock.overlap,
        "num_inference_steps": 5,
        "guidance_scale": 1.0,
        "decode_chunk_size": 25,
        "seed": job["view_root_seed"],
        "seed_policy": "derived-per-call",
        "low_memory_usage": False,
        "frame_start": prefix[0],
        "frame_stop": prefix[1],
        "frame_stride": 1,
        "model_source_schema": _MODEL_SET_SCHEMA,
        "model_source_set_sha256": model_set_id,
        "model_loader_module_sha256": model_binding["loader_sha256"],
        "model_loader_module_bytes": model_binding["loader_bytes"],
    }
    expected_fields = frozenset({*expected, "model_source_manifest_json"})
    _require(
        set(config) == expected_fields,
        "prediction inference-config fields changed",
    )
    changed = [field for field, value in expected.items() if config.get(field) != value]
    _require(not changed, f"prediction inference config changed: {changed}")
    try:
        model_manifest = json.loads(str(config.get("model_source_manifest_json", "")))
    except json.JSONDecodeError as error:
        raise ValueError("prediction model-source manifest is invalid") from error
    _require(model_manifest == model_binding["manifest"], "prediction model set changed")
    windows = manifest.get("overlap_windows")
    _require(isinstance(windows, list) and bool(windows), "prediction windows are missing")
    for index, window in enumerate(windows):
        _require(isinstance(window, Mapping), f"window {index} is not an object")
        start = _integer(window.get("start_frame"), name=f"window {index} start")
        stop = _integer(window.get("stop_frame"), name=f"window {index} stop", minimum=1)
        _require(prefix[0] <= start < stop <= prefix[1], "prediction contains a post-cutoff window")
    return {"run_spec_sha256": run_spec_sha, "member_count": member_count}


def _receipt_base(
    *,
    implementation_revision: str,
    admission: Mapping[str, Any],
    job: Mapping[str, Any],
    provider_lock: Deform360VisualProviderLockV1,
    command_id: str,
) -> dict[str, Any]:
    return {
        "implementation_revision": exact_revision(
            implementation_revision,
            name="implementation_revision",
        ),
        "admission_id": admission["admission_id"],
        "job_id": job["job_id"],
        "object_id": job["object_id"],
        "episode_id": job["episode_id"],
        "stratum": job["stratum"],
        "camera_id": job["camera_id"],
        "provider_revision": provider_lock.provider_revision,
        "motioncrafter_revision": provider_lock.motioncrafter_revision,
        "visual_provider_lock_id": provider_lock.artifact_id,
        "model_set_id": provider_lock.model_set_id,
        "command_id": command_id,
        "output_relative_directory": job["output_relative_directory"],
        "information_boundary": dict(PRODUCTION_INFORMATION_BOUNDARY),
    }


def build_deform360_calibration_visual_prediction_seal(
    *,
    implementation_revision: str,
    admission: Mapping[str, Any],
    job: Mapping[str, Any],
    provider_lock: Deform360VisualProviderLockV1,
    command_id: str,
    prediction_manifest: Mapping[str, object],
    run_spec_sha256: str,
    verified_member_count: int,
) -> dict[str, Any]:
    """Build one content-addressed seal before evaluation frames may be opened."""

    prefix = cast(Sequence[int], job["prefix_source_frame_range_half_open"])
    prediction = cast(Sequence[int], job["prediction_source_frame_range_half_open"])
    identity = {
        "schema": DEFORM360_CALIBRATION_VISUAL_PREDICTION_SEAL_SCHEMA,
        "schema_version": 1,
        "semantics": "integrity-bound-causal-prefix-motioncrafter-prediction-v1",
        **_receipt_base(
            implementation_revision=implementation_revision,
            admission=admission,
            job=job,
            provider_lock=provider_lock,
            command_id=command_id,
        ),
        "source_video": job["source_video"],
        "source_timestamps": job["source_timestamps"],
        "causal_prefix_frame_range_half_open": list(prefix),
        "reserved_evaluation_frame_range_half_open": [prefix[1], prediction[1]],
        "prediction_manifest": dict(prediction_manifest),
        "run_spec_sha256": run_spec_sha256,
        "verified_member_count": verified_member_count,
    }
    return validate_deform360_calibration_visual_prediction_seal(
        {**identity, "seal_id": content_id(identity)}
    )


def validate_deform360_calibration_visual_prediction_seal(
    value: object,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("prediction seal must be a JSON object")
    seal = cast(dict[str, Any], plain_json(value))
    require_exact_fields(seal, expected=_SEAL_FIELDS, name="prediction seal")
    _require(
        seal["schema"] == DEFORM360_CALIBRATION_VISUAL_PREDICTION_SEAL_SCHEMA
        and seal["schema_version"] == 1
        and seal["semantics"]
        == "integrity-bound-causal-prefix-motioncrafter-prediction-v1",
        "prediction seal contract changed",
    )
    for field in ("implementation_revision", "provider_revision", "motioncrafter_revision"):
        exact_revision(seal[field], name=field)
    for field in (
        "admission_id",
        "job_id",
        "visual_provider_lock_id",
        "model_set_id",
        "command_id",
        "run_spec_sha256",
    ):
        sha256_digest(seal[field], name=field)
    _file_record(seal["source_video"], name="source_video")
    _file_record(seal["source_timestamps"], name="source_timestamps")
    _file_record(seal["prediction_manifest"], name="prediction_manifest")
    prefix = seal["causal_prefix_frame_range_half_open"]
    evaluation = seal["reserved_evaluation_frame_range_half_open"]
    _require(
        isinstance(prefix, list)
        and isinstance(evaluation, list)
        and len(prefix) == len(evaluation) == 2
        and all(type(item) is int for item in [*prefix, *evaluation])
        and prefix[0] < prefix[1] == evaluation[0] < evaluation[1]
        and prefix[1] - prefix[0] == 58
        and evaluation[1] - evaluation[0] == 18,
        "prediction seal frame boundary changed",
    )
    _integer(seal["verified_member_count"], name="verified_member_count", minimum=1)
    canonical_relative_posix_path(
        seal["output_relative_directory"],
        name="output_relative_directory",
    )
    _require(
        seal["information_boundary"] == PRODUCTION_INFORMATION_BOUNDARY,
        "prediction seal information boundary changed",
    )
    declared = sha256_digest(seal["seal_id"], name="seal_id")
    _require(
        declared
        == content_id({key: item for key, item in seal.items() if key != "seal_id"}),
        "prediction seal ID mismatch",
    )
    return seal


def build_deform360_calibration_visual_technical_failure(
    *,
    implementation_revision: str,
    admission: Mapping[str, Any],
    job: Mapping[str, Any],
    provider_lock: Deform360VisualProviderLockV1,
    command_id: str,
    stage: str,
    return_code: int,
    detail: bytes,
    stdout: Mapping[str, object],
    stderr: Mapping[str, object],
) -> dict[str, Any]:
    """Build a retained technical failure without replacing the admitted job."""

    identity = {
        "schema": DEFORM360_CALIBRATION_VISUAL_TECHNICAL_FAILURE_SCHEMA,
        "schema_version": 1,
        **_receipt_base(
            implementation_revision=implementation_revision,
            admission=admission,
            job=job,
            provider_lock=provider_lock,
            command_id=command_id,
        ),
        "stage": _string(stage, name="stage"),
        "return_code": _integer(return_code, name="return_code"),
        "detail_sha256": hashlib.sha256(detail).hexdigest(),
        "stdout": dict(stdout),
        "stderr": dict(stderr),
    }
    return validate_deform360_calibration_visual_technical_failure(
        {**identity, "failure_id": content_id(identity)}
    )


def validate_deform360_calibration_visual_technical_failure(
    value: object,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("technical failure must be a JSON object")
    failure = cast(dict[str, Any], plain_json(value))
    require_exact_fields(failure, expected=_FAILURE_FIELDS, name="technical failure")
    _require(
        failure["schema"] == DEFORM360_CALIBRATION_VISUAL_TECHNICAL_FAILURE_SCHEMA
        and failure["schema_version"] == 1,
        "technical-failure contract changed",
    )
    for field in ("implementation_revision", "provider_revision", "motioncrafter_revision"):
        exact_revision(failure[field], name=field)
    for field in (
        "admission_id",
        "job_id",
        "visual_provider_lock_id",
        "model_set_id",
        "command_id",
        "detail_sha256",
    ):
        sha256_digest(failure[field], name=field)
    _string(failure["stage"], name="stage")
    _integer(failure["return_code"], name="return_code")
    _file_record(failure["stdout"], name="stdout", minimum_bytes=0)
    _file_record(failure["stderr"], name="stderr", minimum_bytes=0)
    canonical_relative_posix_path(
        failure["output_relative_directory"],
        name="output_relative_directory",
    )
    _require(
        failure["information_boundary"] == PRODUCTION_INFORMATION_BOUNDARY,
        "technical-failure information boundary changed",
    )
    declared = sha256_digest(failure["failure_id"], name="failure_id")
    _require(
        declared
        == content_id(
            {key: item for key, item in failure.items() if key != "failure_id"}
        ),
        "technical-failure ID mismatch",
    )
    return failure


def build_deform360_calibration_visual_production_result(
    *,
    implementation_revision: str,
    admission: Mapping[str, Any],
    provider_lock: Deform360VisualProviderLockV1,
    jobs: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    """Build complete sorted accounting over every admitted camera job."""

    rows = [plain_json(row) for row in jobs]
    admitted_jobs = admission.get("jobs")
    _require(isinstance(admitted_jobs, list), "admission jobs are missing")
    expected_roster = [
        (row["job_id"], row["object_id"], row["camera_id"])
        for row in admitted_jobs
    ]
    observed_roster = [
        (row.get("job_id"), row.get("object_id"), row.get("camera_id"))
        for row in rows
    ]
    _require(
        observed_roster == expected_roster,
        "production result does not account for the exact admitted job roster",
    )
    statuses = [row["status"] for row in rows]
    object_statuses: dict[str, list[str]] = {}
    for row in rows:
        object_statuses.setdefault(cast(str, row["object_id"]), []).append(
            cast(str, row["status"])
        )
    failed = statuses.count("technical-failure")
    identity = {
        "schema": DEFORM360_CALIBRATION_VISUAL_PRODUCTION_RESULT_SCHEMA,
        "schema_version": 1,
        "semantics": "complete-admitted-calibration-view-accounting-v1",
        "implementation_revision": exact_revision(
            implementation_revision,
            name="implementation_revision",
        ),
        "admission_id": admission["admission_id"],
        "visual_provider_lock_id": provider_lock.artifact_id,
        "provider_revision": provider_lock.provider_revision,
        "motioncrafter_revision": provider_lock.motioncrafter_revision,
        "model_set_id": provider_lock.model_set_id,
        "object_count": admission["object_count"],
        "camera_view_count": admission["camera_view_count"],
        "succeeded_job_count": statuses.count("succeeded"),
        "technical_failure_job_count": failed,
        "completely_succeeded_object_count": sum(
            all(status == "succeeded" for status in values)
            for values in object_statuses.values()
        ),
        "status": (
            "all-jobs-succeeded" if failed == 0 else "technical-failures-retained"
        ),
        "jobs": rows,
        "information_boundary": dict(PRODUCTION_INFORMATION_BOUNDARY),
    }
    return validate_deform360_calibration_visual_production_result(
        {**identity, "result_id": content_id(identity)}
    )


def validate_deform360_calibration_visual_production_result(
    value: object,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("visual production result must be a JSON object")
    result = cast(dict[str, Any], plain_json(value))
    require_exact_fields(result, expected=_RESULT_FIELDS, name="visual production result")
    _require(
        result["schema"] == DEFORM360_CALIBRATION_VISUAL_PRODUCTION_RESULT_SCHEMA
        and result["schema_version"] == 1
        and result["semantics"] == "complete-admitted-calibration-view-accounting-v1",
        "visual production result contract changed",
    )
    for field in ("implementation_revision", "provider_revision", "motioncrafter_revision"):
        exact_revision(result[field], name=field)
    for field in ("admission_id", "visual_provider_lock_id", "model_set_id"):
        sha256_digest(result[field], name=field)
    object_count = _integer(result["object_count"], name="object_count", minimum=1)
    camera_count = _integer(
        result["camera_view_count"],
        name="camera_view_count",
        minimum=1,
    )
    jobs = result["jobs"]
    _require(isinstance(jobs, list) and len(jobs) == camera_count, "result job count changed")
    statuses: list[str] = []
    ordering: list[tuple[str, str]] = []
    object_statuses: dict[str, list[str]] = {}
    for index, row in enumerate(jobs):
        _require(isinstance(row, Mapping), f"result job {index} is not an object")
        row = cast(Mapping[str, Any], row)
        require_exact_fields(row, expected=_RESULT_JOB_FIELDS, name=f"result job {index}")
        sha256_digest(row["job_id"], name=f"result job {index} ID")
        object_id = _string(row["object_id"], name=f"result job {index} object")
        camera_id = _string(row["camera_id"], name=f"result job {index} camera")
        status = _string(row["status"], name=f"result job {index} status")
        _require(status in {"succeeded", "technical-failure"}, "result status changed")
        _file_record(row["receipt"], name=f"result job {index} receipt")
        statuses.append(status)
        ordering.append((object_id, camera_id))
        object_statuses.setdefault(object_id, []).append(status)
    _require(
        ordering == sorted(ordering) and len(set(ordering)) == len(ordering),
        "jobs not sorted",
    )
    _require(len(object_statuses) == object_count, "result object count changed")
    succeeded = statuses.count("succeeded")
    failed = statuses.count("technical-failure")
    _require(result["succeeded_job_count"] == succeeded, "succeeded count changed")
    _require(result["technical_failure_job_count"] == failed, "failure count changed")
    complete = sum(
        all(status == "succeeded" for status in values)
        for values in object_statuses.values()
    )
    _require(
        result["completely_succeeded_object_count"] == complete,
        "object success count changed",
    )
    expected_status = "all-jobs-succeeded" if failed == 0 else "technical-failures-retained"
    _require(result["status"] == expected_status, "terminal status changed")
    _require(
        result["information_boundary"] == PRODUCTION_INFORMATION_BOUNDARY,
        "result information boundary changed",
    )
    declared = sha256_digest(result["result_id"], name="result_id")
    _require(
        declared
        == content_id({key: item for key, item in result.items() if key != "result_id"}),
        "result ID mismatch",
    )
    return result


__all__ = [
    "DEFORM360_CALIBRATION_VISUAL_PREDICTION_SEAL_SCHEMA",
    "DEFORM360_CALIBRATION_VISUAL_PRODUCTION_RESULT_SCHEMA",
    "DEFORM360_CALIBRATION_VISUAL_TECHNICAL_FAILURE_SCHEMA",
    "PRODUCTION_INFORMATION_BOUNDARY",
    "build_deform360_calibration_visual_command",
    "build_deform360_calibration_visual_prediction_seal",
    "build_deform360_calibration_visual_production_result",
    "build_deform360_calibration_visual_technical_failure",
    "deform360_calibration_visual_command_descriptor",
    "validate_deform360_calibration_visual_prediction_seal",
    "validate_deform360_calibration_visual_production_result",
    "validate_deform360_calibration_visual_technical_failure",
    "validate_deform360_motioncrafter_model_set_binding",
    "validate_deform360_motioncrafter_prediction_manifest",
]
