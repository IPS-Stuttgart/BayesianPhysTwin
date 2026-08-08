"""Exact visual-provider jobs for the frozen Deform360 calibration cohort.

The manifest binds the successful ten-object source-preparation execution to all
aligned camera videos before MotionCrafter or Prob4D inference starts. It reads
only ordinary file metadata and hashes; no prediction value, geometry target,
confirmation payload, or target outcome is admitted here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final, Literal, cast

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    plain_json,
)
from ._portable_contracts import (
    canonical_json_bytes,
    content_id,
    exact_revision,
    load_strict_json_object,
    nonempty_string,
    require_exact_fields,
    sha256_digest,
    source_artifact_mapping,
    write_atomic_json,
)
from .deform360_bias_aware_prospective_staging import (
    PREDICTION_FRAME_COUNT,
    PREFIX_FRAME_COUNT,
    STAGING_FRAME_COUNT,
)
from .deform360_calibration_execution import (
    file_sha256,
    load_deform360_stage0_selection,
)
from .deform360_calibration_source_run_record import (
    load_deform360_calibration_source_run_record,
)
from .deform360_visual_provider_lock import (
    Deform360VisualProviderLockV1,
    load_deform360_visual_provider_lock,
)

DEFORM360_CALIBRATION_VISUAL_JOB_SCHEMA: Final = (
    "bayesian-phystwin.deform360-calibration-visual-jobs"
)
DEFORM360_CALIBRATION_VISUAL_JOB_VERSION: Final = 1
DEFORM360_CALIBRATION_VISUAL_JOB_SEMANTICS: Final = (
    "all-aligned-camera-causal-prefix-provider-jobs-v1"
)
DEFORM360_CALIBRATION_VISUAL_JOB_PROTOCOL_ID: Final = (
    "deform360-official-hub-visuotactile-v1"
)
DEFORM360_OBJECT_SEED_SCHEMA: Final = (
    "bayesian-phystwin.deform360-per-object-derived-seed-v1"
)
PROB4D_MOTIONCRAFTER_SEED_SCHEMA: Final = (
    "prob4d.motioncrafter-seed-schedule.v1"
)
PROB4D_MOTIONCRAFTER_SEED_POLICY: Final = "derived-per-call"
MINIMUM_SUPPORTED_OBJECTS: Final = 8
MINIMUM_SUPPORTED_PER_STRATUM: Final = 4
MINIMUM_ALIGNED_CAMERAS: Final = 8

DEFORM360_CALIBRATION_VISUAL_JOB_CLAIM_BOUNDARY: Final = (
    "Calibration-only provider-job provenance. A valid manifest establishes "
    "exact source bytes, causal frame limits, stochastic scheduling, and object "
    "accounting; it does not establish observation quality, physical-query "
    "benefit, tactile benefit, calibrated uncertainty, Causal4D benefit, "
    "deployment safety, or state of the art."
)

ObjectStatus = Literal["planned", "technical_failure_without_replacement"]
Stratum = Literal["sheet", "volumetric"]

_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "manifest_id",
        "protocol_id",
        "status",
        "selection_artifact_sha256",
        "visual_provider_lock_id",
        "calibration_source_run_record_sha256",
        "calibration_source_result_sha256",
        "calibration_source_revision",
        "implementation_revision",
        "provider",
        "staging",
        "objects",
        "support_gate",
        "source_artifacts",
        "information_boundary",
        "claim_boundary",
    }
)
_OBJECT_FIELDS = frozenset(
    {
        "object_id",
        "episode_id",
        "stratum",
        "status",
        "aligned_frame_count",
        "action_window",
        "object_seed",
        "jobs",
        "failure_reason",
    }
)
_JOB_FIELDS = frozenset(
    {
        "job_id",
        "camera",
        "source_video",
        "source_frame_start",
        "source_frame_stop_exclusive",
        "source_frame_count",
        "evaluation_frame_start",
        "evaluation_frame_stop_exclusive",
        "overlap_windows",
        "stochastic_seed_schedule",
        "output_relative_path",
    }
)
_VIDEO_FIELDS = frozenset({"path", "file_sha256", "bytes"})
_WINDOW_FIELDS = frozenset(
    {
        "window_id",
        "source_frame_start",
        "source_frame_stop_exclusive",
    }
)
_CALL_FIELDS = frozenset(
    {
        "call_id",
        "product",
        "effective_seed",
        "window_id",
        "source_frame_start",
        "source_frame_stop_exclusive",
    }
)
_SUPPORT_FIELDS = frozenset(
    {
        "minimum_supported_objects",
        "minimum_supported_objects_per_stratum",
        "planned_object_count",
        "technical_failure_count",
        "planned_by_stratum",
        "technical_failures_retained_without_replacement",
        "support_passed",
    }
)
_BOUNDARY = {
    "calibration_camera_payload_bytes_hashed": True,
    "camera_frames_decoded": False,
    "prediction_outputs_opened": False,
    "calibration_target_metrics_computed": False,
    "confirmation_payloads_opened": False,
    "target_outcomes_used": False,
    "replacement_allowed": False,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _literal(value: object, *, name: str) -> str:
    result = nonempty_string(value, name=name)
    if result != result.strip():
        raise ValueError(f"{name} must not contain surrounding whitespace")
    return result


def _stratum(value: object) -> Stratum:
    if type(value) is not str or value not in {"sheet", "volumetric"}:
        raise ValueError("stratum must be sheet or volumetric")
    return cast(Stratum, value)


def _status(value: object) -> ObjectStatus:
    if type(value) is not str or value not in {
        "planned",
        "technical_failure_without_replacement",
    }:
        raise ValueError("visual-job object status is unsupported")
    return cast(ObjectStatus, value)


def _safe_segment(value: object, *, name: str) -> str:
    result = _literal(value, name=name)
    pure = PurePosixPath(result)
    if pure.as_posix() != result or len(pure.parts) != 1 or result in {".", ".."}:
        raise ValueError(f"{name} must be one safe POSIX path segment")
    return result


def _safe_relative(value: object, *, name: str) -> str:
    result = _literal(value, name=name)
    pure = PurePosixPath(result)
    if (
        pure.is_absolute()
        or pure.as_posix() != result
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"{name} must be a safe relative POSIX path")
    return result


def _ordinary_file(path: str | Path, *, name: str) -> Path:
    absolute = Path(path).absolute()
    if any(candidate.is_symlink() for candidate in (absolute, *absolute.parents)):
        raise ValueError(f"{name} path must not contain symlinks")
    try:
        result = absolute.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{name} does not exist") from error
    if not result.is_file():
        raise ValueError(f"{name} must be an ordinary file")
    return result


def _ordinary_directory(path: str | Path, *, name: str) -> Path:
    absolute = Path(path).absolute()
    if any(candidate.is_symlink() for candidate in (absolute, *absolute.parents)):
        raise ValueError(f"{name} path must not contain symlinks")
    try:
        result = absolute.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{name} does not exist") from error
    if not result.is_dir():
        raise ValueError(f"{name} must be an ordinary directory")
    return result


def _file_descriptor(path: Path, *, root: Path) -> dict[str, object]:
    resolved = _ordinary_file(path, name="source video")
    _require(resolved.is_relative_to(root), "source video escaped processed root")
    return {
        "path": resolved.relative_to(root).as_posix(),
        "file_sha256": file_sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def deform360_object_seed(root_seed: int, *, object_id: str) -> int:
    """Derive the exact object-level seed named by the provider lock."""

    seed = genuine_integer(root_seed, name="root_seed", minimum=0)
    if seed >= 2**32:
        raise ValueError("root_seed must lie in [0, 2**32)")
    identifier = _safe_segment(object_id, name="object_id")
    descriptor = {
        "schema": DEFORM360_OBJECT_SEED_SCHEMA,
        "root_seed": seed,
        "object_id": identifier,
    }
    return int.from_bytes(hashlib.sha256(canonical_json_bytes(descriptor)).digest()[:4], "big")


def prob4d_motioncrafter_seed(root_seed: int, *, call_id: str) -> int:
    """Mirror Prob4D@25d90ef's exact derived-per-call seed function."""

    seed = genuine_integer(root_seed, name="root_seed", minimum=0)
    if seed >= 2**32:
        raise ValueError("root_seed must lie in [0, 2**32)")
    identifier = _literal(call_id, name="call_id")
    descriptor = {
        "schema": PROB4D_MOTIONCRAFTER_SEED_SCHEMA,
        "root_seed": seed,
        "call_id": identifier,
    }
    return int.from_bytes(hashlib.sha256(canonical_json_bytes(descriptor)).digest()[:4], "big")


def _windows(
    *,
    frame_start: int,
    frame_count: int,
    window_size: int,
    overlap: int,
) -> list[dict[str, object]]:
    _require(frame_start >= 0, "frame_start must be nonnegative")
    _require(frame_count >= 1, "frame_count must be positive")
    _require(1 <= window_size <= frame_count, "window_size is incompatible")
    _require(0 <= overlap < window_size, "overlap is incompatible")
    stride = window_size - overlap
    starts = list(range(0, max(1, frame_count - window_size + 1), stride))
    final_start = max(0, frame_count - window_size)
    if final_start not in starts:
        starts.append(final_start)
    starts = sorted(set(starts))
    return [
        {
            "window_id": f"window_{index:04d}",
            "source_frame_start": frame_start + start,
            "source_frame_stop_exclusive": frame_start + start + window_size,
        }
        for index, start in enumerate(starts)
    ]


def _seed_schedule(
    *,
    object_seed: int,
    windows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = [
        {
            "call_id": "baseline-disjoint",
            "product": "disjoint_baseline",
            "window_id": None,
            "source_frame_start": None,
            "source_frame_stop_exclusive": None,
        },
        {
            "call_id": "baseline-latent-linear",
            "product": "latent_linear_baseline",
            "window_id": None,
            "source_frame_start": None,
            "source_frame_stop_exclusive": None,
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
        call["effective_seed"] = prob4d_motioncrafter_seed(
            object_seed,
            call_id=str(call["call_id"]),
        )
    _require(
        len({int(call["effective_seed"]) for call in calls}) == len(calls),
        "derived-per-call seed schedule contains a collision",
    )
    return calls


def _provider_record(lock: Deform360VisualProviderLockV1) -> dict[str, object]:
    return {
        "provider_repository": lock.provider_repository,
        "provider_revision": lock.provider_revision,
        "provider_api_version": lock.provider_api_version,
        "provider_manifest_id": lock.provider_manifest_id,
        "provider_attestation_sha256": lock.provider_attestation_sha256,
        "stream_contract_version": lock.stream_contract_version,
        "full_joint_gauge_covariance": lock.full_joint_gauge_covariance,
        "persistent_material_identities": lock.persistent_material_identities,
        "motioncrafter_repository": lock.motioncrafter_repository,
        "motioncrafter_revision": lock.motioncrafter_revision,
        "model_set_id": lock.model_set_id,
        "root_seed": lock.root_seed,
        "seed_policy": lock.seed_policy,
        "prob4d_seed_policy": PROB4D_MOTIONCRAFTER_SEED_POLICY,
        "window_size": lock.window_size,
        "overlap": lock.overlap,
        "height": lock.height,
        "width": lock.width,
        "storage_dtype": lock.storage_dtype,
        "initial_metric_frame_prior_id": lock.initial_metric_frame_prior_id,
        "additional_metric_anchor_policy": lock.additional_metric_anchor_policy,
        "max_gauge_rank": lock.max_gauge_rank,
        "minimum_retained_gauge_trace": lock.minimum_retained_gauge_trace,
    }


def _staging_record() -> dict[str, object]:
    return {
        "selected_frame_count": STAGING_FRAME_COUNT,
        "prediction_frame_count": PREDICTION_FRAME_COUNT,
        "causal_prefix_frame_count": PREFIX_FRAME_COUNT,
        "cutoff_convention": "exclusive",
        "provider_reads": "causal-prefix-only",
        "evaluation_future_reads": "forbidden-before-prediction-seal",
    }


def _action_window(value: object, *, aligned_frame_count: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("source result lacks action_window")
    result = plain_json(frozen_finite_json_mapping(value, name="action_window"))
    for key, count in (
        ("selected_raw_frame_range_half_open", STAGING_FRAME_COUNT),
        ("prediction_raw_frame_range_half_open", PREDICTION_FRAME_COUNT),
        ("prefix_raw_frame_range_half_open", PREFIX_FRAME_COUNT),
    ):
        raw = result.get(key)
        if not isinstance(raw, list) or len(raw) != 2:
            raise ValueError(f"action_window lacks {key}")
        start = genuine_integer(raw[0], name=f"{key}[0]", minimum=0)
        stop = genuine_integer(raw[1], name=f"{key}[1]", minimum=1)
        if stop - start != count or stop > aligned_frame_count:
            raise ValueError(f"action_window {key} changed")
    selected = result["selected_raw_frame_range_half_open"]
    prediction = result["prediction_raw_frame_range_half_open"]
    prefix = result["prefix_raw_frame_range_half_open"]
    if selected[0] != prediction[0] or selected[0] != prefix[0]:
        raise ValueError("action_window ranges do not share one start")
    if result.get("tactile_read") is not False:
        raise ValueError("action-window selection used tactile data")
    if result.get("object_geometry_read") is not False:
        raise ValueError("action-window selection used object geometry")
    if result.get("object_tracks_read") is not False:
        raise ValueError("action-window selection used object tracks")
    return cast(dict[str, Any], result)


def _support_gate(objects: Sequence[Mapping[str, object]]) -> dict[str, object]:
    planned_by_stratum = {
        stratum: sum(
            item["stratum"] == stratum and item["status"] == "planned"
            for item in objects
        )
        for stratum in ("sheet", "volumetric")
    }
    planned = sum(planned_by_stratum.values())
    passed = planned >= MINIMUM_SUPPORTED_OBJECTS and all(
        value >= MINIMUM_SUPPORTED_PER_STRATUM
        for value in planned_by_stratum.values()
    )
    return {
        "minimum_supported_objects": MINIMUM_SUPPORTED_OBJECTS,
        "minimum_supported_objects_per_stratum": MINIMUM_SUPPORTED_PER_STRATUM,
        "planned_object_count": planned,
        "technical_failure_count": len(objects) - planned,
        "planned_by_stratum": planned_by_stratum,
        "technical_failures_retained_without_replacement": True,
        "support_passed": passed,
    }


def _source_context(
    *,
    stage0_protocol_path: Path,
    selection_lock_path: Path,
    visual_provider_lock_path: Path,
    calibration_source_run_record_path: Path,
    calibration_source_result_path: Path,
) -> tuple[Any, Deform360VisualProviderLockV1, Mapping[str, Any], Mapping[str, Any]]:
    stage0 = _ordinary_file(stage0_protocol_path, name="Stage-0 protocol")
    selection_path = _ordinary_file(selection_lock_path, name="selection lock")
    provider_path = _ordinary_file(
        visual_provider_lock_path,
        name="visual-provider lock",
    )
    run_path = _ordinary_file(
        calibration_source_run_record_path,
        name="calibration-source run record",
    )
    result_path = _ordinary_file(
        calibration_source_result_path,
        name="calibration-source result",
    )
    selection = load_deform360_stage0_selection(
        selection_path,
        protocol_path=stage0,
    )
    provider = load_deform360_visual_provider_lock(provider_path)
    run = load_deform360_calibration_source_run_record(run_path)
    result = load_strict_json_object(
        result_path,
        label="calibration-source result",
    )
    if run.get("status") != "succeeded" or run.get("exit_code") != 0:
        raise ValueError("calibration-source run did not succeed")
    if run.get("confirmation_boundary_verified") is not True:
        raise ValueError("calibration-source confirmation boundary is unverified")
    if run.get("confirmation_payloads_opened") is not False:
        raise ValueError("calibration-source record reports confirmation access")
    gate = run.get("support_gate")
    if not isinstance(gate, Mapping) or gate.get("support_passed") is not True:
        raise ValueError("calibration-source support gate did not pass")
    bindings = (
        ("selection_lock_file_sha256", file_sha256(selection_path)),
        ("selection_artifact_sha256", selection.selection_artifact_sha256),
        ("visual_provider_lock_file_sha256", file_sha256(provider_path)),
        ("visual_provider_lock_id", provider.artifact_id),
        ("result_file_sha256", file_sha256(result_path)),
        ("result_sha256", result.get("result_sha256")),
    )
    for key, observed in bindings:
        if run.get(key) != observed:
            raise ValueError(f"calibration-source {key} changed")
    return selection, provider, run, result


def build_deform360_calibration_visual_job_manifest(
    *,
    stage0_protocol_path: str | Path,
    selection_lock_path: str | Path,
    visual_provider_lock_path: str | Path,
    calibration_source_run_record_path: str | Path,
    calibration_source_result_path: str | Path,
    processed_root: str | Path,
    implementation_revision: str,
) -> dict[str, object]:
    """Build all calibration-camera jobs from the exact successful source run."""

    selection, provider, run, result = _source_context(
        stage0_protocol_path=Path(stage0_protocol_path),
        selection_lock_path=Path(selection_lock_path),
        visual_provider_lock_path=Path(visual_provider_lock_path),
        calibration_source_run_record_path=Path(
            calibration_source_run_record_path
        ),
        calibration_source_result_path=Path(calibration_source_result_path),
    )
    if provider.seed_policy != "per-object-derived-seed-v1":
        raise ValueError("visual-provider seed policy changed")
    root = _ordinary_directory(processed_root, name="processed calibration root")
    implementation = exact_revision(
        implementation_revision,
        name="implementation_revision",
    )
    raw_rows = result.get("objects")
    if not isinstance(raw_rows, list):
        raise ValueError("calibration-source result lacks object rows")
    by_id = {
        row.get("object_id"): row
        for row in raw_rows
        if isinstance(row, Mapping) and type(row.get("object_id")) is str
    }
    expected = {unit.object_id: unit for unit in selection.calibration_units}
    if set(by_id) != set(expected) or len(raw_rows) != len(expected):
        raise ValueError("calibration-source result differs from Stage-0 cohort")

    objects: list[dict[str, object]] = []
    seeds: set[int] = set()
    for object_id in sorted(expected):
        unit = expected[object_id]
        row = cast(Mapping[str, Any], by_id[object_id])
        if row.get("episode_id") != unit.episode_id or row.get("stratum") != unit.stratum:
            raise ValueError("calibration-source object identity changed")
        if row.get("status") != "source_prepared":
            objects.append(
                {
                    "object_id": object_id,
                    "episode_id": unit.episode_id,
                    "stratum": unit.stratum,
                    "status": "technical_failure_without_replacement",
                    "aligned_frame_count": None,
                    "action_window": None,
                    "object_seed": None,
                    "jobs": [],
                    "failure_reason": _literal(
                        row.get("error", "source was not prepared"),
                        name="failure_reason",
                    ),
                }
            )
            continue
        aligned_count = genuine_integer(
            row.get("aligned_frame_count"),
            name="aligned_frame_count",
            minimum=STAGING_FRAME_COUNT,
        )
        action = _action_window(
            row.get("action_window"),
            aligned_frame_count=aligned_count,
        )
        cameras = row.get("cameras")
        if not isinstance(cameras, list):
            raise ValueError("source result lacks cameras")
        normalized_cameras = sorted(
            _safe_segment(camera, name="camera") for camera in cameras
        )
        if len(normalized_cameras) < MINIMUM_ALIGNED_CAMERAS:
            raise ValueError("source result has fewer than eight cameras")
        if len(set(normalized_cameras)) != len(normalized_cameras):
            raise ValueError("source result repeats a camera")
        if row.get("camera_count") != len(normalized_cameras):
            raise ValueError("source result camera_count changed")

        episode_root = root / object_id / "episode_0000"
        alignment_path = episode_root / "alignment.json"
        outputs = row.get("outputs_sha256")
        if not isinstance(outputs, Mapping):
            raise ValueError("source result lacks output identities")
        if file_sha256(_ordinary_file(alignment_path, name="alignment")) != outputs.get(
            "alignment"
        ):
            raise ValueError("processed alignment bytes changed")
        alignment = load_strict_json_object(alignment_path, label="alignment")
        if alignment.get("cameras") != cameras or alignment.get("frame_count") != (
            aligned_count
        ):
            raise ValueError("processed alignment differs from source result")

        prefix_start, prefix_stop = action["prefix_raw_frame_range_half_open"]
        prediction_start, prediction_stop = action[
            "prediction_raw_frame_range_half_open"
        ]
        windows = _windows(
            frame_start=prefix_start,
            frame_count=prefix_stop - prefix_start,
            window_size=provider.window_size,
            overlap=provider.overlap,
        )
        object_seed = deform360_object_seed(provider.root_seed, object_id=object_id)
        if object_seed in seeds:
            raise ValueError("per-object seed schedule contains a collision")
        seeds.add(object_seed)
        schedule = _seed_schedule(object_seed=object_seed, windows=windows)
        jobs: list[dict[str, object]] = []
        for camera in normalized_cameras:
            video = _file_descriptor(
                episode_root / camera / "undistorted.mp4",
                root=root,
            )
            descriptor: dict[str, object] = {
                "camera": camera,
                "source_video": video,
                "source_frame_start": prefix_start,
                "source_frame_stop_exclusive": prefix_stop,
                "source_frame_count": PREFIX_FRAME_COUNT,
                "evaluation_frame_start": prefix_stop,
                "evaluation_frame_stop_exclusive": prediction_stop,
                "overlap_windows": windows,
                "stochastic_seed_schedule": {
                    "schema": PROB4D_MOTIONCRAFTER_SEED_SCHEMA,
                    "policy": PROB4D_MOTIONCRAFTER_SEED_POLICY,
                    "root_seed": object_seed,
                    "calls": schedule,
                },
                "output_relative_path": (
                    f"{object_id}/episode_0000/{camera}"
                ),
            }
            jobs.append({"job_id": content_id(descriptor), **descriptor})
        objects.append(
            {
                "object_id": object_id,
                "episode_id": unit.episode_id,
                "stratum": unit.stratum,
                "status": "planned",
                "aligned_frame_count": aligned_count,
                "action_window": action,
                "object_seed": object_seed,
                "jobs": jobs,
                "failure_reason": None,
            }
        )

    support = _support_gate(objects)
    source_artifacts = {
        "sources/stage0/protocol.json": file_sha256(Path(stage0_protocol_path)),
        "sources/stage0/selection.json": file_sha256(Path(selection_lock_path)),
        "sources/locks/visual-provider-lock.json": file_sha256(
            Path(visual_provider_lock_path)
        ),
        "sources/calibration-source/execution-manifest.json": file_sha256(
            Path(calibration_source_run_record_path)
        ),
        "sources/calibration-source/result.json": file_sha256(
            Path(calibration_source_result_path)
        ),
        "sources/implementation/deform360_calibration_visual_jobs.py": file_sha256(
            Path(__file__)
        ),
    }
    identity: dict[str, object] = {
        "schema": DEFORM360_CALIBRATION_VISUAL_JOB_SCHEMA,
        "schema_version": DEFORM360_CALIBRATION_VISUAL_JOB_VERSION,
        "semantics": DEFORM360_CALIBRATION_VISUAL_JOB_SEMANTICS,
        "protocol_id": DEFORM360_CALIBRATION_VISUAL_JOB_PROTOCOL_ID,
        "status": (
            "locked-pre-provider-inference"
            if support["support_passed"] is True
            else "completed-insufficient-provider-job-support"
        ),
        "selection_artifact_sha256": selection.selection_artifact_sha256,
        "visual_provider_lock_id": provider.artifact_id,
        "calibration_source_run_record_sha256": run["record_sha256"],
        "calibration_source_result_sha256": result["result_sha256"],
        "calibration_source_revision": run["source_revision"],
        "implementation_revision": implementation,
        "provider": _provider_record(provider),
        "staging": _staging_record(),
        "objects": objects,
        "support_gate": support,
        "source_artifacts": source_artifacts,
        "information_boundary": dict(_BOUNDARY),
        "claim_boundary": DEFORM360_CALIBRATION_VISUAL_JOB_CLAIM_BOUNDARY,
    }
    return validate_deform360_calibration_visual_job_manifest(
        {**identity, "manifest_id": content_id(identity)}
    )


def _validate_video(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    require_exact_fields(value, expected=_VIDEO_FIELDS, name=name)
    size = genuine_integer(value["bytes"], name=f"{name}.bytes", minimum=1)
    return {
        "path": _safe_relative(value["path"], name=f"{name}.path"),
        "file_sha256": sha256_digest(
            value["file_sha256"],
            name=f"{name}.file_sha256",
        ),
        "bytes": size,
    }


def _validate_windows(value: object) -> list[dict[str, object]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("overlap_windows must be a sequence")
    result: list[dict[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError("overlap window must be a mapping")
        require_exact_fields(
            item,
            expected=_WINDOW_FIELDS,
            name=f"overlap_windows[{index}]",
        )
        result.append(
            {
                "window_id": _literal(
                    item["window_id"],
                    name=f"overlap_windows[{index}].window_id",
                ),
                "source_frame_start": genuine_integer(
                    item["source_frame_start"],
                    name="window start",
                    minimum=0,
                ),
                "source_frame_stop_exclusive": genuine_integer(
                    item["source_frame_stop_exclusive"],
                    name="window stop",
                    minimum=1,
                ),
            }
        )
    return result


def _validate_schedule(
    value: object,
    *,
    object_seed: int,
    windows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("stochastic_seed_schedule must be a mapping")
    expected_fields = frozenset({"schema", "policy", "root_seed", "calls"})
    require_exact_fields(value, expected=expected_fields, name="seed schedule")
    if value["schema"] != PROB4D_MOTIONCRAFTER_SEED_SCHEMA:
        raise ValueError("seed schedule schema changed")
    if value["policy"] != PROB4D_MOTIONCRAFTER_SEED_POLICY:
        raise ValueError("seed schedule policy changed")
    if value["root_seed"] != object_seed:
        raise ValueError("seed schedule object seed changed")
    calls = value["calls"]
    expected = _seed_schedule(object_seed=object_seed, windows=windows)
    if calls != expected:
        raise ValueError("seed schedule calls changed")
    return {
        "schema": PROB4D_MOTIONCRAFTER_SEED_SCHEMA,
        "policy": PROB4D_MOTIONCRAFTER_SEED_POLICY,
        "root_seed": object_seed,
        "calls": expected,
    }


def _validate_job(
    value: object,
    *,
    object_id: str,
    object_seed: int,
    provider: Mapping[str, object],
    action: Mapping[str, Any],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("visual job must be a mapping")
    require_exact_fields(value, expected=_JOB_FIELDS, name="visual job")
    camera = _safe_segment(value["camera"], name="camera")
    prefix_start, prefix_stop = action["prefix_raw_frame_range_half_open"]
    _prediction_start, prediction_stop = action[
        "prediction_raw_frame_range_half_open"
    ]
    if (
        value["source_frame_start"] != prefix_start
        or value["source_frame_stop_exclusive"] != prefix_stop
        or value["source_frame_count"] != PREFIX_FRAME_COUNT
        or value["evaluation_frame_start"] != prefix_stop
        or value["evaluation_frame_stop_exclusive"] != prediction_stop
    ):
        raise ValueError("visual job causal frame boundary changed")
    windows = _validate_windows(value["overlap_windows"])
    expected_windows = _windows(
        frame_start=prefix_start,
        frame_count=PREFIX_FRAME_COUNT,
        window_size=int(provider["window_size"]),
        overlap=int(provider["overlap"]),
    )
    if windows != expected_windows:
        raise ValueError("visual job overlap windows changed")
    schedule = _validate_schedule(
        value["stochastic_seed_schedule"],
        object_seed=object_seed,
        windows=windows,
    )
    expected_output = f"{object_id}/episode_0000/{camera}"
    if value["output_relative_path"] != expected_output:
        raise ValueError("visual job output path changed")
    descriptor: dict[str, object] = {
        "camera": camera,
        "source_video": _validate_video(value["source_video"], name="source_video"),
        "source_frame_start": prefix_start,
        "source_frame_stop_exclusive": prefix_stop,
        "source_frame_count": PREFIX_FRAME_COUNT,
        "evaluation_frame_start": prefix_stop,
        "evaluation_frame_stop_exclusive": prediction_stop,
        "overlap_windows": windows,
        "stochastic_seed_schedule": schedule,
        "output_relative_path": expected_output,
    }
    job_id = sha256_digest(value["job_id"], name="job_id")
    if job_id != content_id(descriptor):
        raise ValueError("visual job identity changed")
    return {"job_id": job_id, **descriptor}


def validate_deform360_calibration_visual_job_manifest(
    value: Mapping[str, Any],
) -> dict[str, object]:
    """Validate all derived fields in one portable job manifest."""

    require_exact_fields(value, expected=_MANIFEST_FIELDS, name="visual-job manifest")
    if value["schema"] != DEFORM360_CALIBRATION_VISUAL_JOB_SCHEMA:
        raise ValueError("visual-job manifest schema changed")
    version = genuine_integer(value["schema_version"], name="schema_version", minimum=1)
    if version != DEFORM360_CALIBRATION_VISUAL_JOB_VERSION:
        raise ValueError("visual-job manifest version changed")
    if value["semantics"] != DEFORM360_CALIBRATION_VISUAL_JOB_SEMANTICS:
        raise ValueError("visual-job manifest semantics changed")
    if value["protocol_id"] != DEFORM360_CALIBRATION_VISUAL_JOB_PROTOCOL_ID:
        raise ValueError("visual-job manifest protocol changed")
    if value["information_boundary"] != _BOUNDARY:
        raise ValueError("visual-job information boundary changed")
    if value["claim_boundary"] != DEFORM360_CALIBRATION_VISUAL_JOB_CLAIM_BOUNDARY:
        raise ValueError("visual-job claim boundary changed")
    provider = value["provider"]
    if not isinstance(provider, Mapping):
        raise ValueError("provider must be a mapping")
    provider_record = plain_json(
        frozen_finite_json_mapping(provider, name="provider")
    )
    if provider_record.get("seed_policy") != "per-object-derived-seed-v1":
        raise ValueError("provider seed policy changed")
    if provider_record.get("prob4d_seed_policy") != (
        PROB4D_MOTIONCRAFTER_SEED_POLICY
    ):
        raise ValueError("Prob4D seed policy changed")
    if value["staging"] != _staging_record():
        raise ValueError("visual-job staging contract changed")

    raw_objects = value["objects"]
    if isinstance(raw_objects, (str, bytes)) or not isinstance(
        raw_objects,
        Sequence,
    ):
        raise ValueError("objects must be a sequence")
    objects: list[dict[str, object]] = []
    seen_objects: set[str] = set()
    seen_jobs: set[str] = set()
    seen_seeds: set[int] = set()
    for index, raw in enumerate(raw_objects):
        if not isinstance(raw, Mapping):
            raise ValueError("visual-job object must be a mapping")
        require_exact_fields(raw, expected=_OBJECT_FIELDS, name=f"objects[{index}]")
        object_id = _safe_segment(raw["object_id"], name="object_id")
        if object_id in seen_objects:
            raise ValueError("visual-job manifest repeats an object")
        seen_objects.add(object_id)
        episode_id = genuine_integer(raw["episode_id"], name="episode_id", minimum=0)
        stratum = _stratum(raw["stratum"])
        status = _status(raw["status"])
        if status == "planned":
            aligned = genuine_integer(
                raw["aligned_frame_count"],
                name="aligned_frame_count",
                minimum=STAGING_FRAME_COUNT,
            )
            action = _action_window(raw["action_window"], aligned_frame_count=aligned)
            object_seed = genuine_integer(
                raw["object_seed"],
                name="object_seed",
                minimum=0,
            )
            expected_seed = deform360_object_seed(
                int(provider_record["root_seed"]),
                object_id=object_id,
            )
            if object_seed != expected_seed or object_seed in seen_seeds:
                raise ValueError("visual-job object seed changed or collided")
            seen_seeds.add(object_seed)
            raw_jobs = raw["jobs"]
            if isinstance(raw_jobs, (str, bytes)) or not isinstance(
                raw_jobs,
                Sequence,
            ):
                raise ValueError("planned object jobs must be a sequence")
            jobs = [
                _validate_job(
                    item,
                    object_id=object_id,
                    object_seed=object_seed,
                    provider=provider_record,
                    action=action,
                )
                for item in raw_jobs
            ]
            if len(jobs) < MINIMUM_ALIGNED_CAMERAS:
                raise ValueError("planned object has fewer than eight jobs")
            if [job["camera"] for job in jobs] != sorted(
                job["camera"] for job in jobs
            ):
                raise ValueError("planned object jobs are not camera-sorted")
            for job in jobs:
                job_id = str(job["job_id"])
                if job_id in seen_jobs:
                    raise ValueError("visual-job manifest repeats a job")
                seen_jobs.add(job_id)
            failure_reason = raw["failure_reason"]
            if failure_reason is not None:
                raise ValueError("planned object declares a failure reason")
        else:
            if any(
                raw[key] is not None
                for key in ("aligned_frame_count", "action_window", "object_seed")
            ) or raw["jobs"] != []:
                raise ValueError("technical failure carries provider jobs")
            aligned = None
            action = None
            object_seed = None
            jobs = []
            failure_reason = _literal(raw["failure_reason"], name="failure_reason")
        objects.append(
            {
                "object_id": object_id,
                "episode_id": episode_id,
                "stratum": stratum,
                "status": status,
                "aligned_frame_count": aligned,
                "action_window": action,
                "object_seed": object_seed,
                "jobs": jobs,
                "failure_reason": failure_reason,
            }
        )
    if len(objects) != 10 or len(seen_objects) != 10:
        raise ValueError("visual-job manifest must contain ten physical objects")
    if objects != sorted(objects, key=lambda item: str(item["object_id"])):
        raise ValueError("visual-job objects are not in canonical order")
    support = _support_gate(objects)
    if value["support_gate"] != support:
        raise ValueError("visual-job support gate changed")
    expected_status = (
        "locked-pre-provider-inference"
        if support["support_passed"] is True
        else "completed-insufficient-provider-job-support"
    )
    if value["status"] != expected_status:
        raise ValueError("visual-job manifest status changed")

    identity: dict[str, object] = {
        "schema": DEFORM360_CALIBRATION_VISUAL_JOB_SCHEMA,
        "schema_version": DEFORM360_CALIBRATION_VISUAL_JOB_VERSION,
        "semantics": DEFORM360_CALIBRATION_VISUAL_JOB_SEMANTICS,
        "protocol_id": DEFORM360_CALIBRATION_VISUAL_JOB_PROTOCOL_ID,
        "status": expected_status,
        "selection_artifact_sha256": sha256_digest(
            value["selection_artifact_sha256"],
            name="selection_artifact_sha256",
        ),
        "visual_provider_lock_id": sha256_digest(
            value["visual_provider_lock_id"],
            name="visual_provider_lock_id",
        ),
        "calibration_source_run_record_sha256": sha256_digest(
            value["calibration_source_run_record_sha256"],
            name="calibration_source_run_record_sha256",
        ),
        "calibration_source_result_sha256": sha256_digest(
            value["calibration_source_result_sha256"],
            name="calibration_source_result_sha256",
        ),
        "calibration_source_revision": exact_revision(
            value["calibration_source_revision"],
            name="calibration_source_revision",
        ),
        "implementation_revision": exact_revision(
            value["implementation_revision"],
            name="implementation_revision",
        ),
        "provider": provider_record,
        "staging": _staging_record(),
        "objects": objects,
        "support_gate": support,
        "source_artifacts": plain_json(
            source_artifact_mapping(
                cast(Mapping[str, str], value["source_artifacts"]),
                name="source_artifacts",
            )
        ),
        "information_boundary": dict(_BOUNDARY),
        "claim_boundary": DEFORM360_CALIBRATION_VISUAL_JOB_CLAIM_BOUNDARY,
    }
    manifest_id = sha256_digest(value["manifest_id"], name="manifest_id")
    if manifest_id != content_id(identity):
        raise ValueError("visual-job manifest identity changed")
    return {**identity, "manifest_id": manifest_id}


def load_deform360_calibration_visual_job_manifest(
    path: str | Path,
) -> dict[str, object]:
    value = load_strict_json_object(path, label="Deform360 calibration visual jobs")
    return validate_deform360_calibration_visual_job_manifest(value)


def save_deform360_calibration_visual_job_manifest(
    path: str | Path,
    value: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    validated = validate_deform360_calibration_visual_job_manifest(value)
    write_atomic_json(validated, path, overwrite=overwrite)


__all__ = [
    "DEFORM360_CALIBRATION_VISUAL_JOB_CLAIM_BOUNDARY",
    "DEFORM360_CALIBRATION_VISUAL_JOB_SCHEMA",
    "DEFORM360_CALIBRATION_VISUAL_JOB_SEMANTICS",
    "DEFORM360_CALIBRATION_VISUAL_JOB_VERSION",
    "build_deform360_calibration_visual_job_manifest",
    "deform360_object_seed",
    "load_deform360_calibration_visual_job_manifest",
    "prob4d_motioncrafter_seed",
    "save_deform360_calibration_visual_job_manifest",
    "validate_deform360_calibration_visual_job_manifest",
]
