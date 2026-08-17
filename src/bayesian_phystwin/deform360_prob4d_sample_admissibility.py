"""Target-free sample-admissibility preflight for public Deform360 Prob4D."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

import numpy as np

from ._canonical_contracts import (
    canonical_relative_posix_path,
    genuine_boolean,
    genuine_integer,
    plain_json,
)
from ._portable_contracts import (
    content_id,
    exact_revision,
    nonempty_string,
    require_exact_fields,
    sha256_digest,
)
from .deform360_prob4d_sample_admissibility_contract import (
    SAMPLE_ADMISSIBLE_PLAN_SEMANTICS,
    SAMPLE_ADMISSIBLE_PLAN_VERSION,
    SAMPLE_SUPPORT_NEGATIVE_REASON,
    validate_deform360_prob4d_sample_admissibility_policy,
)
from .deform360_prob4d_sample_materializer import (
    METRIC_PREFIX_ARRAYS,
    PLAN_SCHEMA,
    _confined_file,
    _deterministic_rows,
    _load_json,
    _load_plan,
    _sha256_file,
    _spatial_clusters,
    _verify_record,
)

SAMPLE_ADMISSIBILITY_RESULT_SCHEMA: Final = (
    "bayesian-phystwin.deform360-prob4d-sample-admissibility"
)
SAMPLE_ADMISSIBILITY_RESULT_VERSION: Final = 1
SAMPLE_ADMISSIBILITY_RESULT_SEMANTICS: Final = (
    "target-free-per-window-provider-mask-and-released-metric-support-v1"
)
SAMPLE_ADMISSIBILITY_RESULT_FILENAME: Final = "sample-admissibility-result.json"
SAMPLE_ADMISSIBLE_PLAN_FILENAME: Final = "metric-prefix-plan-v3.json"

_WINDOW_FIELDS = frozenset(
    {
        "window_id",
        "start_frame",
        "stop_frame",
        "metric_gauge_correspondence_count",
        "metric_gauge_spatial_cluster_count",
        "held_prefix_point_row_count",
        "status",
    }
)
_JOB_FIELDS = frozenset(
    {
        "job_id",
        "object_id",
        "episode_id",
        "stratum",
        "camera_id",
        "status",
        "window_count",
        "windows",
        "failure_reason",
        "failure_detail_sha256",
    }
)
_RESULT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "result_id",
        "implementation_revision",
        "input_plan_id",
        "sample_admissibility_policy_id",
        "admitted_stream_count",
        "prior_excluded_stream_count",
        "candidate_stream_count",
        "admissible_stream_count",
        "support_negative_stream_count",
        "technical_failure_stream_count",
        "supported_object_count",
        "plan_emitted",
        "plan_id",
        "plan_file",
        "status",
        "jobs",
        "source_artifacts",
        "information_boundary",
        "claim_boundary",
    }
)
_PLAN_V3_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "plan_id",
        "protocol_id",
        "selection_file_sha256",
        "visual_provider_spec_file_sha256",
        "metric_prior_policy_file_sha256",
        "camera_eligibility_policy_file_sha256",
        "camera_eligibility_policy_id",
        "sample_admissibility_policy_file_sha256",
        "sample_admissibility_policy_id",
        "dataset_revision",
        "processing_revision",
        "prob4d_revision",
        "motioncrafter_revision",
        "visual_production_result_id",
        "cases",
        "excluded_streams",
        "information_boundary",
        "claim_boundary",
    }
)
_SOURCE_ARTIFACT_NAMES: Final = frozenset(
    {
        "metric-prefix-plan-v2.json",
        "selection.json",
        "visual-provider-spec.json",
        "metric-prior-policy.json",
        "camera-eligibility-policy.json",
        "sample-admissibility-policy.json",
    }
)
_EXCLUDED_STREAM_FIELDS: Final = frozenset(
    {"job_id", "object_id", "episode_id", "stratum", "camera_id", "reason"}
)
_BOUNDARY: Final = {
    "public_released_measurements_used": True,
    "prediction_support_masks_opened": True,
    "prediction_point_values_used": False,
    "prediction_residuals_used": False,
    "calibration_outcomes_used": False,
    "confirmation_payloads_opened": False,
    "target_outcomes_used": False,
    "future_frames_used": False,
    "replacement_allowed": False,
    "human_approval_required": False,
    "new_measurements_required": False,
}


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            plain_json(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class _SupportWindow:
    window_id: str
    frame_indices: np.ndarray
    valid_mask: np.ndarray


@dataclass(frozen=True)
class _MetricSupport:
    frame_indices: np.ndarray
    valid_mask: np.ndarray


def _load_metric_support(
    path: Path,
    *,
    causal_range: tuple[int, int],
    image_resolution: tuple[int, int],
) -> _MetricSupport:
    """Load only released support masks; point values stay unopened."""

    try:
        with np.load(path, allow_pickle=False) as archive:
            _require(
                set(archive.files) == METRIC_PREFIX_ARRAYS,
                "metric-prefix array names changed",
            )
            frame_indices = np.asarray(archive["frame_indices"])
            valid = np.asarray(archive["valid_mask"])
    except (OSError, ValueError) as error:
        raise ValueError("cannot load metric-prefix support") from error
    start, stop = causal_range
    expected_frames: np.ndarray = np.arange(start, stop, dtype=np.int64)
    _require(
        frame_indices.dtype.kind in "iu"
        and np.array_equal(frame_indices, expected_frames),
        "metric prefix does not contain exactly the causal frame range",
    )
    _require(
        valid.dtype.kind == "b"
        and valid.shape == (len(expected_frames), *image_resolution),
        "metric support mask does not match the causal image grid",
    )
    return _MetricSupport(
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
        valid_mask=np.asarray(valid, dtype=np.bool_),
    )


def _load_prediction_support_windows(
    *,
    api: Any,
    manifest_path: Path,
    causal_range: tuple[int, int],
    image_resolution: tuple[int, int],
) -> list[_SupportWindow]:
    verification = api.verify_motioncrafter_prediction_manifest(
        manifest_path,
        verify_hashes=True,
    )
    _require(
        verification.get("integrity_bound") is True
        and verification.get("hashes_verified") is True,
        "prediction integrity verification failed",
    )
    manifest = _load_json(manifest_path, name="prediction manifest")
    raw_windows = manifest.get("overlap_windows")
    _require(
        isinstance(raw_windows, list) and bool(raw_windows),
        "prediction support windows are missing",
    )
    start, stop = causal_range
    observed: set[str] = set()
    windows: list[_SupportWindow] = []
    for index, raw_window in enumerate(cast(list[object], raw_windows)):
        _require(
            isinstance(raw_window, Mapping),
            f"prediction support window {index} is invalid",
        )
        record = cast(Mapping[str, Any], raw_window)
        window_id = nonempty_string(
            record.get("window_id"), name=f"prediction support window {index} ID"
        )
        _require(window_id not in observed, "prediction support window ID is repeated")
        observed.add(window_id)
        relative = canonical_relative_posix_path(
            record.get("path"), name=f"prediction support window {index} path"
        )
        path = _confined_file(
            manifest_path.parent,
            relative,
            name=f"prediction support window {index} path",
        )
        try:
            with np.load(path, allow_pickle=False) as archive:
                _require(
                    {"frame_indices", "valid_mask"} <= set(archive.files),
                    "prediction support arrays are missing",
                )
                frames = np.asarray(archive["frame_indices"])
                valid = np.asarray(archive["valid_mask"])
        except (OSError, ValueError) as error:
            raise ValueError("cannot load prediction support archive") from error
        expected_start = genuine_integer(
            record.get("start_frame"), name="prediction support start", minimum=start
        )
        expected_stop = genuine_integer(
            record.get("stop_frame"), name="prediction support stop", minimum=1
        )
        _require(
            expected_start < expected_stop <= stop
            and frames.dtype.kind in "iu"
            and np.array_equal(
                frames,
                np.arange(expected_start, expected_stop, dtype=np.int64),
            ),
            "prediction support frame metadata changed",
        )
        _require(
            valid.dtype.kind == "b" and valid.shape == (len(frames), *image_resolution),
            "prediction support mask shape changed",
        )
        windows.append(
            _SupportWindow(
                window_id=window_id,
                frame_indices=np.asarray(frames, dtype=np.int64),
                valid_mask=np.asarray(valid, dtype=np.bool_),
            )
        )
    windows.sort(key=lambda item: (int(item.frame_indices[0]), item.window_id))
    _require(
        int(windows[0].frame_indices[0]) == start,
        "first prediction support window misses prefix start",
    )
    return windows


def _assess_windows(
    *,
    job_id: str,
    windows: Sequence[_SupportWindow],
    metric: _MetricSupport,
    policy: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    all_admissible = True
    cluster_size = cast(int, policy["covariance_cluster_size_pixels"])
    for window in windows:
        metric_indices: np.ndarray = window.frame_indices - int(metric.frame_indices[0])
        _require(
            np.all(
                (metric_indices >= 0) & (metric_indices < len(metric.frame_indices))
            ),
            "prediction support window leaves metric prefix",
        )
        first_metric_index = int(metric_indices[0])
        active = window.valid_mask[0] & metric.valid_mask[first_metric_index]
        active_rows, active_columns = np.nonzero(active)
        selected = _deterministic_rows(
            len(active_rows),
            cast(int, policy["maximum_metric_fit_correspondences"]),
            seed_text=f"{job_id}:{window.window_id}:metric",
        )
        active_rows = active_rows[selected]
        active_columns = active_columns[selected]
        cluster_count = int(
            np.unique(
                _spatial_clusters(
                    active_rows,
                    active_columns,
                    width=window.valid_mask.shape[2],
                    cluster_size=cluster_size,
                )
            ).size
        )
        held_count = 0
        for local_index, metric_index in enumerate(metric_indices[1:], start=1):
            held_count += int(
                np.count_nonzero(
                    window.valid_mask[local_index]
                    & metric.valid_mask[int(metric_index)]
                )
            )
        admissible = (
            len(active_rows)
            >= policy["minimum_metric_gauge_correspondences_per_window"]
            and cluster_count
            >= policy["minimum_metric_gauge_spatial_clusters_per_window"]
            and held_count >= policy["minimum_held_prefix_point_rows_per_window"]
        )
        all_admissible &= bool(admissible)
        rows.append(
            {
                "window_id": window.window_id,
                "start_frame": int(window.frame_indices[0]),
                "stop_frame": int(window.frame_indices[-1]) + 1,
                "metric_gauge_correspondence_count": len(active_rows),
                "metric_gauge_spatial_cluster_count": cluster_count,
                "held_prefix_point_row_count": held_count,
                "status": "admissible" if admissible else "support-negative",
            }
        )
    return rows, all_admissible


def _build_v3_plan(
    *,
    input_plan: Mapping[str, Any],
    policy_path: Path,
    policy: Mapping[str, Any],
    admissible_jobs: set[str],
    support_negative_jobs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    identity = {key: item for key, item in input_plan.items() if key != "plan_id"}
    identity["schema_version"] = SAMPLE_ADMISSIBLE_PLAN_VERSION
    identity["semantics"] = SAMPLE_ADMISSIBLE_PLAN_SEMANTICS
    identity["sample_admissibility_policy_file_sha256"] = _sha256_file(policy_path)
    identity["sample_admissibility_policy_id"] = policy["artifact_id"]
    cases: list[dict[str, Any]] = []
    for raw_case in cast(Sequence[Mapping[str, Any]], input_plan["cases"]):
        case = dict(raw_case)
        case["streams"] = [
            dict(stream)
            for stream in cast(Sequence[Mapping[str, Any]], raw_case["streams"])
            if stream["job_id"] in admissible_jobs
        ]
        cases.append(case)
    identity["cases"] = cases
    exclusions = [
        dict(row)
        for row in cast(Sequence[Mapping[str, Any]], input_plan["excluded_streams"])
    ]
    exclusions.extend(dict(row) for row in support_negative_jobs.values())
    identity["excluded_streams"] = sorted(
        exclusions,
        key=lambda row: (
            str(row["object_id"]),
            str(row["camera_id"]),
            str(row["job_id"]),
        ),
    )
    identity["claim_boundary"] = (
        "Target-free plan retaining only streams that satisfy the frozen per-window "
        "provider-mask and released-metric support requirements. Exclusions are "
        "retained without replacement. No prediction residual, calibration outcome, "
        "confirmation payload, target outcome, or future frame was used."
    )
    return cast(
        dict[str, Any],
        plain_json({**identity, "plan_id": content_id(identity)}),
    )


def materialize_deform360_prob4d_sample_admissibility(
    *,
    plan_path: str | Path,
    prediction_root: str | Path,
    metric_root: str | Path,
    selection_path: str | Path,
    visual_provider_spec_path: str | Path,
    metric_prior_policy_path: str | Path,
    camera_eligibility_policy_path: str | Path,
    sample_admissibility_policy_path: str | Path,
    expected_processing_revision: str,
    implementation_revision: str,
    api: Any,
    output_directory: str | Path,
) -> Mapping[str, Any]:
    """Publish a target-free v3 plan or a retained preflight terminal."""

    plan_source = Path(plan_path).resolve(strict=True)
    selection_source = Path(selection_path).resolve(strict=True)
    provider_source = Path(visual_provider_spec_path).resolve(strict=True)
    metric_policy_source = Path(metric_prior_policy_path).resolve(strict=True)
    camera_policy_source = Path(camera_eligibility_policy_path).resolve(strict=True)
    sample_policy_source = Path(sample_admissibility_policy_path).resolve(strict=True)
    input_plan = _load_plan(
        plan_source,
        selection_path=selection_source,
        visual_provider_spec_path=provider_source,
        metric_prior_policy_path=metric_policy_source,
        camera_eligibility_policy_path=camera_policy_source,
        sample_admissibility_policy_path=None,
    )
    _require(
        input_plan["schema_version"] == 2,
        "sample admissibility requires a visible-stream v2 input plan",
    )
    _require(
        input_plan["processing_revision"]
        == exact_revision(
            expected_processing_revision,
            name="expected_processing_revision",
        ),
        "sample admissibility processing revision changed",
    )
    policy = validate_deform360_prob4d_sample_admissibility_policy(
        _load_json(sample_policy_source, name="sample admissibility policy")
    )
    _require(
        policy["protocol_id"] == input_plan["protocol_id"],
        "sample admissibility policy uses a different protocol",
    )
    provider = _load_json(provider_source, name="visual provider specification")
    motioncrafter = cast(Mapping[str, Any], provider["motioncrafter"])
    image_resolution = (
        genuine_integer(
            motioncrafter["height"], name="MotionCrafter height", minimum=1
        ),
        genuine_integer(motioncrafter["width"], name="MotionCrafter width", minimum=1),
    )
    prediction_root_path = Path(prediction_root).resolve(strict=True)
    metric_root_path = Path(metric_root).resolve(strict=True)
    result_jobs: list[dict[str, Any]] = []
    admissible_jobs: set[str] = set()
    support_negative_jobs: dict[str, dict[str, Any]] = {}
    per_object_admissible: dict[str, int] = {}

    for raw_case in cast(Sequence[Mapping[str, Any]], input_plan["cases"]):
        object_id = cast(str, raw_case["object_id"])
        episode_id = cast(int, raw_case["episode_id"])
        stratum = cast(str, raw_case["stratum"])
        causal_range = cast(
            tuple[int, int], tuple(raw_case["causal_frame_range_half_open"])
        )
        per_object_admissible[object_id] = 0
        for raw_stream in cast(Sequence[Mapping[str, Any]], raw_case["streams"]):
            stream = cast(Mapping[str, Any], raw_stream)
            job_id = cast(str, stream["job_id"])
            camera_id = cast(str, stream["camera_id"])
            windows: list[dict[str, Any]] = []
            status = "admissible"
            failure_reason: str | None = None
            failure_detail_sha256: str | None = None
            try:
                manifest_path, _manifest_record = _verify_record(
                    prediction_root_path,
                    stream["prediction_manifest"],
                    name="prediction manifest",
                )
                metric_path, _metric_record = _verify_record(
                    metric_root_path,
                    stream["metric_prefix"],
                    name="metric prefix",
                )
                metric = _load_metric_support(
                    metric_path,
                    causal_range=causal_range,
                    image_resolution=image_resolution,
                )
                support_windows = _load_prediction_support_windows(
                    api=api,
                    manifest_path=manifest_path,
                    causal_range=causal_range,
                    image_resolution=image_resolution,
                )
                windows, admissible = _assess_windows(
                    job_id=job_id,
                    windows=support_windows,
                    metric=metric,
                    policy=policy,
                )
                if not admissible:
                    status = "support-negative"
                    failure_reason = SAMPLE_SUPPORT_NEGATIVE_REASON
            except ValueError as error:
                status = "technical-failure"
                failure_reason = "sample-admissibility-preflight-failed"
                failure_detail_sha256 = hashlib.sha256(
                    json.dumps(
                        {"type": type(error).__name__, "detail": str(error)},
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
            if status == "admissible":
                admissible_jobs.add(job_id)
                per_object_admissible[object_id] += 1
            elif status == "support-negative":
                support_negative_jobs[job_id] = {
                    "job_id": job_id,
                    "object_id": object_id,
                    "episode_id": episode_id,
                    "stratum": stratum,
                    "camera_id": camera_id,
                    "reason": SAMPLE_SUPPORT_NEGATIVE_REASON,
                }
            result_jobs.append(
                {
                    "job_id": job_id,
                    "object_id": object_id,
                    "episode_id": episode_id,
                    "stratum": stratum,
                    "camera_id": camera_id,
                    "status": status,
                    "window_count": len(windows),
                    "windows": windows,
                    "failure_reason": failure_reason,
                    "failure_detail_sha256": failure_detail_sha256,
                }
            )

    prior_exclusions = cast(Sequence[Mapping[str, Any]], input_plan["excluded_streams"])
    candidate_count = len(result_jobs)
    admitted_count = candidate_count + len(prior_exclusions)
    technical_count = sum(row["status"] == "technical-failure" for row in result_jobs)
    support_negative_count = len(support_negative_jobs)
    supported_object_count = sum(
        count >= policy["minimum_supported_streams_per_object"]
        for count in per_object_admissible.values()
    )
    gate_passed = (
        technical_count == 0
        and supported_object_count >= policy["minimum_supported_object_count"]
        and all(
            count >= policy["minimum_supported_streams_per_object"]
            for count in per_object_admissible.values()
        )
        and len(admissible_jobs) / admitted_count
        >= policy["minimum_supported_stream_fraction"]
    )
    plan: dict[str, Any] | None = None
    plan_record: dict[str, object] | None = None
    target = Path(output_directory).resolve()
    _require(not os.path.lexists(target), "sample admissibility output already exists")
    temporary = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        copied_policy = temporary / "sample-admissibility-policy.json"
        shutil.copyfile(sample_policy_source, copied_policy)
        _require(
            _sha256_file(copied_policy) == _sha256_file(sample_policy_source),
            "copied sample admissibility policy changed",
        )
        if gate_passed:
            plan = _build_v3_plan(
                input_plan=input_plan,
                policy_path=sample_policy_source,
                policy=policy,
                admissible_jobs=admissible_jobs,
                support_negative_jobs=support_negative_jobs,
            )
            plan_path = temporary / SAMPLE_ADMISSIBLE_PLAN_FILENAME
            _write_json(plan_path, plan)
            plan_record = {
                "path": plan_path.name,
                "sha256": _sha256_file(plan_path),
                "byte_count": plan_path.stat().st_size,
            }
        status = (
            "technical-failures-retained"
            if technical_count
            else (
                "target-free-sample-admissibility-supported"
                if gate_passed
                else "sample-admissibility-gate-failed"
            )
        )
        identity = {
            "schema": SAMPLE_ADMISSIBILITY_RESULT_SCHEMA,
            "schema_version": SAMPLE_ADMISSIBILITY_RESULT_VERSION,
            "semantics": SAMPLE_ADMISSIBILITY_RESULT_SEMANTICS,
            "implementation_revision": exact_revision(
                implementation_revision, name="implementation_revision"
            ),
            "input_plan_id": input_plan["plan_id"],
            "sample_admissibility_policy_id": policy["artifact_id"],
            "admitted_stream_count": admitted_count,
            "prior_excluded_stream_count": len(prior_exclusions),
            "candidate_stream_count": candidate_count,
            "admissible_stream_count": len(admissible_jobs),
            "support_negative_stream_count": support_negative_count,
            "technical_failure_stream_count": technical_count,
            "supported_object_count": supported_object_count,
            "plan_emitted": plan is not None,
            "plan_id": None if plan is None else plan["plan_id"],
            "plan_file": plan_record,
            "status": status,
            "jobs": sorted(
                result_jobs,
                key=lambda row: (
                    str(row["object_id"]),
                    str(row["camera_id"]),
                    str(row["job_id"]),
                ),
            ),
            "source_artifacts": {
                "metric-prefix-plan-v2.json": _sha256_file(plan_source),
                "selection.json": _sha256_file(selection_source),
                "visual-provider-spec.json": _sha256_file(provider_source),
                "metric-prior-policy.json": _sha256_file(metric_policy_source),
                "camera-eligibility-policy.json": _sha256_file(camera_policy_source),
                "sample-admissibility-policy.json": _sha256_file(sample_policy_source),
            },
            "information_boundary": dict(_BOUNDARY),
            "claim_boundary": (
                "Target-free structural preflight only. It does not score a "
                "prediction residual, fit calibration, authorize confirmation, or "
                "establish prediction benefit or state of the art."
            ),
        }
        result = {**identity, "result_id": content_id(identity)}
        _write_json(temporary / SAMPLE_ADMISSIBILITY_RESULT_FILENAME, result)
        checksum_paths = sorted(
            path
            for path in temporary.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS"
        )
        (temporary / "SHA256SUMS").write_text(
            "".join(
                f"{_sha256_file(path)}  {path.relative_to(temporary).as_posix()}\n"
                for path in checksum_paths
            ),
            encoding="ascii",
        )
        os.rename(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return cast(Mapping[str, Any], plain_json(result))


def validate_deform360_prob4d_sample_admissibility_result(
    root: str | Path,
) -> dict[str, Any]:
    """Validate one published target-free sample-admissibility artifact."""

    source = Path(root).resolve(strict=True)
    _require(source.is_dir(), "sample admissibility root is not a directory")
    _require(
        not any(path.is_symlink() for path in source.rglob("*")),
        "sample admissibility artifact contains a symbolic link",
    )
    checksum_path = source / "SHA256SUMS"
    lines = checksum_path.read_text(encoding="ascii").splitlines()
    observed_paths: set[str] = set()
    for line in lines:
        digest, separator, relative = line.partition("  ")
        _require(bool(separator), "sample admissibility checksum is malformed")
        sha256_digest(digest, name="sample admissibility checksum")
        path = PurePosixPath(relative)
        _require(
            not path.is_absolute() and ".." not in path.parts,
            "sample admissibility checksum path is unsafe",
        )
        _require(
            relative not in observed_paths, "sample admissibility checksum repeats"
        )
        observed_paths.add(relative)
        _require(
            _sha256_file(source / path) == digest,
            "sample admissibility artifact checksum changed",
        )
    actual_paths = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    _require(
        observed_paths == actual_paths,
        "sample admissibility checksum coverage changed",
    )
    result = _load_json(
        source / SAMPLE_ADMISSIBILITY_RESULT_FILENAME,
        name="sample admissibility result",
    )
    require_exact_fields(
        result, expected=_RESULT_FIELDS, name="sample admissibility result"
    )
    supplied = sha256_digest(result["result_id"], name="result_id")
    _require(
        content_id({key: item for key, item in result.items() if key != "result_id"})
        == supplied,
        "sample admissibility result ID changed",
    )
    _require(
        result["schema"] == SAMPLE_ADMISSIBILITY_RESULT_SCHEMA
        and result["schema_version"] == SAMPLE_ADMISSIBILITY_RESULT_VERSION
        and result["semantics"] == SAMPLE_ADMISSIBILITY_RESULT_SEMANTICS
        and result["information_boundary"] == _BOUNDARY,
        "unsupported sample admissibility result",
    )
    exact_revision(result["implementation_revision"], name="implementation_revision")
    sha256_digest(result["input_plan_id"], name="input_plan_id")
    sample_policy_id = sha256_digest(
        result["sample_admissibility_policy_id"],
        name="sample_admissibility_policy_id",
    )
    counts = {
        field: genuine_integer(result[field], name=field, minimum=0)
        for field in (
            "admitted_stream_count",
            "prior_excluded_stream_count",
            "candidate_stream_count",
            "admissible_stream_count",
            "support_negative_stream_count",
            "technical_failure_stream_count",
            "supported_object_count",
        )
    }
    _require(
        counts["admitted_stream_count"]
        == counts["prior_excluded_stream_count"] + counts["candidate_stream_count"],
        "sample admissibility admitted-stream accounting changed",
    )
    artifacts = result["source_artifacts"]
    _require(
        isinstance(artifacts, Mapping) and set(artifacts) == _SOURCE_ARTIFACT_NAMES,
        "sample admissibility source-artifact roster changed",
    )
    normalized_artifacts = {
        str(name): sha256_digest(digest, name=f"source artifact {name}")
        for name, digest in artifacts.items()
    }
    policy_path = source / "sample-admissibility-policy.json"
    _require(
        _sha256_file(policy_path)
        == normalized_artifacts["sample-admissibility-policy.json"],
        "sample admissibility policy bytes changed",
    )
    policy = validate_deform360_prob4d_sample_admissibility_policy(
        _load_json(policy_path, name="sample admissibility policy")
    )
    _require(
        policy["artifact_id"] == sample_policy_id,
        "sample admissibility result uses a different policy",
    )
    jobs = result["jobs"]
    _require(isinstance(jobs, list), "sample admissibility jobs are invalid")
    _require(
        len(jobs) == counts["candidate_stream_count"],
        "sample admissibility candidate count changed",
    )
    job_order: list[tuple[str, str, str]] = []
    job_ids: set[str] = set()
    status_counts = {"admissible": 0, "support-negative": 0, "technical-failure": 0}
    for index, raw_job in enumerate(cast(list[object], jobs)):
        _require(
            isinstance(raw_job, Mapping), f"sample admissibility job {index} is invalid"
        )
        job = cast(Mapping[str, Any], raw_job)
        require_exact_fields(
            job,
            expected=_JOB_FIELDS,
            name=f"sample admissibility job {index}",
        )
        job_id = sha256_digest(job["job_id"], name=f"job {index} ID")
        _require(job_id not in job_ids, "sample admissibility job is repeated")
        job_ids.add(job_id)
        object_id = nonempty_string(job["object_id"], name=f"job {index} object")
        camera_id = nonempty_string(job["camera_id"], name=f"job {index} camera")
        genuine_integer(job["episode_id"], name=f"job {index} episode", minimum=0)
        nonempty_string(job["stratum"], name=f"job {index} stratum")
        job_order.append((object_id, camera_id, job_id))
        status = nonempty_string(job["status"], name=f"job {index} status")
        _require(status in status_counts, "sample admissibility job status changed")
        status_counts[status] += 1
        windows = job["windows"]
        _require(isinstance(windows, list), "sample admissibility windows are invalid")
        _require(
            genuine_integer(job["window_count"], name="window_count", minimum=0)
            == len(windows),
            "sample admissibility window count changed",
        )
        window_statuses: list[str] = []
        window_ids: set[str] = set()
        for raw_window in cast(Sequence[Mapping[str, Any]], windows):
            require_exact_fields(
                raw_window,
                expected=_WINDOW_FIELDS,
                name="sample admissibility window",
            )
            window_id = nonempty_string(raw_window["window_id"], name="window_id")
            _require(window_id not in window_ids, "sample admissibility window repeats")
            window_ids.add(window_id)
            start = genuine_integer(
                raw_window["start_frame"], name="start_frame", minimum=0
            )
            stop = genuine_integer(
                raw_window["stop_frame"], name="stop_frame", minimum=1
            )
            _require(start < stop, "sample admissibility window is empty")
            for field in (
                "metric_gauge_correspondence_count",
                "metric_gauge_spatial_cluster_count",
                "held_prefix_point_row_count",
            ):
                genuine_integer(raw_window[field], name=field, minimum=0)
            window_status = nonempty_string(raw_window["status"], name="window status")
            _require(
                window_status in {"admissible", "support-negative"},
                "sample admissibility window status changed",
            )
            window_statuses.append(window_status)
        failure_reason = job["failure_reason"]
        failure_detail = job["failure_detail_sha256"]
        if status == "admissible":
            _require(
                bool(windows)
                and set(window_statuses) == {"admissible"}
                and failure_reason is None
                and failure_detail is None,
                "admissible job evidence changed",
            )
        elif status == "support-negative":
            _require(
                SAMPLE_SUPPORT_NEGATIVE_REASON == failure_reason
                and "support-negative" in window_statuses
                and failure_detail is None,
                "support-negative job evidence changed",
            )
        else:
            _require(
                failure_reason == "sample-admissibility-preflight-failed",
                "technical-failure reason changed",
            )
            sha256_digest(failure_detail, name="technical failure detail")
    _require(job_order == sorted(job_order), "sample admissibility jobs are not sorted")
    _require(
        status_counts["admissible"] == counts["admissible_stream_count"]
        and status_counts["support-negative"] == counts["support_negative_stream_count"]
        and status_counts["technical-failure"]
        == counts["technical_failure_stream_count"]
        and sum(status_counts.values()) == counts["candidate_stream_count"],
        "sample admissibility status accounting changed",
    )
    plan_emitted = genuine_boolean(result["plan_emitted"], name="plan_emitted")
    if plan_emitted:
        _require(
            result["status"] == "target-free-sample-admissibility-supported"
            and counts["technical_failure_stream_count"] == 0,
            "sample-admissible plan status changed",
        )
        plan_path = source / SAMPLE_ADMISSIBLE_PLAN_FILENAME
        plan = _load_json(plan_path, name="sample-admissible plan")
        require_exact_fields(
            plan, expected=_PLAN_V3_FIELDS, name="sample-admissible plan"
        )
        supplied_plan_id = sha256_digest(plan["plan_id"], name="plan_id")
        _require(
            plan["schema"] == PLAN_SCHEMA
            and plan["schema_version"] == SAMPLE_ADMISSIBLE_PLAN_VERSION
            and plan["semantics"] == SAMPLE_ADMISSIBLE_PLAN_SEMANTICS
            and content_id(
                {key: item for key, item in plan.items() if key != "plan_id"}
            )
            == supplied_plan_id
            == result["plan_id"],
            "sample-admissible plan changed",
        )
        plan_file = result["plan_file"]
        _require(
            isinstance(plan_file, Mapping), "sample-admissible plan record is invalid"
        )
        require_exact_fields(
            plan_file,
            expected=frozenset({"path", "sha256", "byte_count"}),
            name="plan file",
        )
        _require(
            plan_file["path"] == SAMPLE_ADMISSIBLE_PLAN_FILENAME
            and sha256_digest(plan_file["sha256"], name="plan file sha256")
            == _sha256_file(plan_path)
            and genuine_integer(
                plan_file["byte_count"], name="plan byte_count", minimum=1
            )
            == plan_path.stat().st_size,
            "sample-admissible plan record changed",
        )
        _require(
            plan["sample_admissibility_policy_id"] == sample_policy_id
            and plan["sample_admissibility_policy_file_sha256"]
            == normalized_artifacts["sample-admissibility-policy.json"],
            "sample-admissible plan policy binding changed",
        )
        cases = plan["cases"]
        exclusions = plan["excluded_streams"]
        _require(
            isinstance(cases, list) and isinstance(exclusions, list),
            "sample-admissible plan roster is invalid",
        )
        included_ids: set[str] = set()
        for raw_case in cast(Sequence[Mapping[str, Any]], cases):
            streams = raw_case.get("streams")
            _require(isinstance(streams, list), "sample-admissible streams are invalid")
            for stream in cast(Sequence[Mapping[str, Any]], streams):
                stream_id = sha256_digest(stream.get("job_id"), name="included job ID")
                _require(stream_id not in included_ids, "included job is repeated")
                included_ids.add(stream_id)
        support_negative_ids = {
            sha256_digest(job["job_id"], name="support-negative job ID")
            for job in cast(Sequence[Mapping[str, Any]], jobs)
            if job["status"] == "support-negative"
        }
        excluded_ids: set[str] = set()
        sample_excluded_ids: set[str] = set()
        for raw_exclusion in cast(Sequence[Mapping[str, Any]], exclusions):
            require_exact_fields(
                raw_exclusion,
                expected=_EXCLUDED_STREAM_FIELDS,
                name="sample-admissible exclusion",
            )
            exclusion_id = sha256_digest(
                raw_exclusion["job_id"], name="excluded job ID"
            )
            _require(exclusion_id not in excluded_ids, "excluded job is repeated")
            excluded_ids.add(exclusion_id)
            if raw_exclusion["reason"] == SAMPLE_SUPPORT_NEGATIVE_REASON:
                sample_excluded_ids.add(exclusion_id)
            else:
                _require(
                    raw_exclusion["reason"]
                    == "released-robot-geometry-outside-fixed-camera-prefix",
                    "sample-admissible exclusion reason changed",
                )
        _require(
            included_ids
            == {
                sha256_digest(job["job_id"], name="admissible job ID")
                for job in cast(Sequence[Mapping[str, Any]], jobs)
                if job["status"] == "admissible"
            }
            and sample_excluded_ids == support_negative_ids
            and len(excluded_ids)
            == counts["prior_excluded_stream_count"]
            + counts["support_negative_stream_count"],
            "sample-admissible roster accounting changed",
        )
    else:
        _require(
            result["plan_id"] is None
            and result["plan_file"] is None
            and SAMPLE_ADMISSIBLE_PLAN_FILENAME not in observed_paths,
            "failed sample-admissibility result contains a plan",
        )
        expected_status = (
            "technical-failures-retained"
            if counts["technical_failure_stream_count"]
            else "sample-admissibility-gate-failed"
        )
        _require(result["status"] == expected_status, "failed preflight status changed")
    return cast(dict[str, Any], plain_json(result))


__all__ = [
    "SAMPLE_ADMISSIBILITY_RESULT_FILENAME",
    "SAMPLE_ADMISSIBILITY_RESULT_SCHEMA",
    "SAMPLE_ADMISSIBILITY_RESULT_SEMANTICS",
    "SAMPLE_ADMISSIBILITY_RESULT_VERSION",
    "SAMPLE_ADMISSIBLE_PLAN_FILENAME",
    "materialize_deform360_prob4d_sample_admissibility",
    "validate_deform360_prob4d_sample_admissibility_result",
]
