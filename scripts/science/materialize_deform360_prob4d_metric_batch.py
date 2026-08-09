"""Batch released Deform360 robot gauges for source-only Prob4D calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

from bayesian_phystwin._canonical_contracts import (
    canonical_relative_posix_path,
    genuine_boolean,
    genuine_integer,
    plain_json,
)
from bayesian_phystwin._portable_contracts import (
    content_id,
    exact_revision,
    nonempty_string,
    require_exact_fields,
    sha256_digest,
)
from bayesian_phystwin.deform360_calibration_visual_execution_admission import (
    validate_deform360_prepared_source_inventory,
)
from bayesian_phystwin.deform360_calibration_visual_production import (
    validate_deform360_calibration_visual_prediction_seal,
    validate_deform360_calibration_visual_production_result,
)
from bayesian_phystwin.deform360_prob4d_sample_materializer import (
    PLAN_SCHEMA,
    PLAN_SEMANTICS,
    PLAN_VERSION,
)
from bayesian_phystwin.deform360_public_contact_prefix import (
    _ordinary_directory,
    _ordinary_file,
)
from bayesian_phystwin.deform360_robot_metric_prefix import (
    DEFORM360_ROBOT_METRIC_SOURCE_KIND,
    METRIC_CALIBRATION_FILENAME,
    METRIC_PREFIX_FILENAME,
    materialize_deform360_robot_metric_prefix,
    validate_deform360_robot_metric_prefix,
)

DEFORM360_PROB4D_METRIC_BATCH_SCHEMA: Final = (
    "bayesian-phystwin.deform360-prob4d-metric-batch"
)
DEFORM360_PROB4D_METRIC_BATCH_VERSION: Final = 1
DEFORM360_PROB4D_METRIC_BATCH_SEMANTICS: Final = (
    "all-sealed-calibration-streams-released-robot-gauge-v1"
)
METRIC_BATCH_RESULT_FILENAME: Final = "metric-batch-result.json"
METRIC_PREFIX_PLAN_FILENAME: Final = "metric-prefix-plan.json"
METRIC_DIRECTORY_NAME: Final = "metrics"
SUPPORT_NEGATIVE_DETAIL: Final = "released robot geometry is outside this camera prefix"

METRIC_BATCH_CLAIM_BOUNDARY: Final = (
    "Source-only materialization of released Deform360 robot geometry for every "
    "sealed calibration camera. This artifact does not use new capture, require "
    "human approval, open confirmation payloads or future frames, evaluate "
    "calibration or transfer, authorize confirmation, or establish state of the art."
)
METRIC_PLAN_CLAIM_BOUNDARY: Final = (
    "Source-only plan covering every successful frozen visual-production stream "
    "with causal released robot-gauge evidence. It does not establish calibrated "
    "uncertainty, transfer, confirmation benefit, or state of the art."
)

_FILE_FIELDS = frozenset({"path", "sha256", "byte_count"})
_RESULT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "result_id",
        "implementation_revision",
        "production_result_id",
        "admission_id",
        "object_count",
        "admitted_stream_count",
        "supported_stream_count",
        "support_negative_stream_count",
        "technical_failure_stream_count",
        "supported_object_count",
        "plan_emitted",
        "plan_file",
        "status",
        "jobs",
        "source_artifacts",
        "information_boundary",
        "claim_boundary",
    }
)
_JOB_FIELDS = frozenset(
    {
        "job_id",
        "object_id",
        "episode_id",
        "stratum",
        "camera_id",
        "output_relative_directory",
        "status",
        "metric_artifact_id",
        "projected_point_count",
        "failure_reason",
        "failure_detail_sha256",
    }
)
_BATCH_BOUNDARY_FIELDS = frozenset(
    {
        "calibration_robot_state_access_attempted",
        "calibration_robot_state_opened",
        "calibration_camera_calibration_opened",
        "calibration_camera_images_opened",
        "calibration_tactile_payloads_opened",
        "rendered_depth_opened",
        "confirmation_payloads_opened",
        "target_outcomes_used",
        "future_frames_used",
        "replacement_allowed",
        "human_approval_required",
    }
)
_PLAN_BOUNDARY: Final = {
    "calibration_payloads_opened": True,
    "confirmation_payloads_opened": False,
    "target_outcomes_used": False,
    "future_frames_used": False,
    "replacement_allowed": False,
}


class _InsufficientMultiviewSupport(ValueError):
    """Raised only when a frozen object has fewer than two supported streams."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        plain_json(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _load_json(path: str | Path, *, name: str) -> dict[str, Any]:
    source = _ordinary_file(path, name=name)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return cast(dict[str, Any], value)


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_json_bytes(value) + b"\n")


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return cast(Sequence[Any], value)


def _file_record(path: Path, *, root: Path) -> dict[str, object]:
    source = _ordinary_file(path, name="bound artifact")
    _require(root == source or root in source.parents, "bound artifact escapes root")
    return {
        "path": source.relative_to(root).as_posix(),
        "sha256": _sha256_file(source),
        "byte_count": source.stat().st_size,
    }


def _validate_file_record(value: object, *, name: str) -> dict[str, object]:
    record = _mapping(value, name=name)
    require_exact_fields(record, expected=_FILE_FIELDS, name=name)
    return {
        "path": canonical_relative_posix_path(record["path"], name=f"{name}.path"),
        "sha256": sha256_digest(record["sha256"], name=f"{name}.sha256"),
        "byte_count": genuine_integer(
            record["byte_count"], name=f"{name}.byte_count", minimum=1
        ),
    }


def _verify_file_record(
    root: Path, value: object, *, name: str
) -> tuple[Path, dict[str, object]]:
    record = _validate_file_record(value, name=name)
    relative = cast(str, record["path"])
    candidate = root / PurePosixPath(relative)
    path = _ordinary_file(candidate, name=name)
    _require(root == path or root in path.parents, f"{name} escapes its root")
    _require(path.stat().st_size == record["byte_count"], f"{name} byte count changed")
    _require(_sha256_file(path) == record["sha256"], f"{name} SHA-256 changed")
    return path, record


def _source_selection_rows(selection: Mapping[str, Any]) -> set[tuple[str, int, str]]:
    selected = _mapping(selection.get("selection"), name="selection")
    calibration = _sequence(selected.get("calibration"), name="calibration selection")
    rows: set[tuple[str, int, str]] = set()
    for index, raw in enumerate(calibration):
        row = _mapping(raw, name=f"calibration selection {index}")
        identity = (
            nonempty_string(row.get("object_id"), name="selection object_id"),
            genuine_integer(row.get("episode_id"), name="selection episode_id"),
            nonempty_string(row.get("stratum"), name="selection stratum"),
        )
        _require(identity not in rows, "calibration selection repeats an object")
        rows.add(identity)
    return rows


def _inventory_rows(inventory: Mapping[str, Any]) -> set[tuple[str, int, str]]:
    rows: set[tuple[str, int, str]] = set()
    for index, raw in enumerate(
        _sequence(inventory.get("objects"), name="inventory objects")
    ):
        row = _mapping(raw, name=f"inventory object {index}")
        identity = (
            nonempty_string(row.get("object_id"), name="inventory object_id"),
            genuine_integer(row.get("episode_id"), name="inventory episode_id"),
            nonempty_string(row.get("stratum"), name="inventory stratum"),
        )
        _require(identity not in rows, "prepared inventory repeats an object")
        rows.add(identity)
    return rows


def _validate_inputs(
    *,
    prepared_source_inventory_path: Path,
    production_result_path: Path,
    selection_path: Path,
    visual_provider_spec_path: Path,
    metric_prior_policy_path: Path,
    expected_processing_revision: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    production = validate_deform360_calibration_visual_production_result(
        _load_json(production_result_path, name="visual production result")
    )
    _require(
        production["status"] == "all-jobs-succeeded"
        and production["technical_failure_job_count"] == 0
        and production["succeeded_job_count"] == production["camera_view_count"],
        "metric batch requires a completely successful visual production",
    )
    inventory = validate_deform360_prepared_source_inventory(
        _load_json(prepared_source_inventory_path, name="prepared-source inventory")
    )
    processing_revision = exact_revision(
        expected_processing_revision, name="expected_processing_revision"
    )
    _require(
        inventory["processing_revision"] == processing_revision,
        "prepared-source processing revision changed",
    )
    selection = _load_json(selection_path, name="selection")
    provider = _load_json(
        visual_provider_spec_path, name="visual provider specification"
    )
    policy = _load_json(metric_prior_policy_path, name="metric prior policy")
    protocol_id = nonempty_string(selection.get("protocol_id"), name="protocol_id")
    _require(
        provider.get("protocol_id") == protocol_id
        and policy.get("protocol_id") == protocol_id,
        "source protocol identity changed",
    )
    provider_spec = _mapping(provider.get("provider"), name="Prob4D provider")
    motioncrafter = _mapping(provider.get("motioncrafter"), name="MotionCrafter")
    _require(
        exact_revision(provider_spec.get("revision"), name="Prob4D revision")
        == production["provider_revision"],
        "Prob4D revision differs from visual production",
    )
    _require(
        exact_revision(motioncrafter.get("revision"), name="MotionCrafter revision")
        == production["motioncrafter_revision"],
        "MotionCrafter revision differs from visual production",
    )
    _require(
        policy.get("metric_source_kind") == DEFORM360_ROBOT_METRIC_SOURCE_KIND
        and policy.get("future_frames_used") is False
        and policy.get("confirmation_payloads_opened") is False
        and policy.get("target_outcomes_used") is False
        and policy.get("human_approval_required") is False,
        "robot metric-gauge policy changed",
    )
    selected_rows = _source_selection_rows(selection)
    inventory_rows = _inventory_rows(inventory)
    _require(selected_rows == inventory_rows, "selection and prepared inventory differ")
    _require(
        len(selected_rows) == production["object_count"],
        "production object count differs from selection",
    )
    return production, inventory, selection, provider, policy


def _load_prediction_seal(
    *,
    production_root: Path,
    prediction_root: Path,
    production: Mapping[str, Any],
    row: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, object]]:
    seal_path, _seal_record = _verify_file_record(
        production_root, row.get("receipt"), name="prediction seal"
    )
    seal = validate_deform360_calibration_visual_prediction_seal(
        _load_json(seal_path, name="prediction seal")
    )
    _require(
        seal["job_id"] == row.get("job_id")
        and seal["object_id"] == row.get("object_id")
        and seal["camera_id"] == row.get("camera_id"),
        "prediction seal identity differs from production result",
    )
    _require(
        seal["admission_id"] == production["admission_id"]
        and seal["implementation_revision"] == production["implementation_revision"]
        and seal["provider_revision"] == production["provider_revision"]
        and seal["motioncrafter_revision"] == production["motioncrafter_revision"]
        and seal["visual_provider_lock_id"] == production["visual_provider_lock_id"]
        and seal["model_set_id"] == production["model_set_id"],
        "prediction seal lineage differs from production result",
    )
    output_relative = canonical_relative_posix_path(
        seal["output_relative_directory"], name="prediction output directory"
    )
    sealed_manifest = _validate_file_record(
        seal["prediction_manifest"], name="sealed prediction manifest"
    )
    manifest_relative = (
        PurePosixPath(output_relative) / cast(str, sealed_manifest["path"])
    ).as_posix()
    manifest_path, verified = _verify_file_record(
        prediction_root,
        {**sealed_manifest, "path": manifest_relative},
        name="prediction manifest",
    )
    return seal, _file_record(manifest_path, root=prediction_root) | {
        "sha256": verified["sha256"],
        "byte_count": verified["byte_count"],
    }


def _metric_stream_records(
    *, metric_directory: Path, metric_root: Path
) -> tuple[dict[str, object], dict[str, object]]:
    return (
        _file_record(metric_directory / METRIC_PREFIX_FILENAME, root=metric_root),
        _file_record(metric_directory / METRIC_CALIBRATION_FILENAME, root=metric_root),
    )


def _build_plan(
    *,
    streams: Sequence[Mapping[str, Any]],
    production: Mapping[str, Any],
    selection: Mapping[str, Any],
    provider: Mapping[str, Any],
    selection_path: Path,
    visual_provider_spec_path: Path,
    metric_prior_policy_path: Path,
    processing_revision: str,
) -> dict[str, Any]:
    grouped: dict[tuple[str, int, str, tuple[int, int]], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for stream in streams:
        frame_range = cast(Sequence[int], stream["causal_frame_range_half_open"])
        key = (
            cast(str, stream["object_id"]),
            cast(int, stream["episode_id"]),
            cast(str, stream["stratum"]),
            (int(frame_range[0]), int(frame_range[1])),
        )
        grouped[key].append(
            {
                "job_id": stream["job_id"],
                "camera_id": stream["camera_id"],
                "prediction_manifest": stream["prediction_manifest"],
                "metric_prefix": stream["metric_prefix"],
                "metric_calibration": stream["metric_calibration"],
            }
        )
    cases: list[dict[str, Any]] = []
    for (object_id, episode_id, stratum, frame_range), case_streams in sorted(
        grouped.items()
    ):
        case_streams.sort(key=lambda item: (item["camera_id"], item["job_id"]))
        if len(case_streams) < 2:
            raise _InsufficientMultiviewSupport(
                "every calibration object requires at least two supported streams"
            )
        case_identity = {
            "schema": "bayesian-phystwin.deform360-prob4d-metric-case-id-v1",
            "object_id": object_id,
            "episode_id": episode_id,
        }
        cases.append(
            {
                "case_id": content_id(case_identity),
                "object_id": object_id,
                "episode_id": episode_id,
                "stratum": stratum,
                "causal_frame_range_half_open": list(frame_range),
                "streams": case_streams,
            }
        )
    dataset = _mapping(selection.get("dataset"), name="selection dataset")
    provider_spec = _mapping(provider.get("provider"), name="Prob4D provider")
    motioncrafter = _mapping(provider.get("motioncrafter"), name="MotionCrafter")
    identity: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "schema_version": PLAN_VERSION,
        "semantics": PLAN_SEMANTICS,
        "protocol_id": nonempty_string(
            selection.get("protocol_id"), name="protocol_id"
        ),
        "selection_file_sha256": _sha256_file(selection_path),
        "visual_provider_spec_file_sha256": _sha256_file(visual_provider_spec_path),
        "metric_prior_policy_file_sha256": _sha256_file(metric_prior_policy_path),
        "dataset_revision": exact_revision(
            dataset.get("resolved_revision"), name="dataset revision"
        ),
        "processing_revision": exact_revision(
            processing_revision, name="processing revision"
        ),
        "prob4d_revision": exact_revision(
            provider_spec.get("revision"), name="Prob4D revision"
        ),
        "motioncrafter_revision": exact_revision(
            motioncrafter.get("revision"), name="MotionCrafter revision"
        ),
        "visual_production_result_id": production["result_id"],
        "cases": cases,
        "information_boundary": dict(_PLAN_BOUNDARY),
        "claim_boundary": METRIC_PLAN_CLAIM_BOUNDARY,
    }
    return {**identity, "plan_id": content_id(identity)}


def _write_checksums(root: Path) -> None:
    paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [
        f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in paths
    ]
    (root / "SHA256SUMS").write_text("".join(lines), encoding="ascii")


def _validate_emitted_plan_binding(
    plan: Mapping[str, Any],
    *,
    result: Mapping[str, Any],
    jobs: Sequence[Any],
    source_artifacts: Mapping[str, Any],
    object_count: int,
) -> None:
    """Bind an emitted plan to the exact batch lineage and supported job roster."""

    _require(
        plan.get("visual_production_result_id") == result["production_result_id"],
        "metric-prefix plan uses a different production result",
    )
    _require(
        plan.get("selection_file_sha256") == source_artifacts["selection.json"]
        and plan.get("visual_provider_spec_file_sha256")
        == source_artifacts["visual-provider-spec.json"]
        and plan.get("metric_prior_policy_file_sha256")
        == source_artifacts["metric-prior-policy.json"],
        "metric-prefix plan source artifacts differ from batch",
    )
    _require(
        plan.get("claim_boundary") == METRIC_PLAN_CLAIM_BOUNDARY,
        "metric-prefix plan claim boundary changed",
    )

    expected_jobs: dict[str, tuple[str, int, str, str]] = {}
    for index, raw in enumerate(jobs):
        row = _mapping(raw, name=f"metric batch job {index}")
        if row["status"] != "supported":
            continue
        job_id = sha256_digest(row["job_id"], name="job_id")
        identity = (
            nonempty_string(row["object_id"], name="job object_id"),
            genuine_integer(row["episode_id"], name="job episode_id", minimum=0),
            nonempty_string(row["stratum"], name="job stratum"),
            nonempty_string(row["camera_id"], name="job camera_id"),
        )
        _require(job_id not in expected_jobs, "metric batch repeats a job ID")
        expected_jobs[job_id] = identity

    cases = _sequence(plan.get("cases"), name="metric-prefix plan cases")
    _require(len(cases) == object_count, "metric-prefix plan object roster changed")
    planned_jobs: dict[str, tuple[str, int, str, str]] = {}
    seen_objects: set[str] = set()
    case_order: list[tuple[str, int]] = []
    for case_index, raw_case in enumerate(cases):
        case = _mapping(raw_case, name=f"metric-prefix plan case {case_index}")
        object_id = nonempty_string(
            case.get("object_id"), name=f"metric-prefix plan case {case_index} object_id"
        )
        episode_id = genuine_integer(
            case.get("episode_id"),
            name=f"metric-prefix plan case {case_index} episode_id",
            minimum=0,
        )
        stratum = nonempty_string(
            case.get("stratum"), name=f"metric-prefix plan case {case_index} stratum"
        )
        _require(object_id not in seen_objects, "metric-prefix plan repeats an object")
        seen_objects.add(object_id)
        case_order.append((object_id, episode_id))
        expected_case_id = content_id(
            {
                "schema": "bayesian-phystwin.deform360-prob4d-metric-case-id-v1",
                "object_id": object_id,
                "episode_id": episode_id,
            }
        )
        _require(
            case.get("case_id") == expected_case_id,
            "metric-prefix plan case ID changed",
        )
        raw_range = _sequence(
            case.get("causal_frame_range_half_open"),
            name=f"metric-prefix plan case {case_index} causal range",
        )
        _require(len(raw_range) == 2, "metric-prefix plan causal range is invalid")
        start = genuine_integer(
            raw_range[0], name="metric-prefix plan causal start", minimum=0
        )
        stop = genuine_integer(
            raw_range[1], name="metric-prefix plan causal stop", minimum=0
        )
        _require(start < stop, "metric-prefix plan causal range is empty")
        streams = _sequence(
            case.get("streams"), name=f"metric-prefix plan case {case_index} streams"
        )
        _require(
            len(streams) >= 2,
            "every calibration object requires at least two supported streams",
        )
        stream_order: list[tuple[str, str]] = []
        for stream_index, raw_stream in enumerate(streams):
            stream = _mapping(
                raw_stream,
                name=f"metric-prefix plan case {case_index} stream {stream_index}",
            )
            job_id = sha256_digest(stream.get("job_id"), name="stream job_id")
            camera_id = nonempty_string(
                stream.get("camera_id"), name="stream camera_id"
            )
            _require(job_id not in planned_jobs, "metric-prefix plan repeats a job")
            planned_jobs[job_id] = (object_id, episode_id, stratum, camera_id)
            stream_order.append((camera_id, job_id))
            for field in (
                "prediction_manifest",
                "metric_prefix",
                "metric_calibration",
            ):
                _validate_file_record(
                    stream.get(field), name=f"metric-prefix plan {field}"
                )
        _require(stream_order == sorted(stream_order), "metric-prefix plan streams unsorted")
    _require(case_order == sorted(case_order), "metric-prefix plan cases unsorted")
    _require(
        planned_jobs == expected_jobs,
        "metric-prefix plan does not match supported batch jobs",
    )


def validate_deform360_prob4d_metric_batch(
    directory: str | Path,
) -> dict[str, Any]:
    """Strictly reload a published metric batch and its recursive checksums."""

    root = _ordinary_directory(directory, name="metric batch")
    result = _load_json(root / METRIC_BATCH_RESULT_FILENAME, name="metric batch result")
    require_exact_fields(result, expected=_RESULT_FIELDS, name="metric batch result")
    _require(
        result["schema"] == DEFORM360_PROB4D_METRIC_BATCH_SCHEMA
        and result["schema_version"] == DEFORM360_PROB4D_METRIC_BATCH_VERSION
        and result["semantics"] == DEFORM360_PROB4D_METRIC_BATCH_SEMANTICS,
        "metric batch contract changed",
    )
    identity = dict(result)
    declared_id = sha256_digest(identity.pop("result_id"), name="result_id")
    _require(content_id(identity) == declared_id, "metric batch result ID changed")
    exact_revision(result["implementation_revision"], name="implementation_revision")
    for field in ("production_result_id", "admission_id"):
        sha256_digest(result[field], name=field)
    object_count = genuine_integer(
        result["object_count"], name="object_count", minimum=1
    )
    jobs = _sequence(result["jobs"], name="metric batch jobs")
    counts = {"supported": 0, "support-negative": 0, "technical-failure": 0}
    ordering: list[tuple[str, str, str]] = []
    supported_objects: set[str] = set()
    for index, raw in enumerate(jobs):
        row = _mapping(raw, name=f"metric batch job {index}")
        require_exact_fields(
            row, expected=_JOB_FIELDS, name=f"metric batch job {index}"
        )
        status = nonempty_string(row["status"], name="job status")
        _require(status in counts, "unsupported metric batch job status")
        counts[status] += 1
        object_id = nonempty_string(row["object_id"], name="job object_id")
        genuine_integer(row["episode_id"], name="job episode_id", minimum=0)
        nonempty_string(row["stratum"], name="job stratum")
        camera_id = nonempty_string(row["camera_id"], name="job camera_id")
        canonical_relative_posix_path(
            row["output_relative_directory"], name="job output_relative_directory"
        )
        job_id = sha256_digest(row["job_id"], name="job_id")
        ordering.append((object_id, camera_id, job_id))
        if status == "supported":
            supported_objects.add(object_id)
            sha256_digest(row["metric_artifact_id"], name="metric_artifact_id")
            genuine_integer(
                row["projected_point_count"],
                name="projected_point_count",
                minimum=1,
            )
            _require(
                row["failure_reason"] is None and row["failure_detail_sha256"] is None,
                "supported metric stream contains a failure",
            )
        else:
            _require(
                row["metric_artifact_id"] is None and row["projected_point_count"] == 0,
                "failed metric stream contains a metric artifact",
            )
            nonempty_string(row["failure_reason"], name="failure_reason")
            if status == "technical-failure":
                sha256_digest(
                    row["failure_detail_sha256"], name="failure_detail_sha256"
                )
            else:
                _require(
                    row["failure_detail_sha256"] is None,
                    "support negative contains technical detail",
                )
    _require(ordering == sorted(ordering), "metric batch jobs are not sorted")
    _require(
        len(jobs)
        == genuine_integer(
            result["admitted_stream_count"], name="admitted_stream_count", minimum=1
        )
        and counts["supported"] == result["supported_stream_count"]
        and counts["support-negative"] == result["support_negative_stream_count"]
        and counts["technical-failure"] == result["technical_failure_stream_count"]
        and len(supported_objects) == result["supported_object_count"],
        "metric batch accounting changed",
    )
    _require(
        len(supported_objects) <= object_count,
        "supported object count exceeds the frozen cohort",
    )
    source_artifacts = _mapping(result["source_artifacts"], name="source_artifacts")
    _require(
        set(source_artifacts)
        == {
            "prepared-source-inventory.json",
            "visual-production-result.json",
            "selection.json",
            "visual-provider-spec.json",
            "metric-prior-policy.json",
        },
        "metric batch source-artifact roster changed",
    )
    for name, digest in source_artifacts.items():
        sha256_digest(digest, name=f"source_artifacts.{name}")
    _require(
        result["claim_boundary"] == METRIC_BATCH_CLAIM_BOUNDARY,
        "metric batch claim boundary changed",
    )
    boundary = _mapping(result["information_boundary"], name="information_boundary")
    require_exact_fields(
        boundary, expected=_BATCH_BOUNDARY_FIELDS, name="information_boundary"
    )
    for field in _BATCH_BOUNDARY_FIELDS:
        genuine_boolean(boundary[field], name=f"information_boundary.{field}")
    for field in (
        "calibration_camera_images_opened",
        "calibration_tactile_payloads_opened",
        "rendered_depth_opened",
        "confirmation_payloads_opened",
        "target_outcomes_used",
        "future_frames_used",
        "replacement_allowed",
        "human_approval_required",
    ):
        _require(boundary[field] is False, f"information boundary changed: {field}")
    _require(
        boundary["calibration_robot_state_access_attempted"] is True,
        "robot-state access was not attempted for the frozen batch",
    )
    expected_opened = counts["supported"] + counts["support-negative"] > 0
    _require(
        boundary["calibration_robot_state_opened"] is expected_opened
        and boundary["calibration_camera_calibration_opened"] is expected_opened,
        "robot or camera-calibration access accounting changed",
    )
    plan_emitted = genuine_boolean(result["plan_emitted"], name="plan_emitted")
    if plan_emitted:
        plan_path, _ = _verify_file_record(root, result["plan_file"], name="plan file")
        plan = _load_json(plan_path, name="metric-prefix plan")
        _require(
            plan.get("schema") == PLAN_SCHEMA
            and plan.get("schema_version") == PLAN_VERSION
            and plan.get("semantics") == PLAN_SEMANTICS
            and plan.get("information_boundary") == _PLAN_BOUNDARY,
            "metric-prefix plan contract changed",
        )
        declared_plan_id = sha256_digest(plan.get("plan_id"), name="plan_id")
        _require(
            content_id({key: value for key, value in plan.items() if key != "plan_id"})
            == declared_plan_id,
            "metric-prefix plan ID changed",
        )
        _validate_emitted_plan_binding(
            plan,
            result=result,
            jobs=jobs,
            source_artifacts=source_artifacts,
            object_count=object_count,
        )
        _require(
            counts
            == {
                "supported": len(jobs),
                "support-negative": 0,
                "technical-failure": 0,
            }
            and result["status"] == "all-streams-supported",
            "metric-prefix plan emitted for an incomplete batch",
        )
    else:
        _require(result["plan_file"] is None, "unemitted plan has a file record")
        expected_status = (
            "technical-failures-retained"
            if counts["technical-failure"]
            else (
                "support-negatives-retained"
                if counts["support-negative"]
                else "insufficient-multiview-support"
            )
        )
        _require(
            result["status"] == expected_status,
            "incomplete metric batch status changed",
        )
    checksum_path = _ordinary_file(root / "SHA256SUMS", name="batch SHA256SUMS")
    observed_lines = checksum_path.read_text(encoding="ascii").splitlines()
    expected_paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    expected_lines = [
        f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in expected_paths
    ]
    _require(observed_lines == expected_lines, "metric batch checksums changed")
    return cast(dict[str, Any], plain_json(result))


def materialize_deform360_prob4d_metric_batch(
    *,
    prepared_source_inventory_path: str | Path,
    production_result_path: str | Path,
    production_root: str | Path,
    prediction_root: str | Path,
    processed_root: str | Path,
    selection_path: str | Path,
    visual_provider_spec_path: str | Path,
    metric_prior_policy_path: str | Path,
    expected_processing_revision: str,
    implementation_revision: str,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Materialize every sealed calibration stream and publish one batch."""

    inventory_path = _ordinary_file(
        prepared_source_inventory_path, name="prepared-source inventory"
    )
    production_path = _ordinary_file(
        production_result_path, name="visual production result"
    )
    selection_source = _ordinary_file(selection_path, name="selection")
    provider_source = _ordinary_file(
        visual_provider_spec_path, name="visual provider specification"
    )
    policy_source = _ordinary_file(metric_prior_policy_path, name="metric prior policy")
    production_root_path = _ordinary_directory(
        production_root, name="visual production root"
    )
    prediction_root_path = _ordinary_directory(prediction_root, name="prediction root")
    processed_root_path = _ordinary_directory(processed_root, name="processed root")
    revision = exact_revision(implementation_revision, name="implementation_revision")
    processing_revision = exact_revision(
        expected_processing_revision, name="expected_processing_revision"
    )
    production, _inventory, selection, provider, _policy = _validate_inputs(
        prepared_source_inventory_path=inventory_path,
        production_result_path=production_path,
        selection_path=selection_source,
        visual_provider_spec_path=provider_source,
        metric_prior_policy_path=policy_source,
        expected_processing_revision=processing_revision,
    )
    selected_identities = _source_selection_rows(selection)
    motioncrafter = _mapping(provider["motioncrafter"], name="MotionCrafter")
    target_height = genuine_integer(
        motioncrafter.get("height"), name="MotionCrafter height", minimum=1
    )
    target_width = genuine_integer(
        motioncrafter.get("width"), name="MotionCrafter width", minimum=1
    )
    raw_jobs = _sequence(production["jobs"], name="visual production jobs")
    jobs = sorted(
        (_mapping(row, name="visual production job") for row in raw_jobs),
        key=lambda row: (
            str(row["object_id"]),
            str(row["camera_id"]),
            str(row["job_id"]),
        ),
    )

    target = Path(output_directory).absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    _ordinary_directory(target.parent, name="metric batch output parent")
    _require(not os.path.lexists(target), "metric batch output already exists")
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.partial"
    temporary.mkdir(mode=0o700)
    metric_root = temporary / METRIC_DIRECTORY_NAME
    metric_root.mkdir()
    result_jobs: list[dict[str, Any]] = []
    supported_streams: list[dict[str, Any]] = []
    observed_identities: set[tuple[str, int, str]] = set()
    try:
        for row in jobs:
            seal, prediction_record = _load_prediction_seal(
                production_root=production_root_path,
                prediction_root=prediction_root_path,
                production=production,
                row=row,
            )
            identity = (
                cast(str, seal["object_id"]),
                cast(int, seal["episode_id"]),
                cast(str, seal["stratum"]),
            )
            _require(
                identity in selected_identities,
                "prediction seal is outside the calibration selection",
            )
            observed_identities.add(identity)
            output_relative = canonical_relative_posix_path(
                seal["output_relative_directory"], name="metric output directory"
            )
            metric_directory = metric_root / PurePosixPath(output_relative)
            status = "supported"
            failure_reason: str | None = None
            failure_detail_sha256: str | None = None
            metric_artifact_id: str | None = None
            projected_point_count = 0
            try:
                metric = materialize_deform360_robot_metric_prefix(
                    prepared_source_inventory_path=inventory_path,
                    processed_root=processed_root_path,
                    object_id=cast(str, seal["object_id"]),
                    camera_id=cast(str, seal["camera_id"]),
                    expected_processing_revision=processing_revision,
                    target_height=target_height,
                    target_width=target_width,
                    output_directory=metric_directory,
                )
                metric = validate_deform360_robot_metric_prefix(metric_directory)
                _require(
                    metric["object_id"] == seal["object_id"]
                    and metric["episode_id"] == seal["episode_id"]
                    and metric["stratum"] == seal["stratum"]
                    and metric["camera_id"] == seal["camera_id"]
                    and metric["causal_frame_range_half_open"]
                    == seal["causal_prefix_frame_range_half_open"],
                    "metric-prefix identity differs from prediction seal",
                )
                metric_artifact_id = sha256_digest(
                    metric["artifact_id"], name="metric artifact_id"
                )
                projected_point_count = genuine_integer(
                    metric["projected_point_count"],
                    name="projected_point_count",
                    minimum=1,
                )
                metric_prefix, metric_calibration = _metric_stream_records(
                    metric_directory=metric_directory, metric_root=metric_root
                )
                supported_streams.append(
                    {
                        "job_id": seal["job_id"],
                        "object_id": seal["object_id"],
                        "episode_id": seal["episode_id"],
                        "stratum": seal["stratum"],
                        "camera_id": seal["camera_id"],
                        "causal_frame_range_half_open": seal[
                            "causal_prefix_frame_range_half_open"
                        ],
                        "prediction_manifest": prediction_record,
                        "metric_prefix": metric_prefix,
                        "metric_calibration": metric_calibration,
                    }
                )
            except ValueError as error:
                if str(error) == SUPPORT_NEGATIVE_DETAIL:
                    status = "support-negative"
                    failure_reason = (
                        "released-robot-geometry-outside-fixed-camera-prefix"
                    )
                else:
                    status = "technical-failure"
                    failure_reason = "metric-materialization-failed"
                    failure_detail_sha256 = hashlib.sha256(
                        _canonical_json_bytes(
                            {"type": type(error).__name__, "detail": str(error)}
                        )
                    ).hexdigest()
            except Exception as error:  # pragma: no cover - defensive retention
                status = "technical-failure"
                failure_reason = "unexpected-metric-materialization-failure"
                failure_detail_sha256 = hashlib.sha256(
                    _canonical_json_bytes(
                        {"type": type(error).__name__, "detail": str(error)}
                    )
                ).hexdigest()
            if status != "supported" and metric_directory.exists():
                shutil.rmtree(metric_directory)
            result_jobs.append(
                {
                    "job_id": seal["job_id"],
                    "object_id": seal["object_id"],
                    "episode_id": seal["episode_id"],
                    "stratum": seal["stratum"],
                    "camera_id": seal["camera_id"],
                    "output_relative_directory": output_relative,
                    "status": status,
                    "metric_artifact_id": metric_artifact_id,
                    "projected_point_count": projected_point_count,
                    "failure_reason": failure_reason,
                    "failure_detail_sha256": failure_detail_sha256,
                }
            )

        _require(
            observed_identities == selected_identities,
            "sealed visual production does not cover the calibration selection",
        )

        statuses = [cast(str, row["status"]) for row in result_jobs]
        support_negative_count = statuses.count("support-negative")
        technical_failure_count = statuses.count("technical-failure")
        supported_count = statuses.count("supported")
        plan: dict[str, Any] | None = None
        plan_record: dict[str, object] | None = None
        batch_status: str
        if technical_failure_count:
            batch_status = "technical-failures-retained"
        elif support_negative_count:
            batch_status = "support-negatives-retained"
        else:
            try:
                plan = _build_plan(
                    streams=supported_streams,
                    production=production,
                    selection=selection,
                    provider=provider,
                    selection_path=selection_source,
                    visual_provider_spec_path=provider_source,
                    metric_prior_policy_path=policy_source,
                    processing_revision=processing_revision,
                )
            except _InsufficientMultiviewSupport:
                batch_status = "insufficient-multiview-support"
            else:
                batch_status = "all-streams-supported"
                plan_path = temporary / METRIC_PREFIX_PLAN_FILENAME
                _write_json(plan_path, plan)
                plan_record = _file_record(plan_path, root=temporary)
        opened = supported_count + support_negative_count > 0
        boundary = {
            "calibration_robot_state_access_attempted": bool(result_jobs),
            "calibration_robot_state_opened": opened,
            "calibration_camera_calibration_opened": opened,
            "calibration_camera_images_opened": False,
            "calibration_tactile_payloads_opened": False,
            "rendered_depth_opened": False,
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
            "future_frames_used": False,
            "replacement_allowed": False,
            "human_approval_required": False,
        }
        source_artifacts = {
            "prepared-source-inventory.json": _sha256_file(inventory_path),
            "visual-production-result.json": _sha256_file(production_path),
            "selection.json": _sha256_file(selection_source),
            "visual-provider-spec.json": _sha256_file(provider_source),
            "metric-prior-policy.json": _sha256_file(policy_source),
        }
        result_identity: dict[str, Any] = {
            "schema": DEFORM360_PROB4D_METRIC_BATCH_SCHEMA,
            "schema_version": DEFORM360_PROB4D_METRIC_BATCH_VERSION,
            "semantics": DEFORM360_PROB4D_METRIC_BATCH_SEMANTICS,
            "implementation_revision": revision,
            "production_result_id": production["result_id"],
            "admission_id": production["admission_id"],
            "object_count": production["object_count"],
            "admitted_stream_count": len(result_jobs),
            "supported_stream_count": supported_count,
            "support_negative_stream_count": support_negative_count,
            "technical_failure_stream_count": technical_failure_count,
            "supported_object_count": len(
                {
                    cast(str, row["object_id"])
                    for row in result_jobs
                    if row["status"] == "supported"
                }
            ),
            "plan_emitted": plan is not None,
            "plan_file": plan_record,
            "status": batch_status,
            "jobs": result_jobs,
            "source_artifacts": dict(sorted(source_artifacts.items())),
            "information_boundary": boundary,
            "claim_boundary": METRIC_BATCH_CLAIM_BOUNDARY,
        }
        result = {**result_identity, "result_id": content_id(result_identity)}
        _write_json(temporary / METRIC_BATCH_RESULT_FILENAME, result)
        _write_checksums(temporary)
        validate_deform360_prob4d_metric_batch(temporary)
        _require(not os.path.lexists(target), "metric batch output already exists")
        os.rename(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validate_deform360_prob4d_metric_batch(target)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-source-inventory", type=Path, required=True)
    parser.add_argument("--production-result", type=Path, required=True)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--visual-provider-spec", type=Path, required=True)
    parser.add_argument("--metric-prior-policy", type=Path, required=True)
    parser.add_argument("--processing-revision", required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = materialize_deform360_prob4d_metric_batch(
        prepared_source_inventory_path=arguments.prepared_source_inventory,
        production_result_path=arguments.production_result,
        production_root=arguments.production_root,
        prediction_root=arguments.prediction_root,
        processed_root=arguments.processed_root,
        selection_path=arguments.selection,
        visual_provider_spec_path=arguments.visual_provider_spec,
        metric_prior_policy_path=arguments.metric_prior_policy,
        expected_processing_revision=arguments.processing_revision,
        implementation_revision=arguments.implementation_revision,
        output_directory=arguments.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


__all__ = [
    "DEFORM360_PROB4D_METRIC_BATCH_SCHEMA",
    "DEFORM360_PROB4D_METRIC_BATCH_SEMANTICS",
    "DEFORM360_PROB4D_METRIC_BATCH_VERSION",
    "METRIC_BATCH_RESULT_FILENAME",
    "METRIC_DIRECTORY_NAME",
    "METRIC_PREFIX_PLAN_FILENAME",
    "SUPPORT_NEGATIVE_DETAIL",
    "materialize_deform360_prob4d_metric_batch",
    "validate_deform360_prob4d_metric_batch",
]


if __name__ == "__main__":
    raise SystemExit(main())
