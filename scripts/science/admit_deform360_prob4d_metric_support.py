"""Admit retained Deform360 metric support under the frozen object minimum.

The first source-only metric batch deliberately retained every unsupported camera
stream and therefore emitted no plan.  This versioned admission does not rewrite
that immutable batch.  It verifies the complete batch, applies the already-frozen
minimum number of supported streams per physical object, and emits a plan over
supported streams only while retaining every support-negative in the bound source
artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from types import ModuleType
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
from bayesian_phystwin.deform360_calibration_visual_production import (
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
    validate_deform360_robot_metric_prefix,
)

SUPPORT_ADMISSION_SCHEMA: Final = (
    "bayesian-phystwin.deform360-prob4d-metric-support-admission"
)
SUPPORT_ADMISSION_VERSION: Final = 1
SUPPORT_ADMISSION_SEMANTICS: Final = (
    "retained-stream-outcomes-with-frozen-object-minimum-v2"
)
SUPPORT_ADMISSION_RESULT_FILENAME: Final = "metric-support-admission-result.json"
SUPPORT_ADMISSION_PLAN_FILENAME: Final = "metric-prefix-plan.json"
SUPPORT_ADMISSION_LOCK_FILENAME: Final = "source-gate-lock.json"

SUPPORT_ADMISSION_CLAIM_BOUNDARY: Final = (
    "This source-only admission applies the preregistered minimum number of "
    "supported metric streams per physical object to an immutable complete metric "
    "batch. It retains every support-negative, permits no replacement, opens no "
    "confirmation payload or future frame, and establishes neither provider "
    "calibration nor downstream physical-query benefit."
)

_RESULT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "result_id",
        "implementation_revision",
        "source_metric_batch_result_id",
        "source_metric_batch_result_sha256",
        "source_metric_batch_checksums_sha256",
        "source_gate_lock_id",
        "production_result_id",
        "object_count",
        "exact_stratum_counts",
        "admitted_stream_count",
        "supported_stream_count",
        "support_negative_stream_count",
        "technical_failure_stream_count",
        "supported_object_count",
        "minimum_supported_streams_per_object",
        "support_by_object",
        "plan_emitted",
        "plan_id",
        "plan_file",
        "status",
        "information_boundary",
        "claim_boundary",
    }
)
_SUPPORT_FIELDS = frozenset(
    {
        "object_id",
        "stratum",
        "admitted_stream_count",
        "supported_stream_count",
        "support_negative_stream_count",
        "technical_failure_stream_count",
        "minimum_supported_streams_required",
        "minimum_support_passed",
    }
)
_FILE_FIELDS = frozenset({"path", "sha256", "byte_count"})
_BOUNDARY_FIELDS = frozenset(
    {
        "calibration_payloads_opened",
        "confirmation_payloads_opened",
        "target_outcomes_used",
        "future_frames_used",
        "replacement_allowed",
        "human_approval_required",
        "new_measurements_required",
    }
)
_EXPECTED_BOUNDARY: Final = {
    "calibration_payloads_opened": True,
    "confirmation_payloads_opened": False,
    "target_outcomes_used": False,
    "future_frames_used": False,
    "replacement_allowed": False,
    "human_approval_required": False,
    "new_measurements_required": False,
}


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


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_json_bytes(value) + b"\n")


def _load_json(path: str | Path, *, name: str) -> dict[str, Any]:
    source = _ordinary_file(path, name=name)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return cast(dict[str, Any], value)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return cast(Sequence[Any], value)


def _metric_batch_module() -> ModuleType:
    path = Path(__file__).with_name("materialize_deform360_prob4d_metric_batch.py")
    spec = importlib.util.spec_from_file_location(
        "deform360_prob4d_metric_batch_for_support_admission", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load metric-batch implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _file_record(path: Path, *, root: Path) -> dict[str, object]:
    source = _ordinary_file(path, name="support-admission artifact")
    _require(root == source or root in source.parents, "artifact escapes output root")
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
    path = _ordinary_file(root / PurePosixPath(cast(str, record["path"])), name=name)
    _require(root == path or root in path.parents, f"{name} escapes output root")
    _require(path.stat().st_size == record["byte_count"], f"{name} byte count changed")
    _require(_sha256_file(path) == record["sha256"], f"{name} SHA-256 changed")
    return path, record


def _write_checksums(root: Path) -> None:
    paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [
        f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in paths
    ]
    (root / "SHA256SUMS").write_text("".join(lines), encoding="ascii")


def _validate_gate_lock(path: Path) -> dict[str, Any]:
    value = _load_json(path, name="source-gate lock")
    identity = dict(value)
    declared = sha256_digest(identity.pop("artifact_id"), name="source-gate lock ID")
    _require(content_id(identity) == declared, "source-gate lock ID changed")
    cohort = _mapping(value.get("cohort"), name="source-gate cohort")
    boundary = _mapping(value.get("information_boundary"), name="source-gate boundary")
    for field in (
        "confirmation_payloads_opened",
        "future_frames_used",
        "human_approval_required",
        "new_measurements_required",
        "replacement_allowed",
        "target_outcomes_used",
    ):
        _require(boundary.get(field) is False, f"source-gate boundary changed: {field}")
    genuine_integer(cohort.get("exact_object_count"), name="exact_object_count", minimum=1)
    genuine_integer(
        cohort.get("minimum_metric_streams_per_object"),
        name="minimum_metric_streams_per_object",
        minimum=1,
    )
    strata = _mapping(cohort.get("exact_stratum_counts"), name="exact_stratum_counts")
    _require(set(strata) == {"sheet", "volumetric"}, "source-gate strata changed")
    for name, count in strata.items():
        genuine_integer(count, name=f"exact_stratum_counts.{name}", minimum=1)
    return value


def _support_rows(
    result: Mapping[str, Any], *, minimum: int
) -> tuple[list[dict[str, object]], bool]:
    grouped: dict[tuple[str, str], Counter[str]] = {}
    for index, raw in enumerate(_sequence(result.get("jobs"), name="metric jobs")):
        row = _mapping(raw, name=f"metric job {index}")
        object_id = nonempty_string(row.get("object_id"), name="job object_id")
        stratum = nonempty_string(row.get("stratum"), name="job stratum")
        status = nonempty_string(row.get("status"), name="job status")
        _require(
            status in {"supported", "support-negative", "technical-failure"},
            "metric job status changed",
        )
        key = (object_id, stratum)
        grouped.setdefault(key, Counter())[status] += 1
    rows: list[dict[str, object]] = []
    admitted = True
    for (object_id, stratum), counts in sorted(grouped.items()):
        supported = counts["supported"]
        passed = supported >= minimum
        admitted = admitted and passed
        rows.append(
            {
                "object_id": object_id,
                "stratum": stratum,
                "admitted_stream_count": sum(counts.values()),
                "supported_stream_count": supported,
                "support_negative_stream_count": counts["support-negative"],
                "technical_failure_stream_count": counts["technical-failure"],
                "minimum_supported_streams_required": minimum,
                "minimum_support_passed": passed,
            }
        )
    return rows, admitted


def _build_supported_plan(
    *,
    module: ModuleType,
    metric_batch_root: Path,
    result: Mapping[str, Any],
    production_result_path: Path,
    production_root: Path,
    prediction_root: Path,
    selection_path: Path,
    visual_provider_spec_path: Path,
    metric_prior_policy_path: Path,
    processing_revision: str,
) -> dict[str, Any]:
    production = validate_deform360_calibration_visual_production_result(
        _load_json(production_result_path, name="visual production result")
    )
    _require(
        production["result_id"] == result["production_result_id"],
        "production result differs from metric batch",
    )
    selection = _load_json(selection_path, name="selection")
    provider = _load_json(visual_provider_spec_path, name="visual provider specification")
    source_artifacts = _mapping(result.get("source_artifacts"), name="source artifacts")
    expected_digests = {
        "visual-production-result.json": _sha256_file(production_result_path),
        "selection.json": _sha256_file(selection_path),
        "visual-provider-spec.json": _sha256_file(visual_provider_spec_path),
        "metric-prior-policy.json": _sha256_file(metric_prior_policy_path),
    }
    for name, digest in expected_digests.items():
        _require(source_artifacts.get(name) == digest, f"source artifact changed: {name}")

    production_jobs = {
        nonempty_string(row.get("job_id"), name="production job_id"): row
        for row in (
            _mapping(raw, name="production job")
            for raw in _sequence(production.get("jobs"), name="production jobs")
        )
    }
    metric_root = _ordinary_directory(
        metric_batch_root / cast(str, module.METRIC_DIRECTORY_NAME),
        name="metric stream root",
    )
    streams: list[dict[str, Any]] = []
    for index, raw in enumerate(_sequence(result.get("jobs"), name="metric jobs")):
        row = _mapping(raw, name=f"metric job {index}")
        if row.get("status") != "supported":
            continue
        job_id = nonempty_string(row.get("job_id"), name="metric job_id")
        _require(job_id in production_jobs, "supported metric job is absent from production")
        seal, prediction_record = module._load_prediction_seal(
            production_root=production_root,
            prediction_root=prediction_root,
            production=production,
            row=production_jobs[job_id],
        )
        output_relative = canonical_relative_posix_path(
            row.get("output_relative_directory"), name="metric output directory"
        )
        _require(
            seal["output_relative_directory"] == output_relative,
            "metric and prediction output directories differ",
        )
        metric_directory = _ordinary_directory(
            metric_root / PurePosixPath(output_relative), name="metric stream"
        )
        metric = validate_deform360_robot_metric_prefix(metric_directory)
        _require(
            metric["artifact_id"] == row.get("metric_artifact_id")
            and metric["object_id"] == seal["object_id"]
            and metric["episode_id"] == seal["episode_id"]
            and metric["stratum"] == seal["stratum"]
            and metric["camera_id"] == seal["camera_id"],
            "metric stream identity differs from retained batch",
        )
        metric_prefix, metric_calibration = module._metric_stream_records(
            metric_directory=metric_directory,
            metric_root=metric_root,
        )
        streams.append(
            {
                "job_id": job_id,
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
    plan = module._build_plan(
        streams=streams,
        production=production,
        selection=selection,
        provider=provider,
        selection_path=selection_path,
        visual_provider_spec_path=visual_provider_spec_path,
        metric_prior_policy_path=metric_prior_policy_path,
        processing_revision=processing_revision,
    )
    module._validate_emitted_plan_binding(
        plan,
        result=result,
        jobs=_sequence(result.get("jobs"), name="metric jobs"),
        source_artifacts=source_artifacts,
        object_count=genuine_integer(
            result.get("object_count"), name="object_count", minimum=1
        ),
    )
    return cast(dict[str, Any], plain_json(plan))


def validate_deform360_prob4d_metric_support_admission(
    directory: str | Path,
) -> dict[str, Any]:
    """Strictly reload one support admission and its optional supported plan."""

    root = _ordinary_directory(directory, name="metric support admission")
    result = _load_json(
        root / SUPPORT_ADMISSION_RESULT_FILENAME, name="support admission result"
    )
    require_exact_fields(result, expected=_RESULT_FIELDS, name="support admission result")
    _require(
        result["schema"] == SUPPORT_ADMISSION_SCHEMA
        and result["schema_version"] == SUPPORT_ADMISSION_VERSION
        and result["semantics"] == SUPPORT_ADMISSION_SEMANTICS,
        "support admission contract changed",
    )
    identity = dict(result)
    declared = sha256_digest(identity.pop("result_id"), name="result_id")
    _require(content_id(identity) == declared, "support admission result ID changed")
    exact_revision(result["implementation_revision"], name="implementation_revision")
    for field in (
        "source_metric_batch_result_id",
        "source_metric_batch_result_sha256",
        "source_metric_batch_checksums_sha256",
        "source_gate_lock_id",
        "production_result_id",
    ):
        sha256_digest(result[field], name=field)
    object_count = genuine_integer(result["object_count"], name="object_count", minimum=1)
    minimum = genuine_integer(
        result["minimum_supported_streams_per_object"],
        name="minimum_supported_streams_per_object",
        minimum=1,
    )
    rows = _sequence(result["support_by_object"], name="support_by_object")
    _require(len(rows) == object_count, "support object count changed")
    seen: set[str] = set()
    totals = Counter[str]()
    strata = Counter[str]()
    all_passed = True
    for index, raw in enumerate(rows):
        row = _mapping(raw, name=f"support object {index}")
        require_exact_fields(row, expected=_SUPPORT_FIELDS, name=f"support object {index}")
        object_id = nonempty_string(row["object_id"], name="object_id")
        _require(object_id not in seen, "support admission repeats an object")
        seen.add(object_id)
        stratum = nonempty_string(row["stratum"], name="stratum")
        strata[stratum] += 1
        admitted = genuine_integer(
            row["admitted_stream_count"], name="admitted_stream_count", minimum=1
        )
        supported = genuine_integer(
            row["supported_stream_count"], name="supported_stream_count", minimum=0
        )
        negative = genuine_integer(
            row["support_negative_stream_count"],
            name="support_negative_stream_count",
            minimum=0,
        )
        technical = genuine_integer(
            row["technical_failure_stream_count"],
            name="technical_failure_stream_count",
            minimum=0,
        )
        _require(admitted == supported + negative + technical, "object support accounting changed")
        _require(
            row["minimum_supported_streams_required"] == minimum,
            "object support minimum changed",
        )
        passed = genuine_boolean(row["minimum_support_passed"], name="minimum_support_passed")
        _require(passed is (supported >= minimum), "object support decision changed")
        all_passed = all_passed and passed
        totals.update(
            {
                "admitted": admitted,
                "supported": supported,
                "negative": negative,
                "technical": technical,
            }
        )
    _require(
        totals["admitted"] == result["admitted_stream_count"]
        and totals["supported"] == result["supported_stream_count"]
        and totals["negative"] == result["support_negative_stream_count"]
        and totals["technical"] == result["technical_failure_stream_count"],
        "support admission aggregate accounting changed",
    )
    _require(
        dict(sorted(strata.items())) == result["exact_stratum_counts"],
        "support admission stratum accounting changed",
    )
    boundary = _mapping(result["information_boundary"], name="information_boundary")
    require_exact_fields(boundary, expected=_BOUNDARY_FIELDS, name="information_boundary")
    for field, expected in _EXPECTED_BOUNDARY.items():
        observed = genuine_boolean(boundary[field], name=f"information_boundary.{field}")
        _require(observed is expected, f"information boundary changed: {field}")
    _require(
        result["claim_boundary"] == SUPPORT_ADMISSION_CLAIM_BOUNDARY,
        "support admission claim boundary changed",
    )
    plan_emitted = genuine_boolean(result["plan_emitted"], name="plan_emitted")
    expected_admitted = totals["technical"] == 0 and all_passed
    _require(plan_emitted is expected_admitted, "support admission plan decision changed")
    if plan_emitted:
        plan_path, _ = _verify_file_record(root, result["plan_file"], name="plan file")
        plan = _load_json(plan_path, name="metric-prefix plan")
        _require(
            plan.get("schema") == PLAN_SCHEMA
            and plan.get("schema_version") == PLAN_VERSION
            and plan.get("semantics") == PLAN_SEMANTICS,
            "metric-prefix plan contract changed",
        )
        plan_identity = {key: value for key, value in plan.items() if key != "plan_id"}
        _require(content_id(plan_identity) == plan.get("plan_id"), "plan ID changed")
        _require(result["plan_id"] == plan["plan_id"], "admission plan identity changed")
        expected_status = (
            "all-streams-supported"
            if totals["negative"] == 0
            else "admitted-with-retained-support-negatives"
        )
        _require(result["status"] == expected_status, "admitted support status changed")
    else:
        _require(result["plan_id"] is None and result["plan_file"] is None, "negative admission has a plan")
        expected_status = (
            "technical-failures-retained"
            if totals["technical"]
            else "insufficient-multiview-support"
        )
        _require(result["status"] == expected_status, "negative support status changed")
    lock_path = _ordinary_file(root / SUPPORT_ADMISSION_LOCK_FILENAME, name="copied source-gate lock")
    lock = _validate_gate_lock(lock_path)
    _require(lock["artifact_id"] == result["source_gate_lock_id"], "copied gate lock changed")
    checksum_path = _ordinary_file(root / "SHA256SUMS", name="support admission checksums")
    observed = checksum_path.read_text(encoding="ascii").splitlines()
    paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    expected = [
        f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in paths
    ]
    _require(observed == expected, "support admission checksums changed")
    return cast(dict[str, Any], plain_json(result))


def admit_deform360_prob4d_metric_support(
    *,
    metric_batch_root: str | Path,
    production_result_path: str | Path,
    production_root: str | Path,
    prediction_root: str | Path,
    selection_path: str | Path,
    visual_provider_spec_path: str | Path,
    metric_prior_policy_path: str | Path,
    source_gate_lock_path: str | Path,
    processing_revision: str,
    implementation_revision: str,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Apply the frozen per-object support minimum to one complete metric batch."""

    module = _metric_batch_module()
    source_root = _ordinary_directory(metric_batch_root, name="source metric batch")
    source_result = module.validate_deform360_prob4d_metric_batch(source_root)
    production_path = _ordinary_file(production_result_path, name="visual production result")
    production_root_path = _ordinary_directory(production_root, name="production root")
    prediction_root_path = _ordinary_directory(prediction_root, name="prediction root")
    selection_source = _ordinary_file(selection_path, name="selection")
    provider_source = _ordinary_file(visual_provider_spec_path, name="visual provider specification")
    policy_source = _ordinary_file(metric_prior_policy_path, name="metric prior policy")
    gate_source = _ordinary_file(source_gate_lock_path, name="source-gate lock")
    gate = _validate_gate_lock(gate_source)
    revision = exact_revision(implementation_revision, name="implementation_revision")
    processing = exact_revision(processing_revision, name="processing_revision")

    cohort = _mapping(gate["cohort"], name="source-gate cohort")
    minimum = genuine_integer(
        cohort["minimum_metric_streams_per_object"],
        name="minimum_metric_streams_per_object",
        minimum=1,
    )
    object_count = genuine_integer(
        source_result["object_count"], name="object_count", minimum=1
    )
    _require(object_count == cohort["exact_object_count"], "source object count differs from gate lock")
    _require(
        source_result["supported_object_count"] == object_count,
        "source batch does not retain support for every object",
    )
    admitted_count = genuine_integer(
        source_result["admitted_stream_count"], name="admitted_stream_count", minimum=1
    )
    supported_count = genuine_integer(
        source_result["supported_stream_count"], name="supported_stream_count", minimum=0
    )
    negative_count = genuine_integer(
        source_result["support_negative_stream_count"],
        name="support_negative_stream_count",
        minimum=0,
    )
    technical_count = genuine_integer(
        source_result["technical_failure_stream_count"],
        name="technical_failure_stream_count",
        minimum=0,
    )
    _require(
        admitted_count == supported_count + negative_count + technical_count,
        "source metric accounting changed",
    )
    support_rows, minimum_passed = _support_rows(source_result, minimum=minimum)
    observed_strata = Counter(cast(str, row["stratum"]) for row in support_rows)
    expected_strata = {
        name: genuine_integer(count, name=f"exact_stratum_counts.{name}", minimum=1)
        for name, count in _mapping(
            cohort["exact_stratum_counts"], name="exact_stratum_counts"
        ).items()
    }
    _require(dict(sorted(observed_strata.items())) == expected_strata, "source strata differ from gate lock")

    plan: dict[str, Any] | None = None
    if technical_count == 0 and minimum_passed:
        plan = _build_supported_plan(
            module=module,
            metric_batch_root=source_root,
            result=source_result,
            production_result_path=production_path,
            production_root=production_root_path,
            prediction_root=prediction_root_path,
            selection_path=selection_source,
            visual_provider_spec_path=provider_source,
            metric_prior_policy_path=policy_source,
            processing_revision=processing,
        )

    target = Path(output_directory).absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    _ordinary_directory(target.parent, name="support admission output parent")
    _require(not os.path.lexists(target), "support admission output already exists")
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.partial"
    temporary.mkdir(mode=0o700)
    try:
        shutil.copy2(gate_source, temporary / SUPPORT_ADMISSION_LOCK_FILENAME)
        plan_record: dict[str, object] | None = None
        if plan is not None:
            plan_path = temporary / SUPPORT_ADMISSION_PLAN_FILENAME
            _write_json(plan_path, plan)
            plan_record = _file_record(plan_path, root=temporary)
        status = (
            "technical-failures-retained"
            if technical_count
            else (
                "insufficient-multiview-support"
                if plan is None
                else (
                    "all-streams-supported"
                    if negative_count == 0
                    else "admitted-with-retained-support-negatives"
                )
            )
        )
        boundary = dict(_EXPECTED_BOUNDARY)
        identity: dict[str, Any] = {
            "schema": SUPPORT_ADMISSION_SCHEMA,
            "schema_version": SUPPORT_ADMISSION_VERSION,
            "semantics": SUPPORT_ADMISSION_SEMANTICS,
            "implementation_revision": revision,
            "source_metric_batch_result_id": source_result["result_id"],
            "source_metric_batch_result_sha256": _sha256_file(
                source_root / cast(str, module.METRIC_BATCH_RESULT_FILENAME)
            ),
            "source_metric_batch_checksums_sha256": _sha256_file(
                source_root / "SHA256SUMS"
            ),
            "source_gate_lock_id": gate["artifact_id"],
            "production_result_id": source_result["production_result_id"],
            "object_count": object_count,
            "exact_stratum_counts": dict(sorted(observed_strata.items())),
            "admitted_stream_count": admitted_count,
            "supported_stream_count": supported_count,
            "support_negative_stream_count": negative_count,
            "technical_failure_stream_count": technical_count,
            "supported_object_count": source_result["supported_object_count"],
            "minimum_supported_streams_per_object": minimum,
            "support_by_object": support_rows,
            "plan_emitted": plan is not None,
            "plan_id": None if plan is None else plan["plan_id"],
            "plan_file": plan_record,
            "status": status,
            "information_boundary": boundary,
            "claim_boundary": SUPPORT_ADMISSION_CLAIM_BOUNDARY,
        }
        result = {**identity, "result_id": content_id(identity)}
        _write_json(temporary / SUPPORT_ADMISSION_RESULT_FILENAME, result)
        _write_checksums(temporary)
        validate_deform360_prob4d_metric_support_admission(temporary)
        _require(not os.path.lexists(target), "support admission output already exists")
        os.rename(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validate_deform360_prob4d_metric_support_admission(target)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric-batch-root", type=Path, required=True)
    parser.add_argument("--production-result", type=Path, required=True)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--visual-provider-spec", type=Path, required=True)
    parser.add_argument("--metric-prior-policy", type=Path, required=True)
    parser.add_argument("--source-gate-lock", type=Path, required=True)
    parser.add_argument("--processing-revision", required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = admit_deform360_prob4d_metric_support(
        metric_batch_root=arguments.metric_batch_root,
        production_result_path=arguments.production_result,
        production_root=arguments.production_root,
        prediction_root=arguments.prediction_root,
        selection_path=arguments.selection,
        visual_provider_spec_path=arguments.visual_provider_spec,
        metric_prior_policy_path=arguments.metric_prior_policy,
        source_gate_lock_path=arguments.source_gate_lock,
        processing_revision=arguments.processing_revision,
        implementation_revision=arguments.implementation_revision,
        output_directory=arguments.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


__all__ = [
    "SUPPORT_ADMISSION_CLAIM_BOUNDARY",
    "SUPPORT_ADMISSION_LOCK_FILENAME",
    "SUPPORT_ADMISSION_PLAN_FILENAME",
    "SUPPORT_ADMISSION_RESULT_FILENAME",
    "SUPPORT_ADMISSION_SCHEMA",
    "SUPPORT_ADMISSION_SEMANTICS",
    "SUPPORT_ADMISSION_VERSION",
    "admit_deform360_prob4d_metric_support",
    "validate_deform360_prob4d_metric_support_admission",
]


if __name__ == "__main__":
    raise SystemExit(main())
