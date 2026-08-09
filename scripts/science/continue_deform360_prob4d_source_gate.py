#!/usr/bin/env python3
"""Continue the frozen Deform360 Prob4D source gate from a sealed metric batch.

The first protected source run retained camera-level support negatives even though
all ten physical objects exceeded the preregistered minimum of two metric streams.
This helper validates that immutable batch, excludes only its already-recorded
support-negative streams, and emits the ordinary metric-prefix plan for the
unchanged object-balanced calibration and source gate.
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

from bayesian_phystwin._canonical_contracts import plain_json
from bayesian_phystwin._portable_contracts import (
    content_id,
    exact_revision,
    nonempty_string,
    sha256_digest,
)
from bayesian_phystwin.deform360_public_contact_prefix import (
    _ordinary_directory,
    _ordinary_file,
)

CONTINUATION_SCHEMA: Final = (
    "bayesian-phystwin.deform360-prob4d-source-support-continuation"
)
CONTINUATION_VERSION: Final = 1
CONTINUATION_SEMANTICS: Final = (
    "retain-camera-support-negatives-and-continue-object-balanced-source-gate-v1"
)
CONTINUATION_RECEIPT_FILENAME: Final = "continuation-receipt.json"
METRIC_PREFIX_PLAN_FILENAME: Final = "metric-prefix-plan.json"
EXPECTED_SUPPORT_NEGATIVE_REASON: Final = (
    "released-robot-geometry-outside-fixed-camera-prefix"
)
CLAIM_BOUNDARY: Final = (
    "This artifact continues the already frozen source-only calibration using "
    "only supported streams from every preselected physical object. It changes "
    "no object, threshold, provider, metric source, or target boundary; it does "
    "not establish confirmation performance, physical-query benefit, safety, "
    "benchmark parity, or state of the art."
)
INFORMATION_BOUNDARY: Final = {
    "public_released_measurements_used": True,
    "new_measurements_required": False,
    "human_approval_required": False,
    "confirmation_payloads_opened": False,
    "target_outcomes_used": False,
    "future_frames_used": False,
    "replacement_allowed": False,
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
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
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


def _write_checksums(root: Path) -> None:
    paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
            for path in paths
        ),
        encoding="ascii",
    )


def _load_script(path: Path, *, module_name: str) -> ModuleType:
    source = _ordinary_file(path, name=module_name)
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _metric_batch_module() -> ModuleType:
    return _load_script(
        _repository_root()
        / "scripts/science/materialize_deform360_prob4d_metric_batch.py",
        module_name="deform360_metric_batch_continuation_dependency",
    )


def _source_gate_module() -> ModuleType:
    return _load_script(
        _repository_root()
        / "scripts/science/evaluate_deform360_prob4d_source_gate.py",
        module_name="deform360_source_gate_continuation_dependency",
    )


def validate_supported_batch_against_lock(
    batch: Mapping[str, Any],
    gate_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate support using the preregistered physical-object criterion."""

    cohort = gate_lock.get("cohort")
    if not isinstance(cohort, Mapping):
        raise ValueError("source gate lock cohort is missing")
    exact_object_count = int(cohort["exact_object_count"])
    minimum_streams = int(cohort["minimum_metric_streams_per_object"])
    expected_strata = {
        str(key): int(value)
        for key, value in cast(Mapping[str, Any], cohort["exact_stratum_counts"]).items()
    }

    _require(
        int(batch.get("object_count", -1)) == exact_object_count,
        "metric batch physical-object count differs from the source gate lock",
    )
    _require(
        int(batch.get("technical_failure_stream_count", -1)) == 0,
        "metric batch contains a technical failure",
    )
    _require(
        batch.get("status")
        in {"all-streams-supported", "support-negatives-retained"},
        "metric batch is not eligible for source-gate continuation",
    )

    jobs = batch.get("jobs")
    if isinstance(jobs, (str, bytes)) or not isinstance(jobs, Sequence):
        raise ValueError("metric batch jobs are missing")
    supported_by_object: Counter[str] = Counter()
    object_strata: dict[str, str] = {}
    support_negatives: list[dict[str, str]] = []
    observed_statuses: Counter[str] = Counter()
    for index, raw in enumerate(jobs):
        if not isinstance(raw, Mapping):
            raise ValueError(f"metric batch job {index} is invalid")
        row = cast(Mapping[str, Any], raw)
        object_id = nonempty_string(row.get("object_id"), name="object_id")
        stratum = nonempty_string(row.get("stratum"), name="stratum")
        previous = object_strata.setdefault(object_id, stratum)
        _require(previous == stratum, "one object appears in several strata")
        status = nonempty_string(row.get("status"), name="status")
        _require(
            status in {"supported", "support-negative"},
            "metric batch contains a non-scientific stream failure",
        )
        observed_statuses[status] += 1
        if status == "supported":
            supported_by_object[object_id] += 1
            continue
        reason = nonempty_string(row.get("failure_reason"), name="failure_reason")
        _require(
            reason == EXPECTED_SUPPORT_NEGATIVE_REASON,
            "metric batch contains an unregistered support-negative reason",
        )
        support_negatives.append(
            {
                "job_id": sha256_digest(row.get("job_id"), name="job_id"),
                "object_id": object_id,
                "camera_id": nonempty_string(
                    row.get("camera_id"), name="camera_id"
                ),
                "reason": reason,
            }
        )

    _require(
        len(object_strata) == exact_object_count,
        "metric batch object roster differs from the source gate lock",
    )
    _require(
        all(supported_by_object[object_id] >= minimum_streams for object_id in object_strata),
        "a frozen object has fewer supported streams than preregistered",
    )
    observed_strata = Counter(object_strata.values())
    _require(
        dict(sorted(observed_strata.items())) == dict(sorted(expected_strata.items())),
        "metric batch stratum roster differs from the source gate lock",
    )
    admitted = len(jobs)
    _require(
        int(batch.get("admitted_stream_count", -1)) == admitted
        and int(batch.get("supported_stream_count", -1))
        == observed_statuses["supported"]
        and int(batch.get("support_negative_stream_count", -1))
        == observed_statuses["support-negative"]
        and int(batch.get("supported_object_count", -1)) == exact_object_count,
        "metric batch support accounting changed",
    )
    return {
        "exact_object_count": exact_object_count,
        "minimum_metric_streams_per_object": minimum_streams,
        "supported_stream_counts_by_object": dict(sorted(supported_by_object.items())),
        "support_negatives": sorted(
            support_negatives,
            key=lambda row: (row["object_id"], row["camera_id"], row["job_id"]),
        ),
    }


def build_supported_stream_records(
    *,
    module: ModuleType,
    batch: Mapping[str, Any],
    production: Mapping[str, Any],
    production_root: Path,
    prediction_root: Path,
    metric_root: Path,
) -> list[dict[str, Any]]:
    """Reconstruct plan rows for supported jobs only, without replacement."""

    raw_production_jobs = production.get("jobs")
    if isinstance(raw_production_jobs, (str, bytes)) or not isinstance(
        raw_production_jobs, Sequence
    ):
        raise ValueError("visual production jobs are missing")
    production_jobs: dict[str, Mapping[str, Any]] = {}
    for raw in raw_production_jobs:
        if not isinstance(raw, Mapping):
            raise ValueError("visual production job is invalid")
        row = cast(Mapping[str, Any], raw)
        job_id = sha256_digest(row.get("job_id"), name="production job_id")
        _require(job_id not in production_jobs, "visual production repeats a job")
        production_jobs[job_id] = row

    raw_batch_jobs = batch.get("jobs")
    if isinstance(raw_batch_jobs, (str, bytes)) or not isinstance(
        raw_batch_jobs, Sequence
    ):
        raise ValueError("metric batch jobs are missing")
    streams: list[dict[str, Any]] = []
    for raw in raw_batch_jobs:
        if not isinstance(raw, Mapping):
            raise ValueError("metric batch job is invalid")
        row = cast(Mapping[str, Any], raw)
        if row.get("status") != "supported":
            continue
        job_id = sha256_digest(row.get("job_id"), name="metric job_id")
        _require(job_id in production_jobs, "supported metric job is not in production")
        production_row = production_jobs[job_id]
        seal, prediction_record = module._load_prediction_seal(
            production_root=production_root,
            prediction_root=prediction_root,
            production=production,
            row=production_row,
        )
        expected = {
            "job_id": job_id,
            "object_id": row.get("object_id"),
            "episode_id": row.get("episode_id"),
            "stratum": row.get("stratum"),
            "camera_id": row.get("camera_id"),
            "output_relative_directory": row.get("output_relative_directory"),
        }
        _require(
            {key: seal.get(key) for key in expected} == expected,
            "metric and visual stream identities differ",
        )
        relative = nonempty_string(
            row.get("output_relative_directory"),
            name="output_relative_directory",
        )
        metric_directory = metric_root / PurePosixPath(relative)
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
    return streams


def publish_continuation(
    *,
    metric_batch_root: str | Path,
    production_result_path: str | Path,
    production_root: str | Path,
    prediction_root: str | Path,
    selection_path: str | Path,
    visual_provider_spec_path: str | Path,
    metric_prior_policy_path: str | Path,
    source_gate_lock_path: str | Path,
    expected_processing_revision: str,
    expected_metric_batch_result_id: str,
    expected_metric_batch_implementation_revision: str,
    implementation_revision: str,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Publish a no-overwrite plan from the sealed supported metric streams."""

    module = _metric_batch_module()
    gate_module = _source_gate_module()
    batch_root = _ordinary_directory(metric_batch_root, name="source metric batch")
    batch = module.validate_deform360_prob4d_metric_batch(batch_root)
    _require(
        batch["result_id"]
        == sha256_digest(
            expected_metric_batch_result_id,
            name="expected_metric_batch_result_id",
        ),
        "source metric batch result identity changed",
    )
    source_revision = exact_revision(
        expected_metric_batch_implementation_revision,
        name="expected_metric_batch_implementation_revision",
    )
    _require(
        batch["implementation_revision"] == source_revision,
        "source metric batch implementation revision changed",
    )
    revision = exact_revision(implementation_revision, name="implementation_revision")
    processing_revision = exact_revision(
        expected_processing_revision,
        name="expected_processing_revision",
    )
    lock_path = _ordinary_file(source_gate_lock_path, name="source gate lock")
    gate_lock = gate_module.load_source_gate_lock(lock_path)
    support = validate_supported_batch_against_lock(batch, gate_lock)

    production_path = _ordinary_file(
        production_result_path, name="visual production result"
    )
    production = module.validate_deform360_calibration_visual_production_result(
        _load_json(production_path, name="visual production result")
    )
    _require(
        production["result_id"] == batch["production_result_id"],
        "visual production and source metric batch differ",
    )
    production_root_path = _ordinary_directory(
        production_root, name="visual production root"
    )
    prediction_root_path = _ordinary_directory(
        prediction_root, name="prediction root"
    )
    metric_root = _ordinary_directory(
        batch_root / module.METRIC_DIRECTORY_NAME,
        name="source metric root",
    )
    selection = _load_json(selection_path, name="selection")
    provider = _load_json(visual_provider_spec_path, name="visual provider")
    streams = build_supported_stream_records(
        module=module,
        batch=batch,
        production=production,
        production_root=production_root_path,
        prediction_root=prediction_root_path,
        metric_root=metric_root,
    )
    plan = module._build_plan(
        streams=streams,
        production=production,
        selection=selection,
        provider=provider,
        selection_path=_ordinary_file(selection_path, name="selection"),
        visual_provider_spec_path=_ordinary_file(
            visual_provider_spec_path, name="visual provider"
        ),
        metric_prior_policy_path=_ordinary_file(
            metric_prior_policy_path, name="metric prior policy"
        ),
        processing_revision=processing_revision,
    )
    module._validate_emitted_plan_binding(
        plan,
        result=batch,
        jobs=batch["jobs"],
        source_artifacts=batch["source_artifacts"],
        object_count=batch["object_count"],
    )

    target = Path(output_directory).absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    _ordinary_directory(target.parent, name="continuation output parent")
    _require(not os.path.lexists(target), "continuation output already exists")
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.partial"
    temporary.mkdir(mode=0o700)
    try:
        plan_path = temporary / METRIC_PREFIX_PLAN_FILENAME
        _write_json(plan_path, plan)
        support_negatives = cast(list[dict[str, str]], support["support_negatives"])
        identity: dict[str, Any] = {
            "schema": CONTINUATION_SCHEMA,
            "schema_version": CONTINUATION_VERSION,
            "semantics": CONTINUATION_SEMANTICS,
            "implementation_revision": revision,
            "source_metric_batch_result_id": batch["result_id"],
            "source_metric_batch_implementation_revision": source_revision,
            "visual_production_result_id": production["result_id"],
            "source_gate_lock_id": gate_lock["artifact_id"],
            "metric_prefix_plan_id": plan["plan_id"],
            "metric_prefix_plan_sha256": _sha256_file(plan_path),
            "object_count": batch["object_count"],
            "admitted_stream_count": batch["admitted_stream_count"],
            "supported_stream_count": batch["supported_stream_count"],
            "support_negative_stream_count": batch[
                "support_negative_stream_count"
            ],
            "technical_failure_stream_count": batch[
                "technical_failure_stream_count"
            ],
            "minimum_metric_streams_per_object": support[
                "minimum_metric_streams_per_object"
            ],
            "supported_stream_counts_by_object": support[
                "supported_stream_counts_by_object"
            ],
            "support_negatives": support_negatives,
            "source_files": {
                "metric_batch_result_sha256": _sha256_file(
                    batch_root / module.METRIC_BATCH_RESULT_FILENAME
                ),
                "metric_batch_checksums_sha256": _sha256_file(
                    batch_root / "SHA256SUMS"
                ),
                "source_gate_lock_sha256": _sha256_file(lock_path),
            },
            "information_boundary": dict(INFORMATION_BOUNDARY),
            "claim_boundary": CLAIM_BOUNDARY,
        }
        receipt = {**identity, "result_id": content_id(identity)}
        _write_json(temporary / CONTINUATION_RECEIPT_FILENAME, receipt)
        _write_checksums(temporary)
        validate_continuation(temporary)
        os.rename(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validate_continuation(target)


def validate_continuation(directory: str | Path) -> dict[str, Any]:
    """Validate a compact continuation plan and its content identity."""

    root = _ordinary_directory(directory, name="source support continuation")
    receipt = _load_json(
        root / CONTINUATION_RECEIPT_FILENAME,
        name="continuation receipt",
    )
    _require(
        receipt.get("schema") == CONTINUATION_SCHEMA
        and receipt.get("schema_version") == CONTINUATION_VERSION
        and receipt.get("semantics") == CONTINUATION_SEMANTICS,
        "continuation contract changed",
    )
    identity = dict(receipt)
    declared_id = sha256_digest(identity.pop("result_id"), name="result_id")
    _require(content_id(identity) == declared_id, "continuation result ID changed")
    exact_revision(receipt.get("implementation_revision"), name="implementation_revision")
    exact_revision(
        receipt.get("source_metric_batch_implementation_revision"),
        name="source metric batch implementation revision",
    )
    for field in (
        "source_metric_batch_result_id",
        "visual_production_result_id",
        "source_gate_lock_id",
        "metric_prefix_plan_id",
        "metric_prefix_plan_sha256",
    ):
        sha256_digest(receipt.get(field), name=field)
    _require(
        receipt.get("technical_failure_stream_count") == 0,
        "continuation contains a technical failure",
    )
    _require(
        receipt.get("information_boundary") == INFORMATION_BOUNDARY,
        "continuation information boundary changed",
    )
    _require(
        receipt.get("claim_boundary") == CLAIM_BOUNDARY,
        "continuation claim boundary changed",
    )
    plan_path = _ordinary_file(
        root / METRIC_PREFIX_PLAN_FILENAME,
        name="metric-prefix plan",
    )
    plan = _load_json(plan_path, name="metric-prefix plan")
    _require(
        plan.get("plan_id") == receipt["metric_prefix_plan_id"]
        and _sha256_file(plan_path) == receipt["metric_prefix_plan_sha256"],
        "continuation metric-prefix plan changed",
    )
    checksum_path = _ordinary_file(root / "SHA256SUMS", name="continuation checksums")
    expected = "".join(
        f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
        for path in sorted(
            item
            for item in root.rglob("*")
            if item.is_file() and item.name != "SHA256SUMS"
        )
    )
    _require(checksum_path.read_text(encoding="ascii") == expected, "checksums changed")
    return cast(dict[str, Any], plain_json(receipt))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--metric-batch-root", type=Path, required=True)
    build.add_argument("--production-result", type=Path, required=True)
    build.add_argument("--production-root", type=Path, required=True)
    build.add_argument("--prediction-root", type=Path, required=True)
    build.add_argument("--selection", type=Path, required=True)
    build.add_argument("--visual-provider-spec", type=Path, required=True)
    build.add_argument("--metric-prior-policy", type=Path, required=True)
    build.add_argument("--source-gate-lock", type=Path, required=True)
    build.add_argument("--processing-revision", required=True)
    build.add_argument("--expected-metric-batch-result-id", required=True)
    build.add_argument(
        "--expected-metric-batch-implementation-revision",
        required=True,
    )
    build.add_argument("--implementation-revision", required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("directory", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "validate":
        result = validate_continuation(arguments.directory)
    else:
        result = publish_continuation(
            metric_batch_root=arguments.metric_batch_root,
            production_result_path=arguments.production_result,
            production_root=arguments.production_root,
            prediction_root=arguments.prediction_root,
            selection_path=arguments.selection,
            visual_provider_spec_path=arguments.visual_provider_spec,
            metric_prior_policy_path=arguments.metric_prior_policy,
            source_gate_lock_path=arguments.source_gate_lock,
            expected_processing_revision=arguments.processing_revision,
            expected_metric_batch_result_id=(
                arguments.expected_metric_batch_result_id
            ),
            expected_metric_batch_implementation_revision=(
                arguments.expected_metric_batch_implementation_revision
            ),
            implementation_revision=arguments.implementation_revision,
            output_directory=arguments.output_dir,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
