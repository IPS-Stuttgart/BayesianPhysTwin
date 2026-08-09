#!/usr/bin/env python3
"""Audit the exact retained Deform360 Prob4D support-negative result."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

SOURCE_WORKFLOW_RUN_ID: Final = 31297018948
SOURCE_WORKFLOW_RUN_ATTEMPT: Final = 1
SOURCE_WORKFLOW_CONCLUSION: Final = "failure"
SOURCE_HEAD_SHA: Final = "ded8910becbbffe958dfd18c84ad91069e7087a4"
SOURCE_ARTIFACT_ID: Final = 9033414269
SOURCE_ARTIFACT_NAME: Final = "deform360-prob4d-source-gate-31297018948-1"
SOURCE_ARTIFACT_DIGEST: Final = (
    "sha256:7247a2a260509c4c226e7ca437aff09d090abf6d2ca08f471a2143ea7d4bf7de"
)
PRODUCTION_RESULT_ID: Final = (
    "146f885351b2af0134b8b3d3c28a76deaa899749b1b1306e0d7061807ae95f89"
)
ADMISSION_ID: Final = "715ab8479bad4d97eba766cdba1a161f1f6e83e3fd597bb09a2bf8ab8dc91e15"
PROB4D_REVISION: Final = "25d90ef7f78ba4307f4555cb636d666004e1bf66"
MOTIONCRAFTER_REVISION: Final = "9cb4e9679f5f34e249945544052464ef46324bc2"
METRIC_BATCH_RESULT_ID: Final = (
    "f246394c84fd643b6ec8961dbcb2101a73c34e46d5eaf43961f28429aeb197eb"
)
SUPPORT_NEGATIVE_REASON: Final = "released-robot-geometry-outside-fixed-camera-prefix"
OBJECT_COUNT: Final = 10
ADMITTED_STREAM_COUNT: Final = 324
SUPPORTED_STREAM_COUNT: Final = 313
SUPPORT_NEGATIVE_STREAM_COUNT: Final = 11
TECHNICAL_FAILURE_STREAM_COUNT: Final = 0
SUPPORTED_OBJECT_COUNT: Final = 10
SUPPORT_NEGATIVE_OBJECT_COUNTS_BY_STRATUM: Final = {
    "sheet": 3,
    "volumetric": 3,
}
SUPPORT_NEGATIVE_CAMERA_COUNT: Final = 5
EMPTY_SHA256: Final = hashlib.sha256(b"").hexdigest()

EXPECTED_SOURCE_FILES: Final = (
    "metric-support/metric-batch-result.json",
    "metric-support/support-receipt.json",
    "pipeline-receipt.json",
)
METRIC_RESULT_FIELDS: Final = frozenset(
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
METRIC_JOB_FIELDS: Final = frozenset(
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
SUPPORT_RECEIPT_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "implementation_revision",
        "metric_batch_step_outcome",
        "metric_batch_result_id",
        "metric_batch_status",
        "metric_batch_stderr_sha256",
        "new_measurements_required",
        "human_approval_required",
        "confirmation_payloads_opened",
        "target_outcomes_used",
    }
)
PIPELINE_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "implementation_revision",
        "visual_production_result_id",
        "prob4d_revision",
        "motioncrafter_revision",
        "stage_outcomes",
        "stderr_sha256",
        "source_gate_result_id",
        "source_gate_passed",
        "confirmation_access_authorized",
        "information_boundary",
    }
)
PIPELINE_BOUNDARY: Final = {
    "public_released_measurements_used": True,
    "new_measurements_required": False,
    "human_approval_required": False,
    "confirmation_payloads_opened": False,
    "target_outcomes_used": False,
    "future_frames_used": False,
    "replacement_allowed": False,
}
METRIC_BOUNDARY: Final = {
    "calibration_robot_state_access_attempted": True,
    "calibration_robot_state_opened": True,
    "calibration_camera_calibration_opened": True,
    "calibration_camera_images_opened": False,
    "calibration_tactile_payloads_opened": False,
    "rendered_depth_opened": False,
    "confirmation_payloads_opened": False,
    "target_outcomes_used": False,
    "future_frames_used": False,
    "replacement_allowed": False,
    "human_approval_required": False,
}
CLAIM_BOUNDARY: Final = (
    "Source-only materialization of released Deform360 robot geometry for every "
    "sealed calibration camera. This artifact does not use new capture, require "
    "human approval, open confirmation payloads or future frames, evaluate "
    "calibration or transfer, authorize confirmation, or establish state of the art."
)
_SHA256_LINE = re.compile(r"^([0-9a-f]{64})  ([^\0\r\n]+)$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _load_object(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return cast(dict[str, Any], value)


def _exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], name: str
) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    _require(not missing and not extra, f"{name} fields changed: {missing=} {extra=}")


def _string(value: object, *, name: str) -> str:
    _require(type(value) is str and bool(value), f"{name} must be a nonempty string")
    return cast(str, value)


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    _require(
        type(value) is int and cast(int, value) >= minimum,
        f"{name} must be an integer >= {minimum}",
    )
    return cast(int, value)


def _sha256(value: object, *, name: str) -> str:
    digest = _string(value, name=name)
    _require(
        len(digest) == 64 and all(char in "0123456789abcdef" for char in digest),
        f"{name} must be a lowercase SHA-256 digest",
    )
    return digest


def _revision(value: object, *, name: str) -> str:
    revision = _string(value, name=name)
    _require(
        len(revision) in {40, 64}
        and all(char in "0123456789abcdef" for char in revision),
        f"{name} must be an exact lowercase revision",
    )
    return revision


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    _require(
        not isinstance(value, (str, bytes)) and isinstance(value, Sequence),
        f"{name} must be an array",
    )
    return cast(Sequence[Any], value)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be an object")
    return cast(Mapping[str, Any], value)


def _canonical_path(value: object, *, name: str) -> str:
    path = _string(value, name=name)
    pure = PurePosixPath(path)
    _require(
        not pure.is_absolute()
        and pure.as_posix() == path
        and all(part not in {"", ".", ".."} for part in pure.parts),
        f"{name} must be a canonical relative POSIX path",
    )
    return path


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _content_id(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_path_symlinks(path: Path) -> None:
    absolute = path.absolute()
    for candidate in (absolute, *absolute.parents):
        _require(
            not candidate.is_symlink(),
            f"path contains a symlink: {candidate}",
        )


def _reject_tree_symlinks(root: Path) -> None:
    _reject_path_symlinks(root)
    for candidate in root.rglob("*"):
        _require(
            not candidate.is_symlink(),
            f"artifact contains a symlink: {candidate}",
        )


def _validate_checksums(root: Path) -> dict[str, str]:
    checksum_path = root / "SHA256SUMS"
    lines = checksum_path.read_text(encoding="ascii").splitlines()
    observed: dict[str, str] = {}
    for line in lines:
        match = _SHA256_LINE.fullmatch(line)
        _require(match is not None, "source SHA256SUMS is malformed")
        assert match is not None
        digest, relative = match.groups()
        _canonical_path(relative, name="checksummed path")
        _require(relative not in observed, "source SHA256SUMS repeats a path")
        observed[relative] = digest
    _require(
        tuple(sorted(observed)) == EXPECTED_SOURCE_FILES,
        "source compact artifact file roster changed",
    )
    for relative, digest in observed.items():
        path = root / PurePosixPath(relative)
        _require(path.is_file(), f"checksummed file is missing: {relative}")
        _require(_sha256_file(path) == digest, f"checksummed file changed: {relative}")
    actual = tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS"
        )
    )
    _require(actual == EXPECTED_SOURCE_FILES, "source artifact contains extra files")
    return observed


def _validate_metric_result(value: dict[str, Any]) -> dict[str, object]:
    _exact_fields(value, METRIC_RESULT_FIELDS, "metric batch result")
    _require(
        value["schema"] == "bayesian-phystwin.deform360-prob4d-metric-batch"
        and value["schema_version"] == 1
        and value["semantics"]
        == "all-sealed-calibration-streams-released-robot-gauge-v1",
        "metric batch contract changed",
    )
    identity = dict(value)
    declared_id = _sha256(identity.pop("result_id"), name="metric result_id")
    _require(_content_id(identity) == declared_id, "metric result ID changed")
    _require(declared_id == METRIC_BATCH_RESULT_ID, "metric result identity changed")
    _require(
        _revision(value["implementation_revision"], name="implementation revision")
        == SOURCE_HEAD_SHA,
        "metric implementation revision changed",
    )
    _require(
        _sha256(value["production_result_id"], name="production_result_id")
        == PRODUCTION_RESULT_ID,
        "metric production result changed",
    )
    _require(
        _sha256(value["admission_id"], name="admission_id") == ADMISSION_ID,
        "metric admission changed",
    )
    expected_scalars = {
        "object_count": OBJECT_COUNT,
        "admitted_stream_count": ADMITTED_STREAM_COUNT,
        "supported_stream_count": SUPPORTED_STREAM_COUNT,
        "support_negative_stream_count": SUPPORT_NEGATIVE_STREAM_COUNT,
        "technical_failure_stream_count": TECHNICAL_FAILURE_STREAM_COUNT,
        "supported_object_count": SUPPORTED_OBJECT_COUNT,
        "plan_emitted": False,
        "plan_file": None,
        "status": "support-negatives-retained",
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _require(
        {key: value.get(key) for key in expected_scalars} == expected_scalars,
        "frozen metric support decision changed",
    )
    _require(
        value["information_boundary"] == METRIC_BOUNDARY,
        "metric boundary changed",
    )
    source_artifacts = _mapping(value["source_artifacts"], name="source_artifacts")
    expected_source_names = {
        "prepared-source-inventory.json",
        "visual-production-result.json",
        "selection.json",
        "visual-provider-spec.json",
        "metric-prior-policy.json",
    }
    _require(set(source_artifacts) == expected_source_names, "source roster changed")
    for name, digest in source_artifacts.items():
        _sha256(digest, name=f"source artifact {name}")

    jobs = _sequence(value["jobs"], name="metric jobs")
    _require(
        len(jobs) == ADMITTED_STREAM_COUNT,
        "metric job roster changed",
    )
    counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    negative_objects: dict[str, set[str]] = defaultdict(set)
    negative_cameras: Counter[str] = Counter()
    supported_objects: set[str] = set()
    ordering: list[tuple[str, str, str]] = []
    seen_jobs: set[str] = set()
    seen_streams: set[tuple[str, str]] = set()
    for index, raw in enumerate(jobs):
        row = _mapping(raw, name=f"metric job {index}")
        _exact_fields(row, METRIC_JOB_FIELDS, f"metric job {index}")
        job_id = _sha256(row["job_id"], name="job_id")
        object_id = _string(row["object_id"], name="object_id")
        camera_id = _string(row["camera_id"], name="camera_id")
        _integer(row["episode_id"], name="episode_id")
        stratum = _string(row["stratum"], name="stratum")
        _require(stratum in {"sheet", "volumetric"}, "job stratum changed")
        _canonical_path(
            row["output_relative_directory"], name="output_relative_directory"
        )
        _require(job_id not in seen_jobs, "metric jobs repeat a job ID")
        _require((object_id, camera_id) not in seen_streams, "metric stream repeated")
        seen_jobs.add(job_id)
        seen_streams.add((object_id, camera_id))
        ordering.append((object_id, camera_id, job_id))
        status = _string(row["status"], name="status")
        _require(status in {"supported", "support-negative"}, "job status changed")
        counts[status] += 1
        if status == "supported":
            supported_objects.add(object_id)
            _sha256(row["metric_artifact_id"], name="metric_artifact_id")
            _integer(
                row["projected_point_count"],
                name="projected_point_count",
                minimum=1,
            )
            _require(
                row["failure_reason"] is None and row["failure_detail_sha256"] is None,
                "supported stream contains failure evidence",
            )
        else:
            _require(
                row["metric_artifact_id"] is None
                and row["projected_point_count"] == 0
                and row["failure_detail_sha256"] is None,
                "support-negative stream contains a metric or technical detail",
            )
            reason = _string(row["failure_reason"], name="failure_reason")
            reasons[reason] += 1
            negative_objects[stratum].add(object_id)
            negative_cameras[camera_id] += 1
    _require(ordering == sorted(ordering), "metric jobs are not sorted")
    _require(
        counts
        == Counter(
            {
                "supported": SUPPORTED_STREAM_COUNT,
                "support-negative": SUPPORT_NEGATIVE_STREAM_COUNT,
            }
        ),
        "counts changed",
    )
    _require(
        len(supported_objects) == SUPPORTED_OBJECT_COUNT,
        "supported object count changed",
    )
    _require(
        reasons == Counter({SUPPORT_NEGATIVE_REASON: SUPPORT_NEGATIVE_STREAM_COUNT}),
        "support-negative reason changed",
    )
    _require(
        {key: len(items) for key, items in negative_objects.items()}
        == SUPPORT_NEGATIVE_OBJECT_COUNTS_BY_STRATUM,
        "support-negative object strata changed",
    )
    _require(
        len(negative_cameras) == SUPPORT_NEGATIVE_CAMERA_COUNT,
        "support-negative camera count changed",
    )
    return {
        "supported_stream_count": counts["supported"],
        "support_negative_stream_count": counts["support-negative"],
        "technical_failure_stream_count": TECHNICAL_FAILURE_STREAM_COUNT,
        "support_negative_reason_counts": dict(sorted(reasons.items())),
        "support_negative_object_counts_by_stratum": {
            key: len(items) for key, items in sorted(negative_objects.items())
        },
        "support_negative_camera_count": len(negative_cameras),
    }


def _validate_support_receipt(value: dict[str, Any], metric: Mapping[str, Any]) -> None:
    _exact_fields(value, SUPPORT_RECEIPT_FIELDS, "support receipt")
    expected = {
        "schema": "bayesian-phystwin.deform360-prob4d-source-support-receipt",
        "schema_version": 1,
        "implementation_revision": SOURCE_HEAD_SHA,
        "metric_batch_step_outcome": "success",
        "metric_batch_result_id": METRIC_BATCH_RESULT_ID,
        "metric_batch_status": "support-negatives-retained",
        "metric_batch_stderr_sha256": EMPTY_SHA256,
        "new_measurements_required": False,
        "human_approval_required": False,
        "confirmation_payloads_opened": False,
        "target_outcomes_used": False,
    }
    _require(value == expected, "support receipt changed")
    _require(
        value["metric_batch_result_id"] == metric["result_id"]
        and value["metric_batch_status"] == metric["status"],
        "support receipt does not bind the metric result",
    )


def _validate_pipeline(value: dict[str, Any], support: Mapping[str, Any]) -> None:
    _exact_fields(value, PIPELINE_FIELDS, "pipeline receipt")
    expected_stage_outcomes = {
        "metric_batch": "success",
        "support_gate": "failure",
        "samples": "skipped",
        "calibration": "skipped",
        "source_gate": "skipped",
    }
    _require(
        value["schema"] == "bayesian-phystwin.deform360-prob4d-source-pipeline-receipt"
        and value["schema_version"] == 1,
        "pipeline contract changed",
    )
    _require(
        value["implementation_revision"] == SOURCE_HEAD_SHA,
        "pipeline revision changed",
    )
    _require(
        value["visual_production_result_id"] == PRODUCTION_RESULT_ID,
        "pipeline production result changed",
    )
    _require(value["prob4d_revision"] == PROB4D_REVISION, "Prob4D revision changed")
    _require(
        value["motioncrafter_revision"] == MOTIONCRAFTER_REVISION,
        "MotionCrafter revision changed",
    )
    _require(
        value["stage_outcomes"] == expected_stage_outcomes,
        "stage outcomes changed",
    )
    stderr = _mapping(value["stderr_sha256"], name="stderr_sha256")
    expected_stderr_names = {"metric-batch", "samples", "calibration", "source-gate"}
    _require(set(stderr) == expected_stderr_names, "stderr roster changed")
    for name, digest in stderr.items():
        _require(
            _sha256(digest, name=f"stderr {name}") == EMPTY_SHA256,
            "stderr changed",
        )
    _require(
        stderr["metric-batch"] == support["metric_batch_stderr_sha256"],
        "support and pipeline stderr identities differ",
    )
    _require(
        value["source_gate_result_id"] is None
        and value["source_gate_passed"] is None
        and value["confirmation_access_authorized"] is None,
        "skipped source gate contains a decision",
    )
    _require(
        value["information_boundary"] == PIPELINE_BOUNDARY,
        "pipeline boundary changed",
    )


def audit_support_negative(
    source_root: str | Path,
    output_directory: str | Path,
    *,
    auditor_revision: str,
) -> dict[str, Any]:
    """Validate the exact early-terminal source artifact and publish an audit."""

    revision = _revision(auditor_revision, name="auditor_revision")
    source = Path(source_root).absolute()
    output = Path(output_directory).absolute()
    _require(source.is_dir(), "source compact artifact is missing")
    _reject_tree_symlinks(source)
    _require(not output.exists(), "audit output already exists")
    _reject_path_symlinks(output.parent)

    checksums = _validate_checksums(source)
    metric_path = source / "metric-support/metric-batch-result.json"
    support_path = source / "metric-support/support-receipt.json"
    pipeline_path = source / "pipeline-receipt.json"
    metric = _load_object(metric_path, name="metric batch result")
    support = _load_object(support_path, name="support receipt")
    pipeline = _load_object(pipeline_path, name="pipeline receipt")
    summary = _validate_metric_result(metric)
    _validate_support_receipt(support, metric)
    _validate_pipeline(pipeline, support)

    receipt: dict[str, Any] = {
        "schema": "bayesian-phystwin.deform360-prob4d-source-support-independent-audit",
        "schema_version": 1,
        "source_workflow_run_id": SOURCE_WORKFLOW_RUN_ID,
        "source_workflow_run_attempt": SOURCE_WORKFLOW_RUN_ATTEMPT,
        "source_workflow_conclusion": SOURCE_WORKFLOW_CONCLUSION,
        "source_workflow_head_sha": SOURCE_HEAD_SHA,
        "source_artifact_id": SOURCE_ARTIFACT_ID,
        "source_artifact_name": SOURCE_ARTIFACT_NAME,
        "source_artifact_digest": SOURCE_ARTIFACT_DIGEST,
        "auditor_revision": revision,
        "audit_status": "validated-negative",
        "terminal_stage": "support-gate",
        "metric_batch_result_id": METRIC_BATCH_RESULT_ID,
        "source_gate_result_id": None,
        "source_gate_passed": None,
        "support_gate_passed": False,
        "confirmation_access_authorized": False,
        **summary,
        "source_file_sha256": dict(sorted(checksums.items())),
        "source_artifact_checksums_sha256": _sha256_file(source / "SHA256SUMS"),
        "information_boundary": {
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
            "future_frames_used": False,
            "replacement_allowed": False,
        },
    }
    receipt["audit_id"] = _content_id(receipt)

    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source / "metric-support", output / "metric-support")
    shutil.copy2(pipeline_path, output / "pipeline-receipt.json")
    receipt_path = output / "independent-audit-receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    files = sorted(path for path in output.rglob("*") if path.is_file())
    checksum_lines = [
        f"{_sha256_file(path)}  {path.relative_to(output).as_posix()}" for path in files
    ]
    (output / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n", encoding="ascii"
    )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--auditor-revision", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = audit_support_negative(
        args.source_root,
        args.output_dir,
        auditor_revision=args.auditor_revision,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
