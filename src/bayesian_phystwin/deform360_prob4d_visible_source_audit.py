"""Independent compact-artifact audit for the frozen Deform360 Prob4D v2 run."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

from ._portable_contracts import (
    content_id,
    exact_revision,
    require_exact_fields,
    sha256_digest,
)
from .deform360_prob4d_camera_eligibility import (
    SUPPORT_NEGATIVE_REASON,
    VISIBLE_STREAM_PLAN_SEMANTICS,
    VISIBLE_STREAM_PLAN_VERSION,
)

AUDIT_SCHEMA: Final = (
    "bayesian-phystwin.deform360-prob4d-visible-source-independent-audit"
)
AUDIT_VERSION: Final = 1
EXPECTED_SOURCE_RUN_ID: Final = 31301431579
EXPECTED_SOURCE_RUN_ATTEMPT: Final = 1
EXPECTED_SOURCE_HEAD_SHA: Final = "136f72b996e9c76b0bab3ab5db5d0fe7172e0307"
EXPECTED_SOURCE_ARTIFACT_ID: Final = 9034737368
EXPECTED_SOURCE_ARTIFACT_NAME: Final = "deform360-prob4d-source-gate-31301431579-1"
EXPECTED_SOURCE_ARTIFACT_DIGEST: Final = (
    "sha256:caa8d5ea887ec5273c306dd8de59d57056181ef139c98ac7acb76185032a3828"
)
EXPECTED_PRODUCTION_RESULT_ID: Final = (
    "146f885351b2af0134b8b3d3c28a76deaa899749b1b1306e0d7061807ae95f89"
)
EXPECTED_ADMISSION_ID: Final = (
    "715ab8479bad4d97eba766cdba1a161f1f6e83e3fd597bb09a2bf8ab8dc91e15"
)
EXPECTED_PROB4D_REVISION: Final = "25d90ef7f78ba4307f4555cb636d666004e1bf66"
EXPECTED_MOTIONCRAFTER_REVISION: Final = "9cb4e9679f5f34e249945544052464ef46324bc2"
EXPECTED_CAMERA_POLICY_ID: Final = (
    "1540e20e847d9877a54ca7a1cdc5290f542de25c1e779c4cf145532f9dd3b9d0"
)
EXPECTED_METRIC_BATCH_RESULT_ID: Final = (
    "2e7a16ce502ac877f56457809683f5b30d40eee5ed290547043010eeed1fefa6"
)
EXPECTED_PLAN_ID: Final = (
    "beb40127dc5d673f9a236550e90b6e38924067a667a885228fd4cf8496c20cc4"
)
EXPECTED_SAMPLE_STDERR_SHA256: Final = (
    "5da90e87f5cd814b48200f7a978309a634b7d2a5adddf6aefec1a045ac4e5b7d"
)
EXPECTED_EMPTY_SHA256: Final = hashlib.sha256(b"").hexdigest()

_SOURCE_FILES = frozenset(
    {
        "SHA256SUMS",
        "pipeline-receipt.json",
        "metric-support/metric-batch-result.json",
        "metric-support/metric-prefix-plan.json",
        "metric-support/support-receipt.json",
    }
)
_PIPELINE_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "implementation_revision",
        "eligibility_contract",
        "camera_eligibility_policy_id",
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
_SUPPORT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "implementation_revision",
        "eligibility_contract",
        "camera_eligibility_policy_id",
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
_RESULT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
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
        "result_id",
    }
)
_PLAN_FIELDS = frozenset(
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
_PIPELINE_BOUNDARY: Final = {
    "public_released_measurements_used": True,
    "new_measurements_required": False,
    "human_approval_required": False,
    "confirmation_payloads_opened": False,
    "target_outcomes_used": False,
    "future_frames_used": False,
    "replacement_allowed": False,
}
_METRIC_BOUNDARY: Final = {
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
_PLAN_BOUNDARY: Final = {
    "calibration_payloads_opened": True,
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
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _verify_source_checksums(root: Path) -> dict[str, str]:
    _require(root.is_dir(), "compact source artifact is missing")
    _require(
        not any(path.is_symlink() for path in root.rglob("*")),
        "compact source artifact contains a symbolic link",
    )
    files = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    _require(files == _SOURCE_FILES, "compact source artifact roster changed")
    lines = (root / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    records: dict[str, str] = {}
    for line in lines:
        digest, separator, relative = line.partition("  ")
        _require(bool(separator), "compact checksum record is malformed")
        digest = sha256_digest(digest, name="compact checksum")
        path = PurePosixPath(relative)
        _require(
            not path.is_absolute()
            and ".." not in path.parts
            and relative != "SHA256SUMS",
            "compact checksum path is unsafe",
        )
        _require(relative not in records, "compact checksum path is repeated")
        records[relative] = digest
    expected = _SOURCE_FILES - {"SHA256SUMS"}
    _require(set(records) == expected, "compact checksum roster changed")
    for relative, expected_digest in records.items():
        _require(
            _sha256_file(root / PurePosixPath(relative)) == expected_digest,
            f"compact artifact member changed: {relative}",
        )
    return records


def _validate_pipeline(value: Mapping[str, Any]) -> None:
    require_exact_fields(value, expected=_PIPELINE_FIELDS, name="pipeline receipt")
    _require(
        value["schema"] == "bayesian-phystwin.deform360-prob4d-source-pipeline-receipt"
        and value["schema_version"] == 1
        and value["implementation_revision"] == EXPECTED_SOURCE_HEAD_SHA
        and value["eligibility_contract"] == "v2-target-free-visible"
        and value["camera_eligibility_policy_id"] == EXPECTED_CAMERA_POLICY_ID
        and value["visual_production_result_id"] == EXPECTED_PRODUCTION_RESULT_ID
        and value["prob4d_revision"] == EXPECTED_PROB4D_REVISION
        and value["motioncrafter_revision"] == EXPECTED_MOTIONCRAFTER_REVISION,
        "source pipeline identity changed",
    )
    _require(
        value["stage_outcomes"]
        == {
            "metric_batch": "success",
            "support_gate": "success",
            "samples": "failure",
            "calibration": "skipped",
            "source_gate": "skipped",
        },
        "source pipeline terminal order changed",
    )
    _require(
        value["stderr_sha256"]
        == {
            "metric-batch": EXPECTED_EMPTY_SHA256,
            "samples": EXPECTED_SAMPLE_STDERR_SHA256,
            "calibration": EXPECTED_EMPTY_SHA256,
            "source-gate": EXPECTED_EMPTY_SHA256,
        },
        "source pipeline stderr accounting changed",
    )
    _require(
        value["source_gate_result_id"] is None
        and value["source_gate_passed"] is None
        and value["confirmation_access_authorized"] is None,
        "technical terminal contains a source-gate decision",
    )
    _require(
        value["information_boundary"] == _PIPELINE_BOUNDARY,
        "source pipeline information boundary changed",
    )


def _validate_support(value: Mapping[str, Any]) -> None:
    require_exact_fields(value, expected=_SUPPORT_FIELDS, name="support receipt")
    _require(
        value["schema"] == "bayesian-phystwin.deform360-prob4d-source-support-receipt"
        and value["schema_version"] == 1
        and value["implementation_revision"] == EXPECTED_SOURCE_HEAD_SHA
        and value["eligibility_contract"] == "v2-target-free-visible"
        and value["camera_eligibility_policy_id"] == EXPECTED_CAMERA_POLICY_ID
        and value["metric_batch_step_outcome"] == "success"
        and value["metric_batch_result_id"] == EXPECTED_METRIC_BATCH_RESULT_ID
        and value["metric_batch_status"] == "target-free-visible-streams-supported"
        and value["metric_batch_stderr_sha256"] == EXPECTED_EMPTY_SHA256
        and value["new_measurements_required"] is False
        and value["human_approval_required"] is False
        and value["confirmation_payloads_opened"] is False
        and value["target_outcomes_used"] is False,
        "target-free support receipt changed",
    )


def _validate_metric_result(value: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    require_exact_fields(value, expected=_RESULT_FIELDS, name="metric batch result")
    declared_id = sha256_digest(value["result_id"], name="metric batch result ID")
    _require(
        content_id({key: item for key, item in value.items() if key != "result_id"})
        == declared_id
        == EXPECTED_METRIC_BATCH_RESULT_ID,
        "metric batch result ID changed",
    )
    _require(
        value["schema"] == "bayesian-phystwin.deform360-prob4d-metric-batch"
        and value["schema_version"] == 2
        and value["semantics"]
        == "target-free-robot-visible-calibration-streams-released-robot-gauge-v2"
        and value["implementation_revision"] == EXPECTED_SOURCE_HEAD_SHA
        and value["production_result_id"] == EXPECTED_PRODUCTION_RESULT_ID
        and value["admission_id"] == EXPECTED_ADMISSION_ID
        and value["object_count"] == 10
        and value["admitted_stream_count"] == 324
        and value["supported_stream_count"] == 313
        and value["support_negative_stream_count"] == 11
        and value["technical_failure_stream_count"] == 0
        and value["supported_object_count"] == 10
        and value["plan_emitted"] is True
        and value["status"] == "target-free-visible-streams-supported",
        "metric batch accounting changed",
    )
    _require(
        value["information_boundary"] == _METRIC_BOUNDARY,
        "metric batch information boundary changed",
    )
    source_artifacts = value["source_artifacts"]
    _require(isinstance(source_artifacts, Mapping), "source artifacts are invalid")
    _require(
        set(source_artifacts)
        == {
            "prepared-source-inventory.json",
            "visual-production-result.json",
            "selection.json",
            "visual-provider-spec.json",
            "metric-prior-policy.json",
            "camera-eligibility-policy.json",
        },
        "metric batch source-artifact roster changed",
    )
    for name, digest in source_artifacts.items():
        sha256_digest(digest, name=f"source artifact {name}")

    jobs = value["jobs"]
    _require(isinstance(jobs, Sequence), "metric batch jobs are invalid")
    observed: dict[str, Mapping[str, Any]] = {}
    ordering: list[tuple[str, str, str]] = []
    counts: Counter[str] = Counter()
    supported_objects: set[str] = set()
    for raw in jobs:
        _require(isinstance(raw, Mapping), "metric batch job is invalid")
        row = cast(Mapping[str, Any], raw)
        job_id = sha256_digest(row["job_id"], name="metric batch job ID")
        object_id = str(row["object_id"])
        camera_id = str(row["camera_id"])
        status = str(row["status"])
        _require(job_id not in observed, "metric batch job is repeated")
        _require(
            status in {"supported", "support-negative"},
            "metric batch contains an inadmissible job status",
        )
        if status == "supported":
            _require(
                row["failure_reason"] is None
                and row["failure_detail_sha256"] is None
                and row["metric_artifact_id"] is not None
                and int(row["projected_point_count"]) >= 1,
                "supported metric job changed",
            )
            supported_objects.add(object_id)
        else:
            _require(
                row["failure_reason"] == SUPPORT_NEGATIVE_REASON
                and row["failure_detail_sha256"] is None
                and row["metric_artifact_id"] is None
                and row["projected_point_count"] == 0,
                "visibility exclusion changed",
            )
        observed[job_id] = row
        counts[status] += 1
        ordering.append((object_id, camera_id, job_id))
    _require(
        len(observed) == 324
        and ordering == sorted(ordering)
        and counts == Counter({"supported": 313, "support-negative": 11})
        and len(supported_objects) == 10,
        "metric batch job partition changed",
    )
    return observed


def _validate_plan(
    value: Mapping[str, Any],
    *,
    result: Mapping[str, Any],
    jobs: Mapping[str, Mapping[str, Any]],
    plan_path: Path,
) -> None:
    require_exact_fields(value, expected=_PLAN_FIELDS, name="metric-prefix plan")
    declared_id = sha256_digest(value["plan_id"], name="metric-prefix plan ID")
    _require(
        content_id({key: item for key, item in value.items() if key != "plan_id"})
        == declared_id
        == EXPECTED_PLAN_ID,
        "metric-prefix plan ID changed",
    )
    _require(
        value["schema"] == "bayesian-phystwin.deform360-prob4d-metric-prefix-plan"
        and value["schema_version"] == VISIBLE_STREAM_PLAN_VERSION
        and value["semantics"] == VISIBLE_STREAM_PLAN_SEMANTICS
        and value["camera_eligibility_policy_id"] == EXPECTED_CAMERA_POLICY_ID
        and value["visual_production_result_id"] == EXPECTED_PRODUCTION_RESULT_ID
        and value["prob4d_revision"] == EXPECTED_PROB4D_REVISION
        and value["motioncrafter_revision"] == EXPECTED_MOTIONCRAFTER_REVISION
        and value["information_boundary"] == _PLAN_BOUNDARY,
        "metric-prefix plan identity or boundary changed",
    )
    source_artifacts = cast(Mapping[str, Any], result["source_artifacts"])
    _require(
        value["camera_eligibility_policy_file_sha256"]
        == source_artifacts["camera-eligibility-policy.json"],
        "camera eligibility policy binding changed",
    )
    plan_record = result["plan_file"]
    _require(isinstance(plan_record, Mapping), "metric batch plan record is invalid")
    _require(
        plan_record["path"] == "metric-prefix-plan.json"
        and plan_record["sha256"] == _sha256_file(plan_path)
        and plan_record["byte_count"] == plan_path.stat().st_size,
        "metric-prefix plan file record changed",
    )

    cases = value["cases"]
    exclusions = value["excluded_streams"]
    _require(
        isinstance(cases, Sequence)
        and isinstance(exclusions, Sequence)
        and len(cases) == 10
        and len(exclusions) == 11,
        "metric-prefix plan roster changed",
    )
    included_ids: set[str] = set()
    included_order: list[tuple[str, int]] = []
    case_identity: dict[str, tuple[int, str]] = {}
    for raw_case in cases:
        _require(isinstance(raw_case, Mapping), "metric-prefix case is invalid")
        case = cast(Mapping[str, Any], raw_case)
        object_id = str(case["object_id"])
        episode_id = int(case["episode_id"])
        stratum = str(case["stratum"])
        streams = case["streams"]
        _require(
            isinstance(streams, Sequence) and len(streams) >= 2,
            "metric-prefix case has too few streams",
        )
        _require(object_id not in case_identity, "metric-prefix object is repeated")
        case_identity[object_id] = (episode_id, stratum)
        included_order.append((object_id, episode_id))
        stream_order: list[tuple[str, str]] = []
        for raw_stream in streams:
            _require(isinstance(raw_stream, Mapping), "metric-prefix stream is invalid")
            stream = cast(Mapping[str, Any], raw_stream)
            job_id = sha256_digest(stream["job_id"], name="included job ID")
            camera_id = str(stream["camera_id"])
            _require(job_id not in included_ids, "included stream is repeated")
            row = jobs.get(job_id)
            _require(
                row is not None
                and row["status"] == "supported"
                and row["object_id"] == object_id
                and row["camera_id"] == camera_id,
                "included stream differs from metric batch",
            )
            included_ids.add(job_id)
            stream_order.append((camera_id, job_id))
        _require(stream_order == sorted(stream_order), "included streams are unsorted")
    _require(
        included_order == sorted(included_order) and len(included_ids) == 313,
        "included stream accounting changed",
    )

    excluded_ids: set[str] = set()
    excluded_order: list[tuple[str, str, str]] = []
    for raw_exclusion in exclusions:
        _require(isinstance(raw_exclusion, Mapping), "excluded stream is invalid")
        exclusion = cast(Mapping[str, Any], raw_exclusion)
        job_id = sha256_digest(exclusion["job_id"], name="excluded job ID")
        object_id = str(exclusion["object_id"])
        camera_id = str(exclusion["camera_id"])
        _require(
            job_id not in excluded_ids
            and exclusion["reason"] == SUPPORT_NEGATIVE_REASON
            and case_identity.get(object_id)
            == (int(exclusion["episode_id"]), str(exclusion["stratum"])),
            "retained visibility exclusion changed",
        )
        row = jobs.get(job_id)
        _require(
            row is not None
            and row["status"] == "support-negative"
            and row["object_id"] == object_id
            and row["camera_id"] == camera_id,
            "excluded stream differs from metric batch",
        )
        excluded_ids.add(job_id)
        excluded_order.append((object_id, camera_id, job_id))
    _require(
        excluded_order == sorted(excluded_order)
        and included_ids.isdisjoint(excluded_ids)
        and included_ids | excluded_ids == set(jobs),
        "included/excluded production partition changed",
    )


def audit_deform360_prob4d_visible_source_v2(
    *,
    source_root: str | Path,
    output_directory: str | Path,
    validator_revision: str,
    source_run_id: int,
    source_run_attempt: int,
    source_artifact_id: int,
    source_artifact_name: str,
    source_artifact_digest: str,
) -> Mapping[str, Any]:
    """Validate and atomically publish the exact v2 source technical terminal."""

    source = Path(source_root).resolve(strict=True)
    output = Path(output_directory).resolve()
    _require(not output.exists(), "audit output already exists")
    records = _verify_source_checksums(source)
    pipeline_path = source / "pipeline-receipt.json"
    support_path = source / "metric-support/support-receipt.json"
    result_path = source / "metric-support/metric-batch-result.json"
    plan_path = source / "metric-support/metric-prefix-plan.json"
    pipeline = _load_json(pipeline_path, name="pipeline receipt")
    support = _load_json(support_path, name="support receipt")
    result = _load_json(result_path, name="metric batch result")
    plan = _load_json(plan_path, name="metric-prefix plan")
    _validate_pipeline(pipeline)
    _validate_support(support)
    jobs = _validate_metric_result(result)
    _validate_plan(plan, result=result, jobs=jobs, plan_path=plan_path)
    _require(
        support["metric_batch_result_id"] == result["result_id"],
        "support receipt and metric result differ",
    )
    _require(
        int(source_run_id) == EXPECTED_SOURCE_RUN_ID
        and int(source_run_attempt) == EXPECTED_SOURCE_RUN_ATTEMPT
        and int(source_artifact_id) == EXPECTED_SOURCE_ARTIFACT_ID
        and str(source_artifact_name) == EXPECTED_SOURCE_ARTIFACT_NAME
        and str(source_artifact_digest) == EXPECTED_SOURCE_ARTIFACT_DIGEST,
        "source run or compact artifact identity changed",
    )

    identity = {
        "schema": AUDIT_SCHEMA,
        "schema_version": AUDIT_VERSION,
        "validator_revision": exact_revision(
            validator_revision, name="validator revision"
        ),
        "source_workflow_run_id": int(source_run_id),
        "source_workflow_run_attempt": int(source_run_attempt),
        "source_workflow_head_sha": EXPECTED_SOURCE_HEAD_SHA,
        "source_artifact_id": int(source_artifact_id),
        "source_artifact_name": str(source_artifact_name),
        "source_artifact_digest": str(source_artifact_digest),
        "audit_status": "validated-source-sample-materialization-failure",
        "result_kind": "source-pipeline-technical-terminal",
        "terminal_stage": "source-calibration-samples",
        "support_gate_passed": True,
        "source_gate_evaluated": False,
        "source_gate_result_id": None,
        "source_gate_passed": None,
        "confirmation_access_authorized": False,
        "failure_detail_available_in_compact_artifact": False,
        "sample_stderr_sha256": EXPECTED_SAMPLE_STDERR_SHA256,
        "metric_batch_result_id": EXPECTED_METRIC_BATCH_RESULT_ID,
        "metric_prefix_plan_id": EXPECTED_PLAN_ID,
        "admitted_stream_count": 324,
        "supported_stream_count": 313,
        "support_negative_stream_count": 11,
        "technical_failure_stream_count": 0,
        "supported_object_count": 10,
        "eligibility_contract": "v2-target-free-visible",
        "camera_eligibility_policy_id": EXPECTED_CAMERA_POLICY_ID,
        "compact_member_sha256": dict(sorted(records.items())),
        "compact_sha256s_manifest_sha256": _sha256_file(source / "SHA256SUMS"),
        "information_boundary": dict(_PIPELINE_BOUNDARY),
        "claim_boundary": (
            "This audit validates a public source-pipeline technical terminal. "
            "It is not a source-calibration decision, does not authorize "
            "confirmation, and establishes no prediction or state-of-the-art claim."
        ),
    }
    receipt = {**identity, "audit_id": content_id(identity)}
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        receipt_path = temporary / "independent-audit-receipt.json"
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (temporary / "SHA256SUMS").write_text(
            f"{_sha256_file(receipt_path)}  {receipt_path.name}\n",
            encoding="ascii",
        )
        _require(not os.path.lexists(output), "audit output already exists")
        os.rename(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return cast(Mapping[str, Any], receipt)


__all__ = [
    "AUDIT_SCHEMA",
    "AUDIT_VERSION",
    "EXPECTED_CAMERA_POLICY_ID",
    "EXPECTED_METRIC_BATCH_RESULT_ID",
    "EXPECTED_PLAN_ID",
    "EXPECTED_SAMPLE_STDERR_SHA256",
    "EXPECTED_SOURCE_ARTIFACT_DIGEST",
    "EXPECTED_SOURCE_ARTIFACT_ID",
    "EXPECTED_SOURCE_ARTIFACT_NAME",
    "EXPECTED_SOURCE_HEAD_SHA",
    "EXPECTED_SOURCE_RUN_ATTEMPT",
    "EXPECTED_SOURCE_RUN_ID",
    "audit_deform360_prob4d_visible_source_v2",
]
