"""Terminal record construction for direct Deform360 calibration runs."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ._deform360_calibration_artifact_chain import (
    download_summary,
    plan_summary,
    result_summary,
    source_lock_summary,
)
from ._deform360_calibration_run_common import (
    ARTIFACT_CONTRACT_EXIT_CODE,
    DEFORM360_CALIBRATION_SOURCE_RUN_CLAIM_BOUNDARY,
    DEFORM360_CALIBRATION_SOURCE_RUN_SCHEMA,
    DEFORM360_CALIBRATION_SOURCE_RUN_SEMANTICS,
    DEFORM360_CALIBRATION_SOURCE_RUN_VERSION,
    DEFORM360_DATASET_REVISION,
    RECORD_WRITE_EXIT_CODE,
    SUPPORT_GATE_EXIT_CODE,
    canonical_sha256,
    exit_code,
    positive_integer,
    revision,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _execution_outcome(
    *,
    workload_exit_code: int,
    confirmation_boundary_exit_code: int,
    source_locks: Mapping[str, Any],
    plan: Mapping[str, Any],
    download: Mapping[str, Any],
    result: Mapping[str, Any],
) -> tuple[int, str | None]:
    if confirmation_boundary_exit_code != 0:
        return confirmation_boundary_exit_code, "confirmation-boundary"
    if not source_locks["source_locks_valid"]:
        return ARTIFACT_CONTRACT_EXIT_CODE, "source-lock-contract"
    if plan["plan_available"] and not plan["plan_valid"]:
        return ARTIFACT_CONTRACT_EXIT_CODE, "plan-contract"
    if download["download_available"] and not download["download_valid"]:
        return ARTIFACT_CONTRACT_EXIT_CODE, "download-contract"
    if result["result_available"] and not result["result_valid"]:
        return ARTIFACT_CONTRACT_EXIT_CODE, "result-contract"

    if not plan["plan_valid"]:
        if workload_exit_code != 0:
            return workload_exit_code, "calibration-source-workload"
        return ARTIFACT_CONTRACT_EXIT_CODE, "plan-contract"

    plan_gate = plan["plan_support_gate"]
    if not isinstance(plan_gate, Mapping):
        return ARTIFACT_CONTRACT_EXIT_CODE, "plan-contract"
    if plan_gate.get("support_passed") is False:
        expected_gate_exit = (
            workload_exit_code == SUPPORT_GATE_EXIT_CODE
            and not download["download_available"]
            and not result["result_available"]
        )
        if expected_gate_exit:
            return SUPPORT_GATE_EXIT_CODE, "calibration-source-admission-gate"
        return ARTIFACT_CONTRACT_EXIT_CODE, "plan-contract"

    if not download["download_valid"]:
        if workload_exit_code != 0 and not result["result_available"]:
            return workload_exit_code, "calibration-source-workload"
        return ARTIFACT_CONTRACT_EXIT_CODE, "download-contract"

    if not result["result_valid"]:
        if workload_exit_code != 0:
            return workload_exit_code, "calibration-source-workload"
        return ARTIFACT_CONTRACT_EXIT_CODE, "result-contract"

    support_gate = result["support_gate"]
    if not isinstance(support_gate, Mapping):
        return ARTIFACT_CONTRACT_EXIT_CODE, "result-contract"
    support_passed = support_gate.get("support_passed")
    expected_workload = 0 if support_passed is True else SUPPORT_GATE_EXIT_CODE
    if workload_exit_code != expected_workload:
        return ARTIFACT_CONTRACT_EXIT_CODE, "result-contract"
    if support_passed is False:
        return SUPPORT_GATE_EXIT_CODE, "calibration-source-support-gate"
    return 0, None


def build_deform360_calibration_source_run_record(
    *,
    source_revision: str,
    processing_revision: str,
    workflow_run_id: int,
    workflow_run_attempt: int,
    workload_exit_code: int,
    confirmation_boundary_exit_code: int,
    source_protocol_json: Path,
    stage0_protocol_json: Path,
    selection_lock: Path,
    visual_provider_lock: Path,
    plan_json: Path,
    download_json: Path,
    result_json: Path,
) -> dict[str, Any]:
    """Build one portable record without paths, object IDs, or target outcomes."""

    source_revision = revision(source_revision, name="source_revision")
    processing_revision = revision(
        processing_revision,
        name="processing_revision",
    )
    workflow_run_id = positive_integer(workflow_run_id, name="workflow_run_id")
    workflow_run_attempt = positive_integer(
        workflow_run_attempt,
        name="workflow_run_attempt",
    )
    workload_exit_code = exit_code(
        workload_exit_code,
        name="workload_exit_code",
    )
    confirmation_boundary_exit_code = exit_code(
        confirmation_boundary_exit_code,
        name="confirmation_boundary_exit_code",
    )
    source_locks, expected_units, confirmation_ids = source_lock_summary(
        source_protocol_json=source_protocol_json,
        stage0_protocol_json=stage0_protocol_json,
        selection_lock=selection_lock,
        visual_provider_lock=visual_provider_lock,
        processing_revision=processing_revision,
    )
    plan, identities, planned_ids = plan_summary(
        plan_json,
        processing_revision=processing_revision,
        source_locks=source_locks,
        expected_units=expected_units,
        confirmation_ids=confirmation_ids,
    )
    download = download_summary(
        download_json,
        plan_sha256=plan["plan_sha256"],
        planned_ids=planned_ids,
        confirmation_ids=confirmation_ids,
    )
    result = result_summary(
        result_json,
        processing_revision=processing_revision,
        plan_sha256=plan["plan_sha256"],
        download_sha256=download["download_sha256"],
        expected_identities=identities,
        planned_ids=planned_ids,
    )
    confirmation_boundary_verified = confirmation_boundary_exit_code == 0
    effective_exit_code, failure_stage = _execution_outcome(
        workload_exit_code=workload_exit_code,
        confirmation_boundary_exit_code=confirmation_boundary_exit_code,
        source_locks=source_locks,
        plan=plan,
        download=download,
        result=result,
    )
    record: dict[str, Any] = {
        "schema": DEFORM360_CALIBRATION_SOURCE_RUN_SCHEMA,
        "schema_version": DEFORM360_CALIBRATION_SOURCE_RUN_VERSION,
        "semantics": DEFORM360_CALIBRATION_SOURCE_RUN_SEMANTICS,
        "record_sha256": None,
        "status": "succeeded" if effective_exit_code == 0 else "failed",
        "exit_code": effective_exit_code,
        "workload_exit_code": workload_exit_code,
        "failure_stage": failure_stage,
        "confirmation_boundary_exit_code": confirmation_boundary_exit_code,
        "confirmation_boundary_verified": confirmation_boundary_verified,
        "confirmation_payloads_opened": (
            False if confirmation_boundary_verified else None
        ),
        "source_revision": source_revision,
        "dataset_revision": DEFORM360_DATASET_REVISION,
        "processing_revision": processing_revision,
        "workflow_run_id": workflow_run_id,
        "workflow_run_attempt": workflow_run_attempt,
        **source_locks,
        **plan,
        **download,
        **result,
        "claim_boundary": DEFORM360_CALIBRATION_SOURCE_RUN_CLAIM_BOUNDARY,
    }
    record["record_sha256"] = canonical_sha256(record)
    return record


def save_deform360_calibration_source_run_record(
    record: Mapping[str, Any],
    path: Path,
) -> None:
    """Publish a run record atomically without replacing an earlier record."""

    expected = record.get("record_sha256")
    if type(expected) is not str or _SHA256_RE.fullmatch(expected) is None:
        raise ValueError("record_sha256 is invalid")
    if expected != canonical_sha256(record):
        raise ValueError("record_sha256 does not match record content")
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.tmp-",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--processing-revision", required=True)
    parser.add_argument("--workflow-run-id", type=int, required=True)
    parser.add_argument("--workflow-run-attempt", type=int, required=True)
    parser.add_argument("--workload-exit-code", type=int, required=True)
    parser.add_argument(
        "--confirmation-boundary-exit-code",
        type=int,
        required=True,
    )
    parser.add_argument("--source-protocol-json", type=Path, required=True)
    parser.add_argument("--stage0-protocol-json", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--visual-provider-lock", type=Path, required=True)
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--download-json", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        record = build_deform360_calibration_source_run_record(
            source_revision=args.source_revision,
            processing_revision=args.processing_revision,
            workflow_run_id=args.workflow_run_id,
            workflow_run_attempt=args.workflow_run_attempt,
            workload_exit_code=args.workload_exit_code,
            confirmation_boundary_exit_code=args.confirmation_boundary_exit_code,
            source_protocol_json=args.source_protocol_json,
            stage0_protocol_json=args.stage0_protocol_json,
            selection_lock=args.selection_lock,
            visual_provider_lock=args.visual_provider_lock,
            plan_json=args.plan_json,
            download_json=args.download_json,
            result_json=args.result_json,
        )
        save_deform360_calibration_source_run_record(record, args.output)
    except (OSError, TypeError, ValueError) as error:
        print(f"cannot publish calibration-source run record: {error}", file=sys.stderr)
        return RECORD_WRITE_EXIT_CODE
    print(json.dumps(record, indent=2, sort_keys=True, allow_nan=False))
    return int(record["exit_code"])
