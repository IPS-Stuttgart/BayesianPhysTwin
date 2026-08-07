"""Non-sensitive completion records for direct Deform360 calibration runs."""

# ruff: noqa: I001

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import _deform360_calibration_source_run_record_impl as _impl
from ._deform360_calibration_run_common import (
    DEFORM360_CALIBRATION_DOWNLOAD_SCHEMA,
    DEFORM360_CALIBRATION_SOURCE_PLAN_SCHEMA,
    DEFORM360_CALIBRATION_SOURCE_PROTOCOL_ID,
    DEFORM360_CALIBRATION_SOURCE_RESULT_SCHEMA,
    DEFORM360_CALIBRATION_SOURCE_RUN_SCHEMA,
    DEFORM360_DATASET_REVISION,
    RECORD_WRITE_EXIT_CODE,
    canonical_sha256 as _canonical_sha256,
)
from ._deform360_calibration_source_run_record_validation import (
    load_deform360_calibration_source_run_record,
    validate_deform360_calibration_source_run_record,
)

build_deform360_calibration_source_run_record = (
    _impl.build_deform360_calibration_source_run_record
)
_save_run_record = _impl.save_deform360_calibration_source_run_record


def save_deform360_calibration_source_run_record(
    record: Mapping[str, Any],
    path: Path,
) -> None:
    """Validate the complete contract before atomic non-replacing publication."""

    validated = validate_deform360_calibration_source_run_record(record)
    _save_run_record(validated, path)


def main(argv: Sequence[str] | None = None) -> int:
    """Build, strictly validate, and publish one terminal execution record."""

    args = _impl.build_parser().parse_args(argv)
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


__all__ = [
    "DEFORM360_CALIBRATION_DOWNLOAD_SCHEMA",
    "DEFORM360_CALIBRATION_SOURCE_PLAN_SCHEMA",
    "DEFORM360_CALIBRATION_SOURCE_PROTOCOL_ID",
    "DEFORM360_CALIBRATION_SOURCE_RESULT_SCHEMA",
    "DEFORM360_CALIBRATION_SOURCE_RUN_SCHEMA",
    "DEFORM360_DATASET_REVISION",
    "_canonical_sha256",
    "build_deform360_calibration_source_run_record",
    "load_deform360_calibration_source_run_record",
    "main",
    "save_deform360_calibration_source_run_record",
    "validate_deform360_calibration_source_run_record",
]


if __name__ == "__main__":
    raise SystemExit(main())
