"""Non-sensitive completion records for direct Deform360 calibration runs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ._deform360_calibration_run_common import (
    DEFORM360_CALIBRATION_DOWNLOAD_SCHEMA,
    DEFORM360_CALIBRATION_SOURCE_PLAN_SCHEMA,
    DEFORM360_CALIBRATION_SOURCE_PROTOCOL_ID,
    DEFORM360_CALIBRATION_SOURCE_RESULT_SCHEMA,
    DEFORM360_CALIBRATION_SOURCE_RUN_SCHEMA,
    DEFORM360_DATASET_REVISION,
    canonical_sha256 as _canonical_sha256,
)
from ._deform360_calibration_source_run_record_impl import (
    build_deform360_calibration_source_run_record,
    main,
    save_deform360_calibration_source_run_record as _save_run_record,
)
from ._deform360_calibration_source_run_record_validation import (
    load_deform360_calibration_source_run_record,
    validate_deform360_calibration_source_run_record,
)


def save_deform360_calibration_source_run_record(
    record: Mapping[str, Any],
    path: Path,
) -> None:
    """Validate the complete contract before atomic non-replacing publication."""

    validated = validate_deform360_calibration_source_run_record(record)
    _save_run_record(validated, path)


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
