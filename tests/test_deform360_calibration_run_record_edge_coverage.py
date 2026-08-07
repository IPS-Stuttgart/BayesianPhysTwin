"""Focused edge coverage for Deform360 calibration terminal contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bayesian_phystwin._deform360_calibration_artifact_chain import (
    _selection_units,
)
from bayesian_phystwin._deform360_calibration_run_common import (
    validated_support_gate,
)
from bayesian_phystwin._deform360_calibration_source_run_record_validation import (
    _validate_artifact_summary,
    _validate_source_lock_summary,
    _validated_record_gate,
)
from test_deform360_calibration_source_run_record import (
    _build_chain,
    _record,
    _rewrite,
    _units,
)


def test_selection_rejects_metadata_path_for_another_object() -> None:
    rows = _units(
        prefix="cal",
        sheet=5,
        volumetric=5,
        digest_offset=0,
    )
    rows[0]["metadata_path"] = "raw/another-object/metadata.json"

    with pytest.raises(ValueError, match="metadata path changed"):
        _selection_units(
            rows,
            expected_count=10,
            expected_per_stratum=5,
            name="calibration selection",
        )


def test_source_protocol_rejects_nonmapping_provider_locks(
    tmp_path: Path,
) -> None:
    chain = _build_chain(tmp_path)
    source_protocol = json.loads(
        chain.source_protocol_path.read_text(encoding="utf-8")
    )
    source_protocol["locks"] = []
    _rewrite(
        chain.source_protocol_path,
        source_protocol,
        digest_key="protocol_sha256",
    )

    record = _record(chain, workload_exit_code=1)

    assert record["source_locks_valid"] is False
    assert record["failure_stage"] == "source-lock-contract"


def test_unsupported_plan_row_requires_retained_errors(tmp_path: Path) -> None:
    chain = _build_chain(
        tmp_path,
        planned_sheet=4,
        planned_volumetric=4,
        prepared_sheet=4,
        prepared_volumetric=4,
    )
    plan = json.loads(chain.plan_path.read_text(encoding="utf-8"))
    unsupported = next(
        row
        for row in plan["objects"]
        if row["status"] == "unsupported_without_replacement"
    )
    unsupported["errors"] = []
    _rewrite(chain.plan_path, plan, digest_key="plan_sha256")

    record = _record(chain, workload_exit_code=1)

    assert record["plan_valid"] is False
    assert record["failure_stage"] == "plan-contract"


def test_download_rejects_lfs_identity_mismatch(tmp_path: Path) -> None:
    chain = _build_chain(tmp_path)
    download = json.loads(chain.download_path.read_text(encoding="utf-8"))
    lfs_file = next(
        row for row in download["files"] if row["lfs_sha256"] is not None
    )
    lfs_file["downloaded_sha256"] = "e" * 64
    _rewrite(chain.download_path, download, digest_key="download_sha256")

    record = _record(chain, workload_exit_code=1)

    assert record["download_valid"] is False
    assert record["failure_stage"] == "download-contract"


def test_result_accepts_frozen_unsupported_rows_and_rejects_reclassification(
    tmp_path: Path,
) -> None:
    chain = _build_chain(
        tmp_path,
        planned_sheet=4,
        planned_volumetric=4,
        prepared_sheet=4,
        prepared_volumetric=4,
    )

    assert _record(chain)["status"] == "succeeded"

    result = json.loads(chain.result_path.read_text(encoding="utf-8"))
    prepared = next(
        row for row in result["objects"] if row["status"] == "source_prepared"
    )
    prepared["status"] = "unsupported_without_replacement"
    prepared["errors"] = ["source support unavailable"]
    _rewrite(chain.result_path, result, digest_key="result_sha256")

    record = _record(chain, workload_exit_code=1)

    assert record["result_valid"] is False
    assert record["failure_stage"] == "result-contract"


def test_support_gate_rejects_changed_stratum_shape() -> None:
    gate = {
        "supported_object_count": 8,
        "supported_by_stratum": [],
        "minimum_supported_objects": 8,
        "minimum_supported_per_stratum": 4,
        "support_passed": True,
    }

    with pytest.raises(ValueError, match="supported_by_stratum changed"):
        validated_support_gate(
            {"gate": gate},
            artifact="test",
            object_supported=8,
            object_supported_by_stratum={"sheet": 4, "volumetric": 4},
        )


def test_record_gate_rejects_changed_stratum_shape() -> None:
    with pytest.raises(ValueError, match="supported_by_stratum changed"):
        _validated_record_gate(
            {"supported_by_stratum": []},
            name="support_gate",
        )


def test_source_lock_summary_requires_boolean_flags() -> None:
    record = {
        "source_locks_available": 1,
        "source_locks_valid": False,
        "source_locks_error": "missing",
    }

    with pytest.raises(ValueError, match="flags must be booleans"):
        _validate_source_lock_summary(record)


def test_artifact_summary_requires_boolean_flags() -> None:
    record = {
        "plan_available": 1,
        "plan_valid": False,
        "plan_error": "missing",
        "plan_file_sha256": None,
        "plan_sha256": None,
        "plan_support_gate": None,
    }

    with pytest.raises(ValueError, match="flags must be booleans"):
        _validate_artifact_summary(
            record,
            prefix="plan",
            gate_key="plan_support_gate",
        )
