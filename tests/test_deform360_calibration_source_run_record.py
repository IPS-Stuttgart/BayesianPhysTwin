"""Adversarial contracts for direct Deform360 calibration run records."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from bayesian_phystwin._deform360_calibration_run_common import (
    DEFORM360_CALIBRATION_DOWNLOAD_SCHEMA,
    DEFORM360_CALIBRATION_SOURCE_PLAN_SCHEMA,
    DEFORM360_CALIBRATION_SOURCE_PROTOCOL_ID,
    DEFORM360_CALIBRATION_SOURCE_PROTOCOL_SCHEMA,
    DEFORM360_CALIBRATION_SOURCE_RESULT_SCHEMA,
    DEFORM360_DATASET_REPOSITORY,
    DEFORM360_DATASET_REVISION,
    DEFORM360_EXPECTED_TACTILE_BASELINE_POLICY,
    DEFORM360_PARENT_PROTOCOL_ID,
    DEFORM360_PROCESSING_REPOSITORY,
    DEFORM360_STAGE0_SELECTION_SCHEMA,
    DEFORM360_VISUAL_PROVIDER_LOCK_SCHEMA,
    DEFORM360_VISUAL_PROVIDER_LOCK_SEMANTICS,
    canonical_sha256,
    content_sha256,
)
from bayesian_phystwin.deform360_calibration_source_run_record import (
    DEFORM360_CALIBRATION_SOURCE_RUN_SCHEMA,
    _canonical_sha256,
    build_deform360_calibration_source_run_record,
    main,
    save_deform360_calibration_source_run_record,
)

SOURCE_REVISION = "1" * 40
PROCESSING_REVISION = "d8522a4403b766aeb387510c04e89032a56fdf35"


@dataclass
class Chain:
    source_protocol_path: Path
    stage0_protocol_path: Path
    selection_path: Path
    provider_path: Path
    plan_path: Path
    download_path: Path
    result_path: Path
    source_protocol: dict[str, Any]
    stage0_protocol: dict[str, Any]
    selection: dict[str, Any]
    provider: dict[str, Any]
    plan: dict[str, Any]
    download: dict[str, Any]
    result: dict[str, Any]


def _write(path: Path, value: dict[str, Any]) -> str:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.write_text(encoded, encoding="utf-8")
    import hashlib

    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _gate(sheet: int, volumetric: int) -> dict[str, Any]:
    supported = sheet + volumetric
    return {
        "supported_object_count": supported,
        "supported_by_stratum": {
            "sheet": sheet,
            "volumetric": volumetric,
        },
        "minimum_supported_objects": 8,
        "minimum_supported_per_stratum": 4,
        "support_passed": (
            supported >= 8 and sheet >= 4 and volumetric >= 4
        ),
    }


def _units(
    *,
    prefix: str,
    sheet: int,
    volumetric: int,
    digest_offset: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stratum, count in (("sheet", sheet), ("volumetric", volumetric)):
        for index in range(count):
            object_id = f"{prefix}-{stratum}-{index}"
            rows.append(
                {
                    "object_id": object_id,
                    "episode_id": index,
                    "stratum": stratum,
                    "metadata_path": f"raw/{object_id}/metadata.json",
                    "metadata_sha256": f"{digest_offset + index + 1:064x}",
                }
            )
    return rows


def _selection_lock(
    stage0_protocol: dict[str, Any],
    calibration: list[dict[str, Any]],
    confirmation: list[dict[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": DEFORM360_STAGE0_SELECTION_SCHEMA,
        "schema_version": 1,
        "protocol_id": DEFORM360_PARENT_PROTOCOL_ID,
        "protocol_sha256": content_sha256(stage0_protocol),
        "dataset": {
            "repo_id": DEFORM360_DATASET_REPOSITORY,
            "resolved_revision": DEFORM360_DATASET_REVISION,
        },
        "official_processing": {
            "repository": DEFORM360_PROCESSING_REPOSITORY,
            "revision": PROCESSING_REVISION,
        },
        "replacement_allowed_after_payload_access": False,
        "selection": {
            "calibration": calibration,
            "confirmation": confirmation,
        },
        "selection_sha256": None,
        "content_selection_sha256": None,
        "selection_artifact_sha256": None,
        "implementation_revision": "2" * 40,
    }
    value["selection_sha256"] = content_sha256(value["selection"])
    content = dict(value)
    content.pop("content_selection_sha256")
    content.pop("implementation_revision")
    content.pop("selection_artifact_sha256")
    value["content_selection_sha256"] = content_sha256(content)
    artifact = dict(value)
    artifact.pop("selection_artifact_sha256")
    value["selection_artifact_sha256"] = content_sha256(artifact)
    return value


def _provider_lock() -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": DEFORM360_VISUAL_PROVIDER_LOCK_SCHEMA,
        "schema_version": 1,
        "semantics": DEFORM360_VISUAL_PROVIDER_LOCK_SEMANTICS,
        "protocol_id": DEFORM360_PARENT_PROTOCOL_ID,
        "selected_raw_payloads_opened": False,
        "target_outcomes_used": False,
        "provider_repository": "IPS-Stuttgart/Prob4D",
    }
    value["artifact_id"] = content_sha256(value)
    return value


def _source_protocol(provider_id: str) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": DEFORM360_CALIBRATION_SOURCE_PROTOCOL_SCHEMA,
        "schema_version": 1,
        "protocol_id": DEFORM360_CALIBRATION_SOURCE_PROTOCOL_ID,
        "parent_protocol_id": DEFORM360_PARENT_PROTOCOL_ID,
        "dataset": {
            "repository": DEFORM360_DATASET_REPOSITORY,
            "revision": DEFORM360_DATASET_REVISION,
        },
        "processing": {
            "repository": DEFORM360_PROCESSING_REPOSITORY,
            "revision": PROCESSING_REVISION,
        },
        "locks": {"visual_provider_lock_id": provider_id},
        "protocol_sha256": None,
    }
    value["protocol_sha256"] = canonical_sha256(
        value,
        digest_key="protocol_sha256",
    )
    return value


def _plan_row(
    unit: dict[str, Any],
    *,
    planned: bool,
) -> dict[str, Any]:
    object_id = unit["object_id"]
    return {
        **unit,
        "status": (
            "planned" if planned else "unsupported_without_replacement"
        ),
        "errors": [] if planned else ["source support unavailable"],
        "camera_streams": [f"camera-{index}" for index in range(8)]
        if planned
        else [],
        "tactile_streams": ["touch-0"] if planned else [],
        "selected_files": [
            {
                "path": unit["metadata_path"],
                "size": 10,
                "blob_id": "a" * 40,
                "lfs_sha256": None,
            },
            {
                "path": f"raw/{object_id}/camera-0/frame.mp4",
                "size": 20,
                "blob_id": "b" * 40,
                "lfs_sha256": "c" * 64,
            },
        ],
    }


def _prepared_row(unit: dict[str, Any]) -> dict[str, Any]:
    return {
        "object_id": unit["object_id"],
        "episode_id": unit["episode_id"],
        "stratum": unit["stratum"],
        "status": "source_prepared",
        "completed_stage": "action-window-selection",
        "synthetic_episode_index": 0,
        "bimanual": False,
        "camera_count": 8,
        "cameras": [f"camera-{index}" for index in range(8)],
        "aligned_frame_count": 81,
        "tactile_sensor_count": 1,
        "tactile_sensors": ["touch-0"],
        "action_window": {"start": 0, "stop": 81},
        "outputs_sha256": {
            "alignment": "1" * 64,
            "undistorted_intrinsics": "2" * 64,
            "extrinsics": "3" * 64,
            "robot": "4" * 64,
            "tactile": {"touch-0": "5" * 64},
        },
    }


def _technical_failure(unit: dict[str, Any]) -> dict[str, Any]:
    return {
        "object_id": unit["object_id"],
        "episode_id": unit["episode_id"],
        "stratum": unit["stratum"],
        "status": "technical_failure_without_replacement",
        "completed_stage": "tactile-align",
        "error": "RuntimeError: synthetic failure",
    }


def _unsupported_result(unit: dict[str, Any]) -> dict[str, Any]:
    return {
        "object_id": unit["object_id"],
        "episode_id": unit["episode_id"],
        "stratum": unit["stratum"],
        "status": "unsupported_without_replacement",
        "errors": ["source support unavailable"],
    }


def _build_chain(
    root: Path,
    *,
    planned_sheet: int = 5,
    planned_volumetric: int = 5,
    prepared_sheet: int = 5,
    prepared_volumetric: int = 5,
) -> Chain:
    root.mkdir(parents=True, exist_ok=True)
    calibration = _units(
        prefix="cal",
        sheet=5,
        volumetric=5,
        digest_offset=0,
    )
    confirmation = _units(
        prefix="confirm",
        sheet=6,
        volumetric=6,
        digest_offset=20,
    )
    stage0_protocol = {
        "protocol_id": DEFORM360_PARENT_PROTOCOL_ID,
        "description": "synthetic Stage-0 protocol",
    }
    stage0_protocol_path = root / "stage0-protocol.json"
    _write(stage0_protocol_path, stage0_protocol)

    selection = _selection_lock(
        stage0_protocol,
        calibration,
        confirmation,
    )
    selection_path = root / "selection.json"
    selection_file_sha256 = _write(selection_path, selection)

    provider = _provider_lock()
    provider_path = root / "provider.json"
    provider_file_sha256 = _write(provider_path, provider)

    source_protocol = _source_protocol(provider["artifact_id"])
    source_protocol_path = root / "source-protocol.json"
    _write(source_protocol_path, source_protocol)

    planned_counts = {
        "sheet": planned_sheet,
        "volumetric": planned_volumetric,
    }
    plan_rows: list[dict[str, Any]] = []
    seen = {"sheet": 0, "volumetric": 0}
    for unit in calibration:
        stratum = unit["stratum"]
        planned = seen[stratum] < planned_counts[stratum]
        seen[stratum] += 1
        plan_rows.append(_plan_row(unit, planned=planned))
    plan: dict[str, Any] = {
        "schema": DEFORM360_CALIBRATION_SOURCE_PLAN_SCHEMA,
        "schema_version": 1,
        "protocol_id": DEFORM360_CALIBRATION_SOURCE_PROTOCOL_ID,
        "protocol_sha256": source_protocol["protocol_sha256"],
        "parent_protocol_id": DEFORM360_PARENT_PROTOCOL_ID,
        "selection_source_sha256": selection_file_sha256,
        "visual_provider_lock_id": provider["artifact_id"],
        "visual_provider_source_sha256": provider_file_sha256,
        "dataset_repository": DEFORM360_DATASET_REPOSITORY,
        "dataset_revision": DEFORM360_DATASET_REVISION,
        "processing_repository": DEFORM360_PROCESSING_REPOSITORY,
        "processing_revision": PROCESSING_REVISION,
        "tactile_baseline_policy": dict(
            DEFORM360_EXPECTED_TACTILE_BASELINE_POLICY
        ),
        "objects": plan_rows,
        "gate": _gate(planned_sheet, planned_volumetric),
        "information_boundary": {
            "repository_names_opened": True,
            "calibration_payloads_opened": False,
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
            "replacement_allowed": False,
        },
        "plan_sha256": None,
    }
    plan["plan_sha256"] = canonical_sha256(
        plan,
        digest_key="plan_sha256",
    )
    plan_path = root / "plan.json"
    _write(plan_path, plan)

    planned_units = [
        unit
        for unit, row in zip(calibration, plan_rows, strict=True)
        if row["status"] == "planned"
    ]
    download_files: list[dict[str, Any]] = []
    for row in plan_rows:
        if row["status"] != "planned":
            continue
        for selected in row["selected_files"]:
            downloaded_sha = selected["lfs_sha256"] or "d" * 64
            download_files.append(
                {
                    **selected,
                    "downloaded_size": selected["size"],
                    "downloaded_sha256": downloaded_sha,
                }
            )
    download: dict[str, Any] = {
        "schema": DEFORM360_CALIBRATION_DOWNLOAD_SCHEMA,
        "schema_version": 1,
        "protocol_id": DEFORM360_CALIBRATION_SOURCE_PROTOCOL_ID,
        "plan_sha256": plan["plan_sha256"],
        "dataset_repository": DEFORM360_DATASET_REPOSITORY,
        "dataset_revision": DEFORM360_DATASET_REVISION,
        "data_root": "/not-published",
        "files": download_files,
        "object_ids": sorted(unit["object_id"] for unit in planned_units),
        "information_boundary": {
            "calibration_payloads_opened": True,
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
            "replacement_allowed": False,
        },
        "download_sha256": None,
    }
    download["download_sha256"] = canonical_sha256(
        download,
        digest_key="download_sha256",
    )
    download_path = root / "download.json"
    _write(download_path, download)

    prepared_counts = {
        "sheet": prepared_sheet,
        "volumetric": prepared_volumetric,
    }
    prepared_seen = {"sheet": 0, "volumetric": 0}
    result_rows: list[dict[str, Any]] = []
    for unit, plan_row in zip(calibration, plan_rows, strict=True):
        if plan_row["status"] != "planned":
            result_rows.append(_unsupported_result(unit))
            continue
        stratum = unit["stratum"]
        if prepared_seen[stratum] < prepared_counts[stratum]:
            result_rows.append(_prepared_row(unit))
        else:
            result_rows.append(_technical_failure(unit))
        prepared_seen[stratum] += 1
    result: dict[str, Any] = {
        "schema": DEFORM360_CALIBRATION_SOURCE_RESULT_SCHEMA,
        "schema_version": 1,
        "protocol_id": DEFORM360_CALIBRATION_SOURCE_PROTOCOL_ID,
        "plan_sha256": plan["plan_sha256"],
        "download_sha256": download["download_sha256"],
        "dataset_revision": DEFORM360_DATASET_REVISION,
        "processing_revision": PROCESSING_REVISION,
        "objects": result_rows,
        "gate": _gate(prepared_sheet, prepared_volumetric),
        "information_boundary": {
            "calibration_camera_payloads_opened": True,
            "calibration_tactile_payloads_opened": True,
            "calibration_robot_state_derived": True,
            "calibration_target_metrics_computed": False,
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
            "replacement_allowed": False,
        },
        "result_sha256": None,
    }
    result["result_sha256"] = canonical_sha256(
        result,
        digest_key="result_sha256",
    )
    result_path = root / "result.json"
    _write(result_path, result)
    return Chain(
        source_protocol_path=source_protocol_path,
        stage0_protocol_path=stage0_protocol_path,
        selection_path=selection_path,
        provider_path=provider_path,
        plan_path=plan_path,
        download_path=download_path,
        result_path=result_path,
        source_protocol=source_protocol,
        stage0_protocol=stage0_protocol,
        selection=selection,
        provider=provider,
        plan=plan,
        download=download,
        result=result,
    )


def _record(chain: Chain, **overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "source_revision": SOURCE_REVISION,
        "processing_revision": PROCESSING_REVISION,
        "workflow_run_id": 123,
        "workflow_run_attempt": 2,
        "workload_exit_code": 0,
        "confirmation_boundary_exit_code": 0,
        "source_protocol_json": chain.source_protocol_path,
        "stage0_protocol_json": chain.stage0_protocol_path,
        "selection_lock": chain.selection_path,
        "visual_provider_lock": chain.provider_path,
        "plan_json": chain.plan_path,
        "download_json": chain.download_path,
        "result_json": chain.result_path,
    }
    values.update(overrides)
    return build_deform360_calibration_source_run_record(**values)


def _rewrite(
    path: Path,
    value: dict[str, Any],
    *,
    digest_key: str | None = None,
) -> None:
    if digest_key is not None:
        value[digest_key] = canonical_sha256(value, digest_key=digest_key)
    _write(path, value)


def test_success_record_binds_exact_locks_and_remains_non_sensitive(
    tmp_path: Path,
) -> None:
    chain = _build_chain(tmp_path)

    record = _record(chain)

    assert record["schema"] == DEFORM360_CALIBRATION_SOURCE_RUN_SCHEMA
    assert record["status"] == "succeeded"
    assert record["exit_code"] == 0
    assert record["source_locks_valid"] is True
    assert record["plan_valid"] is True
    assert record["download_valid"] is True
    assert record["result_valid"] is True
    assert record["record_sha256"] == _canonical_sha256(record)
    serialized = json.dumps(record, sort_keys=True)
    assert "cal-sheet" not in serialized
    assert "confirm-sheet" not in serialized
    assert "/not-published" not in serialized
    assert str(tmp_path) not in serialized


def test_tampered_source_lock_fails_before_artifact_interpretation(
    tmp_path: Path,
) -> None:
    chain = _build_chain(tmp_path)
    provider = dict(chain.provider)
    provider["target_outcomes_used"] = True
    provider["artifact_id"] = content_sha256(
        {key: value for key, value in provider.items() if key != "artifact_id"}
    )
    _write(chain.provider_path, provider)

    record = _record(chain, workload_exit_code=1)

    assert record["exit_code"] == 4
    assert record["failure_stage"] == "source-lock-contract"
    assert record["source_locks_valid"] is False


def test_plan_cannot_substitute_the_frozen_cohort(tmp_path: Path) -> None:
    chain = _build_chain(tmp_path)
    plan = json.loads(chain.plan_path.read_text(encoding="utf-8"))
    plan["objects"][0]["object_id"] = "replacement-object"
    _rewrite(chain.plan_path, plan, digest_key="plan_sha256")

    record = _record(chain, workload_exit_code=1)

    assert record["failure_stage"] == "plan-contract"
    assert record["plan_valid"] is False


def test_plan_and_download_reject_confirmation_paths(tmp_path: Path) -> None:
    chain = _build_chain(tmp_path)
    confirmation = chain.selection["selection"]["confirmation"][0]
    plan = json.loads(chain.plan_path.read_text(encoding="utf-8"))
    plan["objects"][0]["selected_files"][0]["path"] = confirmation[
        "metadata_path"
    ]
    _rewrite(chain.plan_path, plan, digest_key="plan_sha256")
    assert _record(chain, workload_exit_code=1)["failure_stage"] == (
        "plan-contract"
    )

    chain = _build_chain(tmp_path / "download-case")
    download = json.loads(chain.download_path.read_text(encoding="utf-8"))
    download["files"][0]["path"] = confirmation["metadata_path"]
    _rewrite(chain.download_path, download, digest_key="download_sha256")
    record = _record(chain, workload_exit_code=1)
    assert record["failure_stage"] == "download-contract"


def test_unplanned_object_cannot_become_prepared(tmp_path: Path) -> None:
    chain = _build_chain(
        tmp_path,
        planned_sheet=4,
        planned_volumetric=4,
        prepared_sheet=4,
        prepared_volumetric=4,
    )
    result = json.loads(chain.result_path.read_text(encoding="utf-8"))
    unsupported = next(
        row
        for row in result["objects"]
        if row["status"] == "unsupported_without_replacement"
    )
    replacement = next(
        row for row in result["objects"] if row["status"] == "source_prepared"
    )
    unsupported.clear()
    unsupported.update(replacement)
    _rewrite(chain.result_path, result, digest_key="result_sha256")

    record = _record(chain, workload_exit_code=1)

    assert record["failure_stage"] == "result-contract"
    assert record["result_valid"] is False


def test_admission_and_preparation_gates_remain_distinct(tmp_path: Path) -> None:
    admission = _build_chain(
        tmp_path / "admission",
        planned_sheet=4,
        planned_volumetric=3,
        prepared_sheet=4,
        prepared_volumetric=3,
    )
    admission.download_path.unlink()
    admission.result_path.unlink()
    record = _record(admission, workload_exit_code=3)
    assert record["failure_stage"] == "calibration-source-admission-gate"

    preparation = _build_chain(
        tmp_path / "preparation",
        prepared_sheet=4,
        prepared_volumetric=3,
    )
    record = _record(preparation, workload_exit_code=3)
    assert record["failure_stage"] == "calibration-source-support-gate"


def test_early_workload_and_boundary_failures_are_retained(tmp_path: Path) -> None:
    chain = _build_chain(tmp_path)
    chain.plan_path.unlink()
    chain.download_path.unlink()
    chain.result_path.unlink()

    workload = _record(chain, workload_exit_code=5)
    assert workload["exit_code"] == 5
    assert workload["failure_stage"] == "calibration-source-workload"
    assert workload["confirmation_payloads_opened"] is False

    boundary = _record(
        chain,
        workload_exit_code=5,
        confirmation_boundary_exit_code=9,
    )
    assert boundary["exit_code"] == 9
    assert boundary["failure_stage"] == "confirmation-boundary"
    assert boundary["confirmation_payloads_opened"] is None


@pytest.mark.parametrize(
    "payload",
    (
        '{"schema":"first","schema":"second"}',
        '{"value":NaN}',
    ),
)
def test_noncanonical_result_json_is_rejected(
    tmp_path: Path,
    payload: str,
) -> None:
    chain = _build_chain(tmp_path)
    chain.result_path.write_text(payload, encoding="utf-8")

    record = _record(chain, workload_exit_code=1)

    assert record["failure_stage"] == "result-contract"
    assert record["result_error"] == "invalid-json"


def test_digest_chain_and_prepared_row_contract_fail_closed(
    tmp_path: Path,
) -> None:
    chain = _build_chain(tmp_path)
    download = json.loads(chain.download_path.read_text(encoding="utf-8"))
    download["plan_sha256"] = "9" * 64
    _rewrite(chain.download_path, download, digest_key="download_sha256")
    assert _record(chain, workload_exit_code=1)["failure_stage"] == (
        "download-contract"
    )

    chain = _build_chain(tmp_path / "prepared-row")
    result = json.loads(chain.result_path.read_text(encoding="utf-8"))
    result["objects"][0]["camera_count"] = 7
    _rewrite(chain.result_path, result, digest_key="result_sha256")
    assert _record(chain, workload_exit_code=1)["failure_stage"] == (
        "result-contract"
    )


def test_atomic_publication_is_durable_and_non_replacing(tmp_path: Path) -> None:
    chain = _build_chain(tmp_path)
    output = tmp_path / "execution-manifest.json"
    record = _record(chain)

    save_deform360_calibration_source_run_record(record, output)
    first = output.read_bytes()
    with pytest.raises(FileExistsError):
        save_deform360_calibration_source_run_record(record, output)
    assert output.read_bytes() == first


def test_concurrent_publication_has_exactly_one_winner(tmp_path: Path) -> None:
    chain = _build_chain(tmp_path)
    output = tmp_path / "execution-manifest.json"
    record = _record(chain)

    def publish() -> bool:
        try:
            save_deform360_calibration_source_run_record(record, output)
        except FileExistsError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: publish(), range(2)))

    assert sorted(outcomes) == [False, True]
    assert json.loads(output.read_text(encoding="utf-8")) == record


def test_cli_writes_the_bound_terminal_record(tmp_path: Path) -> None:
    chain = _build_chain(tmp_path)
    output = tmp_path / "execution-manifest.json"

    status = main(
        [
            "--output",
            str(output),
            "--source-revision",
            SOURCE_REVISION,
            "--processing-revision",
            PROCESSING_REVISION,
            "--workflow-run-id",
            "123",
            "--workflow-run-attempt",
            "1",
            "--workload-exit-code",
            "0",
            "--confirmation-boundary-exit-code",
            "0",
            "--source-protocol-json",
            str(chain.source_protocol_path),
            "--stage0-protocol-json",
            str(chain.stage0_protocol_path),
            "--selection-lock",
            str(chain.selection_path),
            "--visual-provider-lock",
            str(chain.provider_path),
            "--plan-json",
            str(chain.plan_path),
            "--download-json",
            str(chain.download_path),
            "--result-json",
            str(chain.result_path),
        ]
    )

    assert status == 0
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["source_locks_valid"] is True
    assert saved["record_sha256"] == _canonical_sha256(saved)
