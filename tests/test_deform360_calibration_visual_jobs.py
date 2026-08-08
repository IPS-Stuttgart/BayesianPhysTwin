"""Contracts for exact Deform360 calibration visual-provider jobs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import test_deform360_calibration_observability_batch as batch_cases

from bayesian_phystwin._portable_contracts import canonical_json_bytes
from bayesian_phystwin.deform360_calibration_source_run_record import (
    save_deform360_calibration_source_run_record,
)
from bayesian_phystwin.deform360_calibration_visual_jobs import (
    DEFORM360_OBJECT_SEED_SCHEMA,
    PROB4D_MOTIONCRAFTER_SEED_SCHEMA,
    build_deform360_calibration_visual_job_manifest,
    deform360_object_seed,
    load_deform360_calibration_visual_job_manifest,
    prob4d_motioncrafter_seed,
    save_deform360_calibration_visual_job_manifest,
    validate_deform360_calibration_visual_job_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = ROOT / "scripts/science/build_deform360_calibration_visual_jobs.py"
SPEC = importlib.util.spec_from_file_location("deform360_visual_jobs_cli", CLI_PATH)
assert SPEC is not None and SPEC.loader is not None
CLI = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CLI
SPEC.loader.exec_module(CLI)

IMPLEMENTATION_REVISION = "7" * 40
CAMERAS = tuple(f"camera_{index:02d}" for index in range(8))


def _action_window(start: int = 10) -> dict[str, Any]:
    return {
        "selection_rule": "maximum_mean_closed_weighted_gripper_path",
        "selected_raw_frame_range_half_open": [start, start + 81],
        "prediction_raw_frame_range_half_open": [start, start + 76],
        "prefix_raw_frame_range_half_open": [start, start + 58],
        "candidate_first_frame": 8,
        "candidate_stride_frames": 6,
        "candidate_count": 2,
        "tie_break": "earliest start",
        "mean_closed_weighted_path_length_m": 0.25,
        "input_fields": ["robot.actions", "robot.openings"],
        "known_future_action_is_conditioning_input": True,
        "object_geometry_read": False,
        "object_tracks_read": False,
        "tactile_read": False,
    }


def _inputs(
    tmp_path: Path,
    *,
    source_failures: frozenset[str] = frozenset(),
) -> tuple[Any, Path]:
    inputs = batch_cases._batch_inputs(tmp_path / "inputs")
    source = batch_cases.case_inputs.source_run_cases
    processed = tmp_path / "processed"
    result = json.loads(inputs.chain.result_path.read_text(encoding="utf-8"))

    prepared_by_stratum = {"sheet": 0, "volumetric": 0}
    for row in result["objects"]:
        object_id = row["object_id"]
        if object_id in source_failures:
            row["status"] = "technical_failure_without_replacement"
            row["error"] = "synthetic retained source failure"
            continue
        prepared_by_stratum[row["stratum"]] += 1
        episode = processed / object_id / "episode_0000"
        alignment = {
            "cameras": list(CAMERAS),
            "frame_count": 100,
        }
        alignment_path = episode / "alignment.json"
        alignment_path.parent.mkdir(parents=True, exist_ok=True)
        source._write(alignment_path, alignment)
        for camera in CAMERAS:
            camera_root = episode / camera
            camera_root.mkdir(parents=True)
            (camera_root / "undistorted.mp4").write_bytes(
                f"{object_id}:{camera}:synthetic-video\n".encode("utf-8")
            )
        row.update(
            {
                "status": "source_prepared",
                "completed_stage": "action-window-selection",
                "synthetic_episode_index": 0,
                "bimanual": False,
                "camera_count": len(CAMERAS),
                "cameras": list(CAMERAS),
                "aligned_frame_count": 100,
                "tactile_sensor_count": 1,
                "tactile_sensors": ["sensor-a"],
                "action_window": _action_window(),
                "outputs_sha256": {
                    "alignment": source.file_sha256(alignment_path),
                },
            }
        )
    prepared = sum(prepared_by_stratum.values())
    result["gate"] = {
        "supported_object_count": prepared,
        "supported_by_stratum": prepared_by_stratum,
        "minimum_supported_objects": 8,
        "minimum_supported_per_stratum": 4,
        "support_passed": (
            prepared >= 8 and min(prepared_by_stratum.values()) >= 4
        ),
    }
    result["result_sha256"] = source.canonical_sha256(
        result,
        digest_key="result_sha256",
    )
    source._write(inputs.chain.result_path, result)
    inputs.run_record_path.unlink()
    save_deform360_calibration_source_run_record(
        source._record(inputs.chain),
        inputs.run_record_path,
    )
    return inputs, processed


def _build(inputs: Any, processed: Path) -> dict[str, object]:
    return build_deform360_calibration_visual_job_manifest(
        stage0_protocol_path=inputs.chain.stage0_protocol_path,
        selection_lock_path=inputs.chain.selection_path,
        visual_provider_lock_path=inputs.chain.provider_path,
        calibration_source_run_record_path=inputs.run_record_path,
        calibration_source_result_path=inputs.chain.result_path,
        processed_root=processed,
        implementation_revision=IMPLEMENTATION_REVISION,
    )


def _arguments(inputs: Any, processed: Path, output: Path) -> list[str]:
    return [
        "--stage0-protocol",
        str(inputs.chain.stage0_protocol_path),
        "--selection-lock",
        str(inputs.chain.selection_path),
        "--visual-provider-lock",
        str(inputs.chain.provider_path),
        "--calibration-source-run-record",
        str(inputs.run_record_path),
        "--calibration-source-result",
        str(inputs.chain.result_path),
        "--processed-root",
        str(processed),
        "--implementation-revision",
        IMPLEMENTATION_REVISION,
        "--output",
        str(output),
    ]


def test_manifest_plans_every_camera_from_only_the_causal_prefix(
    tmp_path: Path,
) -> None:
    inputs, processed = _inputs(tmp_path)

    manifest = _build(inputs, processed)

    assert manifest["status"] == "locked-pre-provider-inference"
    assert manifest["support_gate"]["planned_object_count"] == 10
    assert manifest["support_gate"]["support_passed"] is True
    assert len(manifest["objects"]) == 10
    assert sum(len(item["jobs"]) for item in manifest["objects"]) == 80
    assert len({item["object_seed"] for item in manifest["objects"]}) == 10

    first = manifest["objects"][0]
    assert first["status"] == "planned"
    assert first["jobs"][0]["source_frame_start"] == 10
    assert first["jobs"][0]["source_frame_stop_exclusive"] == 68
    assert first["jobs"][0]["evaluation_frame_start"] == 68
    assert first["jobs"][0]["evaluation_frame_stop_exclusive"] == 86
    assert first["jobs"][0]["overlap_windows"] == [
        {
            "window_id": "window_0000",
            "source_frame_start": 10,
            "source_frame_stop_exclusive": 35,
        },
        {
            "window_id": "window_0001",
            "source_frame_start": 27,
            "source_frame_stop_exclusive": 52,
        },
        {
            "window_id": "window_0002",
            "source_frame_start": 43,
            "source_frame_stop_exclusive": 68,
        },
    ]
    assert manifest["information_boundary"] == {
        "calibration_camera_payload_bytes_hashed": True,
        "camera_frames_decoded": False,
        "prediction_outputs_opened": False,
        "calibration_target_metrics_computed": False,
        "confirmation_payloads_opened": False,
        "target_outcomes_used": False,
        "replacement_allowed": False,
    }
    serialized = json.dumps(manifest, sort_keys=True)
    assert str(tmp_path) not in serialized
    assert "confirm-sheet" not in serialized


def test_seed_functions_match_the_declared_canonical_algorithms() -> None:
    root_seed = 20260805
    object_id = "cal-sheet-0"
    object_descriptor = {
        "schema": DEFORM360_OBJECT_SEED_SCHEMA,
        "root_seed": root_seed,
        "object_id": object_id,
    }
    object_seed = int.from_bytes(
        hashlib.sha256(canonical_json_bytes(object_descriptor)).digest()[:4],
        "big",
    )
    assert deform360_object_seed(root_seed, object_id=object_id) == object_seed

    call_id = "overlap-window:window_0000:10:35"
    call_descriptor = {
        "schema": PROB4D_MOTIONCRAFTER_SEED_SCHEMA,
        "root_seed": object_seed,
        "call_id": call_id,
    }
    expected = int.from_bytes(
        hashlib.sha256(canonical_json_bytes(call_descriptor)).digest()[:4],
        "big",
    )
    assert prob4d_motioncrafter_seed(object_seed, call_id=call_id) == expected


def test_manifest_roundtrips_and_rejects_derived_field_tampering(
    tmp_path: Path,
) -> None:
    inputs, processed = _inputs(tmp_path)
    manifest = _build(inputs, processed)
    output = tmp_path / "visual-jobs.json"

    save_deform360_calibration_visual_job_manifest(output, manifest)
    loaded = load_deform360_calibration_visual_job_manifest(output)
    assert loaded["manifest_id"] == manifest["manifest_id"]
    with pytest.raises(FileExistsError):
        save_deform360_calibration_visual_job_manifest(output, manifest)

    changed = json.loads(json.dumps(manifest))
    changed["objects"][0]["jobs"][0]["source_frame_stop_exclusive"] = 69
    with pytest.raises(ValueError, match="causal frame boundary"):
        validate_deform360_calibration_visual_job_manifest(changed)

    changed = json.loads(json.dumps(manifest))
    changed["objects"][0]["jobs"][0]["stochastic_seed_schedule"]["calls"][0][
        "effective_seed"
    ] += 1
    with pytest.raises(ValueError, match="seed schedule"):
        validate_deform360_calibration_visual_job_manifest(changed)

    changed = json.loads(json.dumps(manifest))
    changed["support_gate"]["planned_object_count"] = 9
    with pytest.raises(ValueError, match="support gate"):
        validate_deform360_calibration_visual_job_manifest(changed)


def test_two_source_failures_remain_in_the_object_denominator(
    tmp_path: Path,
) -> None:
    inputs, processed = _inputs(
        tmp_path,
        source_failures=frozenset({"cal-sheet-4", "cal-volumetric-4"}),
    )

    manifest = _build(inputs, processed)

    assert manifest["support_gate"]["planned_object_count"] == 8
    assert manifest["support_gate"]["technical_failure_count"] == 2
    assert manifest["support_gate"]["planned_by_stratum"] == {
        "sheet": 4,
        "volumetric": 4,
    }
    failures = [
        item
        for item in manifest["objects"]
        if item["status"] == "technical_failure_without_replacement"
    ]
    assert len(failures) == 2
    assert all(item["jobs"] == [] for item in failures)
    assert all(item["failure_reason"] for item in failures)


def test_exact_source_substitution_and_unsafe_video_paths_fail_closed(
    tmp_path: Path,
) -> None:
    inputs, processed = _inputs(tmp_path)
    with inputs.chain.result_path.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="result_file_sha256"):
        _build(inputs, processed)

    inputs, processed = _inputs(tmp_path / "symlink")
    video = next(processed.glob("*/episode_0000/*/undistorted.mp4"))
    target = video.with_name("real.mp4")
    video.rename(target)
    video.symlink_to(target.name)
    with pytest.raises(ValueError, match="symlink"):
        _build(inputs, processed)


def test_missing_camera_video_is_a_failed_manifest_not_an_exclusion(
    tmp_path: Path,
) -> None:
    inputs, processed = _inputs(tmp_path)
    video = next(processed.glob("*/episode_0000/*/undistorted.mp4"))
    video.unlink()

    with pytest.raises(ValueError, match="does not exist"):
        _build(inputs, processed)


def test_cli_publishes_once_and_returns_support_status(tmp_path: Path) -> None:
    inputs, processed = _inputs(tmp_path)
    output = tmp_path / "visual-jobs.json"
    arguments = _arguments(inputs, processed, output)

    assert CLI.main(arguments) == 0
    assert output.is_file()
    assert CLI.main(arguments) == CLI.CONTRACT_FAILURE_EXIT_CODE
