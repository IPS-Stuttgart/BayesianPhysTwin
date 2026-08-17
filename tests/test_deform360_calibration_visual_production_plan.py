from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import test_deform360_calibration_observability_batch as batch_cases

from bayesian_phystwin.deform360_calibration_source_run_record import (
    save_deform360_calibration_source_run_record,
)
from bayesian_phystwin.deform360_calibration_visual_production_plan import (
    CAMERA_ROSTER_POLICY,
    PROB4D_MOTIONCRAFTER_SEED_POLICY,
    build_deform360_calibration_visual_production_plan,
    deform360_calibration_object_seed,
    deform360_calibration_view_seed,
    load_deform360_calibration_visual_production_plan,
    save_deform360_calibration_visual_production_plan,
    validate_deform360_calibration_visual_production_plan,
)

IMPLEMENTATION_REVISION = "9" * 40


def _inputs(tmp_path: Path) -> batch_cases.case_inputs.Inputs:
    inputs = batch_cases._batch_inputs(tmp_path)
    source = batch_cases.case_inputs.source_run_cases
    result = json.loads(inputs.chain.result_path.read_text(encoding="utf-8"))
    for index, row in enumerate(result["objects"]):
        start = 20 + index * 7
        row["camera_count"] = 8
        row["cameras"] = [f"camera-{camera}" for camera in range(8)]
        row["aligned_frame_count"] = 140
        row["action_window"] = {
            "selected_raw_frame_range_half_open": [start, start + 81],
            "prediction_raw_frame_range_half_open": [start, start + 76],
            "prefix_raw_frame_range_half_open": [start, start + 58],
            "longest_active_segment": [start + 10, start + 40],
            "raw_frame_count": 140,
            "selection_rule": "synthetic-action-only-window-v1",
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
    return inputs


def _build(inputs: batch_cases.case_inputs.Inputs):
    return build_deform360_calibration_visual_production_plan(
        source_protocol_path=inputs.chain.source_protocol_path,
        stage0_protocol_path=inputs.chain.stage0_protocol_path,
        selection_lock_path=inputs.chain.selection_path,
        visual_provider_lock_path=inputs.chain.provider_path,
        calibration_source_plan_path=inputs.chain.plan_path,
        calibration_source_download_path=inputs.chain.download_path,
        calibration_source_run_record_path=inputs.run_record_path,
        calibration_source_result_path=inputs.chain.result_path,
        implementation_revision=IMPLEMENTATION_REVISION,
    )


def test_plan_binds_exact_ten_object_camera_and_frame_roster(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)

    plan = _build(inputs)

    assert plan["object_count"] == 10
    assert plan["camera_view_count"] == 80
    assert plan["production_policy"]["camera_roster_policy"] == (CAMERA_ROSTER_POLICY)
    assert plan["production_policy"]["prob4d_motioncrafter_seed_policy"] == (
        PROB4D_MOTIONCRAFTER_SEED_POLICY
    )
    assert [row["object_id"] for row in plan["objects"]] == sorted(
        row["object_id"] for row in plan["objects"]
    )
    assert {row["stratum"] for row in plan["objects"]} == {
        "sheet",
        "volumetric",
    }
    first = plan["objects"][0]
    selected = first["selected_source_frame_range_half_open"]
    prediction = first["prediction_source_frame_range_half_open"]
    prefix = first["prefix_source_frame_range_half_open"]
    assert selected[1] - selected[0] == 81
    assert prediction[1] - prediction[0] == 76
    assert prefix[1] - prefix[0] == 58
    assert [camera["camera_id"] for camera in first["cameras"]] == [
        f"camera-{camera}" for camera in range(8)
    ]
    assert all(
        camera["source_video_relative_path"].endswith("/undistorted.mp4")
        for camera in first["cameras"]
    )
    serialized = json.dumps(plan, sort_keys=True)
    assert str(tmp_path) not in serialized
    assert "confirm-sheet" not in serialized


def test_object_and_view_seeds_are_deterministic_and_distinct(
    tmp_path: Path,
) -> None:
    plan = _build(_inputs(tmp_path))
    provider = plan["provider"]
    object_seeds = []
    view_seeds = []
    for row in plan["objects"]:
        expected = deform360_calibration_object_seed(
            root_seed=provider["root_seed"],
            visual_provider_lock_id=plan["visual_provider_lock_id"],
            object_id=row["object_id"],
            episode_id=row["episode_id"],
        )
        assert row["object_root_seed"] == expected
        object_seeds.append(expected)
        for camera in row["cameras"]:
            assert camera["view_root_seed"] == deform360_calibration_view_seed(
                object_seed=expected,
                camera_id=camera["camera_id"],
            )
            view_seeds.append(camera["view_root_seed"])
    assert len(set(object_seeds)) == 10
    assert len(set(view_seeds)) == 80


def test_plan_round_trip_is_content_addressed_and_non_replacing(
    tmp_path: Path,
) -> None:
    plan = _build(_inputs(tmp_path))
    path = tmp_path / "visual-production-plan.json"

    save_deform360_calibration_visual_production_plan(path, plan)
    assert load_deform360_calibration_visual_production_plan(path) == plan
    with pytest.raises(FileExistsError):
        save_deform360_calibration_visual_production_plan(path, plan)

    tampered = copy.deepcopy(plan)
    tampered["objects"][0]["cameras"][0]["view_root_seed"] += 1
    with pytest.raises(ValueError, match="camera plan changed"):
        validate_deform360_calibration_visual_production_plan(tampered)


def test_camera_order_and_substitution_fail_closed(tmp_path: Path) -> None:
    plan = _build(_inputs(tmp_path))

    reordered = copy.deepcopy(plan)
    reordered["objects"][0]["cameras"].reverse()
    with pytest.raises(ValueError, match="camera roster must be sorted"):
        validate_deform360_calibration_visual_production_plan(reordered)

    substituted = copy.deepcopy(plan)
    substituted["objects"][0]["cameras"][0]["camera_id"] = "other-camera"
    with pytest.raises(ValueError, match="camera plan changed"):
        validate_deform360_calibration_visual_production_plan(substituted)


def test_frame_range_and_output_collisions_fail_closed(tmp_path: Path) -> None:
    plan = _build(_inputs(tmp_path))

    changed_range = copy.deepcopy(plan)
    changed_range["objects"][0]["prediction_source_frame_range_half_open"][1] += 1
    with pytest.raises(ValueError, match="exactly 76 frames"):
        validate_deform360_calibration_visual_production_plan(changed_range)

    collision = copy.deepcopy(plan)
    collision["objects"][1]["cameras"][0]["output_relative_directory"] = collision[
        "objects"
    ][0]["cameras"][0]["output_relative_directory"]
    with pytest.raises(ValueError, match="camera plan changed|collision"):
        validate_deform360_calibration_visual_production_plan(collision)


def test_plan_requires_all_ten_successfully_prepared_objects(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    source = batch_cases.case_inputs.source_run_cases
    result = json.loads(inputs.chain.result_path.read_text(encoding="utf-8"))
    result["objects"][0] = source._technical_failure(
        inputs.chain.selection["selection"]["calibration"][0]
    )
    result["gate"] = source._gate(4, 5)
    result["result_sha256"] = source.canonical_sha256(
        result,
        digest_key="result_sha256",
    )
    source._write(inputs.chain.result_path, result)
    inputs.run_record_path.unlink()
    save_deform360_calibration_source_run_record(
        source._record(inputs.chain, workload_exit_code=3),
        inputs.run_record_path,
    )

    with pytest.raises(ValueError, match="terminal record did not succeed|all ten"):
        _build(inputs)


def test_confirmation_identity_cannot_enter_object_plan(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    source = batch_cases.case_inputs.source_run_cases
    result = json.loads(inputs.chain.result_path.read_text(encoding="utf-8"))
    result["objects"][0]["object_id"] = "confirm-sheet-0"
    result["result_sha256"] = source.canonical_sha256(
        result,
        digest_key="result_sha256",
    )
    source._write(inputs.chain.result_path, result)
    inputs.run_record_path.unlink()
    save_deform360_calibration_source_run_record(
        source._record(inputs.chain, workload_exit_code=1),
        inputs.run_record_path,
    )

    with pytest.raises(
        ValueError, match="terminal record did not succeed|confirmation"
    ):
        _build(inputs)
