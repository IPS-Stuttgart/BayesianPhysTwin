from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest
import test_deform360_calibration_visual_production_plan as plan_cases

import bayesian_phystwin.deform360_calibration_visual_execution_admission as admission_api
from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.deform360_calibration_visual_execution_admission import (
    DEFORM360_PREPARED_SOURCE_INVENTORY_CLAIM_BOUNDARY,
    build_deform360_calibration_visual_execution_admission,
    load_deform360_calibration_visual_execution_admission,
    save_deform360_calibration_visual_execution_admission,
    validate_deform360_calibration_visual_execution_admission,
)
from bayesian_phystwin.deform360_calibration_visual_production_plan import (
    save_deform360_calibration_visual_production_plan,
)

IMPLEMENTATION_REVISION = "5" * 40


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_record(path: str) -> dict[str, object]:
    return {
        "path": path,
        "sha256": _digest(path),
        "byte_count": 100 + len(path),
    }


def _inventory(plan: dict[str, object]) -> dict[str, object]:
    objects = []
    for plan_object in plan["objects"]:
        object_id = plan_object["object_id"]
        selected_stop = plan_object["selected_source_frame_range_half_open"][1]
        if type(selected_stop) is not int:
            raise AssertionError("selected frame stop must be an integer")
        aligned_frame_count = max(140, selected_stop)
        cameras = []
        for plan_camera in plan_object["cameras"]:
            camera_id = plan_camera["camera_id"]
            camera_root = f"{object_id}/episode_0000/{camera_id}"
            cameras.append(
                {
                    "camera": camera_id,
                    "video": _file_record(plan_camera["source_video_relative_path"]),
                    "preview": _file_record(f"{camera_root}/undistorted_000000.png"),
                    "timestamps": _file_record(
                        plan_camera["source_timestamps_relative_path"]
                    ),
                    "alignment": _file_record(f"{camera_root}/alignment.json"),
                    "metadata": _file_record(f"{camera_root}/metadata.json"),
                    "frame_count": aligned_frame_count,
                    "width": 640,
                    "height": 360,
                    "fps": 30.0,
                    "timeline_sha256": _digest(f"{object_id}:{camera_id}:timeline"),
                }
            )
        objects.append(
            {
                "object_id": object_id,
                "episode_id": plan_object["episode_id"],
                "stratum": plan_object["stratum"],
                "synthetic_episode_index": 0,
                "aligned_frame_count": aligned_frame_count,
                "action_window": {
                    "selected_raw_frame_range_half_open": plan_object[
                        "selected_source_frame_range_half_open"
                    ],
                    "prediction_raw_frame_range_half_open": plan_object[
                        "prediction_source_frame_range_half_open"
                    ],
                    "prefix_raw_frame_range_half_open": plan_object[
                        "prefix_source_frame_range_half_open"
                    ],
                    "raw_frame_count": aligned_frame_count,
                },
                "episode_files": {},
                "cameras": cameras,
                "tactile": [],
            }
        )
    identity: dict[str, object] = {
        "schema": ("bayesian-phystwin.deform360-calibration-prepared-source-inventory"),
        "schema_version": 1,
        "semantics": "exact-retained-calibration-rgb-tactile-robot-inventory-v1",
        "status": "complete-calibration-only-prepared-source",
        "implementation_revision": "8" * 40,
        "calibration_source_revision": "7" * 40,
        "processing_revision": "6" * 40,
        "selection_artifact_sha256": plan["selection_artifact_sha256"],
        "visual_provider_lock_id": plan["visual_provider_lock_id"],
        "calibration_source_run_record_sha256": plan[
            "calibration_source_run_record_sha256"
        ],
        "object_count": 10,
        "objects": objects,
        "source_artifacts": {
            "sources/calibration-source/result.json": plan[
                "calibration_source_result_sha256"
            ],
            "sources/stage0/selection.json": plan["selection_artifact_sha256"],
        },
        "information_boundary": {
            "calibration_camera_payloads_opened": True,
            "calibration_tactile_payloads_opened": True,
            "calibration_robot_state_opened": True,
            "calibration_target_metrics_computed": False,
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
            "replacement_allowed": False,
        },
        "claim_boundary": DEFORM360_PREPARED_SOURCE_INVENTORY_CLAIM_BOUNDARY,
    }
    return {**identity, "inventory_id": content_id(identity)}


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rewrite_inventory(path: Path, value: dict[str, object]) -> None:
    identity = {key: item for key, item in value.items() if key != "inventory_id"}
    value["inventory_id"] = content_id(identity)
    _write_json(path, value)


def _inputs(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    plan = plan_cases._build(plan_cases._inputs(tmp_path))
    plan_path = tmp_path / "visual-production-plan.json"
    save_deform360_calibration_visual_production_plan(plan_path, plan)
    inventory = _inventory(plan)
    inventory_path = tmp_path / "prepared-source-inventory.json"
    _write_json(inventory_path, inventory)
    return plan_path, inventory_path, plan


def _build(tmp_path: Path) -> dict[str, object]:
    plan_path, inventory_path, _plan = _inputs(tmp_path)
    return build_deform360_calibration_visual_execution_admission(
        visual_production_plan_path=plan_path,
        prepared_source_inventory_path=inventory_path,
        implementation_revision=IMPLEMENTATION_REVISION,
    )


def test_admission_binds_all_jobs_to_exact_source_bytes(tmp_path: Path) -> None:
    admission = _build(tmp_path)

    assert admission["object_count"] == 10
    assert admission["camera_view_count"] == 80
    assert len(admission["jobs"]) == 80
    assert [
        (job["object_id"], job["camera_id"]) for job in admission["jobs"]
    ] == sorted((job["object_id"], job["camera_id"]) for job in admission["jobs"])
    first = admission["jobs"][0]
    assert first["source_video"]["path"].endswith("/undistorted.mp4")
    assert first["source_timestamps"]["path"].endswith("/aligned_timestamps.txt")
    assert first["source_video"]["byte_count"] > 0
    assert len(first["source_video"]["sha256"]) == 64
    serialized = json.dumps(admission, sort_keys=True)
    assert str(tmp_path) not in serialized
    assert admission["information_boundary"] == {
        "plan_metadata_opened": True,
        "inventory_metadata_opened": True,
        "retained_calibration_payloads_opened": False,
        "confirmation_payloads_opened": False,
        "target_outcomes_used": False,
        "replacement_allowed": False,
    }


def test_admission_round_trip_is_deterministic_and_non_replacing(
    tmp_path: Path,
) -> None:
    first = _build(tmp_path / "first")
    second = _build(tmp_path / "second")

    assert first["admission_id"] == second["admission_id"]
    output = tmp_path / "admission.json"
    save_deform360_calibration_visual_execution_admission(output, first)
    assert load_deform360_calibration_visual_execution_admission(output) == first
    with pytest.raises(FileExistsError):
        save_deform360_calibration_visual_execution_admission(output, first)


def test_camera_omission_and_source_identity_drift_fail_closed(
    tmp_path: Path,
) -> None:
    plan_path, inventory_path, _plan = _inputs(tmp_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["objects"][0]["cameras"].pop()
    _rewrite_inventory(inventory_path, inventory)

    with pytest.raises(ValueError, match="camera rosters differ"):
        build_deform360_calibration_visual_execution_admission(
            visual_production_plan_path=plan_path,
            prepared_source_inventory_path=inventory_path,
            implementation_revision=IMPLEMENTATION_REVISION,
        )

    plan_path, inventory_path, _plan = _inputs(tmp_path / "digest")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["objects"][0]["cameras"][0]["video"]["sha256"] = "f" * 64
    _rewrite_inventory(inventory_path, inventory)
    admission = build_deform360_calibration_visual_execution_admission(
        visual_production_plan_path=plan_path,
        prepared_source_inventory_path=inventory_path,
        implementation_revision=IMPLEMENTATION_REVISION,
    )
    assert admission["jobs"][0]["source_video"]["sha256"] == "f" * 64
    assert admission["jobs"][0]["job_id"] != first_job_id(_build(tmp_path / "base"))


def first_job_id(admission: dict[str, object]) -> str:
    return admission["jobs"][0]["job_id"]


def test_frame_and_upstream_identity_mismatch_fail_closed(tmp_path: Path) -> None:
    plan_path, inventory_path, _plan = _inputs(tmp_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["objects"][0]["action_window"]["prediction_raw_frame_range_half_open"][
        1
    ] += 1
    _rewrite_inventory(inventory_path, inventory)
    with pytest.raises(ValueError, match="exactly 76 frames"):
        build_deform360_calibration_visual_execution_admission(
            visual_production_plan_path=plan_path,
            prepared_source_inventory_path=inventory_path,
            implementation_revision=IMPLEMENTATION_REVISION,
        )

    plan_path, inventory_path, _plan = _inputs(tmp_path / "provider")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["visual_provider_lock_id"] = "e" * 64
    _rewrite_inventory(inventory_path, inventory)
    with pytest.raises(ValueError, match="visual_provider_lock_id"):
        build_deform360_calibration_visual_execution_admission(
            visual_production_plan_path=plan_path,
            prepared_source_inventory_path=inventory_path,
            implementation_revision=IMPLEMENTATION_REVISION,
        )


def test_duplicate_keys_boolean_versions_and_tampering_are_rejected(
    tmp_path: Path,
) -> None:
    plan_path, inventory_path, _plan = _inputs(tmp_path)
    inventory_path.write_text(
        '{"schema":"a","schema":"b"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        build_deform360_calibration_visual_execution_admission(
            visual_production_plan_path=plan_path,
            prepared_source_inventory_path=inventory_path,
            implementation_revision=IMPLEMENTATION_REVISION,
        )

    plan_path, inventory_path, _plan = _inputs(tmp_path / "boolean")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["schema_version"] = True
    _rewrite_inventory(inventory_path, inventory)
    with pytest.raises(ValueError, match="version changed"):
        build_deform360_calibration_visual_execution_admission(
            visual_production_plan_path=plan_path,
            prepared_source_inventory_path=inventory_path,
            implementation_revision=IMPLEMENTATION_REVISION,
        )

    admission = _build(tmp_path / "tamper")
    tampered = copy.deepcopy(admission)
    tampered["jobs"][0]["source_video"]["byte_count"] += 1
    with pytest.raises(ValueError, match="job_id"):
        validate_deform360_calibration_visual_execution_admission(tampered)


def test_plan_is_parsed_from_the_opened_descriptor_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, inventory_path, _plan = _inputs(tmp_path)
    original_open = admission_api.os.open
    replaced = False

    def open_then_replace(path: object, flags: int, *args: object) -> int:
        nonlocal replaced
        descriptor = original_open(path, flags, *args)
        if not replaced and Path(path) == plan_path:
            replacement = plan_path.with_name("replacement-plan.json")
            replacement.write_text("{}\n", encoding="utf-8")
            os.replace(replacement, plan_path)
            replaced = True
        return descriptor

    monkeypatch.setattr(admission_api.os, "open", open_then_replace)
    admitted = build_deform360_calibration_visual_execution_admission(
        visual_production_plan_path=plan_path,
        prepared_source_inventory_path=inventory_path,
        implementation_revision=IMPLEMENTATION_REVISION,
    )

    assert admitted["camera_view_count"] == 80
    assert json.loads(plan_path.read_text(encoding="utf-8")) == {}
