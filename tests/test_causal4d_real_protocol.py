import csv
import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from causal4d.real_protocol import (
    build_same_object_real_protocol,
    execution_manifest_template,
    object_registration_template,
    protocol_design_sha256,
    scaffold_dataset,
    slip_pilot_template,
    validate_dataset,
    validate_execution_manifest,
    validate_object_registration,
    validate_protocol,
    validate_slip_pilot,
    write_acquisition_schedule,
    write_protocol,
)


def _complete_manifest(protocol: dict, execution_id: str) -> dict:
    manifest = execution_manifest_template(protocol, execution_id)
    manifest["acquisition_status"] = "complete"
    manifest["acquisition"] = {
        "operator_id": "operator-1",
        "hardware_run_id": f"run-{execution_id}",
        "started_at_utc": "2026-07-13T08:00:00+02:00",
    }
    manifest["timing"] = {
        "frame_count": 120,
        "intervention_frame": 30,
        "o_plus_prefix_frames": 6,
    }
    for name in protocol["recording_contract"]["required_artifacts"]:
        descriptor = {
            "path": f"data/{name}.bin",
            "sha256": "0" * 64,
            "bytes": 1,
        }
        if name in protocol["recording_contract"]["timestamped_artifacts"]:
            descriptor["clock_id"] = "ptp-clock-0"
        manifest["artifacts"][name] = descriptor
    manifest["quality"] = {
        "reset_passed": True,
        "rgbd_actuator_sync_error_ms": 1.0,
        "initial_state_chamfer_m": 0.001,
        "end_effector_reset_error_m": 0.001,
        "contact_centroid_error_m": 0.002,
        "dropped_rgbd_frames": 0,
        "slip_displacement_m": None,
        "complete_release_observed": None,
    }
    if manifest["realization_condition_id"] == "slip_low_force":
        manifest["artifacts"]["gripper_normal_force"] = {
            "path": "data/gripper_normal_force.bin",
            "sha256": "1" * 64,
            "bytes": 1,
            "clock_id": "ptp-clock-0",
        }
        manifest["quality"]["slip_displacement_m"] = 0.01
        manifest["quality"]["complete_release_observed"] = False
    manifest["drift_indicators"] = {
        "wear_cycle_count": 4,
        "minutes_since_first_execution": 12.5,
        "object_temperature_c": 22.4,
        "room_temperature_c": 21.8,
        "notes": "none",
    }
    manifest["exclusion"] = {
        "status": "included",
        "reason": None,
        "decided_before_target_evaluation": True,
    }
    return manifest


def _complete_registration(protocol: dict) -> dict:
    registration = object_registration_template(protocol)
    registration["object_instance_serial"] = "sloth-001"
    registration["phystwin_model_id"] = "sloth-twin-v1"
    registration["phystwin_model_sha256"] = "2" * 64
    for index, descriptor in enumerate(registration["contact_regions"].values()):
        descriptor["canonical_node_set_path"] = f"contact_{index}.npz"
        descriptor["canonical_node_set_sha256"] = f"{index + 3:x}" * 64
        descriptor["node_count"] = 24 + index
    return registration


def _passing_slip_pilot(protocol: dict) -> dict:
    pilot = slip_pilot_template(protocol)
    pilot.update(
        {
            "pilot_execution_ids": [f"pilot-{index}" for index in range(5)],
            "contact_region_ids": ["left_forepaw", "right_forepaw"],
            "bounded_slip_successes": 5,
            "slip_displacement_mean_m": 0.009,
            "slip_displacement_coefficient_of_variation": 0.2,
            "complete_release_count": 0,
            "passed": True,
            "decided_before_confirmatory_collection": True,
        }
    )
    return pilot


def _materialize_manifest_artifacts(manifest: dict, execution_root: Path) -> None:
    for name, descriptor in manifest["artifacts"].items():
        if not descriptor.get("path"):
            continue
        artifact_path = execution_root / descriptor["path"]
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        content = f"{manifest['execution_id']}:{name}\n".encode()
        artifact_path.write_bytes(content)
        descriptor["bytes"] = len(content)
        descriptor["sha256"] = hashlib.sha256(content).hexdigest()


def test_protocol_has_balanced_36_execution_factorial() -> None:
    protocol = build_same_object_real_protocol()
    summary = validate_protocol(protocol)
    assert summary["sessions"] == 18
    assert summary["executions"] == 36

    executions = protocol["executions"]
    assert Counter(execution["contact_region_id"] for execution in executions) == {
        "left_forepaw": 12,
        "right_forepaw": 12,
        "upper_torso": 12,
    }
    assert set(
        Counter(execution["command_profile_id"] for execution in executions).values()
    ) == {9}
    assert set(
        Counter(
            execution["realization_condition_id"] for execution in executions
        ).values()
    ) == {6}
    condition_profile_counts = Counter(
        (
            execution["realization_condition_id"],
            execution["command_profile_id"],
        )
        for execution in executions
    )
    assert set(condition_profile_counts.values()) == {1, 2}
    assert len(condition_profile_counts) == 24
    cells = Counter(
        (execution["contact_region_id"], execution["command_profile_id"])
        for execution in executions
    )
    assert len(cells) == 12
    assert set(cells.values()) == {3}


def test_same_grasp_sessions_are_chronological_and_counterbalanced() -> None:
    protocol = build_same_object_real_protocol()
    execution_by_id = {
        execution["execution_id"]: execution for execution in protocol["executions"]
    }
    pairs = protocol["splits"]["same_grasp_intervention_prediction"]
    assert len(pairs) == 18
    for pair in pairs:
        source = execution_by_id[pair["source_execution_id"]]
        target = execution_by_id[pair["target_execution_id"]]
        assert source["session_id"] == target["session_id"]
        assert source["pair_order"] == 0
        assert target["pair_order"] == 1
        assert pair["transfer_phi"]
        assert pair["reuse_kappa"]

    first_counts = Counter(
        execution["command_profile_id"]
        for execution in protocol["executions"]
        if execution["pair_order"] == 0
    )
    second_counts = Counter(
        execution["command_profile_id"]
        for execution in protocol["executions"]
        if execution["pair_order"] == 1
    )
    assert (
        max(
            abs(first_counts[profile] - second_counts[profile])
            for profile in first_counts
        )
        <= 1
    )

    session_by_id = {session["session_id"]: session for session in protocol["sessions"]}
    contact_order = [
        session_by_id[session_id]["contact_region_id"]
        for session_id in protocol["acquisition_session_order"]
    ]
    assert all(left != right for left, right in zip(contact_order, contact_order[1:]))


def test_new_contact_pairs_hold_command_and_phi_fixed() -> None:
    protocol = build_same_object_real_protocol()
    execution_by_id = {
        execution["execution_id"]: execution for execution in protocol["executions"]
    }
    pairs = protocol["splits"]["new_contact_intervention_prediction"]
    assert len(pairs) == 12
    contact_pairs: Counter[tuple[str, str]] = Counter()
    profile_counts: Counter[str] = Counter()
    condition_counts: Counter[str] = Counter()
    for pair in pairs:
        source = execution_by_id[pair["source_execution_id"]]
        target = execution_by_id[pair["target_execution_id"]]
        assert source["contact_region_id"] != target["contact_region_id"]
        assert source["command_profile_id"] == target["command_profile_id"]
        assert source["realization_condition_id"] == target["realization_condition_id"]
        assert pair["resample_kappa_cf"]
        assert not pair["reuse_kappa"]
        assert (
            source["acquisition_execution_index"]
            < target["acquisition_execution_index"]
        )
        contact_pairs[
            tuple(sorted((source["contact_region_id"], target["contact_region_id"])))
        ] += 1
        profile_counts[source["command_profile_id"]] += 1
        condition_counts[source["realization_condition_id"]] += 1
    assert set(contact_pairs.values()) == {4}
    assert set(profile_counts.values()) == {3}
    assert set(condition_counts.values()) == {2}


def test_cross_action_contact_folds_have_no_session_or_factor_leakage() -> None:
    protocol = build_same_object_real_protocol()
    execution_by_id = {
        execution["execution_id"]: execution for execution in protocol["executions"]
    }
    target_counts: Counter[str] = Counter()
    folds = protocol["splits"]["cross_action_contact_calibration_folds"]
    assert len(folds) == 12
    for fold in folds:
        fit = set(fold["fit_execution_ids"])
        calibration = set(fold["calibration_execution_ids"])
        target = set(fold["target_execution_ids"])
        assert (len(fit), len(calibration), len(target)) == (8, 4, 3)
        assert not fit & calibration
        assert not fit & target
        assert not calibration & target
        assert not set(fold["fit_session_ids"]) & set(fold["calibration_session_ids"])
        for identifier in fit | calibration:
            execution = execution_by_id[identifier]
            assert execution["contact_region_id"] != fold["held_out_contact_region_id"]
            assert (
                execution["command_profile_id"] != fold["held_out_command_profile_id"]
            )
        target_counts.update(target)
    assert target_counts == Counter(
        {execution["execution_id"]: 1 for execution in protocol["executions"]}
    )


def test_protocol_digest_detects_split_changes() -> None:
    protocol = build_same_object_real_protocol()
    changed = deepcopy(protocol)
    fold = changed["splits"]["cross_action_contact_calibration_folds"][0]
    fold["fit_execution_ids"].append(fold["target_execution_ids"][0])
    changed["design_sha256"] = protocol_design_sha256(changed)
    with pytest.raises(ValueError, match="fit set"):
        validate_protocol(changed)

    weakened_gate = deepcopy(protocol)
    weakened_gate["quality_gates"]["maximum_rgbd_actuator_sync_error_ms"] = 500.0
    weakened_gate["design_sha256"] = protocol_design_sha256(weakened_gate)
    with pytest.raises(ValueError, match="canonical locked v1 design"):
        validate_protocol(weakened_gate)


def test_checked_in_protocol_is_the_deterministic_design() -> None:
    repository = Path(__file__).resolve().parents[1]
    checked_in = json.loads(
        (repository / "configs/causal4d/sloth_multi_action_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert checked_in == build_same_object_real_protocol()
    assert validate_protocol(checked_in)["passed"]


def test_checked_in_schedule_is_complete_and_in_locked_order(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    protocol = build_same_object_real_protocol()
    generated = write_acquisition_schedule(tmp_path / "schedule.csv", protocol)
    checked_in = repository / "configs/causal4d/sloth_multi_action_v1_schedule.csv"
    assert generated.read_bytes() == checked_in.read_bytes()
    with checked_in.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 36
    assert [int(row["acquisition_execution_index"]) for row in rows] == list(range(36))


def test_scaffold_writes_templates_and_refuses_overwrite(tmp_path: Path) -> None:
    protocol = build_same_object_real_protocol()
    root = tmp_path / "dataset"
    summary = scaffold_dataset(protocol, root)
    assert summary["execution_templates"] == 36
    assert (root / "acquisition_schedule.csv").is_file()
    assert len(list(root.glob("executions/*/manifest.template.json"))) == 36
    assert len(list(root.glob("sessions/*/session.template.json"))) == 18
    with pytest.raises(FileExistsError, match="nonempty"):
        scaffold_dataset(protocol, root)
    template = json.loads(
        next(root.glob("executions/*/manifest.template.json")).read_text(
            encoding="utf-8"
        )
    )
    with pytest.raises(ValueError, match="incomplete"):
        validate_execution_manifest(protocol, template)


def test_execution_manifest_requires_measured_actuation_and_locked_injection() -> None:
    protocol = build_same_object_real_protocol()
    execution_id = next(
        execution["execution_id"]
        for execution in protocol["executions"]
        if execution["realization_condition_id"] == "nominal"
    )
    manifest = _complete_manifest(protocol, execution_id)
    assert validate_execution_manifest(protocol, manifest)["passed"]

    missing_actuator = deepcopy(manifest)
    del missing_actuator["artifacts"]["measured_end_effector_trajectory"]
    with pytest.raises(ValueError, match="measured_end_effector"):
        validate_execution_manifest(protocol, missing_actuator)

    changed_injection = deepcopy(manifest)
    changed_injection["known_injection"]["phi"]["gain_multiplier"] = 0.9
    with pytest.raises(ValueError, match="known injection"):
        validate_execution_manifest(protocol, changed_injection)

    failed_gate = deepcopy(manifest)
    failed_gate["quality"]["initial_state_chamfer_m"] = 0.004
    with pytest.raises(ValueError, match="require a preregistered exclusion"):
        validate_execution_manifest(protocol, failed_gate)
    failed_gate["exclusion"] = {
        "status": "excluded",
        "reason": "initial-state gate",
        "decided_before_target_evaluation": True,
    }
    result = validate_execution_manifest(protocol, failed_gate)
    assert result["quality_gate_failures"] == ["initial_state_chamfer_m"]
    assert not result["included"]


def test_slip_execution_and_pilot_require_bounded_instrumented_slip() -> None:
    protocol = build_same_object_real_protocol()
    execution_id = next(
        execution["execution_id"]
        for execution in protocol["executions"]
        if execution["realization_condition_id"] == "slip_low_force"
    )
    manifest = _complete_manifest(protocol, execution_id)
    assert validate_execution_manifest(protocol, manifest)["passed"]
    manifest["artifacts"]["gripper_normal_force"]["path"] = None
    with pytest.raises(ValueError, match="force/torque"):
        validate_execution_manifest(protocol, manifest)

    pilot = _passing_slip_pilot(protocol)
    validate_slip_pilot(protocol, pilot)
    pilot["slip_displacement_coefficient_of_variation"] = 0.5
    with pytest.raises(ValueError, match="not reproducible"):
        validate_slip_pilot(protocol, pilot)


def test_complete_dataset_validates_all_registered_file_hashes(tmp_path: Path) -> None:
    protocol = build_same_object_real_protocol()
    write_protocol(tmp_path / "protocol.json", protocol)
    write_acquisition_schedule(tmp_path / "acquisition_schedule.csv", protocol)
    registration = _complete_registration(protocol)
    for region_id, descriptor in registration["contact_regions"].items():
        node_path = tmp_path / descriptor["canonical_node_set_path"]
        content = f"canonical-node-set:{region_id}\n".encode()
        node_path.write_bytes(content)
        descriptor["canonical_node_set_sha256"] = hashlib.sha256(content).hexdigest()
    validate_object_registration(protocol, registration)
    (tmp_path / "object_registration.json").write_text(
        json.dumps(registration), encoding="utf-8"
    )
    (tmp_path / "slip_pilot.json").write_text(
        json.dumps(_passing_slip_pilot(protocol)), encoding="utf-8"
    )
    for execution in protocol["executions"]:
        root = tmp_path / "executions" / execution["execution_id"]
        root.mkdir(parents=True)
        manifest = _complete_manifest(protocol, execution["execution_id"])
        _materialize_manifest_artifacts(manifest, root)
        (root / "manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
    result = validate_dataset(protocol, tmp_path, verify_files=True)
    assert result["executions_checked"] == 36
    assert result["included"] == 36
    assert result["file_hashes_verified"]
    assert result["passed"]

    schedule_path = tmp_path / "acquisition_schedule.csv"
    with schedule_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    rows[0]["contact_region_id"] = "wrong-contact"
    with schedule_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="schedule differs from the locked design"):
        validate_dataset(protocol, tmp_path, verify_files=False)
