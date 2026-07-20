from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from deform360_held_test_helpers import (
    bound_file,
    default_frame_zero_config,
    dummy_immutable_bindings,
    write_robot_kinematics_fixture,
    write_robot_metadata_fixture,
)

from bayesian_phystwin.deform360_held_protocol import (
    CALIBRATION_CASE_NAMES,
    CALIBRATION_GATE,
    CALIBRATION_SCORE_EVIDENCE_KIND,
    CONFIRMATION_CASES,
    CONFIRMATION_GATE,
    FRAME_ZERO_KIND,
    METRIC_LOCK,
    ONLINE_ARTIFACT_ROLES,
    PHYSICAL_ARTIFACT_ROLES,
    PRIMARY_METHOD,
    PROTOCOL_ID,
    REQUIRED_IMMUTABLE_BINDING_KEYS,
    SOURCE_FEASIBILITY_AMENDMENT_CONTRACT,
    _FRAME_ZERO_CAMERA_SELECTION_POLICY_ID,
    _FRAME_ZERO_CAMERA_SELECTION_RULE,
    authorize_outcome_phase,
    create_calibration_gate_decision,
    create_confirmation_protocol_lock,
    create_held_protocol_lock,
    create_online_prediction_seal,
    create_physical_prior_seal,
    create_prefix_stage_authorization,
    held_artifact_sha256,
    load_held_protocol_lock,
    locked_case_names,
    run_outcome_operation,
    validate_frame_zero_bundle_manifest,
)
from bayesian_phystwin.deform360_robot_kinematics import (
    load_robot_kinematics_archive,
    robot_kinematics_array_records,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _calibration_lock(tmp_path: Path) -> Path:
    path = tmp_path / "calibration-lock.json"
    create_held_protocol_lock(
        path,
        immutable_bindings=dummy_immutable_bindings(),
    )
    return path


def _passing_scores(*, primary_chamfer_m: float = 0.90) -> dict[str, dict[str, float]]:
    return {
        case_name: {
            "primary_chamfer_m": primary_chamfer_m,
            "comparator_chamfer_m": 1.0,
            "primary_identity_rmse_m": 0.9,
            "comparator_identity_rmse_m": 1.0,
        }
        for case_name in CALIBRATION_CASE_NAMES
    }


def _score_evidence(
    path: Path,
    permit: object,
    scores: dict[str, dict[str, float]],
) -> Path:
    lock_path = Path(permit.lock_path)
    lock = load_held_protocol_lock(lock_path)
    scored_frames = [
        *range(20, 38),
        *range(39, 57),
        *range(58, 76),
    ]
    case_records: dict[str, object] = {}
    evidence_files = path.parent / f"{path.stem}-files"
    evidence_files.mkdir(parents=True, exist_ok=True)
    for case_name in CALIBRATION_CASE_NAMES:
        seal_path = Path(dict(permit.seal_paths)[case_name])
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        authorization_path = Path(seal["prefix_authorization"]["path"])
        authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
        physical_path = Path(authorization["physical_prior_seal"]["path"])
        physical = json.loads(physical_path.read_text(encoding="utf-8"))
        frame_zero_path = Path(physical["frame_zero_manifest"]["path"])
        frame_zero = json.loads(frame_zero_path.read_text(encoding="utf-8"))
        target_path = evidence_files / f"{case_name}-target.bin"
        outcome_path = evidence_files / f"{case_name}-outcome.bin"
        target_path.write_bytes(f"target:{case_name}".encode())
        outcome_path.write_bytes(f"outcome:{case_name}".encode())
        object_id, episode = case_name.rsplit("-ep", maxsplit=1)
        score = scores[case_name]

        def detailed(identity: float, chamfer: float) -> dict[str, object]:
            return {
                "frame_count": len(scored_frames),
                "scored_frames": scored_frames,
                "permanently_excluded_center_count": 1,
                "post_update_hidden_identity_rmse_m": identity,
                "post_update_hidden_symmetric_chamfer_m": chamfer,
                "hidden_identity_count_per_frame": {
                    "minimum": 1,
                    "mean": 1.0,
                    "maximum": 1,
                },
                "by_frame": {
                    "hidden_identity_rmse_m": [identity] * len(scored_frames),
                    "hidden_symmetric_chamfer_m": [chamfer] * len(scored_frames),
                },
            }

        case_records[case_name] = {
            "case_name": case_name,
            "gate_score": dict(score),
            "scored_frames": scored_frames,
            "permanently_excluded_center_ids": [0],
            "identity_transport": {
                "algorithm": "scipy-sparse-minimum-weight-full-bipartite-matching",
                "scipy_version": "test-only",
                "maximum_assignment_distance_m": 0.015,
                "candidate_edge_count": 2,
                "sealed_point_coverage_fraction": 1.0,
                "assigned_official_identity_collision_count": 0,
                "assigned_official_identity_count": 2,
                "official_identity_count": 2,
                "mean_assignment_distance_m": 0.0,
                "p95_assignment_distance_m": 0.0,
                "observed_maximum_assignment_distance_m": 0.0,
                "assignment_ids_sha256": "1" * 64,
                "assignment_distances_sha256": "2" * 64,
                "eligible_official_frame_zero_identity_count": 2,
                "official_identity_ids_sha256": "3" * 64,
                "raw_official_frame_zero_sha256": "4" * 64,
                "sealed_frame_zero_sha256": "5" * 64,
                "transported_frame_zero_replaced_with_sealed_identity": True,
                "claim_limitation": (
                    "one-to-one transported official reconstruction proxy; not native "
                    "official material identity and not Deform360 Table-4 parity"
                ),
            },
            "scores": {
                "primary": detailed(
                    score["primary_identity_rmse_m"], score["primary_chamfer_m"]
                ),
                "selected_raw_backbone": detailed(
                    score["comparator_identity_rmse_m"],
                    score["comparator_chamfer_m"],
                ),
            },
            "sealed_inputs": {
                "online_prediction_seal": _record(seal_path),
                "online_prediction_archive": seal["online_artifacts"][
                    "online_prediction_archive"
                ],
                "physical_prediction_archive": physical["physical_artifacts"][
                    "physical_prediction_archive"
                ],
                "frame_zero_bundle": frame_zero["bundle"],
            },
            "outcome_provenance": {
                "target_artifact_kind": "Deform360OfficialReconstructionTarget",
                "outcome_artifact_kind": "Deform360HeldOfficialOutcome",
                "case_name": case_name,
                "object_id": object_id,
                "episode_id": int(episode),
                "dataset_revision": lock["dataset_revision"],
                "cohort_barrier_sha256": permit.cohort_barrier_sha256,
                "target_file": _record(target_path),
                "outcome_file": _record(outcome_path),
                "array_sha256": {
                    "object_points": "6" * 64,
                    "object_visibilities": "7" * 64,
                    "object_motions_valid": "8" * 64,
                },
                "information_boundary": {
                    "complete_cohort_barrier_validated_before_future_open": True,
                    "official_target_constructed_or_read_after_barrier": True,
                    "prediction_metric_computed_during_target_construction": False,
                },
            },
            "method_selection_or_tuning_performed": False,
        }
    artifact: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": CALIBRATION_SCORE_EVIDENCE_KIND,
        "protocol_id": PROTOCOL_ID,
        "role": "calibration",
        "cohort_barrier_sha256": permit.cohort_barrier_sha256,
        "lock": _record(lock_path),
        "outcome_reconstruction_contract_sha256": lock["immutable_bindings"][
            "outcome_reconstruction_contract"
        ],
        "ordered_case_names": list(CALIBRATION_CASE_NAMES),
        "metric_lock": METRIC_LOCK,
        "case_records": case_records,
        "information_boundary": {
            "all_15_online_predictions_sealed_before_any_outcome": True,
            "outcomes_opened_only_through_live_permit": True,
            "method_selection_or_tuning_performed": False,
            "confirmation_payload_read": False,
        },
    }
    artifact["artifact_sha256"] = held_artifact_sha256(artifact)
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


def _identity(case_name: str) -> tuple[str, int]:
    object_id, episode = case_name.rsplit("-ep", maxsplit=1)
    return object_id, int(episode)


def _frame_zero_manifest(
    root: Path,
    lock_path: Path,
    case_name: str,
    *,
    role: str = "confirmation",
    bundle_suffix: str = ".npz",
    boundary_override: dict[str, object] | None = None,
) -> Path:
    directory = root / case_name
    directory.mkdir(parents=True, exist_ok=True)
    bundle = directory / f"frame-zero{bundle_suffix}"
    config = default_frame_zero_config()
    cameras = tuple(
        sorted(
            (
                str(config["reference_camera"]),
                "cam1",
                "cam2",
                "cam3",
                "cam4",
                "cam5",
                "cam6",
                "cam7",
            )
        )
    )
    if bundle_suffix == ".npz":
        np.savez_compressed(bundle, camera_names=np.asarray(cameras))
    else:
        bundle.write_bytes(b"single extracted frame")
    robot, _selected_robot, action_alignment = write_robot_kinematics_fixture(
        directory
    )
    metadata = write_robot_metadata_fixture(
        directory / "robot.meta.json",
        source_frame_count=150,
        cameras=cameras,
    )
    source_start = int(action_alignment["selected_raw_frame_range_half_open"][0])
    object_id, episode_id = _identity(case_name)
    boundary: dict[str, object] = {
        "maximum_object_rgb_frame_read": 0,
        "object_observation_frames_used": [0],
        "known_aligned_realized_robot_kinematics_read": True,
        "known_robot_trajectory_semantics": action_alignment[
            "trajectory_semantics"
        ],
        "robot_delta_command_read": False,
        "commanded_control_read": False,
        "known_future_robot_action_read": True,
        "future_object_rgb_read": False,
        "future_object_geometry_read": False,
        "future_depth_or_mask_read": False,
        "future_tactile_read": False,
        "outcome_created": False,
        "outcome_read": False,
        "whole_future_container_hashed_or_read": False,
    }
    if boundary_override:
        boundary.update(boundary_override)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": FRAME_ZERO_KIND,
        "protocol_id": PROTOCOL_ID,
        "case_name": case_name,
        "object_id": object_id,
        "episode_id": episode_id,
        "role": role,
        "lock_sha256": _sha256(lock_path),
        "lock_artifact_sha256": load_held_protocol_lock(lock_path)["artifact_sha256"],
        "frame_indices": [0],
        "config": config,
        "bundle": _record(bundle),
        "action_inputs": {
            "robot_trajectory": bound_file(robot),
            "robot_metadata": bound_file(metadata),
        },
        "action_alignment": action_alignment,
        "camera_policy": {
            "policy_id": _FRAME_ZERO_CAMERA_SELECTION_POLICY_ID,
            "rule": _FRAME_ZERO_CAMERA_SELECTION_RULE,
            "reference_camera": config["reference_camera"],
            "minimum_selected_camera_count": config["minimum_camera_count"],
            "candidate_cameras": list(cameras),
            "candidate_camera_count": len(cameras),
            "selected_cameras": list(cameras),
            "selected_camera_count": len(cameras),
            "abstained_cameras": [],
            "abstained_camera_count": 0,
        },
        "camera_frame_zero_access": [
            {
                "camera": camera,
                "path": str((directory / camera / "undistorted.mp4").resolve()),
                "decoded_frame_count": 1,
                "maximum_rgb_frame_read": 0,
                "action_window_frame_index": 0,
                "source_aligned_frame_index": source_start,
                "decoded_rgb_sha256": "d" * 64,
                "whole_file_hashed_or_read": False,
            }
            for camera in cameras
        ],
        "information_boundary": boundary,
    }
    manifest["artifact_sha256"] = held_artifact_sha256(manifest)
    path = directory / "frame-zero-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _artifact_files(
    root: Path,
    case_name: str,
    roles: tuple[str, ...],
) -> dict[str, Path]:
    directory = root / case_name
    directory.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for role in roles:
        suffix = ".json" if "manifest" in role or "summary" in role else ".npz"
        path = directory / f"{role}{suffix}"
        path.write_text(f"{case_name}:{role}\n", encoding="utf-8")
        result[role] = path
    return result


def _seal_case(
    root: Path,
    lock_path: Path,
    case_name: str,
    *,
    role: str = "confirmation",
) -> tuple[Path, dict[str, Path]]:
    manifest = _frame_zero_manifest(root, lock_path, case_name, role=role)
    physical_files = _artifact_files(
        root / "physical-files", case_name, PHYSICAL_ARTIFACT_ROLES
    )
    physical_seal = root / "physical-seals" / f"{case_name}.json"
    create_physical_prior_seal(
        physical_seal,
        lock_path,
        manifest,
        physical_files,
        case_name=case_name,
        role=role,
    )
    prefix_authorization = root / "prefix-authorizations" / f"{case_name}.json"
    create_prefix_stage_authorization(
        prefix_authorization,
        lock_path,
        physical_seal,
    )
    online_files = _artifact_files(
        root / "online-files", case_name, ONLINE_ARTIFACT_ROLES
    )
    online_seal = root / "online-seals" / f"{case_name}.json"
    create_online_prediction_seal(
        online_seal,
        lock_path,
        prefix_authorization,
        online_files,
    )
    return online_seal, online_files


def _seal_confirmation_cohort(
    root: Path,
    lock_path: Path,
) -> tuple[dict[str, Path], dict[str, dict[str, Path]]]:
    seals: dict[str, Path] = {}
    artifacts: dict[str, dict[str, Path]] = {}
    for case in CONFIRMATION_CASES:
        seal, files = _seal_case(root, lock_path, case.case_name)
        seals[case.case_name] = seal
        artifacts[case.case_name] = files
    return seals, artifacts


def _seal_calibration_cohort(root: Path, lock_path: Path) -> dict[str, Path]:
    return {
        case_name: _seal_case(
            root,
            lock_path,
            case_name,
            role="calibration",
        )[0]
        for case_name in CALIBRATION_CASE_NAMES
    }


def _lock(tmp_path: Path) -> Path:
    calibration_lock = _calibration_lock(tmp_path)
    calibration_seals = _seal_calibration_cohort(
        tmp_path / "calibration-chain",
        calibration_lock,
    )
    permit = authorize_outcome_phase(
        calibration_lock,
        calibration_seals,
        role="calibration",
    )
    decision_path = tmp_path / "calibration-decision.json"
    scores = _passing_scores()
    evidence_path = _score_evidence(
        tmp_path / "calibration-score-evidence.json", permit, scores
    )
    create_calibration_gate_decision(
        decision_path,
        permit,
        scores,
        score_evidence_path=evidence_path,
    )
    confirmation_lock = tmp_path / "confirmation-lock.json"
    create_confirmation_protocol_lock(
        confirmation_lock,
        calibration_lock,
        decision_path,
    )
    return confirmation_lock


def test_lock_freezes_exact_cases_method_metrics_and_decision_gates(
    tmp_path: Path,
) -> None:
    lock_path = _lock(tmp_path)
    lock = load_held_protocol_lock(lock_path)

    assert locked_case_names(lock_path) == tuple(
        case.case_name for case in CONFIRMATION_CASES
    )
    assert locked_case_names(lock_path, role="calibration") == CALIBRATION_CASE_NAMES
    assert lock["primary_method"] == PRIMARY_METHOD
    assert lock["primary_method"]["calibration_selects_method"] is False
    assert CALIBRATION_GATE["minimum_case_chamfer_wins"] == 10
    assert CALIBRATION_GATE["no_go_keeps_confirmation_payload_sealed"] is True
    assert CONFIRMATION_GATE["required_case_chamfer_wins"] == 6
    assert CONFIRMATION_GATE["one_sided_sign_test_p"] == 1.0 / 64.0

    mutated = deepcopy(lock)
    mutated["case_whitelist"] = mutated["case_whitelist"][:-1]
    mutated["artifact_sha256"] = held_artifact_sha256(mutated)
    mutated_path = tmp_path / "mutated-lock.json"
    mutated_path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(ValueError, match="confirmation whitelist changed"):
        load_held_protocol_lock(mutated_path)


def test_lock_requires_the_exact_immutable_binding_key_set(tmp_path: Path) -> None:
    missing = dummy_immutable_bindings()
    missing.pop(REQUIRED_IMMUTABLE_BINDING_KEYS[0])
    with pytest.raises(ValueError, match="immutable binding keys changed"):
        create_held_protocol_lock(tmp_path / "missing.json", immutable_bindings=missing)

    extra = dummy_immutable_bindings()
    extra["unlocked_runtime_component"] = "f" * 64
    with pytest.raises(ValueError, match="immutable binding keys changed"):
        create_held_protocol_lock(tmp_path / "extra.json", immutable_bindings=extra)

    invalid_report = dummy_immutable_bindings()
    invalid_report["v1_preoutcome_feasibility_report"] = "not-a-sha256"
    with pytest.raises(ValueError, match="must be a named SHA-256"):
        create_held_protocol_lock(
            tmp_path / "invalid-report.json",
            immutable_bindings=invalid_report,
        )

    assert not (tmp_path / "missing.json").exists()
    assert not (tmp_path / "extra.json").exists()
    assert not (tmp_path / "invalid-report.json").exists()


def test_v2_lock_binds_the_abandoned_v1_source_feasibility_amendment(
    tmp_path: Path,
) -> None:
    bindings = dummy_immutable_bindings()
    lock_path = tmp_path / "v2-lock.json"
    lock = create_held_protocol_lock(lock_path, immutable_bindings=bindings)

    assert lock["protocol_id"] == "deform360-held-online-belief-v2"
    assert set(REQUIRED_IMMUTABLE_BINDING_KEYS) >= {
        "v1_preoutcome_feasibility_report",
        "held_source_feasibility_amendment_contract",
    }
    assert (
        lock["immutable_bindings"]["held_source_feasibility_amendment_contract"]
        == hashlib.sha256(
            json.dumps(
                SOURCE_FEASIBILITY_AMENDMENT_CONTRACT,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    )
    assert SOURCE_FEASIBILITY_AMENDMENT_CONTRACT["parent_execution"] == {
        "protocol_id": "deform360-held-online-belief-v1",
        "disposition": "ABANDONED_PREOUTCOME",
        "evidence_binding_key": "v1_preoutcome_feasibility_report",
        "exact_target_free_census": {
            "requested_case_count": 15,
            "sealed_case_count": 5,
            "frame_zero_failure_count": 9,
            "physical_admission_failure_count": 1,
        },
        "predictions_reused_by_v2": False,
    }
    assert SOURCE_FEASIBILITY_AMENDMENT_CONTRACT["information_boundary"] == {
        "selection_evidence": (
            "frame-zero source inputs and automatic-twin admission diagnostics only"
        ),
        "outcome_payloads_accessed": False,
        "target_payloads_accessed": False,
        "confirmation_payloads_accessed": False,
        "outcome_permit_created": False,
    }

    wrong_contract = dummy_immutable_bindings()
    wrong_contract["held_source_feasibility_amendment_contract"] = "f" * 64
    with pytest.raises(
        ValueError,
        match="source-feasibility amendment contract binding changed",
    ):
        create_held_protocol_lock(
            tmp_path / "wrong-contract.json",
            immutable_bindings=wrong_contract,
        )
    assert not (tmp_path / "wrong-contract.json").exists()

    tampered_lock = deepcopy(lock)
    tampered_lock["immutable_bindings"][
        "held_source_feasibility_amendment_contract"
    ] = "f" * 64
    tampered_lock["artifact_sha256"] = held_artifact_sha256(tampered_lock)
    tampered_path = tmp_path / "tampered-contract-lock.json"
    tampered_path.write_text(json.dumps(tampered_lock), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="source-feasibility amendment contract binding changed",
    ):
        load_held_protocol_lock(tampered_path)


def test_frame_zero_contract_allows_action_but_rejects_object_future_and_hdf5(
    tmp_path: Path,
) -> None:
    lock_path = _lock(tmp_path)
    case_name = CONFIRMATION_CASES[0].case_name
    valid = _frame_zero_manifest(tmp_path / "valid", lock_path, case_name)
    manifest = validate_frame_zero_bundle_manifest(valid, lock_path)
    assert manifest["information_boundary"]["known_future_robot_action_read"] is True
    assert manifest["information_boundary"]["object_observation_frames_used"] == [0]

    future = _frame_zero_manifest(
        tmp_path / "future",
        lock_path,
        case_name,
        boundary_override={"future_object_rgb_read": True},
    )
    with pytest.raises(ValueError, match="object-future boundary"):
        validate_frame_zero_bundle_manifest(future, lock_path)

    future_container = _frame_zero_manifest(
        tmp_path / "hdf5",
        lock_path,
        case_name,
        bundle_suffix=".h5",
    )
    with pytest.raises(ValueError, match="extracted"):
        validate_frame_zero_bundle_manifest(future_container, lock_path)


def _rewrite_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **arrays)


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {name: np.asarray(stored[name]).copy() for name in stored.files}


def _rewrite_frame_manifest(path: Path, manifest: dict[str, object]) -> None:
    manifest["artifact_sha256"] = held_artifact_sha256(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("extra_field", "field set changed"),
        ("wrong_dtype", "actions must have dtype float64"),
        ("parity", "row 0 does not match"),
        ("nonscalar_bimanual", "bimanual must be a bool scalar"),
    ),
)
def test_frame_zero_robot_archive_schema_and_parity_fail_closed(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    lock_path = _calibration_lock(tmp_path)
    case_name = CALIBRATION_CASE_NAMES[0]
    manifest_path = _frame_zero_manifest(
        tmp_path / "frame-zero", lock_path, case_name, role="calibration"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    robot_path = Path(manifest["action_inputs"]["robot_trajectory"]["path"])
    arrays = _load_npz_arrays(robot_path)
    if mutation == "extra_field":
        arrays["unexpected"] = np.asarray(1, dtype=np.int64)
    elif mutation == "wrong_dtype":
        arrays["actions"] = arrays["actions"].astype(np.float32)
    elif mutation == "parity":
        arrays["actions"][0, 0, 0] += 0.1
    else:
        arrays["bimanual"] = np.asarray([False], dtype=np.bool_)
    _rewrite_npz(robot_path, arrays)
    manifest["action_inputs"]["robot_trajectory"] = _record(robot_path)
    _rewrite_frame_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match=message):
        validate_frame_zero_bundle_manifest(
            manifest_path, lock_path, expected_role="calibration"
        )


def test_frame_zero_robot_selection_start_and_selected_slice_fail_closed(
    tmp_path: Path,
) -> None:
    lock_path = _calibration_lock(tmp_path)
    case_name = CALIBRATION_CASE_NAMES[0]
    manifest_path = _frame_zero_manifest(
        tmp_path / "frame-zero", lock_path, case_name, role="calibration"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["action_alignment"]["selected_raw_frame_range_half_open"] = [68, 149]
    _rewrite_frame_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="selection mirrors changed"):
        validate_frame_zero_bundle_manifest(
            manifest_path, lock_path, expected_role="calibration"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_range = manifest["action_alignment"]["selection_audit"][
        "selected_raw_frame_range_half_open"
    ]
    manifest["action_alignment"]["selected_raw_frame_range_half_open"] = expected_range
    manifest["camera_frame_zero_access"][0]["source_aligned_frame_index"] += 1
    _rewrite_frame_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="camera source index"):
        validate_frame_zero_bundle_manifest(
            manifest_path, lock_path, expected_role="calibration"
        )
    manifest["camera_frame_zero_access"][0]["source_aligned_frame_index"] -= 1
    removed_camera = manifest["camera_policy"]["selected_cameras"].pop()
    manifest["camera_policy"]["selected_camera_count"] -= 1
    manifest["camera_policy"]["abstained_cameras"] = [removed_camera]
    manifest["camera_policy"]["abstained_camera_count"] = 1
    _rewrite_frame_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="partition or bundle order"):
        validate_frame_zero_bundle_manifest(
            manifest_path, lock_path, expected_role="calibration"
        )
    manifest["camera_policy"]["selected_cameras"].append(removed_camera)
    manifest["camera_policy"]["selected_camera_count"] += 1
    manifest["camera_policy"]["abstained_cameras"] = []
    manifest["camera_policy"]["abstained_camera_count"] = 0
    for field in ("policy_id", "rule"):
        original = manifest["camera_policy"][field]
        manifest["camera_policy"][field] = "tampered"
        _rewrite_frame_manifest(manifest_path, manifest)
        with pytest.raises(ValueError, match="reference policy changed"):
            validate_frame_zero_bundle_manifest(
                manifest_path, lock_path, expected_role="calibration"
            )
        manifest["camera_policy"][field] = original
    selected_path = Path(
        manifest["action_alignment"]["selected_robot_kinematics_bundle"]["path"]
    )
    selected_arrays = _load_npz_arrays(selected_path)
    selected_arrays["T_worlds"][:, 0, 3] += 0.1
    selected_arrays["actions"][:, 0, 0] += 0.1
    _rewrite_npz(selected_path, selected_arrays)
    selected_record = _record(selected_path)
    selected_state = load_robot_kinematics_archive(selected_path)
    manifest["action_alignment"]["selected_robot_kinematics_bundle"] = selected_record
    manifest["action_alignment"]["selected_action_bundle"] = selected_record
    manifest["action_alignment"]["selected_action_arrays"] = (
        robot_kinematics_array_records(selected_state)
    )
    _rewrite_frame_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="exact source slice"):
        validate_frame_zero_bundle_manifest(
            manifest_path, lock_path, expected_role="calibration"
        )


def test_calibration_authorization_cannot_be_relabelled_as_confirmation(
    tmp_path: Path,
) -> None:
    lock_path = _calibration_lock(tmp_path)
    case_name = "083-blanket-cloth-ep0000"
    manifest = _frame_zero_manifest(
        tmp_path,
        lock_path,
        case_name,
        role="calibration",
    )
    validate_frame_zero_bundle_manifest(
        manifest,
        lock_path,
        expected_role="calibration",
    )
    physical_files = _artifact_files(
        tmp_path / "physical", case_name, PHYSICAL_ARTIFACT_ROLES
    )
    with pytest.raises(ValueError, match="confirmation remains sealed"):
        create_physical_prior_seal(
            tmp_path / "physical-seal.json",
            lock_path,
            manifest,
            physical_files,
            case_name=case_name,
            role="confirmation",
        )

    confirmation_case = CONFIRMATION_CASES[0].case_name
    confirmation_manifest = _frame_zero_manifest(
        tmp_path / "confirmation",
        lock_path,
        confirmation_case,
        role="confirmation",
    )
    with pytest.raises(ValueError, match="confirmation remains sealed"):
        validate_frame_zero_bundle_manifest(
            confirmation_manifest,
            lock_path,
            expected_role="calibration",
        )


def test_calibration_no_go_cannot_create_a_confirmation_lock(tmp_path: Path) -> None:
    calibration_lock = _calibration_lock(tmp_path)
    seals = _seal_calibration_cohort(
        tmp_path / "calibration-chain",
        calibration_lock,
    )
    permit = authorize_outcome_phase(
        calibration_lock,
        seals,
        role="calibration",
    )
    decision_path = tmp_path / "no-go.json"
    scores = _passing_scores(primary_chamfer_m=1.10)
    evidence_path = _score_evidence(tmp_path / "no-go-evidence.json", permit, scores)
    decision = create_calibration_gate_decision(
        decision_path,
        permit,
        scores,
        score_evidence_path=evidence_path,
    )
    assert decision["decision"] == "NO_GO"

    with pytest.raises(ValueError, match="remains sealed"):
        create_confirmation_protocol_lock(
            tmp_path / "forbidden-confirmation-lock.json",
            calibration_lock,
            decision_path,
        )

    target_case = CONFIRMATION_CASES[0].case_name
    target_manifest = _frame_zero_manifest(
        tmp_path / "target",
        calibration_lock,
        target_case,
    )
    with pytest.raises(ValueError, match="confirmation remains sealed"):
        validate_frame_zero_bundle_manifest(target_manifest, calibration_lock)


def test_calibration_decision_scores_must_equal_sealed_evidence(
    tmp_path: Path,
) -> None:
    calibration_lock = _calibration_lock(tmp_path)
    seals = _seal_calibration_cohort(tmp_path / "calibration-chain", calibration_lock)
    permit = authorize_outcome_phase(calibration_lock, seals, role="calibration")
    evidence_scores = _passing_scores(primary_chamfer_m=0.90)
    evidence_path = _score_evidence(
        tmp_path / "calibration-score-evidence.json",
        permit,
        evidence_scores,
    )
    mismatched_scores = _passing_scores(primary_chamfer_m=0.89)

    with pytest.raises(ValueError, match="differ from immutable score evidence"):
        create_calibration_gate_decision(
            tmp_path / "forbidden-decision.json",
            permit,
            mismatched_scores,
            score_evidence_path=evidence_path,
        )

    assert not (tmp_path / "forbidden-decision.json").exists()


def test_calibration_gate_rejects_skeletal_self_hashed_score_evidence(
    tmp_path: Path,
) -> None:
    calibration_lock = _calibration_lock(tmp_path)
    seals = _seal_calibration_cohort(tmp_path / "calibration-chain", calibration_lock)
    permit = authorize_outcome_phase(calibration_lock, seals, role="calibration")
    scores = _passing_scores()
    evidence_path = _score_evidence(tmp_path / "complete-evidence.json", permit, scores)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    for record in evidence["case_records"].values():
        for key in tuple(record):
            if key not in {
                "case_name",
                "gate_score",
                "method_selection_or_tuning_performed",
            }:
                record.pop(key)
    evidence["artifact_sha256"] = held_artifact_sha256(evidence)
    skeletal = tmp_path / "skeletal-evidence.json"
    skeletal.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(ValueError, match="exact score record schema"):
        create_calibration_gate_decision(
            tmp_path / "forbidden-decision.json",
            permit,
            scores,
            score_evidence_path=skeletal,
        )

    assert not (tmp_path / "forbidden-decision.json").exists()


def test_physical_and_online_seals_are_ordered_and_write_once(tmp_path: Path) -> None:
    lock_path = _lock(tmp_path)
    case_name = CONFIRMATION_CASES[0].case_name
    manifest = _frame_zero_manifest(tmp_path, lock_path, case_name)
    physical_files = _artifact_files(
        tmp_path / "physical", case_name, PHYSICAL_ARTIFACT_ROLES
    )
    physical_seal = tmp_path / "physical-seal.json"
    create_physical_prior_seal(
        physical_seal,
        lock_path,
        manifest,
        physical_files,
        case_name=case_name,
    )
    with pytest.raises(FileExistsError):
        create_physical_prior_seal(
            physical_seal,
            lock_path,
            manifest,
            physical_files,
            case_name=case_name,
        )

    online_files = _artifact_files(
        tmp_path / "online", case_name, ONLINE_ARTIFACT_ROLES
    )
    with pytest.raises(ValueError, match="unsupported prefix authorization"):
        create_online_prediction_seal(
            tmp_path / "online-seal.json",
            lock_path,
            physical_seal,
            online_files,
        )

    prefix = tmp_path / "prefix-authorization.json"
    create_prefix_stage_authorization(prefix, lock_path, physical_seal)
    physical_files["physical_prediction_archive"].write_text(
        "changed after physical seal\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="file binding changed"):
        create_online_prediction_seal(
            tmp_path / "online-seal-after-change.json",
            lock_path,
            prefix,
            online_files,
        )


def test_outcome_callback_requires_all_six_valid_prediction_seals(
    tmp_path: Path,
) -> None:
    lock_path = _lock(tmp_path)
    seals, _ = _seal_confirmation_cohort(tmp_path / "cohort", lock_path)
    incomplete = dict(seals)
    incomplete.pop(CONFIRMATION_CASES[-1].case_name)

    with pytest.raises(ValueError, match="every exact cohort seal"):
        authorize_outcome_phase(lock_path, incomplete)

    extra = {**seals, "unlocked-extra-case": next(iter(seals.values()))}
    with pytest.raises(ValueError, match="every exact cohort seal"):
        authorize_outcome_phase(lock_path, extra)

    permit = authorize_outcome_phase(lock_path, seals)
    calls: list[str] = []
    result = run_outcome_operation(
        permit,
        case_name=CONFIRMATION_CASES[0].case_name,
        operation="read",
        callback=lambda: calls.append("opened") or "outcome",
    )
    assert result == "outcome"
    assert calls == ["opened"]


def test_permit_fails_closed_if_a_bound_prediction_changes(tmp_path: Path) -> None:
    lock_path = _lock(tmp_path)
    seals, artifacts = _seal_confirmation_cohort(tmp_path / "cohort", lock_path)
    permit = authorize_outcome_phase(lock_path, seals)
    case_name = CONFIRMATION_CASES[2].case_name
    artifacts[case_name]["online_prediction_archive"].write_text(
        "post-authorization mutation\n",
        encoding="utf-8",
    )
    calls: list[str] = []

    with pytest.raises(ValueError, match="file binding changed"):
        run_outcome_operation(
            permit,
            case_name=case_name,
            operation="create",
            callback=lambda: calls.append("created"),
        )
    assert calls == []


def test_preoutcome_builders_have_no_target_or_outcome_argument() -> None:
    for function in (
        create_physical_prior_seal,
        create_prefix_stage_authorization,
        create_online_prediction_seal,
    ):
        parameters = inspect.signature(function).parameters
        assert "target" not in parameters
        assert "outcome" not in parameters
