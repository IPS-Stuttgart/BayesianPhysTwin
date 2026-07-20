from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import bayesian_phystwin.deform360_held_protocol as held_protocol
import bayesian_phystwin.deform360_held_outcome_scoring as held_scoring
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
    CONFIRMATION_CASE_NAMES,
    CONFIRMATION_DECISION_KIND,
    CONFIRMATION_GATE,
    CONFIRMATION_SCORE_EVIDENCE_KIND,
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
    held_contract_sha256,
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


def _passing_scores(
    *,
    primary_chamfer_m: float = 0.90,
    case_names: tuple[str, ...] = CALIBRATION_CASE_NAMES,
) -> dict[str, dict[str, float]]:
    return {
        case_name: {
            "primary_chamfer_m": primary_chamfer_m,
            "comparator_chamfer_m": 1.0,
            "primary_identity_rmse_m": 0.9,
            "comparator_identity_rmse_m": 1.0,
        }
        for case_name in case_names
    }


def _score_evidence(
    path: Path,
    permit: object,
    scores: dict[str, dict[str, float]],
    *,
    role: str = "calibration",
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
    case_names = (
        CALIBRATION_CASE_NAMES if role == "calibration" else CONFIRMATION_CASE_NAMES
    )
    for case_name in case_names:
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
                    "material identity and not parity with the Deform360 Tables 3-5 "
                    "world-model benchmarks "
                    "or their native tactile-refined material identities"
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
        "artifact_kind": (
            CALIBRATION_SCORE_EVIDENCE_KIND
            if role == "calibration"
            else CONFIRMATION_SCORE_EVIDENCE_KIND
        ),
        "protocol_id": PROTOCOL_ID,
        "role": role,
        "cohort_barrier_sha256": permit.cohort_barrier_sha256,
        "lock": _record(lock_path),
        "outcome_reconstruction_contract_sha256": lock["immutable_bindings"][
            "outcome_reconstruction_contract"
        ],
        "ordered_case_names": list(case_names),
        "metric_lock": METRIC_LOCK,
        "case_records": case_records,
        "information_boundary": (
            {
                "all_15_online_predictions_sealed_before_any_outcome": True,
                "outcomes_opened_only_through_live_permit": True,
                "method_selection_or_tuning_performed": False,
                "confirmation_payload_read": False,
            }
            if role == "calibration"
            else {
                "all_6_online_predictions_sealed_before_any_outcome": True,
                "outcomes_opened_only_through_live_permit": True,
                "method_selection_or_tuning_performed": False,
                "calibration_method_and_gate_unchanged": True,
            }
        ),
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
    robot, _selected_robot, action_alignment = write_robot_kinematics_fixture(directory)
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
        "known_robot_trajectory_semantics": action_alignment["trajectory_semantics"],
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

    mutated = deepcopy(lock)
    mutated["case_whitelist"] = mutated["case_whitelist"][:-1]
    mutated["artifact_sha256"] = held_artifact_sha256(mutated)
    mutated_path = tmp_path / "mutated-lock.json"
    mutated_path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(ValueError, match="confirmation whitelist changed"):
        load_held_protocol_lock(mutated_path)


def test_exact_six_case_confirmation_gate_arithmetic_passes() -> None:
    scores = _passing_scores(case_names=CONFIRMATION_CASE_NAMES)
    normalized, summary = held_protocol._confirmation_gate_summary(scores)

    assert tuple(normalized) == CONFIRMATION_CASE_NAMES
    assert summary["case_chamfer_wins"] == 6
    assert summary["one_sided_sign_test_p"] == 1.0 / 64.0
    assert summary["equal_case_mean_chamfer_improvement_fraction"] == (
        pytest.approx(0.1)
    )
    assert all(summary["checks"].values())
    assert summary["passed"] is True
    assert CONFIRMATION_DECISION_KIND == "Deform360HeldConfirmationDecision"
    assert CONFIRMATION_SCORE_EVIDENCE_KIND == (
        "Deform360HeldConfirmationScoreEvidence"
    )


def test_confirmation_gate_reports_all_cases_and_rejects_one_nonwin() -> None:
    scores = _passing_scores(case_names=CONFIRMATION_CASE_NAMES)
    scores[CONFIRMATION_CASE_NAMES[-1]]["primary_chamfer_m"] = 1.0
    _normalized, summary = held_protocol._confirmation_gate_summary(scores)

    assert summary["passed"] is False
    assert summary["case_chamfer_wins"] == 5
    assert summary["one_sided_sign_test_p"] == 7.0 / 64.0
    assert summary["checks"]["all_6_cases_chamfer_win"] is False
    assert summary["checks"]["one_sided_sign_test_p_is_1_over_64"] is False


def test_confirmation_evidence_and_final_decision_end_to_end_are_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the complete confirmation artifact chain without GPU payloads."""

    lock_path = tmp_path / "confirmation-lock.json"
    lock_path.write_text("test-only confirmation lock\n", encoding="utf-8")
    lock = {
        "stage": "confirmation",
        "case_whitelist": list(CONFIRMATION_CASE_NAMES),
        "calibration_case_whitelist": [],
        "immutable_bindings": {
            "outcome_reconstruction_contract": held_contract_sha256(
                held_scoring.OUTCOME_RECONSTRUCTION_CONTRACT
            )
        },
    }
    seal_paths: dict[str, Path] = {}
    for case_name in CONFIRMATION_CASE_NAMES:
        path = tmp_path / "seals" / f"{case_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"sealed:{case_name}\n", encoding="utf-8")
        seal_paths[case_name] = path

    def fake_online_seal(
        path: str | Path,
        _lock: str | Path,
        *,
        expected_case_name: str,
        expected_role: str,
    ) -> dict[str, str]:
        assert Path(path) == seal_paths[expected_case_name]
        assert expected_role == "confirmation"
        digest = hashlib.sha256(expected_case_name.encode()).hexdigest()
        return {"artifact_sha256": digest}

    monkeypatch.setattr(held_protocol, "load_held_protocol_lock", lambda _path: lock)
    monkeypatch.setattr(
        held_scoring,
        "load_held_protocol_lock",
        lambda _path: lock,
    )
    monkeypatch.setattr(
        held_protocol,
        "validate_online_prediction_seal",
        fake_online_seal,
    )
    monkeypatch.setattr(
        held_protocol,
        "_validate_case_score_record",
        lambda *_args, **_kwargs: None,
    )
    permit = authorize_outcome_phase(
        lock_path,
        seal_paths,
        role="confirmation",
    )
    scores = _passing_scores(case_names=CONFIRMATION_CASE_NAMES)
    records = {
        case_name: {
            "case_name": case_name,
            "gate_score": dict(scores[case_name]),
            "method_selection_or_tuning_performed": False,
        }
        for case_name in CONFIRMATION_CASE_NAMES
    }
    monkeypatch.setattr(
        held_scoring,
        "score_confirmation_cohort",
        lambda observed_permit, _operations: (
            (
                scores,
                records,
            )
            if observed_permit is permit
            else (_ for _ in ()).throw(AssertionError("another permit"))
        ),
    )
    evidence_path = tmp_path / "confirmation-score-evidence.json"
    decision_path = tmp_path / "confirmation-final-decision.json"

    decision, evidence, observed_records = (
        held_scoring.score_and_create_confirmation_gate(
            decision_path,
            permit,
            {},
            evidence_path=evidence_path,
        )
    )

    assert tuple(observed_records) == CONFIRMATION_CASE_NAMES
    assert evidence["role"] == "confirmation"
    assert evidence["cohort_barrier_sha256"] == permit.cohort_barrier_sha256
    assert decision["decision"] == "CONFIRMED"
    assert decision["summary"]["one_sided_sign_test_p"] == 1.0 / 64.0

    wrong_barrier = deepcopy(evidence)
    wrong_barrier["cohort_barrier_sha256"] = "0" * 64
    wrong_barrier["artifact_sha256"] = held_artifact_sha256(wrong_barrier)
    wrong_barrier_path = tmp_path / "wrong-barrier-evidence.json"
    wrong_barrier_path.write_text(json.dumps(wrong_barrier), encoding="utf-8")
    with pytest.raises(ValueError, match="another cohort barrier"):
        held_protocol.validate_confirmation_score_evidence(
            wrong_barrier_path,
            permit,
        )

    wrong_role = deepcopy(evidence)
    wrong_role["role"] = "calibration"
    wrong_role["artifact_sha256"] = held_artifact_sha256(wrong_role)
    wrong_role_path = tmp_path / "wrong-role-evidence.json"
    wrong_role_path.write_text(json.dumps(wrong_role), encoding="utf-8")
    with pytest.raises(ValueError, match="role changed"):
        held_protocol.validate_confirmation_score_evidence(
            wrong_role_path,
            permit,
        )

    tampered_decision = deepcopy(decision)
    tampered_decision["summary"]["case_chamfer_wins"] = 5
    tampered_decision["artifact_sha256"] = held_artifact_sha256(tampered_decision)
    tampered_path = tmp_path / "tampered-final-decision.json"
    tampered_path.write_text(json.dumps(tampered_decision), encoding="utf-8")
    with pytest.raises(ValueError, match="arithmetic changed"):
        held_protocol.validate_confirmation_gate_decision(
            tampered_path,
            lock_path,
        )
    assert CONFIRMATION_GATE["required_case_chamfer_wins"] == 6
    assert CONFIRMATION_GATE["one_sided_sign_test_p"] == 1.0 / 64.0


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


def test_v5_lock_binds_v1_through_failed_closed_v4_lineage(
    tmp_path: Path,
) -> None:
    bindings = dummy_immutable_bindings()
    lock_path = tmp_path / "v5-lock.json"
    lock = create_held_protocol_lock(lock_path, immutable_bindings=bindings)

    assert lock["protocol_id"] == "deform360-held-online-belief-v5"
    assert len(REQUIRED_IMMUTABLE_BINDING_KEYS) == 112
    assert set(REQUIRED_IMMUTABLE_BINDING_KEYS) >= {
        "v1_preoutcome_feasibility_report",
        "v2_design_withdrawal_report",
        "v3_prelock_boundary_incident_report",
        "v4_execution_withdrawal_report",
        "held_frozen_runtime_manifest",
        "held_source_feasibility_amendment_contract",
        "deform360_robot_kinematics_source",
        "robot_kinematics_window_contract",
        "frame_zero_semantic_gate_contract",
        "frame_zero_siglip2_model_tree",
        "frame_zero_siglip2_revision_literal",
        "frame_zero_siglip2_transformers_sources",
        "held_calibration_case_runner_source",
        "held_calibration_outcome_driver_source",
        "held_calibration_shard_runner_source",
        "held_protocol_lock_operator_source",
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
    assert SOURCE_FEASIBILITY_AMENDMENT_CONTRACT["v1_execution"] == {
        "protocol_id": "deform360-held-online-belief-v1",
        "disposition": "ABANDONED_PREOUTCOME",
        "evidence_binding_key": "v1_preoutcome_feasibility_report",
        "exact_target_free_census": {
            "requested_case_count": 15,
            "sealed_case_count": 5,
            "frame_zero_failure_count": 9,
            "physical_admission_failure_count": 1,
        },
        "outcome_payloads_accessed": False,
        "target_payloads_accessed": False,
        "confirmation_payloads_accessed": False,
        "outcome_permit_created": False,
        "execution_artifacts_reused_by_v5": False,
        "predictions_reused_by_v5": False,
    }
    assert SOURCE_FEASIBILITY_AMENDMENT_CONTRACT["v2_design"] == {
        "protocol_id": "deform360-held-online-belief-v2",
        "disposition": "WITHDRAWN_BEFORE_LOCK_AND_PREDICTION",
        "evidence_binding_key": "v2_design_withdrawal_report",
        "exact_execution_census": {
            "calibration_lock_count": 0,
            "case_attempt_count": 0,
            "deployed_snapshot_count": 0,
            "frame_zero_manifest_count": 0,
            "online_prediction_seal_count": 0,
            "outcome_created_count": 0,
            "outcome_permit_count": 0,
            "outcome_read_count": 0,
            "physical_prior_seal_count": 0,
            "prediction_count": 0,
            "prefix_authorization_count": 0,
            "shard_start_count": 0,
            "target_operation_count": 0,
        },
        "information_access": {
            "confirmation_payload_read": False,
            "episode_payload_read": False,
            "frame_zero_payload_read": False,
            "future_tactile_read": False,
            "outcome_read": False,
            "prediction_payload_read": False,
            "target_data_read": False,
            "target_or_outcome_path_accessed": False,
        },
        "execution_artifacts_reused_by_v5": False,
        "predictions_reused_by_v5": False,
    }
    v3_design = SOURCE_FEASIBILITY_AMENDMENT_CONTRACT["v3_design"]
    assert v3_design["protocol_id"] == "deform360-held-online-belief-v3"
    assert v3_design["disposition"] == "WITHDRAWN_BEFORE_LOCK_AND_PREDICTION"
    assert v3_design["evidence_binding_key"] == ("v3_prelock_boundary_incident_report")
    assert set(v3_design["exact_formal_protocol_execution_census"].values()) == {0}
    assert (
        "excludes the separately disclosed rg content scanner"
        in v3_design["formal_protocol_execution_scope"]
    )
    incident = v3_design["prelock_boundary_incident"]
    assert incident["program"] == "rg"
    assert incident["mode"] == "-l"
    assert incident["stdout_consumer"] == "head"
    assert incident["stdout_maximum_line_count"] == 100
    assert incident["content_scanner_may_have_opened_any_regular_file"] is True
    assert incident["protected_file_open_status"] == "NOT_CLAIMED"
    assert incident["only_matching_absolute_filenames_returned"] is True
    assert incident["included_unrelated_171_outcome_or_log_paths"] is True
    assert incident["payload_bytes_metrics_labels_arrays_or_values_returned"] is False
    assert (
        incident["held_cohort_payload_content_or_value_returned_to_research_agent"]
        is False
    )
    assert incident["method_or_gate_choice_used_outcome_values"] is False
    assert "outcome_payloads_accessed" not in v3_design

    repairs = SOURCE_FEASIBILITY_AMENDMENT_CONTRACT["v4_repairs"]
    assert repairs["robot_window_selection"]["archive_fields"] == [
        "format_version",
        "actions",
        "T_worlds",
        "openings",
        "bimanual",
    ]
    assert repairs["robot_window_selection"]["selection_translation"] == (
        "T_worlds[..., :3, 3]"
    )
    assert (
        repairs["robot_window_selection"][
            "selected_bundle_must_replay_exact_source_slice"
        ]
        is True
    )
    assert (
        repairs["robot_window_selection"][
            "camera_frame_zero_must_match_selected_action_window_start"
        ]
        is True
    )
    assert (
        repairs["frame_zero_geometry"][
            "official_current_frame_urdf_robot_exclusion_required"
        ]
        is True
    )
    assert (
        repairs["frame_zero_geometry"][
            "pinned_siglip2_exclusive_semantic_rank_gate_required"
        ]
        is True
    )
    assert repairs["information_boundary"] == {
        "rg_content_scanner_may_have_opened_any_regular_file_under_search_roots": True,
        "protected_file_open_status": "NOT_CLAIMED",
        "held_cohort_payload_content_or_value_returned_to_research_agent": False,
        "outcome_metric_label_array_or_value_returned_to_research_agent": False,
        "method_or_gate_choice_used_outcome_values": False,
    }
    assert "outcome_payloads_accessed" not in repairs["information_boundary"]
    v4_execution = SOURCE_FEASIBILITY_AMENDMENT_CONTRACT["v4_execution"]
    assert v4_execution["protocol_id"] == "deform360-held-online-belief-v4"
    assert v4_execution["disposition"] == (
        "WITHDRAWN_AFTER_FRAME_ZERO_BEFORE_PHYSICAL_PREDICTION"
    )
    assert v4_execution["evidence_binding_key"] == ("v4_execution_withdrawal_report")
    assert (
        v4_execution["exact_execution_census"]["physical_builder_invocation_count"] == 2
    )
    assert (
        v4_execution["exact_execution_census"]["formal_physical_prediction_count"] == 0
    )
    assert v4_execution["exact_execution_census"]["formal_online_prediction_count"] == 0
    assert (
        v4_execution["information_boundary"]["object_future_rgb_depth_or_tracking_read"]
        is False
    )
    assert v4_execution["information_boundary"]["episode_payload_scope"] == (
        "frame-zero RGB-D and masks; the frame-zero pipeline read the full "
        "realized robot archive to select the window, then sealed the aligned "
        "76-frame robot-kinematics window"
    )
    assert v4_execution["failure"]["failure_time_inventory_recorded"] is False

    assert SOURCE_FEASIBILITY_AMENDMENT_CONTRACT["reuse"] == {
        "v1_execution_artifacts_reused_by_v5": False,
        "v1_predictions_reused_by_v5": False,
        "v2_execution_artifacts_reused_by_v5": False,
        "v2_predictions_reused_by_v5": False,
        "v3_execution_artifacts_reused_by_v5": False,
        "v3_predictions_reused_by_v5": False,
        "v4_execution_artifacts_reused_by_v5": False,
        "v4_predictions_reused_by_v5": False,
        "sealed_lineage_reports_bound_by_v5": [
            "v1_preoutcome_feasibility_report",
            "v2_design_withdrawal_report",
            "v3_prelock_boundary_incident_report",
            "v4_execution_withdrawal_report",
        ],
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


def test_held_reference_optional_recognizer_requires_semantic_cross_links(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        held_protocol,
        "validate_semantic_gate_audit",
        lambda _audit: {
            "true_label": "spider toy",
            "selected_cameras": ["camera-00"],
        },
    )
    proposal = {
        "camera": "camera-00",
        "candidate_index": 2,
        "mask_sha256": "a" * 64,
    }
    assignment = {
        "strategy": held_protocol.FRAME_ZERO_REFERENCE_OPTIONAL_ASSIGNMENT_STRATEGY,
        "policy_id": held_protocol.FRAME_ZERO_REFERENCE_OPTIONAL_FALLBACK_POLICY_ID,
        "selected_proposals": [proposal],
    }
    semantic_gate = {
        "selected_exact8": [
            {
                "camera": "camera-00",
                "candidate_index": 2,
                "selected_mask_sha256": "a" * 64,
            }
        ]
    }
    safeguard = {
        "contract_sha256": held_protocol.FRAME_ZERO_SEMANTIC_GATE_CONTRACT_SHA256,
        "assignment": assignment,
        "official_urdf": {},
        "semantic_selected_proposals": [proposal],
        "semantic_gate": semantic_gate,
        "robot_subtraction": {},
    }
    safeguard["artifact_sha256"] = held_artifact_sha256(safeguard)
    manifest = {
        "object_id": "170-spider",
        "geometry_fallback": {
            "policy_id": held_protocol.FRAME_ZERO_REFERENCE_OPTIONAL_FALLBACK_POLICY_ID,
            "ordered_strategies": list(
                held_protocol.FRAME_ZERO_SEMANTIC_GATE_CONTRACT["application_order"]
            ),
            "selected_strategy": (
                held_protocol.FRAME_ZERO_REFERENCE_OPTIONAL_GEOMETRY_STRATEGY
            ),
            "attempts": [
                {"strategy": "legacy", "status": "failed"},
                {
                    "strategy": (
                        held_protocol.FRAME_ZERO_REFERENCE_OPTIONAL_GEOMETRY_STRATEGY
                    ),
                    "status": "passed",
                },
            ],
            "common_assignment": assignment,
            "reference_optional_safeguard": safeguard,
        },
    }
    assert held_protocol._reference_optional_camera_strategy(manifest) is True

    tampered = deepcopy(manifest)
    tampered["geometry_fallback"]["reference_optional_safeguard"]["assignment"][
        "strategy"
    ] = "other"
    with pytest.raises(ValueError, match="strategy audit changed"):
        held_protocol._reference_optional_camera_strategy(tampered)

    wrong_object = deepcopy(manifest)
    wrong_object["object_id"] = "083-blanket-cloth"
    with pytest.raises(ValueError, match="held object/assignment"):
        held_protocol._reference_optional_camera_strategy(wrong_object)

    wrong_semantic_mask = deepcopy(manifest)
    wrong_semantic_mask["geometry_fallback"]["reference_optional_safeguard"][
        "semantic_gate"
    ]["selected_exact8"][0]["selected_mask_sha256"] = "b" * 64
    wrong_safeguard = wrong_semantic_mask["geometry_fallback"][
        "reference_optional_safeguard"
    ]
    wrong_safeguard["artifact_sha256"] = held_artifact_sha256(wrong_safeguard)
    with pytest.raises(ValueError, match="held object/assignment"):
        held_protocol._reference_optional_camera_strategy(wrong_semantic_mask)


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
