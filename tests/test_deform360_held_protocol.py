from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from bayesian_phystwin.deform360_held_protocol import (
    CALIBRATION_CASE_NAMES,
    CALIBRATION_GATE,
    CONFIRMATION_CASES,
    CONFIRMATION_GATE,
    FRAME_ZERO_KIND,
    ONLINE_ARTIFACT_ROLES,
    PHYSICAL_ARTIFACT_ROLES,
    PRIMARY_METHOD,
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
        immutable_bindings={
            "analysis_source": "a" * 64,
            "analysis_configuration": "b" * 64,
            "alltracker_checkpoint": "c" * 64,
        },
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
    bundle.write_bytes(b"single extracted frame")
    robot = directory / "robot.npz"
    robot.write_bytes(b"known robot trajectory")
    metadata = directory / "robot.meta.json"
    metadata.write_text('{"frame_count": 76}\n', encoding="utf-8")
    object_id, episode_id = _identity(case_name)
    boundary: dict[str, object] = {
        "maximum_object_rgb_frame_read": 0,
        "object_observation_frames_used": [0],
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
        "case_name": case_name,
        "object_id": object_id,
        "episode_id": episode_id,
        "role": role,
        "lock_sha256": _sha256(lock_path),
        "frame_indices": [0],
        "bundle": _record(bundle),
        "action_inputs": {
            "robot_trajectory": _record(robot),
            "robot_metadata": _record(metadata),
        },
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
    create_calibration_gate_decision(
        decision_path,
        permit,
        _passing_scores(),
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
    decision = create_calibration_gate_decision(
        decision_path,
        permit,
        _passing_scores(primary_chamfer_m=1.10),
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
