from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bayesian_phystwin.deform360_held_outcome_scoring import (
    OUTCOME_ARTIFACT_KIND,
    TARGET_ARTIFACT_KIND,
    OfficialTarget,
    SealedCasePredictions,
    TargetOperation,
    official_target_array_sha256,
    score_and_create_calibration_gate,
    score_calibration_cohort,
    score_sealed_case,
    scored_frames,
    sparse_min_cost_frame_zero_assignment,
    transport_official_target,
    validate_permitted_target_provenance,
)
from bayesian_phystwin.deform360_held_protocol import (
    CALIBRATION_CASE_NAMES,
    DATASET_REVISION,
    FRAME_ZERO_KIND,
    ONLINE_ARTIFACT_ROLES,
    PHYSICAL_ARTIFACT_ROLES,
    authorize_outcome_phase,
    create_held_protocol_lock,
    create_online_prediction_seal,
    create_physical_prior_seal,
    create_prefix_stage_authorization,
    held_artifact_sha256,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _synthetic_frame_zero() -> np.ndarray:
    return np.array(
        [
            [0.00, 0.00, 0.00],
            [0.10, 0.00, 0.00],
            [0.20, 0.00, 0.00],
            [0.30, 0.00, 0.00],
        ],
        dtype=np.float32,
    )


def _synthetic_target(frame_zero: np.ndarray) -> OfficialTarget:
    points = np.repeat(frame_zero[None], 76, axis=0)
    points[1:, :, 1] += np.float32(0.01)
    mask = np.ones(points.shape[:2], dtype=bool)
    return OfficialTarget(points, mask, mask.copy(), {"synthetic": True})


def _permitted_target(
    target: OfficialTarget,
    permit: object,
    case_name: str,
    artifact_root: Path,
) -> OfficialTarget:
    object_id, episode = case_name.rsplit("-ep", maxsplit=1)
    case_root = artifact_root / case_name
    case_root.mkdir(parents=True, exist_ok=True)
    target_file = case_root / "official-target.pkl"
    outcome_file = case_root / "held-outcome.json"
    target_file.write_bytes(
        official_target_array_sha256(target)["object_points"].encode()
    )
    outcome_file.write_text(f"{case_name}\n", encoding="utf-8")
    provenance = {
        "target_artifact_kind": TARGET_ARTIFACT_KIND,
        "outcome_artifact_kind": OUTCOME_ARTIFACT_KIND,
        "case_name": case_name,
        "object_id": object_id,
        "episode_id": int(episode),
        "dataset_revision": DATASET_REVISION,
        "cohort_barrier_sha256": permit.cohort_barrier_sha256,
        "target_file": _record(target_file),
        "outcome_file": _record(outcome_file),
        "array_sha256": official_target_array_sha256(target),
        "information_boundary": {
            "complete_cohort_barrier_validated_before_future_open": True,
            "official_target_constructed_or_read_after_barrier": True,
            "prediction_metric_computed_during_target_construction": False,
        },
    }
    return OfficialTarget(
        target.object_points,
        target.object_visibilities,
        target.object_motions_valid,
        provenance,
    )


def _synthetic_predictions(frame_zero: np.ndarray) -> SealedCasePredictions:
    target = _synthetic_target(frame_zero).object_points
    comparator = np.repeat(frame_zero[None], 76, axis=0)
    return SealedCasePredictions(
        case_name=CALIBRATION_CASE_NAMES[0],
        center_ids=np.array([0], dtype=np.int64),
        primary_prediction_m=target.copy(),
        selected_raw_backbone_m=comparator,
        frame_zero_points_m=frame_zero,
        seal_path=Path("synthetic-seal.json"),
        archive_path=Path("synthetic-online.npz"),
        bindings={},
    )


def test_scored_frames_are_exact_frozen_post_update_intervals() -> None:
    frames = scored_frames()
    assert frames == (
        *range(20, 38),
        *range(39, 57),
        *range(58, 76),
    )
    assert 19 not in frames
    assert 38 not in frames
    assert 57 not in frames


def test_sparse_min_cost_assignment_solves_greedy_counterexample() -> None:
    # Row zero is within radius of both identities and greedily prefers identity
    # zero.  Row one can use only identity zero.  A row-ordered greedy matcher
    # fails; the full sparse assignment finds [1, 0].
    sealed = np.array([[0.012, 0.0, 0.0], [0.000, 0.0, 0.0]])
    official = np.array([[0.000, 0.0, 0.0], [0.025, 0.0, 0.0]])

    assigned, distances, diagnostics = sparse_min_cost_frame_zero_assignment(
        sealed, official
    )

    np.testing.assert_array_equal(assigned, np.array([1, 0]))
    np.testing.assert_allclose(distances, np.array([0.013, 0.0]), atol=1e-12)
    assert diagnostics["sealed_point_coverage_fraction"] == 1.0
    assert diagnostics["assigned_official_identity_collision_count"] == 0


def test_sparse_assignment_fails_closed_without_unique_radius_match() -> None:
    sealed = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]])
    official = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="collision-free full"):
        sparse_min_cost_frame_zero_assignment(sealed, official)


def test_transport_is_one_to_one_and_frame_zero_is_bit_exact() -> None:
    sealed = np.array([[0.012, 0.0, 0.0], [0.000, 0.0, 0.0]], dtype=np.float32)
    official_zero = np.array([[0.000, 0.0, 0.0], [0.025, 0.0, 0.0]], dtype=np.float32)
    points = np.repeat(official_zero[None], 76, axis=0)
    points[1:, :, 2] += np.array([0.1, 0.2], dtype=np.float32)
    mask = np.ones((76, 2), dtype=bool)

    transported = transport_official_target(
        sealed,
        OfficialTarget(points, mask, mask.copy(), {"synthetic": True}),
    )

    np.testing.assert_array_equal(transported.object_points[0], sealed)
    np.testing.assert_array_equal(transported.official_identity_ids, [1, 0])
    assert len(np.unique(transported.official_identity_ids)) == len(sealed)
    assert transported.diagnostics[
        "transported_frame_zero_replaced_with_sealed_identity"
    ]


def test_permitted_target_rejects_mutated_bound_outcome_file(tmp_path: Path) -> None:
    case_name = CALIBRATION_CASE_NAMES[0]
    permit = SimpleNamespace(cohort_barrier_sha256="f" * 64)
    target = _permitted_target(
        _synthetic_target(_synthetic_frame_zero()),
        permit,
        case_name,
        tmp_path,
    )
    Path(target.provenance["target_file"]["path"]).write_bytes(b"mutated")

    with pytest.raises(ValueError, match="official target file binding changed"):
        validate_permitted_target_provenance(target, permit, case_name)


def test_case_score_excludes_centers_and_unscored_boundaries() -> None:
    frame_zero = _synthetic_frame_zero()
    predictions = _synthetic_predictions(frame_zero)
    target = _synthetic_target(frame_zero)
    primary = predictions.primary_prediction_m.copy()
    primary[:, 0] += np.float32(100.0)  # observed centre: never scored
    primary[0, 1:] += np.float32(100.0)
    primary[19, 1:] += np.float32(100.0)
    primary[38, 1:] += np.float32(100.0)
    primary[57, 1:] += np.float32(100.0)
    predictions = SealedCasePredictions(
        **{
            **predictions.__dict__,
            "primary_prediction_m": primary,
        }
    )

    record = score_sealed_case(predictions, target)

    assert record["gate_score"]["primary_chamfer_m"] == 0.0
    assert record["gate_score"]["primary_identity_rmse_m"] == 0.0
    assert record["gate_score"]["comparator_chamfer_m"] > 0.0
    assert record["gate_score"]["comparator_identity_rmse_m"] > 0.0
    assert record["permanently_excluded_center_ids"] == [0]


def _frame_zero_manifest(
    root: Path,
    lock_path: Path,
    case_name: str,
    frame_zero: np.ndarray,
) -> Path:
    directory = root / case_name
    directory.mkdir(parents=True, exist_ok=True)
    bundle = directory / "frame-zero.npz"
    np.savez_compressed(bundle, object_points_world_m=frame_zero)
    robot = directory / "robot.npz"
    robot.write_bytes(b"known-action")
    metadata = directory / "robot-metadata.json"
    metadata.write_text("{}\n", encoding="utf-8")
    object_id, episode = case_name.rsplit("-ep", maxsplit=1)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": FRAME_ZERO_KIND,
        "case_name": case_name,
        "object_id": object_id,
        "episode_id": int(episode),
        "role": "calibration",
        "lock_sha256": _sha256(lock_path),
        "frame_indices": [0],
        "bundle": _record(bundle),
        "action_inputs": {
            "robot_trajectory": _record(robot),
            "robot_metadata": _record(metadata),
        },
        "information_boundary": {
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
        },
    }
    manifest["artifact_sha256"] = held_artifact_sha256(manifest)
    path = directory / "frame-zero-manifest.json"
    _write_json(path, manifest)
    return path


def _seal_calibration_case(
    root: Path,
    lock_path: Path,
    case_name: str,
    frame_zero: np.ndarray,
    primary: np.ndarray,
    comparator: np.ndarray,
) -> Path:
    manifest = _frame_zero_manifest(
        root / "frame-zero", lock_path, case_name, frame_zero
    )
    physical_root = root / "physical" / case_name
    physical_root.mkdir(parents=True, exist_ok=True)
    physical_files: dict[str, Path] = {}
    for role in PHYSICAL_ARTIFACT_ROLES:
        path = physical_root / f"{role}.npz"
        if role == "physical_prediction_archive":
            np.savez_compressed(path, frame_zero_points_m=frame_zero)
        else:
            path.write_bytes(role.encode())
        physical_files[role] = path
    physical_seal = root / "physical-seals" / f"{case_name}.json"
    create_physical_prior_seal(
        physical_seal,
        lock_path,
        manifest,
        physical_files,
        case_name=case_name,
        role="calibration",
    )
    prefix = root / "prefix" / f"{case_name}.json"
    create_prefix_stage_authorization(prefix, lock_path, physical_seal)

    online_root = root / "online" / case_name
    online_root.mkdir(parents=True, exist_ok=True)
    online_files: dict[str, Path] = {}
    for role in ONLINE_ARTIFACT_ROLES:
        path = online_root / f"{role}.npz"
        if role == "online_prediction_archive":
            np.savez_compressed(
                path,
                center_ids=np.array([0], dtype=np.int64),
                primary_prediction_m=primary,
                selected_raw_backbone_m=comparator,
                frame_zero_points_m=frame_zero,
                ignored_control_m=comparator,
            )
        else:
            path.write_bytes(role.encode())
        online_files[role] = path
    online_seal = root / "online-seals" / f"{case_name}.json"
    create_online_prediction_seal(online_seal, lock_path, prefix, online_files)
    return online_seal


def _complete_calibration_fixture(
    tmp_path: Path,
) -> tuple[Path, dict[str, Path], OfficialTarget]:
    lock_path = tmp_path / "lock.json"
    create_held_protocol_lock(
        lock_path,
        immutable_bindings={
            "held_online_runner": "a" * 64,
            "held_outcome_scorer": "b" * 64,
            "scipy_runtime": "c" * 64,
        },
    )
    frame_zero = _synthetic_frame_zero()
    target = _synthetic_target(frame_zero)
    primary = target.object_points.copy()
    comparator = np.repeat(frame_zero[None], 76, axis=0)
    seals = {
        case_name: _seal_calibration_case(
            tmp_path / "chain",
            lock_path,
            case_name,
            frame_zero,
            primary,
            comparator,
        )
        for case_name in CALIBRATION_CASE_NAMES
    }
    return lock_path, seals, target


def test_outcome_scorer_requires_complete_live_permit(tmp_path: Path) -> None:
    lock_path, seals, target = _complete_calibration_fixture(tmp_path)
    incomplete = dict(seals)
    incomplete.pop(CALIBRATION_CASE_NAMES[-1])
    with pytest.raises(ValueError, match="every exact cohort seal"):
        authorize_outcome_phase(lock_path, incomplete, role="calibration")

    permit = authorize_outcome_phase(lock_path, seals, role="calibration")
    operations = {
        case: TargetOperation(
            "create",
            lambda case=case: _permitted_target(
                target, permit, case, tmp_path / "synthetic-outcomes"
            ),
        )
        for case in CALIBRATION_CASE_NAMES
    }
    with pytest.raises(ValueError, match="lacks a cohort capability"):
        # Copying the public dataclass without its in-process capability cannot
        # cross the callback boundary.
        broken = type(permit)(
            permit.lock_path,
            permit.role,
            permit.seal_paths,
            permit.cohort_barrier_sha256,
            object(),
        )
        score_calibration_cohort(broken, operations)


def test_all_15_permit_scoring_feeds_frozen_gate(tmp_path: Path) -> None:
    lock_path, seals, target = _complete_calibration_fixture(tmp_path)
    permit = authorize_outcome_phase(lock_path, seals, role="calibration")
    calls: list[str] = []
    operations = {
        case: TargetOperation(
            "create",
            lambda case=case, target=target: (
                calls.append(case)
                or _permitted_target(
                    target, permit, case, tmp_path / "synthetic-outcomes"
                )
            ),
        )
        for case in CALIBRATION_CASE_NAMES
    }

    decision, evidence, records = score_and_create_calibration_gate(
        tmp_path / "decision.json", permit, operations
    )

    assert calls == list(CALIBRATION_CASE_NAMES)
    assert set(records) == set(CALIBRATION_CASE_NAMES)
    assert decision["decision"] == "GO"
    assert evidence["artifact_kind"] == "Deform360HeldCalibrationScoreEvidence"
    assert (tmp_path / "calibration-score-evidence.json").is_file()
    assert decision["summary"]["case_chamfer_wins"] == 15
    for case in CALIBRATION_CASE_NAMES:
        assert set(records[case]["gate_score"]) == {
            "primary_chamfer_m",
            "comparator_chamfer_m",
            "primary_identity_rmse_m",
            "comparator_identity_rmse_m",
        }
        assert records[case]["method_selection_or_tuning_performed"] is False
