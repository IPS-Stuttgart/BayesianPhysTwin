from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

import bayesian_phystwin.deform360_frozen_query_field as frozen_field
import bayesian_phystwin.deform360_held_v8_query_artifacts as artifacts


CASE_NAME = "002-rope-silk-ep0003"


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _bound_file(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _write_lock(path: Path) -> None:
    value: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "TestHeldV8Lock",
        "protocol_id": artifacts.PROTOCOL_ID,
    }
    value["artifact_sha256"] = artifacts._artifact_sha256(value)
    _write_json(path, value)
    path.chmod(0o400)


def _write_decision(path: Path) -> None:
    value = {
        "schema_version": 1,
        "artifact_kind": "Deform360Open27QueryFieldDevelopmentDecision",
        "protocol_id": artifacts.DEVELOPMENT_PROTOCOL_ID,
        "selection": {
            "status": "locked using only non-held open-development evidence",
            "selected_candidate_id": artifacts.DEVELOPMENT_CANDIDATE_ID,
            "selected_config": {
                "candidate_id": artifacts.DEVELOPMENT_CANDIDATE_ID,
                "operator_id": artifacts.FIELD_OPERATOR_ID,
                "neighbor_count": artifacts.GAUSSIAN_NEIGHBOR_COUNT,
                "length_scale_fraction": artifacts.GAUSSIAN_LENGTH_SCALE_FRACTION,
                "support_radius_fraction": artifacts.SUPPORT_RADIUS_FRACTION,
            },
            "future_target_scores_used_for_selection": False,
            "future_target_masks_used_for_selection": False,
        },
    }
    _write_json(path, value)


def _source_arrays(
    *, center_count: int = artifacts.CENTER_COUNT
) -> dict[str, np.ndarray]:
    point_count = 32
    x = np.linspace(0.0, 0.155, point_count, dtype=np.float32)
    frame_zero = np.column_stack(
        (x, 0.01 * np.sin(20.0 * x), 0.20 + 0.005 * np.cos(10.0 * x))
    ).astype(np.float32)
    primary = np.repeat(frame_zero[None], artifacts.FRAME_COUNT, axis=0)
    comparator = primary.copy()
    time = np.arange(artifacts.FRAME_COUNT, dtype=np.float32)[:, None]
    node = np.linspace(0.5, 1.5, point_count, dtype=np.float32)[None]
    primary[:, :, 1] += (time * node * np.float32(0.0002)).astype(np.float32)
    comparator[:, :, 2] -= (time * node * np.float32(0.0001)).astype(np.float32)
    primary[0] = frame_zero
    comparator[0] = frame_zero
    return {
        "primary_prediction_m": primary,
        "selected_raw_backbone_m": comparator,
        "frame_zero_points_m": frame_zero,
        "center_ids": np.arange(center_count, dtype=np.int64),
        "unrelated_diagnostic": np.asarray([7], dtype=np.int64),
    }


def _write_online(
    root: Path,
    lock: Path,
    *,
    center_count: int = artifacts.CENTER_COUNT,
) -> tuple[Path, Path]:
    archive = root / "online_prediction.npz"
    np.savez_compressed(archive, **_source_arrays(center_count=center_count))
    archive.chmod(0o400)
    seal: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "TestV8OnlinePredictionSeal",
        "protocol_id": artifacts.PROTOCOL_ID,
        "case_name": CASE_NAME,
        "lock": _bound_file(lock),
        "online_artifacts": {"online_prediction_archive": _bound_file(archive)},
        "information_boundary": {
            "outcome_created": False,
            "outcome_read": False,
            "all_frozen_predictions_hashed_before_outcome": True,
        },
    }
    seal["artifact_sha256"] = artifacts._artifact_sha256(seal)
    seal_path = root / "online_seal.json"
    _write_json(seal_path, seal)
    seal_path.chmod(0o400)
    return archive, seal_path


def _write_field_fixture(
    root: Path,
    *,
    center_count: int = artifacts.CENTER_COUNT,
) -> tuple[Path, Path, Path, Path, Path]:
    lock = root / "lock.json"
    decision = root / "decision.json"
    _write_lock(lock)
    _write_decision(decision)
    online, online_seal = _write_online(root, lock, center_count=center_count)
    manifest = root / "field.json"
    field_source = Path(frozen_field.__file__).resolve()
    module_source = Path(artifacts.__file__).resolve()
    artifacts.write_preoutcome_frozen_field_manifest(
        manifest,
        lock_path=lock,
        lock_sha256=_bound_file(lock)["sha256"],
        online_prediction_archive_path=online,
        online_prediction_seal_path=online_seal,
        field_source_path=field_source,
        field_source_sha256=_bound_file(field_source)["sha256"],
        artifact_module_source_path=module_source,
        artifact_module_source_sha256=_bound_file(module_source)["sha256"],
        development_decision_path=decision,
        development_decision_sha256=_bound_file(decision)["sha256"],
        case_name=CASE_NAME,
    )
    return lock, decision, online, online_seal, manifest


def _query_arrays(online: Path) -> dict[str, np.ndarray]:
    with np.load(online, allow_pickle=False) as stored:
        centers = stored["frame_zero_points_m"][: artifacts.CENTER_COUNT].copy()
    identities = np.arange(1000, 1000 + artifacts.CENTER_COUNT + 2, dtype=np.int64)
    positions = np.concatenate(
        (
            centers,
            np.asarray([[0.08, 0.005, 0.20], [1.0, 1.0, 1.0]], dtype=np.float32),
        ),
        axis=0,
    )
    return {"identity_ids": identities, "positions_m": positions}


def _write_query_and_prediction(
    root: Path,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    lock, _, online, _, field_manifest = _write_field_fixture(root)
    query_archive = root / "official_x0.npz"
    query_manifest = root / "official_x0.json"
    artifacts.write_official_query_artifact(
        query_archive,
        query_manifest,
        lock,
        lock_sha256=_bound_file(lock)["sha256"],
        case_name=CASE_NAME,
        query_arrays=_query_arrays(online),
    )
    prediction_archive = root / "queried_prediction.npz"
    prediction_seal = root / "queried_prediction.json"
    artifacts.write_queried_prediction_artifact(
        prediction_archive,
        prediction_seal,
        lock_path=lock,
        lock_sha256=_bound_file(lock)["sha256"],
        frozen_field_manifest_path=field_manifest,
        official_query_manifest_path=query_manifest,
    )
    return (
        lock,
        field_manifest,
        query_archive,
        query_manifest,
        prediction_archive,
        prediction_seal,
    )


def test_round_trip_freezes_field_then_queries_only_x0(tmp_path: Path) -> None:
    (
        lock,
        field_manifest_path,
        query_archive,
        query_manifest,
        prediction_archive,
        prediction_seal,
    ) = _write_query_and_prediction(tmp_path)

    field_manifest = artifacts.validate_preoutcome_frozen_field_manifest(
        field_manifest_path, lock_path=lock, expected_case_name=CASE_NAME
    )
    artifacts.validate_official_query_artifact(
        query_manifest, lock, expected_case_name=CASE_NAME
    )
    artifacts.validate_queried_prediction_artifact(
        prediction_seal, lock_path=lock, expected_case_name=CASE_NAME
    )

    contract = field_manifest["field_contract"]
    assert contract["operator_id"] == "gaussian-knn-normalized-v1"
    assert contract["gaussian_neighbor_count"] == 4
    assert contract["gaussian_length_scale_fraction"] == 0.05
    assert contract["support_radius_fraction"] == 0.5
    assert contract["center_exclusion"] == {
        "method": "geometry-only-radius-union-v2",
        "maximum_distance_m": 0.015,
        "exclude_every_query_within_radius": True,
        "centers_without_a_query_in_radius_are_allowed": True,
        "per_center_nearest_query_is_audit_only": True,
        "future_coordinates_or_masks_used": False,
        "cohort_coverage_gate_imposed_here": False,
        "contract_sha256": artifacts.CENTER_EXCLUSION_CONTRACT_SHA256,
    }
    for sealed_path in (
        field_manifest_path,
        query_archive,
        query_manifest,
        prediction_archive,
        prediction_seal,
    ):
        assert sealed_path.stat().st_mode & 0o777 == 0o400
    with np.load(query_archive, allow_pickle=False) as stored:
        assert set(stored.files) == {"identity_ids", "positions_m"}
        assert stored["positions_m"].ndim == 2
    with np.load(prediction_archive, allow_pickle=False) as stored:
        assert set(stored.files) == artifacts.QUERIED_PREDICTION_ARRAY_NAMES
        count = len(stored["identity_ids"])
        assert stored["primary_prediction_m"].shape == (76, count, 3)
        assert stored["selected_raw_backbone_m"].shape == (76, count, 3)
        np.testing.assert_array_equal(stored["frame_indices"], np.arange(76))
        np.testing.assert_array_equal(
            stored["primary_prediction_m"][0], stored["positions_m"]
        )
        np.testing.assert_array_equal(
            stored["selected_raw_backbone_m"][0], stored["positions_m"]
        )
        assert stored["shared_support_mask"].dtype == np.dtype(bool)
        assert not bool(stored["shared_support_mask"][-1])
        assert np.all(np.isfinite(stored["primary_prediction_m"][:, -1]))
        # The radius union is deliberately not capped at one query per center:
        # the extra synthetic query lies inside an already covered neighborhood.
        assert int(np.sum(stored["center_exclusion_mask"])) == 17
        assert np.max(stored["center_nearest_query_distance_m"]) == 0.0
        np.testing.assert_array_equal(stored["center_within_radius_mask"], True)


def test_query_artifact_allows_no_official_identity_near_a_center(
    tmp_path: Path,
) -> None:
    lock, _, _, _, field_manifest = _write_field_fixture(tmp_path)
    count = artifacts.CENTER_COUNT + 2
    query_archive = tmp_path / "official_x0.npz"
    query_manifest = tmp_path / "official_x0.json"
    artifacts.write_official_query_artifact(
        query_archive,
        query_manifest,
        lock,
        lock_sha256=_bound_file(lock)["sha256"],
        case_name=CASE_NAME,
        query_arrays={
            "identity_ids": np.arange(2000, 2000 + count, dtype=np.int64),
            "positions_m": np.column_stack(
                (
                    np.linspace(1.0, 2.0, count, dtype=np.float32),
                    np.zeros(count, dtype=np.float32),
                    np.zeros(count, dtype=np.float32),
                )
            ),
        },
    )
    prediction_archive = tmp_path / "queried_prediction.npz"
    prediction_seal = tmp_path / "queried_prediction.json"

    artifacts.write_queried_prediction_artifact(
        prediction_archive,
        prediction_seal,
        lock_path=lock,
        lock_sha256=_bound_file(lock)["sha256"],
        frozen_field_manifest_path=field_manifest,
        official_query_manifest_path=query_manifest,
    )

    artifacts.validate_queried_prediction_artifact(
        prediction_seal,
        lock_path=lock,
        expected_case_name=CASE_NAME,
    )
    with np.load(prediction_archive, allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored["center_within_radius_mask"], False)
        np.testing.assert_array_equal(stored["center_exclusion_mask"], False)


@pytest.mark.parametrize(
    "arrays",
    [
        {
            "identity_ids": np.arange(16, dtype=np.int64),
            "positions_m": np.zeros((16, 3), dtype=np.float32),
            "visibility": np.ones((16,), dtype=bool),
        },
        {
            "identity_ids": np.arange(16, dtype=np.int64),
            "positions_m": np.zeros((76, 16, 3), dtype=np.float32),
        },
        {
            "identity_ids": np.arange(15, dtype=np.int64),
            "positions_m": np.zeros((15, 3), dtype=np.float32),
        },
        {
            "identity_ids": np.asarray(
                [1, 0, *range(2, 16)],
                dtype=np.int64,
            ),
            "positions_m": np.zeros((16, 3), dtype=np.float32),
        },
    ],
)
def test_official_query_rejects_extra_future_or_too_small_inputs(
    tmp_path: Path, arrays: dict[str, np.ndarray]
) -> None:
    lock = tmp_path / "lock.json"
    _write_lock(lock)
    with pytest.raises(ValueError):
        artifacts.write_official_query_artifact(
            tmp_path / "query.npz",
            tmp_path / "query.json",
            lock,
            lock_sha256=_bound_file(lock)["sha256"],
            case_name=CASE_NAME,
            query_arrays=arrays,
        )
    assert not (tmp_path / "query.npz").exists()


def test_preoutcome_field_requires_exactly_sixteen_centers(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly 16"):
        _write_field_fixture(tmp_path, center_count=15)


def test_code_and_development_decision_are_rehashed(tmp_path: Path) -> None:
    lock, decision, _, _, manifest = _write_field_fixture(tmp_path)
    decision.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="development decision file binding"):
        artifacts.validate_preoutcome_frozen_field_manifest(manifest, lock_path=lock)


def test_queried_archive_tampering_and_extra_arm_mask_fail_closed(
    tmp_path: Path,
) -> None:
    lock, _, _, _, archive, seal = _write_query_and_prediction(tmp_path)
    os.chmod(archive, 0o600)
    with np.load(archive, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    arrays["primary_support_mask"] = arrays["shared_support_mask"].copy()
    np.savez_compressed(archive, **arrays)
    seal_value = json.loads(seal.read_text(encoding="utf-8"))
    seal_value["archive"] = _bound_file(archive)
    seal_value["array_records"] = artifacts._array_records(arrays)
    seal_value["artifact_sha256"] = artifacts._artifact_sha256(seal_value)
    os.chmod(seal, 0o600)
    _write_json(seal, seal_value)
    archive.chmod(0o400)
    seal.chmod(0o400)
    with pytest.raises(ValueError, match="arm-specific masks"):
        artifacts.validate_queried_prediction_artifact(seal, lock_path=lock)


def test_exclusive_writes_and_symlinked_inputs_are_rejected(tmp_path: Path) -> None:
    lock = tmp_path / "lock.json"
    _write_lock(lock)
    online = tmp_path / "irrelevant.npz"
    arrays = {
        "identity_ids": np.arange(16, dtype=np.int64),
        "positions_m": np.column_stack(
            (
                np.arange(16, dtype=np.float32) * np.float32(0.001),
                np.zeros(16, dtype=np.float32),
                np.zeros(16, dtype=np.float32),
            )
        ),
    }
    artifacts.write_official_query_artifact(
        online,
        tmp_path / "query.json",
        lock,
        lock_sha256=_bound_file(lock)["sha256"],
        case_name=CASE_NAME,
        query_arrays=arrays,
    )
    with pytest.raises(FileExistsError):
        artifacts.write_official_query_artifact(
            online,
            tmp_path / "second.json",
            lock,
            lock_sha256=_bound_file(lock)["sha256"],
            case_name=CASE_NAME,
            query_arrays=arrays,
        )
    linked_lock = tmp_path / "linked-lock.json"
    linked_lock.symlink_to(lock)
    with pytest.raises(ValueError, match="symlink"):
        artifacts.validate_official_query_artifact(tmp_path / "query.json", linked_lock)
    online.chmod(0o444)
    with pytest.raises(ValueError, match="mode 0400"):
        artifacts.validate_official_query_artifact(tmp_path / "query.json", lock)


def test_module_has_no_scorer_or_outcome_import() -> None:
    source = Path(artifacts.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any("outcome" in name or "scor" in name for name in imported)
