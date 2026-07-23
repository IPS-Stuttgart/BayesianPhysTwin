from __future__ import annotations

import inspect
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import bayesian_phystwin.deform360_adaptive_covariance_confirmation_lock as lock
import bayesian_phystwin.deform360_adaptive_covariance_confirmation_seal as seal
from bayesian_phystwin.deform360_adaptive_covariance_rbf import (
    ADAPTIVE_COVARIANCE_PROTOCOL_ID,
)


H1 = "a" * 40
H2 = "b" * 40


def _write_lock(tmp_path: Path) -> Path:
    path = tmp_path / "lock" / "adaptive-confirmation-h2.json"
    lock.write_confirmation_cohort_lock(path, H1)
    return path


def _arrays() -> dict[str, np.ndarray]:
    base = np.zeros((76, 20, 3), dtype=np.float32)
    result = {}
    for role_index, role in enumerate(seal.ARRAY_ROLES):
        value = base.copy()
        value[1:] += np.float32(role_index + 1)
        result[role] = value
    for role in ("adaptive_prediction_m", "selected_raw_prediction_m"):
        result[role][58:76] = result["physical_prior_m"][58:76]
    return result


def _cameras() -> dict[int, list[str]]:
    eight = [f"camera-{index:02d}" for index in range(8)]
    return {4: eight[:4], 8: eight}


def _budget_record(*, reliable: bool, dispersion: float) -> dict[str, Any]:
    return {
        "valid_covariance_center_count": 8,
        "valid_covariance_center_ids": list(range(8)),
        "normalized_covariance_dispersion": dispersion,
        "reliable": reliable,
    }


def _routing() -> dict[str, Any]:
    cameras = _cameras()
    return {
        "protocol_id": ADAPTIVE_COVARIANCE_PROTOCOL_ID,
        "fallback": {
            "trajectory": "physical_prior",
            "rbf_state_update": False,
            "bit_exact": True,
        },
        "updates": [
            {
                "frame": 19,
                "stop_frame_exclusive": 38,
                "route": "4_view_rbf",
                "selected_camera_budget": 4,
                "tracked_camera_count": 4,
                "tracked_cameras": cameras[4],
                "selected_backbone": "physical_prior",
                "rbf_correction_applied": True,
                "state_updated": True,
                "budget_diagnostics": {
                    "4": _budget_record(reliable=True, dispersion=0.010),
                },
            },
            {
                "frame": 38,
                "stop_frame_exclusive": 57,
                "route": "8_view_rbf",
                "selected_camera_budget": 8,
                "tracked_camera_count": 8,
                "tracked_cameras": cameras[8],
                "selected_backbone": "persistence",
                "rbf_correction_applied": True,
                "state_updated": True,
                "budget_diagnostics": {
                    "4": _budget_record(reliable=False, dispersion=0.020),
                    "8": _budget_record(reliable=True, dispersion=0.012),
                },
            },
            {
                "frame": 57,
                "stop_frame_exclusive": 76,
                "route": "physical_prior_fallback",
                "selected_camera_budget": None,
                "tracked_camera_count": 8,
                "tracked_cameras": cameras[8],
                "selected_backbone": "physical_prior",
                "rbf_correction_applied": False,
                "state_updated": False,
                "budget_diagnostics": {
                    "4": _budget_record(reliable=False, dispersion=0.020),
                    "8": _budget_record(reliable=False, dispersion=0.025),
                },
            },
        ],
    }


def _disposition() -> dict[str, Any]:
    return {
        "status": "prediction_complete",
        "case_retained": True,
        "disposition_based_on_target_or_outcome": False,
        "center_ids": list(range(16)),
        "notes": "fixture target-free execution",
    }


def _selected_cases(lock_path: Path) -> list[str]:
    payload = lock.load_confirmation_cohort_lock(
        lock_path,
        expected_implementation_commit_h1=H1,
    )
    return list(payload["selected_case_ids"])


def _seal_case(
    lock_path: Path,
    cases_root: Path,
    case_id: str,
) -> dict[str, Any]:
    return seal.seal_confirmation_case(
        lock_path,
        H2,
        case_id,
        cases_root / case_id,
        _arrays(),
        _cameras(),
        _routing(),
        _disposition(),
        expected_h1=H1,
    )


def _seal_complete_cohort(
    lock_path: Path,
    cases_root: Path,
) -> dict[str, Path]:
    result = {}
    for case_id in _selected_cases(lock_path):
        _seal_case(lock_path, cases_root, case_id)
        result[case_id] = cases_root / case_id
    return result


def test_case_sealer_has_no_evaluation_parameter_and_binds_all_content(
    tmp_path: Path,
) -> None:
    parameters = inspect.signature(seal.seal_confirmation_case).parameters
    assert all(
        "target" not in name and "outcome" not in name and "metric" not in name
        for name in parameters
    )
    lock_path = _write_lock(tmp_path)
    case_id = _selected_cases(lock_path)[0]
    arrays = _arrays()
    arrays_before = {role: value.copy() for role, value in arrays.items()}
    output = tmp_path / "cases" / case_id

    record = seal.seal_confirmation_case(
        lock_path,
        H2,
        case_id,
        output,
        arrays,
        _cameras(),
        _routing(),
        _disposition(),
        expected_h1=H1,
    )

    assert set(path.name for path in output.iterdir()) == {
        seal.ARRAY_ARCHIVE_FILENAME,
        seal.DIAGNOSTIC_FILENAME,
        seal.CASE_MANIFEST_FILENAME,
    }
    assert record["case_id"] == case_id
    manifest = json.loads((output / seal.CASE_MANIFEST_FILENAME).read_text())
    assert manifest["lock_binding"]["implementation_commit_h1"] == H1
    assert manifest["lock_binding"]["cohort_lock_commit_h2"] == H2
    assert set(manifest["content"]["prediction_archive"]["arrays"]) == set(
        seal.ARRAY_ROLES
    )
    diagnostic = json.loads((output / seal.DIAGNOSTIC_FILENAME).read_text())
    assert diagnostic["nested_selected_cameras"]["4"] == _cameras()[4]
    assert diagnostic["nested_selected_cameras"]["8"][:4] == _cameras()[4]
    assert diagnostic["information_boundary"] == seal.TARGET_FREE_BOUNDARY
    assert (
        seal.validate_confirmation_case_seal(
            output,
            lock_path,
            H2,
            expected_case_id=case_id,
            expected_h1=H1,
        )
        == record
    )
    for role in seal.ARRAY_ROLES:
        np.testing.assert_array_equal(arrays[role], arrays_before[role])

    arrays["adaptive_prediction_m"][1:] = np.float32(999.0)
    assert (
        seal.validate_confirmation_case_seal(
            output,
            lock_path,
            H2,
            expected_case_id=case_id,
            expected_h1=H1,
        )
        == record
    )
    with pytest.raises(ValueError, match="already exists"):
        _seal_case(lock_path, output.parent, case_id)


def test_case_sealer_rejects_wrong_case_nonnesting_and_frame_zero_change(
    tmp_path: Path,
) -> None:
    lock_path = _write_lock(tmp_path)
    case_id = _selected_cases(lock_path)[0]
    with pytest.raises(ValueError, match="outside the exact H2-locked cohort"):
        seal.seal_confirmation_case(
            lock_path,
            H2,
            "unlocked-object-ep0000",
            tmp_path / "cases" / "unlocked-object-ep0000",
            _arrays(),
            _cameras(),
            _routing(),
            _disposition(),
            expected_h1=H1,
        )

    cameras = _cameras()
    cameras[8] = cameras[8][4:] + cameras[8][:4]
    with pytest.raises(ValueError, match="exact ordered prefix"):
        seal.seal_confirmation_case(
            lock_path,
            H2,
            case_id,
            tmp_path / "nonnested" / case_id,
            _arrays(),
            cameras,
            _routing(),
            _disposition(),
            expected_h1=H1,
        )

    arrays = _arrays()
    arrays["fixed_4_rbf_prediction_m"][0, 0, 0] = np.float32(1.0)
    with pytest.raises(ValueError, match="frame zero differs"):
        seal.seal_confirmation_case(
            lock_path,
            H2,
            case_id,
            tmp_path / "identity-change" / case_id,
            arrays,
            _cameras(),
            _routing(),
            _disposition(),
            expected_h1=H1,
        )


def test_case_sealer_requires_frozen_shape_centers_and_routing_semantics(
    tmp_path: Path,
) -> None:
    lock_path = _write_lock(tmp_path)
    case_id = _selected_cases(lock_path)[0]

    wrong_frames = {role: value[:75].copy() for role, value in _arrays().items()}
    with pytest.raises(ValueError, match=r"shape \(76, N, 3\) with N > 16"):
        seal.seal_confirmation_case(
            lock_path,
            H2,
            case_id,
            tmp_path / "wrong-frames" / case_id,
            wrong_frames,
            _cameras(),
            _routing(),
            _disposition(),
            expected_h1=H1,
        )

    no_noncenter_points = {
        role: value[:, :16].copy() for role, value in _arrays().items()
    }
    with pytest.raises(ValueError, match="N > 16"):
        seal.seal_confirmation_case(
            lock_path,
            H2,
            case_id,
            tmp_path / "no-noncenter-points" / case_id,
            no_noncenter_points,
            _cameras(),
            _routing(),
            _disposition(),
            expected_h1=H1,
        )

    disposition = _disposition()
    disposition["center_ids"][-1] = disposition["center_ids"][0]
    with pytest.raises(ValueError, match="16 unique in-range"):
        seal.seal_confirmation_case(
            lock_path,
            H2,
            case_id,
            tmp_path / "duplicate-centers" / case_id,
            _arrays(),
            _cameras(),
            _routing(),
            disposition,
            expected_h1=H1,
        )

    routing = _routing()
    routing["updates"][0]["frame"] = 18
    with pytest.raises(ValueError, match="exactly 19, 38, 57"):
        seal.seal_confirmation_case(
            lock_path,
            H2,
            case_id,
            tmp_path / "wrong-update-frame" / case_id,
            _arrays(),
            _cameras(),
            routing,
            _disposition(),
            expected_h1=H1,
        )

    routing = _routing()
    routing["updates"][1]["stop_frame_exclusive"] = 58
    with pytest.raises(ValueError, match="exactly 38, 57, 76"):
        seal.seal_confirmation_case(
            lock_path,
            H2,
            case_id,
            tmp_path / "wrong-update-stop" / case_id,
            _arrays(),
            _cameras(),
            routing,
            _disposition(),
            expected_h1=H1,
        )

    routing = _routing()
    routing["updates"][0]["tracked_cameras"] = list(
        reversed(routing["updates"][0]["tracked_cameras"])
    )
    with pytest.raises(ValueError, match="differ from the nested plan"):
        seal.seal_confirmation_case(
            lock_path,
            H2,
            case_id,
            tmp_path / "invented-camera-order" / case_id,
            _arrays(),
            _cameras(),
            routing,
            _disposition(),
            expected_h1=H1,
        )

    routing = _routing()
    routing["updates"][2]["state_updated"] = True
    with pytest.raises(ValueError, match="correction/state disposition"):
        seal.seal_confirmation_case(
            lock_path,
            H2,
            case_id,
            tmp_path / "fallback-state-update" / case_id,
            _arrays(),
            _cameras(),
            routing,
            _disposition(),
            expected_h1=H1,
        )

    routing = _routing()
    routing["updates"][1]["budget_diagnostics"]["8"]["valid_covariance_center_ids"][
        -1
    ] = 19
    with pytest.raises(ValueError, match="center IDs are invalid"):
        seal.seal_confirmation_case(
            lock_path,
            H2,
            case_id,
            tmp_path / "foreign-valid-center" / case_id,
            _arrays(),
            _cameras(),
            routing,
            _disposition(),
            expected_h1=H1,
        )

    arrays = _arrays()
    arrays["adaptive_prediction_m"][60, 0, 0] = np.float32(123.0)
    with pytest.raises(ValueError, match="fallback interval is not bit-exact"):
        seal.seal_confirmation_case(
            lock_path,
            H2,
            case_id,
            tmp_path / "nonexact-fallback" / case_id,
            arrays,
            _cameras(),
            _routing(),
            _disposition(),
            expected_h1=H1,
        )


def test_case_sealer_rejects_embedded_evaluation_payloads(
    tmp_path: Path,
) -> None:
    lock_path = _write_lock(tmp_path)
    case_id = _selected_cases(lock_path)[0]
    routing = _routing()
    routing["target_metrics"] = {"rmse_m": 0.0}
    with pytest.raises(ValueError, match="forbidden evaluation field"):
        seal.seal_confirmation_case(
            lock_path,
            H2,
            case_id,
            tmp_path / "routing-leak" / case_id,
            _arrays(),
            _cameras(),
            routing,
            _disposition(),
            expected_h1=H1,
        )

    disposition = _disposition()
    disposition["outcome_score"] = 1.0
    with pytest.raises(ValueError, match="forbidden evaluation field"):
        seal.seal_confirmation_case(
            lock_path,
            H2,
            case_id,
            tmp_path / "disposition-leak" / case_id,
            _arrays(),
            _cameras(),
            _routing(),
            disposition,
            expected_h1=H1,
        )


def test_case_seal_is_clone_independent(
    tmp_path: Path,
) -> None:
    lock_path = _write_lock(tmp_path)
    case_id = _selected_cases(lock_path)[0]
    original = tmp_path / "original" / "cases" / case_id
    seal.seal_confirmation_case(
        lock_path,
        H2,
        case_id,
        original,
        _arrays(),
        _cameras(),
        _routing(),
        _disposition(),
        expected_h1=H1,
    )
    clone_lock = tmp_path / "clone" / "lock" / lock_path.name
    clone_lock.parent.mkdir(parents=True)
    shutil.copy2(lock_path, clone_lock)
    clone_case = tmp_path / "clone" / "cases" / case_id
    shutil.copytree(original, clone_case)

    original_record = seal.validate_confirmation_case_seal(
        original,
        lock_path,
        H2,
        expected_case_id=case_id,
        expected_h1=H1,
    )
    clone_record = seal.validate_confirmation_case_seal(
        clone_case,
        clone_lock,
        H2,
        expected_case_id=case_id,
        expected_h1=H1,
    )

    assert clone_record == original_record
    manifest = json.loads((clone_case / seal.CASE_MANIFEST_FILENAME).read_text())
    assert "path" not in manifest["lock_binding"]


def test_case_sealer_rechecks_h2_lock_before_atomic_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = _write_lock(tmp_path)
    case_id = _selected_cases(lock_path)[0]
    output = tmp_path / "cases" / case_id
    original_write_json = seal._write_json

    def mutate_lock_after_manifest(
        path: Path,
        payload: dict[str, Any],
    ) -> None:
        original_write_json(path, payload)
        if path.name == seal.CASE_MANIFEST_FILENAME:
            lock_payload = json.loads(lock_path.read_text())
            lock_payload["status"] = "changed-during-case-sealing"
            lock_path.write_text(
                json.dumps(lock_payload, indent=2, sort_keys=True) + "\n"
            )

    monkeypatch.setattr(seal, "_write_json", mutate_lock_after_manifest)
    with pytest.raises(ValueError, match="checksum mismatch"):
        seal.seal_confirmation_case(
            lock_path,
            H2,
            case_id,
            output,
            _arrays(),
            _cameras(),
            _routing(),
            _disposition(),
            expected_h1=H1,
        )
    assert not output.exists()
    assert not list(output.parent.glob(f".{case_id}.staging-*"))


def test_complete_barrier_binds_exact_h1_h2_cases_and_seal_hashes(
    tmp_path: Path,
) -> None:
    lock_path = _write_lock(tmp_path)
    case_dirs = _seal_complete_cohort(lock_path, tmp_path / "cases")
    barrier_path = tmp_path / "barriers" / "predictions-sealed.json"

    barrier = seal.create_confirmation_prediction_barrier(
        barrier_path,
        lock_path,
        H2,
        case_dirs,
        expected_h1=H1,
    )

    expected_cases = _selected_cases(lock_path)
    assert barrier["exact_case_ids"] == expected_cases
    assert barrier["case_count"] == len(expected_cases) == 34
    assert barrier["lock_binding"]["implementation_commit_h1"] == H1
    assert barrier["lock_binding"]["cohort_lock_commit_h2"] == H2
    assert [row["case_id"] for row in barrier["ordered_case_seals"]] == expected_cases
    assert all(
        len(row["manifest_file_sha256"]) == 64
        and len(row["manifest_artifact_sha256"]) == 64
        for row in barrier["ordered_case_seals"]
    )
    assert (
        seal.validate_confirmation_prediction_barrier(
            barrier_path,
            lock_path,
            H2,
            case_dirs,
            expected_h1=H1,
        )
        == barrier
    )
    with pytest.raises(ValueError, match="already exists"):
        seal.create_confirmation_prediction_barrier(
            barrier_path,
            lock_path,
            H2,
            case_dirs,
            expected_h1=H1,
        )


def test_barrier_rejects_missing_extra_overlap_and_changed_case(
    tmp_path: Path,
) -> None:
    lock_path = _write_lock(tmp_path)
    case_dirs = _seal_complete_cohort(lock_path, tmp_path / "cases")
    expected_cases = _selected_cases(lock_path)

    missing = dict(case_dirs)
    missing.pop(expected_cases[-1])
    with pytest.raises(ValueError, match="every exact locked case"):
        seal.create_confirmation_prediction_barrier(
            tmp_path / "missing.json",
            lock_path,
            H2,
            missing,
            expected_h1=H1,
        )

    extra = dict(case_dirs)
    extra["unlocked-extra-ep0000"] = next(iter(case_dirs.values()))
    with pytest.raises(ValueError, match="every exact locked case"):
        seal.create_confirmation_prediction_barrier(
            tmp_path / "extra.json",
            lock_path,
            H2,
            extra,
            expected_h1=H1,
        )

    with pytest.raises(ValueError, match="overlaps case seal"):
        seal.create_confirmation_prediction_barrier(
            next(iter(case_dirs.values())) / "barrier.json",
            lock_path,
            H2,
            case_dirs,
            expected_h1=H1,
        )

    barrier_path = tmp_path / "valid.json"
    seal.create_confirmation_prediction_barrier(
        barrier_path,
        lock_path,
        H2,
        case_dirs,
        expected_h1=H1,
    )
    changed_case = case_dirs[expected_cases[0]]
    archive = changed_case / seal.ARRAY_ARCHIVE_FILENAME
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="archive content hash changed"):
        seal.validate_confirmation_prediction_barrier(
            barrier_path,
            lock_path,
            H2,
            case_dirs,
            expected_h1=H1,
        )


def test_barrier_second_pass_rejects_toctou_case_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = _write_lock(tmp_path)
    case_dirs = _seal_complete_cohort(lock_path, tmp_path / "cases")
    first_case = _selected_cases(lock_path)[0]
    archive = case_dirs[first_case] / seal.ARRAY_ARCHIVE_FILENAME
    barrier_path = tmp_path / "barrier.json"
    original_collect = seal._collect_case_records
    call_count = 0

    def mutate_after_first_pass(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        nonlocal call_count
        result = original_collect(*args, **kwargs)
        call_count += 1
        if call_count == 1:
            archive.write_bytes(archive.read_bytes() + b"changed-between-passes")
        return result

    monkeypatch.setattr(seal, "_collect_case_records", mutate_after_first_pass)
    with pytest.raises(ValueError, match="archive content hash changed"):
        seal.create_confirmation_prediction_barrier(
            barrier_path,
            lock_path,
            H2,
            case_dirs,
            expected_h1=H1,
        )
    assert not barrier_path.exists()


def test_complete_barrier_is_clone_independent(
    tmp_path: Path,
) -> None:
    lock_path = _write_lock(tmp_path)
    case_dirs = _seal_complete_cohort(lock_path, tmp_path / "original" / "cases")
    barrier_path = tmp_path / "original" / "barrier.json"
    barrier = seal.create_confirmation_prediction_barrier(
        barrier_path,
        lock_path,
        H2,
        case_dirs,
        expected_h1=H1,
    )

    clone_root = tmp_path / "clone"
    clone_lock = clone_root / "lock" / lock_path.name
    clone_lock.parent.mkdir(parents=True)
    shutil.copy2(lock_path, clone_lock)
    clone_barrier = clone_root / "barrier.json"
    shutil.copy2(barrier_path, clone_barrier)
    clone_cases_root = clone_root / "cases"
    shutil.copytree(tmp_path / "original" / "cases", clone_cases_root)
    clone_case_dirs = {
        case_id: clone_cases_root / case_id for case_id in _selected_cases(clone_lock)
    }

    assert (
        seal.validate_confirmation_prediction_barrier(
            clone_barrier,
            clone_lock,
            H2,
            clone_case_dirs,
            expected_h1=H1,
        )
        == barrier
    )
    assert all("case_dir" not in record for record in barrier["ordered_case_seals"])
