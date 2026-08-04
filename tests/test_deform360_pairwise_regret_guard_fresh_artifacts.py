from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform360_bias_aware_prospective_artifacts import (
    PHYSICAL_ARRAY_NAMES,
    array_sha256,
)
from bayesian_phystwin.deform360_pairwise_regret_guard_fresh_artifacts import (
    FAILURE_SEAL_FILENAME,
    GUARDED_ARCHIVE_FILENAME,
    GUARDED_REPORT_FILENAME,
    GUARDED_SEAL_FILENAME,
    PHYSICAL_SEAL_FILENAME,
    build_fresh_guarded_prediction,
    build_fresh_physical_seal,
    build_fresh_prediction_cohort,
    build_fresh_prediction_failure_seal,
    build_fresh_processing_cohort,
    fresh_case_records,
    validate_fresh_processing_cohort,
)
from bayesian_phystwin.deform360_pairwise_regret_guard_fresh_processing import (
    ADMISSION_KIND,
    PREDICTION_FRAME_COUNT,
    PROCESSING_KIND,
    canonical_sha256,
    seal_case_artifact,
)
from bayesian_phystwin.deform360_pairwise_regret_guard_fresh_protocol import (
    file_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "configs/sota/deform360_pairwise_regret_guard_fresh_technical_v1.json"
PROTOCOL = (
    ROOT / "configs/sota/deform360_pairwise_regret_guard_fresh_processing_v1.json"
)
QUALIFICATION = (
    ROOT
    / "results/sota/deform360_pairwise_regret_guard_source_v1/source_qualification.json"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _admission(
    protocol: dict[str, object],
    case: dict[str, object],
    *,
    accepted: bool,
) -> dict[str, object]:
    reasons = (
        [] if accepted else ["frame-zero point count is outside backend admission"]
    )
    digest = "a" * 64
    return seal_case_artifact(
        ADMISSION_KIND,
        protocol=protocol,
        case=case,
        payload={
            "accepted": accepted,
            "rejection_reasons": reasons,
            "observed_source_contract": {
                "camera_count": 3,
                "cameras": ["cam0", "cam1", "cam2"],
                "frame_zero_point_count": 128 if accepted else 54,
                "split_frame_count": 76,
                "active_frame_count": 76,
                "train": [0, 60],
                "test": [60, 76],
                "contact_start_frame": 0,
                "contact_end_frame": 75,
                "stage_inputs_valid": True,
            },
            "source_files": {
                name: {"basename": f"{name}.bin", "sha256": digest}
                for name in (
                    "metadata",
                    "control_meta",
                    "split",
                    "calibrate",
                    "frame_zero",
                    "future_payload",
                )
            },
            "information_boundary": {
                "future_object_positions_deserialized": False,
                "future_payload_bytes_hashed": True,
                "future_metrics_read": False,
                "held_v8_runtime_or_target_artifact_access": False,
            },
        },
    )


def _processing(
    protocol: dict[str, object],
    case: dict[str, object],
    *,
    status: str,
    admission: dict[str, object] | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": status,
        "information_boundary": {
            "source_rgb_and_masks_read": True,
            "future_geometry_deserialized_for_admission": False,
            "target_metric_read": False,
            "technical_failure_causes_no_implicit_replacement": True,
            "held_v8_runtime_or_target_artifact_access": False,
        },
    }
    if admission is not None:
        payload.update(
            {
                "admission_sha256": admission["result_sha256"],
                "admission_accepted": admission["accepted"],
                "admission_rejection_reasons": admission["rejection_reasons"],
            }
        )
    else:
        payload["error"] = {"type": "RuntimeError", "message": "sealed failure"}
    return seal_case_artifact(
        PROCESSING_KIND,
        protocol=protocol,
        case=case,
        payload=payload,
    )


def _processing_tree(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    lock = _load(LOCK)
    protocol = _load(PROTOCOL)
    root = tmp_path / "processed"
    for index, case in enumerate(fresh_case_records(lock)):
        case_dir = root / str(case["object_id"]) / f"episode_{case['episode_id']:04d}"
        if index == 1:
            admission = _admission(protocol, case, accepted=False)
            status = "source_rejected"
        elif index == 3:
            admission = None
            status = "technical_failure"
        else:
            admission = _admission(protocol, case, accepted=True)
            status = "admitted"
        processing = _processing(protocol, case, status=status, admission=admission)
        _write(case_dir / "fresh_pairwise_processing.json", processing)
        if admission is not None:
            _write(case_dir / "fresh_pairwise_admission.json", admission)
    cohort = build_fresh_processing_cohort(LOCK, PROTOCOL, root)
    cohort_path = tmp_path / "processing_cohort.json"
    _write(cohort_path, cohort)
    return cohort_path, cohort


def _physical_arrays(point_count: int = 128) -> dict[str, np.ndarray]:
    points = np.stack(
        (
            np.linspace(0.0, 0.1, point_count),
            np.zeros(point_count),
            np.zeros(point_count),
        ),
        axis=1,
    ).astype(np.float32)
    persistence = np.repeat(points[None], PREDICTION_FRAME_COUNT, axis=0)
    physical = persistence.copy()
    for frame in range(PREDICTION_FRAME_COUNT):
        physical[frame, :, 1] += np.float32(0.02 * frame / (PREDICTION_FRAME_COUNT - 1))
    return {
        "prediction_m": physical,
        "persistence_m": persistence,
        "driven_readout_m": physical.copy(),
        "zero_action_readout_m": persistence.copy(),
        "action_support": np.ones(point_count, dtype=np.float32),
        "frame_zero_points_m": points,
    }


def _physical_case(
    tmp_path: Path,
    cohort_path: Path,
    *,
    episode_id: int = 0,
) -> Path:
    arrays = _physical_arrays()
    source_archive = tmp_path / f"source-physical-{episode_id}.npz"
    np.savez_compressed(source_archive, **arrays)
    case = next(
        row
        for row in fresh_case_records(_load(LOCK))
        if row["episode_id"] == episode_id
    )
    manifest = {
        **case,
        "result_sha256": "b" * 64,
        "information_boundary": {
            "future_object_rgb_read": False,
            "future_object_geometry_read": False,
            "future_object_track_read": False,
            "outcome_read": False,
        },
    }
    source_manifest = tmp_path / f"source-physical-{episode_id}.json"
    _write(source_manifest, manifest)
    destination = tmp_path / "physical" / str(case["case"])
    build_fresh_physical_seal(
        LOCK,
        PROTOCOL,
        cohort_path,
        destination,
        object_id=str(case["object_id"]),
        episode_id=int(case["episode_id"]),
        physical_archive=source_archive,
        physical_manifest=source_manifest,
    )
    return destination


def _measurement(tmp_path: Path, physical_dir: Path) -> Path:
    seal_path = physical_dir / PHYSICAL_SEAL_FILENAME
    seal = _load(seal_path)
    with np.load(
        physical_dir / "physical_prediction.npz", allow_pickle=False
    ) as stored:
        physical = np.asarray(stored["prediction_m"])
        initial = np.asarray(stored["frame_zero_points_m"])
    centers = np.arange(16, dtype=np.int64)
    measurement = np.full_like(physical, np.nan)
    visible = np.zeros(physical.shape[:2], dtype=bool)
    valid = np.zeros_like(visible)
    measurement[0, centers] = initial[centers]
    visible[0, centers] = True
    valid[0, centers] = True
    for frame in (19, 38, 57):
        measurement[frame, centers] = physical[frame, centers]
        visible[frame, centers] = True
        valid[frame, centers] = True
    output = tmp_path / "measurement" / str(seal["case"])
    output.mkdir(parents=True)
    archive = output / "measurement.npz"
    np.savez_compressed(
        archive,
        measurement_m=measurement,
        measurement_visibility=visible,
        measurement_validity=valid,
        center_ids=centers,
        selected_cameras=np.asarray(["cam0", "cam1", "cam2"]),
        triangulation_inlier_view_count=np.full((3, 16), 3, dtype=np.int16),
        triangulation_median_reprojection_px=np.ones((3, 16), dtype=np.float32),
    )
    manifest: dict[str, object] = {
        "case": seal["case"],
        "object_id": seal["object_id"],
        "episode_id": seal["episode_id"],
        "episode_key": seal["episode_key"],
        "inputs": {"prediction_seal": {"sha256": file_sha256(seal_path)}},
        "output": {"measurement_archive_sha256": file_sha256(archive)},
        "information_boundary": {
            "target_data_read": False,
            "outcome_manifest_read": False,
            "future_reconstruction_after_frame_zero_read": False,
        },
    }
    manifest["result_sha256"] = hashlib.sha256(
        json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    _write(output / "measurement_manifest.json", manifest)
    return output


def test_processing_cohort_binds_all_nine_dispositions(tmp_path: Path) -> None:
    _, cohort = _processing_tree(tmp_path)
    assert cohort["counts"] == {
        "admitted": 7,
        "source_rejected": 1,
        "technical_failure": 1,
    }
    validate_fresh_processing_cohort(cohort, lock=_load(LOCK), protocol=_load(PROTOCOL))
    tampered = json.loads(json.dumps(cohort))
    tampered["cases"][0]["status"] = "technical_failure"
    with pytest.raises(ValueError, match="identity|checksum|counts"):
        validate_fresh_processing_cohort(
            tampered, lock=_load(LOCK), protocol=_load(PROTOCOL)
        )


def test_guarded_prediction_is_outcome_blind_and_checksum_bound(tmp_path: Path) -> None:
    cohort_path, _ = _processing_tree(tmp_path)
    physical_dir = _physical_case(tmp_path, cohort_path)
    measurement_dir = _measurement(tmp_path, physical_dir)
    output = tmp_path / "guarded" / physical_dir.name
    seal = build_fresh_guarded_prediction(
        LOCK,
        PROTOCOL,
        physical_dir,
        measurement_dir,
        QUALIFICATION,
        output,
    )
    assert seal["information_boundary"]["target_or_metric_read"] is False
    assert seal["information_boundary"]["fresh_data_refit"] is False
    assert (output / GUARDED_ARCHIVE_FILENAME).is_file()
    assert (output / GUARDED_REPORT_FILENAME).is_file()


def _write_synthetic_prediction_seal(
    case_dir: Path,
    case: dict[str, object],
) -> None:
    case_dir.mkdir(parents=True)
    arrays = _physical_arrays()
    archive = case_dir / GUARDED_ARCHIVE_FILENAME
    np.savez_compressed(
        archive,
        prediction_m=arrays["prediction_m"],
        selected_raw_backbone_m=arrays["persistence_m"],
    )
    report = {"result_sha256": "c" * 64}
    report_path = case_dir / GUARDED_REPORT_FILENAME
    _write(report_path, report)
    lock = _load(LOCK)
    seal: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "Deform360PairwiseRegretGuardFreshPredictionSeal",
        "protocol_id": lock["protocol_id"],
        "technical_lock_sha256": lock["lock_sha256"],
        **case,
        "episode_key": f"{case['object_id']}/{case['episode_id']}",
        "prediction_archive": {
            "path": str(archive),
            "file_sha256": file_sha256(archive),
            "prediction_array_sha256": array_sha256(arrays["prediction_m"]),
            "baseline_array_sha256": array_sha256(arrays["persistence_m"]),
        },
        "prediction_report": {
            "path": str(report_path),
            "file_sha256": file_sha256(report_path),
            "result_sha256": report["result_sha256"],
        },
        "accepted_interval_count": 0,
        "information_boundary": {
            "prediction_hashed_before_outcome": True,
            "target_or_metric_read": False,
            "outcome_manifest_read": False,
            "fresh_data_refit": False,
            "held_v8_runtime_or_target_artifact_access": False,
        },
    }
    seal["result_sha256"] = canonical_sha256(seal, digest_key="result_sha256")
    _write(case_dir / GUARDED_SEAL_FILENAME, seal)


def test_retained_failures_block_nine_prediction_outcome_barrier(
    tmp_path: Path,
) -> None:
    cohort_path, cohort = _processing_tree(tmp_path)
    lock = _load(LOCK)
    prediction_root = tmp_path / "predictions"
    for row in cohort["cases"]:
        case = {key: row[key] for key in fresh_case_records(lock)[0]}
        case_dir = prediction_root / str(row["case"])
        if row["status"] == "admitted":
            _write_synthetic_prediction_seal(case_dir, case)
        else:
            case_dir.mkdir(parents=True)
            build_fresh_prediction_failure_seal(
                LOCK,
                PROTOCOL,
                cohort_path,
                case_dir / FAILURE_SEAL_FILENAME,
                object_id=str(row["object_id"]),
                episode_id=int(row["episode_id"]),
            )
    barrier = build_fresh_prediction_cohort(LOCK, prediction_root)
    assert barrier["ordinary_prediction_count"] == 7
    assert barrier["retained_failure_count"] == 2
    assert barrier["all_case_dispositions_sealed"] is True
    assert barrier["ordinary_prediction_requirement_satisfied"] is False
    assert barrier["outcome_open_allowed"] is False
    assert barrier["status"] == "predictions_incomplete_outcome_barrier_blocked"


def test_physical_array_fixture_has_frozen_contract() -> None:
    assert set(_physical_arrays()) == PHYSICAL_ARRAY_NAMES
