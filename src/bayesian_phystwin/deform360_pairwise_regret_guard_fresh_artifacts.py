"""Prediction-first artifacts for the fresh pairwise-regret replication.

The sole untouched public Deform360 object is a technical replication, not a
new source panel.  This module keeps preprocessing, prediction, and outcome
opening as separate artifact stages.  None of the builders below accepts a
future object trajectory or a metric target.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .deform360_bias_aware_prospective_artifacts import (
    PHYSICAL_ARRAY_NAMES,
    array_sha256,
    load_physical_archive,
)
from .deform360_pairwise_regret_guard import (
    PairwiseRegretGuardConfig,
    apply_pairwise_regret_guard,
    build_pairwise_regret_candidate_arrays,
)
from .deform360_pairwise_regret_guard_fresh_processing import (
    PREDICTION_FRAME_COUNT,
    PROCESSING_KIND,
    UPDATE_FRAMES,
    canonical_sha256,
    fresh_processing_case,
    validate_case_artifact,
    validate_fresh_processing_protocol,
    validate_fresh_source_admission,
)
from .deform360_pairwise_regret_guard_fresh_protocol import (
    file_sha256,
    validate_fresh_technical_lock,
)
from .deform360_pairwise_regret_guard_source import (
    pairwise_regret_certificate_from_dict,
)
from .deform360_raw_camera_observation import (
    MANIFEST_FILENAME,
    MEASUREMENT_FILENAME,
)

PROCESSING_COHORT_KIND = "Deform360PairwiseRegretGuardFreshProcessingCohort"
PHYSICAL_SEAL_KIND = "Deform360PairwiseRegretGuardFreshPhysicalSeal"
PREDICTION_SEAL_KIND = "Deform360PairwiseRegretGuardFreshPredictionSeal"
FAILURE_SEAL_KIND = "Deform360PairwiseRegretGuardFreshFailureSeal"
PREDICTION_COHORT_KIND = "Deform360PairwiseRegretGuardFreshPredictionCohort"

PROCESSING_FILENAME = "fresh_pairwise_processing.json"
ADMISSION_FILENAME = "fresh_pairwise_admission.json"
PHYSICAL_ARCHIVE_FILENAME = "physical_prediction.npz"
PHYSICAL_MANIFEST_FILENAME = "physical_prediction_manifest.json"
PHYSICAL_SEAL_FILENAME = "prediction_seal.json"
GUARDED_ARCHIVE_FILENAME = "guarded_prediction.npz"
GUARDED_REPORT_FILENAME = "guarded_prediction.json"
GUARDED_SEAL_FILENAME = "guarded_prediction_seal.json"
FAILURE_SEAL_FILENAME = "prediction_failure_seal.json"

_HEX64 = frozenset("0123456789abcdef")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON artifact: {source}") from exc
    _require(isinstance(value, dict), f"JSON artifact is not an object: {source}")
    return value


def _write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(dict(value), allow_nan=False))
    payload["result_sha256"] = canonical_sha256(payload, digest_key="result_sha256")
    return payload


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX64


def fresh_case_records(lock: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return the exact nine valid episodes in their frozen order."""

    validate_fresh_technical_lock(lock)
    selected = lock["selected_physical_object"]
    object_id = str(selected["object_id"])
    records = tuple(
        fresh_processing_case(lock, object_id, int(row["episode_id"]))
        for row in selected["valid_episodes"]
    )
    _require(
        len(records) == int(selected["valid_episode_count"])
        and len({row["case"] for row in records}) == len(records),
        "fresh case panel changed",
    )
    return records


def _validate_processing_result(
    result: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    case: Mapping[str, Any],
) -> None:
    validate_case_artifact(
        result,
        artifact_kind=PROCESSING_KIND,
        protocol=protocol,
        case=case,
    )
    _require(
        result.get("status") in {"admitted", "source_rejected", "technical_failure"},
        "unknown processing disposition",
    )
    boundary = result.get("information_boundary", {})
    _require(
        boundary.get("future_geometry_deserialized_for_admission") is False
        and boundary.get("target_metric_read") is False
        and boundary.get("technical_failure_causes_no_implicit_replacement") is True
        and boundary.get("held_v8_runtime_or_target_artifact_access") is False,
        "processing result crossed its information boundary",
    )


def build_fresh_processing_cohort(
    technical_lock_path: str | Path,
    processing_protocol_path: str | Path,
    processed_root: str | Path,
) -> dict[str, Any]:
    """Bind all nine source-processing dispositions without opening outcomes."""

    lock_path = Path(technical_lock_path).resolve()
    protocol_path = Path(processing_protocol_path).resolve()
    root = Path(processed_root).resolve()
    lock = _load_json(lock_path)
    protocol = _load_json(protocol_path)
    validate_fresh_technical_lock(lock)
    validate_fresh_processing_protocol(protocol)
    _require(
        protocol.get("bindings", {}).get("technical_lock_sha256")
        == lock["lock_sha256"],
        "processing protocol binds another technical lock",
    )
    rows: list[dict[str, Any]] = []
    counts = {"admitted": 0, "source_rejected": 0, "technical_failure": 0}
    for case in fresh_case_records(lock):
        case_dir = root / str(case["object_id"]) / f"episode_{case['episode_id']:04d}"
        result_path = case_dir / PROCESSING_FILENAME
        result = _load_json(result_path)
        _validate_processing_result(result, protocol=protocol, case=case)
        status = str(result["status"])
        counts[status] += 1
        admission_path = case_dir / ADMISSION_FILENAME
        admission_record: dict[str, Any] | None = None
        if status in {"admitted", "source_rejected"}:
            admission = _load_json(admission_path)
            validate_fresh_source_admission(admission, protocol=protocol, case=case)
            _require(
                bool(admission["accepted"]) == (status == "admitted")
                and result.get("admission_sha256") == admission["result_sha256"],
                "processing and admission dispositions disagree",
            )
            admission_record = {
                "path": str(admission_path),
                "file_sha256": file_sha256(admission_path),
                "result_sha256": admission["result_sha256"],
                "accepted": bool(admission["accepted"]),
                "rejection_reasons": list(admission["rejection_reasons"]),
                "observed_source_contract": admission["observed_source_contract"],
            }
        else:
            _require(not admission_path.exists(), "technical failure has an admission")
        rows.append(
            {
                **case,
                "status": status,
                "processing": {
                    "path": str(result_path),
                    "file_sha256": file_sha256(result_path),
                    "result_sha256": result["result_sha256"],
                },
                "admission": admission_record,
                "error": result.get("error"),
            }
        )
    payload = _seal(
        {
            "schema_version": 1,
            "artifact_kind": PROCESSING_COHORT_KIND,
            "technical_protocol_id": lock["protocol_id"],
            "technical_lock_sha256": lock["lock_sha256"],
            "processing_protocol_id": protocol["protocol_id"],
            "processing_protocol_sha256": protocol["protocol_sha256"],
            "expected_case_count": len(rows),
            "counts": counts,
            "complete": len(rows) == len(fresh_case_records(lock)),
            "cases": rows,
            "inputs": {
                "technical_lock": {
                    "path": str(lock_path),
                    "file_sha256": file_sha256(lock_path),
                },
                "processing_protocol": {
                    "path": str(protocol_path),
                    "file_sha256": file_sha256(protocol_path),
                },
            },
            "information_boundary": {
                "future_object_positions_deserialized": False,
                "outcome_or_metric_read": False,
                "technical_failures_retained_without_retry": True,
                "source_rejections_retained_without_replacement": True,
                "held_v8_runtime_or_target_artifact_access": False,
            },
        }
    )
    validate_fresh_processing_cohort(payload, lock=lock, protocol=protocol)
    return payload


def validate_fresh_processing_cohort(
    payload: Mapping[str, Any],
    *,
    lock: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    """Validate accounting and checksums of a processing cohort manifest."""

    validate_fresh_technical_lock(lock)
    validate_fresh_processing_protocol(protocol)
    _require(
        payload.get("artifact_kind") == PROCESSING_COHORT_KIND
        and payload.get("technical_lock_sha256") == lock["lock_sha256"]
        and payload.get("processing_protocol_sha256") == protocol["protocol_sha256"]
        and payload.get("result_sha256")
        == canonical_sha256(payload, digest_key="result_sha256"),
        "processing cohort identity changed",
    )
    expected = fresh_case_records(lock)
    rows = payload.get("cases")
    _require(
        isinstance(rows, Sequence)
        and len(rows) == len(expected)
        and payload.get("expected_case_count") == len(expected)
        and payload.get("complete") is True,
        "processing cohort is incomplete",
    )
    observed_counts = {"admitted": 0, "source_rejected": 0, "technical_failure": 0}
    for row, case in zip(rows, expected, strict=True):
        _require(
            isinstance(row, Mapping)
            and all(row.get(key) == value for key, value in case.items()),
            "processing cohort case order changed",
        )
        status = row.get("status")
        _require(status in observed_counts, "processing cohort status changed")
        observed_counts[str(status)] += 1
        result = row.get("processing")
        _require(
            isinstance(result, Mapping)
            and _valid_sha256(result.get("file_sha256"))
            and _valid_sha256(result.get("result_sha256")),
            "processing provenance is malformed",
        )
    _require(payload.get("counts") == observed_counts, "processing counts changed")
    _require(
        payload.get("information_boundary")
        == {
            "future_object_positions_deserialized": False,
            "outcome_or_metric_read": False,
            "technical_failures_retained_without_retry": True,
            "source_rejections_retained_without_replacement": True,
            "held_v8_runtime_or_target_artifact_access": False,
        },
        "processing cohort boundary changed",
    )


def build_fresh_physical_seal(
    technical_lock_path: str | Path,
    processing_protocol_path: str | Path,
    processing_cohort_path: str | Path,
    output_dir: str | Path,
    *,
    object_id: str,
    episode_id: int,
    physical_archive: str | Path,
    physical_manifest: str | Path,
) -> dict[str, Any]:
    """Copy and seal one outcome-blind physical/persistence backbone."""

    lock_path = Path(technical_lock_path).resolve()
    protocol_path = Path(processing_protocol_path).resolve()
    cohort_path = Path(processing_cohort_path).resolve()
    lock = _load_json(lock_path)
    protocol = _load_json(protocol_path)
    cohort = _load_json(cohort_path)
    validate_fresh_processing_cohort(cohort, lock=lock, protocol=protocol)
    case = fresh_processing_case(lock, object_id, int(episode_id))
    row = next(value for value in cohort["cases"] if value["case"] == case["case"])
    _require(row["status"] == "admitted", "physical seal requires source admission")
    source_archive = Path(physical_archive).resolve()
    source_manifest = Path(physical_manifest).resolve()
    arrays = load_physical_archive(source_archive)
    manifest = _load_json(source_manifest)
    boundary = manifest.get("information_boundary", {})
    _require(
        boundary.get("future_object_rgb_read") is False
        and boundary.get("future_object_geometry_read") is False
        and boundary.get("future_object_track_read") is False
        and boundary.get("outcome_read") is False,
        "physical backbone crossed its future boundary",
    )
    _require(
        all(manifest.get(key) == value for key, value in case.items()),
        "physical manifest case identity changed",
    )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    archive = output / PHYSICAL_ARCHIVE_FILENAME
    copied_manifest = output / PHYSICAL_MANIFEST_FILENAME
    shutil.copy2(source_archive, archive)
    shutil.copy2(source_manifest, copied_manifest)
    payload = _seal(
        {
            "schema_version": 1,
            "artifact_kind": PHYSICAL_SEAL_KIND,
            "protocol_id": lock["protocol_id"],
            "technical_lock_sha256": lock["lock_sha256"],
            "processing_protocol_sha256": protocol["protocol_sha256"],
            **case,
            "episode_key": f"{object_id}/{int(episode_id)}",
            "frame_count": PREDICTION_FRAME_COUNT,
            "material_point_count": int(len(arrays["frame_zero_points_m"])),
            "material_identity_sha256": array_sha256(arrays["frame_zero_points_m"]),
            "prediction_archive": {
                "path": str(archive),
                "file_sha256": file_sha256(archive),
                "array_sha256": {
                    name: array_sha256(arrays[name])
                    for name in sorted(PHYSICAL_ARRAY_NAMES)
                },
            },
            "physical_manifest": {
                "path": str(copied_manifest),
                "file_sha256": file_sha256(copied_manifest),
                "result_sha256": manifest.get("result_sha256"),
            },
            "source_processing_result_sha256": row["processing"]["result_sha256"],
            "inputs": {
                "technical_lock_file_sha256": file_sha256(lock_path),
                "processing_protocol_file_sha256": file_sha256(protocol_path),
                "processing_cohort_file_sha256": file_sha256(cohort_path),
            },
            "information_boundary": {
                "object_observation_frames_used": [0],
                "known_future_robot_action_read": True,
                "future_object_rgb_read": False,
                "future_object_geometry_read": False,
                "future_object_track_read": False,
                "target_metric_read": False,
                "prediction_hashed_before_future_outcome_scoring": True,
                "held_v8_runtime_or_target_artifact_access": False,
            },
        }
    )
    _write_json(output / PHYSICAL_SEAL_FILENAME, payload)
    validate_fresh_physical_seal(payload, case_dir=output, lock=lock, protocol=protocol)
    return payload


def write_fresh_physical_artifacts(
    archive_path: str | Path,
    manifest_path: str | Path,
    arrays: Mapping[str, np.ndarray],
    *,
    case: Mapping[str, Any],
    technical_lock: Mapping[str, Any],
    processing_protocol: Mapping[str, Any],
    physical_mode: str,
    input_files: Mapping[str, str | Path],
    runtime_provenance: Mapping[str, Any],
    fallback_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a fresh physical archive without accepting future observations."""

    _require(
        physical_mode in {"warp_twin", "persistence_fallback"},
        "unknown physical mode",
    )
    _require(
        (physical_mode == "persistence_fallback") == (fallback_diagnostics is not None),
        "fallback diagnostics disagree with physical mode",
    )
    validate_fresh_technical_lock(technical_lock)
    validate_fresh_processing_protocol(processing_protocol)
    expected = fresh_processing_case(
        technical_lock, str(case["object_id"]), int(case["episode_id"])
    )
    _require(
        all(case.get(key) == value for key, value in expected.items()),
        "physical case differs from the technical lock",
    )
    stored = {name: np.asarray(arrays[name]) for name in sorted(PHYSICAL_ARRAY_NAMES)}
    _require(set(stored) == PHYSICAL_ARRAY_NAMES, "physical arrays are incomplete")
    archive = Path(archive_path).resolve()
    archive.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(archive, **stored)
    load_physical_archive(archive)
    payload = _seal(
        {
            "schema_version": 1,
            "artifact_kind": "Deform360PairwiseRegretGuardFreshPhysicalPrediction",
            "protocol_id": technical_lock["protocol_id"],
            "technical_lock_sha256": technical_lock["lock_sha256"],
            "processing_protocol_sha256": processing_protocol["protocol_sha256"],
            **expected,
            "episode_key": f"{expected['object_id']}/{expected['episode_id']}",
            "physical_mode": physical_mode,
            "physical_admitted": physical_mode == "warp_twin",
            "fallback_diagnostics": (
                None if fallback_diagnostics is None else dict(fallback_diagnostics)
            ),
            "physical_prediction_archive": {
                "path": str(archive),
                "file_sha256": file_sha256(archive),
                "array_sha256": {
                    name: array_sha256(value) for name, value in stored.items()
                },
            },
            "input_files": {
                name: {
                    "path": str(Path(path).resolve()),
                    "sha256": file_sha256(path),
                }
                for name, path in sorted(input_files.items())
            },
            "runtime_provenance": dict(runtime_provenance),
            "information_boundary": {
                "object_observation_frames_used": [0],
                "known_future_robot_action_read": True,
                "future_object_rgb_read": False,
                "future_object_geometry_read": False,
                "future_object_track_read": False,
                "future_tactile_read": False,
                "outcome_read": False,
                "prediction_hashed_before_future_outcome_scoring": True,
                "held_v8_runtime_or_target_artifact_access": False,
            },
            "passed": True,
        }
    )
    _write_json(manifest_path, payload)
    return payload


def validate_fresh_physical_seal(
    payload: Mapping[str, Any],
    *,
    case_dir: str | Path,
    lock: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    """Validate one immutable physical backbone and all copied bytes."""

    case = fresh_processing_case(
        lock, str(payload.get("object_id", "")), int(payload.get("episode_id", -1))
    )
    _require(
        payload.get("artifact_kind") == PHYSICAL_SEAL_KIND
        and payload.get("protocol_id") == lock["protocol_id"]
        and payload.get("technical_lock_sha256") == lock["lock_sha256"]
        and payload.get("processing_protocol_sha256") == protocol["protocol_sha256"]
        and all(payload.get(key) == value for key, value in case.items())
        and payload.get("episode_key")
        == f"{case['object_id']}/{int(case['episode_id'])}"
        and payload.get("result_sha256")
        == canonical_sha256(payload, digest_key="result_sha256"),
        "physical seal identity changed",
    )
    _require(
        payload.get("information_boundary")
        == {
            "object_observation_frames_used": [0],
            "known_future_robot_action_read": True,
            "future_object_rgb_read": False,
            "future_object_geometry_read": False,
            "future_object_track_read": False,
            "target_metric_read": False,
            "prediction_hashed_before_future_outcome_scoring": True,
            "held_v8_runtime_or_target_artifact_access": False,
        },
        "physical seal boundary changed",
    )
    root = Path(case_dir).resolve()
    archive = root / PHYSICAL_ARCHIVE_FILENAME
    manifest = root / PHYSICAL_MANIFEST_FILENAME
    _require(
        archive.is_file()
        and file_sha256(archive) == payload["prediction_archive"]["file_sha256"]
        and manifest.is_file()
        and file_sha256(manifest) == payload["physical_manifest"]["file_sha256"],
        "physical artifact bytes changed",
    )
    arrays = load_physical_archive(archive)
    _require(
        all(
            array_sha256(arrays[name])
            == payload["prediction_archive"]["array_sha256"][name]
            for name in PHYSICAL_ARRAY_NAMES
        ),
        "physical arrays changed",
    )


def _camera_identity(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _validate_measurement(
    measurement_dir: Path,
    *,
    physical_seal_path: Path,
    physical_seal: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest_path = measurement_dir / MANIFEST_FILENAME
    archive_path = measurement_dir / MEASUREMENT_FILENAME
    manifest = _load_json(manifest_path)
    _require(
        manifest.get("result_sha256")
        == hashlib.sha256(
            json.dumps(
                {
                    key: value
                    for key, value in manifest.items()
                    if key != "result_sha256"
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        "measurement manifest checksum changed",
    )
    _require(
        all(
            manifest.get(key) == physical_seal.get(key)
            for key in ("case", "object_id", "episode_id", "episode_key")
        ),
        "measurement case identity changed",
    )
    boundary = manifest.get("information_boundary", {})
    _require(
        boundary.get("target_data_read") is False
        and boundary.get("outcome_manifest_read") is False
        and boundary.get("future_reconstruction_after_frame_zero_read") is False,
        "measurement crossed its future boundary",
    )
    _require(
        manifest.get("inputs", {}).get("prediction_seal", {}).get("sha256")
        == file_sha256(physical_seal_path)
        and manifest.get("output", {}).get("measurement_archive_sha256")
        == file_sha256(archive_path),
        "measurement input or output bytes changed",
    )
    with np.load(archive_path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]).copy() for name in stored.files}
    return manifest, arrays


def build_fresh_guarded_prediction(
    technical_lock_path: str | Path,
    processing_protocol_path: str | Path,
    physical_case_dir: str | Path,
    measurement_dir: str | Path,
    source_qualification_path: str | Path,
    output_dir: str | Path,
    *,
    config: PairwiseRegretGuardConfig | None = None,
) -> dict[str, Any]:
    """Apply the frozen source certificate and seal one prediction."""

    lock_path = Path(technical_lock_path).resolve()
    protocol_path = Path(processing_protocol_path).resolve()
    physical_root = Path(physical_case_dir).resolve()
    measurement_root = Path(measurement_dir).resolve()
    qualification_path = Path(source_qualification_path).resolve()
    lock = _load_json(lock_path)
    protocol = _load_json(protocol_path)
    seal_path = physical_root / PHYSICAL_SEAL_FILENAME
    physical_seal = _load_json(seal_path)
    validate_fresh_physical_seal(
        physical_seal, case_dir=physical_root, lock=lock, protocol=protocol
    )
    qualification = _load_json(qualification_path)
    _require(
        qualification.get("source_gate_passed") is True
        and qualification.get("fresh_accuracy_evaluation_allowed") is True
        and qualification.get("information_boundary", {}).get(
            "runtime_candidate_accepts_outcome"
        )
        is False,
        "source qualification did not authorize frozen deployment",
    )
    expected_qualification_sha = lock.get("method", {}).get(
        "source_qualification_file_sha256"
    )
    _require(
        file_sha256(qualification_path) == expected_qualification_sha,
        "source qualification differs from the technical lock",
    )
    certificate = pairwise_regret_certificate_from_dict(
        qualification["deployment_artifact"]["candidate_certificate"]
    )
    measurement_manifest, measurement = _validate_measurement(
        measurement_root,
        physical_seal_path=seal_path,
        physical_seal=physical_seal,
    )
    physical = load_physical_archive(physical_root / PHYSICAL_ARCHIVE_FILENAME)
    report, baseline, candidate = build_pairwise_regret_candidate_arrays(
        physical["prediction_m"],
        physical["persistence_m"],
        measurement["measurement_m"],
        measurement["measurement_visibility"],
        measurement["measurement_validity"],
        center_ids=measurement["center_ids"],
        selected_camera_ids=tuple(
            _camera_identity(value) for value in measurement["selected_cameras"]
        ),
        triangulation_inlier_view_count=measurement["triangulation_inlier_view_count"],
        triangulation_median_reprojection_px=measurement[
            "triangulation_median_reprojection_px"
        ],
        config=config,
    )
    guard_report, selected = apply_pairwise_regret_guard(
        baseline, candidate, report, certificate
    )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    archive = output / GUARDED_ARCHIVE_FILENAME
    np.savez_compressed(
        archive,
        prediction_m=selected,
        selected_raw_backbone_m=baseline,
        eligible_pairwise_candidate_m=candidate,
        physical_prior_m=physical["prediction_m"],
        persistence_m=physical["persistence_m"],
        frame_zero_points_m=physical["frame_zero_points_m"],
        center_ids=np.asarray(measurement["center_ids"], dtype=np.int64),
        update_frames=np.asarray(UPDATE_FRAMES, dtype=np.int64),
    )
    case_keys = (
        "case",
        "object_id",
        "episode_id",
        "action",
        "bimanual",
        "nonprehensile",
        "episode_key",
    )
    report_payload = _seal(
        {
            "schema_version": 1,
            "artifact_kind": "Deform360PairwiseRegretGuardFreshPrediction",
            "protocol_id": lock["protocol_id"],
            "technical_lock_sha256": lock["lock_sha256"],
            **{key: physical_seal[key] for key in case_keys},
            "candidate": report,
            "regret_guard": guard_report,
            "source_certificate": {
                "qualification_file_sha256": file_sha256(qualification_path),
                "finite_sample_coverage": certificate.finite_sample_coverage,
                "calibrated_safety_claim_allowed": False,
                "refit_on_fresh_data": False,
            },
            "output": {
                "prediction_archive": str(archive),
                "file_sha256": file_sha256(archive),
                "prediction_array_sha256": array_sha256(selected),
                "baseline_array_sha256": array_sha256(baseline),
                "candidate_array_sha256": array_sha256(candidate),
                "accepted_interval_count": guard_report["accepted_count"],
                "exact_fallback_interval_count": guard_report["exact_fallback_count"],
            },
            "inputs": {
                "technical_lock_file_sha256": file_sha256(lock_path),
                "processing_protocol_file_sha256": file_sha256(protocol_path),
                "physical_seal_file_sha256": file_sha256(seal_path),
                "physical_seal_result_sha256": physical_seal["result_sha256"],
                "measurement_manifest_file_sha256": file_sha256(
                    measurement_root / MANIFEST_FILENAME
                ),
                "measurement_result_sha256": measurement_manifest["result_sha256"],
                "measurement_archive_file_sha256": file_sha256(
                    measurement_root / MEASUREMENT_FILENAME
                ),
            },
            "information_boundary": {
                "future_camera_or_object_observation_used": False,
                "future_physical_rollout_used_for_action_support_only": True,
                "target_or_metric_argument_accepted": False,
                "outcome_manifest_read": False,
                "fresh_data_refit": False,
                "prediction_hashed_before_outcome": True,
                "held_v8_runtime_or_target_artifact_access": False,
            },
        }
    )
    report_path = output / GUARDED_REPORT_FILENAME
    _write_json(report_path, report_payload)
    seal = _seal(
        {
            "schema_version": 1,
            "artifact_kind": PREDICTION_SEAL_KIND,
            "protocol_id": lock["protocol_id"],
            "technical_lock_sha256": lock["lock_sha256"],
            **{key: physical_seal[key] for key in case_keys},
            "prediction_archive": {
                "path": str(archive),
                "file_sha256": file_sha256(archive),
                "prediction_array_sha256": array_sha256(selected),
                "baseline_array_sha256": array_sha256(baseline),
            },
            "prediction_report": {
                "path": str(report_path),
                "file_sha256": file_sha256(report_path),
                "result_sha256": report_payload["result_sha256"],
            },
            "accepted_interval_count": guard_report["accepted_count"],
            "information_boundary": {
                "prediction_hashed_before_outcome": True,
                "target_or_metric_read": False,
                "outcome_manifest_read": False,
                "fresh_data_refit": False,
                "held_v8_runtime_or_target_artifact_access": False,
            },
        }
    )
    _write_json(output / GUARDED_SEAL_FILENAME, seal)
    validate_fresh_guarded_prediction_seal(seal, prediction_dir=output, lock=lock)
    return seal


def validate_fresh_guarded_prediction_seal(
    payload: Mapping[str, Any],
    *,
    prediction_dir: str | Path,
    lock: Mapping[str, Any],
) -> None:
    """Validate one final no-refit guarded prediction seal."""

    case = fresh_processing_case(
        lock, str(payload.get("object_id", "")), int(payload.get("episode_id", -1))
    )
    _require(
        payload.get("artifact_kind") == PREDICTION_SEAL_KIND
        and payload.get("protocol_id") == lock["protocol_id"]
        and payload.get("technical_lock_sha256") == lock["lock_sha256"]
        and all(payload.get(key) == value for key, value in case.items())
        and payload.get("result_sha256")
        == canonical_sha256(payload, digest_key="result_sha256"),
        "guarded prediction seal changed",
    )
    _require(
        payload.get("information_boundary")
        == {
            "prediction_hashed_before_outcome": True,
            "target_or_metric_read": False,
            "outcome_manifest_read": False,
            "fresh_data_refit": False,
            "held_v8_runtime_or_target_artifact_access": False,
        },
        "guarded prediction boundary changed",
    )
    root = Path(prediction_dir).resolve()
    archive = root / GUARDED_ARCHIVE_FILENAME
    report = root / GUARDED_REPORT_FILENAME
    _require(
        archive.is_file()
        and file_sha256(archive) == payload["prediction_archive"]["file_sha256"]
        and report.is_file()
        and file_sha256(report) == payload["prediction_report"]["file_sha256"],
        "guarded prediction bytes changed",
    )
    with np.load(archive, allow_pickle=False) as stored:
        prediction = np.asarray(stored["prediction_m"])
        baseline = np.asarray(stored["selected_raw_backbone_m"])
    _require(
        array_sha256(prediction)
        == payload["prediction_archive"]["prediction_array_sha256"]
        and array_sha256(baseline)
        == payload["prediction_archive"]["baseline_array_sha256"],
        "guarded prediction arrays changed",
    )


def build_fresh_prediction_failure_seal(
    technical_lock_path: str | Path,
    processing_protocol_path: str | Path,
    processing_cohort_path: str | Path,
    output_path: str | Path,
    *,
    object_id: str,
    episode_id: int,
) -> dict[str, Any]:
    """Convert a retained source disposition into a terminal prediction failure."""

    lock = _load_json(technical_lock_path)
    protocol = _load_json(processing_protocol_path)
    cohort = _load_json(processing_cohort_path)
    validate_fresh_processing_cohort(cohort, lock=lock, protocol=protocol)
    case = fresh_processing_case(lock, object_id, int(episode_id))
    row = next(value for value in cohort["cases"] if value["case"] == case["case"])
    _require(
        row["status"] in {"source_rejected", "technical_failure"},
        "admitted case cannot be sealed as a source failure",
    )
    payload = _seal(
        {
            "schema_version": 1,
            "artifact_kind": FAILURE_SEAL_KIND,
            "protocol_id": lock["protocol_id"],
            "technical_lock_sha256": lock["lock_sha256"],
            **case,
            "stage": "source-processing",
            "disposition": row["status"],
            "error": row.get("error"),
            "admission": row.get("admission"),
            "processing": row["processing"],
            "replacement_allowed": False,
            "retry_allowed": False,
            "ordinary_prediction_created": False,
            "information_boundary": {
                "failure_sealed_before_outcome": True,
                "target_or_metric_read": False,
                "outcome_manifest_read": False,
                "held_v8_runtime_or_target_artifact_access": False,
            },
        }
    )
    _write_json(output_path, payload)
    return payload


def build_fresh_runtime_failure_seal(
    technical_lock_path: str | Path,
    output_path: str | Path,
    *,
    object_id: str,
    episode_id: int,
    stage: str,
    error_type: str,
    error_message: str,
    input_files: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Seal an admitted case's first technical prediction failure."""

    _require(
        stage
        in {"physical-backbone", "prefix-camera-measurement", "guarded-prediction"},
        "unsupported runtime failure stage",
    )
    _require(bool(error_type) and bool(error_message), "runtime error is empty")
    lock = _load_json(technical_lock_path)
    validate_fresh_technical_lock(lock)
    case = fresh_processing_case(lock, object_id, int(episode_id))
    payload = _seal(
        {
            "schema_version": 1,
            "artifact_kind": FAILURE_SEAL_KIND,
            "protocol_id": lock["protocol_id"],
            "technical_lock_sha256": lock["lock_sha256"],
            **case,
            "stage": stage,
            "disposition": "technical_failure",
            "error": {"type": error_type, "message": error_message},
            "inputs": {
                name: {
                    "path": str(Path(path).resolve()),
                    "file_sha256": file_sha256(path),
                }
                for name, path in sorted(input_files.items())
            },
            "replacement_allowed": False,
            "retry_allowed": False,
            "ordinary_prediction_created": False,
            "information_boundary": {
                "failure_sealed_before_outcome": True,
                "target_or_metric_read": False,
                "outcome_manifest_read": False,
                "held_v8_runtime_or_target_artifact_access": False,
            },
        }
    )
    _write_json(output_path, payload)
    return payload


def build_fresh_prediction_cohort(
    technical_lock_path: str | Path,
    prediction_root: str | Path,
) -> dict[str, Any]:
    """Seal all nine dispositions and expose whether outcome opening is legal."""

    lock_path = Path(technical_lock_path).resolve()
    root = Path(prediction_root).resolve()
    lock = _load_json(lock_path)
    records: list[dict[str, Any]] = []
    prediction_count = 0
    failure_count = 0
    for case in fresh_case_records(lock):
        case_dir = root / str(case["case"])
        prediction_path = case_dir / GUARDED_SEAL_FILENAME
        failure_path = case_dir / FAILURE_SEAL_FILENAME
        _require(
            prediction_path.is_file() != failure_path.is_file(),
            f"case needs exactly one terminal disposition: {case['case']}",
        )
        if prediction_path.is_file():
            prediction = _load_json(prediction_path)
            validate_fresh_guarded_prediction_seal(
                prediction, prediction_dir=case_dir, lock=lock
            )
            records.append(
                {
                    **case,
                    "disposition": "ordinary_prediction",
                    "artifact_path": str(prediction_path),
                    "artifact_file_sha256": file_sha256(prediction_path),
                    "artifact_result_sha256": prediction["result_sha256"],
                    "accepted_interval_count": prediction["accepted_interval_count"],
                }
            )
            prediction_count += 1
        else:
            failure = _load_json(failure_path)
            _require(
                failure.get("artifact_kind") == FAILURE_SEAL_KIND
                and failure.get("protocol_id") == lock["protocol_id"]
                and failure.get("technical_lock_sha256") == lock["lock_sha256"]
                and all(failure.get(key) == value for key, value in case.items())
                and failure.get("result_sha256")
                == canonical_sha256(failure, digest_key="result_sha256")
                and failure.get("replacement_allowed") is False
                and failure.get("retry_allowed") is False,
                "prediction failure seal changed",
            )
            records.append(
                {
                    **case,
                    "disposition": str(failure["disposition"]),
                    "artifact_path": str(failure_path),
                    "artifact_file_sha256": file_sha256(failure_path),
                    "artifact_result_sha256": failure["result_sha256"],
                    "accepted_interval_count": 0,
                }
            )
            failure_count += 1
    required = int(
        lock["execution_contract"]["ordinary_predictions_required_before_outcome_open"]
    )
    ordinary_requirement_satisfied = prediction_count == required
    status = (
        "ready_for_outcome_open"
        if ordinary_requirement_satisfied
        else "predictions_incomplete_outcome_barrier_blocked"
    )
    payload = _seal(
        {
            "schema_version": 1,
            "artifact_kind": PREDICTION_COHORT_KIND,
            "protocol_id": lock["protocol_id"],
            "technical_lock_sha256": lock["lock_sha256"],
            "status": status,
            "expected_case_count": len(records),
            "ordinary_predictions_required": required,
            "ordinary_prediction_count": prediction_count,
            "retained_failure_count": failure_count,
            "all_case_dispositions_sealed": len(records)
            == len(fresh_case_records(lock)),
            "ordinary_prediction_requirement_satisfied": ordinary_requirement_satisfied,
            "outcome_open_allowed": ordinary_requirement_satisfied,
            "replacement_count": 0,
            "cases": records,
            "inputs": {
                "technical_lock_file_sha256": file_sha256(lock_path),
            },
            "information_boundary": {
                "all_dispositions_sealed_before_outcome": True,
                "target_or_metric_read": False,
                "outcome_manifest_read": False,
                "blocked_barrier_cannot_authorize_outcome_open": True,
                "held_v8_runtime_or_target_artifact_access": False,
            },
        }
    )
    return payload


__all__ = [
    "ADMISSION_FILENAME",
    "FAILURE_SEAL_FILENAME",
    "GUARDED_ARCHIVE_FILENAME",
    "GUARDED_REPORT_FILENAME",
    "GUARDED_SEAL_FILENAME",
    "PHYSICAL_ARCHIVE_FILENAME",
    "PHYSICAL_MANIFEST_FILENAME",
    "PHYSICAL_SEAL_FILENAME",
    "PROCESSING_FILENAME",
    "build_fresh_guarded_prediction",
    "build_fresh_physical_seal",
    "build_fresh_prediction_cohort",
    "build_fresh_prediction_failure_seal",
    "build_fresh_runtime_failure_seal",
    "build_fresh_processing_cohort",
    "fresh_case_records",
    "validate_fresh_guarded_prediction_seal",
    "validate_fresh_physical_seal",
    "validate_fresh_processing_cohort",
    "write_fresh_physical_artifacts",
]
