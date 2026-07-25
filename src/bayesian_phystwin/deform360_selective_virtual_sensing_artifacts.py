"""Sealed target-free artifacts for the prospective Deform360 study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .deform360_online_belief_evaluation import _resolve_prediction_archive, _sha256
from .deform360_raw_camera_observation import (
    MANIFEST_FILENAME,
    MEASUREMENT_FILENAME,
    AllTrackerPrefixRuntime,
    RawCameraObservationConfig,
    build_raw_camera_measurement_case_with_contract,
)
from .deform360_raw_pairwise_correspondence_diagnostic import CPD_ARM, UNGATED_RBF_ARM
from .deform360_selective_virtual_sensing_prediction import (
    predict_persistence_control_arrays,
    predict_persistence_pairwise_rbf_arrays,
)
from .deform360_selective_virtual_sensing_protocol import (
    EXPECTED_FRAME_COUNT,
    EXPECTED_STRATA,
    EXPECTED_UPDATE_FRAMES,
    PROTOCOL_ID,
    load_selective_virtual_sensing_protocol,
)


BACKBONE_ARCHIVE_FILENAME = "backbone_prediction.npz"
BACKBONE_SEAL_FILENAME = "prediction_seal.json"
VIRTUAL_SENSING_ARCHIVE_FILENAME = "virtual_sensing_prediction.npz"
VIRTUAL_SENSING_REPORT_FILENAME = "virtual_sensing_prediction.json"
VIRTUAL_SENSING_SEAL_FILENAME = "virtual_sensing_prediction_seal.json"
QUALITY_FAILURE_FILENAME = "quality_failure.json"
PREDICTION_COHORT_SEAL_FILENAME = "prediction_cohort_seal.json"
BACKBONE_ARTIFACT_KIND = "Deform360SelectiveVirtualSensingBackboneSeal"
PREDICTION_ARTIFACT_KIND = "Deform360SelectiveVirtualSensingPredictionSeal"
QUALITY_FAILURE_ARTIFACT_KIND = "Deform360SelectiveVirtualSensingQualityFailure"
PREDICTION_COHORT_ARTIFACT_KIND = (
    "Deform360SelectiveVirtualSensingPredictionCohortSeal"
)
TARGET_FREE_FAILURE_STAGES = frozenset(
    {
        "source-preparation",
        "prediction-prefix-staging",
        "frame-zero-reconstruction",
        "sparse-camera-measurement",
        "virtual-sensing-prediction",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_sha256(payload: Mapping[str, Any], *, digest_key: str) -> str:
    canonical = dict(payload)
    canonical.pop(digest_key, None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def selective_case_records(
    protocol_path: str | Path,
) -> tuple[dict[str, Any], ...]:
    """Return the exact case order and grouping from the canonical lock."""

    protocol = load_selective_virtual_sensing_protocol(protocol_path)
    rows = []
    for stratum in EXPECTED_STRATA:
        for object_id, episode_ids in protocol["normalized_cohort"][stratum].items():
            for episode_id in episode_ids:
                rows.append(
                    {
                        "case": f"{object_id}-ep{episode_id:04d}",
                        "object_id": object_id,
                        "episode_id": int(episode_id),
                        "episode_key": f"{object_id}/{episode_id}",
                        "stratum": stratum,
                    }
                )
    _require(len(rows) == 24, "prospective case panel is incomplete")
    _require(len({row["case"] for row in rows}) == len(rows), "case repeated")
    return tuple(rows)


def _case_record(
    protocol_path: str | Path, *, object_id: str, episode_id: int
) -> dict[str, Any]:
    matches = [
        row
        for row in selective_case_records(protocol_path)
        if row["object_id"] == object_id and row["episode_id"] == int(episode_id)
    ]
    _require(len(matches) == 1, "object/episode is outside the locked cohort")
    return matches[0]


def build_selective_backbone_seal(
    protocol_path: str | Path,
    output_dir: str | Path,
    *,
    object_id: str,
    episode_id: int,
    frame_zero_points_m: np.ndarray,
    frame_zero_reconstruction_manifest: str | Path,
    prediction_stage_manifest: str | Path,
) -> dict[str, Any]:
    """Seal frame-zero persistence without accepting any future target input."""

    protocol = load_selective_virtual_sensing_protocol(protocol_path)
    record = _case_record(
        protocol_path, object_id=object_id, episode_id=episode_id
    )
    points = np.asarray(frame_zero_points_m, dtype=np.float32)
    _require(
        points.ndim == 2
        and points.shape[1] == 3
        and len(points) > 16
        and np.all(np.isfinite(points)),
        "frame-zero points must be finite (N,3) with hidden points",
    )
    source_manifest = Path(frame_zero_reconstruction_manifest).resolve()
    stage_manifest = Path(prediction_stage_manifest).resolve()
    _require(source_manifest.is_file(), "frame-zero reconstruction manifest is missing")
    _require(stage_manifest.is_file(), "prediction-stage manifest is missing")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    persistence = np.repeat(points[None], EXPECTED_FRAME_COUNT, axis=0)
    archive_path = output / BACKBONE_ARCHIVE_FILENAME
    np.savez_compressed(
        archive_path,
        prediction_m=persistence,
        persistence_m=persistence,
        frame_zero_points_m=points,
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": BACKBONE_ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": protocol["config_sha256"],
        **record,
        "frame_count": EXPECTED_FRAME_COUNT,
        "material_point_count": len(points),
        "material_identity_sha256": _array_sha256(points),
        "prediction_archive": {
            "path": str(archive_path),
            "file_sha256": _sha256(archive_path),
            "persistence_array_sha256": _array_sha256(persistence),
        },
        "input_sha256": {
            "frame_zero_reconstruction_manifest": _sha256(source_manifest),
            "prediction_stage_manifest": _sha256(stage_manifest),
        },
        "information_boundary": {
            "object_observation_frames_used": [0],
            "future_object_track_read": False,
            "future_dense_reconstruction_read": False,
            "target_metric_read": False,
            "prediction_hashed_before_future_outcome_scoring": True,
        },
    }
    payload["result_sha256"] = _canonical_sha256(
        payload, digest_key="result_sha256"
    )
    seal_path = output / BACKBONE_SEAL_FILENAME
    seal_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    validate_selective_backbone_seal(
        payload, protocol_path=protocol_path, case_dir=output
    )
    return payload


def validate_selective_backbone_seal(
    seal: Mapping[str, Any],
    *,
    protocol_path: str | Path,
    case_dir: str | Path | None = None,
) -> None:
    protocol = load_selective_virtual_sensing_protocol(protocol_path)
    _require(seal.get("artifact_kind") == BACKBONE_ARTIFACT_KIND, "wrong backbone seal kind")
    _require(seal.get("protocol_id") == PROTOCOL_ID, "backbone protocol changed")
    _require(
        seal.get("protocol_config_sha256") == protocol["config_sha256"],
        "backbone protocol checksum changed",
    )
    expected = _case_record(
        protocol_path,
        object_id=str(seal.get("object_id", "")),
        episode_id=int(seal.get("episode_id", -1)),
    )
    _require(
        all(seal.get(key) == value for key, value in expected.items()),
        "backbone case identity changed",
    )
    _require(
        seal.get("result_sha256")
        == _canonical_sha256(seal, digest_key="result_sha256"),
        "backbone seal content checksum changed",
    )
    boundary = seal.get("information_boundary", {})
    _require(
        boundary
        == {
            "object_observation_frames_used": [0],
            "future_object_track_read": False,
            "future_dense_reconstruction_read": False,
            "target_metric_read": False,
            "prediction_hashed_before_future_outcome_scoring": True,
        },
        "backbone information boundary changed",
    )
    if case_dir is not None:
        archive = _resolve_prediction_archive(Path(case_dir).resolve(), seal)
        _require(
            _sha256(archive) == seal["prediction_archive"]["file_sha256"],
            "backbone archive checksum changed",
        )


def build_selective_raw_camera_measurement_case(
    protocol_path: str | Path,
    backbone_case_dir: str | Path,
    processed_episode_dir: str | Path,
    output_dir: str | Path,
    runtime: AllTrackerPrefixRuntime,
    *,
    config: RawCameraObservationConfig | None = None,
) -> dict[str, Any]:
    """Run causal prefixes for one locked case without exposing a target."""

    records = selective_case_records(protocol_path)
    case_dir = Path(backbone_case_dir).resolve()
    expected_names = tuple(str(record["case"]) for record in records)
    _require(case_dir.name in expected_names, "backbone case directory name changed")

    def validate(seal: Mapping[str, Any]) -> None:
        validate_selective_backbone_seal(
            seal, protocol_path=protocol_path, case_dir=case_dir
        )

    return build_raw_camera_measurement_case_with_contract(
        case_dir,
        processed_episode_dir,
        output_dir,
        runtime,
        protocol_id=PROTOCOL_ID,
        expected_case_names=expected_names,
        prediction_seal_validator=validate,
        claim_boundary=(
            "prospective target-free sparse RGB-prefix measurement; dense future "
            "outcomes remain sealed"
        ),
        config=config,
    )


def _load_selective_measurement(
    backbone_case_dir: Path,
    measurement_dir: Path,
    seal: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest_path = measurement_dir / MANIFEST_FILENAME
    archive_path = measurement_dir / MEASUREMENT_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(
        manifest.get("result_sha256")
        == _canonical_sha256(manifest, digest_key="result_sha256"),
        "measurement manifest content checksum changed",
    )
    _require(
        manifest.get("artifact_kind") == "Deform360CausalRawCameraMeasurement",
        "unsupported measurement artifact",
    )
    _require(manifest.get("protocol_id") == PROTOCOL_ID, "measurement protocol changed")
    for key in ("case", "object_id", "episode_id", "episode_key"):
        _require(manifest.get(key) == seal.get(key), f"measurement {key} changed")
    boundary = manifest.get("information_boundary", {})
    _require(
        boundary.get("target_data_read") is False
        and boundary.get("outcome_manifest_read") is False
        and boundary.get("future_reconstruction_after_frame_zero_read") is False,
        "measurement crossed its target boundary",
    )
    _require(
        manifest.get("output", {}).get("measurement_archive_sha256")
        == _sha256(archive_path),
        "measurement archive checksum changed",
    )
    _require(
        manifest.get("inputs", {}).get("prediction_seal", {}).get("sha256")
        == _sha256(backbone_case_dir / BACKBONE_SEAL_FILENAME),
        "measurement used another backbone seal",
    )
    with np.load(archive_path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]).copy() for name in stored.files}
    return manifest, arrays


def build_selective_virtual_sensing_prediction_case(
    protocol_path: str | Path,
    backbone_case_dir: str | Path,
    measurement_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Hash the frozen primary prediction before any dense future can open."""

    protocol = load_selective_virtual_sensing_protocol(protocol_path)
    case_dir = Path(backbone_case_dir).resolve()
    measurement_path = Path(measurement_dir).resolve()
    output = Path(output_dir).resolve()
    seal_path = case_dir / BACKBONE_SEAL_FILENAME
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    validate_selective_backbone_seal(
        seal, protocol_path=protocol_path, case_dir=case_dir
    )
    manifest, measurement = _load_selective_measurement(
        case_dir, measurement_path, seal
    )
    archive = _resolve_prediction_archive(case_dir, seal)
    with np.load(archive, allow_pickle=False) as stored:
        persistence = np.asarray(stored["persistence_m"]).copy()
    center_ids = np.asarray(measurement["center_ids"], dtype=np.int64)
    _require(len(center_ids) == 16, "measurement center count changed")
    selected_cameras = np.asarray(measurement["selected_cameras"]).astype(str)
    _require(
        selected_cameras.shape == (8,)
        and len(np.unique(selected_cameras)) == len(selected_cameras),
        "measurement camera panel changed",
    )
    _require(
        tuple(np.asarray(measurement["update_frames"], dtype=np.int64))
        == EXPECTED_UPDATE_FRAMES,
        "measurement update frames changed",
    )
    method_report, prediction = predict_persistence_pairwise_rbf_arrays(
        persistence,
        measurement["measurement_m"],
        measurement["measurement_visibility"],
        measurement["measurement_validity"],
        center_ids=center_ids,
        update_frames=EXPECTED_UPDATE_FRAMES,
    )
    control_report, controls = predict_persistence_control_arrays(
        persistence,
        measurement["measurement_m"],
        measurement["measurement_visibility"],
        measurement["measurement_validity"],
        center_ids=center_ids,
        update_frames=EXPECTED_UPDATE_FRAMES,
    )
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / VIRTUAL_SENSING_ARCHIVE_FILENAME
    np.savez_compressed(
        prediction_path,
        prediction_m=prediction,
        persistence_m=persistence,
        ungated_rbf_m=controls[UNGATED_RBF_ARM],
        independent_cpd_m=controls[CPD_ARM],
        center_ids=center_ids,
        selected_cameras=selected_cameras,
        update_frames=np.asarray(EXPECTED_UPDATE_FRAMES, dtype=np.int64),
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360SelectiveVirtualSensingPrediction",
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": protocol["config_sha256"],
        "case": seal["case"],
        "object_id": seal["object_id"],
        "episode_id": seal["episode_id"],
        "episode_key": seal["episode_key"],
        "stratum": seal["stratum"],
        "method": {
            "primary": method_report,
            "controls": control_report,
        },
        "inputs_sha256": {
            "backbone_seal": _sha256(seal_path),
            "backbone_archive": _sha256(archive),
            "measurement_manifest": _sha256(measurement_path / MANIFEST_FILENAME),
            "measurement_archive": _sha256(measurement_path / MEASUREMENT_FILENAME),
        },
        "output": {
            "prediction_archive": str(prediction_path),
            "prediction_archive_sha256": _sha256(prediction_path),
            "prediction_array_sha256": _array_sha256(prediction),
            "persistence_array_sha256": _array_sha256(persistence),
            "ungated_rbf_array_sha256": _array_sha256(
                controls[UNGATED_RBF_ARM]
            ),
            "independent_cpd_array_sha256": _array_sha256(controls[CPD_ARM]),
            "selected_cameras": selected_cameras.tolist(),
        },
        "information_boundary": {
            "measurement_manifest_verified_before_prediction": True,
            "measurement_archive_verified_before_prediction": True,
            "target_data_read": False,
            "outcome_manifest_read": False,
            "future_dense_reconstruction_read": False,
            "future_particle_tracks_read": False,
        },
    }
    report["result_sha256"] = _canonical_sha256(
        report, digest_key="result_sha256"
    )
    report_path = output / VIRTUAL_SENSING_REPORT_FILENAME
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    prediction_seal: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": PREDICTION_ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": protocol["config_sha256"],
        "case": seal["case"],
        "object_id": seal["object_id"],
        "episode_id": seal["episode_id"],
        "episode_key": seal["episode_key"],
        "stratum": seal["stratum"],
        "prediction_archive": {
            "path": str(prediction_path),
            "file_sha256": _sha256(prediction_path),
        },
        "prediction_report": {
            "path": str(report_path),
            "file_sha256": _sha256(report_path),
            "result_sha256": report["result_sha256"],
        },
        "input_sha256": report["inputs_sha256"],
        "measurement_result_sha256": manifest["result_sha256"],
        "information_boundary": {
            "measurement_and_prediction_hashed_before_target_open": True,
            "target_data_read": False,
            "outcome_manifest_read": False,
            "future_dense_reconstruction_read": False,
            "future_particle_tracks_read": False,
        },
    }
    prediction_seal["result_sha256"] = _canonical_sha256(
        prediction_seal, digest_key="result_sha256"
    )
    final_seal_path = output / VIRTUAL_SENSING_SEAL_FILENAME
    final_seal_path.write_text(
        json.dumps(prediction_seal, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    validate_selective_prediction_seal(
        prediction_seal, protocol_path=protocol_path, prediction_dir=output
    )
    return prediction_seal


def validate_selective_prediction_seal(
    seal: Mapping[str, Any],
    *,
    protocol_path: str | Path,
    prediction_dir: str | Path,
) -> None:
    protocol = load_selective_virtual_sensing_protocol(protocol_path)
    _require(seal.get("artifact_kind") == PREDICTION_ARTIFACT_KIND, "wrong prediction seal kind")
    _require(seal.get("protocol_id") == PROTOCOL_ID, "prediction protocol changed")
    _require(
        seal.get("protocol_config_sha256") == protocol["config_sha256"],
        "prediction protocol checksum changed",
    )
    expected = _case_record(
        protocol_path,
        object_id=str(seal.get("object_id", "")),
        episode_id=int(seal.get("episode_id", -1)),
    )
    _require(
        all(seal.get(key) == value for key, value in expected.items()),
        "prediction case identity changed",
    )
    _require(
        seal.get("result_sha256")
        == _canonical_sha256(seal, digest_key="result_sha256"),
        "prediction seal content checksum changed",
    )
    boundary = seal.get("information_boundary", {})
    _require(
        boundary.get("measurement_and_prediction_hashed_before_target_open") is True
        and boundary.get("target_data_read") is False
        and boundary.get("outcome_manifest_read") is False
        and boundary.get("future_dense_reconstruction_read") is False
        and boundary.get("future_particle_tracks_read") is False,
        "prediction crossed its target boundary",
    )
    root = Path(prediction_dir).resolve()
    prediction = root / Path(str(seal["prediction_archive"]["path"])).name
    report = root / Path(str(seal["prediction_report"]["path"])).name
    _require(
        prediction.is_file()
        and _sha256(prediction) == seal["prediction_archive"]["file_sha256"],
        "sealed prediction archive changed",
    )
    _require(
        report.is_file()
        and _sha256(report) == seal["prediction_report"]["file_sha256"],
        "sealed prediction report changed",
    )


def record_selective_quality_failure(
    protocol_path: str | Path,
    output_dir: str | Path,
    *,
    object_id: str,
    episode_id: int,
    stage: str,
    error_type: str,
    error_message: str,
    evidence_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Seal a target-free technical failure without selecting a replacement."""

    protocol = load_selective_virtual_sensing_protocol(protocol_path)
    record = _case_record(
        protocol_path, object_id=object_id, episode_id=episode_id
    )
    _require(stage in TARGET_FREE_FAILURE_STAGES, "failure stage is not target-free")
    _require(error_type.strip() == error_type and error_type, "error type is empty")
    _require(
        error_message.strip() == error_message and error_message,
        "error message is empty",
    )
    evidence: dict[str, str] = {}
    for name, path_value in sorted((evidence_paths or {}).items()):
        _require(name and name.strip() == name, "failure evidence name is invalid")
        path = Path(path_value).resolve()
        _require(path.is_file(), f"failure evidence is missing: {name}")
        evidence[name] = _sha256(path)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": QUALITY_FAILURE_ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": protocol["config_sha256"],
        **record,
        "stage": stage,
        "error_type": error_type,
        "error_message": error_message,
        "evidence_sha256": evidence,
        "disposition": "episode excluded without replacement",
        "information_boundary": {
            "failure_determined_without_target_data": True,
            "target_data_read": False,
            "outcome_manifest_read": False,
            "future_dense_reconstruction_read": False,
            "future_particle_tracks_read": False,
            "replacement_episode_selected": False,
        },
    }
    payload["result_sha256"] = _canonical_sha256(
        payload, digest_key="result_sha256"
    )
    (output / QUALITY_FAILURE_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    validate_selective_quality_failure(payload, protocol_path=protocol_path)
    return payload


def validate_selective_quality_failure(
    failure: Mapping[str, Any], *, protocol_path: str | Path
) -> None:
    protocol = load_selective_virtual_sensing_protocol(protocol_path)
    _require(
        failure.get("artifact_kind") == QUALITY_FAILURE_ARTIFACT_KIND,
        "wrong quality-failure kind",
    )
    _require(failure.get("protocol_id") == PROTOCOL_ID, "failure protocol changed")
    _require(
        failure.get("protocol_config_sha256") == protocol["config_sha256"],
        "failure protocol checksum changed",
    )
    expected = _case_record(
        protocol_path,
        object_id=str(failure.get("object_id", "")),
        episode_id=int(failure.get("episode_id", -1)),
    )
    _require(
        all(failure.get(key) == value for key, value in expected.items()),
        "quality-failure case identity changed",
    )
    _require(
        failure.get("stage") in TARGET_FREE_FAILURE_STAGES,
        "quality-failure stage changed",
    )
    _require(
        failure.get("result_sha256")
        == _canonical_sha256(failure, digest_key="result_sha256"),
        "quality-failure content checksum changed",
    )
    _require(
        failure.get("information_boundary")
        == {
            "failure_determined_without_target_data": True,
            "target_data_read": False,
            "outcome_manifest_read": False,
            "future_dense_reconstruction_read": False,
            "future_particle_tracks_read": False,
            "replacement_episode_selected": False,
        },
        "quality failure crossed its target boundary",
    )


def build_selective_prediction_cohort_seal(
    protocol_path: str | Path,
    prediction_root: str | Path,
    failure_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Account for the full panel before permitting any target construction."""

    protocol = load_selective_virtual_sensing_protocol(protocol_path)
    predictions = Path(prediction_root).resolve()
    failures = Path(failure_root).resolve()
    rows: list[dict[str, Any]] = []
    successful_objects: dict[str, set[str]] = {
        stratum: set() for stratum in EXPECTED_STRATA
    }
    for record in selective_case_records(protocol_path):
        case_name = str(record["case"])
        prediction_dir = predictions / case_name
        prediction_path = prediction_dir / VIRTUAL_SENSING_SEAL_FILENAME
        failure_path = failures / case_name / QUALITY_FAILURE_FILENAME
        _require(
            prediction_path.is_file() != failure_path.is_file(),
            f"case must have exactly one prediction or quality failure: {case_name}",
        )
        if prediction_path.is_file():
            prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
            validate_selective_prediction_seal(
                prediction,
                protocol_path=protocol_path,
                prediction_dir=prediction_dir,
            )
            successful_objects[str(record["stratum"])].add(
                str(record["object_id"])
            )
            rows.append(
                {
                    **record,
                    "status": "prediction-sealed",
                    "artifact_file_sha256": _sha256(prediction_path),
                    "artifact_result_sha256": prediction["result_sha256"],
                }
            )
        else:
            failure = json.loads(failure_path.read_text(encoding="utf-8"))
            validate_selective_quality_failure(
                failure, protocol_path=protocol_path
            )
            _require(
                all(failure.get(key) == value for key, value in record.items()),
                "quality-failure panel position changed",
            )
            rows.append(
                {
                    **record,
                    "status": "quality-failure",
                    "failure_stage": failure["stage"],
                    "artifact_file_sha256": _sha256(failure_path),
                    "artifact_result_sha256": failure["result_sha256"],
                }
            )

    per_stratum = {
        stratum: len(successful_objects[stratum]) for stratum in EXPECTED_STRATA
    }
    total_objects = sum(per_stratum.values())
    cohort = protocol["config"]["cohort"]
    minimum_total = int(cohort["minimum_evaluable_objects"])
    minimum_per_stratum = int(cohort["minimum_evaluable_objects_per_stratum"])
    eligible = total_objects >= minimum_total and all(
        count >= minimum_per_stratum for count in per_stratum.values()
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": PREDICTION_COHORT_ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": protocol["config_sha256"],
        "case_count": len(rows),
        "prediction_count": sum(row["status"] == "prediction-sealed" for row in rows),
        "quality_failure_count": sum(
            row["status"] == "quality-failure" for row in rows
        ),
        "evaluable_object_count": total_objects,
        "evaluable_object_count_by_stratum": per_stratum,
        "minimum_evaluable_objects": minimum_total,
        "minimum_evaluable_objects_per_stratum": minimum_per_stratum,
        "eligible_to_open_targets": eligible,
        "cases": rows,
        "information_boundary": {
            "all_locked_cases_accounted_before_target_open": True,
            "prediction_or_failure_exclusive_per_case": True,
            "target_data_read": False,
            "outcome_manifest_read": False,
            "future_dense_reconstruction_read": False,
            "future_particle_tracks_read": False,
            "replacement_episode_selected": False,
        },
    }
    payload["result_sha256"] = _canonical_sha256(
        payload, digest_key="result_sha256"
    )
    destination = Path(output_path).resolve()
    _require(not destination.exists(), "prediction cohort seal already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    validate_selective_prediction_cohort_seal(
        payload, protocol_path=protocol_path
    )
    return payload


def validate_selective_prediction_cohort_seal(
    seal: Mapping[str, Any],
    *,
    protocol_path: str | Path,
    require_eligible: bool = False,
    prediction_root: str | Path | None = None,
    failure_root: str | Path | None = None,
) -> None:
    protocol = load_selective_virtual_sensing_protocol(protocol_path)
    _require(
        seal.get("artifact_kind") == PREDICTION_COHORT_ARTIFACT_KIND,
        "wrong prediction-cohort seal kind",
    )
    _require(seal.get("protocol_id") == PROTOCOL_ID, "cohort protocol changed")
    _require(
        seal.get("protocol_config_sha256") == protocol["config_sha256"],
        "cohort protocol checksum changed",
    )
    _require(
        seal.get("result_sha256")
        == _canonical_sha256(seal, digest_key="result_sha256"),
        "prediction-cohort content checksum changed",
    )
    expected_cases = [dict(record) for record in selective_case_records(protocol_path)]
    actual_cases = seal.get("cases")
    _require(
        isinstance(actual_cases, list)
        and all(isinstance(row, Mapping) for row in actual_cases)
        and len(actual_cases) == len(expected_cases),
        "prediction-cohort case panel changed",
    )
    _require(
        all(
            all(actual.get(key) == value for key, value in expected.items())
            for actual, expected in zip(actual_cases, expected_cases, strict=True)
        ),
        "prediction-cohort case order changed",
    )
    statuses = [row.get("status") for row in actual_cases]
    _require(
        all(status in {"prediction-sealed", "quality-failure"} for status in statuses),
        "prediction-cohort case status changed",
    )
    prediction_count = statuses.count("prediction-sealed")
    failure_count = statuses.count("quality-failure")
    _require(
        seal.get("case_count") == len(actual_cases)
        and seal.get("prediction_count") == prediction_count
        and seal.get("quality_failure_count") == failure_count,
        "prediction-cohort case counts changed",
    )
    successful_objects: dict[str, set[str]] = {
        stratum: set() for stratum in EXPECTED_STRATA
    }
    for row in actual_cases:
        if row["status"] == "prediction-sealed":
            successful_objects[str(row["stratum"])].add(str(row["object_id"]))
    per_stratum = {
        stratum: len(successful_objects[stratum]) for stratum in EXPECTED_STRATA
    }
    total_objects = sum(per_stratum.values())
    cohort = protocol["config"]["cohort"]
    minimum_total = int(cohort["minimum_evaluable_objects"])
    minimum_per_stratum = int(cohort["minimum_evaluable_objects_per_stratum"])
    eligible = total_objects >= minimum_total and all(
        count >= minimum_per_stratum for count in per_stratum.values()
    )
    _require(
        seal.get("evaluable_object_count") == total_objects
        and seal.get("evaluable_object_count_by_stratum") == per_stratum
        and seal.get("minimum_evaluable_objects") == minimum_total
        and seal.get("minimum_evaluable_objects_per_stratum")
        == minimum_per_stratum
        and seal.get("eligible_to_open_targets") is eligible,
        "prediction-cohort evaluability calculation changed",
    )
    _require(
        seal.get("information_boundary")
        == {
            "all_locked_cases_accounted_before_target_open": True,
            "prediction_or_failure_exclusive_per_case": True,
            "target_data_read": False,
            "outcome_manifest_read": False,
            "future_dense_reconstruction_read": False,
            "future_particle_tracks_read": False,
            "replacement_episode_selected": False,
        },
        "prediction cohort crossed its target boundary",
    )
    _require(
        (prediction_root is None) == (failure_root is None),
        "prediction and failure roots must be verified together",
    )
    if prediction_root is not None and failure_root is not None:
        predictions = Path(prediction_root).resolve()
        failures = Path(failure_root).resolve()
        for row in actual_cases:
            case_name = str(row["case"])
            if row["status"] == "prediction-sealed":
                prediction_dir = predictions / case_name
                artifact_path = prediction_dir / VIRTUAL_SENSING_SEAL_FILENAME
                _require(
                    artifact_path.is_file()
                    and _sha256(artifact_path) == row["artifact_file_sha256"],
                    f"cohort prediction artifact changed: {case_name}",
                )
                artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
                validate_selective_prediction_seal(
                    artifact,
                    protocol_path=protocol_path,
                    prediction_dir=prediction_dir,
                )
            else:
                artifact_path = failures / case_name / QUALITY_FAILURE_FILENAME
                _require(
                    artifact_path.is_file()
                    and _sha256(artifact_path) == row["artifact_file_sha256"],
                    f"cohort quality-failure artifact changed: {case_name}",
                )
                artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
                validate_selective_quality_failure(
                    artifact, protocol_path=protocol_path
                )
            _require(
                artifact["result_sha256"] == row["artifact_result_sha256"],
                f"cohort artifact result changed: {case_name}",
            )
    if require_eligible:
        _require(
            seal.get("eligible_to_open_targets") is True,
            "prediction cohort did not meet the locked evaluability threshold",
        )


def authorize_selective_target_case(
    cohort_seal: Mapping[str, Any],
    *,
    protocol_path: str | Path,
    prediction_root: str | Path,
    failure_root: str | Path,
    object_id: str,
    episode_id: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authorize one sealed successful case after the all-case cohort gate."""

    validate_selective_prediction_cohort_seal(
        cohort_seal,
        protocol_path=protocol_path,
        require_eligible=True,
        prediction_root=prediction_root,
        failure_root=failure_root,
    )
    record = _case_record(
        protocol_path, object_id=object_id, episode_id=episode_id
    )
    rows = [
        row
        for row in cohort_seal["cases"]
        if row.get("case") == record["case"]
    ]
    _require(len(rows) == 1, "authorized case is absent from the cohort seal")
    _require(
        rows[0].get("status") == "prediction-sealed",
        "quality-failure case cannot open a target",
    )
    prediction_dir = Path(prediction_root).resolve() / str(record["case"])
    prediction_path = prediction_dir / VIRTUAL_SENSING_SEAL_FILENAME
    _require(
        prediction_path.is_file()
        and _sha256(prediction_path) == rows[0]["artifact_file_sha256"],
        "authorized prediction seal changed",
    )
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    validate_selective_prediction_seal(
        prediction,
        protocol_path=protocol_path,
        prediction_dir=prediction_dir,
    )
    _require(
        prediction["result_sha256"] == rows[0]["artifact_result_sha256"],
        "authorized prediction result changed",
    )
    return record, prediction


__all__ = [
    "BACKBONE_ARCHIVE_FILENAME",
    "BACKBONE_SEAL_FILENAME",
    "VIRTUAL_SENSING_ARCHIVE_FILENAME",
    "VIRTUAL_SENSING_REPORT_FILENAME",
    "VIRTUAL_SENSING_SEAL_FILENAME",
    "authorize_selective_target_case",
    "PREDICTION_COHORT_SEAL_FILENAME",
    "QUALITY_FAILURE_FILENAME",
    "build_selective_backbone_seal",
    "build_selective_prediction_cohort_seal",
    "build_selective_raw_camera_measurement_case",
    "build_selective_virtual_sensing_prediction_case",
    "record_selective_quality_failure",
    "selective_case_records",
    "validate_selective_backbone_seal",
    "validate_selective_prediction_cohort_seal",
    "validate_selective_prediction_seal",
    "validate_selective_quality_failure",
]
