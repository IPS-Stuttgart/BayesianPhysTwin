"""Prediction-first artifacts for the fresh Deform360 bias-aware protocol.

Every builder in this module is outcome-blind.  Dense future object geometry,
particle identities, and target metrics are deliberately absent from every
function signature.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360_bias_aware_belief_development import (
    Deform360BiasAwareDevelopmentConfig,
    predict_bias_aware_candidate_arrays,
)
from .deform360_bias_aware_prospective_protocol import (
    EXPECTED_FRAME_COUNT,
    EXPECTED_STRATA,
    EXPECTED_UPDATE_FRAMES,
    PROTOCOL_ID,
    SOURCE_LOCK_GROUP_COUNT,
    SOURCE_LOCK_SHA256,
    load_bias_aware_prospective_protocol,
)
from .deform360_cpd_diagnostic import _symmetric_set_chamfer_m
from .deform360_online_belief_evaluation import _resolve_prediction_archive
from .deform360_raw_camera_observation import (
    MANIFEST_FILENAME,
    MEASUREMENT_FILENAME,
    AllTrackerPrefixRuntime,
    RawCameraObservationConfig,
    build_raw_camera_measurement_case_with_contract,
)


PHYSICAL_ARCHIVE_FILENAME = "physical_prediction.npz"
PHYSICAL_MANIFEST_FILENAME = "physical_prediction_manifest.json"
BACKBONE_SEAL_FILENAME = "prediction_seal.json"
MEASUREMENT_CYCLE_ARCHIVE_FILENAME = "measurement_cycle_uncertainty.npz"
MEASUREMENT_CYCLE_MANIFEST_FILENAME = "measurement_cycle_uncertainty_manifest.json"
PREDICTION_ARCHIVE_FILENAME = "bias_aware_prediction.npz"
PREDICTION_REPORT_FILENAME = "bias_aware_prediction.json"
PREDICTION_SEAL_FILENAME = "bias_aware_prediction_seal.json"
QUALITY_FAILURE_FILENAME = "quality_failure.json"
PREDICTION_COHORT_SEAL_FILENAME = "prediction_cohort_seal.json"
CALIBRATION_SUPPORT_REJECTION_FILENAME = "calibration_support_rejection.json"

BACKBONE_ARTIFACT_KIND = "Deform360BiasAwareProspectiveBackboneSeal"
PREDICTION_ARTIFACT_KIND = "Deform360BiasAwareProspectivePredictionSeal"
QUALITY_FAILURE_ARTIFACT_KIND = "Deform360BiasAwareProspectiveQualityFailure"
PREDICTION_COHORT_ARTIFACT_KIND = (
    "Deform360BiasAwareProspectivePredictionCohortSeal"
)
CALIBRATION_SUPPORT_REJECTION_ARTIFACT_KIND = (
    "Deform360BiasAwareProspectiveCalibrationSupportRejection"
)

PHYSICAL_ARRAY_NAMES = frozenset(
    {
        "prediction_m",
        "persistence_m",
        "driven_readout_m",
        "zero_action_readout_m",
        "action_support",
        "frame_zero_points_m",
    }
)
TARGET_FREE_FAILURE_STAGES = frozenset(
    {
        "source-preparation",
        "prediction-prefix-staging",
        "frame-zero-reconstruction",
        "automatic-physical-twin",
        "physical-rollout",
        "sparse-camera-measurement",
        "measurement-uncertainty",
        "measurement-cycle-uncertainty",
        "bias-aware-prediction",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of one regular file."""

    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    """Hash dtype, shape, and exact array bytes."""

    array = np.ascontiguousarray(value)
    descriptor = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(
        descriptor + b"\0" + array.view(np.uint8).tobytes()
    ).hexdigest()


def canonical_sha256(payload: Mapping[str, Any], *, digest_key: str) -> str:
    """Hash a JSON object after removing its self-declared digest."""

    canonical = dict(payload)
    canonical.pop(digest_key, None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def prospective_case_records(
    protocol_path: str | Path,
    *,
    role: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return the canonical role, stratum, object, and episode order."""

    protocol = load_bias_aware_prospective_protocol(protocol_path)
    roles = (role,) if role is not None else ("calibration", "target")
    _require(all(value in {"calibration", "target"} for value in roles), "bad role")
    rows: list[dict[str, Any]] = []
    for current_role in roles:
        cohort = protocol[f"{current_role}_cohort"]
        for stratum in EXPECTED_STRATA:
            for object_id, episode_ids in cohort[stratum].items():
                for episode_id in episode_ids:
                    rows.append(
                        {
                            "case": f"{object_id}-ep{episode_id:04d}",
                            "object_id": object_id,
                            "episode_id": int(episode_id),
                            "episode_key": f"{object_id}/{episode_id}",
                            "stratum": stratum,
                            "role": current_role,
                        }
                    )
    expected_count = 9 if role == "calibration" else 24 if role == "target" else 33
    _require(len(rows) == expected_count, "prospective case panel is incomplete")
    _require(len({row["case"] for row in rows}) == len(rows), "case repeated")
    return tuple(rows)


def prospective_case_record(
    protocol_path: str | Path,
    *,
    object_id: str,
    episode_id: int,
) -> dict[str, Any]:
    """Resolve one case and reject every object or episode outside the lock."""

    matches = [
        row
        for row in prospective_case_records(protocol_path)
        if row["object_id"] == object_id and row["episode_id"] == int(episode_id)
    ]
    _require(len(matches) == 1, "object/episode is outside the prospective lock")
    return matches[0]


def _validate_physical_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    _require(PHYSICAL_ARRAY_NAMES <= set(arrays), "physical archive is incomplete")
    prediction = np.asarray(arrays["prediction_m"])
    persistence = np.asarray(arrays["persistence_m"])
    driven = np.asarray(arrays["driven_readout_m"])
    zero = np.asarray(arrays["zero_action_readout_m"])
    support = np.asarray(arrays["action_support"])
    frame_zero = np.asarray(arrays["frame_zero_points_m"])
    _require(
        prediction.ndim == 3
        and prediction.shape[0] == EXPECTED_FRAME_COUNT
        and prediction.shape[2] == 3,
        "physical prediction shape changed",
    )
    _require(
        persistence.shape == prediction.shape
        and driven.shape == prediction.shape
        and zero.shape == prediction.shape,
        "physical trajectory shapes differ",
    )
    _require(frame_zero.shape == prediction.shape[1:], "frame-zero shape changed")
    _require(support.shape == (prediction.shape[1],), "action support shape changed")
    _require(
        all(
            np.all(np.isfinite(np.asarray(arrays[name])))
            for name in PHYSICAL_ARRAY_NAMES
        ),
        "physical archive is non-finite",
    )
    _require(np.all((support >= 0.0) & (support <= 1.0)), "invalid action support")
    _require(
        np.array_equal(persistence, np.repeat(frame_zero[None], len(prediction), axis=0)),
        "persistence is not the exact frame-zero trajectory",
    )
    _require(
        np.array_equal(prediction[0], frame_zero)
        and np.array_equal(driven[0], frame_zero)
        and np.array_equal(zero[0], frame_zero),
        "physical trajectories changed material identity at frame zero",
    )


def load_physical_archive(path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate the frozen physical/archive interface."""

    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]).copy() for name in stored.files}
    _validate_physical_arrays(arrays)
    return arrays


def build_prospective_backbone_seal(
    protocol_path: str | Path,
    output_dir: str | Path,
    *,
    object_id: str,
    episode_id: int,
    physical_archive: str | Path,
    physical_manifest: str | Path,
) -> dict[str, Any]:
    """Copy and seal one target-free physical/persistence backbone."""

    protocol = load_bias_aware_prospective_protocol(protocol_path)
    record = prospective_case_record(
        protocol_path, object_id=object_id, episode_id=episode_id
    )
    source_archive = Path(physical_archive).resolve()
    source_manifest = Path(physical_manifest).resolve()
    _require(source_archive.is_file(), "physical archive is missing")
    _require(source_manifest.is_file(), "physical manifest is missing")
    physical = json.loads(source_manifest.read_text(encoding="utf-8"))
    _require(isinstance(physical, Mapping), "physical manifest is invalid")
    _require(
        physical.get("information_boundary", {}).get("future_object_rgb_read") is False
        and physical.get("information_boundary", {}).get("future_object_geometry_read")
        is False
        and physical.get("information_boundary", {}).get("outcome_read") is False,
        "physical runner crossed its prediction boundary",
    )
    arrays = load_physical_archive(source_archive)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    archive = output / PHYSICAL_ARCHIVE_FILENAME
    shutil.copy2(source_archive, archive)
    _require(file_sha256(archive) == file_sha256(source_archive), "archive copy changed")
    copied_manifest = output / PHYSICAL_MANIFEST_FILENAME
    shutil.copy2(source_manifest, copied_manifest)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": BACKBONE_ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": protocol["config_sha256"],
        **record,
        "frame_count": EXPECTED_FRAME_COUNT,
        "material_point_count": int(len(arrays["frame_zero_points_m"])),
        "material_identity_sha256": array_sha256(arrays["frame_zero_points_m"]),
        "prediction_archive": {
            "path": str(archive),
            "file_sha256": file_sha256(archive),
            "array_sha256": {
                name: array_sha256(arrays[name]) for name in sorted(PHYSICAL_ARRAY_NAMES)
            },
        },
        "physical_manifest": {
            "path": str(copied_manifest),
            "file_sha256": file_sha256(copied_manifest),
        },
        "information_boundary": {
            "object_observation_frames_used": [0],
            "known_future_robot_action_read": True,
            "future_object_rgb_read": False,
            "future_object_geometry_read": False,
            "future_object_track_read": False,
            "future_tactile_read": False,
            "target_metric_read": False,
            "prediction_hashed_before_future_outcome_scoring": True,
        },
    }
    payload["result_sha256"] = canonical_sha256(payload, digest_key="result_sha256")
    _write_json(output / BACKBONE_SEAL_FILENAME, payload)
    validate_prospective_backbone_seal(
        payload, protocol_path=protocol_path, case_dir=output
    )
    return payload


def validate_prospective_backbone_seal(
    seal: Mapping[str, Any],
    *,
    protocol_path: str | Path,
    case_dir: str | Path | None = None,
) -> None:
    """Validate case identity, hashes, and the frame-zero-only boundary."""

    protocol = load_bias_aware_prospective_protocol(protocol_path)
    _require(seal.get("artifact_kind") == BACKBONE_ARTIFACT_KIND, "wrong backbone kind")
    _require(seal.get("protocol_id") == PROTOCOL_ID, "backbone protocol changed")
    _require(
        seal.get("protocol_config_sha256") == protocol["config_sha256"],
        "backbone protocol checksum changed",
    )
    expected = prospective_case_record(
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
        == canonical_sha256(seal, digest_key="result_sha256"),
        "backbone seal checksum changed",
    )
    expected_boundary = {
        "object_observation_frames_used": [0],
        "known_future_robot_action_read": True,
        "future_object_rgb_read": False,
        "future_object_geometry_read": False,
        "future_object_track_read": False,
        "future_tactile_read": False,
        "target_metric_read": False,
        "prediction_hashed_before_future_outcome_scoring": True,
    }
    _require(seal.get("information_boundary") == expected_boundary, "boundary changed")
    if case_dir is not None:
        root = Path(case_dir).resolve()
        archive = _resolve_prediction_archive(root, seal)
        manifest = root / Path(str(seal["physical_manifest"]["path"])).name
        _require(
            file_sha256(archive) == seal["prediction_archive"]["file_sha256"],
            "backbone archive checksum changed",
        )
        _require(
            manifest.is_file()
            and file_sha256(manifest) == seal["physical_manifest"]["file_sha256"],
            "physical manifest checksum changed",
        )
        arrays = load_physical_archive(archive)
        for name, digest in seal["prediction_archive"]["array_sha256"].items():
            _require(array_sha256(arrays[name]) == digest, f"{name} array changed")


def build_prospective_raw_camera_measurement_case(
    protocol_path: str | Path,
    backbone_case_dir: str | Path,
    processed_episode_dir: str | Path,
    output_dir: str | Path,
    runtime: AllTrackerPrefixRuntime,
    *,
    config: RawCameraObservationConfig | None = None,
) -> dict[str, Any]:
    """Run AllTracker on causal prefixes under the fresh case contract."""

    records = prospective_case_records(protocol_path)
    case_dir = Path(backbone_case_dir).resolve()
    expected_names = tuple(str(record["case"]) for record in records)
    _require(case_dir.name in expected_names, "backbone directory is outside the lock")

    def validate(seal: Mapping[str, Any]) -> None:
        validate_prospective_backbone_seal(
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
            "prospective target-free sparse RGB-prefix measurement; object "
            "futures and metrics remain sealed"
        ),
        config=config,
    )


def load_prospective_measurement(
    protocol_path: str | Path,
    backbone_case_dir: str | Path,
    measurement_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    """Checksum-validate one sparse measurement without opening an outcome."""

    case_dir = Path(backbone_case_dir).resolve()
    measurement_root = Path(measurement_dir).resolve()
    seal_path = case_dir / BACKBONE_SEAL_FILENAME
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    validate_prospective_backbone_seal(
        seal, protocol_path=protocol_path, case_dir=case_dir
    )
    manifest_path = measurement_root / MANIFEST_FILENAME
    archive_path = measurement_root / MEASUREMENT_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(
        manifest.get("result_sha256")
        == canonical_sha256(manifest, digest_key="result_sha256"),
        "measurement manifest checksum changed",
    )
    _require(
        manifest.get("artifact_kind") == "Deform360CausalRawCameraMeasurement"
        and manifest.get("protocol_id") == PROTOCOL_ID,
        "measurement contract changed",
    )
    for key in ("case", "object_id", "episode_id", "episode_key"):
        _require(manifest.get(key) == seal.get(key), f"measurement {key} changed")
    boundary = manifest.get("information_boundary", {})
    _require(
        boundary.get("target_data_read") is False
        and boundary.get("outcome_manifest_read") is False
        and boundary.get("future_reconstruction_after_frame_zero_read") is False,
        "measurement crossed its future boundary",
    )
    _require(
        manifest.get("output", {}).get("measurement_archive_sha256")
        == file_sha256(archive_path),
        "measurement archive checksum changed",
    )
    _require(
        manifest.get("inputs", {}).get("prediction_seal", {}).get("sha256")
        == file_sha256(seal_path),
        "measurement used another backbone seal",
    )
    with np.load(archive_path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]).copy() for name in stored.files}
    return manifest, arrays, seal


def select_raw_backbone_arrays(
    physical_prior_m: np.ndarray,
    persistence_m: np.ndarray,
    measurement_m: np.ndarray,
    measurement_visibility: np.ndarray,
    measurement_validity: np.ndarray,
    *,
    center_ids: np.ndarray,
    update_frames: Sequence[int] = EXPECTED_UPDATE_FRAMES,
    minimum_support: int = 3,
) -> tuple[dict[str, Any], np.ndarray]:
    """Select physical or persistence intervals using current observations only."""

    physical_input = np.asarray(physical_prior_m)
    persistence_input = np.asarray(persistence_m)
    physical = np.asarray(physical_input, dtype=np.float64)
    persistence = np.asarray(persistence_input, dtype=np.float64)
    measurement = np.asarray(measurement_m, dtype=np.float64)
    visible = np.asarray(measurement_visibility, dtype=bool)
    valid = np.asarray(measurement_validity, dtype=bool)
    centers = np.asarray(center_ids, dtype=np.int64)
    updates = tuple(int(value) for value in update_frames)
    _require(physical.shape == persistence.shape == measurement.shape, "shape mismatch")
    _require(visible.shape == valid.shape == physical.shape[:2], "mask shape changed")
    _require(
        centers.ndim == 1
        and len(centers) == len(np.unique(centers))
        and np.all((centers >= 0) & (centers < physical.shape[1])),
        "invalid center IDs",
    )
    _require(minimum_support >= 1, "minimum support must be positive")
    selected = physical_input.copy()
    records: list[dict[str, Any]] = []
    backbones = {"physical_prior": physical, "persistence": persistence}
    for index, update in enumerate(updates):
        stop = updates[index + 1] if index + 1 < len(updates) else len(physical)
        available = (
            visible[update, centers]
            & valid[update, centers]
            & np.all(np.isfinite(measurement[update, centers]), axis=1)
            & np.all(np.isfinite(physical[update, centers]), axis=1)
            & np.all(np.isfinite(persistence[update, centers]), axis=1)
        )
        ids = centers[available]
        observed = measurement[update, ids]
        sufficient = len(ids) >= minimum_support
        chamfer: dict[str, float | None]
        if sufficient:
            chamfer = {
                name: _symmetric_set_chamfer_m(backbone[update, ids], observed)
                for name, backbone in backbones.items()
            }
            chosen = min(
                ("physical_prior", "persistence"),
                key=lambda name: (
                    float(chamfer[name]),
                    0 if name == "physical_prior" else 1,
                ),
            )
        else:
            chamfer = (
                {
                    name: _symmetric_set_chamfer_m(backbone[update, ids], observed)
                    for name, backbone in backbones.items()
                }
                if len(ids)
                else {"physical_prior": None, "persistence": None}
            )
            chosen = "persistence"
        selected[update + 1 : stop] = (
            physical_input if chosen == "physical_prior" else persistence_input
        )[update + 1 : stop]
        records.append(
            {
                "frame": update,
                "interval_end_exclusive": stop,
                "available_center_count": int(len(ids)),
                "support_sufficient": sufficient,
                "decision": (
                    "current_observation_chamfer"
                    if sufficient
                    else "insufficient_support_persistence_default"
                ),
                "selected_backbone": chosen,
                "current_observation_chamfer_m": chamfer,
            }
        )
    return {
        "minimum_support": minimum_support,
        "tie_break": "physical_prior",
        "updates": records,
        "information_boundary": {
            "target_argument_accepted": False,
            "future_observation_read": False,
            "selection_uses_current_update_observation_only": True,
        },
    }, selected


def source_reliability_and_variance(
    measurement_arrays: Mapping[str, np.ndarray],
    cycle_arrays: Mapping[str, np.ndarray],
    *,
    center_ids: np.ndarray,
    config: Deform360BiasAwareDevelopmentConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the source-v4 residual-independent reliability contract."""

    cfg = config or Deform360BiasAwareDevelopmentConfig()
    selected_camera_count = len(measurement_arrays["selected_cameras"])
    _require(selected_camera_count >= 2, "fewer than two selected cameras")
    centers = np.asarray(center_ids, dtype=np.int64)
    inlier_count = np.asarray(
        measurement_arrays["triangulation_inlier_view_count"], dtype=np.float64
    )
    reprojection = np.asarray(
        measurement_arrays["triangulation_median_reprojection_px"], dtype=np.float64
    )
    expected = (len(cfg.update_frames), len(centers))
    _require(inlier_count.shape == expected, "inlier-count shape changed")
    _require(reprojection.shape == expected, "reprojection shape changed")
    redundancy = np.clip(
        (inlier_count - 1.0) / (selected_camera_count - 1.0), 0.0, 1.0
    )
    geometry = np.exp(-0.5 * np.square(reprojection / cfg.reprojection_scale_px))
    reliability = redundancy * geometry
    reliability[~np.isfinite(reliability)] = 0.0

    covariance = np.asarray(cycle_arrays["measurement_covariance_m2"], dtype=np.float64)
    covariance_valid = np.asarray(
        cycle_arrays["measurement_covariance_valid"], dtype=bool
    )
    _require(covariance.shape[-2:] == (3, 3), "cycle covariance shape changed")
    variance = np.empty(expected, dtype=np.float64)
    for update_index, update in enumerate(cfg.update_frames):
        selected = covariance[update, centers]
        isotropic = np.trace(selected, axis1=1, axis2=2) / 3.0
        valid_covariance = covariance_valid[update, centers] & np.isfinite(isotropic)
        reliability[update_index, ~valid_covariance] = 0.0
        variance[update_index] = np.where(
            valid_covariance,
            np.maximum(isotropic, cfg.observation_variance_floor_m2),
            cfg.observation_variance_floor_m2,
        )
    return reliability, variance


def _load_cycle_artifact(
    cycle_dir: Path,
    *,
    measurement_manifest: Mapping[str, Any],
    measurement_dir: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest_path = cycle_dir / MEASUREMENT_CYCLE_MANIFEST_FILENAME
    archive_path = cycle_dir / MEASUREMENT_CYCLE_ARCHIVE_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(
        manifest.get("result_sha256")
        == canonical_sha256(manifest, digest_key="result_sha256"),
        "cycle manifest checksum changed",
    )
    boundary = manifest.get("information_boundary", {})
    _require(
        boundary.get("target_data_read") is False
        and boundary.get("outcome_manifest_read") is False
        and boundary.get("future_frame_read") is False,
        "cycle uncertainty crossed its future boundary",
    )
    _require(
        manifest.get("inputs", {}).get("measurement_manifest", {}).get("sha256")
        == file_sha256(measurement_dir / MANIFEST_FILENAME)
        and manifest.get("inputs", {})
        .get("measurement_manifest", {})
        .get("result_sha256")
        == measurement_manifest["result_sha256"],
        "cycle uncertainty used another measurement",
    )
    _require(
        manifest.get("output", {}).get("archive_sha256") == file_sha256(archive_path),
        "cycle archive checksum changed",
    )
    with np.load(archive_path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]).copy() for name in stored.files}
    return manifest, arrays


def build_prospective_bias_aware_prediction_case(
    protocol_path: str | Path,
    backbone_case_dir: str | Path,
    measurement_dir: str | Path,
    cycle_uncertainty_dir: str | Path,
    source_lock_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Construct and hash one frozen-v4 prediction before outcomes can open."""

    protocol = load_bias_aware_prospective_protocol(protocol_path)
    case_dir = Path(backbone_case_dir).resolve()
    measurement_root = Path(measurement_dir).resolve()
    cycle_root = Path(cycle_uncertainty_dir).resolve()
    source_lock_file = Path(source_lock_path).resolve()
    output = Path(output_dir).resolve()
    _require(file_sha256(source_lock_file) == SOURCE_LOCK_SHA256, "source lock changed")
    source_lock = json.loads(source_lock_file.read_text(encoding="utf-8"))
    _require(
        source_lock.get("candidate_certified") is True
        and source_lock.get("fresh_accuracy_evaluation_allowed") is True,
        "source lock did not certify deployment",
    )
    measurement_manifest, measurement, seal = load_prospective_measurement(
        protocol_path, case_dir, measurement_root
    )
    cycle_manifest, cycle = _load_cycle_artifact(
        cycle_root,
        measurement_manifest=measurement_manifest,
        measurement_dir=measurement_root,
    )
    physical_path = _resolve_prediction_archive(case_dir, seal)
    physical = load_physical_archive(physical_path)
    centers = np.asarray(measurement["center_ids"], dtype=np.int64)
    update_frames = tuple(int(value) for value in measurement["update_frames"])
    _require(update_frames == EXPECTED_UPDATE_FRAMES, "update frames changed")
    baseline_report, baseline = select_raw_backbone_arrays(
        physical["prediction_m"],
        physical["persistence_m"],
        measurement["measurement_m"],
        measurement["measurement_visibility"],
        measurement["measurement_validity"],
        center_ids=centers,
        update_frames=update_frames,
    )
    config = Deform360BiasAwareDevelopmentConfig()
    reliability, variance = source_reliability_and_variance(
        measurement, cycle, center_ids=centers, config=config
    )
    candidate_report, candidate = predict_bias_aware_candidate_arrays(
        baseline,
        np.asarray(physical["driven_readout_m"], dtype=np.float64)
        - np.asarray(physical["zero_action_readout_m"], dtype=np.float64),
        physical["frame_zero_points_m"],
        physical["action_support"],
        measurement["measurement_m"],
        measurement["measurement_visibility"],
        measurement["measurement_validity"],
        center_ids=centers,
        prior_reliability=reliability,
        observation_variance_m2=variance,
        config=config,
    )
    prediction = candidate.copy()
    for update in candidate_report["updates"]:
        if update["candidate_available"]:
            continue
        start = int(update["frame"]) + 1
        stop = int(update["interval_end_exclusive"])
        _require(
            np.array_equal(prediction[start:stop], baseline[start:stop]),
            "ineligible interval is not exact fallback",
        )
    output.mkdir(parents=True, exist_ok=False)
    archive = output / PREDICTION_ARCHIVE_FILENAME
    np.savez_compressed(
        archive,
        prediction_m=prediction,
        selected_raw_backbone=baseline,
        bias_aware_candidate_unguarded=candidate,
        physical_prior_m=physical["prediction_m"],
        persistence_m=physical["persistence_m"],
        center_ids=centers,
        update_frames=np.asarray(update_frames, dtype=np.int64),
        prior_reliability=reliability.astype(np.float32),
        observation_variance_m2=variance.astype(np.float32),
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360BiasAwareProspectivePrediction",
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": protocol["config_sha256"],
        **{key: seal[key] for key in ("case", "object_id", "episode_id", "episode_key", "stratum", "role")},
        "method": {
            "config": asdict(config),
            "source_lock_candidate_certified": True,
            "source_lock_upper_regret_m": source_lock["upper_regret_m"],
            "source_lock_finite_sample_coverage": source_lock["finite_sample_coverage"],
            "baseline_selector": baseline_report,
            "bias_aware_candidate": candidate_report,
        },
        "inputs_sha256": {
            "protocol": file_sha256(protocol_path),
            "backbone_seal": file_sha256(case_dir / BACKBONE_SEAL_FILENAME),
            "physical_archive": file_sha256(physical_path),
            "measurement_manifest": file_sha256(measurement_root / MANIFEST_FILENAME),
            "measurement_archive": file_sha256(measurement_root / MEASUREMENT_FILENAME),
            "cycle_manifest": file_sha256(
                cycle_root / MEASUREMENT_CYCLE_MANIFEST_FILENAME
            ),
            "cycle_archive": file_sha256(
                cycle_root / MEASUREMENT_CYCLE_ARCHIVE_FILENAME
            ),
            "source_lock": file_sha256(source_lock_file),
        },
        "input_result_sha256": {
            "backbone": seal["result_sha256"],
            "measurement": measurement_manifest["result_sha256"],
            "cycle": cycle_manifest["result_sha256"],
        },
        "output": {
            "prediction_archive": str(archive),
            "prediction_archive_sha256": file_sha256(archive),
            "prediction_array_sha256": array_sha256(prediction),
            "baseline_array_sha256": array_sha256(baseline),
            "candidate_array_sha256": array_sha256(candidate),
            "candidate_update_count": candidate_report["candidate_update_count"],
            "exact_fallback_update_count": int(
                sum(not row["candidate_available"] for row in candidate_report["updates"])
            ),
        },
        "information_boundary": {
            "target_argument_accepted": False,
            "target_data_read": False,
            "outcome_manifest_read": False,
            "future_object_rgb_read": False,
            "future_object_geometry_read": False,
            "future_particle_tracks_read": False,
            "source_lock_read_before_prediction": True,
            "prediction_hashed_before_outcome": True,
        },
    }
    report["result_sha256"] = canonical_sha256(report, digest_key="result_sha256")
    report_path = output / PREDICTION_REPORT_FILENAME
    _write_json(report_path, report)
    prediction_seal: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": PREDICTION_ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": protocol["config_sha256"],
        **{key: seal[key] for key in ("case", "object_id", "episode_id", "episode_key", "stratum", "role")},
        "prediction_archive": {
            "path": str(archive),
            "file_sha256": file_sha256(archive),
            "prediction_array_sha256": array_sha256(prediction),
            "baseline_array_sha256": array_sha256(baseline),
        },
        "prediction_report": {
            "path": str(report_path),
            "file_sha256": file_sha256(report_path),
            "result_sha256": report["result_sha256"],
        },
        "source_lock_sha256": file_sha256(source_lock_file),
        "information_boundary": {
            "prediction_hashed_before_outcome": True,
            "target_data_read": False,
            "outcome_manifest_read": False,
            "future_object_rgb_read": False,
            "future_object_geometry_read": False,
            "future_particle_tracks_read": False,
        },
    }
    prediction_seal["result_sha256"] = canonical_sha256(
        prediction_seal, digest_key="result_sha256"
    )
    _write_json(output / PREDICTION_SEAL_FILENAME, prediction_seal)
    validate_prospective_prediction_seal(
        prediction_seal, protocol_path=protocol_path, prediction_dir=output
    )
    return prediction_seal


def validate_prospective_prediction_seal(
    seal: Mapping[str, Any],
    *,
    protocol_path: str | Path,
    prediction_dir: str | Path,
) -> None:
    """Validate one immutable target-free v4 prediction."""

    protocol = load_bias_aware_prospective_protocol(protocol_path)
    _require(seal.get("artifact_kind") == PREDICTION_ARTIFACT_KIND, "wrong seal kind")
    _require(seal.get("protocol_id") == PROTOCOL_ID, "prediction protocol changed")
    _require(
        seal.get("protocol_config_sha256") == protocol["config_sha256"],
        "prediction protocol checksum changed",
    )
    expected = prospective_case_record(
        protocol_path,
        object_id=str(seal.get("object_id", "")),
        episode_id=int(seal.get("episode_id", -1)),
    )
    _require(
        all(seal.get(key) == value for key, value in expected.items()),
        "prediction case identity changed",
    )
    _require(seal.get("source_lock_sha256") == SOURCE_LOCK_SHA256, "lock changed")
    _require(
        seal.get("result_sha256")
        == canonical_sha256(seal, digest_key="result_sha256"),
        "prediction seal checksum changed",
    )
    boundary = seal.get("information_boundary", {})
    _require(
        boundary
        == {
            "prediction_hashed_before_outcome": True,
            "target_data_read": False,
            "outcome_manifest_read": False,
            "future_object_rgb_read": False,
            "future_object_geometry_read": False,
            "future_particle_tracks_read": False,
        },
        "prediction boundary changed",
    )
    root = Path(prediction_dir).resolve()
    archive = root / Path(str(seal["prediction_archive"]["path"])).name
    report = root / Path(str(seal["prediction_report"]["path"])).name
    _require(
        archive.is_file()
        and file_sha256(archive) == seal["prediction_archive"]["file_sha256"],
        "prediction archive changed",
    )
    _require(
        report.is_file()
        and file_sha256(report) == seal["prediction_report"]["file_sha256"],
        "prediction report changed",
    )
    with np.load(archive, allow_pickle=False) as stored:
        prediction = np.asarray(stored["prediction_m"])
        baseline = np.asarray(stored["selected_raw_backbone"])
    _require(
        array_sha256(prediction)
        == seal["prediction_archive"]["prediction_array_sha256"]
        and array_sha256(baseline)
        == seal["prediction_archive"]["baseline_array_sha256"],
        "sealed prediction arrays changed",
    )


def build_prospective_prediction_cohort_seal(
    protocol_path: str | Path,
    role: str,
    artifact_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Seal every prediction or target-free quality failure for one role."""

    _require(role in {"calibration", "target"}, "invalid prospective role")
    protocol = load_bias_aware_prospective_protocol(protocol_path)
    expected = prospective_case_records(protocol_path, role=role)
    root = Path(artifact_root).resolve()
    records: list[dict[str, Any]] = []
    prediction_count = 0
    failure_count = 0
    for case in expected:
        case_dir = root / str(case["case"])
        prediction_path = case_dir / PREDICTION_SEAL_FILENAME
        failure_path = case_dir / QUALITY_FAILURE_FILENAME
        _require(
            prediction_path.is_file() != failure_path.is_file(),
            f"case must contain exactly one sealed disposition: {case['case']}",
        )
        if prediction_path.is_file():
            prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
            validate_prospective_prediction_seal(
                prediction,
                protocol_path=protocol_path,
                prediction_dir=case_dir,
            )
            record = {
                **case,
                "disposition": "prediction",
                "artifact_file": str(prediction_path),
                "artifact_file_sha256": file_sha256(prediction_path),
                "artifact_result_sha256": prediction["result_sha256"],
            }
            prediction_count += 1
        else:
            failure = json.loads(failure_path.read_text(encoding="utf-8"))
            _require(
                failure.get("artifact_kind") == QUALITY_FAILURE_ARTIFACT_KIND
                and failure.get("protocol_id") == PROTOCOL_ID
                and failure.get("protocol_config_sha256")
                == protocol["config_sha256"]
                and failure.get("result_sha256")
                == canonical_sha256(failure, digest_key="result_sha256"),
                f"quality failure is incompatible: {case['case']}",
            )
            _require(
                all(failure.get(key) == value for key, value in case.items()),
                f"quality-failure identity changed: {case['case']}",
            )
            _require(
                failure.get("replacement_allowed") is False
                and failure.get("information_boundary")
                == {
                    "target_data_read": False,
                    "outcome_manifest_read": False,
                    "failure_recorded_before_future_open": True,
                },
                f"quality-failure boundary changed: {case['case']}",
            )
            record = {
                **case,
                "disposition": "quality_failure",
                "failure_stage": failure["stage"],
                "failure_type": failure["error_type"],
                "artifact_file": str(failure_path),
                "artifact_file_sha256": file_sha256(failure_path),
                "artifact_result_sha256": failure["result_sha256"],
            }
            failure_count += 1
        records.append(record)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": PREDICTION_COHORT_ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": protocol["config_sha256"],
        "role": role,
        "expected_case_count": len(expected),
        "prediction_count": prediction_count,
        "quality_failure_count": failure_count,
        "replacement_count": 0,
        "cases": records,
        "complete": len(records) == len(expected),
        "information_boundary": {
            "predictions_or_failures_sealed_before_future_open": True,
            "target_data_read": False,
            "outcome_manifest_read": False,
            "replacement_allowed": False,
        },
    }
    payload["result_sha256"] = canonical_sha256(payload, digest_key="result_sha256")
    destination = Path(output_path).resolve()
    _require(not destination.exists(), "prediction cohort is already sealed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json(destination, payload)
    return payload


def validate_prospective_prediction_cohort_seal(
    seal: Mapping[str, Any],
    *,
    protocol_path: str | Path,
    role: str,
    artifact_root: str | Path,
) -> None:
    """Validate one complete prediction-first cohort and every disposition."""

    _require(role in {"calibration", "target"}, "invalid prospective role")
    protocol = load_bias_aware_prospective_protocol(protocol_path)
    expected = prospective_case_records(protocol_path, role=role)
    _require(
        seal.get("artifact_kind") == PREDICTION_COHORT_ARTIFACT_KIND
        and seal.get("protocol_id") == PROTOCOL_ID
        and seal.get("protocol_config_sha256") == protocol["config_sha256"]
        and seal.get("role") == role,
        "prediction cohort contract changed",
    )
    _require(
        seal.get("result_sha256") == canonical_sha256(seal, digest_key="result_sha256"),
        "prediction cohort checksum changed",
    )
    _require(
        seal.get("information_boundary")
        == {
            "predictions_or_failures_sealed_before_future_open": True,
            "target_data_read": False,
            "outcome_manifest_read": False,
            "replacement_allowed": False,
        },
        "prediction cohort boundary changed",
    )
    rows = seal.get("cases")
    _require(
        isinstance(rows, Sequence)
        and len(rows) == len(expected)
        and seal.get("expected_case_count") == len(expected)
        and seal.get("complete") is True
        and seal.get("replacement_count") == 0,
        "prediction cohort is incomplete",
    )
    root = Path(artifact_root).resolve()
    prediction_count = 0
    failure_count = 0
    for row, case in zip(rows, expected, strict=True):
        _require(isinstance(row, Mapping), "prediction cohort row is invalid")
        _require(
            all(row.get(key) == value for key, value in case.items()),
            f"prediction cohort case order changed: {case['case']}",
        )
        case_dir = root / str(case["case"])
        disposition = row.get("disposition")
        if disposition == "prediction":
            path = case_dir / PREDICTION_SEAL_FILENAME
            prediction = json.loads(path.read_text(encoding="utf-8"))
            validate_prospective_prediction_seal(
                prediction,
                protocol_path=protocol_path,
                prediction_dir=case_dir,
            )
            expected_result = prediction["result_sha256"]
            prediction_count += 1
        elif disposition == "quality_failure":
            path = case_dir / QUALITY_FAILURE_FILENAME
            failure = json.loads(path.read_text(encoding="utf-8"))
            _require(
                failure.get("artifact_kind") == QUALITY_FAILURE_ARTIFACT_KIND
                and failure.get("protocol_id") == PROTOCOL_ID
                and failure.get("protocol_config_sha256") == protocol["config_sha256"]
                and failure.get("result_sha256")
                == canonical_sha256(failure, digest_key="result_sha256"),
                f"quality failure changed: {case['case']}",
            )
            _require(
                all(failure.get(key) == value for key, value in case.items()),
                f"quality-failure identity changed: {case['case']}",
            )
            expected_result = failure["result_sha256"]
            failure_count += 1
        else:
            raise ValueError(f"unknown cohort disposition: {case['case']}")
        _require(
            path.is_file()
            and row.get("artifact_file_sha256") == file_sha256(path)
            and row.get("artifact_result_sha256") == expected_result,
            f"cohort artifact changed: {case['case']}",
        )
    _require(
        seal.get("prediction_count") == prediction_count
        and seal.get("quality_failure_count") == failure_count
        and prediction_count + failure_count == len(expected),
        "prediction cohort counts changed",
    )


def _calibration_support_rejection_payload(
    protocol: Mapping[str, Any],
    cohort_seal: Mapping[str, Any],
) -> dict[str, Any]:
    rows = tuple(cohort_seal["cases"])
    prediction_rows = tuple(
        row for row in rows if row["disposition"] == "prediction"
    )
    evaluable_objects = {str(row["object_id"]) for row in prediction_rows}
    evaluable_by_stratum = {
        stratum: len(
            {
                str(row["object_id"])
                for row in prediction_rows
                if row["stratum"] == stratum
            }
        )
        for stratum in EXPECTED_STRATA
    }
    maximum_new_groups = len(evaluable_objects)
    maximum_combined_groups = SOURCE_LOCK_GROUP_COUNT + maximum_new_groups
    finite_sample_rank = min(
        maximum_combined_groups,
        int(np.ceil((maximum_combined_groups + 1) * 0.90)),
    )
    maximum_finite_sample_coverage = finite_sample_rank / (
        maximum_combined_groups + 1
    )
    gate = protocol["config"]["calibration_gate"]
    support_upper_bound_gates = {
        "minimum_evaluable_objects": len(evaluable_objects)
        >= int(gate["minimum_evaluable_objects"]),
        "minimum_evaluable_objects_per_stratum": all(
            count >= int(gate["minimum_evaluable_objects_per_stratum"])
            for count in evaluable_by_stratum.values()
        ),
        "minimum_new_eligible_object_groups_possible": maximum_new_groups
        >= int(gate["minimum_new_eligible_object_groups"]),
        "minimum_combined_eligible_object_groups_possible": (
            maximum_combined_groups
            >= int(gate["minimum_combined_eligible_object_groups"])
        ),
        "required_finite_sample_coverage_possible": (
            maximum_finite_sample_coverage
            >= float(gate["required_finite_sample_coverage"])
        ),
    }
    failed_support_gates = sorted(
        name for name, passed in support_upper_bound_gates.items() if not passed
    )
    _require(
        failed_support_gates,
        "calibration support remains sufficient; authorized outcomes are required",
    )
    quality_failures = [
        {
            "case": str(row["case"]),
            "object_id": str(row["object_id"]),
            "episode_id": int(row["episode_id"]),
            "stratum": str(row["stratum"]),
            "stage": str(row["failure_stage"]),
            "error_type": str(row["failure_type"]),
            "artifact_result_sha256": str(row["artifact_result_sha256"]),
        }
        for row in rows
        if row["disposition"] == "quality_failure"
    ]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": CALIBRATION_SUPPORT_REJECTION_ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": protocol["config_sha256"],
        "calibration_prediction_cohort_result_sha256": cohort_seal[
            "result_sha256"
        ],
        "source_lock_sha256": SOURCE_LOCK_SHA256,
        "decision_stage": "pre-outcome-support",
        "evaluable_object_count": len(evaluable_objects),
        "evaluable_object_count_by_stratum": evaluable_by_stratum,
        "quality_failure_count": len(quality_failures),
        "quality_failures": quality_failures,
        "maximum_possible_new_eligible_object_group_count": maximum_new_groups,
        "maximum_possible_combined_eligible_object_group_count": (
            maximum_combined_groups
        ),
        "maximum_possible_finite_sample_rank": finite_sample_rank,
        "maximum_possible_finite_sample_coverage": (
            maximum_finite_sample_coverage
        ),
        "support_upper_bound_gates": support_upper_bound_gates,
        "failed_support_gates": failed_support_gates,
        "post_outcome_gates_evaluated": False,
        "calibration_gate_passed": False,
        "target_access_authorized": False,
        "failed_gate_action": (
            "publish calibration support failure and keep every calibration and "
            "target future sealed"
        ),
        "information_boundary": {
            "method_family_changed": False,
            "candidate_threshold_changed": False,
            "observation_model_changed": False,
            "calibration_future_read": False,
            "calibration_outcome_read": False,
            "target_object_media_read": False,
            "target_future_read": False,
            "support_rejection_is_non_authorizing": True,
        },
        "claim_boundary": (
            "target-free rejection from irreversible calibration-support loss; "
            "no accuracy or non-regression claim"
        ),
    }
    payload["result_sha256"] = canonical_sha256(
        payload, digest_key="result_sha256"
    )
    return payload


def build_prospective_calibration_support_rejection(
    protocol_path: str | Path,
    cohort_seal: Mapping[str, Any],
    artifact_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Reject an impossible calibration gate without opening any future."""

    protocol = load_bias_aware_prospective_protocol(protocol_path)
    validate_prospective_prediction_cohort_seal(
        cohort_seal,
        protocol_path=protocol_path,
        role="calibration",
        artifact_root=artifact_root,
    )
    payload = _calibration_support_rejection_payload(protocol, cohort_seal)
    destination = Path(output_path).resolve()
    _require(not destination.exists(), "calibration support is already rejected")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json(destination, payload)
    return payload


def validate_prospective_calibration_support_rejection(
    rejection: Mapping[str, Any],
    *,
    protocol_path: str | Path,
    cohort_seal: Mapping[str, Any],
    artifact_root: str | Path,
) -> None:
    """Validate a fail-closed rejection against its complete prediction cohort."""

    protocol = load_bias_aware_prospective_protocol(protocol_path)
    validate_prospective_prediction_cohort_seal(
        cohort_seal,
        protocol_path=protocol_path,
        role="calibration",
        artifact_root=artifact_root,
    )
    expected = _calibration_support_rejection_payload(protocol, cohort_seal)
    _require(dict(rejection) == expected, "calibration support rejection changed")


def authorize_prospective_outcome_case(
    cohort_seal: Mapping[str, Any],
    *,
    protocol_path: str | Path,
    role: str,
    artifact_root: str | Path,
    object_id: str,
    episode_id: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authorize one future only after the complete role cohort was sealed."""

    validate_prospective_prediction_cohort_seal(
        cohort_seal,
        protocol_path=protocol_path,
        role=role,
        artifact_root=artifact_root,
    )
    record = prospective_case_record(
        protocol_path, object_id=object_id, episode_id=episode_id
    )
    _require(record["role"] == role, "case role differs from cohort authorization")
    matches = [row for row in cohort_seal["cases"] if row.get("case") == record["case"]]
    _require(len(matches) == 1, "authorized case is absent from cohort seal")
    _require(
        matches[0].get("disposition") == "prediction",
        "a pre-outcome quality failure has no authorized future",
    )
    prediction_dir = Path(artifact_root).resolve() / str(record["case"])
    prediction_path = prediction_dir / PREDICTION_SEAL_FILENAME
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    validate_prospective_prediction_seal(
        prediction,
        protocol_path=protocol_path,
        prediction_dir=prediction_dir,
    )
    _require(
        matches[0].get("artifact_result_sha256") == prediction["result_sha256"],
        "authorized prediction differs from cohort seal",
    )
    return record, prediction


def record_prospective_quality_failure(
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
    """Seal one target-free failure without selecting a replacement."""

    protocol = load_bias_aware_prospective_protocol(protocol_path)
    record = prospective_case_record(
        protocol_path, object_id=object_id, episode_id=episode_id
    )
    _require(stage in TARGET_FREE_FAILURE_STAGES, "failure stage is not target-free")
    _require(error_type and error_type.strip() == error_type, "invalid error type")
    _require(error_message and error_message.strip() == error_message, "invalid error")
    evidence: dict[str, str] = {}
    for name, value in sorted((evidence_paths or {}).items()):
        path = Path(value).resolve()
        _require(path.is_file(), f"failure evidence is missing: {name}")
        evidence[name] = file_sha256(path)
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
        "replacement_allowed": False,
        "information_boundary": {
            "target_data_read": False,
            "outcome_manifest_read": False,
            "failure_recorded_before_future_open": True,
        },
    }
    payload["result_sha256"] = canonical_sha256(payload, digest_key="result_sha256")
    _write_json(output / QUALITY_FAILURE_FILENAME, payload)
    return payload


__all__ = [
    "BACKBONE_SEAL_FILENAME",
    "CALIBRATION_SUPPORT_REJECTION_FILENAME",
    "MEASUREMENT_CYCLE_ARCHIVE_FILENAME",
    "MEASUREMENT_CYCLE_MANIFEST_FILENAME",
    "PHYSICAL_ARCHIVE_FILENAME",
    "PHYSICAL_MANIFEST_FILENAME",
    "PREDICTION_ARCHIVE_FILENAME",
    "PREDICTION_COHORT_SEAL_FILENAME",
    "PREDICTION_REPORT_FILENAME",
    "PREDICTION_SEAL_FILENAME",
    "QUALITY_FAILURE_FILENAME",
    "array_sha256",
    "authorize_prospective_outcome_case",
    "build_prospective_backbone_seal",
    "build_prospective_bias_aware_prediction_case",
    "build_prospective_calibration_support_rejection",
    "build_prospective_prediction_cohort_seal",
    "build_prospective_raw_camera_measurement_case",
    "canonical_sha256",
    "file_sha256",
    "load_physical_archive",
    "load_prospective_measurement",
    "prospective_case_record",
    "prospective_case_records",
    "record_prospective_quality_failure",
    "select_raw_backbone_arrays",
    "source_reliability_and_variance",
    "validate_prospective_backbone_seal",
    "validate_prospective_calibration_support_rejection",
    "validate_prospective_prediction_cohort_seal",
    "validate_prospective_prediction_seal",
]
