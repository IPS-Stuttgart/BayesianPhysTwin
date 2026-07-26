"""MVTracker competence control on one outcome-open Deform360 source case.

The first study deliberately uses rendered depth produced by the released
full-sequence reconstruction.  It is therefore a privileged reconstruction
control, not a causal observation arm.  Prediction artifacts must be sealed
before the already-open source target is loaded.  A positive result only
authorizes a later prefix-only learned-depth study.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .deform360_online_belief_evaluation import (
    _resolve_prediction_archive,
    _validate_deform360_outcome_manifest,
)


PROTOCOL_ID = "deform360-mvtracker-privileged-depth-competence-v1"
CASE_NAME = "092-squirrel-ep0008"
OBJECT_ID = "092-squirrel"
EPISODE_ID = 8
PREDICTION_FILENAME = "mvtracker_prediction.npz"
REPORT_FILENAME = "mvtracker_prediction_report.json"
SEAL_FILENAME = "mvtracker_prediction_seal.json"
EVALUATION_FILENAME = "mvtracker_source_evaluation.json"
MVTRACKER_REVISION = "ceea8ad2af77ed9b44148ef8e9eeba4ea3c3f072"
MVTRACKER_CHECKPOINT_SHA256 = (
    "a7fa86f2a7223e3e0aa4c1d3eff0dec5fe8a9227a48572ce943b8e49d8a4f8e6"
)
_NUMPY_PICKLE_MODULE_ALIASES = {
    "numpy._core.numeric": "numpy.core.numeric",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: str | Path) -> str:
    """Hash one file without interpreting its payload."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    """Hash array dtype, shape, and bytes."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a JSON artifact while excluding its self-hash."""

    unsigned = dict(payload)
    unsigned.pop("result_sha256", None)
    encoded = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class _NumpyCompatibilityUnpickler(pickle.Unpickler):
    """Read NumPy 2 archives with the NumPy 1 private-module layout."""

    def find_class(self, module: str, name: str) -> Any:
        compatible_module = _compatible_pickle_module(module)
        return super().find_class(compatible_module, name)


def _compatible_pickle_module(module: str) -> str:
    """Use the producer namespace when present and its exact legacy alias otherwise."""

    compatible_module = _NUMPY_PICKLE_MODULE_ALIASES.get(module)
    if compatible_module is None:
        return module
    try:
        importlib.import_module(module)
    except ModuleNotFoundError:
        return compatible_module
    return module


def _load_manifest_validated_target_pickle(path: str | Path) -> Any:
    """Load a validated source target across the NumPy 1/2 namespace change."""

    with Path(path).open("rb") as stream:
        return _NumpyCompatibilityUnpickler(stream).load()


def _evaluation_implementation_sha256() -> dict[str, str]:
    """Bind the exact scorer, command wrapper, and frozen protocol."""

    adapter_path = Path(__file__).resolve()
    repository_root = adapter_path.parents[2]
    paths = {
        "adapter_and_evaluator": adapter_path,
        "runner": (
            repository_root
            / "scripts"
            / "remote"
            / "run_deform360_mvtracker_source.py"
        ),
        "protocol": (
            repository_root
            / "configs"
            / "sota"
            / "deform360_mvtracker_privileged_depth_competence_v1.json"
        ),
    }
    return {name: file_sha256(path) for name, path in paths.items()}


def _config_payload(config: MVTrackerSourceConfig) -> dict[str, Any]:
    """Return the dataclass in its canonical JSON representation."""

    return json.loads(json.dumps(asdict(config), allow_nan=False))


@dataclass(frozen=True)
class MVTrackerSourceConfig:
    """Frozen choices for the privileged-depth competence control."""

    case_name: str = CASE_NAME
    prefix_frame_count: int = 20
    update_frame: int = 19
    center_ids: tuple[int, ...] = (
        69,
        678,
        350,
        660,
        563,
        605,
        526,
        347,
        590,
        130,
        624,
        134,
        548,
        637,
        399,
        371,
    )
    selected_cameras: tuple[str, ...] = (
        "brics-odroid-001_cam0",
        "brics-odroid-006_cam0",
        "brics-odroid-007_cam0",
        "brics-odroid-008_cam0",
        "brics-odroid-013_cam0",
        "brics-odroid-014_cam1",
        "brics-odroid-015_cam1",
        "brics-odroid-024_cam1",
    )
    depth_scale_to_m: float = 0.001
    visibility_threshold: float = 0.5
    observation_std_floor_m: float = 0.005
    minimum_supported_fraction: float = 0.75
    minimum_relative_gain_over_best_baseline: float = 0.10
    maximum_tracker_rmse_m: float = 0.010
    normalization_target_camera_radius: float = 6.3

    def __post_init__(self) -> None:
        _require(self.case_name == CASE_NAME, "source case is not frozen")
        _require(
            self.prefix_frame_count == self.update_frame + 1,
            "prefix and update frame disagree",
        )
        _require(self.prefix_frame_count >= 2, "prefix is too short")
        _require(len(set(self.center_ids)) == len(self.center_ids), "duplicate centers")
        _require(len(self.center_ids) >= 3, "too few source identities")
        _require(
            len(set(self.selected_cameras)) == len(self.selected_cameras),
            "duplicate cameras",
        )
        _require(len(self.selected_cameras) >= 2, "too few cameras")
        _require(self.depth_scale_to_m > 0.0, "depth scale must be positive")
        _require(
            0.0 < self.visibility_threshold < 1.0,
            "visibility threshold must lie in (0, 1)",
        )
        _require(
            self.observation_std_floor_m > 0.0,
            "observation floor must be positive",
        )
        _require(
            0.0 < self.minimum_supported_fraction <= 1.0,
            "support fraction must lie in (0, 1]",
        )
        _require(
            0.0 <= self.minimum_relative_gain_over_best_baseline < 1.0,
            "relative gain must lie in [0, 1)",
        )
        _require(self.maximum_tracker_rmse_m > 0.0, "RMSE cap must be positive")
        _require(
            self.normalization_target_camera_radius > 0.0,
            "normalization radius must be positive",
        )


def exact_anchor_trajectory(
    raw_trajectory_m: np.ndarray,
    frame_zero_points_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Make the query frame exact while preserving every predicted displacement."""

    raw = np.asarray(raw_trajectory_m, dtype=np.float64)
    initial = np.asarray(frame_zero_points_m, dtype=np.float64)
    _require(raw.ndim == 3 and raw.shape[2] == 3, "trajectory must be (T, N, 3)")
    _require(initial.shape == raw.shape[1:], "frame-zero shape differs")
    _require(np.all(np.isfinite(raw)), "tracker trajectory is not finite")
    _require(np.all(np.isfinite(initial)), "frame-zero points are not finite")
    correction = initial - raw[0]
    anchored = raw + correction[None]
    _require(
        np.array_equal(anchored[0], initial),
        "exact frame-zero anchoring failed",
    )
    return anchored.astype(np.float32), correction.astype(np.float32)


def metric_observation_variance_m2(
    visibility_probability: np.ndarray,
    anchor_correction_m: np.ndarray,
    *,
    standard_deviation_floor_m: float,
) -> np.ndarray:
    """Convert residual-independent tracker cues into metric variance.

    Visibility controls only an inflation above the metric floor.  The
    query-frame anchoring correction is also treated as uncertainty, not as
    evidence that the physical state is wrong.  No physical-state innovation
    enters this function.
    """

    visibility = np.asarray(visibility_probability, dtype=np.float64)
    correction = np.asarray(anchor_correction_m, dtype=np.float64)
    _require(visibility.ndim == 2, "visibility must be (T, N)")
    _require(
        correction.shape == (visibility.shape[1], 3),
        "anchor correction shape differs",
    )
    _require(np.all(np.isfinite(visibility)), "visibility is not finite")
    _require(
        np.all((visibility >= 0.0) & (visibility <= 1.0)),
        "visibility must lie in [0, 1]",
    )
    _require(standard_deviation_floor_m > 0.0, "variance floor is not positive")
    floor_m2 = standard_deviation_floor_m**2
    anchor_m2 = np.sum(np.square(correction), axis=1)
    bounded_visibility = np.clip(visibility, 0.05, 1.0)
    variance = (floor_m2 + anchor_m2[None]) / bounded_visibility
    return variance.astype(np.float32)


def validate_source_contract(
    source_dir: str | Path,
    *,
    config: MVTrackerSourceConfig | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Validate the original seal and the frozen open-27 camera plan."""

    cfg = config or MVTrackerSourceConfig()
    source = Path(source_dir).resolve()
    seal_path = source / "prediction_seal.json"
    plan_path = source / "open27_measurement_manifest.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    _require(
        seal.get("artifact_kind") == "Deform360IndependentSourcePredictionSeal",
        "unsupported physical prediction seal",
    )
    _require(
        seal.get("object_id") == OBJECT_ID and seal.get("episode_id") == EPISODE_ID,
        "physical prediction seal names a different source case",
    )
    boundary = seal.get("information_boundary", {})
    _require(
        boundary.get("object_observation_frames_used") == [0]
        and boundary.get("future_object_track_read") is False
        and boundary.get("prediction_hashed_before_future_outcome_scoring") is True,
        "physical prediction crossed its source boundary",
    )
    frozen_plan = plan.get("plan", {})
    _require(
        tuple(frozen_plan.get("center_ids", ())) == cfg.center_ids,
        "frozen center identities changed",
    )
    _require(
        tuple(frozen_plan.get("selected_cameras", ())) == cfg.selected_cameras,
        "frozen camera panel changed",
    )
    archive_path = _resolve_prediction_archive(source, seal)
    with np.load(archive_path, allow_pickle=False) as stored:
        physical = np.asarray(stored["prediction_m"]).copy()
        persistence = np.asarray(stored["persistence_m"]).copy()
        frame_zero = np.asarray(stored["frame_zero_points_m"]).copy()
    _require(
        physical.shape == persistence.shape
        and physical.ndim == 3
        and physical.shape[2] == 3,
        "physical trajectory shape is invalid",
    )
    _require(frame_zero.shape == physical.shape[1:], "frame-zero shape is invalid")
    _require(
        cfg.update_frame < len(physical) and max(cfg.center_ids) < physical.shape[1],
        "frozen query exceeds physical trajectory",
    )
    return seal, physical, persistence, frame_zero


def write_prediction_artifact(
    output_dir: str | Path,
    *,
    raw_tracker_m: np.ndarray,
    visibility_probability: np.ndarray,
    physical_prior_m: np.ndarray,
    persistence_m: np.ndarray,
    frame_zero_points_m: np.ndarray,
    input_provenance: Mapping[str, Any],
    runtime_provenance: Mapping[str, Any],
    config: MVTrackerSourceConfig | None = None,
) -> dict[str, Any]:
    """Write one target-free MVTracker prediction artifact."""

    cfg = config or MVTrackerSourceConfig()
    output = Path(output_dir).resolve()
    _require(not output.exists(), "prediction output already exists")
    raw = np.asarray(raw_tracker_m, dtype=np.float32)
    visibility = np.asarray(visibility_probability, dtype=np.float32)
    centers = np.asarray(cfg.center_ids, dtype=np.int64)
    initial = np.asarray(frame_zero_points_m, dtype=np.float32)[centers]
    _require(
        raw.shape == (cfg.prefix_frame_count, len(centers), 3),
        "tracker trajectory shape differs from protocol",
    )
    _require(
        visibility.shape == raw.shape[:2],
        "tracker visibility shape differs",
    )
    anchored, correction = exact_anchor_trajectory(raw, initial)
    variance_m2 = metric_observation_variance_m2(
        visibility,
        correction,
        standard_deviation_floor_m=cfg.observation_std_floor_m,
    )
    physical = np.asarray(physical_prior_m, dtype=np.float32)
    persistence = np.asarray(persistence_m, dtype=np.float32)
    _require(
        physical.shape == persistence.shape
        and cfg.prefix_frame_count <= len(physical),
        "baseline trajectories differ or are too short",
    )
    output.mkdir(parents=True)
    archive_path = output / PREDICTION_FILENAME
    np.savez_compressed(
        archive_path,
        raw_tracker_m=raw,
        anchored_tracker_m=anchored,
        visibility_probability=visibility,
        observation_variance_m2=variance_m2,
        frame_zero_anchor_correction_m=correction,
        center_ids=centers,
        physical_prior_centers_m=physical[: cfg.prefix_frame_count, centers],
        persistence_centers_m=persistence[: cfg.prefix_frame_count, centers],
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360MVTrackerPrivilegedDepthPrediction",
        "protocol_id": PROTOCOL_ID,
        "case": CASE_NAME,
        "object_id": OBJECT_ID,
        "episode_id": EPISODE_ID,
        "config": _config_payload(cfg),
        "tracker": {
            "name": "MVTracker",
            "repository_revision": MVTRACKER_REVISION,
            "checkpoint_sha256": MVTRACKER_CHECKPOINT_SHA256,
            **dict(runtime_provenance),
        },
        "inputs": dict(input_provenance),
        "diagnostics": {
            "raw_query_anchor_mean_m": float(
                np.mean(np.linalg.norm(correction, axis=1))
            ),
            "raw_query_anchor_max_m": float(
                np.max(np.linalg.norm(correction, axis=1))
            ),
            "visible_fraction": float(
                np.mean(visibility >= cfg.visibility_threshold)
            ),
            "observation_variance_m2_min": float(np.min(variance_m2)),
            "observation_variance_m2_max": float(np.max(variance_m2)),
        },
        "output": {
            "archive": str(archive_path),
            "archive_sha256": file_sha256(archive_path),
        },
        "information_boundary": {
            "source_target_read": False,
            "source_outcome_read": False,
            "maximum_rgb_frame_read": cfg.update_frame,
            "rendered_depth_indices_read": list(range(cfg.prefix_frame_count)),
            "rendered_depth_derived_from_full_sequence_splat": True,
            "deployable_predictive_observation": False,
            "exact_frame_zero_identity_preserved": True,
            "state_innovation_used_in_prior_reliability": False,
        },
        "claim_boundary": (
            "privileged rendered-depth competence control on an outcome-open "
            "source case; not a causal observation, Bayesian-PhysTwin gain, "
            "confirmation, or state-of-the-art result"
        ),
    }
    report["result_sha256"] = canonical_sha256(report)
    (output / REPORT_FILENAME).write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def seal_prediction(
    prediction_dir: str | Path,
    *,
    config: MVTrackerSourceConfig | None = None,
) -> dict[str, Any]:
    """Seal the one-case prediction before source target loading."""

    cfg = config or MVTrackerSourceConfig()
    root = Path(prediction_dir).resolve()
    report_path = root / REPORT_FILENAME
    archive_path = root / PREDICTION_FILENAME
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(
        report.get("result_sha256") == canonical_sha256(report),
        "prediction report checksum changed",
    )
    _require(
        report.get("protocol_id") == PROTOCOL_ID
        and report.get("config") == _config_payload(cfg),
        "prediction report differs from frozen protocol",
    )
    _require(
        report.get("information_boundary", {}).get("source_target_read") is False,
        "prediction report crossed source target boundary",
    )
    _require(
        report.get("output", {}).get("archive_sha256")
        == file_sha256(archive_path),
        "prediction archive checksum changed",
    )
    seal: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360MVTrackerPrivilegedDepthPredictionSeal",
        "protocol_id": PROTOCOL_ID,
        "case": CASE_NAME,
        "config": _config_payload(cfg),
        "prediction_report_sha256": file_sha256(report_path),
        "prediction_archive_sha256": file_sha256(archive_path),
        "information_boundary": {
            "prediction_hashed_before_source_target_loading": True,
            "privileged_depth_control": True,
        },
        "claim_boundary": (
            "one-case competence-control prediction seal; source target remains "
            "outside this operation"
        ),
    }
    seal["result_sha256"] = canonical_sha256(seal)
    seal_path = root / SEAL_FILENAME
    _require(not seal_path.exists(), "MVTracker prediction is already sealed")
    seal_path.write_text(
        json.dumps(seal, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return seal


def _identity_rmse_m(
    prediction_m: np.ndarray,
    target_m: np.ndarray,
    support: np.ndarray,
) -> float:
    squared = np.sum(np.square(prediction_m - target_m), axis=2)
    _require(np.any(support), "identity score has no supported samples")
    return float(np.sqrt(np.mean(squared[support])))


def score_competence_arrays(
    *,
    tracker_m: np.ndarray,
    tracker_visibility_probability: np.ndarray,
    physical_prior_centers_m: np.ndarray,
    persistence_centers_m: np.ndarray,
    target_centers_m: np.ndarray,
    target_visibility: np.ndarray,
    target_validity: np.ndarray,
    config: MVTrackerSourceConfig | None = None,
) -> dict[str, Any]:
    """Score the sealed direct tracker on common visible identity support."""

    cfg = config or MVTrackerSourceConfig()
    tracker = np.asarray(tracker_m, dtype=np.float64)
    tracker_visibility = np.asarray(
        tracker_visibility_probability, dtype=np.float64
    )
    physical = np.asarray(physical_prior_centers_m, dtype=np.float64)
    persistence = np.asarray(persistence_centers_m, dtype=np.float64)
    target = np.asarray(target_centers_m, dtype=np.float64)
    target_visible = np.asarray(target_visibility, dtype=bool)
    target_valid = np.asarray(target_validity, dtype=bool)
    expected_shape = (cfg.prefix_frame_count, len(cfg.center_ids), 3)
    _require(
        tracker.shape
        == physical.shape
        == persistence.shape
        == target.shape
        == expected_shape,
        "competence trajectories have different shapes",
    )
    _require(
        tracker_visibility.shape == expected_shape[:2]
        and target_visible.shape == expected_shape[:2]
        and target_valid.shape == expected_shape[:2],
        "competence support arrays have different shapes",
    )
    future = np.zeros(expected_shape[:2], dtype=bool)
    future[1:] = True
    eligible = (
        future
        & target_visible
        & target_valid
        & np.all(np.isfinite(target), axis=2)
    )
    supported = (
        eligible
        & (tracker_visibility >= cfg.visibility_threshold)
        & np.all(np.isfinite(tracker), axis=2)
    )
    eligible_count = int(np.sum(eligible))
    supported_count = int(np.sum(supported))
    _require(eligible_count > 0, "source target has no eligible prefix identities")
    supported_fraction = supported_count / eligible_count
    if supported_count == 0:
        return {
            "eligible_identity_frames": eligible_count,
            "supported_identity_frames": 0,
            "supported_fraction": 0.0,
            "scores": {
                "mvtracker_identity_rmse_m": None,
                "physical_prior_identity_rmse_m": None,
                "persistence_identity_rmse_m": None,
            },
            "relative_gain_over_best_baseline": None,
            "gates": {
                "supported_fraction": False,
                "relative_gain_over_best_baseline": False,
                "absolute_rmse": False,
            },
            "passed": False,
        }
    scores = {
        "mvtracker_identity_rmse_m": _identity_rmse_m(
            tracker, target, supported
        ),
        "physical_prior_identity_rmse_m": _identity_rmse_m(
            physical, target, supported
        ),
        "persistence_identity_rmse_m": _identity_rmse_m(
            persistence, target, supported
        ),
    }
    best_baseline = min(
        scores["physical_prior_identity_rmse_m"],
        scores["persistence_identity_rmse_m"],
    )
    relative_gain = 1.0 - scores["mvtracker_identity_rmse_m"] / best_baseline
    gates = {
        "supported_fraction": (
            supported_fraction >= cfg.minimum_supported_fraction
        ),
        "relative_gain_over_best_baseline": (
            relative_gain >= cfg.minimum_relative_gain_over_best_baseline
        ),
        "absolute_rmse": (
            scores["mvtracker_identity_rmse_m"] <= cfg.maximum_tracker_rmse_m
        ),
    }
    return {
        "eligible_identity_frames": eligible_count,
        "supported_identity_frames": supported_count,
        "supported_fraction": float(supported_fraction),
        "scores": scores,
        "relative_gain_over_best_baseline": float(relative_gain),
        "gates": gates,
        "passed": bool(all(gates.values())),
    }


def evaluate_prediction(
    prediction_dir: str | Path,
    source_case_dir: str | Path,
    output_path: str | Path,
    *,
    config: MVTrackerSourceConfig | None = None,
) -> dict[str, Any]:
    """Open the authorized source target only after validating the seal."""

    cfg = config or MVTrackerSourceConfig()
    prediction = Path(prediction_dir).resolve()
    source = Path(source_case_dir).resolve()
    seal_path = prediction / SEAL_FILENAME
    report_path = prediction / REPORT_FILENAME
    archive_path = prediction / PREDICTION_FILENAME
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    _require(
        seal.get("result_sha256") == canonical_sha256(seal),
        "MVTracker prediction seal checksum changed",
    )
    _require(
        seal.get("prediction_report_sha256") == file_sha256(report_path)
        and seal.get("prediction_archive_sha256") == file_sha256(archive_path),
        "sealed MVTracker prediction artifact changed",
    )
    original_seal_path = source / "prediction_seal.json"
    target_path = source / "target_data.pkl"
    outcome_path = source / "outcome.json"
    original_seal = json.loads(original_seal_path.read_text(encoding="utf-8"))
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    _validate_deform360_outcome_manifest(
        original_seal_path,
        target_path,
        original_seal,
        outcome,
    )
    with np.load(archive_path, allow_pickle=False) as stored:
        tracker = np.asarray(stored["anchored_tracker_m"]).copy()
        tracker_visibility = np.asarray(
            stored["visibility_probability"]
        ).copy()
        physical = np.asarray(stored["physical_prior_centers_m"]).copy()
        persistence = np.asarray(stored["persistence_centers_m"]).copy()
        centers = np.asarray(stored["center_ids"], dtype=np.int64)
    _require(
        np.array_equal(centers, np.asarray(cfg.center_ids, dtype=np.int64)),
        "sealed MVTracker centers changed",
    )
    target_data = _load_manifest_validated_target_pickle(target_path)
    target = np.asarray(target_data["object_points"])
    target_visibility = np.asarray(
        target_data["object_visibilities"], dtype=bool
    )
    target_validity = np.asarray(
        target_data["object_motions_valid"], dtype=bool
    )
    prefix = slice(0, cfg.prefix_frame_count)
    score = score_competence_arrays(
        tracker_m=tracker,
        tracker_visibility_probability=tracker_visibility,
        physical_prior_centers_m=physical,
        persistence_centers_m=persistence,
        target_centers_m=target[prefix][:, centers],
        target_visibility=target_visibility[prefix][:, centers],
        target_validity=target_validity[prefix][:, centers],
        config=cfg,
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360MVTrackerPrivilegedDepthSourceEvaluation",
        "protocol_id": PROTOCOL_ID,
        "case": CASE_NAME,
        "config": _config_payload(cfg),
        "competence": score,
        "decision": (
            "authorize-prefix-only-learned-depth-study"
            if score["passed"]
            else "stop-mvtracker-deform360-route"
        ),
        "inputs_sha256": {
            "mvtracker_prediction_seal": file_sha256(seal_path),
            "source_prediction_seal": file_sha256(original_seal_path),
            "source_outcome": file_sha256(outcome_path),
            "source_target": file_sha256(target_path),
        },
        "implementation_sha256": _evaluation_implementation_sha256(),
        "compatibility": {
            "target_pickle_module_aliases": dict(_NUMPY_PICKLE_MODULE_ALIASES),
            "target_bytes_modified": False,
        },
        "information_boundary": {
            "mvtracker_prediction_validated_before_source_target_loading": True,
            "already_open_source_target_used_for_scoring": True,
            "fresh_or_held_target_used": False,
            "privileged_depth_control": True,
        },
        "claim_boundary": (
            "outcome-open privileged-depth competence result; a pass only "
            "authorizes causal learned-depth development"
        ),
    }
    result["result_sha256"] = canonical_sha256(result)
    output = Path(output_path).resolve()
    _require(not output.exists(), "MVTracker evaluation output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result
