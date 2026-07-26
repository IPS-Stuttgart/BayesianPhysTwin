"""Prefix-only MVTracker competence control for released PhysTwin RGB-D data.

The control deliberately stops before simulator assimilation.  A benchmark
query position from one allowed prefix frame initializes MVTracker, while a
separately withheld slice of the same prefix is used only after the tracker
prediction has been sealed.  No frame at or after the released training
boundary is staged or read.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PROTOCOL_ID = "phystwin-mvtracker-prefix-competence-v1"
CASE_NAME = "single_lift_cloth"
QUERY_FILENAME = "query_input.npz"
WITHHELD_FILENAME = "withheld_prefix_target.npz"
SOURCE_REPORT_FILENAME = "source_artifact_report.json"
PREDICTION_FILENAME = "mvtracker_prediction.npz"
PREDICTION_REPORT_FILENAME = "mvtracker_prediction_report.json"
PREDICTION_SEAL_FILENAME = "mvtracker_prediction_seal.json"
EVALUATION_FILENAME = "mvtracker_prefix_evaluation.json"
MVTRACKER_REVISION = "ceea8ad2af77ed9b44148ef8e9eeba4ea3c3f072"
MVTRACKER_CHECKPOINT_SHA256 = (
    "a7fa86f2a7223e3e0aa4c1d3eff0dec5fe8a9227a48572ce943b8e49d8a4f8e6"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    """Hash an array's dtype, shape, and contiguous bytes."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a JSON payload while excluding its self-hash."""

    unsigned = dict(payload)
    unsigned.pop("result_sha256", None)
    encoded = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PhysTwinMVTrackerCompetenceConfig:
    """Frozen choices for the one-case source competence control."""

    case_name: str = CASE_NAME
    source_frame_start: int = 90
    source_frame_end_exclusive: int = 121
    selected_identity_ids: tuple[int, ...] = (3, 4, 6, 8)
    selected_cameras: tuple[int, ...] = (0, 1, 2)
    depth_scale_to_m: float = 0.001
    visibility_threshold: float = 0.5
    observation_std_floor_m: float = 0.005
    normalization_target_camera_radius: float = 6.3
    minimum_supported_fraction: float = 0.75
    minimum_relative_gain_over_persistence: float = 0.10
    maximum_identity_rmse_m: float = 0.015
    maximum_endpoint_rmse_m: float = 0.015
    endpoint_frame_count: int = 6

    def __post_init__(self) -> None:
        _require(self.case_name == CASE_NAME, "source case is not frozen")
        _require(self.source_frame_start >= 0, "source start must be nonnegative")
        _require(
            self.source_frame_end_exclusive > self.source_frame_start + 1,
            "source interval is too short",
        )
        _require(
            len(self.selected_identity_ids) >= 3
            and len(set(self.selected_identity_ids))
            == len(self.selected_identity_ids),
            "selected identities must be unique and contain at least three points",
        )
        _require(
            all(identity >= 0 for identity in self.selected_identity_ids),
            "identity indices must be nonnegative",
        )
        _require(
            len(self.selected_cameras) >= 2
            and len(set(self.selected_cameras)) == len(self.selected_cameras),
            "selected cameras must be unique and multiview",
        )
        _require(
            all(camera >= 0 for camera in self.selected_cameras),
            "camera indices must be nonnegative",
        )
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
            self.normalization_target_camera_radius > 0.0,
            "normalization radius must be positive",
        )
        _require(
            0.0 < self.minimum_supported_fraction <= 1.0,
            "support threshold must lie in (0, 1]",
        )
        _require(
            0.0 <= self.minimum_relative_gain_over_persistence < 1.0,
            "gain threshold must lie in [0, 1)",
        )
        _require(
            self.maximum_identity_rmse_m > 0.0
            and self.maximum_endpoint_rmse_m > 0.0,
            "RMSE thresholds must be positive",
        )
        _require(
            1 <= self.endpoint_frame_count < self.prefix_frame_count,
            "endpoint interval is invalid",
        )

    @property
    def prefix_frame_count(self) -> int:
        return self.source_frame_end_exclusive - self.source_frame_start


def _load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def prepare_source_artifacts(
    manual_tracks_path: str | Path,
    split_path: str | Path,
    output_dir: str | Path,
    *,
    config: PhysTwinMVTrackerCompetenceConfig | None = None,
) -> dict[str, Any]:
    """Stage one query frame and a separate prefix-only evaluation target."""

    cfg = config or PhysTwinMVTrackerCompetenceConfig()
    output = Path(output_dir).resolve()
    _require(not output.exists(), "source artifact output already exists")
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    train_end = int(split["train"][1])
    _require(
        cfg.source_frame_end_exclusive <= train_end,
        "source interval crosses the released training boundary",
    )
    tracks = np.asarray(_load_pickle(manual_tracks_path), dtype=np.float64)
    _require(
        tracks.ndim == 3 and tracks.shape[2] == 3,
        "manual tracks must have shape (T, N, 3)",
    )
    _require(
        cfg.source_frame_end_exclusive <= len(tracks),
        "manual tracks are shorter than the source interval",
    )
    selected_ids = np.asarray(cfg.selected_identity_ids, dtype=np.int64)
    _require(
        int(np.max(selected_ids)) < tracks.shape[1],
        "selected identity exceeds the manual track array",
    )
    withheld = tracks[
        cfg.source_frame_start : cfg.source_frame_end_exclusive,
        selected_ids,
    ].copy()
    query = withheld[0].copy()
    _require(
        np.all(np.isfinite(query)),
        "every frozen query identity must be finite at the query frame",
    )
    identity_ids = selected_ids

    input_dir = output / "prediction_input"
    withheld_dir = output / "withheld_evaluation"
    input_dir.mkdir(parents=True)
    withheld_dir.mkdir()
    query_path = input_dir / QUERY_FILENAME
    withheld_path = withheld_dir / WITHHELD_FILENAME
    np.savez_compressed(
        query_path,
        query_points_world_m=query.astype(np.float32),
        identity_ids=identity_ids,
        source_frame=np.asarray(cfg.source_frame_start, dtype=np.int64),
        train_end_frame_exclusive=np.asarray(train_end, dtype=np.int64),
    )
    np.savez_compressed(
        withheld_path,
        target_tracks_world_m=withheld.astype(np.float32),
        identity_ids=identity_ids,
        source_frame_start=np.asarray(cfg.source_frame_start, dtype=np.int64),
        source_frame_end_exclusive=np.asarray(
            cfg.source_frame_end_exclusive,
            dtype=np.int64,
        ),
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinMVTrackerCompetenceSourceArtifacts",
        "protocol_id": PROTOCOL_ID,
        "case": CASE_NAME,
        "config": asdict(cfg),
        "inputs": {
            "manual_tracks_sha256": file_sha256(manual_tracks_path),
            "split_sha256": file_sha256(split_path),
            "released_train_end_frame_exclusive": train_end,
        },
        "prediction_input": {
            "path": str(query_path),
            "sha256": file_sha256(query_path),
            "query_count": len(query),
            "query_array_sha256": array_sha256(query.astype(np.float32)),
        },
        "withheld_evaluation": {
            "path": str(withheld_path),
            "sha256": file_sha256(withheld_path),
            "frame_count": len(withheld),
            "target_array_sha256": array_sha256(withheld.astype(np.float32)),
        },
        "information_boundary": {
            "manual_source_file_loaded_during_staging": True,
            "prediction_input_retains_only_source_frame": cfg.source_frame_start,
            "withheld_numeric_frame_range_half_open": [
                cfg.source_frame_start,
                cfg.source_frame_end_exclusive,
            ],
            "frame_at_or_after_train_end_retained": False,
            "withheld_artifact_available_to_prediction": False,
        },
    }
    report["result_sha256"] = canonical_sha256(report)
    report_path = output / SOURCE_REPORT_FILENAME
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def validate_query_input(
    query_path: str | Path,
    expected_sha256: str,
    *,
    config: PhysTwinMVTrackerCompetenceConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and load the prediction-visible frame-zero query artifact."""

    cfg = config or PhysTwinMVTrackerCompetenceConfig()
    path = Path(query_path).resolve()
    _require(file_sha256(path) == expected_sha256, "query artifact hash changed")
    with np.load(path, allow_pickle=False) as stored:
        query = np.asarray(stored["query_points_world_m"], dtype=np.float32)
        identity_ids = np.asarray(stored["identity_ids"], dtype=np.int64)
        source_frame = int(stored["source_frame"])
        train_end = int(stored["train_end_frame_exclusive"])
    _require(
        query.ndim == 2 and query.shape[1] == 3,
        "query points must have shape (N, 3)",
    )
    _require(
        identity_ids.shape == (len(query),)
        and np.array_equal(
            identity_ids,
            np.asarray(cfg.selected_identity_ids, dtype=np.int64),
        ),
        "query identity IDs are invalid",
    )
    _require(np.all(np.isfinite(query)), "query points are not finite")
    _require(
        source_frame == cfg.source_frame_start,
        "query frame differs from the frozen source frame",
    )
    _require(
        cfg.source_frame_end_exclusive <= train_end,
        "query artifact crosses the training boundary",
    )
    return query, identity_ids


def exact_anchor_trajectory(
    raw_trajectory_m: np.ndarray,
    query_points_world_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Preserve predicted displacement while making the query frame exact."""

    raw = np.asarray(raw_trajectory_m, dtype=np.float64)
    query = np.asarray(query_points_world_m, dtype=np.float64)
    _require(
        raw.ndim == 3 and raw.shape[2] == 3,
        "tracker trajectory must have shape (T, N, 3)",
    )
    _require(query.shape == raw.shape[1:], "query shape differs from trajectory")
    _require(
        np.all(np.isfinite(raw)) and np.all(np.isfinite(query)),
        "tracker trajectory and query must be finite",
    )
    correction = query - raw[0]
    anchored = raw + correction[None]
    _require(np.array_equal(anchored[0], query), "exact query anchoring failed")
    return anchored.astype(np.float32), correction.astype(np.float32)


def metric_observation_variance_m2(
    visibility_probability: np.ndarray,
    anchor_correction_m: np.ndarray,
    *,
    standard_deviation_floor_m: float,
) -> np.ndarray:
    """Build residual-independent metric variance for a later Bayesian update."""

    visibility = np.asarray(visibility_probability, dtype=np.float64)
    correction = np.asarray(anchor_correction_m, dtype=np.float64)
    _require(visibility.ndim == 2, "visibility must have shape (T, N)")
    _require(
        correction.shape == (visibility.shape[1], 3),
        "anchor correction shape differs",
    )
    _require(
        np.all(np.isfinite(visibility))
        and np.all((visibility >= 0.0) & (visibility <= 1.0)),
        "visibility must be finite and lie in [0, 1]",
    )
    _require(standard_deviation_floor_m > 0.0, "variance floor must be positive")
    floor_m2 = standard_deviation_floor_m**2
    anchor_m2 = np.sum(np.square(correction), axis=1)
    variance = (floor_m2 + anchor_m2[None]) / np.clip(
        visibility,
        0.05,
        1.0,
    )
    return variance.astype(np.float32)


def write_prediction_artifact(
    output_dir: str | Path,
    *,
    raw_tracker_m: np.ndarray,
    visibility_probability: np.ndarray,
    query_points_world_m: np.ndarray,
    identity_ids: np.ndarray,
    input_provenance: Mapping[str, Any],
    runtime_provenance: Mapping[str, Any],
    implementation_sha256: Mapping[str, str],
    config: PhysTwinMVTrackerCompetenceConfig | None = None,
) -> dict[str, Any]:
    """Write one target-free tracker prediction before competence scoring."""

    cfg = config or PhysTwinMVTrackerCompetenceConfig()
    output = Path(output_dir).resolve()
    _require(not output.exists(), "prediction output already exists")
    raw = np.asarray(raw_tracker_m, dtype=np.float32)
    visibility = np.asarray(visibility_probability, dtype=np.float32)
    query = np.asarray(query_points_world_m, dtype=np.float32)
    ids = np.asarray(identity_ids, dtype=np.int64)
    _require(
        raw.shape == (cfg.prefix_frame_count, len(query), 3),
        "tracker trajectory shape differs from the protocol",
    )
    _require(visibility.shape == raw.shape[:2], "visibility shape differs")
    _require(
        ids.shape == (len(query),)
        and np.array_equal(
            ids,
            np.asarray(cfg.selected_identity_ids, dtype=np.int64),
        ),
        "identity IDs differ from the query contract",
    )
    anchored, correction = exact_anchor_trajectory(raw, query)
    variance = metric_observation_variance_m2(
        visibility,
        correction,
        standard_deviation_floor_m=cfg.observation_std_floor_m,
    )

    output.mkdir(parents=True)
    archive_path = output / PREDICTION_FILENAME
    np.savez_compressed(
        archive_path,
        raw_tracker_m=raw,
        anchored_tracker_m=anchored,
        visibility_probability=visibility,
        observation_variance_m2=variance,
        frame_zero_anchor_correction_m=correction,
        identity_ids=ids,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinMVTrackerPrefixPrediction",
        "protocol_id": PROTOCOL_ID,
        "case": CASE_NAME,
        "config": asdict(cfg),
        "tracker": {
            "name": "MVTracker",
            "repository_revision": MVTRACKER_REVISION,
            "checkpoint_sha256": MVTRACKER_CHECKPOINT_SHA256,
            **dict(runtime_provenance),
        },
        "inputs": dict(input_provenance),
        "implementation_sha256": dict(implementation_sha256),
        "diagnostics": {
            "query_count": len(query),
            "raw_query_anchor_mean_m": float(
                np.mean(np.linalg.norm(correction, axis=1))
            ),
            "raw_query_anchor_max_m": float(
                np.max(np.linalg.norm(correction, axis=1))
            ),
            "visible_fraction": float(
                np.mean(visibility >= cfg.visibility_threshold)
            ),
            "observation_variance_m2_min": float(np.min(variance)),
            "observation_variance_m2_max": float(np.max(variance)),
        },
        "output": {
            "archive": str(archive_path),
            "archive_sha256": file_sha256(archive_path),
        },
        "information_boundary": {
            "rgb_depth_frame_range_half_open": [
                cfg.source_frame_start,
                cfg.source_frame_end_exclusive,
            ],
            "query_frame": cfg.source_frame_start,
            "withheld_prefix_target_read": False,
            "manual_track_file_read": False,
            "frame_at_or_after_train_end_read": False,
            "state_innovation_used_in_prior_reliability": False,
        },
        "claim_boundary": (
            "target-free tracker prediction for one already-open source competence "
            "control; not simulator assimilation, confirmation, calibration, or "
            "state-of-the-art evidence"
        ),
    }
    report["result_sha256"] = canonical_sha256(report)
    report_path = output / PREDICTION_REPORT_FILENAME
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def seal_prediction(
    prediction_dir: str | Path,
) -> dict[str, Any]:
    """Seal a prediction before the withheld prefix target is opened."""

    prediction = Path(prediction_dir).resolve()
    report_path = prediction / PREDICTION_REPORT_FILENAME
    archive_path = prediction / PREDICTION_FILENAME
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(
        report.get("artifact_kind") == "PhysTwinMVTrackerPrefixPrediction",
        "prediction report kind is invalid",
    )
    _require(
        report.get("result_sha256") == canonical_sha256(report),
        "prediction report self-hash changed",
    )
    archive_sha = file_sha256(archive_path)
    _require(
        report.get("output", {}).get("archive_sha256") == archive_sha,
        "prediction archive hash differs from its report",
    )
    seal: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinMVTrackerPrefixPredictionSeal",
        "protocol_id": PROTOCOL_ID,
        "case": CASE_NAME,
        "prediction_report_sha256": file_sha256(report_path),
        "prediction_archive_sha256": archive_sha,
        "prediction_result_sha256": report["result_sha256"],
        "information_boundary": {
            "prediction_hashed_before_withheld_prefix_scoring": True,
            "future_outcome_scoring_authorized": False,
        },
    }
    seal["result_sha256"] = canonical_sha256(seal)
    seal_path = prediction / PREDICTION_SEAL_FILENAME
    _require(not seal_path.exists(), "prediction seal already exists")
    seal_path.write_text(
        json.dumps(seal, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return seal


def _radial_rmse_m(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> float:
    error = np.asarray(prediction, dtype=float) - np.asarray(target, dtype=float)
    selected = error[np.asarray(mask, dtype=bool)]
    _require(len(selected) > 0, "RMSE requires supported point-frames")
    return float(np.sqrt(np.mean(np.sum(np.square(selected), axis=1))))


def evaluate_competence(
    prediction_dir: str | Path,
    withheld_prefix_path: str | Path,
    expected_withheld_sha256: str,
    output_path: str | Path,
    *,
    config: PhysTwinMVTrackerCompetenceConfig | None = None,
) -> dict[str, Any]:
    """Open only the staged prefix target and evaluate the frozen tracker."""

    cfg = config or PhysTwinMVTrackerCompetenceConfig()
    prediction = Path(prediction_dir).resolve()
    seal_path = prediction / PREDICTION_SEAL_FILENAME
    report_path = prediction / PREDICTION_REPORT_FILENAME
    archive_path = prediction / PREDICTION_FILENAME
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    _require(
        seal.get("artifact_kind") == "PhysTwinMVTrackerPrefixPredictionSeal",
        "prediction seal kind is invalid",
    )
    _require(
        seal.get("result_sha256") == canonical_sha256(seal),
        "prediction seal self-hash changed",
    )
    _require(
        seal.get("prediction_report_sha256") == file_sha256(report_path)
        and seal.get("prediction_archive_sha256") == file_sha256(archive_path),
        "sealed prediction files changed",
    )
    withheld_path = Path(withheld_prefix_path).resolve()
    _require(
        file_sha256(withheld_path) == expected_withheld_sha256,
        "withheld prefix target hash changed",
    )
    with np.load(archive_path, allow_pickle=False) as stored:
        candidate = np.asarray(stored["anchored_tracker_m"], dtype=np.float32)
        visibility = np.asarray(
            stored["visibility_probability"],
            dtype=np.float32,
        )
        identity_ids = np.asarray(stored["identity_ids"], dtype=np.int64)
    with np.load(withheld_path, allow_pickle=False) as stored:
        target = np.asarray(stored["target_tracks_world_m"], dtype=np.float32)
        target_ids = np.asarray(stored["identity_ids"], dtype=np.int64)
        source_start = int(stored["source_frame_start"])
        source_end = int(stored["source_frame_end_exclusive"])
    _require(
        candidate.shape == target.shape
        == (cfg.prefix_frame_count, len(identity_ids), 3),
        "candidate and withheld target shapes differ",
    )
    _require(
        np.array_equal(identity_ids, target_ids),
        "prediction and target identity IDs differ",
    )
    _require(
        source_start == cfg.source_frame_start
        and source_end == cfg.source_frame_end_exclusive,
        "withheld source interval differs from the protocol",
    )
    target_valid = np.all(np.isfinite(target), axis=2)
    supported = (
        target_valid
        & np.all(np.isfinite(candidate), axis=2)
        & (visibility >= cfg.visibility_threshold)
    )
    score_rows = np.arange(cfg.prefix_frame_count) > 0
    scored_target = target_valid & score_rows[:, None]
    scored_supported = supported & score_rows[:, None]
    supported_fraction = float(
        np.sum(scored_supported) / max(np.sum(scored_target), 1)
    )
    persistence = np.repeat(target[:1], cfg.prefix_frame_count, axis=0)
    candidate_rmse = _radial_rmse_m(candidate, target, scored_supported)
    persistence_rmse = _radial_rmse_m(
        persistence,
        target,
        scored_supported,
    )
    relative_gain = (
        (persistence_rmse - candidate_rmse) / persistence_rmse
        if persistence_rmse > 0.0
        else -1.0
    )
    endpoint_rows = np.arange(cfg.prefix_frame_count) >= (
        cfg.prefix_frame_count - cfg.endpoint_frame_count
    )
    endpoint_supported = supported & endpoint_rows[:, None]
    endpoint_rmse = _radial_rmse_m(
        candidate,
        target,
        endpoint_supported,
    )
    gates = {
        "supported_fraction": (
            supported_fraction >= cfg.minimum_supported_fraction
        ),
        "relative_gain_over_persistence": (
            relative_gain >= cfg.minimum_relative_gain_over_persistence
        ),
        "identity_rmse": candidate_rmse <= cfg.maximum_identity_rmse_m,
        "endpoint_rmse": endpoint_rmse <= cfg.maximum_endpoint_rmse_m,
    }
    passed = all(gates.values())
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinMVTrackerPrefixCompetenceResult",
        "protocol_id": PROTOCOL_ID,
        "case": CASE_NAME,
        "config": asdict(cfg),
        "metrics": {
            "supported_fraction": supported_fraction,
            "supported_point_frame_count": int(np.sum(scored_supported)),
            "eligible_point_frame_count": int(np.sum(scored_target)),
            "candidate_identity_rmse_m": candidate_rmse,
            "persistence_identity_rmse_m": persistence_rmse,
            "relative_gain_over_persistence": relative_gain,
            "candidate_endpoint_rmse_m": endpoint_rmse,
        },
        "gates": gates,
        "competence_gate_passed": passed,
        "decision": (
            "advance-to-separately-locked-assimilation-smoke"
            if passed
            else "stop-mvtracker-phystwin-route"
        ),
        "inputs": {
            "prediction_seal_sha256": file_sha256(seal_path),
            "withheld_prefix_sha256": expected_withheld_sha256,
        },
        "information_boundary": {
            "prediction_sealed_before_target_open": True,
            "scored_numeric_frame_range_half_open": [
                cfg.source_frame_start,
                cfg.source_frame_end_exclusive,
            ],
            "frame_at_or_after_train_end_scored": False,
            "future_simulator_outcome_read": False,
        },
        "claim_boundary": (
            "one-case prefix-only competence control on an already-open source "
            "interaction; not a Bayesian-PhysTwin gain or state-of-the-art result"
        ),
    }
    result["result_sha256"] = canonical_sha256(result)
    output = Path(output_path).resolve()
    _require(not output.exists(), "evaluation output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


__all__ = [
    "CASE_NAME",
    "EVALUATION_FILENAME",
    "MVTRACKER_CHECKPOINT_SHA256",
    "MVTRACKER_REVISION",
    "PREDICTION_FILENAME",
    "PREDICTION_REPORT_FILENAME",
    "PREDICTION_SEAL_FILENAME",
    "PROTOCOL_ID",
    "PhysTwinMVTrackerCompetenceConfig",
    "array_sha256",
    "canonical_sha256",
    "evaluate_competence",
    "exact_anchor_trajectory",
    "file_sha256",
    "metric_observation_variance_m2",
    "prepare_source_artifacts",
    "seal_prediction",
    "validate_query_input",
    "write_prediction_artifact",
]
