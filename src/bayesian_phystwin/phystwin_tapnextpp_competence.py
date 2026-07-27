"""Sealed source competence control for causal TAPNext++ observations.

Only a query frame, calibrated causal RGB-D frames, and prefix object masks are
available to prediction.  The manual material trajectories over the same
allowed prefix are staged separately and cannot be opened until the tracker
prediction has been sealed.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .phystwin_mvtracker_competence import (
    array_sha256,
    canonical_sha256,
    exact_anchor_trajectory,
    file_sha256,
)
from .tapnextpp_multiview import TAPNextPPMultiviewConfig

PROTOCOL_ID = "phystwin-tapnextpp-prefix-competence-v1"
CASE_NAME = "single_lift_cloth"
QUERY_FILENAME = "prediction_input.npz"
WITHHELD_FILENAME = "withheld_prefix_target.npz"
SOURCE_REPORT_FILENAME = "source_artifact_report.json"
PREDICTION_FILENAME = "tapnextpp_prediction.npz"
PREDICTION_REPORT_FILENAME = "tapnextpp_prediction_report.json"
PREDICTION_SEAL_FILENAME = "tapnextpp_prediction_seal.json"
EVALUATION_FILENAME = "tapnextpp_prefix_evaluation.json"
TAPNEXTPP_REVISION = "c2cbab81cc06092b5f05bfe2da7bfec54e2079c9"
TAPNEXTPP_CHECKPOINT_SHA256 = (
    "6cd0e793fdcface3063d63f8ed3819bcf74c2c0468fe1fef85acee4de2f3609f"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class PhysTwinTAPNextPPCompetenceConfig:
    """Frozen one-case source competence choices."""

    case_name: str = CASE_NAME
    source_frame_start: int = 68
    source_frame_end_exclusive: int = 88
    selected_identity_ids: tuple[int, ...] = (3, 4, 6, 8)
    selected_cameras: tuple[int, ...] = (0, 1, 2)
    depth_scale_to_m: float = 0.001
    input_resolution: int = 512
    support_points_per_query: int = 64
    support_radius_model_px: float = 32.0
    visibility_threshold: float = 0.5
    maximum_reprojection_error_px: float = 3.0
    maximum_depth_residual_m: float = 0.03
    minimum_triangulation_views: int = 2
    mask_patch_radius_px: int = 2
    minimum_object_mask_fraction: float = 0.20
    pixel_standard_deviation_px: float = 1.5
    shared_bias_standard_deviation_m: float = 0.005
    two_view_covariance_inflation: float = 4.0
    minimum_supported_fraction: float = 0.75
    minimum_relative_gain_over_persistence: float = 0.10
    maximum_identity_rmse_m: float = 0.015
    maximum_endpoint_rmse_m: float = 0.015
    endpoint_frame_count: int = 5

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
            "selected identities must be unique and contain at least three",
        )
        _require(
            all(identity >= 0 for identity in self.selected_identity_ids),
            "identity indices must be nonnegative",
        )
        _require(
            len(self.selected_cameras) >= self.minimum_triangulation_views
            and len(set(self.selected_cameras)) == len(self.selected_cameras),
            "selected cameras must satisfy the multiview contract",
        )
        _require(
            all(camera >= 0 for camera in self.selected_cameras),
            "camera indices must be nonnegative",
        )
        _require(self.depth_scale_to_m > 0.0, "depth scale must be positive")
        _require(
            self.input_resolution > 0 and self.input_resolution % 16 == 0,
            "input resolution must be a positive patch multiple",
        )
        _require(
            self.support_points_per_query >= 0,
            "support-point count must be nonnegative",
        )
        _require(
            self.support_radius_model_px > 0.0,
            "support radius must be positive",
        )
        _require(
            0.0 < self.minimum_supported_fraction <= 1.0,
            "support gate must lie in (0, 1]",
        )
        _require(
            0.0 <= self.minimum_relative_gain_over_persistence < 1.0,
            "gain gate must lie in [0, 1)",
        )
        _require(
            self.maximum_identity_rmse_m > 0.0
            and self.maximum_endpoint_rmse_m > 0.0,
            "RMSE gates must be positive",
        )
        _require(
            1 <= self.endpoint_frame_count < self.prefix_frame_count,
            "endpoint interval is invalid",
        )
        _ = self.multiview_config

    @property
    def prefix_frame_count(self) -> int:
        return self.source_frame_end_exclusive - self.source_frame_start

    @property
    def multiview_config(self) -> TAPNextPPMultiviewConfig:
        return TAPNextPPMultiviewConfig(
            visibility_threshold=self.visibility_threshold,
            maximum_reprojection_error_px=(
                self.maximum_reprojection_error_px
            ),
            maximum_depth_residual_m=self.maximum_depth_residual_m,
            minimum_view_count=self.minimum_triangulation_views,
            mask_patch_radius_px=self.mask_patch_radius_px,
            minimum_object_mask_fraction=(
                self.minimum_object_mask_fraction
            ),
            pixel_standard_deviation_px=self.pixel_standard_deviation_px,
            shared_bias_standard_deviation_m=(
                self.shared_bias_standard_deviation_m
            ),
            two_view_covariance_inflation=(
                self.two_view_covariance_inflation
            ),
        )


def _load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def _extract_prefix_object_masks(
    masks: Mapping[int, Mapping[int, Mapping[str, Any]]],
    config: PhysTwinTAPNextPPCompetenceConfig,
) -> np.ndarray:
    camera_sequences = []
    for camera in config.selected_cameras:
        frames = []
        for frame in range(
            config.source_frame_start,
            config.source_frame_end_exclusive,
        ):
            _require(frame in masks, f"object masks omit frame {frame}")
            _require(camera in masks[frame], f"object masks omit camera {camera}")
            frame_mask = np.asarray(
                masks[frame][camera]["object"],
                dtype=bool,
            )
            _require(
                frame_mask.ndim == 2,
                "object masks must be two-dimensional",
            )
            frames.append(frame_mask)
        camera_sequences.append(np.stack(frames))
    output = np.stack(camera_sequences)
    _require(
        output.shape[:2]
        == (len(config.selected_cameras), config.prefix_frame_count),
        "prefix mask shape differs from the protocol",
    )
    return output


def prepare_source_artifacts(
    manual_tracks_path: str | Path,
    split_path: str | Path,
    processed_masks_path: str | Path,
    output_dir: str | Path,
    *,
    config: PhysTwinTAPNextPPCompetenceConfig | None = None,
) -> dict[str, Any]:
    """Stage causal prediction inputs separately from manual prefix targets."""

    cfg = config or PhysTwinTAPNextPPCompetenceConfig()
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
    prefix_masks = _extract_prefix_object_masks(
        _load_pickle(processed_masks_path),
        cfg,
    )

    input_dir = output / "prediction_input"
    withheld_dir = output / "withheld_evaluation"
    input_dir.mkdir(parents=True)
    withheld_dir.mkdir()
    query_path = input_dir / QUERY_FILENAME
    withheld_path = withheld_dir / WITHHELD_FILENAME
    np.savez_compressed(
        query_path,
        query_points_world_m=query.astype(np.float32),
        identity_ids=selected_ids,
        object_masks=prefix_masks,
        selected_cameras=np.asarray(cfg.selected_cameras, dtype=np.int64),
        source_frame=np.asarray(cfg.source_frame_start, dtype=np.int64),
        train_end_frame_exclusive=np.asarray(train_end, dtype=np.int64),
    )
    np.savez_compressed(
        withheld_path,
        target_tracks_world_m=withheld.astype(np.float32),
        identity_ids=selected_ids,
        source_frame_start=np.asarray(cfg.source_frame_start, dtype=np.int64),
        source_frame_end_exclusive=np.asarray(
            cfg.source_frame_end_exclusive,
            dtype=np.int64,
        ),
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinTAPNextPPCompetenceSourceArtifacts",
        "protocol_id": PROTOCOL_ID,
        "case": CASE_NAME,
        "config": asdict(cfg),
        "inputs": {
            "manual_tracks_sha256": file_sha256(manual_tracks_path),
            "split_sha256": file_sha256(split_path),
            "processed_masks_sha256": file_sha256(processed_masks_path),
            "released_train_end_frame_exclusive": train_end,
        },
        "prediction_input": {
            "path": str(query_path),
            "sha256": file_sha256(query_path),
            "query_count": len(query),
            "query_array_sha256": array_sha256(query.astype(np.float32)),
            "object_mask_array_sha256": array_sha256(prefix_masks),
        },
        "withheld_evaluation": {
            "path": str(withheld_path),
            "sha256": file_sha256(withheld_path),
            "frame_count": len(withheld),
            "target_array_sha256": array_sha256(withheld.astype(np.float32)),
        },
        "information_boundary": {
            "source_files_loaded_during_staging": [
                "manual material tracks",
                "processed object masks",
            ],
            "prediction_input_retains_manual_rows": [
                cfg.source_frame_start,
            ],
            "prediction_input_retains_mask_frame_range_half_open": [
                cfg.source_frame_start,
                cfg.source_frame_end_exclusive,
            ],
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


def validate_prediction_input(
    query_path: str | Path,
    expected_sha256: str,
    *,
    config: PhysTwinTAPNextPPCompetenceConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and load the prediction-visible query and prefix masks."""

    cfg = config or PhysTwinTAPNextPPCompetenceConfig()
    path = Path(query_path).resolve()
    _require(file_sha256(path) == expected_sha256, "query artifact hash changed")
    with np.load(path, allow_pickle=False) as stored:
        query = np.asarray(stored["query_points_world_m"], dtype=np.float32)
        identity_ids = np.asarray(stored["identity_ids"], dtype=np.int64)
        object_masks = np.asarray(stored["object_masks"], dtype=bool)
        selected_cameras = np.asarray(
            stored["selected_cameras"],
            dtype=np.int64,
        )
        source_frame = int(stored["source_frame"])
        train_end = int(stored["train_end_frame_exclusive"])
    _require(
        query.shape == (len(cfg.selected_identity_ids), 3),
        "query points have the wrong shape",
    )
    _require(
        np.array_equal(
            identity_ids,
            np.asarray(cfg.selected_identity_ids, dtype=np.int64),
        ),
        "query identity IDs are invalid",
    )
    _require(
        np.array_equal(
            selected_cameras,
            np.asarray(cfg.selected_cameras, dtype=np.int64),
        ),
        "selected cameras differ from the protocol",
    )
    _require(
        object_masks.ndim == 4
        and object_masks.shape[:2]
        == (len(cfg.selected_cameras), cfg.prefix_frame_count),
        "prefix object masks have the wrong shape",
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
    return query, identity_ids, object_masks


def write_prediction_artifact(
    output_dir: str | Path,
    *,
    raw_tracker_m: np.ndarray,
    accepted_support: np.ndarray,
    observation_reliability: np.ndarray,
    observation_covariance_m2: np.ndarray,
    support_view_count: np.ndarray,
    reprojection_rmse_px: np.ndarray,
    depth_residual_rmse_m: np.ndarray,
    per_camera_tracks_xy: np.ndarray,
    per_camera_visibility_probability: np.ndarray,
    query_points_world_m: np.ndarray,
    identity_ids: np.ndarray,
    input_provenance: Mapping[str, Any],
    runtime_provenance: Mapping[str, Any],
    implementation_sha256: Mapping[str, str],
    config: PhysTwinTAPNextPPCompetenceConfig | None = None,
) -> dict[str, Any]:
    """Write a target-free causal tracker prediction before manual scoring."""

    cfg = config or PhysTwinTAPNextPPCompetenceConfig()
    output = Path(output_dir).resolve()
    _require(not output.exists(), "prediction output already exists")
    raw = np.asarray(raw_tracker_m, dtype=np.float32)
    support = np.asarray(accepted_support, dtype=bool)
    reliability = np.asarray(observation_reliability, dtype=np.float32)
    covariance = np.asarray(observation_covariance_m2, dtype=np.float32)
    view_count = np.asarray(support_view_count, dtype=np.int16)
    reprojection = np.asarray(reprojection_rmse_px, dtype=np.float32)
    depth_residual = np.asarray(depth_residual_rmse_m, dtype=np.float32)
    camera_tracks = np.asarray(per_camera_tracks_xy, dtype=np.float32)
    camera_visibility = np.asarray(
        per_camera_visibility_probability,
        dtype=np.float32,
    )
    query = np.asarray(query_points_world_m, dtype=np.float32)
    ids = np.asarray(identity_ids, dtype=np.int64)
    expected_shape = (cfg.prefix_frame_count, len(query))
    _require(
        raw.shape == (*expected_shape, 3),
        "tracker trajectory shape differs from the protocol",
    )
    _require(support.shape == expected_shape, "support shape differs")
    _require(reliability.shape == expected_shape, "reliability shape differs")
    _require(
        covariance.shape == (*expected_shape, 3, 3),
        "metric covariance shape differs",
    )
    _require(view_count.shape == expected_shape, "view-count shape differs")
    _require(reprojection.shape == expected_shape, "reprojection shape differs")
    _require(
        depth_residual.shape == expected_shape,
        "depth-residual shape differs",
    )
    _require(
        camera_tracks.shape
        == (
            len(cfg.selected_cameras),
            cfg.prefix_frame_count,
            len(query),
            2,
        ),
        "per-camera track shape differs",
    )
    _require(
        camera_visibility.shape == camera_tracks.shape[:3],
        "per-camera visibility shape differs",
    )
    _require(
        ids.shape == (len(query),)
        and np.array_equal(
            ids,
            np.asarray(cfg.selected_identity_ids, dtype=np.int64),
        ),
        "identity IDs differ from the query contract",
    )
    _require(
        np.all(np.isfinite(raw))
        and np.all(np.isfinite(reliability))
        and np.all((reliability >= 0.0) & (reliability <= 1.0)),
        "prediction arrays contain invalid values",
    )
    _require(
        np.all(np.isfinite(covariance)),
        "metric covariance must be finite",
    )
    anchored, correction = exact_anchor_trajectory(raw, query)
    for point_index, offset in enumerate(correction):
        covariance[:, point_index] += np.outer(offset, offset).astype(
            np.float32
        )

    output.mkdir(parents=True)
    archive_path = output / PREDICTION_FILENAME
    np.savez_compressed(
        archive_path,
        raw_tracker_m=raw,
        anchored_tracker_m=anchored,
        accepted_support=support,
        observation_reliability=reliability,
        observation_covariance_m2=covariance,
        support_view_count=view_count,
        reprojection_rmse_px=reprojection,
        depth_residual_rmse_m=depth_residual,
        per_camera_tracks_xy=camera_tracks,
        per_camera_visibility_probability=camera_visibility,
        frame_zero_anchor_correction_m=correction,
        identity_ids=ids,
    )
    supported_covariance = covariance[support]
    minimum_eigenvalue = (
        float(
            np.min(
                np.linalg.eigvalsh(
                    supported_covariance.astype(np.float64)
                )
            )
        )
        if len(supported_covariance)
        else None
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinTAPNextPPPrefixPrediction",
        "protocol_id": PROTOCOL_ID,
        "case": CASE_NAME,
        "config": asdict(cfg),
        "tracker": {
            "name": "TAPNext++",
            "repository_revision": TAPNEXTPP_REVISION,
            "checkpoint_sha256": TAPNEXTPP_CHECKPOINT_SHA256,
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
            "accepted_fraction": float(np.mean(support)),
            "two_view_fraction_among_accepted": (
                float(np.mean(view_count[support] == 2))
                if np.any(support)
                else None
            ),
            "mean_reliability_among_accepted": (
                float(np.mean(reliability[support]))
                if np.any(support)
                else None
            ),
            "minimum_covariance_eigenvalue_m2": minimum_eigenvalue,
        },
        "output": {
            "archive": str(archive_path),
            "archive_sha256": file_sha256(archive_path),
        },
        "information_boundary": {
            "rgb_depth_mask_frame_range_half_open": [
                cfg.source_frame_start,
                cfg.source_frame_end_exclusive,
            ],
            "manual_query_frame": cfg.source_frame_start,
            "withheld_prefix_target_read": False,
            "manual_track_file_read": False,
            "frame_at_or_after_train_end_read": False,
            "physical_state_innovation_used_in_prior_reliability": False,
            "future_observation_used": False,
        },
        "claim_boundary": (
            "target-free online-tracker prediction for one already-open source "
            "competence control; not simulator assimilation, confirmation, "
            "calibration, or state-of-the-art evidence"
        ),
    }
    report["result_sha256"] = canonical_sha256(report)
    report_path = output / PREDICTION_REPORT_FILENAME
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def seal_prediction(prediction_dir: str | Path) -> dict[str, Any]:
    """Seal a TAPNext++ prediction before opening its manual prefix target."""

    prediction = Path(prediction_dir).resolve()
    report_path = prediction / PREDICTION_REPORT_FILENAME
    archive_path = prediction / PREDICTION_FILENAME
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(
        report.get("artifact_kind") == "PhysTwinTAPNextPPPrefixPrediction",
        "prediction report kind is invalid",
    )
    _require(
        report.get("result_sha256") == canonical_sha256(report),
        "prediction report self-hash changed",
    )
    archive_sha256 = file_sha256(archive_path)
    _require(
        report.get("output", {}).get("archive_sha256") == archive_sha256,
        "prediction archive hash differs from its report",
    )
    seal: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinTAPNextPPPrefixPredictionSeal",
        "protocol_id": PROTOCOL_ID,
        "case": CASE_NAME,
        "prediction_report_sha256": file_sha256(report_path),
        "prediction_archive_sha256": archive_sha256,
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
) -> float | None:
    selected = (
        np.asarray(prediction, dtype=float)
        - np.asarray(target, dtype=float)
    )[np.asarray(mask, dtype=bool)]
    if len(selected) == 0:
        return None
    return float(np.sqrt(np.mean(np.sum(np.square(selected), axis=1))))


def evaluate_competence(
    prediction_dir: str | Path,
    withheld_prefix_path: str | Path,
    expected_withheld_sha256: str,
    output_path: str | Path,
    *,
    config: PhysTwinTAPNextPPCompetenceConfig | None = None,
) -> dict[str, Any]:
    """Open only the staged manual prefix target after prediction sealing."""

    cfg = config or PhysTwinTAPNextPPCompetenceConfig()
    prediction = Path(prediction_dir).resolve()
    seal_path = prediction / PREDICTION_SEAL_FILENAME
    report_path = prediction / PREDICTION_REPORT_FILENAME
    archive_path = prediction / PREDICTION_FILENAME
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    _require(
        seal.get("artifact_kind") == "PhysTwinTAPNextPPPrefixPredictionSeal",
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
        support = np.asarray(stored["accepted_support"], dtype=bool)
        identity_ids = np.asarray(stored["identity_ids"], dtype=np.int64)
    with np.load(withheld_path, allow_pickle=False) as stored:
        target = np.asarray(stored["target_tracks_world_m"], dtype=np.float32)
        target_ids = np.asarray(stored["identity_ids"], dtype=np.int64)
        source_start = int(stored["source_frame_start"])
        source_end = int(stored["source_frame_end_exclusive"])
    expected_shape = (
        cfg.prefix_frame_count,
        len(identity_ids),
        3,
    )
    _require(
        candidate.shape == target.shape == expected_shape,
        "candidate and withheld target shapes differ",
    )
    _require(
        support.shape == expected_shape[:2],
        "prediction support shape differs",
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
    score_rows = np.arange(cfg.prefix_frame_count) > 0
    scored_target = target_valid & score_rows[:, None]
    scored_supported = support & scored_target
    supported_fraction = float(
        np.sum(scored_supported) / max(np.sum(scored_target), 1)
    )
    persistence = np.repeat(target[:1], cfg.prefix_frame_count, axis=0)
    candidate_rmse = _radial_rmse_m(
        candidate,
        target,
        scored_supported,
    )
    persistence_rmse = _radial_rmse_m(
        persistence,
        target,
        scored_supported,
    )
    relative_gain = (
        (persistence_rmse - candidate_rmse) / persistence_rmse
        if candidate_rmse is not None
        and persistence_rmse is not None
        and persistence_rmse > 0.0
        else None
    )
    endpoint_rows = np.arange(cfg.prefix_frame_count) >= (
        cfg.prefix_frame_count - cfg.endpoint_frame_count
    )
    endpoint_supported = support & target_valid & endpoint_rows[:, None]
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
            relative_gain is not None
            and relative_gain >= cfg.minimum_relative_gain_over_persistence
        ),
        "identity_rmse": (
            candidate_rmse is not None
            and candidate_rmse <= cfg.maximum_identity_rmse_m
        ),
        "endpoint_rmse": (
            endpoint_rmse is not None
            and endpoint_rmse <= cfg.maximum_endpoint_rmse_m
        ),
    }
    passed = all(gates.values())
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinTAPNextPPPrefixCompetenceResult",
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
            "advance-to-separately-locked-guarded-assimilation-smoke"
            if passed
            else "stop-tapnextpp-phystwin-route"
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
            "held_v8_accessed": False,
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
    "PREDICTION_FILENAME",
    "PREDICTION_REPORT_FILENAME",
    "PREDICTION_SEAL_FILENAME",
    "PROTOCOL_ID",
    "PhysTwinTAPNextPPCompetenceConfig",
    "SOURCE_REPORT_FILENAME",
    "TAPNEXTPP_CHECKPOINT_SHA256",
    "TAPNEXTPP_REVISION",
    "WITHHELD_FILENAME",
    "evaluate_competence",
    "prepare_source_artifacts",
    "seal_prediction",
    "validate_prediction_input",
    "write_prediction_artifact",
]
