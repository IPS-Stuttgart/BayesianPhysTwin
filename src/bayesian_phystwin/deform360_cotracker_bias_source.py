"""Source-only CoTracker3 prefix study for the reusable penguin twin.

Prediction construction reads the selected physical response, calibration,
frame-zero mask/depth, and exact RGB prefixes.  Source PCD outcomes are opened
only by :func:`evaluate_penguin_source_predictions` after a complete
checksummed prediction seal exists.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .bias_aware_belief import (
    apply_group_regret_bound,
    fit_source_group_regret_bound,
)
from .deform360_bias_aware_belief_development import (
    Deform360BiasAwareDevelopmentConfig,
    predict_bias_aware_candidate_arrays,
)
from .deform360_online_belief_evaluation import (
    PRIMARY_METRICS,
    score_deform360_hidden_trajectory,
)
from .deform360_raw_camera_observation import (
    RawCameraObservationConfig,
    frame_zero_camera_support,
    select_frame_zero_observation_plan,
    triangulate_observation_ransac,
)
from .phystwin_online_belief import deterministic_farthest_point_ids


PROTOCOL_ID = "deform360-penguin-cotracker-bias-source-v1"
SOURCE_EPISODE_IDS = (1, 3, 4, 6, 7, 9)
PREDICTION_FILENAME = "prediction.npz"
REPORT_FILENAME = "prediction_report.json"
COHORT_SEAL_FILENAME = "prediction_cohort_seal.json"
PRIMARY_IDENTITY = "post_update_hidden_identity_rmse_m"
PRIMARY_CHAMFER = "post_update_hidden_symmetric_chamfer_m"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("result_sha256", None)
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def penguin_episode_directory(staged_root: str | Path, episode_id: int) -> Path:
    """Resolve the source-staging naming convention without target discovery."""

    _require(episode_id in SOURCE_EPISODE_IDS, "episode is outside the source panel")
    name = "171-penguin" if episode_id == 1 else f"171-penguin-ep{episode_id:04d}"
    return Path(staged_root).resolve() / name / "episode_0000"


class PrefixTracker(Protocol):
    """Small runtime boundary shared by GPU and deterministic test trackers."""

    def track_prefix(
        self,
        video_path: str | Path,
        query_pixels_xy: np.ndarray,
        update_frame: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]: ...


@dataclass(frozen=True)
class PenguinCoTrackerSourceConfig:
    """Frozen target-free settings for this source diagnostic."""

    update_frames: tuple[int, ...] = (19, 38, 57)
    center_count: int = 16
    selected_camera_count: int = 5
    minimum_initial_view_count: int = 2
    minimum_triangulation_view_count: int = 2
    observation_variance_floor_m2: float = 0.005**2
    reprojection_scale_px: float = 3.0
    two_view_variance_multiplier: float = 4.0
    minimum_improvement_m: float = 0.000005
    nominal_regret_coverage: float = 0.90

    def __post_init__(self) -> None:
        _require(
            tuple(sorted(set(self.update_frames))) == self.update_frames,
            "update frames must be strictly increasing",
        )
        _require(self.center_count >= 1, "center count must be positive")
        _require(self.selected_camera_count >= 2, "at least two cameras are needed")
        _require(
            2 <= self.minimum_initial_view_count <= self.selected_camera_count,
            "invalid initial view count",
        )
        _require(
            2 <= self.minimum_triangulation_view_count <= self.selected_camera_count,
            "invalid triangulation view count",
        )
        positive = (
            self.observation_variance_floor_m2,
            self.reprojection_scale_px,
            self.two_view_variance_multiplier,
        )
        _require(
            all(np.isfinite(value) and value > 0.0 for value in positive),
            "measurement scales must be positive",
        )
        _require(self.minimum_improvement_m >= 0.0, "improvement is negative")
        _require(
            0.0 < self.nominal_regret_coverage < 1.0,
            "regret coverage must lie in (0, 1)",
        )

    def observation_config(self) -> RawCameraObservationConfig:
        """Translate source choices into the shared camera geometry contract."""

        return RawCameraObservationConfig(
            center_count=self.center_count,
            selected_camera_count=self.selected_camera_count,
            minimum_initial_view_count=self.minimum_initial_view_count,
            minimum_triangulation_view_count=(
                self.minimum_triangulation_view_count
            ),
            update_frames=self.update_frames,
        )

    def belief_config(self) -> Deform360BiasAwareDevelopmentConfig:
        """Reuse the frozen source-v4 state/bias estimator unchanged."""

        return Deform360BiasAwareDevelopmentConfig(
            update_frames=self.update_frames,
            observation_variance_floor_m2=self.observation_variance_floor_m2,
            reprojection_scale_px=self.reprojection_scale_px,
            regret_nominal_coverage=self.nominal_regret_coverage,
            regret_minimum_improvement_m=self.minimum_improvement_m,
        )


def conservative_triangulation_variance_m2(
    *,
    inlier_view_count: int,
    selected_camera_count: int,
    leave_one_view_points_m: np.ndarray,
    fused_point_m: np.ndarray,
    variance_floor_m2: float,
    two_view_variance_multiplier: float,
) -> float:
    """Return a correlation-conservative scalar metric variance.

    The base variance is inflated by view redundancy.  Leave-one-view spread
    can only increase it.  In particular, duplicating a coherent view cannot
    drive the variance below the fixed metric floor.
    """

    _require(inlier_view_count >= 2, "triangulation needs at least two views")
    _require(
        selected_camera_count >= inlier_view_count,
        "inlier count exceeds selected views",
    )
    fused = np.asarray(fused_point_m, dtype=np.float64)
    _require(fused.shape == (3,) and np.all(np.isfinite(fused)), "invalid point")
    leave_one_out = np.asarray(leave_one_view_points_m, dtype=np.float64)
    _require(
        leave_one_out.ndim == 2 and leave_one_out.shape[1] == 3,
        "leave-one-view points must have shape (K, 3)",
    )
    redundancy_inflation = (selected_camera_count / inlier_view_count) ** 2
    if inlier_view_count == 2:
        redundancy_inflation = max(
            redundancy_inflation, two_view_variance_multiplier
        )
    result = variance_floor_m2 * redundancy_inflation
    if len(leave_one_out):
        _require(
            np.all(np.isfinite(leave_one_out)),
            "leave-one-view points are not finite",
        )
        squared_spread = np.sum(np.square(leave_one_out - fused[None]), axis=1)
        result = max(result, float(np.max(squared_spread)))
    return float(result)


def _projection_matrix(
    intrinsics: np.ndarray, camera_to_world: np.ndarray
) -> np.ndarray:
    return (
        np.asarray(intrinsics, dtype=np.float64)
        @ np.linalg.inv(np.asarray(camera_to_world, dtype=np.float64))[:3]
    )


def _load_calibration(
    episode_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    intrinsics = np.load(
        episode_dir / "undistorted_intrinsics.npy", allow_pickle=True
    ).item()
    extrinsics = np.load(
        episode_dir / "extrinsics.npy", allow_pickle=True
    ).item()
    _require(isinstance(intrinsics, dict), "intrinsics archive is invalid")
    _require(isinstance(extrinsics, dict), "extrinsics archive is invalid")
    return intrinsics, extrinsics


def _default_center_ids(points_m: np.ndarray, center_count: int) -> np.ndarray:
    candidates = np.arange(len(points_m), dtype=np.int64)
    return deterministic_farthest_point_ids(
        points_m, candidates, min(center_count, len(candidates))
    )


def build_causal_cotracker_measurement(
    episode_dir: str | Path,
    frame_zero_points_m: np.ndarray,
    trajectory_shape: tuple[int, int, int],
    tracker: PrefixTracker,
    *,
    config: PenguinCoTrackerSourceConfig | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build sparse 3-D measurements without reading source PCD outcomes."""

    cfg = config or PenguinCoTrackerSourceConfig()
    camera_cfg = cfg.observation_config()
    root = Path(episode_dir).resolve()
    frame_zero = np.asarray(frame_zero_points_m, dtype=np.float64)
    _require(
        frame_zero.shape == trajectory_shape[1:],
        "frame-zero geometry differs from trajectory",
    )
    intrinsics, extrinsics = _load_calibration(root)
    cameras, support, projected = frame_zero_camera_support(
        frame_zero,
        root,
        intrinsics,
        extrinsics,
        depth_tolerance_m=camera_cfg.frame_zero_depth_tolerance_m,
    )
    plan = select_frame_zero_observation_plan(
        frame_zero,
        cameras,
        support,
        projected,
        extrinsics,
        config=camera_cfg,
    )
    centers = np.asarray(plan["center_ids"], dtype=np.int64)
    selected_cameras = tuple(plan["selected_cameras"])
    projection_matrices = {
        camera: _projection_matrix(intrinsics[camera], extrinsics[camera])
        for camera in selected_cameras
    }
    camera_origins = {
        camera: np.asarray(extrinsics[camera], dtype=np.float64)[:3, 3]
        for camera in selected_cameras
    }
    measurement = np.full(trajectory_shape, np.nan, dtype=np.float32)
    visibility = np.zeros(trajectory_shape[:2], dtype=bool)
    validity = np.zeros(trajectory_shape[:2], dtype=bool)
    measurement[0, centers] = frame_zero[centers]
    visibility[0, centers] = True
    validity[0, centers] = True
    reliability = np.zeros((len(cfg.update_frames), len(centers)), dtype=np.float64)
    variance = np.full(
        reliability.shape, cfg.observation_variance_floor_m2, dtype=np.float64
    )
    inlier_count = np.zeros(reliability.shape, dtype=np.int16)
    reprojection = np.full(reliability.shape, np.nan, dtype=np.float32)
    update_reports: list[dict[str, Any]] = []

    for update_index, update_frame in enumerate(cfg.update_frames):
        tracks_by_camera: dict[str, dict[int, np.ndarray]] = {}
        tracker_reports: list[dict[str, Any]] = []
        for camera in selected_cameras:
            query_ids = np.asarray(plan["query_ids"][camera], dtype=np.int64)
            query_pixels = np.asarray(plan["query_pixels"][camera], dtype=np.float64)
            tracks, visible, tracker_report = tracker.track_prefix(
                root / camera / "undistorted.mp4",
                query_pixels,
                update_frame,
            )
            _require(
                tracks.shape == query_pixels.shape
                and visible.shape == (len(query_ids),),
                "tracker output shape changed",
            )
            tracks_by_camera[camera] = {
                int(point_id): tracks[index]
                for index, point_id in enumerate(query_ids)
                if visible[index]
            }
            tracker_reports.append(
                {
                    **tracker_report,
                    "camera": camera,
                    "query_ids": query_ids.tolist(),
                }
            )
        center_reports: list[dict[str, Any]] = []
        for center_index, center_id in enumerate(centers):
            observations = {
                camera: tracks_by_camera[camera][int(center_id)]
                for camera in selected_cameras
                if int(center_id) in tracks_by_camera[camera]
            }
            point, diagnostic = triangulate_observation_ransac(
                observations,
                projection_matrices,
                camera_origins,
                frame_zero[center_id],
                config=camera_cfg,
            )
            diagnostic["center_id"] = int(center_id)
            center_reports.append(diagnostic)
            if point is None:
                continue
            inlier_cameras = tuple(diagnostic["inlier_cameras"])
            leave_one_out: list[np.ndarray] = []
            if len(inlier_cameras) >= 3:
                for excluded in inlier_cameras:
                    reduced = {
                        camera: observations[camera]
                        for camera in inlier_cameras
                        if camera != excluded
                    }
                    reduced_point, _ = triangulate_observation_ransac(
                        reduced,
                        projection_matrices,
                        camera_origins,
                        frame_zero[center_id],
                        config=camera_cfg,
                    )
                    if reduced_point is not None:
                        leave_one_out.append(reduced_point)
            count = int(diagnostic["inlier_view_count"])
            median_reprojection = float(
                diagnostic["median_reprojection_error_px"]
            )
            redundancy = np.clip(
                (count - 1.0) / (len(selected_cameras) - 1.0), 0.0, 1.0
            )
            geometry = np.exp(
                -0.5
                * np.square(median_reprojection / cfg.reprojection_scale_px)
            )
            measurement[update_frame, center_id] = point
            visibility[update_frame, center_id] = True
            validity[update_frame, center_id] = True
            reliability[update_index, center_index] = float(redundancy * geometry)
            variance[update_index, center_index] = (
                conservative_triangulation_variance_m2(
                    inlier_view_count=count,
                    selected_camera_count=len(selected_cameras),
                    leave_one_view_points_m=np.asarray(
                        leave_one_out, dtype=np.float64
                    ).reshape(-1, 3),
                    fused_point_m=point,
                    variance_floor_m2=cfg.observation_variance_floor_m2,
                    two_view_variance_multiplier=(
                        cfg.two_view_variance_multiplier
                    ),
                )
            )
            inlier_count[update_index, center_index] = count
            reprojection[update_index, center_index] = median_reprojection
        update_reports.append(
            {
                "frame": update_frame,
                "prefix_frame_range_half_open": [0, update_frame + 1],
                "maximum_video_frame_read": update_frame,
                "accepted_center_count": int(
                    np.sum(validity[update_frame, centers])
                ),
                "tracker": tracker_reports,
                "centers": center_reports,
            }
        )
    arrays = {
        "measurement_m": measurement,
        "measurement_visibility": visibility,
        "measurement_validity": validity,
        "center_ids": centers,
        "selected_cameras": np.asarray(selected_cameras),
        "update_frames": np.asarray(cfg.update_frames, dtype=np.int64),
        "prior_reliability": reliability,
        "observation_variance_m2": variance,
        "triangulation_inlier_view_count": inlier_count,
        "triangulation_median_reprojection_px": reprojection,
    }
    report = {
        "selected_cameras": list(selected_cameras),
        "center_ids": centers.tolist(),
        "selection_score": list(plan["selection_score"]),
        "updates": update_reports,
        "information_boundary": {
            "pcd_clean_read": False,
            "future_rgb_read": False,
            "maximum_video_frame_read_by_update": list(cfg.update_frames),
            "frame_zero_mask_and_depth_indices_read": [0],
            "prior_reliability_uses_state_innovation": False,
        },
    }
    return arrays, report


def _load_selected_response(
    response_root: Path, episode_id: int
) -> dict[str, np.ndarray]:
    path = response_root / f"episode_{episode_id:04d}.npz"
    with np.load(path, allow_pickle=False) as stored:
        required = {
            "prediction_m",
            "persistence_m",
            "driven_readout_m",
            "zero_action_readout_m",
            "action_support",
            "frame_zero_points_m",
        }
        _require(required <= set(stored.files), "physical response fields changed")
        return {name: np.asarray(stored[name]).copy() for name in required}


def build_penguin_source_prediction(
    *,
    episode_id: int,
    episode_dir: str | Path,
    response_root: str | Path,
    output_dir: str | Path,
    tracker: PrefixTracker,
    config: PenguinCoTrackerSourceConfig | None = None,
) -> dict[str, Any]:
    """Construct and seal one target-free source prediction."""

    cfg = config or PenguinCoTrackerSourceConfig()
    root = Path(episode_dir).resolve()
    response_path = (
        Path(response_root).resolve() / f"episode_{episode_id:04d}.npz"
    )
    response = _load_selected_response(Path(response_root).resolve(), episode_id)
    baseline = np.asarray(response["prediction_m"])
    frame_zero = np.asarray(response["frame_zero_points_m"])
    fallback_reason = None
    try:
        measurement, measurement_report = build_causal_cotracker_measurement(
            root,
            frame_zero,
            baseline.shape,
            tracker,
            config=cfg,
        )
        candidate_report, candidate = predict_bias_aware_candidate_arrays(
            baseline,
            np.asarray(response["driven_readout_m"], dtype=np.float64)
            - np.asarray(response["zero_action_readout_m"], dtype=np.float64),
            frame_zero,
            np.asarray(response["action_support"], dtype=np.float64),
            measurement["measurement_m"],
            measurement["measurement_visibility"],
            measurement["measurement_validity"],
            center_ids=measurement["center_ids"],
            prior_reliability=measurement["prior_reliability"],
            observation_variance_m2=measurement["observation_variance_m2"],
            config=cfg.belief_config(),
        )
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        fallback_reason = f"{type(error).__name__}: {error}"
        centers = _default_center_ids(frame_zero, cfg.center_count)
        measurement = {
            "center_ids": centers,
            "selected_cameras": np.asarray([], dtype="U1"),
            "prior_reliability": np.zeros(
                (len(cfg.update_frames), len(centers)), dtype=np.float64
            ),
            "observation_variance_m2": np.full(
                (len(cfg.update_frames), len(centers)),
                cfg.observation_variance_floor_m2,
                dtype=np.float64,
            ),
        }
        measurement_report = {
            "selected_cameras": [],
            "center_ids": centers.tolist(),
            "updates": [],
            "fallback_reason": fallback_reason,
            "information_boundary": {
                "pcd_clean_read": False,
                "future_rgb_read": False,
            },
        }
        candidate = baseline.copy()
        candidate_report = {
            "protocol_id": PROTOCOL_ID,
            "arm": "exact-physical-fallback",
            "updates": [],
            "candidate_update_count": 0,
            "fallback_reason": fallback_reason,
        }
    output = Path(output_dir).resolve()
    _require(not output.exists(), "prediction output already exists")
    output.mkdir(parents=True)
    archive_path = output / PREDICTION_FILENAME
    np.savez_compressed(
        archive_path,
        baseline_m=baseline,
        candidate_m=candidate,
        center_ids=measurement["center_ids"],
        prior_reliability=measurement["prior_reliability"],
        observation_variance_m2=measurement["observation_variance_m2"],
        selected_cameras=measurement["selected_cameras"],
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360PenguinCoTrackerBiasSourcePrediction",
        "protocol_id": PROTOCOL_ID,
        "object_id": "171-penguin",
        "episode_id": episode_id,
        "config": asdict(cfg),
        "measurement": measurement_report,
        "candidate": candidate_report,
        "technical_fallback": fallback_reason is not None,
        "fallback_reason": fallback_reason,
        "inputs_sha256": {
            "selected_physical_response": _sha256_file(response_path),
            "intrinsics": _sha256_file(root / "undistorted_intrinsics.npy"),
            "extrinsics": _sha256_file(root / "extrinsics.npy"),
        },
        "output": {
            "prediction_archive": str(archive_path),
            "prediction_archive_sha256": _sha256_file(archive_path),
        },
        "information_boundary": {
            "source_outcome_read": False,
            "pcd_clean_read": False,
            "future_rgb_read": False,
            "known_action_physical_response_used": True,
            "state_innovation_likelihood_count": 1,
            "exact_physical_fallback": True,
        },
        "claim_boundary": (
            "target-free prediction on an already outcome-authorized source "
            "episode; no accuracy claim until the complete cohort is sealed"
        ),
    }
    report["result_sha256"] = _canonical_sha256(report)
    (output / REPORT_FILENAME).write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def seal_penguin_source_predictions(
    prediction_root: str | Path,
    *,
    config: PenguinCoTrackerSourceConfig | None = None,
) -> dict[str, Any]:
    """Require and hash every source prediction before outcome scoring."""

    cfg = config or PenguinCoTrackerSourceConfig()
    root = Path(prediction_root).resolve()
    cases: list[dict[str, Any]] = []
    for episode_id in SOURCE_EPISODE_IDS:
        case = root / f"episode_{episode_id:04d}"
        report_path = case / REPORT_FILENAME
        archive_path = case / PREDICTION_FILENAME
        report = json.loads(report_path.read_text(encoding="utf-8"))
        _require(
            report.get("result_sha256") == _canonical_sha256(report),
            "prediction report checksum changed",
        )
        _require(
            report.get("information_boundary", {}).get("source_outcome_read")
            is False,
            "prediction report crossed the source-outcome boundary",
        )
        _require(
            report["output"]["prediction_archive_sha256"]
            == _sha256_file(archive_path),
            "prediction archive checksum changed",
        )
        cases.append(
            {
                "episode_id": episode_id,
                "report_sha256": _sha256_file(report_path),
                "archive_sha256": _sha256_file(archive_path),
                "technical_fallback": bool(report["technical_fallback"]),
            }
        )
    seal: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360PenguinCoTrackerBiasSourcePredictionSeal",
        "protocol_id": PROTOCOL_ID,
        "config": asdict(cfg),
        "episodes": cases,
        "prediction_count": len(cases),
        "information_boundary": {
            "all_predictions_hashed_before_source_outcome_open": True,
            "pcd_clean_read": False,
        },
        "claim_boundary": (
            "complete source-only prediction seal; outcomes remain outside this "
            "operation"
        ),
    }
    seal["result_sha256"] = _canonical_sha256(seal)
    output = root / COHORT_SEAL_FILENAME
    _require(not output.exists(), "prediction cohort is already sealed")
    output.write_text(
        json.dumps(seal, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return seal


def _load_pcd_target(episode_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = sorted((episode_dir / "pcd_clean").glob("*.npz"))
    _require(paths, "source PCD outcome is missing")
    points: list[np.ndarray] = []
    visibility: list[np.ndarray] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as stored:
            point = np.asarray(stored["pts"], dtype=np.float32)
            visible = np.any(
                np.asarray(stored["visibility_matrix"], dtype=bool), axis=1
            )
        points.append(point)
        visibility.append(visible)
    target = np.stack(points)
    visible = np.stack(visibility)
    valid = visible & np.all(np.isfinite(target), axis=2)
    return target, visible, valid


def _scored_frames(
    frame_count: int, update_frames: tuple[int, ...]
) -> tuple[int, ...]:
    result: list[int] = []
    for index, update in enumerate(update_frames):
        stop = (
            update_frames[index + 1]
            if index + 1 < len(update_frames)
            else frame_count
        )
        result.extend(range(update + 1, stop))
    return tuple(result)


def _score(
    trajectory: np.ndarray,
    target: np.ndarray,
    visibility: np.ndarray,
    validity: np.ndarray,
    centers: np.ndarray,
    frames: tuple[int, ...],
) -> dict[str, object]:
    return score_deform360_hidden_trajectory(
        trajectory,
        target,
        visibility,
        validity,
        center_ids=centers,
        scored_frames=frames,
    )


def _relative_change(candidate: float, baseline: float) -> float:
    _require(baseline > 0.0, "baseline metric is not positive")
    return float(candidate / baseline - 1.0)


def _aggregate_case_scores(
    cases: Sequence[Mapping[str, Any]], arm: str
) -> dict[str, float]:
    return {
        metric: float(np.mean([case["scores"][arm][metric] for case in cases]))
        for metric in PRIMARY_METRICS
    }


def evaluate_penguin_source_predictions(
    *,
    prediction_root: str | Path,
    staged_root: str | Path,
    output_path: str | Path,
    config: PenguinCoTrackerSourceConfig | None = None,
) -> dict[str, Any]:
    """Open authorized source PCD outcomes and run episode-held-out guards."""

    cfg = config or PenguinCoTrackerSourceConfig()
    prediction = Path(prediction_root).resolve()
    seal_path = prediction / COHORT_SEAL_FILENAME
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    _require(
        seal.get("result_sha256") == _canonical_sha256(seal),
        "prediction cohort seal checksum changed",
    )
    _require(
        seal.get("prediction_count") == len(SOURCE_EPISODE_IDS),
        "prediction cohort is incomplete",
    )
    cases: list[dict[str, Any]] = []
    for episode_id in SOURCE_EPISODE_IDS:
        case_dir = prediction / f"episode_{episode_id:04d}"
        report_path = case_dir / REPORT_FILENAME
        archive_path = case_dir / PREDICTION_FILENAME
        report = json.loads(report_path.read_text(encoding="utf-8"))
        sealed = next(
            row for row in seal["episodes"] if row["episode_id"] == episode_id
        )
        _require(
            _sha256_file(report_path) == sealed["report_sha256"]
            and _sha256_file(archive_path) == sealed["archive_sha256"],
            "sealed prediction artifact changed",
        )
        with np.load(archive_path, allow_pickle=False) as stored:
            baseline = np.asarray(stored["baseline_m"]).copy()
            candidate = np.asarray(stored["candidate_m"]).copy()
            centers = np.asarray(stored["center_ids"], dtype=np.int64)
        episode_dir = penguin_episode_directory(staged_root, episode_id)
        target, visibility, validity = _load_pcd_target(episode_dir)
        _require(target.shape == baseline.shape, "target and prediction shapes differ")
        frames = _scored_frames(len(target), cfg.update_frames)
        interval_outcomes: list[dict[str, float | int | bool]] = []
        update_records = report["candidate"].get("updates", [])
        by_frame = {int(row["frame"]): row for row in update_records}
        for index, update in enumerate(cfg.update_frames):
            stop = (
                cfg.update_frames[index + 1]
                if index + 1 < len(cfg.update_frames)
                else len(target)
            )
            interval_frames = tuple(range(update + 1, stop))
            baseline_interval = _score(
                baseline, target, visibility, validity, centers, interval_frames
            )
            candidate_interval = _score(
                candidate, target, visibility, validity, centers, interval_frames
            )
            identity_regret = float(
                candidate_interval[PRIMARY_IDENTITY]
                - baseline_interval[PRIMARY_IDENTITY]
            )
            chamfer_regret = float(
                candidate_interval[PRIMARY_CHAMFER]
                - baseline_interval[PRIMARY_CHAMFER]
            )
            interval_outcomes.append(
                {
                    "frame": update,
                    "candidate_available": bool(
                        by_frame.get(update, {}).get("candidate_available", False)
                    ),
                    "identity_regret_m": identity_regret,
                    "chamfer_regret_m": chamfer_regret,
                    "worst_primary_regret_m": max(
                        identity_regret, chamfer_regret
                    ),
                }
            )
        cases.append(
            {
                "episode_id": episode_id,
                "baseline": baseline,
                "candidate": candidate,
                "target": target,
                "visibility": visibility,
                "validity": validity,
                "center_ids": centers,
                "frames": frames,
                "scores": {
                    "physical_baseline": _score(
                        baseline, target, visibility, validity, centers, frames
                    ),
                    "unguarded_bias_aware": _score(
                        candidate, target, visibility, validity, centers, frames
                    ),
                },
                "interval_outcomes": interval_outcomes,
                "guarded": None,
                "guard": None,
            }
        )

    for held in cases:
        regrets: list[float] = []
        groups: list[str] = []
        for source in cases:
            if source is held:
                continue
            for interval in source["interval_outcomes"]:
                if interval["candidate_available"]:
                    regrets.append(float(interval["worst_primary_regret_m"]))
                    groups.append(f"episode-{source['episode_id']}")
        unique_groups = sorted(set(groups))
        if len(unique_groups) < 3:
            selected = held["baseline"].copy()
            guard = {
                "candidate_accepted": False,
                "reason": "fewer-than-three-eligible-source-episodes",
                "eligible_source_episode_count": len(unique_groups),
            }
        else:
            bound = fit_source_group_regret_bound(
                np.asarray(regrets, dtype=np.float64),
                groups,
                nominal_coverage=cfg.nominal_regret_coverage,
                within_group_coverage=1.0,
                minimum_improvement_m=cfg.minimum_improvement_m,
            )
            decision = apply_group_regret_bound(
                held["baseline"], held["candidate"], bound
            )
            selected = decision.selected_value
            guard = {
                "candidate_accepted": decision.candidate_accepted,
                "reason": decision.reason,
                "upper_regret_m": decision.upper_regret,
                "eligible_source_episode_count": len(unique_groups),
                "finite_sample_rank": bound.finite_sample_rank,
                "finite_sample_coverage": bound.finite_sample_coverage,
            }
        if not guard["candidate_accepted"]:
            _require(
                selected.tobytes() == held["baseline"].tobytes(),
                "guarded fallback changed the physical baseline",
            )
        held["guarded"] = selected
        held["guard"] = guard
        held["scores"]["loo_guarded_bias_aware"] = _score(
            selected,
            held["target"],
            held["visibility"],
            held["validity"],
            held["center_ids"],
            held["frames"],
        )

    serializable_cases = []
    for case in cases:
        late_frames = tuple(range(cfg.update_frames[-1] + 1, len(case["target"])))
        late_scores = {
            "physical_baseline": _score(
                case["baseline"],
                case["target"],
                case["visibility"],
                case["validity"],
                case["center_ids"],
                late_frames,
            ),
            "loo_guarded_bias_aware": _score(
                case["guarded"],
                case["target"],
                case["visibility"],
                case["validity"],
                case["center_ids"],
                late_frames,
            ),
        }
        serializable_cases.append(
            {
                "episode_id": case["episode_id"],
                "center_ids": case["center_ids"].tolist(),
                "scores": case["scores"],
                "late_scores": late_scores,
                "interval_outcomes": case["interval_outcomes"],
                "guard": case["guard"],
            }
        )
    aggregate = {
        arm: _aggregate_case_scores(serializable_cases, arm)
        for arm in (
            "physical_baseline",
            "unguarded_bias_aware",
            "loo_guarded_bias_aware",
        )
    }
    late = {
        arm: {
            metric: float(
                np.mean(
                    [case["late_scores"][arm][metric] for case in serializable_cases]
                )
            )
            for metric in PRIMARY_METRICS
        }
        for arm in ("physical_baseline", "loo_guarded_bias_aware")
    }
    relative = {
        metric: _relative_change(
            aggregate["loo_guarded_bias_aware"][metric],
            aggregate["physical_baseline"][metric],
        )
        for metric in PRIMARY_METRICS
    }
    late_relative = {
        metric: _relative_change(
            late["loo_guarded_bias_aware"][metric],
            late["physical_baseline"][metric],
        )
        for metric in PRIMARY_METRICS
    }
    maximum_degradation = max(
        _relative_change(
            case["scores"]["loo_guarded_bias_aware"][metric],
            case["scores"]["physical_baseline"][metric],
        )
        for case in serializable_cases
        for metric in PRIMARY_METRICS
    )
    gates = {
        "minimum_five_percent_identity_improvement": relative[PRIMARY_IDENTITY]
        <= -0.05,
        "minimum_five_percent_chamfer_improvement": relative[PRIMARY_CHAMFER]
        <= -0.05,
        "minimum_five_percent_late_identity_improvement": late_relative[
            PRIMARY_IDENTITY
        ]
        <= -0.05,
        "minimum_five_percent_late_chamfer_improvement": late_relative[
            PRIMARY_CHAMFER
        ]
        <= -0.05,
        "maximum_episode_metric_degradation_at_most_ten_percent": (
            maximum_degradation <= 0.10
        ),
        "at_least_one_held_episode_update_accepted": any(
            case["guard"]["candidate_accepted"] for case in serializable_cases
        ),
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360PenguinCoTrackerBiasSourceEvaluation",
        "protocol_id": PROTOCOL_ID,
        "config": asdict(cfg),
        "cases": serializable_cases,
        "aggregate": aggregate,
        "late": late,
        "relative_change": relative,
        "late_relative_change": late_relative,
        "maximum_episode_metric_degradation": maximum_degradation,
        "gates": gates,
        "accuracy_transfer_gates_pass": all(gates.values()),
        "calibration_gate_evaluated": False,
        "inputs_sha256": {"prediction_cohort_seal": _sha256_file(seal_path)},
        "information_boundary": {
            "source_outcomes_opened_after_prediction_seal": True,
            "held_v8_accessed": False,
            "fresh_target_accessed": False,
            "cross_fit_unit": "episode within one physical object",
        },
        "claim_boundary": (
            "already-authorized same-object source development; episode-held-out "
            "accuracy is not object-held-out confirmation or a calibration claim"
        ),
    }
    result["result_sha256"] = _canonical_sha256(result)
    destination = Path(output_path).resolve()
    _require(not destination.exists(), "evaluation output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


__all__ = [
    "COHORT_SEAL_FILENAME",
    "PenguinCoTrackerSourceConfig",
    "SOURCE_EPISODE_IDS",
    "build_causal_cotracker_measurement",
    "build_penguin_source_prediction",
    "conservative_triangulation_variance_m2",
    "evaluate_penguin_source_predictions",
    "penguin_episode_directory",
    "seal_penguin_source_predictions",
]
