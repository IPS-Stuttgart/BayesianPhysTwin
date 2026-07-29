"""Causal CoTracker3 complement for the frozen TAPNext++ source prediction.

The TAPNext++ prediction is never modified where it has accepted support.
CoTracker3 may bridge a later unsupported row only after the same material
identity has accumulated enough earlier cross-provider agreement.  The
complement therefore adds temporal coverage without relaxing either tracker's
geometric acceptance thresholds or using a PhysTwin innovation as perception
reliability.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .phystwin_mvtracker_competence import canonical_sha256, file_sha256

PROTOCOL_ID = "phystwin-tapnextpp-cotracker-complement-v1"
CASE_NAME = "single_lift_cloth"
PREDICTION_FILENAME = "complement_prediction.npz"
PREDICTION_REPORT_FILENAME = "complement_prediction_report.json"
PREDICTION_SEAL_FILENAME = "complement_prediction_seal.json"
EVALUATION_FILENAME = "complement_prefix_evaluation.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class TAPNextPPCoTrackerComplementConfig:
    """Frozen target-free choices for the complementary provider."""

    source_frame_start: int = 68
    source_frame_end_exclusive: int = 88
    minimum_camera_count: int = 2
    maximum_reprojection_error_px: float = 3.0
    minimum_quality_probability: float = 0.1
    association_radius_m: float = 0.015
    association_kernel_scale_m: float = 0.0075
    maximum_association_neighbors: int = 32
    minimum_neighbor_count: int = 3
    robust_displacement_scale_m: float = 0.010
    robust_reweight_iterations: int = 2
    minimum_prior_overlap_rows: int = 5
    maximum_prior_overlap_rmse_m: float = 0.010
    maximum_bridge_gap_frames: int = 5
    maximum_spatial_spread_m: float = 0.005
    two_view_shared_bias_standard_deviation_m: float = 0.010
    minimum_supported_fraction: float = 0.75
    minimum_added_row_gain_over_persistence: float = 0.10
    maximum_identity_rmse_m: float = 0.015
    maximum_endpoint_rmse_m: float = 0.015
    endpoint_frame_count: int = 5

    def __post_init__(self) -> None:
        _require(
            self.source_frame_end_exclusive > self.source_frame_start + 1,
            "source interval is too short",
        )
        positive = (
            self.maximum_reprojection_error_px,
            self.minimum_quality_probability,
            self.association_radius_m,
            self.association_kernel_scale_m,
            self.robust_displacement_scale_m,
            self.maximum_prior_overlap_rmse_m,
            self.maximum_spatial_spread_m,
            self.two_view_shared_bias_standard_deviation_m,
            self.maximum_identity_rmse_m,
            self.maximum_endpoint_rmse_m,
        )
        _require(all(value > 0.0 for value in positive), "scales must be positive")
        _require(self.minimum_camera_count >= 2, "at least two cameras are required")
        _require(
            self.maximum_association_neighbors >= self.minimum_neighbor_count >= 1,
            "association-neighbor bounds are invalid",
        )
        _require(
            self.robust_reweight_iterations >= 1,
            "robust reweighting needs at least one iteration",
        )
        _require(
            self.minimum_prior_overlap_rows >= 1,
            "cross-provider history must be nonempty",
        )
        _require(
            self.maximum_bridge_gap_frames >= 1,
            "bridge gap must be positive",
        )
        _require(
            0.0 < self.minimum_supported_fraction <= 1.0,
            "support gate must lie in (0, 1]",
        )
        _require(
            0.0 <= self.minimum_added_row_gain_over_persistence < 1.0,
            "gain gate must lie in [0, 1)",
        )
        _require(
            1 <= self.endpoint_frame_count < self.prefix_frame_count,
            "endpoint interval is invalid",
        )

    @property
    def prefix_frame_count(self) -> int:
        return self.source_frame_end_exclusive - self.source_frame_start


def _weighted_mean_and_covariance(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    robust_scale_m: float,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    vectors = np.asarray(values, dtype=np.float64)
    current = np.asarray(weights, dtype=np.float64).copy()
    _require(
        vectors.ndim == 2 and vectors.shape[1] == 3,
        "values must have shape (N, 3)",
    )
    _require(current.shape == (len(vectors),), "weights must match values")
    _require(
        np.all(np.isfinite(vectors))
        and np.all(np.isfinite(current))
        and np.all(current > 0.0),
        "weighted values must be finite with positive weights",
    )
    base_weights = current.copy()
    mean = np.average(vectors, axis=0, weights=current)
    for _ in range(iterations):
        residual = np.linalg.norm(vectors - mean, axis=1)
        robust = np.minimum(1.0, robust_scale_m / np.maximum(residual, 1e-12))
        current = base_weights * robust
        mean = np.average(vectors, axis=0, weights=current)
    normalized = current / np.sum(current)
    centered = vectors - mean
    covariance = np.einsum("n,ni,nj->ij", normalized, centered, centered)
    radial_spread = float(
        np.sqrt(np.sum(normalized * np.sum(np.square(centered), axis=1)))
    )
    return mean, covariance, radial_spread


def _association_entropy(weights: np.ndarray) -> float:
    probability = np.asarray(weights, dtype=np.float64)
    probability = probability / np.sum(probability)
    if len(probability) <= 1:
        return 0.0
    entropy = -float(np.sum(probability * np.log(np.maximum(probability, 1e-12))))
    return entropy / float(np.log(len(probability)))


def _load_prediction_inputs(
    query_path: Path,
    tapnextpp_path: Path,
    cotracker_path: Path,
    config: TAPNextPPCoTrackerComplementConfig,
) -> dict[str, np.ndarray]:
    with np.load(query_path, allow_pickle=False) as stored:
        query = np.asarray(stored["query_points_world_m"], dtype=np.float64)
        identity_ids = np.asarray(stored["identity_ids"], dtype=np.int64)
        source_frame = int(stored["source_frame"])
    with np.load(tapnextpp_path, allow_pickle=False) as stored:
        tap_trajectory = np.asarray(stored["anchored_tracker_m"])
        tap_support = np.asarray(stored["accepted_support"], dtype=bool)
        tap_covariance = np.asarray(
            stored["observation_covariance_m2"],
            dtype=np.float64,
        )
        tap_reliability = np.asarray(
            stored["observation_reliability"],
            dtype=np.float64,
        )
        tap_identity_ids = np.asarray(stored["identity_ids"], dtype=np.int64)
    with np.load(cotracker_path, allow_pickle=False) as stored:
        cotracker_points = np.asarray(
            stored["multiview_points_world_m"],
            dtype=np.float64,
        )
        cotracker_valid = np.asarray(
            stored["multiview_point_valid"],
            dtype=bool,
        )
        camera_count = np.asarray(
            stored["multiview_camera_count"],
            dtype=np.int16,
        )
        reprojection_error = np.asarray(
            stored["multiview_reprojection_error_px"],
            dtype=np.float64,
        )
        quality = np.asarray(
            stored["cotracker_quality_probability"],
            dtype=np.float64,
        )
    expected_shape = (config.prefix_frame_count, len(query))
    _require(source_frame == config.source_frame_start, "query frame changed")
    _require(query.ndim == 2 and query.shape[1] == 3, "query shape is invalid")
    _require(
        np.array_equal(identity_ids, tap_identity_ids),
        "TAPNext++ identity order changed",
    )
    _require(
        tap_trajectory.shape == (*expected_shape, 3)
        and tap_support.shape == expected_shape
        and tap_covariance.shape == (*expected_shape, 3, 3)
        and tap_reliability.shape == expected_shape,
        "TAPNext++ prediction shape changed",
    )
    _require(
        cotracker_points.ndim == 3
        and cotracker_points.shape[2] == 3
        and cotracker_valid.shape == cotracker_points.shape[:2]
        and camera_count.shape == cotracker_points.shape[:2]
        and reprojection_error.shape == cotracker_points.shape[:2]
        and quality.shape == cotracker_points.shape[:2],
        "CoTracker3 cue shape changed",
    )
    _require(
        config.source_frame_end_exclusive <= len(cotracker_points),
        "CoTracker3 cues are shorter than the frozen prefix",
    )
    return {
        "query": query,
        "identity_ids": identity_ids,
        "tap_trajectory": tap_trajectory,
        "tap_support": tap_support,
        "tap_covariance": tap_covariance,
        "tap_reliability": tap_reliability,
        "cotracker_points": cotracker_points,
        "cotracker_valid": cotracker_valid,
        "camera_count": camera_count,
        "reprojection_error": reprojection_error,
        "quality": quality,
    }


def build_complementary_prediction_arrays(
    query_points_world_m: np.ndarray,
    tapnextpp_trajectory_m: np.ndarray,
    tapnextpp_support: np.ndarray,
    tapnextpp_covariance_m2: np.ndarray,
    tapnextpp_reliability: np.ndarray,
    cotracker_points_world_m: np.ndarray,
    cotracker_valid: np.ndarray,
    cotracker_camera_count: np.ndarray,
    cotracker_reprojection_error_px: np.ndarray,
    cotracker_quality_probability: np.ndarray,
    *,
    config: TAPNextPPCoTrackerComplementConfig | None = None,
) -> dict[str, np.ndarray]:
    """Build a target-free causal provider union with exact TAPNext++ retention."""

    cfg = config or TAPNextPPCoTrackerComplementConfig()
    query = np.asarray(query_points_world_m, dtype=np.float64)
    tap_input = np.asarray(tapnextpp_trajectory_m)
    tap = np.asarray(tap_input, dtype=np.float64)
    tap_support = np.asarray(tapnextpp_support, dtype=bool)
    tap_covariance = np.asarray(tapnextpp_covariance_m2, dtype=np.float64)
    tap_reliability = np.asarray(tapnextpp_reliability, dtype=np.float64)
    points = np.asarray(cotracker_points_world_m, dtype=np.float64)
    valid = np.asarray(cotracker_valid, dtype=bool)
    camera_count = np.asarray(cotracker_camera_count)
    reprojection = np.asarray(cotracker_reprojection_error_px, dtype=np.float64)
    quality = np.asarray(cotracker_quality_probability, dtype=np.float64)
    frame_count, identity_count, coordinate_count = tap.shape
    _require(
        coordinate_count == 3
        and frame_count == cfg.prefix_frame_count
        and query.shape == (identity_count, 3),
        "query and TAPNext++ shapes differ from the frozen prefix",
    )
    _require(tap_support.shape == (frame_count, identity_count), "support changed")
    _require(
        tap_covariance.shape == (frame_count, identity_count, 3, 3)
        and tap_reliability.shape == (frame_count, identity_count),
        "TAPNext++ uncertainty shape changed",
    )
    _require(
        points.ndim == 3
        and points.shape[2] == 3
        and valid.shape == points.shape[:2]
        and camera_count.shape == points.shape[:2]
        and reprojection.shape == points.shape[:2]
        and quality.shape == points.shape[:2],
        "CoTracker3 cue shape changed",
    )
    _require(
        cfg.source_frame_end_exclusive <= len(points),
        "CoTracker3 cues do not cover the source prefix",
    )

    candidate = tap_input.copy()
    support = tap_support.copy()
    covariance = tap_covariance.copy()
    reliability = tap_reliability.copy()
    provider_code = np.where(tap_support, 1, 0).astype(np.int8)
    complement_available = np.zeros_like(tap_support)
    complement_trajectory = np.repeat(query[None], frame_count, axis=0)
    complement_covariance = np.zeros((frame_count, identity_count, 3, 3))
    complement_reliability = np.zeros_like(tap_reliability)
    association_neighbor_count = np.zeros(identity_count, dtype=np.int16)
    association_nearest_distance_m = np.full(identity_count, np.inf)
    association_entropy = np.ones(identity_count)
    spatial_spread_m = np.full_like(tap_reliability, np.nan)
    prior_overlap_count = np.zeros_like(tap_support, dtype=np.int16)
    prior_overlap_rmse_m = np.full_like(tap_reliability, np.nan)
    bridge_anchor_frame = np.full_like(tap_support, -1, dtype=np.int16)

    start = cfg.source_frame_start
    stop = cfg.source_frame_end_exclusive
    prefix_points = points[start:stop]
    prefix_valid = valid[start:stop]
    prefix_camera_count = camera_count[start:stop]
    prefix_reprojection = reprojection[start:stop]
    prefix_quality = quality[start:stop]
    base = prefix_points[0]
    source_eligible = (
        prefix_valid[0]
        & (prefix_camera_count[0] >= cfg.minimum_camera_count)
        & (prefix_reprojection[0] <= cfg.maximum_reprojection_error_px)
        & (prefix_quality[0] >= cfg.minimum_quality_probability)
        & np.all(np.isfinite(base), axis=1)
    )

    for identity in range(identity_count):
        distance = np.linalg.norm(base - query[identity], axis=1)
        neighbors = np.flatnonzero(
            source_eligible & (distance <= cfg.association_radius_m)
        )
        neighbors = neighbors[np.argsort(distance[neighbors])][
            : cfg.maximum_association_neighbors
        ]
        association_neighbor_count[identity] = len(neighbors)
        if len(neighbors) < cfg.minimum_neighbor_count:
            continue
        association_nearest_distance_m[identity] = float(
            np.min(distance[neighbors])
        )
        radial_weight = np.exp(
            -0.5
            * np.square(
                distance[neighbors] / cfg.association_kernel_scale_m
            )
        )
        association_entropy[identity] = _association_entropy(radial_weight)

        for frame in range(frame_count):
            frame_valid = (
                prefix_valid[frame, neighbors]
                & (
                    prefix_camera_count[frame, neighbors]
                    >= cfg.minimum_camera_count
                )
                & (
                    prefix_reprojection[frame, neighbors]
                    <= cfg.maximum_reprojection_error_px
                )
                & (
                    prefix_quality[frame, neighbors]
                    >= cfg.minimum_quality_probability
                )
                & np.all(np.isfinite(prefix_points[frame, neighbors]), axis=1)
            )
            if int(np.sum(frame_valid)) < cfg.minimum_neighbor_count:
                continue
            selected = neighbors[frame_valid]
            displacement = prefix_points[frame, selected] - base[selected]
            weights = radial_weight[frame_valid] * np.clip(
                prefix_quality[frame, selected],
                cfg.minimum_quality_probability,
                1.0,
            )
            mean, spread_covariance, spread = _weighted_mean_and_covariance(
                displacement,
                weights,
                robust_scale_m=cfg.robust_displacement_scale_m,
                iterations=cfg.robust_reweight_iterations,
            )
            complement_trajectory[frame, identity] = query[identity] + mean
            spatial_spread_m[frame, identity] = spread
            complement_available[frame, identity] = (
                spread <= cfg.maximum_spatial_spread_m
            )
            shared_variance = (
                cfg.two_view_shared_bias_standard_deviation_m**2
            )
            complement_covariance[frame, identity] = (
                spread_covariance + np.eye(3) * shared_variance
            )
            reprojection_score = float(
                np.exp(
                    -0.5
                    * np.square(
                        np.median(prefix_reprojection[frame, selected])
                        / cfg.maximum_reprojection_error_px
                    )
                )
            )
            quality_score = float(
                np.exp(
                    np.mean(
                        np.log(
                            np.clip(
                                prefix_quality[frame, selected],
                                1e-6,
                                1.0,
                            )
                        )
                    )
                )
            )
            association_score = float(
                np.exp(
                    -0.5
                    * np.square(
                        association_nearest_distance_m[identity]
                        / cfg.association_radius_m
                    )
                )
            )
            complement_reliability[frame, identity] = float(
                np.clip(
                    quality_score * reprojection_score * association_score,
                    0.0,
                    1.0,
                )
            )

        for frame in range(frame_count):
            if tap_support[frame, identity] or not complement_available[
                frame, identity
            ]:
                continue
            history = np.arange(frame)
            overlap = history[
                tap_support[:frame, identity]
                & complement_available[:frame, identity]
            ]
            prior_overlap_count[frame, identity] = len(overlap)
            if len(overlap) < cfg.minimum_prior_overlap_rows:
                continue
            differences = (
                tap[:frame, identity][overlap]
                - complement_trajectory[:frame, identity][overlap]
            )
            agreement_rmse = float(
                np.sqrt(np.mean(np.sum(np.square(differences), axis=1)))
            )
            prior_overlap_rmse_m[frame, identity] = agreement_rmse
            if agreement_rmse > cfg.maximum_prior_overlap_rmse_m:
                continue
            anchor = int(overlap[-1])
            if frame - anchor > cfg.maximum_bridge_gap_frames:
                continue
            bridged = (
                tap[anchor, identity]
                + complement_trajectory[frame, identity]
                - complement_trajectory[anchor, identity]
            )
            candidate[frame, identity] = bridged.astype(candidate.dtype)
            support[frame, identity] = True
            provider_code[frame, identity] = 2
            bridge_anchor_frame[frame, identity] = anchor
            agreement_covariance = (
                differences.T @ differences / max(len(differences), 1)
            )
            covariance[frame, identity] = (
                tap_covariance[anchor, identity]
                + complement_covariance[anchor, identity]
                + complement_covariance[frame, identity]
                + agreement_covariance
            )
            reliability[frame, identity] = float(
                complement_reliability[frame, identity]
                * np.exp(
                    -0.5
                    * np.square(
                        agreement_rmse / cfg.maximum_prior_overlap_rmse_m
                    )
                )
            )

    _require(
        np.array_equal(
            candidate[tap_support],
            tap_input[tap_support],
        ),
        "complement changed an accepted TAPNext++ row",
    )
    _require(
        np.array_equal(support[tap_support], tap_support[tap_support]),
        "complement removed TAPNext++ support",
    )
    return {
        "trajectory_world_m": candidate,
        "accepted_support": support,
        "observation_covariance_m2": covariance.astype(np.float32),
        "observation_reliability": reliability.astype(np.float32),
        "provider_code": provider_code,
        "complement_available": complement_available,
        "complement_trajectory_world_m": complement_trajectory.astype(np.float32),
        "association_neighbor_count": association_neighbor_count,
        "association_nearest_distance_m": association_nearest_distance_m.astype(
            np.float32
        ),
        "association_entropy": association_entropy.astype(np.float32),
        "spatial_spread_m": spatial_spread_m.astype(np.float32),
        "prior_overlap_count": prior_overlap_count,
        "prior_overlap_rmse_m": prior_overlap_rmse_m.astype(np.float32),
        "bridge_anchor_frame": bridge_anchor_frame,
    }


def write_complementary_prediction(
    query_path: str | Path,
    tapnextpp_prediction_path: str | Path,
    tapnextpp_seal_path: str | Path,
    cotracker_cues_path: str | Path,
    output_dir: str | Path,
    *,
    expected_query_sha256: str,
    expected_tapnextpp_prediction_sha256: str,
    expected_tapnextpp_seal_sha256: str,
    expected_cotracker_cues_sha256: str,
    config: TAPNextPPCoTrackerComplementConfig | None = None,
) -> dict[str, Any]:
    """Write a target-free complement prediction from frozen source artifacts."""

    cfg = config or TAPNextPPCoTrackerComplementConfig()
    query = Path(query_path).resolve()
    tapnextpp = Path(tapnextpp_prediction_path).resolve()
    tapnextpp_seal = Path(tapnextpp_seal_path).resolve()
    cotracker = Path(cotracker_cues_path).resolve()
    expected = {
        query: expected_query_sha256,
        tapnextpp: expected_tapnextpp_prediction_sha256,
        tapnextpp_seal: expected_tapnextpp_seal_sha256,
        cotracker: expected_cotracker_cues_sha256,
    }
    for path, digest in expected.items():
        _require(file_sha256(path) == digest, f"input checksum changed: {path}")
    seal = json.loads(tapnextpp_seal.read_text(encoding="utf-8"))
    _require(
        seal.get("artifact_kind") == "PhysTwinTAPNextPPPrefixPredictionSeal",
        "TAPNext++ seal kind changed",
    )
    _require(
        seal.get("prediction_archive_sha256")
        == expected_tapnextpp_prediction_sha256,
        "TAPNext++ seal does not bind the supplied archive",
    )
    arrays = _load_prediction_inputs(query, tapnextpp, cotracker, cfg)
    result = build_complementary_prediction_arrays(
        arrays["query"],
        arrays["tap_trajectory"],
        arrays["tap_support"],
        arrays["tap_covariance"],
        arrays["tap_reliability"],
        arrays["cotracker_points"],
        arrays["cotracker_valid"],
        arrays["camera_count"],
        arrays["reprojection_error"],
        arrays["quality"],
        config=cfg,
    )
    output = Path(output_dir).resolve()
    _require(not output.exists(), "prediction output already exists")
    output.mkdir(parents=True)
    archive_path = output / PREDICTION_FILENAME
    np.savez_compressed(
        archive_path,
        identity_ids=arrays["identity_ids"],
        **result,
    )
    original_support = arrays["tap_support"]
    added_support = result["provider_code"] == 2
    score_rows = np.arange(cfg.prefix_frame_count) > 0
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinTAPNextPPCoTrackerComplementPrediction",
        "protocol_id": PROTOCOL_ID,
        "case": CASE_NAME,
        "config": asdict(cfg),
        "inputs": {
            "query_sha256": expected_query_sha256,
            "tapnextpp_prediction_sha256": expected_tapnextpp_prediction_sha256,
            "tapnextpp_seal_sha256": expected_tapnextpp_seal_sha256,
            "cotracker_cues_sha256": expected_cotracker_cues_sha256,
        },
        "diagnostics": {
            "tapnextpp_supported_point_frames": int(
                np.sum(original_support & score_rows[:, None])
            ),
            "added_point_frames": int(np.sum(added_support & score_rows[:, None])),
            "union_supported_point_frames": int(
                np.sum(result["accepted_support"] & score_rows[:, None])
            ),
            "eligible_point_frames": int(
                len(arrays["identity_ids"]) * np.sum(score_rows)
            ),
            "tapnextpp_rows_bit_identical": bool(
                np.array_equal(
                    result["trajectory_world_m"][original_support],
                    arrays["tap_trajectory"][original_support],
                )
            ),
        },
        "output": {
            "archive": str(archive_path),
            "archive_sha256": file_sha256(archive_path),
        },
        "information_boundary": {
            "manual_query_frame": cfg.source_frame_start,
            "cotracker_frame_range_half_open": [
                cfg.source_frame_start,
                cfg.source_frame_end_exclusive,
            ],
            "withheld_prefix_target_read": False,
            "future_observation_read": False,
            "physical_state_innovation_used_in_prior_reliability": False,
            "complement_requires_prior_cross_provider_agreement": True,
        },
        "claim_boundary": (
            "post-open one-case source provider diagnostic; not simulator "
            "assimilation, independent confirmation, or state-of-the-art evidence"
        ),
    }
    report["result_sha256"] = canonical_sha256(report)
    report_path = output / PREDICTION_REPORT_FILENAME
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def seal_complementary_prediction(prediction_dir: str | Path) -> dict[str, Any]:
    """Seal the complementary provider output before manual-prefix rescoring."""

    prediction = Path(prediction_dir).resolve()
    report_path = prediction / PREDICTION_REPORT_FILENAME
    archive_path = prediction / PREDICTION_FILENAME
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(
        report.get("artifact_kind")
        == "PhysTwinTAPNextPPCoTrackerComplementPrediction",
        "complement report kind changed",
    )
    _require(
        report.get("result_sha256") == canonical_sha256(report),
        "complement report self-hash changed",
    )
    archive_sha256 = file_sha256(archive_path)
    _require(
        report.get("output", {}).get("archive_sha256") == archive_sha256,
        "complement archive changed",
    )
    seal: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinTAPNextPPCoTrackerComplementPredictionSeal",
        "protocol_id": PROTOCOL_ID,
        "case": CASE_NAME,
        "prediction_report_sha256": file_sha256(report_path),
        "prediction_archive_sha256": archive_sha256,
        "prediction_result_sha256": report["result_sha256"],
        "information_boundary": {
            "prediction_hashed_before_manual_prefix_rescoring": True,
            "simulator_future_scoring_authorized": False,
        },
    }
    seal["result_sha256"] = canonical_sha256(seal)
    seal_path = prediction / PREDICTION_SEAL_FILENAME
    _require(not seal_path.exists(), "complement seal already exists")
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
    difference = (
        np.asarray(prediction, dtype=np.float64)
        - np.asarray(target, dtype=np.float64)
    )[np.asarray(mask, dtype=bool)]
    if len(difference) == 0:
        return None
    return float(np.sqrt(np.mean(np.sum(np.square(difference), axis=1))))


def evaluate_complementary_prediction(
    prediction_dir: str | Path,
    withheld_prefix_path: str | Path,
    output_path: str | Path,
    *,
    expected_withheld_sha256: str,
    config: TAPNextPPCoTrackerComplementConfig | None = None,
) -> dict[str, Any]:
    """Score the sealed complement on the already-open staged prefix target."""

    cfg = config or TAPNextPPCoTrackerComplementConfig()
    prediction = Path(prediction_dir).resolve()
    report_path = prediction / PREDICTION_REPORT_FILENAME
    archive_path = prediction / PREDICTION_FILENAME
    seal_path = prediction / PREDICTION_SEAL_FILENAME
    report = json.loads(report_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    _require(
        seal.get("artifact_kind")
        == "PhysTwinTAPNextPPCoTrackerComplementPredictionSeal",
        "complement seal kind changed",
    )
    _require(
        seal.get("result_sha256") == canonical_sha256(seal),
        "complement seal self-hash changed",
    )
    _require(
        seal.get("prediction_report_sha256") == file_sha256(report_path)
        and seal.get("prediction_archive_sha256") == file_sha256(archive_path),
        "sealed complement files changed",
    )
    target_path = Path(withheld_prefix_path).resolve()
    _require(
        file_sha256(target_path) == expected_withheld_sha256,
        "withheld prefix checksum changed",
    )
    with np.load(archive_path, allow_pickle=False) as stored:
        candidate = np.asarray(stored["trajectory_world_m"])
        support = np.asarray(stored["accepted_support"], dtype=bool)
        provider_code = np.asarray(stored["provider_code"], dtype=np.int8)
        identity_ids = np.asarray(stored["identity_ids"], dtype=np.int64)
    with np.load(target_path, allow_pickle=False) as stored:
        target = np.asarray(stored["target_tracks_world_m"], dtype=np.float64)
        target_ids = np.asarray(stored["identity_ids"], dtype=np.int64)
        source_start = int(stored["source_frame_start"])
        source_stop = int(stored["source_frame_end_exclusive"])
    _require(candidate.shape == target.shape, "candidate and target shapes differ")
    _require(support.shape == target.shape[:2], "support shape changed")
    _require(provider_code.shape == support.shape, "provider-code shape changed")
    _require(np.array_equal(identity_ids, target_ids), "identity order changed")
    _require(
        (source_start, source_stop)
        == (cfg.source_frame_start, cfg.source_frame_end_exclusive),
        "withheld prefix interval changed",
    )
    target_valid = np.all(np.isfinite(target), axis=2)
    score_rows = np.arange(cfg.prefix_frame_count) > 0
    eligible = target_valid & score_rows[:, None]
    scored = support & eligible
    added = (provider_code == 2) & eligible
    tap_rows = (provider_code == 1) & eligible
    persistence = np.repeat(target[:1], cfg.prefix_frame_count, axis=0)
    candidate_rmse = _radial_rmse_m(candidate, target, scored)
    persistence_rmse = _radial_rmse_m(persistence, target, scored)
    added_candidate_rmse = _radial_rmse_m(candidate, target, added)
    added_persistence_rmse = _radial_rmse_m(persistence, target, added)
    added_gain = (
        (added_persistence_rmse - added_candidate_rmse) / added_persistence_rmse
        if added_candidate_rmse is not None
        and added_persistence_rmse is not None
        and added_persistence_rmse > 0.0
        else None
    )
    endpoint_rows = np.arange(cfg.prefix_frame_count) >= (
        cfg.prefix_frame_count - cfg.endpoint_frame_count
    )
    endpoint = support & target_valid & endpoint_rows[:, None]
    endpoint_rmse = _radial_rmse_m(candidate, target, endpoint)
    supported_fraction = float(np.sum(scored) / max(np.sum(eligible), 1))
    tap_rows_unchanged = bool(
        report.get("diagnostics", {}).get("tapnextpp_rows_bit_identical")
    )
    gates = {
        "supported_fraction": (
            supported_fraction >= cfg.minimum_supported_fraction
        ),
        "identity_rmse": (
            candidate_rmse is not None
            and candidate_rmse <= cfg.maximum_identity_rmse_m
        ),
        "endpoint_rmse": (
            endpoint_rmse is not None
            and endpoint_rmse <= cfg.maximum_endpoint_rmse_m
        ),
        "added_row_gain": (
            added_gain is not None
            and added_gain >= cfg.minimum_added_row_gain_over_persistence
        ),
        "tapnextpp_rows_bit_identical": tap_rows_unchanged,
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinTAPNextPPCoTrackerComplementEvaluation",
        "protocol_id": PROTOCOL_ID,
        "case": CASE_NAME,
        "config": asdict(cfg),
        "metrics": {
            "supported_fraction": supported_fraction,
            "supported_point_frame_count": int(np.sum(scored)),
            "eligible_point_frame_count": int(np.sum(eligible)),
            "tapnextpp_point_frame_count": int(np.sum(tap_rows)),
            "added_point_frame_count": int(np.sum(added)),
            "candidate_identity_rmse_m": candidate_rmse,
            "persistence_identity_rmse_m": persistence_rmse,
            "candidate_endpoint_rmse_m": endpoint_rmse,
            "added_candidate_rmse_m": added_candidate_rmse,
            "added_persistence_rmse_m": added_persistence_rmse,
            "added_relative_gain_over_persistence": added_gain,
        },
        "gates": gates,
        "competence_gate_passed": all(gates.values()),
        "decision": (
            "provider competence rescued; a separately locked disjoint-identity "
            "assimilation study may be designed"
            if all(gates.values())
            else "complementary provider route remains closed"
        ),
        "prediction": {
            "report_sha256": file_sha256(report_path),
            "archive_sha256": file_sha256(archive_path),
            "seal_sha256": file_sha256(seal_path),
        },
        "withheld_prefix_sha256": expected_withheld_sha256,
        "information_boundary": {
            "already_open_source_prefix_only": True,
            "simulator_future_opened": False,
            "held_v8_opened": False,
        },
        "claim_boundary": (
            "post-open one-case source provider diagnostic; a pass is not "
            "Bayesian-PhysTwin improvement, independent transfer, or SOTA"
        ),
    }
    result["result_sha256"] = canonical_sha256(result)
    destination = Path(output_path).resolve()
    _require(not destination.exists(), "evaluation output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
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
    "TAPNextPPCoTrackerComplementConfig",
    "build_complementary_prediction_arrays",
    "evaluate_complementary_prediction",
    "seal_complementary_prediction",
    "write_complementary_prediction",
]
