"""Frozen source evaluation for the dynamic TAPNext++ provider.

Provider competence and downstream assimilation are deliberately separate.
The former asks whether a dynamically born identity is tracked accurately and
with calibrated covariance. The latter scores only material identities that
were never queried and applies a leave-two-object source regret certificate
with exact interval fallback.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .bias_aware_belief import (
    SourceRegretCertificate,
    fit_source_regret_certificate,
)
from .deform360_dynamic_tapnextpp_assimilation import (
    CANDIDATE_ARM,
    PERSISTENCE_ARM,
    PHYSICAL_ARM,
    SELECTED_BACKBONE_ARM,
    UPDATE_FRAMES,
)
from .phystwin_official_evaluation import _nearest_distances
from .tapnextpp_dynamic_multiview import PROTOCOL_ID

SOURCE_EVALUATION_KIND = (
    "Deform360DynamicTAPNextPPSourceEvaluationProtocol"
)
SOURCE_EVALUATION_FILENAME = (
    "deform360_dynamic_tapnextpp_source_evaluation_v1.json"
)
SOURCE_OBJECT_COUNT = 8
EXPECTED_QUERY_COUNT = 72
PROVIDER_LATE_FRAME_COUNT = 5
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 720
SCORED_RANGES = ((20, 38), (39, 57), (58, 76))
LATE_RANGES = ((32, 38), (51, 57), (70, 76))
REGRET_FEATURE_NAMES = (
    "available_measurement_fraction",
    "pairwise_inlier_fraction",
    "mean_prior_reliability",
    "candidate_correction_to_backbone_motion_ratio",
)
PRIMARY_METRICS = (
    "hidden_identity_rmse_m",
    "hidden_symmetric_chamfer_m",
)
SECONDARY_METRICS = (
    "late_hidden_identity_rmse_m",
    "hidden_target_to_prediction_chamfer_m",
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _valid_digest(value: Any, *, length: int = 64) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_sha256(
    payload: Mapping[str, Any],
    *,
    digest_key: str,
) -> str:
    canonical = dict(payload)
    canonical.pop(digest_key, None)
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def load_source_evaluation_protocol(
    path: str | Path,
) -> dict[str, Any]:
    """Validate the exact source-scoring contract frozen with the cohort."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(
        isinstance(payload, dict)
        and payload.get("schema_version") == 1
        and payload.get("artifact_kind") == SOURCE_EVALUATION_KIND
        and payload.get("protocol_id") == PROTOCOL_ID
        and payload.get("status")
        == "locked-before-cohort-and-provider-outcomes",
        "source evaluation protocol is incompatible",
    )
    provider = payload.get("provider_competence", {})
    calibration = provider.get("calibration", {})
    assimilation = payload.get("assimilation", {})
    cross_fitting = assimilation.get("cross_fitting", {})
    aggregation = payload.get("aggregation", {})
    _require(
        provider.get("primary_error")
        == "coordinate RMSE at the registered update"
        and provider.get("late_error")
        == (
            "coordinate RMSE over the final five frames ending at each "
            "registered update"
        )
        and calibration.get("within_object_score_quantile") == 0.9
        and assimilation.get("scored_ranges_half_open")
        == [list(values) for values in SCORED_RANGES]
        and assimilation.get("late_ranges_half_open")
        == [list(values) for values in LATE_RANGES]
        and assimilation.get("primary_metrics") == list(PRIMARY_METRICS)
        and assimilation.get("secondary_metrics") == list(SECONDARY_METRICS)
        and assimilation.get("regret_features")
        == list(REGRET_FEATURE_NAMES)
        and cross_fitting.get("ridge_penalty") == 10.0
        and cross_fitting.get("nominal_upper_coverage") == 0.9
        and cross_fitting.get("within_object_coverage") == 1.0
        and cross_fitting.get("support_margin_standard_deviations") == 0.0
        and aggregation.get("bootstrap_draws") == BOOTSTRAP_DRAWS
        and aggregation.get("bootstrap_seed") == BOOTSTRAP_SEED,
        "source evaluation settings changed",
    )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get(
            "source_barrier_required_before_future_deserialization"
        )
        is True
        and boundary.get(
            "all_prediction_seals_validated_before_first_future_deserialization"
        )
        is True
        and boundary.get("target_artifacts_opened") is False
        and boundary.get(
            "held_v8_target_query_score_barrier_or_outcome_access"
        )
        is False,
        "source evaluation crossed its information boundary",
    )
    return payload


def _coordinate_rmse_m(
    predicted_m: np.ndarray,
    target_m: np.ndarray,
) -> float:
    predicted = np.asarray(predicted_m, dtype=np.float64)
    target = np.asarray(target_m, dtype=np.float64)
    _require(
        predicted.shape == target.shape
        and predicted.ndim == 2
        and predicted.shape[1] == 3
        and len(predicted) > 0
        and np.all(np.isfinite(predicted))
        and np.all(np.isfinite(target)),
        "RMSE inputs must have finite nonempty shape (N, 3)",
    )
    return float(np.sqrt(np.mean(np.square(predicted - target))))


def _symmetric_chamfer_m(
    predicted_m: np.ndarray,
    target_m: np.ndarray,
) -> tuple[float, float]:
    predicted = np.asarray(predicted_m, dtype=np.float64)
    target = np.asarray(target_m, dtype=np.float64)
    _require(
        predicted.ndim == target.ndim == 2
        and predicted.shape[1:] == target.shape[1:] == (3,)
        and len(predicted) > 0
        and len(target) > 0,
        "Chamfer inputs must have nonempty shape (N, 3)",
    )
    prediction_to_target, _ = _nearest_distances(
        target,
        predicted,
        p=2,
    )
    target_to_prediction, _ = _nearest_distances(
        predicted,
        target,
        p=2,
    )
    target_to_prediction_mean = float(np.mean(target_to_prediction))
    return (
        0.5
        * (
            float(np.mean(prediction_to_target))
            + target_to_prediction_mean
        ),
        target_to_prediction_mean,
    )


def _finite_sample_quantile(
    values: np.ndarray,
    coverage: float,
) -> tuple[float, int]:
    observed = np.asarray(values, dtype=np.float64)
    _require(
        observed.ndim == 1
        and len(observed) > 0
        and np.all(np.isfinite(observed))
        and 0.0 < coverage <= 1.0,
        "finite-sample quantile inputs are invalid",
    )
    rank = min(len(observed), int(np.ceil((len(observed) + 1) * coverage)))
    return float(np.partition(observed, rank - 1)[rank - 1]), rank


def score_provider_case_arrays(
    *,
    trajectory_world_m: np.ndarray,
    accepted_support: np.ndarray,
    local_covariance_m2: np.ndarray,
    shared_bias_standard_deviation_m: float,
    target_m: np.ndarray,
    target_visibility: np.ndarray,
    target_validity: np.ndarray,
    entity_ids: np.ndarray,
    birth_frames: np.ndarray,
    update_frames: np.ndarray,
) -> dict[str, Any]:
    """Score one provider without consuming queried identities downstream."""

    trajectory = np.asarray(trajectory_world_m, dtype=np.float64)
    accepted = np.asarray(accepted_support, dtype=bool)
    covariance = np.asarray(local_covariance_m2, dtype=np.float64)
    target = np.asarray(target_m, dtype=np.float64)
    visibility = np.asarray(target_visibility, dtype=bool)
    validity = np.asarray(target_validity, dtype=bool)
    entities = np.asarray(entity_ids, dtype=np.int64)
    births = np.asarray(birth_frames, dtype=np.int64)
    updates = np.asarray(update_frames, dtype=np.int64)
    _require(
        trajectory.ndim == 3
        and trajectory.shape[2] == 3
        and accepted.shape == trajectory.shape[:2]
        and covariance.shape == (*trajectory.shape[:2], 3, 3),
        "provider arrays changed shape",
    )
    _require(
        target.ndim == 3
        and target.shape[0] == 76
        and target.shape[2] == 3
        and visibility.shape == validity.shape == target.shape[:2],
        "target arrays changed shape",
    )
    _require(
        entities.shape == births.shape == updates.shape
        == (trajectory.shape[1],)
        and len(entities) == EXPECTED_QUERY_COUNT
        and len(np.unique(entities)) == len(entities)
        and np.all((entities >= 0) & (entities < target.shape[1]))
        and np.all((births >= 0) & (births <= updates))
        and np.all(updates < trajectory.shape[0]),
        "provider schedule changed",
    )
    _require(
        np.isfinite(shared_bias_standard_deviation_m)
        and shared_bias_standard_deviation_m > 0.0,
        "shared bias scale is invalid",
    )

    endpoint_rows: list[int] = []
    late_frames: list[int] = []
    late_rows: list[int] = []
    for row, (entity, birth, update) in enumerate(
        zip(entities, births, updates, strict=True)
    ):
        endpoint_ok = bool(
            accepted[birth, row]
            and accepted[update, row]
            and visibility[birth, entity]
            and validity[birth, entity]
            and visibility[update, entity]
            and validity[update, entity]
            and np.all(np.isfinite(target[[birth, update], entity]))
        )
        if endpoint_ok:
            endpoint_rows.append(row)
        for frame in range(
            max(int(birth), int(update) - PROVIDER_LATE_FRAME_COUNT + 1),
            int(update) + 1,
        ):
            if (
                accepted[birth, row]
                and accepted[frame, row]
                and visibility[birth, entity]
                and validity[birth, entity]
                and visibility[frame, entity]
                and validity[frame, entity]
                and np.all(np.isfinite(target[[birth, frame], entity]))
            ):
                late_frames.append(frame)
                late_rows.append(row)

    supported_fraction = len(endpoint_rows) / len(entities)
    if not endpoint_rows:
        return {
            "supported_identity_count": 0,
            "scheduled_identity_count": int(len(entities)),
            "supported_fraction": supported_fraction,
            "provider_rmse_m": None,
            "persistence_rmse_m": None,
            "relative_gain_over_persistence": None,
            "late_provider_rmse_m": None,
            "late_persistence_rmse_m": None,
            "provider_wins": False,
            "mahalanobis_squared": [],
            "within_object_90pct_score": None,
            "within_object_quantile_rank": None,
        }

    endpoint = np.asarray(endpoint_rows, dtype=np.int64)
    endpoint_entities = entities[endpoint]
    endpoint_updates = updates[endpoint]
    endpoint_births = births[endpoint]
    provider_endpoint = trajectory[endpoint_updates, endpoint]
    persistence_endpoint = trajectory[endpoint_births, endpoint]
    target_endpoint = target[endpoint_updates, endpoint_entities]
    provider_rmse = _coordinate_rmse_m(provider_endpoint, target_endpoint)
    persistence_rmse = _coordinate_rmse_m(
        persistence_endpoint,
        target_endpoint,
    )
    gain = (
        None
        if persistence_rmse <= 0.0
        else 1.0 - provider_rmse / persistence_rmse
    )

    late_index = np.asarray(late_rows, dtype=np.int64)
    late_frame = np.asarray(late_frames, dtype=np.int64)
    late_entities = entities[late_index]
    late_provider = trajectory[late_frame, late_index]
    late_persistence = trajectory[births[late_index], late_index]
    late_target = target[late_frame, late_entities]
    late_provider_rmse = _coordinate_rmse_m(late_provider, late_target)
    late_persistence_rmse = _coordinate_rmse_m(
        late_persistence,
        late_target,
    )

    residual = provider_endpoint - target_endpoint
    endpoint_covariance = covariance[endpoint_updates, endpoint].copy()
    endpoint_covariance += (
        shared_bias_standard_deviation_m**2 * np.eye(3)[None]
    )
    mahalanobis = np.asarray(
        [
            float(delta @ np.linalg.solve(matrix, delta))
            for delta, matrix in zip(
                residual,
                endpoint_covariance,
                strict=True,
            )
        ]
    )
    within_score, within_rank = _finite_sample_quantile(mahalanobis, 0.9)
    return {
        "supported_identity_count": int(len(endpoint)),
        "scheduled_identity_count": int(len(entities)),
        "supported_fraction": float(supported_fraction),
        "provider_rmse_m": provider_rmse,
        "persistence_rmse_m": persistence_rmse,
        "relative_gain_over_persistence": gain,
        "late_provider_rmse_m": late_provider_rmse,
        "late_persistence_rmse_m": late_persistence_rmse,
        "provider_wins": bool(
            provider_rmse < persistence_rmse
            and late_provider_rmse < late_persistence_rmse
        ),
        "mahalanobis_squared": mahalanobis.tolist(),
        "within_object_90pct_score": within_score,
        "within_object_quantile_rank": within_rank,
    }


def aggregate_provider_source_gate(
    reports: Sequence[Mapping[str, Any]],
    *,
    expected_source_count: int = SOURCE_OBJECT_COUNT,
) -> dict[str, Any]:
    """Apply the frozen competence and covariance gates by physical object."""

    rows = [dict(report) for report in reports]
    _require(
        len(rows) == expected_source_count
        and len({row["object_hash"] for row in rows}) == expected_source_count,
        "provider source reports must cover distinct source objects",
    )
    scored = [
        row
        for row in rows
        if row.get("technical_failure") is False
        and row.get("provider_rmse_m") is not None
    ]
    supported_fraction = float(
        np.mean(
            [
                0.0
                if row.get("technical_failure")
                else float(row["supported_fraction"])
                for row in rows
            ]
        )
    )
    support_pass_count = sum(
        not row.get("technical_failure")
        and float(row.get("supported_fraction", 0.0)) >= 0.75
        for row in rows
    )
    provider_rmse: float | None = (
        float(np.mean([float(row["provider_rmse_m"]) for row in scored]))
        if scored
        else None
    )
    late_rmse: float | None = (
        float(np.mean([float(row["late_provider_rmse_m"]) for row in scored]))
        if scored
        else None
    )
    persistence_rmse = (
        float(np.mean([float(row["persistence_rmse_m"]) for row in scored]))
        if scored
        else 0.0
    )
    relative_gain: float | None = (
        None
        if provider_rmse is None or persistence_rmse <= 0.0
        else 1.0 - provider_rmse / persistence_rmse
    )
    wins = sum(bool(row.get("provider_wins")) for row in rows)

    calibration_records: list[dict[str, Any]] = []
    for held in scored:
        calibration_scores = [
            float(row["within_object_90pct_score"])
            for row in scored
            if row["object_hash"] != held["object_hash"]
        ]
        if not calibration_scores:
            coverage = 0.0
            threshold = None
        else:
            threshold = float(np.max(calibration_scores))
            values = np.asarray(held["mahalanobis_squared"], dtype=np.float64)
            coverage = float(np.mean(values <= threshold))
        calibration_records.append(
            {
                "object_hash": held["object_hash"],
                "threshold": threshold,
                "coverage": coverage,
            }
        )
    coverage = (
        float(
            np.mean(
                [
                    value
                    for row in calibration_records
                    for value in [row["coverage"]]
                ]
            )
        )
        if calibration_records
        else 0.0
    )
    worst_coverage = (
        float(np.min([row["coverage"] for row in calibration_records]))
        if calibration_records
        else 0.0
    )
    checks = {
        "object_balanced_supported_fraction": supported_fraction >= 0.75,
        "cases_meeting_support_gate": support_pass_count >= 6,
        "provider_rmse": (
            provider_rmse is not None and provider_rmse <= 0.015
        ),
        "late_provider_rmse": (
            late_rmse is not None and late_rmse <= 0.015
        ),
        "relative_gain_over_persistence": (
            relative_gain is not None and relative_gain >= 0.10
        ),
        "case_wins": wins >= 6,
        "out_of_fold_90pct_coverage": 0.80 <= coverage <= 1.0,
        "worst_object_coverage": worst_coverage >= 0.70,
    }
    return {
        "expected_source_count": expected_source_count,
        "ordinary_scored_count": len(scored),
        "technical_failure_count": sum(
            bool(row.get("technical_failure")) for row in rows
        ),
        "object_balanced_supported_fraction": supported_fraction,
        "cases_meeting_support_gate": support_pass_count,
        "provider_rmse_m": provider_rmse,
        "late_provider_rmse_m": late_rmse,
        "persistence_rmse_m": persistence_rmse,
        "relative_gain_over_persistence": relative_gain,
        "case_wins": wins,
        "calibration": {
            "records": calibration_records,
            "out_of_fold_coverage": coverage,
            "worst_object_coverage": worst_coverage,
            "frozen_target_ellipsoid_threshold": (
                float(
                    np.max(
                        [
                            float(row["within_object_90pct_score"])
                            for row in scored
                        ]
                    )
                )
                if scored
                else None
            ),
            "finite_sample_threshold_rule": (
                "maximum other-object within-object 90% score"
            ),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _frames_from_ranges(
    ranges: Sequence[Sequence[int]],
) -> tuple[int, ...]:
    return tuple(
        frame
        for start, stop in ranges
        for frame in range(int(start), int(stop))
    )


def _score_hidden_frames(
    trajectory_m: np.ndarray,
    target_m: np.ndarray,
    visibility: np.ndarray,
    validity: np.ndarray,
    hidden_entity_ids: np.ndarray,
    frames: Sequence[int],
) -> dict[str, float]:
    trajectory = np.asarray(trajectory_m, dtype=np.float64)
    target = np.asarray(target_m, dtype=np.float64)
    visible = np.asarray(visibility, dtype=bool)
    valid = np.asarray(validity, dtype=bool)
    hidden = np.asarray(hidden_entity_ids, dtype=np.int64)
    _require(
        trajectory.shape == target.shape
        and target.ndim == 3
        and target.shape[0] == 76
        and target.shape[2] == 3,
        "hidden-score trajectories changed shape",
    )
    _require(
        visible.shape == valid.shape == target.shape[:2]
        and hidden.ndim == 1
        and len(hidden) > 0
        and len(np.unique(hidden)) == len(hidden)
        and np.all((hidden >= 0) & (hidden < target.shape[1])),
        "hidden-score identity contract changed",
    )
    identity: list[float] = []
    symmetric: list[float] = []
    target_to_prediction: list[float] = []
    for frame in frames:
        _require(0 <= int(frame) < len(target), "scored frame is invalid")
        supported = (
            visible[frame, hidden]
            & valid[frame, hidden]
            & np.all(np.isfinite(target[frame, hidden]), axis=1)
            & np.all(np.isfinite(trajectory[frame, hidden]), axis=1)
        )
        _require(np.any(supported), "scored frame has no hidden identity")
        predicted = trajectory[frame, hidden[supported]]
        observed = target[frame, hidden[supported]]
        identity.append(_coordinate_rmse_m(predicted, observed))
        chamfer, one_sided = _symmetric_chamfer_m(predicted, observed)
        symmetric.append(chamfer)
        target_to_prediction.append(one_sided)
    return {
        "identity_rmse_m": float(np.mean(identity)),
        "symmetric_chamfer_m": float(np.mean(symmetric)),
        "target_to_prediction_chamfer_m": float(
            np.mean(target_to_prediction)
        ),
    }


def score_assimilation_trajectory(
    trajectory_m: np.ndarray,
    target_m: np.ndarray,
    visibility: np.ndarray,
    validity: np.ndarray,
    hidden_entity_ids: np.ndarray,
) -> dict[str, float]:
    """Score one complete candidate on permanently hidden identities."""

    primary = _score_hidden_frames(
        trajectory_m,
        target_m,
        visibility,
        validity,
        hidden_entity_ids,
        _frames_from_ranges(SCORED_RANGES),
    )
    late = _score_hidden_frames(
        trajectory_m,
        target_m,
        visibility,
        validity,
        hidden_entity_ids,
        _frames_from_ranges(LATE_RANGES),
    )
    return {
        "hidden_identity_rmse_m": primary["identity_rmse_m"],
        "hidden_symmetric_chamfer_m": primary["symmetric_chamfer_m"],
        "late_hidden_identity_rmse_m": late["identity_rmse_m"],
        "hidden_target_to_prediction_chamfer_m": primary[
            "target_to_prediction_chamfer_m"
        ],
    }


def _interval_score(
    trajectory_m: np.ndarray,
    target_m: np.ndarray,
    visibility: np.ndarray,
    validity: np.ndarray,
    hidden_entity_ids: np.ndarray,
    interval: tuple[int, int],
) -> tuple[float, float]:
    values = _score_hidden_frames(
        trajectory_m,
        target_m,
        visibility,
        validity,
        hidden_entity_ids,
        tuple(range(*interval)),
    )
    return values["identity_rmse_m"], values["symmetric_chamfer_m"]


def interval_regret_features(
    report: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    interval_index: int,
) -> np.ndarray:
    """Extract the four frozen target-free guard features."""

    _require(
        0 <= interval_index < len(SCORED_RANGES),
        "interval index is invalid",
    )
    updates = report.get("updates", [])
    _require(
        isinstance(updates, list) and len(updates) == len(SCORED_RANGES),
        "assimilation update report changed",
    )
    row = updates[interval_index]
    baseline = np.asarray(arrays[SELECTED_BACKBONE_ARM], dtype=np.float64)
    candidate = np.asarray(arrays[CANDIDATE_ARM], dtype=np.float64)
    _require(
        baseline.shape == candidate.shape
        and baseline.ndim == 3
        and baseline.shape[0] == 76
        and baseline.shape[2] == 3,
        "assimilation feature trajectories changed shape",
    )
    center_count = len(report.get("center_ids", []))
    _require(center_count == EXPECTED_QUERY_COUNT, "assimilation centres changed")
    start, stop = SCORED_RANGES[interval_index]
    update = UPDATE_FRAMES[interval_index]
    correction_rms = float(
        np.sqrt(np.mean(np.square(candidate[start:stop] - baseline[start:stop])))
    )
    backbone_motion_rms = float(
        np.sqrt(
            np.mean(
                np.square(
                    baseline[start:stop]
                    - baseline[update][None]
                )
            )
        )
    )
    pairwise = row.get("pairwise_gate", {})
    return np.asarray(
        [
            float(row["available_center_count"]) / center_count,
            float(pairwise["inlier_fraction"]),
            float(row["mean_prior_reliability"]),
            correction_rms / max(backbone_motion_rms, 0.005),
        ],
        dtype=np.float64,
    )


def _certificate_descriptor(
    certificate: SourceRegretCertificate,
) -> dict[str, Any]:
    return {
        **asdict(certificate),
        "feature_center": certificate.feature_center.tolist(),
        "feature_scale": certificate.feature_scale.tolist(),
        "standardized_feature_lower": (
            certificate.standardized_feature_lower.tolist()
        ),
        "standardized_feature_upper": (
            certificate.standardized_feature_upper.tolist()
        ),
        "coefficients": certificate.coefficients.tolist(),
    }


def crossfit_guarded_assimilation(
    cases: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Apply deterministic leave-two-object regret guards to eight sources."""

    rows = [dict(case) for case in cases]
    _require(
        len(rows) == SOURCE_OBJECT_COUNT
        and len({row["object_hash"] for row in rows}) == SOURCE_OBJECT_COUNT,
        "cross-fitting requires eight distinct source objects",
    )
    ordinary = [row for row in rows if not row.get("technical_failure")]
    _require(len(ordinary) >= 6, "too few ordinary sources for assimilation")

    interval_rows: list[dict[str, Any]] = []
    for case_index, row in enumerate(rows):
        if row.get("technical_failure"):
            continue
        arrays = row["arrays"]
        report = row["assimilation_report"]
        baseline = np.asarray(arrays[SELECTED_BACKBONE_ARM])
        candidate = np.asarray(arrays[CANDIDATE_ARM])
        for interval_index, interval in enumerate(SCORED_RANGES):
            baseline_identity, baseline_chamfer = _interval_score(
                baseline,
                row["target_m"],
                row["visibility"],
                row["validity"],
                row["hidden_entity_ids"],
                interval,
            )
            candidate_identity, candidate_chamfer = _interval_score(
                candidate,
                row["target_m"],
                row["visibility"],
                row["validity"],
                row["hidden_entity_ids"],
                interval,
            )
            interval_rows.append(
                {
                    "case_index": case_index,
                    "interval_index": interval_index,
                    "object_hash": row["object_hash"],
                    "features": interval_regret_features(
                        report,
                        arrays,
                        interval_index,
                    ),
                    "regret_m": max(
                        candidate_identity - baseline_identity,
                        candidate_chamfer - baseline_chamfer,
                    ),
                    "raw_candidate_identity_regret_m": (
                        candidate_identity - baseline_identity
                    ),
                    "raw_candidate_chamfer_regret_m": (
                        candidate_chamfer - baseline_chamfer
                    ),
                    "pairwise_accepted": bool(
                        report["updates"][interval_index][
                            "pairwise_gate"
                        ]["accepted"]
                    ),
                }
            )

    output: list[dict[str, Any]] = []
    for fold_start in range(0, SOURCE_OBJECT_COUNT, 2):
        held_case_indices = {fold_start, fold_start + 1}
        training = [
            row
            for row in interval_rows
            if row["case_index"] not in held_case_indices
        ]
        held = [
            row
            for row in interval_rows
            if row["case_index"] in held_case_indices
        ]
        _require(
            len({row["object_hash"] for row in training}) >= 3,
            "cross-fit training lost source objects",
        )
        certificate = fit_source_regret_certificate(
            np.stack([row["features"] for row in training]),
            np.asarray([row["regret_m"] for row in training]),
            [row["object_hash"] for row in training],
            nominal_coverage=0.9,
            within_group_coverage=1.0,
            minimum_improvement=0.0,
            ridge_penalty=10.0,
            support_margin_std=0.0,
        )
        for row in held:
            upper = certificate.upper_regret(row["features"])
            accepted = bool(
                row["pairwise_accepted"]
                and np.isfinite(upper)
                and upper < 0.0
            )
            output.append(
                {
                    **row,
                    "guard_accepted": accepted,
                    "predicted_regret_m": certificate.predict_regret(
                        row["features"]
                    ),
                    "upper_regret_m": upper,
                    "certificate": _certificate_descriptor(certificate),
                }
            )
    _require(
        len(output) == len(interval_rows),
        "cross-fitting did not score every ordinary interval",
    )
    return output


def _cluster_bootstrap(
    differences: Sequence[float],
) -> dict[str, float]:
    values = np.asarray(differences, dtype=np.float64)
    _require(
        values.shape == (SOURCE_OBJECT_COUNT,)
        and np.all(np.isfinite(values)),
        "bootstrap requires one finite difference per source object",
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    selected = rng.integers(
        0,
        len(values),
        size=(BOOTSTRAP_DRAWS, len(values)),
    )
    means = np.mean(values[selected], axis=1)
    return {
        "object_balanced_mean_difference_m": float(np.mean(values)),
        "object_cluster_lower_95_m": float(np.quantile(means, 0.025)),
        "object_cluster_upper_95_m": float(np.quantile(means, 0.975)),
        "probability_improved": float(np.mean(means < 0.0)),
    }


def evaluate_guarded_assimilation_gate(
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Cross-fit, score, and gate the dynamic discrepancy candidate."""

    rows = [dict(case) for case in cases]
    decisions = crossfit_guarded_assimilation(rows)
    final_certificate = fit_source_regret_certificate(
        np.stack([row["features"] for row in decisions]),
        np.asarray([row["regret_m"] for row in decisions]),
        [row["object_hash"] for row in decisions],
        nominal_coverage=0.9,
        within_group_coverage=1.0,
        minimum_improvement=0.0,
        ridge_penalty=10.0,
        support_margin_std=0.0,
    )
    by_case_interval = {
        (int(row["case_index"]), int(row["interval_index"])): row
        for row in decisions
    }
    case_reports: list[dict[str, Any]] = []
    for case_index, row in enumerate(rows):
        if row.get("technical_failure"):
            case_reports.append(
                {
                    "object_hash": row["object_hash"],
                    "technical_failure": True,
                    "scores": None,
                    "decisions": [],
                }
            )
            continue
        arrays = row["arrays"]
        baseline = np.asarray(arrays[SELECTED_BACKBONE_ARM])
        guarded = baseline.copy()
        interval_decisions = []
        for interval_index, (start, stop) in enumerate(SCORED_RANGES):
            decision = by_case_interval[(case_index, interval_index)]
            if decision["guard_accepted"]:
                guarded[start:stop] = arrays[CANDIDATE_ARM][start:stop]
            else:
                _require(
                    np.array_equal(guarded[start:stop], baseline[start:stop]),
                    "rejected guard changed the selected backbone",
                )
            interval_decisions.append(
                {
                    key: (
                        value.tolist()
                        if isinstance(value, np.ndarray)
                        else (
                            None
                            if isinstance(value, float)
                            and not np.isfinite(value)
                            else value
                        )
                    )
                    for key, value in decision.items()
                    if key not in {"certificate"}
                }
                | {"certificate": decision["certificate"]}
            )
        trajectories = {
            PHYSICAL_ARM: np.asarray(arrays[PHYSICAL_ARM]),
            PERSISTENCE_ARM: np.asarray(arrays[PERSISTENCE_ARM]),
            SELECTED_BACKBONE_ARM: baseline,
            CANDIDATE_ARM: np.asarray(arrays[CANDIDATE_ARM]),
            "guarded_candidate": guarded,
        }
        scores = {
            name: score_assimilation_trajectory(
                trajectory,
                row["target_m"],
                row["visibility"],
                row["validity"],
                row["hidden_entity_ids"],
            )
            for name, trajectory in trajectories.items()
        }
        case_reports.append(
            {
                "object_hash": row["object_hash"],
                "technical_failure": False,
                "scores": scores,
                "decisions": interval_decisions,
            }
        )

    ordinary = [row for row in case_reports if not row["technical_failure"]]
    comparators = (PHYSICAL_ARM, PERSISTENCE_ARM, SELECTED_BACKBONE_ARM)
    comparisons: dict[str, Any] = {}
    all_primary_checks: list[bool] = []
    for comparator in comparators:
        metric_reports: dict[str, Any] = {}
        comparator_checks: list[bool] = []
        for metric in PRIMARY_METRICS:
            differences = [
                0.0
                if row["technical_failure"]
                else float(
                    row["scores"]["guarded_candidate"][metric]
                    - row["scores"][comparator][metric]
                )
                for row in case_reports
            ]
            bootstrap = _cluster_bootstrap(differences)
            candidate_mean = float(
                np.mean(
                    [
                        row["scores"]["guarded_candidate"][metric]
                        for row in ordinary
                    ]
                )
            )
            comparator_mean = float(
                np.mean(
                    [row["scores"][comparator][metric] for row in ordinary]
                )
            )
            relative_improvement = (
                float("-inf")
                if comparator_mean <= 0.0
                else 1.0 - candidate_mean / comparator_mean
            )
            maximum_regression = max(
                [
                    0.0
                    if row["technical_failure"]
                    else max(
                        0.0,
                        row["scores"]["guarded_candidate"][metric]
                        / max(row["scores"][comparator][metric], 1e-12)
                        - 1.0,
                    )
                    for row in case_reports
                ]
            )
            passed = bool(
                relative_improvement >= 0.05
                and maximum_regression <= 0.10
                and bootstrap["object_cluster_upper_95_m"] < 0.0
            )
            comparator_checks.append(passed)
            metric_reports[metric] = {
                **bootstrap,
                "candidate_mean_m": candidate_mean,
                "comparator_mean_m": comparator_mean,
                "relative_improvement": relative_improvement,
                "maximum_single_object_regression": maximum_regression,
                "passed": passed,
            }
        joint_wins = sum(
            not row["technical_failure"]
            and all(
                row["scores"]["guarded_candidate"][metric]
                < row["scores"][comparator][metric]
                for metric in PRIMARY_METRICS
            )
            for row in case_reports
        )
        joint_pass = all(comparator_checks) and joint_wins >= 6
        all_primary_checks.append(joint_pass)
        comparisons[comparator] = {
            "metrics": metric_reports,
            "joint_object_wins": joint_wins,
            "passed": joint_pass,
        }

    secondary = {
        metric: {
            "guarded_mean_m": float(
                np.mean(
                    [
                        row["scores"]["guarded_candidate"][metric]
                        for row in ordinary
                    ]
                )
            ),
            "selected_backbone_mean_m": float(
                np.mean(
                    [
                        row["scores"][SELECTED_BACKBONE_ARM][metric]
                        for row in ordinary
                    ]
                )
            ),
        }
        for metric in SECONDARY_METRICS
    }
    for values in secondary.values():
        values["improved"] = (
            values["guarded_mean_m"]
            < values["selected_backbone_mean_m"]
        )
    checks = {
        "primary_against_all_comparators": all(all_primary_checks),
        "late_hidden_identity_improved": secondary[
            "late_hidden_identity_rmse_m"
        ]["improved"],
        "one_sided_chamfer_improved": secondary[
            "hidden_target_to_prediction_chamfer_m"
        ]["improved"],
        "minimum_ordinary_sources": len(ordinary) >= 6,
    }
    return {
        "case_reports": case_reports,
        "comparisons": comparisons,
        "secondary": secondary,
        "checks": checks,
        "passed": all(checks.values()),
        "frozen_target_regret_certificate": _certificate_descriptor(
            final_certificate
        ),
        "claim_boundary": (
            "source transfer on disjoint hidden material identities; no target "
            "or official Deform360 SOTA claim"
        ),
    }


__all__ = [
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "LATE_RANGES",
    "PRIMARY_METRICS",
    "REGRET_FEATURE_NAMES",
    "SCORED_RANGES",
    "SECONDARY_METRICS",
    "SOURCE_EVALUATION_FILENAME",
    "SOURCE_EVALUATION_KIND",
    "aggregate_provider_source_gate",
    "crossfit_guarded_assimilation",
    "evaluate_guarded_assimilation_gate",
    "interval_regret_features",
    "load_source_evaluation_protocol",
    "score_assimilation_trajectory",
    "score_provider_case_arrays",
]
