"""Source-calibrated regret guard for PokeFlex independent-depth updates."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .bias_aware_belief import (
    SourceGroupRegretBound,
    SourceRegretCertificate,
    fit_source_group_regret_bound,
    fit_source_regret_certificate,
)
from .pokeflex_independent_depth_protocol import (
    POKEFLEX_INDEPENDENT_DEPTH_PROTOCOL_SHA256,
)
from .pokeflex_independent_depth_source_validation_protocol import (
    POKEFLEX_INDEPENDENT_DEPTH_SOURCE_VALIDATION_PROTOCOL_SHA256,
)


FEATURE_NAMES = (
    "d405_upper_regret_mm",
    "d405_mean_regret_mm",
    "d405_sensor_disagreement_mm",
    "candidate_scale",
    "candidate_radius_fraction",
    "kinect_update_rms_mm",
    "log1p_correction_to_prior_motion_ratio",
    "prior_motion_rms_mm",
    "correction_prior_motion_cosine",
    "previous_correction_cosine",
    "force_y_over_50n",
    "force_y_delta_over_20n",
    "median_robust_weight",
    "downweighted_fraction",
    "assignment_std_mm",
    "maximum_d405_calibration_residual_over_10mm",
)

_CANDIDATE_PATTERN = re.compile(
    r"^checkpoint_action_local_state_relative_"
    r"(?P<radius>0\.4|0\.55|0\.7)_residual_scale_"
    r"(?P<scale>0\.125|0\.25|0\.5|1)$"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class PokeFlexRegretGuardConfig:
    """Frozen source-fit and exploratory transfer gates."""

    candidate_nominal_coverage: float = 0.90
    candidate_within_take_coverage: float = 0.80
    selector_nominal_coverage: float = 0.90
    selector_within_take_coverage: float = 0.80
    ridge_penalty: float = 10.0
    support_margin_std: float = 0.25
    minimum_improvement_mm: float = 0.0
    minimum_object_balanced_improvement: float = 0.01
    minimum_object_wins: int = 4
    maximum_object_regression: float = 0.0
    maximum_false_safe_rate: float = 0.10
    minimum_candidate_upper_coverage: float = 0.80

    def __post_init__(self) -> None:
        for value in (
            self.candidate_nominal_coverage,
            self.candidate_within_take_coverage,
            self.selector_nominal_coverage,
            self.selector_within_take_coverage,
        ):
            _require(0.0 < value < 1.0, "coverage must lie in (0, 1)")
        _require(self.ridge_penalty >= 0.0, "ridge penalty is negative")
        _require(self.support_margin_std >= 0.0, "support margin is negative")
        _require(self.minimum_improvement_mm >= 0.0, "minimum improvement is negative")
        _require(
            self.minimum_object_balanced_improvement >= 0.0,
            "transfer gate is negative",
        )
        _require(self.minimum_object_wins >= 1, "object-win gate is invalid")
        _require(self.maximum_object_regression >= 0.0, "regression gate is negative")
        _require(
            0.0 <= self.maximum_false_safe_rate <= 1.0,
            "false-safe gate is invalid",
        )
        _require(
            0.0 <= self.minimum_candidate_upper_coverage <= 1.0,
            "coverage gate is invalid",
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _take_identity(take_id: str) -> tuple[str, str]:
    object_name, separator, take_number = take_id.rpartition("_T")
    _require(
        bool(separator) and bool(object_name) and take_number.isdigit(),
        f"invalid PokeFlex take id: {take_id}",
    )
    return object_name, f"T{take_number}"


def _optional_float(value: object) -> float:
    return 0.0 if value is None else float(value)


def _candidate_feature(
    candidate_name: str,
    evidence: Mapping[str, Any],
    update: Mapping[str, Any],
    maximum_calibration_residual_mm: float,
) -> np.ndarray:
    match = _CANDIDATE_PATTERN.match(candidate_name)
    _require(match is not None, f"unexpected candidate arm: {candidate_name}")
    per_sensor = np.asarray(evidence.get("per_sensor_mm"), dtype=np.float64)
    _require(
        per_sensor.ndim == 1
        and len(per_sensor) >= 1
        and np.all(np.isfinite(per_sensor)),
        "candidate D405 regret is invalid",
    )
    assignment_variance = float(update.get("assignment_variance_m2_mean", 0.0))
    _require(assignment_variance >= 0.0, "assignment variance is negative")
    ratio = max(0.0, float(update.get("correction_to_prior_motion_ratio", 0.0)))
    feature = np.asarray(
        [
            np.max(per_sensor),
            np.mean(per_sensor),
            np.ptp(per_sensor),
            float(match.group("scale")),
            float(match.group("radius")),
            1000.0 * float(update.get("rms_update_m", 0.0)),
            np.log1p(ratio),
            1000.0 * float(update.get("prior_motion_rms_m", 0.0)),
            _optional_float(update.get("correction_prior_motion_cosine")),
            _optional_float(update.get("previous_correction_cosine")),
            float(update.get("force_y", 0.0)) / 50.0,
            float(update.get("force_y_delta", 0.0)) / 20.0,
            float(update.get("median_robust_weight", 0.0)),
            float(update.get("downweighted_fraction", 0.0)),
            1000.0 * np.sqrt(assignment_variance),
            maximum_calibration_residual_mm / 10.0,
        ],
        dtype=np.float64,
    )
    _require(feature.shape == (len(FEATURE_NAMES),), "feature schema changed")
    _require(np.all(np.isfinite(feature)), "candidate features are non-finite")
    return feature


def extract_pokeflex_regret_guard_rows(
    payloads: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract causal candidate features and hidden source outcomes."""

    _require(bool(payloads), "at least one source artifact is required")
    rows: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    seen_takes: set[str] = set()
    protocol_hashes: set[str] = set()
    for payload in payloads:
        _require(
            payload.get("artifact_kind")
            == "PokeFlexCheckpointBayesianRegistrationDevelopmentSmoke",
            "unexpected source artifact kind",
        )
        _require(payload.get("future_observation_used") is False, "future input was used")
        take_id = str(payload.get("take", {}).get("id", ""))
        object_name, take_number = _take_identity(take_id)
        _require(take_id not in seen_takes, f"duplicate source take: {take_id}")
        seen_takes.add(take_id)
        anchor = payload.get("independent_depth_anchor")
        _require(isinstance(anchor, Mapping), "independent-depth metadata is missing")
        protocol_hashes.add(str(anchor.get("protocol_sha256", "")))
        calibration = np.asarray(anchor.get("median_residual_mm"), dtype=np.float64)
        _require(
            calibration.ndim == 1
            and len(calibration) >= 1
            and np.all(np.isfinite(calibration)),
            "D405 calibration inventory changed",
        )
        maximum_calibration = float(np.max(calibration))
        updates = {
            int(value["target_frame"]): value for value in payload.get("updates", ())
        }
        targets = payload.get("targets")
        _require(isinstance(targets, list) and targets, "source targets are missing")
        for target in targets:
            target_frame = int(target["target_frame"])
            frame_id = f"{take_id}:f{target_frame:05d}"
            baseline = float(target["released_checkpoint_CD_UL1_mm"])
            _require(np.isfinite(baseline) and baseline > 0.0, "baseline error is invalid")
            frames.append(
                {
                    "frame_id": frame_id,
                    "take_id": take_id,
                    "object": object_name,
                    "take": take_number,
                    "target_frame": target_frame,
                    "baseline_error_mm": baseline,
                }
            )
            evidence_bank = target.get("independent_anchor_regret", {})
            _require(isinstance(evidence_bank, Mapping), "D405 evidence bank is invalid")
            update = updates.get(target_frame)
            if not evidence_bank:
                continue
            _require(update is not None, "candidate evidence has no source update")
            for candidate_name, evidence in sorted(evidence_bank.items()):
                if _CANDIDATE_PATTERN.match(str(candidate_name)) is None:
                    continue
                _require(isinstance(evidence, Mapping), "candidate evidence is invalid")
                outcome = float(target[candidate_name])
                _require(np.isfinite(outcome), "candidate outcome is non-finite")
                rows.append(
                    {
                        "frame_id": frame_id,
                        "take_id": take_id,
                        "object": object_name,
                        "take": take_number,
                        "target_frame": target_frame,
                        "candidate": str(candidate_name),
                        "features": _candidate_feature(
                            str(candidate_name),
                            evidence,
                            update,
                            maximum_calibration,
                        ),
                        "baseline_error_mm": baseline,
                        "candidate_error_mm": outcome,
                        "regret_mm": outcome - baseline,
                    }
                )
    approved_protocols = {
        POKEFLEX_INDEPENDENT_DEPTH_PROTOCOL_SHA256,
        POKEFLEX_INDEPENDENT_DEPTH_SOURCE_VALIDATION_PROTOCOL_SHA256,
    }
    _require(
        bool(protocol_hashes)
        and "" not in protocol_hashes
        and protocol_hashes <= approved_protocols,
        "source artifact protocol is not an approved frozen lock",
    )
    _require(bool(rows), "source candidate bank is empty")
    return rows, frames


def _fit_candidate_certificate(
    rows: Sequence[Mapping[str, Any]], config: PokeFlexRegretGuardConfig
) -> SourceRegretCertificate:
    return fit_source_regret_certificate(
        np.stack([np.asarray(row["features"], dtype=np.float64) for row in rows]),
        np.asarray([row["regret_mm"] for row in rows], dtype=np.float64),
        [str(row["take_id"]) for row in rows],
        nominal_coverage=config.candidate_nominal_coverage,
        within_group_coverage=config.candidate_within_take_coverage,
        minimum_improvement=config.minimum_improvement_mm,
        ridge_penalty=config.ridge_penalty,
        support_margin_std=config.support_margin_std,
    )


def _certificate_dict(value: SourceRegretCertificate) -> dict[str, Any]:
    return {
        "feature_center": value.feature_center.tolist(),
        "feature_scale": value.feature_scale.tolist(),
        "standardized_feature_lower": value.standardized_feature_lower.tolist(),
        "standardized_feature_upper": value.standardized_feature_upper.tolist(),
        "coefficients": value.coefficients.tolist(),
        "upper_residual_quantile": value.upper_residual_quantile,
        "nominal_coverage": value.nominal_coverage,
        "minimum_improvement": value.minimum_improvement,
        "ridge_penalty": value.ridge_penalty,
        "support_margin_std": value.support_margin_std,
        "source_group_count": value.source_group_count,
        "finite_sample_rank": value.finite_sample_rank,
        "finite_sample_coverage": value.finite_sample_coverage,
    }


def _certificate_from_dict(value: Mapping[str, Any]) -> SourceRegretCertificate:
    return SourceRegretCertificate(
        feature_center=np.asarray(value["feature_center"], dtype=np.float64),
        feature_scale=np.asarray(value["feature_scale"], dtype=np.float64),
        standardized_feature_lower=np.asarray(
            value["standardized_feature_lower"], dtype=np.float64
        ),
        standardized_feature_upper=np.asarray(
            value["standardized_feature_upper"], dtype=np.float64
        ),
        coefficients=np.asarray(value["coefficients"], dtype=np.float64),
        upper_residual_quantile=float(value["upper_residual_quantile"]),
        nominal_coverage=float(value["nominal_coverage"]),
        minimum_improvement=float(value["minimum_improvement"]),
        ridge_penalty=float(value["ridge_penalty"]),
        support_margin_std=float(value["support_margin_std"]),
        source_group_count=int(value["source_group_count"]),
        finite_sample_rank=int(value["finite_sample_rank"]),
        finite_sample_coverage=float(value["finite_sample_coverage"]),
    )


def _bound_dict(value: SourceGroupRegretBound) -> dict[str, Any]:
    return {
        "upper_regret_mm": value.upper_regret_m,
        "group_scores_mm": value.group_scores_m.tolist(),
        "nominal_coverage": value.nominal_coverage,
        "finite_sample_rank": value.finite_sample_rank,
        "finite_sample_coverage": value.finite_sample_coverage,
        "within_group_coverage": value.within_group_coverage,
        "minimum_improvement_mm": value.minimum_improvement_m,
    }


def _bound_from_dict(value: Mapping[str, Any]) -> SourceGroupRegretBound:
    scores = np.asarray(value["group_scores_mm"], dtype=np.float64)
    return SourceGroupRegretBound(
        upper_regret_m=float(value["upper_regret_mm"]),
        group_scores_m=scores,
        nominal_coverage=float(value["nominal_coverage"]),
        finite_sample_rank=int(value["finite_sample_rank"]),
        finite_sample_coverage=float(value["finite_sample_coverage"]),
        within_group_coverage=float(value["within_group_coverage"]),
        minimum_improvement_m=float(value["minimum_improvement_mm"]),
    )


def _select_candidates(
    rows: Sequence[Mapping[str, Any]],
    upper_by_index: Mapping[int, float],
) -> dict[str, tuple[float, int]]:
    selected: dict[str, tuple[float, int]] = {}
    for index, row in enumerate(rows):
        upper = float(upper_by_index[index])
        if not np.isfinite(upper):
            continue
        frame_id = str(row["frame_id"])
        incumbent = selected.get(frame_id)
        candidate_key = (upper, str(row["candidate"]))
        if incumbent is None:
            selected[frame_id] = (upper, index)
            continue
        incumbent_key = (incumbent[0], str(rows[incumbent[1]]["candidate"]))
        if candidate_key < incumbent_key:
            selected[frame_id] = (upper, index)
    return selected


def evaluate_pokeflex_regret_guard_cross_object(
    payloads: Sequence[Mapping[str, Any]],
    *,
    config: PokeFlexRegretGuardConfig | None = None,
) -> dict[str, Any]:
    """Leave one object out, including selector-aware regret calibration."""

    cfg = config or PokeFlexRegretGuardConfig()
    rows, frames = extract_pokeflex_regret_guard_rows(payloads)
    objects = sorted({str(row["object"]) for row in rows})
    _require(len(objects) >= 3, "at least three source objects are required")
    row_upper: dict[int, float] = {}
    fold_certificates = {}
    for held_object in objects:
        training = [row for row in rows if row["object"] != held_object]
        _require(
            len({row["take_id"] for row in training}) >= 3,
            "cross-object training has too few takes",
        )
        certificate = _fit_candidate_certificate(training, cfg)
        fold_certificates[held_object] = _certificate_dict(certificate)
        for index, row in enumerate(rows):
            if row["object"] == held_object:
                row_upper[index] = certificate.upper_regret(row["features"])
    _require(len(row_upper) == len(rows), "cross-object prediction inventory changed")
    selected = _select_candidates(rows, row_upper)

    selector_bounds: dict[str, SourceGroupRegretBound] = {}
    decisions: dict[str, dict[str, Any]] = {}
    candidate_covered = []
    candidate_supported = 0
    for index, row in enumerate(rows):
        upper = row_upper[index]
        if np.isfinite(upper):
            candidate_supported += 1
            candidate_covered.append(float(row["regret_mm"]) <= upper + 1e-12)

    for held_object in objects:
        calibration_residual = []
        calibration_groups = []
        for _, (upper, index) in selected.items():
            row = rows[index]
            if row["object"] == held_object:
                continue
            calibration_residual.append(float(row["regret_mm"]) - upper)
            calibration_groups.append(str(row["take_id"]))
        selector_bound = fit_source_group_regret_bound(
            np.asarray(calibration_residual, dtype=np.float64),
            calibration_groups,
            nominal_coverage=cfg.selector_nominal_coverage,
            within_group_coverage=cfg.selector_within_take_coverage,
            minimum_improvement_m=cfg.minimum_improvement_mm,
        )
        selector_bounds[held_object] = selector_bound
        for frame in frames:
            if frame["object"] != held_object:
                continue
            frame_id = str(frame["frame_id"])
            baseline = float(frame["baseline_error_mm"])
            candidate = selected.get(frame_id)
            selected_error = baseline
            selected_arm = "released_checkpoint"
            candidate_upper = None
            corrected_upper = None
            if candidate is not None:
                candidate_upper, row_index = candidate
                row = rows[row_index]
                corrected_upper = candidate_upper + selector_bound.upper_regret_m
                if corrected_upper < -cfg.minimum_improvement_mm:
                    selected_error = float(row["candidate_error_mm"])
                    selected_arm = str(row["candidate"])
            decisions[frame_id] = {
                **frame,
                "selected_arm": selected_arm,
                "candidate_upper_regret_mm": candidate_upper,
                "selector_adjusted_upper_regret_mm": corrected_upper,
                "selected_error_mm": selected_error,
                "hidden_regret_mm": selected_error - baseline,
                "accepted": selected_arm != "released_checkpoint",
            }

    take_rows = []
    for take_id in sorted({str(frame["take_id"]) for frame in frames}):
        current = [value for value in decisions.values() if value["take_id"] == take_id]
        baseline = float(np.mean([value["baseline_error_mm"] for value in current]))
        selected_mean = float(np.mean([value["selected_error_mm"] for value in current]))
        take_rows.append(
            {
                "take_id": take_id,
                "object": current[0]["object"],
                "take": current[0]["take"],
                "target_frame_count": len(current),
                "baseline_mean_CD_UL1_mm": baseline,
                "selected_mean_CD_UL1_mm": selected_mean,
                "relative_improvement": (baseline - selected_mean) / baseline,
            }
        )
    object_rows = []
    for object_name in objects:
        current = [value for value in take_rows if value["object"] == object_name]
        baseline = float(np.mean([value["baseline_mean_CD_UL1_mm"] for value in current]))
        selected_mean = float(
            np.mean([value["selected_mean_CD_UL1_mm"] for value in current])
        )
        object_rows.append(
            {
                "object": object_name,
                "take_count": len(current),
                "baseline_mean_CD_UL1_mm": baseline,
                "selected_mean_CD_UL1_mm": selected_mean,
                "relative_improvement": (baseline - selected_mean) / baseline,
                "selector_bound": _bound_dict(selector_bounds[object_name]),
            }
        )
    baseline_object_mean = float(
        np.mean([value["baseline_mean_CD_UL1_mm"] for value in object_rows])
    )
    selected_object_mean = float(
        np.mean([value["selected_mean_CD_UL1_mm"] for value in object_rows])
    )
    accepted = [value for value in decisions.values() if value["accepted"]]
    false_safe = [value for value in accepted if value["hidden_regret_mm"] > 1e-12]
    improvement = (baseline_object_mean - selected_object_mean) / baseline_object_mean
    object_wins = sum(value["relative_improvement"] > 1e-12 for value in object_rows)
    maximum_regression = max(
        0.0, max(-float(value["relative_improvement"]) for value in object_rows)
    )
    false_safe_rate = len(false_safe) / len(accepted) if accepted else 0.0
    candidate_coverage = float(np.mean(candidate_covered)) if candidate_covered else 0.0
    gate_checks = {
        "object_balanced_improvement": (
            improvement >= cfg.minimum_object_balanced_improvement
        ),
        "object_wins": object_wins >= cfg.minimum_object_wins,
        "maximum_object_regression": (
            maximum_regression <= cfg.maximum_object_regression
        ),
        "false_safe_rate": false_safe_rate <= cfg.maximum_false_safe_rate,
        "candidate_upper_coverage": (
            candidate_coverage >= cfg.minimum_candidate_upper_coverage
        ),
        "nonempty_acceptance": bool(accepted),
    }

    final_certificate = _fit_candidate_certificate(rows, cfg)
    selector_residual = []
    selector_groups = []
    for _, (upper, index) in selected.items():
        selector_residual.append(float(rows[index]["regret_mm"]) - upper)
        selector_groups.append(str(rows[index]["take_id"]))
    final_selector_bound = fit_source_group_regret_bound(
        np.asarray(selector_residual, dtype=np.float64),
        selector_groups,
        nominal_coverage=cfg.selector_nominal_coverage,
        within_group_coverage=cfg.selector_within_take_coverage,
        minimum_improvement_m=cfg.minimum_improvement_mm,
    )
    return {
        "schema_version": 1,
        "artifact_kind": "PokeFlexIndependentDepthRegretGuardCrossObjectEvaluation",
        "claim_status": "post-open source-only method development",
        "aggregation": "equal frames within take, equal takes within object, equal objects",
        "feature_names": list(FEATURE_NAMES),
        "config": cfg.as_dict(),
        "source": {
            "object_count": len(objects),
            "take_count": len({row["take_id"] for row in rows}),
            "frame_count": len(frames),
            "candidate_row_count": len(rows),
        },
        "cross_object": {
            "baseline_object_mean_CD_UL1_mm": baseline_object_mean,
            "selected_object_mean_CD_UL1_mm": selected_object_mean,
            "object_balanced_relative_improvement": improvement,
            "object_wins": object_wins,
            "object_ties": sum(
                abs(float(value["relative_improvement"])) <= 1e-12
                for value in object_rows
            ),
            "maximum_object_regression": maximum_regression,
            "accepted_frame_count": len(accepted),
            "accepted_frame_wins": sum(
                value["hidden_regret_mm"] < -1e-12 for value in accepted
            ),
            "accepted_frame_losses": len(false_safe),
            "false_safe_rate": false_safe_rate,
            "candidate_supported_count": candidate_supported,
            "candidate_upper_coverage": candidate_coverage,
            "gate_checks": gate_checks,
            "gate_passed": all(gate_checks.values()),
        },
        "objects": object_rows,
        "takes": take_rows,
        "fold_candidate_certificates": fold_certificates,
        "deployment_artifact": {
            "candidate_certificate": _certificate_dict(final_certificate),
            "selector_correction_bound": _bound_dict(final_selector_bound),
            "selection_rule": (
                "choose the in-support candidate with minimum candidate UCB; accept "
                "only when candidate UCB plus selector correction is below zero"
            ),
            "exact_fallback": "released Kinect checkpoint vertices byte-for-byte",
        },
    }


def evaluate_pokeflex_regret_guard_prospective(
    payloads: Sequence[Mapping[str, Any]],
    source_evaluation: Mapping[str, Any],
    *,
    expected_take_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Apply the frozen source certificate once to prospective take artifacts."""

    _require(
        source_evaluation.get("artifact_kind")
        == "PokeFlexIndependentDepthRegretGuardCrossObjectEvaluation",
        "unexpected source evaluation kind",
    )
    _require(
        source_evaluation.get("cross_object", {}).get("gate_passed") is True,
        "source gate did not pass",
    )
    _require(
        tuple(source_evaluation.get("feature_names", ())) == FEATURE_NAMES,
        "deployment feature schema changed",
    )
    deployment = source_evaluation.get("deployment_artifact")
    _require(isinstance(deployment, Mapping), "deployment artifact is missing")
    certificate = _certificate_from_dict(deployment["candidate_certificate"])
    selector_bound = _bound_from_dict(deployment["selector_correction_bound"])
    rows, frames = extract_pokeflex_regret_guard_rows(payloads)
    observed_takes = sorted({str(frame["take_id"]) for frame in frames})
    if expected_take_ids is not None:
        _require(
            observed_takes == sorted(map(str, expected_take_ids)),
            "prospective take inventory changed",
        )
    upper_by_index = {
        index: certificate.upper_regret(row["features"])
        for index, row in enumerate(rows)
    }
    selected = _select_candidates(rows, upper_by_index)
    decisions = []
    for frame in frames:
        baseline = float(frame["baseline_error_mm"])
        selected_error = baseline
        selected_arm = "released_checkpoint"
        candidate_upper = None
        corrected_upper = None
        candidate = selected.get(str(frame["frame_id"]))
        if candidate is not None:
            candidate_upper, row_index = candidate
            corrected_upper = candidate_upper + selector_bound.upper_regret_m
            row = rows[row_index]
            if corrected_upper < -certificate.minimum_improvement:
                selected_error = float(row["candidate_error_mm"])
                selected_arm = str(row["candidate"])
        decisions.append(
            {
                **frame,
                "selected_arm": selected_arm,
                "candidate_upper_regret_mm": candidate_upper,
                "selector_adjusted_upper_regret_mm": corrected_upper,
                "selected_error_mm": selected_error,
                "hidden_regret_mm": selected_error - baseline,
                "accepted": selected_arm != "released_checkpoint",
            }
        )

    take_rows = []
    for take_id in observed_takes:
        current = [value for value in decisions if value["take_id"] == take_id]
        baseline = float(np.mean([value["baseline_error_mm"] for value in current]))
        selected_mean = float(np.mean([value["selected_error_mm"] for value in current]))
        take_rows.append(
            {
                "take_id": take_id,
                "object": current[0]["object"],
                "target_frame_count": len(current),
                "baseline_mean_CD_UL1_mm": baseline,
                "selected_mean_CD_UL1_mm": selected_mean,
                "relative_improvement": (baseline - selected_mean) / baseline,
            }
        )
    object_rows = []
    for object_name in sorted({str(value["object"]) for value in take_rows}):
        current = [value for value in take_rows if value["object"] == object_name]
        baseline = float(np.mean([value["baseline_mean_CD_UL1_mm"] for value in current]))
        selected_mean = float(
            np.mean([value["selected_mean_CD_UL1_mm"] for value in current])
        )
        object_rows.append(
            {
                "object": object_name,
                "take_count": len(current),
                "baseline_mean_CD_UL1_mm": baseline,
                "selected_mean_CD_UL1_mm": selected_mean,
                "relative_improvement": (baseline - selected_mean) / baseline,
            }
        )
    baseline_mean = float(
        np.mean([value["baseline_mean_CD_UL1_mm"] for value in object_rows])
    )
    selected_mean = float(
        np.mean([value["selected_mean_CD_UL1_mm"] for value in object_rows])
    )
    accepted = [value for value in decisions if value["accepted"]]
    return {
        "schema_version": 1,
        "artifact_kind": "PokeFlexIndependentDepthRegretGuardProspectiveEvaluation",
        "claim_status": "prospective development-take replication",
        "aggregation": "equal frames within take, equal takes within object, equal objects",
        "take_ids": observed_takes,
        "object_count": len(object_rows),
        "take_count": len(take_rows),
        "baseline_object_mean_CD_UL1_mm": baseline_mean,
        "selected_object_mean_CD_UL1_mm": selected_mean,
        "object_balanced_relative_improvement": (
            baseline_mean - selected_mean
        ) / baseline_mean,
        "object_wins": sum(
            value["relative_improvement"] > 1e-12 for value in object_rows
        ),
        "object_losses": sum(
            value["relative_improvement"] < -1e-12 for value in object_rows
        ),
        "accepted_frame_count": len(accepted),
        "accepted_frame_wins": sum(
            value["hidden_regret_mm"] < -1e-12 for value in accepted
        ),
        "accepted_frame_losses": sum(
            value["hidden_regret_mm"] > 1e-12 for value in accepted
        ),
        "exact_fallback_frame_count": len(decisions) - len(accepted),
        "objects": object_rows,
        "takes": take_rows,
        "decisions": decisions,
        "deployment_artifact": deployment,
    }
