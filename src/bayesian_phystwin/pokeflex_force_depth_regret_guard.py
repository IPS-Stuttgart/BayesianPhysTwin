"""Cross-object regret guard for force-reachable PokeFlex depth updates."""

from __future__ import annotations

import math
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


FROZEN_CANDIDATE_RUNNER_SHA256 = (
    "7927deb862dac8783b5415197ff65854ec3c0235a01db88689997c9b97f22e25"
)
FORCE_FIELDS = (
    "force_parallel_local_state",
    "action_axis_local_state",
    "force_action_plane_local_state",
    "force_mean_local_state",
)
FORCE_FIELD_LOCK = tuple(
    f"{field}_relative_{radius:g}"
    for field in FORCE_FIELDS
    for radius in (0.25, 0.4, 0.55, 0.7)
)
FEATURE_NAMES = (
    "d405_upper_regret_mm",
    "d405_mean_regret_mm",
    "d405_sensor_disagreement_mm",
    "candidate_scale",
    "candidate_radius_fraction",
    "field_force_parallel",
    "field_action_axis",
    "field_force_action_plane",
    "field_force_mean",
    "update_rms_mm",
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
    r"^checkpoint_(?P<field>"
    + "|".join(FORCE_FIELDS)
    + r")_relative_(?P<radius>0\.25|0\.4|0\.55|0\.7)_residual_scale_"
    r"(?P<scale>0\.125|0\.25|0\.5|1)$"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class PokeFlexForceDepthGuardConfig:
    """Frozen development gates for a force-aware exact-fallback selector."""

    candidate_nominal_coverage: float = 0.90
    candidate_within_take_coverage: float = 0.80
    selector_nominal_coverage: float = 0.90
    selector_within_take_coverage: float = 0.80
    ridge_penalty: float = 10.0
    support_margin_std: float = 0.25
    minimum_improvement_mm: float = 0.0
    minimum_object_balanced_improvement: float = 0.01
    minimum_object_win_fraction: float = 0.75
    maximum_object_regression: float = 0.01
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
        _require(
            0.0 < self.minimum_object_win_fraction <= 1.0,
            "object-win fraction is invalid",
        )
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
    field = str(match.group("field"))
    field_indicators = [float(field == name) for name in FORCE_FIELDS]
    feature = np.asarray(
        [
            np.max(per_sensor),
            np.mean(per_sensor),
            np.ptp(per_sensor),
            float(match.group("scale")),
            float(match.group("radius")),
            *field_indicators,
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


def _validate_force_depth_lock(payload: Mapping[str, Any]) -> None:
    lock = payload.get("force_depth_regret_development")
    _require(isinstance(lock, Mapping), "force-depth development lock is missing")
    _require(
        lock.get("candidate_runner_sha256") == FROZEN_CANDIDATE_RUNNER_SHA256,
        "frozen candidate runner checksum changed",
    )
    _require(
        tuple(lock.get("candidate_fields", ())) == FORCE_FIELD_LOCK,
        "force candidate field lock changed",
    )
    _require(
        tuple(map(float, lock.get("candidate_scales", ())))
        == (0.0, 0.125, 0.25, 0.5, 1.0),
        "force candidate scale lock changed",
    )
    _require(lock.get("measured_force_and_tool_motion_used") is True, "force missing")
    _require(lock.get("d405_evidence") == "frame f-1 only", "D405 timing changed")
    _require(lock.get("prediction_target") == "frame f", "target timing changed")
    _require(lock.get("future_observation_used") is False, "future input was used")
    _require(lock.get("target_objects_opened") is False, "target cohort was opened")


def extract_pokeflex_force_depth_rows(
    payloads: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract target-free force/depth features and hidden development outcomes."""

    _require(bool(payloads), "at least one development artifact is required")
    rows: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    seen_takes: set[str] = set()
    protocol_hashes: set[str] = set()
    for payload in payloads:
        _require(
            payload.get("artifact_kind")
            == "PokeFlexCheckpointBayesianRegistrationDevelopmentSmoke",
            "unexpected development artifact kind",
        )
        _require(
            payload.get("future_observation_used") is False, "future input was used"
        )
        _validate_force_depth_lock(payload)
        take_id = str(payload.get("take", {}).get("id", ""))
        object_name, take_number = _take_identity(take_id)
        _require(take_id not in seen_takes, f"duplicate development take: {take_id}")
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
        _require(
            isinstance(targets, list) and targets, "development targets are missing"
        )
        for target in targets:
            target_frame = int(target["target_frame"])
            frame_id = f"{take_id}:f{target_frame:05d}"
            baseline = float(target["released_checkpoint_CD_UL1_mm"])
            _require(
                np.isfinite(baseline) and baseline > 0.0, "baseline error is invalid"
            )
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
            _require(
                isinstance(evidence_bank, Mapping), "D405 evidence bank is invalid"
            )
            update = updates.get(target_frame)
            if not evidence_bank:
                continue
            _require(update is not None, "candidate evidence has no source update")
            for candidate_name, evidence in sorted(evidence_bank.items()):
                name = str(candidate_name)
                if _CANDIDATE_PATTERN.match(name) is None:
                    continue
                _require(isinstance(evidence, Mapping), "candidate evidence is invalid")
                outcome = float(target[name])
                _require(np.isfinite(outcome), "candidate outcome is non-finite")
                rows.append(
                    {
                        "frame_id": frame_id,
                        "take_id": take_id,
                        "object": object_name,
                        "take": take_number,
                        "target_frame": target_frame,
                        "candidate": name,
                        "features": _candidate_feature(
                            name,
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
        "development artifact protocol is not an approved frozen lock",
    )
    _require(bool(rows), "force-depth candidate bank is empty")
    return rows, frames


def _fit_candidate_certificate(
    rows: Sequence[Mapping[str, Any]], config: PokeFlexForceDepthGuardConfig
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


def _select_candidates(
    rows: Sequence[Mapping[str, Any]],
    certificate: SourceRegretCertificate,
) -> dict[str, tuple[float, int]]:
    selected: dict[str, tuple[float, int]] = {}
    for index, row in enumerate(rows):
        upper = certificate.upper_regret(row["features"])
        if not np.isfinite(upper):
            continue
        frame_id = str(row["frame_id"])
        incumbent = selected.get(frame_id)
        key = (upper, str(row["candidate"]))
        if incumbent is None:
            selected[frame_id] = (upper, index)
            continue
        incumbent_key = (incumbent[0], str(rows[incumbent[1]]["candidate"]))
        if key < incumbent_key:
            selected[frame_id] = (upper, index)
    return selected


def _cross_fitted_selector_residuals(
    training_rows: Sequence[Mapping[str, Any]],
    config: PokeFlexForceDepthGuardConfig,
) -> tuple[list[float], list[str]]:
    """Generate selector residuals without fitting on the selected object's outcomes."""

    objects = sorted({str(row["object"]) for row in training_rows})
    _require(len(objects) >= 3, "selector calibration has too few objects")
    residuals: list[float] = []
    groups: list[str] = []
    for held_object in objects:
        fit_rows = [row for row in training_rows if row["object"] != held_object]
        held_rows = [row for row in training_rows if row["object"] == held_object]
        certificate = _fit_candidate_certificate(fit_rows, config)
        for _, (upper, index) in _select_candidates(held_rows, certificate).items():
            residuals.append(float(held_rows[index]["regret_mm"]) - upper)
            groups.append(str(held_rows[index]["take_id"]))
    _require(len(set(groups)) >= 3, "selector calibration has too few supported takes")
    return residuals, groups


def _fit_selector_bound(
    training_rows: Sequence[Mapping[str, Any]],
    config: PokeFlexForceDepthGuardConfig,
) -> SourceGroupRegretBound:
    residuals, groups = _cross_fitted_selector_residuals(training_rows, config)
    return fit_source_group_regret_bound(
        np.asarray(residuals, dtype=np.float64),
        groups,
        nominal_coverage=config.selector_nominal_coverage,
        within_group_coverage=config.selector_within_take_coverage,
        minimum_improvement_m=config.minimum_improvement_mm,
    )


def _summarize_decisions(
    decisions: Sequence[Mapping[str, Any]], objects: Sequence[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    take_rows = []
    for take_id in sorted({str(value["take_id"]) for value in decisions}):
        current = [value for value in decisions if value["take_id"] == take_id]
        baseline = float(np.mean([value["baseline_error_mm"] for value in current]))
        selected = float(np.mean([value["selected_error_mm"] for value in current]))
        take_rows.append(
            {
                "take_id": take_id,
                "object": current[0]["object"],
                "take": current[0]["take"],
                "target_frame_count": len(current),
                "baseline_mean_CD_UL1_mm": baseline,
                "selected_mean_CD_UL1_mm": selected,
                "relative_improvement": (baseline - selected) / baseline,
            }
        )
    object_rows = []
    for object_name in objects:
        current = [value for value in take_rows if value["object"] == object_name]
        baseline = float(
            np.mean([value["baseline_mean_CD_UL1_mm"] for value in current])
        )
        selected = float(
            np.mean([value["selected_mean_CD_UL1_mm"] for value in current])
        )
        object_rows.append(
            {
                "object": object_name,
                "take_count": len(current),
                "baseline_mean_CD_UL1_mm": baseline,
                "selected_mean_CD_UL1_mm": selected,
                "relative_improvement": (baseline - selected) / baseline,
            }
        )
    return take_rows, object_rows


def _oracle_summary(
    rows: Sequence[Mapping[str, Any]], frames: Sequence[Mapping[str, Any]]
) -> dict[str, float]:
    by_frame: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_frame.setdefault(str(row["frame_id"]), []).append(row)
    values = []
    for frame in frames:
        baseline = float(frame["baseline_error_mm"])
        candidate_rows = by_frame.get(str(frame["frame_id"]), ())
        oracle = min(
            [baseline] + [float(row["candidate_error_mm"]) for row in candidate_rows]
        )
        values.append({**frame, "selected_error_mm": oracle})
    objects = sorted({str(value["object"]) for value in values})
    take_rows, object_rows = _summarize_decisions(values, objects)
    baseline = float(np.mean([row["baseline_mean_CD_UL1_mm"] for row in object_rows]))
    selected = float(np.mean([row["selected_mean_CD_UL1_mm"] for row in object_rows]))
    return {
        "baseline_object_mean_CD_UL1_mm": baseline,
        "oracle_object_mean_CD_UL1_mm": selected,
        "object_balanced_relative_improvement": (baseline - selected) / baseline,
        "take_count": len(take_rows),
    }


def _fixed_arm_decisions(
    arm: str,
    frames: Sequence[Mapping[str, Any]],
    candidate_error: Mapping[tuple[str, str], float],
    objects: set[str],
) -> list[dict[str, Any]]:
    decisions = []
    for frame in frames:
        if frame["object"] not in objects:
            continue
        frame_id = str(frame["frame_id"])
        baseline = float(frame["baseline_error_mm"])
        selected = float(candidate_error.get((frame_id, arm), baseline))
        decisions.append(
            {
                **frame,
                "selected_arm": (
                    arm if (frame_id, arm) in candidate_error else "released_checkpoint"
                ),
                "selected_error_mm": selected,
            }
        )
    return decisions


def _fixed_arm_improvement_by_object(
    arm: str,
    frames: Sequence[Mapping[str, Any]],
    candidate_error: Mapping[tuple[str, str], float],
    objects: Sequence[str],
) -> dict[str, float]:
    decisions = _fixed_arm_decisions(arm, frames, candidate_error, set(objects))
    _, object_rows = _summarize_decisions(decisions, objects)
    return {
        str(row["object"]): float(row["relative_improvement"]) for row in object_rows
    }


def _fixed_arm_cross_object_control(
    rows: Sequence[Mapping[str, Any]],
    frames: Sequence[Mapping[str, Any]],
    objects: Sequence[str],
    config: PokeFlexForceDepthGuardConfig,
    *,
    criterion: str,
) -> dict[str, Any]:
    """Select one global arm on training objects and evaluate the held object."""

    _require(criterion in {"mean", "maximin"}, "fixed-arm criterion is invalid")
    arms = sorted({str(row["candidate"]) for row in rows})
    candidate_error = {
        (str(row["frame_id"]), str(row["candidate"])): float(row["candidate_error_mm"])
        for row in rows
    }
    improvement = {
        arm: _fixed_arm_improvement_by_object(
            arm,
            frames,
            candidate_error,
            objects,
        )
        for arm in arms
    }

    def selection_key(arm: str, training: Sequence[str]) -> tuple[float, float, str]:
        values = [improvement[arm][object_name] for object_name in training]
        mean_value = float(np.mean(values))
        minimum_value = min(values)
        if criterion == "maximin":
            return minimum_value, mean_value, arm
        return mean_value, minimum_value, arm

    decisions: list[dict[str, Any]] = []
    selected_by_object = {}
    for held_object in objects:
        training = [value for value in objects if value != held_object]
        selected_arm = max(arms, key=lambda arm: selection_key(arm, training))
        selected_by_object[held_object] = selected_arm
        decisions.extend(
            _fixed_arm_decisions(
                selected_arm,
                frames,
                candidate_error,
                {held_object},
            )
        )
    take_rows, object_rows = _summarize_decisions(decisions, objects)
    baseline = float(np.mean([row["baseline_mean_CD_UL1_mm"] for row in object_rows]))
    selected = float(np.mean([row["selected_mean_CD_UL1_mm"] for row in object_rows]))
    relative_improvement = (baseline - selected) / baseline
    object_wins = sum(row["relative_improvement"] > 1e-12 for row in object_rows)
    maximum_regression = max(
        0.0, max(-float(row["relative_improvement"]) for row in object_rows)
    )
    required_object_wins = math.ceil(config.minimum_object_win_fraction * len(objects))
    deployment_arm = max(arms, key=lambda arm: selection_key(arm, objects))
    deployment_improvement = improvement[deployment_arm]
    gate_checks = {
        "object_balanced_improvement": (
            relative_improvement >= config.minimum_object_balanced_improvement
        ),
        "object_wins": object_wins >= required_object_wins,
        "maximum_object_regression": (
            maximum_regression <= config.maximum_object_regression
        ),
    }
    return {
        "selection_rule": (
            "maximize the worst training-object relative improvement, then the "
            "training-object mean"
            if criterion == "maximin"
            else "maximize training-object mean, then worst-object improvement"
        ),
        "baseline_object_mean_CD_UL1_mm": baseline,
        "selected_object_mean_CD_UL1_mm": selected,
        "object_balanced_relative_improvement": relative_improvement,
        "required_object_wins": required_object_wins,
        "object_wins": object_wins,
        "maximum_object_regression": maximum_regression,
        "gate_checks": gate_checks,
        "gate_passed": all(gate_checks.values()),
        "selected_arm_by_held_object": selected_by_object,
        "deployment_arm": deployment_arm,
        "deployment_opened_object_relative_improvement": deployment_improvement,
        "objects": object_rows,
        "takes": take_rows,
    }


def evaluate_pokeflex_force_depth_cross_object(
    payloads: Sequence[Mapping[str, Any]],
    *,
    config: PokeFlexForceDepthGuardConfig | None = None,
) -> dict[str, Any]:
    """Nested leave-one-object-out evaluation with selector-aware calibration."""

    cfg = config or PokeFlexForceDepthGuardConfig()
    rows, frames = extract_pokeflex_force_depth_rows(payloads)
    objects = sorted({str(row["object"]) for row in rows})
    _require(len(objects) >= 5, "at least five development objects are required")
    decisions: list[dict[str, Any]] = []
    fold_certificates: dict[str, dict[str, Any]] = {}
    selector_bounds: dict[str, dict[str, Any]] = {}
    candidate_covered: list[bool] = []
    candidate_supported = 0

    for held_object in objects:
        training_rows = [row for row in rows if row["object"] != held_object]
        held_rows = [row for row in rows if row["object"] == held_object]
        certificate = _fit_candidate_certificate(training_rows, cfg)
        selector_bound = _fit_selector_bound(training_rows, cfg)
        fold_certificates[held_object] = _certificate_dict(certificate)
        selector_bounds[held_object] = _bound_dict(selector_bound)
        selected = _select_candidates(held_rows, certificate)
        for row in held_rows:
            upper = certificate.upper_regret(row["features"])
            if np.isfinite(upper):
                candidate_supported += 1
                candidate_covered.append(float(row["regret_mm"]) <= upper + 1e-12)
        held_frames = [frame for frame in frames if frame["object"] == held_object]
        for frame in held_frames:
            baseline = float(frame["baseline_error_mm"])
            selected_error = baseline
            selected_arm = "released_checkpoint"
            candidate_upper = None
            corrected_upper = None
            candidate = selected.get(str(frame["frame_id"]))
            if candidate is not None:
                candidate_upper, row_index = candidate
                corrected_upper = candidate_upper + selector_bound.upper_regret_m
                row = held_rows[row_index]
                if corrected_upper < -cfg.minimum_improvement_mm:
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

    take_rows, object_rows = _summarize_decisions(decisions, objects)
    baseline_object_mean = float(
        np.mean([value["baseline_mean_CD_UL1_mm"] for value in object_rows])
    )
    selected_object_mean = float(
        np.mean([value["selected_mean_CD_UL1_mm"] for value in object_rows])
    )
    improvement = (baseline_object_mean - selected_object_mean) / baseline_object_mean
    accepted = [value for value in decisions if value["accepted"]]
    false_safe = [value for value in accepted if value["hidden_regret_mm"] > 1e-12]
    object_wins = sum(value["relative_improvement"] > 1e-12 for value in object_rows)
    maximum_regression = max(
        0.0, max(-float(value["relative_improvement"]) for value in object_rows)
    )
    false_safe_rate = len(false_safe) / len(accepted) if accepted else 0.0
    candidate_coverage = float(np.mean(candidate_covered)) if candidate_covered else 0.0
    required_object_wins = math.ceil(cfg.minimum_object_win_fraction * len(objects))
    gate_checks = {
        "object_balanced_improvement": (
            improvement >= cfg.minimum_object_balanced_improvement
        ),
        "object_wins": object_wins >= required_object_wins,
        "maximum_object_regression": maximum_regression
        <= cfg.maximum_object_regression,
        "false_safe_rate": false_safe_rate <= cfg.maximum_false_safe_rate,
        "candidate_upper_coverage": (
            candidate_coverage >= cfg.minimum_candidate_upper_coverage
        ),
        "nonempty_acceptance": bool(accepted),
    }

    final_certificate = _fit_candidate_certificate(rows, cfg)
    final_selector_bound = _fit_selector_bound(rows, cfg)
    return {
        "schema_version": 1,
        "artifact_kind": "PokeFlexForceDepthRegretGuardCrossObjectEvaluation",
        "claim_status": "post-open source/calibration method development",
        "aggregation": "equal frames within take, equal takes within object, equal objects",
        "cross_fitting": (
            "outer leave-one-object-out evaluation; inner leave-one-object-out "
            "selector-residual calibration"
        ),
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
            "required_object_wins": required_object_wins,
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
        "candidate_bank_oracle": _oracle_summary(rows, frames),
        "fixed_arm_controls": {
            criterion: _fixed_arm_cross_object_control(
                rows,
                frames,
                objects,
                cfg,
                criterion=criterion,
            )
            for criterion in ("mean", "maximin")
        },
        "objects": object_rows,
        "takes": take_rows,
        "decisions": decisions,
        "fold_candidate_certificates": fold_certificates,
        "fold_selector_bounds": selector_bounds,
        "deployment_artifact": {
            "candidate_certificate": _certificate_dict(final_certificate),
            "selector_correction_bound": _bound_dict(final_selector_bound),
            "selection_rule": (
                "choose the in-support force-reachable candidate with minimum "
                "candidate UCB; accept only when candidate UCB plus nested "
                "selector correction is below zero"
            ),
            "exact_fallback": "released Kinect checkpoint vertices byte-for-byte",
        },
    }
