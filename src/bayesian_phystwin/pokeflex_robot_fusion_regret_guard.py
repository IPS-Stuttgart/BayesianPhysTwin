"""Source-only regret guard for PokeFlex robot-checkpoint fusion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .bias_aware_belief import (
    SourceGroupRegretBound,
    SourceRegretCertificate,
    fit_source_group_regret_bound,
    fit_source_regret_certificate,
)
from .pokeflex_robot_fusion_protocol import (
    EXPECTED_DEVELOPMENT_OBJECTS,
    POKEFLEX_ROBOT_FUSION_SOURCE_PROTOCOL_SHA256,
)


BASE_FEATURE_NAMES = (
    "baseline_deformation_rms_m",
    "robot_deformation_rms_m",
    "model_disagreement_rms_m",
    "deformation_cosine",
    "force_norm_n",
    "force_delta_norm_n",
    "tool_step_m",
)
FEATURE_NAMES = (
    "candidate_scale",
    "candidate_scale_squared",
    *(f"candidate_scale_x_{name}" for name in BASE_FEATURE_NAMES),
)
EXPECTED_SCALES = (0.0, 0.05, 0.1, 0.2)
EXPECTED_TAKES = ("T1", "T4", "T5", "T6")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class PokeFlexRobotFusionRegretConfig:
    """Frozen source-fit settings and transfer gates."""

    candidate_nominal_coverage: float = 0.90
    candidate_within_take_coverage: float = 0.80
    selector_nominal_coverage: float = 0.90
    selector_within_take_coverage: float = 0.80
    ridge_penalty: float = 10.0
    support_margin_std: float = 0.25
    minimum_improvement_mm: float = 0.0
    minimum_object_balanced_relative_improvement: float = 0.05
    minimum_object_wins: int = 4
    maximum_object_relative_regression: float = 0.10
    maximum_false_safe_rate: float = 0.10

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
        _require(
            self.minimum_improvement_mm >= 0.0,
            "minimum improvement is negative",
        )
        _require(
            self.minimum_object_balanced_relative_improvement >= 0.05,
            "object-balanced gate was weakened",
        )
        _require(self.minimum_object_wins >= 4, "object-win gate was weakened")
        _require(
            self.maximum_object_relative_regression <= 0.10,
            "object-regression gate was weakened",
        )
        _require(
            self.maximum_false_safe_rate <= 0.10,
            "false-safe gate was weakened",
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _take_identity(take_id: str) -> tuple[str, str]:
    object_name, separator, number = take_id.rpartition("_T")
    _require(
        bool(separator) and bool(object_name) and number.isdigit(),
        f"invalid PokeFlex take id: {take_id}",
    )
    return object_name, f"T{number}"


def _candidate_name(scale: float) -> str:
    return f"robot_convex_scale_{scale:g}"


def _candidate_feature(
    scale: float,
    source_features: Mapping[str, object],
) -> np.ndarray:
    """Encode a response that vanishes with the candidate contribution."""

    _require(scale > 0.0, "fallback is not a regret candidate")
    base = np.asarray(
        [float(source_features[name]) for name in BASE_FEATURE_NAMES],
        dtype=np.float64,
    )
    _require(np.all(np.isfinite(base)), "fusion features are non-finite")
    feature = np.concatenate(([scale, scale * scale], scale * base))
    _require(feature.shape == (len(FEATURE_NAMES),), "feature schema changed")
    return feature


def extract_pokeflex_robot_fusion_rows(
    payloads: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate locked source artifacts and expose causal candidate rows."""

    _require(bool(payloads), "at least one source artifact is required")
    rows: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    seen_takes: set[str] = set()
    seen_frames: set[str] = set()
    inventory: set[tuple[str, str]] = set()
    for payload in payloads:
        _require(
            payload.get("artifact_kind") == "PokeFlexRobotFusionSourceTake",
            "unexpected robot-fusion source artifact",
        )
        protocol = payload.get("protocol")
        _require(isinstance(protocol, Mapping), "source protocol record is missing")
        _require(
            protocol.get("sha256")
            == POKEFLEX_ROBOT_FUSION_SOURCE_PROTOCOL_SHA256,
            "source protocol checksum changed",
        )
        boundary = payload.get("causal_boundary")
        _require(isinstance(boundary, Mapping), "causal boundary is missing")
        _require(
            boundary.get("future_observation_used") is False,
            "future observation was used",
        )
        _require(
            boundary.get("target_objects_opened") is False,
            "target object was opened",
        )
        config = payload.get("candidate_config")
        _require(isinstance(config, Mapping), "candidate config is missing")
        _require(
            tuple(map(float, config.get("scales", ()))) == EXPECTED_SCALES,
            "candidate scale bank changed",
        )

        take = payload.get("take")
        _require(isinstance(take, Mapping), "take metadata is missing")
        take_id = str(take.get("id", ""))
        object_name, take_number = _take_identity(take_id)
        _require(take_id not in seen_takes, f"duplicate source take: {take_id}")
        seen_takes.add(take_id)
        inventory.add((object_name, take_number))
        targets = payload.get("targets")
        _require(isinstance(targets, list) and targets, "source targets are missing")
        for target in targets:
            _require(isinstance(target, Mapping), "source target is invalid")
            target_frame = int(target["target_frame"])
            frame_id = f"{take_id}:f{target_frame:05d}"
            _require(frame_id not in seen_frames, f"duplicate source frame: {frame_id}")
            seen_frames.add(frame_id)
            baseline = float(target["released_checkpoint_CD_UL1_mm"])
            fallback = float(target["robot_convex_scale_0_CD_UL1_mm"])
            _require(
                np.isfinite(baseline) and baseline > 0.0,
                "baseline error is invalid",
            )
            _require(fallback == baseline, "fallback outcome changed")
            features = target.get("fusion_features")
            _require(isinstance(features, Mapping), "fusion features are missing")
            _candidate_feature(EXPECTED_SCALES[1], features)
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
            for scale in EXPECTED_SCALES[1:]:
                candidate = _candidate_name(scale)
                error = float(target[f"{candidate}_CD_UL1_mm"])
                _require(np.isfinite(error), "candidate error is non-finite")
                rows.append(
                    {
                        "frame_id": frame_id,
                        "take_id": take_id,
                        "object": object_name,
                        "take": take_number,
                        "target_frame": target_frame,
                        "candidate": candidate,
                        "scale": scale,
                        "features": _candidate_feature(scale, features),
                        "baseline_error_mm": baseline,
                        "candidate_error_mm": error,
                        "regret_mm": error - baseline,
                    }
                )

    expected_inventory = {
        (object_name, take)
        for object_name in EXPECTED_DEVELOPMENT_OBJECTS
        for take in EXPECTED_TAKES
    }
    _require(inventory == expected_inventory, "source take inventory changed")
    return rows, frames


def _fit_candidate_certificate(
    rows: Sequence[Mapping[str, Any]],
    config: PokeFlexRobotFusionRegretConfig,
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
    upper_by_index: Mapping[int, float],
) -> dict[str, tuple[float, int]]:
    selected: dict[str, tuple[float, int]] = {}
    for index, row in enumerate(rows):
        upper = float(upper_by_index[index])
        if not np.isfinite(upper):
            continue
        frame_id = str(row["frame_id"])
        candidate_key = (upper, float(row["scale"]))
        incumbent = selected.get(frame_id)
        if incumbent is None:
            selected[frame_id] = (upper, index)
            continue
        incumbent_key = (incumbent[0], float(rows[incumbent[1]]["scale"]))
        if candidate_key < incumbent_key:
            selected[frame_id] = (upper, index)
    return selected


def _summarize_take_and_object(
    decisions: Sequence[Mapping[str, Any]],
    objects: Sequence[str],
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


def _candidate_bank_summary(
    rows: Sequence[Mapping[str, Any]],
    frames: Sequence[Mapping[str, Any]],
    objects: Sequence[str],
) -> dict[str, Any]:
    row_by_frame: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        row_by_frame.setdefault(str(row["frame_id"]), []).append(row)
    fixed = {}
    for scale in EXPECTED_SCALES:
        decisions = []
        for frame in frames:
            if scale == 0.0:
                error = float(frame["baseline_error_mm"])
            else:
                match = next(
                    row
                    for row in row_by_frame[str(frame["frame_id"])]
                    if float(row["scale"]) == scale
                )
                error = float(match["candidate_error_mm"])
            decisions.append({**frame, "selected_error_mm": error})
        _, object_rows = _summarize_take_and_object(decisions, objects)
        baseline = float(
            np.mean([value["baseline_mean_CD_UL1_mm"] for value in object_rows])
        )
        candidate = float(
            np.mean([value["selected_mean_CD_UL1_mm"] for value in object_rows])
        )
        fixed[_candidate_name(scale)] = {
            "baseline_object_balanced_CD_UL1_mm": baseline,
            "candidate_object_balanced_CD_UL1_mm": candidate,
            "relative_improvement": (baseline - candidate) / baseline,
            "object_wins": sum(
                value["relative_improvement"] > 1e-12 for value in object_rows
            ),
            "maximum_object_relative_regression": max(
                0.0,
                max(-float(value["relative_improvement"]) for value in object_rows),
            ),
            "objects": object_rows,
        }

    frame_oracle = []
    for frame in frames:
        candidates = row_by_frame[str(frame["frame_id"])]
        error = min(
            [float(frame["baseline_error_mm"])]
            + [float(row["candidate_error_mm"]) for row in candidates]
        )
        frame_oracle.append({**frame, "selected_error_mm": error})
    _, oracle_objects = _summarize_take_and_object(frame_oracle, objects)
    baseline = float(
        np.mean([value["baseline_mean_CD_UL1_mm"] for value in oracle_objects])
    )
    oracle = float(
        np.mean([value["selected_mean_CD_UL1_mm"] for value in oracle_objects])
    )
    return {
        "fixed_candidates": fixed,
        "frame_oracle": {
            "baseline_object_balanced_CD_UL1_mm": baseline,
            "oracle_object_balanced_CD_UL1_mm": oracle,
            "relative_improvement": (baseline - oracle) / baseline,
            "status": "diagnostic upper bound using hidden source outcomes",
            "objects": oracle_objects,
        },
    }


def evaluate_pokeflex_robot_fusion_cross_object(
    payloads: Sequence[Mapping[str, Any]],
    *,
    config: PokeFlexRobotFusionRegretConfig | None = None,
) -> dict[str, Any]:
    """Run the locked leave-one-object-out source admission study."""

    cfg = config or PokeFlexRobotFusionRegretConfig()
    rows, frames = extract_pokeflex_robot_fusion_rows(payloads)
    objects = sorted({str(row["object"]) for row in rows})
    _require(
        tuple(objects) == tuple(sorted(EXPECTED_DEVELOPMENT_OBJECTS)),
        "source object inventory changed",
    )

    row_upper: dict[int, float] = {}
    fold_certificates = {}
    for held_object in objects:
        training = [row for row in rows if row["object"] != held_object]
        certificate = _fit_candidate_certificate(training, cfg)
        fold_certificates[held_object] = _certificate_dict(certificate)
        for index, row in enumerate(rows):
            if row["object"] == held_object:
                row_upper[index] = certificate.upper_regret(row["features"])
    _require(len(row_upper) == len(rows), "cross-object prediction inventory changed")
    selected = _select_candidates(rows, row_upper)

    selector_bounds: dict[str, SourceGroupRegretBound] = {}
    decisions: list[dict[str, Any]] = []
    for held_object in objects:
        residual = []
        groups = []
        for upper, index in selected.values():
            row = rows[index]
            if row["object"] == held_object:
                continue
            residual.append(float(row["regret_mm"]) - upper)
            groups.append(str(row["take_id"]))
        selector_bound = fit_source_group_regret_bound(
            np.asarray(residual, dtype=np.float64),
            groups,
            nominal_coverage=cfg.selector_nominal_coverage,
            within_group_coverage=cfg.selector_within_take_coverage,
            minimum_improvement_m=cfg.minimum_improvement_mm,
        )
        selector_bounds[held_object] = selector_bound
        for frame in frames:
            if frame["object"] != held_object:
                continue
            baseline = float(frame["baseline_error_mm"])
            selected_error = baseline
            selected_arm = "released_checkpoint"
            candidate_upper = None
            corrected_upper = None
            candidate = selected.get(str(frame["frame_id"]))
            if candidate is not None:
                candidate_upper, row_index = candidate
                row = rows[row_index]
                corrected_upper = candidate_upper + selector_bound.upper_regret_m
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

    take_rows, object_rows = _summarize_take_and_object(decisions, objects)
    baseline_mean = float(
        np.mean([value["baseline_mean_CD_UL1_mm"] for value in object_rows])
    )
    selected_mean = float(
        np.mean([value["selected_mean_CD_UL1_mm"] for value in object_rows])
    )
    improvement = (baseline_mean - selected_mean) / baseline_mean
    object_wins = sum(
        value["relative_improvement"] > 1e-12 for value in object_rows
    )
    maximum_regression = max(
        0.0,
        max(-float(value["relative_improvement"]) for value in object_rows),
    )
    accepted = [value for value in decisions if value["accepted"]]
    false_safe = [value for value in accepted if value["hidden_regret_mm"] > 1e-12]
    false_safe_rate = len(false_safe) / len(accepted) if accepted else 0.0
    covered = [
        float(row["regret_mm"]) <= row_upper[index] + 1e-12
        for index, row in enumerate(rows)
        if np.isfinite(row_upper[index])
    ]
    gate_checks = {
        "object_balanced_improvement": (
            improvement >= cfg.minimum_object_balanced_relative_improvement
        ),
        "object_wins": object_wins >= cfg.minimum_object_wins,
        "maximum_object_regression": (
            maximum_regression <= cfg.maximum_object_relative_regression
        ),
        "false_safe_rate": false_safe_rate <= cfg.maximum_false_safe_rate,
        "nonempty_acceptance": bool(accepted),
    }
    gate_passed = all(gate_checks.values())

    final_certificate = _fit_candidate_certificate(rows, cfg)
    selector_residual = []
    selector_groups = []
    for upper, index in selected.values():
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
        "artifact_kind": "PokeFlexRobotFusionCrossObjectSourceEvaluation",
        "claim_status": "post-open source-only method development",
        "protocol_sha256": POKEFLEX_ROBOT_FUSION_SOURCE_PROTOCOL_SHA256,
        "aggregation": "equal frames within take, equal takes within object, equal objects",
        "feature_names": list(FEATURE_NAMES),
        "config": cfg.as_dict(),
        "source": {
            "object_count": len(objects),
            "take_count": len({row["take_id"] for row in rows}),
            "frame_count": len(frames),
            "candidate_row_count": len(rows),
        },
        "candidate_bank": _candidate_bank_summary(rows, frames, objects),
        "cross_object": {
            "baseline_object_balanced_CD_UL1_mm": baseline_mean,
            "selected_object_balanced_CD_UL1_mm": selected_mean,
            "object_balanced_relative_improvement": improvement,
            "object_wins": object_wins,
            "object_ties": sum(
                abs(float(value["relative_improvement"])) <= 1e-12
                for value in object_rows
            ),
            "maximum_object_relative_regression": maximum_regression,
            "accepted_frame_count": len(accepted),
            "accepted_frame_wins": sum(
                value["hidden_regret_mm"] < -1e-12 for value in accepted
            ),
            "accepted_frame_losses": len(false_safe),
            "false_safe_rate": false_safe_rate,
            "candidate_upper_coverage": (
                float(np.mean(covered)) if covered else 0.0
            ),
            "gate_checks": gate_checks,
            "gate_passed": gate_passed,
            "decision": (
                "PASS: a fresh-object protocol may be authored"
                if gate_passed
                else "FAIL: do not inspect calibration or sealed target objects"
            ),
        },
        "objects": object_rows,
        "takes": take_rows,
        "fold_candidate_certificates": fold_certificates,
        "fold_selector_bounds": {
            name: _bound_dict(value) for name, value in selector_bounds.items()
        },
        "source_fit_diagnostic": {
            "deployment_authorized": gate_passed,
            "candidate_certificate": _certificate_dict(final_certificate),
            "selector_correction_bound": _bound_dict(final_selector_bound),
            "selection_rule": (
                "choose the in-support candidate with minimum candidate upper "
                "regret; accept only when its selector-adjusted upper regret "
                "is below zero; otherwise preserve the released checkpoint exactly"
            ),
        },
    }
