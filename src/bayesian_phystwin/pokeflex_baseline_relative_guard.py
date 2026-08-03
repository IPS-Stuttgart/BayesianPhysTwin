"""Baseline-relative admission for the causal PokeFlex state update."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .bias_aware_belief import (
    SourceRegretCertificate,
    fit_source_regret_certificate,
)

FEATURE_NAMES = (
    "log1p_ratio",
    "prior_motion_rms_mm",
    "correction_prior_motion_cosine",
    "previous_correction_cosine",
    "log1p_association_count",
    "update_rms_mm",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_field(value: np.ndarray, name: str) -> np.ndarray:
    field = np.asarray(value, dtype=np.float64)
    _require(field.ndim == 2 and field.shape[1] == 3, f"{name} must be (N, 3)")
    _require(np.all(np.isfinite(field)), f"{name} contains non-finite values")
    return field


def _field_rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.square(value), axis=1))))


def _field_cosine(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-15 else 0.0


@dataclass(frozen=True)
class PokeFlexBaselineRelativeGuardConfig:
    """Frozen group-calibration settings for one fixed candidate update."""

    nominal_coverage: float = 0.80
    within_object_coverage: float = 0.80
    ridge_penalty: float = 10.0
    support_margin_std: float = 1.0
    minimum_improvement_mm: float = 0.0

    def __post_init__(self) -> None:
        _require(0.0 < self.nominal_coverage < 1.0, "coverage is invalid")
        _require(
            0.0 < self.within_object_coverage <= 1.0,
            "within-object coverage is invalid",
        )
        _require(self.ridge_penalty >= 0.0, "ridge penalty is negative")
        _require(self.support_margin_std >= 0.0, "support margin is negative")
        _require(
            self.minimum_improvement_mm >= 0.0,
            "minimum improvement is negative",
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def extract_baseline_relative_guard_features(
    raw_correction_m: np.ndarray,
    source_prior_vertices_m: np.ndarray,
    target_prior_vertices_m: np.ndarray,
    previous_raw_correction_m: np.ndarray,
    *,
    association_count: int,
) -> np.ndarray:
    """Build target-free features available before target frame ``f``.

    The raw correction is inferred from frame ``f-1``. The prior motion is the
    physical checkpoint's causal displacement from ``f-1`` to ``f``. No target
    mesh, frame-``f`` depth, or candidate error enters this feature vector.
    """

    correction = _finite_field(raw_correction_m, "raw correction")
    source = _finite_field(source_prior_vertices_m, "source prior")
    target = _finite_field(target_prior_vertices_m, "target prior")
    previous = _finite_field(previous_raw_correction_m, "previous correction")
    _require(
        correction.shape == source.shape == target.shape == previous.shape,
        "guard fields use different graph shapes",
    )
    _require(association_count >= 0, "association count is negative")
    prior_motion = target - source
    correction_rms = _field_rms(correction)
    prior_rms = _field_rms(prior_motion)
    ratio = correction_rms / max(prior_rms, 1e-12)
    features = np.asarray(
        [
            np.log1p(ratio),
            1000.0 * prior_rms,
            _field_cosine(correction, prior_motion),
            _field_cosine(correction, previous),
            np.log1p(association_count),
            1000.0 * correction_rms,
        ],
        dtype=np.float64,
    )
    _require(features.shape == (len(FEATURE_NAMES),), "feature schema changed")
    _require(np.all(np.isfinite(features)), "guard features are non-finite")
    return features


def feature_mapping_to_vector(features: Mapping[str, Any]) -> np.ndarray:
    """Decode the frozen named feature schema."""

    _require(
        set(FEATURE_NAMES) <= set(features),
        "baseline-relative guard feature is missing",
    )
    vector = np.asarray([features[name] for name in FEATURE_NAMES], dtype=np.float64)
    _require(np.all(np.isfinite(vector)), "guard features are non-finite")
    return vector


def certificate_to_payload(certificate: SourceRegretCertificate) -> dict[str, Any]:
    """Serialize a source regret certificate without changing its arithmetic."""

    return {
        "feature_center": certificate.feature_center.tolist(),
        "feature_scale": certificate.feature_scale.tolist(),
        "standardized_feature_lower": (
            certificate.standardized_feature_lower.tolist()
        ),
        "standardized_feature_upper": (
            certificate.standardized_feature_upper.tolist()
        ),
        "coefficients": certificate.coefficients.tolist(),
        "upper_residual_quantile": certificate.upper_residual_quantile,
        "nominal_coverage": certificate.nominal_coverage,
        "minimum_improvement": certificate.minimum_improvement,
        "ridge_penalty": certificate.ridge_penalty,
        "support_margin_std": certificate.support_margin_std,
        "source_group_count": certificate.source_group_count,
        "finite_sample_rank": certificate.finite_sample_rank,
        "finite_sample_coverage": certificate.finite_sample_coverage,
    }


def certificate_from_payload(value: Mapping[str, Any]) -> SourceRegretCertificate:
    """Load an immutable source regret certificate."""

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


def fit_baseline_relative_guard_certificate(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: PokeFlexBaselineRelativeGuardConfig | None = None,
) -> SourceRegretCertificate:
    """Fit one object-group-calibrated upper bound on candidate regret."""

    cfg = config or PokeFlexBaselineRelativeGuardConfig()
    _require(bool(rows), "guard development rows are empty")
    features = np.stack(
        [feature_mapping_to_vector(row["features"]) for row in rows]
    )
    regret = np.asarray([row["regret_mm"] for row in rows], dtype=np.float64)
    groups = [str(row["object"]) for row in rows]
    _require(np.all(np.isfinite(regret)), "guard regret is non-finite")
    _require(len(set(groups)) >= 3, "too few physical-object groups")
    return fit_source_regret_certificate(
        features,
        regret,
        groups,
        nominal_coverage=cfg.nominal_coverage,
        within_group_coverage=cfg.within_object_coverage,
        minimum_improvement=cfg.minimum_improvement_mm,
        ridge_penalty=cfg.ridge_penalty,
        support_margin_std=cfg.support_margin_std,
    )


def baseline_relative_guard_decision(
    certificate: SourceRegretCertificate,
    features: np.ndarray | Mapping[str, Any],
) -> dict[str, Any]:
    """Admit only a candidate with a negative calibrated upper regret bound."""

    vector = (
        feature_mapping_to_vector(features)
        if isinstance(features, Mapping)
        else np.asarray(features, dtype=np.float64)
    )
    _require(vector.shape == (len(FEATURE_NAMES),), "feature shape changed")
    upper = certificate.upper_regret(vector)
    accepted = bool(
        np.isfinite(upper) and upper < -certificate.minimum_improvement
    )
    return {
        "accepted": accepted,
        "in_source_support": bool(np.isfinite(upper)),
        "predicted_regret_mm": certificate.predict_regret(vector),
        "upper_regret_mm": float(upper) if np.isfinite(upper) else None,
        "reason": "negative-upper-regret" if accepted else "exact-fallback",
    }


def apply_baseline_relative_guard(
    baseline_vertices_m: np.ndarray,
    candidate_vertices_m: np.ndarray,
    certificate: SourceRegretCertificate,
    features: np.ndarray | Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select the update or return an exact copy of the released baseline."""

    baseline = _finite_field(baseline_vertices_m, "baseline vertices")
    candidate = _finite_field(candidate_vertices_m, "candidate vertices")
    _require(candidate.shape == baseline.shape, "candidate graph shape changed")
    decision = baseline_relative_guard_decision(certificate, features)
    selected = candidate.copy() if decision["accepted"] else baseline.copy()
    if not decision["accepted"] and not np.array_equal(selected, baseline):
        raise AssertionError("baseline-relative fallback changed checkpoint bytes")
    return selected, decision


def leave_one_physical_object_out_decisions(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: PokeFlexBaselineRelativeGuardConfig | None = None,
) -> list[dict[str, Any]]:
    """Cross-fit every decision with the evaluated physical object withheld."""

    cfg = config or PokeFlexBaselineRelativeGuardConfig()
    objects = sorted({str(row["object"]) for row in rows})
    _require(len(objects) >= 4, "too few objects for leave-one-object-out")
    decisions: list[dict[str, Any]] = []
    for held_object in objects:
        training = [row for row in rows if str(row["object"]) != held_object]
        held = [row for row in rows if str(row["object"]) == held_object]
        certificate = fit_baseline_relative_guard_certificate(training, config=cfg)
        for row in held:
            decision = baseline_relative_guard_decision(
                certificate, row["features"]
            )
            decisions.append(
                {
                    **{
                        name: row[name]
                        for name in (
                            "domain",
                            "object",
                            "take_id",
                            "target_frame",
                            "take_target_frame_count",
                            "regret_mm",
                        )
                    },
                    **decision,
                    "selected_regret_mm": (
                        float(row["regret_mm"]) if decision["accepted"] else 0.0
                    ),
                }
            )
    _require(len(decisions) == len(rows), "cross-fit decision inventory changed")
    return decisions


def summarize_guard_decisions(
    decisions: Sequence[Mapping[str, Any]],
    *,
    domain: str,
    object_baseline_mm: Mapping[str, float],
    take_inventory: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate frame decisions equally within take, then across objects."""

    current = [row for row in decisions if str(row["domain"]) == domain]
    seen_frames: set[tuple[str, int]] = set()
    for row in current:
        take_id = str(row["take_id"])
        _require(take_id in take_inventory, f"decision take is unknown: {take_id}")
        _require(
            int(row["take_target_frame_count"])
            == int(take_inventory[take_id]["frame_count"]),
            f"decision frame count changed for {take_id}",
        )
        key = (take_id, int(row["target_frame"]))
        _require(key not in seen_frames, f"duplicate guard decision: {key}")
        seen_frames.add(key)
    by_take: dict[str, float] = {}
    for take_id, record in take_inventory.items():
        _require(int(record["frame_count"]) >= 1, "take frame count is invalid")
        by_take[take_id] = float(
            sum(
                row["selected_regret_mm"]
                for row in current
                if str(row["take_id"]) == take_id
            )
            / int(record["frame_count"])
        )
    objects = []
    for object_name, baseline_value in sorted(object_baseline_mm.items()):
        takes = [
            take_id
            for take_id, record in take_inventory.items()
            if str(record["object"]) == object_name
        ]
        _require(bool(takes), f"object has no take inventory: {object_name}")
        baseline = float(baseline_value)
        _require(np.isfinite(baseline) and baseline > 0.0, "baseline is invalid")
        delta = float(np.mean([by_take[take_id] for take_id in takes]))
        guarded = baseline + delta
        accepted_count = int(
            sum(
                bool(row["accepted"])
                for row in current
                if str(row["object"]) == object_name
            )
        )
        objects.append(
            {
                "object": object_name,
                "take_count": len(takes),
                "baseline_CD_UL1_mm": baseline,
                "guarded_CD_UL1_mm": guarded,
                "relative_improvement": float(-delta / baseline),
                "accepted_frame_count": accepted_count,
            }
        )
    baseline_mean = float(
        np.mean([row["baseline_CD_UL1_mm"] for row in objects])
    )
    guarded_mean = float(
        np.mean([row["guarded_CD_UL1_mm"] for row in objects])
    )
    improvements = np.asarray(
        [row["relative_improvement"] for row in objects], dtype=np.float64
    )
    return {
        "object_count": len(objects),
        "baseline_object_balanced_CD_UL1_mm": baseline_mean,
        "guarded_object_balanced_CD_UL1_mm": guarded_mean,
        "object_balanced_relative_improvement": float(
            (baseline_mean - guarded_mean) / baseline_mean
        ),
        "object_wins": int(np.sum(improvements > 1e-12)),
        "object_ties": int(np.sum(np.abs(improvements) <= 1e-12)),
        "object_losses": int(np.sum(improvements < -1e-12)),
        "minimum_object_improvement": float(np.min(improvements)),
        "supported_object_count": int(
            sum(row["accepted_frame_count"] > 0 for row in objects)
        ),
        "objects": objects,
    }


def decision_audit(decisions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Report support, false-safe rate, and held-object upper coverage."""

    finite = [row for row in decisions if row["in_source_support"]]
    accepted = [row for row in decisions if row["accepted"]]
    return {
        "row_count": len(decisions),
        "in_support_count": len(finite),
        "accepted_count": len(accepted),
        "accepted_win_count": int(
            sum(float(row["regret_mm"]) < -1e-12 for row in accepted)
        ),
        "accepted_loss_count": int(
            sum(float(row["regret_mm"]) > 1e-12 for row in accepted)
        ),
        "false_safe_rate": float(
            sum(float(row["regret_mm"]) > 1e-12 for row in accepted)
            / len(accepted)
            if accepted
            else 0.0
        ),
        "upper_coverage": float(
            sum(
                float(row["regret_mm"]) <= float(row["upper_regret_mm"]) + 1e-12
                for row in finite
            )
            / len(finite)
            if finite
            else 0.0
        ),
    }
