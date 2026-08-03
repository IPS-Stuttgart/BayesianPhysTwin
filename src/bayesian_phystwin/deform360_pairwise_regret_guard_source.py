"""Source-only qualification for the Deform360 pairwise regret guard.

The runtime candidate never accepts target or outcome data. This module is the
separate, explicitly outcome-open source stage that calibrates its regret
certificate. It evaluates every physical object in an outer leave-one-object-
out fold, checks exact fallback accounting, and runs deterministic placebo and
synthetic-positive controls before emitting a deployment certificate.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .bias_aware_belief import (
    SourceRegretCertificate,
    fit_source_regret_certificate,
)
from .deform360_online_belief_evaluation import (
    _post_update_scored_frames,
    score_deform360_hidden_trajectory,
)
from .deform360_pairwise_regret_guard import (
    FEATURE_NAMES,
    PairwiseRegretGuardConfig,
    build_pairwise_regret_candidate_arrays,
)
from .deform360_raw_camera_observation import (
    MANIFEST_FILENAME,
    _load_measurement_artifact,
    _load_open_case_for_evaluation,
    _sha256,
    _validate_prediction_seal,
    expected_open_case_names,
)

PRIMARY_METRICS = (
    "post_update_hidden_identity_rmse_m",
    "post_update_hidden_symmetric_chamfer_m",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class PairwiseRegretSourceConfig:
    """Frozen source qualification and control thresholds."""

    nominal_coverage: float = 0.50
    within_group_coverage: float = 1.0
    minimum_improvement_m: float = 0.0
    ridge_penalty: float = 10.0
    support_margin_std: float = 1.0
    minimum_relative_improvement: float = 0.02
    minimum_accepted_interval_count: int = 4
    minimum_accepted_object_count: int = 2
    maximum_harmful_accepted_count: int = 0
    placebo_trial_count: int = 1024
    maximum_placebo_pass_rate: float = 0.05
    placebo_seed: int = 20260803
    synthetic_positive_regret_m: float = -0.002

    def __post_init__(self) -> None:
        _require(
            0.0 < self.nominal_coverage < 1.0,
            "nominal coverage must lie in (0, 1)",
        )
        _require(
            0.0 < self.within_group_coverage <= 1.0,
            "within-group coverage must lie in (0, 1]",
        )
        _require(self.minimum_improvement_m >= 0.0, "improvement is negative")
        _require(self.ridge_penalty >= 0.0, "ridge penalty is negative")
        _require(self.support_margin_std >= 0.0, "support margin is negative")
        _require(
            0.0 <= self.minimum_relative_improvement < 1.0,
            "relative improvement must lie in [0, 1)",
        )
        _require(
            self.minimum_accepted_interval_count >= 1,
            "accepted interval count must be positive",
        )
        _require(
            self.minimum_accepted_object_count >= 1,
            "accepted object count must be positive",
        )
        _require(
            self.maximum_harmful_accepted_count >= 0,
            "harmful acceptance count is negative",
        )
        _require(self.placebo_trial_count >= 1, "placebo trial count must be positive")
        _require(
            0.0 <= self.maximum_placebo_pass_rate <= 1.0,
            "placebo pass rate must lie in [0, 1]",
        )
        _require(
            np.isfinite(self.synthetic_positive_regret_m)
            and self.synthetic_positive_regret_m < 0.0,
            "synthetic positive regret must be finite and negative",
        )


@dataclass(frozen=True)
class _Outcome:
    worst_regret_m: float
    metric_regret_m: tuple[float, float]


@dataclass(frozen=True)
class _Interval:
    row_index: int
    interval_index: int
    case: str
    object_id: str
    frame: int
    available: bool
    features: np.ndarray
    outcome: _Outcome


@dataclass(frozen=True)
class _Case:
    case: str
    object_id: str
    baseline_m: tuple[float, float]
    intervals: tuple[_Interval, ...]


def _camera_identity(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def build_pairwise_regret_source_case(
    panel_case_dir: str | Path,
    measurement_dir: str | Path,
    *,
    config: PairwiseRegretGuardConfig | None = None,
) -> dict[str, Any]:
    """Build and score one explicitly outcome-open source case."""

    case_dir = Path(panel_case_dir).resolve()
    measurement_path = Path(measurement_dir).resolve()
    _require(
        case_dir.name in expected_open_case_names(),
        "case is outside the explicit outcome-open panel",
    )
    seal_path = case_dir / "prediction_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    _validate_prediction_seal(seal)
    measurement_manifest, measurement_arrays = _load_measurement_artifact(
        case_dir, measurement_path, seal
    )
    boundary = measurement_manifest.get("information_boundary", {})
    _require(
        boundary.get("target_data_read") is False
        and boundary.get("outcome_manifest_read") is False,
        "measurement construction crossed the target boundary",
    )
    open_seal, physical, persistence, target, visibility, validity = (
        _load_open_case_for_evaluation(case_dir)
    )
    _require(open_seal == seal, "prediction seal changed while opening outcome")
    report, baseline, candidate = build_pairwise_regret_candidate_arrays(
        physical,
        persistence,
        measurement_arrays["measurement_m"],
        measurement_arrays["measurement_visibility"],
        measurement_arrays["measurement_validity"],
        center_ids=measurement_arrays["center_ids"],
        selected_camera_ids=tuple(
            _camera_identity(value)
            for value in measurement_arrays["selected_cameras"]
        ),
        triangulation_inlier_view_count=measurement_arrays[
            "triangulation_inlier_view_count"
        ],
        triangulation_median_reprojection_px=measurement_arrays[
            "triangulation_median_reprojection_px"
        ],
        config=config,
    )
    center_ids = np.asarray(measurement_arrays["center_ids"], dtype=np.int64)
    scored_frames = _post_update_scored_frames(len(target))
    scores = {
        "baseline": score_deform360_hidden_trajectory(
            baseline,
            target,
            visibility,
            validity,
            center_ids=center_ids,
            scored_frames=scored_frames,
        ),
        "candidate": score_deform360_hidden_trajectory(
            candidate,
            target,
            visibility,
            validity,
            center_ids=center_ids,
            scored_frames=scored_frames,
        ),
    }
    intervals = []
    for update in report["updates"]:
        interval_frames = tuple(
            range(int(update["frame"]) + 1, int(update["interval_end_exclusive"]))
        )
        interval_scores = {
            "baseline": score_deform360_hidden_trajectory(
                baseline,
                target,
                visibility,
                validity,
                center_ids=center_ids,
                scored_frames=interval_frames,
            ),
            "candidate": score_deform360_hidden_trajectory(
                candidate,
                target,
                visibility,
                validity,
                center_ids=center_ids,
                scored_frames=interval_frames,
            ),
        }
        regret = {
            metric: float(
                interval_scores["candidate"][metric]
                - interval_scores["baseline"][metric]
            )
            for metric in PRIMARY_METRICS
        }
        intervals.append(
            {
                "frame": int(update["frame"]),
                "available": bool(update["candidate_available"]),
                "features": list(update["features"]),
                "regret": regret,
                "worst_regret_m": max(regret.values()),
                "rejection_reasons": list(update["rejection_reasons"]),
                "applied_correction_scale": float(
                    update["applied_correction_scale"]
                ),
            }
        )
    return {
        "case": case_dir.name,
        "object_id": str(seal["object_id"]),
        "scores": scores,
        "intervals": intervals,
        "report": report,
        "input_sha256": {
            "prediction_seal": _sha256(seal_path),
            "measurement_manifest": _sha256(
                measurement_path / MANIFEST_FILENAME
            ),
            "measurement_result": str(measurement_manifest["result_sha256"]),
        },
        "information_boundary": {
            "measurement_verified_before_outcome_open": True,
            "source_outcome_opened_for_scoring": True,
            "fresh_or_target_outcome_read": False,
        },
    }


def build_pairwise_regret_source_payload(
    panel_root: str | Path,
    measurement_root: str | Path,
    *,
    config: PairwiseRegretGuardConfig | None = None,
) -> dict[str, Any]:
    """Build the complete open-27 source payload in canonical case order."""

    panel = Path(panel_root).resolve()
    measurements = Path(measurement_root).resolve()
    expected = expected_open_case_names()
    missing = [
        case
        for case in expected
        if not (panel / case).is_dir()
        or not (measurements / case / MANIFEST_FILENAME).is_file()
    ]
    _require(not missing, f"open source panel is incomplete: {missing}")
    rows = [
        build_pairwise_regret_source_case(
            panel / case,
            measurements / case,
            config=config,
        )
        for case in expected
    ]
    return {
        "schema_version": 1,
        "artifact_kind": "Deform360PairwiseRegretGuardOpenSourcePayload",
        "claim_status": "post-open source-only method development",
        "rows": rows,
        "source": {
            "case_count": len(rows),
            "physical_object_count": len({row["object_id"] for row in rows}),
            "case_order": list(expected),
        },
        "information_boundary": {
            "source_outcomes_opened_for_scoring": True,
            "runtime_candidate_accepts_target": False,
            "runtime_candidate_accepts_outcome": False,
            "fresh_or_target_outcome_read": False,
        },
    }


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


def pairwise_regret_certificate_from_dict(
    value: Mapping[str, Any],
) -> SourceRegretCertificate:
    """Restore a checked runtime certificate from a source lock."""

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


def _parse_source_payload(payload: Mapping[str, Any]) -> tuple[_Case, ...]:
    rows = payload.get("rows")
    _require(isinstance(rows, Sequence) and bool(rows), "source rows are missing")
    cases: list[_Case] = []
    seen_cases: set[str] = set()
    for row_index, row_value in enumerate(rows):
        _require(isinstance(row_value, Mapping), "source row is not a mapping")
        case = str(row_value.get("case", ""))
        object_id = str(row_value.get("object_id", ""))
        _require(bool(case) and bool(object_id), "case identity is missing")
        _require(case not in seen_cases, "source case is duplicated")
        seen_cases.add(case)
        scores = row_value.get("scores")
        _require(isinstance(scores, Mapping), "source scores are missing")
        baseline = scores.get("baseline")
        _require(isinstance(baseline, Mapping), "baseline scores are missing")
        baseline_m = tuple(float(baseline[metric]) for metric in PRIMARY_METRICS)
        _require(
            all(np.isfinite(value) and value > 0.0 for value in baseline_m),
            "baseline score is invalid",
        )
        interval_values = row_value.get("intervals")
        _require(
            isinstance(interval_values, Sequence) and len(interval_values) == 3,
            "every case must contain three update intervals",
        )
        intervals: list[_Interval] = []
        seen_frames: set[int] = set()
        for interval_index, interval_value in enumerate(interval_values):
            _require(isinstance(interval_value, Mapping), "interval is not a mapping")
            frame = int(interval_value["frame"])
            _require(frame not in seen_frames, "update frame is duplicated")
            seen_frames.add(frame)
            available = bool(interval_value["available"])
            features = np.asarray(interval_value["features"], dtype=np.float64)
            _require(
                features.shape == (len(FEATURE_NAMES),)
                and np.all(np.isfinite(features)),
                "source feature schema changed",
            )
            regret = interval_value.get("regret")
            _require(isinstance(regret, Mapping), "interval regret is missing")
            metric_regret = tuple(float(regret[metric]) for metric in PRIMARY_METRICS)
            worst = float(interval_value["worst_regret_m"])
            _require(
                np.all(np.isfinite(metric_regret)) and np.isfinite(worst),
                "interval regret is non-finite",
            )
            _require(
                np.isclose(worst, max(metric_regret), rtol=0.0, atol=1e-12),
                "worst regret is inconsistent with the co-primary metrics",
            )
            if not available:
                _require(
                    worst == 0.0 and metric_regret == (0.0, 0.0),
                    "unavailable interval did not preserve the exact baseline",
                )
            intervals.append(
                _Interval(
                    row_index=row_index,
                    interval_index=interval_index,
                    case=case,
                    object_id=object_id,
                    frame=frame,
                    available=available,
                    features=features,
                    outcome=_Outcome(worst, metric_regret),
                )
            )
        cases.append(
            _Case(
                case=case,
                object_id=object_id,
                baseline_m=baseline_m,
                intervals=tuple(sorted(intervals, key=lambda value: value.frame)),
            )
        )
    _require(
        len({case.object_id for case in cases}) >= 5,
        "at least five physical source objects are required",
    )
    return tuple(cases)


def _fit_certificate(
    intervals: Sequence[_Interval],
    outcomes: Mapping[tuple[int, int], _Outcome],
    config: PairwiseRegretSourceConfig,
) -> SourceRegretCertificate:
    return fit_source_regret_certificate(
        np.stack([interval.features for interval in intervals]),
        np.asarray(
            [
                outcomes[(interval.row_index, interval.interval_index)].worst_regret_m
                for interval in intervals
            ],
            dtype=np.float64,
        ),
        [interval.object_id for interval in intervals],
        nominal_coverage=config.nominal_coverage,
        within_group_coverage=config.within_group_coverage,
        minimum_improvement=config.minimum_improvement_m,
        ridge_penalty=config.ridge_penalty,
        support_margin_std=config.support_margin_std,
    )


def _object_balanced(
    cases: Sequence[_Case], values: Mapping[str, tuple[float, float]]
) -> tuple[float, float]:
    objects = sorted({case.object_id for case in cases})
    result = []
    for metric_index in range(len(PRIMARY_METRICS)):
        object_values = []
        for object_id in objects:
            object_values.append(
                float(
                    np.mean(
                        [
                            values[case.case][metric_index]
                            for case in cases
                            if case.object_id == object_id
                        ]
                    )
                )
            )
        result.append(float(np.mean(object_values)))
    return tuple(result)  # type: ignore[return-value]


def _gate_checks(
    relative_improvement: tuple[float, float],
    accepted_count: int,
    accepted_object_count: int,
    harmful_accepted_count: int,
    config: PairwiseRegretSourceConfig,
) -> dict[str, bool]:
    return {
        "identity_improvement": (
            relative_improvement[0] >= config.minimum_relative_improvement
        ),
        "chamfer_improvement": (
            relative_improvement[1] >= config.minimum_relative_improvement
        ),
        "accepted_interval_count": (
            accepted_count >= config.minimum_accepted_interval_count
        ),
        "accepted_object_count": (
            accepted_object_count >= config.minimum_accepted_object_count
        ),
        "harmful_accepted_count": (
            harmful_accepted_count <= config.maximum_harmful_accepted_count
        ),
    }


def _cross_fit(
    cases: Sequence[_Case],
    outcomes: Mapping[tuple[int, int], _Outcome],
    config: PairwiseRegretSourceConfig,
    *,
    include_decisions: bool,
) -> dict[str, Any]:
    objects = sorted({case.object_id for case in cases})
    baseline_values = {case.case: case.baseline_m for case in cases}
    selected_values: dict[str, tuple[float, float]] = {}
    decisions: list[dict[str, Any]] = []
    fold_certificates: dict[str, dict[str, Any]] = {}
    for held_object in objects:
        training = [
            interval
            for case in cases
            if case.object_id != held_object
            for interval in case.intervals
            if interval.available
        ]
        _require(
            len({interval.object_id for interval in training}) >= 4,
            "cross-fit training has too few physical objects",
        )
        certificate = _fit_certificate(training, outcomes, config)
        if include_decisions:
            fold_certificates[held_object] = _certificate_dict(certificate)
        for case in cases:
            if case.object_id != held_object:
                continue
            selected = list(case.baseline_m)
            for interval in case.intervals:
                accepted = False
                supported: bool | None = None
                upper: float | None = None
                if interval.available:
                    supported = certificate.in_source_support(interval.features)
                    candidate_upper = certificate.upper_regret(interval.features)
                    upper = (
                        float(candidate_upper)
                        if np.isfinite(candidate_upper)
                        else None
                    )
                    accepted = bool(
                        supported
                        and candidate_upper < -config.minimum_improvement_m
                    )
                outcome = outcomes[(interval.row_index, interval.interval_index)]
                if accepted:
                    for metric_index in range(len(PRIMARY_METRICS)):
                        selected[metric_index] += (
                            outcome.metric_regret_m[metric_index] / 3.0
                        )
                decisions.append(
                    {
                        "case": case.case,
                        "object_id": case.object_id,
                        "frame": interval.frame,
                        "candidate_available": interval.available,
                        "candidate_accepted": accepted,
                        "in_source_support": supported,
                        "upper_regret_m": upper,
                        "actual_worst_regret_m": outcome.worst_regret_m,
                        "exact_baseline_fallback_selected": not accepted,
                    }
                )
            selected_values[case.case] = tuple(selected)  # type: ignore[assignment]

    baseline = _object_balanced(cases, baseline_values)
    selected = _object_balanced(cases, selected_values)
    relative_improvement = tuple(
        (baseline[index] - selected[index]) / baseline[index]
        for index in range(len(PRIMARY_METRICS))
    )
    accepted = [value for value in decisions if value["candidate_accepted"]]
    harmful = [
        value for value in accepted if value["actual_worst_regret_m"] > 0.0
    ]
    accepted_objects = {value["object_id"] for value in accepted}
    checks = _gate_checks(
        relative_improvement,
        len(accepted),
        len(accepted_objects),
        len(harmful),
        config,
    )
    result = {
        "baseline_object_balanced_m": dict(zip(PRIMARY_METRICS, baseline, strict=True)),
        "selected_object_balanced_m": dict(zip(PRIMARY_METRICS, selected, strict=True)),
        "relative_improvement": dict(
            zip(PRIMARY_METRICS, relative_improvement, strict=True)
        ),
        "candidate_available_count": int(
            sum(interval.available for case in cases for interval in case.intervals)
        ),
        "accepted_interval_count": len(accepted),
        "accepted_object_count": len(accepted_objects),
        "beneficial_accepted_count": int(
            sum(value["actual_worst_regret_m"] < 0.0 for value in accepted)
        ),
        "harmful_accepted_count": len(harmful),
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }
    if include_decisions:
        result["decisions"] = decisions
        result["fold_certificates"] = fold_certificates
    return result


def _outcome_map(cases: Sequence[_Case]) -> dict[tuple[int, int], _Outcome]:
    return {
        (interval.row_index, interval.interval_index): interval.outcome
        for case in cases
        for interval in case.intervals
    }


def _placebo_controls(
    cases: Sequence[_Case],
    original: Mapping[tuple[int, int], _Outcome],
    config: PairwiseRegretSourceConfig,
) -> dict[str, Any]:
    rng = np.random.default_rng(config.placebo_seed)
    available_by_object = {
        object_id: [
            interval
            for case in cases
            if case.object_id == object_id
            for interval in case.intervals
            if interval.available
        ]
        for object_id in sorted({case.object_id for case in cases})
    }
    pass_count = 0
    nonempty_count = 0
    zero_harm_count = 0
    for _ in range(config.placebo_trial_count):
        placebo = dict(original)
        for intervals in available_by_object.values():
            order = rng.permutation(len(intervals))
            for target, source_index in zip(intervals, order, strict=True):
                source = intervals[int(source_index)]
                placebo[(target.row_index, target.interval_index)] = original[
                    (source.row_index, source.interval_index)
                ]
        result = _cross_fit(
            cases, placebo, config, include_decisions=False
        )
        pass_count += int(result["gate_passed"])
        nonempty_count += int(result["accepted_interval_count"] > 0)
        zero_harm_count += int(
            result["accepted_interval_count"] > 0
            and result["harmful_accepted_count"] == 0
        )
    pass_rate = pass_count / config.placebo_trial_count
    return {
        "kind": "within-object outcome-permutation placebo",
        "seed": config.placebo_seed,
        "trial_count": config.placebo_trial_count,
        "gate_pass_count": pass_count,
        "gate_pass_rate": pass_rate,
        "nonempty_acceptance_rate": nonempty_count / config.placebo_trial_count,
        "nonempty_zero_harm_rate": zero_harm_count / config.placebo_trial_count,
        "maximum_allowed_gate_pass_rate": config.maximum_placebo_pass_rate,
        "passed": pass_rate <= config.maximum_placebo_pass_rate,
    }


def _synthetic_positive_control(
    cases: Sequence[_Case],
    original: Mapping[tuple[int, int], _Outcome],
    config: PairwiseRegretSourceConfig,
) -> dict[str, Any]:
    positive = dict(original)
    injected = _Outcome(
        config.synthetic_positive_regret_m,
        (
            config.synthetic_positive_regret_m,
            config.synthetic_positive_regret_m,
        ),
    )
    for case in cases:
        for interval in case.intervals:
            if interval.available:
                positive[(interval.row_index, interval.interval_index)] = injected
    result = _cross_fit(cases, positive, config, include_decisions=False)
    return {
        "kind": "known negative-regret injection on production feature rows",
        "injected_regret_m": config.synthetic_positive_regret_m,
        "result": result,
        "passed": bool(result["gate_passed"]),
    }


def evaluate_pairwise_regret_guard_source(
    payload: Mapping[str, Any],
    *,
    config: PairwiseRegretSourceConfig | None = None,
) -> dict[str, Any]:
    """Cross-fit and qualify one outcome-open source payload."""

    cfg = config or PairwiseRegretSourceConfig()
    cases = _parse_source_payload(payload)
    outcomes = _outcome_map(cases)
    cross_fit = _cross_fit(cases, outcomes, cfg, include_decisions=True)
    placebo = _placebo_controls(cases, outcomes, cfg)
    positive = _synthetic_positive_control(cases, outcomes, cfg)
    available = [
        interval
        for case in cases
        for interval in case.intervals
        if interval.available
    ]
    deployment = _fit_certificate(available, outcomes, cfg)
    source_gate_passed = bool(
        cross_fit["gate_passed"] and placebo["passed"] and positive["passed"]
    )
    return {
        "schema_version": 1,
        "artifact_kind": "Deform360PairwiseRegretGuardSourceQualification",
        "claim_status": "post-open source-only method development",
        "feature_names": list(FEATURE_NAMES),
        "primary_metrics": list(PRIMARY_METRICS),
        "config": asdict(cfg),
        "source": {
            "case_count": len(cases),
            "physical_object_count": len({case.object_id for case in cases}),
            "candidate_available_count": len(available),
            "aggregation": (
                "equal update intervals within case, equal cases within object, "
                "equal physical objects"
            ),
        },
        "cross_object": cross_fit,
        "controls": {"placebo": placebo, "synthetic_positive": positive},
        "source_gate_passed": source_gate_passed,
        "fresh_accuracy_evaluation_allowed": source_gate_passed,
        "calibrated_safety_claim_allowed": False,
        "deployment_artifact": {
            "candidate_certificate": _certificate_dict(deployment),
            "selection_rule": (
                "admit only target-free eligible intervals that lie in source "
                "support and have a strictly negative source regret upper bound"
            ),
            "fallback": "bit-exact selected physical/persistence baseline",
        },
        "information_boundary": {
            "source_outcomes_used_to_fit_certificate": True,
            "runtime_candidate_accepts_target": False,
            "runtime_candidate_accepts_outcome": False,
            "placebo_permutations_remain_within_physical_object": True,
            "fresh_outcomes_may_not_select_or_refit_this_lock": True,
        },
        "claim_boundary": (
            "Passing this source gate authorizes only a no-refit evaluation on "
            "genuinely fresh physical objects. Four-group outer folds provide 60% "
            "finite-sample coverage, while the five-group deployment certificate "
            "provides 50%; neither is a calibrated safety or uniformly non-"
            "worsening guarantee."
        ),
    }


__all__ = [
    "PRIMARY_METRICS",
    "PairwiseRegretSourceConfig",
    "build_pairwise_regret_source_case",
    "build_pairwise_regret_source_payload",
    "evaluate_pairwise_regret_guard_source",
    "pairwise_regret_certificate_from_dict",
]
