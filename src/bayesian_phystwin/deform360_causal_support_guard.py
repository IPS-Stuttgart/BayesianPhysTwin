"""Interpretable causal-support guard for Deform360 belief updates.

The guard admits a camera-derived candidate only when at least one frozen,
source-fitted route provides independent or temporal support for the update.
Rejected intervals preserve the selected physical/persistence baseline exactly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

RouteDirection = Literal["at_most", "at_least"]

CAUSAL_SUPPORT_FEATURE_NAMES = (
    "sensor_ratio_max",
    "correction_change_rms_over_object_scale",
    "prior_consistency_gain",
)

CAUSAL_SUPPORT_ROUTE_SPECS = (
    (
        "low_tactile_loading",
        "sensor_ratio_max",
        "at_most",
        "independent-tactile-support",
    ),
    (
        "large_causal_correction_change",
        "correction_change_rms_over_object_scale",
        "at_least",
        "causal-response-support",
    ),
    (
        "temporally_consistent_prior",
        "prior_consistency_gain",
        "at_least",
        "temporal-belief-support",
    ),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class CausalSupportRoute:
    """One monotone, source-fitted admission route."""

    name: str
    feature_name: str
    direction: RouteDirection
    evidence_kind: str
    threshold: float
    enabled: bool
    source_beneficial_admission_count: int
    source_regressive_admission_count: int

    def __post_init__(self) -> None:
        _require(bool(self.name), "causal-support route needs a name")
        _require(
            self.feature_name in CAUSAL_SUPPORT_FEATURE_NAMES,
            "unknown causal-support feature",
        )
        _require(
            self.direction in ("at_most", "at_least"),
            "unknown causal-support direction",
        )
        _require(bool(self.evidence_kind), "causal-support route needs evidence")
        _require(np.isfinite(self.threshold), "route threshold must be finite")
        _require(
            self.source_beneficial_admission_count >= 0
            and self.source_regressive_admission_count >= 0,
            "route admission counts must be nonnegative",
        )
        if self.enabled:
            _require(
                self.source_beneficial_admission_count > 0,
                "enabled route must admit source benefit",
            )
            _require(
                self.source_regressive_admission_count == 0,
                "enabled route cannot admit source regression",
            )

    def passes(self, value: float) -> bool:
        """Return whether a finite feature value satisfies this route."""

        _require(np.isfinite(value), "causal-support feature is non-finite")
        if not self.enabled:
            return False
        if self.direction == "at_most":
            return bool(value <= self.threshold)
        return bool(value >= self.threshold)


@dataclass(frozen=True)
class CausalSupportGuardModel:
    """Frozen union of interpretable admission routes."""

    routes: tuple[CausalSupportRoute, ...]
    source_object_count: int
    source_row_count: int
    source_informative_row_count: int
    regret_tolerance_m: float = 0.0

    def __post_init__(self) -> None:
        _require(
            tuple(route.name for route in self.routes)
            == tuple(spec[0] for spec in CAUSAL_SUPPORT_ROUTE_SPECS),
            "causal-support route order changed",
        )
        _require(
            len({route.feature_name for route in self.routes}) == len(self.routes),
            "causal-support features must be unique",
        )
        _require(
            self.source_object_count >= 2 and self.source_row_count >= 2,
            "causal-support guard needs multiple source objects and rows",
        )
        _require(
            1 <= self.source_informative_row_count <= self.source_row_count,
            "invalid informative source-row count",
        )
        _require(
            np.isfinite(self.regret_tolerance_m) and self.regret_tolerance_m >= 0.0,
            "regret tolerance must be finite and nonnegative",
        )
        _require(
            any(route.enabled for route in self.routes),
            "causal-support guard has no enabled route",
        )


def causal_support_feature_vector(
    tactile_features: Mapping[str, Any],
    pairwise_features: Mapping[str, Any],
) -> np.ndarray:
    """Extract the three target-free support features in frozen order."""

    merged = {**pairwise_features, **tactile_features}
    values = np.asarray(
        [float(merged[name]) for name in CAUSAL_SUPPORT_FEATURE_NAMES],
        dtype=np.float64,
    )
    _require(
        values.shape == (len(CAUSAL_SUPPORT_FEATURE_NAMES),)
        and np.all(np.isfinite(values)),
        "causal-support feature vector is invalid",
    )
    return values


def _fit_route(
    values: np.ndarray,
    regret_m: np.ndarray,
    *,
    name: str,
    feature_name: str,
    direction: RouteDirection,
    evidence_kind: str,
    tolerance_m: float,
) -> CausalSupportRoute:
    beneficial = regret_m < -tolerance_m
    regressive = regret_m > tolerance_m
    candidate_thresholds = np.unique(values[beneficial])
    best: tuple[tuple[int, int, float], float, int, int] | None = None
    for threshold in candidate_thresholds.tolist():
        admitted = (
            values <= threshold if direction == "at_most" else values >= threshold
        )
        beneficial_count = int(np.count_nonzero(admitted & beneficial))
        regressive_count = int(np.count_nonzero(admitted & regressive))
        if regressive_count:
            continue
        admitted_count = int(np.count_nonzero(admitted))
        strictness = -float(threshold) if direction == "at_most" else float(threshold)
        key = (beneficial_count, -admitted_count, strictness)
        if best is None or key > best[0]:
            best = (key, float(threshold), beneficial_count, regressive_count)

    if best is None:
        return CausalSupportRoute(
            name=name,
            feature_name=feature_name,
            direction=direction,
            evidence_kind=evidence_kind,
            threshold=0.0,
            enabled=False,
            source_beneficial_admission_count=0,
            source_regressive_admission_count=0,
        )
    return CausalSupportRoute(
        name=name,
        feature_name=feature_name,
        direction=direction,
        evidence_kind=evidence_kind,
        threshold=best[1],
        enabled=True,
        source_beneficial_admission_count=best[2],
        source_regressive_admission_count=best[3],
    )


def fit_causal_support_guard(
    feature_vectors: np.ndarray,
    maximum_regret_m: np.ndarray,
    object_ids: Sequence[str],
    *,
    candidate_nontrivial: Sequence[bool] | None = None,
    regret_tolerance_m: float = 0.0,
) -> CausalSupportGuardModel:
    """Fit monotone routes while admitting no regressive source update.

    Thresholds are selected independently for each route. The objective first
    maximizes beneficial admissions, then minimizes total admissions, and then
    chooses the stricter threshold. Exact candidate no-ops do not influence the
    fitted boundaries.
    """

    features = np.asarray(feature_vectors, dtype=np.float64)
    regret = np.asarray(maximum_regret_m, dtype=np.float64)
    groups = tuple(str(value) for value in object_ids)
    _require(
        features.ndim == 2
        and features.shape[1] == len(CAUSAL_SUPPORT_FEATURE_NAMES),
        "causal-support source features have the wrong shape",
    )
    _require(
        len(features) == len(regret) == len(groups) and len(features) >= 2,
        "causal-support source rows differ",
    )
    _require(
        np.all(np.isfinite(features)) and np.all(np.isfinite(regret)),
        "causal-support source rows contain non-finite values",
    )
    _require(
        np.isfinite(regret_tolerance_m) and regret_tolerance_m >= 0.0,
        "regret tolerance must be finite and nonnegative",
    )
    unique_groups = tuple(sorted(set(groups)))
    _require(len(unique_groups) >= 2, "causal-support guard needs multiple objects")

    if candidate_nontrivial is None:
        informative = np.ones(len(features), dtype=bool)
    else:
        informative = np.asarray(candidate_nontrivial, dtype=bool)
        _require(
            informative.shape == (len(features),),
            "candidate nontrivial mask has the wrong shape",
        )
    _require(np.any(informative), "causal-support source has no candidate updates")
    fit_features = features[informative]
    fit_regret = regret[informative]
    _require(
        np.any(fit_regret < -regret_tolerance_m),
        "causal-support source has no beneficial update",
    )

    routes = []
    for feature_index, (name, feature_name, direction, evidence_kind) in enumerate(
        CAUSAL_SUPPORT_ROUTE_SPECS
    ):
        routes.append(
            _fit_route(
                fit_features[:, feature_index],
                fit_regret,
                name=name,
                feature_name=feature_name,
                direction=direction,
                evidence_kind=evidence_kind,
                tolerance_m=regret_tolerance_m,
            )
        )
    return CausalSupportGuardModel(
        routes=tuple(routes),
        source_object_count=len(unique_groups),
        source_row_count=len(features),
        source_informative_row_count=int(np.count_nonzero(informative)),
        regret_tolerance_m=float(regret_tolerance_m),
    )


def causal_support_decisions(
    feature_vectors: np.ndarray,
    model: CausalSupportGuardModel,
) -> list[dict[str, Any]]:
    """Evaluate every route without using a state innovation or target outcome."""

    features = np.asarray(feature_vectors, dtype=np.float64)
    _require(
        features.ndim == 2
        and features.shape[1] == len(CAUSAL_SUPPORT_FEATURE_NAMES)
        and np.all(np.isfinite(features)),
        "causal-support features have the wrong shape or values",
    )
    decisions = []
    for values in features:
        route_decisions = []
        for index, route in enumerate(model.routes):
            value = float(values[index])
            route_decisions.append(
                {
                    "name": route.name,
                    "feature_name": route.feature_name,
                    "evidence_kind": route.evidence_kind,
                    "direction": route.direction,
                    "value": value,
                    "threshold": route.threshold,
                    "enabled": route.enabled,
                    "passed": route.passes(value),
                }
            )
        decisions.append(
            {
                "support_available": bool(
                    any(route["passed"] for route in route_decisions)
                ),
                "admitting_routes": [
                    str(route["name"])
                    for route in route_decisions
                    if route["passed"]
                ],
                "routes": route_decisions,
            }
        )
    return decisions


def apply_causal_support_guard(
    baseline: np.ndarray,
    candidate: np.ndarray,
    feature_vectors: np.ndarray,
    model: CausalSupportGuardModel,
    *,
    update_frames: Sequence[int] = (19, 38, 57),
) -> tuple[dict[str, Any], np.ndarray]:
    """Admit supported candidate intervals and preserve exact fallback."""

    baseline_input = np.asarray(baseline)
    candidate_input = np.asarray(candidate)
    updates = tuple(int(frame) for frame in update_frames)
    _require(
        baseline_input.shape == candidate_input.shape and baseline_input.ndim >= 1,
        "causal-support candidate and baseline differ",
    )
    _require(
        updates
        and tuple(sorted(set(updates))) == updates
        and updates[-1] < len(baseline_input),
        "invalid causal-support update frames",
    )
    decisions = causal_support_decisions(feature_vectors, model)
    _require(len(decisions) == len(updates), "causal-support update count changed")

    guarded = baseline_input.copy()
    reports = []
    for index, (update, support) in enumerate(zip(updates, decisions, strict=True)):
        stop = updates[index + 1] if index + 1 < len(updates) else len(guarded)
        interval = slice(update + 1, stop)
        candidate_nontrivial = not np.array_equal(
            candidate_input[interval], baseline_input[interval]
        )
        accepted = bool(support["support_available"] and candidate_nontrivial)
        if accepted:
            guarded[interval] = candidate_input[interval]
        exact_fallback = bool(
            not accepted
            and np.array_equal(guarded[interval], baseline_input[interval])
        )
        if not accepted and not exact_fallback:
            raise AssertionError("causal-support rejection changed the exact baseline")
        reports.append(
            {
                "frame": update,
                "interval_end_exclusive": stop,
                "candidate_nontrivial": candidate_nontrivial,
                "candidate_accepted": accepted,
                "support_available": bool(support["support_available"]),
                "admitting_routes": list(support["admitting_routes"]),
                "routes": support["routes"],
                "reason": (
                    "causal-support-admission"
                    if accepted
                    else (
                        "exact-candidate-noop"
                        if not candidate_nontrivial
                        else "exact-baseline-fallback"
                    )
                ),
                "bit_exact_baseline_fallback": exact_fallback,
            }
        )
    return {
        "arm": "dual_backbone_causal_support_union_guarded",
        "feature_names": list(CAUSAL_SUPPORT_FEATURE_NAMES),
        "updates": reports,
        "information_boundary": {
            "target_argument_accepted": False,
            "future_tactile_read": False,
            "future_camera_observation_read": False,
            "state_innovation_used_as_prior_reliability": False,
            "source_outcomes_used_only_to_fit_route_thresholds": True,
            "rejection_is_bit_exact_selected_backbone": True,
        },
    }, guarded


__all__ = [
    "CAUSAL_SUPPORT_FEATURE_NAMES",
    "CAUSAL_SUPPORT_ROUTE_SPECS",
    "CausalSupportGuardModel",
    "CausalSupportRoute",
    "apply_causal_support_guard",
    "causal_support_decisions",
    "causal_support_feature_vector",
    "fit_causal_support_guard",
]
