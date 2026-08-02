"""Cross-modal extension of the Deform360 causal-support guard.

The v1 guard remains unchanged. This module adds one conjunctive route that
requires both a stable or decreasing tactile load and a spatially coherent
camera correction. Either signal alone is insufficient for this route.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from .deform360_causal_support_guard import (
    CAUSAL_SUPPORT_FEATURE_NAMES,
    CausalSupportGuardModel,
    causal_support_decisions,
    fit_causal_support_guard,
)

CROSS_MODAL_SUPPORT_FEATURE_NAMES = (
    *CAUSAL_SUPPORT_FEATURE_NAMES,
    "cumulative_energy_change_from_frame0_fraction",
    "correction_coherence",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class StableTactileCoherentCorrectionRoute:
    """Conjunctive support from independent tactile and camera structure."""

    maximum_cumulative_energy_change: float
    minimum_correction_coherence: float
    enabled: bool
    source_beneficial_admission_count: int
    source_regressive_admission_count: int

    def __post_init__(self) -> None:
        _require(
            np.isfinite(self.maximum_cumulative_energy_change)
            and np.isfinite(self.minimum_correction_coherence),
            "cross-modal thresholds must be finite",
        )
        _require(
            self.source_beneficial_admission_count >= 0
            and self.source_regressive_admission_count >= 0,
            "cross-modal admission counts must be nonnegative",
        )
        if self.enabled:
            _require(
                self.source_beneficial_admission_count > 0,
                "enabled cross-modal route must admit source benefit",
            )
            _require(
                self.source_regressive_admission_count == 0,
                "enabled cross-modal route cannot admit source regression",
            )

    def passes(
        self,
        cumulative_energy_change: float,
        correction_coherence: float,
    ) -> bool:
        """Return whether both finite cross-modal conditions pass."""

        _require(
            np.isfinite(cumulative_energy_change)
            and np.isfinite(correction_coherence),
            "cross-modal feature is non-finite",
        )
        return bool(
            self.enabled
            and cumulative_energy_change <= self.maximum_cumulative_energy_change
            and correction_coherence >= self.minimum_correction_coherence
        )


@dataclass(frozen=True)
class CrossModalSupportGuardModel:
    """V1 causal support plus one conjunctive cross-modal route."""

    causal_support: CausalSupportGuardModel
    stable_tactile_coherent_correction: StableTactileCoherentCorrectionRoute
    source_object_count: int
    source_row_count: int
    source_informative_row_count: int
    regret_tolerance_m: float = 0.0

    def __post_init__(self) -> None:
        _require(
            self.source_object_count == self.causal_support.source_object_count
            and self.source_row_count == self.causal_support.source_row_count
            and self.source_informative_row_count
            == self.causal_support.source_informative_row_count,
            "cross-modal and causal-support source contracts differ",
        )
        _require(
            self.regret_tolerance_m == self.causal_support.regret_tolerance_m,
            "cross-modal and causal-support regret tolerances differ",
        )


def cross_modal_support_feature_vector(
    tactile_features: Mapping[str, Any],
    pairwise_features: Mapping[str, Any],
) -> np.ndarray:
    """Extract all v2 support features in frozen order."""

    merged = {**pairwise_features, **tactile_features}
    values = np.asarray(
        [float(merged[name]) for name in CROSS_MODAL_SUPPORT_FEATURE_NAMES],
        dtype=np.float64,
    )
    _require(
        values.shape == (len(CROSS_MODAL_SUPPORT_FEATURE_NAMES),)
        and np.all(np.isfinite(values)),
        "cross-modal support feature vector is invalid",
    )
    return values


def _fit_cross_modal_route(
    cumulative_energy_change: np.ndarray,
    correction_coherence: np.ndarray,
    regret_m: np.ndarray,
    *,
    tolerance_m: float,
) -> StableTactileCoherentCorrectionRoute:
    beneficial = regret_m < -tolerance_m
    regressive = regret_m > tolerance_m
    energy_thresholds = np.unique(cumulative_energy_change[beneficial])
    coherence_thresholds = np.unique(correction_coherence[beneficial])
    best: tuple[tuple[int, int, float, float], float, float, int, int] | None = None
    for maximum_energy, minimum_coherence in product(
        energy_thresholds.tolist(),
        coherence_thresholds.tolist(),
    ):
        admitted = (
            (cumulative_energy_change <= maximum_energy)
            & (correction_coherence >= minimum_coherence)
        )
        beneficial_count = int(np.count_nonzero(admitted & beneficial))
        regressive_count = int(np.count_nonzero(admitted & regressive))
        if regressive_count:
            continue
        admitted_count = int(np.count_nonzero(admitted))
        key = (
            beneficial_count,
            -admitted_count,
            -float(maximum_energy),
            float(minimum_coherence),
        )
        if best is None or key > best[0]:
            best = (
                key,
                float(maximum_energy),
                float(minimum_coherence),
                beneficial_count,
                regressive_count,
            )
    if best is None:
        return StableTactileCoherentCorrectionRoute(
            maximum_cumulative_energy_change=0.0,
            minimum_correction_coherence=0.0,
            enabled=False,
            source_beneficial_admission_count=0,
            source_regressive_admission_count=0,
        )
    return StableTactileCoherentCorrectionRoute(
        maximum_cumulative_energy_change=best[1],
        minimum_correction_coherence=best[2],
        enabled=True,
        source_beneficial_admission_count=best[3],
        source_regressive_admission_count=best[4],
    )


def fit_cross_modal_support_guard(
    feature_vectors: np.ndarray,
    maximum_regret_m: np.ndarray,
    object_ids: Sequence[str],
    *,
    candidate_nontrivial: Sequence[bool] | None = None,
    regret_tolerance_m: float = 0.0,
) -> CrossModalSupportGuardModel:
    """Fit v1 routes and the conjunctive v2 route on source rows only."""

    features = np.asarray(feature_vectors, dtype=np.float64)
    regret = np.asarray(maximum_regret_m, dtype=np.float64)
    groups = tuple(str(value) for value in object_ids)
    _require(
        features.ndim == 2
        and features.shape[1] == len(CROSS_MODAL_SUPPORT_FEATURE_NAMES),
        "cross-modal source features have the wrong shape",
    )
    _require(
        len(features) == len(regret) == len(groups) and len(features) >= 2,
        "cross-modal source rows differ",
    )
    _require(
        np.all(np.isfinite(features)) and np.all(np.isfinite(regret)),
        "cross-modal source rows contain non-finite values",
    )
    if candidate_nontrivial is None:
        informative = np.ones(len(features), dtype=bool)
    else:
        informative = np.asarray(candidate_nontrivial, dtype=bool)
        _require(
            informative.shape == (len(features),),
            "candidate nontrivial mask has the wrong shape",
        )
    _require(np.any(informative), "cross-modal source has no candidate updates")

    causal = fit_causal_support_guard(
        features[:, : len(CAUSAL_SUPPORT_FEATURE_NAMES)],
        regret,
        groups,
        candidate_nontrivial=informative,
        regret_tolerance_m=regret_tolerance_m,
    )
    route = _fit_cross_modal_route(
        features[informative, 3],
        features[informative, 4],
        regret[informative],
        tolerance_m=regret_tolerance_m,
    )
    return CrossModalSupportGuardModel(
        causal_support=causal,
        stable_tactile_coherent_correction=route,
        source_object_count=causal.source_object_count,
        source_row_count=causal.source_row_count,
        source_informative_row_count=causal.source_informative_row_count,
        regret_tolerance_m=causal.regret_tolerance_m,
    )


def cross_modal_support_decisions(
    feature_vectors: np.ndarray,
    model: CrossModalSupportGuardModel,
) -> list[dict[str, Any]]:
    """Evaluate v1 and conjunctive support without target outcomes."""

    features = np.asarray(feature_vectors, dtype=np.float64)
    _require(
        features.ndim == 2
        and features.shape[1] == len(CROSS_MODAL_SUPPORT_FEATURE_NAMES)
        and np.all(np.isfinite(features)),
        "cross-modal features have the wrong shape or values",
    )
    causal = causal_support_decisions(
        features[:, : len(CAUSAL_SUPPORT_FEATURE_NAMES)],
        model.causal_support,
    )
    output = []
    route = model.stable_tactile_coherent_correction
    for values, causal_decision in zip(features, causal, strict=True):
        cross_modal_passed = route.passes(float(values[3]), float(values[4]))
        admitting_routes = list(causal_decision["admitting_routes"])
        if cross_modal_passed:
            admitting_routes.append("stable_tactile_coherent_correction")
        output.append(
            {
                "support_available": bool(
                    causal_decision["support_available"] or cross_modal_passed
                ),
                "admitting_routes": admitting_routes,
                "causal_support_routes": causal_decision["routes"],
                "cross_modal_route": {
                    "name": "stable_tactile_coherent_correction",
                    "evidence_kind": "cross-modal-regime-support",
                    "conditions": [
                        {
                            "feature_name": (
                                "cumulative_energy_change_from_frame0_fraction"
                            ),
                            "direction": "at_most",
                            "value": float(values[3]),
                            "threshold": route.maximum_cumulative_energy_change,
                        },
                        {
                            "feature_name": "correction_coherence",
                            "direction": "at_least",
                            "value": float(values[4]),
                            "threshold": route.minimum_correction_coherence,
                        },
                    ],
                    "enabled": route.enabled,
                    "passed": cross_modal_passed,
                },
            }
        )
    return output


def apply_cross_modal_support_guard(
    baseline: np.ndarray,
    candidate: np.ndarray,
    feature_vectors: np.ndarray,
    model: CrossModalSupportGuardModel,
    *,
    update_frames: Sequence[int] = (19, 38, 57),
) -> tuple[dict[str, Any], np.ndarray]:
    """Admit v2-supported intervals and retain exact fallback otherwise."""

    baseline_input = np.asarray(baseline)
    candidate_input = np.asarray(candidate)
    updates = tuple(int(frame) for frame in update_frames)
    _require(
        baseline_input.shape == candidate_input.shape and baseline_input.ndim >= 1,
        "cross-modal candidate and baseline differ",
    )
    _require(
        updates
        and tuple(sorted(set(updates))) == updates
        and updates[-1] < len(baseline_input),
        "invalid cross-modal update frames",
    )
    support = cross_modal_support_decisions(feature_vectors, model)
    _require(len(support) == len(updates), "cross-modal update count changed")

    guarded = baseline_input.copy()
    reports = []
    for index, (update, decision) in enumerate(zip(updates, support, strict=True)):
        stop = updates[index + 1] if index + 1 < len(updates) else len(guarded)
        interval = slice(update + 1, stop)
        candidate_nontrivial = not np.array_equal(
            candidate_input[interval], baseline_input[interval]
        )
        accepted = bool(decision["support_available"] and candidate_nontrivial)
        if accepted:
            guarded[interval] = candidate_input[interval]
        exact_fallback = bool(
            not accepted
            and np.array_equal(guarded[interval], baseline_input[interval])
        )
        if not accepted and not exact_fallback:
            raise AssertionError("cross-modal rejection changed the exact baseline")
        reports.append(
            {
                "frame": update,
                "interval_end_exclusive": stop,
                "candidate_nontrivial": candidate_nontrivial,
                "candidate_accepted": accepted,
                "support_available": bool(decision["support_available"]),
                "admitting_routes": decision["admitting_routes"],
                "causal_support_routes": decision["causal_support_routes"],
                "cross_modal_route": decision["cross_modal_route"],
                "reason": (
                    "cross-modal-support-admission"
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
        "arm": "dual_backbone_cross_modal_support_union_guarded",
        "feature_names": list(CROSS_MODAL_SUPPORT_FEATURE_NAMES),
        "updates": reports,
        "information_boundary": {
            "target_argument_accepted": False,
            "future_tactile_read": False,
            "future_camera_observation_read": False,
            "state_innovation_used_as_prior_reliability": False,
            "cross_modal_route_requires_both_conditions": True,
            "source_outcomes_used_only_to_fit_route_thresholds": True,
            "rejection_is_bit_exact_selected_backbone": True,
        },
    }, guarded


__all__ = [
    "CROSS_MODAL_SUPPORT_FEATURE_NAMES",
    "CrossModalSupportGuardModel",
    "StableTactileCoherentCorrectionRoute",
    "apply_cross_modal_support_guard",
    "cross_modal_support_decisions",
    "cross_modal_support_feature_vector",
    "fit_cross_modal_support_guard",
]
