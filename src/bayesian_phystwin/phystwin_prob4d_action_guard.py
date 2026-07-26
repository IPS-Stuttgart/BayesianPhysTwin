"""Action-conditioned propagation for guarded Prob4D prefix updates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np

from .phystwin_prob4d_bias_guard import (
    Prob4DBiasGuardConfig,
    _prefix_validation_metrics,
    build_guarded_prob4d_prefix_candidate,
)


@dataclass(frozen=True)
class Prob4DActionGuardConfig:
    """Frozen controls for one signed physical-progress candidate."""

    static_guard: Prob4DBiasGuardConfig = field(
        default_factory=Prob4DBiasGuardConfig
    )
    minimum_reference_response_rms_m: float = 0.0005
    maximum_absolute_progress: float = 2.0
    minimum_dynamic_validation_improvement_fraction: float = 0.001

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.minimum_reference_response_rms_m)
            or self.minimum_reference_response_rms_m <= 0.0
        ):
            raise ValueError("minimum reference response must be positive")
        if (
            not np.isfinite(self.maximum_absolute_progress)
            or self.maximum_absolute_progress < 1.0
        ):
            raise ValueError("maximum absolute progress must be at least one")
        if (
            not np.isfinite(
                self.minimum_dynamic_validation_improvement_fraction
            )
            or self.minimum_dynamic_validation_improvement_fraction < 0.0
        ):
            raise ValueError("minimum dynamic improvement is invalid")


def _signed_physical_progress(
    baseline_m: np.ndarray,
    *,
    reference_frame: int,
    maximum_absolute_progress: float,
    minimum_reference_response_rms_m: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    baseline = np.asarray(baseline_m, dtype=np.float64)
    if baseline.ndim != 3 or baseline.shape[2] != 3:
        raise ValueError("baseline must have shape (T, N, 3)")
    if not 0 < reference_frame < len(baseline):
        raise ValueError("reference frame is outside the baseline")
    if not np.all(np.isfinite(baseline)):
        raise ValueError("baseline contains non-finite values")

    response = baseline - baseline[0]
    reference = response[reference_frame]
    reference_rms = float(
        np.sqrt(np.mean(np.sum(np.square(reference), axis=1)))
    )
    if reference_rms < minimum_reference_response_rms_m:
        raise ValueError("reference physical response is below threshold")
    denominator = float(np.sum(np.square(reference)))
    raw = np.einsum("tnc,nc->t", response, reference) / denominator
    clipped = np.clip(raw, -maximum_absolute_progress, maximum_absolute_progress)
    return clipped, {
        "reference_response_rms_m": reference_rms,
        "minimum_raw_progress": float(np.min(raw)),
        "maximum_raw_progress": float(np.max(raw)),
        "minimum_clipped_progress": float(np.min(clipped)),
        "maximum_clipped_progress": float(np.max(clipped)),
        "clipped_frame_count": int(np.sum(raw != clipped)),
    }


def build_guarded_action_conditioned_prob4d_candidate(
    selected_baseline_m: np.ndarray,
    physical_prefix_m: np.ndarray,
    prob4d_prefix_positions_m: np.ndarray,
    prob4d_prefix_validity: np.ndarray,
    prob4d_prefix_prior_reliability: np.ndarray,
    prob4d_prefix_observation_covariance_m2: np.ndarray,
    prefix_object_points_m: np.ndarray,
    prefix_object_visibility: np.ndarray,
    prefix_object_motion_validity: np.ndarray,
    *,
    num_surface_points: int,
    source_lock: Mapping[str, Any],
    config: Prob4DActionGuardConfig | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Build and guard a signed action-progress correction.

    The endpoint correction is inferred by the existing bias-aware update. Its
    future amplitude is determined only by the selected physical prediction's
    signed progress along the causal fit-prefix response. Future observations
    are not inputs to candidate construction or admission.
    """

    cfg = config or Prob4DActionGuardConfig()
    baseline_input = np.asarray(selected_baseline_m)
    baseline = np.asarray(baseline_input, dtype=np.float64)
    static_report, static_candidate, _ = build_guarded_prob4d_prefix_candidate(
        baseline_input,
        physical_prefix_m,
        prob4d_prefix_positions_m,
        prob4d_prefix_validity,
        prob4d_prefix_prior_reliability,
        prob4d_prefix_observation_covariance_m2,
        prefix_object_points_m,
        prefix_object_visibility,
        prefix_object_motion_validity,
        num_surface_points=num_surface_points,
        source_lock=source_lock,
        config=cfg.static_guard,
    )
    fit_end = int(static_report["fit_end_exclusive"])
    fit_frame = int(static_report["fit_frame"])
    temporal_candidate = baseline_input.copy()
    progress_report: dict[str, Any]
    candidate_available = bool(static_report["candidate_available"])
    if candidate_available:
        try:
            progress, progress_report = _signed_physical_progress(
                baseline,
                reference_frame=fit_frame,
                maximum_absolute_progress=cfg.maximum_absolute_progress,
                minimum_reference_response_rms_m=(
                    cfg.minimum_reference_response_rms_m
                ),
            )
            endpoint_correction = (
                np.asarray(static_candidate[fit_end], dtype=np.float64)
                - baseline[fit_end]
            )
            temporal_candidate[fit_end:] = (
                baseline[fit_end:]
                + progress[fit_end:, None, None] * endpoint_correction[None]
            ).astype(baseline_input.dtype, copy=False)
            progress_report.update(
                {
                    "endpoint_correction_rms_m": float(
                        np.sqrt(
                            np.mean(
                                np.sum(np.square(endpoint_correction), axis=1)
                            )
                        )
                    ),
                    "endpoint_correction_maximum_m": float(
                        np.max(np.linalg.norm(endpoint_correction, axis=1))
                    ),
                }
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            candidate_available = False
            progress_report = {
                "fallback_reason": f"{type(error).__name__}: {error}"
            }
    else:
        progress_report = {
            "fallback_reason": "static physical-state candidate unavailable"
        }

    baseline_validation = static_report["validation"]["selected_baseline"]
    static_validation = static_report["validation"]["candidate"]
    temporal_validation = _prefix_validation_metrics(
        temporal_candidate,
        prefix_object_points_m,
        prefix_object_visibility,
        prefix_object_motion_validity,
        num_surface_points=num_surface_points,
        start_frame=fit_end,
        end_frame=len(physical_prefix_m),
    )
    metric_names = ("chamfer_distance_m", "pseudo_identity_error_m")
    temporal_over_baseline = {
        name: temporal_validation[name] / baseline_validation[name]
        for name in metric_names
    }
    temporal_over_static = {
        name: temporal_validation[name] / static_validation[name]
        for name in metric_names
    }
    baseline_improvement = 1.0 - float(
        np.mean(list(temporal_over_baseline.values()))
    )
    dynamic_improvement = 1.0 - float(
        np.mean(list(temporal_over_static.values()))
    )
    no_baseline_regression = all(
        ratio <= 1.0 for ratio in temporal_over_baseline.values()
    )
    no_static_regression = all(
        ratio <= 1.0 for ratio in temporal_over_static.values()
    )
    accepted = bool(
        candidate_available
        and no_baseline_regression
        and no_static_regression
        and baseline_improvement
        >= cfg.static_guard.minimum_balanced_validation_improvement_fraction
        and dynamic_improvement
        >= cfg.minimum_dynamic_validation_improvement_fraction
    )
    guarded = baseline_input.copy()
    if accepted:
        guarded[fit_end:] = temporal_candidate[fit_end:]
        decision = "accepted-action-conditioned-prefix-validation"
    else:
        decision = "exact-selected-baseline-fallback"
    selected_baseline_unchanged = np.array_equal(guarded, baseline_input)
    bit_exact_fallback = bool(not accepted and selected_baseline_unchanged)
    if not (accepted or bit_exact_fallback):
        raise AssertionError("action-conditioned guard violated exact fallback")

    report = {
        "artifact_kind": "PhysTwinProb4DActionConditionedPrefixCandidate",
        "schema_version": 1,
        "config": asdict(cfg),
        "fit_end_exclusive": fit_end,
        "fit_frame": fit_frame,
        "candidate_available": candidate_available,
        "candidate_accepted": accepted,
        "decision": decision,
        "bit_exact_selected_baseline_fallback": bit_exact_fallback,
        "progress": progress_report,
        "validation": {
            "selected_baseline": baseline_validation,
            "static_candidate": static_validation,
            "action_conditioned_candidate": temporal_validation,
            "action_conditioned_over_baseline": temporal_over_baseline,
            "action_conditioned_over_static": temporal_over_static,
            "balanced_baseline_improvement_fraction": baseline_improvement,
            "balanced_dynamic_improvement_fraction": dynamic_improvement,
            "no_primary_baseline_regression": no_baseline_regression,
            "no_primary_static_regression": no_static_regression,
        },
        "static_candidate_report": static_report,
        "information_boundary": {
            "future_prob4d_observation_read": False,
            "future_object_observation_read": False,
            "future_manual_track_read": False,
            "future_selected_physical_prediction_read": True,
            "future_selected_physical_prediction_role": (
                "action-progress propagation only"
            ),
            "candidate_fit_frames": [0, fit_frame],
            "family_validation_frames": [fit_end, len(physical_prefix_m) - 1],
            "prior_reliability_uses_state_innovation": False,
            "state_innovation_likelihood_count": 1,
        },
    }
    return report, temporal_candidate, guarded


__all__ = [
    "Prob4DActionGuardConfig",
    "build_guarded_action_conditioned_prob4d_candidate",
]
