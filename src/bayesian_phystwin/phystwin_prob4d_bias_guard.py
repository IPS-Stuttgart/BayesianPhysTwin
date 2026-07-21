"""Causal bias-aware admission for Prob4D prefix observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping

import numpy as np

from .deform360_bias_aware_belief_development import (
    predict_bias_aware_candidate_arrays,
)
from .deform360_selective_bias_guard_diagnostic import config_from_source_lock
from .phystwin_official_evaluation import _nearest_distances


@dataclass(frozen=True)
class Prob4DBiasGuardConfig:
    """Choices fixed before the exploratory future is scored."""

    fit_fraction: float = 0.75
    minimum_validation_frame_count: int = 3
    minimum_balanced_validation_improvement_fraction: float = 0.001

    def __post_init__(self) -> None:
        if not 0.0 < self.fit_fraction < 1.0:
            raise ValueError("fit fraction must lie in (0, 1)")
        if self.minimum_validation_frame_count < 1:
            raise ValueError("minimum validation frame count must be positive")
        if (
            not np.isfinite(
                self.minimum_balanced_validation_improvement_fraction
            )
            or self.minimum_balanced_validation_improvement_fraction < 0.0
        ):
            raise ValueError("minimum validation improvement is invalid")


def _prefix_validation_metrics(
    trajectory_m: np.ndarray,
    object_points_m: np.ndarray,
    object_visibility: np.ndarray,
    object_motion_validity: np.ndarray,
    *,
    num_surface_points: int,
    start_frame: int,
    end_frame: int,
) -> dict[str, float]:
    trajectory = np.asarray(trajectory_m, dtype=np.float64)
    observed = np.asarray(object_points_m, dtype=np.float64)
    visible = np.asarray(object_visibility, dtype=bool)
    motion_valid = np.asarray(object_motion_validity, dtype=bool)
    if observed.ndim != 3 or observed.shape[2] != 3:
        raise ValueError("object points must have shape (T, M, 3)")
    if visible.shape != observed.shape[:2] or motion_valid.shape != visible.shape:
        raise ValueError("prefix object masks have changed shape")
    if trajectory.ndim != 3 or trajectory.shape[2] != 3:
        raise ValueError("trajectory must have shape (T, N, 3)")
    if not 0 <= start_frame < end_frame <= len(observed):
        raise ValueError("prefix validation interval is invalid")
    if len(trajectory) < end_frame or trajectory.shape[1] < observed.shape[1]:
        raise ValueError("trajectory does not cover prefix object identities")
    if not 1 <= num_surface_points <= trajectory.shape[1]:
        raise ValueError("surface point count exceeds trajectory")

    chamfer: list[float] = []
    identity: list[float] = []
    original_count = observed.shape[1]
    for frame in range(start_frame, end_frame):
        visible_points = visible[frame] & np.all(
            np.isfinite(observed[frame]), axis=1
        )
        if np.any(visible_points):
            distance, _ = _nearest_distances(
                trajectory[frame, :num_surface_points],
                observed[frame, visible_points],
                p=1,
            )
            chamfer.append(float(np.mean(distance)))
        valid_identity = (
            visible_points
            & motion_valid[frame]
            & np.all(
                np.isfinite(trajectory[frame, :original_count]), axis=1
            )
        )
        if np.any(valid_identity):
            residual = (
                trajectory[frame, :original_count][valid_identity]
                - observed[frame, valid_identity]
            )
            identity.append(float(np.mean(np.linalg.norm(residual, axis=1))))
    if not chamfer or not identity:
        raise ValueError("prefix validation interval has no finite observations")
    return {
        "chamfer_distance_m": float(np.mean(chamfer)),
        "pseudo_identity_error_m": float(np.mean(identity)),
        "chamfer_frame_count": len(chamfer),
        "identity_frame_count": len(identity),
    }


def build_guarded_prob4d_prefix_candidate(
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
    config: Prob4DBiasGuardConfig | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Build and prefix-gate one future-blind Prob4D state update.

    The physical and Prob4D arrays end at the released training boundary.
    Only the selected deterministic baseline may extend into the future.
    """

    cfg = config or Prob4DBiasGuardConfig()
    baseline_input = np.asarray(selected_baseline_m)
    baseline = np.asarray(baseline_input, dtype=np.float64)
    physical_prefix = np.asarray(physical_prefix_m, dtype=np.float64)
    measurement_prefix = np.asarray(
        prob4d_prefix_positions_m, dtype=np.float64
    )
    direct_valid = np.asarray(prob4d_prefix_validity, dtype=bool)
    prior_reliability = np.asarray(
        prob4d_prefix_prior_reliability, dtype=np.float64
    )
    covariance = np.asarray(
        prob4d_prefix_observation_covariance_m2, dtype=np.float64
    )
    if baseline.ndim != 3 or baseline.shape[2] != 3:
        raise ValueError("selected baseline must have shape (T, N, 3)")
    train_end = len(physical_prefix)
    prefix_shape = (train_end, baseline.shape[1])
    if physical_prefix.shape != (train_end, baseline.shape[1], 3):
        raise ValueError("physical prefix shape differs from selected baseline")
    if measurement_prefix.shape != physical_prefix.shape:
        raise ValueError("Prob4D position prefix shape changed")
    for name, value in (
        ("prob4d validity", direct_valid),
        ("prob4d prior reliability", prior_reliability),
    ):
        if value.shape != prefix_shape:
            raise ValueError(f"{name} shape changed")
    if covariance.shape != (*prefix_shape, 3, 3):
        raise ValueError("Prob4D covariance must have shape (T, N, 3, 3)")
    if train_end >= len(baseline):
        raise ValueError("selected baseline has no untouched future")

    fit_end = int(np.floor(cfg.fit_fraction * train_end))
    fit_end = min(max(fit_end, 2), train_end - cfg.minimum_validation_frame_count)
    fit_frame = fit_end - 1
    if train_end - fit_end < cfg.minimum_validation_frame_count:
        raise ValueError("training prefix is too short for the validation gate")

    source_config = config_from_source_lock(source_lock)
    candidate_config = replace(source_config, update_frames=(fit_frame,))
    physical_response = np.zeros_like(baseline, dtype=np.float64)
    physical_response[:train_end] = physical_prefix - physical_prefix[0]
    response_norm = np.linalg.norm(physical_response[:train_end], axis=2)
    action_support = np.any(response_norm > 0.0, axis=0).astype(np.float64)

    measurement = np.full_like(baseline, np.nan, dtype=np.float64)
    measurement[:train_end] = measurement_prefix
    visibility = np.zeros(baseline.shape[:2], dtype=bool)
    validity = np.zeros_like(visibility)
    covariance_variance = np.trace(covariance, axis1=2, axis2=3) / 3.0
    covariance_valid = (
        np.all(np.isfinite(covariance), axis=(2, 3))
        & np.isfinite(covariance_variance)
        & (covariance_variance > 0.0)
    )
    prefix_valid = (
        direct_valid
        & covariance_valid
        & np.all(np.isfinite(measurement_prefix), axis=2)
        & np.isfinite(prior_reliability)
        & (prior_reliability > 0.0)
    )
    visibility[:train_end] = prefix_valid
    validity[:train_end] = prefix_valid
    fit_reliability = prior_reliability[fit_frame]
    reliability_row = np.where(
        np.isfinite(fit_reliability),
        np.clip(fit_reliability, 0.0, 1.0),
        0.0,
    )[None]
    fit_variance = covariance_variance[fit_frame]
    variance_row = np.where(
        np.isfinite(fit_variance) & (fit_variance > 0.0),
        np.maximum(
            fit_variance,
            candidate_config.observation_variance_floor_m2,
        ),
        candidate_config.observation_variance_floor_m2,
    )[None]
    centers = np.arange(baseline.shape[1], dtype=np.int64)
    candidate_report, candidate = predict_bias_aware_candidate_arrays(
        baseline_input,
        physical_response,
        baseline[0],
        action_support,
        measurement,
        visibility,
        validity,
        center_ids=centers,
        prior_reliability=reliability_row,
        observation_variance_m2=variance_row,
        config=candidate_config,
    )

    baseline_validation = _prefix_validation_metrics(
        baseline,
        prefix_object_points_m,
        prefix_object_visibility,
        prefix_object_motion_validity,
        num_surface_points=num_surface_points,
        start_frame=fit_end,
        end_frame=train_end,
    )
    candidate_validation = _prefix_validation_metrics(
        candidate,
        prefix_object_points_m,
        prefix_object_visibility,
        prefix_object_motion_validity,
        num_surface_points=num_surface_points,
        start_frame=fit_end,
        end_frame=train_end,
    )
    metric_names = ("chamfer_distance_m", "pseudo_identity_error_m")
    ratios = {
        name: candidate_validation[name] / baseline_validation[name]
        for name in metric_names
    }
    balanced_improvement = 1.0 - float(np.mean(list(ratios.values())))
    candidate_available = bool(candidate_report["candidate_update_count"] == 1)
    no_regression = all(ratios[name] <= 1.0 for name in metric_names)
    accepted = bool(
        candidate_available
        and no_regression
        and balanced_improvement
        >= cfg.minimum_balanced_validation_improvement_fraction
    )
    guarded = baseline_input.copy()
    if accepted:
        guarded[fit_end:] = candidate[fit_end:]
        decision = "accepted-by-disjoint-prefix-validation"
    else:
        guarded[fit_end:] = baseline_input[fit_end:]
        decision = "exact-selected-baseline-fallback"
    exact_fallback = accepted or np.array_equal(guarded, baseline_input)
    if not exact_fallback:
        raise AssertionError("Prob4D validation guard violated exact fallback")

    report = {
        "artifact_kind": "PhysTwinProb4DBiasAwarePrefixCandidate",
        "config": asdict(cfg),
        "source_bias_aware_config": asdict(source_config),
        "train_end_exclusive": train_end,
        "fit_end_exclusive": fit_end,
        "fit_frame": fit_frame,
        "validation_frame_count": train_end - fit_end,
        "candidate": candidate_report,
        "candidate_available": candidate_available,
        "candidate_accepted": accepted,
        "decision": decision,
        "bit_exact_selected_baseline_fallback": exact_fallback,
        "physical_response_rms_m": float(
            np.sqrt(np.mean(np.square(response_norm[:fit_end])))
        ),
        "physical_response_maximum_m": float(
            np.max(response_norm[:fit_end])
        ),
        "action_support_count": int(np.sum(action_support > 0.0)),
        "validation": {
            "selected_baseline": baseline_validation,
            "candidate": candidate_validation,
            "candidate_over_baseline": ratios,
            "balanced_improvement_fraction": balanced_improvement,
            "no_primary_regression": no_regression,
        },
        "information_boundary": {
            "future_prob4d_observation_read": False,
            "future_object_observation_read": False,
            "future_manual_track_read": False,
            "physical_response_frames": [0, fit_frame],
            "candidate_fit_frames": [0, fit_frame],
            "family_validation_frames": [fit_end, train_end - 1],
            "prior_reliability_uses_state_innovation": False,
            "state_innovation_likelihood_count": 1,
            "selected_baseline_future_is_action_conditioned_prediction": True,
        },
    }
    return report, candidate, guarded


__all__ = [
    "Prob4DBiasGuardConfig",
    "build_guarded_prob4d_prefix_candidate",
]
