"""Post-open application of the unchanged source-v4 bias-aware guard."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .deform360_bias_aware_belief_development import (
    predict_bias_aware_candidate_arrays,
)
from .deform360_selective_bias_guard_diagnostic import (
    config_from_source_lock,
    selective_reliability_and_variance,
)


ARTIFACT_KIND = "Deform360FreshPairwiseBiasGuardDiagnostic"


def apply_frozen_fresh_bias_guard_arrays(
    selected_baseline_m: np.ndarray,
    physical_prior_m: np.ndarray,
    persistence_m: np.ndarray,
    measurement_m: np.ndarray,
    measurement_visibility: np.ndarray,
    measurement_validity: np.ndarray,
    *,
    center_ids: np.ndarray,
    update_frames: np.ndarray,
    selected_camera_count: int,
    triangulation_inlier_view_count: np.ndarray,
    triangulation_median_reprojection_px: np.ndarray,
    source_lock: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    """Apply source v4 without accepting a target or future observation."""

    config = config_from_source_lock(source_lock)
    updates = tuple(int(value) for value in np.asarray(update_frames).tolist())
    if updates != config.update_frames:
        raise ValueError("fresh update frames differ from the source lock")
    baseline_input = np.asarray(selected_baseline_m)
    physical = np.asarray(physical_prior_m)
    persistence = np.asarray(persistence_m)
    if physical.shape != persistence.shape or physical.shape != baseline_input.shape:
        raise ValueError("fresh physical, persistence, and baseline shapes differ")
    physical_response = (
        physical.astype(np.float64) - persistence.astype(np.float64)
    )
    response_norm = np.linalg.norm(physical_response, axis=2)
    action_support = np.any(response_norm > 0.0, axis=0).astype(np.float64)
    reliability, variance = selective_reliability_and_variance(
        triangulation_inlier_view_count,
        triangulation_median_reprojection_px,
        selected_camera_count=selected_camera_count,
        observation_variance_floor_m2=config.observation_variance_floor_m2,
        reprojection_scale_px=config.reprojection_scale_px,
    )
    candidate_report, candidate = predict_bias_aware_candidate_arrays(
        baseline_input,
        physical_response,
        persistence[0],
        action_support,
        measurement_m,
        measurement_visibility,
        measurement_validity,
        center_ids=np.asarray(center_ids, dtype=np.int64),
        prior_reliability=reliability,
        observation_variance_m2=variance,
        config=config,
    )
    selected = baseline_input.copy()
    decisions: list[dict[str, Any]] = []
    for update in candidate_report["updates"]:
        start = int(update["frame"]) + 1
        stop = int(update["interval_end_exclusive"])
        accepted = bool(
            update["candidate_available"]
            and source_lock.get("candidate_certified") is True
        )
        if accepted:
            selected[start:stop] = candidate[start:stop]
            reason = "unchanged-source-v4-lock-accepted"
        else:
            selected[start:stop] = baseline_input[start:stop]
            reason = "candidate-unavailable-exact-baseline-fallback"
        exact_fallback = accepted or np.array_equal(
            selected[start:stop], baseline_input[start:stop]
        )
        if not exact_fallback:
            raise AssertionError("fresh source-v4 guard violated exact fallback")
        decisions.append(
            {
                "frame": int(update["frame"]),
                "interval_end_exclusive": stop,
                "candidate_available": bool(update["candidate_available"]),
                "candidate_accepted": accepted,
                "reason": reason,
                "bit_exact_baseline_fallback": exact_fallback,
            }
        )
    report = {
        "artifact_kind": ARTIFACT_KIND,
        "source_protocol_id": source_lock.get("protocol_id"),
        "source_candidate_certified": bool(
            source_lock.get("candidate_certified", False)
        ),
        "source_upper_regret_m": float(source_lock["upper_regret_m"]),
        "candidate": candidate_report,
        "decisions": decisions,
        "candidate_available_count": sum(
            record["candidate_available"] for record in decisions
        ),
        "accepted_count": sum(record["candidate_accepted"] for record in decisions),
        "exact_fallback_interval_count": sum(
            not record["candidate_accepted"]
            and record["bit_exact_baseline_fallback"]
            for record in decisions
        ),
        "physical_response_rms_m": float(
            np.sqrt(np.mean(np.square(response_norm)))
        ),
        "physical_response_maximum_m": float(np.max(response_norm)),
        "action_support_count": int(np.sum(action_support > 0.0)),
        "selected_bit_exact_baseline": bool(
            np.array_equal(selected, baseline_input)
        ),
        "information_boundary": {
            "target_argument_accepted": False,
            "future_observation_read": False,
            "source_v4_lock_changed": False,
            "physical_response_source": (
                "sealed physical prior minus sealed persistence"
            ),
            "prior_reliability_uses_state_innovation": False,
            "state_innovation_likelihood_count": 1,
            "missing_cycle_covariance_policy": (
                "frozen source variance floor"
            ),
        },
        "claim_boundary": (
            "post-open mechanism diagnostic using unchanged source v4; not "
            "prospective confirmation, selector tuning, or SOTA evidence"
        ),
    }
    return report, selected


__all__ = ["ARTIFACT_KIND", "apply_frozen_fresh_bias_guard_arrays"]
