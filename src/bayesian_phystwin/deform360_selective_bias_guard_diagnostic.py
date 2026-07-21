"""Post-open diagnostic for the frozen Deform360 bias-aware guard.

The candidate path is target-free. Open outcomes may be joined only after the
selected trajectory has been frozen, and exact fallback is verified bytewise.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any, Mapping

import numpy as np

from .bias_aware_belief import BiasAwareStateUpdateConfig
from .deform360_bias_aware_belief_development import (
    Deform360BiasAwareDevelopmentConfig,
    predict_bias_aware_candidate_arrays,
)


ARTIFACT_KIND = "Deform360SelectiveBiasAwareGuardDiagnostic"


def config_from_source_lock(
    source_lock: Mapping[str, Any],
) -> Deform360BiasAwareDevelopmentConfig:
    """Reconstruct the candidate config recorded in a frozen source lock."""

    if not bool(source_lock.get("candidate_certified", False)):
        raise ValueError("source lock did not certify the candidate")
    raw_config = source_lock.get("config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("source lock lacks a candidate config")
    config_values = dict(raw_config)
    state_update = config_values.pop("state_update", None)
    if not isinstance(state_update, Mapping):
        raise ValueError("source lock lacks the state-update config")
    config_values["state_update"] = BiasAwareStateUpdateConfig(**state_update)
    if "update_frames" in config_values:
        config_values["update_frames"] = tuple(config_values["update_frames"])
    config = Deform360BiasAwareDevelopmentConfig(**config_values)
    reconstructed = json.loads(json.dumps(asdict(config), allow_nan=False))
    recorded = json.loads(json.dumps(dict(raw_config), allow_nan=False))
    if reconstructed != recorded:
        raise ValueError("source-lock config changed during reconstruction")
    return config


def selective_reliability_and_variance(
    triangulation_inlier_view_count: np.ndarray,
    triangulation_median_reprojection_px: np.ndarray,
    *,
    selected_camera_count: int,
    observation_variance_floor_m2: float,
    reprojection_scale_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build residual-independent reliability for the selective RGB path."""

    inlier_count = np.asarray(
        triangulation_inlier_view_count, dtype=np.float64
    )
    reprojection = np.asarray(
        triangulation_median_reprojection_px, dtype=np.float64
    )
    if inlier_count.shape != reprojection.shape or inlier_count.ndim != 2:
        raise ValueError("triangulation diagnostics must share shape (U, C)")
    if selected_camera_count < 2:
        raise ValueError("at least two selected cameras are required")
    if observation_variance_floor_m2 <= 0.0:
        raise ValueError("observation variance floor must be positive")
    if reprojection_scale_px <= 0.0:
        raise ValueError("reprojection scale must be positive")

    redundancy = np.clip(
        (inlier_count - 1.0) / (selected_camera_count - 1.0), 0.0, 1.0
    )
    geometry = np.exp(-0.5 * np.square(reprojection / reprojection_scale_px))
    reliability = redundancy * geometry
    reliability[~np.isfinite(reliability)] = 0.0
    variance = np.full(
        reliability.shape, observation_variance_floor_m2, dtype=np.float64
    )
    return reliability, variance


def apply_frozen_selective_bias_guard_arrays(
    persistence_m: np.ndarray,
    driven_backbone_m: np.ndarray,
    frame_zero_points_m: np.ndarray,
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
    """Apply a frozen source guard without accepting targets or future cues."""

    config = config_from_source_lock(source_lock)
    updates = tuple(int(value) for value in np.asarray(update_frames).tolist())
    if updates != config.update_frames:
        raise ValueError("selective update frames differ from the source lock")

    baseline_input = np.asarray(persistence_m)
    driven = np.asarray(driven_backbone_m)
    if driven.shape != baseline_input.shape:
        raise ValueError("driven backbone and persistence shapes differ")
    physical_response = (
        driven.astype(np.float64) - baseline_input.astype(np.float64)
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
        frame_zero_points_m,
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
            and source_lock["candidate_certified"]
        )
        if accepted:
            selected[start:stop] = candidate[start:stop]
            reason = "frozen-source-group-bound-accepted"
        else:
            selected[start:stop] = baseline_input[start:stop]
            reason = (
                "candidate-unavailable-exact-baseline-fallback"
                if not update["candidate_available"]
                else "source-lock-rejected-exact-baseline-fallback"
            )
        exact_fallback = accepted or np.array_equal(
            selected[start:stop], baseline_input[start:stop]
        )
        if not exact_fallback:
            raise AssertionError("frozen guard violated exact fallback")
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
        "source_candidate_certified": bool(source_lock["candidate_certified"]),
        "source_upper_regret_m": float(source_lock["upper_regret_m"]),
        "candidate": candidate_report,
        "decisions": decisions,
        "candidate_available_count": int(
            sum(record["candidate_available"] for record in decisions)
        ),
        "accepted_count": int(
            sum(record["candidate_accepted"] for record in decisions)
        ),
        "exact_fallback_interval_count": int(
            sum(
                not record["candidate_accepted"]
                and record["bit_exact_baseline_fallback"]
                for record in decisions
            )
        ),
        "driven_backbone_bit_exact_persistence": bool(
            np.array_equal(driven, baseline_input)
        ),
        "physical_response_rms_m": float(
            np.sqrt(np.mean(np.square(response_norm)))
        ),
        "physical_response_maximum_m": float(np.max(response_norm)),
        "action_support_count": int(np.sum(action_support > 0.0)),
        "selected_bit_exact_persistence": bool(
            np.array_equal(selected, baseline_input)
        ),
        "information_boundary": {
            "target_argument_accepted": False,
            "future_observation_read": False,
            "physical_response_source": (
                "sealed driven backbone minus sealed persistence"
            ),
            "action_support_source": (
                "nonzero sealed physical response; no outcome used"
            ),
            "prior_reliability_uses_state_innovation": False,
            "state_innovation_likelihood_count": 1,
            "missing_cycle_covariance_policy": (
                "frozen source variance floor; eligibility is evaluated first"
            ),
        },
        "claim_boundary": (
            "post-open mechanism diagnostic using the unchanged source-v4 "
            "guard; not prospective confirmation and not selector tuning"
        ),
    }
    return report, selected


__all__ = [
    "ARTIFACT_KIND",
    "apply_frozen_selective_bias_guard_arrays",
    "config_from_source_lock",
    "selective_reliability_and_variance",
]
